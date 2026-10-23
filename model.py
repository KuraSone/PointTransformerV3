"""
Point Transformer V3（Mode1）
脱离 Pointcept 框架的独立版本。

作者：Xiaoyang Wu（xiaoyang.wu.cs@gmail.com）
如果这份代码对你有帮助，请引用我们的工作。

注释中的张量形状记号：
B 为批次中的点云数，N 为点数，C 为通道数，O 为序列化顺序数，
P 为补丁数，K 为补丁点数，H 为注意力头数，N' = P * K 为填充后的序列长度，
M 为池化后的点数。
"""

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn
import torch_scatter
from addict import Dict
from timm.layers.drop import DropPath

from serialization import encode
from subm_conv import SparseConvTensor, SubMConv3d


@torch.inference_mode()
def offset2bincount(offset: torch.Tensor) -> torch.Tensor:
    """将各点云的累计点数转换为每组点数。

    offset 形状为 (B,)，返回同形状张量，例如 [3, 5] 转为 [3, 2]。
    """
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """根据累计点数生成每个点所属的批次编号。

    输入 offset 形状为 (B,)，返回形状为 (N,) 的整数张量；
    例如 [2, 5] 对应 [0, 0, 1, 1, 1]，其中 N 为总点数。
    """
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch: torch.Tensor) -> torch.Tensor:
    """将按样本分组的批次编号转换为各组累计点数。

    batch 形状为 (N,)，编号从 0 开始且按组排列；
    例如 [0, 0, 1, 1, 1] 返回 [2, 5]。
    """
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):  # type: ignore[misc]
    """
    Pointcept 的点云数据结构。

    Point 继承自字典，用于存储批量点云的坐标、特征及其他属性。
    各字段的含义如下：

    - "coord"：点云的原始坐标；
    - "grid_coord"：按指定网格大小离散化后的坐标，与网格采样有关。

    可选字段：
    - "offset"：批次中各点云的累计点数，可根据 "batch" 自动生成；
    - "batch"：每个点所属样本的批次编号，可根据 "offset" 自动生成；
    - "feat"：点云特征，作为模型的默认输入；
    - "grid_size"：网格大小，与网格采样有关。

    序列化相关字段：
    - "serialized_depth"：序列化深度，2 ** depth * grid_size 表示可编码的空间边长；
    - "serialized_code"：不同序列化顺序对应的编码；
    - "serialized_order"：根据编码得到的排序索引；
    - "serialized_inverse"：用于恢复原始点顺序的逆映射索引。

    稀疏化相关字段：
    - "sparse_shape"：稀疏卷积张量的空间形状；
    - "sparse_conv_feat"：根据 Point 中的信息构建的稀疏卷积张量。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """从字典或关键字字段创建点云，并补齐缺失的 batch 或 offset。"""
        super().__init__(*args, **kwargs)
        # 缺少 "offset" 或 "batch" 时，根据已有字段生成另一个字段。
        if "batch" not in self and "offset" in self:
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self and "batch" in self:
            self["offset"] = batch2offset(self.batch)

    def serialization(
        self,
        order: str | Sequence[str] = "z",
        depth: int | None = None,
        shuffle_orders: bool = False,
    ) -> None:
        """为点云生成空间编码、排序索引和恢复点顺序所需的逆索引。

        order 指定编码顺序，通常传入 ("z", "z-trans") 等顺序集合；
        depth 为 None 时按网格坐标推断，shuffle_orders 控制是否打乱各顺序。
        依赖 grid_coord 或 coord + grid_size，以及 batch 和 offset。
        结果原地写入 serialized_code、serialized_order、serialized_inverse
        和 serialized_depth，实际特征重排由注意力模块使用这些索引完成。
        """
        if "grid_coord" not in self:
            # 未提供 grid_coord 时，输入中必须包含 grid_size。
            # (N, 3) 连续坐标 -> (N, 3) 整数网格坐标。
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # 根据网格坐标自适应确定序列化立方体的深度，边长为 2 ** depth。
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # 编码使用 int64，当前实现要求 depth <= 16（最多 48 位坐标编码）。
        codes = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(codes)  # O 个 (N,) 编码堆叠为 (O, N)。
        # (O, N)：第 o 种顺序下，第 i 个序列位置对应的点索引。
        serialized_order = torch.argsort(code)
        # (O, N)：每个点在序列中的位置，满足 inverse[o, order[o, i]] = i。
        inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(0, code.shape[1], device=serialized_order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            serialized_order = serialized_order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = serialized_order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad: int = 96) -> None:
        """
        构建点云的稀疏卷积表示。

        点云本身是稀疏的，此处的“稀疏化”特指构建具有
        SparseConvTensor 接口的坐标和特征容器。

        依赖字段：["grid_coord" 或 "coord" + "grid_size", "batch", "feat"]。

        pad：计算稀疏张量空间形状时增加的边界填充量。
        结果原地写入 sparse_shape 和 sparse_conv_feat，特征形状为 (N, C)，
        稀疏坐标形状为 (N, 4)，依次存储批次编号及三个网格坐标。
        """
        if "grid_coord" not in self:
            # 未提供 grid_coord 时，输入中必须包含 grid_size。
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self:
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = SparseConvTensor(
            features=self.feat,  # (N, C)
            # (N, 1) 批次编号与 (N, 3) 网格坐标拼接为 (N, 4)。
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""点云模块标记类；其子类在 PointSequential 中接收完整的 Point。"""


class PointSequential(PointModule):
    r"""顺序执行模块，并适配 Point、稀疏卷积和普通 PyTorch 层。"""

    def __init__(
        self,
        *args: nn.Module | Mapping[str, nn.Module],
        **kwargs: nn.Module,
    ) -> None:
        """按位置参数、映射或关键字顺序注册子模块。"""
        super().__init__()
        if len(args) == 1 and isinstance(args[0], Mapping):
            for key, named_module in args[0].items():
                self.add_module(key, named_module)
        else:
            for idx, positional_module in enumerate(args):
                if isinstance(positional_module, Mapping):
                    raise TypeError("Mapping must be the only positional argument")
                self.add_module(str(idx), positional_module)
        for name, keyword_module in kwargs.items():
            self.add_module(name, keyword_module)

    def __len__(self) -> int:
        """返回容器中已注册模块的数量。"""
        return len(self._modules)

    def add(self, module: nn.Module, name: str | None = None) -> None:
        """将模块追加到容器；省略 name 时使用当前模块数量作为数字名称。"""
        if name is None:
            name = str(len(self._modules))
        self.add_module(name, module)

    def forward(self, point: Point) -> Point:
        """依次执行子模块，并同步 Point 中的特征表示。"""
        for module in self._modules.values():
            if isinstance(module, PointModule):
                point = module(point)
            elif isinstance(module, SubMConv3d):
                point.sparse_conv_feat = module(point.sparse_conv_feat)
                point.feat = point.sparse_conv_feat.features
            else:
                point.feat = module(point.feat)  # type: ignore[misc]
                if "sparse_conv_feat" in point:
                    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                        point.feat
                    )
        return point


class PDNorm(PointModule):
    def __init__(
        self,
        num_features: int,
        norm_layer: Callable[[int], nn.Module],
        context_channels: int = 256,
        conditions: Sequence[str] = ("ScanNet", "S3DIS", "Structured3D"),
        decouple: bool = True,
        adaptive: bool = False,
    ) -> None:
        """构建按数据集条件选择的归一化层及可选上下文调制分支。

        norm_layer 根据 num_features 创建归一化模块；decouple 控制各条件
        独立建层还是共享一层。adaptive 启用由 context_channels 维上下文
        预测的逐通道缩放与偏移，conditions 定义可接受的条件名称。
        """
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.shared_norm = norm_layer(num_features)
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point: Point) -> Point:
        """根据 condition 归一化 feat，并在启用 adaptive 时应用上下文调制。

        输入需要 feat 和 condition；条件为序列时取首项，同批次应使用同一条件。
        自适应模式还需要 context，其调制结果须能广播到特征形状 (N, C)。
        返回已更新 feat 的 Point。
        """
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.shared_norm
        point.feat = norm(point.feat)
        if self.adaptive:
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size: int, num_heads: int) -> None:
        """按补丁大小确定相对坐标范围，为三个坐标轴创建每个注意力头的偏置表。"""
        super().__init__()
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        """将点对的相对网格坐标查表转换为注意力偏置。

        coord 形状为 (P, K, K, 3)，返回 (P, H, K, K)；
        P 为补丁数量，K 为每个补丁的点数，H 为注意力头数。
        """
        # (P, K, K, 3)：三个轴的相对坐标分别平移到偏置表中各自的区段。
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))  # (P*K*K*3, H)
        out = out.view(idx.shape + (-1,)).sum(3)  # (P, K, K, 3, H) -> (P, K, K, H)
        out = out.permute(0, 3, 1, 2)  # (P, K, K, H) -> (P, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        order_index: int = 0,
        enable_rpe: bool = False,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ) -> None:
        """构建按空间序列分块的多头自注意力模块。

        channels 和 num_heads 决定特征分头方式，patch_size 指定补丁点数上限，
        order_index 选择已缓存的序列化顺序；qkv_bias、qk_scale 和两个 drop
        参数控制投影偏置、注意力缩放与随机丢弃。
        使用 PyTorch 张量运算计算注意力，可通过 enable_rpe、
        upcast_attention、upcast_softmax 配置相对位置编码与计算精度。
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.patch_size_max = patch_size
        self.patch_size = 0
        self.attn_dropout = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point: Point, order: torch.Tensor) -> torch.Tensor:
        """按给定序列索引计算补丁内点对的相对网格坐标，并缓存在 Point 中。

        order 包含填充后的取点索引，返回形状为 (P, K, K, 3) 的坐标差；
        K 为当前补丁点数，缓存键按 order_index 区分不同序列化顺序。
        """
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point:
            grid_coord = point.grid_coord[order]  # (N, 3) -> (N', 3)
            grid_coord = grid_coord.reshape(-1, K, 3)  # (N', 3) -> (P, K, 3)
            # (P, K, 1, 3) - (P, 1, K, 3) -> (P, K, K, 3)。
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        rel_pos: torch.Tensor = point[rel_pos_key]
        return rel_pos

    @torch.no_grad()
    def get_padding_and_inverse(
        self, point: Point
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据每组点数和补丁大小生成填充与去填充索引。

        pad 将填充后的序列位置映射到未填充位置，unpad 找回有效点的位置；
        点数超过补丁大小的组会重复已有点以补齐末块，结果缓存在 Point 中。
        """
        pad_key = "pad"
        unpad_key = "unpad"
        if pad_key not in point or unpad_key not in point:
            offset = point.offset  # (B,)
            bincount = offset2bincount(offset)  # (B,)
            # (B,)：点数大于 K 的点云向上补齐到 K 的整数倍后的点数。
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # 仅对点数大于补丁大小的点云进行填充。
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            # Python 数值参数显式取出标量，索引张量仍保留在原设备上。
            padded_num_points = int(_offset_pad[-1].item())
            num_points = int(_offset[-1].item())
            pad = torch.arange(padded_num_points, device=offset.device)  # (N',)
            unpad = torch.arange(num_points, device=offset.device)  # (N,)
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            point[pad_key] = pad
            point[unpad_key] = unpad
        return point[pad_key], point[unpad_key]

    def forward(self, point: Point) -> Point:
        """按空间序列分块计算注意力，并恢复输入点顺序。

        输入需要形状为 (N, C) 的 feat、offset 和序列化索引；相对位置编码
        还需要 grid_coord。完成 QKV 投影、分块注意力和输出投影后，
        返回已更新 feat 的 Point，特征仍为 (N, C)。
        """
        min_num_points = int(offset2bincount(point.offset).min().item())
        self.patch_size = min(min_num_points, self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad = self.get_padding_and_inverse(point)

        # (N',)：填充后序列中每个位置要读取的点索引。
        order = point.serialized_order[self.order_index][pad]
        # (N,)：每个点在填充后序列中的位置，用于恢复原始点顺序。
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # 按序列化顺序提取填充后的分块特征：(N, C) -> (N, 3C) -> (N', 3C)。
        qkv = self.qkv(point.feat)[order]

        # (N', 3C) -> (P, K, 3, H, C/H) -> (3, P, H, K, C/H)，
        # 拆分后 q、k、v 均为 (P, H, K, C/H)。
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        # 使用 PyTorch 矩阵乘法与 Softmax 计算注意力权重。
        if self.upcast_attention:
            q = q.float()
            k = k.float()
        attn = (q * self.scale) @ k.transpose(-2, -1)  # 形状为 (P, H, K, K)。
        if self.enable_rpe:
            # 相对位置偏置同为 (P, H, K, K)，逐元素相加。
            attn = attn + self.rpe(self.get_rel_pos(point, order))  # type: ignore[misc]
        if self.upcast_softmax:
            attn = attn.float()
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn).to(qkv.dtype)
        # (P, H, K, K) @ (P, H, K, C/H) -> (P, H, K, C/H) -> (P, K, H, C/H)
        # -> (N', C)。
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        feat = feat[inverse]  # (N', C) -> (N, C)，去掉填充并恢复原始点顺序。

        # 输出投影，形状保持 (N, C)。
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """构建两层线性映射，中间使用激活函数，两层输出均应用 Dropout。

        hidden_channels 和 out_channels 未指定时使用 in_channels；
        act_layer 提供激活层构造函数，drop 指定丢弃概率。
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将特征最后一维从 in_channels 映射为 out_channels，保留其他维度。"""
        x = self.fc1(x)  # (N, C_in) -> (N, C_hidden)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # (N, C_hidden) -> (N, C_out)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 48,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        pre_norm: bool = True,
        order_index: int = 0,
        cpe_indice_key: str | None = None,
        enable_rpe: bool = False,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ) -> None:
        """构建包含局部稀疏卷积、序列化注意力和 MLP 的残差块。

        channels 为特征宽度，mlp_ratio 控制 MLP 隐藏宽度，pre_norm 选择
        残差分支的归一化位置；norm_layer、act_layer 提供层构造函数。
        drop_path 控制分支随机丢弃，其余注意力参数传入 SerializedAttention。
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point) -> Point:
        """依次执行局部位置编码、注意力和 MLP 三个残差更新。

        输入应已完成序列化和稀疏化；输出保持点数与特征宽度，
        最后将更新后的 feat 写回 sparse_conv_feat，供后续稀疏卷积使用。
        """
        # cpe、注意力、MLP 三个残差分支的输入输出均为 (N, C)。
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] | None = None,
        reduce: Literal["sum", "mean", "min", "max"] = "max",
        shuffle_orders: bool = True,
        traceable: bool = True,  # 记录父级点云与聚类映射。
    ) -> None:
        """构建按空间编码聚类的下采样层。

        stride 取 2 的整数次幂，reduce 指定簇内特征的聚合方式，
        in_channels 和 out_channels 指定投影前后的宽度。
        norm_layer、act_layer 配置后处理层，shuffle_orders 打乱编码顺序；
        traceable 记录父级点云与逆映射，为后续反池化提供依据。
        """
        super().__init__()
        # TODO：支持任意步长的网格池化。
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point) -> Point:
        """将落在同一粗网格中的点聚合，返回较低分辨率的新 Point。

        特征按 reduce 聚合，连续坐标取均值，网格坐标和序列化索引同步更新。
        输出特征形状为 (M, out_channels)，M 不超过输入点数；
        开启 traceable 时在结果中保存 pooling_parent 和 pooling_inverse。
        """
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        # (O, N)：右移后，同一粗网格内的点编码相同。
        code = point.serialized_code >> pooling_depth * 3
        # cluster 为 (N,)，即每个点所属粗网格的编号；counts 为 (M,)。
        _, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        _, indices = torch.sort(cluster)  # (N,)：让同一粗网格的点连续排列。
        # (M + 1,)：第 m 个粗网格对应 indices[idx_ptr[m] : idx_ptr[m + 1]]。
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]  # (M,)：每个粗网格的代表点。
        code = code[:, head_indices]  # (O, N) -> (O, M)
        order = torch.argsort(code)  # (O, M)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        point_dict = Dict(
            # (N, C_in) -> (N, C_out) -> 组内取最大值 -> (M, C_out)。
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            # (N, 3) -> (M, 3)，取组内均值。
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,  # (M, 3)
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],  # (M,)
        )

        if "condition" in point:
            point_dict["condition"] = point.condition
        if "context" in point:
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] | None = None,
        traceable: bool = False,  # 保留反池化前的点云，便于追踪。
    ) -> None:
        """构建上采样特征与父级跳跃连接特征的投影层。

        in_channels、skip_channels 分别指定两路输入宽度，out_channels 为
        共同输出宽度；norm_layer 和 act_layer 配置投影后的处理层。
        traceable 控制是否在恢复后的父级点云中记录当前子级点云。
        """
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point: Point) -> Point:
        """按池化逆映射恢复父级点数，并与跳跃连接特征相加。

        从输入中取出并移除 pooling_parent、pooling_inverse，分别投影两路
        特征后，将子级特征广播回父级点位置；返回更新特征后的父级 Point。
        """
        parent: Point = point.pop("pooling_parent")  # 池化前的 N 个点。
        inverse = point.pop("pooling_inverse")  # (N,)：每个父点对应的子点编号。
        point = self.proj(point)  # (M, C_in) -> (M, C_out)
        parent = self.proj_skip(parent)  # (N, C_skip) -> (N, C_out)
        # 子点特征按 inverse 广播：(M, C_out) -> (N, C_out)，再与跳跃连接相加。
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] | None = None,
    ) -> None:
        """使用 5×5×5 子流形稀疏卷积将输入特征映射到嵌入宽度。

        in_channels 为输入特征维数，embed_channels 为输出维数；
        norm_layer 和 act_layer 可配置卷积后的归一化与激活层。
        """
        super().__init__()
        self.stem = PointSequential(
            conv=SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point) -> Point:
        """处理已稀疏化的 Point，保持点坐标与点数，返回嵌入后的点特征。"""
        point = self.stem(point)  # (N, C_in) -> (N, C_embed)
        return point


class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels: int = 6,
        order: str | Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (64, 64, 128, 256),
        dec_num_head: Sequence[int] = (4, 4, 8, 16),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cls_mode: bool = False,
        pdnorm_bn: bool = False,
        pdnorm_ln: bool = False,
        pdnorm_decouple: bool = True,
        pdnorm_adaptive: bool = False,
        pdnorm_affine: bool = True,
        pdnorm_conditions: Sequence[str] = ("ScanNet", "S3DIS", "Structured3D"),
    ) -> None:
        """按阶段配置构建 Point Transformer V3 编码器和可选解码器。

        in_channels 指定输入宽度，order 指定空间编码顺序；enc_* 和 dec_*
        分别配置各阶段的块数、特征宽度、注意力头数和补丁点数。
        编码阶段数为 len(enc_depths)，stride 数量应少一；启用解码器时，
        每组 dec_* 配置也应比编码阶段少一项。
        cls_mode 为 True 时保留编码器最粗粒度特征，为 False 时解码恢复点数。
        其余参数配置注意力、随机丢弃、归一化位置和 pdnorm_* 条件归一化。
        """
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        # 配置归一化层。
        bn_layer: Callable[[int], nn.Module]
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer: Callable[[int], nn.Module]
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # 配置激活层。
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # 构建编码器。
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # 构建解码器。
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict: Mapping[str, Any]) -> Point:
        """从批量点云字典提取特征，返回包含特征和空间信息的 Point。

        data_dict 需提供 feat (N, C)、coord (N, 3)，以及 offset (B,) 或
        batch (N,)；网格位置由 grid_coord (N, 3) 提供，或通过 coord 和
        grid_size 计算。下采样同时使用连续坐标 coord 聚合每个簇的位置。
        完成序列化、稀疏化和编码后，按 cls_mode 决定是否解码恢复输入点数；
        返回的 feat 是点特征，可继续连接任务相关的预测头。
        批次字段说明：
        https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point: Point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)  # (N, in_channels) -> (N, enc_channels[0])
        # 每个下采样阶段点数减少、通道增加，最终为 (M, enc_channels[-1])。
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)  # 恢复为 (N, dec_channels[0])。
        return point


def main() -> None:
    """随机生成分别含 60、100 个点的两组点云，在 CPU 上执行前向传播。

    batch 为 2，并通过钩子打印各阶段的特征形状和每组点数，
    展示编码器下采样与解码器恢复点数的过程。
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    # 每行是一个点的 [x, y, z]，第一组生成 60 行，第二组生成 100 行。
    num_points_1 = 60
    num_points_2 = 100
    coord_1 = torch.rand(num_points_1, 3, device=device)
    coord_2 = torch.rand(num_points_2, 3, device=device)
    # 按行拼接两组点，共 160 行；offset 记录两组各自的结束位置。
    coord = torch.cat([coord_1, coord_2], dim=0)
    # 模型还需要整数网格位置：将空间按边长 0.05 划成小格子。
    grid_coord = (coord / 0.05).int()
    color = torch.rand(coord.shape[0], 3, device=device)
    feat = torch.cat([coord, color], dim=1)  # 每点 6 维输入：XYZ 坐标 + RGB 颜色。
    offset = torch.tensor(
        [num_points_1, num_points_1 + num_points_2], dtype=torch.long, device=device
    )
    data_dict = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": offset,
    }

    # 使用较小的三阶段模型，保留完整的编码、下采样、上采样过程。
    # 使用较小的补丁，便于在 CPU 上运行示例。
    model = PointTransformerV3(
        in_channels=6,
        order=("z",),
        stride=(2, 2),
        enc_depths=(1, 1, 1),
        enc_channels=(16, 32, 64),
        enc_num_head=(2, 4, 8),
        enc_patch_size=(16, 16, 16),
        dec_depths=(1, 1),
        dec_channels=(16, 32),
        dec_num_head=(2, 4),
        dec_patch_size=(16, 16),
        drop_path=0.0,
        shuffle_orders=False,
        cls_mode=False,
    ).to(device)
    model.eval()  # 推理模式：使用固定的归一化统计量，并关闭随机丢弃。

    # 钩子只打印阶段输出；实际前向传播仍由下面的 model(data_dict) 完整执行。
    def print_stage(
        _name: str,
        _module: nn.Module,
        _inputs: object,
        _output: Point,
    ) -> None:
        """作为前向钩子，打印 name 所指阶段的特征形状与各点云点数。"""
        print(
            f"{_name}: feat={tuple(_output.feat.shape)}, "
            f"每组点数={offset2bincount(_output.offset).tolist()}"
        )

    hooks = [model.embedding.register_forward_hook(partial(print_stage, "嵌入层"))]
    for name, stage in model.enc.named_children():
        hooks.append(
            stage.register_forward_hook(partial(print_stage, f"编码器 {name}"))
        )
    for name, stage in model.dec.named_children():
        hooks.append(
            stage.register_forward_hook(partial(print_stage, f"解码器 {name}"))
        )

    print(f"输入: feat={tuple(feat.shape)}, 每组点数={[num_points_1, num_points_2]}")
    with torch.inference_mode():  # 本例只做前向计算，不构建反向传播所需的计算图。
        output = model(data_dict)
    for hook in hooks:
        hook.remove()

    print(f"输出: feat={tuple(output.feat.shape)}")


if __name__ == "__main__":
    main()
