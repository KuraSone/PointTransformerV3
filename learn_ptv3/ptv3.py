# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.12.1"]
# ///
"""PTv3默认网络的单文件前向。Python 3.12 / PyTorch 2.12.1，只依赖torch。

直接运行本文件，从末尾main()逐步调试。默认：6维输入，5级编码+4级解码，
22个注意力块，4种路线，组长上限1024，最终输出每个点64维特征。
模型设置与 Pointcept 训练默认配置（ScanNet semseg-pt-v3m1-0-base）逐项对应，
见下方常量区；与训练配置的差异仅有 fp32 注意力和不实现训练期组件。
固定推理模式，不包含训练、反向传播、DropPath或命令行参数。
运行：uv --no-config run learn_ptv3/ptv3.py
文件头声明Python和torch版本；--no-config跳过项目额外的包源配置。
参考根目录model.py（Pointcept，MIT）；卷积、空间编码也在本文件中展开。
"""

import math
import sys
from itertools import product
from typing import NotRequired, TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InputData(TypedDict):
    """PTv3.forward 的输入字典；grid_coord 与 grid_size 至少提供其一。"""

    feat: Tensor
    coord: Tensor
    grid_coord: NotRequired[Tensor]
    grid_size: NotRequired[float]
    offset: NotRequired[Tensor]
    batch: NotRequired[Tensor]


class OutputData(TypedDict):
    """PTv3.forward 的输出字典：输入的键全部保留，且各键必有值。"""

    feat: Tensor
    coord: Tensor
    grid_coord: Tensor
    grid_size: NotRequired[float]
    batch: Tensor
    offset: Tensor
    serialized_code: Tensor
    serialized_order: Tensor
    serialized_inverse: Tensor
    serialized_depth: int


class _Parent(TypedDict):
    """下采样前缓存的细分辨率数据，解码器据此恢复点数与坐标。"""

    feat: Tensor
    coord: Tensor
    grid: Tensor
    batch: Tensor
    offset: Tensor
    codes: Tensor
    names: list[str]
    depth: int
    inverse: Tensor


# ==== 模型设置：与 Pointcept 训练默认配置 configs/scannet/semseg-pt-v3m1-0-base.py
# 的 model 段逐项对应（README 模型库中 ScanNet 77.6% 的默认条目）。====
IN_CHANNELS = 6
ORDER = ("z", "z-trans", "hilbert", "hilbert-trans")
STRIDE = (2, 2, 2, 2)
ENC_DEPTHS = (2, 2, 2, 6, 2)
ENC_CHANNELS = (32, 64, 128, 256, 512)
ENC_NUM_HEAD = (2, 4, 8, 16, 32)
DEC_DEPTHS = (2, 2, 2, 2)
DEC_CHANNELS = (64, 64, 128, 256)
DEC_NUM_HEAD = (4, 4, 8, 16)
MLP_RATIO = 4
QKV_BIAS = True
# 注意力分组大小：enc_patch_size 与 dec_patch_size 在训练配置中全为 1024。
PATCH_SIZE = 1024
# 训练配置中其余项与本文件的关系：drop_path=0.3（训练期随机深度，推理恒等，
# 本文件不实现）；enable_flash=True（本文件改用原生 SDPA、全程 fp32）；
# enable_rpe=False；pdnorm_* 全部关闭；upcast_attention/softmax=False；
# cls_mode=False；attn_drop=proj_drop=0；BN 为 eps=1e-3、momentum=0.01。


def spatial_code(grid: Tensor, depth: int, route: str) -> Tensor:
    """(N,3)整数网格坐标 -> (N,)空间编号；编号用于排序，不加入特征。"""
    xyz = grid[:, [1, 0, 2]] if route.endswith("trans") else grid
    xyz = xyz.long().clone()
    code = xyz.new_zeros(xyz.shape[0])
    if depth == 0:
        return code
    if route.startswith("hilbert"):
        # Hilbert编码：依次按各位所在的子立方体交换/翻转坐标，再转为Gray码。
        # xyz一直是(N,3)，这些操作只改变排序规则，特征尚未参与计算。
        bit = 1 << (depth - 1)
        while bit > 1:
            for axis in range(3):
                high = (xyz[:, axis] & bit) != 0
                exchange = torch.where(high, 0, (xyz[:, 0] ^ xyz[:, axis]) & (bit - 1))
                xyz[:, 0] ^= torch.where(high, bit - 1, exchange)
                if axis != 0:
                    xyz[:, axis] ^= exchange
            bit >>= 1
        xyz[:, 1] ^= xyz[:, 0]
        xyz[:, 2] ^= xyz[:, 1]
        flip = code.clone()
        bit = 1 << (depth - 1)
        while bit > 1:
            flip ^= torch.where((xyz[:, 2] & bit) != 0, bit - 1, 0)
            bit >>= 1
        xyz ^= flip[:, None]  # (N,3)与(N,1)按行广播。
    # Z序直接交错x/y/z的位；Hilbert序交错变换后的位。结果均为(N,)。
    for bit in range(depth):
        for axis in range(3):
            code |= ((xyz[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return code


class SparseConv(nn.Module):
    """子流形稀疏卷积：读取相邻网格里的特征，保持原点数和坐标顺序。"""

    def __init__(self, cin: int, cout: int, kernel: int, bias: bool = True) -> None:
        super().__init__()
        self.kernel = kernel
        # 和项目spconv权重布局一致：(输出通道,K,K,K,输入通道)。
        self.weight = nn.Parameter(torch.empty(cout, kernel, kernel, kernel, cin))
        self.bias = nn.Parameter(torch.empty(cout)) if bias else None
        bound = 1 / math.sqrt(cin * kernel**3)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, grid: Tensor, batch: Tensor) -> Tensor:
        # x:(N,Cin)，grid:(N,3)，batch:(N,)。输出始终为(N,Cout)。
        size = grid.amax(0).long() + 96

        def key(position: Tensor) -> Tensor:
            # 把batch和三维坐标编码为(N,)索引，避免不同点云相互卷积。
            return (
                (batch * size[0] + position[:, 0]) * size[1] + position[:, 1]
            ) * size[2] + position[:, 2]

        keys, sorted_rows = key(grid).sort()
        center = self.kernel // 2
        out = x @ self.weight[:, center, center, center, :].T  # (N,Cin) -> (N,Cout)。
        for i, j, k in product(range(self.kernel), repeat=3):
            if (i, j, k) == (center, center, center):
                continue
            neighbor = grid + grid.new_tensor([i - center, j - center, k - center])
            valid = ((neighbor >= 0) & (neighbor < size)).all(1)  # (N,)。
            neighbor_keys = key(neighbor)  # (N,)。
            found = torch.searchsorted(keys, neighbor_keys).clamp_max(len(keys) - 1)
            rows = torch.where(valid & (keys[found] == neighbor_keys))[
                0
            ]  # 有邻居的输出行。
            source = sorted_rows[found[rows]]  # 对应的输入邻居行。
            contribution = x[source] @ self.weight[:, i, j, k, :].T  # (邻居数,Cout)。
            out.index_add_(0, rows, contribution)  # 把各方向邻居的贡献加回(N,Cout)。
        return out if self.bias is None else out + self.bias  # 偏置(Cout,)按行广播。


class Block(nn.Module):
    """一个块：卷积位置编码+残差 -> 多头注意力+残差 -> MLP+残差。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.cpe_conv = SparseConv(channels, channels, 3)
        self.cpe_linear = nn.Linear(channels, channels)
        self.cpe_norm = nn.LayerNorm(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=QKV_BIAS)
        self.attention_projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels * MLP_RATIO)
        self.fc2 = nn.Linear(channels * MLP_RATIO, channels)

    def forward(
        self,
        x: Tensor,
        grid: Tensor,
        batch: Tensor,
        order: Tensor,
        offset: Tensor,
        cpe_input: Tensor | None = None,
    ) -> Tensor:
        n, c = x.shape
        h, d = self.heads, c // self.heads  # 默认各层每头都是16维。
        # 1. CPE：卷积、线性层、LayerNorm都保持(N,C)，再与原特征相加。
        conv_x = x if cpe_input is None else cpe_input
        position = self.cpe_conv(conv_x, grid, batch)  # (N,C) -> (N,C)。
        x = x + self.cpe_norm(self.cpe_linear(position))  # 第一条残差，(N,C)。

        # 2. QKV投影并按一条路线排序：(N,C) -> (N,3C)。
        shortcut = x
        # 全程fp32：不复刻原版FlashAttention在注意力前后的fp16转换。
        qkv = self.qkv(self.norm1(x))[order]
        outputs = []
        start = 0
        for end in offset.tolist():
            # 每片点云单独切组，组间、点云间均不在当前层直接做注意力。
            chunks = list(qkv[start:end].split(PATCH_SIZE))  # 每段(L,3C)。
            start = end
            for number, chunk in enumerate(chunks):
                real_length = len(chunk)
                if len(chunks) > 1 and real_length < PATCH_SIZE:
                    # 末组不足 patch 大小时，借前一组的尾部；借点位置与原版一致。
                    chunk = torch.cat((chunk, chunks[-2][real_length:]), dim=0)
                length = len(chunk)
                q, k, v = chunk.reshape(length, 3, h, d).permute(1, 2, 0, 3).unbind(0)
                # 各(H,L,D)。原生SDPA计算softmax(QK转置/sqrt(D))V；
                # 内部注意力权重为(H,L,L)，结果为(H,L,D)，使用PyTorch原生实现。
                mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                mixed = mixed.transpose(0, 1).reshape(length, c)  # (H,L,D) -> (L,C)。
                outputs.append(mixed[:real_length])  # 去掉借来的行，保留本组真实点。
                if number == 0:
                    print(f"    Q/K/V {tuple(q.shape)}；权重({h},{length},{length})")
        mixed = torch.cat(outputs, dim=0)  # 各组拼回(N,C)。
        mixed = mixed[order.argsort()]  # (N,C)，恢复输入点的行顺序。
        x = shortcut + self.attention_projection(mixed)  # 第二条残差，(N,C)。

        # 3. MLP逐点改变通道：(N,C) -> (N,4C) -> (N,C)，再加残差。
        hidden = F.gelu(self.fc1(self.norm2(x)))  # (N,4C)。
        x = x + self.fc2(hidden)  # 第三条残差，(N,C)。
        print(f"    MLP：({n},{c}) -> ({n},{c * MLP_RATIO}) -> {tuple(x.shape)}")
        return x


class PTv3(nn.Module):
    """只展开项目的默认推理网络，输入/输出使用普通字典。"""

    def __init__(self) -> None:
        super().__init__()
        enc_c, enc_h, enc_depth = ENC_CHANNELS, ENC_NUM_HEAD, ENC_DEPTHS
        dec_c, dec_h = DEC_CHANNELS, DEC_NUM_HEAD
        self.embedding_conv = SparseConv(IN_CHANNELS, ENC_CHANNELS[0], 5, bias=False)
        self.embedding_norm = nn.BatchNorm1d(
            ENC_CHANNELS[0], eps=1e-3, momentum=0.01
        )
        # ModuleList 负责参数登记（state_dict、设备迁移）；enc_stages 保存
        # 同一批 Block 的类型化引用，前向按 list[Block] 遍历。
        self.encoder = nn.ModuleList()
        self.enc_stages: list[list[Block]] = []
        for c, h, depth in zip(enc_c, enc_h, enc_depth, strict=True):
            blocks = [Block(c, h) for _ in range(depth)]
            self.encoder.append(nn.ModuleList(blocks))
            self.enc_stages.append(blocks)
        self.down_linear = nn.ModuleList(
            nn.Linear(enc_c[s], enc_c[s + 1]) for s in range(4)
        )
        self.down_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in enc_c[1:]
        )
        self.up_low_linear = nn.ModuleList(
            nn.Linear(cin, cout)
            for cin, cout in zip((64, 128, 256, 512), dec_c, strict=True)
        )
        self.up_skip_linear = nn.ModuleList(
            nn.Linear(cin, cout) for cin, cout in zip(enc_c[:4], dec_c, strict=True)
        )
        self.up_low_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in dec_c
        )
        self.up_skip_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in dec_c
        )
        self.decoder = nn.ModuleList()
        self.dec_stages: list[list[Block]] = []
        for c, h in zip(dec_c, dec_h, strict=True):
            blocks = [Block(c, h) for _ in range(2)]
            self.decoder.append(nn.ModuleList(blocks))
            self.dec_stages.append(blocks)
        self.eval()  # 固定推理，BN使用保存的均值与方差。

    @torch.inference_mode()
    def forward(self, data: InputData) -> OutputData:
        # 输入：N个点，feat(N,6)，coord/grid_coord(N,3)，offset(B,)或batch(N,)。
        x, coord = data["feat"], data["coord"]
        assert x.ndim == 2 and x.shape[1] == IN_CHANNELS, (
            f"默认输入特征必须是(N,{IN_CHANNELS})。"
        )
        if "offset" in data:
            offset = data["offset"].long()
            counts = torch.diff(offset, prepend=offset.new_zeros(1))  # (B,)。
            batch = torch.arange(len(counts), device=x.device).repeat_interleave(
                counts
            )  # (N,)。
        else:
            batch = data["batch"].long()  # (N,)。
            offset = batch.bincount().cumsum(0)  # (B,)。
        grid = data.get("grid_coord")
        if grid is None:
            if "grid_size" not in data:
                raise KeyError("输入需要 grid_coord 或 grid_size 之一。")
            grid = torch.div(
                coord - coord.amin(0), data["grid_size"], rounding_mode="trunc"
            ).int()  # (N,3)。
        depth = int(grid.max().item()).bit_length()
        assert depth <= 16, "空间编码最多支持每轴16位。"
        names = list(ORDER)
        # 四种(N,)编码 -> (4,N)。batch占高位，使排序后各片点云仍连续分开。
        codes = torch.stack([spatial_code(grid, depth, name) for name in names])
        codes |= batch[None] << (3 * depth)  # (4,N)与(1,N)广播。
        permutation = torch.randperm(4)
        codes, names = codes[permutation], [names[i] for i in permutation.tolist()]

        # 输入嵌入：(N,6) -> (N,32)，点数不变。
        x = F.gelu(self.embedding_norm(self.embedding_conv(x, grid, batch)))
        print("输入嵌入：", tuple(data["feat"].shape), "->", tuple(x.shape))
        parents: list[_Parent] = []  # 保存4次下采样之前的数据，之后从后往前恢复。

        # 编码器：N0/32 -> N1/64 -> N2/128 -> N3/256 -> N4/512。
        # Ni由粗网格内实际有多少组决定，不能直接按点数除以8计算。
        for stage, blocks in enumerate(self.enc_stages):
            if stage > 0:
                shift = (STRIDE[stage - 1] - 1).bit_length()  # stride=2 -> 右移1位。
                shift = shift if depth >= shift else 0  # 深度不足时不再缩小。
                coarse_codes = codes >> (3 * shift)  # (4,N)，去掉最低一级空间位。
                _, cluster, counts = torch.unique(
                    coarse_codes[0],
                    sorted=True,
                    return_inverse=True,
                    return_counts=True,
                )
                # cluster(N,)给出每个细点的组号；counts(M,)给出M个粗点各含几个细点。
                parents.append(
                    {
                        "feat": x,
                        "coord": coord,
                        "grid": grid,
                        "batch": batch,
                        "offset": offset,
                        "codes": codes,
                        "names": names,
                        "depth": depth,
                        "inverse": cluster,
                    }
                )
                sorted_rows = cluster.argsort()  # (N,)，把同组点排在一起。
                first = sorted_rows[
                    F.pad(counts.cumsum(0), (1, 0))[:-1]
                ]  # (M,)，组代表。
                projected = self.down_linear[stage - 1](x)  # (N,Cin) -> (N,Cout)。
                x = torch.segment_reduce(
                    projected[sorted_rows], "max", lengths=counts
                )  # -> (M,Cout)。
                x = F.gelu(self.down_norm[stage - 1](x))  # (M,Cout)，BN+GELU。
                coord = torch.segment_reduce(
                    coord[sorted_rows], "mean", lengths=counts
                )  # (N,3) -> (M,3)。
                grid, batch = grid[first] >> shift, batch[first]  # (M,3)、(M,)。
                offset = batch.bincount().cumsum(0)  # (B,)，粗分辨率下各点云边界。
                codes, depth = coarse_codes[:, first], depth - shift  # (4,M)。
                permutation = torch.randperm(4)  # 原版每次下采样也会重新洗牌。
                codes, names = (
                    codes[permutation],
                    [names[i] for i in permutation.tolist()],
                )
                print(
                    f"下采样stage{stage}：{tuple(projected.shape)} -> {tuple(x.shape)}"
                )
            for number, block in enumerate(blocks):
                route = number % 4  # 每个阶段重新从0开始，阶段内按0、1、2、3轮换。
                print(f"编码stage{stage}/block{number}，路线={names[route]}")
                x = block(x, grid, batch, codes[route].argsort(), offset)

        # 解码器：恢复父级点数，先将两条支路投影到相同通道，再相加。
        for stage in (3, 2, 1, 0):
            low = F.gelu(
                self.up_low_norm[stage](self.up_low_linear[stage](x))
            )  # (M,Cout)。
            parent = parents.pop()
            skip = F.gelu(
                self.up_skip_norm[stage](self.up_skip_linear[stage](parent["feat"]))
            )  # (N,Cout)。
            x = (
                skip + low[parent["inverse"]]
            )  # (M,Cout)按归属复制到(N,Cout)，然后残差相加。
            coord, grid, batch, offset = (
                parent["coord"],
                parent["grid"],
                parent["batch"],
                parent["offset"],
            )
            codes, names, depth = parent["codes"], parent["names"], parent["depth"]
            print(f"上采样到解码stage{stage}：{tuple(low.shape)} -> {tuple(x.shape)}")
            for number, block in enumerate(self.dec_stages[stage]):
                print(f"解码stage{stage}/block{number}，路线={names[number % 4]}")
                # 原版上采样未立即刷新卷积缓存：首块CPE读skip，残差读相加结果。
                cpe_input = skip if number == 0 else None
                x = block(
                    x, grid, batch, codes[number % 4].argsort(), offset, cpe_input
                )

        # feat(N0,64)，恢复原点数、原点顺序和坐标；这是主干特征，尚未接预测头。
        order = codes.argsort(dim=1)  # (4,N0)，四种序列位置到原点编号的映射。
        return {
            **data,
            "feat": x,
            "coord": coord,
            "grid_coord": grid,
            "batch": batch,
            "offset": offset,
            "serialized_code": codes,
            "serialized_order": order,
            "serialized_inverse": order.argsort(dim=1),
            "serialized_depth": depth,
        }


def main() -> None:
    # 直接在这里改输入点数和设备，不需要任何命令行选项。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = "cpu"
    n = 88
    flat = torch.randperm(32**3, device=device)[:n]
    grid = torch.stack(
        (flat // 1024, flat // 32 % 32, flat % 32), dim=1
    ).int()  # (88,3)，无重复网格。
    data: InputData = {
        "feat": torch.rand(n, 6, device=device),  # (88,6)，示例逐点特征。
        "coord": grid.float() * 0.02,  # (88,3)，连续坐标。
        "grid_coord": grid,  # (88,3)，整数网格坐标。
        "offset": torch.tensor([n], device=device),  # (1,)，只有一片点云。
    }
    model = PTv3().to(device)  # 完整默认结构，22个注意力块。
    output = model(data)  # 进入上面的forward；仅推理，不构建反向图。
    print("\n最终逐点特征：", tuple(output["feat"].shape))  # (88,64)。
    print("参数总数：", sum(parameter.numel() for parameter in model.parameters()))
    print("运行版本：Python", sys.version.split()[0], "PyTorch", torch.__version__)


if __name__ == "__main__":
    main()
