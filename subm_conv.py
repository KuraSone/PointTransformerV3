"""仅依赖 PyTorch 的简易三维子流形稀疏卷积，支持 CPU 和 CUDA。

坐标为 [N, 4]，列顺序是 [batch, x, y, z]；特征为 [N, C]。
卷积只更新已有坐标的特征，不创建新体素，也不展开稠密三维网格。
"""

import copy
import math
from collections.abc import Sequence
from itertools import product
from typing import Any, Protocol, Self, cast

import torch
from torch import nn


class SparseTensorLike(Protocol):
    """SubMConv3d 所需的接口；原版 spconv.SparseConvTensor 也满足它。"""

    features: torch.Tensor
    indices: torch.Tensor
    spatial_shape: Sequence[int]
    batch_size: int

    def replace_feature(self, feature: torch.Tensor) -> Self:
        """以形状为 (N, C) 的新特征构建保留空间信息的稀疏张量。"""
        ...


class SparseConvTensor:
    """供本项目使用的简易容器，无需安装 spconv。

    与原版一样，通过 replace_feature 返回新对象，保留坐标和其他元信息。
    原版的高级功能（索引缓存、量化、计时等）不在此容器的实现范围内。
    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Sequence[int],
        batch_size: int,
    ) -> None:
        """保存有效体素的特征、坐标和批次信息。

        features 形状为 (N, C)，indices 为 (N, 4)，列顺序为 batch、x、y、z；
        spatial_shape 指定三维网格范围，batch_size 为批次中的点云数量。
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = list(spatial_shape)
        self.batch_size = batch_size

    def replace_feature(self, feature: torch.Tensor) -> Self:
        """返回使用新特征的浅拷贝，保留坐标、空间范围与批次信息。

        feature 必须为二维张量且点数不变，通道数可以改变。
        """
        result = copy.copy(self)
        result.features = feature
        return result


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    """将单个整数扩展为三元组，或转换已有的三个整数。

    value 应为整数或含三个整数的序列，返回用于三维卷积参数的整数元组。
    """
    values = (value, value, value) if isinstance(value, int) else tuple(value)
    return cast(tuple[int, int, int], values)


class SubMConv3d(nn.Module):
    """保持有效坐标不变的卷积，接收和返回稀疏张量，而非普通 Tensor。

    可直接接收原版 spconv.SparseConvTensor，输出仍通过其 replace_feature
    构建，因此保留输入类型、坐标顺序、空间形状及批次大小。

    权重沿用 spconv 2.2+ 的 KRSC 布局：[C_out, K_x, K_y, K_z, C_in]。
    与原版 SubM 一样，仅支持奇数卷积核和 groups=1；实际步长固定为 1，
    实际 padding 自动取 kernel_size // 2 * dilation，接收的 stride/padding
    参数不改变有效坐标。algo、large_kernel_fast_algo 仅保留调用接口；
    indice_key 不缓存索引，每次前向重新查找邻居，方便阅读与调试。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        indice_key: str | None = None,
        algo: Any = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name: str | None = None,
    ) -> None:
        """创建保留有效坐标的三维稀疏卷积权重和可选偏置。

        in_channels、out_channels 指定特征宽度，kernel_size 须为正奇数，
        dilation 控制邻居间距，groups 仅支持 1。
        stride、padding 及索引、算法相关参数用于接口兼容；实际输出坐标
        与输入一致。fp32_accum 可启用半精度输入的单精度累加。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.indice_key = indice_key
        self.algo = algo
        self.fp32_accum = fp32_accum
        self.large_kernel_fast_algo = large_kernel_fast_algo
        self.name = name

        self.weight = nn.Parameter(
            torch.empty(out_channels, *self.kernel_size, in_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """根据输入通道数与卷积核体积，以均匀分布重新初始化权重和偏置。"""
        bound = 1 / math.sqrt(self.in_channels * math.prod(self.kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward[SparseTensorT: SparseTensorLike](
        self, input: SparseTensorT, add_input: SparseTensorLike | None = None
    ) -> SparseTensorT:
        """在输入的有效体素位置执行卷积，返回相同类型的稀疏张量。

        input 需提供 (N, C_in) 特征、整数坐标和空间范围，输出特征为
        (N, C_out)。add_input 可叠加同坐标、同输出形状的残差特征。
        同批次内的有效坐标须唯一，并位于 spatial_shape 指定的范围内。
        启用 fp32_accum 时先提升半精度输入的计算精度，再恢复特征类型。
        """
        features = input.features  # (N, C_in)
        shape = _triple(input.spatial_shape)

        # 邻居索引也在特征所在设备上生成，GPU 输入无需搬回 CPU；坐标为 (N, 4)。
        coordinates = input.indices.to(device=features.device, dtype=torch.long)

        accumulate_fp32 = self.fp32_accum is True and features.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        if accumulate_fp32:
            with torch.autocast(device_type=features.device.type, enabled=False):
                output = self._convolve(
                    coordinates, shape, features.float(), self.weight.float()
                )
            output = output.to(features.dtype)
        else:
            output = self._convolve(coordinates, shape, features, self.weight)
        # output 为 (N, C_out)；偏置 (C_out,) 按行广播。
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        if add_input is not None:
            output = output + add_input.features
        return input.replace_feature(output)

    def _convolve(
        self,
        coordinates: torch.Tensor,
        shape: tuple[int, int, int],
        features: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """通过坐标编码、排序和邻居查询累加卷积贡献，返回 (N, C_out) 特征。

        coordinates 为包含批次编号的 (N, 4) 坐标，shape 为三维空间范围，
        weight 使用 (C_out, K_x, K_y, K_z, C_in) 布局。
        1×1×1 卷积沿用 spconv 的矩阵映射方式，其余卷积核逐个偏移查找邻居。
        """
        if self.kernel_size == (1, 1, 1):
            # 沿用 spconv 2.3.8 的 1x1 快路径：按 [C_in, C_out] 展平权重，
            # 以便直接加载其 state_dict，而非按普通 KRSC 卷积重新解释。
            # (N, C_in) @ (C_in, C_out) -> (N, C_out)。
            return features @ weight.reshape(self.in_channels, self.out_channels)
        center = tuple(size // 2 for size in self.kernel_size)
        # 中心体素一定存在；空输入也保留对特征和权重的自动求导关系。
        # (N, C_in) @ (C_in, C_out) -> (N, C_out)，其余偏移的贡献累加到其中。
        output = features @ weight[:, center[0], center[1], center[2], :].T
        count = coordinates.shape[0]
        if count == 0:
            return output

        def encode(coords: torch.Tensor) -> torch.Tensor:
            """将 (N, 4) 的批次及空间坐标编码为一维整数键，用于查找同批次邻居。"""
            # 编码包含 batch，防止不同点云中相同坐标相互影响。
            return (
                (coords[:, 0] * shape[0] + coords[:, 1]) * shape[1] + coords[:, 2]
            ) * shape[2] + coords[:, 3]

        keys, order = torch.sort(encode(coordinates))  # 均为 (N,)：有序键与原索引。
        spatial_limits = coordinates.new_tensor(shape)  # (3,)
        for kernel_index in product(*(range(size) for size in self.kernel_size)):
            if kernel_index == center:
                continue
            # 对输出位置 p，当前卷积核读取 p + (核索引 - 核中心) * dilation。
            offset = coordinates.new_tensor(
                [
                    (index - middle) * step
                    for index, middle, step in zip(
                        kernel_index, center, self.dilation, strict=True
                    )
                ]
            )  # (3,)
            neighbors = coordinates.clone()  # (N, 4)：每个输出位置要读取的邻居坐标。
            neighbors[:, 1:] += offset
            # (N,)：邻居坐标是否落在空间范围内；记满足条件的数量为 V。
            valid = ((neighbors[:, 1:] >= 0) & (neighbors[:, 1:] < spatial_limits)).all(
                1
            )
            output_indices = torch.where(valid)[0]  # (V,)
            neighbor_keys = encode(neighbors[valid])  # (V,)
            positions = torch.searchsorted(keys, neighbor_keys).clamp(max=count - 1)
            # (V,)：邻居体素是否真实存在；记存在的数量为 F。
            found = keys[positions] == neighbor_keys
            input_indices = order[positions[found]]  # (F,)
            kernel_weight = weight[
                :, kernel_index[0], kernel_index[1], kernel_index[2], :
            ]  # (C_out, C_in)
            # (F, C_in) @ (C_in, C_out) -> (F, C_out)，再按行累加到 (N, C_out)。
            contribution = features[input_indices] @ kernel_weight.T
            output.index_add_(0, output_indices[found], contribution)
        return output

    def extra_repr(self) -> str:
        """返回打印模块结构时展示的通道、卷积核、膨胀率和偏置信息。"""
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, bias={self.bias is not None}, "
            f"indice_key={self.indice_key!r}"
        )
