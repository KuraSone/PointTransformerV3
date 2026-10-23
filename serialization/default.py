import torch

from .hilbert import encode as hilbert_encode_
from .z_order import xyz2key as z_order_encode_


@torch.inference_mode()
def encode(
    grid_coord: torch.Tensor,
    batch: torch.Tensor | None = None,
    depth: int = 16,
    order: str = "z",
) -> torch.Tensor:
    """按指定空间顺序将 (N, 3) 网格坐标编码为整数，并可叠加批次编号。

    order 支持 Z 序、Hilbert 序及交换 x/y 轴的变体，depth 指定每轴位数。
    batch 为可选的 (N,) 编号，写入坐标编码以上的高位。
    返回逐点编码；Hilbert 路径仅有一个点时可能返回标量张量。
    """
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    # code 为 (N,)，占用低 depth * 3 位。
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code  # (N,)：批次编号写入高位。
    return code


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """将 (N, 3) 网格坐标转换为 (N,) Z 序编码，depth 指定每轴编码位数。"""
    # (N, 3) 拆成三个 (N,) 坐标分量。
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # 此处不处理批次编号，由 Point 类统一维护包含批次信息的编码。
    code = z_order_encode_(x, y, z, depth=depth)
    return code


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """使用每轴 depth 位编码三维网格坐标，返回逐点 Hilbert 编码张量。

    编码展平点维度，单个点的结果会被底层实现压缩为标量。
    """
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)
