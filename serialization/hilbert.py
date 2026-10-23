"""
Hilbert 序（希尔伯特曲线编码）
修改自 https://github.com/PrincetonLIPS/numpy-hilbert-curve

作者：Xiaoyang Wu（xiaoyang.wu.cs@gmail.com）、Kaixin Xu
如果这份代码对你有帮助，请引用我们的工作。
"""

import torch


def right_shift(
    binary: torch.Tensor, k: int | torch.Tensor = 1, axis: int = -1
) -> torch.Tensor:
    """对二进制张量执行右移，在前端补零并截断末尾。

    参数：
      binary：二进制位组成的张量。
      k：右移位数，默认为 1，可为整数或仅含单个整数的张量。
      axis：用于计算切片的维度，默认为最后一维。

    返回：
      右移后的张量。
    """

    k = int(k)
    # 右移位数覆盖整个目标维度时，直接返回全零张量。
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # 构造切片，去掉目标维度末尾的 k 位。
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def gray2binary(gray: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """将格雷码张量还原为二进制位张量。

    参数：
      gray：格雷码张量。
      axis：执行格雷码解码的维度，默认为最后一维。

    返回：
      二进制位组成的张量。
    """

    # 按位数的二进制对数进行多轮移位和异或。
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    """将多维超立方体内的坐标编码为 Hilbert 整数。

    本实现对 John Skilling 的 Hilbert 曲线算法进行了向量化处理。
    参考文献：Skilling, J.（2004 年 4 月），《Programming the Hilbert curve》，
    AIP Conference Proceedings，第 707 卷第 1 期，第 381—387 页。

    参数：
      locs：坐标张量，最后一维须为 num_dims，各坐标的取值范围为
            0 到 2 ** num_bits - 1。
      num_dims：超立方体的维数。
      num_bits：每个坐标维度使用的位数，总位数 num_dims * num_bits 须不超过 63。

    返回：
      使用 int64 存储的 Hilbert 编码张量，点维度展平，单个编码会压缩为标量。
    """

    # 以下记 N 为点数，D 为 num_dims，b 为 num_bits。
    bitpack_mask: torch.Tensor = 1 << torch.arange(0, 8).to(locs.device)  # (8,)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    # 将坐标整数按 64 位位模式拆分成字节序列，保留各坐标维度的对应关系。
    # (N, D) int64 -> (N, D * 8) uint8 -> (N, D, 8)，翻转后高位字节在前。
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # 将字节展开为二进制位，并保留末尾 num_bits 位：
    # (N, D, 8, 1) 与 (8,) 按位与 -> (N, D, 8, 8) -> (N, D, 64) -> (N, D, b)。
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    # 按解码的逆过程编码，依次遍历各二进制位。
    for bit in range(0, num_bits):
        # 正向遍历各坐标维度。
        for dim in range(0, num_dims):
            # 找出当前位为 1 的坐标，形状为 (N,)。
            mask = gray[:, dim, bit]

            # 当前位为 1 时，翻转第 0 维中更低的位；低位切片为 (N, b - bit - 1)。
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # 当前位为 0 时，将该维度的低位与第 0 维的低位交换。
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # 调整维度并展平二进制位：(N, D, b) -> (N, b, D) -> (N, b * D)，各维按位交错。
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # 将格雷码还原为二进制位，形状保持 (N, b * D)。
    hh_bin = gray2binary(gray)

    # 在前端补零，使每个编码达到 64 位：(N, b * D) -> (N, 64)。
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    # 将二进制位打包为字节：(N, 64) -> (N, 8, 8) -> (N, 8)。
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # 将字节按 int64 位模式重新解释：(N, 8) uint8 -> (N, 1) int64 -> (N,)。
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64
