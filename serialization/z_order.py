# --------------------------------------------------------
# 基于八叉树的稀疏卷积神经网络
# 版权所有 (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# 基于 MIT 许可证，详情见 LICENSE 文件。
# 作者：Peng-Shuai Wang
# --------------------------------------------------------

import torch

_CPU_DEVICE = torch.device("cpu")


class KeyLUT:
    def __init__(self) -> None:
        """在 CPU 上预计算三个坐标轴的编码查找表，并初始化设备缓存。"""
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        self._encode = {
            _CPU_DEVICE: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }

    def encode_lut(
        self, device: torch.device = _CPU_DEVICE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回指定设备上的三个轴编码表，首次访问该设备时复制并缓存。"""
        if device not in self._encode:
            x, y, z = self._encode[_CPU_DEVICE]
            self._encode[device] = (x.to(device), y.to(device), z.to(device))
        return self._encode[device]

    def xyz2key(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, depth: int
    ) -> torch.Tensor:
        """逐位交错三个轴的低 depth 位，返回同形状的 Z 序编码。

        x、y、z 应为整数张量；此实现用于初始化查找表。
        """
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key


_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    depth: int = 16,
) -> torch.Tensor:
    r"""通过预计算的查找表，将 x、y、z 坐标编码为 Z 序编码。

    查表实现比逐位循环更快。

    参数：
      x：x 轴坐标张量。
      y：y 轴坐标张量。
      z：z 轴坐标张量。
      depth：编码深度，须小于 17。

    返回：
      坐标的整数编码张量。
    """

    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    return key
