import math
import sys
from itertools import pairwise, product
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InputData(TypedDict):
    """PTv3.forward 的单点云输入，所有字段必填。"""

    feat: Tensor
    coord: Tensor
    grid_coord: Tensor


class OutputData(TypedDict):
    """PTv3.forward 的输出字典：输入的键全部保留，且各键必有值。"""

    feat: Tensor
    coord: Tensor
    grid_coord: Tensor
    batch: Tensor
    serialized_code: Tensor
    serialized_order: Tensor
    serialized_inverse: Tensor
    serialized_depth: int


class _Parent(TypedDict):
    """下采样前缓存的细分辨率数据，解码器据此恢复点数与坐标。"""

    feat: Tensor
    coord: Tensor
    grid: Tensor
    codes: Tensor
    names: list[str]
    depth: int
    inverse: Tensor


def spatial_code(grid: Tensor, depth: int, route: str) -> Tensor:
    """(N,3)整数网格坐标 -> (N,)空间编号；编号用于排序，不加入特征。"""
    # trans 路线交换前两轴。
    xyz = grid[:, [1, 0, 2]] if route.endswith("trans") else grid
    # 转 int64，便于按位运算。
    xyz = xyz.to(torch.int64)
    # (N,)编号累加器，初值全 0。
    code = xyz.new_zeros(xyz.shape[0])
    if depth == 0:
        return code
    if route.startswith("hilbert"):
        # 后续会就地修改坐标，先复制以保护输入。
        xyz = xyz.clone()
        # Hilbert编码：依次按各位所在的子立方体交换/翻转坐标，再转为Gray码。
        # 当前处理的二进制位，从最高位开始。
        bit = 1 << (depth - 1)
        while bit > 1:
            for axis in range(3):
                # (N,)该轴当前位是否为 1。
                high = (xyz[:, axis] & bit) != 0
                # (N,)低位交换掩码：该位为 0 时记录与 x 轴低位的差异。
                exchange = torch.where(high, 0, (xyz[:, 0] ^ xyz[:, axis]) & (bit - 1))
                xyz[:, 0] ^= torch.where(high, bit - 1, exchange)
                if axis != 0:
                    xyz[:, axis] ^= exchange
            bit >>= 1
        # Gray 码变换：三轴级联异或，再按第三轴累加翻转掩码。
        xyz[:, 1] ^= xyz[:, 0]
        xyz[:, 2] ^= xyz[:, 1]
        # (N,)翻转掩码，初值全 0。
        flip = code.clone()
        bit = 1 << (depth - 1)
        while bit > 1:
            flip ^= torch.where((xyz[:, 2] & bit) != 0, bit - 1, 0)
            bit >>= 1
        # (N,3)与(N,1)按行广播。
        xyz ^= flip[:, None]
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
        # 权重形状：(输出通道,K,K,K,输入通道)。
        self.weight = nn.Parameter(torch.empty(cout, kernel, kernel, kernel, cin))
        self.bias = nn.Parameter(torch.empty(cout)) if bias else None
        # 均匀初始化上界，1/sqrt(扇入)。
        bound = 1 / math.sqrt(cin * kernel**3)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, grid: Tensor) -> Tensor:
        # x:(N,Cin)，grid:(N,3)。输出始终为(N,Cout)。
        # (3,)各轴的网格大小。
        size = grid.amax(0).to(torch.int64) + 1

        def key(xyz: Tensor) -> Tensor:
            # 把三维坐标编码为(N,)索引。
            return (xyz[:, 0] * size[1] + xyz[:, 1]) * size[2] + xyz[:, 2]

        # keys：(N,)排序后键；sorted_rows：键所在的原行号。
        keys, sorted_rows = key(grid).sort()
        # 核中心下标。
        center = self.kernel // 2
        # (N,Cin) -> (N,Cout)。
        out = x @ self.weight[:, center, center, center, :].T
        for i, j, k in product(range(self.kernel), repeat=3):
            if (i, j, k) == (center, center, center):
                continue
            # (N,3)当前偏移方向的邻居坐标。
            neighbor = grid + grid.new_tensor([i - center, j - center, k - center])
            # (N,)布尔，邻居是否落在网格范围内。
            valid = ((neighbor >= 0) & (neighbor < size)).all(1)
            # (N,)邻居点的键。
            neighbor_keys = key(neighbor)
            # (N,)各邻居键在 keys 中的插入位置。
            found = torch.searchsorted(keys, neighbor_keys).clamp_max(len(keys) - 1)
            # (N,)有邻居的输出行。
            rows = torch.where(valid & (keys[found] == neighbor_keys))[0]
            # 对应的输入邻居行。
            source = sorted_rows[found[rows]]
            # (邻居数,Cout)。
            contribution = x[source] @ self.weight[:, i, j, k, :].T
            # 把各方向邻居的贡献加回(N,Cout)。
            out.index_add_(0, rows, contribution)
        # 偏置(Cout,)按行广播。
        return out if self.bias is None else out + self.bias


class Block(nn.Module):
    """一个块：卷积位置编码+残差 -> 多头注意力+残差 -> MLP+残差。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        # 注意力头数。
        self.heads = heads
        # 注意力分组的点数上限。
        self.patch_size = 1024
        # 卷积位置编码：3×3×3 邻域稀疏卷积。
        self.cpe_conv = SparseConv(channels, channels, kernel=3)
        # 位置编码的线性变换。
        self.cpe_linear = nn.Linear(channels, channels)
        # 位置编码的归一化。
        self.cpe_norm = nn.LayerNorm(channels)
        # 注意力前的归一化。
        self.norm1 = nn.LayerNorm(channels)
        # 联合投影出 Q/K/V。
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        # 注意力输出投影。
        self.attention_projection = nn.Linear(channels, channels)
        # MLP 前的归一化。
        self.norm2 = nn.LayerNorm(channels)
        hidden_channels = channels * 4
        # MLP 升维。
        self.fc1 = nn.Linear(channels, hidden_channels)
        # MLP 降维。
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, x: Tensor, grid: Tensor, order: Tensor) -> Tensor:
        # x:(N,C)特征，grid:(N,3)网格坐标，order:(N,)某条路线的排序索引。
        # 点数、通道数。
        n, c = x.shape
        # 默认各层每头都是16维。
        h, d = self.heads, c // self.heads
        # 1. CPE：卷积、线性层、LayerNorm都保持(N,C)，再与原特征相加。
        # (N,C) -> (N,C)。
        position = self.cpe_conv(x, grid)
        # 第一条残差，(N,C)。
        x = x + self.cpe_norm(self.cpe_linear(position))

        # 2. QKV投影并按一条路线排序：(N,C) -> (N,3C)。
        qkv = self.qkv(self.norm1(x))[order]
        # 按路线排序后，直接把整片点云切成每段最多1024个点的组。
        chunks = qkv.split(self.patch_size)
        # 依次收集各组注意力输出，(L,C)。
        outputs = []
        for number, chunk in enumerate(chunks):
            # 本组真实点数（不含借点）。
            real_length = len(chunk)
            if len(chunks) > 1 and real_length < self.patch_size:
                # 末组不足 patch 大小时，借前一组的尾部；借点位置与原版一致。
                chunk = torch.cat((chunk, chunks[-2][real_length:]), dim=0)
            # 借点后的组长度。
            length = len(chunk)
            q, k, v = chunk.reshape(length, 3, h, d).permute(1, 2, 0, 3).unbind(0)
            # Q/K/V各为(H,L,D)，注意力权重为(H,L,L)。
            # SDPA计算softmax(QK转置/sqrt(D))V，输出(H,L,D)。
            mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            # (H,L,D) -> (L,C)。
            mixed = mixed.transpose(0, 1).reshape(length, c)
            # 去掉借来的行，保留本组真实点。
            outputs.append(mixed[:real_length])
            if number == 0:
                print(f"    Q/K/V {tuple(q.shape)}；权重({h},{length},{length})")
        # 各组拼回(N,C)。
        mixed = torch.cat(outputs, dim=0)
        # (N,C)，恢复输入点的行顺序。
        mixed = mixed[order.argsort()]
        # 第二条残差，(N,C)。
        x = x + self.attention_projection(mixed)

        # 3. MLP逐点改变通道：(N,C) -> (N,4C) -> (N,C)，再加残差。
        # (N,4C)。
        hidden = F.gelu(self.fc1(self.norm2(x)))
        # 第三条残差，(N,C)。
        x = x + self.fc2(hidden)
        print(f"    MLP：({n},{c}) -> {tuple(hidden.shape)} -> {tuple(x.shape)}")
        return x


class PTv3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 6
        # 编码器各阶段的通道数、头数、块数。
        enc_channels = (32, 64, 128, 256, 512)
        enc_heads = (2, 4, 8, 16, 32)
        enc_depths = (2, 2, 2, 6, 2)
        self.embedding_conv = SparseConv(
            self.in_channels, enc_channels[0], kernel=5, bias=False
        )
        self.embedding_norm = nn.BatchNorm1d(enc_channels[0], eps=1e-3, momentum=0.01)
        # ModuleList登记参数；阶段列表引用同一批Block，供前向直接遍历。
        self.encoder = nn.ModuleList()
        self.enc_stages: list[list[Block]] = []
        for c, h, depth in zip(enc_channels, enc_heads, enc_depths, strict=True):
            # 该级 depth 个块。
            blocks = [Block(c, h) for _ in range(depth)]
            self.encoder.append(nn.ModuleList(blocks))
            self.enc_stages.append(blocks)
        self.down_linear = nn.ModuleList(
            nn.Linear(cin, cout) for cin, cout in pairwise(enc_channels)
        )
        self.down_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in enc_channels[1:]
        )
        # 解码器各阶段的通道数、头数、块数。
        dec_channels = (64, 64, 128, 256)
        dec_heads = (4, 4, 8, 16)
        dec_depths = (2, 2, 2, 2)
        self.up_low_linear = nn.ModuleList(
            nn.Linear(cin, cout)
            for cin, cout in zip(enc_channels[1:], dec_channels, strict=True)
        )
        self.up_skip_linear = nn.ModuleList(
            nn.Linear(cin, cout)
            for cin, cout in zip(enc_channels[:-1], dec_channels, strict=True)
        )
        self.up_low_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in dec_channels
        )
        self.up_skip_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in dec_channels
        )
        self.decoder = nn.ModuleList()
        self.dec_stages: list[list[Block]] = []
        for c, h, depth in zip(dec_channels, dec_heads, dec_depths, strict=True):
            # 该级 depth 个块。
            blocks = [Block(c, h) for _ in range(depth)]
            self.decoder.append(nn.ModuleList(blocks))
            self.dec_stages.append(blocks)
        # 固定推理，BN使用保存的均值与方差。
        self.eval()

    @torch.inference_mode()
    def forward(self, data: InputData) -> OutputData:
        # 单片点云：N个点，feat(N,6)，coord(N,3)，grid_coord(N,3)。
        # x：(N,6)逐点特征；coord：(N,3)连续坐标。
        x, coord = data["feat"], data["coord"]
        # (N,3)整数网格坐标。
        grid = data["grid_coord"]
        # 每轴坐标需要的二进制位数。
        depth = int(grid.max().item()).bit_length()
        # 四种序列化路线。
        names = ["z", "z-trans", "hilbert", "hilbert-trans"]
        # 四种(N,)空间编码叠成(4,N)。
        codes = torch.stack([spatial_code(grid, depth, name) for name in names])
        # 随机打乱路线次序，codes 与 names 同步重排。
        permutation = torch.randperm(len(names))
        codes, names = codes[permutation], [names[i] for i in permutation.tolist()]

        # 输入嵌入：(N,6) -> (N,32)，点数不变。
        x = F.gelu(self.embedding_norm(self.embedding_conv(x, grid)))
        # 保存4次下采样之前的数据，之后从后往前恢复。
        parents: list[_Parent] = []

        # 编码器：N0/32 -> N1/64 -> N2/128 -> N3/256 -> N4/512。
        # Ni由粗网格内实际有多少组决定，不能直接按点数除以8计算。
        for stage, blocks in enumerate(self.enc_stages):
            if stage > 0:
                # 各次下采样步长均为2：右移1位；深度为0时不再缩小。
                shift = 1 if depth > 0 else 0
                # (4,N)，去掉最低一级空间位。
                coarse_codes = codes >> (3 * shift)
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
                        "codes": codes,
                        "names": names,
                        "depth": depth,
                        "inverse": cluster,
                    }
                )
                # (N,)，把同组点排在一起。
                sorted_rows = cluster.argsort()
                # 每组的结束位置减去组大小，就是该组在排序后的起始位置。
                group_starts = counts.cumsum(0) - counts
                # (M,)，每组的代表点。
                first = sorted_rows[group_starts]
                # (N,Cin) -> (N,Cout)。
                projected = self.down_linear[stage - 1](x)
                # -> (M,Cout)。
                x = torch.segment_reduce(projected[sorted_rows], "max", lengths=counts)
                # (M,Cout)，BN+GELU。
                x = F.gelu(self.down_norm[stage - 1](x))
                # 特征取组内最大值；连续坐标取组内均值，保持同一级点数。
                # (N,3) -> (M,3)。
                coord = torch.segment_reduce(coord[sorted_rows], "mean", lengths=counts)
                # (M,3)。
                grid = grid[first] >> shift
                # (4,M)。
                codes, depth = coarse_codes[:, first], depth - shift
                permutation = torch.randperm(len(names))
                codes, names = (
                    codes[permutation],
                    [names[i] for i in permutation.tolist()],
                )
                print(
                    f"下采样stage{stage}：{tuple(projected.shape)} -> {tuple(x.shape)}"
                )
            for number, block in enumerate(blocks):
                # 各阶段从0开始轮换路线。
                route = number % len(names)
                print(f"编码stage{stage}/block{number}，路线={names[route]}")
                x = block(x, grid, codes[route].argsort())

        # 解码器：恢复父级点数，先将两条支路投影到相同通道，再相加。
        for stage in reversed(range(len(self.dec_stages))):
            # (M,Cout)。
            low = F.gelu(self.up_low_norm[stage](self.up_low_linear[stage](x)))
            # 弹出当前级下采样前缓存的细分辨率数据。
            parent = parents.pop()
            # (N,Cout)。
            skip = F.gelu(
                self.up_skip_norm[stage](self.up_skip_linear[stage](parent["feat"]))
            )
            # (M,Cout)按归属复制到(N,Cout)，然后残差相加。
            x = skip + low[parent["inverse"]]
            coord, grid = parent["coord"], parent["grid"]
            codes, names, depth = parent["codes"], parent["names"], parent["depth"]
            print(f"上采样到解码stage{stage}：{tuple(low.shape)} -> {tuple(x.shape)}")
            for number, block in enumerate(self.dec_stages[stage]):
                route = number % len(names)
                print(f"解码stage{stage}/block{number}，路线={names[route]}")
                x = block(x, grid, codes[route].argsort())

        # feat(N0,64)，恢复原点数、原点顺序和坐标；这是主干特征，尚未接预测头。
        # (4,N0)，四种序列位置到原点编号的映射。
        order = codes.argsort(dim=1)
        return {
            **data,
            "feat": x,
            "coord": coord,
            "grid_coord": grid,
            "batch": torch.ones(len(x), dtype=torch.int64, device=x.device),
            "serialized_code": codes,
            "serialized_order": order,
            "serialized_inverse": order.argsort(dim=1),
            "serialized_depth": depth,
        }


def main() -> None:
    # 运算设备：演示在 CPU 上跑。
    device = "cpu"
    # 模拟点云的点数。
    n = 88
    # (88,)32³网格的一维展开下标，随机取 n 个互不重复。
    flat = torch.randperm(32**3, device=device)[:n]
    # (88,3)：一维下标 i=a*1024+b*32+c 解码回 32³ 网格的三维体素坐标，互不重复。
    grid = torch.stack((flat // 1024, flat // 32 % 32, flat % 32), dim=1).to(
        torch.int32
    )
    data: InputData = {
        # (88,6)，示例逐点特征。
        "feat": torch.rand(n, 6, device=device),
        # (88,3)，连续坐标。
        "coord": grid.to(torch.float32) * 0.02,
        # (88,3)，整数网格坐标。
        "grid_coord": grid,
    }
    # 构建模型并搬到设备。
    model = PTv3().to(device)
    # 前向一次，得到输出字典。
    output = model(data)
    print(f"\n最终逐点特征： {tuple(output['feat'].shape)}")
    print("参数总数：", sum(parameter.numel() for parameter in model.parameters()))
    print(f"运行版本：Python {sys.version.split()[0]} PyTorch {torch.__version__}")


if __name__ == "__main__":
    main()
