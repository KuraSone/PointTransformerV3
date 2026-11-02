# 阅读约定：N/M 表示下采样前/后的点数，C 表示特征通道数。
# H 表示注意力头数，D=C/H 表示每头通道数，L 表示一个注意力组的点数。
# 一行对应一个点；feat、coord、grid 的第 i 行始终描述同一个点。
# 本文件是单点云的推理演示：稀疏卷积 -> 编码器逐级下采样 -> 解码器逐级恢复。

import math
import sys
from itertools import pairwise, product
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InputData(TypedDict):
    """PTv3.forward 的单点云输入，所有字段必填。"""

    # TypedDict 只描述字典的类型，不会在运行时检查形状或转换数据。
    feat: Tensor  # (N,6)，每个点的输入特征，具体六个通道的含义由调用方决定。
    coord: Tensor  # (N,3)，连续空间坐标，可用于计算粗点的平均位置。
    grid_coord: Tensor  # (N,3)，非负整数体素坐标；此实现要求一个体素对应一个输入点。


class OutputData(TypedDict):
    """PTv3.forward 的输出字典：输入的键全部保留，且各键必有值。"""

    feat: Tensor  # (N,64)，主干输出特征，行顺序与输入 feat 一致。
    coord: Tensor  # (N,3)，恢复到输入点的连续坐标。
    grid_coord: Tensor  # (N,3)，恢复到输入点的体素坐标。
    batch: Tensor  # (N,)，各点所属点云的标签；这里所有点属于同一片点云。
    serialized_code: Tensor  # (4,N)，四条路线的编码，列仍按输入点顺序排列。
    serialized_order: Tensor  # (4,N)，序列位置 -> 输入点行号。
    serialized_inverse: Tensor  # (4,N)，输入点行号 -> 序列位置。
    serialized_depth: int  # 编码每个坐标轴所使用的二进制位数。


class _Parent(TypedDict):
    """下采样前缓存的细分辨率数据，解码器据此恢复点数与坐标。"""

    feat: Tensor  # (N,C)，下采样前的细点特征，解码时作为跳连分支。
    coord: Tensor  # (N,3)，下采样前的连续坐标。
    grid: Tensor  # (N,3)，下采样前的体素坐标。
    codes: Tensor  # (4,N)，下采样前各路线的编码。
    names: list[str]  # 四条路线名称，顺序与 codes 的行一一对应。
    depth: int  # 下采样前的编码深度。
    inverse: Tensor  # (N,)，每个细点对应的粗点行号，供上采样时查表。


def spatial_code(grid: Tensor, depth: int, route: str) -> Tensor:
    """(N,3)整数网格坐标 -> (N,)空间编号；编号用于排序，不加入特征。"""
    # trans 路线交换 x/y 轴来改变遍历路线，不改变点的行顺序和实际坐标。
    xyz = grid[:, [1, 0, 2]] if route.endswith("trans") else grid
    # 转 int64，便于按位运算。
    xyz = xyz.to(torch.int64)
    # (N,)编号累加器，初值全 0。
    code = xyz.new_zeros(xyz.shape[0])
    # 所有坐标均为 0 时不需要编码任何位，所有编号都是 0。
    if depth == 0:
        return code
    # Hilbert 路线先变换坐标的位；Z 路线跳过这段，直接交错原坐标的位。
    if route.startswith("hilbert"):
        # 后续会就地修改坐标，先复制以保护输入。
        xyz = xyz.clone()
        # Hilbert编码：依次按各位所在的子立方体交换/翻转坐标，再转为Gray码。
        # 当前处理的二进制位，从最高位开始。
        bit = 1 << (depth - 1)  # 如 depth=3，则 bit=0b100，只选中最高位。
        # 每轮处理一级空间划分，最低位无需再执行这里的方向变换。
        while bit > 1:
            # 依次处理 x、y、z 三轴，每次同时处理所有点。
            for axis in range(3):
                # (N,)该轴当前位是否为 1。
                high = (xyz[:, axis] & bit) != 0
                # (N,)低位交换掩码：该位为 0 时记录与 x 轴低位的差异。
                exchange = torch.where(high, 0, (xyz[:, 0] ^ xyz[:, axis]) & (bit - 1))
                # ^= 是原地异或：当前位为 1 时翻转 x 的低位，否则交换两轴低位。
                xyz[:, 0] ^= torch.where(high, bit - 1, exchange)
                # axis=0 时两轴其实是同一轴，不要把同一份交换再做一次。
                if axis != 0:
                    xyz[:, axis] ^= exchange
            bit >>= 1  # 右移一位，转向下一级更小的子立方体。
        # Gray 码变换：三轴级联异或，再按第三轴累加翻转掩码。
        xyz[:, 1] ^= xyz[:, 0]  # y <- y XOR x。
        xyz[:, 2] ^= xyz[:, 1]  # z <- z XOR 已更新的 y。
        # (N,)翻转掩码，初值全 0。
        flip = code.clone()
        bit = 1 << (depth - 1)  # 重新从最高位构造各点的翻转掩码。
        while bit > 1:
            # z 的当前位为 1，就把该位以下的掩码异或到 flip；否则异或 0。
            flip ^= torch.where((xyz[:, 2] & bit) != 0, bit - 1, 0)
            bit >>= 1  # 继续检查下一位。
        # flip[:,None] 是 (N,1)，同一个点的三轴共用其翻转掩码。
        xyz ^= flip[:, None]
    # Z序直接交错x/y/z的位；Hilbert序交错变换后的位。结果均为(N,)。
    for bit in range(depth):  # 从每轴最低位开始，依次提取 depth 个二进制位。
        for axis in range(3):
            # >> bit 再 & 1：提取该轴第 bit 位；左移后放到总编码的对应位置。
            # 每层占 3 位，x/y/z 分别放在 3*bit+2、3*bit+1、3*bit。
            # |= 把这一位写入 code，保留此前写入的其他位。
            code |= ((xyz[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return code  # (N,)，这里仅生成编号；调用方 argsort 后才会得到点的遍历顺序。


class SparseConv(nn.Module):
    """子流形稀疏卷积：读取相邻网格里的特征，保持原点数和坐标顺序。"""

    def __init__(self, cin: int, cout: int, kernel: int, bias: bool = True) -> None:
        # 初始化 nn.Module，使其能够登记参数和子模块。
        super().__init__()
        self.kernel = kernel  # 三个轴各有 kernel 个采样位置，本例使用奇数大小。
        # 权重形状：(输出通道,K,K,K,输入通道)。
        self.weight = nn.Parameter(torch.empty(cout, kernel, kernel, kernel, cin))
        # 每个输出通道一个偏置；bias=False 时不创建偏置参数。
        self.bias = nn.Parameter(torch.empty(cout)) if bias else None
        # 均匀初始化上界，1/sqrt(扇入)。
        bound = 1 / math.sqrt(cin * kernel**3)
        nn.init.uniform_(self.weight, -bound, bound)  # 用均匀分布填入初始权重。
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)  # 启用偏置时使用相同初始化范围。

    def forward(self, x: Tensor, grid: Tensor) -> Tensor:
        # x:(N,Cin)，grid:(N,3)。输出始终为(N,Cout)。
        # (3,)，每轴最大坐标加 1，得到容纳这些非负坐标的网格边长。
        size = grid.amax(0).to(torch.int64) + 1

        def key(xyz: Tensor) -> Tensor:
            # 把三维坐标展开为一维键：(x*Y+y)*Z+z；范围内的不同体素键不同。
            # 这是查邻居用的键，与前面的 Z/Hilbert 空间编码无关。
            return (xyz[:, 0] * size[1] + xyz[:, 1]) * size[2] + xyz[:, 2]

        # keys：(N,)升序的键；sorted_rows：这些键对应的输入行号。
        # 只给查找表排序，不把输入 x/grid 的行顺序改掉。
        keys, sorted_rows = key(grid).sort()
        # 核中心下标，例如 kernel=3 时中心是 (1,1,1)。
        center = self.kernel // 2
        # 中心位置就是点自身，先算所有点自身的贡献：(N,Cin) @ (Cin,Cout)。
        out = x @ self.weight[:, center, center, center, :].T
        # product 枚举卷积核的所有 (i,j,k)，共 kernel³ 个相对位置。
        for i, j, k in product(range(self.kernel), repeat=3):
            if (i, j, k) == (center, center, center):
                continue  # 中心已经计算过，不再重复累加。
            # (N,3)当前偏移方向的邻居坐标。
            neighbor = grid + grid.new_tensor([i - center, j - center, k - center])
            # (N,)布尔；all(1) 要求邻居的三个坐标都在 [0,size) 内。
            valid = ((neighbor >= 0) & (neighbor < size)).all(1)
            # (N,)邻居点的键。
            neighbor_keys = key(neighbor)
            # (N,)，二分查找邻居键的插入位置；插入位置不代表邻居一定存在。
            # 位置可能等于 N，先截到 N-1 防止后面 keys[found] 越界。
            found = torch.searchsorted(keys, neighbor_keys).clamp_max(len(keys) - 1)
            # 只有坐标合法且键恰好匹配才算找到邻居；where(...)[0] 取行号。
            # rows 的长度是本方向实际找到邻居的点数，不一定是 N。
            rows = torch.where(valid & (keys[found] == neighbor_keys))[0]
            # found 是排序表位置，经 sorted_rows 换回邻居在输入 x 中的行号。
            source = sorted_rows[found[rows]]
            # 按当前核位置的权重变换邻居特征，得到 (邻居数,Cout)。
            contribution = x[source] @ self.weight[:, i, j, k, :].T
            # 沿第 0 维，把 contribution 加到 out 的 rows 行；无邻居就不贡献。
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
        self.patch_size = 1024  # 一个注意力组最多 1024 点，不是整片点云的点数上限。
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
        hidden_channels = channels * 4  # MLP 的中间通道数是输入的 4 倍。
        # MLP 升维。
        self.fc1 = nn.Linear(channels, hidden_channels)
        # MLP 降维。
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, x: Tensor, grid: Tensor, order: Tensor) -> Tensor:
        # x:(N,C)特征，grid:(N,3)网格坐标，order:(N,)某条路线的排序索引。
        # 点数、通道数。
        n, c = x.shape
        # 将 C 个通道拆成 H 个头，每头 D=C/H 维；当前配置每头都是 16 维。
        h, d = self.heads, c // self.heads
        # 1. CPE：卷积、线性层、LayerNorm都保持(N,C)，再与原特征相加。
        # grid 给出每行特征所在的体素，卷积据此查找空间邻居；输出仍按原行排列。
        position = self.cpe_conv(x, grid)
        # 第一条残差，(N,C)。
        x = x + self.cpe_norm(self.cpe_linear(position))

        # 2. 每点先归一化并生成 Q/K/V，再用 order 把行排成空间路线顺序。
        # order[j]=i 表示序列第 j 个位置放输入的第 i 个点，结果为 (N,3C)。
        qkv = self.qkv(self.norm1(x))[order]
        # split 沿第 0 维切分，返回张量元组；最后一组可能不足 1024 点。
        chunks = qkv.split(self.patch_size)
        # 收集每组真实点的输出；之后沿点维拼回整片点云。
        outputs = []
        # enumerate 同时给出组号与组张量；number 是组号，当前实现未使用它。
        for number, chunk in enumerate(chunks):
            # 本组真实点数（不含借点）。
            real_length = len(chunk)
            # 只有多于一组且末组不满时才补齐；整片点云不足 1024 点则直接算。
            if len(chunks) > 1 and real_length < self.patch_size:
                # chunks[-2] 是倒数第二组，完整含 1024 点；[r:] 取它的后 1024-r 点。
                # 真实点放前，借点放后，因此后面 mixed[:real_length] 能直接去掉借点。
                chunk = torch.cat((chunk, chunks[-2][real_length:]), dim=0)
            # 借点后的组长度。
            length = len(chunk)
            # (L,3C) -> (L,3,H,D) -> (3,H,L,D)，再沿第 0 维拆成 Q、K、V。
            # unbind(0) 去掉大小为 3 的那一维，返回三个 (H,L,D) 张量。
            q, k, v = chunk.reshape(length, 3, h, d).permute(1, 2, 0, 3).unbind(0)
            # Q/K/V各为(H,L,D)，注意力权重为(H,L,L)。
            # SDPA 计算 softmax(QK转置/sqrt(D))V，组内点相互注意，输出 (H,L,D)。
            # dropout_p=0.0 表示不随机丢弃注意力权重；借来的点也参与注意力计算。
            mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            # (H,L,D) -> (L,H,D) -> (L,C)，把所有头的输出拼回通道维。
            mixed = mixed.transpose(0, 1).reshape(length, c)
            # 去掉借来的行，保留本组真实点。
            outputs.append(mixed[:real_length])
        # 各组按切分顺序拼回 (N,C)，此时仍是 order 指定的空间路线顺序。
        mixed = torch.cat(outputs, dim=0)
        # argsort 得到 order 的逆排列，恢复进入本 Block 时 x 的行顺序。
        # 例：order=[2,0,1]，逆排列=[1,2,0]，将 [点2,点0,点1] 还原为 [点0,点1,点2]。
        mixed = mixed[order.argsort()]
        # 第二条残差，(N,C)。
        x = x + self.attention_projection(mixed)

        # 3. MLP逐点改变通道：(N,C) -> (N,4C) -> (N,C)，再加残差。
        # 先 LayerNorm，再升到 4C 维并经过 GELU 激活，得到 (N,4C)。
        hidden = F.gelu(self.fc1(self.norm2(x)))
        # 第三条残差，(N,C)。
        x = x + self.fc2(hidden)
        return x


class PTv3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 6  # 每个输入点必须有 6 维特征，与 feat 的第二维一致。
        # 编码器各阶段的通道数、头数、块数。
        enc_channels = (32, 64, 128, 256, 512)  # stage 0~4 各自的输出通道数。
        enc_heads = (2, 4, 8, 16, 32)  # 与各阶段通道数对应，每头都是 16 维。
        enc_depths = (2, 2, 2, 6, 2)  # 各阶段包含几个 Block，与空间编码 depth 无关。
        # 5×5×5 稀疏卷积把 6 维输入嵌入为 32 维特征，不增减点。
        self.embedding_conv = SparseConv(
            self.in_channels, enc_channels[0], kernel=5, bias=False
        )
        # 对 (N,32) 的每个通道做 BN；eps 防止除零，momentum 控制运行统计量更新。
        self.embedding_norm = nn.BatchNorm1d(enc_channels[0], eps=1e-3, momentum=0.01)
        # ModuleList登记参数；阶段列表引用同一批Block，供前向直接遍历。
        self.encoder = nn.ModuleList()
        self.enc_stages: list[list[Block]] = []
        # zip 将同一阶段的配置配在一起；strict=True 要求三个配置长度相同。
        for c, h, depth in zip(enc_channels, enc_heads, enc_depths, strict=True):
            # 该阶段创建 depth 个独立 Block，各自有自己的权重。
            blocks = [Block(c, h) for _ in range(depth)]
            # 登记到模型，参与保存和设备迁移。
            self.encoder.append(nn.ModuleList(blocks))
            self.enc_stages.append(blocks)  # 普通列表方便 forward 按阶段访问。
        # pairwise 取相邻通道对：32->64、64->128、128->256、256->512。
        # 四次下采样分别使用自己的线性层，先变通道，再按粗体素聚合。
        self.down_linear = nn.ModuleList(
            nn.Linear(cin, cout) for cin, cout in pairwise(enc_channels)
        )
        # 四次下采样聚合后，分别对 64、128、256、512 维特征做 BN。
        self.down_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in enc_channels[1:]
        )
        # 解码器配置按细->粗列出，实际前向按 stage 3、2、1、0 执行。
        dec_channels = (64, 64, 128, 256)
        dec_heads = (4, 4, 8, 16)
        dec_depths = (2, 2, 2, 2)
        # 粗点分支的通道投影；按 stage 0~3 存储为 64->64、128->64、256->128、512->256。
        # stage 3 输出 256 维，正好作为下一次 stage 2 的粗点输入，依次类推。
        self.up_low_linear = nn.ModuleList(
            nn.Linear(cin, cout)
            for cin, cout in zip(enc_channels[1:], dec_channels, strict=True)
        )
        # 细点跳连分支的通道投影：32->64、64->64、128->128、256->256。
        # 两个分支在同一 stage 投影到相同维度，才能逐元素相加。
        self.up_skip_linear = nn.ModuleList(
            nn.Linear(cin, cout)
            for cin, cout in zip(enc_channels[:-1], dec_channels, strict=True)
        )
        # 粗点分支在线性投影后做 BN，各阶段统计量独立。
        self.up_low_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in dec_channels
        )
        # 细点跳连分支也有独立 BN，不与粗点分支共用统计量。
        self.up_skip_norm = nn.ModuleList(
            nn.BatchNorm1d(c, eps=1e-3, momentum=0.01) for c in dec_channels
        )
        # 与编码器一样，ModuleList 登记参数，普通阶段列表供 forward 遍历。
        self.decoder = nn.ModuleList()
        self.dec_stages: list[list[Block]] = []
        for c, h, depth in zip(dec_channels, dec_heads, dec_depths, strict=True):
            # 按该阶段的通道数、头数创建 depth 个解码 Block。
            blocks = [Block(c, h) for _ in range(depth)]
            self.decoder.append(nn.ModuleList(blocks))  # 登记解码器参数。
            self.dec_stages.append(blocks)  # 保存同一批 Block 的引用。
        # 构造后默认进入 eval 模式，BN 使用运行均值/方差；这不等于加载预训练权重。
        self.eval()

    # 整个 forward 禁用梯度记录和部分自动求导开销，用于推理演示。
    # inference_mode 控制梯度，eval 控制 BN 等模块的行为，两者作用不同。
    @torch.inference_mode()
    def forward(self, data: InputData) -> OutputData:
        # 单片点云：N个点，feat(N,6)，coord(N,3)，grid_coord(N,3)。
        # x：(N,6)逐点特征；coord：(N,3)连续坐标。
        x, coord = data["feat"], data["coord"]
        # (N,3)整数网格坐标。
        grid = data["grid_coord"]
        # 取三轴中的最大整数坐标，其 bit_length 作为三轴统一的编码深度。
        # 如最大坐标 31=0b11111，则 depth=5；item() 把单值张量转为 Python 数值。
        depth = int(grid.max().item()).bit_length()
        # 四种序列化路线。
        names = ["z", "z-trans", "hilbert", "hilbert-trans"]
        # 四种(N,)空间编码叠成(4,N)。
        codes = torch.stack([spatial_code(grid, depth, name) for name in names])
        # randperm(4) 生成 0~3 的随机排列，只打乱路线次序，不打乱点的行顺序。
        permutation = torch.randperm(len(names))
        # codes 重排第 0 维，names 用相同排列重排，确保名称与编码仍对应。
        codes, names = codes[permutation], [names[i] for i in permutation.tolist()]

        # 输入嵌入：(N,6) -> (N,32)，点数不变。
        x = F.gelu(self.embedding_norm(self.embedding_conv(x, grid)))
        # 保存4次下采样之前的数据，之后从后往前恢复。
        parents: list[_Parent] = []

        # 编码器：N0/32 -> N1/64 -> N2/128 -> N3/256 -> N4/512。
        # Ni由粗网格内实际有多少组决定，不能直接按点数除以8计算。
        for stage, blocks in enumerate(self.enc_stages):
            # stage 0 直接处理嵌入后的细点；stage 1~4 先下采样，再运行该级 Block。
            if stage > 0:
                # 各次下采样步长均为2：右移1位；深度为0时不再缩小。
                shift = 1 if depth > 0 else 0
                # (4,N)，每级空间划分占 3 位，所以右移 3*shift 位。
                # 得到即将进入的更粗 stage 的编码；shift=1 时把 2×2×2 个体素合为一格。
                coarse_codes = codes >> (3 * shift)
                # 按粗编码分组：相同粗编码的细点将合成一个粗点。
                # unique 返回：去重后的编码、每个细点的组号、每组点数；第一个结果忽略。
                _, cluster, counts = torch.unique(
                    coarse_codes[0],  # 第一条路线的全部点；一条路线足以分组。
                    sorted=True,  # 去重编码按升序排列，组号 0~M-1 对应此顺序。
                    return_inverse=True,  # cluster:(N,)，给每个原行标注所属组号。
                    return_counts=True,  # counts:(M,)，按组号统计各组含几个细点。
                )
                # cluster(N,)给出每个细点的组号；counts(M,)给出M个粗点各含几个细点。
                # 例：粗编码 [5,2,5,7,2] -> cluster=[1,0,1,2,0]，counts=[2,2,1]。
                # 后续聚合结果按组号排列，即粗点行序由本次第一条路线的粗编码升序决定。
                # 先缓存当前细层；后续变量 x/coord/grid 会改为粗层数据。
                parents.append(
                    {
                        "feat": x,  # 保存细层特征的引用，解码时用于跳连。
                        "coord": coord,  # 保存细点连续坐标。
                        "grid": grid,  # 保存细点体素坐标。
                        "codes": codes,  # 保存细层四条路线的编码。
                        "names": names,  # 保存此时的路线次序。
                        "depth": depth,  # 保存细层空间编码深度。
                        "inverse": cluster,  # 细点 -> 粗点行号，供解码时复制粗特征。
                    }
                )
                # (N,)，按组号排序得到细点行号，使第 0、1、2...组依次连续排列。
                # 这里只要求同组连续，不依赖同组内部的具体点序。
                sorted_rows = cluster.argsort()
                # 每组的结束位置减去组大小，就是该组在排序后的起始位置。
                # 如 counts=[2,2,1]，cumsum(0)=[2,4,5]，相减得到起点 [0,2,4]。
                group_starts = counts.cumsum(0) - counts
                # (M,)，取每组首个点的原行号；代表点只用于提取体素坐标和编码。
                # 粗点特征和连续坐标仍由整组聚合，不直接使用代表点的特征/坐标。
                first = sorted_rows[group_starts]
                # stage 1 使用第 0 个下采样层，以此类推；先逐点投影 (N,Cin)->(N,Cout)。
                projected = self.down_linear[stage - 1](x)
                # 按 sorted_rows 排好组，再按 counts 指定的长度逐段取通道最大值。
                # (N,Cout)->(M,Cout)，一个粗点的不同通道可以来自组内不同细点。
                x = torch.segment_reduce(projected[sorted_rows], "max", lengths=counts)
                # (M,Cout)，BN+GELU。
                x = F.gelu(self.down_norm[stage - 1](x))
                # 特征取组内最大值；连续坐标取组内均值，保持同一级点数。
                # 用同一分组，对 x/y/z 分别求平均：(N,3)->(M,3)。
                coord = torch.segment_reduce(coord[sorted_rows], "mean", lengths=counts)
                # (M,3)，代表点各轴右移 shift 位，即整数除以 2**shift 向下取整。
                # 同组细点右移后的体素坐标相同，所以任取一个代表即可。
                grid = grid[first] >> shift
                # 每组只保留一列粗编码，得到 (4,M)，同时减少每轴编码位数。
                codes, depth = coarse_codes[:, first], depth - shift
                # 到新分辨率后再次打乱四条路线的使用次序，点的行顺序仍不变。
                permutation = torch.randperm(len(names))
                codes, names = (
                    codes[permutation],  # 重排四行编码，不重排 M 列粗点。
                    [names[i] for i in permutation.tolist()],  # 路线名称同步重排。
                )
            # 本阶段不再改变点数；依次运行该阶段所有 Block。
            for number, block in enumerate(blocks):
                # 块号对路线数取余，按 0,1,2,3,0,1...轮换；每个阶段重新从 0 开始。
                route = number % len(names)
                # argsort 得到该路线的点排列；Block 内部排序做注意力，返回前恢复行顺序。
                x = block(x, grid, codes[route].argsort())

        # 解码器：按 stage 3->2->1->0，从最粗层依次恢复到输入点数。
        # low 是当前粗点分支，skip 是下采样前保存的细点分支，先统一通道再融合。
        for stage in reversed(range(len(self.dec_stages))):
            # 粗点先经过 Linear、BN、GELU，得到 (M,Cout)，此时尚未恢复点数。
            low = F.gelu(self.up_low_norm[stage](self.up_low_linear[stage](x)))
            # pop 取出并移除最后一次缓存，正好对应本次要恢复的细层。
            parent = parents.pop()
            # 下采样前的细点特征也经过 Linear、BN、GELU，得到 (N,Cout)。
            skip = F.gelu(
                self.up_skip_norm[stage](self.up_skip_linear[stage](parent["feat"]))
            )
            # inverse[i]=j 表示细点 i 来自粗点 j，因此 low[inverse] 按细点行序复制特征。
            # 如 inverse=[1,0,1]，则取 [low[1],low[0],low[1]]，从 M 行扩展为 N 行。
            # 同组细点拿到相同粗特征，再加各自的 skip，得到 (N,Cout)。
            x = skip + low[parent["inverse"]]
            # 点数恢复后，坐标也取回该细层的原值，而不是从粗坐标估计。
            coord, grid = parent["coord"], parent["grid"]
            # 同步恢复细层编码、路线次序和深度，保证后续注意力与细点对应。
            codes, names, depth = parent["codes"], parent["names"], parent["depth"]
            # 用该细分辨率的解码 Block 进一步融合特征。
            for number, block in enumerate(self.dec_stages[stage]):
                route = number % len(names)  # 与编码器一样，逐块轮换序列化路线。
                # 在恢复后的细点上做注意力，Block 输出仍保持进入时的细点行序。
                x = block(x, grid, codes[route].argsort())

        # feat(N0,64)，恢复原点数、原点顺序和坐标；这是主干特征，尚未接预测头。
        # (4,N0)，四种序列位置到原点编号的映射。
        order = codes.argsort(dim=1)
        return {
            **data,  # 先展开输入字典，后面的同名键覆盖输入值。
            "feat": x,  # (N0,64)，替换输入的 6 维特征，不是分类/分割预测结果。
            "coord": coord,  # (N0,3)，输入层连续坐标。
            "grid_coord": grid,  # (N0,3)，输入层体素坐标。
            # (N0,)，此实现用全 1 标记同一片点云，没有在 forward 中按 batch 分组。
            "batch": torch.ones(len(x), dtype=torch.int64, device=x.device),
            "serialized_code": codes,  # (4,N0)，每个输入点在各路线上的空间编码。
            "serialized_order": order,  # (4,N0)，沿每行索引特征即可排成对应路线。
            "serialized_inverse": order.argsort(dim=1),  # (4,N0)，各路线排序的逆排列。
            "serialized_depth": depth,  # 恢复后的输入层编码深度。
        }


def main() -> None:
    # 运算设备：演示在 CPU 上跑。
    device = "cpu"
    # 模拟点云的点数。
    n = 88
    # randperm 随机排列 0~32³-1，再取前 n 个，保证采样的体素互不重复。
    flat = torch.randperm(32**3, device=device)[:n]
    # 展开下标满足 i=x*1024+y*32+z，用整除和取余分别还原三个坐标。
    # stack(dim=1) 把三个 (88,) 张量作为三列组成 (88,3)，最后转为 int32。
    grid = torch.stack((flat // 1024, flat // 32 % 32, flat % 32), dim=1).to(
        torch.int32
    )
    data: InputData = {  # 按 forward 的输入约定，组织同一批点的特征和两类坐标。
        # (88,6)，各通道是 [0,1) 的随机数，只用于演示形状，并无实际属性含义。
        "feat": torch.rand(n, 6, device=device),
        # (88,3)，转成浮点后乘 0.02，模拟边长为 0.02 的体素所对应的连续坐标。
        "coord": grid.to(torch.float32) * 0.02,
        # (88,3)，整数网格坐标。
        "grid_coord": grid,
    }
    # 构建随机初始化的模型并搬到 CPU；构造函数已将其设为 eval 模式。
    model = PTv3().to(device)
    # 前向一次，得到输出字典。
    output = model(data)
    # 最终应为 (88,64)，点数和行序恢复到输入，通道数变为解码器 stage 0 的 64。
    print(f"\n最终逐点特征： {tuple(output['feat'].shape)}")
    # numel 返回单个参数张量的元素数，遍历全部参数后求和得到模型参数总数。
    print("参数总数：", sum(parameter.numel() for parameter in model.parameters()))
    # 显示实际运行时的 Python 与 PyTorch 版本，便于复现实验环境。
    print(f"运行版本：Python {sys.version.split()[0]} PyTorch {torch.__version__}")


if __name__ == "__main__":  # 直接运行本文件时执行演示；被其他文件导入时不执行。
    main()
