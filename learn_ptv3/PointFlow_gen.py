"""PointFlow 点云生成：先从高斯先验生成形状码，再按该形状采样三维点。"""

from collections.abc import Callable
from itertools import pairwise

import torch
from torch import Tensor, nn
from torchdiffeq import odeint

# 潜变量流有两个状态：(形状向量 [B,128]，累计密度变化 [B,1])。
# 点流另带固定的形状条件：(点位置 [B,N,3]，累计密度变化 [B,N,1]，形状码 [B,128])。
type FlowState = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]


class ConcatSquashLinear(nn.Module):
    """用条件向量控制线性映射的逐通道门值及平移量。"""

    def __init__(
        self, input_features: int, output_features: int, context_features: int
    ) -> None:
        super().__init__()
        self.value: nn.Linear = nn.Linear(input_features, output_features)
        # context_features 包含时间分量：潜变量流为 1，点流为 1+128。
        self.shift: nn.Linear = nn.Linear(context_features, output_features, bias=False)
        self.gate: nn.Linear = nn.Linear(context_features, output_features)

    def forward(self, context: Tensor, features: Tensor) -> Tensor:
        """按 [B,K] 条件映射最后一维，支持潜变量 [B,C_in] 和点特征 [B,N,C_in]。

        K=1 时条件只有时间；K=129 时第 0 列为时间，后 128 列为形状码。
        输出仅将 C_in 改为 C_out，其他维度及其索引对应关系不变。
        """
        gate: Tensor = torch.sigmoid(self.gate(context))  # [B,C_out]，取值在 (0,1)。
        shift: Tensor = self.shift(context)  # [B,C_out]
        if features.ndim == 3:
            # 点流插入长度为 1 的点维，向 N 个点广播同一组条件。
            # 潜变量流只有 [B,C_in] 两维，直接使用上面的 [B,C_out] 门值和平移量。
            gate = gate.unsqueeze(1)  # [B,1,C_out]
            shift = shift.unsqueeze(1)
        # value 改变特征通道，* 和 + 逐元素调节；批次维和点维不变。
        transformed: Tensor = self.value(features) * gate + shift
        return transformed


class VelocityField(nn.Module):
    """条件速度网络，输入和输出的末维相同，隐藏层使用 tanh。"""

    def __init__(
        self, dimension: int, hidden_dimensions: tuple[int, ...], context_features: int
    ) -> None:
        super().__init__()
        # 两端宽度均为 dimension，保证位置和速度具有相同形状。
        dimensions: tuple[int, ...] = (dimension, *hidden_dimensions, dimension)
        # pairwise 将相邻宽度配成各层的输入/输出，例如 (128,256,256,128) 生成三层。
        self.layer_refs: list[ConcatSquashLinear] = [
            ConcatSquashLinear(input_width, output_width, context_features)
            for input_width, output_width in pairwise(dimensions)
        ]
        # 两个容器引用同一批层：列表保留具体类型，ModuleList 负责登记参数。
        self.layers: nn.ModuleList = nn.ModuleList(self.layer_refs)

    def forward(self, context: Tensor, position: Tensor) -> Tensor:
        """计算当前状态的时间导数，输出形状与 position 相同。

        潜变量流输入 [B,128] 状态与 [B,1] 时间；点流输入 [B,N,3] 坐标
        与 [B,129] 的时间及形状条件。末层输出是速度，由外部求解器进行积分。
        """
        features: Tensor = position
        for index, layer in enumerate(self.layer_refs):
            features = layer(context, features)
            if index < len(self.layer_refs) - 1:
                features = torch.tanh(features)
        # 最后一层无激活，允许每个分量的速度为任意正数或负数。
        return features


class InverseMovingBatchNorm(nn.Module):
    """使用保存的运行统计量，逐分量撤销流中的归一化和仿射变换。"""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        # 每个潜变量分量或坐标轴各有一组参数；weight 保存对数尺度。
        self.weight: nn.Parameter = nn.Parameter(torch.zeros(dimension))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(dimension))
        self.running_mean: Tensor
        self.running_var: Tensor
        # buffer 会随模型保存及迁移 dtype/设备；推理中不更新这些统计量。
        self.register_buffer("running_mean", torch.zeros(dimension))
        self.register_buffer("running_var", torch.ones(dimension))

    def forward(self, normalized: Tensor) -> Tensor:
        """还原 [B,D] 或 [B,N,D] 的归一化结果，仅对最后一维做独立仿射变换。"""
        # 正变换为 y=(x-mean)/sqrt(var+eps)*exp(weight)+bias。
        # [D] 参数沿前面的批次维、点维广播，先撤销平移和仿射尺度。
        unscaled: Tensor = (normalized - self.bias) * torch.exp(-self.weight)
        # 再乘 sqrt(var+eps) 并加回均值。这里保留源码 exp(0.5*log(...))
        # 的浮点计算路径和 eps=1e-4，以便与相同权重的原实现对照。
        return (
            unscaled * torch.exp(0.5 * torch.log(self.running_var + 1e-4))
            + self.running_mean
        )


class InverseFlow(nn.Module):
    """撤销一个包含起端归一化、连续流、末端归一化的完整流块。"""

    def __init__(
        self, dimension: int, hidden_dimensions: tuple[int, ...], context_features: int
    ) -> None:
        super().__init__()
        self.field: VelocityField = VelocityField(
            dimension, hidden_dimensions, context_features
        )
        # T=sqrt_end_time² 是可学习的积分终点，初始 T=0.5；使用时须 T>0。
        self.sqrt_end_time: nn.Parameter = nn.Parameter(torch.tensor(0.5).sqrt())
        self.before: InverseMovingBatchNorm = InverseMovingBatchNorm(dimension)
        self.after: InverseMovingBatchNorm = InverseMovingBatchNorm(dimension)

    def forward(self, noise: Tensor, shape_code: Tensor | None) -> Tensor:
        """将高斯噪声沿流的逆方向变换，输出形状与 noise 相同。

        潜变量流的 noise 为 [B,128]，shape_code 为 None，条件只有时间。
        点流的 noise 为 [B,N,3]，shape_code 为 [B,128]，条件为 [时间,形状码]。
        每次调用都会抽取新的散度估计噪声；请在 torch.no_grad() 中调用。
        """
        # 密度建模时的链为 before→CNF→after，因此采样先撤销 after。
        position: Tensor = self.after(noise)
        # 初值为零，仅累计本次 CNF 中的密度变化；它不是完整的对数概率。
        # 保留批次/点维，末尾补长度 1；new_zeros 沿用 position 的 dtype 和设备。
        density_change: Tensor = position.new_zeros((*position.shape[:-1], 1))
        # 每个状态分量抽取一个标准高斯分量，形状与 position 相同。
        # 本次积分的所有试探步共用这一份 epsilon，不在 dynamics 内重复采样。
        epsilon: Tensor = torch.randn_like(position)
        initial_state: FlowState
        if shape_code is None:
            initial_state = (position, density_change)
        else:
            initial_state = (position, density_change, shape_code)

        def dynamics(time: Tensor, state: FlowState) -> FlowState:
            """返回每个积分状态的时间导数，保留无条件/条件流各自的状态数量。"""
            current: Tensor = state[0]  # 潜变量为 [B,128]，点位置为 [B,N,3]。
            # 标量时间广播为 [B,1]，并对齐当前位置的 dtype/设备。
            time_column: Tensor = time.to(current).expand(current.shape[0], 1)
            context: Tensor = time_column
            if len(state) == 3:
                context = torch.cat((time_column, state[2]), dim=-1)  # [B,129]

            def velocity_at(current_position: Tensor) -> Tensor:
                """固定本次的时间和形状条件，仅把当前位置作为求导输入。"""
                current_velocity: Tensor = self.field(context, current_position)
                return current_velocity

            # vjp 内部调用 velocity_at(current)，返回速度和一个求导函数。
            # J=∂velocity/∂current；对固定的 epsilon 调用该函数，得到 Jᵀ epsilon。
            vjp_result = torch.func.vjp(velocity_at, current)
            velocity: Tensor = vjp_result[0]
            pullback: Callable[[Tensor], tuple[Tensor]] = vjp_result[1]
            # 只有一个求导输入 current，返回元组的第 0 项就是它的导数。
            input_cotangent: Tensor = pullback(epsilon)[0]

            # epsilonᵀJ epsilon 是散度 tr(J) 的无偏估计，衡量局部体积变化率。
            # 潜变量流沿 128 维求和，得到 [B,1]；点流仅沿 xyz 求和，得到 [B,N,1]。
            # 这里不把不同样本或不同点的密度变化相加。
            divergence: Tensor = (input_cotangent * epsilon).sum(dim=-1, keepdim=True)
            if len(state) == 3:
                # d(log p)/dt=-divergence；形状码固定，其时间导数为零。
                return velocity, -divergence, torch.zeros_like(state[2])
            return velocity, -divergence

        times: Tensor = torch.stack(
            (self.sqrt_end_time.square(), position.new_zeros(()))
        ).to(position)  # [2]，依次请求 t=T 和 t=0 的状态。
        # 逆向由 [T,0] 的时间顺序指定，dynamics 中保留原微分方程的符号。
        # Dopri5 的误差控制包含密度和形状状态，因此保留完整的二元或三元状态。
        trajectory: FlowState = odeint(
            dynamics,
            initial_state,
            times,
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
        )
        # 第一个元组分量是位置轨迹；其时间维长度为 2，[-1] 取 t=0 的终点。
        transported: Tensor = trajectory[0][-1]
        restored: Tensor = self.before(transported)
        return restored


class PointFlow(nn.Module):
    """用潜变量流采样形状码，再用条件点流生成该形状的点云。"""

    def __init__(self) -> None:
        super().__init__()
        # 点流的条件为 1 维时间加 128 维形状码，三层隐藏宽度均为 512。
        self.point_cnf: InverseFlow = InverseFlow(3, (512, 512, 512), 129)
        # 潜变量流只以时间为条件，128 维状态经过两层 256 维隐藏层。
        self.latent_cnf: InverseFlow = InverseFlow(128, (256, 256), 1)

    def forward(self, shape_noise: Tensor, num_points: int) -> tuple[Tensor, Tensor]:
        """从给定的形状高斯噪声生成形状码和三维点集。

        参数:
            shape_noise: 浮点张量 [B,128]，B>=1，每行是一个标准高斯样本。
                张量的设备和 dtype 须与模型参数一致。
            num_points: 每个形状的输出点数 N，须为正整数。

        返回:
            (shape_code, points)：分别为 [B,128] 的形状码与 [B,N,3] 的点坐标。
            points[b,i,:] 来自内部第 i 个点噪声，处于训练数据的归一化空间。

        推理前调用 eval()，外部使用 torch.no_grad()；散度内部仍会计算局部位置导数。
        """
        shape_code: Tensor = self.latent_cnf(shape_noise, None)  # [B,128]
        # 在形状码生成之后采样三维点，保持形状采样和点采样的随机操作顺序。
        point_noise: Tensor = torch.randn(
            (shape_code.shape[0], num_points, 3),
            dtype=shape_code.dtype,
            device=shape_code.device,
        )
        # 同一行形状码控制该样本的全部 N 个点；点噪声各自独立通过同一个速度场。
        points: Tensor = self.point_cnf(point_noise, shape_code)
        return shape_code, points


def main() -> None:
    """在 CPU 上采样一个 32 点形状。

    模型随机初始化，不加载训练权重；_shape_code 为 [1,128]，_points 为 [1,32,3]。
    """
    torch.manual_seed(7)
    model: PointFlow = PointFlow().to(device="cpu", dtype=torch.float32)
    model.eval()
    shape_noise: Tensor = torch.randn((1, 128), device="cpu", dtype=torch.float32)
    _shape_code: Tensor
    _points: Tensor
    # no_grad 不记录整条前向的训练图；vjp 会为局部位置导数临时启用微分。
    # inference_mode 会妨碍这里的局部微分，因此该路径使用 no_grad。
    with torch.no_grad():
        _shape_code, _points = model(shape_noise, num_points=32)


if __name__ == "__main__":
    main()
