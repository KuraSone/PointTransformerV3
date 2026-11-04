"""PointFlow 自编码重建：将输入点集编码为形状条件，再从高斯噪声采样重建点集。"""

from collections.abc import Callable
from itertools import pairwise

import torch
from torch import Tensor, nn
from torchdiffeq import odeint

# 三个积分状态依次为：点位置 [B,N,3]、累计密度变化 [B,N,1]、形状码 [B,128]。
# 这里只累计 CNF 内的对数密度变化，没有计算完整点云的概率或训练损失。
type FlowState = tuple[Tensor, Tensor, Tensor]


class Encoder(nn.Module):
    """将一个点云编码为一个 128 维向量，推理时编码结果不含随机采样。"""

    def __init__(self) -> None:
        super().__init__()
        # 1×1 卷积在所有点上共享参数，逐点通道依次为 3→128→128→256→512。
        self.conv1: nn.Conv1d = nn.Conv1d(3, 128, kernel_size=1)
        self.conv2: nn.Conv1d = nn.Conv1d(128, 128, kernel_size=1)
        self.conv3: nn.Conv1d = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4: nn.Conv1d = nn.Conv1d(256, 512, kernel_size=1)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.bn4: nn.BatchNorm1d = nn.BatchNorm1d(512)
        self.fc1: nn.Linear = nn.Linear(512, 256)
        self.fc2: nn.Linear = nn.Linear(256, 128)
        self.fc_bn1: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.fc_bn2: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.fc3: nn.Linear = nn.Linear(128, 128)

    def forward(self, input_points: Tensor) -> Tensor:
        """将 [B,N_in,3] 坐标编码为 [B,128] 形状码，N_in>=1。"""
        # input_points[b,i,:] 是点 i 的 xyz；mT 交换最后两维，得到 [B,3,N_in]。
        # Conv1d 的中间维是通道，最后一维对应点；转置不改变点的排列。
        features: Tensor = input_points.mT
        features = torch.relu(self.bn1(self.conv1(features)))  # [B,128,N_in]
        features = torch.relu(self.bn2(self.conv2(features)))  # [B,128,N_in]
        features = torch.relu(self.bn3(self.conv3(features)))  # [B,256,N_in]
        features = self.bn4(self.conv4(features))  # [B,512,N_in]，此层后没有 ReLU。

        # 每个通道独立在所有点中取最大值，得到整个点云的 [B,512] 描述。
        # 不同通道的最大值可以来自不同点；输入点重排不改变最大值。
        global_features: Tensor = features.amax(dim=-1)
        hidden: Tensor = torch.relu(self.fc_bn1(self.fc1(global_features)))  # [B,256]
        hidden = torch.relu(self.fc_bn2(self.fc2(hidden)))  # [B,128]
        shape_code: Tensor = self.fc3(hidden)  # [B,128]，最后一层无激活。
        return shape_code


class ConcatSquashLinear(nn.Module):
    """用时间和形状码控制线性映射的逐通道门值及平移量。"""

    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.value: nn.Linear = nn.Linear(input_features, output_features)
        # 条件向量为 [t,z]，共 1+128 维；平移分支采用无偏置的线性映射。
        self.shift: nn.Linear = nn.Linear(129, output_features, bias=False)
        self.gate: nn.Linear = nn.Linear(129, output_features)

    def forward(self, context: Tensor, features: Tensor) -> Tensor:
        """将 [B,N,C_in] 映射为 [B,N,C_out]，context 的形状为 [B,129]。"""
        # 同一形状的所有点共享 [t,z]。先得到 [B,C_out] 的门值和平移量，
        # 再插入长度为 1 的点维，广播到各点；没有复制 N 套参数。
        gate: Tensor = torch.sigmoid(self.gate(context)).unsqueeze(1)  # [B,1,C_out]
        shift: Tensor = self.shift(context).unsqueeze(1)  # [B,1,C_out]
        # value 在最后一维做线性映射；* 按元素调节各通道，+ 加上条件平移。
        transformed: Tensor = self.value(features) * gate + shift  # [B,N,C_out]
        return transformed


class VelocityField(nn.Module):
    """条件速度场：3→512→512→512→3，隐藏层使用 tanh。"""

    def __init__(self) -> None:
        super().__init__()
        dimensions: tuple[int, ...] = (3, 512, 512, 512, 3)
        # pairwise 依次取相邻宽度，构造四个条件线性层。
        self.layer_refs: list[ConcatSquashLinear] = [
            ConcatSquashLinear(input_width, output_width)
            for input_width, output_width in pairwise(dimensions)
        ]
        # 两个容器引用同一批层：列表保留具体类型，ModuleList 负责登记参数。
        self.layers: nn.ModuleList = nn.ModuleList(self.layer_refs)

    def forward(self, context: Tensor, position: Tensor) -> Tensor:
        """根据 [B,129] 条件和 [B,N,3] 当前位置，返回同形状的位置导数 dx/dt。"""
        features: Tensor = position
        for index, layer in enumerate(self.layer_refs):
            features = layer(context, features)
            if index < len(self.layer_refs) - 1:
                features = torch.tanh(features)
        # 最后一层输出 xyz 三个方向的速度；实际点坐标由 ODE 积分得到。
        return features


class InverseMovingBatchNorm(nn.Module):
    """使用保存的运行统计量，撤销流中的逐坐标轴归一化和仿射变换。"""

    def __init__(self) -> None:
        super().__init__()
        # weight 保存对数尺度，实际的乘法尺度为 exp(weight)。
        self.weight: nn.Parameter = nn.Parameter(torch.zeros(3))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(3))
        self.running_mean: Tensor
        self.running_var: Tensor
        # buffer 会随模型保存及迁移 dtype/设备；推理中不更新这些统计量。
        self.register_buffer("running_mean", torch.zeros(3))
        self.register_buffer("running_var", torch.ones(3))

    def forward(self, normalized: Tensor) -> Tensor:
        """将 [B,N,3] 的归一化结果还原为进入该归一化层之前的坐标。"""
        # 正变换为 y=(x-mean)/sqrt(var+eps)*exp(weight)+bias。
        # 先撤销平移和仿射尺度；[3] 参数沿批次维、点维广播。
        unscaled: Tensor = (normalized - self.bias) * torch.exp(-self.weight)
        # 再乘 sqrt(var+eps) 并加回均值。这里保留源码 exp(0.5*log(...))
        # 的浮点计算路径和 eps=1e-4，以便与相同权重的原实现对照。
        return (
            unscaled * torch.exp(0.5 * torch.log(self.running_var + 1e-4))
            + self.running_mean
        )


class InversePointFlow(nn.Module):
    """以固定形状码为条件，将三维高斯点噪声变换为重建点坐标。"""

    def __init__(self) -> None:
        super().__init__()
        self.field: VelocityField = VelocityField()
        # T=sqrt_end_time² 是可学习的积分终点，初始 T=0.5；使用时须 T>0。
        self.sqrt_end_time: nn.Parameter = nn.Parameter(torch.tensor(0.5).sqrt())
        self.before: InverseMovingBatchNorm = InverseMovingBatchNorm()
        self.after: InverseMovingBatchNorm = InverseMovingBatchNorm()

    def forward(self, point_noise: Tensor, shape_code: Tensor) -> Tensor:
        """把 [B,N,3] 点噪声解码为同形状点云，形状条件为 [B,128]。

        输出第 i 个点对应 point_noise 的第 i 行在流中的搬运结果。
        每次调用都会抽取新的散度估计噪声；请在 torch.no_grad() 中调用。
        """
        # 密度建模时的链为 before→CNF→after，因此采样先撤销 after。
        position: Tensor = self.after(point_noise)  # [B,N,3]
        # 这个零初值只用于累计 CNF 中的密度变化，不是点的初始对数概率。
        # 保留批次/点维，将坐标维换成 1；new_zeros 沿用 position 的 dtype 和设备。
        density_change: Tensor = position.new_zeros((*position.shape[:-1], 1))
        # epsilon 与位置同形状，每个点有三个标准高斯分量。
        # 它在本次积分的所有求解器试探中保持固定，避免右端函数反复随机变化。
        epsilon: Tensor = torch.randn_like(position)
        initial_state: FlowState = (position, density_change, shape_code)

        def dynamics(time: Tensor, state: FlowState) -> FlowState:
            """返回点位置、累计密度变化、形状码这三个状态各自对时间的导数。"""
            current: Tensor = state[0]  # [B,N,3]，求解器本次试探的点位置。
            # 标量时间先对齐位置的 dtype/设备，再广播为每个样本一列 [B,1]。
            time_column: Tensor = time.to(current).expand(current.shape[0], 1)
            context: Tensor = torch.cat((time_column, state[2]), dim=-1)  # [B,129]

            def velocity_at(current_position: Tensor) -> Tensor:
                """固定本次的时间和形状码，仅把点位置作为求导输入。"""
                current_velocity: Tensor = self.field(context, current_position)
                return current_velocity

            # vjp 内部调用 velocity_at(current)，同时返回速度和一个求导函数。
            # J=∂velocity/∂current；pullback(epsilon) 给出 Jᵀ epsilon。
            vjp_result = torch.func.vjp(velocity_at, current)
            velocity: Tensor = vjp_result[0]  # [B,N,3]
            pullback: Callable[[Tensor], tuple[Tensor]] = vjp_result[1]
            # 这里只有 current 一个求导输入，因此返回元组的第 0 项就是它的导数。
            input_cotangent: Tensor = pullback(epsilon)[0]  # [B,N,3]

            # 散度 tr(J) 描述局部体积的膨胀或收缩。标准高斯 epsilon 满足
            # E[epsilon·epsilonᵀ]=I，所以 epsilonᵀJ epsilon 是它的无偏估计。
            # 各点独立经过速度网络，仅沿 xyz 求和，得到每点一个估计值 [B,N,1]。
            divergence: Tensor = (input_cotangent * epsilon).sum(dim=-1, keepdim=True)
            # d(log p)/dt=-divergence；形状码固定，其时间导数为零。
            # 时间反向由下面的 [T,0] 指定，这里仍使用原微分方程的负号。
            return velocity, -divergence, torch.zeros_like(state[2])

        times: Tensor = torch.stack(
            (self.sqrt_end_time.square(), position.new_zeros(()))
        ).to(position)  # [2]，依次请求 t=T 和 t=0 的状态。
        # Dopri5 在内部选择自适应步长；密度和形状状态也参与误差控制，
        # 因此保留完整三元状态，即使此处最终只取坐标。
        trajectory: FlowState = odeint(
            dynamics,
            initial_state,
            times,
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
        )
        # 第一个元组分量是位置轨迹 [2,B,N,3]；[-1] 取终点 t=0 的所有点。
        transported: Tensor = trajectory[0][-1]
        restored: Tensor = self.before(transported)
        return restored


class PointFlow(nn.Module):
    """确定性点云编码器和条件点流组成的自编码重建模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder: Encoder = Encoder()
        self.point_cnf: InversePointFlow = InversePointFlow()

    def forward(self, input_points: Tensor, num_points: int) -> Tensor:
        """根据输入点云的形状，重新采样指定数量的点。

        参数:
            input_points: 浮点坐标 [B,N_in,3]，B>=1、N_in>=1。
                坐标须有限，并采用与训练时相同的数据归一化方式。
                张量的设备和 dtype 须与模型参数一致。
            num_points: 每个重建点云的输出点数 N_out，须为正整数。

        返回:
            坐标张量 [B,N_out,3]，与输入处于同一归一化空间。
            输出点由新噪声产生，与输入点没有逐行对应关系。

        推理前调用 eval()，外部使用 torch.no_grad()；散度内部仍会计算局部位置导数。
        """
        shape_code: Tensor = self.encoder(input_points)  # [B,128]，每个点云一个条件。
        # 编码没有随机采样；解码从新的标准高斯点噪声开始，仍然是随机过程。
        point_noise: Tensor = torch.randn(
            (shape_code.shape[0], num_points, 3),
            dtype=shape_code.dtype,
            device=shape_code.device,
        )
        reconstructed: Tensor = self.point_cnf(point_noise, shape_code)
        return reconstructed


def main() -> None:
    """在 CPU 上用一个 [1,32,3] 模拟点云执行重建。

    模型随机初始化，不加载训练权重；输出 [1,32,3] 保存在 _reconstructed 中。
    """
    torch.manual_seed(7)
    model: PointFlow = PointFlow().to(device="cpu", dtype=torch.float32)
    # eval 决定编码器 BatchNorm 的统计量来源；本例使用保存的运行统计量。
    model.eval()
    input_points: Tensor = (
        torch.randn((1, 32, 3), device="cpu", dtype=torch.float32) * 0.25
    )
    # no_grad 不记录整条前向的训练图；vjp 会为局部位置导数临时启用微分。
    # inference_mode 会妨碍这里的局部微分，因此该路径使用 no_grad。
    with torch.no_grad():
        _reconstructed: Tensor = model(input_points, num_points=32)


if __name__ == "__main__":
    main()
