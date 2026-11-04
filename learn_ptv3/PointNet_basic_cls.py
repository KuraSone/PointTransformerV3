"""PointNet 基础点云分类：独立映射各点，再取全局最大值完成整个点云的分类。"""

import torch
from torch import Tensor, nn


class PointNetBasic(nn.Module):
    """使用五层逐点卷积和全局最大池化的 40 类分类器。

    每层 1×1 卷积在所有点上共享参数，通道数依次为 3→64→64→64→128→1024，
    每层卷积后接 BatchNorm 和 ReLU，五层卷积均保留点数 N。
    沿点维取最大值，得到每个点云的 1024 维全局特征，再经 512 维隐藏层
    输出 40 个类别分数。
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1: nn.Conv1d = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2: nn.Conv1d = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3: nn.Conv1d = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4: nn.Conv1d = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5: nn.Conv1d = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.bn4: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.bn5: nn.BatchNorm1d = nn.BatchNorm1d(1024)
        self.linear1: nn.Linear = nn.Linear(1024, 512, bias=False)
        self.bn6: nn.BatchNorm1d = nn.BatchNorm1d(512)
        self.linear2: nn.Linear = nn.Linear(512, 40)

    def forward(self, points: Tensor) -> Tensor:
        """计算每个输入点云的 40 个类别分数。

        推理前调用 eval()，使 BatchNorm 使用保存的运行统计量。

        参数:
            points: 浮点坐标张量 [B,3,N]，B>=1，N>=1。
                points[b,:,i] 是第 b 个点云中第 i 个点的 [x,y,z] 坐标。
                坐标须有限，张量的设备和 dtype 须与模型参数一致。

        返回:
            浮点张量 [B,40]，logits[b,c] 是第 b 个点云属于第 c 类的分数。
            全局最大池化仅归约点维，B=1 时仍保留批次维。
            分数未经过 softmax，不是概率。
        """
        point_features: Tensor = torch.relu(self.bn1(self.conv1(points)))
        point_features = torch.relu(self.bn2(self.conv2(point_features)))
        point_features = torch.relu(self.bn3(self.conv3(point_features)))
        point_features = torch.relu(self.bn4(self.conv4(point_features)))
        point_embedding: Tensor = torch.relu(self.bn5(self.conv5(point_features)))

        global_descriptor: Tensor = point_embedding.amax(dim=-1)
        hidden: Tensor = torch.relu(self.bn6(self.linear1(global_descriptor)))
        logits: Tensor = self.linear2(hidden)
        return logits


def main() -> None:
    """用一个 [1,3,32] 的 float32 随机点云，在 CPU 上执行一次推理。

    模型随机初始化，不加载权重。[1,40] 的输出保存在局部变量 _logits 中，
    可通过断点检查；模拟输入和随机参数只用于演示前向计算。
    """
    torch.manual_seed(7)
    model: PointNetBasic = PointNetBasic().to(device="cpu", dtype=torch.float32)
    model.eval()
    points: Tensor = torch.randn((1, 3, 32), device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        _logits: Tensor = model(points)


if __name__ == "__main__":
    main()
