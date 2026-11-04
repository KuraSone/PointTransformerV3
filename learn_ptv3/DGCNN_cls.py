"""DGCNN 点云分类：逐层重建近邻图，提取边特征，再汇聚为整个点云的类别分数。"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def knn(features: Tensor, k: int) -> Tensor:
    """在每个样本内部，为每个点选择特征距离最近的 k 个点。

    参数:
        features: 浮点张量 [B,C,N]，B 为样本数，C 为特征通道数，N 为点数。
            features[b,:,i] 是第 b 个样本中第 i 个点的 C 维特征。
        k: 每个点选取的近邻数，满足 1<=k<=N。

    返回:
        与 features 同设备的 int64 张量 [B,N,k]。
        indices[b,i,r]=j 表示中心点 i 的第 r 个邻居是同一样本中的点 j。
        邻居按计算得到的平方欧氏距离从近到远排列。

    中心点自身参与近邻候选；距离并列时，具体索引及次序由 topk 决定。
    """
    negative_twice_inner: Tensor = -2 * (features.mT @ features)
    squared_norm: Tensor = features.square().sum(dim=1, keepdim=True)
    negative_squared_distance: Tensor = (
        -squared_norm - negative_twice_inner - squared_norm.mT
    )
    indices: Tensor = negative_squared_distance.topk(
        k=k, dim=-1, largest=True, sorted=True
    ).indices
    return indices


def get_graph_feature(features: Tensor, k: int) -> Tensor:
    """重新计算近邻，并将逐点特征组织为中心点与邻居之间的边特征。

    参数:
        features: 浮点张量 [B,C,N]，最后一维的 i 对应输入中的第 i 个点。
        k: 每个中心点的近邻数，满足 1<=k<=N。

    返回:
        与 features 同设备、同 dtype 的张量 [B,2C,N,k]。
        若中心点 i 的第 r 个邻居为 j，则前 C 个通道存储
        features[b,:,j]-features[b,:,i]，后 C 个通道存储 features[b,:,i]。
        中心点的排列不变；最后一维 r 的排列取决于本次近邻搜索结果。

    每次调用均使用传入的 features 建图，因此后续层的邻居由当前层特征决定。
    """
    neighbor_indices: Tensor = knn(features, k)
    points: Tensor = features.mT
    batch_indices: Tensor = torch.arange(
        features.shape[0], device=features.device
    ).reshape(-1, 1, 1)
    neighbors: Tensor = points[batch_indices, neighbor_indices]
    centers: Tensor = points.unsqueeze(2).expand(-1, -1, k, -1)
    edges: Tensor = torch.cat((neighbors - centers, centers), dim=-1)
    edges = edges.permute(0, 3, 1, 2).contiguous()
    return edges


class DGCNN(nn.Module):
    """固定每点 20 个近邻、输出 40 类分数的 DGCNN 分类器。

    四个 EdgeConv 阶段均先按当前特征建图，再对边做卷积并在邻居维取最大值，
    分别产生 64、64、128、256 通道的逐点特征。
    将四阶段结果按通道拼接并映射为 1024 通道后，沿点维分别取最大值和均值，
    得到每个点云的 2048 维全局特征，再通过 512、256 维隐藏层完成分类。
    """

    def __init__(self) -> None:
        super().__init__()
        self.k: int = 20
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.bn3: nn.BatchNorm2d = nn.BatchNorm2d(128)
        self.bn4: nn.BatchNorm2d = nn.BatchNorm2d(256)
        self.bn5: nn.BatchNorm1d = nn.BatchNorm1d(1024)

        self.conv1: nn.Conv2d = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2: nn.Conv2d = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv3: nn.Conv2d = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv4: nn.Conv2d = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv5: nn.Conv1d = nn.Conv1d(512, 1024, kernel_size=1, bias=False)

        self.linear1: nn.Linear = nn.Linear(2048, 512, bias=False)
        self.bn6: nn.BatchNorm1d = nn.BatchNorm1d(512)
        self.linear2: nn.Linear = nn.Linear(512, 256)
        self.bn7: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.linear3: nn.Linear = nn.Linear(256, 40)

    def forward(self, points: Tensor) -> Tensor:
        """计算每个输入点云的 40 个类别分数。

        推理前调用 eval()，使 BatchNorm 使用保存的运行统计量。

        参数:
            points: 浮点坐标张量 [B,3,N]，B>=1，N>=20。
                points[b,:,i] 是第 b 个点云中第 i 个点的 [x,y,z] 坐标。
                坐标须有限，张量的设备和 dtype 须与模型参数一致。

        返回:
            浮点张量 [B,40]，logits[b,c] 是第 b 个点云属于第 c 类的分数。
            每个点云对应一行；分数未经过 softmax，不是概率。
        """
        edges1: Tensor = get_graph_feature(points, self.k)
        edge_features1: Tensor = F.leaky_relu(
            self.bn1(self.conv1(edges1)), negative_slope=0.2
        )
        stage1: Tensor = edge_features1.amax(dim=-1)

        edges2: Tensor = get_graph_feature(stage1, self.k)
        edge_features2: Tensor = F.leaky_relu(
            self.bn2(self.conv2(edges2)), negative_slope=0.2
        )
        stage2: Tensor = edge_features2.amax(dim=-1)

        edges3: Tensor = get_graph_feature(stage2, self.k)
        edge_features3: Tensor = F.leaky_relu(
            self.bn3(self.conv3(edges3)), negative_slope=0.2
        )
        stage3: Tensor = edge_features3.amax(dim=-1)

        edges4: Tensor = get_graph_feature(stage3, self.k)
        edge_features4: Tensor = F.leaky_relu(
            self.bn4(self.conv4(edges4)), negative_slope=0.2
        )
        stage4: Tensor = edge_features4.amax(dim=-1)

        multiscale: Tensor = torch.cat((stage1, stage2, stage3, stage4), dim=1)
        point_embedding: Tensor = F.leaky_relu(
            self.bn5(self.conv5(multiscale)), negative_slope=0.2
        )

        global_max: Tensor = point_embedding.amax(dim=-1)
        global_mean: Tensor = point_embedding.mean(dim=-1)
        global_descriptor: Tensor = torch.cat((global_max, global_mean), dim=1)
        hidden: Tensor = F.leaky_relu(
            self.bn6(self.linear1(global_descriptor)), negative_slope=0.2
        )
        hidden = F.leaky_relu(self.bn7(self.linear2(hidden)), negative_slope=0.2)
        logits: Tensor = self.linear3(hidden)
        return logits


def main() -> None:
    """用一个 [1,3,32] 的 float32 随机点云，在 CPU 上执行一次推理。

    模型随机初始化，不加载权重。[1,40] 的输出保存在局部变量 _logits 中，
    可通过断点检查；模拟输入和随机参数只用于演示前向计算。
    """
    torch.manual_seed(7)
    model: DGCNN = DGCNN().to(device="cpu", dtype=torch.float32)
    model.eval()
    points: Tensor = torch.randn((1, 3, 32), device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        _logits: Tensor = model(points)


if __name__ == "__main__":
    main()
