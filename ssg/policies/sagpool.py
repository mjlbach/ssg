import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_scatter import scatter

from ssg.policies.mlp import MLP


def weighted_pool(x, weight, batch, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    x * weight.reshape(-1, 1)
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")


class SAG(torch.nn.Module):
    def __init__(self, in_features, out_features=256):
        super(SAG, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.mlp = MLP([64, 512, 512], batch_norm=False)
        self.sag_pool = SAGPooling(in_channels=512)

        self.mlp2 = MLP([512, 512, out_features], batch_norm=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        x, edge_index, _, batch, perm, score = self.sag_pool(
            x, edge_index, batch=data.batch
        )
        x = weighted_pool(x, score, batch)
        x = self.mlp2(x)

        return x
