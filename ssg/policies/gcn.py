import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from ssg.policies.mlp import MLP

class GNN(torch.nn.Module):
    def __init__(self, in_features, out_features=256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # The scene graph encoding is a 3 layer deep graph convolution,
        # Equivalent to 3 MLP layers
        self.conv1 = GCNConv(in_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)

        self.mlp = MLP([128, 128, out_features], batch_norm=False)

    def forward(self, data):
        # Extract the features
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Head 1, no feature reduction, compute the per-node encoding
        # including effects propogated from local neighbors
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Pool the model
        x = global_mean_pool(x , batch)

        # Final encoding layers
        x = self.mlp(x)

        return x
