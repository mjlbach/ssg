import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphMultisetTransformer

from ssg.policies.mlp import MLP


class GMA(torch.nn.Module):
    def __init__(self, in_features, out_features=128):
        super(GMA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # The scene graph encoding is a 3 layer deep graph convolution,
        # Equivalent to 3 MLP layers
        self.conv1 = GCNConv(in_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)

        # The attention mask is a 3 layer deep graph convolution, equivalent to 3 MLP
        # layers
        self.multiset = GraphMultisetTransformer(128, 128, 128)

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

        # Head 2, feature reduction to a single value per node, compute
        # the per node attention
        x = self.multiset(x, batch, edge_index)

        # Final encoding layers
        x = self.mlp(x)

        return x
