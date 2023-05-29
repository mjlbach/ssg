import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax
from torch_scatter import scatter

from ssg.policies.mlp import MLP


def weighted_pool(x, weight, batch, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    x * weight.reshape(-1, 1)
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")


def graph_sum(x, batch, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")


class SAMR(torch.nn.Module):
    def __init__(self, in_features, out_features=256):
        super(SAMR, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # The scene graph encoding is a 3 layer deep graph convolution,
        # Equivalent to 3 MLP layers
        self.conv1 = GCNConv(in_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)

        # The attention mask is a 3 layer deep graph convolution, equivalent to 3 MLP
        # layers
        self.a_conv1 = GCNConv(in_features, 64)
        self.a_conv2 = GCNConv(64, 32)
        self.a_conv3 = GCNConv(32, 1)

        self.mlp = MLP([128, 128, out_features], batch_norm=False)

    def forward(self, data):
        # Extract the features
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Head 1, no feature reduction, compute the per-node encoding
        # including effects propogated from local neighbors
        features = self.conv1(x, edge_index)
        features = F.relu(features)
        features = self.conv2(features, edge_index)
        features = F.relu(features)
        features = self.conv3(features, edge_index)
        features = F.relu(features)

        # Head 2, feature reduction to a single value per node, compute
        # the per node attention
        weight = self.a_conv1(x, edge_index)
        weight = F.relu(weight)
        weight = self.a_conv2(weight, edge_index)
        weight = F.relu(weight)
        weight = self.a_conv3(weight, edge_index)
        weight = F.relu(weight)

        # Softmax of the weight
        weight = softmax(weight, batch)

        # Compute the attention per node
        # Weight the node features by the node attention
        feature_attention = graph_sum(features * weight, batch)

        # Final encoding layers
        feature_attention = self.mlp(feature_attention)

        return feature_attention
