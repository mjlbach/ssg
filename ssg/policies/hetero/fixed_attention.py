import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

from ssg.policies.mlp import MLP

CLASS_IDX = 7

def graph_sum(x, batch, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")

class HFAM(torch.nn.Module):
    def __init__(self, in_features, out_features=256, metadata=None, num_heads=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # The scene graph encoding is a 3 layer deep graph convolution,
        # Equivalent to 3 MLP layers
        self.conv1 = HGTConv(in_features, 128, metadata, num_heads)
        self.conv2 = HGTConv(128, 128, metadata, num_heads)
        self.conv3 = HGTConv(128, 128, metadata, num_heads)

        self.mlp = MLP([128, 128, out_features], batch_norm=False)

    def forward(self, data_combined):
        # Extract the features
        data, task_batch = data_combined
        x_dict_init = data.x_dict
        edge_index_dict = data.edge_index_dict
        batch = data.batch_dict

        # First round of graph convolution
        x_dict = self.conv1(x_dict_init, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        num_task_object = task_batch.shape[-1]
        
        weight = (x_dict_init['node'][:, CLASS_IDX].unsqueeze(1).repeat(1, num_task_object) == task_batch[batch['node']]).any(1)
        weight = weight.unsqueeze(1).float()

        # Weight the node features by the node attention
        x = graph_sum(x_dict['node'] * weight, batch['node'])

        # Final encoding layers
        x = self.mlp(x)

        return x
