import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions

# TODO (sven): add IMPALA-style option.
# from ray.rllib.examples.models.impala_vision_nets import TorchImpalaVisionNet
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.torch_utils import one_hot

import torch
from torch import nn
import torch_geometric as pyg
from gym.spaces import Box, Discrete, MultiDiscrete

from ssg.policies import REGISTERED_MODELS
from torch_geometric.data import HeteroData

def generate_pyg_homogeneous_batch(input_dict):
    edges = input_dict["obs"]["scene_graph"]["edges"]
    nodes = input_dict["obs"]["scene_graph"]["nodes"]
    graphs = []
    idx = 0
    for _ in nodes.lengths:
        # TODO (mjlbach): This basically ensures there is at least a single node per graph, otherwise we cannot guarantee there is an input to the output MLP. Should we instead pad with sentinel values?
        node_length = max(int(nodes.lengths[idx]), 1)
        graph_edges = edges.values[idx][
            : edges.lengths[idx].type(torch.int32)
        ].type(torch.int64)
        graph_nodes = nodes.values[idx][:node_length]
        graph = pyg.data.Data(x=graph_nodes, edge_index=graph_edges.T)
        graphs.append(graph)
        idx += 1
    batch = pyg.data.Batch.from_data_list(graphs)
    if nodes.values.device.type == "cpu":
        batch.cpu()
    else:
        batch.cuda()
    return batch

def generate_pyg_heterogeneous_batch(input_dict, value):
    nodes = input_dict["obs"]["scene_graph"]["nodes"]
    graphs = []
    idx = 0
    for _ in nodes.lengths:
        # TODO (mjlbach): This basically ensures there is at least a single node per graph, otherwise we cannot guarantee there is an input to the output MLP. Should we instead pad with sentinel values?
        node_length = max(int(nodes.lengths[idx]), 1)
        data = HeteroData()
        data['node'].x = nodes.values[idx][:node_length]
        for key in value:
            if key == 'nodes':
                continue
            else:
                data['node', key, 'node'].edge_index = value[key].values[idx].T.long()
        graphs.append(data)
        idx += 1

    batch = pyg.data.Batch.from_data_list(graphs)
    
    if nodes.values.device.type == "cpu":
        batch.cpu()
    else:
        batch.cuda()
    return batch

def tile_goal(input_dict, goal):
    nodes = input_dict["obs"]["scene_graph"]["nodes"]
    idx = 0
    sz = 0
    for l in nodes.lengths:
        sz += max(int(l), 1)
    filler = torch.zeros((sz, goal.shape[1]))

    c = 0
    for _ in nodes.lengths:
        # TODO (mjlbach): This basically ensures there is at least a single node per graph, otherwise we cannot guarantee there is an input to the output MLP. Should we instead pad with sentinel values?
        node_length = max(int(nodes.lengths[idx]), 1)
        filler[c:c+node_length] = goal[idx].reshape(1, -1).repeat_interleave(node_length, 0)
        c += node_length
        idx += 1
    if nodes.values.device.type == "cpu":
        filler = filler.cpu()
    else:
        filler = filler.cuda()
    return filler

class ComplexInputNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and vaulue heads.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        self.model_config = model_config
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        # Atari type CNNs or IMPALA type CNNs (with residual layers)?
        # self.cnn_type = self.model_config["custom_model_config"].get(
        #     "conv_type", "atari")

        # Build the CNN(s) given obs_space's image components.
        self.feature_extractors = {}
        self.flatten_dims = {}
        concat_size = 0
        for key, component in self.original_space.spaces.items():
            # Image space.
            if key == "scene_graph":
                name = "gnn_{}".format(key)
                self.graph_architecture = kwargs.get("graph_model", "SAM")
                GraphModel = REGISTERED_MODELS[self.graph_architecture]
                if self.graph_architecture in ["HGNN", "HSAM", "HFAM", "HSAMGE", "HSAML"]:
                    node_metadata = ["node"]
                    edge_metadata = []
                    for candidate_edge in component:
                        if candidate_edge not in ["node", "nodes"]:
                            edge_metadata.append(['node', candidate_edge, 'node'])

                    self.feature_extractors["scene_graph"] = GraphModel(
                        in_features=component["nodes"].child_space.shape[0],
                        metadata=(node_metadata, edge_metadata)
                    )
                else:
                    self.feature_extractors["scene_graph"] = GraphModel(
                        in_features=component["nodes"].child_space.shape[0]
                    )
                # THIS IS CRITICAL DO NOT FORGET THIS
                self.add_module(name, self.feature_extractors[key])
                concat_size += self.feature_extractors["scene_graph"].out_features #type: ignore
            elif key == "task_graph_obs":
                size = int(np.product(component.shape))
                self.flatten_dims[key] = size
            elif key == "object_set":
                name = "transformer_{}".format(key)
                self.feature_extractors["object_set"] = TransformerModel(
                    num_features=component.child_space.shape[0],
                    ntoken=128,
                    d_model=256,
                    d_hid=200,
                    nhead=8,
                    dropout=0.2,
                    nlayers=2,
                )
                # THIS IS CRITICAL DO NOT FORGET THIS
                self.add_module(name, self.feature_extractors[key])
                concat_size += self.feature_extractors["object_set"].out_features
            elif len(component.shape) == 3:
                name = "cnn_{}".format(key)
                config = {
                    "conv_filters": model_config["conv_filters"]
                    if "conv_filters" in model_config
                    else get_filter_config(obs_space.shape),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                }
                # if self.cnn_type == "atari":
                self.feature_extractors[key] = ModelCatalog.get_model_v2(
                    component,
                    action_space,
                    num_outputs=None, #type: ignore
                    model_config=config,
                    framework="torch",
                    name=name,
                )

                # THIS IS CRITICAL DO NOT FORGET THIS
                self.add_module(name, self.feature_extractors[key])
                concat_size += self.feature_extractors[key].num_outputs
            # Discrete|MultiDiscrete inputs -> One-hot encode.
            elif isinstance(component, (Discrete, MultiDiscrete)):
                name = "discrete_{}".format(key)
                if isinstance(component, Discrete):
                    size = component.n
                else:
                    size = sum(component.nvec)
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.feature_extractors[key] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None, #type: ignore
                    model_config=config,
                    framework="torch",
                    name=name,
                )
                # THIS IS CRITICAL DO NOT FORGET THIS
                self.add_module(name, self.feature_extractors[key])
                concat_size += self.feature_extractors[key].num_outputs
            # Everything else (1D Box).
            else:
                name="flat_{}".format(key)
                size = int(np.product(component.shape))
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.feature_extractors[key] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None, #type: ignore
                    model_config=config,
                    framework="torch",
                    name="flatten_{}".format(key)
                )
                # THIS IS CRITICAL DO NOT FORGET THIS
                self.add_module(name, self.feature_extractors[key])
                self.flatten_dims[key] = size
                concat_size += self.feature_extractors[key].num_outputs

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", [ 128, 128, 128]),
            "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack",
        )

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        if num_outputs:
            # Action-distribution head.
            self.logits_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=num_outputs,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
        else:
            self.num_outputs = concat_size
        self.value_head = ModelCatalog.get_model_v2(
                    Box(float("-inf"), float("inf"), shape=(self.post_fc_stack.num_outputs,), dtype=np.float32),
                    action_space,
                    num_outputs=1, #type: ignore
                    model_config=post_fc_stack_config,
                    framework="torch",
                    name='value_head',
                )
        self.logits_head = ModelCatalog.get_model_v2(
                    Box(float("-inf"), float("inf"), shape=(self.post_fc_stack.num_outputs,), dtype=np.float32),
                    action_space,
                    num_outputs=num_outputs, #type: ignore
                    model_config=post_fc_stack_config,
                    framework="torch",
                    name='value_head',
                )


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )

        # Push observations through the different components
        # (CNNs, one-hot + FC, etc..).
        outs = []
        for key, value in orig_obs.items():
            if key in ["task_graph_obs"]:
                continue
            elif key in ["rgb", "depth", "ego_sem_map", "agent_path", "multichannel_map"]:
                cnn_out, _ = self.feature_extractors[key](
                    SampleBatch({SampleBatch.OBS: value})
                )
                outs.append(cnn_out)
            elif key in ["one_hot"]:
                if value.dtype in [torch.int32, torch.int64, torch.uint8]:
                    one_hot_in = {
                        SampleBatch.OBS: one_hot(
                            value, self.flattened_input_space[value]
                        )
                    }
                else:
                    one_hot_in = {SampleBatch.OBS: value}
                one_hot_out, _ = self.feature_extractors[key](SampleBatch(one_hot_in))
                outs.append(one_hot_out)
            elif key in ["scene_graph"]:
                if self.graph_architecture in ["HFAM"]:
                    batch = generate_pyg_heterogeneous_batch(input_dict, value)
                    outs.append(self.feature_extractors[key]((batch, orig_obs["task_graph_obs"])))
                elif self.graph_architecture in ["HSAML"]:
                    batch = generate_pyg_heterogeneous_batch(input_dict, value)
                    goal = tile_goal(goal=orig_obs["task_obs"], input_dict=input_dict)
                    outs.append(self.feature_extractors[key](batch, goal))
                elif self.graph_architecture in ["HSAMGE"]:
                    goal_embedding, _ = self.feature_extractors['task_obs'](
                        SampleBatch(
                            {
                                SampleBatch.OBS: torch.reshape(
                                    orig_obs['task_obs'], [-1, self.flatten_dims['task_obs']]
                                )
                            }
                        )
                    )
                    goal_embedding = tile_goal(goal=goal_embedding, input_dict=input_dict)
                    batch = generate_pyg_heterogeneous_batch(input_dict, value)
                    outs.append(self.feature_extractors[key](batch, goal_embedding))
                else:
                    if self.graph_architecture in ["HGNN", "HSAM", ]:
                        batch = generate_pyg_heterogeneous_batch(input_dict, value)
                    else:
                        batch = generate_pyg_homogeneous_batch(input_dict)
                    outs.append(self.feature_extractors[key](batch))
            elif key in ["object_set"]:
                val = SampleBatch({SampleBatch.OBS: value})
                val = val["obs"]
                out = self.feature_extractors[key](val.values, val.lengths)
                outs.append(out)
            elif key in ["task_obs"]:
                nn_out, _ = self.feature_extractors[key](
                    SampleBatch(
                        {
                            SampleBatch.OBS: torch.reshape(
                                value, [-1, self.flatten_dims[key]]
                            )
                        }
                    )
                )
                outs.append(nn_out)
            else:
                raise Exception("Unsupported observation modality")

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        logits, _ = self.logits_head(SampleBatch({SampleBatch.OBS: out}))
        values, _ = self.value_head(SampleBatch({SampleBatch.OBS: out}))

        # Logits- and value branches.
        # logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])

        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
