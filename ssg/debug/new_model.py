import numpy as np
from gym.spaces import Box
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions

# TODO (sven): add IMPALA-style option.
# from ray.rllib.examples.models.impala_vision_nets import TorchImpalaVisionNet
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import torch
from torch import nn
import torch_geometric as pyg
from gym.spaces import Box

class ComplexInputNetworkNew(TorchModelV2, nn.Module):
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
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        self.flatten_dims = {}
        self.feature_extractors = {}
        concat_size = 0

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


        for key, component in self.original_space.spaces.items():
            # Image space.
            size = int(np.product(component.shape))
            config = {
                "fcnet_hiddens": model_config["fcnet_hiddens"],
                "fcnet_activation": model_config.get("fcnet_activation"),
                "post_fcnet_hiddens": [],
            }
            self.feature_extractors[key] = ModelCatalog.get_model_v2(
                Box(-1.0, 1.0, (size,), np.float32),
                action_space,
                num_outputs=None,
                model_config=config,
                framework="torch",
                name="flatten_{}".format(key),
            )
            self.flatten_dims[key] = size
            concat_size += self.feature_extractors[key].num_outputs

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
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

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )

        graphs = []
        idx = 0

        # Push observations through the different components
        # (CNNs, one-hot + FC, etc..).
        outs = []
        for key, value in orig_obs.items():
            if key in ["scene_graph"]:
                edges = input_dict["obs"]["scene_graph"]["edges"]
                nodes = input_dict["obs"]["scene_graph"]["nodes"]
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

                outs.append(self.feature_extractors[key](batch))
            else:
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

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])

        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

