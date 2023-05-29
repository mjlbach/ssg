import numpy as np
from gym.spaces import Box
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2

# TODO (sven): add IMPALA-style option.
# from ray.rllib.examples.models.impala_vision_nets import TorchImpalaVisionNet
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import torch
from torch import nn
from gym.spaces import Box

class ComplexInputNetworkIllustration(TorchModelV2, nn.Module):
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


        # Image space.
        size = int(np.product(self.processed_obs_space.shape))
        config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": model_config.get("fcnet_activation"),
            "post_fcnet_hiddens": [],
        }
        self.feature_extractor = ModelCatalog.get_model_v2(
            Box(-1.0, 1.0, (size,), np.float32),
            action_space,
            num_outputs=None,
            model_config=config,
            framework="torch",
            name="flatten",
        )
        self.flatten_dims = size
        #
        # # Optional post-concat FC-stack.
        # post_fc_stack_config = {
        #     "fcnet_hiddens": model_config.get("post_fcnet_hiddens", [256, 256, 256]),
        #     "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        # }
        # self.post_fc_stack = ModelCatalog.get_model_v2(
        #     Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
        #     self.action_space,
        #     None,
        #     post_fc_stack_config,
        #     framework="torch",
        #     name="post_fc_stack",
        # )
        #
        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        # Action-distribution head.
        self.logits_layer = SlimFC(
            in_size=self.feature_extractor.num_outputs,
            out_size=num_outputs,
            activation_fn=None,
            initializer=torch_normc_initializer(0.01),
        )
        # Create the value branch model.
        self.value_layer = SlimFC(
            in_size=self.feature_extractor.num_outputs,
            out_size=1,
            activation_fn=None,
            initializer=torch_normc_initializer(0.01),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        # Push observations through the different components
        # (CNNs, one-hot + FC, etc..).
        out, _ = self.feature_extractor(
            SampleBatch(
                {
                    SampleBatch.OBS: torch.reshape(
                        input_dict['obs_flat'], [-1, self.flatten_dims]
                    )
                }
            )
        )

        logits = self.logits_layer(out)
        value = self.value_layer(out)
        # Logits- and value branches.
        self._value_out = torch.reshape(value, [-1])

        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

