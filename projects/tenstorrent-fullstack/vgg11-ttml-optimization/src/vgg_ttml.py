\
\
\
   

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from vgg import TTConv2d, tt_max_pool


class FrozenVGG11:
                                       
    conv_indices = (0, 3, 6, 8, 11, 13, 16, 18)
                          
    channels = (3, 64, 128, 256, 256, 512, 512, 512, 512)
                                               
    pool_after = {0, 1, 3, 5, 7}

    def __init__(self, device, image_size=112, weights_path=None):
        if image_size < 32:
            raise ValueError("VGG11 requires image_size >= 32")
                                 
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
        else:
            from torchvision.models import VGG11_Weights
            state = VGG11_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        self.device = device
        self.image_size = image_size
                                             
        self.out_features = 512 * (image_size // 32) ** 2
        self.layers = []
        self.state = {}
                                          
        for i, index in enumerate(self.conv_indices):
            conv_config_args = dict(
                weights_dtype=ttnn.bfloat16,
                config_tensors_in_dram=True,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                act_block_h_override=128 if i == 0 else 0,
                enable_act_double_buffer=False,
                enable_weights_double_buffer=False,
                reshard_if_not_optimal=i == 0,
                deallocate_activation=i == 0,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if i in {0, 1, 3}:
                conv_config_args["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            conv_config = ttnn.Conv2dConfig(**conv_config_args)
            layer = TTConv2d(
                self.channels[i],
                self.channels[i + 1],
                device,
                conv_config=conv_config,
                memory_config=ttnn.L1_MEMORY_CONFIG if i == 0 else None,
            )
            prefix = f"features.{index}"
            weight = state[f"{prefix}.weight"].detach().cpu().contiguous()
            bias = state[f"{prefix}.bias"].detach().cpu().contiguous()
            expected = (self.channels[i + 1], self.channels[i], 3, 3)
            if tuple(weight.shape) != expected or tuple(bias.shape) != (self.channels[i + 1],):
                raise ValueError(f"Invalid VGG11 weights at {prefix}")
            self.state[f"{prefix}.weight"] = weight
            self.state[f"{prefix}.bias"] = bias
            layer.weight = ttnn.from_torch(weight.to(torch.bfloat16), dtype=ttnn.bfloat16)
            layer.bias = ttnn.from_torch(bias.to(torch.bfloat16).reshape(1, 1, 1, -1),
                                         dtype=ttnn.bfloat16)
            self.layers.append(layer)

    def __call__(self, x, batch):
        h = w = self.image_size
        for i, layer in enumerate(self.layers):
            x, h, w = layer(x, batch, h, w)
            if i in self.pool_after:
                x, h, w = tt_max_pool(x, batch, h, w, self.channels[i + 1])

        x = ttnn.reshape(x, (1, 1, batch, self.out_features))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttml.autograd.create_tensor(x, requires_grad=False)


class VGGClassifier(AbstractModuleBase):
                                         
    def __init__(self, in_features, num_classes=37, hidden_size=512):
        super().__init__()
        self.fc1 = LinearLayer(in_features, hidden_size)
        self.fc2 = LinearLayer(hidden_size, hidden_size)
        self.fc3 = LinearLayer(hidden_size, num_classes)

    def forward(self, x):
        x = ttml.ops.unary.relu(self.fc1(x))
        x = ttml.ops.unary.relu(self.fc2(x))
        return self.fc3(x)

    def cpu_state_dict(self):
                                                         
        state = {}
        for name, parameter in self.parameters().items():
            array = parameter.to_numpy(ttnn.DataType.FLOAT32)
            state[name] = torch.from_numpy(np.array(array, copy=True))
        return state

    def load_cpu_state_dict(self, state):
                                        
        parameters = self.parameters()
        if parameters.keys() != state.keys():
            raise ValueError("Classifier checkpoint keys do not match")
        for name, parameter in parameters.items():
            if tuple(parameter.shape) != tuple(state[name].shape):
                raise ValueError(f"Classifier checkpoint shape mismatch: {name}")
            value = ttml.autograd.Tensor.from_numpy(state[name].float().numpy(),
                                                     new_type=ttnn.DataType.BFLOAT16)
            parameter.set_value(value.get_value())
