
import torch
import torch.nn as nn


from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode
from module.conv import Transpose


class SConv2dSubsampling(nn.Module):
    """Conv2d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)

    """

    def __init__(self, config):
        super(SConv2dSubsampling, self).__init__()
        self.config = config
        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(3, 0, 1, 2) # (B,C,D,T) => (T,B,C,D)
        self.trans3 = Transpose(1, 2, 3, 0) # (T,B,C,D) => (B,C,D,T)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, self.config.conv_hidden, kernel_size=3, stride=2,
                      bias=self.config.bias, padding=1),
            nn.BatchNorm2d(self.config.conv_hidden),
            self.trans2,
            self.create_lif_node(self.config.spike_mode, self.config),
            self.trans3,
            # layer.Dropout(self.config.dropout_l, step_mode='m')
        )
    def create_lif_node(self, spike_mode, config):
        if spike_mode == 'plif':
            return ParametricLIFNode(
                init_tau=self.config.init_tau, v_threshold=self.config.v_threshold,
                surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset,
                step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)
        return LIFNode(
            tau=self.config.init_tau, v_threshold=self.config.v_threshold,
            surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)

    def forward(self, x):
        x = self.trans1(x)

        # (B, D, T) -> (B, 1, D, T)
        # (32,80,1579) => (32,1,80,1579)

        # (B, C, D//2, T//2) -> (B, C, D//4, T//4) -> (B, C*D//4, T//4)
        # (32,1,80,1579) => (32,144,40,797) => LIF(797, 32, 144, 40)=> (32,144,40,797)
        #                => (32,144,20,395) => LIF(395,32,144,20) => (32,144,20,395)
        #                => (32,144*20, 395)

        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.conv_block:
            x = layer(x)

        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        #
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)

        return x