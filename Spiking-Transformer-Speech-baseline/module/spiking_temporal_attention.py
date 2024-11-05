import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from spikingjelly.activation_based import neuron, layer
# import torch.nn.functional as F
import math
from .conv import Transpose

class MS_MLP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.config = config
        if self.config.use_ln:
            self.ln = nn.LayerNorm(in_features)
        # batch, time, dim
        self.trans1 = Transpose(1,0,2)


        self.fc1_conv = layer.Linear(in_features, hidden_features, bias=False, step_mode='m')
        self.fc1_bn = layer.BatchNorm1d(hidden_features, step_mode='m')


        if spike_mode == "lif":
            self.fc1_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True ,backend=config.backend)

        elif spike_mode == "plif":
            self.fc1_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_p, step_mode='m')

        self.fc2_conv = layer.Linear(hidden_features, out_features, bias=False, step_mode='m')
        self.fc2_bn = layer.BatchNorm1d(out_features, step_mode='m')
        # self.fc2_ln = nn.LayerNorm(out_features)

        if spike_mode == "lif":
            self.fc2_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.fc2_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        if self.config.use_dp:
            self.dropout2 = layer.Dropout(config.dropout_p, step_mode='m')



    def forward(self, x):
        if self.config.use_ln:
            #  batch, time, dim
            x = self.trans1(x)
            x = self.ln(x)
            # time, batch, dim
            x = self.trans1(x)

        x = self.fc1_conv(x)
        x = self.fc1_bn(x).contiguous()
        # fc1_lif 可以放在fc1前面
        x = self.fc1_lif(x)
        if self.config.use_dp:
            x = self.dropout1(x)

        x = self.fc2_conv(x)
        # Transformer原版中，没有LN和LIF
        x = self.fc2_bn(x).contiguous()
        # fc2_lif可以放在fc2前面
        x = self.fc2_lif(x)
        if self.config.use_dp:
            x = self.dropout2(x)
        # x = x + identity
        return x



class MS_SSA(nn.Module):
    def __init__(
        self,
        dim,
        config,
        num_heads=8,
        spike_mode="lif",
        layers=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        # self.dvs = dvs
        self.num_heads = num_heads
        self.config = config
        if self.config.use_ln:
            self.ln = nn.LayerNorm(dim)
            if spike_mode == 'lif':
                self.head_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                        step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)
            elif spike_mode == 'plif':
                self.head_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                                                  surrogate_function=config.surrogate_function,
                                                  detach_reset=config.detach_reset,
                                                  step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)
            if self.config.use_dp:
                self.head_dp = layer.Dropout(config.dropout_p, step_mode='m')
        # batch, time, dim
        self.trans1 = Transpose(1,0,2)

        self.scale = 1/math.sqrt(self.dim//self.num_heads)



        self.q_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias, step_mode='m')
        self.q_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')

        if spike_mode == "lif":
            self.q_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)

            self.q_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)

        elif spike_mode == "plif":
            self.q_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)

            self.q_lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True, backend=config.backend)


        self.k_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias,step_mode='m')

        self.k_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')

        if spike_mode == "lif":
            self.k_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

            self.k_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.k_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)

            self.k_lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)

        self.v_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias,step_mode='m')

        self.v_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')

        if spike_mode == "lif":
            self.v_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)


        elif spike_mode == "plif":
            self.v_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)

        if self.config.use_dp:
            self.attn_dropout = layer.Dropout(config.dropout_p, step_mode='m')

        if spike_mode == "lif":
            self.attn_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.attn_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)

        self.proj_conv = layer.Linear(dim, dim, bias=self.config.bias,step_mode='m')
        self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_p, step_mode='m')

        if spike_mode == "lif":
            self.proj_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.proj_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=True,backend=config.backend)

        self.layers = layers

    def forward(self, x):
        if self.config.use_ln:
            #  batch, time, dim
            x = self.trans1(x)
            x = self.ln(x)
            # time, batch, dim
            x = self.trans1(x)
            x = self.head_lif(x)


        T, B, N = x.shape

        x_qkv = x # B*N*T


        q_conv_out = self.q_conv(x_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B, H, T, D)
                .contiguous()
        )
        k_conv_out = self.k_conv(x_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)

        k = (
            k_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B,H,T,D)
                .contiguous()
        )

        v_conv_out = self.v_conv(x_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out)

        v = (
            v_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B,H,T,D)
                .contiguous()
        )


        if self.config.attn_mode == 'v1':
            qk = q.mul(k) # (B,H,T,D)
            qk = qk.sum(dim=-2, keepdim=True) # (B,H,1,D)
            qk = qk * self.scale

            qk = self.attn_lif(qk)
            qk = self.attn_dropout(qk)

            x = v.mul(qk)  # (B,H,T,D)

            x = x.permute(2,0,1,3) # (T,B,H,D)

            # Flatten the last two dimensions
            x = x.reshape(T, B, -1).contiguous()  # Ensure the tensor is stored in a contiguous chunk of memory
            # x = x.permute(0,2,1)
            x = self.proj_bn(self.proj_conv(x)).contiguous()

            x = self.proj_lif(x)
            x = self.dropout(x)
            # x = x + identity
            return x
        elif self.config.attn_mode == 'v2':

            attn = (q@ k.transpose(-2, -1))

            x = attn@v # (B,H,T,T) * (B,H,T,D)
            x = x*self.scale  # (B,H,T,T)
            x = x.permute(2, 0, 1, 3)  # (T,B,H,D)

            x = self.attn_lif(x)
            if self.config.use_dp:
                x = self.attn_dropout(x)
            x = x.reshape(T, B, -1).contiguous()
            x = self.proj_bn(self.proj_conv(x)).contiguous()
            # 注意在原版的SDT 中没有这个LIF
            x = self.proj_lif(x)
            if self.config.use_dp:
                x = self.dropout(x)
            # x = x + identity
            return x



class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        init_tau=2.0,
        spike_mode="lif",
        layers=0,
        # norm_first = False
    ):
        super().__init__()
        self.config = config

        # SDSA
        self.attn = MS_SSA(
            dim,
            config,
            num_heads=num_heads,
            spike_mode=spike_mode,
            layers=layers,
        )

        mlp_hidden_dim = config.hidden_dims
        # MLP
        self.mlp1 = MS_MLP(
            config,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
        )


    def forward(self, x):

        # Attention with residual
        attn_output = self.attn(x)
        x = x + attn_output

        # First MLP with residual
        mlp1_output = self.mlp1(x)
        x = x + mlp1_output


        return x
