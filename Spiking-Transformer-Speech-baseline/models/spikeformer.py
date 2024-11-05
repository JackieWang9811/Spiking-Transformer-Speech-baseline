from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from spikingjelly.clock_driven.neuron import (
#     MultiStepLIFNode,
#     MultiStepParametricLIFNode,)s
# from spikingjelly.activation_based.neuron_kernel import MultiStepLIFNodePTT,MultiStepParametricLIFNodePTT
from spikingjelly.activation_based.neuron import LIFNode,ParametricLIFNode
from module import *
# ---- module:
# -- from .ms_conv import MS_Block_Conv
# -- from .sps import MS_SPS

# sps: Spiking Patch Splitting (SPS)

class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        # 这是个用法
        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            # self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            self.head_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            # self.head_lif = MultiStepParametricLIFNode(
            #     init_tau=2.0, detach_reset=True, backend="cupy"
            # )
            self.head_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):

        #反射属性, getattr(object, name) 从指定的对象中获取与给定名称对应的属性值。
        block = getattr(self, f"block") # 从 self中获取block
        patch_embed = getattr(self, f"patch_embed") # 从 self中获取patch_embed

                                               # size:(Time_Step, Batch_Size, C, H, W)
        x, _, hook = patch_embed(x, hook=hook) # x:(4, 64, 256, 8, 8)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        # if len(x.shape) < 5:
        #     x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1) #将输入重复T次，这与时间步有关
        # else:
        #     x = x.transpose(0, 1).contiguous()

        # 主干网络()
        x, hook = self.forward_features(x, hook=hook)


        # LIF
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()
        # 分类头
        x = self.head(x)
        # if not self.TET:
        #     x = x.mean(0)


        return x, hook


@register_model
# spiking-driven transformer
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
