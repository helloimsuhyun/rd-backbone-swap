from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

# 원본 코드에서 필요없는 부분 삭제 및 forward와 init 간단한 수정


__all__ = [
    "de_convnext_base",
]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)
   


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class de_ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []


        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage)) # layer 1 2 3 4
            if cnf.out_channels is not None:
                # upsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        deconv2x2(cnf.input_channels, cnf.out_channels, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)

    

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        #8 8 1024

        for i, layer in enumerate(self.features): # stg1 -> up -> stg2-> up -> stg3 -> up -> stg4
            x = layer(x)
            if i == 1 : feature_a = x 
            elif i == 3 : feature_b = x 
            elif i == 5 : feature_c = x 
        
        feature_d = x

        
        print(f"decoder feature_a : {feature_a.size()}") # 16 16 512
        print(f"decoder feature_b : {feature_b.size()}") # 32 32 256
        print(f"decoder feature_c : {feature_c.size()}") # 128 128 64
        
        
        return [feature_c,feature_b,feature_a]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    


############-----------------------------------------------------------------------------------아래는 보조함수들
def _deconvnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    progress: bool,
    **kwargs: Any,
) -> de_ConvNeXt:

    model = de_ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    return model


def de_convnext_base(*, progress: bool = True, **kwargs: Any) -> de_ConvNeXt:

    block_setting = [
        CNBlockConfig(1024 , 512, 3),
        CNBlockConfig(512, 256, 3),
        CNBlockConfig(256, 128, 27),
        CNBlockConfig(128, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _deconvnext(block_setting, stochastic_depth_prob, progress, **kwargs)

