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

# 기존 conv의 함수를 조금만 수정함 


__all__ = [
    "bn_conv",
]

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
            nn.Linear(in_features=dim, out_features= 4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features = dim, bias=True),
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


class BN_conv(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
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

        #Downsamplings
        layers.append( 
            nn.Sequential(
                norm_layer(512*3),
                nn.Conv2d(512*3, 512*2, kernel_size=2, stride=2),
                )
            )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage)) 

        self.features = nn.Sequential(*layers)


        self.expansion = 2
        self.conv1 = conv3x3(64 * self.expansion, 128 * self.expansion, 2) #stride 2 해상도 절반 줄임
        self.bn1 = norm_layer(128 * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * self.expansion, 256 * self.expansion, 2) 
        self.bn2 = norm_layer(256 * self.expansion)

        self.conv3 = conv3x3(128 * self.expansion, 256 * self.expansion, 2) 
        self.bn3 = norm_layer(256 * self.expansion)

        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        feature_a size : torch.Size([16, 128, 64, 64])
        feature_b size : torch.Size([16, 256, 32, 32])
        feature_c size : torch.Size([16, 512, 16, 16])
        """
        l1 = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0])))))
        l1 = self.relu(l1)

        l2 = self.bn3(self.conv3(x[1]))
        l2 = self.relu(l2)
        feature = torch.cat([l1,l2,x[2]],1) 

        """
        print(f"x0 size : {x[0].size()}")
        print(f"x1 size : {x[1].size()}")
        print(f"x2 size : {x[2].size()}")

        print(f"l1 size : {l1.size()}")
        print(f"l2 size : {l2.size()}")
        print(f"feature cat size : {feature.size()}")
        """
        for x , layers in enumerate(self.features): #norm -> conv downsampling -> 
            feature = layers(feature)
            #print(f"{x} layer output size :{feature.size()}")

        output = feature.contiguous()

        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    


############-----------------------------------------------------------------------------------아래는 보조함수들


def _bn_conv(
        
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    progress: bool,
    **kwargs: Any,
) -> BN_conv:

    model = BN_conv(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    return model

def bn_conv(*, progress: bool = True, **kwargs: Any) -> _bn_conv:

    block_setting = [
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)

    return _bn_conv(block_setting, stochastic_depth_prob,progress,**kwargs)




