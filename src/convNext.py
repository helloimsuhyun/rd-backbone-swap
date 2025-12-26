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
    "ConvNeXt",
    "ConvNeXt_Base_Weights",
    "convnext_base",
]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


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


class ConvNeXt(nn.Module):
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

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append( # layer 0
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

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
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)

        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )
        

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def _forward_impl(self, x: Tensor) -> Tensor:

        for i, layer in enumerate(self.features): #stem -> stage1 (3)-> ds -> stage2 (3) -> ds -> stage3 (27) -> ds -> stage4 (3)
            x = layer(x)
            if i == 0 : feature_stem = x # stem
            elif i == 1 : feature_a = x # stage 1
            elif i == 3 : feature_b = x # stage 2
            elif i == 5 : feature_c = x # stage 3
            elif i == 7 : feature_d = x # stage 4

        
        print(f"encoder feature_stem size : {feature_a.size()}") #64 64 128
        print(f"encoder feature_a size : {feature_a.size()}") #64 64 128
        print(f"encoder feature_b size : {feature_b.size()}") #32 32 256
        print(f"encoder feature_c size : {feature_c.size()}") #16 16 512
        
        
        #x = self.avgpool(x)
        #x = self.classifier(x)

        return [feature_a,feature_b,feature_c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    


############-----------------------------------------------------------------------------------아래는 보조함수들
def _convnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ConvNeXt:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
    "_docs": """
        These weights improve upon the results of the original paper by using a modified version of TorchVision's
        `new training recipe
        <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    """,
}


class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88591464,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.062,
                    "acc@5": 96.870,
                }
            },
            "_ops": 15.355,
            "_file_size": 338.064,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", ConvNeXt_Base_Weights.IMAGENET1K_V1))
def convnext_base(*, weights: Optional[ConvNeXt_Base_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Base_Weights
        :members:
    """
    weights = ConvNeXt_Base_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)

