# -*- coding: utf-8 -*-

"""
@date: 2020/12/7 下午7:58
@file: build_resnet_backbone.py
@author: zj
@description: 
"""

from torchvision.models.utils import load_state_dict_from_url

from .. import registry
from .basicblock import BasicBlock
from .bottleneck import Bottleneck
from .resnet_backbone import ResNetBackbone
from ..norm_helper import get_norm

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

arch_settings = {
    'resnet18': (BasicBlock, (2, 2, 2, 2)),
    'resnet34': (BasicBlock, (3, 4, 6, 3)),
    'resnet50': (Bottleneck, (3, 4, 6, 3)),
    'resnet101': (Bottleneck, (3, 4, 23, 3)),
    'resnet152': (Bottleneck, (3, 8, 36, 3))
}


@registry.BACKBONE.register("ResNetBackbone")
def build_resnet_backbone(cfg, map_location=None):
    arch = cfg.MODEL.BACKBONE.ARCH
    norm_layer = get_norm(cfg)

    block_layer, layer_blocks = arch_settings[arch]
    zero_init_residual = cfg.MODEL.BACKBONE.ZERO_INIT_RESIDUAL

    backbone = ResNetBackbone(
        layer_blocks=layer_blocks,
        block_layer=block_layer,
        norm_layer=norm_layer,
        zero_init_residual=zero_init_residual
    )

    pretrained = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True, map_location=map_location)
        backbone.load_state_dict(state_dict, strict=False)

    return backbone
