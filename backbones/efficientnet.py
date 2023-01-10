import math
from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn

from utils.backbone import ConvNormActivation, SqueezeExcitation, StochasticDepth

__all__ = [
  'EfficientNet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
  'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'
]


class MBConvConfig:
    def __init__(
      self, expand_ratio: float, kernel_size: int, stride: int, in_channels: int, out_channels: int, num_layers: int,
      width_multiplier: float, depth_multiplier: float
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = self.adjust_channels(in_channels, width_multiplier)
        self.out_channels = self.adjust_channels(out_channels, width_multiplier)
        self.num_layers = self.adjust_depth(num_layers, depth_multiplier)
    
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', in_channels={in_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers})'
        return s.format(**self.__dict__)
    
    @staticmethod
    def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    @staticmethod
    def adjust_channels(channels: int, width_multiplier: float, min_value: Optional[int] = None) -> int:
        return MBConvConfig._make_divisible(channels * width_multiplier, 8, min_value)
    
    @staticmethod
    def adjust_depth(num_layers: int, depth_multiplier: float) -> int:
        return int(math.ceil(num_layers * depth_multiplier))


class MBConv(nn.Module):
    def __init__(
      self,
      cfg: MBConvConfig,
      stochastic_depth_prob: float,
      norm_layer: Callable[..., nn.Module],
      se_layer: Callable[..., nn.Module] = SqueezeExcitation
    ) -> None:
        super(MBConv, self).__init__()
        
        if not (1 <= cfg.stride <= 2):
            raise ValueError('Illegal stride value')
        
        self.use_res_connection = cfg.stride == 1 and cfg.in_channels == cfg.out_channels
        layers: List[nn.module] = []
        act_layer = nn.SiLU
        
        # expansion phase
        expanded_channels = cfg.adjust_channels(cfg.in_channels, cfg.expand_ratio)
        if expanded_channels != cfg.in_channels:
            layers.append(
              ConvNormActivation(
                cfg.in_channels, expanded_channels, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer
              )
            )
        
        # depth-wise phase
        layers.append(
          ConvNormActivation(
            expanded_channels,
            expanded_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            groups=expanded_channels,
            norm_layer=norm_layer,
            act_layer=act_layer
          )
        )
        
        # squeeze and excitation
        squeeze_channels = max(1, cfg.in_channels // 4)
        layers.extend((
          se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)),
          ConvNormActivation(
            expanded_channels, cfg.out_channels, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer
          ),
        ))
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')
        self.out_channels = cfg.out_channels
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.block(inputs)
        if self.use_res_connection:
            result = self.stochastic_depth(result)
            result += inputs
        return result


class EfficientNet(nn.Module):
    def __init__(
      self,
      inverted_residual_settings: List[MBConvConfig],
      dropout: float,
      stochastic_depth_prob: float = 0.2,
      num_classes: int = 1000,
      block: Optional[Callable[..., nn.Module]] = None,
      norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(EfficientNet, self).__init__()
        if not inverted_residual_settings:
            raise ValueError('The inverted_residual_settings should not be empty')
        elif not (
          isinstance(inverted_residual_settings, Sequence) and
          all(isinstance(s, MBConvConfig) for s in inverted_residual_settings)
        ):
            raise TypeError('The inverted_residual_settings should be List[MBConvConfig]')
        
        block = block or MBConv
        norm_layer = norm_layer or nn.BatchNorm2d
        
        # stem
        stem_out = inverted_residual_settings[0].in_channels
        self.stem = nn.Sequential(*[
            ConvNormActivation(3, stem_out, kernel_size=3, stride=2, norm_layer=norm_layer, act_layer=nn.SiLU)
        ])
        
        # inverted residual blocks
        total_stage_blocks = sum(cfg.num_layers for cfg in inverted_residual_settings)
        stage_block_id = 0
        self.blocks = nn.ModuleList([])
        for cfg in inverted_residual_settings:
            stage: List[nn.Module] = []
            for _ in range(cfg.num_layers):
                # copy to avoid modifications. Shallow copy is enough
                block_cfg = deepcopy(cfg)
                
                # overwrite info if not the first conv in the stage
                if stage:
                    block_cfg.in_channels = block_cfg.out_channels
                    block_cfg.stride = 1
                
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                
                stage.append(block(block_cfg, sd_prob, norm_layer))
                stage_block_id += 1
            self.blocks.extend(stage)
        
        # head
        head_in = inverted_residual_settings[-1].out_channels
        head_out = head_in * 4
        self.head = nn.Sequential(*[
            ConvNormActivation(head_in, head_out, kernel_size=1, norm_layer=norm_layer, act_layer=nn.SiLU)
        ])
        
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
          nn.Dropout(p=dropout, inplace=True),
          nn.Linear(head_out, num_classes),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.head(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _efficientnet_config(width_multiplier: float, depth_multiplier: float) -> List[MBConvConfig]:
    bottleneck_config = partial(MBConvConfig, width_multiplier=width_multiplier, depth_multiplier=depth_multiplier)
    
    return [
      bottleneck_config(1, 3, 1, 32, 16, 1),
      bottleneck_config(6, 3, 2, 16, 24, 2),
      bottleneck_config(6, 5, 2, 24, 40, 2),
      bottleneck_config(6, 3, 2, 40, 80, 3),
      bottleneck_config(6, 5, 1, 80, 112, 3),
      bottleneck_config(6, 5, 2, 112, 192, 4),
      bottleneck_config(6, 3, 1, 192, 320, 1),
    ]


def _efficientnet_model(inverted_residual_setting: List[MBConvConfig], dropout: float) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout)
    return model


def efficientnet_b0() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.0, depth_multiplier=1.0)
    return _efficientnet_model(inverted_residual_setting, 0.2)


def efficientnet_b1() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.0, depth_multiplier=1.1)
    return _efficientnet_model(inverted_residual_setting, 0.2)


def efficientnet_b2() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.1, depth_multiplier=1.2)
    return _efficientnet_model(inverted_residual_setting, 0.3)


def efficientnet_b3() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.2, depth_multiplier=1.4)
    return _efficientnet_model(inverted_residual_setting, 0.3)


def efficientnet_b4() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.4, depth_multiplier=1.8)
    return _efficientnet_model(inverted_residual_setting, 0.4)


def efficientnet_b5() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.6, depth_multiplier=2.2)
    return _efficientnet_model(inverted_residual_setting, 0.4)


def efficientnet_b6() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=1.8, depth_multiplier=2.6)
    return _efficientnet_model(inverted_residual_setting, 0.5)


def efficientnet_b7() -> EfficientNet:
    inverted_residual_setting = _efficientnet_config(width_multiplier=2.0, depth_multiplier=3.1)
    return _efficientnet_model(inverted_residual_setting, 0.5)
