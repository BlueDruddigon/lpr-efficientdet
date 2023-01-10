from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      channel_multiplier: float = 1.0,
      norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
      act_layer: Optional[Callable[..., nn.Module]] = nn.SiLU
    ) -> None:
        super(SeparableConv2d, self).__init__()
        self.conv_depthwise = nn.Conv2d(
          in_channels, int(in_channels * channel_multiplier), kernel_size=3, strride=1, groups=in_channels, bias=False
        )
        self.conv_pointwise = nn.Conv2d(int(in_channels * channel_multiplier), out_channels, kernel_size=1, stride=1)
        self.norm_layer = norm_layer(out_channels, momentum=0.01, eps=1e-3) if norm_layer is not None else None
        self.act_layer = act_layer(inplace=True) if act_layer is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x


class ResampleFeatureMap(nn.Module):
    def __init__(self, ) -> None:
        super(ResampleFeatureMap, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
