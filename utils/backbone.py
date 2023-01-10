from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ['ConvNormActivation', 'SqueezeExcitation', 'StochasticDepth']


class ConvNormActivation(nn.Sequential):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      padding: Optional[Union[int, Tuple[int, int]]] = None,
      groups: int = 1,
      norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
      act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
      dilation: int = 1,
      inplace: bool = True,
      bias: bool = False
    ) -> None:
        if padding is None:
            padding = (kernel_size-1) // 2 * dilation
        layers = [
          nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias or norm_layer is None
          )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels, momentum=0.01, eps=1e-3))
        if act_layer is not None:
            layers.append(act_layer(inplace=inplace))
        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_channels
        self.in_channels = in_channels


class SqueezeExcitation(nn.Module):
    def __init__(
      self,
      input_channels: int,
      squeeze_channels: int,
      activation: Callable[..., nn.Module] = nn.ReLU,
      scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
    
    def _scale(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(inputs)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self._scale(inputs)
        return scale * inputs


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super(StochasticDepth, self).__init__()
        self.p = p
        self.mode = mode
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.p < 0.0 or self.p > 1.0:
            raise ValueError(f'drop probability has to be between 0 and 1, but got {self.p}')
        if self.mode not in ["batch", "row"]:
            raise ValueError(f"mode has to be either 'batch' or 'row', but got {self.mode}")
        if not self.training or self.p == 0.0:
            return inputs
        
        survival_rate = 1.0 - self.p
        if self.mode == 'row':
            size = [inputs.shape[0]] + [1] * (inputs.ndim - 1)
        else:
            size = [1] * inputs.ndim
        noise = torch.empty(size, dtype=inputs.dtype, device=inputs.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return inputs * noise
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, mode={self.mode})'
