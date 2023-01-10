from typing import Any

from .efficientnet import *

__all__ = ['backbone_builder']

backbone_mappings = {
  'efficientnet-b0': efficientnet_b0,
  'efficientnet-b1': efficientnet_b1,
  'efficientnet-b2': efficientnet_b2,
  'efficientnet-b3': efficientnet_b3,
  'efficientnet-b4': efficientnet_b4,
  'efficientnet-b5': efficientnet_b5,
  'efficientnet-b6': efficientnet_b6,
  'efficientnet-b7': efficientnet_b7,
}


def backbone_builder(model_name: str, **kwargs: Any) -> EfficientNet:
    assert model_name in backbone_mappings.keys()
    return backbone_mappings[model_name]()
