from typing import Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class SpyNetConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.SPYNET
    model: Union[ModelType, str] = ModelType.SpyNet


SpyNetConfigs = [
    # BasicSR SpyNet
    SpyNetConfig(
        name=ConfigType.SpyNet_spynet_sintel_final,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SpyNet_spynet_sintel_final.pth",
        hash="3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714",
    ),
]

for cfg in SpyNetConfigs:
    CONFIG_REGISTRY.register(cfg)
