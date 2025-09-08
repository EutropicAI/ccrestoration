from typing import Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class BasicVSRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.BASICVSR
    model: Union[ModelType, str] = ModelType.BasicVSR
    spynet: Union[ConfigType, str] = ConfigType.SpyNet_spynet_sintel_final
    scale: int = 4
    length: int = 7
    num_feat: int = 64
    num_block: int = 30


BasicVSRConfigs = [
    BasicVSRConfig(
        name=ConfigType.BasicVSR_REDS_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/BasicVSR_REDS_4x.pth",
        hash="543c826113a9efdc971572320a9e259833fc230f843d94f7ef7270c92b5ea4dc",
    ),
    BasicVSRConfig(
        name=ConfigType.BasicVSR_Vimeo90K_BD_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/BasicVSR_Vimeo90K_BD_4x.pth",
        hash="e9bf46ebb478abfc3002572ae67194f6b99fed5070262d75ec537f9b9df4477e",
    ),
    BasicVSRConfig(
        name=ConfigType.BasicVSR_Vimeo90K_BI_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/BasicVSR_Vimeo90K_BI_4x.pth",
        hash="2a29695a2607a10db193234cae388a2b486862aacf636e0b6f8bc218a9cba401",
    ),
]

for cfg in BasicVSRConfigs:
    CONFIG_REGISTRY.register(cfg)
