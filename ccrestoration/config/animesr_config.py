from typing import Tuple, Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class AnimeSRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.MSRSWVSR
    model: Union[ModelType, str] = ModelType.AnimeSR
    scale: int = 4
    length: int = 3
    num_feat: int = 64
    num_block: Tuple[int, int, int] = (5, 3, 2)


AnimeSRConfigs = [
    AnimeSRConfig(
        name=ConfigType.AnimeSR_v1_PaperModel_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/AnimeSR_v1_PaperModel_4x.pth",
        hash="915ef7f0f7067f04219516b50e88c362581300e48902e3b7f540650e32a20c10",
        scale=4,
    ),
    AnimeSRConfig(
        name=ConfigType.AnimeSR_v2_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/AnimeSR_v2_4x.pth",
        hash="d0f29c8966b53718828bd424bbdc306e7ff0cbf6350beadaf8b5b2500b108548",
        scale=4,
    ),
]

for cfg in AnimeSRConfigs:
    CONFIG_REGISTRY.register(cfg)
