from typing import Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class SRCNNConfig(BaseConfig):
    arch: ArchType = ArchType.SRCNN
    model: Union[ModelType, str] = ModelType.SRCNN
    scale: int = 2
    num_channels: int = 1


SRCNNConfigs = [
    SRCNNConfig(
        name=ConfigType.SRCNN_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SRCNN_2x.pth",
        hash="e803ec6e0230ae12b1fa7fd1c67bd57d2e744b4f4fbbc861bf9790070fc4d19e",
        scale=2,
    ),
    SRCNNConfig(
        name=ConfigType.SRCNN_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SRCNN_3x.pth",
        hash="364ec936313d0fd1052c641b20cefd8153a2c1d89712f357f804f0119ab7ab90",
        scale=3,
    ),
    SRCNNConfig(
        name=ConfigType.SRCNN_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SRCNN_4x.pth",
        hash="f07978e521ede367d55ef7ca83f4f4979e2339c594bead101cfbb9611023f70e",
        scale=4,
    ),
]

for cfg in SRCNNConfigs:
    CONFIG_REGISTRY.register(cfg)
