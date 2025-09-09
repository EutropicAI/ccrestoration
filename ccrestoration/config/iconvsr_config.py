from typing import Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class IconVSRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.ICONVSR
    model: Union[ModelType, str] = ModelType.IconVSR
    spynet: Union[ConfigType, str] = ConfigType.SpyNet_spynet_sintel_final
    edvr_feature_extractor: Union[ConfigType, str] = ConfigType.EDVRFeatureExtractor_REDS_pretrained_for_IconVSR
    scale: int = 4
    length: int = 7
    num_feat: int = 64
    num_block: int = 15
    keyframe_stride: int = 5
    temporal_padding: int = 2


IconVSRConfigs = [
    IconVSRConfig(
        name=ConfigType.IconVSR_REDS_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/IconVSR_REDS_4x.pth",
        hash="aaa5367f64c6fc1e5839e13d65e596fc012608f0d22b02477324e6fe4edf99ad",
        edvr_feature_extractor=ConfigType.EDVRFeatureExtractor_REDS_pretrained_for_IconVSR,
        num_feat=64,
        num_block=30,
        temporal_padding=2,
    ),
    IconVSRConfig(
        name=ConfigType.IconVSR_Vimeo90K_BD_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/IconVSR_Vimeo90K_BD_4x.pth",
        hash="cfcb7e002dbd89b5e0e3c06c122da033e21d43698a6a819c22f9ab3cf4c91dcd",
        edvr_feature_extractor=ConfigType.EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR,
        num_feat=64,
        num_block=30,
        temporal_padding=3,
    ),
    IconVSRConfig(
        name=ConfigType.IconVSR_Vimeo90K_BI_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/IconVSR_Vimeo90K_BI_4x.pth",
        hash="35fec07c2bc278e2b9f7180cf6d2aada49d6ed216da09803235ffd07de561771",
        edvr_feature_extractor=ConfigType.EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR,
        num_feat=64,
        num_block=30,
        temporal_padding=3,
    ),
]

for cfg in IconVSRConfigs:
    CONFIG_REGISTRY.register(cfg)
