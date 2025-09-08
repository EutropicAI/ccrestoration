from typing import Optional, Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class EDVRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.EDVR
    model: Union[ModelType, str] = ModelType.EDVR
    scale: int = 4
    length: int = 5
    num_in_ch: int = 3
    num_out_ch: int = 3
    num_feat: int = 64
    deformable_groups: int = 8
    num_extract_block: int = 5
    num_reconstruct_block: int = 10
    center_frame_idx: Optional[int] = None
    hr_in: bool = False
    with_predeblur: bool = False
    with_tsa: bool = True


EDVRConfigs = [
    # Official Medium size models
    EDVRConfig(
        name=ConfigType.EDVR_M_SR_REDS_official_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDVR_M_SR_REDS_official_4x.pth",
        hash="32075921635eb07d56c83ec4009341bd82d882900ca7043f9c51cc7951033efd",
        scale=4,
    ),
    EDVRConfig(
        name=ConfigType.EDVR_M_woTSA_SR_REDS_official_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDVR_M_woTSA_SR_REDS_official_4x.pth",
        hash="1edf645c117ba34ea1c7fcf5506079cf9dd361c2bd26b6f3cad9197a0ba7adbf",
        scale=4,
        with_tsa=False,
    ),
]

for cfg in EDVRConfigs:
    CONFIG_REGISTRY.register(cfg)


# ----------------------------------------------------------------------------------------------------------------------


class EDVRFeatureExtractorConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.EDVRFEATUREEXTACTOR
    model: Union[ModelType, str] = ModelType.EDVRFeatureExtractor
    num_input_frame: int = 5
    num_feat: int = 64


EDVRFeatureExtractorConfigs = [
    # BasicSR SpyNet
    EDVRFeatureExtractorConfig(
        name=ConfigType.EDVRFeatureExtractor_REDS_pretrained_for_IconVSR,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDVRFeatureExtractor_REDS_pretrained_for_IconVSR.pth",
        hash="f62a2f1ea25dfa0d38d82b58beef3888edcc5433b8d39ae599bc4e1d1b261e15",
        num_input_frame=5,
        num_feat=64,
    ),
    EDVRFeatureExtractorConfig(
        name=ConfigType.EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR.pth",
        hash="ee48ee9240055f86a1c55cf3fbe8d7802bf017409fda9f9ea2ca09cfb57bcfe7",
        num_input_frame=7,
        num_feat=64,
    ),
]

for cfg2 in EDVRFeatureExtractorConfigs:
    CONFIG_REGISTRY.register(cfg2)
