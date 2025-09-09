from typing import Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class EDSRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.EDSR
    model: Union[ModelType, str] = ModelType.EDSR
    scale: int = 2
    num_in_ch: int = 3
    num_out_ch: int = 3
    num_feat: int = 64
    num_block: int = 16
    res_scale: int = 1
    img_range: float = 255.0
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040)


EDSRConfigs = [
    # Official Medium size models
    EDSRConfig(
        name=ConfigType.EDSR_Mx2_f64b16_DIV2K_official_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDSR_Mx2_f64b16_DIV2K_official_2x.pth",
        hash="3ba7b0861913de93740110738fb621410651897e391e8057b7b6104c4f999254",
        scale=2,
    ),
    EDSRConfig(
        name=ConfigType.EDSR_Mx3_f64b16_DIV2K_official_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDSR_Mx3_f64b16_DIV2K_official_3x.pth",
        hash="6908f88a1be95e7112f480b7b1d9608ad83b4ffa0c227416a6376f6b036a77f3",
        scale=3,
    ),
    EDSRConfig(
        name=ConfigType.EDSR_Mx4_f64b16_DIV2K_official_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDSR_Mx4_f64b16_DIV2K_official_4x.pth",
        hash="0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f",
        scale=4,
    ),
    # Official Large size models
    # EDSRConfig(
    #     name=ConfigType.EDSR_Lx2_f256b32_DIV2K_official_2x,
    #     url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDSR_Lx2_f256b32_DIV2K_official_2x.pth",
    #     hash="be38e77dcff9ec95225cea6326b5a616d57869824688674da317df37f3d87d1b",
    #     scale=2,
    #     num_feat=256,
    #     num_block=32,
    # ),
    # EDSRConfig(
    #     name=ConfigType.EDSR_Lx3_f256b32_DIV2K_official_3x,
    #     url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDSR_Lx3_f256b32_DIV2K_official_3x.pth",
    #     hash="3660f70d306481ef4867731500c4e3d901a1b8547996cf4245a09ffbc151b70b",
    #     scale=3,
    #     num_feat=256,
    #     num_block=32,
    # ),
    # EDSRConfig(
    #     name=ConfigType.EDSR_Lx4_f256b32_DIV2K_official_4x,
    #     url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/EDSR_Lx4_f256b32_DIV2K_official_4x.pth",
    #     hash="76ee1c8f48813f46024bee8d2700f417f6b2db070e899954ff1552fbae343e93",
    #     scale=4,
    #     num_feat=256,
    #     num_block=32,
    # ),
]

for cfg in EDSRConfigs:
    CONFIG_REGISTRY.register(cfg)
