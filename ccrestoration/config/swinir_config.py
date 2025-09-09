from typing import Any, List, Optional, Tuple, Union

from pydantic import field_validator
from torch import nn

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class SwinIRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.SWINIR
    model: Union[ModelType, str] = ModelType.SwinIR
    scale: int = 2
    img_size: Union[int, Tuple[int, ...]] = 64
    patch_size: Union[int, Tuple[int, ...]] = 1
    in_chans: int = 3
    embed_dim: int = 96
    depths: Union[List[int], Tuple[int, ...]] = [6, 6, 6, 6]  # noqa
    num_heads: Union[List[int], Tuple[int, ...]] = [6, 6, 6, 6]  # noqa
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Any = nn.LayerNorm
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    img_range: float = 1.0
    upsampler: str = ""
    resi_connection: str = "1conv"

    @field_validator("scale")
    def scale_match(cls, v: int) -> int:
        if v not in [1, 2, 3, 4, 8]:
            raise ValueError("Upscale factor must be one of [1, 2, 3, 4, 8]")
        return v

    @field_validator("upsampler")
    def upsampler_match(cls, v: str) -> str:
        if v not in ["pixelshuffle", "pixelshuffledirect", "nearest+conv", ""]:
            raise ValueError("Upsampler must be one of ['pixelshuffle','pixelshuffledirect','nearest+conv', '']")
        return v

    @field_validator("resi_connection")
    def resi_connection_match(cls, v: str) -> str:
        if v not in ["1conv", "3conv"]:
            raise ValueError("Residual connection must be one of ['1conv', '3conv']")
        return v


SwinIRConfigs = [
    # official models
    SwinIRConfig(
        name=ConfigType.SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x.pth",
        hash="2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac",
        scale=2,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2,
        upsampler="pixelshuffle",
    ),
    SwinIRConfig(
        name=ConfigType.SwinIR_lightweightSR_DIV2K_s64w8_SwinIR_S_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SwinIR_lightweightSR_DIV2K_s64w8_SwinIR_S_2x.pth",
        hash="193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7",
        scale=2,
        window_size=8,
        embed_dim=60,
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
    ),
    SwinIRConfig(
        name=ConfigType.SwinIR_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_GAN_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SwinIR_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_GAN_4x.pth",
        hash="99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a",
        scale=4,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        embed_dim=240,
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="3conv",
    ),
    SwinIRConfig(
        name=ConfigType.SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_2x.pth",
        hash="f397408977a3e07eb06afb7238d453a12ef35ebab7328a54241f307860dbe342",
        scale=2,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2,
        upsampler="nearest+conv",
    ),
    SwinIRConfig(
        name=ConfigType.SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_4x.pth",
        hash="b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6",
        scale=4,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2,
        upsampler="nearest+conv",
    ),
    # community models
    SwinIRConfig(
        name=ConfigType.SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x.pth",
        hash="aea80c061b41e1cbc5b0c0f9bb2603a82c7d00d2451bfd5b98a495244dd5fb2f",
        scale=2,
        img_size=32,
        window_size=8,
        embed_dim=60,
        mlp_ratio=2,
        upsampler="pixelshuffle",
    ),
]

for cfg in SwinIRConfigs:
    CONFIG_REGISTRY.register(cfg)
