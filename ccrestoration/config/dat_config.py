from typing import Any, List, Optional, Tuple, Union

from pydantic import field_validator
from torch import nn

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class DATConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.DAT
    model: Union[ModelType, str] = ModelType.DAT
    scale: int = 2
    in_chans: int = 3
    img_size: Union[int, Tuple[int, ...]] = 64
    img_range: float = 1.0
    split_size: Union[List[int], Tuple[int, ...]] = [2, 4]  # noqa
    depth: Union[List[int], Tuple[int, ...]] = [6, 6, 6, 6, 6, 6]  # noqa
    embed_dim: int = 180
    num_heads: Union[List[int], Tuple[int, ...]] = [6, 6, 6, 6, 6, 6]  # noqa
    expansion_factor: float = 4.0
    resi_connection: str = "1conv"
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    act_layer: Any = nn.GELU
    norm_layer: Any = nn.LayerNorm
    use_chk: bool = False
    upsampler: str = "pixelshuffle"

    @field_validator("scale")
    def scale_match(cls, v: int) -> int:
        if v not in [1, 2, 3, 4, 8]:
            raise ValueError("scale factor must be one of [1, 2, 3, 4, 8]")
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


DATConfigs = [
    # official models
    # dat_s
    DATConfig(
        name=ConfigType.DAT_S_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_S_2x.pth",
        hash="160330dd8a40b141e12713ca9bfde09e36a03c533b455965f157d023672cb794",
        scale=2,
        split_size=[8, 16],
        expansion_factor=2,
    ),
    DATConfig(
        name=ConfigType.DAT_S_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_S_3x.pth",
        hash="d1446d3eb2fbaad472c6fd6c2ac03a3467265c5f93822a741b09838b80f18b62",
        scale=3,
        split_size=[8, 16],
        expansion_factor=2,
    ),
    DATConfig(
        name=ConfigType.DAT_S_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_S_4x.pth",
        hash="2ba8a7cbe2fd88f3499d08d1b353fa17a98b815832f4426a7144a0ef9f3bfcf7",
        scale=4,
        split_size=[8, 16],
        expansion_factor=2,
    ),
    # dat_m
    DATConfig(
        name=ConfigType.DAT_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_2x.pth",
        hash="7760aa96e4ee77e29d4f89c3a4486200042e019461fdb8aa286f49aa00b89b51",
        scale=2,
        split_size=[8, 32],
        expansion_factor=4,
    ),
    DATConfig(
        name=ConfigType.DAT_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_3x.pth",
        hash="581973e02c06f90d4eb90acf743ec9604f56f3c2c6f9e1e2c2b38ded1f80d197",
        scale=3,
        split_size=[8, 32],
        expansion_factor=4,
    ),
    DATConfig(
        name=ConfigType.DAT_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_4x.pth",
        hash="391a6ce69899dff5ea3214557e9d585608254579217169faf3d4c353caff049e",
        scale=4,
        split_size=[8, 32],
        expansion_factor=4,
    ),
    # dat_2
    DATConfig(
        name=ConfigType.DAT_2_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_2_2x.pth",
        hash="aea2c76996c2b3e7cb034380e07738608bd59cc34c667331df7269e4b670ac18",
        scale=2,
        split_size=[8, 32],
        expansion_factor=2,
    ),
    DATConfig(
        name=ConfigType.DAT_2_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_2_3x.pth",
        hash="e19fbb2e6addf5cecf90472937fabdd905853b79c4ef807b4dd184c30bb22a28",
        scale=3,
        split_size=[8, 32],
        expansion_factor=2,
    ),
    DATConfig(
        name=ConfigType.DAT_2_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_2_4x.pth",
        hash="05b5c17bb5d1939ec0ec6b9368368d82d8c45b80c134e370f798efec0aeec395",
        scale=4,
        split_size=[8, 32],
        expansion_factor=2,
    ),
    # dat_light
    DATConfig(
        name=ConfigType.DAT_light_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_light_2x.pth",
        hash="3888bf067c1a3790adb64210e108690859f4900ffd158fe39a2cfe057ae8300c",
        scale=2,
        split_size=[8, 32],
        expansion_factor=2,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        resi_connection="3conv",
        upsampler="pixelshuffledirect",
    ),
    DATConfig(
        name=ConfigType.DAT_light_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_light_3x.pth",
        hash="dfb99deab865db05771080c73c50a06b67c9499c3a27ca64ab3729fa573d5cfe",
        scale=3,
        split_size=[8, 32],
        expansion_factor=2,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        resi_connection="3conv",
        upsampler="pixelshuffledirect",
    ),
    DATConfig(
        name=ConfigType.DAT_light_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_light_4x.pth",
        hash="e18223fc41500e72e6bd576f2a576eae0818f58fa8cc8646a2233f5faf7d0f74",
        scale=4,
        split_size=[8, 32],
        expansion_factor=2,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        resi_connection="3conv",
        upsampler="pixelshuffledirect",
    ),
    # community models
    DATConfig(
        name=ConfigType.DAT_APISR_GAN_generator_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/DAT_APISR_GAN_generator_4x.pth",
        hash="cc625c4ec5242a57e46f0941a3c36e3a731dcd859750c5f72a5251045b1e6d72",
        scale=4,
        split_size=[8, 16],
        expansion_factor=2,
        upsampler="pixelshuffledirect",
    ),
]

for cfg in DATConfigs:
    CONFIG_REGISTRY.register(cfg)
