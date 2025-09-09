from typing import Any, List, Optional, Tuple, Union

from pydantic import field_validator
from torch import nn

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class HATConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.HAT
    model: Union[ModelType, str] = ModelType.HAT
    scale: int = 2
    patch_size: int = 1
    in_chans: int = 3
    img_size: Union[int, Tuple[int, ...]] = 64
    img_range: float = 1.0
    depth: Union[List[int], Tuple[int, ...]] = (6, 6, 6, 6)
    embed_dim: int = 96
    num_heads: Union[List[int], Tuple[int, ...]] = (6, 6, 6, 6)
    window_size: int = 7
    compress_ratio: int = 3
    squeeze_factor: int = 30
    conv_scale: float = 0.01
    overlap_ratio: float = 0.5
    mlp_ratio: float = 4.0
    resi_connection: str = "1conv"
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    ape: bool = False
    patch_norm: bool = True
    act_layer: Any = nn.GELU
    norm_layer: Any = nn.LayerNorm
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


HATConfigs = [
    # official models
    # hat_s
    HATConfig(
        name=ConfigType.HAT_S_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_S_2x.pth",
        hash="3e249a901c2aed6b82548875e21847ef6c015a40c814237a7a0abb10c69d5ddf",
        scale=2,
        in_chans=3,
        window_size=16,
        compress_ratio=24,
        squeeze_factor=24,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=144,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_S_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_S_3x.pth",
        hash="3733345caeb2b6cb0a7514d634b143f318e2441faf32316195a47e4cda67669e",
        scale=3,
        in_chans=3,
        window_size=16,
        compress_ratio=24,
        squeeze_factor=24,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=144,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_S_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_S_4x.pth",
        hash="a92f81bd2c0c1aaa371a6e4d6cac69e749fde2e36196885ee47a4a3667542c9a",
        scale=4,
        in_chans=3,
        window_size=16,
        compress_ratio=24,
        squeeze_factor=24,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=144,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    # hat_m
    HATConfig(
        name=ConfigType.HAT_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_2x.pth",
        hash="35e9409849016ab2bfc0c549639ea4f1d4d5b2bdd75f856a925e124dd05670d0",
        scale=2,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_3x.pth",
        hash="453c90b94d494d041c8bff2d222987af311f07b4a09b4dafbd9ab6492f15206c",
        scale=3,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_4x.pth",
        hash="02dabea478aa5902a7170ad89350124e691bd89c91356f24b3267022622dc030",
        scale=4,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    # hat_l
    HATConfig(
        name=ConfigType.HAT_L_ImageNet_pretrain_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_L_ImageNet_pretrain_2x.pth",
        hash="2818c7ca8d72ec4cc5f31c93203d55252a662dd35cda34ce1a69661f97dcd38f",
        scale=2,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_L_ImageNet_pretrain_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_L_ImageNet_pretrain_3x.pth",
        hash="78af181bedf1e805fd7517d3738bc5824f7ebdc477fc9f757b708bcf49ad4e3d",
        scale=3,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_L_ImageNet_pretrain_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_L_ImageNet_pretrain_4x.pth",
        hash="5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043",
        scale=4,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    # hat_image_net
    HATConfig(
        name=ConfigType.HAT_ImageNet_pretrain_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_ImageNet_pretrain_2x.pth",
        hash="82ebd911263bcc886fbef46b30cf97b92a932a27a3cba30163d4577afb09b9d7",
        scale=2,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_ImageNet_pretrain_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_ImageNet_pretrain_3x.pth",
        hash="8469db903d464497f94419587054c9e83dff994edc409d563ab8f5a503767be8",
        scale=3,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_ImageNet_pretrain_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_ImageNet_pretrain_4x.pth",
        hash="4ee053c42461187846dc0e93aa5abd34591c0725a8e044a59000e92ee215e833",
        scale=4,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    # hat_real_gan
    HATConfig(
        name=ConfigType.HAT_Real_GAN_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_Real_GAN_4x.pth",
        hash="f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7",
        scale=4,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    HATConfig(
        name=ConfigType.HAT_Real_GAN_sharper_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/HAT_Real_GAN_sharper_4x.pth",
        hash="5800b67136006eb8cab3b4ed7c8d73b6a195bb18e6cc709b674f9aa069c00271",
        scale=4,
        in_chans=3,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
]

for cfg in HATConfigs:
    CONFIG_REGISTRY.register(cfg)
