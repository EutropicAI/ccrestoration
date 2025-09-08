from typing import Union

from pydantic import field_validator

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class RealCUGANConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.UPCUNET
    model: Union[ModelType, str] = ModelType.RealCUGAN
    scale: int = 2
    in_channels: int = 3
    out_channels: int = 3
    cache_mode: int = 0
    alpha: float = 1
    pro: bool = False

    @field_validator("scale")
    def scale_match(cls, v: int) -> int:
        if v not in [2, 3, 4]:
            raise ValueError("real cugan scale must be one of 2, 3, 4")
        return v

    @field_validator("cache_mode")
    def cache_mode_match(cls, v: int) -> int:
        if v not in [0, 1, 2, 3]:
            raise ValueError("cache mode must be one of 0, 1, 2, 3")
        return v

    @field_validator("alpha")
    def alpha_match(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("alpha must be in [0, 2]")
        return v


RealCUGANConfigs = [
    # RealCUGAN
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Conservative_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Conservative_2x.pth",
        hash="6cfe3b23687915d08ba96010f25198d9cfe8a683aa4131f1acf7eaa58ee1de93",
        scale=2,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Denoise1x_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Denoise1x_2x.pth",
        hash="2e783c39da6a6394fbc250fdd069c55eaedc43971c4f2405322f18949ce38573",
        scale=2,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Denoise2x_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Denoise2x_2x.pth",
        hash="8188b3faef4258cf748c59360cbc8086ebedf4a63eb9d5d6637d45f819d32496",
        scale=2,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Denoise3x_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Denoise3x_2x.pth",
        hash="0a14739f3f5fcbd74ec3ce2806d13a47916c916b20afe4a39d95f6df4ca6abd8",
        scale=2,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_No_Denoise_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_No_Denoise_2x.pth",
        hash="f491f9ecf6964ead9f3a36bf03e83527f32c6a341b683f7378ac6c1e2a5f0d16",
        scale=2,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Conservative_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Conservative_3x.pth",
        hash="f6ea5fd20380413beb2701182483fd80c2e86f3b3f08053eb3df4975184aefe3",
        scale=3,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Denoise3x_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Denoise3x_3x.pth",
        hash="39f1e6e90d50e5528a63f4ba1866bad23365a737cbea22a80769b2ec4c1c3285",
        scale=3,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_No_Denoise_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_No_Denoise_3x.pth",
        hash="763f0a87e70d744673f1a41db5396d5f334d22de97fff68ffc40deb91404a584",
        scale=3,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Conservative_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Conservative_4x.pth",
        hash="a8c8185def699b0883662a02df0ef2e6db3b0275170b6cc0d28089b64b273427",
        scale=4,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Denoise3x_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Denoise3x_4x.pth",
        hash="42bd8fcdae37c12c5b25ed59625266bfa65780071a8d38192d83756cb85e98dd",
        scale=4,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_No_Denoise_4x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_No_Denoise_4x.pth",
        hash="aaf3ef78a488cce5d3842154925eb70ff8423b8298e2cd189ec66eb7f6f66fae",
        scale=4,
    ),
    # RealCUGAN Pro
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_Conservative_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_Conservative_2x.pth",
        hash="b8ae5225d2d515aa3c33ef1318aadc532a42ea5ed8d564471b5a5b586783e964",
        scale=2,
        pro=True,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_Denoise3x_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_Denoise3x_2x.pth",
        hash="e80ca8fc7c261e3dc8f4c0ce0656ac5501d71a476543071615c43392dbeb4c0d",
        scale=2,
        pro=True,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_No_Denoise_2x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_No_Denoise_2x.pth",
        hash="ccce1f535d94c50ce38e268a53687bc7e68ef7215e3c5e6b3bfd1bfc1dacf0fa",
        scale=2,
        pro=True,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_Conservative_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_Conservative_3x.pth",
        hash="a9f3c783a04b15c793b95e332bfdac524cfa30ba186cb829c1290593e28ad9e7",
        scale=3,
        pro=True,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_Denoise3x_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_Denoise3x_3x.pth",
        hash="4ddd14e2430db0d75d186c6dda934db34929c50da8a88a0c6f4accb871fe4b70",
        scale=3,
        pro=True,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_No_Denoise_3x,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_No_Denoise_3x.pth",
        hash="c14d693a6d3316b8a3eba362e7576f178aea3407e1d89ca0bcb34e1c61269b0f",
        scale=3,
        pro=True,
    ),
]

for cfg in RealCUGANConfigs:
    CONFIG_REGISTRY.register(cfg)
