from typing import Any

from ccrestoration import AutoConfig, AutoModel
from ccrestoration.config import RealESRGANConfig
from ccrestoration.model import RealESRGANModel


def test_auto_class_register() -> None:
    cfg_name = "TESTCONFIG.pth"
    model_name = "TESTMODEL"

    cfg = RealESRGANConfig(
        name=cfg_name,
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth",
        arch="RRDB",
        model=model_name,
        scale=4,
        num_block=6,
    )

    AutoConfig.register(cfg)

    @AutoModel.register(name=model_name)
    class TESTMODEL(RealESRGANModel):
        def get_cfg(self) -> Any:
            return self.config

    model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
    assert model.get_cfg() == cfg
