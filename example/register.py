from typing import Any

from ccrestoration import AutoConfig, AutoModel, SRBaseModel
from ccrestoration.arch import RRDBNet
from ccrestoration.config import RealESRGANConfig

# define your own config name and model name
cfg_name = "TESTCONFIG.pth"
model_name = "TESTMODEL"

# this should be your own config, not RealESRGANConfig
# extend from ccrestoration.BaseConfig then implement your own config parameters
cfg = RealESRGANConfig(
    name=cfg_name,
    url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth",
    arch="RRDB",
    model=model_name,
    scale=4,
    num_block=6,
)

AutoConfig.register(cfg)


# this should be your own model
# extend from ccrestoration.SRBaseModel or ccrestoration.VSRBaseModel then implement your own model
# self.one_frame_out: bool = False  for this kind of vsr model: f1, f2, f3, f4 -> f1', f2', f3', f4'
# self.one_frame_out: bool = True  for this kind of vsr model: f-2, f-1, f0, f1, f2 -> f0'
# override self.one_frame_out in self.load_model() if you want
@AutoModel.register(name=model_name)
class TESTMODEL(SRBaseModel):
    def load_model(self) -> Any:
        cfg: RealESRGANConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "model_state_dict" in state_dict:
            # For APISR's model
            state_dict = state_dict["model_state_dict"]

        model = RRDBNet(
            num_in_ch=cfg.num_in_ch,
            num_out_ch=cfg.num_out_ch,
            scale=cfg.scale,
            num_feat=cfg.num_feat,
            num_block=cfg.num_block,
            num_grow_ch=cfg.num_grow_ch,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model


model: TESTMODEL = AutoModel.from_pretrained(cfg_name)
