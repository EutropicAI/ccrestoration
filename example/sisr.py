import cv2
import numpy as np

from ccrestoration import ArchType, AutoConfig, AutoModel, BaseConfig, ConfigType, SRBaseModel
from ccrestoration.config import RealESRGANConfig

example = 4

if example == 0:
    # fast load a pre-trained model
    model: SRBaseModel = AutoModel.from_pretrained(ConfigType.RealESRGAN_APISR_RRDB_GAN_generator_2x)
elif example == 1:
    # edit the configuration
    config: BaseConfig = AutoConfig.from_pretrained(
        pretrained_model_name=ConfigType.RealESRGAN_APISR_RRDB_GAN_generator_2x
    )
    print(config)
    config.scale = 2
    model: SRBaseModel = AutoModel.from_config(config=config)
elif example == 3:
    # use your own configuration
    config = RealESRGANConfig(
        name="114514.pth",
        url="https://github.com/EutropicAI/ccrestoration/releases/download/model_zoo/RealESRGAN_APISR_RRDB_GAN_generator_2x.pth",
        hash="3b0d2b3a3c0461ac17d00f4f32240666fb832b738ea5a48449b1acf07fbb07e5",
        arch=ArchType.RRDB,
        scale=2,
        num_block=6,
    )
    model: SRBaseModel = AutoModel.from_config(config=config)
elif example == 4:
    # use custom model dir and gh proxy
    model: SRBaseModel = AutoModel.from_pretrained(
        pretrained_model_name=ConfigType.RealESRGAN_APISR_RRDB_GAN_generator_2x,
        model_dir="./",
        gh_proxy="https://github.abskoop.workers.dev/",
    )


else:
    raise ValueError("example not found")


img = cv2.imdecode(np.fromfile("../assets/test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = model.inference_image(img)
cv2.imwrite("../assets/test_sisr_out.jpg", img)
