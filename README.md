# ccrestoration

[![codecov](https://codecov.io/gh/EutropicAI/ccrestoration/graph/badge.svg?token=VK0BHDUXAI)](https://codecov.io/gh/EutropicAI/ccrestoration)
[![CI-test](https://github.com/EutropicAI/ccrestoration/actions/workflows/CI-test.yml/badge.svg)](https://github.com/EutropicAI/ccrestoration/actions/workflows/CI-test.yml)
[![Release-pypi](https://github.com/EutropicAI/ccrestoration/actions/workflows/Release.yml/badge.svg)](https://github.com/EutropicAI/ccrestoration/actions/workflows/Release.yml)
[![PyPI version](https://badge.fury.io/py/ccrestoration.svg)](https://badge.fury.io/py/ccrestoration)
![GitHub](https://img.shields.io/github/license/EutropicAI/ccrestoration)

an inference lib for image/video restoration with VapourSynth support, compatible with [many community models](https://openmodeldb.info/)

### Install

Make sure you have Python >= 3.9 and PyTorch >= 1.13 installed

```bash
pip install ccrestoration
```

- Install VapourSynth (optional)

### Start

#### cv2

a simple example to use the SISR (Single Image Super-Resolution) model to process an image (APISR)

```python
import cv2
import numpy as np

from ccrestoration import AutoModel, ConfigType, SRBaseModel

model: SRBaseModel = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.RealESRGAN_APISR_RRDB_GAN_generator_2x,
)

img = cv2.imdecode(np.fromfile("test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = model.inference_image(img)
cv2.imwrite("test_out.jpg", img)
```

#### VapourSynth

a simple example to use the VSR (Video Super-Resolution) model to process a video (AnimeSR)

```python
import vapoursynth as vs
from vapoursynth import core

from ccrestoration import AutoModel, BaseModelInterface, ConfigType

model: BaseModelInterface = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
    tile=None
)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
```

See more examples in the [example](./example) directory, ccrestoration can register custom configurations and models to extend the functionality

### Current Support

It still in development, the following models are supported:

- [Architecture](./ccrestoration/type/arch.py)
- [Model](./ccrestoration/type/model.py)
- [Weight(Config)](./ccrestoration/type/config.py)

### Notice

- All the architectures have been edited to normalize input and output, and automatic padding is applied. The input and output tensor shapes may differ from the original architectures. For SR models, the input and output are both 4D tensors in the shape of `(b, c, h, w)`. For VSR models, the input and output are both 5D tensors in the shape of `(b, l, c, h, w)`.

- For VSR models with equal l in input and output `(f1, f2, f3, f4 -> f1', f2', f3', f4')`, you can directly extend from `class VSRBaseModel`. For VSR models that output only one frame `(f-2, f-1, f0, f1, f2 -> f0')`, you also need to set `self.one_frame_out = True`.

### Reference

- [PyTorch](https://github.com/pytorch/pytorch)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [huggingface transformers](https://github.com/huggingface/transformers)
- [VapourSynth](https://www.vapoursynth.com/)
- [HolyWu's functions](https://github.com/HolyWu)

### License

This project is licensed under the MIT - see
the [LICENSE file](./LICENSE) for details.
