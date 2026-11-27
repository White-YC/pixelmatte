# PixelMatte

**High-Fidelity Image Matting with Pixel-Space Diffusion Transformers**

[![arXiv](https://img.shields.io/badge/arXiv-2611.34929-b31b1b.svg)](https://arxiv.org/abs/2611.34929)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://white-yc.github.io/pixelmatte/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow.svg)](https://huggingface.co/spaces/WhiteYC/PixelMatte)
[![GitHub stars](https://img.shields.io/github/stars/White-YC/pixelmatte?style=social)](https://github.com/White-YC/pixelmatte/stargazers)

Baihan Yang, Xin Lin, Yufan Wei, Baiang Li, Yuheng Liu, Juanxi Tian, Xiaohui Xie, Truong Nguyen

This repository contains the official inference and evaluation code.

PixelMatte predicts an alpha matte from an image and a trimap with a pixel-space diffusion transformer, in a single network
evaluation and without a VAE. Two modules complete the model: **LayerShift**, a learned per-layer shift of the tokens in the
unknown region, and the **High-Pass Refiner**, a small network that corrects fine detail at full resolution.

## Models

| Model | Backbone | Network input | Checkpoint |
|---|---|---|---|
| PixelMatte (HiDream-8B) | [HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) | short side 2048 | [WhiteYC/PixelMatte-HiDream-8B](https://huggingface.co/WhiteYC/PixelMatte-HiDream-8B) |
| PixelMatte (PRX-7B) | [PRX](https://huggingface.co/Photoroom/prxpixel-t2i) | short side 1024 | [WhiteYC/PixelMatte-PRX-7B](https://huggingface.co/WhiteYC/PixelMatte-PRX-7B) |

Each checkpoint is one folder with `config.json` and `model.safetensors` and contains the backbone, LayerShift and the
High-Pass Refiner. No text encoder or VAE is needed.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.10 and CUDA 12.8. Inference runs in bfloat16 and needs about 18 GB of GPU memory for a
2048 x 2048 input with the HiDream-8B model and about 16 GB for a 1024 x 1024 input with the PRX-7B model.

## Usage

Command line:

```bash
# one image
python infer.py --model WhiteYC/PixelMatte-HiDream-8B --image photo.jpg --trimap trimap.png --output alpha.png

# a folder of images and a folder of trimaps with the same file names
python infer.py --model WhiteYC/PixelMatte-HiDream-8B --image images/ --trimap trimaps/ --output mattes/
```

Python:

```python
from PIL import Image
from pixelmatte import PixelMattePipeline

pipe = PixelMattePipeline.from_pretrained("WhiteYC/PixelMatte-HiDream-8B")
alpha = pipe(Image.open("photo.jpg"), Image.open("trimap.png"))      # float32 array [H, W] in [0, 1]
```

The trimap is a single-channel image of the same size as the photo: 0 marks background, 255 foreground and any value in
between (usually 128) the unknown region. The matte is returned at the size of the input image. Pixels that the trimap marks
as foreground or background keep the values 1 and 0.

## Evaluation

The paper reports results on four test sets. Each is distributed by its authors with images, ground-truth mattes and trimaps;
we use these trimaps as they are.

| Test set | Images | Download | Folder used below |
|---|---|---|---|
| P3M-500-NP | 500 | [P3M-10k](https://github.com/JizhiziLi/P3M) | `P3M-10k/validation/P3M-500-NP` (`original_image`, `mask`, `trimap`) |
| AM-2K | 200 | [AM-2k](https://github.com/JizhiziLi/GFM) | `AM-2k/validation` (`original`, `mask`, `trimap`) |
| AIM-500 | 500 | [AIM-500](https://github.com/JizhiziLi/AIM) | `AIM-500` (`original`, `mask`, `trimap`) |
| RM-2K | 2000 | [RenderMatte dataset](https://huggingface.co/datasets/Renz-7/RenderMatte-dataset) | `RenderMatte-dataset/RenderMatte-2k` (`original`, `mask`, `trimap`) |

As in the paper, each test set is predicted twice, with the noise seeds 42 and 43, and the two scores are averaged
(adjust the paths to where you put the sets):

```bash
MODEL=WhiteYC/PixelMatte-HiDream-8B

# P3M-500-NP
for SEED in 42 43; do python infer.py --model $MODEL --seed $SEED --image P3M-10k/validation/P3M-500-NP/original_image --trimap P3M-10k/validation/P3M-500-NP/trimap --output results/p3m_500_np/seed$SEED; done
python eval.py --pred results/p3m_500_np/seed42 results/p3m_500_np/seed43 --alpha P3M-10k/validation/P3M-500-NP/mask --trimap P3M-10k/validation/P3M-500-NP/trimap

# AM-2K
for SEED in 42 43; do python infer.py --model $MODEL --seed $SEED --image AM-2k/validation/original --trimap AM-2k/validation/trimap --output results/am_2k/seed$SEED; done
python eval.py --pred results/am_2k/seed42 results/am_2k/seed43 --alpha AM-2k/validation/mask --trimap AM-2k/validation/trimap

# AIM-500
for SEED in 42 43; do python infer.py --model $MODEL --seed $SEED --image AIM-500/original --trimap AIM-500/trimap --output results/aim_500/seed$SEED; done
python eval.py --pred results/aim_500/seed42 results/aim_500/seed43 --alpha AIM-500/mask --trimap AIM-500/trimap

# RM-2K
for SEED in 42 43; do python infer.py --model $MODEL --seed $SEED --image RenderMatte-dataset/RenderMatte-2k/original --trimap RenderMatte-dataset/RenderMatte-2k/trimap --output results/rm_2k/seed$SEED; done
python eval.py --pred results/rm_2k/seed42 results/rm_2k/seed43 --alpha RenderMatte-dataset/RenderMatte-2k/mask --trimap RenderMatte-dataset/RenderMatte-2k/trimap
```

Set `MODEL=WhiteYC/PixelMatte-PRX-7B` for the PRX model. `eval.py` prints the score of each run and their mean. All four
metrics are computed over the whole image; `eval.py` documents their definitions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=white-yc/pixelmatte&type=Date)](https://star-history.com/#white-yc/pixelmatte&Date)

## Citation

```bibtex
@misc{yang2026pixelmatte,
  title         = {High-Fidelity Image Matting with Pixel-Space Diffusion Transformers},
  author        = {Baihan Yang and Xin Lin and Yufan Wei and Baiang Li and Yuheng Liu and Juanxi Tian and Xiaohui Xie and Truong Nguyen},
  year          = {2026},
  eprint        = {2611.34929},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.34929}
}
```

## Acknowledgements

The backbones are [HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) (MIT) and
[PRX](https://huggingface.co/Photoroom/prxpixel-t2i) (Apache-2.0). The folder `pixelmatte/third_party/hidream` contains the
unmodified model code of HiDream-O1-Image under its own licence; `pixelmatte/prx.py` follows the PRX transformer of the
[diffusers](https://github.com/huggingface/diffusers) library.

## Licence

The code and the model weights are released under the [Apache License 2.0](LICENSE). The folder
`pixelmatte/third_party/hidream` keeps the MIT licence of HiDream-O1-Image.
