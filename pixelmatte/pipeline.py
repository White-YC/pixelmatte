"""PixelMatte inference: image + trimap -> alpha matte, in one network evaluation."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from .refiner import HighPassRefiner
from .trimap import BACKGROUND, FOREGROUND, normalize_trimap


def _working_size(width: int, height: int, short_side: int, multiple: int, max_long_side: int = 0):
    """Size the network runs at: the short side becomes `short_side`, the aspect ratio is kept, both sides are multiples of the
    patch size. `max_long_side` (0 = no limit) bounds the longer side for very elongated images."""
    scale = short_side / min(width, height)
    if max_long_side and max(width, height) * scale > max_long_side:
        scale = max_long_side / max(width, height)
    return max(multiple, round(height * scale / multiple) * multiple), max(multiple, round(width * scale / multiple) * multiple)


class PixelMattePipeline:
    def __init__(self, model, refiner, config: dict, device, dtype=torch.bfloat16):
        self.model, self.refiner, self.config = model, refiner, config
        self.device, self.dtype = torch.device(device), dtype
        self.prompt_embeds = None

    @classmethod
    def from_pretrained(cls, path: str, device="cuda", revision: str = None, token=None):
        """`path` is a local folder with config.json and model.safetensors, or the name of a Hugging Face model repository."""
        folder = Path(path)
        if not folder.is_dir():
            from huggingface_hub import snapshot_download
            folder = Path(snapshot_download(path, revision=revision, token=token, allow_patterns=["config.json", "model.safetensors"]))
        config = json.loads((folder / "config.json").read_text())
        with safe_open(str(folder / "model.safetensors"), "pt", device=str(device)) as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
        take = lambda prefix: {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if config["backbone_type"] == "hidream-o1":
            from .hidream import build_hidream_matting
            model = build_hidream_matting(config)
        elif config["backbone_type"] == "prx":
            from .prx import build_prx_matting
            model = build_prx_matting(config)
        else:
            raise ValueError(f"unknown backbone_type {config['backbone_type']!r}")
        model.load_state_dict(take("model."), strict=True, assign=True)
        model = model.to(device).eval()
        r = config["refiner"]
        refiner = HighPassRefiner(r["channels"], tuple(r["dilations"]), r["sigma"], r["amplitude"])
        refiner.load_state_dict(take("refiner."), strict=True)
        refiner = refiner.to(device).eval()
        pipe = cls(model, refiner, config, device)
        if "prompt_embeds" in state:
            pipe.prompt_embeds = state["prompt_embeds"]
        return pipe

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.refiner.to(self.device)
        if self.prompt_embeds is not None:
            self.prompt_embeds = self.prompt_embeds.to(self.device)
        return self

    @torch.no_grad()
    def __call__(self, image: Image.Image, trimap, seed: int = 42, refine: bool = True, layer_shift: bool = True,
                 max_long_side: int = 0) -> np.ndarray:
        """image: PIL image. trimap: PIL image or array of the same size (0 background, 128 unknown, 255 foreground).
        Returns the alpha matte as a float32 array [H, W] in [0, 1] at the size of the input image."""
        cfg, dev = self.config, self.device
        image = image.convert("RGB")
        W0, H0 = image.size
        tri = normalize_trimap(trimap)
        if tri.shape != (H0, W0):
            tri = np.asarray(Image.fromarray(tri).resize((W0, H0), Image.NEAREST))
        p = cfg["patch_size"]
        H, W = _working_size(W0, H0, cfg["resolution"], p, max_long_side)

        to_tensor = lambda a: torch.from_numpy(np.asarray(a).copy()).to(dev)
        rgb = to_tensor(image.resize((W, H), Image.BILINEAR)).permute(2, 0, 1)[None].float() / 255 * 2 - 1
        tri_in = to_tensor(Image.fromarray(tri).resize((W, H), Image.NEAREST))[None, None].float() / 255 * 2 - 1
        noise = torch.randn((1, 3, H, W), generator=torch.Generator("cpu").manual_seed(seed)) * cfg["noise_scale"]

        with torch.autocast(dev.type, dtype=self.dtype):
            kwargs = {} if self.prompt_embeds is None else {"prompt_embeds": self.prompt_embeds}
            x0 = self.model(noise.to(dev), rgb, tri_in, sigma=1.0, layer_shift=layer_shift, **kwargs).float()
        alpha = ((x0.mean(1, keepdim=True) + 1) / 2).clamp(0, 1)

        if (H, W) != (H0, W0):
            if cfg.get("downscale", "bilinear") == "area" and H >= H0 and W >= W0:
                alpha = F.interpolate(alpha, size=(H0, W0), mode="area")
            else:
                alpha = F.interpolate(alpha, size=(H0, W0), mode="bilinear", align_corners=False)
            alpha = alpha.clamp(0, 1)

        tri_full = to_tensor(tri)[None, None]
        if refine:
            rgb_full = to_tensor(image).permute(2, 0, 1)[None].float() / 255 * 2 - 1
            sign = (tri_full == FOREGROUND).float() - (tri_full == BACKGROUND).float()
            alpha = self.refiner(rgb_full.contiguous(memory_format=torch.channels_last), sign, alpha)
        alpha = torch.where(tri_full == BACKGROUND, torch.zeros_like(alpha), alpha)
        alpha = torch.where(tri_full == FOREGROUND, torch.ones_like(alpha), alpha)
        return alpha[0, 0].float().cpu().numpy()
