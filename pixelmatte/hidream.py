"""PixelMatte on the HiDream-O1-Image backbone (8B, pixel space, patch size 32).

The backbone reads one token sequence:

    [prompt tokens, timestep token] [noisy matte] [image] [trimap]

The three picture segments share the pretrained patch embedder and the same (row, column) rotary positions; they differ only in
the first rotary axis (matte t, image t + 1, trimap t + 2). The matte is predicted directly (x-prediction) from the tokens of the
first segment. LayerShift adds one learned vector per block to the matte tokens that touch the unknown region of the trimap.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_SIZE = 32
NOISE_SCALE = 8.0
POSITION_ORIGIN = 4096      # rotary origin of generated images in HiDream-O1-Image
NUM_SEGMENTS = 3            # matte, image, trimap


def _backbone_modules():
    """The HiDream-O1-Image model code, imported on first use. Attention runs through torch's scaled_dot_product_attention."""
    assert os.environ.get("USE_BF16_ROPE", "0") in ("", "0"), "USE_BF16_ROPE must be unset or 0"
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_parameters)        # not registered in transformers 5
    from .third_party.hidream import qwen3_vl_transformers as Q
    from .third_party.hidream import utils as U
    if Q._flash_attn_func is None:
        Q._flash_attn_func = _attention
    return Q, U


def _default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
    """Inverse frequencies of the standard rotary embedding."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    theta = getattr(config, "rope_theta", None) or (getattr(config, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)), 1.0


def _attention(q, k, v, softmax_scale=None, causal=False):
    """Drop-in for flash_attn_func: q [B, S, H, D], k / v [B, S, H_kv, D] (grouped-query attention) -> [B, S, H, D]."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if q.is_cuda:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale, enable_gqa=True).transpose(1, 2)
        except RuntimeError:
            pass
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale).transpose(1, 2)


def patchify(x: torch.Tensor, p: int = PATCH_SIZE) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H / p) (W / p), C p p]"""
    B, C, H, W = x.shape
    return x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(s: torch.Tensor, rows: int, cols: int, p: int = PATCH_SIZE, C: int = 3) -> torch.Tensor:
    """[B, rows cols, C p p] -> [B, C, rows p, cols p]"""
    return s.reshape(s.shape[0], rows, cols, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(s.shape[0], C, rows * p, cols * p)


class _NoVisionTower(nn.Module):
    """The image and the trimap enter as pixel patches, so the backbone's vision encoder is not needed and is left out."""
    spatial_merge_size, dtype = 1, torch.float32

    def forward(self, pixel_values, grid_thw):
        return pixel_values.new_zeros(int(grid_thw.prod(-1).sum()), 1), []


class LayerShift(nn.Module):
    """One learned vector per transformer block. It is added, before the block's MLP, to the matte tokens whose patch contains at
    least one unknown pixel of the trimap."""

    def __init__(self, depth: int, dim: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(depth, dim))

    @staticmethod
    def token_mask(trimap: torch.Tensor, patch_size: int) -> torch.Tensor:
        """trimap [B, 1, H, W] in [-1, 1] -> [B, tokens, 1], 1 where the patch touches the unknown region"""
        unknown = (trimap.float().abs() < 0.9).float()
        return (F.avg_pool2d(unknown, patch_size) > 0).flatten(1).unsqueeze(-1)


class HiDreamMatting(nn.Module):
    def __init__(self, backbone: nn.Module, prompt_ids: torch.Tensor):
        super().__init__()
        rot = backbone.language_model.rotary_emb
        rot.original_inv_freq = rot.original_inv_freq.detach().clone().float()   # rotary angles stay in float32
        backbone.visual = _NoVisionTower()
        self.backbone = backbone
        self.prompt_ids = prompt_ids.reshape(1, -1).long().cpu()
        self.layer_shift = LayerShift(len(self.blocks), backbone.config.text_config.hidden_size)
        self._layouts = {}
        self._shift_mask = None
        for index, block in enumerate(self.blocks):
            block.mlp.register_forward_pre_hook(self._shift_hook(index))

    @property
    def blocks(self) -> nn.ModuleList:
        return self.backbone.language_model.layers

    def _shift_hook(self, index: int):
        def hook(module, args):
            mask = self._shift_mask
            if mask is None:
                return None
            h = args[0]
            t, n = self.prompt_ids.shape[1], mask.shape[1]
            shifted = h[:, t:t + n] + (self.layer_shift.shift[index].to(h.dtype) * mask.to(h.dtype))
            return (torch.cat([h[:, :t], shifted, h[:, t + n:]], 1),) + tuple(args[1:])
        return hook

    def _layout(self, rows: int, cols: int, device):
        """-> prompt ids [1, T], rotary position ids [3, 1, T + 3 n], token types [1, T + 3 n] for an image of rows x cols patches"""
        key = (rows, cols, str(device))
        if key not in self._layouts:
            _, U = _backbone_modules()
            cfg = self.backbone.config
            ids, n = self.prompt_ids, rows * cols
            t = ids.shape[1]
            picture = torch.full((1, n), cfg.image_token_id, dtype=ids.dtype)
            picture[0, 0] = cfg.vision_start_token_id
            pos, _ = U.get_rope_index_fix_point(1, cfg.image_token_id, cfg.video_token_id, cfg.vision_start_token_id,
                                                input_ids=torch.cat([ids, picture], -1),
                                                image_grid_thw=torch.tensor([[1, rows, cols]], dtype=torch.int64), video_grid_thw=None,
                                                attention_mask=None, skip_vision_start_token=[1], fix_point=POSITION_ORIGIN)
            matte = pos[:, :, t:]
            step = torch.tensor([1, 0, 0], dtype=pos.dtype).view(3, 1, 1)
            pos = torch.cat([pos[:, :, :t]] + [matte + k * step for k in range(NUM_SEGMENTS)], -1)
            types = torch.zeros((1, t + NUM_SEGMENTS * n), dtype=ids.dtype)
            types[0, t - 1:] = 1                                  # the timestep token and all picture tokens attend in both directions
            self._layouts[key] = (ids.to(device), pos.to(device), types.to(device))
        return self._layouts[key]

    def forward(self, x: torch.Tensor, image: torch.Tensor, trimap: torch.Tensor, sigma: float = 1.0, layer_shift: bool = True):
        """x [B, 3, H, W] noisy matte; image [B, 3, H, W] and trimap [B, 1, H, W] in [-1, 1]; sigma in [0, 1] (1 = pure noise).
        Returns the predicted clean matte [B, 3, H, W] in [-1, 1]."""
        B, C, H, W = x.shape
        p = PATCH_SIZE
        assert C == 3 and H % p == 0 and W % p == 0, f"height and width must be multiples of {p}, got {H} x {W}"
        rows, cols = H // p, W // p
        n = rows * cols
        ids, pos, types = self._layout(rows, cols, x.device)
        t = ids.shape[1]
        dtype = self.backbone.x_embedder.proj1.weight.dtype
        tokens = torch.cat([patchify(x), patchify(image), patchify(trimap.expand(-1, 3, -1, -1))], 1).to(dtype)
        timestep = torch.full((B,), 1.0 - float(sigma), device=x.device)
        self._shift_mask = LayerShift.token_mask(trimap, p) if layer_shift else None
        try:
            out = self.backbone(input_ids=ids.expand(B, -1), position_ids=pos.expand(-1, B, -1), vinputs=tokens, timestep=timestep,
                                token_types=types.expand(B, -1), use_flash_attn=True)
        finally:
            self._shift_mask = None
        return unpatchify(out.x_pred[:, t:t + n], rows, cols)


def build_hidream_matting(config: dict) -> HiDreamMatting:
    """An uninitialised model on the meta device, to be filled with load_state_dict(..., assign=True)."""
    Q, _ = _backbone_modules()
    cfg = Q.Qwen3VLConfig.from_dict(json.loads(json.dumps(config["backbone"])))
    with torch.device("meta"):
        backbone = Q.Qwen3VLModel._from_config(cfg, attn_implementation="sdpa")
    backbone.language_model.rotary_emb = Q.Qwen3VLTextRotaryEmbedding(config=backbone.language_model.config)
    return HiDreamMatting(backbone, torch.tensor(config["prompt_ids"]))
