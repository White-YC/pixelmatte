"""PixelMatte on the PRX backbone (7B, pixel space, patch size 16).

The transformer follows PRXTransformer2DModel of the diffusers library (Apache-2.0) and keeps its parameter names. The image
tokens form one sequence

    [noisy matte] [image] [trimap]

in which all three segments use the same 2-D rotary positions; a learned segment vector is added to the image and trimap tokens.
The text condition is a fixed prompt whose encoder features are stored with the checkpoint, so no text encoder is needed. The matte
is predicted directly (x-prediction) from the tokens of the first segment. LayerShift adds one learned vector per block to the
matte tokens that touch the unknown region of the trimap.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hidream import LayerShift

NUM_SEGMENTS = 3            # matte, image, trimap


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0, scale: float = 1.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    emb = scale * t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def rope_table(ids: torch.Tensor, axes_dim, theta: float) -> torch.Tensor:
    """ids [L, axes] -> rotation matrices [1, 1, L, sum(axes_dim) / 2, 2, 2]"""
    out = []
    for i, d in enumerate(axes_dim):
        omega = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device=ids.device) / d))
        o = ids[:, i].to(torch.float64)[:, None] * omega[None, :]
        o = torch.stack([torch.cos(o), -torch.sin(o), torch.sin(o), torch.cos(o)], dim=-1)
        out.append(o.reshape(*o.shape[:-1], 2, 2).float())
    return torch.cat(out, dim=-3)[None, None]


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def img2seq(img: torch.Tensor, p: int) -> torch.Tensor:
    b, c, h, w = img.shape
    return img.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * p * p)


def seq2img(seq: torch.Tensor, p: int, h: int, w: int) -> torch.Tensor:
    b, _, d = seq.shape
    c = d // (p * p)
    return seq.reshape(b, h // p, w // p, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


class ResolutionEmbedder(nn.Module):
    def __init__(self, hidden: int, max_period: float):
        super().__init__()
        self.max_period = max_period
        self.mlp = MLPEmbedder(256, hidden)

    def forward(self, h: torch.Tensor, w: torch.Tensor, dtype) -> torch.Tensor:
        e = torch.cat([timestep_embedding(h, 128, self.max_period), timestep_embedding(w, 128, self.max_period)], -1)
        return self.mlp(e.to(dtype))


class Attention(nn.Module):
    """Image tokens attend to [text tokens, image tokens]; rotary positions on the image tokens only."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.img_qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)
        self.txt_kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.norm_added_k = RMSNorm(self.head_dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])

    def forward(self, x, txt, rope):
        B, L, _ = x.shape
        q, k, v = self.img_qkv_proj(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.norm_q(q), self.norm_k(k)
        tk, tv = self.txt_kv_proj(txt).reshape(B, txt.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        tk = self.norm_added_k(tk)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        k = torch.cat([tk.to(k.dtype), k], dim=2)
        v = torch.cat([tv.to(v.dtype), v], dim=2)
        o = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
        return self.to_out[0](o.transpose(1, 2).reshape(B, L, self.heads * self.head_dim))


class Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim)

    def forward(self, vec):
        out = self.lin(F.silu(vec))[:, None, :].chunk(6, dim=-1)
        return out[:3], out[3:]


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.img_pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(dim, heads)
        self.post_attention_layernorm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(dim)

    def forward(self, x, txt, vec, rope, shift=None):
        """shift [B, n, dim] or None: added to the first n tokens (the matte segment) before the MLP"""
        (a_shift, a_scale, a_gate), (m_shift, m_scale, m_gate) = self.modulation(vec)
        x = x + a_gate * self.attention((1 + a_scale) * self.img_pre_norm(x) + a_shift, txt, rope)
        h = (1 + m_scale) * self.post_attention_layernorm(x) + m_shift
        if shift is not None:
            n = shift.shape[1]
            h = torch.cat([h[:, :n] + shift, h[:, n:]], 1)
        return x + m_gate * self.down_proj(self.mlp_act(self.gate_proj(h)) * self.up_proj(h))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])


class PRXMatting(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, context_in_dim=2048, hidden_size=3584, mlp_ratio=3.5, num_heads=28, depth=24,
                 axes_dim=(64, 64), theta=10000, time_factor=1000.0, time_max_period=10000, bottleneck_size=768, **_):
        super().__init__()
        self.patch_size, self.hidden_size = patch_size, hidden_size
        self.axes_dim, self.theta = list(axes_dim), theta
        self.time_factor, self.time_max_period = time_factor, time_max_period
        patch_dim = in_channels * patch_size ** 2
        self.img_in = nn.Sequential(nn.Linear(patch_dim, bottleneck_size), nn.Linear(bottleneck_size, hidden_size))
        self.time_in = MLPEmbedder(256, hidden_size)
        self.txt_in = nn.Linear(context_in_dim, hidden_size)
        self.resolution_embedder = ResolutionEmbedder(hidden_size, time_max_period)
        self.blocks = nn.ModuleList(Block(hidden_size, num_heads, mlp_ratio) for _ in range(depth))
        self.final_layer = FinalLayer(hidden_size, patch_dim)
        self.cond_seg = nn.Parameter(torch.zeros(hidden_size))
        self.layer_shift = LayerShift(depth, hidden_size)
        self._rope = {}

    def _positions(self, rows: int, cols: int, device):
        key = (rows, cols, str(device))
        if key not in self._rope:
            ids = torch.zeros(rows, cols, 2, device=device)
            ids[..., 0] = torch.arange(rows, device=device)[:, None]
            ids[..., 1] = torch.arange(cols, device=device)[None, :]
            self._rope[key] = rope_table(torch.cat([ids.reshape(-1, 2)] * NUM_SEGMENTS, 0), self.axes_dim, self.theta)
        return self._rope[key]

    def forward(self, x: torch.Tensor, image: torch.Tensor, trimap: torch.Tensor, sigma: float = 1.0, layer_shift: bool = True,
                prompt_embeds: torch.Tensor = None):
        """x [B, 3, H, W] noisy matte; image [B, 3, H, W] and trimap [B, 1, H, W] in [-1, 1]; sigma in [0, 1] (1 = pure noise);
        prompt_embeds [1, T, 2048]. Returns the predicted clean matte [B, 3, H, W] in [-1, 1]."""
        B, C, H, W = x.shape
        p = self.patch_size
        assert C == 3 and H % p == 0 and W % p == 0, f"height and width must be multiples of {p}, got {H} x {W}"
        rows, cols = H // p, W // p
        n = rows * cols
        img = self.img_in(torch.cat([img2seq(x, p), img2seq(image, p), img2seq(trimap.expand(-1, 3, -1, -1), p)], 1))
        img = torch.cat([img[:, :n], img[:, n:] + self.cond_seg.to(img.dtype)], 1)
        txt = self.txt_in(prompt_embeds.to(img.dtype).expand(B, -1, -1))
        s = torch.full((B,), float(sigma), device=x.device)
        vec = self.time_in(timestep_embedding(s, 256, self.time_max_period, self.time_factor).to(img.dtype))
        size = lambda v: torch.full((B,), float(v), device=x.device)
        vec = vec + self.resolution_embedder(size(H), size(W), vec.dtype)
        rope = self._positions(rows, cols, x.device)
        mask = LayerShift.token_mask(trimap, p).to(img.dtype) if layer_shift else None
        for index, block in enumerate(self.blocks):
            shift = None if mask is None else self.layer_shift.shift[index].to(img.dtype) * mask
            img = block(img, txt, vec, rope, shift)
        return seq2img(self.final_layer(img[:, :n], vec), p, H, W)


def build_prx_matting(config: dict) -> PRXMatting:
    """An uninitialised model on the meta device, to be filled with load_state_dict(..., assign=True)."""
    with torch.device("meta"):
        return PRXMatting(**config["backbone"])
