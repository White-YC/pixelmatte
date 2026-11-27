"""High-Pass Refiner: a small convolutional network that corrects the fine detail of a predicted matte inside the unknown region."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = int(math.ceil(3 * sigma))
    u = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-u ** 2 / (2 * sigma ** 2))
    return k / k.sum()


class HighPassRefiner(nn.Module):
    """Predicts a residual for the matte from (image, trimap, matte). The local mean of the residual over the unknown region is
    removed with a Gaussian window, so the refiner only changes detail finer than the window and leaves the coarse matte as it is."""

    def __init__(self, channels: int = 96, dilations=(1, 2, 4, 8, 1, 2, 4, 8), sigma: float = 24.0, amplitude: float = 0.5):
        super().__init__()
        self.stem = nn.Conv2d(5, channels, 3, padding=1)
        self.c1 = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=d, dilation=d) for d in dilations)
        self.c2 = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1) for _ in dilations)
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        self.amplitude = amplitude
        self.register_buffer("kernel", _gaussian_kernel(sigma), persistent=False)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        r = self.kernel.numel() // 2
        x = F.conv2d(x, self.kernel.view(1, 1, 1, -1), padding=(0, r))
        return F.conv2d(x, self.kernel.view(1, 1, -1, 1), padding=(r, 0))

    def residual(self, image: torch.Tensor, trimap: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        h = self.stem(torch.cat([image, trimap, alpha * 2 - 1], 1))
        for c1, c2 in zip(self.c1, self.c2):
            h = h + c2(F.gelu(c1(F.gelu(h))))
        return self.amplitude * torch.tanh(self.head(F.gelu(h)))

    def forward(self, image: torch.Tensor, trimap: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """image [B, 3, H, W] in [-1, 1]; trimap [B, 1, H, W] with -1 (background), 0 (unknown), 1 (foreground);
        alpha [B, 1, H, W] in [0, 1]. Returns the refined alpha."""
        unknown = (trimap == 0).to(alpha.dtype)
        d = self.residual(image, trimap, alpha)
        d = d - self._blur(unknown * d) / self._blur(unknown).clamp_min(1e-3)
        return (alpha + unknown * d).clamp(0, 1)
