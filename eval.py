"""Score predicted mattes against the ground truth with the four metrics of the paper: SAD, MSE, Grad and Conn.

    python eval.py --pred results/am2k --alpha AM-2k/validation/mask --trimap AM-2k/validation/trimap

With several prediction folders (runs with different noise seeds), each run is scored and the mean over the runs is printed last:

    python eval.py --pred results/am2k/seed42 results/am2k/seed43 --alpha AM-2k/validation/mask --trimap AM-2k/validation/trimap

Files are matched by name (any image extension). Every metric is computed over the whole image, after the pixels that the trimap
marks as background or foreground are set to 0 and 1 in the prediction:

    SAD   sum of |prediction - ground truth| / 1000
    MSE   mean of (prediction - ground truth)^2, printed in units of 10^-3
    Grad  sum of (|grad prediction| - |grad ground truth|)^2 / 1000, gradients from Gaussian derivative filters (sigma 1.4)
    Conn  connectivity error / 1000, computed as in the evaluation code of RenderMatte (https://github.com/Nislab-r/RenderMatte)

Each metric is averaged over the images of the set.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
METRICS = ("SAD", "MSE", "Grad", "Conn")
FOUR_CONNECTED = ndi.generate_binary_structure(2, 1)


def _gradient_kernels(sigma: float = 1.4, epsilon: float = 1e-2):
    half = int(np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))))
    u = np.arange(-half, half + 1, dtype=np.float64)
    gauss = np.exp(-u ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    kernel = gauss[:, None] * (-u * gauss / sigma ** 2)[None, :]          # smooth along rows, differentiate along columns
    kernel = kernel / np.sqrt(np.sum(kernel ** 2))
    return kernel, kernel.T


KERNEL_X, KERNEL_Y = _gradient_kernels()


def gradient_magnitude(x: np.ndarray) -> np.ndarray:
    return np.sqrt(ndi.convolve(x, KERNEL_X, mode="nearest") ** 2 + ndi.convolve(x, KERNEL_Y, mode="nearest") ** 2)


def _largest_region(mask: np.ndarray) -> np.ndarray:
    """The largest 4-connected region of the label map of `mask`. The complement of the mask is one of the regions, as in
    RenderMatte's evaluation code."""
    labels, _ = ndi.label(mask, structure=FOUR_CONNECTED)
    return labels == np.argmax(np.bincount(labels.ravel()))


def connectivity_error(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    levels = np.arange(0, 1 + step, step)
    reach = np.full(pred.shape, -1.0, np.float32)          # highest level at which a pixel still belongs to the largest region
    for i in range(1, len(levels)):
        omega = _largest_region((pred >= levels[i]) & (gt >= levels[i]))
        reach[(reach == -1) & ~omega] = levels[i - 1]
    reach[reach == -1] = 1
    d_pred, d_gt = pred - reach, gt - reach
    phi_pred, phi_gt = 1 - d_pred * (d_pred >= 0.15), 1 - d_gt * (d_gt >= 0.15)
    return float(np.sum(np.abs(phi_pred - phi_gt)) / 1000)


def matting_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: float64 arrays [H, W] in [0, 1]."""
    diff = pred - gt
    return {"SAD": float(np.abs(diff).sum() / 1000), "MSE": float((diff ** 2).mean()),
            "Grad": float(np.sum((gradient_magnitude(pred) - gradient_magnitude(gt)) ** 2) / 1000),
            "Conn": connectivity_error(pred, gt)}


def read_unit(path) -> np.ndarray:
    """8-bit or 16-bit image -> float64 [H, W] in [0, 1]"""
    image = Image.open(path)
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        return np.asarray(image, np.float64) / 65535
    return np.asarray(image.convert("L"), np.float64) / 255


def score(paths) -> dict:
    pred_path, alpha_path, trimap_path = paths
    gt = read_unit(alpha_path)
    pred = read_unit(pred_path)
    if pred.shape != gt.shape:
        raise ValueError(f"{pred_path}: prediction is {pred.shape[::-1]}, ground truth is {gt.shape[::-1]}")
    trimap = Image.open(trimap_path).convert("L")
    if trimap.size != (gt.shape[1], gt.shape[0]):
        trimap = trimap.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
    trimap = np.asarray(trimap)
    pred[trimap <= 10] = 0
    pred[trimap >= 245] = 1
    return matting_metrics(pred, gt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path, nargs="+", help="folder of predicted mattes; several folders are averaged")
    ap.add_argument("--alpha", required=True, type=Path, help="folder of ground-truth mattes")
    ap.add_argument("--trimap", required=True, type=Path, help="folder of trimaps")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = lambda folder: {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTENSIONS}
    alphas, trimaps = files(args.alpha), files(args.trimap)
    line = lambda m: f"SAD {m['SAD']:.2f} | MSE {m['MSE'] * 1000:.2f} x 10^-3 | Grad {m['Grad']:.2f} | Conn {m['Conn']:.2f}"
    runs = []
    with Pool(args.workers) as pool:
        for folder in args.pred:
            preds = files(folder)
            missing = sorted(set(alphas) - set(preds))
            if missing:
                raise FileNotFoundError(f"{len(missing)} of {len(alphas)} images have no prediction in {folder}, e.g. {missing[0]}")
            rows = pool.map(score, [(preds[s], alphas[s], trimaps[s]) for s in sorted(alphas)], chunksize=4)
            runs.append({k: float(np.mean([r[k] for r in rows])) for k in METRICS})
            print(f"{folder}: {len(rows)} images | {line(runs[-1])}")
    if len(runs) > 1:
        print(f"mean of {len(runs)} runs | {line({k: float(np.mean([r[k] for r in runs])) for k in METRICS})}")


if __name__ == "__main__":
    main()
