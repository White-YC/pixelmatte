"""Run PixelMatte on one image or on a folder of images.

    python infer.py --model WhiteYC/PixelMatte --subfolder HiDream-8B --image photo.jpg --trimap trimap.png --output alpha.png
    python infer.py --model WhiteYC/PixelMatte --subfolder HiDream-8B --image images/ --trimap trimaps/ --output mattes/

In folder mode every image needs a trimap with the same file name (any image extension). The matte is written as a 16-bit PNG;
--cutout additionally writes the image with the matte as its alpha channel.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pixelmatte import PixelMattePipeline

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def find_pairs(image: Path, trimap: Path, output: Path):
    if image.is_file():
        return [(image, trimap, output)]
    trimaps = {p.stem: p for p in sorted(trimap.iterdir()) if p.suffix.lower() in EXTENSIONS}
    pairs = []
    for p in sorted(image.iterdir()):
        if p.suffix.lower() in EXTENSIONS:
            if p.stem not in trimaps:
                raise FileNotFoundError(f"no trimap named {p.stem}.* in {trimap}")
            pairs.append((p, trimaps[p.stem], output / f"{p.stem}.png"))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face repository or local folder")
    ap.add_argument("--subfolder", default=None, help="checkpoint inside the repository or folder: HiDream-8B or PRX-7B")
    ap.add_argument("--image", required=True, type=Path, help="image file or folder")
    ap.add_argument("--trimap", required=True, type=Path, help="trimap file or folder (0 background, 128 unknown, 255 foreground)")
    ap.add_argument("--output", required=True, type=Path, help="output file or folder")
    ap.add_argument("--seed", type=int, default=42, help="seed of the input noise")
    ap.add_argument("--no-refiner", action="store_true", help="skip the High-Pass Refiner")
    ap.add_argument("--no-layershift", action="store_true", help="skip LayerShift")
    ap.add_argument("--max-long-side", type=int, default=0, help="limit the longer side of the network input (0 = no limit)")
    ap.add_argument("--cutout", action="store_true", help="also save <name>_rgba.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pipe = PixelMattePipeline.from_pretrained(args.model, args.subfolder, device=args.device)
    pairs = find_pairs(args.image, args.trimap, args.output)
    if args.image.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
    for image_path, trimap_path, out_path in pairs:
        image = Image.open(image_path).convert("RGB")
        alpha = pipe(image, Image.open(trimap_path), seed=args.seed, refine=not args.no_refiner, layer_shift=not args.no_layershift,
                     max_long_side=args.max_long_side)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((alpha * 65535 + 0.5).astype(np.uint16)).save(out_path)
        if args.cutout:
            rgba = np.dstack([np.asarray(image), (alpha * 255 + 0.5).astype(np.uint8)])
            Image.fromarray(rgba, "RGBA").save(out_path.with_name(out_path.stem + "_rgba.png"))
        print(f"{image_path} -> {out_path}")


if __name__ == "__main__":
    main()
