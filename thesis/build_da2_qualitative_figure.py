#!/usr/bin/env python3
"""Build the thesis DA2 qualitative figure from official Depth Anything V2 assets.

This script downloads the official metric-depth comparison image used by the
Depth Anything V2 repository/project page, extracts two representative
RGB/DA2 pairs, and assembles them into the compact 1x4 strip used in the
thesis.

Usage:
    ./.venv/bin/python thesis/build_da2_qualitative_figure.py
    ./.venv/bin/python thesis/build_da2_qualitative_figure.py --output thesis/figures/depth_anything_v2_demo.png
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OFFICIAL_SOURCE_URL = (
    "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/"
    "metric_depth/assets/compare_zoedepth.png"
)

# Crop coordinates in the official comparison grid. They correspond to:
# - row 2: living room
# - row 3: library
ROW_BANDS = [
    (708, 1249),
    (1288, 1828),
]
LEFT_IMAGE_X = (2, 835)
LEFT_DA2_X = (1766, 2600)
RIGHT_IMAGE_X = (2719, 3556)
RIGHT_DA2_X = (4484, 5315)
PANEL_SELECTIONS = [
    (ROW_BANDS[0], RIGHT_IMAGE_X, RIGHT_DA2_X),
    (ROW_BANDS[1], LEFT_IMAGE_X, LEFT_DA2_X),
]


def _download_source(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(OFFICIAL_SOURCE_URL, path)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return ImageFont.truetype(str(candidate_path), size)
    return ImageFont.load_default()


def build_figure(source_path: Path, output_path: Path):
    if not source_path.exists():
        _download_source(source_path)

    src = Image.open(source_path).convert("RGB")

    panels: list[Image.Image] = []
    for (y0, y1), image_x, da2_x in PANEL_SELECTIONS:
        panels.append(src.crop((image_x[0], y0, image_x[1], y1)))
        panels.append(src.crop((da2_x[0], y0, da2_x[1], y1)))

    cell_w = min(im.size[0] for im in panels)
    cell_h = min(im.size[1] for im in panels)
    panels = [im.resize((cell_w, cell_h), Image.Resampling.LANCZOS) for im in panels]

    labels = ["RGB", "DA2", "RGB", "DA2"]
    gap = 12
    outer = 10
    header_h = 30

    canvas_w = outer * 2 + cell_w * 4 + gap * 3
    canvas_h = outer * 2 + header_h + cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(24)

    for idx, (label, panel) in enumerate(zip(labels, panels)):
        x0 = outer + idx * (cell_w + gap)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x0 + (cell_w - text_w) / 2, outer - 2), label, fill="black", font=font)
        canvas.paste(panel, (x0, outer + header_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("thesis/figures/raw/depth_anything_v2_compare_zoedepth.png"),
        help="Local cache path for the official source comparison image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thesis/figures/depth_anything_v2_demo.png"),
        help="Output path for the assembled thesis figure.",
    )
    args = parser.parse_args()
    build_figure(args.source, args.output)


if __name__ == "__main__":
    main()
