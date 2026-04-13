"""Procedural texture generation for terrain surfaces.

Generates 512x512 RGB textures with bold, macro-scale patterns designed to be
clearly visible from a robot-mounted camera at 1-3m distance.  These textures
give monocular depth estimators (e.g. Depth Anything V2) strong visual cues
for inferring surface geometry.

Used by both the Genesis simulator (training) and the thesis figure capture
script, ensuring a single canonical implementation.
"""

from __future__ import annotations

import numpy as np


TEX_SIZE = 512

# All supported texture modes.
TEXTURE_MODES = [
    "checkerboard",
    "bricks",
    "concrete",
    "wood",
    "grass",
    "stone_tiles",
    "gravel",
    "noise",
    "solid_red",
    "solid_green",
    "solid_blue",
    "solid_yellow",
    "solid_white",
    "solid_dark",
]


def generate_texture_image(mode: str, size: int = TEX_SIZE) -> np.ndarray:
    """Return a (size, size, 3) uint8 RGB texture image for *mode*.

    Patterns are designed to be clearly visible at macro scale when tiled
    with ``uv_scale=1.0`` over typical parkour terrain (6m lanes).
    """
    if mode == "checkerboard":
        return _checkerboard(size)
    if mode == "bricks":
        return _bricks(size)
    if mode == "concrete":
        return _concrete(size)
    if mode == "wood":
        return _wood(size)
    if mode == "grass":
        return _grass(size)
    if mode == "stone_tiles":
        return _stone_tiles(size)
    if mode == "gravel":
        return _gravel(size)
    if mode == "noise":
        return np.random.RandomState(123).randint(0, 256, (size, size, 3), dtype=np.uint8)
    if mode.startswith("solid_"):
        return _solid(mode, size)
    if mode == "random_color":
        import time
        rng = np.random.RandomState(int(time.time()) % 2**31)
        color = rng.randint(30, 230, size=3).tolist()
        return np.full((size, size, 3), color, dtype=np.uint8)
    # Fallback: plain grey
    return np.full((size, size, 3), [180, 175, 170], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Individual texture generators — bold, high-contrast patterns
# ---------------------------------------------------------------------------

def _checkerboard(size: int) -> np.ndarray:
    """Large alternating light/dark squares with subtle noise."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // 4  # 4x4 grid of 128px cells
    light = np.array([210, 205, 200], dtype=np.uint8)
    dark = np.array([120, 115, 110], dtype=np.uint8)
    for row in range(4):
        for col in range(4):
            y0, y1 = row * cell, (row + 1) * cell
            x0, x1 = col * cell, (col + 1) * cell
            color = light if (row + col) % 2 == 0 else dark
            img[y0:y1, x0:x1] = color
    noise = np.random.RandomState(42).randint(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _bricks(size: int) -> np.ndarray:
    """Large brick pattern with visible mortar lines."""
    brick_h, brick_w, mortar = size // 4, size // 2, 6
    mortar_color = np.array([195, 190, 185], dtype=np.uint8)
    img = np.full((size, size, 3), mortar_color, dtype=np.uint8)
    rng = np.random.RandomState(42)
    row_idx = 0
    for y in range(0, size, brick_h):
        offset = (brick_w // 2) if (row_idx % 2) else 0
        for x in range(-brick_w, size + brick_w, brick_w):
            bx0 = max(0, x + offset + mortar)
            bx1 = min(size, x + offset + brick_w)
            by0 = max(0, y + mortar)
            by1 = min(size, y + brick_h)
            if bx0 < bx1 and by0 < by1:
                shade = rng.randint(-15, 16)
                color = np.clip(np.array([180, 100, 70], dtype=np.int16) + shade, 0, 255)
                brick_noise = rng.randint(-6, 7, (by1 - by0, bx1 - bx0, 3), dtype=np.int16)
                img[by0:by1, bx0:bx1] = np.clip(color + brick_noise, 0, 255).astype(np.uint8)
        row_idx += 1
    return img


def _concrete(size: int) -> np.ndarray:
    """Concrete with visible expansion joints and aggregate spots."""
    rng = np.random.RandomState(44)
    img = np.full((size, size, 3), [194, 192, 188], dtype=np.uint8)
    # Large expansion joints
    joint_spacing = size // 4
    for i in range(joint_spacing, size, joint_spacing):
        img[i:i + 4, :] = [150, 150, 146]
        img[:, i:i + 4] = [150, 150, 146]
    # Surface noise
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Larger, more visible aggregate spots
    for _ in range(120):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(5, 14)
        shade = np.full(3, rng.randint(110, 160), dtype=np.uint8)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = xx * xx + yy * yy <= radius * radius
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        local_mask = mask[y0 - (cy - radius):y1 - (cy - radius), x0 - (cx - radius):x1 - (cx - radius)]
        img[y0:y1, x0:x1][local_mask] = shade
    return img


def _wood(size: int) -> np.ndarray:
    """Wide wood grain bands with high contrast."""
    rng = np.random.RandomState(45)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    phase = np.cumsum(rng.normal(0, 0.04, size))
    for y in range(size):
        freq = 0.04 + 0.015 * np.sin(y * 0.008)
        wave = np.sin(np.arange(size) * freq + phase + y * 0.06)
        intensity = ((wave * 0.5 + 0.5) * 70).astype(np.int16)
        img[y, :, 0] = np.clip(150 + intensity, 0, 255)
        img[y, :, 1] = np.clip(105 + intensity, 0, 255)
        img[y, :, 2] = np.clip(65 + (intensity * 0.4).astype(np.int16), 0, 255)
    noise = rng.randint(-6, 7, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _grass(size: int) -> np.ndarray:
    """Grass with large visible patches of varying shade."""
    rng = np.random.RandomState(46)
    img = np.full((size, size, 3), [70, 135, 50], dtype=np.uint8)
    # Per-channel noise for base variation
    for ch, sigma in enumerate([10, 18, 8]):
        noise = rng.normal(0, sigma, (size, size)).astype(np.int16)
        img[:, :, ch] = np.clip(img[:, :, ch].astype(np.int16) + noise, 0, 255)
    # Large, clearly visible patches
    for _ in range(25):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(30, 80)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = xx * xx + yy * yy <= radius * radius
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        local_mask = mask[y0 - (cy - radius):y1 - (cy - radius), x0 - (cx - radius):x1 - (cx - radius)]
        # Alternate between dark and light patches
        base = np.array([45, 100, 30]) if rng.random() < 0.5 else np.array([100, 170, 70])
        patch_noise = rng.normal(0, 6, (local_mask.sum(), 3)).astype(np.int16)
        img[y0:y1, x0:x1][local_mask] = np.clip(base + patch_noise, 0, 255).astype(np.uint8)
    return img


def _stone_tiles(size: int) -> np.ndarray:
    """Large stone tiles with prominent mortar gaps."""
    rng = np.random.RandomState(47)
    tile_size = size // 4  # 4x4 grid
    mortar_w = 6
    mortar_color = np.array([95, 90, 85], dtype=np.uint8)
    img = np.full((size, size, 3), mortar_color, dtype=np.uint8)
    for ty in range(0, size, tile_size):
        for tx in range(0, size, tile_size):
            grey = rng.randint(145, 210)
            color = np.array([grey, grey - 3, grey - 7], dtype=np.int16)
            y0 = ty + mortar_w
            x0 = tx + mortar_w
            y1 = min(ty + tile_size, size)
            x1 = min(tx + tile_size, size)
            if y0 < y1 and x0 < x1:
                tile_noise = rng.randint(-10, 11, (y1 - y0, x1 - x0, 3), dtype=np.int16)
                img[y0:y1, x0:x1] = np.clip(color + tile_noise, 0, 255).astype(np.uint8)
    return img


def _gravel(size: int) -> np.ndarray:
    """Coarse gravel with large, visible stones."""
    rng = np.random.RandomState(48)
    img = np.full((size, size, 3), [160, 155, 150], dtype=np.uint8)
    for _ in range(600):
        cx, cy = rng.randint(0, size, 2)
        radius = rng.randint(6, 20)
        grey = rng.randint(90, 230)
        color = np.array([grey, grey - 3, grey - 6], dtype=np.int16)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = xx * xx + yy * yy <= radius * radius
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        local_mask = mask[y0 - (cy - radius):y1 - (cy - radius), x0 - (cx - radius):x1 - (cx - radius)]
        img[y0:y1, x0:x1][local_mask] = np.clip(color, 0, 255).astype(np.uint8)
    return img


def _solid(mode: str, size: int) -> np.ndarray:
    color_map = {
        "solid_red": [200, 60, 60],
        "solid_green": [60, 180, 60],
        "solid_blue": [60, 60, 200],
        "solid_yellow": [200, 200, 60],
        "solid_white": [240, 240, 240],
        "solid_dark": [40, 40, 40],
    }
    return np.full((size, size, 3), color_map.get(mode, [180, 180, 180]), dtype=np.uint8)
