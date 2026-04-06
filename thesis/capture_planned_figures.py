#!/usr/bin/env python3
"""Generate thesis figure assets from the local parkour codebase.

This script covers the figures that should be regenerated from this repo:

1. `parkour_obstacle_types.png`
2. `terrain_curriculum.png`
3. `da2_texture_comparison.png`

Usage:
    ./.venv/bin/python thesis/capture_planned_figures.py
    ./.venv/bin/python thesis/capture_planned_figures.py --figures obstacle_types terrain_curriculum
    ./.venv/bin/python thesis/capture_planned_figures.py --figures robot_perception --output-dir thesis/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SIMULATOR", "genesis")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import genesis as gs
import genesis.utils.geom as gu
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from legged_gym.envs.go2.go2_parkour_teacher.go2_parkour_teacher_config import Go2ParkourTeacherCfg
from legged_gym.perception import create_depth_estimator
from legged_gym.utils.parkour_terrain import ParkourLaneBuilder


FAMILY_LABELS = {
    "stairs": "Stairs",
    "hurdle_block": "Hurdle",
    "gap": "Gap",
}

MODEL_RUNS = {
    False: {"load_run": "Mar29_08-06-01_BASE1_da2_base", "ckpt": 7000},
    True: {"load_run": "Mar29_13-36-08_BASE3_da2_base_texture", "ckpt": 7000},
}

TEXTURE_COMPARISON_COLUMNS = [
    {
        "label": "No texture",
        "perturbation": "tex_none",
        "texture_mode": None,
        "add_texture": False,
    },
    {
        "label": "Checkerboard",
        "perturbation": "baseline",
        "texture_mode": "checkerboard",
        "add_texture": True,
    },
    {
        "label": "Grass",
        "perturbation": "tex_grass",
        "texture_mode": "grass",
        "add_texture": True,
    },
    {
        "label": "Wood",
        "perturbation": "tex_wood",
        "texture_mode": "wood",
        "add_texture": True,
    },
]

TERRAIN_RESOLUTION = (720, 420)
ROBOT_RGB_RESOLUTION = (424, 240)
FIGURE_VERTICAL_GAIN = {
    "stairs": 1.45,
    "hurdle_block": 2.6,
    "gap": 1.15,
}

_GENESIS_INITIALIZED = False
_TERRAIN_RENDER_CACHE: dict[tuple[str, int], np.ndarray] = {}
_TEXTURE_CACHE: dict[str, np.ndarray] = {}
_DEPTH_ESTIMATOR = None
_DOMAIN_INVARIANCE_RESULTS = None


def _figure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_genesis_initialized():
    global _GENESIS_INITIALIZED
    if not _GENESIS_INITIALIZED:
        gs.init(backend=gs.cuda)
        _GENESIS_INITIALIZED = True


def _build_lane(family: str, row: int):
    builder = ParkourLaneBuilder(Go2ParkourTeacherCfg.terrain)
    return builder.build_lane(family=family, difficulty_row=row, variant_id=0)


def _meters_to_pixels(x_m: float, y_m: float):
    scale = float(Go2ParkourTeacherCfg.terrain.horizontal_scale)
    return int(round(x_m / scale)), int(round(y_m / scale))


def _crop_first_obstacle(
    height_field_raw: np.ndarray,
    metadata,
    x_pad: float,
    y_pad: float,
):
    heights_m = height_field_raw.astype(np.float32) * float(Go2ParkourTeacherCfg.terrain.vertical_scale)
    section_x0, section_x1 = metadata.section_bounds[0]
    sec_px0, _ = _meters_to_pixels(float(section_x0), 0.0)
    sec_px1, _ = _meters_to_pixels(float(section_x1), 0.0)
    sec_px1 = max(sec_px1, sec_px0 + 1)

    # Parkour height fields are indexed as [x, y], not [y, x].
    strip = heights_m[sec_px0:sec_px1, :]
    nonzero = np.argwhere(np.abs(strip) > 1e-6)
    if nonzero.size == 0:
        y0, y1 = metadata.lane_bounds[2], metadata.lane_bounds[3]
        x0 = max(0.0, float(section_x0) - x_pad)
        x1 = min(float(Go2ParkourTeacherCfg.terrain.terrain_length), float(section_x1) + x_pad)
    else:
        x_idx0, y_idx0 = nonzero.min(axis=0)
        x_idx1, y_idx1 = nonzero.max(axis=0) + 1
        scale = float(Go2ParkourTeacherCfg.terrain.horizontal_scale)
        x0 = max(0.0, (sec_px0 + x_idx0) * scale - x_pad)
        x1 = min(float(Go2ParkourTeacherCfg.terrain.terrain_length), (sec_px0 + x_idx1) * scale + x_pad)
        y0 = max(0.0, y_idx0 * scale - y_pad)
        y1 = min(float(Go2ParkourTeacherCfg.terrain.terrain_width), y_idx1 * scale + y_pad)

    px0, py0 = _meters_to_pixels(x0, y0)
    px1, py1 = _meters_to_pixels(x1, y1)
    px1 = max(px1, px0 + 1)
    py1 = max(py1, py0 + 1)
    return height_field_raw[px0:px1, py0:py1].copy(), (x0, x1, y0, y1)


def _texture_image(mode: str) -> np.ndarray:
    cached = _TEXTURE_CACHE.get(mode)
    if cached is not None:
        return cached

    tex_size = 512

    if mode == "checkerboard":
        img = np.full((tex_size, tex_size, 3), [178, 174, 170], dtype=np.uint8)
        spacing = 32
        for i in range(0, tex_size, spacing):
            img[i : i + 2, :] = [118, 114, 110]
            img[:, i : i + 2] = [118, 114, 110]
        noise = np.random.RandomState(42).randint(-10, 11, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif mode == "concrete":
        rng = np.random.RandomState(44)
        img = np.full((tex_size, tex_size, 3), [194, 192, 188], dtype=np.uint8)
        grid_spacing = 56
        for i in range(0, tex_size, grid_spacing):
            img[i : i + 3, :] = [150, 150, 146]
            img[:, i : i + 3] = [150, 150, 146]
        noise = rng.normal(0, 6, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for _ in range(180):
            cx, cy = rng.randint(0, tex_size, 2)
            radius = rng.randint(2, 5)
            shade = np.full(3, rng.randint(122, 154), dtype=np.uint8)
            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            mask = xx * xx + yy * yy <= radius * radius
            y0, y1 = max(0, cy - radius), min(tex_size, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(tex_size, cx + radius + 1)
            local_mask = mask[y0 - (cy - radius) : y1 - (cy - radius), x0 - (cx - radius) : x1 - (cx - radius)]
            img[y0:y1, x0:x1][local_mask] = shade
    elif mode == "wood":
        rng = np.random.RandomState(45)
        img = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
        phase = np.cumsum(rng.normal(0, 0.02, tex_size))
        for y in range(tex_size):
            freq = 0.15 + 0.05 * np.sin(y * 0.01)
            wave = np.sin(np.arange(tex_size) * freq + phase + y * 0.3)
            intensity = ((wave * 0.5 + 0.5) * 40).astype(np.int16)
            img[y, :, 0] = np.clip(160 + intensity, 0, 255)
            img[y, :, 1] = np.clip(120 + intensity, 0, 255)
            img[y, :, 2] = np.clip(80 + (intensity * 0.5).astype(np.int16), 0, 255)
        noise = rng.randint(-5, 6, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif mode == "grass":
        rng = np.random.RandomState(46)
        img = np.full((tex_size, tex_size, 3), [80, 140, 60], dtype=np.uint8)
        noise_r = rng.normal(0, 12, (tex_size, tex_size)).astype(np.int16)
        noise_g = rng.normal(0, 20, (tex_size, tex_size)).astype(np.int16)
        noise_b = rng.normal(0, 10, (tex_size, tex_size)).astype(np.int16)
        img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + noise_r, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + noise_g, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + noise_b, 0, 255)
        for _ in range(40):
            cx, cy = rng.randint(0, tex_size, 2)
            radius = rng.randint(8, 20)
            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            mask = xx * xx + yy * yy <= radius * radius
            y0, y1 = max(0, cy - radius), min(tex_size, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(tex_size, cx + radius + 1)
            local_mask = mask[y0 - (cy - radius) : y1 - (cy - radius), x0 - (cx - radius) : x1 - (cx - radius)]
            patch = rng.normal(0, 5, (local_mask.sum(), 3)).astype(np.int16)
            img[y0:y1, x0:x1][local_mask] = np.clip(
                np.array([110, 95, 55]) + patch, 0, 255
            ).astype(np.uint8)
    elif mode == "stone_tiles":
        rng = np.random.RandomState(47)
        tile_size, mortar_w = 48, 3
        mortar_color = np.array([105, 100, 95], dtype=np.uint8)
        img = np.full((tex_size, tex_size, 3), mortar_color, dtype=np.uint8)
        for ty in range(0, tex_size, tile_size):
            for tx in range(0, tex_size, tile_size):
                grey = rng.randint(150, 200)
                color = np.array([grey, grey - 2, grey - 5], dtype=np.int16)
                y0 = ty + mortar_w
                x0 = tx + mortar_w
                y1 = min(ty + tile_size, tex_size)
                x1 = min(tx + tile_size, tex_size)
                if y0 < y1 and x0 < x1:
                    tile_noise = rng.randint(-8, 9, (y1 - y0, x1 - x0, 3), dtype=np.int16)
                    img[y0:y1, x0:x1] = np.clip(color + tile_noise, 0, 255).astype(np.uint8)
    elif mode == "gravel":
        rng = np.random.RandomState(48)
        img = np.full((tex_size, tex_size, 3), [160, 155, 150], dtype=np.uint8)
        for _ in range(2500):
            cx, cy = rng.randint(0, tex_size, 2)
            radius = rng.randint(2, 7)
            grey = rng.randint(100, 220)
            color = np.array([grey, grey - 3, grey - 6], dtype=np.int16)
            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            mask = xx * xx + yy * yy <= radius * radius
            y0, y1 = max(0, cy - radius), min(tex_size, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(tex_size, cx + radius + 1)
            local_mask = mask[y0 - (cy - radius) : y1 - (cy - radius), x0 - (cx - radius) : x1 - (cx - radius)]
            img[y0:y1, x0:x1][local_mask] = np.clip(color, 0, 255).astype(np.uint8)
    elif mode == "noise":
        rng = np.random.RandomState(123)
        img = rng.randint(0, 256, (tex_size, tex_size, 3), dtype=np.uint8)
    elif mode == "bricks":
        img = np.full((tex_size, tex_size, 3), [180, 100, 70], dtype=np.uint8)
        brick_h, brick_w, mortar = 32, 64, 2
        for row in range(0, tex_size, brick_h):
            img[row : row + mortar, :] = [160, 160, 155]
            offset = (brick_w // 2) if ((row // brick_h) % 2) else 0
            for col in range(offset, tex_size, brick_w):
                img[row : row + brick_h, col : col + mortar] = [160, 160, 155]
        noise = np.random.RandomState(42).randint(-10, 11, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif mode.startswith("solid_"):
        color_map = {
            "solid_red": [200, 60, 60],
            "solid_green": [60, 180, 60],
            "solid_blue": [60, 60, 200],
            "solid_yellow": [200, 200, 60],
            "solid_white": [240, 240, 240],
            "solid_dark": [40, 40, 40],
        }
        img = np.full((tex_size, tex_size, 3), color_map.get(mode, [180, 180, 180]), dtype=np.uint8)
    else:
        img = np.full((tex_size, tex_size, 3), [180, 180, 180], dtype=np.uint8)

    _TEXTURE_CACHE[mode] = img
    return img


def _create_texture_surface(mode: str):
    return gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(image_array=_texture_image(mode)),
    )


def _create_checker_surface():
    return _create_texture_surface("checkerboard")


def _create_solid_surface():
    solid = np.full((8, 8, 3), [150, 150, 150], dtype=np.uint8)
    return gs.surfaces.Rough(
        diffuse_texture=gs.textures.ImageTexture(image_array=solid),
    )


def _surface_for_texture_mode(texture_mode: str | None, add_texture: bool):
    if not add_texture or texture_mode is None:
        return _create_solid_surface()
    if texture_mode == "checkerboard":
        return _create_checker_surface()
    return _create_texture_surface(texture_mode)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    if rgb.max() <= 1.01:
        rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _terrain_camera_pose(
    family: str,
    row: int,
    obstacle_center_x: float,
    obstacle_center_y: float,
    span_x: float,
    span_y: float,
):
    if family == "stairs":
        return {
            "pos": (
                obstacle_center_x - 0.68 * span_x,
                obstacle_center_y - 0.44 * span_y,
                0.88,
            ),
            "lookat": (
                obstacle_center_x + 0.05 * span_x,
                obstacle_center_y,
                0.16 + 0.10 * row,
            ),
            "fov": 32,
        }

    if family == "gap":
        return {
            "pos": (
                obstacle_center_x - 0.66 * span_x,
                obstacle_center_y - 0.44 * span_y,
                0.88,
            ),
            "lookat": (
                obstacle_center_x + 0.03 * span_x,
                obstacle_center_y,
                -0.18,
            ),
            "fov": 30,
        }

    return {
        "pos": (
            obstacle_center_x - 0.92 * span_x,
            obstacle_center_y - 0.50 * span_y,
            0.84,
        ),
        "lookat": (
            obstacle_center_x + 0.02 * span_x,
            obstacle_center_y,
            -0.16,
        ),
        "fov": 40,
    }


def _render_lane_panel(family: str, row: int) -> np.ndarray:
    cache_key = (family, row)
    cached = _TERRAIN_RENDER_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    if family == "hurdle_block":
        image = _render_hurdle_box_panel(row)
        _TERRAIN_RENDER_CACHE[cache_key] = image.copy()
        return image

    _ensure_genesis_initialized()
    terrain, metadata = _build_lane(family, row)
    crop_raw, extent = _crop_first_obstacle(terrain.height_field_raw, metadata, 1.50, 1.24)
    x0, x1, y0, y1 = extent
    span_x = max(x1 - x0, 1e-3)
    span_y = max(y1 - y0, 1e-3)
    obstacle_center_x = 0.5 * span_x
    obstacle_center_y = 0.5 * span_y
    camera = _terrain_camera_pose(family, row, obstacle_center_x, obstacle_center_y, span_x, span_y)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        vis_options=gs.options.VisOptions(
            shadow=True,
            background_color=(0.87, 0.91, 0.98),
            ambient_light=(0.22, 0.22, 0.24),
            lights=[
                {
                    "type": "directional",
                    "dir": (-0.8, -0.55, -0.75),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 24.0,
                },
                {
                    "type": "directional",
                    "dir": (0.55, -0.2, -0.4),
                    "color": (0.95, 0.97, 1.0),
                    "intensity": 4.0,
                },
            ],
        ),
        show_viewer=False,
    )
    try:
        scene.add_entity(
            gs.morphs.Terrain(
                pos=(0.0, 0.0, 0.0),
                horizontal_scale=Go2ParkourTeacherCfg.terrain.horizontal_scale,
                vertical_scale=Go2ParkourTeacherCfg.terrain.vertical_scale * FIGURE_VERTICAL_GAIN[family],
                height_field=crop_raw,
                uv_scale=8.0,
                visualization=True,
            ),
            surface=_create_checker_surface(),
        )
        camera_entity = scene.add_camera(
            res=TERRAIN_RESOLUTION,
            pos=camera["pos"],
            lookat=camera["lookat"],
            fov=camera["fov"],
            near=0.01,
            far=50.0,
            GUI=False,
        )
        scene.build()
        rgb, _, _, _ = camera_entity.render(depth=False, segmentation=False, normal=False)
        image = _to_uint8_rgb(rgb)
    finally:
        scene.destroy()

    _TERRAIN_RENDER_CACHE[cache_key] = image.copy()
    return image


def _render_hurdle_box_panel(row: int) -> np.ndarray:
    _ensure_genesis_initialized()
    height = ParkourLaneBuilder._HURDLE_HEIGHTS[row] * 1.65
    length = ParkourLaneBuilder._HURDLE_LENGTHS[row]
    width = 2.0 * ParkourLaneBuilder._HURDLE_Y_HALVES[row]
    obstacle_center_x = 1.75
    obstacle_center_y = 1.80
    obstacle_span_x = max(length + 0.80, 1.20)
    obstacle_span_y = max(width + 1.10, 1.60)
    camera = _terrain_camera_pose(
        "hurdle_block",
        row,
        obstacle_center_x,
        obstacle_center_y,
        obstacle_span_x,
        obstacle_span_y,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        vis_options=gs.options.VisOptions(
            shadow=True,
            background_color=(0.87, 0.91, 0.98),
            ambient_light=(0.18, 0.18, 0.20),
            lights=[
                {
                    "type": "directional",
                    "dir": (-0.55, -0.85, -0.60),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 24.0,
                },
                {
                    "type": "directional",
                    "dir": (0.25, 0.45, -0.30),
                    "color": (0.92, 0.96, 1.0),
                    "intensity": 3.0,
                },
            ],
        ),
        show_viewer=False,
    )
    try:
        scene.add_entity(
            gs.morphs.Box(
                pos=(1.55, 1.80, -0.03),
                size=(3.1, 3.4, 0.06),
                visualization=True,
                fixed=True,
            ),
            surface=_create_checker_surface(),
        )
        scene.add_entity(
            gs.morphs.Box(
                pos=(1.75, 1.80, 0.5 * height),
                size=(length, width, height),
                visualization=True,
                fixed=True,
            ),
            surface=_create_solid_surface(),
        )
        camera = scene.add_camera(
            res=TERRAIN_RESOLUTION,
            pos=camera["pos"],
            lookat=(
                camera["lookat"][0],
                camera["lookat"][1],
                0.16 + 0.06 * row,
            ),
            fov=camera["fov"],
            near=0.01,
            far=50.0,
            GUI=False,
        )
        scene.build()
        rgb, _, _, _ = camera.render(depth=False, segmentation=False, normal=False)
        return _to_uint8_rgb(rgb)
    finally:
        scene.destroy()


def render_obstacle_types(output_dir: Path):
    out_path = output_dir / "parkour_obstacle_types.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), constrained_layout=True)
    families = ["stairs", "hurdle_block", "gap"]
    for ax, family in zip(axes, families):
        ax.imshow(_render_lane_panel(family, row=2))
        ax.set_axis_off()
        ax.set_title(FAMILY_LABELS[family], fontsize=10)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_terrain_curriculum(output_dir: Path):
    out_path = output_dir / "terrain_curriculum.png"
    families = ["stairs", "hurdle_block", "gap"]
    rows = [0, 1, 2, 3]

    fig, axes = plt.subplots(
        len(families),
        len(rows),
        figsize=(11.4, 7.2),
        constrained_layout=True,
    )
    for i, family in enumerate(families):
        for j, row in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(_render_lane_panel(family, row))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"Row {row}", fontsize=10)
            if j == 0:
                ax.set_ylabel(FAMILY_LABELS[family], fontsize=10)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _runtime_imports():
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import ensure_runtime_initialized, task_registry

    return ensure_runtime_initialized, task_registry


def _base_args(task: str, family: str, row: int) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        headless=True,
        cpu=False,
        num_envs=1,
        resume=False,
        sync_wandb=False,
        debug=False,
        follow_robot=False,
        motion_file=None,
        max_iterations=None,
        load_run=-1,
        ckpt=-1,
        parkour_force_family=family,
        parkour_force_row=row,
        depth_ablation="none",
    )


def _make_perception_env_and_policy(family: str, row: int, textured: bool):
    ensure_runtime_initialized, task_registry = _runtime_imports()
    args = _base_args("go2_parkour_depth_est_student", family, row)
    args.resume = True
    args.load_run = MODEL_RUNS[textured]["load_run"]
    args.ckpt = MODEL_RUNS[textured]["ckpt"]
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.parkour.force_family = family
    env_cfg.terrain.parkour.force_row = row
    env_cfg.terrain.add_texture = bool(textured)
    env_cfg.terrain.texture_mode = "checkerboard"
    env_cfg.terrain.shadow = True
    env_cfg.terrain.background_color = (0.84, 0.90, 0.98)
    env_cfg.terrain.ambient_light = (0.18, 0.18, 0.20)
    env_cfg.terrain.light_direction = (-0.85, -0.45, -0.72)
    env_cfg.terrain.light_intensity = 26.0

    # Force fresh sensor data every step so capture is deterministic and quick.
    env_cfg.sensor.depth_noise_level = 0.0
    env_cfg.sensor.depth_estimation.update_interval = 1
    env_cfg.sensor.depth_camera_config.decimation = 1
    env_cfg.sensor.rgb_camera_config.resolution = ROBOT_RGB_RESOLUTION
    env_cfg.sensor.rgb_camera_config.euler = (0.0, 0.24, 0.0)

    # Freeze nuisance randomness for figure capture.
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    policy = runner.get_inference_policy(device=env.device)
    return env, policy


def _tile_origin_xy(env) -> torch.Tensor:
    return env.simulator.env_origins[0, :2] - env.tile_half_extent


def _world_xy_from_lane_xy(env, lane_xy: torch.Tensor) -> torch.Tensor:
    return _tile_origin_xy(env) + lane_xy


def _teleport_before_first_obstacle(env, approach_margin: float = 0.55):
    env.reset()
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    lane_spawn = env.simulator.lane_spawn_pose[0]
    section_start = env.simulator.lane_section_bounds[0, 0, 0]

    local_x = torch.maximum(section_start - approach_margin, lane_spawn[0] + 0.10)
    local_y = lane_spawn[1]
    world_xy = _world_xy_from_lane_xy(env, torch.stack([local_x, local_y]))

    base_z = lane_spawn[2] + env.simulator.base_init_pos[2]
    base_pos = torch.tensor([[world_xy[0].item(), world_xy[1].item(), float(base_z)]], device=env.device)
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)
    zero_vel = torch.zeros((1, 3), device=env.device)
    zero_dof_vel = torch.zeros_like(env.simulator.dof_vel[env_ids])

    env.simulator.reset_dofs(env_ids, env.simulator.default_dof_pos[env_ids], zero_dof_vel)
    env.simulator.reset_root_states(env_ids, base_pos, base_quat, zero_vel, zero_vel)

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(2):
        env.step(zero_actions)


def _camera_vertical_fov_deg(camera_cfg) -> float:
    width, height = camera_cfg.resolution
    half_horizontal = np.deg2rad(camera_cfg.horizontal_fov_deg) * 0.5
    half_vertical = np.arctan(np.tan(half_horizontal) * (height / max(width, 1)))
    return float(np.rad2deg(half_vertical * 2.0))


def _robot_sensor_cfg():
    return SimpleNamespace(
        rgb_camera_config=SimpleNamespace(
            resolution=ROBOT_RGB_RESOLUTION,
            horizontal_fov_deg=87,
            near_plane=0.1,
            far_plane=10.0,
            pos=(0.327, 0.0, 0.043),
            euler=(0.0, 0.22, 0.0),
        ),
        depth_estimation=SimpleNamespace(
            enabled=True,
            model_type="depth_anything_v2_metric_outdoor",
            model_size="base",
        ),
    )


def _robot_view_camera_pose(metadata, family: str):
    section_start, section_end = metadata.section_bounds[0]
    lane_center_y = 0.5 * float(metadata.lane_bounds[2] + metadata.lane_bounds[3])
    camera_height = float(Go2ParkourTeacherCfg.init_state.pos[2]) + 0.043

    if family == "gap":
        return {
            "pos": (float(section_start) - 0.34, lane_center_y - 0.14, camera_height),
            "lookat": (float(section_end) + 0.34, lane_center_y + 0.10, -0.30),
        }
    return {
        "pos": (float(section_start) - 0.50, lane_center_y - 0.15, camera_height),
        "lookat": (float(section_end) + 0.22, lane_center_y + 0.09, 0.16),
    }


def _render_robot_view_rgb(
    family: str,
    row: int,
    textured: bool,
    texture_mode: str | None,
    sensor_cfg,
) -> np.ndarray:
    _ensure_genesis_initialized()
    terrain, metadata = _build_lane(family, row)
    camera_pose = _robot_view_camera_pose(metadata, family)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        vis_options=gs.options.VisOptions(
            shadow=True,
            background_color=(0.84, 0.90, 0.98),
            ambient_light=(0.06, 0.06, 0.07),
            lights=[
                {
                    "type": "directional",
                    "dir": (-0.20, -1.0, -0.55),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 30.0,
                },
                {
                    "type": "directional",
                    "dir": (0.25, 0.55, -0.35),
                    "color": (0.92, 0.95, 1.0),
                    "intensity": 4.0,
                },
            ],
        ),
        show_viewer=False,
    )
    try:
        scene.add_entity(
            gs.morphs.Terrain(
                pos=(0.0, 0.0, 0.0),
                horizontal_scale=Go2ParkourTeacherCfg.terrain.horizontal_scale,
                vertical_scale=Go2ParkourTeacherCfg.terrain.vertical_scale,
                height_field=terrain.height_field_raw,
                uv_scale=10.0,
                visualization=True,
            ),
            surface=_surface_for_texture_mode(texture_mode, textured),
        )
        camera = scene.add_camera(
            res=sensor_cfg.rgb_camera_config.resolution,
            pos=camera_pose["pos"],
            lookat=camera_pose["lookat"],
            fov=_camera_vertical_fov_deg(sensor_cfg.rgb_camera_config),
            near=sensor_cfg.rgb_camera_config.near_plane,
            far=sensor_cfg.rgb_camera_config.far_plane,
            GUI=False,
        )
        scene.build()
        rgb, _, _, _ = camera.render(depth=False, segmentation=False, normal=False)
        return _to_uint8_rgb(rgb)
    finally:
        scene.destroy()


def _estimate_depth_anything(rgb: np.ndarray, sensor_cfg) -> np.ndarray:
    global _DEPTH_ESTIMATOR
    if _DEPTH_ESTIMATOR is None:
        _DEPTH_ESTIMATOR = create_depth_estimator(sensor_cfg, "cuda")
    rgb = np.ascontiguousarray(rgb)
    output = _DEPTH_ESTIMATOR.estimate({"rgb": torch.from_numpy(rgb).unsqueeze(0)})
    return output.depth[0].detach().cpu().numpy()


def _select_capture_pose(env, policy, family: str):
    best_pose = None
    best_score = None
    target_dist = 0.18 if family == "gap" else 0.14

    obs, _teacher_obs, student_depth, _ = env.get_observations()
    for _ in range(120):
        with torch.inference_mode():
            actions = policy(obs, student_depth)
        obs, _teacher_obs, student_depth, depth_updated, _rews, dones, _infos = env.step(actions.detach())
        if not depth_updated[0]:
            continue

        dist = float(env.simulator.lane_section_bounds[0, 0, 0].item() - env.local_base_pos[0, 0].item())
        roll_pitch = torch.abs(env.simulator.base_euler[0, :2]).sum().item()
        score = -abs(dist - target_dist) - 0.08 * roll_pitch
        if best_score is None or score > best_score:
            best_score = score
            best_pose = {
                "local_base_pos": env.local_base_pos[0].detach().cpu().numpy().copy(),
                "base_quat_gs": env.simulator._base_quat_gs[0].detach().cpu().numpy().copy(),
            }
        if dist < 0.08 or bool(dones[0].item()):
            break

    if best_pose is None:
        best_pose = {
            "local_base_pos": env.local_base_pos[0].detach().cpu().numpy().copy(),
            "base_quat_gs": env.simulator._base_quat_gs[0].detach().cpu().numpy().copy(),
        }
    return best_pose


def _camera_pose_from_robot_pose(simulator, sensor_cfg, pose):
    base_transform = gu.trans_R_to_T(
        np.asarray(pose["local_base_pos"], dtype=np.float32),
        gu.quat_to_R(np.asarray(pose["base_quat_gs"], dtype=np.float32)),
    )
    camera_local = simulator._camera_offset_transform(
        sensor_cfg.rgb_camera_config.pos,
        sensor_cfg.rgb_camera_config.euler,
    )
    camera_world = base_transform @ camera_local
    camera_pos = camera_world[:3, 3]
    forward = -camera_world[:3, 2]
    lookat = camera_pos + 1.25 * forward
    return camera_pos, lookat


def _render_pose_rgb(
    simulator,
    sensor_cfg,
    family: str,
    row: int,
    textured: bool,
    texture_mode: str | None,
    pose,
) -> np.ndarray:
    _ensure_genesis_initialized()
    terrain, _metadata = _build_lane(family, row)
    camera_pos, lookat = _camera_pose_from_robot_pose(simulator, sensor_cfg, pose)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        vis_options=gs.options.VisOptions(
            shadow=True,
            background_color=(0.84, 0.90, 0.98),
            ambient_light=(0.14, 0.14, 0.15),
            lights=[
                {
                    "type": "directional",
                    "dir": (-0.85, -0.42, -0.72),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 24.0,
                },
            ],
        ),
        show_viewer=False,
    )
    try:
        scene.add_entity(
            gs.morphs.Terrain(
                pos=(0.0, 0.0, 0.0),
                horizontal_scale=Go2ParkourTeacherCfg.terrain.horizontal_scale,
                vertical_scale=Go2ParkourTeacherCfg.terrain.vertical_scale,
                height_field=terrain.height_field_raw,
                uv_scale=10.0,
                visualization=True,
            ),
            surface=_surface_for_texture_mode(texture_mode, textured),
        )
        camera = scene.add_camera(
            res=sensor_cfg.rgb_camera_config.resolution,
            pos=tuple(float(x) for x in camera_pos),
            lookat=tuple(float(x) for x in lookat),
            fov=_camera_vertical_fov_deg(sensor_cfg.rgb_camera_config),
            near=sensor_cfg.rgb_camera_config.near_plane,
            far=sensor_cfg.rgb_camera_config.far_plane,
            GUI=False,
        )
        scene.build()
        rgb, _, _, _ = camera.render(depth=False, segmentation=False, normal=False)
        return _to_uint8_rgb(rgb)
    finally:
        scene.destroy()


def _colorize_processed_depth(depth_2d: np.ndarray) -> np.ndarray:
    valid = depth_2d[np.isfinite(depth_2d)]
    if valid.size == 0:
        valid = np.array([0.0, 1.0], dtype=np.float32)
    lo = float(np.quantile(valid, 0.02))
    hi = float(np.quantile(valid, 0.98))
    if hi - lo < 1e-4:
        hi = lo + 1.0
    normalized = 1.0 - np.clip((depth_2d - lo) / (hi - lo), 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(normalized)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _save_raw_image(path: Path, image: np.ndarray):
    Image.fromarray(image).save(path)


def _domain_invariance_results():
    global _DOMAIN_INVARIANCE_RESULTS
    if _DOMAIN_INVARIANCE_RESULTS is None:
        results_path = Path(__file__).with_name("domain_invariance_results.json")
        import json

        with results_path.open("r", encoding="utf-8") as f:
            _DOMAIN_INVARIANCE_RESULTS = json.load(f)
    return _DOMAIN_INVARIANCE_RESULTS


def _dsq_for_perturbation(perturbation: str) -> float:
    results = _domain_invariance_results()["results"]
    for row in results:
        if row.get("student") == "DA2-base+tex" and row.get("perturbation") == perturbation:
            return float(row["depth_signal_quality"])
    raise KeyError(f"Missing DSQ for perturbation {perturbation}")


def _panel_pair(rgb: np.ndarray, depth_rgb: np.ndarray, gutter: int = 4) -> np.ndarray:
    height = max(rgb.shape[0], depth_rgb.shape[0])
    width = rgb.shape[1] + depth_rgb.shape[1] + gutter
    pair = np.full((height, width, 3), 255, dtype=np.uint8)
    pair[: rgb.shape[0], : rgb.shape[1]] = rgb
    x1 = rgb.shape[1] + gutter
    pair[: depth_rgb.shape[0], x1 : x1 + depth_rgb.shape[1]] = depth_rgb
    return pair


def _capture_robot_view(family: str, row: int, texture_mode: str | None, add_texture: bool, raw_dir: Path):
    sensor_cfg = _robot_sensor_cfg()
    rgb = _render_robot_view_rgb(
        family=family,
        row=row,
        textured=add_texture,
        texture_mode=texture_mode,
        sensor_cfg=sensor_cfg,
    )
    inferred = _estimate_depth_anything(rgb, sensor_cfg)

    top_crop = max(8, int(round(0.08 * rgb.shape[0])))
    bottom_crop = max(10, int(round(0.10 * rgb.shape[0])))
    side_crop = max(2, int(round(0.01 * rgb.shape[1])))
    rgb = rgb[top_crop : rgb.shape[0] - bottom_crop, side_crop : rgb.shape[1] - side_crop, :]
    inferred = inferred[top_crop : inferred.shape[0] - bottom_crop, side_crop : inferred.shape[1] - side_crop]
    depth_rgb = _colorize_processed_depth(inferred)

    suffix = texture_mode or "none"
    family_tag = "hurdle" if family == "hurdle_block" else family
    _save_raw_image(raw_dir / f"{family_tag}_{suffix}_rgb.png", rgb)
    _save_raw_image(raw_dir / f"{family_tag}_{suffix}_da2.png", depth_rgb)
    return rgb, depth_rgb


def render_robot_perception(output_dir: Path):
    out_path = output_dir / "da2_texture_comparison.png"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    families = [("gap", "Gap"), ("hurdle_block", "Hurdle")]
    columns = TEXTURE_COMPARISON_COLUMNS

    fig, axes = plt.subplots(len(families), len(columns), figsize=(13.6, 2.55))
    for row_idx, (family, family_label) in enumerate(families):
        for col_idx, column in enumerate(columns):
            rgb, depth = _capture_robot_view(
                family=family,
                row=2,
                texture_mode=column["texture_mode"],
                add_texture=column["add_texture"],
                raw_dir=raw_dir,
            )
            ax = axes[row_idx, col_idx]
            ax.imshow(_panel_pair(rgb, depth))
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(column["label"], fontsize=10, pad=8)
            if col_idx == 0:
                ax.set_ylabel(family_label, fontsize=10)

            for spine in ax.spines.values():
                spine.set_visible(False)

    for col_idx, column in enumerate(columns):
        dsq = _dsq_for_perturbation(column["perturbation"])
        x = (col_idx + 0.5) / len(columns)
        fig.text(x, 0.035, f"DSQ {dsq:.2f}", ha="center", va="bottom", fontsize=9, color="#4a4a4a")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.14, wspace=0.025, hspace=0.01)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["obstacle_types", "terrain_curriculum", "robot_perception"],
        choices=["obstacle_types", "terrain_curriculum", "robot_perception"],
        help="Which figure groups to render.",
    )
    parser.add_argument(
        "--output-dir",
        default="thesis/figures",
        help="Directory where final figure assets should be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = _figure_dir(args.output_dir)

    rendered = []
    if "obstacle_types" in args.figures:
        rendered.append(render_obstacle_types(output_dir))
    if "terrain_curriculum" in args.figures:
        rendered.append(render_terrain_curriculum(output_dir))
    if "robot_perception" in args.figures:
        rendered.append(render_robot_perception(output_dir))

    print("Generated figure assets:")
    for path in rendered:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
