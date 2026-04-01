from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from legged_gym import PROJECT_CACHE_DIR


class DepthEstimatorError(RuntimeError):
    """Raised when a depth-estimation backend cannot be initialized or executed."""


@dataclass
class DepthEstimatorOutput:
    depth: torch.Tensor
    latency_ms: float
    backend_name: str


@dataclass(frozen=True)
class DepthModelSpec:
    name: str
    model_type: str
    model_size: str
    backend: str
    model_id: str | None = None
    hf_filename: str | None = None
    encoder: str | None = None
    backbone: str | None = None
    repo_url: str | None = None
    torch_hub_repo: str | None = None
    torch_hub_entry: str | None = None
    display_name: str | None = None


DEPTH_MODEL_SPECS: dict[str, DepthModelSpec] = {
    "depth_anything_v2_small": DepthModelSpec(
        name="depth_anything_v2_small",
        model_type="depth_anything_v2",
        model_size="small",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        display_name="Depth Anything V2 Small",
    ),
    "depth_anything_v2_base": DepthModelSpec(
        name="depth_anything_v2_base",
        model_type="depth_anything_v2",
        model_size="base",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Base-hf",
        display_name="Depth Anything V2 Base",
    ),
    "depth_anything_v2_large": DepthModelSpec(
        name="depth_anything_v2_large",
        model_type="depth_anything_v2",
        model_size="large",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Large-hf",
        display_name="Depth Anything V2 Large",
    ),
    "depth_anything_v2_metric_indoor_small": DepthModelSpec(
        name="depth_anything_v2_metric_indoor_small",
        model_type="depth_anything_v2_metric_indoor",
        model_size="small",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        display_name="Depth Anything V2 Metric Indoor Small",
    ),
    "depth_anything_v2_metric_indoor_base": DepthModelSpec(
        name="depth_anything_v2_metric_indoor_base",
        model_type="depth_anything_v2_metric_indoor",
        model_size="base",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        display_name="Depth Anything V2 Metric Indoor Base",
    ),
    "depth_anything_v2_metric_indoor_large": DepthModelSpec(
        name="depth_anything_v2_metric_indoor_large",
        model_type="depth_anything_v2_metric_indoor",
        model_size="large",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        display_name="Depth Anything V2 Metric Indoor Large",
    ),
    "depth_anything_v2_metric_outdoor_small": DepthModelSpec(
        name="depth_anything_v2_metric_outdoor_small",
        model_type="depth_anything_v2_metric_outdoor",
        model_size="small",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        display_name="Depth Anything V2 Metric Outdoor Small",
    ),
    "depth_anything_v2_metric_outdoor_base": DepthModelSpec(
        name="depth_anything_v2_metric_outdoor_base",
        model_type="depth_anything_v2_metric_outdoor",
        model_size="base",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        display_name="Depth Anything V2 Metric Outdoor Base",
    ),
    "depth_anything_v2_metric_outdoor_large": DepthModelSpec(
        name="depth_anything_v2_metric_outdoor_large",
        model_type="depth_anything_v2_metric_outdoor",
        model_size="large",
        backend="transformers",
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        display_name="Depth Anything V2 Metric Outdoor Large",
    ),
    "video_depth_anything_metric_small": DepthModelSpec(
        name="video_depth_anything_metric_small",
        model_type="video_depth_anything_metric",
        model_size="small",
        backend="video_depth_anything",
        model_id="depth-anything/Metric-Video-Depth-Anything-Small",
        hf_filename="metric_video_depth_anything_vits.pth",
        encoder="vits",
        repo_url="https://github.com/DepthAnything/Video-Depth-Anything.git",
        display_name="Video Depth Anything Metric Small",
    ),
    "video_depth_anything_metric_base": DepthModelSpec(
        name="video_depth_anything_metric_base",
        model_type="video_depth_anything_metric",
        model_size="base",
        backend="video_depth_anything",
        model_id="depth-anything/Metric-Video-Depth-Anything-Base",
        hf_filename="metric_video_depth_anything_vitb.pth",
        encoder="vitb",
        repo_url="https://github.com/DepthAnything/Video-Depth-Anything.git",
        display_name="Video Depth Anything Metric Base",
    ),
    "video_depth_anything_metric_large": DepthModelSpec(
        name="video_depth_anything_metric_large",
        model_type="video_depth_anything_metric",
        model_size="large",
        backend="video_depth_anything",
        model_id="depth-anything/Metric-Video-Depth-Anything-Large",
        hf_filename="metric_video_depth_anything_vitl.pth",
        encoder="vitl",
        repo_url="https://github.com/DepthAnything/Video-Depth-Anything.git",
        display_name="Video Depth Anything Metric Large",
    ),
    "metric3d_v2_small": DepthModelSpec(
        name="metric3d_v2_small",
        model_type="metric3d_v2",
        model_size="small",
        backend="metric3d",
        torch_hub_repo="yvanyin/metric3d",
        torch_hub_entry="metric3d_vit_small",
        display_name="Metric3D v2 Small",
    ),
    "metric3d_v2_large": DepthModelSpec(
        name="metric3d_v2_large",
        model_type="metric3d_v2",
        model_size="large",
        backend="metric3d",
        torch_hub_repo="yvanyin/metric3d",
        torch_hub_entry="metric3d_vit_large",
        display_name="Metric3D v2 Large",
    ),
    "metric3d_v2_giant": DepthModelSpec(
        name="metric3d_v2_giant",
        model_type="metric3d_v2",
        model_size="giant",
        backend="metric3d",
        torch_hub_repo="yvanyin/metric3d",
        torch_hub_entry="metric3d_vit_giant2",
        display_name="Metric3D v2 Giant2",
    ),
    "unidepth_v2_small": DepthModelSpec(
        name="unidepth_v2_small",
        model_type="unidepth_v2",
        model_size="small",
        backend="unidepth",
        torch_hub_repo="lpiccinelli-eth/UniDepth",
        torch_hub_entry="UniDepth",
        backbone="vits14",
        display_name="UniDepth V2 Small",
    ),
    "unidepth_v2_base": DepthModelSpec(
        name="unidepth_v2_base",
        model_type="unidepth_v2",
        model_size="base",
        backend="unidepth",
        torch_hub_repo="lpiccinelli-eth/UniDepth",
        torch_hub_entry="UniDepth",
        backbone="vitb14",
        display_name="UniDepth V2 Base",
    ),
    "unidepth_v2_large": DepthModelSpec(
        name="unidepth_v2_large",
        model_type="unidepth_v2",
        model_size="large",
        backend="unidepth",
        torch_hub_repo="lpiccinelli-eth/UniDepth",
        torch_hub_entry="UniDepth",
        backbone="vitl14",
        display_name="UniDepth V2 Large",
    ),
}

DEPTH_MODEL_VARIANTS = {
    (spec.model_type, spec.model_size): spec.name for spec in DEPTH_MODEL_SPECS.values()
}


class DepthEstimatorBackend(ABC):
    def __init__(self, sensor_cfg, device: str):
        self.sensor_cfg = sensor_cfg
        self.cfg = sensor_cfg.depth_estimation
        self.device = torch.device(device)
        self.model_name = resolve_depth_model_name(
            model_type=getattr(self.cfg, "model_type", None),
            model_size=getattr(self.cfg, "model_size", None),
        )
        self.spec = DEPTH_MODEL_SPECS[self.model_name]

    @property
    def backend_name(self) -> str:
        return self.model_name

    @property
    def display_name(self) -> str:
        return self.spec.display_name or self.spec.name.replace("_", " ")

    @abstractmethod
    def estimate(self, inputs: Mapping[str, Any]) -> DepthEstimatorOutput:
        """Run depth estimation for the given inputs."""

    def _autocast_enabled(self) -> bool:
        return self.device.type == "cuda"

    def _autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.float32

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare_rgb_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise DepthEstimatorError(
                f"Expected RGB tensor shaped [B, H, W, 3], got {tuple(rgb.shape)}"
            )
        return rgb

    def _rgb_batch_to_chw(self, rgb: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        rgb = self._prepare_rgb_batch(rgb)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float()
        return rgb.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)

    def _rgb_batch_to_numpy(self, rgb: torch.Tensor) -> list[np.ndarray]:
        rgb = self._prepare_rgb_batch(rgb)
        return [frame.detach().cpu().numpy() for frame in rgb]

    def _camera_intrinsics(self, width: int, height: int) -> torch.Tensor:
        fov_deg = float(self.sensor_cfg.rgb_camera_config.horizontal_fov_deg)
        cx = width * 0.5
        cy = height * 0.5
        fx = width * 0.5 / np.tan(np.deg2rad(fov_deg) * 0.5)
        fy = fx
        return torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )


class DepthAnythingV2Backend(DepthEstimatorBackend):
    def __init__(self, sensor_cfg, device: str):
        super().__init__(sensor_cfg, device)
        self._image_processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise DepthEstimatorError(
                "Depth Anything V2 backends require the optional 'transformers' dependency."
            ) from exc

        self._image_processor = transformers.AutoImageProcessor.from_pretrained(
            self.spec.model_id,
            use_fast=True,
        )
        self._model = transformers.AutoModelForDepthEstimation.from_pretrained(
            self.spec.model_id,
            dtype=self._autocast_dtype(),
        ).to(self.device)
        self._model.eval()

    def estimate(self, inputs: Mapping[str, Any]) -> DepthEstimatorOutput:
        self._ensure_loaded()
        rgb = self._prepare_rgb_batch(inputs["rgb"])
        target_size = tuple(int(dim) for dim in rgb.shape[1:3])

        processed = self._image_processor(images=self._rgb_batch_to_numpy(rgb), return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device, dtype=self._autocast_dtype())

        self._synchronize()
        started = time.perf_counter()
        max_batch = 128
        n = pixel_values.shape[0]
        with torch.inference_mode():
            if n <= max_batch:
                outputs = self._model(pixel_values=pixel_values)
                raw_depth = outputs.predicted_depth
            else:
                chunks = []
                for i in range(0, n, max_batch):
                    out = self._model(pixel_values=pixel_values[i : i + max_batch])
                    chunks.append(out.predicted_depth)
                raw_depth = torch.cat(chunks, dim=0)
        depth = F.interpolate(
            raw_depth.unsqueeze(1).float(),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        self._synchronize()
        return DepthEstimatorOutput(
            depth=depth,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            backend_name=self.display_name,
        )


class VideoDepthAnythingStreamingBackend(DepthEstimatorBackend):
    def __init__(self, sensor_cfg, device: str):
        super().__init__(sensor_cfg, device)
        self._model = None
        self._states: dict[int, dict[str, Any]] = {}

    def _repo_root(self) -> Path:
        return Path(PROJECT_CACHE_DIR) / "external" / "video_depth_anything"

    def _ensure_repo_checkout(self):
        repo_root = self._repo_root()
        if repo_root.exists():
            return repo_root
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(
            ["git", "clone", "--depth", "1", self.spec.repo_url, str(repo_root)],
            check=True,
        )
        return repo_root

    def _ensure_loaded(self):
        if self._model is not None:
            return
        repo_root = self._ensure_repo_checkout()
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        try:
            module = importlib.import_module("video_depth_anything.video_depth_stream")
        except ModuleNotFoundError as exc:
            raise DepthEstimatorError(
                "Video Depth Anything requires the optional dependencies declared in the upstream repository."
            ) from exc

        checkpoint_path = hf_hub_download(repo_id=self.spec.model_id, filename=self.spec.hf_filename)
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        model = module.VideoDepthAnything(**model_configs[self.spec.encoder])
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        self._model = model.to(self.device).eval()

    def _state_for_env(self, env_id: int) -> dict[str, Any]:
        return self._states.setdefault(
            env_id,
            {
                "transform": None,
                "frame_id_list": [],
                "frame_cache_list": [],
                "id": -1,
                "frame_height": None,
                "frame_width": None,
            },
        )

    def _swap_state(self, state: dict[str, Any]):
        self._model.transform = state["transform"]
        self._model.frame_id_list = list(state["frame_id_list"])
        self._model.frame_cache_list = list(state["frame_cache_list"])
        self._model.id = int(state["id"])
        if state["frame_height"] is not None:
            self._model.frame_height = int(state["frame_height"])
            self._model.frame_width = int(state["frame_width"])
        elif hasattr(self._model, "frame_height"):
            delattr(self._model, "frame_height")
            delattr(self._model, "frame_width")

    def _capture_state(self, state: dict[str, Any]):
        state["transform"] = self._model.transform
        state["frame_id_list"] = list(self._model.frame_id_list)
        state["frame_cache_list"] = list(self._model.frame_cache_list)
        state["id"] = int(self._model.id)
        state["frame_height"] = getattr(self._model, "frame_height", None)
        state["frame_width"] = getattr(self._model, "frame_width", None)

    def estimate(self, inputs: Mapping[str, Any]) -> DepthEstimatorOutput:
        self._ensure_loaded()
        rgb = self._prepare_rgb_batch(inputs["rgb"])
        env_ids = [int(env_id) for env_id in inputs["env_ids"]]

        self._synchronize()
        started = time.perf_counter()
        depth_frames = []
        for env_id, frame in zip(env_ids, self._rgb_batch_to_numpy(rgb), strict=True):
            state = self._state_for_env(env_id)
            self._swap_state(state)
            depth_np = self._model.infer_video_depth_one(
                frame,
                input_size=518,
                device=self.device.type,
                fp32=self.device.type != "cuda",
            )
            self._capture_state(state)
            depth_frames.append(torch.from_numpy(depth_np))
        depth = torch.stack(depth_frames, dim=0).to(self.device, dtype=torch.float32)
        self._synchronize()
        return DepthEstimatorOutput(
            depth=depth,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            backend_name=self.display_name,
        )


class Metric3Dv2Backend(DepthEstimatorBackend):
    INPUT_SIZE = (616, 1064)
    _MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
    _STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)

    def __init__(self, sensor_cfg, device: str):
        super().__init__(sensor_cfg, device)
        self._model = None

    @staticmethod
    def _depth_map_2d(depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float()
        while depth.ndim > 2 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        while depth.ndim > 2 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.ndim != 2:
            raise DepthEstimatorError(
                f"Metric3D returned an unexpected depth shape: {tuple(depth.shape)}"
            )
        return depth

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            self._model = torch.hub.load(
                self.spec.torch_hub_repo,
                self.spec.torch_hub_entry,
                pretrain=True,
                trust_repo=True,
            ).to(self.device).eval()
        except Exception as exc:
            raise DepthEstimatorError(
                "Metric3D v2 could not be loaded from the official torch.hub entry."
            ) from exc

    def estimate(self, inputs: Mapping[str, Any]) -> DepthEstimatorOutput:
        self._ensure_loaded()
        rgb = self._prepare_rgb_batch(inputs["rgb"])
        batch_size, height, width, _ = rgb.shape
        original_size = (height, width)
        intrinsics = self._camera_intrinsics(width, height)

        resized_tensors = []
        pad_infos = []
        for frame in self._rgb_batch_to_numpy(rgb):
            scale = min(self.INPUT_SIZE[0] / height, self.INPUT_SIZE[1] / width)
            resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
            pad_h = self.INPUT_SIZE[0] - resized.shape[0]
            pad_w = self.INPUT_SIZE[1] - resized.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(123.675, 116.28, 103.53),
            )
            tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
            tensor = (tensor - self._MEAN) / self._STD
            resized_tensors.append(tensor)
            pad_infos.append((pad_top, pad_bottom, pad_left, pad_right, scale))

        model_input = torch.stack(resized_tensors, dim=0).to(self.device)
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._autocast_dtype(),
                enabled=self._autocast_enabled(),
            ):
                pred_depth, _, _ = self._model.inference({"input": model_input})

        output_depth = []
        for idx in range(batch_size):
            pad_top, pad_bottom, pad_left, pad_right, scale = pad_infos[idx]
            depth = self._depth_map_2d(pred_depth[idx])
            depth = depth[
                pad_top : depth.shape[0] - pad_bottom,
                pad_left : depth.shape[1] - pad_right,
            ]
            depth = F.interpolate(
                depth[None, None, :, :],
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            focal_px = intrinsics[0, 0] * scale
            depth = (depth * (focal_px / 1000.0)).clamp(0.0, 300.0)
            output_depth.append(depth)
        self._synchronize()
        return DepthEstimatorOutput(
            depth=torch.stack(output_depth, dim=0),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            backend_name=self.display_name,
        )


class UniDepthV2Backend(DepthEstimatorBackend):
    def __init__(self, sensor_cfg, device: str):
        super().__init__(sensor_cfg, device)
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            self._model = torch.hub.load(
                self.spec.torch_hub_repo,
                self.spec.torch_hub_entry,
                version="v2",
                backbone=self.spec.backbone,
                pretrained=True,
                trust_repo=True,
            ).to(self.device).eval()
        except Exception as exc:
            raise DepthEstimatorError(
                "UniDepth V2 could not be loaded from the official torch.hub entry."
            ) from exc

    def estimate(self, inputs: Mapping[str, Any]) -> DepthEstimatorOutput:
        self._ensure_loaded()
        rgb = self._prepare_rgb_batch(inputs["rgb"])
        batch_size, height, width, _ = rgb.shape
        rgb_chw = self._rgb_batch_to_chw(rgb).to(self.device, dtype=torch.uint8)
        intrinsics = self._camera_intrinsics(width, height).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)

        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._autocast_dtype(),
                enabled=self._autocast_enabled(),
            ):
                predictions = self._model.infer(rgb_chw, intrinsics)
        depth = predictions["depth"].squeeze(1).float()
        self._synchronize()
        return DepthEstimatorOutput(
            depth=depth,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            backend_name=self.display_name,
        )


BACKEND_BY_TYPE = {
    "transformers": DepthAnythingV2Backend,
    "video_depth_anything": VideoDepthAnythingStreamingBackend,
    "metric3d": Metric3Dv2Backend,
    "unidepth": UniDepthV2Backend,
}


def create_depth_estimator(sensor_cfg, device: str) -> DepthEstimatorBackend:
    model_name = resolve_depth_model_name(
        model_type=getattr(sensor_cfg.depth_estimation, "model_type", None),
        model_size=getattr(sensor_cfg.depth_estimation, "model_size", None),
    )
    spec = DEPTH_MODEL_SPECS[model_name]
    backend_cls = BACKEND_BY_TYPE[spec.backend]
    return backend_cls(sensor_cfg, device)


def resolve_depth_model_name(
    model_type: str | None,
    model_size: str | None,
) -> str:
    model_type = model_type or "depth_anything_v2"
    model_size = model_size or "small"
    key = (model_type, model_size)
    if key not in DEPTH_MODEL_VARIANTS:
        available = ", ".join(
            f"{variant_type}:{variant_size}"
            for variant_type, variant_size in sorted(DEPTH_MODEL_VARIANTS)
        )
        raise DepthEstimatorError(
            f"Unsupported depth model selection '{model_type}:{model_size}'. "
            f"Expected one of: {available}."
        )
    return DEPTH_MODEL_VARIANTS[key]
