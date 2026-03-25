"""Depth-input ablations for parkour student policies (play / offline eval).

Use to test whether behavior depends on the depth stream vs proprioception-only shortcuts.
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_student_depth_ablation(student_depth: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    """Replace or corrupt `student_depth` before the policy forward.

    Args:
        student_depth: ``(B, C, H, W)`` tensor from the environment.
        mode:
            - ``none``: no change.
            - ``zero``: zeros — removes all structure; CNN+GRU still run.
            - ``noise``: IID uniform in ``[-0.5, 0.5]`` — same shape as normalized depth, no scene.
            - ``shuffle``: permute along batch — each env sees another env's depth (needs ``B >= 2``).

    Returns:
        Tensor to pass to ``policy(obs, depth)``.
    """
    if mode in (None, "", "none"):
        return student_depth
    if mode == "zero":
        return torch.zeros_like(student_depth)
    if mode == "noise":
        return torch.rand_like(student_depth) - 0.5
    if mode == "shuffle":
        b = int(student_depth.shape[0])
        if b < 2:
            return torch.rand_like(student_depth) - 0.5
        perm = torch.randperm(b, device=student_depth.device)
        return student_depth[perm]
    raise ValueError(f"Unknown depth ablation mode: {mode!r}")
