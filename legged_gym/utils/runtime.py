from __future__ import annotations

from legged_gym import SIMULATOR


def ensure_runtime_initialized(args=None) -> None:
    """Initialize the active simulator runtime exactly once."""
    if SIMULATOR != "genesis":
        return

    import genesis as gs

    if gs._initialized:
        return

    use_cpu = bool(getattr(args, "cpu", False)) if args is not None else False
    gs.init(
        backend=gs.cpu if use_cpu else gs.gpu,
        logging_level="warning",
    )
