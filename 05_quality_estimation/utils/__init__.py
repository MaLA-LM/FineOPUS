"""Shared utility helpers for scoring workflows."""

__all__ = [
    "build_frames",
    "collect_directions",
    "run_scoring",
]


def __getattr__(name: str):
    if name == "build_frames":
        from utils.frames import build_frames

        return build_frames
    if name == "collect_directions":
        from utils.runner import collect_directions

        return collect_directions
    if name == "run_scoring":
        from utils.runner import run_scoring

        return run_scoring
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
