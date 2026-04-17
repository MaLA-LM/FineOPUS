from __future__ import annotations

__all__ = ["OpusQueueExecutor"]


def __getattr__(name: str):
    if name == "OpusQueueExecutor":
        from execution.opus_queue.executor import OpusQueueExecutor

        return OpusQueueExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
