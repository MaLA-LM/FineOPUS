from __future__ import annotations

from execution.base import ExecutionStrategy


def _load_flores_array() -> ExecutionStrategy:
    from execution.flores_array.runner import FloresArrayExecutor

    return FloresArrayExecutor()


def _load_opus_queue() -> ExecutionStrategy:
    from execution.opus_queue.executor import OpusQueueExecutor

    return OpusQueueExecutor()


_EXECUTOR_LOADERS = {
    "flores_array": _load_flores_array,
    "opus_queue": _load_opus_queue,
}

_EXECUTORS: dict[str, ExecutionStrategy] = {}


def get_executor(name: str) -> ExecutionStrategy:
    key = name.strip().lower()
    try:
        return _EXECUTORS[key]
    except KeyError:
        pass

    try:
        executor = _EXECUTOR_LOADERS[key]()
    except KeyError as exc:
        supported = ", ".join(list_executor_names())
        raise SystemExit(
            f"Unknown execution strategy '{name}'. Supported: {supported}."
        ) from exc

    _EXECUTORS[key] = executor
    return executor


def list_executor_names() -> list[str]:
    return sorted(_EXECUTOR_LOADERS.keys())


__all__ = ["ExecutionStrategy", "get_executor", "list_executor_names"]
