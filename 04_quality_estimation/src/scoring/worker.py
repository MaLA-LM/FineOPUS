from __future__ import annotations

import time
from pathlib import Path


def lock_path(lock_root: Path, lock_id: str) -> Path:
    return lock_root / lock_id


def _clear_lock(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        try:
            child.unlink()  # delete file
        except FileNotFoundError:
            pass
    try:
        path.rmdir()
    except FileNotFoundError:
        pass


def try_acquire_lock(
    lock_root: Path,
    lock_id: str,
    owner: str,
    *,
    timeout_seconds: int = 2 * 60 * 60,  # 2 hours
) -> Path | None:
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_path(lock_root, lock_id)
    if path.exists() and timeout_seconds > 0:
        age = time.time() - path.stat().st_mtime
        if age > timeout_seconds:
            _clear_lock(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None
    info_path = path / "owner.txt"
    info_path.write_text(owner, encoding="utf-8")
    return path


def release_lock(path: Path | None) -> None:
    if path is None:
        return
    _clear_lock(path)
