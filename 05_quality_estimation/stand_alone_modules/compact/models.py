from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelStats:
    model: str
    files: int = 0
    seen: int = 0
    kept: int = 0
    invalid: int = 0
    uncommitted: int = 0
    skipped: int = 0

    def add(self, other: ModelStats) -> None:
        self.files += other.files
        self.seen += other.seen
        self.kept += other.kept
        self.invalid += other.invalid
        self.uncommitted += other.uncommitted
        self.skipped += other.skipped


@dataclass
class PendingRows:
    pending_rows: list[dict] = field(default_factory=list)
    current_pair: list[dict] = field(default_factory=list)
    pending_pairs: int = 0
