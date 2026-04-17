from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, Protocol

from dataset.mediator import DatasetAdapter

ScoreEntry = Callable[[Any], object]


class ExecutionStrategy(Protocol):
    name: str

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None: ...

    def run(
        self,
        args: argparse.Namespace,
        dataset: DatasetAdapter,
        model_tag: str,
        score_entry: ScoreEntry,
    ) -> None: ...
