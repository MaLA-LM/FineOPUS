from src.backends.metricx.backend import (
    apply_score_adjustment,
    load_metricx,
    score_metricx,
    select_device,
)
from src.backends.metricx.cli import main, parse_args, resolve_model
from src.backends.metricx.runner import score_entry

__all__ = [
    "apply_score_adjustment",
    "load_metricx",
    "score_metricx",
    "select_device",
    "parse_args",
    "resolve_model",
    "score_entry",
    "main",
]


if __name__ == "__main__":
    main()
