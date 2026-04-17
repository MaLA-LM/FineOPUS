from src.backends.comet.backend import load_comet_model, score_comet
from src.backends.comet.cli import main, parse_args, resolve_model
from src.backends.comet.runner import score_entry

__all__ = [
    "load_comet_model",
    "score_comet",
    "parse_args",
    "resolve_model",
    "score_entry",
    "main",
]


if __name__ == "__main__":
    main()
