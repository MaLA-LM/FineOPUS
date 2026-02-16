"""Model-layer exports."""

__all__ = [
    "DimScores",
    "QEResult",
    "CometLanguages",
    "QwenLanguages",
    "MetricX24Languages",
    "ModelSpec",
    "ScoreAdjuster",
    "resolve_model_spec",
    "supported_model_keys",
]


def __getattr__(name: str):
    if name in {"DimScores", "QEResult"}:
        from models.llm_qe import DimScores, QEResult

        return {"DimScores": DimScores, "QEResult": QEResult}[name]
    if name in {"CometLanguages", "QwenLanguages", "MetricX24Languages"}:
        from models.language_support import CometLanguages, MetricX24Languages, QwenLanguages

        return {
            "CometLanguages": CometLanguages,
            "QwenLanguages": QwenLanguages,
            "MetricX24Languages": MetricX24Languages,
        }[name]
    if name in {
        "ModelSpec",
        "ScoreAdjuster",
        "resolve_model_spec",
        "supported_model_keys",
    }:
        from models.model_registry import (
            ModelSpec,
            ScoreAdjuster,
            resolve_model_spec,
            supported_model_keys,
        )

        return {
            "ModelSpec": ModelSpec,
            "ScoreAdjuster": ScoreAdjuster,
            "resolve_model_spec": resolve_model_spec,
            "supported_model_keys": supported_model_keys,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
