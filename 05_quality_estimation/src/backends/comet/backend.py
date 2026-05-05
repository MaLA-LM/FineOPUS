from __future__ import annotations

from contextlib import contextmanager

from dataset.mediator import Example

__all__ = ["load_comet_model", "score_comet"]


class _NonMistralModelInfo:
    config = {"model_type": "xlm-roberta"}
    tags: list[str] = []
    siblings: list[object] = []

    def __getattr__(self, name: str):
        return None


@contextmanager
def _skip_xlmr_mistral_hub_probe():
    """Avoid a Transformers Hub metadata probe for known non-Mistral encoders."""
    patched = []

    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None and hasattr(huggingface_hub, "model_info"):
        patched.append((huggingface_hub, "model_info", huggingface_hub.model_info))

    try:
        import transformers.tokenization_utils_base as tokenization_utils_base
    except ImportError:
        tokenization_utils_base = None
    if tokenization_utils_base is not None and hasattr(
        tokenization_utils_base, "model_info"
    ):
        patched.append(
            (
                tokenization_utils_base,
                "model_info",
                tokenization_utils_base.model_info,
            )
        )

    def make_model_info(original):
        def model_info(model_id, *args, **kwargs):
            normalized = str(model_id).lower()
            if "xlm-roberta" in normalized or "xlmr" in normalized:
                return _NonMistralModelInfo()
            return original(model_id, *args, **kwargs)

        return model_info

    for module, name, original in patched:
        setattr(module, name, make_model_info(original))
    try:
        yield
    finally:
        for module, name, original in patched:
            setattr(module, name, original)


def load_comet_model(model_id: str):
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_id)
    with _skip_xlmr_mistral_hub_probe():
        return load_from_checkpoint(model_path)


def score_comet(
    examples: list[Example],
    model,
    batch_size: int,
    gpus: int,
) -> list[float]:
    samples = [{"src": ex["src"], "mt": ex["tgt"]} for ex in examples]
    prediction = model.predict(samples, batch_size=batch_size, gpus=gpus)
    return [float(score) for score in prediction["scores"]]
