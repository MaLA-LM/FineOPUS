from __future__ import annotations

from dataset.mediator import Example

__all__ = ["load_comet_model", "score_comet"]


def load_comet_model(model_id: str):
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_id)
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
