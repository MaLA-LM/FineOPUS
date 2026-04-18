"""Regression tests for queue-facing worker error messages.

Run with:
    python -m execution.opus_queue.tests.test_worker_error_messages
"""

from __future__ import annotations

import traceback

from execution.opus_queue.worker.loop import _format_failure_detail


def test_format_failure_detail_leads_with_summary_and_context() -> None:
    try:
        raise RuntimeError("CUDA out of memory while running MetricX")
    except RuntimeError as exc:
        detail = _format_failure_detail(
            exc,
            traceback.format_exc(),
            direction_key="eng_Latn-fra_Latn",
            shard_id=3,
            start_idx=10,
            end_idx=20,
            attempt=2,
            queue_model="metricx24",
            scorer_model="metricx24",
            backend="metricx",
        )

    assert detail.startswith("RuntimeError: CUDA out of memory while running MetricX")
    assert (
        "context: queue_model=metricx24 scorer_model=metricx24 backend=metricx "
        "direction=eng_Latn-fra_Latn shard=3 range=[10,20) attempt=2"
    ) in detail
    assert "traceback:" in detail
    assert "raise RuntimeError" in detail


def run_test() -> None:
    test_format_failure_detail_leads_with_summary_and_context()
    print("OK: worker DB errors start with the exception summary and shard context.")


if __name__ == "__main__":
    run_test()
