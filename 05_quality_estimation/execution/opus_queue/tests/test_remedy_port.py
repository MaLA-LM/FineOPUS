"""Regression tests for ReMedy torch distributed port selection.

Run with:
    python -m execution.opus_queue.tests.test_remedy_port
"""

from __future__ import annotations

from src.backends.remedy.backend import (
    PORT_RANGE_SIZE,
    PORT_RANGE_START,
    _resolve_master_port,
)


def test_valid_master_port_is_preserved() -> None:
    env = {
        "MASTER_PORT": "23456",
        "SLURM_ARRAY_JOB_ID": "123000",
        "OPUS_ARRAY_TASK_ID": "16",
        "OPUS_LOCAL_ID": "7",
    }

    assert _resolve_master_port(env, 0) == 23456
    assert _resolve_master_port(env, 2) == 23458


def test_fallback_port_uses_standard_g_local_id() -> None:
    base_env = {
        "SLURM_ARRAY_JOB_ID": "123000",
        "OPUS_ARRAY_TASK_ID": "16",
    }

    ports = [
        _resolve_master_port({**base_env, "OPUS_LOCAL_ID": str(local_id)}, 0)
        for local_id in range(8)
    ]

    assert len(set(ports)) == 8
    assert all(
        PORT_RANGE_START <= port < PORT_RANGE_START + PORT_RANGE_SIZE
        for port in ports
    )
    assert ports[1] - ports[0] == 101


def test_out_of_range_master_port_falls_back_per_local_id() -> None:
    base_env = {
        "MASTER_PORT": "40535",
        "SLURM_ARRAY_JOB_ID": "123000",
        "OPUS_ARRAY_TASK_ID": "16",
    }

    port_a = _resolve_master_port({**base_env, "OPUS_LOCAL_ID": "3"}, 0)
    port_b = _resolve_master_port({**base_env, "OPUS_LOCAL_ID": "4"}, 0)

    assert port_a != 40535
    assert port_a != port_b
    assert PORT_RANGE_START <= port_a < PORT_RANGE_START + PORT_RANGE_SIZE
    assert PORT_RANGE_START <= port_b < PORT_RANGE_START + PORT_RANGE_SIZE


def run_test() -> None:
    test_valid_master_port_is_preserved()
    test_fallback_port_uses_standard_g_local_id()
    test_out_of_range_master_port_falls_back_per_local_id()
    print("OK: ReMedy distributed port fallback is local-rank aware.")


if __name__ == "__main__":
    run_test()
