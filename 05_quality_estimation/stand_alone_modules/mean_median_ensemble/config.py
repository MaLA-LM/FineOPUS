"""
config.py
---------
Shared constants: model registry, partition count, and default path roots.
All paths are overridable via CLI arguments.
"""

from pathlib import Path

# ── Models: (hive partition name, safe SQL column name) ──
MODELS = [
    ("bicleaner-ai",             "bicleaner_ai"),
    ("m-prometheus-7b",          "m_prometheus_7b"),
    ("qwen3-4b-instruct-2507",   "qwen3_4b_instruct_2507"),
    ("shaomutan_remedy-9b-22",   "shaomutan_remedy_9b_22"),
    ("xcomet-xl",                "xcomet_xl"),
    ("metricx24",                "metricx24"),
    ("qwen3-14b",                "qwen3_14b"),
    ("qwen3-8b",                 "qwen3_8b"),
    ("wmt23-cometkiwi-da-xl",    "wmt23_cometkiwi_da_xl"),
]

NUM_BUCKETS = 3

# ── Default path roots (override via --src / --dst / --norm / --ens / --tmp) ──
_QE_BASE = Path("/scratch/project_462001050/QE_flores200_scores/dataset=flores200")

DEFAULT_NORM_ROOT   = _QE_BASE / "buckets=normalized_scores"
DEFAULT_ENS_ROOT    = _QE_BASE / "buckets=mean_median_ensembles"
DEFAULT_WAVG_ROOT   = _QE_BASE / "buckets=weighted_avg_ensembles"
DEFAULT_SCRATCH_TMP = _QE_BASE / "duckdb_tmp"
