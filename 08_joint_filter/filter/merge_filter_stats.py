#!/usr/bin/env python3
"""Merge only validated, completed pair results; incomplete runs fail by default."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import atomic_json, read_json, write_csv
from apply_thresholds import output_inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stats_output", help="Default <out_dir>/filter_stats.csv")
    p.add_argument("--allow_partial", action="store_true")
    args = p.parse_args()
    root = Path(args.out_dir)
    run = read_json(root / "run.json")
    results, missing = [], []
    corpora = defaultdict(lambda: {"rows_before": 0, "rows_after": 0, "kept_tokens": 0.0})
    languages = defaultdict(lambda: {"pair_incident_input_tokens": 0.0, "pair_incident_kept_tokens": 0.0})
    reasons = defaultdict(int)
    for pair in run["pairs"]:
        done = root / pair / "_DONE.json"
        if not done.exists():
            missing.append(pair)
            continue
        row = read_json(done)
        if row["output_run_id"] != run["output_run_id"] or row["output_inventory"] != output_inventory(done.parent):
            raise ValueError(f"Stale/incomplete output: {pair}")
        fields = ["lang_pair", "rows_before", "rows_eligible", "rows_after", "kept_fraction",
                  "rows_hard_rejected_or_missing", "rows_budget_rejected", "input_tokens",
                  "kept_tokens", "kept_tokens_kind", "budget_tokens"]
        results.append({k: row[k] for k in fields})
        for corpus, stats in row["corpora"].items():
            for k, v in stats.items():
                corpora[corpus][k] += v
        # Incident pair tokens include both sides, so do NOT sum this table to
        # recover corpus size. Language-specific source/target counts differ.
        for lang in set(pair.split("-")):
            languages[lang]["pair_incident_input_tokens"] += row["input_tokens"]
            languages[lang]["pair_incident_kept_tokens"] += row["kept_tokens"]
        for reason, count in row["reason_counts"].items():
            reasons[reason] += count
    if missing and not args.allow_partial:
        raise ValueError(f"{len(missing)} pairs incomplete; first: {missing[:5]}")
    if not results:
        raise ValueError("No completed pairs")
    dest = Path(args.stats_output) if args.stats_output else root / "filter_stats.csv"
    write_csv(dest, results)
    write_csv(dest.with_name(dest.stem + "_corpora.csv"), [{"corpus": k, **v} for k, v in sorted(corpora.items())])
    write_csv(dest.with_name(dest.stem + "_languages.csv"), [{"language": k, **v} for k, v in sorted(languages.items())])
    totals = {k: sum(r[k] for r in results) for k in ("rows_before", "rows_after", "input_tokens", "kept_tokens", "budget_tokens")}
    totals.update({"completed_pairs": len(results), "missing_pairs": missing,
                   "kept_tokens_kind": results[0]["kept_tokens_kind"], "reason_counts_nonexclusive": dict(reasons)})
    atomic_json(dest.with_suffix(".json"), totals)
    print(f"{len(results)} pairs; retained {totals['rows_after']:,} rows, {totals['kept_tokens']:,.0f} tokens ({totals['kept_tokens_kind']}); {len(missing)} incomplete")


if __name__ == "__main__":
    main()
