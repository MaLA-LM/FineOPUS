#!/usr/bin/env python3
"""Run the shared evaluator once with the exact language plan for each dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
import mt_eval


def discover_planned_files(data_root, specs, language_plan):
    cases, skipped = [], []
    for spec in specs:
        root = data_root / spec.name
        english = root / 'eng_Latn'
        if not (english / f'{spec.test_split}.parquet').is_file():
            raise FileNotFoundError(f'{spec.name}: English test parquet is missing')
        # Codes have already been mapped by the coverage planner. Resolve them
        # exactly, without applying the from_scratch language aliases again.
        for code in language_plan[spec.name]:
            target = root / code
            if not (target / f'{spec.test_split}.parquet').is_file():
                raise FileNotFoundError(f'{spec.name}: planned language {code} is missing')
            cases.append((spec, code, code, english, target))
    return cases, skipped


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset-languages-json', required=True)
    own, remaining = parser.parse_known_args()
    plan = json.loads(own.dataset_languages_json)
    args = mt_eval.parse_eval_args(remaining)
    names = mt_eval.split_csv(args.datasets)
    if set(plan) != set(names):
        raise ValueError('Dataset language plan must match --datasets exactly')
    for name, codes in plan.items():
        if not isinstance(codes, list) or len(codes) != len(set(codes)):
            raise ValueError(f'{name}: expected a list of unique benchmark language codes')
    # This override is local to this FT process. The shared evaluator and its
    # from_scratch entry points remain unchanged.
    mt_eval.discover_files = lambda root, specs, _: discover_planned_files(root, specs, plan)
    return mt_eval.evaluate(args)


if __name__ == '__main__':
    raise SystemExit(main())
