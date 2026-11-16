#!/usr/bin/env python3
"""Plan isolated HF fine-tuning evaluations with one Slurm job per model."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).absolute().parent
EVALUATION = HERE.parent
sys.path.insert(0, str(EVALUATION))
import mt_eval

DEFAULT_MODELS = Path('/scratch/project_462001509/members/zihao/FineOPUS/ft_models')


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mt_eval.atomic_json(path, value)


def languages(report: Path, policy: dict) -> list[dict]:
    with report.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw = row['aligned_key'].partition('|')[2].strip()
        override = policy['label_overrides'].get(row['original_label'])
        code = override['code'] if override else raw.rstrip('?')
        row['raw_code'] = raw
        row['code'] = code if re.fullmatch(r'[a-z]{3}_[A-Z][a-z]{3}', code) else ''
        row['uncertain'] = raw.endswith('?')
        row['normalization_note'] = override['reason'] if override else (
            'Removed source uncertainty marker; retained reported language and script.'
            if row['uncertain'] and row['code'] else '')
    return rows


def coverage(data_root: Path, names: list[str], rows: list[dict], policy: dict) -> tuple[list[dict], dict]:
    details = []
    summary = {}
    for name in names:
        spec = mt_eval.DATASETS[name]
        root = data_root / name
        if not (root / 'eng_Latn' / f'{spec.test_split}.parquet').is_file():
            raise FileNotFoundError(f'{name}: missing English test parquet under {root}')
        entries = []
        for row in rows:
            code = row['code']
            targets = []
            if not code:
                status = 'unresolved'
            elif code == 'eng_Latn':
                status = 'english_anchor'
            else:
                # Exact benchmark codes always take precedence over aliases.
                candidates = [code] if (root / code / f'{spec.test_split}.parquet').is_file() else (
                    policy['dataset_aliases'].get(name, {}).get(code, []))
                targets = [c for c in candidates if (root / c / f'{spec.test_split}.parquet').is_file()]
                status = ('exact' if targets == [code] else 'mapped') if targets else 'unavailable'
            entry = {
                'dataset': name, 'original_label': row['original_label'],
                'reported_code': row['raw_code'], 'language': code,
                'status': status, 'dataset_languages': ','.join(targets),
                'source_uncertain': row['uncertain'], 'note': row['normalization_note'],
            }
            entries.append(entry)
        codes = {r['language'] for r in entries if r['language'] and r['status'] != 'english_anchor'}
        covered = {r['language'] for r in entries if r['status'] in ('exact', 'mapped')}
        exact = {r['language'] for r in entries if r['status'] == 'exact'}
        targets = sorted({c for r in entries for c in r['dataset_languages'].split(',') if c})
        summary[name] = {
            'source_rows': len(rows), 'english_rows': sum(r['status'] == 'english_anchor' for r in entries),
            'unresolved_rows': sum(r['status'] == 'unresolved' for r in entries),
            'target_languages': len(codes), 'covered_languages': len(covered),
            'exact_languages': len(exact), 'mapped_languages': len(covered - exact),
            'coverage_percent': round(100 * len(covered) / len(codes), 2) if codes else 0,
            'covered_source_rows': sum(r['status'] in ('exact', 'mapped') for r in entries),
            'benchmark_languages': targets, 'directions_per_model': 2 * len(targets),
            'unavailable_languages': sorted(codes - covered),
        }
        details.extend(entries)
    return details, summary


def validate_model(path: Path) -> dict:
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        if not (path / name).is_file():
            raise FileNotFoundError(f'Missing HF model file: {path / name}')
    index = path / 'model.safetensors.index.json'
    weights = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else ['model.safetensors']
    for weight in weights:
        if not (path / weight).is_file():
            raise FileNotFoundError(f'Missing HF weights: {path / weight}')
    return {name: [int((path / name).stat().st_size), (path / name).stat().st_mtime_ns]
            for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', *weights)}


def check_output_root(root: Path, models_root: Path) -> None:
    resolved = root.resolve()
    for protected in (EVALUATION / 'results', models_root):
        target = protected.resolve()
        if resolved == target or target in resolved.parents or resolved in target.parents:
            raise ValueError(f'FT results must be separate from {protected}: {root}')


def write_coverage(root: Path, details: list[dict], summary: dict, source: str) -> None:
    write_json(root / 'coverage.json', {'source': source, 'datasets': summary})
    with (root / 'coverage.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(details)


def plan(args: argparse.Namespace) -> int:
    names = mt_eval.split_csv(args.datasets)
    mt_eval.selected_specs(names)
    if args.few_shot < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError('few-shot must be nonnegative; limit must be positive')
    if min(args.comet_batch_size, args.max_model_len, args.max_new_tokens) < 1:
        raise ValueError('Batch size, context length and output length must be positive')
    output_root = args.results_root or HERE / 'results' / (
        f'{args.few_shot}shot' if args.limit is None else f'smoke_{args.few_shot}shot_limit{args.limit}')
    check_output_root(output_root, args.models_root)
    policy = json.loads(args.language_policy.read_text())
    rows = languages(args.language_report, policy)
    details, summary = coverage(args.data_root, names, rows, policy)
    selected = set(mt_eval.split_csv(args.languages)) if args.languages else None
    known = {row['code'] for row in rows}
    if selected and selected - known:
        raise ValueError(f'--languages must use normalized FT codes: {sorted(selected - known)}')
    source_hash = hashlib.sha256(args.language_report.read_bytes() + args.language_policy.read_bytes()).hexdigest()
    patterns = args.model_glob or ['FT-*']
    models = [p for p in sorted(args.models_root.iterdir()) if p.is_dir()
              and any(fnmatch.fnmatchcase(p.name, pattern) for pattern in patterns)]
    if not models:
        raise ValueError(f'No HF models match {patterns} in {args.models_root}')
    signatures = {str(model.absolute()): validate_model(model) for model in models}
    profile = {k: v for k, v in vars(args).items() if k not in (
        'command', 'manifest_pointer', 'results_root', 'force', 'overwrite')}
    profile['layout'] = 'one_job_per_model_v2'
    profile_id = hashlib.sha256(json.dumps(
        [profile, source_hash, signatures], sort_keys=True, default=str).encode()).hexdigest()[:16]
    # Settings changes and replaced checkpoints get separate aggregate roots.
    output_root = output_root / profile_id
    root = output_root / 'task_manifests' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    write_coverage(root, details, summary, policy['source_url'])
    tasks = []
    all_outputs = []
    complete = 0
    dataset_languages = {
        name: sorted({c for row in details if row['dataset'] == name
                      and (selected is None or row['language'] in selected)
                      for c in row['dataset_languages'].split(',') if c})
        for name in names
    }
    if not any(dataset_languages.values()):
        raise ValueError('No selected FT language is covered by the requested datasets')
    for model in models:
        signature = signatures[str(model.absolute())]
        settings = {
            'model': str(model.absolute()), 'model_name': model.name, 'checkpoint_name': 'final',
            'data_root': str(args.data_root.absolute()), 'datasets': ','.join(names),
            'languages': ','.join(sorted({c for codes in dataset_languages.values() for c in codes})),
            'few_shot': args.few_shot, 'limit': args.limit, 'no_comet': args.no_comet,
            'comet_model': args.comet_model, 'comet_batch_size': args.comet_batch_size,
            'enforce_eager': args.enforce_eager, 'max_model_len': args.max_model_len,
            'max_new_tokens': args.max_new_tokens,
        }
        fingerprint = hashlib.sha256(json.dumps(
            [settings, dataset_languages, source_hash, signature], sort_keys=True).encode()).hexdigest()
        output = output_root / model.name / 'final'
        all_outputs.append(str(output.absolute()))
        marker = '_SUCCESS' if args.limit is None else f'_SMOKE_SUCCESS_limit_{args.limit}'
        if (output / marker).is_file() and not (args.force or args.overwrite):
            complete += 1
            continue
        settings['output_dir'] = str(output.absolute())
        tasks.append({'task_id': len(tasks), 'settings': settings, 'fingerprint': fingerprint,
                      'model_signature': signature, 'overwrite': args.overwrite,
                      'dataset_languages': dataset_languages})
    manifest = root / 'tasks.json'
    write_json(manifest, {'tasks': tasks, 'source': policy['source_url'], 'source_sha256': source_hash,
                         'coverage': summary, 'results_root': str(output_root.absolute()),
                         'all_outputs': all_outputs, 'completion_root': str(root / 'completed')})
    args.manifest_pointer.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_pointer.write_text(str(manifest.absolute()) + '\n')
    print('Dataset                  covered/targets  exact  mapped  benchmark  directions/model')
    for name, stats in summary.items():
        print(f"{name:25} {stats['covered_languages']:3}/{stats['target_languages']:<8} "
              f"{stats['exact_languages']:5} {stats['mapped_languages']:7} "
              f"{len(stats['benchmark_languages']):10} {stats['directions_per_model']:17}")
    print(f'Models: {len(models)}; tasks: {len(tasks)}; already complete: {complete}')
    print(f'Manifest: {manifest}')
    print(f'Coverage: {root / "coverage.tsv"}')
    print(f'Results root: {output_root}')
    return 0


def eval_command(settings: dict, *, dataset_languages: dict | None = None,
                 preflight: bool = False, overwrite: bool = False) -> list[str]:
    command = [sys.executable, str(HERE / 'evaluate_model.py') if dataset_languages is not None
               else str(EVALUATION / 'mt_eval.py')]
    if dataset_languages is not None:
        command.extend(['--dataset-languages-json', json.dumps(dataset_languages)])
    for key, value in settings.items():
        if value is None or value is False:
            continue
        command.append('--' + key.replace('_', '-'))
        if value is not True:
            command.append(str(value))
    if preflight:
        command.extend(['--preflight-only', '--check-runtime'])
    if overwrite:
        command.append('--overwrite')
    return command


def aggregate_if_complete(manifest: dict) -> None:
    """The last successful model job aggregates, without an extra Slurm job."""
    if not manifest['tasks'] or manifest['tasks'][0]['settings']['limit'] is not None:
        return
    outputs = [Path(p) for p in manifest['all_outputs']]
    root = Path(manifest['results_root'])
    completion = Path(manifest['completion_root'])
    with (root / '.aggregate.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if (completion / '_AGGREGATED').is_file():
            return
        if not all((completion / str(task['task_id'])).is_file() for task in manifest['tasks']):
            return
        if not all((path / '_SUCCESS').is_file() for path in outputs):
            return
        subprocess.run([sys.executable, str(EVALUATION / 'aggregate_mt_results.py'),
                        '--results-root', str(root)], check=True)
        (completion / '_AGGREGATED').touch()


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    if args.command == 'preflight':
        if not manifest['tasks']:
            return 0
        task = manifest['tasks'][0]
        subprocess.run(eval_command(task['settings'], dataset_languages=task['dataset_languages'], preflight=True), check=True)
        return 0
    task_id = args.task_id if args.task_id is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if task_id < 0 or task_id >= len(manifest['tasks']):
        raise ValueError(f'Task ID out of range: {task_id}')
    task = manifest['tasks'][task_id]
    settings = task['settings']
    if validate_model(Path(settings['model'])) != task['model_signature']:
        raise RuntimeError(f'Model files changed since planning: {settings["model"]}; create a new manifest')
    output = Path(settings['output_dir'])
    check_output_root(output, Path(settings['model']).parent)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'task.json', task)
    print(f'Task {task_id}: {settings["model_name"]} / {settings["datasets"]} / {settings["languages"]}', flush=True)
    subprocess.run(eval_command(settings, dataset_languages=task['dataset_languages'], overwrite=task['overwrite']), check=True)
    completion = Path(manifest['completion_root'])
    completion.mkdir(parents=True, exist_ok=True)
    (completion / str(task_id)).touch()
    aggregate_if_complete(manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('plan')
    p.add_argument('--models-root', type=Path, default=DEFAULT_MODELS)
    p.add_argument('--data-root', type=Path, default=EVALUATION / 'data')
    p.add_argument('--results-root', type=Path)
    p.add_argument('--language-report', type=Path, default=HERE / 'wikiann_langs_script_report.csv')
    p.add_argument('--language-policy', type=Path, default=HERE / 'language_policy.json')
    p.add_argument('--manifest-pointer', type=Path, required=True)
    p.add_argument('--datasets', default=','.join(mt_eval.DATASETS))
    p.add_argument('--languages', help='Optional subset of normalized FT codes (for smoke tests)')
    p.add_argument('--model-glob', action='append', default=[])
    p.add_argument('--few-shot', type=int, default=3)
    p.add_argument('--limit', type=int)
    p.add_argument('--comet-model', default=mt_eval.DEFAULT_COMET_MODEL)
    p.add_argument('--comet-batch-size', type=int, default=8)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=256)
    for flag in ('force', 'overwrite', 'no-comet', 'enforce-eager'):
        p.add_argument('--' + flag, action='store_true')
    for name in ('worker', 'preflight'):
        p = subs.add_parser(name)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--task-id', type=int)
    p = subs.add_parser('coverage', help='Regenerate coverage reports without loading models or GPU packages')
    p.add_argument('--data-root', type=Path, default=EVALUATION / 'data')
    p.add_argument('--datasets', default=','.join(mt_eval.DATASETS))
    p.add_argument('--language-report', type=Path, default=HERE / 'wikiann_langs_script_report.csv')
    p.add_argument('--language-policy', type=Path, default=HERE / 'language_policy.json')
    p.add_argument('--output-dir', type=Path, default=HERE / 'reports')
    args = parser.parse_args()
    if args.command == 'coverage':
        policy = json.loads(args.language_policy.read_text())
        names = mt_eval.split_csv(args.datasets)
        mt_eval.selected_specs(names)
        details, summary = coverage(args.data_root, names, languages(args.language_report, policy), policy)
        write_coverage(args.output_dir, details, summary, policy['source_url'])
        print(args.output_dir / 'coverage.tsv')
        return 0
    return plan(args) if args.command == 'plan' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
