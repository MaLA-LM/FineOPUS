"""Checks for FT coverage semantics, output isolation and resume identity."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workflow as w
from evaluate_model import discover_planned_files


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = json.loads((w.HERE / 'language_policy.json').read_text())

    def test_source_label_corrections_and_unknowns(self):
        rows = {r['original_label']: r for r in w.languages(w.HERE / 'wikiann_langs_script_report.csv', self.policy)}
        self.assertEqual(rows['ja']['code'], 'jpn_Jpan')
        self.assertEqual(rows['simple']['code'], 'eng_Latn')
        self.assertEqual(rows['zh-classical']['code'], 'lzh_Hant')
        self.assertEqual(rows['zh-min-nan']['code'], 'nan_Latn')
        self.assertEqual(rows['zh-yue']['code'], 'yue_Hant')
        self.assertEqual(rows['sh']['code'], '')
        self.assertTrue(rows['zh']['uncertain'])

    def test_exact_match_precedes_alias_and_shared_targets_are_deduplicated(self):
        root = self.root / 'FLORES-200'
        for code in ('eng_Latn', 'ara_Arab', 'arb_Arab', 'als_Latn'):
            (root / code).mkdir(parents=True)
            (root / code / 'test.parquet').touch()
        rows = [{'code': c, 'original_label': c, 'raw_code': c, 'uncertain': False, 'normalization_note': ''}
                for c in ('ara_Arab', 'als_Latn', 'sqi_Latn', 'eng_Latn', '')]
        details, stats = w.coverage(self.root, ['FLORES-200'], rows, self.policy)
        self.assertEqual(details[0]['dataset_languages'], 'ara_Arab')
        self.assertEqual(details[2]['status'], 'mapped')
        self.assertEqual(details[3]['status'], 'english_anchor')
        self.assertEqual(details[4]['status'], 'unresolved')
        self.assertEqual(stats['FLORES-200']['covered_languages'], 3)
        self.assertEqual(stats['FLORES-200']['directions_per_model'], 4)

    def test_outputs_cannot_overlap_baseline_or_models(self):
        for path in (w.EVALUATION / 'results', w.EVALUATION / 'results' / 'ft',
                     w.EVALUATION, self.root / 'models' / 'model'):
            with self.assertRaises(ValueError):
                w.check_output_root(path, self.root / 'models')
        w.check_output_root(self.root / 'results', self.root / 'models')

    def test_missing_shard_is_rejected(self):
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
            (self.root / name).write_text('{}')
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'missing.safetensors'}}))
        with self.assertRaises(FileNotFoundError):
            w.validate_model(self.root)

    def test_resume_and_config_changes(self):
        data = self.root / 'data' / 'FLORES-200'
        for code in ('eng_Latn', 'deu_Latn'):
            (data / code).mkdir(parents=True)
            (data / code / 'test.parquet').touch()
        model = self.root / 'models' / 'FT-test'
        model.mkdir(parents=True)
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'):
            (model / name).write_text('{}')
        args = argparse.Namespace(
            command='plan', models_root=model.parent, data_root=data.parent, results_root=self.root / 'results',
            language_report=w.HERE / 'wikiann_langs_script_report.csv', language_policy=w.HERE / 'language_policy.json',
            datasets='FLORES-200', languages='deu_Latn', model_glob=[], few_shot=3, limit=None,
            comet_model='Unbabel/wmt22-comet-da', comet_batch_size=8,
            no_comet=False, enforce_eager=False, max_model_len=4096, max_new_tokens=256,
            force=False, overwrite=False, manifest_pointer=self.root / 'pointer')
        def plan():
            with contextlib.redirect_stdout(io.StringIO()):
                w.plan(args)
            return json.loads(Path(args.manifest_pointer.read_text().strip()).read_text())
        first = plan()
        output = Path(first['tasks'][0]['settings']['output_dir'])
        output.mkdir(parents=True)
        (output / '_SUCCESS').touch()
        self.assertEqual(plan()['tasks'], [])
        args.no_comet = True
        changed = plan()
        self.assertEqual(len(changed['tasks']), 1)
        self.assertNotEqual(first['results_root'], changed['results_root'])
        args.limit = 3
        smoke = plan()
        self.assertNotEqual(changed['results_root'], smoke['results_root'])

    def test_worker_forwards_existing_three_shot_evaluator_flags(self):
        command = w.eval_command({'few_shot': 3, 'limit': None, 'no_comet': False, 'enforce_eager': True})
        self.assertEqual(command[1], str(w.EVALUATION / 'mt_eval.py'))
        self.assertEqual(command[2:], ['--few-shot', '3', '--enforce-eager'])

    def test_model_runner_keeps_dataset_languages_separate(self):
        for dataset in ('FLORES-200', 'BOUQuET_Sentence'):
            for code in ('eng_Latn', 'deu_Latn', 'zho_Hant', 'cmn_Hant'):
                path = self.root / dataset / code
                path.mkdir(parents=True)
                (path / 'test.parquet').touch()
        plan = {'FLORES-200': ['deu_Latn', 'zho_Hant'], 'BOUQuET_Sentence': ['deu_Latn', 'cmn_Hant']}
        cases, skipped = discover_planned_files(self.root, list(w.mt_eval.DATASETS.values())[::2], plan)
        self.assertEqual([(c[0].name, c[2]) for c in cases], [
            ('FLORES-200', 'deu_Latn'), ('FLORES-200', 'zho_Hant'),
            ('BOUQuET_Sentence', 'deu_Latn'), ('BOUQuET_Sentence', 'cmn_Hant')])
        self.assertEqual(skipped, [])
        command = w.eval_command({'datasets': ','.join(plan)}, dataset_languages=plan)
        self.assertEqual(command[1], str(w.HERE / 'evaluate_model.py'))
        self.assertEqual(json.loads(command[3]), plan)

    def test_aggregation_waits_for_every_current_model_job(self):
        outputs = [self.root / 'a', self.root / 'b']
        completed = self.root / 'completed'
        completed.mkdir()
        for output in outputs:
            output.mkdir()
            (output / '_SUCCESS').touch()
        manifest = {'tasks': [{'task_id': i, 'settings': {'limit': None}} for i in range(2)],
                    'all_outputs': [str(p) for p in outputs], 'results_root': str(self.root),
                    'completion_root': str(completed)}
        with patch.object(w.subprocess, 'run') as run:
            (completed / '0').touch()
            w.aggregate_if_complete(manifest)
            run.assert_not_called()
            (completed / '1').touch()
            w.aggregate_if_complete(manifest)
            run.assert_called_once()
            w.aggregate_if_complete(manifest)
            run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
