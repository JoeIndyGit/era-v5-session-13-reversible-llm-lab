"""Real miniature runs and fault injection; these are never submission benchmarks."""
from pathlib import Path
import copy
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.model import TinyGPT, ModelConfig
from src.train import run_experiment, _train_step, _search_max_stable_batch
from src.diagnostics import gradient_check
from scripts.audit_results import check_run, check_probe
from scripts.run_full_submission import execute_notebook


def configuration(root, output):
    return {'seed': 17, 'target_tokens': 201, 'batch_size': 3, 'eval_batch_size': 2, 'eval_batches': 2,
            'eval_every_tokens': 96, 'speed_warmup_steps': 1, 'max_lr': 0.0003, 'min_lr': 0.00003,
            'warmup_tokens': 48, 'betas': [0.9, 0.95], 'weight_decay': 0.1, 'grad_clip': 1.0,
            'train_path': str(root/'train.bin'), 'val_path': str(root/'val.bin'),
            'results_dir': str(root/output/'results'), 'checkpoint_dir': str(root/output/'checkpoints'),
            'checkpoint_every_steps': 2, 'log_every_steps': 100, 'allow_cpu': True, 'precision': 'auto'}


def small_model(variant=None):
    return ModelConfig(vocab_size=64, seq_len=16, d_model=16, n_heads=4, n_layers=4, d_ff=32,
                       reversible=variant is not None, reversible_variant=variant or 'midpoint')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        for name in ['train', 'val']: rng.integers(0, 64, 4096, dtype=np.uint16).tofile(self.root/f'{name}.bin')
    def tearDown(self): self.temp.cleanup()

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor): self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a: self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.assert_nested_equal(x, y)
        else: self.assertEqual(a, b)

    def test_resume_is_identical_for_all_architectures(self):
        for variant in [None, 'midpoint', 'leapfrog']:
            with self.subTest(variant=variant):
                full_cfg = configuration(self.root, f'{variant}-full')
                resume_cfg = configuration(self.root, f'{variant}-resume')
                full = run_experiment(full_cfg, small_model(variant), 'run')
                paused = run_experiment(resume_cfg, small_model(variant), 'run', stop_after_steps=2)
                self.assertEqual(paused['status'], 'interrupted_for_test')
                self.assertFalse((Path(resume_cfg['results_dir'])/'run.json').exists())
                # Simulate an uncommitted CSV tail after a crash.
                with (Path(resume_cfg['results_dir'])/'run_steps.csv').open('a') as f: f.write('invalid tail after checkpoint\n')
                resumed = run_experiment(resume_cfg, small_model(variant), 'run')
                self.assertEqual(resumed['tokens_seen'], 201)
                self.assertEqual(resumed['optimizer_steps'], 5)
                self.assertEqual(resumed['final_val_loss'], full['final_val_loss'])
                self.assertEqual(resumed['final_train_loss_100step_mean'], full['final_train_loss_100step_mean'])
                a = torch.load(full['checkpoint_file'], weights_only=True)
                b = torch.load(resumed['checkpoint_file'], weights_only=True)
                for key in ['model', 'optimizer', 'rng']: self.assert_nested_equal(a[key], b[key])
                self.assertEqual(check_run(resumed, Path(resume_cfg['results_dir'])/'run_steps.csv', 201), [])

    def test_changed_protocol_and_data_cannot_resume(self):
        cfg = configuration(self.root, 'guard')
        run_experiment(cfg, small_model(), 'run', stop_after_steps=2)
        changed = dict(cfg, max_lr=0.003)
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint settings'): run_experiment(changed, small_model(), 'run')
        with (self.root/'train.bin').open('ab') as f: f.write(b'\x01\x00')
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint settings'): run_experiment(cfg, small_model(), 'run')

    def test_metrics_audit_rejects_tampering_and_missing_logs(self):
        cfg = configuration(self.root, 'audit')
        result = run_experiment(cfg, small_model(), 'run'); path = Path(cfg['results_dir'])/'run_steps.csv'
        self.assertEqual(check_run(result, path, 201), [])
        corrupt = copy.deepcopy(result); corrupt['median_tokens_per_s'] *= 2
        self.assertTrue(any('median_tokens_per_s' in e for e in check_run(corrupt, path, 201)))
        with path.open() as stream: rows = list(csv.DictReader(stream))
        rows[-1]['tokens_seen'] = '200'
        with path.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        self.assertTrue(any('cumulative-token' in e for e in check_run(result, path, 201)))
        path.unlink(); self.assertIn('step CSV is missing', check_run(result, path, 201))

    def test_full_model_gradient_diagnostic(self):
        for variant in ['midpoint', 'leapfrog']:
            torch.manual_seed(3)
            model = TinyGPT(small_model(variant))
            x = torch.randint(0, 64, (2, 16)); y=torch.randint(0, 64, (2, 16))
            report = gradient_check(model, x, y, torch.float32)
            self.assertTrue(report['passed'], report)

    def test_overflow_replays_batch_before_counting_an_update(self):
        class SkipOnceScaler:
            value = 2.0; skipped = False
            def get_scale(self): return self.value
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def step(self, optimizer):
                if self.skipped: optimizer.step()
            def update(self):
                if not self.skipped: self.value=1.0; self.skipped=True
        torch.manual_seed(6); model=TinyGPT(small_model()); reference=copy.deepcopy(model)
        optimizer=torch.optim.AdamW(model.parameters()); refopt=torch.optim.AdamW(reference.parameters())
        x=torch.randint(0,64,(2,16)); y=torch.randint(0,64,(2,16))
        _, _, retries = _train_step(model, optimizer, SkipOnceScaler(), x, y, torch.float32, 1.0)
        _train_step(reference, refopt, None, x, y, torch.float32, 1.0)
        self.assertEqual(retries, 1)
        self.assert_nested_equal(model.state_dict(), reference.state_dict())
        self.assertTrue(all(int(s['step']) == 1 for s in optimizer.state.values()))


class CapacityTests(unittest.TestCase):
    def test_search_finds_adjacent_memory_boundary(self):
        def probe(cfg, model, batch, steps, limit):
            return {'batch_size':batch, 'stable':batch<=37, 'error':None if batch<=37 else 'CUDA OOM',
                    'successful_updates':steps if batch<=37 else 0}
        with patch('torch.cuda.is_available', return_value=True), patch('src.train._probe_one_batch', side_effect=probe):
            result=_search_max_stable_batch({'batch_size':16}, None)
        self.assertEqual(result['largest_stable_batch'],37)
        self.assertEqual(result['first_failed_batch'],38)
        self.assertEqual(check_probe(result), [])
        result['attempts']=[r for r in result['attempts'] if r['batch_size']!=38]
        self.assertTrue(check_probe(result))

    def test_cap_is_only_a_lower_bound(self):
        with patch('torch.cuda.is_available', return_value=True), patch('src.train._probe_one_batch', side_effect=lambda c,m,b,s,r: {'batch_size':b,'stable':True,'error':None,'successful_updates':s}):
            result=_search_max_stable_batch({'batch_size':4},None,max_batch_cap=16)
        self.assertFalse(result['search_complete'])
        self.assertTrue(check_probe(result))


class NotebookTests(unittest.TestCase):
    def test_failed_cell_stops_execution_and_preserves_error(self):
        import nbformat
        from nbclient.exceptions import CellExecutionError
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'input.ipynb'; output=root/'executed.ipynb'
            nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print("first cell executed")'),
                nbformat.v4.new_code_cell('raise RuntimeError("audit rejected evidence")'),
                nbformat.v4.new_code_cell('raise AssertionError("must never run")')])
            nbformat.write(nb,source)
            with self.assertRaises(CellExecutionError): execute_notebook(source,output,root)
            saved=nbformat.read(output,as_version=4)
            self.assertEqual(saved.metadata.execution_record.status,'failed')
            self.assertIsNone(saved.cells[2].execution_count)
            self.assertTrue(any(o.output_type=='error' for o in saved.cells[1].outputs))


if __name__ == '__main__': unittest.main()
