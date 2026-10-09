"""Seed matrix, actual child-process concurrency and failure containment."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from pedestrian_behavior.experiments.e001_sweep import attempts, attempt_command, run_sweep
from pedestrian_behavior.experiments.sweep import run_sweep as execute_sweep


CHILD = """
import json, os, sys, time
from pathlib import Path
path = Path(sys.argv[1]); path.mkdir()
start = time.monotonic()
time.sleep(.3)
(path/'timing.json').write_text(json.dumps({'start': start, 'end': time.monotonic(),
    'pid': os.getpid(), 'threads': [os.environ[k] for k in
    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')]}))
print('child output', flush=True)
sys.exit(int(sys.argv[2]))
"""

CUDA_CHILD = """
import hashlib, json, sys
from pathlib import Path
import torch
from pedestrian_behavior.experiments.e001 import build_model
from pedestrian_behavior.training import seed_run
seed_run(int(sys.argv[2]))
path = Path(sys.argv[1]); path.mkdir()
model = build_model('K+T+R', 'B').cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
inputs = torch.randn(64, 128, 12).cuda()
lengths = torch.full((64,), 128, dtype=torch.int64)
for _ in range(2):
    optimizer.zero_grad()
    model(inputs, lengths).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
digest = hashlib.sha256()
for value in model.state_dict().values(): digest.update(value.detach().cpu().numpy().tobytes())
(path/'gpu.json').write_text(json.dumps({'state_sha256': digest.hexdigest(),
    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}))
"""


class E001SweepTest(unittest.TestCase):
    def test_matrix_and_explicit_run_settings(self):
        plan = attempts(range(5))
        self.assertEqual(len(plan), 80)
        self.assertEqual(len({r['name'] for r in plan}), 80)
        self.assertTrue(all(sum(r['seed'] == seed for r in plan) == 16 for seed in range(5)))
        command = attempt_command(plan[-1], Path('/tmp/attempts'), 'cuda')
        options = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(options['--seed'], '4')
        self.assertEqual((options['--epochs'], options['--patience']), ('75', '8'))
        self.assertEqual(options['--output'], '/tmp/attempts/loki-RAW-BiLSTM-seed4')

    def test_real_processes_obey_concurrency_and_preserve_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for jobs in (1, 2):
                output = Path(tmp)/str(jobs)
                plan = attempts([0])[:4]
                def command(row, destination, device):
                    return [sys.executable, '-c', CHILD, str(destination/row['name']), '0']
                with patch('pedestrian_behavior.experiments.e001_sweep.attempts', return_value=plan), \
                     patch('pedestrian_behavior.experiments.e001_sweep.attempt_command', side_effect=command):
                    state = run_sweep(output, jobs=jobs, seeds=[0], device='cpu')
                self.assertEqual(state['status'], 'complete')
                timings = [json.loads((output/r['name']/'timing.json').read_text()) for r in plan]
                events = sorted([(t['start'], 1) for t in timings]+[(t['end'], -1) for t in timings])
                active = peak = 0
                for _, delta in events:
                    active += delta; peak = max(peak, active)
                self.assertEqual(peak, jobs)
                self.assertEqual(len({t['pid'] for t in timings}), 4)
                self.assertTrue(all(t['threads'] == ['2']*3 for t in timings))
                self.assertTrue(all('child output' in (output/'logs'/(r['name']+'.log')).read_text() for r in plan))
                with self.assertRaises(FileExistsError):
                    run_sweep(output)

    def test_failure_stops_queue_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'failed'
            plan = attempts([0])[:4]
            def command(row, destination, device):
                return [sys.executable, '-c', CHILD, str(destination/row['name']), '7']
            with patch('pedestrian_behavior.experiments.e001_sweep.attempts', return_value=plan), \
                 patch('pedestrian_behavior.experiments.e001_sweep.attempt_command', side_effect=command):
                with self.assertRaisesRegex(RuntimeError, 'Attempt failed'):
                    run_sweep(output, jobs=2, seeds=[0], device='cpu')
            state = json.loads((output/'sweep.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertTrue(state['failures'])
            self.assertEqual([r['status'] for r in state['attempts'][2:]], ['not-started']*2)
            self.assertEqual(len(list((output/'logs').glob('*.log'))), 2)

    def test_invalid_settings_create_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'invalid'
            for jobs, seeds in ((0, [0]), (2, []), (2, [0, 0]), (2, [-1]), (2, [2**32])):
                with self.assertRaises(ValueError):
                    run_sweep(output, jobs=jobs, seeds=seeds)
                self.assertFalse(output.exists())

    def test_shared_scheduler_rejects_duplicate_or_escaping_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'invalid-plan'
            for names in ([], ['a', 'a'], [''], ['..'], ['a/b']):
                with self.assertRaises(ValueError):
                    execute_sweep(output, [{'name': name} for name in names], lambda *args: [])
                self.assertFalse(output.exists())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_parallel_cuda_matches_serial_seeded_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcomes = []
            for jobs in (1, 2):
                output = Path(tmp)/str(jobs)
                plan = [attempts([seed])[1] for seed in (0, 1)]
                def command(row, destination, device):
                    return [sys.executable, '-c', CUDA_CHILD, str(destination/row['name']), str(row['seed'])]
                with patch('pedestrian_behavior.experiments.e001_sweep.attempts', return_value=plan), \
                     patch('pedestrian_behavior.experiments.e001_sweep.attempt_command', side_effect=command):
                    run_sweep(output, jobs=jobs, seeds=[0, 1])
                outcomes.append([json.loads((output/r['name']/'gpu.json').read_text())['state_sha256'] for r in plan])
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertNotEqual(outcomes[0][0], outcomes[0][1])
