import csv
import json
from pathlib import Path
import tempfile
import unittest

from admtune.static_report import export_static


class StaticReportTests(unittest.TestCase):
    def records(self, root):
        trial = dict(trial_id='id-1', phase_id='phase-1', config={'knob': 4}, status='measured',
                     complete=True, effective=True, constraints_ok=True, value=100,
                     objective_key='tps:maximize', prediction={'mean': 4.6, 'latent_std': .1},
                     metadata={'kind': 'candidate', 'simulated': False, 'accepted': False, 'p95_latency_ms': 5})
        (root / 'trials.jsonl').write_text(json.dumps(trial) + '\n')
        (root / 'summary.json').write_text(json.dumps({'objective': 'tps:maximize',
                                                     'independent_verification_values': [101, 99, 100]}))
        return trial

    def test_export_preserves_objective_prediction_and_verification(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t);self.records(root)
            report = export_static(root)
            self.assertEqual(report['trials'], 1)
            self.assertEqual(report['verification_repeats'], 3)
            self.assertEqual(report['usage_records'], 0)
            with (root / 'static_trials.csv').open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row['config_knob'], '4')
            self.assertEqual(row['accepted'], 'False')
            self.assertEqual(row['prediction_log_mean'], '4.6')
            self.assertEqual(row['prediction_log_observation_std'], '')
            self.assertEqual((root / 'static_verification.csv').read_text().count('\n'), 4)

    def test_usage_rows_and_duplicates(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t);trial = self.records(root)
            (root / 'llm_usage.json').write_text(json.dumps({'records': [{'operation': 'selector', 'total_tokens': 10}]}))
            self.assertEqual(export_static(root)['usage_records'], 1)
            with (root / 'trials.jsonl').open('a') as f:
                f.write(json.dumps(trial) + '\n')
            with self.assertRaises(ValueError):
                export_static(root)
