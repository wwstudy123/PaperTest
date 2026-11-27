import unittest
import json
from pathlib import Path
import tempfile

from tools.plot_dynamic_matrix import service_metrics, collect, configuration_counts


class DynamicPlotTests(unittest.TestCase):
    def test_failed_configuration_is_attempted_not_successfully_measured(self):
        rows = [{"config": {"x": 1}, "status": "measured", "complete": True, "effective": True},
                {"config": {"x": 2}, "status": "execution_failed", "complete": False, "effective": False}]
        self.assertEqual(configuration_counts(rows), {
            "unique_measured_configurations": 1, "unique_attempted_configurations": 2})

    @staticmethod
    def requests():
        # 人工公式测试仅在内存中；不写入任何论文实验数据目录。
        return [{"seq": i, "scheduled": i / 10, "emitted": i / 10,
                 "completed": i / 10 + (.1 if i < 6000 else .02), "ok": True}
                for i in range(18000)]

    def test_whole_stream_quantile_is_not_mean_of_window_quantiles(self):
        result, windows, phases, completions = service_metrics(self.requests())
        self.assertAlmostEqual(result["overall_p95_ms"], 100)
        self.assertEqual(len(windows), 90)
        self.assertAlmostEqual(phases[1]["p95_ms"], 20)
        self.assertEqual(completions[-1]["successful_requests"], 18000)
        self.assertLess(completions[0]["successful_requests"], 1)

    def test_missing_or_duplicate_requests_are_not_filled(self):
        rs = self.requests()
        with self.assertRaises(ValueError):
            service_metrics(rs[:-1])
        rs[-1] = rs[0]
        with self.assertRaises(ValueError):
            service_metrics(rs)

    def test_failures_are_not_silently_excluded(self):
        rs = self.requests()
        rs[10]["ok"] = False
        with self.assertRaisesRegex(ValueError, "Failures"):
            service_metrics(rs)

    def test_incomplete_cohort_not_scored_as_complete(self):
        rs = self.requests()
        rs[0]["scheduled"] = 21
        with self.assertRaisesRegex(ValueError, "cohort"):
            service_metrics(rs)

    def test_static_or_smoke_online_tune_cannot_become_dynamic_curve(self):
        for manifest in [
            {"method": "OnlineTune", "smoke": True, "formal_protocol": False, "backend": "onlinetune"},
            {"method": "OnlineTune", "smoke": False, "formal_protocol": False, "backend": "llm"},
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                batch, extra = root / "batch", root / "extra"
                batch.mkdir(); extra.mkdir()
                (batch / "matrix.json").write_text('{"status":"completed"}')
                (extra / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "formal dynamic"):
                    collect(batch, 43, [extra])
