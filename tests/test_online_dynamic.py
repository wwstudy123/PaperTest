import math
import json
from pathlib import Path
import tempfile
import unittest
from admtune.controller import Policy
from admtune.online_dynamic_calibration import evaluate, plan
from admtune.replay import ReplayPlan, make_schedule
from admtune.surrogate import GaussianProcess
from admtune.contracts import fingerprint
from tools.analyze_online_dynamic import analyze
from tools.analyze_dynamic_stage1 import adaptation
from test_calibration import space


class OnlineDynamicTests(unittest.TestCase):
    def test_six_probe_plan_changes_one_parameter_and_is_readonly(self):
        s = space(); specs = plan(s, s.defaults)
        self.assertEqual(len(specs), 6)
        for r in specs:
            changed = [k for k in r["config"] if r["config"][k] != s.defaults[k]]
            self.assertEqual(len(changed), int(r["kind"] == "candidate"))
            self.assertEqual(r["read_percent"], 100)

    def test_signal_gate_and_noise_do_not_depend_on_candidate_variation(self):
        records = [{"block": b, "valid": True, "value_p95_ms": y}
                   for b in ("scan", "sort") for y in (100, 75, 104)]
        result = evaluate(records)
        self.assertTrue(result["passed"])
        self.assertEqual(result["repeat_degrees_of_freedom"], 2)
        self.assertGreaterEqual(result["fixed_noise_log_std"], .02)
        records[1]["value_p95_ms"] = 50
        self.assertEqual(evaluate(records)["fixed_noise_log_std"], result["fixed_noise_log_std"])
        records[-1]["value_p95_ms"] = 150
        self.assertFalse(evaluate(records)["passed"])

    def test_noise_floor_is_training_covariance_not_only_plot_interval(self):
        floor = Policy(fixed_noise_log_std=.03).fixed_noise_log_std
        g = GaussianProcess()
        g.fit([[0.], [.5], [1.]], [1., 2., 1.], fixed_noise_log_std=floor)
        self.assertEqual(g.fixed_noise_log_std, floor)
        self.assertGreaterEqual(g.predict([[0.]])[0].observation_std, floor)
        with self.assertRaises(ValueError):
            Policy(fixed_noise_log_std=float("nan"))

    def test_common_seed_and_readonly_aba_schedule(self):
        kwargs = dict(phase_seconds=600, table_size=2000000, read_percent=100,
                      query_profile="range-mixed", range_rows=32768, top_k=8192)
        a = ReplayPlan(seed=42, **kwargs); b = ReplayPlan(seed=42+0, **kwargs)
        self.assertEqual(make_schedule(a), make_schedule(b))
        rs = make_schedule(a)
        self.assertEqual(len(rs), 18000)
        self.assertTrue(all(r.operation == "read" for r in rs))
        self.assertEqual(sum(r.top_k == 0 for r in rs[:6000]), 5400)
        self.assertEqual(sum(r.top_k == 0 for r in rs[6000:12000]), 600)

    def test_audit_matches_frozen_phase_composition(self):
        for pct in (10, 90):
            p = ReplayPlan(scenario="steady", phase_seconds=28, read_percent=100,
                query_profile="range-mixed", range_rows=32768, top_k=8192,
                table_size=2000000, steady_scan_percent=pct)
            rs = make_schedule(p)
            self.assertEqual(sum(r.top_k == 0 for r in rs), 840*pct//100)
        with self.assertRaises(ValueError):
            ReplayPlan(steady_scan_percent=90)

    def test_attainment_requires_three_complete_windows_and_none_not_zero(self):
        rows = [{"end_s": x, "p95_ms": y, "requests": 200}
                for x, y in [(20, 8), (40, 11), (60, 8), (80, 8), (100, 8)]]
        self.assertEqual(adaptation(rows, 0, 10), 100)
        rows[-1]["requests"] = 199
        self.assertIsNone(adaptation(rows, 0, 10))

    def test_analysis_includes_whole_stream_not_only_measurement_windows(self):
        # 人工数据只存在临时目录，用于公式/导出测试，绝不作为实验记录。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); batch = root/"fixture"; (batch/"calibration").mkdir(parents=True)
            cfg = space().defaults
            records = [{"block": b, "kind": "candidate" if i == 1 else "reference",
                        "name": f"{b}-{i}", "value_p95_ms": y, "valid": True,
                        "config": cfg, "process_metrics_full_stream": {"cpu_core_equivalents": 1}}
                       for b in ("scan", "sort") for i, y in enumerate((100, 80, 101))]
            (batch/"calibration/trials.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            (batch/"calibration/summary.json").write_text('{"completed":true}')
            for method, delay in (("Default", .02), ("ADMTune", .01)):
                d = batch/method; d.mkdir()
                (d/"manifest.json").write_text(json.dumps({"preflight": {"current_config": cfg}}))
                (d/"summary.json").write_text(json.dumps({"complete": True, "schedule_hash": "fixture",
                    "counts": {"succeeded": 18000}, "phase_ready": []}))
                with (d/"requests.jsonl").open("w") as f:
                    for i in range(18000):
                        f.write(json.dumps({"seq": i, "scheduled": i/10, "emitted": i/10,
                            "completed": i/10+delay, "ok": True, "config_generation": 0,
                            "session_config_hash": fingerprint(cfg)})+"\n")
                if method == "ADMTune":
                    (d/"trials.jsonl").write_text(json.dumps({"trial_id": "fixture",
                        "phase_id": "phase-1", "config": cfg, "value": 10, "prediction": None,
                        "status": "measured", "metadata": {"kind": "baseline"}})+"\n")
                    (d/"llm_usage.json").write_text('{"total_tokens":50,"model":"fixture"}')
            result = analyze(batch, root/"analysis")
            self.assertAlmostEqual(result["cumulative_response_time_reduction_pct"], 50, places=6)
            self.assertTrue((root/"analysis/dynamic_comparison.pdf").exists())


if __name__ == "__main__":
    unittest.main()
