import unittest

from admtune.paper_pilot import curve_rows


def record(tps, p95, kind="candidate", **extra):
    return dict(value=tps, complete=True, effective=True, constraints_ok=True,
                status="measured", config={"knob": tps},
                metadata={"kind": kind, "p95_latency_ms": p95, "benchmark_log": f"{tps}.log"}, **extra)


class PaperPilotTests(unittest.TestCase):
    def test_p95_is_associated_with_tps_winner_not_independent_minimum(self):
        records = [record(100, 10, "baseline"), record(110, 12), record(90, 5)]
        rows = curve_rows("ADMTune", records, [], seed=0)
        self.assertEqual([r["best_tps"] for r in rows], [100, 110, 110])
        self.assertEqual([r["best_config_p95_ms"] for r in rows], [10, 12, 12])

    def test_verification_never_overwrites_search_best(self):
        rows = curve_rows("ADMTune", [record(100, 10), record(150, 2, "verification")], [], seed=0)
        self.assertEqual(rows[-1]["best_tps"], 100)
        self.assertEqual(rows[-1]["tps"], 150)

    def test_failure_counts_in_budget_but_does_not_improve_curve(self):
        bad = record(1000, 1)
        bad["constraints_ok"] = False
        rows = curve_rows("ADMTune", [record(100, 10), bad], [], seed=1)
        self.assertEqual(rows[-1]["trial"], 2)
        self.assertFalse(rows[-1]["valid"])
        self.assertEqual(rows[-1]["best_tps"], 100)

    def test_unmeasured_baselines_are_not_created(self):
        self.assertEqual(curve_rows("Default", [], [], seed=0), [])
        rows = curve_rows("Default", [record(100, 10, "default")],
                          [{"metadata": {"benchmark_log": "100.log", "ended_seconds": 85}}], seed=0)
        self.assertEqual(rows[0]["ended_seconds"], 85)
        self.assertEqual({r["method"] for r in rows}, {"Default"})
