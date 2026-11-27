import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from admtune.onlinetune_adapter import LinearCatalog
from admtune.onlinetune_dynamic import observed_sql_context, ContextRouter, DynamicOnlineTuneStrategy
from test_calibration import space


def window(sort_count=4, count=40, start=0):
    return [{"scheduled": start + i * 4 / count, "operation": "read", "range_rows": 32768,
             "top_k": 8192 if i < sort_count else 0, "phase": "DO_NOT_USE"}
            for i in range(count)]


def features(sort_count):
    return observed_sql_context(window(sort_count), 0, 4, table_size=2000000)


class ObservedContextTests(unittest.TestCase):
    def test_dynamic_features_change_with_actual_sql_composition(self):
        a, b = np.array(features(4)), np.array(features(36))
        self.assertEqual(len(a), 10)
        self.assertGreater(np.linalg.norm(a - b), .25)
        self.assertTrue(np.all((a >= 0) & (a <= 1)))

    def test_phase_labels_configuration_and_latency_cannot_leak(self):
        offers = window()
        original = observed_sql_context(offers, 0, 4, table_size=2000000)
        for r in offers:
            r.update(phase="FUTURE_ANSWER", latency_ms=12345, config={"pretend": 1})
        self.assertEqual(original, observed_sql_context(offers, 0, 4, table_size=2000000))

    def test_future_requests_are_refused(self):
        offers = window()
        offers[-1]["scheduled"] = 4
        with self.assertRaisesRegex(ValueError, "future"):
            observed_sql_context(offers, 0, 4, table_size=2000000)

    def test_intensity_is_observed_not_constant(self):
        a = observed_sql_context(window(4), 0, 4, table_size=2000000)
        b = observed_sql_context(window(12, 120), 0, 4, table_size=2000000)
        self.assertAlmostEqual(a[3], b[3])
        self.assertGreater(b[-1], a[-1])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional sklearn dynamic adapter dependency")
class RouterTests(unittest.TestCase):
    def test_aba_routes_to_same_region_without_phase_names(self):
        router = ContextRouter(lambda *a: None)
        a, b = features(4), features(36)
        router.fit([a] * 8)
        first = router.route(a)
        router.fit([a] * 8 + [b] * 8)
        second = router.route(b)
        returned = router.route(a)
        self.assertNotEqual(first, second)
        self.assertEqual(first, returned)
        self.assertGreaterEqual(router.svm_fits, 1)

    def test_novel_short_context_is_not_forced_into_old_model(self):
        router = ContextRouter(lambda *a: None)
        a, b = features(4), features(36)
        router.fit([a] * 8)
        old = router.route(a)
        router.fit([a] * 8 + [b] * 3)
        provisional = router.route(b)
        self.assertNotEqual(old, provisional)
        router.fit([a] * 8 + [b] * 8)
        self.assertEqual(provisional, router.route(b))
        self.assertEqual(old, router.route(a))

    def test_return_reuses_model_and_rebases_only_real_past_facts(self):
        strategy = DynamicOnlineTuneStrategy.__new__(DynamicOnlineTuneStrategy)
        s = space()
        strategy.codec = LinearCatalog(s)
        strategy.event = Mock()
        strategy.router = ContextRouter(strategy.event)
        a, b = features(4), features(36)
        strategy.router.fit([a] * 8 + [b] * 8)
        region = strategy.router.route(a)
        model = SimpleNamespace(model=Mock(), s=[Mock()])
        strategy.bank = {region: model}
        strategy.anchors = {}
        strategy.facts = {region: {"old": {"trial_id": "old", "config": s.defaults,
                                        "value": .125, "context": a, "phase_id": "prior"}}}
        strategy.persist = Mock()
        record = {"trial_id": "new", "value": .2, "config": s.defaults, "phase_id": "now"}
        strategy.start_phase(record, a, [a] * 8 + [b] * 8 + [a] * 8)
        self.assertIs(strategy.algorithm, model)
        x, y = model.model.train_gp.call_args.args
        self.assertEqual(len(x), 2)
        np.testing.assert_allclose(y, [-.075, 0])
        self.assertEqual(set(strategy.facts[region]), {"old", "new"})
        activated = [v for (k, v), _ in strategy.event.call_args_list
                     if k == "onlinetune_region_activated"]
        self.assertTrue(activated[0]["reused_model"])

    def test_cross_region_transfer_does_not_fabricate_other_region_safety(self):
        strategy = DynamicOnlineTuneStrategy.__new__(DynamicOnlineTuneStrategy)
        strategy.codec = LinearCatalog(space())
        strategy.event = Mock()
        strategy.region, strategy.baseline = 1, .1
        strategy.anchors = {0: .08, 1: .1}
        target = SimpleNamespace(model=Mock(), s=[Mock()])
        strategy.bank = {0: target, 1: SimpleNamespace()}
        baseline = {"trial_id": "baseline", "config": space().defaults, "context": features(36),
                    "value": .1, "anchor_at_observation": .1, "origin_region": 1}
        better = {**baseline, "trial_id": "better", "value": .13}
        strategy.facts = {0: {}, 1: {"baseline": baseline, "better": better}}
        strategy.share_good_observation({"trial_id": "better"})
        self.assertIn("better", strategy.facts[0])
        target.model.add_data.assert_called_once()
        target.s[0].add_data.assert_not_called()
