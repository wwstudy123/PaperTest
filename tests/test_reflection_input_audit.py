import unittest

from admtune.contracts import Trial
from tools.replay_reflection_inputs import replay


class ReflectionInputAuditTests(unittest.TestCase):
    def test_same_records_and_triggers_without_real_llm(self):
        records = [Trial(str(i), phase, "env", "p95_latency_ms:minimize", {"x": 1}, [0],
                         "measured", 10, True, True, True).to_dict()
                   for i, phase in enumerate(["a", "a", "b", "b", "c"])]
        inc, full = replay(records, "incremental"), replay(records, "full")
        self.assertEqual([r["input_records"] for r in inc], [2, 2, 1])
        self.assertEqual([r["input_records"] for r in full], [2, 4, 5])
        self.assertEqual(len(inc), len(full))
        self.assertGreater(sum(r["serialized_input_bytes"] for r in full),
                           sum(r["serialized_input_bytes"] for r in inc))
