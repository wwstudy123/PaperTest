import unittest
from tools.run_priority_dynamic import DISPLAY, JOBS, enough_time


class PriorityDynamicTests(unittest.TestCase):
    def test_same_budget_reserves_reporting_time(self):
        self.assertTrue(enough_time(2700, 0))
        self.assertFalse(enough_time(2699, 0))

    def test_four_methods_precede_four_ablations(self):
        self.assertEqual([m for m, _ in JOBS[:4]],
                         ["Default", "AgenticDB", "AgentTune", "ADMTune"])
        self.assertEqual([v for _, v in JOBS[4:]],
                         ["no_history", "no_gp", "single_slice", "no_reflection"])
        self.assertTrue(all(m == "ADMTune" for m, _ in JOBS[4:]))

    def test_single_slice_is_not_labelled_as_removing_all_monitoring(self):
        self.assertEqual(DISPLAY["ADMTune-single_slice"], "Single time slice")
        self.assertEqual(DISPLAY["ADMTune-no_gp"], "w/o GP prefilter")


if __name__ == "__main__":
    unittest.main()
