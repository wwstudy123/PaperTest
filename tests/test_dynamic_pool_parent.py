import unittest
from unittest.mock import patch

from admtune.agents import Team
from admtune.dynamic import PooledTeam


class PoolParentTests(unittest.TestCase):
    def pool(self):
        team = PooledTeam.__new__(PooledTeam)
        team.pool = None
        team.pool_parent = None
        return team

    def test_same_parent_uses_same_batch_without_extra_llm(self):
        team = self.pool()
        with patch.object(Team, "recommend", autospec=True, return_value=["candidate"]) as original:
            self.assertEqual(team.recommend({"x": 1}, "p1", [0], []), ["candidate"])
            self.assertEqual(team.recommend({"x": 1}, "p1", [0], []), ["candidate"])
            self.assertEqual(original.call_count, 1)

    def test_accepted_new_parent_invalidates_old_absolute_candidates(self):
        team = self.pool()
        with patch.object(Team, "recommend", autospec=True, side_effect=[["old"], ["new"]]) as original:
            self.assertEqual(team.recommend({"buffer": 1}, "p1", [0], []), ["old"])
            self.assertEqual(team.recommend({"buffer": 2}, "p1", [0], []), ["new"])
            self.assertEqual(original.call_count, 2)
            self.assertEqual(original.call_args.args[1], {"buffer": 2})

    def test_phase_change_also_requires_fresh_batch(self):
        team = self.pool()
        with patch.object(Team, "recommend", autospec=True, side_effect=[["a"], ["b"]]) as original:
            team.recommend({"x": 1}, "p1", [0], [])
            self.assertEqual(team.recommend({"x": 1}, "p2", [1], []), ["b"])
            self.assertEqual(original.call_count, 2)

    def test_changed_effective_bounds_do_not_reuse_old_batch(self):
        team = self.pool()
        team.ranges = {"x": {"lower": 1, "upper": 2}}
        with patch.object(Team, "recommend", autospec=True, side_effect=[["a"], ["b"]]) as original:
            team.recommend({"x": 1}, "p1", [0], [])
            team.ranges = {"x": {"lower": 1, "upper": 4}}
            self.assertEqual(team.recommend({"x": 1}, "p1", [0], []), ["b"])
            self.assertEqual(original.call_count, 2)

    def test_recommender_receives_bounded_reflection_not_only_first_measurement(self):
        team = self.pool()
        experiences = [{"type": "measurement", "trial_id": "m1"},
                       {"type": "measurement", "trial_id": "m2"},
                       {"type": "reflection", "evidence_ids": ["real-id"],
                        "conditions": {"scope": "test"}, "hypotheses": ["x" * 500]}]
        with patch.object(Team, "recommend", autospec=True, return_value=["candidate"]) as original:
            team.recommend({"x": 1}, "p1", [0], experiences)
            sent = original.call_args.args[-1]
            self.assertEqual([x["type"] for x in sent], ["measurement", "reflection"])
            self.assertEqual(sent[1]["evidence_ids"], ["real-id"])
            self.assertEqual(len(sent[1]["hypothesis_previews"][0]["text"]), 160)
            self.assertTrue(sent[1]["hypothesis_previews"][0]["truncated"])
        self.assertEqual(experiences[-1]["hypotheses"], ["x" * 500])
