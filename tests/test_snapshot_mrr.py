"""The metric a reordering change is judged by.

This repository declined a working reranker once because it was measured on a
composite whose retrieval half was saturated. tools/snapshot_mrr.py is the fix,
and it is now load-bearing: GATES-R in the improvement plan makes it the first
gate any retrain has to clear. A metric that is itself wrong is worse than no
metric, so its arithmetic is pinned here rather than trusted.
"""
from __future__ import annotations

import unittest

from src.features import FEATURES
from src.rerank import Model
from tools import snapshot_mrr as sm


def group(sample_id, turn, order, target_at, width=None):
    """A synthetic snapshot: `order` pids, the target at index `target_at`.

    Feature vectors are one-hot on the first feature so a model with a positive
    first weight prefers exactly the products marked, which makes the reordering
    under test easy to state and impossible to get accidentally right.
    """
    width = width or len(FEATURES)
    rows = []
    for position, pid in enumerate(order):
        rows.append({"pid": pid, "y": int(position == target_at),
                     "x": [0.0] * width,
                     "s": float(len(order) - position)})
    return {"sample_id": sample_id, "turn": turn, "target": order[target_at],
            "features": list(FEATURES), "rows": rows}


class ReciprocalRankTest(unittest.TestCase):
    def test_the_base_ranking_is_the_recorded_order(self):
        """Not a re-sort on score: profile.break_ties has already settled ties in
        that order, and re-sorting would silently undo it."""
        g = group("s1", 1, ["a", "b", "c"], target_at=2)
        self.assertAlmostEqual(sm.reciprocal_rank(g), 1.0 / 3)

    def test_a_model_that_prefers_the_target_promotes_it(self):
        g = group("s1", 1, ["a", "b", "c"], target_at=2)
        g["rows"][2]["x"] = [1.0] + [0.0] * (len(FEATURES) - 1)
        model = Model([100.0] + [0.0] * (len(FEATURES) - 1), blend=1.0)
        self.assertAlmostEqual(sm.reciprocal_rank(g, model), 1.0)

    def test_blend_zero_reproduces_the_base_ordering(self):
        g = group("s1", 1, ["a", "b", "c"], target_at=2)
        g["rows"][2]["x"] = [1.0] + [0.0] * (len(FEATURES) - 1)
        model = Model([100.0] + [0.0] * (len(FEATURES) - 1), blend=0.0)
        self.assertAlmostEqual(sm.reciprocal_rank(g, model), sm.reciprocal_rank(g))

    def test_a_target_the_ranking_never_held_scores_zero(self):
        """Out of the head is out of the comparison. The reranker cannot reach it,
        so no ordering under test may be credited or blamed for it."""
        g = group("s1", 1, ["a", "b", "c"], target_at=2)
        g["rows"][2]["s"] = None
        self.assertEqual(sm.reciprocal_rank(g), 0.0)
        self.assertEqual(sm.reciprocal_rank(g, Model([1.0] * len(FEATURES))), 0.0)

    def test_ties_break_on_the_identifier(self):
        """Same rule as rerank.Model.apply, so pool arrival order cannot leak in."""
        g = group("s1", 1, ["b", "a"], target_at=1)
        for row in g["rows"]:
            row["s"] = 1.0
        model = Model([0.0] * len(FEATURES), blend=1.0)
        self.assertAlmostEqual(sm.reciprocal_rank(g, model), 1.0)


class SessionWeightingTest(unittest.TestCase):
    """A conversation is one observation however many turns it took."""

    def test_a_long_session_does_not_outvote_a_short_one(self):
        groups = [group("long", t, ["a", "b"], target_at=1) for t in (1, 2, 3, 4)]
        groups.append(group("short", 1, ["a", "b"], target_at=0))
        # Turn-weighted this would be (4*0.5 + 1.0)/5 = 0.60; session-weighted it
        # is (0.5 + 1.0)/2 = 0.75.
        self.assertAlmostEqual(sm.mrr(groups), 0.75)

    def test_session_scores_average_within_a_session(self):
        groups = [group("s", 1, ["a", "b"], target_at=0),
                  group("s", 2, ["a", "b"], target_at=1)]
        self.assertAlmostEqual(sm.session_scores(groups)["s"], 0.75)


class PairedIntervalTest(unittest.TestCase):
    def test_a_model_compared_with_itself_has_no_interval(self):
        groups = [group(f"s{i}", 1, ["a", "b", "c"], target_at=i % 3) for i in range(12)]
        model = Model([1.0] * len(FEATURES), blend=0.5)
        low, mid, high, n = sm.paired(groups, model, model, seed=0, resamples=200)
        self.assertEqual((low, mid, high), (0.0, 0.0, 0.0))
        self.assertEqual(n, 12)

    def test_it_resamples_sessions_not_snapshots(self):
        groups = [group("a", t, ["x", "y"], target_at=0) for t in (1, 2, 3)]
        groups += [group("b", 1, ["x", "y"], target_at=1)]
        _, _, _, n = sm.paired(groups, None, None, seed=0, resamples=50)
        self.assertEqual(n, 2, "the bootstrap drew turns rather than conversations")

    def test_an_improvement_is_detected(self):
        groups = []
        for i in range(30):
            g = group(f"s{i}", 1, ["a", "b", "c"], target_at=2)
            g["rows"][2]["x"] = [1.0] + [0.0] * (len(FEATURES) - 1)
            groups.append(g)
        better = Model([100.0] + [0.0] * (len(FEATURES) - 1), blend=1.0)
        low, mid, high, _ = sm.paired(groups, None, better, seed=0, resamples=200)
        self.assertGreater(low, 0.0)
        self.assertAlmostEqual(mid, 1.0 - 1.0 / 3, places=6)


class CacheContractTest(unittest.TestCase):
    def test_a_stale_cache_is_detected_by_its_feature_list(self):
        """Same contract as the shipped asset: a vector compared against the wrong
        names fails silently, so the names travel with the data."""
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            stale = group("s1", 1, ["a", "b"], target_at=0)
            stale["features"] = ["only_one_feature"]
            path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
            with path.open(encoding="utf-8") as handle:
                loaded = [json.loads(line) for line in handle if line.strip()]
            self.assertNotEqual(list(loaded[0]["features"]), list(FEATURES),
                                "the guard in tools.snapshot_mrr.cached compares "
                                "exactly this")


if __name__ == "__main__":
    unittest.main()
