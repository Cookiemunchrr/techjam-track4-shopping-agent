"""End-to-end score floors on the real public set.

Slow (loads the 50k catalog and simulates 200 sessions). Skips itself when the
catalog is absent so the fast suite still runs on a clean checkout.

Floors are set below the measured score, not at it -- a floor that equals the
current number turns every future improvement into a failing test.
"""
from __future__ import annotations

import os
import unittest

CATALOG = "data/catalog.jsonl"
PUBLIC = "data/public_set.jsonl"
DEV = "analysis/dev.jsonl"
HOLDOUT = "analysis/holdout.jsonl"

# Measured after the move to confidence-driven recommendation width:
#   official  full 0.90408, dev 0.91221, holdout 0.89383
#   shadow    0.89594
#
# The official floors moved DOWN by roughly 0.04 and that was deliberate, not a
# regression. A flat `base_width = 1` committed one item on 85% of turns, which the
# official evaluator rewards because a session ends the moment the target appears in
# the *shown* slate. src/policy.py has the full reasoning. The number that did not
# move is the shadow score, which is the point.
#
# So the shadow floor below is the one that actually guards ranking quality. A change
# that lifts the official score while degrading real retrieval must fail a test, or
# the suite is only measuring how well we play the simulator.
BASELINE_SCORE = 0.10671
FLOOR_FULL = 0.88
FLOOR_HOLDOUT = 0.87
FLOOR_SHADOW = 0.87


def _score(dataset: str) -> dict:
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from src.agent import Agent
    ids, categories, products = catalog_index(CATALOG)
    return evaluate(Agent(CATALOG), load_jsonl(dataset), ids, categories, products)


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class RegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _score(PUBLIC)

    def test_beats_the_provided_baseline_by_a_wide_margin(self):
        self.assertGreater(self.result["recommended_technical_score"], BASELINE_SCORE * 5)

    def test_score_floor_on_the_public_set(self):
        self.assertGreaterEqual(self.result["recommended_technical_score"], FLOOR_FULL)

    def test_no_scenario_collapses(self):
        """An aggregate score can hide a scenario scoring near zero -- override did, once."""
        for name, metrics in self.result["scenario_metrics"].items():
            self.assertGreaterEqual(metrics["hit_rate_at_10"], 0.60, f"{name} collapsed")

    def test_reports_no_token_usage(self):
        self.assertEqual(self.result["reported_token_usage"]["total_tokens"], 0)

    def test_finds_most_targets_well_inside_the_turn_budget(self):
        self.assertLess(self.result["mttc"], 6.0)

    def test_ranking_quality_floor_independent_of_recommendation_width(self):
        """The floor that cannot be met by showing fewer candidates.

        tools/shadow.py scores the untruncated internal ranking, so this number
        moves only when retrieval actually changes. Guarding it stops the suite
        from being satisfiable by presentation policy alone.
        """
        from src.agent import Agent
        from evaluator.local_evaluator import catalog_index, load_jsonl
        from tools.shadow import run as shadow_run
        ids, categories, products = catalog_index(CATALOG)
        base = Agent(CATALOG)
        result = shadow_run(lambda: Agent.sharing_index(base), load_jsonl(PUBLIC),
                            ids, categories, products)
        self.assertGreaterEqual(result["shadow_score"], FLOOR_SHADOW)
        self.assertEqual(result["retrieval_hit_rate_at_10"], 1.0,
                         "retrieval stopped reaching every target")


@unittest.skipUnless(os.path.exists(CATALOG) and os.path.exists(HOLDOUT),
                     "held-out split not present")
class HeldOutTest(unittest.TestCase):
    def test_generalises_to_the_untuned_half(self):
        """Weights were fitted on dev only. A large dev-holdout gap means overfitting."""
        dev = _score(DEV)["recommended_technical_score"]
        holdout = _score(HOLDOUT)["recommended_technical_score"]
        self.assertGreaterEqual(holdout, FLOOR_HOLDOUT)
        self.assertLess(dev - holdout, 0.06,
                        f"dev {dev:.4f} vs holdout {holdout:.4f}: tuned too hard on dev")


if __name__ == "__main__":
    unittest.main()
