"""Group E — metric integrity.

The existing regression suite pins score floors. These are the tests that stop a
floor from being met for the wrong reason.

E5 exists because a monkeypatch that broke every single turn once produced a clean
run reporting score=0.00000 with no traceback and no non-zero exit: `respond`'s
blanket `except Exception` swallowed 2,000 failures in silence.

E6 exists because MRR 0.954 was an artifact of `base_width = 1`, not of ranking
quality -- 186/200 hits landed at rank 1 because the agent emitted one item. At
width 10 the same retrieval gives MRR 0.656.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CATALOG = str(REPO / "data" / "catalog.jsonl")
PUBLIC = str(REPO / "data" / "public_set.jsonl")

# Re-baselined when recommendation width moved from a flat 1 to confidence-driven.
# The drop is deliberate; see tests/test_regression.py and src/policy.py. E6 below
# is the floor that presentation policy cannot satisfy.
FLOOR_FULL = 0.88
FLOOR_MRR_AT_WIDTH_10 = 0.55
FLOOR_POOL_RECALL = 0.97


def _evaluate(agent, dataset=PUBLIC):
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    ids, categories, products = catalog_index(CATALOG)
    return evaluate(agent, load_jsonl(dataset), ids, categories, products)


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class ExceptionSilenceTest(unittest.TestCase):
    """E5 — a broken agent must be loud, not merely low-scoring."""

    def test_e5_no_internal_exception_during_a_full_evaluation(self):
        from src.agent import Agent
        agent = Agent(CATALOG)
        result = _evaluate(agent)
        self.assertGreaterEqual(result["recommended_technical_score"], FLOOR_FULL)
        self.assertEqual(
            getattr(agent, "failures", 0), 0,
            f"{getattr(agent, 'failures', 0)} turns were swallowed by the exception guard")

    def test_e5b_the_guard_counts_rather_than_hides(self):
        from src.agent import Agent
        from tests.fixtures import PROFILE, RichCatalog
        with RichCatalog() as path:
            agent = Agent(path)
            agent.reset("boom", PROFILE)
            original = agent.scorer.rank

            def explode(*args, **kwargs):
                raise RuntimeError("induced")

            agent.scorer.rank = explode
            try:
                response = agent.respond("boom", "I'm looking for Accessories Belts.", 1, 10)
            finally:
                agent.scorer.rank = original
            self.assertIsInstance(response["message"], str)
            self.assertGreaterEqual(agent.failures, 1,
                                    "a swallowed exception left no trace on the agent")


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class RankingQualityTest(unittest.TestCase):
    """E6 — separate ranking quality from disclosure policy."""

    def test_e6_mrr_floor_at_fixed_width_ten(self):
        from src.agent import Agent
        os.environ["P_PROBE"], os.environ["P_WIDEN"] = "10", "1"
        try:
            import importlib
            import src.agent as module
            importlib.reload(module)
            result = _evaluate(module.Agent(CATALOG))
        finally:
            os.environ.pop("P_PROBE", None)
            os.environ.pop("P_WIDEN", None)
            import importlib
            import src.agent as module
            importlib.reload(module)
        self.assertGreaterEqual(
            result["mrr"], FLOOR_MRR_AT_WIDTH_10,
            "MRR at width 10 is the honest ranking number; the headline MRR is "
            "mostly a consequence of committing one candidate at a time")


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class RecallTest(unittest.TestCase):
    """E7 — measure recall before ranking, so a scorer change cannot mask a routing loss."""

    def test_e7_target_survives_routing_independently_of_the_scorer(self):
        from evaluator.local_evaluator import (catalog_index, coarse_category,
                                               initial_message, load_jsonl,
                                               materialize_hidden_fields)
        from src.agent import Agent
        from src.routing import candidates, category_key
        from src.text import split_clauses

        ids, categories, products = catalog_index(CATALOG)
        samples = load_jsonl(PUBLIC)
        agent = Agent(CATALOG)
        kept = 0
        for sample in samples:
            target = str(sample["ground_truth"]["parent_asin"])
            card, behaviour = materialize_hidden_fields(sample, products)
            message = initial_message({**sample, "intent_card": card, "behavior": behaviour},
                                      coarse_category(categories.get(target, [])), set())
            key = category_key(split_clauses(message))
            kept += target in set(candidates(agent.catalog, key))
        recall = kept / len(samples)
        self.assertGreaterEqual(recall, FLOOR_POOL_RECALL,
                                f"pool recall {recall:.3f}: routing is dropping targets")


class NoiseTest(unittest.TestCase):
    """E9 — a reported delta smaller than the harness's own noise is not a result."""

    def test_e9_the_paraphrase_harness_is_seed_stable(self):
        from tools.paraphrase import ParaphrasingAgent

        class Echo:
            def reset(self, session_id, profile):
                pass

            def respond(self, session_id, message, turn, top_k):
                return {"message": message, "ask_attribute": None,
                        "recommendations": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        message = "I'm looking for Accessories Belts. A key requirement is: 100% Leather."
        seen = set()
        for _ in range(4):
            wrapper = ParaphrasingAgent(Echo(), level=3, seed=0)
            wrapper.reset("public_0001", {})
            seen.add(wrapper.respond("public_0001", message, 1, 10)["message"])
        self.assertEqual(
            len(seen), 1,
            "the paraphraser is not reproducible; hash() on str is PYTHONHASHSEED-salted, "
            f"so reported paraphrase deltas are noise: {seen}")


if __name__ == "__main__":
    unittest.main()


class BenchPairingTest(unittest.TestCase):
    """The paired comparison, across every mode that produces rows.

    `--against` used to key on (split, paraphrase_level), so only the plain mode
    could be compared with an earlier run -- and the adversarial axes and the
    long-tail slice, which are the rows a reranker retrain is actually held to,
    reported a point estimate and nothing else. Generalising the key broke two
    things at once, and both are pinned here: a summary row carrying no per-session
    detail crashed the comparison, and the long-tail rows lost the sample size that
    is the entire point of that slice -- 25 sessions is the finding.
    """

    def test_rows_from_different_modes_do_not_collide(self):
        from tools.bench import _row_key
        keys = [_row_key({"split": "dev", "paraphrase_level": 0}),
                _row_key({"split": "full", "paraphrase_level": 0}),
                _row_key({"axis": "category"}),
                _row_key({"axis": "control"}),
                _row_key({"slice": "below p90 popularity"}),
                _row_key({"slice": "at or above"})]
        self.assertEqual(len(set(keys)), len(keys))

    def test_a_summary_row_without_session_detail_is_skipped(self):
        """The long-tail percentile line reports a count under a name the metric
        rows use for a list. Checked by type, not by truthiness."""
        from tools.bench import _row_key
        summary = {"slice": "target popularity percentile (median)",
                   "value": 0.9926, "session_count": 200}
        self.assertIsNotNone(_row_key(summary))
        self.assertNotIsInstance(summary.get("sessions"), list)

    def test_the_longtail_baseline_carries_both_the_count_and_the_detail(self):
        import json
        import os
        if not os.path.exists("analysis/longtail.json"):
            self.skipTest("baseline not present")
        with open("analysis/longtail.json", encoding="utf-8") as handle:
            rows = json.load(handle)
        scored = [row for row in rows if "technical_score" in row]
        self.assertTrue(scored, "no scored slice in the long-tail baseline")
        for row in scored:
            self.assertIsInstance(row.get("sessions"), list,
                                  "--against has nothing to pair on")
            self.assertEqual(row["session_count"], len(row["sessions"]),
                             "the reported sample size disagrees with the detail")

    def test_an_identical_run_pairs_to_exactly_zero(self):
        from tools.bench import _by_id, paired_interval
        sessions = [{"sample_id": f"s{i}", "hit": True,
                     "reciprocal_rank": 1.0 / (1 + i % 4), "first_hit_turn": 1 + i % 3}
                    for i in range(40)]
        low, mid, high, n = paired_interval(_by_id(sessions), _by_id(sessions),
                                            resamples=200)
        self.assertEqual((low, mid, high), (0.0, 0.0, 0.0))
        self.assertEqual(n, 40)
