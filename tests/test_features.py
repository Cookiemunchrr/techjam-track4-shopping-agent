"""The feature vector is one definition, shared by training and serving.

src/features.py exists because a reranker has one failure mode that dwarfs every
other: the features it was trained on and the features it is served differ, by a
normalisation, a default or an ordering. The model then behaves plausibly and ranks
badly, and nothing in the training report shows it. The module docstring promises
there is exactly one definition; this checks the promise on real data instead of
trusting it.

It earned its place during the feature experiment recorded in
analysis/reranker_features.json, where six features were appended, measured and
removed. The features went; the check that they were extracted identically on both
sides stayed, because the next batch will need it too.
"""
from __future__ import annotations

import os
import unittest

from src.features import FEATURES, vector


class TrainingServingParityTest(unittest.TestCase):
    """The recorded training row must be the vector the live path computed.

    tools/rerank_data.py yields a snapshot immediately after building its rows from
    `Agent.last_context`, so at the moment of the yield the agent still holds the
    context that produced them -- which is what makes the comparison possible
    without a second replay that could itself drift.
    """

    def test_a_recorded_snapshot_matches_a_live_extraction(self):
        if not (os.path.exists("data/catalog.jsonl")
                and os.path.exists("analysis/dev.jsonl")):
            self.skipTest("full catalog not present")
        from evaluator.local_evaluator import catalog_index, load_jsonl
        from tools.rerank_data import base_agent, snapshots

        ids, categories, products = catalog_index("data/catalog.jsonl")
        agent = base_agent("data/catalog.jsonl")
        sample = load_jsonl("analysis/dev.jsonl")[:1]
        checked = 0
        for group in snapshots(agent, sample, ids, categories, products):
            session_id = list(agent.sessions)[-1]
            context, _ = agent.last_context(session_id)
            self.assertIsNotNone(context, "the agent stopped recording its context")
            self.assertEqual(list(group["features"]), list(FEATURES))
            for row in group["rows"]:
                live = [round(value, 6) for value in vector(row["pid"], context)]
                for name, recorded, served in zip(FEATURES, row["x"], live):
                    self.assertAlmostEqual(
                        recorded, served, places=6,
                        msg=f"{name} differs between training and serving on "
                            f"{row['pid']} at turn {group['turn']}")
                checked += 1
        self.assertGreater(checked, 0, "no snapshot was produced to compare")


if __name__ == "__main__":
    unittest.main()
