"""V4-0 data foundations for the later metric-aligned objective.

These tests do not implement or bless a V4-1 candidate model.  They pin the
information boundary that every A0..C rung must share: the complete raw serving
head, the literal rounded legacy projection, and an explicit distinction between
rerankable, depth-missed, and upstream-pool-missed targets.
"""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from src.features import FEATURES as LIVE_FEATURES
from tools import rerank_train
from tools.rerank_data import snapshot_record, snapshots, training_contract
from tools.rerank_provenance import write_cache


FEATURES = ["lexical", "prior"]


def vector(pid: str):
    number = int(pid[1:])
    return [number / 7.0, number / 13.0]


def head(size: int = 200):
    return [(1000.0 - index / 17.0, f"p{index}")
            for index in range(1, size + 1)]


class RawServingViewTest(unittest.TestCase):
    def record(self, target="p100", pool=None, compatibility_depth=40,
               with_scores=False):
        candidates = head()
        return snapshot_record(
            sample_id="session", scenario_type="buying", turn=2,
            target=target, message="I need a durable blue item.",
            features=FEATURES, ranked=candidates,
            vector_for=vector,
            catalog_ids={pid for _, pid in candidates} | {target},
            candidate_pool=(set(pool) if pool is not None
                            else {pid for _, pid in candidates} | {target}),
            compatibility_depth=compatibility_depth,
            compatibility_scores=with_scores,
        )

    def test_full_live_head_keeps_raw_values_and_exact_serving_ranks(self):
        record = self.record(target="p100")
        self.assertEqual(len(record["live_rows"]), 200)
        target = record["live_rows"][99]
        self.assertEqual(target["pid"], "p100")
        self.assertEqual(target["live_rank"], 100)
        self.assertEqual(record["target_live_rank"], 100)
        self.assertEqual(target["x"], vector("p100"))
        self.assertEqual(target["s"], head()[99][0])
        self.assertEqual(record["reachability"], "rerankable")

    def test_legacy_projection_is_literal_dedupe_head_40_plus_target(self):
        inside = self.record(target="p10")
        self.assertEqual(len(inside["rows"]), 40)
        self.assertEqual(sum(not row["y"] for row in inside["rows"]), 39)
        self.assertEqual([row["pid"] for row in inside["rows"]],
                         [f"p{i}" for i in range(1, 41)])

        outside = self.record(target="p100")
        self.assertEqual(len(outside["rows"]), 41)
        self.assertEqual(sum(not row["y"] for row in outside["rows"]), 40)
        self.assertEqual(outside["rows"][-1]["pid"], "p100")
        # Compatibility values are deliberately rounded; live values are not.
        self.assertEqual(outside["rows"][-1]["x"],
                         [round(value, 6) for value in vector("p100")])
        self.assertNotEqual(outside["rows"][-1]["x"],
                            outside["live_rows"][99]["x"])

    def test_rank_100_is_not_relabelled_as_synthetic_position_41(self):
        record = self.record(target="p100", with_scores=True)
        self.assertEqual(record["target_live_rank"], 100)
        self.assertEqual(record["rows"][-1]["pid"], "p100")
        self.assertEqual(record["rows"][-1]["s"],
                         round(record["live_rows"][99]["s"], 6))

    def test_compatibility_score_flag_changes_only_legacy_rows(self):
        without = self.record(target="p10", with_scores=False)
        with_scores = self.record(target="p10", with_scores=True)
        self.assertNotIn("s", without["rows"][0])
        self.assertIn("s", with_scores["rows"][0])
        self.assertEqual(without["live_rows"], with_scores["live_rows"])


class ReachabilityTest(unittest.TestCase):
    def record(self, *, target: str, pool):
        return snapshot_record(
            sample_id="session", scenario_type="browsing", turn=1,
            target=target, message="Something practical.", features=FEATURES,
            ranked=head(5), vector_for=vector,
            catalog_ids={f"p{i}" for i in range(1, 8)},
            candidate_pool=set(pool), compatibility_depth=3,
            compatibility_scores=True,
        )

    def test_target_in_pool_but_outside_head_is_a_depth_miss(self):
        record = self.record(target="p6", pool={f"p{i}" for i in range(1, 7)})
        self.assertFalse(record["target_in_rerank_head"])
        self.assertTrue(record["target_in_pool"])
        self.assertIsNone(record["target_live_rank"])
        self.assertEqual(record["reachability"], "rerank_depth_miss")
        self.assertIsNone(record["rows"][-1]["s"])
        self.assertFalse(any(row["y"] for row in record["live_rows"]))

    def test_target_outside_pool_is_a_route_pool_miss(self):
        record = self.record(target="p7", pool={f"p{i}" for i in range(1, 6)})
        self.assertFalse(record["target_in_pool"])
        self.assertFalse(record["target_in_rerank_head"])
        self.assertEqual(record["reachability"], "route_pool_miss")

    def test_target_in_head_is_also_in_pool(self):
        with self.assertRaisesRegex(ValueError, "live target is absent from candidate_pool"):
            self.record(target="p2", pool={"p1", "p3", "p4", "p5"})


class DeterminismTest(unittest.TestCase):
    def test_same_observable_inputs_build_equal_records(self):
        kwargs = dict(
            sample_id="s", scenario_type="boundary", turn=3, target="p3",
            message="No preference.", features=FEATURES, ranked=head(5),
            vector_for=vector, catalog_ids={f"p{i}" for i in range(1, 6)},
            candidate_pool={f"p{i}" for i in range(1, 6)},
            compatibility_depth=4, compatibility_scores=True,
        )
        self.assertEqual(snapshot_record(**kwargs), snapshot_record(**kwargs))


class ReplayFailureTest(unittest.TestCase):
    def test_swallowed_agent_failure_aborts_instead_of_reusing_trace_state(self):
        class FailingAgent:
            failures = 0
            trace_features = False
            trace_pool = False

            def reset(self, session_id, profile):
                pass

            def respond(self, session_id, message, turn, top_k):
                self.failures += 1
                return {"message": "fallback", "ask_attribute": None,
                        "recommendations": []}

        sample = {
            "sample_id": "s", "scenario_type": "buying",
            "ground_truth": {"parent_asin": "p1"}, "user_profile": {},
            "intent_card": {"hard_constraints": ["cotton"],
                            "soft_preferences": []},
            "behavior": {},
        }
        with self.assertRaisesRegex(RuntimeError, "swallowed Agent.respond failure"):
            list(snapshots(
                FailingAgent(), [sample], {"p1"}, {}, {}, negatives=1
            ))


class TrainingLoaderTest(unittest.TestCase):
    def fixture(self):
        return snapshot_record(
            sample_id="s", scenario_type="buying", turn=1, target="p3",
            message="fixture", features=LIVE_FEATURES, ranked=head(5),
            vector_for=lambda pid: [float(int(pid[1:]))] * len(LIVE_FEATURES),
            catalog_ids={f"p{i}" for i in range(1, 6)},
            candidate_pool={f"p{i}" for i in range(1, 6)},
            compatibility_depth=3, compatibility_scores=False,
        )

    def test_trainer_uses_manifested_cache_and_legacy_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, split = root / "catalog.jsonl", root / "split.jsonl"
            catalog.write_text(
                "".join(f'{{"parent_asin":"p{i}"}}\n' for i in range(1, 6)),
                encoding="utf-8",
            )
            split.write_text(
                '{"sample_id":"s","scenario_type":"buying",'
                '"ground_truth":{"parent_asin":"p3"}}\n',
                encoding="utf-8",
            )
            cache = root / "training.jsonl"
            contract = training_contract(catalog, split, negatives=3)
            self.assertNotIn("tools/rerank_train.py", contract["sources"])
            write_cache(cache, [self.fixture()], contract)

            groups = rerank_train.load(
                cache, catalog=catalog, split_path=split, negatives=3
            )
            self.assertEqual(groups[0]["target_live_rank"], 3)
            self.assertEqual(len(groups[0]["live_rows"]), 5)
            self.assertEqual(len(rerank_train.pairs(groups)), 2)

    def test_unmanifested_training_data_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, split = root / "catalog.jsonl", root / "split.jsonl"
            catalog.write_text(
                "".join(f'{{"parent_asin":"p{i}"}}\n' for i in range(1, 6)),
                encoding="utf-8",
            )
            split.write_text(
                '{"sample_id":"s","scenario_type":"buying",'
                '"ground_truth":{"parent_asin":"p3"}}\n',
                encoding="utf-8",
            )
            cache = root / "legacy.jsonl"
            cache.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "legacy/unprovenanced"):
                rerank_train.load(
                    cache, catalog=catalog, split_path=split, negatives=3
                )


if __name__ == "__main__":
    unittest.main()
