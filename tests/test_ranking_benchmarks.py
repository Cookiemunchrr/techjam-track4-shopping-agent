"""V4-0 aggregate report arithmetic and identifier-safe diagnostics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.features import FEATURES
from tools import v4_baseline


class ZeroModel:
    blend = 0.05

    @staticmethod
    def score_vector(values):
        return 0.0


class ReverseModel:
    blend = 2.0

    @staticmethod
    def score_vector(values):
        return values[0]


def group(sample_id: str, turn: int, target_position: int, *, scenario="buying"):
    pids = [f"{sample_id}_a", f"{sample_id}_b", f"{sample_id}_c"]
    rows = [
        {"pid": pid, "y": int(index == target_position),
         "x": [float(index)] * len(FEATURES),
         "s": float(3 - index), "live_rank": index + 1,
         "in_rerank_head": True}
        for index, pid in enumerate(pids)
    ]
    return {
        "sample_id": sample_id, "scenario_type": scenario, "turn": turn,
        "target": pids[target_position], "rows": rows, "live_rows": rows,
        "reference_message": f"message {turn}", "reachability": "rerankable",
        "diagnostics": {"active_constraints": turn - 1, "route": "exact"},
    }


class RankReportTest(unittest.TestCase):
    def test_rank_histogram_and_session_changes_use_live_order(self):
        groups = [group("long", turn, 2) for turn in (1, 2, 3)]
        groups.append(group("short", 1, 0, scenario="browsing"))
        histogram = v4_baseline.rank_histogram(groups)
        self.assertEqual(histogram, {"1": 1, "3": 3})
        report = v4_baseline.snapshot_report(groups, ZeroModel())
        self.assertEqual(report["comparison"]["sessions"], 2)
        self.assertEqual(report["session_changes"], {
            "improved": 0, "worsened": 0, "tied": 2,
        })
        self.assertIn("Buying".lower(), {
            name.lower() for name in report["slices"]["scenario"]
        })

    def test_transcript_hash_is_deterministic_and_order_sensitive(self):
        groups = [group("a", 1, 0), group("b", 1, 1)]
        digest = v4_baseline.transcript_sha256(groups)
        self.assertEqual(digest, v4_baseline.transcript_sha256(list(groups)))
        self.assertNotEqual(digest, v4_baseline.transcript_sha256(list(reversed(groups))))

    def test_internal_order_hash_is_model_and_order_sensitive(self):
        groups = [group("a", 1, 2), group("b", 1, 0)]
        base = v4_baseline.internal_order_sha256(groups)
        self.assertEqual(base, v4_baseline.internal_order_sha256(list(groups)))
        self.assertNotEqual(base, v4_baseline.internal_order_sha256(groups, ReverseModel()))
        self.assertNotEqual(base, v4_baseline.internal_order_sha256(list(reversed(groups))))
        paired = v4_baseline.order_pair_sha256(groups, ReverseModel())
        self.assertNotEqual(paired, v4_baseline.order_pair_sha256(
            list(reversed(groups)), ReverseModel()
        ))

    def test_protected_snapshot_pin_fails_closed_on_every_contract_field(self):
        expected = {
            "payload_sha256": "a" * 64,
            "catalog_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "model_sha256": "d" * 64,
            "transcript_sha256": "e" * 64,
            "order_pair_sha256": "f" * 64,
            "comparison": {"before": 0.5, "after": 0.6, "delta": 0.1},
        }
        v4_baseline.validate_snapshot_pin(dict(expected), expected)
        for key in expected:
            actual = dict(expected)
            actual[key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(
                v4_baseline.CacheError, key
            ):
                v4_baseline.validate_snapshot_pin(actual, expected)

    def test_report_delta_is_observed_arithmetic_not_bootstrap_median(self):
        groups = [group("a", 1, 2), group("b", 1, 0)]
        report = v4_baseline.snapshot_mrr.report(groups, None, ReverseModel())
        self.assertEqual(
            report["delta"], round(report["after"] - report["before"], 5)
        )
        self.assertIn("bootstrap_median_delta", report)


class PairMassTest(unittest.TestCase):
    def test_full_split_is_evaluation_only(self):
        self.assertEqual(v4_baseline.TRAINING_SPLITS, {"dev", "holdout"})
        self.assertNotIn("full", v4_baseline.TRAINING_SPLITS)

    def test_pair_mass_is_aggregated_by_session_not_snapshot(self):
        groups = [group("a", 1, 0), group("a", 2, 1), group("b", 1, 2)]
        groups[2]["reachability"] = "rerank_depth_miss"
        report = v4_baseline.pair_mass_report(groups)
        a0 = report["a0_compatibility_pairs_per_session"]
        self.assertEqual((a0["min"], a0["max"]), (2, 4))
        actionable = report["actionable_pairs_per_session"]
        self.assertEqual(actionable["sessions"], 2)
        self.assertEqual((actionable["min"], actionable["max"]), (0, 4))
        self.assertEqual(report["reachability"]["rerank_depth_miss"], 1)

    def test_cache_paths_inside_repo_are_machine_independent(self):
        missing = v4_baseline.REPO / "analysis" / "definitely_missing_v4_cache"
        report = v4_baseline.cache_summary(missing, validated=False)
        self.assertEqual(report["path"], "analysis/definitely_missing_v4_cache")
        self.assertNotIn(str(v4_baseline.REPO), report["path"])


class PriorProbeTest(unittest.TestCase):
    def test_date_and_popularity_probe_has_an_exact_synthetic_oracle(self):
        popularity = FEATURES.index("popularity")
        width = len(FEATURES)
        winner_x, target_x = [0.0] * width, [0.0] * width
        winner_x[popularity], target_x[popularity] = 0.9, 0.1
        rows = [
            {"pid": "winner", "y": 0, "x": winner_x, "s": 2.0},
            {"pid": "target", "y": 1, "x": target_x, "s": 1.0},
        ]
        snapshot = {
            "sample_id": "s", "scenario_type": "buying", "turn": 1,
            "target": "target", "rows": rows, "live_rows": rows,
        }
        products = [
            {"parent_asin": "winner", "rating_number": 100,
             "details": {"Date First Available": "January 1, 2020"}},
            {"parent_asin": "target", "rating_number": 10,
             "details": {"Date First Available": "January 1, 2021"}},
            {"parent_asin": "other", "rating_number": 1, "details": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in products),
                encoding="utf-8",
            )
            report = v4_baseline.catalog_prior_probe(
                catalog, [snapshot], ZeroModel()
            )
        self.assertEqual(report["turn1_shipped_losses"], 1)
        self.assertEqual(report["target_less_popular_than_winner"], 1)
        self.assertEqual(report["catalog_dates"], {"products": 3, "parseable": 2})
        self.assertEqual(report["target_newer_than_winner"], 1)
        self.assertEqual(report["target_has_fewer_reviews"], 1)

    def test_holdout_mode_reports_popularity_without_reading_recency_metadata(self):
        popularity = FEATURES.index("popularity")
        width = len(FEATURES)
        winner_x, target_x = [0.0] * width, [0.0] * width
        winner_x[popularity], target_x[popularity] = 0.9, 0.1
        rows = [
            {"pid": "winner", "y": 0, "x": winner_x, "s": 2.0},
            {"pid": "target", "y": 1, "x": target_x, "s": 1.0},
        ]
        snapshot = {"sample_id": "s", "scenario_type": "buying", "turn": 1,
                    "target": "target", "rows": rows, "live_rows": rows}
        report = v4_baseline.catalog_prior_probe(
            "/path/must/not/be/read.jsonl", [snapshot], ZeroModel(),
            include_recency=False,
        )
        self.assertEqual(report["target_less_popular_than_winner"], 1)
        self.assertEqual(report["recency_probe"]["status"], "withheld")
        self.assertNotIn("target_newer_than_winner", report)


if __name__ == "__main__":
    unittest.main()
