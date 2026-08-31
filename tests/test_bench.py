"""Split/axis isolation and paired-comparison refusal for tools.bench.

A benchmark that measures the wrong sessions is worse than one that crashes,
because it reads as a pass. These tests pin the V6 Phase-0 repairs: an explicit
--splits reads exactly the requested file, --axes runs exactly the requested
adversarial rows, a missing split fails closed, and `--against` refuses to
compare two runs whose inputs or session sets disagree instead of silently
intersecting them.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import bench


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_harness(recorded: list):
    """A harness double that records which split files were opened."""
    def catalog_index(catalog):
        return None, None, None

    def load_jsonl(path):
        recorded.append(path)
        return []

    def evaluate(agent, samples, ids, categories, products):
        return {
            "recommended_technical_score": 0.5,
            "hit_rate_at_10": 1.0,
            "mrr": 0.5,
            "mttc": 2.0,
            "sessions": [],
            "scenario_metrics": {},
        }

    return catalog_index, evaluate, load_jsonl


class SplitFileCase(unittest.TestCase):
    """Common fixture: bench.SPLITS pointed at three distinct temp files."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.paths = {}
        for split in ("dev", "holdout", "full"):
            path = root / f"{split}.jsonl"
            path.write_text(f'{{"sample_id": "{split}-1"}}\n', encoding="utf-8")
            self.paths[split] = str(path)
        self._splits = mock.patch.object(bench, "SPLITS", dict(self.paths))
        self._splits.start()
        self.addCleanup(self._splits.stop)
        self.addCleanup(self.directory.cleanup)


class SplitIsolationTest(SplitFileCase):
    def _run_clean(self, splits):
        recorded = []
        with mock.patch.object(bench, "_harness", lambda: _fake_harness(recorded)), \
                mock.patch.object(bench, "_agent", return_value=object()):
            rows = bench.run(splits, [0])
        return rows, recorded

    def test_dev_split_reads_only_the_dev_file(self):
        """--splits dev opens analysis/dev.jsonl and nothing else (T13)."""
        rows, recorded = self._run_clean(["dev"])
        self.assertEqual(recorded, [self.paths["dev"]])
        self.assertEqual([row["split"] for row in rows], ["dev"])
        self.assertEqual(rows[0]["dataset_sha256"],
                         _sha256(Path(self.paths["dev"])))

    def test_output_rows_carry_split_and_dataset_hash(self):
        rows, _ = self._run_clean(["dev"])
        for row in rows:
            self.assertIn("split", row)
            self.assertIn("dataset_sha256", row)

    def test_a_missing_split_is_an_error_not_an_empty_pass(self):
        """The old behavior skipped an absent split and reported the remainder
        as though it were the requested measurement."""
        Path(self.paths["holdout"]).unlink()
        with self.assertRaises(SystemExit):
            self._run_clean(["holdout"])

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._run_clean(["private"])


class AdversarialSelectionTest(SplitFileCase):
    def _run_adversarial(self, *args, **kwargs):
        import tools.adversarial as adversarial
        recorded = []
        with mock.patch.object(bench, "_harness", lambda: _fake_harness(recorded)), \
                mock.patch.object(bench, "_agent", return_value=object()), \
                mock.patch.object(adversarial, "Adversarial",
                                  side_effect=lambda agent, axes, seed: agent), \
                mock.patch.object(adversarial, "drifted_categories",
                                  side_effect=lambda categories, components: categories):
            rows = bench.run_adversarial(*args, **kwargs)
        return rows, recorded

    def test_axes_category_runs_only_control_and_category(self):
        """--axes category runs control/category and no other row (T13)."""
        rows, recorded = self._run_adversarial(0, splits=["dev"], axes=["category"])
        self.assertEqual([row["axis"] for row in rows], ["control", "category"])
        self.assertTrue(all(row["split"] == "dev" for row in rows))
        self.assertEqual(recorded, [self.paths["dev"]])

    def test_granularity_rows_run_only_when_requested(self):
        rows, _ = self._run_adversarial(0, splits=["dev"], axes=["granularity=1"])
        self.assertEqual([row["axis"] for row in rows], ["control", "granularity=1"])

    def test_default_adversarial_run_is_the_unchanged_full_matrix(self):
        """Behavior-neutrality: no flags means the full set, every axis."""
        rows, recorded = self._run_adversarial(0)
        self.assertEqual([row["axis"] for row in rows], [
            "control", "category", "natural", "scaffold", "constraint",
            "all paraphrase axes", "granularity=1", "granularity=3",
        ])
        self.assertEqual({row["split"] for row in rows}, {"full"})
        self.assertEqual(recorded, [self.paths["full"]])


class ArgparseTest(unittest.TestCase):
    def test_help_prints_cleanly(self):
        """A literal % in the --against help string crashed parser construction
        outright; --help is the canary that the fix holds."""
        with mock.patch.object(sys, "argv", ["bench", "--help"]), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            with self.assertRaises(SystemExit) as ctx:
                bench.main()
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--axes", out.getvalue())

    def test_unknown_axis_is_rejected(self):
        argv = ["bench", "--adversarial", "--axes", "typo-axis"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                bench.main()


class ComparabilityTest(unittest.TestCase):
    """`--against` refuses mismatched split/axis/source hashes instead of
    silently comparing only the intersecting session ids (T14)."""

    @staticmethod
    def _row(axis, ids, digest):
        return {"axis": axis, "split": "full", "dataset_sha256": digest,
                "sessions": [{"sample_id": i, "hit": True,
                              "reciprocal_rank": 1.0, "first_hit_turn": 1}
                             for i in ids]}

    def test_mismatched_dataset_hash_is_refused(self):
        before = self._row("category", ["a", "b"], "0" * 64)
        after = self._row("category", ["a", "b"], "1" * 64)
        with self.assertRaises(SystemExit):
            bench._require_comparable(before, after)

    def test_mismatched_session_sets_are_refused(self):
        before = self._row("category", ["a", "b"], "0" * 64)
        after = self._row("category", ["a", "c"], "0" * 64)
        with self.assertRaises(SystemExit):
            bench._require_comparable(before, after)

    def test_identical_inputs_are_comparable(self):
        before = self._row("category", ["a", "b"], "0" * 64)
        after = self._row("category", ["a", "b"], "0" * 64)
        bench._require_comparable(before, after)

    def test_legacy_artifact_without_hash_still_requires_same_sessions(self):
        """Committed baselines predate the hash field; they can pair, but only
        on proof the sessions agree."""
        before = self._row("category", ["a", "b"], "0" * 64)
        del before["dataset_sha256"]
        bench._require_comparable(before, self._row("category", ["a", "b"], "1" * 64))
        with self.assertRaises(SystemExit):
            bench._require_comparable(before, self._row("category", ["a", "c"], "1" * 64))

    def test_against_pairs_legacy_axis_rows_through_main(self):
        """Rows written before the split key existed still find their pair."""
        earlier = [{"axis": "category",
                    "sessions": [{"sample_id": "a", "hit": True,
                                  "reciprocal_rank": 1.0, "first_hit_turn": 1},
                                 {"sample_id": "b", "hit": False,
                                  "reciprocal_rank": 0.0, "first_hit_turn": None}]}]
        current = [dict(earlier[0], split="full", dataset_sha256="0" * 64,
                        technical_score=0.5)]
        with tempfile.TemporaryDirectory() as directory:
            earlier_path = Path(directory) / "earlier.json"
            output_path = Path(directory) / "out.json"
            earlier_path.write_text(json.dumps(earlier), encoding="utf-8")
            argv = ["bench", "--adversarial", "--axes", "category",
                    "--against", str(earlier_path), "--output", str(output_path)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(bench, "run_adversarial",
                                      return_value=current), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                bench.main()
            rows = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["paired_delta"]["sessions"], 2)
        self.assertEqual(rows[0]["paired_delta"]["median"], 0.0)

    def test_against_refuses_a_mismatched_run_through_main(self):
        earlier = [{"axis": "category",
                    "sessions": [{"sample_id": "a", "hit": True,
                                  "reciprocal_rank": 1.0, "first_hit_turn": 1}]}]
        current = [{"axis": "category", "split": "full", "dataset_sha256": "0" * 64,
                    "sessions": [{"sample_id": "b", "hit": True,
                                  "reciprocal_rank": 1.0, "first_hit_turn": 1}]}]
        with tempfile.TemporaryDirectory() as directory:
            earlier_path = Path(directory) / "earlier.json"
            output_path = Path(directory) / "out.json"
            earlier_path.write_text(json.dumps(earlier), encoding="utf-8")
            argv = ["bench", "--adversarial", "--axes", "category",
                    "--against", str(earlier_path), "--output", str(output_path)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(bench, "run_adversarial",
                                      return_value=current), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    bench.main()
            # Fail closed: nothing was written.
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
