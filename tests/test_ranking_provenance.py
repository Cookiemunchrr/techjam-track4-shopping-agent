"""V4-0 cache provenance is strict, complete, and deterministic."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.features import FEATURES
from src.rerank import DEPTH
from tools.rerank_data import snapshot_record
from tools import rerank_provenance as provenance
from tools import snapshot_mrr


class CacheContractFixture(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.catalog = self._file(
            "catalog.jsonl",
            '{"parent_asin":"p1"}\n'
            '{"parent_asin":"p2"}\n'
            '{"parent_asin":"p3"}\n',
        )
        self.split = self._file(
            "split.jsonl",
            '{"sample_id":"s1","scenario_type":"buying",'
            '"ground_truth":{"parent_asin":"p2"}}\n'
            '{"sample_id":"s2","scenario_type":"browsing",'
            '"ground_truth":{"parent_asin":"p3"}}\n',
        )
        self.evaluator = self._file("evaluator.py", "MAX_TURNS = 10\n")
        self.model = self._file("reranker.json", '{"weights":[0.0,0.0]}\n')
        self.source_a = self._file("source_a.py", "VALUE = 1\n")
        self.source_b = self._file("source_b.py", "VALUE = 2\n")
        self.features = ["first", "second"]
        self.options = {
            "kind": "snapshot_mrr",
            "numeric_precision": "six_decimal_compatibility",
            "with_scores": True,
        }

    def tearDown(self):
        self._temporary.cleanup()

    def _file(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def contract(self, *, depth=2, env=None, split=None):
        return provenance.build_contract(
            catalog_path=self.catalog,
            split_path=split or self.split,
            features=self.features,
            depth=depth,
            generator_options=self.options,
            evaluator_path=self.evaluator,
            model_path=self.model,
            source_paths={"src/a.py": self.source_a, "tools/b.py": self.source_b},
            env={} if env is None else env,
        )

    def groups(self):
        return [
            {
                "sample_id": "s1", "scenario_type": "buying", "turn": 1,
                "target": "p2", "features": list(self.features),
                "rows": [
                    {"pid": "p1", "y": 0, "x": [0.1, 0.2], "s": 2.0},
                    {"pid": "p2", "y": 1, "x": [0.2, 0.4], "s": 1.0},
                ],
            },
            {
                "sample_id": "s2", "scenario_type": "browsing", "turn": 1,
                "target": "p3", "features": list(self.features),
                "rows": [
                    {"pid": "p1", "y": 0, "x": [0.3, 0.1], "s": 3.0},
                    {"pid": "p3", "y": 1, "x": [0.4, 0.2], "s": None},
                ],
            },
        ]

    def _stored(self, path: Path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _rewrite_payload(self, path: Path, groups) -> None:
        payload = provenance.canonical_payload(groups)
        path.write_bytes(payload)
        sidecar = provenance.manifest_path(path)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        sidecar.write_text(provenance.canonical_json(manifest) + "\n", encoding="utf-8")


class ManifestTest(CacheContractFixture):
    def test_contract_hashes_every_semantic_input_and_effective_config(self):
        contract = self.contract(env={"W_POP": "2.5", "P_PROBE": "3"})
        self.assertEqual(contract["catalog"]["sha256"],
                         provenance.sha256_file(self.catalog))
        self.assertEqual(contract["ordered_split"]["sha256"],
                         provenance.sha256_file(self.split))
        self.assertEqual(contract["evaluator_protocol"]["sha256"],
                         provenance.sha256_file(self.evaluator))
        self.assertEqual(set(contract["sources"]), {"src/a.py", "tools/b.py"})
        self.assertTrue(contract["model_asset"]["present"])
        self.assertEqual(contract["feature_schema"]["names"], self.features)
        self.assertEqual(contract["depth"], 2)
        self.assertEqual(contract["generator_options"], self.options)
        self.assertEqual(contract["catalog"]["identity"]["count"], 3)
        self.assertEqual(contract["ordered_split"]["identity"]["sessions"], 2)
        self.assertEqual(contract["effective_runtime"]["scoring_weights"]["popularity"], 2.5)
        self.assertEqual(contract["effective_runtime"]["commit_policy"]["base_width"], 3)

    def test_canonical_contract_and_cache_are_byte_deterministic(self):
        first_contract = self.contract(env={"W_POP": "1.40"})
        second_contract = self.contract(env={})
        self.assertEqual(first_contract, second_contract,
                         "equivalent effective env values changed provenance")
        first, second = self.root / "first.jsonl", self.root / "second.jsonl"
        first_manifest = provenance.write_cache(first, self.groups(), first_contract)
        second_manifest = provenance.write_cache(second, self.groups(), second_contract)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(provenance.manifest_path(first).read_bytes(),
                         provenance.manifest_path(second).read_bytes())
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(provenance.load_cache(first, first_contract), self.groups())

    def test_missing_inputs_are_unavailable_not_a_pass(self):
        with self.assertRaisesRegex(provenance.CacheUnavailable, "ordered split unavailable"):
            self.contract(split=self.root / "missing.jsonl")

    def test_split_sessions_and_targets_must_both_be_disjoint(self):
        dev = self._file(
            "dev.jsonl",
            '{"sample_id":"d1","scenario_type":"buying",'
            '"ground_truth":{"parent_asin":"a"}}\n'
            '{"sample_id":"d2","scenario_type":"browsing",'
            '"ground_truth":{"parent_asin":"b"}}\n',
        )
        holdout = self._file(
            "holdout.jsonl",
            '{"sample_id":"h1","scenario_type":"boundary",'
            '"ground_truth":{"parent_asin":"c"}}\n',
        )
        report = provenance.split_isolation(dev, holdout)
        self.assertEqual(report["session_overlap"], 0)
        self.assertEqual(report["target_overlap"], 0)
        self.assertEqual(report["first"]["sessions"], 2)

        target_leak = self._file(
            "target_leak.jsonl",
            '{"sample_id":"h2","scenario_type":"buying",'
            '"ground_truth":{"parent_asin":"a"}}\n',
        )
        with self.assertRaisesRegex(provenance.CacheStale, "1 targets overlap"):
            provenance.split_isolation(dev, target_leak)

        session_leak = self._file(
            "session_leak.jsonl",
            '{"sample_id":"d2","scenario_type":"buying",'
            '"ground_truth":{"parent_asin":"z"}}\n',
        )
        with self.assertRaisesRegex(provenance.CacheStale, "1 session ids"):
            provenance.split_isolation(dev, session_leak)


class CompletePayloadValidationTest(CacheContractFixture):
    def test_payload_is_bound_to_exact_split_semantics_and_catalog_membership(self):
        contract = self.contract()
        mutations = {
            "outside ordered split": lambda groups: groups[0].update(
                sample_id="foreign_session"),
            "target/scenario": lambda groups: groups[0].update(
                scenario_type="holdout_only"),
            "outside catalog": lambda groups: groups[0]["rows"][0].update(
                pid="foreign_product"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                groups = self.groups()
                mutate(groups)
                with self.assertRaisesRegex(provenance.CacheCorrupt, message):
                    provenance.write_cache(
                        self.root / f"semantic-{message}.jsonl", groups, contract)

        with self.assertRaisesRegex(provenance.CacheCorrupt,
                                    "exactly match ordered split"):
            provenance.write_cache(
                self.root / "missing-session.jsonl", self.groups()[:1], contract)
        with self.assertRaisesRegex(provenance.CacheCorrupt,
                                    "exactly match ordered split"):
            provenance.write_cache(
                self.root / "wrong-order.jsonl", list(reversed(self.groups())), contract)

    def test_manifest_target_counts_are_session_level(self):
        groups = self.groups()
        extra = json.loads(json.dumps(groups[0]))
        extra["turn"] = 2
        manifest = provenance.write_cache(
            self.root / "counts.jsonl", [groups[0], extra, groups[1]], self.contract()
        )
        self.assertEqual(manifest["counts"]["groups"], 3)
        self.assertEqual(manifest["counts"]["sessions"], 2)
        self.assertEqual(manifest["counts"]["targets"], 2)

    def test_corruption_after_group_one_is_checked(self):
        path = self.root / "cache.jsonl"
        contract = self.contract()
        provenance.write_cache(path, self.groups(), contract)
        stored = self._stored(path)
        stored[1]["features"] = ["first", "wrong"]
        self._rewrite_payload(path, stored)
        with self.assertRaisesRegex(provenance.CacheCorrupt, "group 2 feature list"):
            provenance.load_cache(path, contract)

    def test_every_row_vector_dimension_is_checked(self):
        path = self.root / "cache.jsonl"
        contract = self.contract()
        provenance.write_cache(path, self.groups(), contract)
        stored = self._stored(path)
        stored[1]["rows"][1]["x"] = [0.4]
        self._rewrite_payload(path, stored)
        with self.assertRaisesRegex(provenance.CacheCorrupt, "group 2 row 2 vector width"):
            provenance.load_cache(path, contract)

    def test_unique_ids_one_target_finite_values_and_scores_are_enforced(self):
        contract = self.contract()
        mutations = {
            "duplicate pid": lambda rows: rows[0]["rows"][1].update(pid="p1"),
            "target rows": lambda rows: rows[0]["rows"][0].update(y=1),
            "not finite": lambda rows: rows[0]["rows"][0]["x"].__setitem__(0, float("inf")),
            "base score": lambda rows: rows[0]["rows"][0].update(s="high"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                groups = self.groups()
                mutate(groups)
                with self.assertRaisesRegex(provenance.CacheCorrupt, message):
                    provenance.write_cache(self.root / f"{message}.jsonl", groups, contract)

    def test_group_provenance_and_manifest_counts_are_enforced(self):
        contract = self.contract()
        path = self.root / "cache.jsonl"
        provenance.write_cache(path, self.groups(), contract)
        stored = self._stored(path)
        stored[1]["_provenance"] = "wrong"
        self._rewrite_payload(path, stored)
        with self.assertRaisesRegex(provenance.CacheCorrupt, "group 2 provenance"):
            provenance.load_cache(path, contract)

        provenance.write_cache(path, self.groups(), contract)
        sidecar = provenance.manifest_path(path)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        manifest["counts"]["rows"] += 1
        sidecar.write_text(provenance.canonical_json(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(provenance.CacheCorrupt, "counts"):
            provenance.load_cache(path, contract)


class StalenessTest(CacheContractFixture):
    def test_source_config_depth_and_ordered_split_changes_are_stale(self):
        path = self.root / "cache.jsonl"
        original = self.contract()
        provenance.write_cache(path, self.groups(), original)

        changed_split = self._file(
            "reordered.jsonl",
            '{"sample_id":"s2","scenario_type":"browsing",'
            '"ground_truth":{"parent_asin":"p3"}}\n'
            '{"sample_id":"s1","scenario_type":"buying",'
            '"ground_truth":{"parent_asin":"p2"}}\n',
        )
        candidates = [
            self.contract(depth=3),
            self.contract(env={"P_PROBE": "9"}),
            self.contract(split=changed_split),
        ]
        self.source_a.write_text("VALUE = 99\n", encoding="utf-8")
        candidates.append(self.contract())
        for expected in candidates:
            with self.subTest(expected=provenance.contract_sha256(expected)):
                with self.assertRaisesRegex(provenance.CacheStale,
                                            "cache provenance mismatch"):
                    provenance.load_cache(path, expected)

    def test_legacy_jsonl_without_manifest_is_rejected(self):
        path = self.root / "legacy.jsonl"
        path.write_text(json.dumps(self.groups()[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(provenance.CacheStale, "legacy/unprovenanced"):
            provenance.load_cache(path, self.contract())


class SnapshotCacheIntegrationTest(unittest.TestCase):
    def test_regeneration_requires_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            split = root / "split.jsonl"
            catalog.write_text(
                '{"parent_asin":"a"}\n{"parent_asin":"b"}\n',
                encoding="utf-8",
            )
            split.write_text(
                '{"sample_id":"s","scenario_type":"buying",'
                '"ground_truth":{"parent_asin":"b"}}\n',
                encoding="utf-8",
            )
            cache_pattern = str(root / "snapshots_%s.jsonl")
            group = snapshot_record(
                sample_id="s", scenario_type="buying", turn=1, target="b",
                message="fixture", features=FEATURES,
                ranked=[(2.0, "a"), (1.0, "b")],
                vector_for=lambda pid: ([0.0] if pid == "a" else [1.0])
                * len(FEATURES),
                catalog_ids={"a", "b"}, candidate_pool={"a", "b"},
                compatibility_depth=DEPTH, compatibility_scores=True,
            )
            with mock.patch.object(snapshot_mrr, "CACHE", cache_pattern), \
                    mock.patch.dict(snapshot_mrr.SPLITS, {"fixture": str(split)}, clear=True), \
                    mock.patch.object(snapshot_mrr, "collect", return_value=[group]) as collect:
                with self.assertRaisesRegex(SystemExit, "pass --rebuild"):
                    snapshot_mrr.cached("fixture", False, str(catalog), DEPTH)
                collect.assert_not_called()

                self.assertEqual(snapshot_mrr.cached(
                    "fixture", True, str(catalog), DEPTH), [group])
                collect.assert_called_once()
                self.assertTrue(provenance.manifest_path(
                    Path(cache_pattern % "fixture")).exists())
                self.assertEqual(snapshot_mrr.cached(
                    "fixture", False, str(catalog), DEPTH), [group])


if __name__ == "__main__":
    unittest.main()
