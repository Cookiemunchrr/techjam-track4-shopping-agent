"""Tests for the V6 blind shelf-alias asset and its shared mechanism.

Synthetic mappings exercise the MappingTransform contract; the shipped asset is
checked for hash integrity, manifest completeness (T24), prohibited fields, and
one live transform over a generated alias.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.shelf_aliases import build_transform
from src.shelf_transform import MappingTransform, payload_hash

ROOT = Path(__file__).resolve().parent.parent
ASSET_PATH = ROOT / "assets" / "v6_shelf_aliases.json"
MANIFEST_PATH = ROOT / "assets" / "v6_shelf_aliases.manifest.json"

PROHIBITED = ("parent_asin", "sample_id", "session_id", "scenario",
              "target", "answer", "ground_truth")

B1_RULES = ("considered", "junk_token", "uppercase_fragment",
            "empty_after_normalization", "banner_only", "product_floor",
            "retained")


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


class TestMappingTransform(unittest.TestCase):
    """The frozen mechanism, over small synthetic mappings."""

    def test_longest_span_wins(self):
        t = MappingTransform("aliases",
                             {"shoes": ["footwearx"],
                              "running shoes": ["sneakerx"]},
                             "x")
        self.assertEqual(t.transform("red running shoes"),
                         ["sneakerx red"])

    def test_rightmost_span_breaks_ties(self):
        t = MappingTransform("aliases",
                             {"walkers": ["heada"], "talkers": ["headb"]},
                             "x")
        self.assertEqual(t.transform("walkers talkers"), ["headb walkers"])

    def test_two_head_hedge_emits_both(self):
        t = MappingTransform("aliases", {"kicks": ["sneakers", "shoes"]}, "x")
        self.assertEqual(t.transform("kicks"), ["sneakers", "shoes"])

    def test_more_than_two_heads_emits_nothing(self):
        t = MappingTransform("aliases",
                             {"kicks": ["sneakers", "shoes", "footwear"]}, "x")
        self.assertEqual(t.transform("kicks"), [])

    def test_no_match_emits_nothing(self):
        t = MappingTransform("aliases", {"kicks": ["sneakers"]}, "x")
        self.assertEqual(t.transform("hiking footwear"), [])
        self.assertEqual(t.transform(""), [])


class TestBuilderCollisions(unittest.TestCase):
    """The builder's collision and drop rules, over a synthetic raw file."""

    def test_collision_dropped_alias_absent_from_asset(self):
        import tools.build_v6_aliases as builder

        heads_doc = {
            "eligible": {"sneakers": ["Shoes Sneakers"],
                         "shoes": ["Men Shoes"],
                         "boots": ["Shoes Boots"]},
            "counts": {"considered": 3, "junk_token": 0,
                       "uppercase_fragment": 0, "empty_after_normalization": 0,
                       "banner_only": 0, "product_floor": 0, "retained": 3},
        }
        raw = {
            # Canonical-identical: dropped by rule (b).
            "sneakers": ["sneakers",
                         # Equals a DIFFERENT canonical head token: dropped (e).
                         "shoes",
                         # Kept.
                         "runners"],
            # Claimed by three heads: drops entirely (e).
            "shoes": ["kicks"],
            "boots": ["kicks"],
        }
        # "kicks" claimed by only two heads here; add a third claimant.
        raw["sneakers"].append("kicks")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            heads_path = tmp / "heads.json"
            raw_path = tmp / "raw.json"
            asset_path = tmp / "asset.json"
            manifest_path = tmp / "manifest.json"
            audit_path = tmp / "audit-absent.json"  # isolation: no dispositions
            heads_path.write_text(json.dumps(heads_doc), encoding="utf-8")
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            original = (builder.HEADS_PATH, builder.RAW_PATH,
                        builder.ASSET_PATH, builder.MANIFEST_PATH,
                        builder.AUDIT_PATH)
            builder.HEADS_PATH = heads_path
            builder.RAW_PATH = raw_path
            builder.ASSET_PATH = asset_path
            builder.MANIFEST_PATH = manifest_path
            builder.AUDIT_PATH = audit_path
            try:
                builder.cmd_build()
            finally:
                (builder.HEADS_PATH, builder.RAW_PATH,
                 builder.ASSET_PATH, builder.MANIFEST_PATH,
                 builder.AUDIT_PATH) = original

            asset = json.loads(asset_path.read_text(encoding="utf-8"))
            mapping = asset["mapping"]
            # Collision-dropped aliases are absent.
            self.assertNotIn("shoes", mapping)
            self.assertNotIn("kicks", mapping)
            # The honest alias survives.
            self.assertEqual(mapping["runners"], ["sneakers"])
            # Drop accounting.
            dropped = asset["support"]["dropped"]
            self.assertEqual(dropped["canonical_identical"], 1)
            self.assertEqual(dropped["collision_other_head"], 1)
            self.assertEqual(dropped["collision_multi_head"], 3)
            self.assertEqual(asset["support"]["retained"], 1)
            # Hash verifies over the emitted mapping.
            self.assertEqual(asset["payload_sha256"], payload_hash(mapping))

    def test_audit_removals_drop_exactly_the_listed_aliases(self):
        import tools.build_v6_aliases as builder

        heads_doc = {
            "eligible": {"sneakers": ["Shoes Sneakers"],
                         "thongs": ["Underwear Thongs"]},
            "counts": {"considered": 2, "junk_token": 0,
                       "uppercase_fragment": 0, "empty_after_normalization": 0,
                       "banner_only": 0, "product_floor": 0, "retained": 2},
        }
        raw = {"sneakers": ["runners"], "thongs": ["strings", "tangas"]}
        audit = {"reviewer": "synthetic", "sampled": 3, "supported": 2,
                 "precision": 2 / 3, "wilson95": [0.2, 0.9],
                 "removals": [{"alias": "strings", "reason": "too generic"},
                              {"alias": "absent-alias", "reason": "not present"}]}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, doc in (("heads.json", heads_doc), ("raw.json", raw),
                              ("audit.json", audit)):
                (tmp / name).write_text(json.dumps(doc), encoding="utf-8")

            original = (builder.HEADS_PATH, builder.RAW_PATH,
                        builder.ASSET_PATH, builder.MANIFEST_PATH,
                        builder.AUDIT_PATH)
            builder.HEADS_PATH = tmp / "heads.json"
            builder.RAW_PATH = tmp / "raw.json"
            builder.ASSET_PATH = tmp / "asset.json"
            builder.MANIFEST_PATH = tmp / "manifest.json"
            builder.AUDIT_PATH = tmp / "audit.json"
            try:
                builder.cmd_build()
            finally:
                (builder.HEADS_PATH, builder.RAW_PATH,
                 builder.ASSET_PATH, builder.MANIFEST_PATH,
                 builder.AUDIT_PATH) = original

            asset = json.loads((tmp / "asset.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (tmp / "manifest.json").read_text(encoding="utf-8"))
            # The listed alias is removed; everything else is untouched.
            self.assertNotIn("strings", asset["mapping"])
            self.assertEqual(asset["mapping"]["runners"], ["sneakers"])
            self.assertEqual(asset["mapping"]["tangas"], ["thongs"])
            self.assertEqual(asset["support"]["dropped"]["audit_removal"], 1)
            self.assertEqual(asset["support"]["retained"], 2)
            # The manifest carries the audit reference and pre/post counts.
            record = manifest["audit"]
            self.assertEqual(record["reviewer"], "synthetic")
            self.assertEqual(record["counts"],
                             {"pre_removal": 3, "removed": 1,
                              "post_removal": 2})
            self.assertEqual([r["alias"] for r in record["removed"]],
                             ["strings"])
            self.assertEqual(record["removed"][0]["reason"], "too generic")


class TestShippedAsset(unittest.TestCase):
    """The real asset and manifest on disk."""

    @classmethod
    def setUpClass(cls):
        cls.asset = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_asset_payload_hash_verifies(self):
        self.assertEqual(self.asset["payload_sha256"],
                         payload_hash(self.asset["mapping"]))
        self.assertEqual(self.manifest["hashes"]["payload_sha256"],
                         self.asset["payload_sha256"])

    def test_manifest_carries_b1_exclusion_counts(self):
        counts = self.manifest["b1_exclusion_counts"]
        for rule in B1_RULES:
            self.assertIn(rule, counts)
        self.assertEqual(counts["retained"],
                         self.manifest["heads"]["retained"])
        self.assertEqual(counts["considered"],
                         self.manifest["heads"]["considered"])

    def test_no_prohibited_identifier_fields(self):
        for doc in (self.asset, self.manifest):
            for key in _walk_keys(doc):
                lowered = key.lower()
                for field in PROHIBITED:
                    self.assertNotIn(field, lowered)

    def test_real_asset_transforms_an_authored_alias(self):
        t = build_transform(self.asset["mapping"],
                            self.asset["payload_sha256"])
        # "runners" is an authored alias for the "sneakers" head.
        self.assertEqual(t.transform("women runners"), ["sneakers women"])

    def test_transform_receives_only_the_category_phrase(self):
        """T22 shape: a constraint sentence after the category never enters."""
        t = build_transform(self.asset["mapping"],
                            self.asset["payload_sha256"])
        message = ("runners. It has to be breathable, and I'd like it under "
                   "forty dollars if possible")
        category_phrase = message.split(".")[0]

        seen = []
        original = t.transform

        def spy(phrase):
            seen.append(phrase)
            return original(phrase)

        self.assertEqual(spy(category_phrase), ["sneakers"])
        for phrase in seen:
            self.assertNotIn("breathable", phrase)
            self.assertNotIn("forty", phrase)


if __name__ == "__main__":
    unittest.main()
