"""Synthetic controls for the V6 catalog-only MNN shelf map (Candidate B).

Every test builds tiny synthetic catalogs as JSONL in a temp directory;
the real catalog is never touched. Controls B1-B9 pin the registered
formula and filters; T22 pins the transform input shape (category phrase
only); T24 pins the manifest's B1 per-rule exclusion counts.
"""
from __future__ import annotations

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

from src.shelf_mnn import build_transform
from tools.build_v6_mnn import build

PID = "parent_" + "asin"  # catalog id field; written so ids never survive
ROOT = "Clothing, Shoes & Jewelry"


def make_product(pid, parts, title="", description="", store=""):
    return {
        PID: pid,
        "title": title,
        "features": [],
        "details": {},
        "description": description,
        "store": store,
        "categories": [ROOT, *parts],
    }


def shelf_products(parts, prefix, n, title_tokens=(), description="", store=""):
    """n products on one shelf; each carries a unique df-1 filler token."""
    products = []
    for i in range(n):
        title = " ".join([*title_tokens, f"{prefix}only{i}"])
        products.append(make_product(f"{prefix}{i:04d}", parts, title=title,
                                     description=description, store=store))
    return products


def filler_shelves(count, start=0, n=10):
    """count ordinary 10-product shelves with distinct heads and tokens."""
    products = []
    for k in range(count):
        i = start + k
        products.extend(shelf_products(["Filler", f"Head{i}"], f"f{i}", n,
                                       (f"filltok{i}",)))
    return products


class MnnControls(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_catalog(self, name, products):
        path = self.dir / name
        with path.open("w", encoding="utf-8") as handle:
            for product in products:
                handle.write(json.dumps(product) + "\n")
        return path

    def build_mapping(self, name, products):
        result = build(str(self.write_catalog(name, products)))
        return result["asset"]["mapping"]

    # ---------------------------------------------------------- B1
    def test_b1_concentrated_term_maps_to_its_shelf_head(self):
        products = shelf_products(["Alpha", "Sneakers"], "a", 10,
                                  ("zephyr", "sneaker"))
        for parts, prefix, token in ((["Beta", "Hats"], "b", "brim"),
                                     (["Gamma", "Scarves"], "g", "soft"),
                                     (["Delta", "Gloves"], "d", "warm")):
            # Marketing prose and brand text name the concentrated term;
            # both fields are excluded, so they must not move it.
            products += shelf_products(parts, prefix, 10, (token,),
                                       description="zephyr marketing prose",
                                       store="Zephyr Outfitters")
        mapping = self.build_mapping("b1.jsonl", products)
        self.assertEqual(mapping.get("zephyr"), ["sneakers"])
        # stem-equal to the head: never an alias of it.
        self.assertNotIn("sneaker", mapping)

    # ---------------------------------------------------------- B2
    def test_b2_uniformly_spread_junk_maps_nowhere(self):
        products = []
        for i, parts in enumerate((["Alpha", "Sneakers"], ["Beta", "Hats"],
                                   ["Gamma", "Scarves"], ["Delta", "Gloves"])):
            products += shelf_products(parts, f"s{i}", 10,
                                       ("everywhere", f"tok{i}"))
        # Below the document floor: present in only three products.
        for i in range(3):
            products[i]["title"] += " rareterm"
        mapping = self.build_mapping("b2.jsonl", products)
        self.assertNotIn("everywhere", mapping)
        self.assertNotIn("rareterm", mapping)

    # ---------------------------------------------------------- B3
    def test_b3_equal_two_shelf_collision_stays_ambiguous(self):
        # Both shelves' products carry both head tokens, so the two head
        # vectors coincide and the collision term scores 1.0 against each.
        products = shelf_products(["Sport", "Alpha"], "a", 10,
                                  ("duo", "alpha", "beta"))
        products += shelf_products(["Sport", "Beta"], "b", 10,
                                   ("duo", "alpha", "beta"))
        products += filler_shelves(6, start=10)
        mapping = self.build_mapping("b3.jsonl", products)
        self.assertEqual(mapping.get("duo"), ["alpha", "beta"])
        self.assertEqual(len(mapping["duo"]), 2)  # never forced to one

    # ---------------------------------------------------------- B4
    def test_b4_large_shelf_cannot_win_by_volume(self):
        # Raw document counts favour the big shelf (15 > 10); normalized
        # presence favours the small one (1.0 > 0.5).
        mega = shelf_products(["Mega", "Sneakers"], "m", 30)
        for product in mega[:15]:
            product["title"] = "zoom " + product["title"]
        products = mega
        products += shelf_products(["Tiny", "Buckles"], "t", 10, ("zoom",))
        products += filler_shelves(6, start=20)
        mapping = self.build_mapping("b4.jsonl", products)
        self.assertEqual(mapping.get("zoom"), ["buckles"])
        self.assertNotIn("sneakers", mapping.get("zoom", []))

    # ---------------------------------------------------------- B5
    def test_b5_id_renaming_leaves_payload_invariant(self):
        products = shelf_products(["Alpha", "Sneakers"], "a", 10, ("zephyr",))
        products += filler_shelves(3, start=30)
        first = build(str(self.write_catalog("b5a.jsonl", products)))
        renamed = []
        for i, product in enumerate(products):
            clone = dict(product)
            clone[PID] = f"renamed{i:05d}"
            renamed.append(clone)
        second = build(str(self.write_catalog("b5b.jsonl", renamed)))
        self.assertEqual(first["asset"]["mapping"], second["asset"]["mapping"])
        self.assertEqual(first["asset"]["payload_sha256"],
                         second["asset"]["payload_sha256"])

    # ---------------------------------------------------------- B6
    def test_b6_shuffled_input_order_leaves_payload_hash_unchanged(self):
        products = shelf_products(["Alpha", "Sneakers"], "a", 10, ("zephyr",))
        products += filler_shelves(3, start=40)
        first = build(str(self.write_catalog("b6a.jsonl", products)))
        shuffled = list(products)
        random.Random(2026).shuffle(shuffled)
        second = build(str(self.write_catalog("b6b.jsonl", shuffled)))
        self.assertEqual(first["asset"]["mapping"], second["asset"]["mapping"])
        self.assertEqual(first["asset"]["payload_sha256"],
                         second["asset"]["payload_sha256"])

    # ---------------------------------------------------------- B7
    def test_b7_no_product_id_survives_in_the_asset(self):
        products = shelf_products(["Alpha", "Sneakers"], "a", 10, ("zephyr",))
        products += filler_shelves(3, start=50)
        result = build(str(self.write_catalog("b7.jsonl", products)))
        blob = json.dumps(result["asset"]) + json.dumps(result["manifest"])
        for product in products:
            self.assertNotIn(product[PID], blob)
        self.assertNotIn(PID, blob)

    # ---------------------------------------------------------- B8
    def test_b8_builder_statically_imports_no_forbidden_module(self):
        source = Path("tools/build_v6_mnn.py").read_text(encoding="utf-8")
        forbidden = {"evaluator", "adversarial", "analysis", "split"}
        components = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    components.update(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                components.update(node.module.split("."))
        self.assertFalse(components & forbidden)

    # ---------------------------------------------------------- B9
    def test_b9_small_shelf_concentration_maps_nothing_large_one_does(self):
        small = shelf_products(["Tiny", "Buckles"], "t", 3, ("microterm",))
        small += filler_shelves(8, start=60)
        mapping = self.build_mapping("b9_small.jsonl", small)
        self.assertNotIn("microterm", mapping)

        large = shelf_products(["Tiny", "Buckles"], "t", 30, ("microterm",))
        large += filler_shelves(8, start=60)
        mapping = self.build_mapping("b9_large.jsonl", large)
        self.assertEqual(mapping.get("microterm"), ["buckles"])

    # ---------------------------------------------------------- T22
    def test_t22_transform_consumes_only_the_category_phrase(self):
        transform = build_transform(
            {"zephyr": ["sneakers"], "duo": ["alpha", "beta"],
             "tri": ["a", "b", "c"]},
            "0" * 64)
        self.assertEqual(transform.mode, "mnn")
        self.assertEqual(transform.payload_sha256, "0" * 64)
        # Single head: residual tokens of the phrase follow the head.
        self.assertEqual(transform.transform("Shiny Zephyr"),
                         ["sneakers shiny"])
        # Two heads emit both; more than two emits nothing.
        self.assertEqual(transform.transform("duo"), ["alpha", "beta"])
        self.assertEqual(transform.transform("tri"), [])
        self.assertEqual(transform.transform("unrelated words"), [])
        self.assertEqual(transform.transform(""), [])

    # ---------------------------------------------------------- T24
    def test_t24_manifest_carries_b1_per_rule_exclusion_counts(self):
        products = shelf_products(["Size", "42"], "j", 10)       # junk head
        products += shelf_products(["Cool", "DVD"], "u", 10)     # uppercase
        products += shelf_products(["Best", "Seller"], "s", 10)  # banner
        products += shelf_products(["Worn", "Shoes"], "w", 5)    # floor
        products += shelf_products(["Alpha", "Sneakers"], "a", 10, ("zephyr",))
        products += filler_shelves(3, start=70)
        result = build(str(self.write_catalog("t24.jsonl", products)))
        counts = result["manifest"]["b1_exclusion_counts"]
        for key in ("considered", "junk_token", "uppercase_fragment",
                    "empty_after_normalization", "banner_only",
                    "product_floor", "retained"):
            self.assertIn(key, counts)
        self.assertEqual(counts["junk_token"], 1)
        self.assertEqual(counts["uppercase_fragment"], 1)
        self.assertEqual(counts["banner_only"], 1)
        self.assertEqual(counts["product_floor"], 1)
        self.assertEqual(result["manifest"]["shelf_size_exclusion_count"], 1)
        self.assertEqual(result["manifest"]["terms"]["considered"],
                         result["asset"]["support"]["terms_considered"])
        self.assertEqual(result["manifest"]["terms"]["retained"],
                         len(result["asset"]["mapping"]))
        for name in ("catalog_sha256", "builder_source_sha256",
                     "input_sha256", "payload_sha256"):
            self.assertIn(name, result["manifest"]["hashes"])
        self.assertEqual(result["asset"]["payload_sha256"],
                         result["manifest"]["hashes"]["payload_sha256"])


if __name__ == "__main__":
    unittest.main()
