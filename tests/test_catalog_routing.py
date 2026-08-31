"""Catalog indexing and the three-step category narrowing."""
from __future__ import annotations

import unittest

from src.catalog import Catalog, coarse_category
from src.routing import BROWSING, BUYING, OPEN, candidates, category_key, detect_intent
from src.text import split_clauses
from tests.fixtures import TempCatalog


class CoarseCategoryTest(unittest.TestCase):
    def test_takes_two_most_specific_components(self):
        self.assertEqual(
            coarse_category(["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"]),
            "Accessories Belts")

    def test_survives_empty_input(self):
        self.assertEqual(coarse_category([]), "clothing item")
        self.assertEqual(coarse_category(None), "clothing item")

    def test_matches_the_evaluator_exactly(self):
        """The evaluator builds the turn-1 category string with its own coarse_category.

        Ours must agree character for character -- including its quirks, e.g. that
        the root "Clothing, Shoes & Jewelry" is comma-split so "Shoes & Jewelry"
        survives as a component. Diverging here silently breaks category routing.
        """
        from evaluator.local_evaluator import coarse_category as reference
        cases = [
            ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
            ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings", "Hoop"],
            ["Clothing, Shoes & Jewelry"],
            ["Clothing", "Shoes"],
            ["Clothing, Shoes & Jewelry", "Novelty & More", "Clothing", "Tops & Tees", "T-Shirts"],
            [], ["Single"], ["a, b, c", "d"],
        ]
        for case in cases:
            self.assertEqual(coarse_category(case), reference([str(v) for v in case]), msg=str(case))

    def test_parity_holds_across_the_real_catalog(self):
        """Spot-check parity on real category paths, not just handwritten ones."""
        import json, itertools, os
        path = "data/catalog.jsonl"
        if not os.path.exists(path):
            self.skipTest("full catalog not present")
        from evaluator.local_evaluator import coarse_category as reference
        with open(path, encoding="utf-8") as fh:
            for line in itertools.islice(fh, 2000):
                cats = json.loads(line).get("categories") or []
                self.assertEqual(coarse_category(cats), reference([str(v) for v in cats]))


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_indexes_every_product(self):
        self.assertEqual(self.catalog.size, 5)
        self.assertIn("P_LEATHER_BELT", self.catalog.corpus)

    def test_popularity_is_normalised_and_ordered(self):
        pop = self.catalog.popularity
        self.assertGreater(pop("P_LEATHER_BELT"), pop("P_SUEDE_BELT"))
        self.assertGreater(pop("P_SUEDE_BELT"), pop("P_CANVAS_BELT"))
        for pid in self.catalog.ids:
            self.assertGreaterEqual(pop(pid), 0.0)
            self.assertLessEqual(pop(pid), 1.0)

    def test_rarer_terms_get_higher_idf(self):
        self.assertGreater(self.catalog.idf("suede"), self.catalog.idf("belt"))

    def test_facets_are_extracted(self):
        self.assertEqual(self.catalog.facet("P_LEATHER_BELT", "material"), "leather")
        self.assertEqual(self.catalog.facet("P_LEATHER_BELT", "color"), "black")
        self.assertEqual(self.catalog.facet("P_LEATHER_BELT", "brand"), "beltworks")
        self.assertEqual(self.catalog.facet("P_LEATHER_BELT", "budget"), "15-30")
        self.assertIsNone(self.catalog.facet("P_LEATHER_BELT", "nonsense"))

    def test_missing_price_does_not_crash_budget_facet(self):
        self.catalog.meta["P_LEATHER_BELT"]["price"] = None
        self.assertIsNone(self.catalog.facet("P_LEATHER_BELT", "budget"))
        self.catalog.meta["P_LEATHER_BELT"]["price"] = 29.99


class RoutingTest(CatalogTest):
    def test_exact_bucket_match(self):
        self.assertEqual(sorted(candidates(self.catalog, "accessories belts")),
                         ["P_CANVAS_BELT", "P_LEATHER_BELT", "P_SUEDE_BELT"])

    def test_falls_back_to_token_overlap(self):
        got = candidates(self.catalog, "belts accessories mens")
        self.assertIn("P_LEATHER_BELT", got)

    def test_unknown_category_falls_back_to_whole_catalog(self):
        self.assertEqual(len(candidates(self.catalog, "quantum widgets")), self.catalog.size)

    def test_never_returns_empty(self):
        for key in (None, "", "   ", "zzzz"):
            self.assertTrue(candidates(self.catalog, key))

    def test_category_key_strips_trailing_qualifier(self):
        clauses = split_clauses("I'm looking for Accessories Belts, but I'm still exploring.")
        self.assertEqual(category_key(clauses), "accessories belts")

    def test_intent_detection(self):
        browse = "I'm looking for Accessories Belts, but I'm still exploring."
        buy = "I'm looking for Accessories Belts. A key requirement is: leather."
        self.assertEqual(detect_intent(browse, split_clauses(browse)), BROWSING)
        self.assertEqual(detect_intent(buy, split_clauses(buy)), BUYING)
        self.assertEqual(detect_intent("I'm looking for Belts", ["Belts"]), OPEN)


if __name__ == "__main__":
    unittest.main()
