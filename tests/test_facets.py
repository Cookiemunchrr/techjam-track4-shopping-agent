"""Facet equality: is the product the thing they named, or does it just mention it?

The phrase term already rewards "leather" appearing anywhere in a product's blob,
which a canvas bag with a leather trim satisfies. This asks the narrower question,
and it is a question shoppers care about: someone who says "leather" wants a
leather one, not one with a leather tag.

Only material and colour, because they are the only free-text attributes this
catalog resolves to a single value per product. The value comes from
`Catalog.facet`, which reads the first match in the blob -- a good guess at the
primary material and not a fact, which is why the weight sits below the phrase
term it refines rather than replaces.
"""
from __future__ import annotations

import unittest

from src.catalog import Catalog
from src.scoring import Scorer, Weights
from src.state import DialogState
from tests.fixtures import TempCatalog


class ExtractionTest(unittest.TestCase):
    def _facets(self, *messages):
        state = DialogState()
        for turn, message in enumerate(messages, start=1):
            state.observe(message, turn, None)
        return state.facets()

    def test_reads_material_and_colour_from_constraint_text(self):
        found = self._facets("I want a belt. For that, what matters is: 100% Leather; black.")
        self.assertEqual(found.get("material"), "leather")
        self.assertEqual(found.get("color"), "black")

    def test_says_nothing_when_nothing_was_said(self):
        self.assertEqual(self._facets("I'm looking for Accessories Belts."), {})

    def test_a_refusal_is_not_a_preference(self):
        """"not polyester" must never become "wants polyester"."""
        found = self._facets("I want a shirt. I don't want polyester.")
        self.assertNotEqual(found.get("material"), "polyester")

    def test_the_most_recent_value_wins(self):
        found = self._facets("I want a belt. For that, what matters is: Leather.",
                             "Actually, what matters is: Suede.")
        self.assertEqual(found.get("material"), "suede",
                         "an overridden material was still being scored for")


class ScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.catalog = Catalog(cls._catalog.__enter__())
        cls.scorer = Scorer(cls.catalog)
        cls.belts = [pid for pid in cls.catalog.ids if "belt" in cls.catalog.corpus[pid]]

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def test_being_the_material_beats_mentioning_it(self):
        suede = self.scorer._facet("P_SUEDE_BELT", {"material": "suede"})
        leather = self.scorer._facet("P_LEATHER_BELT", {"material": "suede"})
        self.assertGreater(suede, leather)

    def test_every_matching_facet_counts(self):
        one = self.scorer._facet("P_LEATHER_BELT", {"material": "leather"})
        two = self.scorer._facet("P_LEATHER_BELT", {"material": "leather", "color": "black"})
        self.assertGreater(two, one)

    def test_it_never_penalises_a_product(self):
        """A facet the catalog could not read must not push a product down.

        Facet values are extracted by regex over a text blob; a miss is at least
        as often our failure to parse as the product's failure to match.
        """
        for pid in self.catalog.ids:
            self.assertGreaterEqual(self.scorer._facet(pid, {"material": "leather"}), 0.0)

    def test_it_is_a_preference_and_not_a_filter(self):
        weights = Weights()
        self.assertLess(weights.facet, weights.popularity,
                        "a facet mismatch can outvote everything the catalog knows")
        self.assertLess(weights.facet, weights.phrase,
                        "the refinement outweighs the signal it refines")

    def test_rank_drops_nothing(self):
        plain = sorted(pid for _, pid in self.scorer.rank(self.belts, []))
        faceted = sorted(pid for _, pid in
                         self.scorer.rank(self.belts, [], facets={"material": "suede"}))
        self.assertEqual(plain, faceted, "the facet bonus dropped candidates")


if __name__ == "__main__":
    unittest.main()


class BrandIsNotAFacetTest(unittest.TestCase):
    """A shop's name is not a claim about what it sells.

    This catalog is full of brands that read like facet values -- Sabrina Silver,
    White Mountain, Pink Queen, Yellow Box, Claddagh Gold, Coastal Blue, Fools
    Gold T-shirts. `store` is in SEARCH_FIELDS, correctly, because a shopper who
    names a brand should find it by BM25; it was also reaching the loose facet
    scan, where it made 32 of the 50,000 shipped products claim a colour or a
    material on the strength of the shop's name alone.

    It cuts both ways: such a product is offered when that value is asked for and
    penalised when it is refused. "Fools Gold T-shirts" are not gold.
    """

    def _catalog(self, store, title="Plain Tee", desc="A comfortable tee."):
        import json, tempfile
        from pathlib import Path
        from src.catalog import Catalog
        product = {
            "parent_asin": "P_TEE", "title": title, "features": ["Machine wash"],
            "description": [desc], "price": 10.0,
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts"],
            "details": {"Department": "Mens"}, "average_rating": 4.0,
            "rating_number": 10, "store": store,
        }
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "catalog.jsonl"
        path.write_text(json.dumps(product) + "\n", encoding="utf-8")
        return Catalog(str(path))

    def test_a_brand_name_does_not_make_the_product_that_colour(self):
        catalog = self._catalog("Fools Gold T-shirts")
        self.assertEqual(catalog.facet_values("P_TEE", "color", loose=True), ())
        self.assertIsNone(catalog.facet("P_TEE", "color"))

    def test_a_brand_name_does_not_make_the_product_that_material(self):
        """Smartwool socks are not wool, and Suedeco belts are not always suede."""
        catalog = self._catalog("Smartwool")
        self.assertEqual(catalog.facet_values("P_TEE", "material", loose=True), ())

    def test_the_brand_is_still_searchable(self):
        """The exclusion is from the facet scan only, never from the BM25 corpus."""
        catalog = self._catalog("Fools Gold T-shirts")
        self.assertIn("gold", catalog.corpus["P_TEE"])
        self.assertIn("gold", catalog.tf["P_TEE"])

    def test_the_product_s_own_words_still_count(self):
        """The exclusion must not cost a facet the product actually claims."""
        catalog = self._catalog("Fools Gold T-shirts", title="Gold Foil Tee")
        self.assertIn("gold", catalog.facet_values("P_TEE", "color"))
