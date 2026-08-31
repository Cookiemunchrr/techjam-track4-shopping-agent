"""Budget honouring: correct product behaviour that this harness cannot reward.

A shopper who says "around $40" and is shown $200 boots has been ignored, and no
shopping agent that ignores a stated budget is finished. So the term exists.

It also scores exactly nothing on the public set, and the reason is worth stating
precisely because it decides how the term was built. `local_evaluator.intent_card`
appends `budget around $price` *last*, then truncates the list to `cleaned[:4]`;
product features and details always fill those four slots first. Verified over all
200 public sessions: a budget is disclosed in zero of them, which the agent's own
learned answerability table corroborates independently (budget: asked 5, answered 0).

That has a consequence for how the weight and the tolerance were chosen. There is
nothing here to fit them on, and fitting them on generated sessions would be worse
than not fitting them at all: the simulator sets the budget to the target's *exact*
price, so any tolerance optimised against it collapses toward an exact-price match
-- an answer-key detector wearing the costume of a product feature. Both constants
come from what a budget means to a shopper instead.

These tests therefore pin *behaviour*, not score: proportional tolerance, a ceiling
that is satisfied from below, no penalty for a product the catalog has no price for,
and a preference that can always be outvoted by a better match.
"""
from __future__ import annotations

import math
import unittest

from src.catalog import Catalog
from src.scoring import BUDGET_CLIP, Scorer, Weights
from src.state import Budget, DialogState
from tests.fixtures import TempCatalog


class ParsingTest(unittest.TestCase):
    def _budget(self, message: str):
        state = DialogState()
        state.observe("I'm looking for a belt. " + message, 1, None)
        return state.budget()

    def test_reads_a_stated_figure(self):
        self.assertAlmostEqual(self._budget("My budget is around $40.").amount, 40.0)

    def test_a_range_collapses_to_its_midpoint(self):
        self.assertAlmostEqual(self._budget("My budget is between $20 and $40.").amount, 30.0)

    def test_under_is_a_ceiling_and_around_is_a_target(self):
        self.assertTrue(self._budget("I want to stay under $30.").cap)
        self.assertFalse(self._budget("My budget is around $30.").cap)

    def test_numbers_in_feature_text_are_not_prices(self):
        """The single most likely way this feature turns into a bug.

        Product metadata is full of numbers -- dimensions, weights, counts -- and
        reading one as a budget would silently reprice the whole ranking. Only a
        slot classified as `budget` is ever consulted.
        """
        self.assertIsNone(self._budget("Product Dimensions: 7 x 3 x 0.5 inches; 8 Ounces."))
        self.assertIsNone(self._budget("It should have 4 pockets."))

    def test_no_budget_means_none(self):
        self.assertIsNone(self._budget("I'd like something in leather."))

    def test_a_decimal_point_is_not_a_sentence_boundary(self):
        """Splitting "$29.99" into "$29" and "99" turns a price into another price."""
        self.assertAlmostEqual(self._budget("Budget around $29.99.").amount, 29.99)

    def test_a_percentage_is_never_a_price(self):
        """Composition strings are the catalog's most common numbers by far."""
        self.assertIsNone(self._budget("What matters is: 65%Polyester30%Viscose5%Spandex."))
        self.assertIsNone(self._budget("The bit I actually care about is 100% Leather."))


class MisclassificationTest(unittest.TestCase):
    """`budget` is in DialogState.SETTLED, so a misfiled slot is worse than a wasted
    one -- it also convinces the agent it has already asked what the customer can
    afford. Found by instrumenting the scaffold adversarial axis, where 19 of 52
    stored slots were being filed as budgets against 0 on the clean set.
    """

    def _classify(self, text: str) -> str:
        from src.state import classify
        return classify(text)

    def test_a_brand_that_reads_like_a_comparison_is_not_a_budget(self):
        """"Under" is a brand in this catalog, not only a preposition."""
        self.assertNotEqual(self._classify("Under Armour"), "budget")

    def test_about_is_not_money_without_money_next_to_it(self):
        """Natural phrasings of a requirement are full of "about"."""
        self.assertNotEqual(self._classify("The bit I actually care about is Leather"),
                            "budget")
        self.assertNotEqual(self._classify("what I care about is the fit"), "budget")

    def test_the_words_that_mean_money_on_their_own_still_do(self):
        for text in ("my budget is tight", "what's the price", "under $30",
                     "around 40 dollars", "$25"):
            self.assertEqual(self._classify(text), "budget", text)


class ScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.catalog = Catalog(cls._catalog.__enter__())
        cls.scorer = Scorer(cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def test_tolerance_is_proportional_not_absolute(self):
        """$15-against-$20 is the same miss as $150-against-$200.

        A shopper's sense of "too expensive" scales with the figure they named.
        Absolute distance would make every cheap category tolerant and every
        expensive one unusable.
        """
        cheap = self.scorer._budget("P_CANVAS_BELT", Budget(12.0 * 1.5))
        dear = self.scorer._budget("P_SILK_SCARF", Budget(60.0 * 1.5))
        self.assertAlmostEqual(cheap, dear, places=6)

    def test_a_ceiling_is_satisfied_from_below(self):
        under = Budget(30.0, cap=True)
        self.assertEqual(self.scorer._budget("P_CANVAS_BELT", under), 0.0)   # $12
        self.assertLess(self.scorer._budget("P_SUEDE_BELT", under), 0.0)     # $45

    def test_a_target_is_missed_from_both_sides(self):
        around = Budget(30.0)
        self.assertLess(self.scorer._budget("P_CANVAS_BELT", around), 0.0)   # $12
        self.assertLess(self.scorer._budget("P_SUEDE_BELT", around), 0.0)    # $45

    def test_a_product_with_no_price_is_neutral_never_penalised(self):
        """78.9% of the real catalog has no price.

        Penalising them would hide four fifths of every shelf the moment a budget
        was mentioned -- a worse answer than ignoring the budget entirely.
        """
        priceless = dict(self.catalog.meta["P_LEATHER_BELT"])
        self.catalog.meta["P_LEATHER_BELT"] = {**priceless, "price": None}
        try:
            self.assertEqual(self.scorer._budget("P_LEATHER_BELT", Budget(5.0)), 0.0)
        finally:
            self.catalog.meta["P_LEATHER_BELT"] = priceless

    def test_the_miss_is_bounded(self):
        self.assertGreaterEqual(self.scorer._budget("P_SILK_SCARF", Budget(0.5)), -BUDGET_CLIP)
        self.assertAlmostEqual(BUDGET_CLIP, math.log(4.0), places=3)

    def test_being_at_budget_is_never_a_bonus(self):
        """Proximity may not earn rank, or the agent pushes the cheapest thing on
        the shelf at a shopper who only meant "not more than about this"."""
        for pid in self.catalog.ids:
            self.assertLessEqual(self.scorer._budget(pid, Budget(30.0)), 0.0)

    def test_it_is_a_preference_and_not_a_filter(self):
        """A better match that costs more must still be able to win.

        The repo's standing rule -- a stated constraint is evidence, never an AND
        -- applies here too. Verified against the strongest competing term rather
        than asserted in a comment.
        """
        weights = Weights()
        worst_case = weights.budget * BUDGET_CLIP
        self.assertLess(worst_case, weights.popularity,
                        "a price miss can outvote everything the catalog knows")

    def test_rank_honours_a_budget_without_a_budget_being_required(self):
        belts = [pid for pid in self.catalog.ids if "belt" in self.catalog.corpus[pid]]
        plain = [pid for _, pid in self.scorer.rank(belts, [])]
        capped = [pid for _, pid in self.scorer.rank(belts, [], budget=Budget(15.0, cap=True))]
        self.assertEqual(sorted(plain), sorted(capped), "the budget dropped candidates")
        self.assertLess(capped.index("P_CANVAS_BELT"), plain.index("P_CANVAS_BELT"),
                        "the only belt under the cap did not move up")


if __name__ == "__main__":
    unittest.main()
