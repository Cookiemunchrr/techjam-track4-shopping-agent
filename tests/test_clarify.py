"""Proactive shelf clarification: the question the catalog cannot answer.

analysis/shelf_election_finding.json establishes that near-duplicate shelves are
not separable from product evidence -- the posterior over them never sharpens, and
shelf identity is worth +0.217 hit rate when known. src/clarify.py is the
consequence: ask the customer, because they are the only one who knows.

These tests pin the behaviour that makes the question worth asking -- it is closed,
it is asked once, the answer locks the shelf without discarding what the customer
already said, and a refusal leaves no trace.
"""
from __future__ import annotations

import unittest

from src.answerability import AnswerModel
from src.catalog import Catalog
from src.clarify import MIN_CANDIDATES, match, options, phrasing, should_ask
from src.elicitation import choose
from tests.fixtures import PROFILE, TempCatalog

TWINS = ["Jewelry Necklaces", "Necklaces Chains", "Accessories Necklaces"]


class OptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.catalog = Catalog(cls._catalog.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def test_only_offers_shelves_that_exist(self):
        real = sorted(self.catalog.buckets)[0]
        self.assertEqual(options(self.catalog, [real, "Not A Shelf"]), [real])

    def test_preserves_retrieval_order(self):
        shelves = sorted(self.catalog.buckets)
        self.assertEqual(options(self.catalog, shelves, limit=2), shelves[:2])

    def test_does_not_repeat_a_shelf(self):
        real = sorted(self.catalog.buckets)[0]
        self.assertEqual(options(self.catalog, [real, real, real]), [real])

    def test_is_silent_when_there_is_nothing_to_disambiguate(self):
        real = sorted(self.catalog.buckets)[0]
        self.assertFalse(should_ask(self.catalog, [real], already_asked=False),
                         "asked which shelf when only one was in contention")
        self.assertFalse(should_ask(self.catalog, [], already_asked=False))

    def test_asks_at_most_once_per_session(self):
        shelves = sorted(self.catalog.buckets)
        self.assertGreaterEqual(len(shelves), MIN_CANDIDATES)
        self.assertTrue(should_ask(self.catalog, shelves, already_asked=False))
        self.assertFalse(should_ask(self.catalog, shelves, already_asked=True),
                         "re-asked a question the customer has already answered")


class PhrasingTest(unittest.TestCase):
    def test_names_every_option(self):
        text = phrasing(TWINS).lower()
        for name in ("jewelry necklaces", "necklaces chains", "accessories necklaces"):
            self.assertIn(name.split()[-2] if " " in name else name, text)

    def test_reads_as_a_closed_question(self):
        self.assertTrue(phrasing(TWINS).rstrip().endswith("?"))

    def test_drops_the_repeated_parent_from_a_taxonomy_name(self):
        self.assertIn("baseball caps", phrasing(["Hats & Caps Baseball Caps",
                                                 "Hats & Caps Sun Hats"]))
        self.assertNotIn("hats & caps baseball caps",
                         phrasing(["Hats & Caps Baseball Caps", "Hats & Caps Sun Hats"]))


class MatchTest(unittest.TestCase):
    def test_recognises_a_choice_by_its_distinctive_words(self):
        self.assertEqual(match("chains please", TWINS), "Necklaces Chains")
        self.assertEqual(match("the jewelry ones", TWINS), "Jewelry Necklaces")

    def test_ignores_the_word_every_option_shares(self):
        """"necklaces" identifies nothing here -- it is why we had to ask."""
        self.assertIsNone(match("necklaces", TWINS),
                          "matched on a word common to every option")

    def test_a_refusal_is_not_a_choice(self):
        self.assertIsNone(match("I don't have an additional preference for category.", TWINS))
        self.assertIsNone(match("", TWINS))
        self.assertIsNone(match("chains please", []))


class CompetesLikeAnyQuestionTest(unittest.TestCase):
    """`category` earns its turn or loses it -- nothing special-cases it in."""

    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.catalog = Catalog(cls._catalog.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def _ranked(self):
        return [(1.0, pid) for pid in self.catalog.ids]

    def test_wins_when_it_is_the_most_valuable_question(self):
        model = AnswerModel()
        for _ in range(20):
            model.observe("category", True)
            model.observe("material", False)
        self.assertEqual(choose(self.catalog, self._ranked(), set(), model,
                                extra={"category": 0.9}), "category")

    def test_loses_once_the_customer_has_shown_they_cannot_answer_it(self):
        """The learned answerability model is what bounds the cost of asking.

        Against a customer who never answers it, the question stops being asked --
        no rule says so, it simply loses the comparison. Against one who does, it
        keeps winning. Same code, opposite policies, decided by the interlocutor.
        """
        model = AnswerModel()
        for _ in range(40):
            model.observe("category", False)
        for _ in range(40):
            model.observe("material", True)
        self.assertNotEqual(choose(self.catalog, self._ranked(), set(), model,
                                   extra={"category": 0.9}), "category")

    def test_is_never_asked_twice(self):
        model = AnswerModel()
        for _ in range(20):
            model.observe("category", True)
        self.assertNotEqual(choose(self.catalog, self._ranked(), {"category"}, model,
                                   extra={"category": 0.9}), "category")


class EndToEndTest(unittest.TestCase):
    """The answer locks the shelf, and locking it does not discard the conversation."""

    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.path = cls._catalog.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def test_answering_sets_the_category_without_a_product_reset(self):
        from src.agent import Agent
        from src.policy import TOP_K

        agent = Agent(self.path)
        agent.reset("s", PROFILE)
        agent.respond("s", "I'm looking for Accessories Belts. A key requirement is: 100% Leather.",
                      1, TOP_K)
        state = agent.sessions["s"]
        before = {slot.text for slot in state.dialog.slots if not slot.superseded}
        self.assertTrue(before, "nothing was learned from the opening turn")

        state.offered = ["Accessories Scarves", "Accessories Belts"]
        agent.respond("s", "I'm after accessories scarves.", 2, TOP_K)

        self.assertEqual(state.dialog.category, "Accessories Scarves")
        self.assertEqual(state.offered, [], "the offer was not cleared after it was answered")
        still = {slot.text for slot in state.dialog.slots if not slot.superseded}
        self.assertTrue(before & still,
                        "answering which shelf erased constraints the customer still holds")


if __name__ == "__main__":
    unittest.main()
