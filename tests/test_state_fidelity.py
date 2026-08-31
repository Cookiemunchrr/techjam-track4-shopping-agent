"""Did the agent understand the sentence?

Three defects this file exists to keep fixed, each one invisible to the official
harness by construction and each one ordinary shopper behaviour:

  opening clause    "black leather boots" states a category and two facets at
                    once. The opening clause used to be dropped whole.
  correction scope  "Actually, brown" used to retire the material, the budget and
                    the size along with the colour.
  shelf position    a shelf name used to be found only at the end of a sentence,
                    so "do you carry Underwear Briefs, just seeing what is out
                    there" resolved to nothing.

The official turn-1 message is `coarse_category(target.categories)` verbatim, so
none of this can be measured on the leaderboard. It is measured here and on
tools/bench.py --adversarial.
"""
from __future__ import annotations

import unittest

from src.catalog import Catalog
from src.routing import exact_bucket, locate_bucket
from src.state import DialogState
from src.text import is_correction, is_restart
from tests.fixtures import TempCatalog


class ShelfPositionTest(unittest.TestCase):
    """A shelf name is a phrase in a sentence, not a suffix of one."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.path = cls._ctx.__enter__()
        cls.catalog = Catalog(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_bare_shelf_name_resolves(self):
        self.assertEqual(exact_bucket(self.catalog, "accessories belts"),
                         "Accessories Belts")

    def test_leading_scaffolding_is_looked_past(self):
        self.assertEqual(
            exact_bucket(self.catalog, "hoping to pick up accessories belts"),
            "Accessories Belts")

    def test_trailing_chat_is_looked_past(self):
        """The regression this rewrite was for: the shopper keeps talking.

        Anchoring the scan at the end of the phrase assumes a shopper stops
        speaking once they have named the thing. Measured on the scaffold axis,
        that assumption lost the shelf on 47% of openings.
        """
        for key in ("do you carry accessories belts, just seeing what is out there",
                    "accessories belts, nothing settled yet",
                    "wondering if you stock accessories belts, still weighing it up"):
            with self.subTest(key=key):
                self.assertEqual(exact_bucket(self.catalog, key), "Accessories Belts")

    def test_word_order_inside_the_name_still_resolves(self):
        self.assertEqual(exact_bucket(self.catalog, "belts accessories"),
                         "Accessories Belts")

    def test_prefers_the_later_ending_over_the_longer_window(self):
        """The taxonomy puts the leaf noun last, and that outranks window length.

        "shoes & jewelry women dresses" contains both a longer early window and
        the shelf the shopper meant. Preferring length alone cost 0.084 under
        granularity drift.
        """
        located = locate_bucket(self.catalog, "men accessories belts")
        self.assertIsNotNone(located)
        self.assertEqual(located[0], "Accessories Belts")

    def test_a_phrase_naming_nothing_resolves_to_nothing(self):
        self.assertIsNone(exact_bucket(self.catalog, "something entirely unrelated"))

    def test_span_is_returned_with_the_bucket(self):
        located = locate_bucket(self.catalog, "hoping to pick up accessories belts")
        self.assertEqual(located, ("Accessories Belts", 4, 2))


class OpeningClauseTest(unittest.TestCase):
    """Constraints stated in the same breath as the category."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def observe(self, message, turn=1):
        state = DialogState()
        state.observe(message, turn, self.catalog, skip_first=True)
        return state

    def test_facets_in_the_opening_clause_survive(self):
        state = self.observe("black leather boots")
        self.assertEqual(state.facets(), {"material": "leather", "color": "black"})

    def test_a_pure_category_opening_adds_nothing(self):
        """The official turn-1 message. This path must stay a no-op.

        `coarse_category` is fed back verbatim, so if the shelf name leaked into
        the slot table it would be scored as a constraint on every single session.
        """
        for message in ("Accessories Belts", "accessories belts"):
            with self.subTest(message=message):
                self.assertEqual(self.observe(message).slots, [])

    def test_the_category_noun_does_not_become_a_constraint(self):
        """Whatever is left after the shelf name is the product noun, not a spec."""
        state = self.observe("black leather boots")
        self.assertNotIn("boots", [slot.text for slot in state.slots])
        self.assertEqual({slot.attribute for slot in state.slots}, {"material", "color"})

    def test_budget_in_the_opening_clause_is_kept_as_the_money_span_only(self):
        state = self.observe("a black cotton dress under $40")
        budget = state.budget()
        self.assertIsNotNone(budget)
        self.assertEqual(budget.amount, 40.0)
        self.assertTrue(budget.cap)
        # The slot text is scored verbatim by the phrase term, so it must not drag
        # the category words in with it.
        spans = [slot.text for slot in state.slots if slot.attribute == "budget"]
        self.assertEqual(spans, ["under $40"])

    def test_a_composition_percentage_is_not_a_budget(self):
        self.assertIsNone(self.observe("100% cotton hoodies").budget())

    def test_multiple_materials_in_one_clause_are_all_kept(self):
        state = self.observe("a cotton and wool blend scarf")
        self.assertEqual({slot.text for slot in state.slots if slot.attribute == "material"},
                         {"cotton", "wool"})


class CorrectionScopeTest(unittest.TestCase):
    """A correction changes something, not everything."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def opened(self):
        state = DialogState()
        state.observe("Black leather boots under $100.", 1, self.catalog, skip_first=True)
        return state

    def test_attribute_correction_replaces_only_that_attribute(self):
        """The defect: one "actually" used to erase the whole conversation.

        It is wrong on this simulator too, not only in principle -- the override
        sessions derive the old and the new value from the same target product.
        """
        state = self.opened()
        state.observe("Actually, brown.", 2, self.catalog)
        active = state.phrases()
        self.assertIn("brown", active)
        self.assertNotIn("black", active)
        self.assertIn("leather", active)              # not the attribute corrected
        self.assertIsNotNone(state.budget())          # nor is the budget

    def test_superseded_value_is_held_not_deleted(self):
        state = self.opened()
        state.observe("Actually, brown.", 2, self.catalog)
        black = [s for s in state.slots if s.text == "black"]
        self.assertEqual(len(black), 1)
        self.assertTrue(black[0].superseded)

    def test_explicit_restart_does_retire_everything(self):
        state = self.opened()
        state.observe("Forget everything, let's start over. I need a silk scarf.",
                      2, self.catalog)
        active = state.phrases()
        self.assertNotIn("leather", active)
        self.assertNotIn("black", active)
        self.assertIn("a silk scarf", active)

    def test_the_restart_phrase_is_not_stored_as_a_requirement(self):
        """The defect this file's parent docstring opens with, in a new costume."""
        state = DialogState()
        state.observe("Forget everything. Start over.", 1, self.catalog)
        self.assertEqual(state.phrases(), [])

    def test_a_bare_restart_retires_an_existing_brief(self):
        state = self.opened()
        state.observe("Forget everything. Start over.", 2, self.catalog)
        self.assertEqual(state.phrases(), [])

    def test_restart_is_not_triggered_by_an_ordinary_correction(self):
        self.assertTrue(is_correction("Actually, brown."))
        self.assertFalse(is_restart("Actually, brown."))
        self.assertFalse(is_restart("Actually, forget that -- I meant brown."))
        self.assertTrue(is_restart("Let's start over."))
        self.assertTrue(is_restart("Forget everything I said."))

    def test_official_override_message_keeps_the_earlier_constraint(self):
        """The organizer's override template, scoped correctly.

        `behavior_for` sets new_value from the target's own hard_constraints and
        old_value from its soft_preferences, so both describe the target. Retiring
        the old one on the cue is measurably wrong.
        """
        state = self.opened()
        state.observe("Actually, ignore my earlier preference. What I need is: silk.",
                      3, self.catalog)
        self.assertIn("silk", state.phrases())
        self.assertIsNotNone(state.budget())


if __name__ == "__main__":
    unittest.main()


class AlternativesTest(unittest.TestCase):
    """"Blue or green would work" is one constraint with two answers."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def opened(self):
        state = DialogState()
        state.observe("Accessories Belts", 1, self.catalog, skip_first=True)
        return state

    def test_both_values_are_kept(self):
        state = self.opened()
        state.observe("Blue or green would work.", 2, self.catalog)
        self.assertEqual(state.facets()["color"], ("blue", "green"))

    def test_neither_alternative_supersedes_the_other(self):
        state = self.opened()
        state.observe("Blue or green would work.", 2, self.catalog)
        live = [s for s in state.slots if s.attribute == "color" and not s.superseded]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].operator, "one_of")

    def test_a_later_single_value_supersedes_the_whole_set(self):
        state = self.opened()
        state.observe("Blue or green would work.", 2, self.catalog)
        state.observe("Actually, navy.", 3, self.catalog)
        self.assertEqual(state.facets()["color"], "navy")

    def test_alternatives_in_the_opening_clause(self):
        state = DialogState()
        state.observe("blue or green belts", 1, self.catalog, skip_first=True)
        self.assertEqual(state.facets()["color"], ("blue", "green"))

    def test_an_or_that_is_not_a_list_of_values_is_left_alone(self):
        """"leather or something cheap" is not two materials.

        Reading it as one would invent a constraint the shopper did not state.
        """
        state = DialogState()
        state.observe("belts. leather or something cheap", 1, self.catalog, skip_first=True)
        self.assertEqual(state.facets().get("material"), "leather")

    def test_a_one_of_set_scores_its_best_member_not_its_sum(self):
        from src.scoring import Scorer
        scorer = Scorer(self.catalog)
        both = scorer.rank(["P_CANVAS_BELT"], [], facets={"color": ("blue", "green")})
        one = scorer.rank(["P_CANVAS_BELT"], [], facets={"color": "blue"})
        self.assertAlmostEqual(both[0][0], one[0][0])


class BudgetDirectionTest(unittest.TestCase):
    """Which side of the number the shopper meant."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def budget(self, phrase):
        state = DialogState()
        state.observe("Accessories Belts. " + phrase, 1, self.catalog, skip_first=True)
        return state.budget()

    def test_ceiling_floor_and_tier_are_distinguished(self):
        for phrase, amount, cap, floor in (
                ("under $30", 30.0, True, False),
                ("no more than 30 dollars", 30.0, True, False),
                ("less than $20", 20.0, True, False),
                ("at least $50", 50.0, False, True),
                ("no less than 75 dollars", 75.0, False, True),
                ("more than $60", 60.0, False, True),
                ("around $30", 30.0, False, False)):
            with self.subTest(phrase=phrase):
                budget = self.budget(phrase)
                self.assertIsNotNone(budget, phrase)
                self.assertEqual(budget.amount, amount)
                self.assertEqual((budget.cap, budget.floor), (cap, floor))

    def test_a_comparison_is_not_a_refusal(self):
        """"no more than $30" used to be parsed as refusing the word "more".

        The budget vanished and every product whose text contains "more" was
        penalised for it.
        """
        state = DialogState()
        state.observe("Accessories Belts. no more than 30 dollars", 1, self.catalog,
                      skip_first=True)
        self.assertEqual(state.rejected(), [])

    def test_a_refusal_is_still_a_refusal(self):
        state = DialogState()
        state.observe("Accessories Belts. not polyester", 1, self.catalog, skip_first=True)
        self.assertEqual(state.rejected(), ["polyester"])

    def test_a_range_is_neither_a_cap_nor_a_floor(self):
        budget = self.budget("between $20 and $40")
        self.assertEqual(budget.amount, 30.0)
        self.assertFalse(budget.cap)
        self.assertFalse(budget.floor)

    def test_a_floor_is_satisfied_from_above(self):
        from src.scoring import Scorer
        from src.state import Budget
        scorer = Scorer(self.catalog)
        # P_SILK_SCARF is $60, P_CANVAS_BELT is $12.
        dear = scorer.rank(["P_SILK_SCARF"], [], budget=Budget(50, floor=True))[0][0]
        cheap = scorer.rank(["P_CANVAS_BELT"], [], budget=Budget(50, floor=True))[0][0]
        bare_dear = scorer.rank(["P_SILK_SCARF"], [])[0][0]
        self.assertAlmostEqual(dear, bare_dear)      # above the floor: no penalty
        self.assertLess(cheap, scorer.rank(["P_CANVAS_BELT"], [])[0][0])


class RefusalTest(unittest.TestCase):
    """Refusals resolved against facets, not against substrings."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.path = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def setUp(self):
        from src.agent import Agent
        self.agent = Agent(self.path)

    def test_a_product_that_is_the_refused_material_is_penalised_hardest(self):
        pool = ["P_SUEDE_BELT", "P_LEATHER_BELT", "P_CANVAS_BELT"]
        penalties = dict(self.agent._refusals(["leather"], pool))
        self.assertEqual(penalties.get("P_LEATHER_BELT"), 0.9)

    def test_a_product_merely_mentioning_it_is_penalised_less(self):
        """P_CANVAS_BELT's description says "canvas belt"; it is cotton.

        The point of the tier: a mention is evidence, a composition is a fact.
        """
        pool = ["P_LEATHER_BELT", "P_CANVAS_BELT"]
        penalties = dict(self.agent._refusals(["cotton"], pool))
        self.assertGreater(penalties.get("P_CANVAS_BELT", 0.0), 0.0)
        self.assertNotIn("P_LEATHER_BELT", penalties)

    def test_an_unresolvable_refusal_still_uses_the_text(self):
        """"no buckle" names no facet, so the text is the only evidence there is."""
        penalties = dict(self.agent._refusals(["buckle"], ["P_LEATHER_BELT", "P_SILK_SCARF"]))
        self.assertIn("P_LEATHER_BELT", penalties)
        self.assertNotIn("P_SILK_SCARF", penalties)

    def test_no_refusals_means_no_penalties(self):
        self.assertEqual(list(self.agent._refusals([], ["P_LEATHER_BELT"])), [])

    def test_the_text_fallback_matches_words_not_letters(self):
        """A refusal the catalog cannot resolve is read at the word boundary.

        `catalog.corpus` is one normalised string, so the substring test this
        replaces made "silk" match "silky" -- 802 products in the shipped catalog
        against the 478 that contain the word -- and "right" match "bright" and
        "upright", 3,397 against 2,135. The fallback fires precisely when the text
        is the only evidence there is, which is why it has to read the text
        correctly.
        """
        pool = ["P_LEATHER_BELT", "P_SILK_SCARF"]
        # "buck" is a substring of "buckle" and is not a word in any product here.
        self.assertEqual(list(self.agent._refusals(["buck"], pool)), [])
        self.assertIn("P_LEATHER_BELT", dict(self.agent._refusals(["buckle"], pool)))

    def test_a_refusal_that_ends_a_sentence_still_matches(self):
        """Punctuation is stripped by tokenising, not left attached.

        A raw split produced "buckle." as the refused word, which matches no index
        term at all -- so the fallback silently did nothing for any refusal that
        happened to end a sentence.
        """
        pool = ["P_LEATHER_BELT", "P_SILK_SCARF"]
        self.assertIn("P_LEATHER_BELT",
                      dict(self.agent._refusals(["no buckle."], pool)))


class IntentEvidenceTest(unittest.TestCase):
    """The track is re-read every turn, not decided once."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.path = cls._ctx.__enter__()
        cls.catalog = Catalog(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def walk(self, messages):
        from src.text import split_clauses
        state = DialogState()
        out = []
        for turn, message in enumerate(messages, 1):
            state.observe(message, turn, self.catalog, skip_first=(turn == 1))
            out.append(state.read_intent(message, split_clauses(message)))
        return out

    def test_an_exploring_opening_is_browsing(self):
        self.assertEqual(
            self.walk(["I'm looking for belts, but I'm still exploring."])[0], "browsing")

    def test_an_underspecified_opening_is_open(self):
        self.assertEqual(self.walk(["Accessories Belts"])[0], "open")

    def test_a_committed_opening_is_buying(self):
        self.assertEqual(self.walk(["I need black leather belts."])[0], "buying")

    def test_accumulating_constraints_turn_browsing_into_buying(self):
        """The defect: the track was assigned once and never revisited.

        A shopper who opens vaguely and then names a material, a colour and a
        budget kept retrieving from the wide `open`/`browsing` pool for the rest
        of the session.
        """
        track = self.walk(["I'm looking for belts, but I'm still exploring.",
                           "black", "leather", "under $80"])
        self.assertEqual(track[0], "browsing")
        self.assertEqual(track[-1], "buying")

    def test_an_exploration_cue_pushes_back(self):
        track = self.walk(["I need black leather belts.",
                           "Actually I'm still deciding, just browsing."])
        self.assertEqual(track[0], "buying")
        self.assertNotEqual(track[-1], "buying")

    def test_late_browsing_cue_widens_the_track_without_erasing_constraints(self):
        from src.agent import Agent

        agent = Agent(self.path)
        agent.ask_mode = "none"
        agent.trace_pool = True
        agent.reset("late-browse", {})
        messages = ["I'm looking for Accessories Belts.", "black", "leather",
                    "under $80", "I'm still exploring."]
        sizes = []
        for turn, message in enumerate(messages, 1):
            agent.respond("late-browse", message, turn, 10)
            sizes.append(len(agent.candidate_pool("late-browse")))
        state = agent.sessions["late-browse"].dialog
        self.assertEqual(state.intent, "browsing")
        self.assertGreater(sizes[-1], sizes[-2], "browsing cue did not widen the pool")
        self.assertEqual(state.facets().get("material"), "leather")
        self.assertEqual(state.facets().get("color"), "black")
        self.assertIsNotNone(state.budget())

    def test_a_retraction_lowers_the_evidence_rather_than_stranding_it(self):
        """Constraints are counted fresh, not incremented into a high-water mark."""
        state = DialogState()
        state.observe("Accessories Belts", 1, self.catalog, skip_first=True)
        state.observe("black leather", 2, self.catalog)
        with_both = state.read_intent("black leather", ["black leather"])
        state.observe("Forget everything, start over.", 3, self.catalog)
        after = state.read_intent("Forget everything, start over.", [])
        self.assertEqual(with_both, "buying")
        self.assertNotEqual(after, "buying")
