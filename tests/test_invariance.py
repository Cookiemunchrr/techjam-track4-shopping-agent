"""Metamorphic invariance: the same request, said differently, gets the same answer.

The ideal-behaviour spec asks for one property that no score can check on its own:

    behaviour is invariant to *how* something is said. Clause order, the order of
    two constraints stated in one breath, case, punctuation and whitespace must not
    move the recommendation.

Metamorphic testing is the right instrument here because it needs no ground truth.
Each test transforms an input in a way that provably should not change the meaning,
then asserts the output did not change. That catches the failure mode a leaderboard
cannot see -- an agent that has learned the organizer's exact templates rather than
what a customer means -- without anyone having to label a single session.

These are floors, not targets. Every one of them describes something the agent
already does; they exist so that a future change cannot quietly stop doing it.
tools/adversarial.py measures how far robustness extends; this file pins the part
that must never regress at all.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.agent import Agent
from src.policy import TOP_K
from src.state import DialogState
from tests.fixtures import PROFILE, TempCatalog

OPENER = "I'm looking for Accessories Belts. A key requirement is: 100% Leather."


def slate(agent: Agent, turns, session: str = "s") -> list[str]:
    """Run a transcript and return the final recommendation ids."""
    agent.reset(session, PROFILE)
    response = {}
    for index, message in enumerate(turns, start=1):
        response = agent.respond(session, message, index, TOP_K)
    return [item["parent_asin"] for item in response.get("recommendations", [])]


class InvarianceTest(unittest.TestCase):
    """Transformations that preserve meaning must preserve the slate."""

    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.path = cls._catalog.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def agent(self) -> Agent:
        return Agent(self.path)

    def test_constraint_order_within_a_turn_does_not_matter(self):
        """"what matters is: A; B" and "...: B; A" are the same sentence.

        The simulator emits both orders depending on which constraint it drew
        first, and a customer listing two requirements has not ranked them by
        saying one first.
        """
        a = slate(self.agent(), [OPENER, "For that, what matters is: Buckle closure; Imported."])
        b = slate(self.agent(), [OPENER, "For that, what matters is: Imported; Buckle closure."])
        self.assertEqual(a, b, "swapping two constraints stated in one breath moved the slate")

    def test_case_does_not_matter(self):
        a = slate(self.agent(), [OPENER])
        b = slate(self.agent(), [OPENER.upper()])
        c = slate(self.agent(), [OPENER.lower()])
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_whitespace_and_trailing_punctuation_do_not_matter(self):
        a = slate(self.agent(), [OPENER])
        b = slate(self.agent(), ["  " + OPENER.replace(". ", ".   ").rstrip(".") + "  ..  "])
        self.assertEqual(a, b, "whitespace or trailing punctuation moved the slate")

    def test_curly_apostrophe_keeps_the_verbatim_phrase_bonus(self):
        from src.scoring import Scorer
        from src.text import normalise

        scorer = Scorer(SimpleNamespace(corpus={"dress": normalise("Women's Classic Dress")}))
        curly = scorer._phrase("dress", ["Women\u2019s Classic Dress"])
        straight = scorer._phrase("dress", ["Women's Classic Dress"])
        self.assertEqual(curly, straight)
        self.assertGreater(curly, 0.0)

    def test_a_repeated_constraint_is_not_double_counted(self):
        """Saying the same thing twice is emphasis, not twice the evidence.

        A customer who repeats themselves has given one constraint. Scoring it
        twice would let insistence outrank information.
        """
        once = slate(self.agent(), [OPENER, "For that, what matters is: Buckle closure."])
        twice = slate(self.agent(), [OPENER, "For that, what matters is: Buckle closure.",
                                     "For that, what matters is: Buckle closure."])
        self.assertEqual(once[:3], twice[:3],
                         "repeating a constraint changed the top of the slate")

    def test_determinism_across_identical_transcripts(self):
        self.assertEqual(slate(self.agent(), [OPENER], "a"), slate(self.agent(), [OPENER], "b"))

    def test_two_agents_over_the_same_catalog_agree(self):
        """Nothing may depend on process state -- PYTHONHASHSEED included."""
        self.assertEqual(slate(self.agent(), [OPENER]), slate(self.agent(), [OPENER]))


class MonotonicityTest(unittest.TestCase):
    """New information must never make the ranking worse.

    Stated as a property of the state layer rather than of a whole session,
    because a session ends at the first hit and so cannot observe what happens
    to a rank after it. Measured end-to-end in analysis/shelf_election_finding.json:
    over 120 sessions, turn 1 to turn 2, the target's within-shelf rank improved 0
    times, held 112 and slipped 8 -- and every slip is a one-rank drift from slot
    decay with an unchanged slot set, never new evidence scoring against the target.
    """

    @classmethod
    def setUpClass(cls):
        cls._catalog = TempCatalog()
        cls.path = cls._catalog.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._catalog.__exit__(None, None, None)

    def test_a_matching_constraint_never_lowers_the_matching_product(self):
        from src.catalog import Catalog
        from src.scoring import Scorer

        catalog = Catalog(self.path)
        scorer = Scorer(catalog)
        belts = [pid for pid in catalog.ids if "belt" in catalog.corpus[pid]]

        before = DialogState()
        before.observe("I'm looking for Accessories Belts.", 1, catalog)
        after = DialogState()
        after.observe("I'm looking for Accessories Belts.", 1, catalog)
        after.observe("For that, what matters is: Suede.", 2, catalog)

        def rank_of(state, turn):
            state.turn = turn
            order = [pid for _, pid in scorer.rank(belts, state.weighted_phrases())]
            return order.index("P_SUEDE_BELT") + 1

        self.assertLessEqual(rank_of(after, 2), rank_of(before, 2),
                             "stating a constraint the product satisfies lowered its rank")

    def test_a_declined_question_changes_nothing(self):
        """A decline is not evidence, and must not be stored as any."""
        state = DialogState()
        state.observe(OPENER, 1, None)
        before = list(state.weighted_phrases())
        state.observe("I don't have a preference for color; please use your judgment.", 2, None)
        self.assertEqual([text for text, _ in state.weighted_phrases()],
                         [text for text, _ in before],
                         "a declined question was stored as a constraint")


if __name__ == "__main__":
    unittest.main()
