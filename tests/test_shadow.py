"""Group S — the artifact-sensitivity harness itself.

tools/shadow.py exists to tell us when a change is buying score from the simulator
rather than from better retrieval. A harness that quietly stopped doing that would
be worse than not having one, because we would trust it. These are the properties
it has to keep.
"""
from __future__ import annotations

import unittest

from src.agent import Agent
from src.policy import TOP_K
from tests.fixtures import PROFILE, RichCatalog
from tools.shadow import strict_reply

CARD = {"hard_constraints": ["leather", "black"], "soft_preferences": ["Buckle closure"]}
SAMPLE = {"scenario_type": "buying", "intent_card": CARD}


class WildcardRemovedTest(unittest.TestCase):
    """The official `customer_reply` matches on
    `attribute == "other" or classify_constraint(v) == attribute`, so "other"
    unlocks every undisclosed constraint at once. Here it must unlock nothing."""

    def test_s1_other_discloses_nothing(self):
        reply, _ = strict_reply(SAMPLE, "other", set(), False)
        self.assertIn("don't have an additional preference", reply,
                      f"'other' still behaves as a disclosure wildcard: {reply}")

    def test_s2_a_real_attribute_still_discloses(self):
        """The control. If this fails the harness is simply broken, not strict."""
        reply, _ = strict_reply(SAMPLE, "material", set(), False)
        self.assertIn("leather", reply.lower(),
                      f"a genuine attribute stopped disclosing: {reply}")

    def test_s3_a_declined_boundary_turn_is_preserved(self):
        reply, used = strict_reply({**SAMPLE, "scenario_type": "boundary"},
                                   "material", set(), False)
        self.assertTrue(used)
        self.assertIn("use your judgment", reply)


class InternalRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def _agent(self, width: int) -> Agent:
        agent = Agent(self.path)
        agent.commit = type(agent.commit)(base_width=width)
        agent.reset("s", PROFILE)
        return agent

    def test_s4_the_internal_ranking_does_not_depend_on_how_much_is_shown(self):
        """The whole point of the hook. If width moves the internal ranking then
        the harness is measuring presentation again and cannot separate the two."""
        narrow = self._agent(1)
        wide = self._agent(TOP_K)
        message = "I'm looking for Accessories Belts. A key requirement is: leather."
        narrow.respond("s", message, 1, TOP_K)
        wide.respond("s", message, 1, TOP_K)
        self.assertEqual(narrow.internal_ranking("s"), wide.internal_ranking("s"),
                         "recommendation width changed the internal ranking")

    def test_s5_width_still_changes_what_the_harness_is_shown(self):
        """The control for S4: the slates must actually differ, or S4 proves nothing."""
        narrow = self._agent(1)
        wide = self._agent(TOP_K)
        message = "I'm looking for Accessories Belts."
        a = narrow.respond("s", message, 1, TOP_K)["recommendations"]
        b = wide.respond("s", message, 1, TOP_K)["recommendations"]
        self.assertLess(len(a), len(b), "the two widths produced the same slate")

    def test_s6_the_hook_never_leaks_into_the_contract(self):
        agent = self._agent(1)
        response = agent.respond("s", "I'm looking for Accessories Belts.", 1, TOP_K)
        self.assertEqual(set(response) - {"message", "ask_attribute", "recommendations",
                                          "usage"}, set(),
                         "a diagnostic field reached the harness response")
        self.assertTrue(agent.internal_ranking("s"), "the hook recorded nothing")

    def test_s7_unknown_session_yields_an_empty_ranking(self):
        self.assertEqual(self._agent(1).internal_ranking("nope"), [])


class SharedIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.base = Agent(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_s8_the_frozen_index_is_shared_and_the_learned_state_is_not(self):
        """Cold start is only measurable if `sharing_index` resets what accumulates
        while reusing what is read-only."""
        self.base.answers.observe("material", True)
        clone = Agent.sharing_index(self.base)
        self.assertIs(clone.catalog, self.base.catalog, "the catalog was rebuilt")
        self.assertIs(clone.semantic, self.base.semantic, "the vocabulary was remined")
        self.assertEqual(clone.answers.confidence("material"), 0,
                         "the clone inherited learned answerability")
        self.assertEqual(len(clone.memory), 0, "the clone inherited long-term memory")

    def test_s9_a_clone_is_a_working_agent(self):
        clone = Agent.sharing_index(self.base)
        clone.reset("c", PROFILE)
        response = clone.respond("c", "I'm looking for Accessories Belts.", 1, TOP_K)
        self.assertTrue(response["recommendations"])
        self.assertIsInstance(response["message"], str)


if __name__ == "__main__":
    unittest.main()
