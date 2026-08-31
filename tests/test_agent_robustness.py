"""The agent must never crash, hang, or forfeit a turn.

docs/competition_specification.md: "Exceptions, invalid output, and timeouts may
count as a miss." Every one of these inputs is therefore a scoring risk.
"""
from __future__ import annotations

import time
import unittest

from src.agent import Agent
from src.elicitation import ALLOWED
from src.policy import TOP_K
from tests.fixtures import PROFILE, TempCatalog

HOSTILE_MESSAGES = [
    "", "   ", "\n\t\r", "!!!", "?" * 500, "a" * 20000,
    "🙂🙂🙂", "上衣 皮带", "Ω≈ç√∫˜µ", "<script>alert(1)</script>",
    "'; DROP TABLE products; --", "\\x00\\x01", "NaN", "None", "null",
    "For that, what matters is:", "For that, what matters is: ;;;;",
    "I don't have a preference for", "Actually,", "%s %d {} {0}",
    "-" * 200, "; " * 300,
]


class RobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.path = cls._ctx.__enter__()
        cls.agent = Agent(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_survives_hostile_messages_on_every_turn(self):
        for index, message in enumerate(HOSTILE_MESSAGES):
            session = f"h{index}"
            self.agent.reset(session, PROFILE)
            for turn in (1, 2, 10):
                response = self.agent.respond(session, message, turn, TOP_K)
                self.assertIsInstance(response, dict, repr(message[:40]))
                self.assertIn(response["ask_attribute"], set(ALLOWED) | {None})
                self.assertLessEqual(len(response["recommendations"]), TOP_K)

    def test_respond_without_reset_does_not_crash(self):
        response = self.agent.respond("never-reset", "Belts", 1, TOP_K)
        self.assertIsInstance(response, dict)

    def test_survives_a_none_message(self):
        self.agent.reset("n", PROFILE)
        self.assertIsInstance(self.agent.respond("n", None, 1, TOP_K), dict)

    def test_survives_odd_profiles(self):
        for profile in ({}, None, {"preference_tags": None},
                        {"average_prior_rating": "high"}, {"unexpected": [1, 2, 3]}):
            self.agent.reset("p", profile)
            self.assertIsInstance(self.agent.respond("p", "Belts", 1, TOP_K), dict)

    def test_survives_out_of_range_turn_numbers(self):
        self.agent.reset("o", PROFILE)
        for turn in (0, -1, 1, 99, 10 ** 9):
            self.assertIsInstance(self.agent.respond("o", "Belts", turn, TOP_K), dict)

    def test_survives_odd_top_k(self):
        self.agent.reset("k", PROFILE)
        for top_k in (0, 1, -5, 10 ** 6):
            response = self.agent.respond("k", "Belts", 1, top_k)
            self.assertLessEqual(len(response["recommendations"]), TOP_K)

    def test_non_positive_top_k_returns_no_recommendations(self):
        self.agent.reset("empty-k", PROFILE)
        for top_k in (0, -1, -10 ** 6):
            response = self.agent.respond("empty-k", "Belts", 1, top_k)
            self.assertEqual(response["recommendations"], [])

    def test_a_broken_scorer_degrades_to_an_empty_turn_not_a_crash(self):
        """If any internal stage throws, the harness must still get a valid dict."""
        agent = Agent(self.path)
        agent.reset("boom", PROFILE)

        def explode(*args, **kwargs):
            raise RuntimeError("scorer exploded")

        agent.scorer.rank = explode
        response = agent.respond("boom", "Belts", 1, TOP_K)
        self.assertIsInstance(response, dict)
        self.assertEqual(response["recommendations"], [])
        self.assertIn(response["ask_attribute"], set(ALLOWED) | {None})

    def test_a_turn_is_fast_enough_for_the_harness(self):
        self.agent.reset("speed", PROFILE)
        start = time.perf_counter()
        for turn in range(1, 11):
            self.agent.respond("speed", "I want a leather belt with a buckle", turn, TOP_K)
        self.assertLess(time.perf_counter() - start, 5.0, "ten turns took over five seconds")

    def test_never_recommends_an_id_outside_the_catalog(self):
        for index, message in enumerate(HOSTILE_MESSAGES):
            self.agent.reset(f"c{index}", PROFILE)
            for item in self.agent.respond(f"c{index}", message, 1, TOP_K)["recommendations"]:
                self.assertIn(item["parent_asin"], self.agent.catalog.corpus)


if __name__ == "__main__":
    unittest.main()
