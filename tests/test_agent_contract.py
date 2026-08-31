"""The Agent must satisfy docs/agent_api_contract.json on every turn, always.

The harness treats invalid output as a miss, so a contract violation is a scoring
bug, not a cosmetic one.
"""
from __future__ import annotations

import unittest

from src.agent import Agent
from src.elicitation import ALLOWED
from src.policy import MAX_TURNS, TOP_K
from tests.fixtures import PROFILE, TempCatalog

VALID_ATTRIBUTES = set(ALLOWED) | {None}


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.path = cls._ctx.__enter__()
        cls.agent = Agent(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def assert_valid(self, response, message=""):
        self.assertIsInstance(response, dict, message)
        self.assertIsInstance(response["message"], str, message)
        self.assertIn(response["ask_attribute"], VALID_ATTRIBUTES, message)

        recs = response["recommendations"]
        self.assertIsInstance(recs, list, message)
        self.assertLessEqual(len(recs), TOP_K, "more than top_k recommendations")
        seen = set()
        for item in recs:
            self.assertIsInstance(item, dict, message)
            self.assertEqual(set(item), {"parent_asin"}, "unexpected keys in a recommendation")
            pid = item["parent_asin"]
            self.assertIsInstance(pid, str)
            self.assertTrue(pid, "empty parent_asin")
            self.assertNotIn(pid, seen, "duplicate parent_asin in one response")
            seen.add(pid)
            self.assertIn(pid, self.agent.catalog.corpus, "parent_asin outside the catalog")

        usage = response["usage"]
        self.assertGreaterEqual(usage["prompt_tokens"], 0)
        self.assertGreaterEqual(usage["completion_tokens"], 0)

    def test_first_turn_is_valid(self):
        self.agent.reset("s1", PROFILE)
        self.assert_valid(self.agent.respond("s1", "I'm looking for Accessories Belts.", 1, TOP_K))

    def test_every_turn_of_a_full_session_is_valid(self):
        self.agent.reset("s2", PROFILE)
        messages = ["I'm looking for Accessories Belts. A key requirement is: 100% Leather.",
                    "For that, what matters is: Buckle closure; Imported.",
                    "I don't have a preference for color; please use your judgment.",
                    "Actually, ignore my earlier preference. What I need is: Suede."]
        for turn in range(1, MAX_TURNS + 1):
            message = messages[min(turn - 1, len(messages) - 1)]
            self.assert_valid(self.agent.respond("s2", message, turn, TOP_K),
                              message=f"turn {turn}")

    def test_recommendations_are_ordered_best_first(self):
        """The contract says best to worst; verify against the scorer directly."""
        self.agent.reset("s3", PROFILE)
        response = self.agent.respond("s3", "I'm looking for Accessories Belts. Suede.", 1, TOP_K)
        picks = [r["parent_asin"] for r in response["recommendations"]]
        if len(picks) > 1:
            state = self.agent.sessions["s3"]
            ranked = self.agent.scorer.rank(picks, state.clauses)
            self.assertEqual([p for _, p in ranked], picks, "not ordered best to worst")

    def test_never_exceeds_a_smaller_top_k(self):
        self.agent.reset("s4", PROFILE)
        for turn in range(1, 6):
            response = self.agent.respond("s4", "Belts please", turn, 2)
            self.assertLessEqual(len(response["recommendations"]), 2)

    def test_sessions_are_isolated(self):
        self.agent.reset("a", PROFILE)
        self.agent.reset("b", PROFILE)
        self.agent.respond("a", "I'm looking for Accessories Belts. Suede.", 1, TOP_K)
        self.assertEqual(self.agent.sessions["b"].clauses, [],
                         "state leaked between concurrent sessions")

    def test_is_deterministic(self):
        outputs = []
        for _ in range(3):
            agent = Agent(self.path)
            agent.reset("d", PROFILE)
            outputs.append([r["parent_asin"] for r in
                            agent.respond("d", "I'm looking for Accessories Belts.", 1, TOP_K)
                            ["recommendations"]])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_reports_zero_tokens_because_it_runs_offline(self):
        self.agent.reset("t", PROFILE)
        usage = self.agent.respond("t", "Belts", 1, TOP_K)["usage"]
        self.assertEqual(usage, {"prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main()
