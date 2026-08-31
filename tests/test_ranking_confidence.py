"""V4-0 confidence plumbing: preserve legacy behaviour, expose the channel.

The current recommendation width is driven by the top-two margin *after* the
reranker.  V4-0 does not change that policy.  It only makes the confidence value
an explicit input so a later, separately measured migration cannot accidentally
change width by changing the arbitrary scale of a ranking residual.
"""
from __future__ import annotations

import unittest

from src.agent import Agent
from src.policy import CommitPolicy
from tests.fixtures import PROFILE, RichCatalog


class LegacyConfidenceParityTest(unittest.TestCase):
    def setUp(self):
        self.policy = CommitPolicy()

    def test_legacy_confidence_is_the_post_rank_top_two_margin(self):
        ranked = [(1.25, "a"), (0.75, "b"), (-4.0, "c")]
        self.assertEqual(self.policy.legacy_confidence(ranked), 0.5)
        self.assertEqual(self.policy.legacy_confidence([]), 0.0)
        self.assertEqual(self.policy.legacy_confidence([(1.0, "a")]), 0.0)

    def test_explicit_legacy_confidence_reproduces_every_width_tier(self):
        rankings = [
            [(1.0, "a"), (0.95, "b")],
            [(1.0, "a"), (0.85, "b")],
            [(1.0, "a"), (0.55, "b")],
            [(1.0, "a"), (0.10, "b")],
            [(1.0, "a")],
            [],
        ]
        for turn in (1, 7, 8):
            for browsing in (False, True):
                for ranked in rankings:
                    with self.subTest(turn=turn, browsing=browsing, ranked=ranked):
                        legacy = self.policy.width(turn, ranked, browsing)
                        explicit = self.policy.width(
                            turn,
                            ranked,
                            browsing,
                            confidence=self.policy.legacy_confidence(ranked),
                        )
                        self.assertEqual(explicit, legacy)

    def test_adding_a_constant_preserves_confidence_and_width(self):
        ranked = [(0.8, "a"), (0.51, "b"), (0.2, "c")]
        shifted = [(score + 1000.0, pid) for score, pid in ranked]
        confidence = self.policy.legacy_confidence(ranked)
        self.assertAlmostEqual(self.policy.legacy_confidence(shifted), confidence)
        self.assertEqual(
            self.policy.width(2, ranked, confidence=confidence),
            self.policy.width(2, shifted, confidence=confidence),
        )

    def test_explicit_confidence_is_invariant_to_rank_identical_rescaling(self):
        ranked = [(0.20, "a"), (0.19, "b"), (0.0, "c")]
        rescaled = [(score * 100.0, pid) for score, pid in ranked]
        self.assertEqual([pid for _, pid in ranked], [pid for _, pid in rescaled])

        # Canary: the legacy implicit margins really do cross width tiers.  If they
        # did not, this fixture could not detect accidental score-scale coupling.
        self.assertNotEqual(self.policy.width(2, ranked), self.policy.width(2, rescaled))

        confidence = self.policy.legacy_confidence(ranked)
        self.assertEqual(
            self.policy.width(2, ranked, confidence=confidence),
            self.policy.width(2, rescaled, confidence=confidence),
        )

    def test_explicit_zero_is_not_treated_as_an_absent_value(self):
        clear = [(1.0, "a"), (0.0, "b")]
        self.assertNotEqual(
            self.policy.width(2, clear),
            self.policy.width(2, clear, confidence=0.0),
        )


class AgentConfidencePlumbingTest(unittest.TestCase):
    def test_agent_passes_the_declared_legacy_channel_explicitly(self):
        class RecordingPolicy:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = []

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def width(self, turn, ranked, browsing=False, *, confidence=None):
                self.calls.append((list(ranked), confidence))
                return self.wrapped.width(
                    turn, ranked, browsing, confidence=confidence
                )

        with RichCatalog() as path:
            agent = Agent(path)
            agent.ask_mode = "none"
            recorder = RecordingPolicy(agent.commit)
            agent.commit = recorder
            agent.reset("confidence", PROFILE)
            response = agent.respond(
                "confidence",
                "I'm looking for Accessories Belts. A key requirement is: leather.",
                1,
                10,
            )

        self.assertTrue(response["recommendations"])
        self.assertEqual(len(recorder.calls), 1)
        ranked, confidence = recorder.calls[0]
        self.assertIsNotNone(confidence)
        self.assertEqual(confidence, recorder.legacy_confidence(ranked))


if __name__ == "__main__":
    unittest.main()
