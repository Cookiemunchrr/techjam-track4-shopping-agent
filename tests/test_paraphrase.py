"""Paraphrase robustness.

The specification warns the private set may paraphrase the customer's utterances.
An agent that matches the organizer's exact templates scores well here and collapses
there; these tests exist to stop us drifting into that shape.
"""
from __future__ import annotations

import os
import unittest

from src.agent import Agent
from src.policy import TOP_K
from tests.fixtures import PROFILE, TempCatalog
from tools.paraphrase import Paraphraser, ParaphrasingAgent

CATALOG = "data/catalog.jsonl"
PUBLIC = "data/public_set.jsonl"

TEMPLATES = [
    "I'm looking for Accessories Belts. A key requirement is: 100% Leather.",
    "I'm looking for Accessories Belts, but I'm still exploring.",
    "For that, what matters is: Buckle closure; Imported.",
    "I don't have a preference for material; please use your judgment.",
    "Actually, ignore my earlier preference. What I need is: Suede.",
]


class ParaphraserTest(unittest.TestCase):
    def test_actually_changes_the_wording(self):
        rewriter = Paraphraser(level=2, seed=1)
        changed = sum(rewriter.rewrite(t) != t for t in TEMPLATES)
        self.assertGreaterEqual(changed, 4, "paraphraser barely altered the templates")

    def test_is_deterministic_for_a_seed(self):
        a = [Paraphraser(2, seed=5).rewrite(t) for t in TEMPLATES]
        b = [Paraphraser(2, seed=5).rewrite(t) for t in TEMPLATES]
        self.assertEqual(a, b)

    def test_preserves_the_category_and_the_constraint_at_low_levels(self):
        rewritten = Paraphraser(level=1, seed=2).rewrite(TEMPLATES[0])
        self.assertIn("Accessories Belts", rewritten)
        self.assertIn("100% Leather", rewritten)

    def test_level_zero_is_a_no_op(self):
        rewriter = Paraphraser(level=0, seed=0)
        for template in TEMPLATES:
            self.assertEqual(rewriter.rewrite(template), template)

    def test_handles_junk_without_crashing(self):
        rewriter = Paraphraser(level=3, seed=0)
        for message in ("", None, "!!!", "a" * 5000, "🙂"):
            self.assertIsInstance(rewriter.rewrite(message), (str, type(None)))


class AgentUnderParaphraseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.agent = Agent(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_still_extracts_the_category_from_paraphrased_openings(self):
        for opening in ("I need Accessories Belts.", "Can you find me Accessories Belts.",
                        "Show me Accessories Belts.", "hmm, i'd like accessories belts"):
            self.agent.reset("p", PROFILE)
            self.agent.respond("p", opening, 1, TOP_K)
            self.assertEqual(self.agent.sessions["p"].category, "accessories belts", msg=opening)

    def test_still_detects_correction_in_paraphrased_form(self):
        """After a stated change of intent, an item shown earlier must be offerable again.

        This is the behaviour that rescues intent-override sessions -- see
        analysis/ablations_v2.csv, where hard deletion costs 0.101.
        """
        for message in ("Never mind what I said. It has to be Suede.",
                        "I changed my mind - I specifically need Suede.",
                        "Scratch that. Must have Suede.",
                        "Actually, forget that. Must have Suede."):
            self.agent.reset("c", PROFILE)
            self.agent.respond("c", "I'm looking for Accessories Belts.", 1, TOP_K)
            state = self.agent.sessions["c"]
            calls = []
            state.rejection.on_correction = lambda: calls.append(1)
            self.agent.respond("c", message, 2, TOP_K)
            self.assertTrue(calls, msg=f"correction not registered: {message}")

    def test_an_ordinary_turn_does_not_clear_the_slate(self):
        self.agent.reset("o", PROFILE)
        first = self.agent.respond("o", "I'm looking for Accessories Belts.", 1, TOP_K)
        shown = {r["parent_asin"] for r in first["recommendations"]}
        state = self.agent.sessions["o"]
        calls = []
        state.rejection.on_correction = lambda: calls.append(1)
        self.agent.respond("o", "What matters is Buckle closure.", 2, TOP_K)
        self.assertFalse(calls, "ordinary turn wrongly treated as a correction")
        self.assertTrue(state.rejection.penalised & shown,
                        "rejections were forgotten without a stated correction")

    def test_wrapper_never_breaks_the_contract(self):
        wrapped = ParaphrasingAgent(self.agent, level=3, seed=0)
        wrapped.reset("w", PROFILE)
        for turn, template in enumerate(TEMPLATES, start=1):
            response = wrapped.respond("w", template, turn, TOP_K)
            self.assertIsInstance(response, dict)
            self.assertLessEqual(len(response["recommendations"]), TOP_K)


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class ParaphraseRegressionTest(unittest.TestCase):
    """End-to-end floors under paraphrase.

    Measured 2026-08-26: clean 0.937, L1 0.908, L2 0.897, L3 0.888.

    Note that this harness varies scaffolding vocabulary and almost nothing else --
    see tools/adversarial.py and the note at the top of tests/test_adversarial.py
    for why that makes it a weak robustness measure on its own. These floors guard
    against a parser regression; they are not evidence of robustness.

    For contrast, the template-matching probe agent collapses from 0.883 to 0.064
    at L1 -- below the provided BM25 baseline. Floors sit well under our measured
    numbers so honest improvements never trip them.
    """

    FLOOR = 0.82
    MAX_DROP = 0.06

    @classmethod
    def setUpClass(cls):
        from tools.bench import run
        cls.rows = {row["paraphrase_level"]: row for row in run(["full"], [0, 1, 3])}

    def test_survives_scaffolding_paraphrase(self):
        self.assertGreaterEqual(self.rows[1]["technical_score"], self.FLOOR)

    def test_survives_lossy_paraphrase_of_the_quoted_text(self):
        self.assertGreaterEqual(self.rows[3]["technical_score"], self.FLOOR)

    def test_degrades_gracefully_rather_than_collapsing(self):
        clean = self.rows[0]["technical_score"]
        for level in (1, 3):
            drop = clean - self.rows[level]["technical_score"]
            self.assertLess(drop, self.MAX_DROP,
                            f"L{level} lost {drop:.4f} -- the agent is template-dependent")


if __name__ == "__main__":
    unittest.main()
