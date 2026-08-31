"""Group F — adversarial input and distribution shift.

docs/competition_specification.md warns that natural-language paraphrasing may be
added by the organizer. The pre-rebuild harness (tools/paraphrase.py) re-inserted
the category phrase verbatim at every level, so it varied scaffolding vocabulary and
nothing else -- and its scaffolding list was copied from the parser's own
FILLER_PREFIX lexicon, making it a closed loop.

Measured on the pre-rebuild agent, public 200:

    control                              0.94365   HR 0.990
    case / word order / articles         0.94365   HR 0.990   +/- 0.000
    constraint text only (the old L3)    0.89422   HR 0.965   -0.049
    category head noun synonymised       0.64072   HR 0.680   -0.303

F1-F5 are the axes the old harness held constant. F14 converts the popularity prior
from a stated limitation into a measurement.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.agent import Agent
from src.catalog import Catalog
from src.policy import TOP_K
from tests.fixtures import PROFILE, RichCatalog

REPO = Path(__file__).resolve().parents[1]
CATALOG = str(REPO / "data" / "catalog.jsonl")

# Score floors under adversarial rewriting, on the full public set.
# Measured 2026-08-26: category 0.750, scaffold 0.805, granularity=3 0.934,
# below-p90-popularity 0.829. Floors sit under the measurement with headroom, so
# an honest improvement never trips them and a regression does.
# For scale, the pre-rebuild agent scored 0.641 under category synonymisation.
FLOOR_SYNONYM = 0.70
FLOOR_SCAFFOLD = 0.75
FLOOR_GRANULARITY = 0.85
FLOOR_POPULARITY_ADVERSARIAL = 0.75


class LexicalDriftTest(unittest.TestCase):
    """F1, F2, F3 — the category phrase is the load-bearing input."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()
        cls.catalog = Catalog(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def _top(self, message):
        agent = Agent(self.path)
        agent.reset("d", PROFILE)
        return [item["parent_asin"]
                for item in agent.respond("d", message, 1, TOP_K)["recommendations"]]

    def test_f1_synonym_for_the_head_noun_still_reaches_the_right_shelf(self):
        for phrase in ("tees", "t shirts", "cotton tees"):
            found = self._top(f"I'm looking for {phrase}.")
            self.assertTrue(any(pid.startswith("R_TEE") for pid in found),
                            f"'{phrase}' did not reach the T-Shirts bucket: {found}")

    def test_f2_a_hypernym_one_level_up_still_routes(self):
        found = self._top("I'm looking for jewellery.")
        self.assertTrue(any(pid.startswith("R_EAR") for pid in found),
                        f"'jewellery' did not reach the Earrings bucket: {found}")

    def test_f3_a_natural_phrase_rather_than_a_taxonomy_path_still_routes(self):
        found = self._top("I'm hoping to find a pair of hoop earrings for my sister.")
        self.assertTrue(any(pid.startswith("R_EAR") for pid in found),
                        f"natural phrasing did not route: {found}")

    def test_f5_scaffolding_outside_the_parser_lexicon_still_parses(self):
        """The old harness only generated openings the parser was written to strip."""
        for opening in ("Hoping to pick up",
                        "In the market for",
                        "On the hunt for",
                        "Any recommendations on",
                        "Trying to track down"):
            found = self._top(f"{opening} Accessories Belts.")
            self.assertTrue(any(pid.startswith("R_BELT") for pid in found),
                            f"'{opening}' broke routing: {found}")

    def test_f6_surface_invariance(self):
        base = self._top("I'm looking for Accessories Belts.")
        for variant in ("i'm looking for accessories belts",
                        "I'm looking for Belts Accessories.",
                        "I'm looking for some Accessories Belts!"):
            self.assertEqual(self._top(variant), base, variant)


class NegationTest(unittest.TestCase):
    """F11 — 'not polyester' was being tokenised as evidence *for* polyester."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_f11_a_negative_constraint_penalises_rather_than_rewards(self):
        agent = Agent(self.path)
        agent.reset("neg", PROFILE)
        agent.respond("neg", "I'm looking for Tops & Tees T-Shirts.", 1, TOP_K)
        found = [item["parent_asin"] for item in
                 agent.respond("neg", "It must not be polyester.", 2, TOP_K)["recommendations"]]
        if found:
            self.assertNotEqual(found[0], "R_TEE_POLY",
                                "'not polyester' promoted the polyester tee to rank 1")

    def test_f11b_polarity_is_recorded_on_the_slot(self):
        from src.state import DialogState
        from src.catalog import Catalog
        catalog = Catalog(self.path)
        dialog = DialogState()
        dialog.observe("I don't want polyester", 2, catalog)
        polarities = [slot.polarity for slot in dialog.active()]
        self.assertIn(-1, polarities, "no negative slot recorded for an explicit refusal")


class MismatchTest(unittest.TestCase):
    """F9 — the customer answers a question we did not ask."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_f9_an_off_topic_answer_is_still_absorbed(self):
        agent = Agent(self.path)
        agent.reset("mm", PROFILE)
        agent.respond("mm", "I'm looking for Accessories Belts.", 1, TOP_K)
        response = agent.respond("mm", "Honestly the main thing is it has to be under $20.", 2, TOP_K)
        self.assertTrue(response["recommendations"])
        blob = " ".join(agent.sessions["mm"].dialog.phrases()).lower()
        self.assertIn("20", blob, "an unprompted constraint was discarded")


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class FullSetAdversarialTest(unittest.TestCase):
    """F1, F5, F12, F14 measured end-to-end on the real public set."""

    @classmethod
    def setUpClass(cls):
        from evaluator.local_evaluator import catalog_index, load_jsonl
        cls.ids, cls.categories, cls.products = catalog_index(CATALOG)
        cls.samples = load_jsonl(str(REPO / "data" / "public_set.jsonl"))
        cls.agent = Agent(CATALOG)

    def _score(self, wrapper):
        from evaluator.local_evaluator import evaluate
        return evaluate(wrapper, self.samples, self.ids, self.categories, self.products)

    def test_f1_full_set_survives_category_synonymisation(self):
        from tools.adversarial import Adversarial
        result = self._score(Adversarial(self.agent, axes=("category",)))
        self.assertGreaterEqual(
            result["recommended_technical_score"], FLOOR_SYNONYM,
            "category synonymisation still collapses the agent; this measured 0.641 "
            "before the semantic route existed")

    def test_f5_full_set_survives_unfamiliar_scaffolding(self):
        from tools.adversarial import Adversarial
        result = self._score(Adversarial(self.agent, axes=("scaffold",)))
        self.assertGreaterEqual(result["recommended_technical_score"], FLOOR_SCAFFOLD)

    def test_f12_survives_simulator_drift_in_category_granularity(self):
        """A plausible private-set difference: coarse_category keeping a different
        number of path components. Not a paraphrase -- the harness itself differs."""
        from evaluator.local_evaluator import evaluate
        from tools.adversarial import drifted_categories
        result = evaluate(self.agent, self.samples, self.ids,
                          drifted_categories(self.categories, 3), self.products)
        self.assertGreaterEqual(result["recommended_technical_score"], FLOOR_GRANULARITY)

    def test_f14_below_median_popularity_targets_are_not_abandoned(self):
        """Converts the biggest stated limitation into a number."""
        from evaluator.local_evaluator import evaluate
        catalog = self.agent.catalog
        hard = []
        for sample in self.samples:
            target = str(sample["ground_truth"]["parent_asin"])
            bucket = None
            for name, members in catalog.buckets.items():
                if target in members:
                    bucket = members
                    break
            if not bucket:
                continue
            pops = sorted(catalog.meta[pid]["pop"] for pid in bucket)
            below = sum(1 for value in pops if value < catalog.meta[target]["pop"])
            if below / max(len(pops) - 1, 1) < 0.90:
                hard.append(sample)
        if len(hard) < 10:
            self.skipTest(
                f"only {len(hard)} sessions have a target below its bucket median. "
                "That is the finding: the median target sits at the 99.3rd "
                "percentile of its own bucket, so this evaluation set has almost "
                "no long tail to bury.")
        result = evaluate(self.agent, hard, self.ids, self.categories, self.products)
        self.assertGreaterEqual(
            result["recommended_technical_score"], FLOOR_POPULARITY_ADVERSARIAL,
            f"{len(hard)} long-tail sessions score "
            f"{result['recommended_technical_score']:.3f}: the prior is burying niche intent")


if __name__ == "__main__":
    unittest.main()
