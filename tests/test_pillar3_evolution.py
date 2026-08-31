"""Group D — Pillar III: self-evolution and dynamic context programming.

Nothing in the pre-rebuild agent addressed this pillar. `reset()` accepted
`user_profile` and discarded it; the pipeline was a fixed parse -> route -> score ->
commit -> elicit on every turn of every session.

The brief asks for "Personalized Context Distillation", "continuously updating
short-term session states and long-term user profiles", and "runtime workflow
re-orchestration and strategy alignment".

D8 matters more than the features it guards: re-orchestration that degrades the
sessions which already work is worse than no re-orchestration at all.
"""
from __future__ import annotations

import unittest

from src.agent import Agent
from src.catalog import Catalog
from src.policy import TOP_K
from tests.fixtures import PROFILE, RichCatalog

OTHER_PROFILE = {
    "purchase_frequency": "10+ prior purchases", "average_prior_rating": 2.0,
    "rating_style": "critical", "preference_tags": ["durability", "waterproof"],
    "summary": "Prior purchases emphasize durability, waterproof; ratings are critical.",
}


class EvolutionBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()
        cls.catalog = Catalog(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)


class DistillationTest(EvolutionBase):
    """D1, D2 — compress the history without changing what it means."""

    def test_d1_distilled_state_ranks_within_epsilon_of_the_full_history(self):
        """Distillation must lose detail, not meaning.

        Mirrors production: the category is held on the state, not as a constraint
        slot, and ranking happens inside the routed pool.
        """
        from src.scoring import Scorer
        from src.state import DialogState
        dialog = DialogState()
        dialog.category = "accessories belts"
        turns = ["Leather", "brown", "buckle closure", "for work",
                 "under thirty dollars", "full grain", "durable", "imported"]
        for turn, text in enumerate(turns, start=1):
            dialog.observe(text, turn, self.catalog)
        pool = self.catalog.buckets["Accessories Belts"]
        scorer = Scorer(self.catalog)
        full = [pid for _, pid in scorer.rank(pool, dialog.weighted_phrases())]
        dialog.distil(cap=4)
        after = [pid for _, pid in scorer.rank(pool, dialog.weighted_phrases())]
        self.assertEqual(full[0], after[0],
                         f"distillation changed the best candidate: {full} vs {after}")
        self.assertGreaterEqual(len(set(full[:3]) & set(after[:3])), 2,
                                f"distillation reordered the head: {full} vs {after}")

    def test_d15_unmatchable_text_does_not_outrank_a_stated_material(self):
        """IDF peaks at df == 0, so words absent from the catalog used to score as
        maximally informative and survived distillation ahead of the material the
        shopper actually stated."""
        from src.state import _information
        stated = _information("Leather", self.catalog)
        noise = _information("under thirty dollars", self.catalog)
        self.assertGreater(stated, noise,
                           "unmatchable phrasing scored as more informative than "
                           "a material the catalog actually contains")

    def test_d2_distilled_state_stays_bounded_as_turns_grow(self):
        from src.state import DialogState
        dialog = DialogState()
        for turn in range(1, 40):
            dialog.observe(f"constraint number {turn} matters to me", turn, self.catalog)
            dialog.distil(cap=12)
        self.assertLessEqual(len(dialog.phrases()), 12, "distilled state is unbounded")


class ProfilePriorTest(EvolutionBase):
    """D3, D4 — personalization that is real but can never take the wheel."""

    def test_d3_a_matching_preference_tag_lifts_an_otherwise_equal_rival(self):
        from src.profile import ProfilePrior
        prior = ProfilePrior({"preference_tags": ["leather", "durability"],
                              "average_prior_rating": 5.0})
        with_tag = prior.bonus(self.catalog, "R_BELT_LEATHER")
        without = prior.bonus(self.catalog, "R_BELT_CANVAS")
        self.assertGreater(with_tag, without, "profile tags have no measurable effect")

    def test_d4_profile_influence_can_never_outrank_a_stated_constraint(self):
        from src.profile import ProfilePrior
        from src.scoring import Scorer, Weights
        prior = ProfilePrior({"preference_tags": ["leather"], "average_prior_rating": 5.0})
        weights = Weights()
        strongest = max(prior.bonus(self.catalog, pid) for pid in self.catalog.ids)
        self.assertLessEqual(
            strongest, weights.phrase,
            "the profile prior can outweigh a verbatim stated constraint")
        self.assertLessEqual(strongest, ProfilePrior.CAP)

    def test_d3b_an_empty_or_malformed_profile_is_harmless(self):
        from src.profile import ProfilePrior
        for profile in ({}, None, {"preference_tags": None}, {"preference_tags": [1, 2]}):
            prior = ProfilePrior(profile)
            self.assertIsInstance(prior.bonus(self.catalog, "R_BELT_LEATHER"), float)


class LongTermMemoryTest(EvolutionBase):
    """D5 — survives reset for the same profile, never leaks across profiles."""

    def test_d5_memory_persists_across_reset_for_the_same_profile(self):
        from src.memory import LongTermMemory
        memory = LongTermMemory()
        signature = memory.signature(PROFILE)
        memory.observe(signature, bucket="Accessories Belts", attribute="material")
        self.assertEqual(memory.signature(PROFILE), signature)
        self.assertIn("Accessories Belts", memory.recall(signature).get("buckets", {}))

    def test_d5b_memory_never_leaks_across_different_profiles(self):
        from src.memory import LongTermMemory
        memory = LongTermMemory()
        mine = memory.signature(PROFILE)
        theirs = memory.signature(OTHER_PROFILE)
        self.assertNotEqual(mine, theirs)
        memory.observe(mine, bucket="Accessories Belts", attribute="material")
        self.assertEqual(memory.recall(theirs).get("buckets", {}), {})

    def test_d5c_the_agent_keeps_memory_across_sessions(self):
        agent = Agent(self.path)
        agent.reset("m1", PROFILE)
        agent.respond("m1", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.reset("m2", PROFILE)
        self.assertTrue(hasattr(agent, "memory"), "no long-term memory layer on the agent")
        signature = agent.memory.signature(PROFILE)
        self.assertTrue(agent.memory.recall(signature),
                        "nothing was distilled into long-term memory")


class LongTermConsumptionTest(EvolutionBase):
    """W1 — the loop closes: recall() is read by ranking and by elicitation.

    The harness simulates isolated single-user sessions, so these tests build the
    two-session situation directly: session A teaches the store, session B for the
    same profile signature must see it, and a different profile must not.
    """

    def test_w1a_recalled_shelf_lifts_a_tied_candidate_inside_cap(self):
        from src.profile import BUCKET_BONUS, CAP, ProfilePrior
        prior = ProfilePrior({}, {"Accessories Belts": 1})
        lifted = prior.bonus(self.catalog, "R_BELT_LEATHER")
        plain = ProfilePrior({}).bonus(self.catalog, "R_BELT_LEATHER")
        self.assertAlmostEqual(lifted - plain, BUCKET_BONUS,
                               msg="recalled shelf affinity did not fire")
        self.assertLessEqual(lifted, CAP, "the guarantee broke: prior exceeds CAP")

    def test_w1a_a_tie_breaks_toward_the_recalled_shelf(self):
        from src.profile import ProfilePrior, break_ties
        tied = [(1.00, "R_SCARF_SILK"), (1.00, "R_BELT_LEATHER")]
        fresh = break_ties(list(tied), ProfilePrior({}), self.catalog)
        recalled = break_ties(list(tied), ProfilePrior({}, {"Accessories Belts": 2}),
                              self.catalog)
        self.assertEqual([pid for _, pid in fresh], [pid for _, pid in tied])
        self.assertEqual(recalled[0][1], "R_BELT_LEATHER",
                         "the recalled shelf did not settle the tie")

    def test_w1b_recalled_answer_history_seeds_answerability(self):
        from src.answerability import AnswerModel
        fresh = AnswerModel()
        seeded = AnswerModel()
        seeded.set_prior({"material": 5}, {"material": 5}, sessions=10)
        self.assertEqual(seeded.probability("material"), fresh.probability("material"),
                         "the measured-value path must stay the Beta it always was")
        self.assertGreater(seeded.prior_probability("material"),
                           fresh.prior_probability("material"),
                           "recalled answer history did not move the long-term prior")
        seeded.set_prior({"budget": 5}, {"budget": 0}, sessions=10)
        self.assertLess(seeded.prior_probability("budget"),
                        fresh.prior_probability("budget"),
                        "recalled dead-question history did not lower the prior")

    def test_w1b_in_session_evidence_overrides_the_prior(self):
        from src.answerability import PRIOR_STRENGTH, AnswerModel
        model = AnswerModel()
        model.set_prior({"brand": 8}, {"brand": 8}, sessions=10)   # prior says yes
        for _ in range(int(PRIOR_STRENGTH) * 4):
            model.observe("brand", False)                          # session says no
        prior_only = AnswerModel()
        prior_only.set_prior({"brand": 8}, {"brand": 8}, sessions=10)
        self.assertLess(model.prior_probability("brand"),
                        prior_only.prior_probability("brand"),
                        "sixteen real observations did not outweigh four pseudo-ones")

    def test_w1b_the_prior_settles_a_near_tie_and_only_a_near_tie(self):
        from src.answerability import AnswerModel
        from src.elicitation import ASK_TIE_EPSILON, choose

        class FixedModel(AnswerModel):
            def __init__(self, prior):
                super().__init__()
                self._p = prior

            def probability(self, attribute):
                return 0.5

            def has_prior(self):
                return True

            def prior_probability(self, attribute):
                return 0.9 if attribute == self._p else 0.1

        class FakeCatalog:
            def facet(self, pid, attribute):
                return None

        ranked = [(1.0, "R_BELT_LEATHER"), (0.9, "R_SCARF_SILK")]
        # A decisive gap wins regardless of the prior: category leads other by far.
        picked = choose(FakeCatalog(), ranked, set(), FixedModel("other"),
                        extra={"category": 0.60, "other": 0.40})
        self.assertEqual(picked, "category", "the prior overruled a decisive value")
        # An exact tie between two measured values goes to the recalled attribute.
        picked = choose(FakeCatalog(), ranked, set(), FixedModel("other"),
                        extra={"category": 0.60, "other": 0.60})
        self.assertEqual(picked, "other", "the prior failed to settle an exact tie")
        # A tie against an assumed value is a placeholder, not an equivalence:
        # the prior does not touch it -- loop order decides, as it always has.
        picked = choose(FakeCatalog(), ranked, set(), FixedModel("size"),
                        extra={"category": 0.60})
        self.assertEqual(picked, "category",
                         "the prior disturbed an assumed-value ordering")

    def test_w1_first_ever_session_is_exactly_todays_behaviour(self):
        agent = Agent(self.path)
        agent.reset("first", PROFILE)
        state = agent.sessions["first"]
        self.assertEqual(state.profile.recalled_buckets, {})
        self.assertFalse(agent.answers.has_prior())
        from src.answerability import AnswerModel
        self.assertEqual(agent.answers.prior_probability("material"),
                         AnswerModel().prior_probability("material"))

    def test_w1_session_b_consumes_what_session_a_taught(self):
        agent = Agent(self.path)
        agent.reset("a", PROFILE)
        out = agent.respond("a", "I'm looking for Accessories Belts.", 1, TOP_K)
        asked = out.get("ask_attribute")
        self.assertIsNotNone(asked, "fixture dynamics changed: the agent asked nothing")
        agent.respond("a", "Leather, please.", 2, TOP_K)
        control = Agent.sharing_index(agent)      # same index, fresh memory
        agent.reset("b", PROFILE)
        control.reset("b", PROFILE)
        state_b, control_b = agent.sessions["b"], control.sessions["b"]
        self.assertTrue(state_b.profile.recalled_buckets,
                        "session B recalled nothing: the loop is not closed")
        self.assertEqual(control_b.profile.recalled_buckets, {})
        self.assertTrue(agent.answers.has_prior())
        self.assertFalse(control.answers.has_prior())
        self.assertNotEqual(agent.answers.prior_probability(asked),
                            control.answers.prior_probability(asked),
                            "question selection input identical with and without memory")

    def test_w1_no_leakage_across_profiles(self):
        agent = Agent(self.path)
        agent.reset("a", PROFILE)
        agent.respond("a", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.reset("other", OTHER_PROFILE)
        state = agent.sessions["other"]
        self.assertEqual(state.profile.recalled_buckets, {},
                         "one profile's history leaked into another's ranking")
        self.assertFalse(agent.answers.has_prior(),
                         "one profile's answer history leaked into another's prior")


class ReOrchestrationTest(EvolutionBase):
    """D6, D7, D8 — adapt when stalled, and only when stalled."""

    def test_d6_two_zero_yield_turns_switch_strategy(self):
        from src.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        self.assertEqual(orchestrator.strategy(), "default")
        orchestrator.observe(informative=False)
        orchestrator.observe(informative=False)
        self.assertNotEqual(orchestrator.strategy(), "default",
                            "two dead turns and the agent did not change anything")

    def test_d7_repeated_failure_flips_to_cover_mode_before_the_turn_limit(self):
        from src.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        for _ in range(4):
            orchestrator.observe(informative=False)
        self.assertEqual(orchestrator.strategy(), "cover")

    def test_d8_re_orchestration_is_a_no_op_on_healthy_sessions(self):
        from src.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        for _ in range(6):
            orchestrator.observe(informative=True)
        self.assertEqual(orchestrator.strategy(), "default",
                         "a healthy session was re-orchestrated anyway")

    def test_d8b_one_informative_turn_resets_the_stall_counter(self):
        from src.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        orchestrator.observe(informative=False)
        orchestrator.observe(informative=True)
        self.assertEqual(orchestrator.strategy(), "default")

    def test_d6b_the_agent_detects_a_stall_end_to_end(self):
        agent = Agent(self.path)
        agent.reset("st", PROFILE)
        agent.respond("st", "I'm looking for Accessories Belts.", 1, TOP_K)
        for turn in (2, 3, 4):
            agent.respond("st", "I don't have a preference for that.", turn, TOP_K)
        self.assertNotEqual(agent.sessions["st"].orchestrator.strategy(), "default",
                            "three dead turns and the agent kept the same plan")


class ExplainabilityTest(EvolutionBase):
    """D9 — "transparent recommendation explanations" is a listed innovation direction."""

    def test_d9_a_recommendation_can_name_the_signals_that_lifted_it(self):
        from src.scoring import Scorer
        scorer = Scorer(self.catalog)
        breakdown = scorer.explain("R_BELT_LEATHER", ["Leather", "black"])
        self.assertIsInstance(breakdown, dict)
        for term in ("popularity", "lexical", "phrase"):
            self.assertIn(term, breakdown)


if __name__ == "__main__":
    unittest.main()
