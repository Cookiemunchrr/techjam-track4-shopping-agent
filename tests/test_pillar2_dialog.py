"""Group C — Pillar II: the multi-turn state machine.

The brief names two behaviours explicitly: "Information Accumulation (incremental
slots)" and "abrupt Intent Override (slot erasure and rewriting)", plus a retrieval
cutoff on "Over-Generality (candidate pool overload)".

Before the rebuild the agent accumulated a flat list of clauses and never erased
anything; `state.category` was assigned only under `if turn == 1`, so a customer who
changed their mind kept getting the original category. This transcript was real:

    T1  "I'm looking for Men Hoodies. A key requirement is: cotton."   -> a hoodie
    T3  "Actually, forget that. What I need is: a leather belt."       -> the same hoodie
    state after T4: ['cotton', 'navy blue', 'Actually, forget that',
                     'a leather belt', 'full grain leather']

C4-C7 are that transcript, turned into assertions.
"""
from __future__ import annotations

import unittest

from src.agent import Agent
from src.catalog import Catalog
from src.policy import TOP_K
from src.text import is_correction, split_clauses
from tests.fixtures import PROFILE, RichCatalog


def ids(response) -> list[str]:
    return [item["parent_asin"] for item in response["recommendations"]]


class DialogBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()
        cls.catalog = Catalog(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def fresh(self, session="s"):
        agent = Agent(self.path)
        agent.reset(session, PROFILE)
        return agent

    def state(self, agent, session="s"):
        return agent.sessions[session]


class AccumulationTest(DialogBase):
    def test_c1_constraints_accumulate_across_turns(self):
        agent = self.fresh()
        agent.respond("s", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.respond("s", "For that, what matters is: Suede.", 2, TOP_K)
        agent.respond("s", "For that, what matters is: brown.", 3, TOP_K)
        blob = " ".join(self.state(agent).dialog.phrases()).lower()
        self.assertIn("suede", blob)
        self.assertIn("brown", blob)

    def test_c2_a_declined_turn_adds_no_constraint(self):
        agent = self.fresh()
        agent.respond("s", "I'm looking for Accessories Belts.", 1, TOP_K)
        before = list(self.state(agent).dialog.phrases())
        agent.respond("s", "I don't have a preference for colour; please use your judgment.", 2, TOP_K)
        self.assertEqual(list(self.state(agent).dialog.phrases()), before)

    def test_c3_a_repeated_constraint_is_not_double_counted(self):
        agent = self.fresh()
        agent.respond("s", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.respond("s", "For that, what matters is: Suede.", 2, TOP_K)
        once = list(self.state(agent).dialog.phrases())
        agent.respond("s", "For that, what matters is: Suede.", 3, TOP_K)
        self.assertEqual(list(self.state(agent).dialog.phrases()), once)


class OverrideTest(DialogBase):
    """C4-C7 — the behaviour the brief names and the old agent did not have."""

    def _override_session(self):
        agent = self.fresh("ov")
        agent.respond("ov", "I'm looking for Men Fashion Hoodies & Sweatshirts. "
                            "A key requirement is: Cotton.", 1, TOP_K)
        agent.respond("ov", "For that, what matters is: navy.", 2, TOP_K)
        after = agent.respond("ov", "Actually, forget that. What I need is: "
                                    "Accessories Belts, leather.", 3, TOP_K)
        return agent, after

    def test_c4_the_retracted_constraint_stops_influencing_the_ranking(self):
        agent, _ = self._override_session()
        active = " ".join(agent.sessions["ov"].dialog.phrases()).lower()
        self.assertNotIn("navy", active,
                         "a superseded preference is still weighted as active evidence")

    def test_c5_an_override_to_another_category_re_routes_the_pool(self):
        agent, after = self._override_session()
        returned = ids(after)
        self.assertTrue(returned, "override turn returned nothing")
        self.assertTrue(
            any(pid.startswith("R_BELT") for pid in returned),
            f"still recommending the old category after an override: {returned}")

    def test_c6_the_correction_phrase_is_never_stored_as_a_constraint(self):
        agent, _ = self._override_session()
        for phrase in agent.sessions["ov"].dialog.phrases():
            lowered = phrase.lower()
            for cue in ("actually", "forget that", "never mind", "changed my mind"):
                self.assertNotIn(cue, lowered,
                                 f"correction cue stored as evidence: {phrase!r}")

    def test_c7_an_item_shown_before_the_override_is_not_immediately_re_offered(self):
        agent = self.fresh("re")
        first = ids(agent.respond("re", "I'm looking for Accessories Belts. "
                                        "A key requirement is: Leather.", 1, TOP_K))
        after = ids(agent.respond("re", "Actually, forget that. What I need is: Suede.", 2, TOP_K))
        if first and after:
            self.assertNotEqual(first[0], after[0],
                                "re-offered the exact item the customer just rejected")

    def test_c8_override_detection_fires_on_natural_phrasings(self):
        for phrase in ("Actually, I'd rather have leather.",
                       "Never mind, show me something else.",
                       "I changed my mind about the colour.",
                       "Forget that, let's try wool.",
                       "On second thought, something cheaper.",
                       "Scratch that."):
            self.assertTrue(is_correction(phrase), phrase)

    def test_c9_override_detection_does_not_fire_on_ordinary_turns(self):
        for phrase in ("For that, what matters is: 100% cotton.",
                       "I need a belt for work.",
                       "Something in brown, please."):
            self.assertFalse(is_correction(phrase), phrase)

    def test_c10b_a_contradiction_without_a_cue_still_supersedes(self):
        """F10 in the catalogue -- a real customer does not say 'actually'."""
        agent = self.fresh("cx")
        agent.respond("cx", "I'm looking for Accessories Belts. A key requirement is: Cotton.", 1, TOP_K)
        agent.respond("cx", "For that, what matters is: Leather.", 2, TOP_K)
        active = " ".join(agent.sessions["cx"].dialog.phrases()).lower()
        self.assertNotIn("cotton", active,
                         "two conflicting materials are both being scored as active")


class ErasureModeTest(DialogBase):
    """W3 — literal slot erasure as a measured mode beside the decay default.

    Pillar II names "slot erasure and rewriting". decay (default) keeps a
    SUPERSEDED_WEIGHT trace; erase removes the slot from the structure, so its
    text cannot reach BM25, the phrase channel, budget() or facets().
    """

    def _rewrite(self, mode):
        from src.state import DialogState
        dialog = DialogState(mode)
        dialog.observe("A key requirement is: cotton.", 1, self.catalog)
        dialog.observe("Actually, forget that. What I need is: leather.", 2,
                       self.catalog)
        return dialog

    def test_w3_decay_keeps_a_superseded_trace_by_default(self):
        dialog = self._rewrite("decay")
        cotton = [s for s in dialog.slots if "cotton" in s.text.lower()]
        self.assertTrue(cotton and all(s.superseded for s in cotton),
                        "decay mode lost or failed to mark the replaced slot")
        weighted = dict(dialog.weighted_phrases())
        self.assertTrue(any("cotton" in text.lower() for text in weighted),
                        "decay mode dropped the trace it exists to keep")

    def test_w3_erase_removes_the_slot_from_the_structure(self):
        dialog = self._rewrite("erase")
        self.assertFalse([s for s in dialog.slots if "cotton" in s.text.lower()],
                         "erase mode left the replaced slot in the structure")
        blob = " ".join(text for text, _ in dialog.weighted_phrases()).lower()
        self.assertNotIn("cotton", blob, "erased text still reaches scoring")
        self.assertIn("leather", blob)

    def test_w3_erase_applies_to_restart_and_product_reset(self):
        from src.state import DialogState
        dialog = DialogState("erase")
        dialog.observe("A key requirement is: cotton.", 1, self.catalog)
        dialog.observe("brown, please.", 2, self.catalog)
        dialog.product_reset(3)
        self.assertFalse([s for s in dialog.slots
                          if s.attribute in dialog.PRODUCT_BOUND],
                         "product_reset left product-bound slots behind in erase mode")
        dialog.restart(4)
        self.assertFalse(dialog.slots, "restart left slots behind in erase mode")

    def test_w3_agent_level_erasure_and_fail_closed_default(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"P_SUPERSEDE": "erase"}):
            agent = Agent(self.path)
        agent.reset("e", PROFILE)
        agent.respond("e", "I'm looking for Men Fashion Hoodies & Sweatshirts. "
                           "A key requirement is: Cotton.", 1, TOP_K)
        agent.respond("e", "Actually, forget that. What I need is: leather.", 2, TOP_K)
        slots = agent.sessions["e"].dialog.slots
        self.assertFalse([s for s in slots if "cotton" in s.text.lower()],
                         "P_SUPERSEDE=erase did not reach the live dialog state")
        with mock.patch.dict(os.environ, {"P_SUPERSEDE": "nonsense"}):
            agent = Agent(self.path)
        self.assertEqual(agent.supersede_mode, "decay",
                         "an unrecognised mode must fail closed to the default")


class ProactiveGuidanceTest(DialogBase):
    """C10, C11 — the over-generality cutoff."""

    def test_c10_an_overloaded_pool_triggers_a_cutoff(self):
        from src.policy import CommitPolicy
        policy = CommitPolicy()
        self.assertTrue(policy.cutoff(5000), "no cutoff on an overloaded pool")
        self.assertFalse(policy.cutoff(6), "cutoff firing on an already-narrow pool")

    def test_c10c_the_agent_withholds_and_asks_when_the_pool_is_huge(self):
        agent = self.fresh("wide")
        response = agent.respond("wide", "I'm looking for something, I'm still exploring.", 1, TOP_K)
        self.assertIsNotNone(response["ask_attribute"],
                             "over-general opening turn produced no clarification")

    def test_c11_the_prose_and_the_structured_attribute_agree(self):
        agent = self.fresh("pr")
        for turn in range(1, 6):
            response = agent.respond("pr", "I'm looking for Accessories Belts.", turn, TOP_K)
            attribute = response["ask_attribute"]
            if attribute and attribute not in ("other", "feature"):
                self.assertIn(attribute.replace("_", " ").split()[0],
                              response["message"].lower(),
                              f"ask_attribute={attribute!r} not named in {response['message']!r}")


class PreRetrievalCutoffTest(DialogBase):
    """W4 — the cutoff predicts overload from bucket sizes before scoring.

    One gate predicts (CommitPolicy.pre_cutoff), one confirms
    (CommitPolicy.cutoff). When the prediction fires, the full scoring pass is
    skipped and a bounded representative sample is scored instead.
    """

    def test_w4_the_estimate_mirrors_the_route_exactly(self):
        from src.routing import estimated_pool_size, route_detail
        for intent in (None, "browsing", "buying"):
            estimate = estimated_pool_size(self.catalog, None,
                                           "Accessories Belts", intent)
            pool, _, _ = route_detail(self.catalog, None, "Accessories Belts", intent)
            self.assertEqual(estimate, len(pool),
                             f"estimate and route disagree under intent={intent}")

    def test_w4_the_gate_requires_breadth_and_no_constraints(self):
        from src.policy import CommitPolicy
        policy = CommitPolicy()
        self.assertTrue(policy.pre_cutoff(5000, has_constraints=False))
        self.assertFalse(policy.pre_cutoff(5000, has_constraints=True),
                        "a filed constraint must keep the full scoring pass")
        self.assertFalse(policy.pre_cutoff(6, has_constraints=False))

    def test_w4_the_full_scoring_pass_is_skipped_when_it_fires(self):
        import os
        from unittest import mock
        import src.agent as agent_mod
        calls = []
        with mock.patch.dict(os.environ, {"P_OVERLOAD": "3"}), \
                mock.patch.object(agent_mod, "PRE_CUTOFF_SAMPLE", 2):
            agent = Agent(self.path)
            real_rank = agent.scorer.rank
            agent.scorer.rank = lambda pool, *a, **k: (calls.append(len(pool)),
                                                       real_rank(pool, *a, **k))[1]
            agent.reset("wide", PROFILE)
            out = agent.respond("wide", "I'm looking for Accessories Belts, "
                                        "but I'm still exploring.", 1, TOP_K)
        self.assertEqual(calls, [2],
                         f"scoring pass not bounded to the sample: {calls}")
        self.assertIsNotNone(out.get("ask_attribute"),
                             "the cutoff must go straight to a question")

    def test_w4_a_filed_constraint_keeps_the_full_pass(self):
        import os
        from unittest import mock
        import src.agent as agent_mod
        calls = []
        with mock.patch.dict(os.environ, {"P_OVERLOAD": "3"}), \
                mock.patch.object(agent_mod, "PRE_CUTOFF_SAMPLE", 2):
            agent = Agent(self.path)
            real_rank = agent.scorer.rank
            agent.scorer.rank = lambda pool, *a, **k: (calls.append(len(pool)),
                                                       real_rank(pool, *a, **k))[1]
            agent.reset("narrow", PROFILE)
            agent.respond("narrow", "I'm looking for Accessories Belts. "
                                    "A key requirement is: Leather.", 1, TOP_K)
        self.assertGreater(calls[0], 2,
                           "a stated constraint should keep the full pool")

    def test_w4_the_trace_records_the_prediction(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"P_OVERLOAD": "3"}):
            agent = Agent(self.path)
        agent.trace_route = True
        agent.reset("t", PROFILE)
        agent.respond("t", "I'm looking for Accessories Belts, "
                           "but I'm still exploring.", 1, TOP_K)
        trace = agent.route_trace("t")
        self.assertTrue(trace.get("pre_cutoff"), "prediction missing from the trace")
        self.assertIn("estimated_pool_size", trace)


class ElicitationQualityTest(DialogBase):
    """C12-C15 — ask questions that can be answered."""

    def test_c12_never_repeats_a_question_in_one_session(self):
        agent = self.fresh("q")
        asked = []
        for turn in range(1, 11):
            attribute = agent.respond("q", "Tell me more about belts.", turn, TOP_K)["ask_attribute"]
            if attribute:
                asked.append(attribute)
        self.assertEqual(len(asked), len(set(asked)), f"repeated a question: {asked}")

    def test_c13_the_agent_converges_away_from_unanswerable_questions(self):
        """`brand` was asked 130x across 200 sessions for a yield of exactly zero.

        No catalog-derived criterion can avoid it -- see src/answerability.py -- so
        the agent learns it from evidence instead. An online learner must therefore
        be *allowed* to ask it a few times; what it must not do is keep leading with
        it. Counts only the opening question of each session, because once the
        productive attributes are exhausted a bad question is the best one left.
        """
        from src.agent import Agent
        agent = Agent(self.path)
        opening = []
        for index in range(60):
            session = f"c13_{index}"
            agent.reset(session, PROFILE)
            opening.append(agent.respond(
                session, "I'm looking for Accessories Belts.", 1, TOP_K)["ask_attribute"])
        early = opening[:20].count("brand")
        late = opening[40:].count("brand")
        self.assertLess(late, max(1, early),
                        f"still opening with 'brand' after 40 sessions "
                        f"(early={early}, late={late}): no convergence")

    def test_c14_question_yield_stays_above_the_floor(self):
        """Pre-rebuild: 97 of 368 questions carried information -- 26%."""
        from src.agent import Agent
        agent = Agent(self.path)
        for index in range(40):
            session = f"c14_{index}"
            agent.reset(session, PROFILE)
            agent.respond(session, "I'm looking for Accessories Belts.", 1, TOP_K)
            for turn in (2, 3):
                agent.respond(session, "For that, what matters is: Suede.", turn, TOP_K)
        snapshot = agent.answers.snapshot()
        asked = sum(row["asked"] for row in snapshot.values())
        answered = sum(row["answered"] for row in snapshot.values())
        if asked < 20:
            self.skipTest("too few questions to judge yield")
        self.assertGreater(answered / asked, 0.30,
                           f"question yield {answered}/{asked} is barely above the "
                           "pre-rebuild 26%")

    def test_c15_a_recent_constraint_outweighs_an_old_one(self):
        agent = self.fresh("decay")
        agent.respond("decay", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.respond("decay", "For that, what matters is: Cotton.", 2, TOP_K)
        for turn in range(3, 8):
            agent.respond("decay", f"For that, what matters is: detail {turn}.", turn, TOP_K)
        dialog = agent.sessions["decay"].dialog
        weights = dialog.weights()
        old = min(weights.values())
        new = max(weights.values())
        self.assertGreater(new, old, "no recency decay across the constraint history")


class ProductResetTest(DialogBase):
    """C24, C25 — switching product is a different transition from changing a value.

    The organizer's own override sessions never exercise this: `behavior_for` swaps
    one constraint for another drawn from the same target, so the category never
    moves. These are the cases a real shopper produces and the public set does not.
    """

    def test_c24_switching_product_retires_product_bound_constraints(self):
        from src.state import DialogState
        dialog = DialogState()
        dialog.observe("red cotton", 1, self.catalog)
        dialog.observe("under thirty dollars", 2, self.catalog)
        self.assertIn("red cotton", dialog.phrases())
        dialog.product_reset(3)
        live = dialog.phrases()
        self.assertNotIn("red cotton", live,
                         "a colour/material from the abandoned product still binds")
        self.assertIn("under thirty dollars", live,
                      "budget describes the shopper, not the product; it must survive")

    def test_c24b_a_product_switch_stops_scoring_the_old_product(self):
        agent = self.fresh("swap")
        agent.respond("swap", "I'm looking for Accessories Scarves. "
                              "A key requirement is: silk.", 1, TOP_K)
        agent.respond("swap", "Actually, forget that. I need Accessories Belts "
                              "instead, in full grain leather.", 2, TOP_K)
        dialog = agent.sessions["swap"].dialog
        retired = [slot for slot in dialog.slots if slot.superseded]
        self.assertTrue(retired, "a change of product retired nothing at all")
        self.assertTrue(any("silk" in slot.text.lower() for slot in retired),
                        f"the abandoned material is still current: {dialog.slots}")
        self.assertFalse(any("accessories belts" in slot.text.lower()
                             for slot in dialog.slots if not slot.superseded),
                         "the routed category was also stored as feature evidence")
        self.assertTrue(any(slot.attribute == "material" and "leather" in slot.text.lower()
                            for slot in dialog.slots if not slot.superseded),
                        "removing the routed category also discarded its attached facet")

    def test_product_switch_starts_a_new_question_epoch(self):
        agent = self.fresh("question-epoch")
        agent.ask_mode = "none"
        agent.respond("question-epoch", "I'm looking for Accessories Scarves.", 1, TOP_K)
        state = agent.sessions["question-epoch"]
        state.asked.update({"category", "material", "color", "size", "style",
                            "use_case", "feature", "budget", "brand"})

        agent.respond("question-epoch", "Actually, I need Accessories Belts instead.",
                      2, TOP_K)
        self.assertTrue(set(state.dialog.PRODUCT_BOUND).isdisjoint(state.asked))
        self.assertNotIn("category", state.asked)
        self.assertTrue({"budget", "brand"} <= state.asked,
                        "person-level question history should survive a product switch")

    def test_explicit_restart_reopens_every_question_but_keeps_learning(self):
        agent = self.fresh("restart-epoch")
        agent.ask_mode = "none"
        agent.respond("restart-epoch", "I'm looking for Accessories Scarves.", 1, TOP_K)
        state = agent.sessions["restart-epoch"]
        state.asked.update({"category", "material", "budget", "brand"})
        before = agent.answers.snapshot()

        agent.respond("restart-epoch", "Forget everything. Start over.", 2, TOP_K)
        self.assertEqual(state.asked, set())
        self.assertIsNone(state.dialog.category)
        self.assertEqual(agent.answers.snapshot(), before,
                         "restart must not erase cross-session answerability learning")

    def test_restart_instruction_does_not_become_the_new_category(self):
        agent = self.fresh("restart-category")
        agent.ask_mode = "none"
        agent.respond("restart-category", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.respond("restart-category",
                      "Let us forget everything I said. I need Accessories Scarves.",
                      2, TOP_K)
        state = agent.sessions["restart-category"]
        from src.routing import exact_bucket
        self.assertEqual(exact_bucket(agent.catalog, state.dialog.category),
                         "Accessories Scarves")
        self.assertFalse(any("let us" in slot.text.lower() or "i said" in slot.text.lower()
                             for slot in state.dialog.slots))

    def test_nfkc_restart_is_stripped_on_the_same_canonical_text_it_matches(self):
        agent = self.fresh("nfkc-restart")
        agent.ask_mode = "none"
        agent.respond("nfkc-restart", "I'm looking for Accessories Belts.", 1, TOP_K)
        agent.respond("nfkc-restart",
                      "\uff26\uff4f\uff52\uff47\uff45\uff54 everything. I need Accessories Scarves.",
                      2, TOP_K)
        state = agent.sessions["nfkc-restart"]
        from src.routing import exact_bucket
        self.assertEqual(exact_bucket(agent.catalog, state.dialog.category),
                         "Accessories Scarves")

    def test_c25_with_no_question_left_the_full_slate_is_shown(self):
        """Withholding is only ever justified by a question that might sharpen the
        ranking. With nothing left to ask, holding candidates back is pure loss."""
        opening = "I'm looking for Accessories Belts."
        asking = self.fresh("ask")
        with_question = asking.respond("ask", opening, 1, TOP_K)
        self.assertIsNotNone(with_question["ask_attribute"])

        mute = self.fresh("mute")
        mute.ask_mode = "none"                        # forces `_ask` to return None
        without_question = mute.respond("mute", opening, 1, TOP_K)
        self.assertIsNone(without_question["ask_attribute"])
        self.assertGreater(len(without_question["recommendations"]),
                           len(with_question["recommendations"]),
                           "went silent and still withheld candidates")


class BoundaryTest(DialogBase):
    def test_c16_boundary_reply_does_not_stall_elicitation(self):
        agent = self.fresh("bd")
        agent.respond("bd", "I'm looking for Accessories Belts.", 1, TOP_K)
        before = len(agent.sessions["bd"].asked)
        response = agent.respond("bd", "I don't have a preference for material; "
                                       "please use your judgment.", 2, TOP_K)
        self.assertTrue(response["recommendations"], "a declined turn emptied the pool")
        self.assertGreaterEqual(len(agent.sessions["bd"].asked), before)


if __name__ == "__main__":
    unittest.main()


class CounterfactualQuestionValueTest(unittest.TestCase):
    """The question criterion that was specified, built, measured and declined.

    `expected_reduction` asks how much of the pool an answer removes.
    `expected_gain` asks how much it is expected to improve where the *target*
    lands, which is what the score actually pays for. The second is the right
    quantity and it does not help here -- analysis/question_value.json for the
    numbers, README for why. What these tests defend is that the alternative is
    still correct and still measurable, and above all that turning it on is the
    only way to get it.
    """

    @classmethod
    def setUpClass(cls):
        from src.catalog import Catalog
        from src.scoring import Scorer
        from tests.fixtures import RichCatalog
        cls._ctx = RichCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())
        cls.scorer = Scorer(cls.catalog)
        cls.ranked = cls.scorer.rank(cls.catalog.ids, ["belt"])

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_it_demotes_the_question_impurity_prefers(self):
        """The whole reason src/answerability.py exists, arrived at from the other
        side. Gini puts `brand` first because store is high-cardinality; the
        counterfactual puts it last because knowing the brand barely moves the head.
        Neither needs a rule saying "do not ask about brand"."""
        from src.elicitation import expected_gain, expected_reduction
        head = [pid for _, pid in self.ranked[:120]]
        by_reduction = max(("brand", "material"),
                           key=lambda a: expected_reduction(self.catalog, head, a))
        by_gain = max(("brand", "material"),
                      key=lambda a: expected_gain(self.catalog, self.scorer,
                                                  self.ranked, a))
        self.assertEqual(by_reduction, "brand")
        self.assertEqual(by_gain, "material")

    def test_a_question_no_answer_can_move_is_worth_nothing(self):
        """An attribute the head has no values for cannot reorder it, and the
        counterfactual says so rather than assuming a middling impurity."""
        from src.elicitation import expected_gain
        self.assertEqual(
            expected_gain(self.catalog, self.scorer, self.ranked, "size"), 0.0)

    def test_a_head_of_one_has_nothing_to_reorder(self):
        from src.elicitation import expected_gain
        self.assertEqual(
            expected_gain(self.catalog, self.scorer, self.ranked[:1], "material"), 0.0)

    def test_the_default_criterion_is_unchanged(self):
        """Declined means off. The counterfactual is reachable only by passing a
        scorer, which src/agent.py does only under P_ASK=counterfactual."""
        from src.elicitation import choose
        asked: set = set()
        self.assertEqual(choose(self.catalog, self.ranked, asked),
                         choose(self.catalog, self.ranked, asked, None, None, None))

    def test_the_category_question_still_competes_under_the_alternative(self):
        """tools/shelfbench --realistic measures the clarification at +0.167 hit
        rate. Its value is computed outside the catalog, so a change of units could
        switch it off silently; it is rescaled instead, and this is the check."""
        from src.elicitation import choose
        extra = {"category": 0.95}
        self.assertEqual(
            choose(self.catalog, self.ranked, set(), None, extra), "category")
        self.assertEqual(
            choose(self.catalog, self.ranked, set(), None, extra, self.scorer),
            "category")


class NegationPolarityTest(unittest.TestCase):
    """A negation is only evidence when it refuses something a product could be.

    Both defects pinned here were live on this branch until they were measured.
    The first fires on the official harness whenever the agent asks nothing and
    the simulator nudges; the second inverts the polarity of a hard requirement,
    which is the worst thing this parser can do -- the slot ends up voting
    against exactly the products the shopper insisted on.

    Adopted from phase/hybrid-refinement (commit 7925120), which found both.
    """

    def _slots(self, message):
        from src.state import DialogState
        state = DialogState()
        state.observe(message, 1, None)
        return [(slot.text, slot.polarity) for slot in state.slots]

    def test_c27_a_complaint_is_not_a_refusal(self):
        """The evaluator's own nudge, verbatim from local_evaluator.py.

        "quite right yet" names nothing a product could be, and reading it as a
        refused value penalised every product whose text contained those words --
        3,512 of the 50,000 in the shipped catalog, at full NEGATIVE_PENALTY.
        """
        message = ("Those options are not quite right yet. "
                   "Ask me about one specific attribute.")
        self.assertEqual([text for text, polarity in self._slots(message)
                          if polarity < 0], [])

    def test_c27b_negation_under_a_dealbreaker_is_a_requirement(self):
        """"Dealbreaker if it is not X" asks *for* X.

        The phrasing is REQUIREMENTS in tools/adversarial.py, so it is live on
        the robustness harness throughout the scaffold axis.
        """
        slots = self._slots("Dealbreaker if it is not Water Resistant.")
        self.assertIn(("Water Resistant", 1), slots)
        self.assertEqual([text for text, polarity in slots if polarity < 0], [])

    def test_c27c_ordinary_refusals_still_refuse(self):
        """The fix must not cost the behaviour it is protecting."""
        for message, value in (("I do not want polyester.", "polyester"),
                               ("Nothing in suede, please.", "suede"),
                               ("No wool.", "wool")):
            with self.subTest(message=message):
                self.assertIn((value, -1), self._slots(message))

    def test_c27d_courtesy_words_are_not_part_of_the_refused_value(self):
        """Hollow words are trimmed from the ends, not filtered throughout.

        "quite bright" is a refusal of bright; a hollow word *between* two real
        ones is part of the phrase and stays.
        """
        self.assertIn(("bright", -1), self._slots("Not quite bright enough."))

    def test_c27e_unless_is_left_alone(self):
        """"Anything unless it is polyester" reads the other way.

        DOUBLE_NEGATIVE is deliberately narrow: it fires on a *negative
        consequence* ("dealbreaker", "no good", "won't work"), never on a bare
        conditional, because the bare conditional really is a refusal.
        """
        from src.state import DOUBLE_NEGATIVE
        self.assertIsNone(DOUBLE_NEGATIVE.search("Anything unless it is polyester."))
        self.assertIsNotNone(
            DOUBLE_NEGATIVE.search("Dealbreaker if it is not water resistant."))

    def test_c27f_a_hollow_refusal_never_reaches_the_refusal_penalty(self):
        """The end-to-end consequence, not just the parse.

        `Agent._refusals` falls back to a text scan for refusals the catalog
        cannot resolve to a facet, which is exactly where a hollow value lands.
        Nothing hollow may reach it.
        """
        from src.state import substantive
        for hollow in ("quite right yet", "really working", "that one",
                       "exactly what", "those options"):
            with self.subTest(value=hollow):
                self.assertIsNone(substantive(hollow))
        self.assertEqual(substantive("quite bright"), "bright")
        self.assertEqual(substantive("full grain leather"), "full grain leather")


class DeclineRecognitionTest(unittest.TestCase):
    """Handing an attribute back to us is an answer, and it is not a refusal.

    `DECLINE` requires the literal word "preference" (or one of four synonyms),
    which is how the organizer's templates are phrased. A shopper says "I don't
    mind" or "up to you". Those fell through to NEGATION, which read "do not mind
    about brand" as a refusal of the phrase "mind about brand" and penalised
    every product containing any of those words -- 22.1% of this catalog for
    brand, 32.9% for style.

    Neither this branch nor phase/hybrid-refinement had this; it was found while
    verifying that branch's negation work.
    """

    def _slots(self, message):
        from src.state import DialogState
        state = DialogState()
        state.observe(message, 1, None)
        return [(slot.text, slot.polarity) for slot in state.slots]

    def test_c28_a_decline_in_shopper_english_carries_no_constraint(self):
        for message in ("Genuinely do not mind about brand.",
                        "brand is up to you.",
                        "Whatever you recommend.",
                        "Either is fine.",
                        "I'm easy.",
                        "Not fussed about the colour.",
                        "It doesn't matter to me.",
                        "Surprise me."):
            with self.subTest(message=message):
                self.assertEqual(self._slots(message), [])

    def test_c28b_the_organizer_templates_still_decline(self):
        """The narrow DECLINE path must keep working; this is a widening."""
        for message in ("I don't have a preference for color; please use your judgment.",
                        "I don't have an additional preference for style."):
            with self.subTest(message=message):
                self.assertEqual(self._slots(message), [])

    def test_c28c_a_decline_never_becomes_a_refusal(self):
        """The defect, stated as the thing that must not happen.

        A declined turn that reaches NEGATION produces a negative slot, and the
        refusal penalty is applied per-product against its words.
        """
        for message in ("Genuinely do not mind about brand.",
                        "I do not really mind about style."):
            with self.subTest(message=message):
                self.assertEqual(
                    [t for t, polarity in self._slots(message) if polarity < 0], [])

    def test_c28d_declining_one_attribute_does_not_discard_the_others(self):
        """Why this is per-clause and DECLINE is per-message.

        "I need cotton, and the colour is up to you" declines one attribute and
        states another. Dropping the whole turn to handle the first would throw
        away the second -- which is what widening DECLINE itself would have done.
        """
        slots = self._slots("I need cotton, and the colour is up to you.")
        self.assertIn(("cotton", 1), slots)
        self.assertEqual([t for t, polarity in slots if polarity < 0], [])

    def test_c28e_real_constraints_are_not_mistaken_for_declines(self):
        """The false-positive direction. "I mind about quality" is not a decline."""
        from src.text import NO_PREFERENCE
        for message in ("I need cotton.", "A key requirement is: full grain leather.",
                        "I do not want polyester.", "I care about durability.",
                        "I mind about quality a great deal.",
                        "Dealbreaker if it is not water resistant.",
                        "Budget is around $40."):
            with self.subTest(message=message):
                self.assertIsNone(NO_PREFERENCE.search(message))


class MetaClauseTest(unittest.TestCase):
    """Telling us how to behave is not telling us what to buy.

    The evaluator's nudge is two clauses. The complaint is handled by the
    negation path (test_c27); the instruction -- "Ask me about one specific
    attribute" -- was stored as a positive constraint and fed to BM25, which
    ranks products by whether their text contains "specific" or "option".

    Same defect as the correction phrase that CORRECTION_LEAD strips, and it
    survived on both this branch and phase/hybrid-refinement.
    """

    def _slots(self, message):
        from src.state import DialogState
        state = DialogState()
        state.observe(message, 1, None)
        return [slot.text for slot in state.slots]

    def test_c29_the_nudge_leaves_no_constraint_at_all(self):
        """Both clauses of the evaluator's nudge, verbatim."""
        self.assertEqual(
            self._slots("Those options are not quite right yet. "
                        "Ask me about one specific attribute."), [])

    def test_c29b_an_instruction_that_also_names_a_property_is_kept(self):
        """Why every content word must be conversational before a clause is cut.

        A shopper steering the conversation usually says something in the same
        breath, and dropping the clause would drop that too.
        """
        self.assertTrue(any("cheaper" in text for text in
                            self._slots("Show me something cheaper.")))
        self.assertTrue(any("leather" in text.lower() for text in
                            self._slots("Tell me more about the leather one.")))

    def test_c29c_ordinary_constraints_are_untouched(self):
        from src.text import is_meta
        for message in ("What matters is: full grain leather.",
                        "I need a cotton shirt.", "Budget is around $40.",
                        "Something in navy blue."):
            with self.subTest(message=message):
                self.assertFalse(is_meta(message))

    def test_c29d_an_empty_clause_is_not_meta(self):
        """`all()` over an empty sequence is True; that must not cut a clause."""
        from src.text import is_meta
        self.assertFalse(is_meta(""))
        self.assertFalse(is_meta("   "))


class RestartScopeTest(unittest.TestCase):
    """Pillar II names "slot erasure and rewriting" explicitly.

    Erasure is the widest thing a correction can ask for, so the cue that
    triggers it stays narrow on purpose: "actually, brown" replaces the colour
    and leaves the material, the budget and the size standing. That narrowness
    was right and is unchanged. What it missed is the shopper who retracts the
    whole brief in words other than "start over".
    """

    def test_c30_retracting_the_whole_brief_reaches_global_scope(self):
        from src.text import is_restart
        for message in ("Ignore my earlier preferences.",
                        "Disregard my previous request.",
                        "Forget my previous requirements.",
                        "Ignore my preferences, let's try again.",
                        "Start over.",
                        "Forget everything."):
            with self.subTest(message=message):
                self.assertTrue(is_restart(message))

    def test_c30d_one_preference_is_not_the_whole_brief(self):
        """The singular/plural boundary, which is where this nearly went wrong.

        "Ignore my earlier preference" retracts one attribute; "...preferences"
        retracts the set. That is ordinary English, and it is also load-bearing:
        the organizer's override template is "Actually, ignore my earlier
        preference. What I need is: X", and both of its values derive from the
        same target product, so sweeping the slot table on it is measurably wrong.
        CORRECTION_LEAD handles that phrasing at attribute scope instead.
        """
        from src.text import is_restart
        self.assertFalse(is_restart("Actually, ignore my earlier preference. "
                                    "What I need is: silk."))
        self.assertFalse(is_restart("Ignore my earlier preference."))
        self.assertTrue(is_restart("Ignore my earlier preferences."))

    def test_c30b_an_attribute_correction_does_not(self):
        """The narrowness is the feature, not an oversight.

        "Forget that" retracts the last thing said, not the session. Promoting it
        to global scope is what made a colour change erase the material -- and on
        this simulator it is measurably wrong as well, because the override
        sessions derive the old and the new value from the same target product.
        """
        from src.text import is_restart
        for message in ("Actually, brown.", "Forget that.", "Scratch that.",
                        "Never mind.", "Ignore that.", "No wait, cotton."):
            with self.subTest(message=message):
                self.assertFalse(is_restart(message))

    def test_c30c_a_global_retraction_erases_the_earlier_slots(self):
        from src.state import DialogState
        state = DialogState()
        state.observe("I need a leather belt.", 1, None)
        state.observe("Black, and under $40.", 2, None)
        self.assertTrue([s for s in state.slots if not s.superseded])
        state.observe("Ignore my earlier preferences. I want a silk scarf.", 3, None)
        standing = [s.text for s in state.slots if not s.superseded]
        self.assertTrue(any("silk" in text.lower() for text in standing))
        self.assertFalse(any("leather" in text.lower() for text in standing),
                         f"a global retraction left the old brief standing: {standing}")
