"""Scoring terms and the commit / rejection policies."""
from __future__ import annotations

import unittest

from src.catalog import Catalog
from src.elicitation import ALLOWED, choose
from src.policy import TOP_K, CommitPolicy, RejectionModel
from src.scoring import Scorer, Weights
from tests.fixtures import TempCatalog


class ScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())
        cls.scorer = Scorer(cls.catalog)
        cls.belts = ["P_LEATHER_BELT", "P_SUEDE_BELT", "P_CANVAS_BELT"]

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_a_short_material_constraint_still_scores(self):
        """"leather" is 7 characters and below PHRASE_MIN_CHARS, but it is the most
        discriminative thing a belt shopper can say. 27.6% of the constraints this
        simulator discloses are shorter than the floor."""
        plain = self.scorer._phrase("P_SUEDE_BELT", ["leather"])
        self.assertEqual(plain, 0.0, "suede belt matched a leather constraint")
        self.assertGreater(self.scorer._phrase("P_LEATHER_BELT", ["leather"]), 0.0,
                           "a stated material shorter than the floor scored nothing")

    def test_a_short_non_constraint_word_is_still_ignored(self):
        """The exemption is for materials and colours, not for every short word."""
        self.assertEqual(self.scorer._phrase("P_LEATHER_BELT", ["belt"]), 0.0,
                         "the floor stopped applying to ordinary short words")

    def test_with_no_evidence_popularity_decides(self):
        ranked = self.scorer.rank(self.belts, [])
        self.assertEqual(ranked[0][1], "P_LEATHER_BELT")   # 9000 reviews

    def test_stated_evidence_overrides_the_popularity_prior(self):
        """A 40-review item the shopper described must beat a 9000-review one they did not.

        The prior is a prior. If it cannot be overridden by what the customer
        actually said, the conversation is decorative.
        """
        prior_only = dict((p, s) for s, p in self.scorer.rank(self.belts, []))
        self.assertEqual(max(prior_only, key=prior_only.get), "P_LEATHER_BELT")
        with_evidence = self.scorer.rank(self.belts, ["A brown suede belt"])
        self.assertEqual(with_evidence[0][1], "P_SUEDE_BELT")

    def test_evidence_outweighs_prior_on_the_real_catalog(self):
        """Regression guard on the weight balance -- see analysis/priors.json.

        Measured over real sessions, the target's lexical+phrase contribution is
        roughly 5.8x the popularity contribution. The floor below is deliberate:
        if a weight change drops it under 3:1, retrieval has become popularity-only,
        the conversation has stopped mattering, and this fails.
        """
        import json, os, collections
        if not (os.path.exists("data/catalog.jsonl") and os.path.exists("data/public_set.jsonl")):
            self.skipTest("full catalog not present")
        from evaluator.local_evaluator import catalog_index, coarse_category, intent_card
        from src.catalog import Catalog as FullCatalog
        from src.scoring import Scorer as FullScorer
        from src.routing import candidates as narrow
        from src.text import tokens as content_tokens

        full = FullCatalog("data/catalog.jsonl")
        scorer = FullScorer(full)
        _, cats, prods = catalog_index("data/catalog.jsonl")
        with open("data/public_set.jsonl", encoding="utf-8") as fh:
            samples = [json.loads(line) for line in fh][:20]

        prior_total = evidence_total = 0.0
        for sample in samples:
            target = sample["ground_truth"]["parent_asin"]
            card = intent_card(prods[target])
            clauses = list(dict.fromkeys(card["hard_constraints"] + card["soft_preferences"]))
            query = collections.Counter()
            for clause in clauses:
                for term in content_tokens(clause):
                    query[term] += 1
            prior_total += scorer.w.popularity * full.popularity(target)
            evidence_total += (scorer.w.lexical * scorer._bm25(target, query)
                               + scorer.w.phrase * scorer._phrase(target, clauses))
        self.assertGreater(evidence_total, prior_total * 3,
                           "popularity prior is drowning out what the customer said")

    def test_verbatim_phrase_beats_a_bag_of_words(self):
        with_phrase = dict((p, s) for s, p in self.scorer.rank(self.belts, ["Buckle closure"]))
        weightless = Scorer(self.catalog, Weights(phrase=0.0))
        without = dict((p, s) for s, p in weightless.rank(self.belts, ["Buckle closure"]))
        self.assertGreater(with_phrase["P_LEATHER_BELT"] - without["P_LEATHER_BELT"], 0)

    def test_short_clauses_do_not_count_as_phrases(self):
        self.assertEqual(self.scorer._phrase("P_LEATHER_BELT", ["belt"]), 0.0)

    def test_shown_items_are_penalised_by_exactly_the_weight_not_removed(self):
        clean = dict((p, s) for s, p in self.scorer.rank(self.belts, []))
        marked = dict((p, s) for s, p in self.scorer.rank(self.belts, [], shown={"P_LEATHER_BELT"}))
        self.assertIn("P_LEATHER_BELT", marked, "a shown item must stay in contention")
        self.assertAlmostEqual(clean["P_LEATHER_BELT"] - marked["P_LEATHER_BELT"],
                               self.scorer.w.shown_penalty, places=6)
        self.assertEqual(clean["P_SUEDE_BELT"], marked["P_SUEDE_BELT"], "others unaffected")

    def test_a_shown_item_loses_to_an_otherwise_equal_rival(self):
        near_tie = Scorer(self.catalog, Weights(popularity=0.0, lexical=0.0,
                                                phrase=0.0, shown_penalty=0.55))
        ranked = near_tie.rank(["P_SUEDE_BELT", "P_LEATHER_BELT"], [], shown={"P_LEATHER_BELT"})
        self.assertEqual(ranked[0][1], "P_SUEDE_BELT")

    def test_ranking_is_deterministic_and_totally_ordered(self):
        a = self.scorer.rank(self.belts, ["leather"])
        b = self.scorer.rank(list(reversed(self.belts)), ["leather"])
        self.assertEqual(a, b, "ranking must not depend on candidate order")

    def test_scores_are_finite(self):
        for score, _ in self.scorer.rank(self.catalog.ids, ["leather", "x" * 300]):
            self.assertEqual(score, score)                   # not NaN
            self.assertLess(abs(score), 1e6)

    def test_weights_from_env_fall_back_on_garbage(self):
        w = Weights.from_env({"W_POP": "not-a-number"})
        self.assertEqual(w.popularity, Weights().popularity)


class CommitPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = CommitPolicy()

    def test_commits_narrow_when_the_leader_is_clear(self):
        # 0.5 clears mid_margin but not high_margin: base_width + 1.
        self.assertEqual(self.policy.width(2, [(1.0, "a"), (0.5, "b")]),
                         self.policy.base_width + 1)
        # A runaway leader is a verdict and gets the narrowest slate.
        self.assertEqual(self.policy.width(2, [(1.0, "a"), (0.1, "b")]),
                         self.policy.base_width)

    def test_widens_when_the_top_is_flat(self):
        flat = self.policy.width(2, [(1.0, "a"), (0.999, "b")])
        clear = self.policy.width(2, [(1.0, "a"), (0.1, "b")])
        self.assertGreater(flat, clear,
                           "a flat ranking must be covered more widely than a clear one")
        self.assertEqual(flat, self.policy.base_width + 6)

    def test_covers_fully_once_turns_run_short(self):
        self.assertEqual(self.policy.width(8, [(1.0, "a"), (0.5, "b")]), TOP_K)
        self.assertEqual(self.policy.width(10, [(1.0, "a"), (0.5, "b")]), TOP_K)

    def test_handles_degenerate_rankings(self):
        self.assertGreaterEqual(self.policy.width(1, []), 1)
        self.assertGreaterEqual(self.policy.width(1, [(1.0, "a")]), 1)

    def test_env_override_is_robust(self):
        self.assertEqual(CommitPolicy.from_env({"P_PROBE": "junk"}).base_width, 2)
        self.assertEqual(CommitPolicy.from_env({"P_PROBE": "4"}).base_width, 4)


class RejectionModelTest(unittest.TestCase):
    def test_hard_mode_removes_candidates(self):
        model = RejectionModel(RejectionModel.HARD)
        model.record(["a"])
        self.assertEqual(model.filter(["a", "b"]), ["b"])
        self.assertEqual(model.penalised, frozenset())

    def test_soft_mode_keeps_but_penalises(self):
        model = RejectionModel(RejectionModel.SOFT)
        model.record(["a"])
        self.assertEqual(model.filter(["a", "b"]), ["a", "b"])
        self.assertEqual(model.penalised, frozenset({"a"}))

    def test_correction_retires_old_rejections_but_not_the_newest(self):
        """A change of intent means earlier rejections no longer bind -- earlier,
        not the slate we just put in front of them. Clearing everything bounced the
        product the customer had just turned down straight back to rank one."""
        model = RejectionModel(RejectionModel.RESET)
        model.record(["a", "b"])
        model.record(["c"])
        model.on_correction()
        self.assertNotIn("a", model.shown, "an old rejection still binds after a correction")
        self.assertIn("c", model.shown, "the item just shown was immediately forgiven")

    def test_correction_does_not_clear_in_soft_or_hard_mode(self):
        for mode in (RejectionModel.SOFT, RejectionModel.HARD):
            model = RejectionModel(mode)
            model.record(["a"])
            model.on_correction()
            self.assertIn("a", model.shown, mode)

    def test_unknown_mode_defaults_to_reset(self):
        self.assertEqual(RejectionModel("nonsense").mode, RejectionModel.RESET)


class ElicitationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def _ranked(self, ids):
        return [(1.0, pid) for pid in ids]

    def test_returns_an_allowed_attribute(self):
        got = choose(self.catalog, self._ranked(self.catalog.ids), set())
        self.assertIn(got, ALLOWED)

    def test_prefers_the_attribute_that_splits_the_pool(self):
        """All three belts differ in material, so material carries the most entropy."""
        got = choose(self.catalog, self._ranked(
            ["P_LEATHER_BELT", "P_SUEDE_BELT", "P_CANVAS_BELT"]), set())
        self.assertEqual(got, "material")

    def test_never_repeats_a_question(self):
        asked, seen = set(), []
        for _ in range(8):
            got = choose(self.catalog, self._ranked(self.catalog.ids), asked)
            if got is None:
                break
            self.assertNotIn(got, seen)
            seen.append(got)
            asked.add(got)

    def test_falls_back_when_the_pool_is_empty(self):
        self.assertIn(choose(self.catalog, [], set()), ALLOWED)

    def test_eventually_returns_none_rather_than_looping(self):
        asked = set(ALLOWED)
        self.assertIsNone(choose(self.catalog, self._ranked(self.catalog.ids), asked))


if __name__ == "__main__":
    unittest.main()
