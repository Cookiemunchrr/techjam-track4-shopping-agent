"""The learned reranker: what it is allowed to do, and what it must not.

The model was built, measured, nearly declined on a measurement that turned out to
be too blunt, re-measured, and shipped. The story is in
analysis/reranker_experiment.json; what these tests defend is the shape of it.

It reorders the head of a ranking the weighted sum produced. It never retrieves, so
a bad model can cost ordering and can never cost recall -- which matters because
tools/recall.py says recall is already 1.000 on every official session. It runs at
a deliberately small blend, so it argues with the weighted sum rather than replacing
it. And it refuses an asset whose feature list is not the one it was trained on,
because a vector served in the wrong order fails silently.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.catalog import Catalog
from src.features import FEATURES, Context, vector
from src.rerank import Model, load
from src.scoring import Scorer
from tests.fixtures import PathologyCatalog, TempCatalog

# The blend the plan calls the headroom guard. The shipped asset sits at 0.05;
# asserting the invariant only there would pass a model that is one cautious knob
# turn away from breaking it, and raising the blend is exactly the change somebody
# reaches for after a retrain that looks good on paper.
HEADROOM_BLEND = 0.2


class ShippedAssetTest(unittest.TestCase):
    """The committed model, and the properties the measurement depended on."""

    path = Path("analysis/reranker.json")

    def test_the_asset_is_committed(self):
        self.assertTrue(self.path.exists(),
                        "analysis/reranker.json is the shipped model; see "
                        "analysis/reranker_experiment.json for what it had to beat")

    def test_it_loads_and_matches_the_current_feature_vector(self):
        model = load(self.path)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.weights), len(FEATURES))

    def test_the_blend_stays_small(self):
        """Blend is the whole safety margin, and it is not a free parameter.

        At blend 1.0 the model's wider score range inflates rank-1-to-rank-2
        margins, CommitPolicy.width reads margin as confidence, the slate narrows
        from 4.61 items to 3.28, and the official score rises 0.025 without the
        ranking improving at all. The measurement that justifies shipping was taken
        at 0.05. Raising this re-opens that question rather than tuning a knob.
        """
        self.assertLessEqual(load(self.path).blend, 0.1)

    def test_the_agent_picks_it_up(self):
        from src.agent import Agent
        with TempCatalog() as path:
            self.assertIsNotNone(Agent(path).reranker)

    def test_a_missing_asset_is_a_supported_state(self):
        """Absence must degrade to the weighted sum, never to an exception.

        The submission bundle is assembled by hand and the agent has to start in an
        environment that has only what the bundle carries.
        """
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load(Path(directory) / "nothing.json"))


class AssetContractTest(unittest.TestCase):
    """A model served a vector it was not trained on fails silently. Not here."""

    def test_wrong_feature_count_is_refused(self):
        with self.assertRaises(ValueError):
            Model([0.0] * (len(FEATURES) - 1))

    def test_reordered_features_are_refused(self):
        scrambled = list(FEATURES)
        scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
        with self.assertRaises(ValueError):
            Model.from_dict({"features": scrambled, "weights": [0.0] * len(FEATURES)})

    def test_a_matching_asset_loads(self):
        model = Model.from_dict({"features": list(FEATURES),
                                 "weights": [0.0] * len(FEATURES), "blend": 0.5})
        self.assertEqual(model.blend, 0.5)


class ReordersButNeverRetrievesTest(unittest.TestCase):
    """The safety property: a bad model can cost ordering, never recall."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())
        cls.scorer = Scorer(cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def context(self):
        return Context(self.catalog, self.scorer, ["leather belt"],
                       facets={"material": "leather"}, constraints=1)

    def ranked(self):
        return self.scorer.rank(self.catalog.ids, ["leather belt"])

    def test_the_candidate_set_is_unchanged(self):
        model = Model([1.0] * len(FEATURES))
        before = self.ranked()
        after = model.apply(before, self.context())
        self.assertEqual({pid for _, pid in before}, {pid for _, pid in after})
        self.assertEqual(len(before), len(after))

    def test_the_tail_beyond_depth_is_untouched(self):
        model = Model([1.0] * len(FEATURES))
        before = self.ranked()
        after = model.apply(before, self.context(), depth=2)
        self.assertEqual(before[2:], after[2:])

    def test_a_zero_model_changes_nothing(self):
        model = Model([0.0] * len(FEATURES))
        before = self.ranked()
        self.assertEqual([p for _, p in before],
                         [p for _, p in model.apply(before, self.context())])

    def test_blend_zero_changes_nothing(self):
        model = Model([5.0] * len(FEATURES), blend=0.0)
        before = self.ranked()
        self.assertEqual([p for _, p in before],
                         [p for _, p in model.apply(before, self.context())])

    def test_reordering_is_deterministic_and_order_independent(self):
        """Ties break on the identifier, so pool arrival order cannot leak in."""
        model = Model([1.0] * len(FEATURES))
        before = self.ranked()
        once = model.apply(before, self.context())
        twice = model.apply(list(reversed(before)), self.context())
        self.assertEqual([p for _, p in once], [p for _, p in twice])


class FeatureVectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TempCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())
        cls.scorer = Scorer(cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_width_matches_the_declared_features(self):
        ctx = Context(self.catalog, self.scorer, ["leather"])
        self.assertEqual(len(vector("P_LEATHER_BELT", ctx)), len(FEATURES))

    def test_every_value_is_finite(self):
        import math
        ctx = Context(self.catalog, self.scorer, ["leather belt"],
                      facets={"material": "leather"}, constraints=2)
        for pid in self.catalog.ids:
            for name, value in zip(FEATURES, vector(pid, ctx)):
                self.assertTrue(math.isfinite(value), f"{name} on {pid}")

    def test_an_empty_turn_produces_a_vector(self):
        """Turn 1 of a session that stated nothing must not crash the extractor."""
        ctx = Context(self.catalog, self.scorer, [])
        self.assertEqual(len(vector("P_LEATHER_BELT", ctx)), len(FEATURES))

    def test_the_refusal_feature_is_negative(self):
        """Sign convention: a refused product must push the score down, not up."""
        ctx = Context(self.catalog, self.scorer, ["belt"],
                      refused={"P_LEATHER_BELT": 0.9})
        index = FEATURES.index("refused")
        self.assertLess(vector("P_LEATHER_BELT", ctx)[index], 0.0)
        self.assertEqual(vector("P_SUEDE_BELT", ctx)[index], 0.0)

    def test_shelf_popularity_separates_the_long_tail(self):
        """The whole reason the feature exists: global popularity hides it."""
        index = FEATURES.index("shelf_popularity")
        ctx = Context(self.catalog, self.scorer, ["belt"])
        top = vector("P_LEATHER_BELT", ctx)[index]     # 9000 ratings
        tail = vector("P_CANVAS_BELT", ctx)[index]     # 5 ratings
        self.assertGreater(top, tail)


if __name__ == "__main__":
    unittest.main()


class PopularityPathologyTest(unittest.TestCase):
    """The prior must not be allowed to overrule what the shopper said.

    src/scoring.py holds a 1.4:1 popularity-to-phrase ratio and
    tests/test_scoring_policy.py pins a 3:1 evidence floor under it. The trained
    model does not share that opinion: analysis/reranker_experiment.json records
    that it wants 7.6:1, and on this fixture it prefers the popular canvas belt to
    the leather one the shopper asked for by a full point of its own score. The
    blend is what stands between that preference and the ranking.

    So the floor the weighted sum owns is worth nothing unless the *combined*
    ordering keeps it, and that is what this asserts -- at the shipped blend, and
    again at four times it, because a retrain that leans harder on the prior shows
    up here before it shows up in analysis/longtail.json, where the slice is 25
    sessions and the interval is wide.

    Deliberately not asserted: that the model prefers the popular item. That is
    today's pathology, not an invariant, and a retrain that fixes it should not
    have to delete a test to prove it.
    """

    @classmethod
    def setUpClass(cls):
        cls._ctx = PathologyCatalog()
        cls.catalog = Catalog(cls._ctx.__enter__())
        cls.scorer = Scorer(cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def context(self):
        return Context(self.catalog, self.scorer, ["leather"],
                       facets={"material": "leather"}, constraints=1)

    def ranked(self):
        return self.scorer.rank(self.catalog.ids, ["leather"],
                                facets={"material": "leather"})

    def combined(self, blend):
        """Top of the ranking the agent would actually show, at this blend."""
        shipped = load(ShippedAssetTest.path)
        model = Model(shipped.weights, shipped.bias, blend)
        ordered = model.apply(self.ranked(), self.context())
        return ordered[0][1], ordered[0][0] - ordered[1][0]

    def test_the_weighted_sum_puts_evidence_first(self):
        """The premise. If this fails the fixture has drifted, not the model."""
        self.assertEqual(self.ranked()[0][1], "LONGTAIL_LEATHER")

    def test_the_shipped_blend_keeps_evidence_first(self):
        winner, _ = self.combined(load(ShippedAssetTest.path).blend)
        self.assertEqual(winner, "LONGTAIL_LEATHER",
                         "the reranker overturned a stated material constraint in "
                         "favour of the popular item; see F-C in the improvement "
                         "plan and analysis/longtail.json")

    def test_it_still_holds_with_four_times_the_blend(self):
        winner, margin = self.combined(HEADROOM_BLEND)
        self.assertEqual(winner, "LONGTAIL_LEATHER",
                         f"evidence lost to popularity at blend {HEADROOM_BLEND}: "
                         "the shipped blend is now the only thing holding the "
                         "invariant, which is not a margin")
        self.assertGreater(margin, 0.1,
                           "evidence wins by a rounding error at the headroom "
                           "blend, which is the pathology arriving rather than "
                           "the guard holding")

    def test_a_model_that_only_reads_popularity_is_caught(self):
        """The canary has to be able to fail. A model that has learned nothing but
        the prior must lose this ranking at the headroom blend."""
        weights = [0.0] * len(FEATURES)
        weights[FEATURES.index("popularity")] = 30.0
        ordered = Model(weights, 0.0, HEADROOM_BLEND).apply(self.ranked(), self.context())
        self.assertEqual(ordered[0][1], "POPULAR_CANVAS",
                         "a popularity-only model kept the evidence ranking, so "
                         "this test cannot detect the failure it exists for")


class BoostedTreeAssetTest(unittest.TestCase):
    """The declined tree experiment, pinned so its record stays reproducible.

    analysis/reranker_stumps.json closes the last of the three levers on this
    reranker. Nothing from it ships, so there is no serving path to defend -- but a
    recorded experiment that cannot be re-run is a claim, not a measurement, and
    the tree format is the part of it most likely to rot.
    """

    def trees(self):
        return [{"f": 0, "t": 0.5, "l": -1.0, "r": 1.0},
                {"f": 3, "t": 0.25, "l": 0.5,
                 "r": {"f": 1, "t": 0.5, "l": -0.25, "r": 2.0}}]

    def test_a_tree_asset_round_trips_through_json(self):
        from tools.rerank_stumps import TreeModel
        model = TreeModel(self.trees(), rate=0.1, blend=0.2)
        restored = TreeModel(**{k: v for k, v in
                                json.loads(json.dumps(model.to_dict())).items()
                                if k in ("trees", "rate", "blend", "meta")})
        for x in ([0.0] * len(FEATURES), [1.0] * len(FEATURES),
                  [0.4, 0.9, 0.0, 0.3] + [0.0] * (len(FEATURES) - 4)):
            self.assertAlmostEqual(model.score_vector(x), restored.score_vector(x),
                                   places=9)

    def test_a_split_sends_equal_values_left(self):
        """The convention the trainer fits under. Flipping it silently would change
        every recorded number without changing a single stored threshold."""
        from tools.rerank_stumps import TreeModel
        model = TreeModel([{"f": 0, "t": 0.5, "l": -1.0, "r": 1.0}], rate=1.0)
        exactly_at = [0.5] + [0.0] * (len(FEATURES) - 1)
        self.assertEqual(model.score_vector(exactly_at), -1.0)

    def test_training_is_deterministic_without_a_seed(self):
        """No RNG anywhere in the trainer: same data, same trees, byte for byte.
        The repository's seeding rule exists because str hash() is salted; the
        answer here is to have nothing to seed."""
        from tools.rerank_stumps import boost
        groups = [{"rows": [{"x": [float(i % 3), float(i % 2)] + [0.0] * (len(FEATURES) - 2),
                             "y": int(i == 0)} for i in range(8)]}
                  for _ in range(4)]
        first, _ = boost(groups, rounds=5, rate=0.1, depth=1, min_leaf=2)
        second, _ = boost(groups, rounds=5, rate=0.1, depth=1, min_leaf=2)
        self.assertEqual(json.dumps(first), json.dumps(second))
