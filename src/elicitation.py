"""Which attribute to ask about next.

The value of a question is how much it narrows the candidate set, discounted by how
likely the customer is to answer it:

    value(a) = expected_reduction(a) * P(answerable | a)

`expected_reduction` is computed from the live candidate pool. For a facet whose
values have distribution p, the expected number of candidates surviving a correct
answer is n * sum(p_v^2), so the fraction removed is 1 - sum(p_v^2) -- the Gini
impurity of the facet. That is the standard splitting criterion, and on its own it
prefers `brand` over everything, which is a wasted turn every single time. See
src/answerability.py for why no catalog-derived criterion can fix that, and why the
second factor has to be learned instead.

Attributes we cannot compute a facet for still get asked: they carry an assumed
middling impurity and are ranked entirely on learned answerability, which is how
`feature` -- the highest-yield question in the simulator, at 97% -- rises to the top
without anyone hard-coding it.
"""
from __future__ import annotations

import collections
import math

from .answerability import AnswerModel
from .catalog import Catalog

# Attributes the harness will accept (docs/agent_api_contract.json).
ALLOWED = ("category", "material", "color", "size", "style", "brand",
           "budget", "feature", "use_case", "other")

# Attributes we can compute a value for, so impurity is measurable.
MEASURABLE = ("material", "color", "brand", "budget")

# Attributes we cannot measure. Ranked on learned answerability alone.
UNMEASURABLE = ("feature", "style", "use_case", "size")

POOL = 120              # how many top candidates to reason over
MIN_COVERAGE = 0.25     # skip an attribute most candidates have no value for
ASSUMED_IMPURITY = 0.60  # stand-in reduction for an attribute with no facet
ASK_TIE_EPSILON = 0.0   # exact ties only: values identical to the last digit are
                        # the same question; the prior settles those and nothing else

# What one extra turn costs, in the units the counterfactual below produces.
# TechnicalScore = 0.50*HitRate + 0.30*MRR + 0.20*clip((11 - MTTC)/10), so a turn
# is worth 0.20/10 = 0.02 of score and a unit of MRR is worth 0.30. A question has
# to be expected to buy more than 0.02/0.30 of reciprocal rank to pay for itself.
# Read off the published scoring formula, not fitted to anything -- fitting a
# threshold to simulator-generated answers is the trap in the working rules.
TURN_COST_IN_MRR = 0.02 / 0.30
# How many of an attribute's values to simulate answers for. The tail of the value
# distribution is where the mass is not.
SIMULATED_ANSWERS = 4


def expected_reduction(catalog: Catalog, pool, attribute: str) -> float | None:
    """Fraction of the pool a correct answer would remove, or None if unmeasurable."""
    counts = collections.Counter(catalog.facet(pid, attribute) for pid in pool)
    counts.pop(None, None)
    total = sum(counts.values())
    if not total or total < len(pool) * MIN_COVERAGE or len(counts) < 2:
        return None
    gini = 1.0 - sum((count / total) ** 2 for count in counts.values())
    # Scale by coverage: an attribute only two thirds of the pool has a value for
    # can only ever split that two thirds.
    return gini * (total / len(pool))


def _softmax(scores) -> list[float]:
    """Which candidate is the target, as a distribution over the head.

    The scoring weights already set the scale of these scores -- the reranker is
    fitted under exactly this exponential-of-score model, pairwise -- so there is no
    temperature to choose here, and choosing one would be a knob fitted to the
    simulator.
    """
    top = max(scores)
    weights = [math.exp(value - top) for value in scores]
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def _expected_reciprocal_rank(beliefs, order) -> float:
    """Where we expect the target to land, under a belief and an ordering."""
    return sum(beliefs[index] / position
               for position, index in enumerate(order, start=1))


def expected_gain(catalog: Catalog, scorer, ranked, attribute: str,
                  depth: int = POOL) -> float:
    """How much answering this question is expected to improve the target's rank.

    The question the current criterion cannot ask. `expected_reduction` measures how
    much of the pool an answer removes, and a question that splits the bottom of the
    pool scores well on it while being unable to move the head at all. What the
    score actually pays for is where the *target* lands, so that is what this
    simulates:

      1. Read a belief over which candidate is the target from the scores already
         computed -- softmax over the head.
      2. For each of the likeliest answers, apply the facet term the scorer would
         apply if the shopper gave that answer, and re-sort. That is not a model of
         the answer's effect; it is literally the effect, computed by the same
         `Weights.facet` the next turn would use.
      3. Update the belief by the same rule, so the belief and the ranking never
         disagree, and read the expected reciprocal rank off the new ordering.
      4. Average over answers, weighted by how much belief mass each one carries.

    A product whose facet is unknown keeps its mass under every answer. That is the
    repository's standing invariant -- missing metadata is uncertainty, not a
    contradiction -- and it is why a question cannot be made to look valuable by
    the thin half of the catalog.

    Returns a difference in expected reciprocal rank, which is directly comparable
    to TURN_COST_IN_MRR.
    """
    head = list(ranked[:depth])
    if len(head) < 2:
        return 0.0
    scores = [score for score, _ in head]
    values = [catalog.facet(pid, attribute) for _, pid in head]
    if not any(value is not None for value in values):
        return 0.0

    beliefs = _softmax(scores)
    order = sorted(range(len(head)), key=lambda i: (-scores[i], head[i][1]))
    before = _expected_reciprocal_rank(beliefs, order)

    mass: dict = collections.defaultdict(float)
    for index, value in enumerate(values):
        if value is not None:
            mass[value] += beliefs[index]
    if not mass:
        return 0.0

    bonus = scorer.w.facet
    gain = 0.0
    for value, probability in sorted(mass.items(), key=lambda pair: -pair[1])[:SIMULATED_ANSWERS]:
        answered = [score + (bonus if values[i] == value else 0.0)
                    for i, score in enumerate(scores)]
        after_order = sorted(range(len(head)),
                             key=lambda i: (-answered[i], head[i][1]))
        gain += probability * (_expected_reciprocal_rank(_softmax(answered), after_order)
                               - before)
    return gain


def _gains(catalog: Catalog, scorer, ranked, asked: set[str]) -> dict:
    """Expected reciprocal-rank improvement for every measurable attribute left.

    Attributes with no values in the head are absent rather than zero, so the
    caller can tell "asking this cannot help" from "this cannot be asked", which
    are different reasons not to ask and only one of them is a measurement.
    """
    out: dict = {}
    for attribute in MEASURABLE:
        if attribute in asked:
            continue
        pool = [pid for _, pid in ranked[:POOL]]
        if expected_reduction(catalog, pool, attribute) is None:
            continue
        out[attribute] = expected_gain(catalog, scorer, ranked, attribute)
    return out


def choose(catalog: Catalog, ranked, asked: set[str],
           model: AnswerModel | None = None, extra: dict | None = None,
           scorer=None) -> str | None:
    """Pick the next attribute to ask about, or None if nothing is left.

    `extra` supplies attributes whose expected reduction the caller has already
    computed because it depends on something outside the catalog -- `category`,
    whose value is how much of the pool the wrong shelves account for (src/clarify.py).
    They compete on the same terms as everything else: a reduction the caller
    measured, multiplied by an answerability the model learned. Nothing is
    special-cased in, which is the point -- a question that no customer answers
    loses this comparison on its own, without a rule saying so.

    Passing `scorer` switches the first factor from pool reduction to the
    counterfactual in `expected_gain`: not how much of the pool an answer removes,
    but how much it is expected to improve where the target lands. See
    analysis/question_value.json for what that is worth.
    """
    pool = [pid for _, pid in ranked[:POOL]]
    model = model or AnswerModel()
    gains = _gains(catalog, scorer, ranked, asked) if scorer is not None else None
    # A stand-in for the attributes the catalog cannot resolve to values, on the
    # scale of the ones it can -- the same role ASSUMED_IMPURITY plays for pool
    # reduction. Taking the mean of what was measured keeps `feature`, the highest
    # yielding question on this simulator, competing rather than winning by units.
    assumed = (sum(gains.values()) / len(gains)) if gains else ASSUMED_IMPURITY
    scale = (assumed / ASSUMED_IMPURITY) if gains else 1.0

    best, best_value = None, 0.0
    contenders: list[tuple[str, float, bool]] = []
    for attribute, reduction in (extra or {}).items():
        if attribute in asked:
            continue
        # Rescaled rather than recomputed: what the category answer removes is
        # shelves, not facet values, so `expected_gain` cannot see it at all. The
        # rescaling leaves its standing against everything else exactly as it was,
        # which is what keeps the clarification question that shelfbench measures
        # at +0.167 hit rate from being quietly switched off by a change of units.
        value = float(reduction) * scale * model.probability(attribute)
        contenders.append((attribute, value, True))
        if value > best_value:
            best, best_value = attribute, value
    for attribute in (*MEASURABLE, *UNMEASURABLE):
        if attribute in asked:
            continue
        measured = attribute in MEASURABLE
        if gains is not None:
            if measured:
                if attribute not in gains:
                    continue      # measurable but the pool has no values for it
                reduction = gains[attribute]
            else:
                reduction = assumed
        else:
            reduction = expected_reduction(catalog, pool, attribute) if pool else None
            if reduction is None:
                if measured:
                    continue      # measurable but the pool has no values for it
                reduction = ASSUMED_IMPURITY
                measured = False
        value = reduction * model.probability(attribute)
        contenders.append((attribute, value, measured))
        if value > best_value:
            best, best_value = attribute, value

    # W1b: the profile's long-term answer history settles an exact tie between
    # questions whose values are MEASURED -- two real estimates the catalog cannot
    # separate. Ties between assumed values are placeholders, not equivalences;
    # the prior does not touch them (it moved the public transcript when it did --
    # see analysis/v8_w1_longterm.json). The prior can never overrule a gap.
    if best is not None and model.has_prior():
        near = [(index, attribute)
                for index, (attribute, value, measured) in enumerate(contenders)
                if measured and best_value - value <= ASK_TIE_EPSILON]
        if len(near) > 1:
            # -index keeps today's loop order when the prior has no opinion;
            # only a genuine recalled-history difference may reorder.
            best = max(near, key=lambda t: (model.prior_probability(t[1]),
                                            -t[0]))[1]

    if best is not None:
        return best
    # Everything measurable is exhausted or covered; fall back in a natural order.
    return next((a for a in UNMEASURABLE if a not in asked), None)
