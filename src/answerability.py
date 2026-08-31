"""Learning which questions a customer can actually answer.

The pre-rebuild agent chose questions by maximum entropy over the candidate set and
asked `brand` 130 times across 200 sessions for a yield of exactly zero. The obvious
diagnosis -- "entropy is biased toward high-cardinality facets" -- turns out to be
the wrong one, and it is worth writing down why, because it determines the fix.

Take the honest information-theoretic criterion: minimise the expected number of
candidates still standing after the answer. For a facet with value distribution p,

    E[remaining | answered] = n * sum(p_v^2)

A high-cardinality facet like `store` has a tiny collision probability, so it
*maximises* expected reduction. Gain ratio, normalised entropy and expected-pool-size
all agree: brand is the best question. Every one of them is right, and every one of
them is useless, because they all condition on the answer arriving.

The missing term is P(the customer can answer at all), and no property of the catalog
can supply it -- it is a fact about the person, not the products. So it has to be
learned from what happens when we ask. That is what this module does, and it is also
the honest version of Pillar III's "long-term user profile": the agent asks, watches
whether anything came back, and stops asking questions that never land.

Deliberately not shipped with a pre-trained table. A table fitted to the public
simulator would score better from turn one and would be exactly the kind of harness
fitting this rebuild exists to remove. The model starts uniform and pays a real
cold-start cost; `snapshot()` exists so the learning curve can be reported.
"""
from __future__ import annotations

import collections

# Beta(1, 1) -- uniform. Two pseudo-observations is enough to keep early estimates
# from swinging on a single answer without drowning out real evidence.
PRIOR_YES = 1.0
PRIOR_NO = 1.0

# W1b: the long-term prior. A profile's recalled asked/answered counts enter as
# pseudo-counts worth at most PRIOR_STRENGTH observations, reaching full weight
# only after PRIOR_FULL_SESSIONS sessions with this signature. Both constants are
# declared structure, not fitted knobs: nothing about them was chosen against
# simulator output (C9). A first-ever session has an empty recall, the prior is
# absent, and the model behaves exactly as it always has.
PRIOR_STRENGTH = 4.0
PRIOR_FULL_SESSIONS = 10


class AnswerModel:
    """Per-attribute P(the customer supplies new information | we ask).

    Shared across sessions on one Agent instance, which is what makes it adaptive
    rather than a per-conversation heuristic. W1b adds the cross-session half the
    pillar asks for: at reset the agent seeds this model with the recalled
    asked/answered counts for the session's profile signature, as a prior that
    in-session evidence overrides -- the prior is worth at most PRIOR_STRENGTH
    pseudo-observations, while real observations accumulate without bound.
    """

    __slots__ = ("asked", "answered", "_prior")

    def __init__(self) -> None:
        self.asked: collections.Counter = collections.Counter()
        self.answered: collections.Counter = collections.Counter()
        self._prior: tuple[dict, dict, float] | None = None

    def set_prior(self, asked: dict | None, answered: dict | None,
                  sessions: int) -> None:
        """Seed from LongTermMemory.recall(). Empty recall means no prior."""
        if not asked or not sessions:
            self._prior = None
            return
        weight = min(1.0, sessions / PRIOR_FULL_SESSIONS)
        self._prior = (dict(asked), dict(answered or {}), weight)

    def observe(self, attribute: str | None, informative: bool) -> None:
        if not attribute:
            return
        self.asked[attribute] += 1
        if informative:
            self.answered[attribute] += 1

    def has_prior(self) -> bool:
        return self._prior is not None

    def probability(self, attribute: str) -> float:
        """Posterior mean of a Beta(1, 1) prior updated with what we have seen."""
        yes = self.answered.get(attribute, 0) + PRIOR_YES
        total = self.asked.get(attribute, 0) + PRIOR_YES + PRIOR_NO
        return yes / total

    def prior_probability(self, attribute: str) -> float:
        """The long-term view: the Beta posterior plus this profile's recalled
        history, worth at most PRIOR_STRENGTH pseudo-observations. Consulted
        only to settle near-ties (elicitation.choose); in-session evidence
        overrides it because real counts accumulate unbounded."""
        value = self.probability(attribute)
        if self._prior:
            prior_asked, prior_answered, weight = self._prior
            asked_count = prior_asked.get(attribute, 0)
            if asked_count:
                yes = self.answered.get(attribute, 0) + PRIOR_YES
                total = self.asked.get(attribute, 0) + PRIOR_YES + PRIOR_NO
                yes += weight * PRIOR_STRENGTH * (prior_answered.get(attribute, 0)
                                                  / asked_count)
                total += weight * PRIOR_STRENGTH
                value = yes / total
        return value

    def confidence(self, attribute: str) -> int:
        """How many times we have actually asked. Used only for reporting."""
        return self.asked.get(attribute, 0)

    def snapshot(self) -> dict:
        """Everything learned so far, for the write-up's learning curve."""
        return {
            attribute: {
                "asked": self.asked[attribute],
                "answered": self.answered[attribute],
                "p_answerable": round(self.probability(attribute), 4),
            }
            for attribute in sorted(self.asked)
        }
