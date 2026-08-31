"""The anonymized user profile, used as a tie-breaker and never as a driver.

docs/agent_api_contract.json supplies `preference_tags`, `average_prior_rating`,
`purchase_frequency` and `rating_style`. The signal is real but weak: a preference
tag appears in the target's text 44.0% of the time against 31.2% for a random
product in the same bucket, a 1.41x lift, and wiring it in as a scoring term
previously cost 0.005 because the lift was smaller than the noise it introduced.

So it is bounded here rather than weighted. CAP sits strictly below the phrase
weight in src/scoring.py, which makes the guarantee testable: nothing the profile
knows can outrank something the customer actually said. That is the right shape for
personalization in a shopping surface regardless of what it does to the score. A
system that overrides a stated requirement because of purchase history is broken
even on the occasions when it happens to be right.
"""
from __future__ import annotations

from .catalog import Catalog
from .text import normalise

CAP = 0.30              # strictly below Weights.phrase, which is 1.00
PER_TAG = 0.10
RATING_BONUS = 0.06
BUCKET_BONUS = 0.05     # long-term shelf affinity (W1a), still inside CAP
TIE_EPSILON = 0.02      # scores this close are, for our purposes, the same score
TIE_DEPTH = 60          # only reorder near the top; the tail is not worth touching


class ProfilePrior:
    """A small, capped preference bonus derived from the aggregate profile.

    W1a adds the long-term half: shelves this profile signature has converged
    on in earlier sessions (LongTermMemory.recall's "buckets"). Same guarantee
    as everything else here -- bounded, inside CAP, tie-break only.
    """

    __slots__ = ("tags", "prior_rating", "recalled_buckets")

    CAP = CAP           # the guarantee, reachable from the class it constrains

    def __init__(self, user_profile: dict | None,
                 recalled_buckets: dict | None = None) -> None:
        profile = user_profile if isinstance(user_profile, dict) else {}
        raw = profile.get("preference_tags")
        self.tags = ([normalise(str(tag)) for tag in raw
                      if isinstance(tag, str) and len(str(tag)) > 2]
                     if isinstance(raw, list) else [])
        rating = profile.get("average_prior_rating")
        self.prior_rating = float(rating) if isinstance(rating, (int, float)) else None
        # Recalled keys are dialog category phrases; they match bucket_of names
        # when the earlier session's shelf resolved exactly, and simply never
        # fire when it did not. Conservative by construction.
        self.recalled_buckets = (dict(recalled_buckets)
                                 if isinstance(recalled_buckets, dict) else {})

    def bonus(self, catalog: Catalog, pid: str) -> float:
        """Bounded lift for one product. Never exceeds CAP."""
        if pid not in catalog.corpus:
            return 0.0
        total = 0.0
        if self.tags:
            blob = catalog.corpus[pid]
            total += PER_TAG * sum(1 for tag in self.tags if tag and tag in blob)
        if self.prior_rating is not None:
            stars = catalog.meta[pid]["stars"]
            if stars and abs(stars - self.prior_rating) <= 0.5:
                total += RATING_BONUS
        if self.recalled_buckets:
            shelf = catalog.bucket_of.get(pid)
            if shelf is not None and shelf in self.recalled_buckets:
                total += BUCKET_BONUS
        return min(total, CAP)

    def __bool__(self) -> bool:
        return bool(self.tags) or self.prior_rating is not None \
            or bool(self.recalled_buckets)


def break_ties(ranked, prior: ProfilePrior, catalog: Catalog,
               epsilon: float = TIE_EPSILON, depth: int = TIE_DEPTH):
    """Reorder only runs of candidates the ranking cannot separate.

    Applied as an additive bonus instead, this was actively harmful: measured on
    the public set it cost 0.20, because `preference_tags` are category-level
    abstractions -- the literal strings "material", "fit", "comfort" -- and every
    clothing listing contains those words. Substring-matching them against product
    text is close to a random number, and a random number worth up to CAP is more
    than enough to displace a correct top-1.

    So it only ever chooses between candidates that are already tied. A profile can
    settle a coin flip. It cannot overrule the conversation.
    """
    if not prior or not ranked:
        return ranked
    head, tail = list(ranked[:depth]), list(ranked[depth:])
    out: list = []
    index = 0
    while index < len(head):
        run_end = index + 1
        while run_end < len(head) and abs(head[index][0] - head[run_end][0]) < epsilon:
            run_end += 1
        run = head[index:run_end]
        if len(run) > 1:
            run.sort(key=lambda pair: (-(pair[0] + prior.bonus(catalog, pair[1])), pair[1]))
        out.extend(run)
        index = run_end
    return out + tail
