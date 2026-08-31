"""Candidate scoring: popularity prior + BM25 + phrase containment + budget proximity.

Weights are deliberately few and were fitted on public sessions 1-100 only; see
analysis/ablations_v2.csv for the held-out numbers behind each term.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

from .catalog import COLORS, MATERIALS, Catalog
from .text import normalise, tokens

BM25_K1 = 2.5
BM25_B = 0.75
BM25_SCALE = 12.0          # maps a typical BM25 sum into roughly [0, 1]
PHRASE_MIN_CHARS = 9       # below this a "phrase" is really just a word
# ...unless the whole phrase *is* the constraint. "leather" (7), "cotton" (6) and
# "denim" (5) fall under the floor, yet they are among the most discriminative
# things a shopper says. 27.6% of the constraints this simulator discloses are
# shorter than the floor, and dropping them costs 0.008.
SHORT_EXACT = frozenset(MATERIALS) | frozenset(COLORS)
PHRASE_CAP = 2.0           # stop rewarding ever-longer verbatim quotes

# How far off a stated budget a product can be before the miss stops getting worse.
# log(4), so four times or a quarter of the stated figure is the floor. Beyond that
# the product is not what they asked for and there is nothing left to express.
BUDGET_CLIP = 1.3863



@dataclass(frozen=True)
class Weights:
    """Scoring weights, fitted on public sessions 1-100 only.

    The popularity prior is large because the targets are real 5-core purchases,
    but stated evidence still outweighs it roughly 5.8:1 on the measured sessions.
    test_scoring_policy pins that ratio at 3:1 -- drifting below it would mean the
    conversation has stopped mattering, which is both worse product behaviour and
    the long-tail pathology the report calls out.
    """
    popularity: float = 1.40
    lexical: float = 0.40
    phrase: float = 1.00
    shown_penalty: float = 0.80
    # Not fitted, and there is nothing here to fit it on: no public session ever
    # discloses a budget (0/200 -- `intent_card` appends it last and truncates it
    # away). Set from what a budget means to a shopper: a strong preference, never
    # a filter. At this weight a product priced at twice the stated figure loses
    # 0.42, which is real pressure and still less than the popularity term can
    # answer with -- so a genuinely better match that costs more still wins.
    budget: float = 0.60
    # A product that *is* the stated material outranks one that merely mentions it.
    # Worth less than the phrase term it refines rather than replaces: the phrase
    # term already fires on the same word, and Catalog.facet reads the first match
    # in the blob, which is a good guess at the primary material and not a fact.
    facet: float = 0.50

    @classmethod
    def from_env(cls, env) -> "Weights":
        def f(name, default):
            try:
                return float(env.get(name, default))
            except (TypeError, ValueError):
                return default
        return cls(f("W_POP", 1.40), f("W_TXT", 0.40), f("W_PHRASE", 1.00),
                   f("P_SOFT", 0.80), f("W_BUDGET", 0.60), f("W_FACET", 0.50))


class Scorer:
    def __init__(self, catalog: Catalog, weights: Weights | None = None) -> None:
        self.catalog = catalog
        self.w = weights or Weights()

    def _bm25(self, pid: str, query: collections.Counter) -> float:
        cat = self.catalog
        tf, length = cat.tf[pid], cat.length[pid]
        norm = BM25_B * length / cat.avg_length + (1.0 - BM25_B)
        total = 0.0
        for term, qn in query.items():
            freq = tf.get(term, 0)
            if freq:
                total += qn * cat.idf(term) * (freq * BM25_K1) / (freq + BM25_K1 * norm)
        return total / BM25_SCALE

    @staticmethod
    def _weighted(clauses):
        """Accept either plain strings or (text, weight) pairs, uniformly."""
        out = []
        for clause in clauses or ():
            if isinstance(clause, (tuple, list)) and len(clause) == 2:
                out.append((str(clause[0]), float(clause[1])))
            else:
                out.append((str(clause), 1.0))
        return out

    def _phrase(self, pid: str, clauses) -> float:
        """Reward verbatim containment of a stated clause.

        Real shoppers do quote spec strings ("100% cotton"), so this is a genuine
        product-search signal -- but it is one term among several, never a filter.
        """
        blob = self.catalog.corpus[pid]
        total = 0.0
        for text, weight in self._weighted(clauses):
            text = normalise(text)
            if not text or text not in blob:
                continue
            if len(text) >= PHRASE_MIN_CHARS or text in SHORT_EXACT:
                total += weight * (1.0 + min(len(text) / 60.0, PHRASE_CAP))
        return total

    def _budget(self, pid: str, budget: float) -> float:
        """How far this product's price is from the stated budget, in log space.

        Log space so that $15-against-$20 and $150-against-$200 are the same miss;
        a shopper's tolerance is proportional, not absolute. Returns zero --
        neutral, not penalised -- when the catalog has no price for the product:
        78.9% of this catalog does not, and hiding four fifths of a shelf because
        the metadata is thin is a worse answer than ignoring the budget entirely.

        Negative-only by construction, so a product cannot *earn* rank for being
        cheap. Being at budget is the absence of a problem, not a virtue -- an
        agent that rewarded proximity would push the cheapest thing on the shelf
        at a shopper who only meant "not more than about this".
        """
        price = self.catalog.meta[pid]["price"]
        if not price or price <= 0:
            return 0.0
        amount = getattr(budget, "amount", budget)
        if not amount or amount <= 0:
            return 0.0
        if getattr(budget, "cap", False) and price <= amount:
            return 0.0                      # a ceiling is satisfied from below
        if getattr(budget, "floor", False) and price >= amount:
            return 0.0                      # ...and a floor from above
        return -min(abs(math.log(price / amount)), BUDGET_CLIP)

    def explain(self, pid: str, clauses, budget: float | None = None) -> dict:
        """The score, broken into the terms that produced it.

        "Transparent recommendation explanations" is a listed innovation direction,
        and it is also how anyone debugs a ranking they disagree with.
        """
        query: collections.Counter = collections.Counter()
        for text, weight in self._weighted(clauses):
            for term in tokens(text):
                query[term] += weight
        w = self.w
        parts = {
            "popularity": w.popularity * self.catalog.popularity(pid),
            "lexical": w.lexical * self._bm25(pid, query) if query else 0.0,
            "phrase": w.phrase * self._phrase(pid, clauses or ()) if clauses else 0.0,
            "budget": w.budget * self._budget(pid, budget) if budget else 0.0,
        }
        parts["total"] = sum(parts.values())
        return parts

    def _facet(self, pid: str, facets: dict) -> float:
        """How many stated facets this product actually has.

        Equality against the product's *primary* value for the attribute, not
        membership in the set of every value its text contains. That is a measured
        choice and the measurement went against the intuition:

            equality on the primary value      dev 0.91517  holdout 0.90252
            membership in the stated set       dev 0.91575  holdout 0.89597
            membership, description included   dev 0.91575  holdout 0.89382

        Multi-valued facets were supposed to fix the canvas-bag-with-leather-trim
        case. They do, and they break a larger one: a listing that reads
        "polyester, cotton lining" matches a shopper asking for cotton just as
        strongly as a cotton shirt does, and there are many more of those than
        there are trims. Membership is the less discriminative test, and on this
        catalog that costs more than the precision it buys. Widening the credit
        made it monotonically worse, which is the signature of a term that has
        stopped separating anything.

        The set is still built and still used -- by the refusal logic in
        src/agent.py, where the asymmetry runs the other way: wrongly penalising
        a product for a word in its prose is a worse error than missing one.

        A value may itself be a set: "blue or green" is satisfied by either, so an
        alternative set scores its best member rather than the sum.
        """
        total = 0.0
        for attribute, value in facets.items():
            wanted = value if isinstance(value, (tuple, list, set, frozenset)) else (value,)
            got = self.catalog.facet(pid, attribute)
            if got is not None and any(got == one for one in wanted):
                total += 1.0
        return total

    def rank(self, candidates, clauses, shown=frozenset(), boost=None,
             budget: float | None = None, facets: dict | None = None) -> list[tuple[float, str]]:
        """Score every candidate. Returns [(score, product_id)] best first.

        `boost` is an optional per-item additive bonus. The browsing track uses it
        to keep the named category ahead of the adjacent categories it pulls in:
        a customer exploring sneakers should see sandals, but below the sneakers.
        """
        query: collections.Counter = collections.Counter()
        for text, weight in self._weighted(clauses):
            for term in tokens(text):
                query[term] += weight

        w, cat = self.w, self.catalog
        scored: list[tuple[float, str]] = []
        for pid in candidates:
            score = w.popularity * cat.popularity(pid)
            if query:
                score += w.lexical * self._bm25(pid, query)
            if clauses:
                score += w.phrase * self._phrase(pid, clauses)
            if budget:
                score += w.budget * self._budget(pid, budget)
            if facets:
                score += w.facet * self._facet(pid, facets)
            if pid in shown:
                # Weak negative evidence: the shopper saw it and moved on. Not proof.
                score -= w.shown_penalty
            if boost:
                score += boost.get(pid, 0.0)
            scored.append((score, pid))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored
