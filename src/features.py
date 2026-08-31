"""The feature vector, defined once for both training and serving.

A learned reranker has one failure mode that dwarfs every other: the features it
was trained on and the features it is served differ, by a normalisation, a default
or an ordering. The model then behaves plausibly and ranks badly, and nothing in
the training report shows it. So there is exactly one definition of a feature
vector in this repository and it is this file. `tools/rerank_data.py` calls it to
build the training set and `src/rerank.py` calls it to score a live turn; neither
has its own copy.

Everything here is already computed by the existing pipeline -- this is a logging
layer over `src/scoring.py`, not new retrieval. The point of the exercise is to ask
whether a model can combine those same signals better than the hand-fitted weighted
sum in `Weights`, which is a question about the combination and not about the
evidence.

Two features are deliberately absent, and the omissions are the interesting part:

  the category string      `local_evaluator.initial_message` builds turn 1 from
                           `coarse_category(target.categories)` verbatim, so any
                           feature encoding "the query string equals this product's
                           category path" is an inversion of the answer key rather
                           than a retrieval signal. Shelf *membership* is kept --
                           a shelf holds hundreds of products and identifies none
                           of them -- and exact string equality is not.
  anything from the sample  no session id, no ordering, no scenario type. A model
                           that learns "sample 47's target is B0..." would score
                           beautifully on the public set and zero on the private
                           one.
"""
from __future__ import annotations

import collections
import math

# Order is part of the contract with the trained asset. Appending is safe;
# reordering or removing silently invalidates every committed weight, so
# `rerank.Model` checks this list against the asset it loads and refuses a
# mismatch rather than serving a scrambled vector.
FEATURES = (
    "popularity",          # global, [0, 1]
    "shelf_popularity",    # within its own shelf, [0, 1] -- separates the long tail
    "stars",               # mean rating, scaled
    "bm25",                # lexical agreement with the stated constraints
    "phrase",              # verbatim containment of a stated clause
    "facet",               # is the stated material/colour, not merely mentions it
    "budget",              # signed distance from a stated price, log space
    "primary",             # sits on the shelf we read as named
    "exact_shelf",         # ...and that shelf was named outright, not inferred
    "refused",             # matches something the shopper explicitly refused
    "shown",               # already put in front of them and passed over
    "has_price",           # metadata completeness: 78.9% of the catalog has none
    "blob",                # how much text there is to match against, log-scaled
    "bm25_x_constraints",  # interaction: lexical evidence is worth more once the
                           # shopper has actually said several things
    "phrase_x_constraints",
)

BM25_SCALE = 12.0      # matches scoring.BM25_SCALE; keeps the feature near [0, 1]
BLOB_SCALE = 8.0       # log(len) is ~5-8 across this catalog


class Context:
    """Everything about one turn that every candidate is scored against.

    Built once per turn and reused for all candidates, because the expensive part
    (tokenising the constraint text, resolving facets, assembling the refusal sets)
    does not depend on which product is being looked at.
    """

    __slots__ = ("catalog", "scorer", "query", "clauses", "facets", "budget",
                 "primary", "exact", "shown", "refused", "constraints")

    def __init__(self, catalog, scorer, clauses, facets=None, budget=None,
                 primary=frozenset(), exact=False, shown=frozenset(),
                 refused=None, constraints=0) -> None:
        from .text import tokens
        self.catalog = catalog
        self.scorer = scorer
        self.clauses = list(clauses or ())
        self.query: collections.Counter = collections.Counter()
        for text, weight in scorer._weighted(self.clauses):
            for term in tokens(text):
                self.query[term] += weight
        self.facets = facets or {}
        self.budget = budget
        self.primary = primary or frozenset()
        self.exact = bool(exact)
        self.shown = shown or frozenset()
        # pid -> penalty, from Agent._refusals. Passed in rather than recomputed so
        # the training set sees exactly what the live path saw.
        self.refused = refused or {}
        # How much the shopper has actually pinned down. Constant across candidates,
        # so it earns its place only through the interaction terms below -- a
        # per-group constant cannot separate two members of that group.
        self.constraints = float(constraints)


def vector(pid: str, ctx: Context) -> list[float]:
    """The feature vector for one candidate under one turn's context."""
    catalog, scorer = ctx.catalog, ctx.scorer
    meta = catalog.meta[pid]

    bm25 = scorer._bm25(pid, ctx.query) if ctx.query else 0.0
    phrase = scorer._phrase(pid, ctx.clauses) if ctx.clauses else 0.0
    return [
        catalog.popularity(pid),
        catalog.bucket_rank.get(pid, 0.5),
        meta["stars"] / 5.0,
        bm25,
        phrase,
        scorer._facet(pid, ctx.facets) if ctx.facets else 0.0,
        scorer._budget(pid, ctx.budget) if ctx.budget else 0.0,
        1.0 if pid in ctx.primary else 0.0,
        1.0 if (ctx.exact and pid in ctx.primary) else 0.0,
        -float(ctx.refused.get(pid, 0.0)),
        1.0 if pid in ctx.shown else 0.0,
        1.0 if meta["price"] else 0.0,
        math.log1p(catalog.length[pid]) / BLOB_SCALE,
        bm25 * ctx.constraints,
        phrase * ctx.constraints,
    ]
