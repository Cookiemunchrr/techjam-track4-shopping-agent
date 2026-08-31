"""Reciprocal rank fusion.

Cormack et al., SIGIR 2009. Combines rankings by position rather than by score, so
routes with incomparable scales -- a BM25 sum, a popularity prior in [0, 1], a
vocabulary similarity -- can be merged without fitting a weight per route.

    score(d) = sum over routes of 1 / (k + rank(d))

Offered as an alternative to the tuned weighted sum in src/scoring.py, not as a
replacement for it. The weighted sum was fitted on dev and is hard to beat on this
data; RRF earns its place by needing no fitting at all, which matters when the
private distribution differs from the public one. Both are measured.
"""
from __future__ import annotations

K = 60.0        # Cormack's default; damps the influence of the very top ranks


def rrf(rankings, k: float = K, weights=None) -> list[tuple[float, str]]:
    """Fuse ranked id lists into one. Deterministic and order-invariant.

    Ties break on the identifier, so passing the same routes in a different order
    cannot change the result -- a fusion that depends on argument order is a bug
    that only shows up when someone reorders a dict.
    """
    scores: dict[str, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[index])
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + rank)
    return sorted(((score, identifier) for identifier, score in scores.items()),
                  key=lambda pair: (-pair[0], pair[1]))
