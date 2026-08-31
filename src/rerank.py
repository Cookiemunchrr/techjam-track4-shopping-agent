"""A learned reordering of the head of the ranking.

The weighted sum in `src/scoring.py` has six terms whose weights were fitted by
hand on public sessions 1-100. The question this module asks is narrow and worth
asking: given the same evidence, can a model trained on the same sessions combine
it better?

Narrow on purpose. It reorders the top `RERANK_DEPTH` of a ranking the existing
pipeline produced; it never retrieves. Nothing below that line can be promoted and
nothing above it can be lost, so a bad model costs ordering and can never cost
recall -- which matters because `tools/recall.py` says recall is already 1.000 on
every official session and ordering is where all the remaining headroom is.

Runtime is stdlib and offline, as everything here is: the trained asset is a JSON
file of coefficients, and inference is a dot product per candidate. Training lives
in `tools/rerank_train.py` and runs at build time only.

If `analysis/reranker.json` is absent the agent runs exactly as it did before --
`load()` returns None and `Agent.reranker` stays None. That is deliberate: the
model is an experiment that has to earn its place against a ship bar
(README, "Two doors"), and the code has to be honest about the case where it does
not.
"""
from __future__ import annotations

import json
from pathlib import Path

from .features import FEATURES, vector
from .paths import ROOT

ASSET = "analysis/reranker.json"
# How much of the existing ranking the model is allowed to reorder. Deep enough to
# matter -- the target is inside the top ten on every official session, so the work
# is all near the head -- and bounded so per-turn latency stays flat.
DEPTH = 200


class Model:
    """Linear scorer over `features.FEATURES`, loaded from a committed asset."""

    __slots__ = ("weights", "bias", "blend", "meta")

    def __init__(self, weights, bias: float = 0.0, blend: float = 1.0, meta=None) -> None:
        if len(weights) != len(FEATURES):
            raise ValueError(
                f"reranker asset has {len(weights)} weights, features.FEATURES has "
                f"{len(FEATURES)}. The asset was trained against a different vector; "
                f"retrain it rather than serving a scrambled one.")
        self.weights = [float(w) for w in weights]
        self.bias = float(bias)
        # How far the model is allowed to move things, relative to the score the
        # weighted sum already produced. 1.0 replaces the ordering of the head
        # outright; below that the model argues and the weighted sum decides ties.
        self.blend = float(blend)
        self.meta = dict(meta or {})

    def score_vector(self, x) -> float:
        """The model's opinion of one already-extracted feature vector.

        Split out from `score` so an offline measurement (tools/snapshot_mrr.py)
        can score recorded vectors without a catalog, and score them through the
        same arithmetic the live path uses rather than a re-implementation of it.
        """
        total = self.bias
        for weight, value in zip(self.weights, x):
            total += weight * value
        return total

    def score(self, pid: str, ctx) -> float:
        return self.score_vector(vector(pid, ctx))

    def apply(self, ranked, ctx, depth: int = DEPTH):
        """Reorder the head of `ranked`. The tail is returned untouched."""
        head, tail = list(ranked[:depth]), list(ranked[depth:])
        if not head:
            return ranked
        rescored = [(base + self.blend * self.score(pid, ctx), pid)
                    for base, pid in head]
        # Ties break on the identifier, as everywhere else here, so the result does
        # not depend on the order the pool happened to arrive in.
        rescored.sort(key=lambda pair: (-pair[0], pair[1]))
        return rescored + tail

    @classmethod
    def from_dict(cls, payload: dict) -> "Model":
        names = payload.get("features")
        if names is not None and list(names) != list(FEATURES):
            raise ValueError(
                "reranker asset lists different features from features.FEATURES; "
                "retrain it. Serving a vector in the wrong order fails silently, "
                "which is the whole reason this check exists.")
        return cls(payload["weights"], payload.get("bias", 0.0),
                   payload.get("blend", 1.0), payload.get("meta"))


def load(path: str | Path | None = None) -> "Model | None":
    """The committed model, or None if there is not one.

    Never raises on a missing file: shipping without the asset is a supported
    state, and the agent must start in an environment that has only what the
    submission bundle carries.
    """
    candidate = Path(path) if path else (ROOT / ASSET)
    if not candidate.exists():
        return None
    with candidate.open(encoding="utf-8") as handle:
        return Model.from_dict(json.load(handle))
