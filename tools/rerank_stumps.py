"""Gradient-boosted stumps over the same features: is it the model class?

Two of the three levers on this reranker are now measured and closed. More
training data buys nothing (analysis/reranker_experiment.json, ceiling_probe), and
the obvious next batch of features buys nothing either
(analysis/reranker_features.json). The third is the shape of the model: the shipped
one is linear in fifteen features, and a linear model cannot say "popularity counts
for less once the shopper has named a material", however the weights are set. Only
an interaction-capable class can, and this is the cheapest honest one.

## The model

Boosted depth-1 or depth-2 regression trees under the same pairwise objective the
linear model uses -- RankNet with trees rather than with a dot product. For every
(target, other) pair in a snapshot,

    loss = -log sigmoid(F(x_target) - F(x_other))

Each round computes the per-candidate gradient of that loss summed over the pairs
the candidate takes part in, fits one small tree to those gradients by least
squares, and adds it at a learning rate. Inference is a handful of comparisons per
candidate, so a tree model is no more expensive to serve than the dot product.

## Why it is expected to overfit, and what is done about it

148 snapshots and 5,558 pairwise comparisons is thin for trees, which can carve
that many points into memorised regions. Three things push back, and none of them
is a judgement call made after seeing the answer: the depth is 1 or 2, the learning
rate is small, and the number of rounds is chosen by the same grouped 5-fold
cross-validation inside dev that picks the linear model's regularisation -- folds
split on session, never on snapshot. Holdout is read once, at the end.

## The bar

The plan's GATES-R, with (a) strengthened: a tree model must beat the *linear*
model on holdout snapshot MRR with an interval excluding zero. Not the weighted
sum, and not on dev. If it cannot, the linear model stays -- it is simpler, its
weights are readable, its asset is fifteen numbers, and explainability is a judged
criterion here rather than a preference.

    python3 -m tools.rerank_stumps
    python3 -m tools.rerank_stumps --depth 2 --rounds 300
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.features import FEATURES
from tools import rerank_train as trainer
from tools import snapshot_mrr as metric

ROUNDS = 200
RATE = 0.1
DEPTH = 1
MIN_LEAF = 20          # a split that isolates fewer candidates than this is memory
FOLDS = 5


class TreeModel:
    """A boosted-tree scorer with the interface tools/snapshot_mrr.py measures.

    Deliberately not in src/. Nothing here reaches the agent unless the model
    clears the bar, and an experiment that has not cleared its bar has no business
    adding a serving path.
    """

    __slots__ = ("trees", "rate", "blend", "meta")

    def __init__(self, trees, rate: float = RATE, blend: float = 1.0, meta=None) -> None:
        self.trees = list(trees)
        self.rate = float(rate)
        self.blend = float(blend)
        self.meta = dict(meta or {})

    def score_vector(self, x) -> float:
        return self.rate * sum(_evaluate(tree, x) for tree in self.trees)

    def to_dict(self) -> dict:
        return {"kind": "boosted_stumps", "features": list(FEATURES),
                "trees": self.trees, "rate": self.rate, "blend": self.blend,
                "meta": self.meta}


def _evaluate(node, x) -> float:
    while isinstance(node, dict):
        node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
    return float(node)


def rows_of(groups):
    """Every candidate once, with the group it belongs to. Trees fit points, not
    differences: the pairwise loss lives in the gradient, not in the base learner."""
    xs, index = [], []
    for group in groups:
        start = len(xs)
        for row in group["rows"]:
            xs.append(row["x"])
        index.append((start, len(xs), [i for i, row in enumerate(group["rows"], start)
                                       if row["y"]]))
    return xs, index


def gradients(scores, index) -> list[float]:
    """-dL/dF for every candidate, summed over the pairs it appears in.

    A target is pushed up once for each non-target it has not yet beaten
    convincingly; each non-target is pushed down once for each target above it. The
    push is (1 - sigmoid(margin)), so a comparison the model already wins by a wide
    margin contributes almost nothing -- which is what stops the easy majority of
    pairs from drowning out the ones still being lost.
    """
    grad = [0.0] * len(scores)
    for start, end, positives in index:
        for p in positives:
            for o in range(start, end):
                if o == p:
                    continue
                margin = scores[p] - scores[o]
                slack = 1.0 / (1.0 + math.exp(margin)) if margin > -30 else 1.0
                grad[p] += slack
                grad[o] -= slack
    return grad


def _order(xs) -> list[list[int]]:
    """Row indices sorted by each feature. Computed once; splits never change it."""
    return [sorted(range(len(xs)), key=lambda i: xs[i][f]) for f in range(len(FEATURES))]


def _best_split(xs, grad, rows, order, min_leaf: int):
    """The (feature, threshold) minimising squared error on the gradients.

    Standard regression-stump fitting: for one feature, walk the rows in sorted
    order and track the running sums, which gives every candidate threshold's split
    quality in one pass.
    """
    member = set(rows)
    total = sum(grad[i] for i in rows)
    count = len(rows)
    if count < 2 * min_leaf:
        return None
    best = None
    for feature in range(len(FEATURES)):
        left_sum = 0.0
        left_count = 0
        ordered = [i for i in order[feature] if i in member]
        for position, index in enumerate(ordered[:-1]):
            left_sum += grad[index]
            left_count += 1
            if xs[index][feature] == xs[ordered[position + 1]][feature]:
                continue          # cannot split between equal values
            right_count = count - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            # Reduction in squared error from splitting here.
            gain = (left_sum * left_sum / left_count
                    + (total - left_sum) ** 2 / right_count - total * total / count)
            if best is None or gain > best[0]:
                threshold = (xs[index][feature] + xs[ordered[position + 1]][feature]) / 2.0
                best = (gain, feature, threshold)
    return best


def _grow(xs, grad, rows, order, depth: int, min_leaf: int):
    if depth <= 0:
        return sum(grad[i] for i in rows) / len(rows) if rows else 0.0
    split = _best_split(xs, grad, rows, order, min_leaf)
    if split is None:
        return sum(grad[i] for i in rows) / len(rows) if rows else 0.0
    _, feature, threshold = split
    left = [i for i in rows if xs[i][feature] <= threshold]
    right = [i for i in rows if xs[i][feature] > threshold]
    if not left or not right:
        return sum(grad[i] for i in rows) / len(rows)
    return {"f": feature, "t": round(threshold, 6),
            "l": _grow(xs, grad, left, order, depth - 1, min_leaf),
            "r": _grow(xs, grad, right, order, depth - 1, min_leaf)}


def boost(groups, rounds: int, rate: float, depth: int, min_leaf: int,
          watch=None, report_every: int = 10):
    """Fit `rounds` trees, optionally scoring a watch set after each block.

    Returns (trees, curve). The curve is what early stopping reads; it is never
    computed on the data being fitted.
    """
    xs, index = rows_of(groups)
    order = _order(xs)
    scores = [0.0] * len(xs)
    everything = list(range(len(xs)))
    trees, curve = [], []
    for step in range(1, rounds + 1):
        tree = _grow(xs, gradients(scores, index), everything, order, depth, min_leaf)
        if not isinstance(tree, dict):
            break                 # nothing left to split on
        trees.append(tree)
        for i in everything:
            scores[i] += rate * _evaluate(tree, xs[i])
        if watch is not None and (step % report_every == 0 or step == rounds):
            curve.append({"rounds": step,
                          "held_pairwise_accuracy": round(
                              pairwise_accuracy(watch, TreeModel(trees, rate)), 5)})
    return trees, curve


def pairwise_accuracy(groups, model) -> float:
    """Share of (target, other) comparisons the model gets the right way round."""
    right = total = 0
    for group in groups:
        scored = [(model.score_vector(row["x"]), row["y"]) for row in group["rows"]]
        for value, is_target in scored:
            if not is_target:
                continue
            for other, other_target in scored:
                if other_target:
                    continue
                total += 1
                right += int(value > other)
    return right / total if total else 0.0


def choose_rounds(dev, rounds: int, rate: float, depth: int, min_leaf: int):
    """Grouped 5-fold CV inside dev. Folds split on session, never on snapshot."""
    accumulated: dict = {}
    for train_groups, held_groups in trainer.folds(dev, FOLDS):
        _, curve = boost(train_groups, rounds, rate, depth, min_leaf, watch=held_groups)
        for point in curve:
            accumulated.setdefault(point["rounds"], []).append(
                point["held_pairwise_accuracy"])
    curve = [{"rounds": k, "cv_pairwise_accuracy": round(sum(v) / len(v), 5)}
             for k, v in sorted(accumulated.items()) if len(v) == FOLDS]
    best = max(curve, key=lambda point: point["cv_pairwise_accuracy"])
    return best["rounds"], best["cv_pairwise_accuracy"], curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Boosted stumps for the reranker")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--rate", type=float, default=RATE)
    parser.add_argument("--depth", type=int, default=DEPTH, choices=(1, 2))
    parser.add_argument("--min-leaf", type=int, default=MIN_LEAF)
    parser.add_argument("--blends", default="0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--baseline", default="analysis/reranker.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="analysis/reranker_stumps.json")
    args = parser.parse_args()

    dev, holdout = trainer.load(trainer.DEV), trainer.load(trainer.HOLDOUT)
    print(f"dev: {len(dev)} snapshots, {len(trainer.pairs(dev))} pairwise comparisons")

    rounds, cv, curve = choose_rounds(dev, args.rounds, args.rate, args.depth,
                                      args.min_leaf)
    print(f"grouped-CV chose {rounds} rounds at pairwise accuracy {cv:.4f}")
    trees, _ = boost(dev, rounds, args.rate, args.depth, args.min_leaf)
    model = TreeModel(trees, args.rate)
    print("pairwise accuracy   dev %.4f   holdout %.4f"
          % (pairwise_accuracy(dev, model), pairwise_accuracy(holdout, model)))

    snapshots = {split: metric.cached(split) for split in ("dev", "holdout")}
    linear = metric.load_model(args.baseline)
    blends = [float(b) for b in args.blends.split(",") if b.strip()]
    sweep = []
    for blend in blends:
        model.blend = blend
        row = {"blend": blend}
        for split, groups in snapshots.items():
            low, mid, high, _ = metric.paired(groups, linear, model, args.seed)
            row[split] = {"snapshot_mrr": round(metric.mrr(groups, model), 5),
                          "delta_vs_linear": round(mid, 5),
                          "ci95": [round(low, 5), round(high, 5)],
                          "verdict": "significant" if low > 0 or high < 0 else "within noise"}
        sweep.append(row)
        print("  blend %-6s dev %+.4f [%+.4f, %+.4f]   holdout %+.4f [%+.4f, %+.4f]  %s"
              % (blend, row["dev"]["delta_vs_linear"], *row["dev"]["ci95"],
                 row["holdout"]["delta_vs_linear"], *row["holdout"]["ci95"],
                 row["holdout"]["verdict"]))

    # The blend is chosen on dev, as the linear model's was; holdout is reported
    # at that choice and never used to pick it.
    chosen = max(sweep, key=lambda row: row["dev"]["delta_vs_linear"])
    model.blend = chosen["blend"]
    payload = {
        "question": "Two levers on this reranker are closed -- more data buys nothing, "
                    "and the obvious next features buy nothing. Is the third one, the "
                    "model class, where the remaining ordering headroom is?",
        "model": {"kind": "boosted regression trees under the pairwise logistic "
                          "objective (RankNet with trees)",
                  "depth": args.depth, "learning_rate": args.rate,
                  "min_leaf": args.min_leaf, "rounds_offered": args.rounds,
                  "rounds_chosen": rounds,
                  "rounds_chosen_by": "grouped 5-fold CV within dev; folds split on "
                                      "session, never on snapshot",
                  "cv_pairwise_accuracy": cv,
                  "dev_pairwise_accuracy": round(pairwise_accuracy(dev, model), 5),
                  "holdout_pairwise_accuracy": round(pairwise_accuracy(holdout, model), 5),
                  "trees": len(trees)},
        "baseline": args.baseline,
        "cv_curve": curve,
        "blend_sweep_snapshot_mrr_vs_linear": sweep,
        "blend_chosen_on_dev": chosen["blend"],
        "asset": model.to_dict(),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
