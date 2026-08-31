"""Train the legacy A0 pairwise reranker compatibility control. Build time only.

This is intentionally the pre-V4 objective: standalone target/negative differences,
shared dev scaling, grouped-CV pair accuracy selection, and a final one-time holdout
report. It exists to reproduce and diagnose A0; it does not implement V4-1's deployed-
residual, nested session-MRR, reachability, or fold-local objective ladder.

A ranking model is not a classifier. "Is this product the target" is answerable by
predicting "no" every time, and a model that learns the popularity prior scores well
on it. What matters is whether the target is ordered above the products it is
competing with, so the objective is pairwise: for every snapshot, every
(target, non-target) pair contributes

    loss = -log sigmoid(w . (x_target - x_other))

which is ordinary logistic regression on feature *differences*. It has no intercept
-- a constant added to every candidate in a group cannot reorder that group -- and
that is why `Model.bias` exists only to absorb standardisation.

Pure standard library, full-batch gradient descent with L2. The set is small enough
(a few thousand pairs) that this converges in a second and needs no sampling, and
avoiding numpy keeps the training story as reproducible as the runtime one.

## What is fitted where

R1 in the repository's working rules: fit on dev, validate on holdout, report on
both. The regularisation strength is chosen by 5-fold **grouped** cross-validation
*within dev* -- folds split on session, never on snapshot, because two snapshots
from the same session share a target and splitting between them leaks it. Holdout
is read once, at the end, and never used to choose anything.

    python3 -m tools.rerank_train
    python3 -m tools.rerank_train --blend 0.5

The candidate it writes is inert. `src/rerank.py` loads the already-shipped
`analysis/reranker.json` and nothing else, so this compatibility trainer cannot replace
that asset unless somebody explicitly selects the serving output path after completing
the applicable gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.features import FEATURES
from tools.rerank_data import (CATALOG, NEGATIVES, SPLITS,
                               load_training_cache)
from tools.rerank_provenance import CacheError

DEV = "analysis/rerank_dev.jsonl"
HOLDOUT = "analysis/rerank_holdout.jsonl"
LAMBDAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
EPOCHS = 400
RATE = 0.5
FOLDS = 5


def load(path: str | Path, *, catalog: str | Path = CATALOG,
         split_path: str | Path | None = None,
         negatives: int = NEGATIVES) -> list[dict]:
    """Load only a fully validated V4 training cache.

    A feature-list check alone admits semantically stale rows.  Default cache
    paths have an unambiguous split; custom cache paths must name their ordered
    source split explicitly so provenance can never be guessed from a filename.
    """
    candidate = Path(path)
    if split_path is None:
        known = {
            Path(DEV): Path(SPLITS["dev"]),
            Path(HOLDOUT): Path(SPLITS["holdout"]),
        }
        split_path = known.get(candidate)
    if split_path is None:
        raise SystemExit(
            f"cannot validate {path}: pass the exact ordered split path")
    try:
        return load_training_cache(candidate, catalog, split_path, negatives)
    except CacheError as exc:
        raise SystemExit(
            f"{path} is not a valid V4 training cache: {exc}. "
            "Rebuild it explicitly with tools/rerank_data.py.") from exc


def pairs(groups) -> list[list[float]]:
    """x_target - x_other, one row per comparison the model has to win."""
    out: list[list[float]] = []
    for group in groups:
        positives = [row["x"] for row in group["rows"] if row["y"]]
        negatives = [row["x"] for row in group["rows"] if not row["y"]]
        for positive in positives:
            for negative in negatives:
                out.append([a - b for a, b in zip(positive, negative)])
    return out


def standardise(rows):
    """Per-feature mean and standard deviation, folded into the weights later.

    Gradient descent on raw features would be dominated by whichever one happens to
    have the largest scale. Standardising is a property of the optimiser, not of the
    model, so it is undone before export and the runtime stays a plain dot product.

    The width is read from the rows rather than from `FEATURES`, so the same
    trainer fits a subset of the vector -- which is what tools/rerank_ablate.py
    needs to ask what any one feature is worth.
    """
    n = len(rows)
    if not n:
        return [], []
    width = len(rows[0])
    mean = [sum(row[i] for row in rows) / n for i in range(width)]
    var = [sum((row[i] - mean[i]) ** 2 for row in rows) / n for i in range(width)]
    std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, std


def fit(rows, lam: float, epochs: int = EPOCHS, rate: float = RATE) -> list[float]:
    """Full-batch gradient descent on the pairwise logistic loss.

    The pair rows are already differences, so the target label is always 1: the
    model should rank the target above the other. There is no intercept, because a
    constant cannot reorder a group.
    """
    n = len(rows)
    if not n:
        return []
    width = len(rows[0])
    weights = [0.0] * width
    for _ in range(epochs):
        gradient = [0.0] * width
        for row in rows:
            margin = sum(w * x for w, x in zip(weights, row))
            # d/dw of -log sigmoid(margin) is -(1 - sigmoid(margin)) * x
            slack = 1.0 / (1.0 + math.exp(margin)) if margin > -30 else 1.0
            for i, x in enumerate(row):
                gradient[i] -= slack * x
        for i in range(width):
            weights[i] -= rate * (gradient[i] / n + lam * weights[i])
    return weights


def accuracy(rows, weights) -> float:
    """Share of comparisons the model gets the right way round."""
    if not rows:
        return 0.0
    right = sum(1 for row in rows
                if sum(w * x for w, x in zip(weights, row)) > 0)
    return right / len(rows)


def folds(groups, count: int = FOLDS):
    """Grouped by session: every snapshot of one session lands in one fold.

    Splitting snapshots instead of sessions would put turn 1 and turn 2 of the same
    conversation on opposite sides of the split. They share a target, so the model
    would be validated on an answer it had already been shown.
    """
    by_session: dict[str, list[dict]] = {}
    for group in groups:
        by_session.setdefault(str(group["sample_id"]), []).append(group)
    sessions = sorted(by_session)
    for index in range(count):
        held = {s for position, s in enumerate(sessions) if position % count == index}
        yield ([g for s in sessions if s not in held for g in by_session[s]],
               [g for s in sessions if s in held for g in by_session[s]])


def standardised(rows, mean, std):
    return [[(x - 0.0) / s for x, s in zip(row, std)] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the legacy A0 pairwise reranker compatibility control"
    )
    parser.add_argument("--dev", default=DEV)
    parser.add_argument("--holdout", default=HOLDOUT)
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dev-split", default=SPLITS["dev"],
                        help="ordered source split used to build --dev")
    parser.add_argument("--holdout-split", default=SPLITS["holdout"],
                        help="ordered source split used to build --holdout")
    parser.add_argument("--blend", type=float, default=1.0,
                        help="how far the model may move things, relative to the "
                             "score the weighted sum already produced")
    # Deliberately NOT analysis/reranker.json, which is the path src/rerank.py
    # loads. Training is an experiment; enabling a model in the shipped agent is a
    # decision, and it should take an explicit `--output analysis/reranker.json`
    # rather than happening as a side effect of running the trainer. See the ship
    # bar in analysis/reranker_experiment.json and the protected serving-asset tests.
    parser.add_argument("--output", default="analysis/reranker_candidate.json")
    parser.add_argument("--report", default="analysis/reranker_training.json")
    args = parser.parse_args()

    dev = load(args.dev, catalog=args.catalog, split_path=args.dev_split)
    holdout = load(args.holdout, catalog=args.catalog,
                   split_path=args.holdout_split)
    dev_pairs = pairs(dev)
    # Differences have mean zero by nature of the objective; only the scale matters.
    _, std = standardise(dev_pairs)
    print(f"dev: {len(dev)} snapshots, {len(dev_pairs)} pairwise comparisons")

    chosen, best = LAMBDAS[0], -1.0
    curve = []
    for lam in LAMBDAS:
        scores = []
        for train_groups, held_groups in folds(dev):
            train_rows = standardised(pairs(train_groups), None, std)
            held_rows = standardised(pairs(held_groups), None, std)
            if not train_rows or not held_rows:
                continue
            scores.append(accuracy(held_rows, fit(train_rows, lam)))
        mean = sum(scores) / len(scores) if scores else 0.0
        curve.append({"lambda": lam, "cv_pairwise_accuracy": round(mean, 5)})
        print(f"  lambda={lam:<6} grouped-CV pairwise accuracy {mean:.4f}")
        if mean > best:
            chosen, best = lam, mean

    scaled = standardised(dev_pairs, None, std)
    weights = fit(scaled, chosen)
    # Undo the standardisation so the runtime never has to know about it.
    exported = [w / s for w, s in zip(weights, std)]

    dev_accuracy = accuracy(scaled, weights)
    holdout_pairs = standardised(pairs(holdout), None, std)
    holdout_accuracy = accuracy(holdout_pairs, weights)
    print(f"\nchosen lambda={chosen}  CV={best:.4f}")
    print(f"pairwise accuracy   dev {dev_accuracy:.4f}   holdout {holdout_accuracy:.4f}")
    print("\nweights (exported, un-standardised):")
    for name, weight in sorted(zip(FEATURES, exported), key=lambda p: -abs(p[1])):
        print(f"  {name:<22} {weight:+.4f}")

    payload = {
        "features": list(FEATURES),
        "weights": [round(w, 6) for w in exported],
        "bias": 0.0,
        "blend": args.blend,
        "meta": {
            "objective": "pairwise logistic on feature differences",
            "lambda": chosen,
            "trained_on": args.dev,
            "snapshots": len(dev),
            "pairs": len(dev_pairs),
            "cv_pairwise_accuracy": round(best, 5),
            "dev_pairwise_accuracy": round(dev_accuracy, 5),
            "holdout_pairwise_accuracy": round(holdout_accuracy, 5),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    Path(args.report).write_text(json.dumps(
        {"lambda_curve": curve, **payload["meta"]}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
