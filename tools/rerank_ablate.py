"""Which of the appended features carry weight, one at a time.

The ceiling probe in analysis/reranker_experiment.json established that the
training set is not what limits this model: fitting on the evaluation sessions
themselves buys +0.0280 against the dev-fitted model's +0.0279. What binds is
what the feature vector can express. So the next lever is features -- and a batch
of six that lands as one commit teaches nothing about which of the six did it, or
whether one of them is quietly costing.

This trains the same pairwise model under each feature subset, on the same data,
with the same grouped cross-validation, and scores each one where it matters: the
snapshot MRR of the resulting ordering against the model that is already shipped.
Both directions are run, because they answer different questions --

  add-one    the 15 shipped features plus one appended feature. "Does this
             feature carry anything at all?"
  drop-one   all 21 minus one appended feature. "Does it carry anything the
             others do not?" A feature can pass the first and fail the second,
             and that is the one worth knowing about: it means the batch is
             paying for it twice.

The control is the 15 shipped features on their own, retrained here. It has to
reproduce the committed asset's weights, and if it does not, something in the
data or the trainer has drifted and no other row on the table means anything.

    python3 -m tools.rerank_ablate
    python3 -m tools.rerank_ablate --features constraint_coverage,shelf_lexical_rank
    python3 -m tools.rerank_ablate --emit analysis/candidates

`--emit` writes each config as a loadable asset. Snapshot MRR is the first gate and
not the only one -- a feature can be flat on ordering and still change what the
agent shows under paraphrase, which is what tools/violations.py measures -- and
those gates need a model that the agent can actually run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.features import FEATURES
from src.rerank import Model
from tools import rerank_train as trainer
from tools import snapshot_mrr as metric

# Everything appended after the fifteen the shipped asset was trained on.
SHIPPED = 15
APPENDED = tuple(FEATURES[SHIPPED:])
BASELINE_ASSET = "analysis/reranker.json"
BLEND = 0.05


def columns(names) -> list[int]:
    return [FEATURES.index(name) for name in names]


def restrict(rows, keep: list[int]) -> list[list[float]]:
    return [[row[i] for i in keep] for row in rows]


def train(dev_groups, keep: list[int]) -> tuple[list[float], dict]:
    """Fit on dev under one feature subset, returning full-width weights.

    The subset is trained on its own columns and then scattered back into a
    full-width vector with zeros elsewhere, so every config produces something
    `rerank.Model` can serve and `snapshot_mrr` can score without special cases.
    """
    dev_pairs = restrict(trainer.pairs(dev_groups), keep)
    _, std = trainer.standardise(dev_pairs)
    chosen, best = trainer.LAMBDAS[0], -1.0
    curve = []
    for lam in trainer.LAMBDAS:
        scores = []
        for train_groups, held_groups in trainer.folds(dev_groups):
            train_rows = trainer.standardised(restrict(trainer.pairs(train_groups), keep), None, std)
            held_rows = trainer.standardised(restrict(trainer.pairs(held_groups), keep), None, std)
            if not train_rows or not held_rows:
                continue
            scores.append(trainer.accuracy(held_rows, trainer.fit(train_rows, lam)))
        mean = sum(scores) / len(scores) if scores else 0.0
        curve.append({"lambda": lam, "cv_pairwise_accuracy": round(mean, 5)})
        if mean > best:
            chosen, best = lam, mean
    scaled = trainer.standardised(dev_pairs, None, std)
    fitted = trainer.fit(scaled, chosen)
    exported = [w / s for w, s in zip(fitted, std)]
    weights = [0.0] * len(FEATURES)
    for position, index in enumerate(keep):
        weights[index] = exported[position]
    return weights, {"lambda": chosen, "cv_pairwise_accuracy": round(best, 5),
                     "dev_pairwise_accuracy": round(trainer.accuracy(scaled, fitted), 5),
                     "lambda_curve": curve}


def evaluate(weights, snapshots, baseline, seed: int = 0) -> dict:
    model = Model(weights, 0.0, BLEND)
    out = {}
    for split, groups in snapshots.items():
        low, mid, high, _ = metric.paired(groups, baseline, model, seed)
        out[split] = {"snapshot_mrr": round(metric.mrr(groups, model), 5),
                      "delta_vs_shipped": round(mid, 5),
                      "ci95": [round(low, 5), round(high, 5)],
                      "verdict": "significant" if low > 0 or high < 0 else "within noise"}
    return out


def configs(names) -> list[tuple[str, tuple]]:
    """The subsets to try, in the order a reader wants to see them."""
    plans = [("shipped 15 (control)", tuple(FEATURES[:SHIPPED])),
             ("all appended", tuple(FEATURES[:SHIPPED]) + tuple(names))]
    for name in names:
        plans.append((f"+ {name}", tuple(FEATURES[:SHIPPED]) + (name,)))
    if len(names) > 1:
        for name in names:
            plans.append((f"all appended - {name}",
                          tuple(FEATURES[:SHIPPED]) + tuple(n for n in names if n != name)))
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-feature ablation for the reranker")
    parser.add_argument("--features", default=",".join(APPENDED),
                        help="the appended features to ablate over")
    parser.add_argument("--baseline", default=BASELINE_ASSET,
                        help="the ordering every config is compared against")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emit", default="",
                        help="directory to write each config as a loadable asset")
    parser.add_argument("--output", default="analysis/reranker_features.json")
    args = parser.parse_args()

    names = [n.strip() for n in args.features.split(",") if n.strip()]
    unknown = [n for n in names if n not in FEATURES]
    if unknown:
        raise SystemExit(f"not in features.FEATURES: {', '.join(unknown)}")

    dev = trainer.load(trainer.DEV)
    snapshots = {split: metric.cached(split) for split in ("dev", "holdout")}
    baseline = metric.load_model(args.baseline)
    print(f"dev: {len(dev)} snapshots, {len(trainer.pairs(dev))} pairwise comparisons")
    print(f"baseline: {args.baseline} at blend {getattr(baseline, 'blend', None)}\n")

    emit = Path(args.emit) if args.emit else None
    if emit:
        emit.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, subset in configs(names):
        weights, meta = train(dev, columns(subset))
        if emit:
            slug = label.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
            (emit / f"{slug}.json").write_text(json.dumps({
                "features": list(FEATURES), "weights": [round(w, 6) for w in weights],
                "bias": 0.0, "blend": BLEND,
                "meta": {"config": label, "trained_on": trainer.DEV,
                         "fitted_features": list(subset), **meta},
            }, indent=2) + "\n", encoding="utf-8")
        row = {"config": label, "features": list(subset), **meta,
               **evaluate(weights, snapshots, baseline, args.seed),
               "weights": {name: round(weights[FEATURES.index(name)], 6) for name in subset}}
        rows.append(row)
        print("  %-34s CV %.4f   dev %+.4f [%+.4f, %+.4f]   holdout %+.4f [%+.4f, %+.4f]"
              % (label, row["cv_pairwise_accuracy"],
                 row["dev"]["delta_vs_shipped"], *row["dev"]["ci95"],
                 row["holdout"]["delta_vs_shipped"], *row["holdout"]["ci95"]))

    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
