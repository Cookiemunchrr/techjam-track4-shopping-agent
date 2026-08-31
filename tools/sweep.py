"""Policy sweep. Builds the catalog index once and swaps policy objects between runs,
so a 40-configuration search takes seconds instead of ten minutes.

Always sweeps on dev and reports holdout for the winner -- never tune on all 200.
"""
from __future__ import annotations

import argparse
import itertools
import json

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent import Agent
from src.policy import CommitPolicy
from src.scoring import Scorer, Weights

CATALOG = "data/catalog.jsonl"


def score(agent, samples, index) -> dict:
    ids, categories, products = index
    result = evaluate(agent, samples, ids, categories, products)
    return {"score": result["recommended_technical_score"], "hr": result["hit_rate_at_10"],
            "mrr": result["mrr"], "mttc": result["mttc"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="analysis/sweep.json")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    index = catalog_index(CATALOG)
    agent = Agent(CATALOG)
    dev = load_jsonl("analysis/dev.jsonl")
    holdout = load_jsonl("analysis/holdout.jsonl")

    grid = list(itertools.product(
        (0.01, 0.02, 0.04, 0.08, 0.15, 0.30),   # margin_floor
        (2, 3, 4, 6),                            # unsure_width
        (7, 8, 9),                               # widen_turn
    ))
    rows = []
    for margin, unsure, widen in grid:
        agent.commit = CommitPolicy(base_width=1, widen_turn=widen,
                                    margin_floor=margin, unsure_width=unsure)
        row = {"margin": margin, "unsure": unsure, "widen": widen, **score(agent, dev, index)}
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])

    print("top configurations on DEV (100 sessions)")
    for row in rows[:args.top]:
        print("  margin=%-5s unsure=%d widen=%d  dev=%.5f  HR=%.3f MRR=%.3f MTTC=%.2f" % (
            row["margin"], row["unsure"], row["widen"], row["score"], row["hr"], row["mrr"], row["mttc"]))

    print("\nholdout check for the top 5 (never tuned on)")
    for row in rows[:5]:
        agent.commit = CommitPolicy(base_width=1, widen_turn=row["widen"],
                                    margin_floor=row["margin"], unsure_width=row["unsure"])
        held = score(agent, holdout, index)
        row["holdout"] = held["score"]
        print("  margin=%-5s unsure=%d widen=%d  dev=%.5f  holdout=%.5f  gap=%+.4f" % (
            row["margin"], row["unsure"], row["widen"], row["score"], held["score"],
            row["score"] - held["score"]))

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
