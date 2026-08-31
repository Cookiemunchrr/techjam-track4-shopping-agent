"""What a question is expected to be worth, against what a turn costs.

The elicitation criterion ranks attributes by how much of the pool an answer
removes (Gini impurity), discounted by learned answerability. The standing
objection, written down in the V1 specification as P8 and never tested until now,
is that pool reduction is the wrong quantity: a question that splits the bottom of
the candidate set scores well on it and cannot move the head at all, and what the
score pays for is where the *target* lands.

`src/elicitation.expected_gain` computes the right quantity -- simulate the likely
answers, apply the facet term the scorer would apply, and read the change in
expected reciprocal rank -- and this measures it over real sessions, next to the
cost of the turn it would spend. TURN_COST_IN_MRR is read off the published scoring
formula: a turn is 0.02 of TechnicalScore and a unit of MRR is 0.30, so a question
has to buy 0.0667 of reciprocal rank to pay for itself.

    python3 analysis/question_value.py
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

OUTPUT = "analysis/question_value.json"
SPLITS = {"dev": "analysis/dev.jsonl", "holdout": "analysis/holdout.jsonl"}


def collect(split: str) -> dict:
    from src.agent import Agent
    from src.elicitation import (MEASURABLE, TURN_COST_IN_MRR, expected_gain,
                                 expected_reduction, POOL)
    from tools.rerank_data import base_agent
    from tools.snapshot_mrr import load_model

    seen: dict = collections.defaultdict(list)
    original = Agent._ask

    def probe(self, ranked, state, pool=(), hedge=()):
        head = [pid for _, pid in ranked[:POOL]]
        for attribute in MEASURABLE:
            reduction = expected_reduction(self.catalog, head, attribute)
            if reduction is None:
                continue
            seen[attribute].append({
                "gain": expected_gain(self.catalog, self.scorer, ranked, attribute),
                "reduction": reduction,
                "answerability": self.answers.probability(attribute)})
        return original(self, ranked, state, pool, hedge)

    Agent._ask = probe
    try:
        ids, categories, products = catalog_index("data/catalog.jsonl")
        agent = base_agent("data/catalog.jsonl")
        agent.reranker = load_model("analysis/reranker.json")
        evaluate(agent, load_jsonl(SPLITS[split]), ids, categories, products)
    finally:
        Agent._ask = original

    rows = []
    for attribute, values in sorted(seen.items()):
        gains = sorted(row["gain"] for row in values)
        rows.append({
            "attribute": attribute,
            "observations": len(values),
            "gain_median": round(statistics.median(gains), 5),
            "gain_p90": round(gains[int(0.9 * len(gains))], 5),
            "gain_max": round(max(gains), 5),
            "reduction_median": round(
                statistics.median(row["reduction"] for row in values), 4),
            "answerability_at_the_end": round(values[-1]["answerability"], 3),
            "share_paying_for_the_turn": round(
                sum(1 for row in values
                    if row["gain"] * row["answerability"] > TURN_COST_IN_MRR)
                / len(values), 4),
        })
    return {"split": split, "turn_cost_in_mrr": round(TURN_COST_IN_MRR, 4),
            "attributes": rows}


def main() -> None:
    report = [collect(split) for split in ("dev", "holdout")]
    for block in report:
        print(f"{block['split']}: a turn costs {block['turn_cost_in_mrr']} of "
              f"reciprocal rank")
        for row in block["attributes"]:
            print("  %-8s n=%-4d gain median %.4f p90 %.4f max %.4f | "
                  "reduction median %.3f | P(answer) %.2f | pays for the turn %.1f%%"
                  % (row["attribute"], row["observations"], row["gain_median"],
                     row["gain_p90"], row["gain_max"], row["reduction_median"],
                     row["answerability_at_the_end"],
                     100 * row["share_paying_for_the_turn"]))
    Path(OUTPUT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
