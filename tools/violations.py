"""How often the agent recommends something the shopper ruled out.

`facet_disagreement_material` and `facet_disagreement_color` exist for one
mechanism: a product that states a material the shopper did not ask for is
evidence *against*, where a product that states nothing is merely unknown, and the
weighted sum scores both as zero. Snapshot MRR on the clean public set cannot see
whether that mechanism works -- the simulator discloses constraints the target
satisfies, so a contradiction in the top ten is rare there by construction.

This measures the mechanism directly. Replay the official loop, and for every item
the agent actually shows, ask whether the product claims a value for a stated
attribute and none of the claimed values is one the shopper asked for. Strict facet
sets only, exactly as the feature reads them: prose that mentions leather is not a
leather claim, and it is not a denial of cotton either.

    python3 -m tools.violations                        # control, shipped model
    python3 -m tools.violations --axis constraint      # under paraphrase
    python3 -m tools.violations --axis constraint --model analysis/reranker_candidate.json

The rate is per shown item, over every turn of every session -- not per session --
because showing one contradicting product in a slate of ten is a tenth of a
failure, not a whole one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATALOG = "data/catalog.jsonl"
FULL = "data/public_set.jsonl"
ATTRIBUTES = ("material", "color")


class Counting:
    """Wraps an Agent and tallies contradictions in what it shows.

    Between the harness and the agent rather than inside it: the agent must not
    know it is being measured, and nothing here may change what it returns.
    """

    def __init__(self, agent, catalog) -> None:
        self._agent = agent
        self._catalog = catalog
        self.shown = 0
        self.violating = 0
        self.turns_with_facets = 0
        self.sessions_touched: set = set()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self._agent.respond(session_id, user_message, turn, top_k)
        state = self._agent.sessions.get(str(session_id))
        facets = state.dialog.facets() if state is not None else {}
        wanted = {a: v for a, v in facets.items() if a in ATTRIBUTES and v}
        if not wanted:
            return response
        self.turns_with_facets += 1
        for item in response.get("recommendations") or ():
            pid = item.get("parent_asin")
            if pid not in self._catalog.meta:
                continue
            self.shown += 1
            if any(_contradicts(self._catalog, pid, attribute, value)
                   for attribute, value in wanted.items()):
                self.violating += 1
                self.sessions_touched.add(str(session_id))
        return response


def _contradicts(catalog, pid: str, attribute: str, wanted) -> bool:
    """The product claims values for `attribute` and none of them was asked for."""
    stated = catalog.facet_values(pid, attribute)
    if not stated:
        return False              # says nothing: unknown, not contradicting
    options = wanted if isinstance(wanted, (tuple, list, set, frozenset)) else (wanted,)
    return not any(value in stated for value in options)


def run(axis: str, model_path: str, catalog_path: str = CATALOG, seed: int = 0) -> dict:
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from tools.adversarial import Adversarial
    from tools.rerank_data import base_agent
    from tools.snapshot_mrr import load_model

    ids, categories, products = catalog_index(catalog_path)
    agent = base_agent(catalog_path)
    agent.reranker = load_model(model_path)
    counter = Counting(agent, agent.catalog)
    wrapped = counter if axis == "control" else Adversarial(counter, (axis,), seed)
    result = evaluate(wrapped, load_jsonl(FULL), ids, categories, products)
    return {
        "axis": axis,
        "model": model_path,
        "blend": getattr(agent.reranker, "blend", None),
        "shown_under_a_stated_facet": counter.shown,
        "contradicting": counter.violating,
        "rate": round(counter.violating / counter.shown, 5) if counter.shown else 0.0,
        "turns_with_a_stated_facet": counter.turns_with_facets,
        "sessions_with_any": len(counter.sessions_touched),
        "technical_score": round(result["recommended_technical_score"], 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint contradictions in the slate")
    parser.add_argument("--axis", default="control",
                        choices=("control", "category", "natural", "scaffold", "constraint"))
    parser.add_argument("--model", default="analysis/reranker.json",
                        help="'none' for the weighted sum alone")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    row = run(args.axis, args.model, args.catalog, args.seed)
    print("  %-11s %-40s  %d/%d shown contradict a stated facet  rate %.4f  "
          "(score %.5f)" % (row["axis"], row["model"], row["contradicting"],
                            row["shown_under_a_stated_facet"], row["rate"],
                            row["technical_score"]))
    if args.output:
        Path(args.output).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
