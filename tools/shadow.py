"""Artifact-sensitivity harness: the official loop with known simulator quirks removed.

This is a diagnostic, not a second leaderboard. `evaluator/local_evaluator.py` remains
the only source of the competition score and is never modified. What this measures is
narrower and more useful during development:

    how much of our score depends on behaviours that exist only in the simulator?

Three artifacts are removed, each one identified by reading the evaluator:

  wildcard    `customer_reply` matches a constraint when `attribute == "other" or
              classify_constraint(v) == attribute`, so "other" unlocks every
              undisclosed constraint at once. Here it matches only its own class,
              which is to say nothing -- `classify_constraint` never returns "other".

  dead zone   the official hit check is `if override_applied and target in ranked`,
              and the flag only flips at turn 3 or 4, so a correct recommendation
              made earlier in an override session cannot register. Here it counts
              whenever it is made.

  width       the official session ends when the target appears in the *shown*
              slate, so narrowing the slate suppresses low-rank exposure and
              inflates MRR. Here scoring reads `Agent.internal_ranking` -- the
              untruncated top ten, before the dialogue policy decides how much of
              it to show. Recommendation width therefore cannot move the rank,
              which separates "did we rank the item correctly" from "how much of
              that ranking did the user see".

Rank is *not* removed. The brief asks for the target as early and as highly ranked
as possible and weights MRR at 30%, so the shadow score uses the official weighting
(0.50 HR + 0.30 MRR + 0.20 efficiency) on the internal ranking. Dropping the rank
term would answer a different question than the one the challenge asks.

Two operational axes are included for the same reason:

  fresh       a new Agent per session, so cross-session learning cannot carry the
              result if the private harness constructs one agent per session
  shuffle     a permuted session order, so nothing depends on public-set ordering

    python3 -m tools.shadow                    # every axis
    python3 -m tools.shadow --axis clean       # artifacts removed, one shared agent
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import uuid
from pathlib import Path

from evaluator.local_evaluator import (ALLOWED_ATTRIBUTES, MAX_TURNS, TOP_K,
                                       catalog_index, classify_constraint,
                                       coarse_category, initial_message, load_jsonl,
                                       materialize_hidden_fields)

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"


def strict_reply(sample: dict, ask_attribute: object, disclosed: set[str],
                 boundary_used: bool) -> tuple[str, bool]:
    """`customer_reply` with the disclosure wildcard removed.

    The only change from the official version is that `attribute == "other"` no
    longer short-circuits the class test. Everything else is identical, so a
    difference between the two runs is attributable to the wildcard alone.
    """
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [value for value in constraints
               if value not in disclosed and classify_constraint(value) == attribute][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", boundary_used


def _ranking(agent, session_id: str, response: dict, catalog_ids: set[str]) -> list[str]:
    """The agent's internal top ten, falling back to what it chose to show.

    A wrapper (tools/paraphrase.py) may not forward the diagnostic hook; scoring
    the visible slate in that case is strictly harsher, never more generous.
    """
    getter = getattr(agent, "internal_ranking", None)
    ranked: list[str] = list(getter(session_id)) if callable(getter) else []
    if not ranked:
        for item in response.get("recommendations") or []:
            value = item.get("parent_asin", "") if isinstance(item, dict) else item
            if value:
                ranked.append(str(value))
    seen: set[str] = set()
    out: list[str] = []
    for pid in ranked:
        if pid and pid not in seen and pid in catalog_ids:
            seen.add(pid)
            out.append(pid)
        if len(out) >= TOP_K:
            break
    return out


def run(agent_factory, samples, catalog_ids, categories, products,
        fresh: bool = False) -> dict:
    """One pass over the sessions. `fresh` builds a new agent for each one."""
    agent = None if fresh else agent_factory()
    sessions: list[dict] = []
    for sample in samples:
        if fresh:
            agent = agent_factory()
        session_id = f"shadow_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective, coarse_category(categories.get(target, [])),
                                       disclosed)
        hit_turn = best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = _ranking(agent, session_id, response, catalog_ids)
            # No dead zone: a correct ranking counts whenever it is produced.
            if target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message",
                                                "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = strict_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used)
        sessions.append({
            "sample_id": sample["sample_id"], "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None, "first_hit_turn": hit_turn,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    return summarise(sessions)


def summarise(sessions: list[dict]) -> dict:
    hit_rate = sum(int(s["hit"]) for s in sessions) / len(sessions)
    mrr = statistics.fmean(s["reciprocal_rank"] for s in sessions)
    mttc = statistics.fmean(
        s["first_hit_turn"] if s["first_hit_turn"] is not None else MAX_TURNS + 1
        for s in sessions)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "sample_count": len(sessions),
        "retrieval_hit_rate_at_10": round(hit_rate, 6),
        "retrieval_mrr": round(mrr, 6),
        "retrieval_mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        # Official weighting, deliberately. The task is "early and highly ranked".
        "shadow_score": round(0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-sensitivity harness")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--axis", choices=("clean", "fresh", "shuffle", "all"), default="all")
    parser.add_argument("--output", default="analysis/shadow.json")
    args = parser.parse_args()

    from src.agent import Agent
    samples = load_jsonl(args.dataset)
    ids, categories, products = catalog_index(args.catalog)
    base = Agent(args.catalog)
    # The frozen index is read-only and shared; only the learned state is rebuilt.
    factory = lambda: Agent.sharing_index(base)

    rows: dict[str, dict] = {}
    if args.axis in ("clean", "all"):
        rows["clean"] = run(factory, samples, ids, categories, products)
    if args.axis in ("fresh", "all"):
        rows["fresh_agent"] = run(factory, samples, ids, categories, products, fresh=True)
    if args.axis in ("shuffle", "all"):
        for seed in (1, 2):
            order = list(samples)
            random.Random(seed).shuffle(order)
            rows[f"shuffle_seed{seed}"] = run(factory, order, ids, categories, products)

    for name, row in rows.items():
        print("  %-16s shadow=%.5f  HR=%.3f  MRR=%.4f  MTTC=%.2f" % (
            name, row["shadow_score"], row["retrieval_hit_rate_at_10"],
            row["retrieval_mrr"], row["retrieval_mttc"]))
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
