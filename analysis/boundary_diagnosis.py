"""Why the boundary slice ranks worst, turn by turn.

Boundary is the weakest scenario in the evaluator output and nobody had looked at
it: 10 sessions, MRR 0.671 against 0.749 buying and 0.776 browsing. n=10 cannot
carry a ship decision, so this is diagnosis. It asks one question of every turn --
where was the target, and which layer lost it --

  retrieval   the target was not in the candidate pool at all. Nothing downstream
              can recover it, and the fix would be routing.
  ordering    it was in the pool and outside the top ten of the ranking.
  width       it was inside the top ten of the ranking and the commit policy chose
              to show fewer than that. The ranking was right and the agent held it
              back, which is a dialogue decision, not a retrieval one.

The third is the one worth separating out, because it is the only one where the
agent already knew the answer.

All two hundred sessions are replayed, in order, and only the boundary ones are
recorded. That is not thoroughness, it is the only way to get the right answer:
`src/answerability.py` learns which questions this customer will answer *across*
sessions on one Agent instance, so replaying the ten boundary sessions on their own
measures a cold-start agent that the official run never has. The first draft of
this script did exactly that and produced a much worse-looking result -- six
consecutive refused questions -- which was an artifact of the isolation, not a
defect of the agent.

    python3 analysis/boundary_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (MAX_TURNS, TOP_K, catalog_index,
                                       coarse_category, customer_reply,
                                       initial_message, load_jsonl,
                                       materialize_hidden_fields,
                                       normalize_recommendations)

CATALOG = "data/catalog.jsonl"
FULL = "data/public_set.jsonl"
OUTPUT = "analysis/boundary_diagnosis.json"


def replay(agent, sample, catalog_ids, categories, products) -> dict:
    session_id = f"diag_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])),
                              disclosed)

    turns, hit_turn, best_rank = [], None, None
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, message, turn, TOP_K)
        shown = normalize_recommendations(response.get("recommendations"), catalog_ids)
        internal = agent.internal_ranking(session_id)
        pool = agent.candidate_pool(session_id)
        state = agent.sessions.get(session_id)
        record = {
            "turn": turn,
            "said_to_agent": message,
            "asked": response.get("ask_attribute"),
            "width": len(shown),
            "in_pool": target in pool,
            "internal_rank": (internal.index(target) + 1) if target in internal else None,
            "shown_rank": (shown.index(target) + 1) if target in shown else None,
            "slots": len([s for s in state.dialog.slots if not s.superseded]) if state else 0,
        }
        record["lost_to"] = classify(record)
        turns.append(record)
        if target in shown:
            hit_turn, best_rank = turn, shown.index(target) + 1
            break
        if turn == MAX_TURNS:
            break
        message, boundary_used = customer_reply(
            effective, response.get("ask_attribute"), disclosed, boundary_used)
    return {"sample_id": sample["sample_id"], "target": target,
            "first_hit_turn": hit_turn, "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else round(1.0 / best_rank, 4),
            "turns": turns}


def classify(record: dict) -> str:
    if record["shown_rank"] is not None:
        return "hit"
    if not record["in_pool"]:
        return "retrieval"
    rank = record["internal_rank"]
    if rank is None or rank > TOP_K:
        return "ordering"
    return "width"


def pinned_width(sessions) -> dict:
    """What this slice would have scored had the agent always shown ten.

    The commit policy narrows the slate while a question is still outstanding, on
    the argument that a later turn can rank better than this one and rank is worth
    about twelve turns of delay. Boundary is the slice where that argument is under
    the most strain: the opening states nothing, so the first ranking is the
    popularity prior and there is the most to gain from waiting -- or the most to
    lose by hiding an answer the agent already had.

    Read straight off the recorded turns: the first turn whose internal rank is
    within ten is the turn a ten-wide slate would have hit on, at that rank.
    """
    total_rr = total_turn = 0.0
    for session in sessions:
        for record in session["turns"]:
            rank = record["internal_rank"]
            if rank is not None and rank <= TOP_K:
                total_rr += 1.0 / rank
                total_turn += record["turn"]
                break
        else:
            total_turn += MAX_TURNS + 1
    count = len(sessions)
    return {"mrr": round(total_rr / count, 4), "mttc": round(total_turn / count, 3)}


def main() -> None:
    from tools.rerank_data import base_agent
    from tools.snapshot_mrr import load_model

    ids, categories, products = catalog_index(CATALOG)
    agent = base_agent(CATALOG)
    agent.reranker = load_model("analysis/reranker.json")
    agent.trace_pool = True

    sessions = []
    for sample in load_jsonl(FULL):
        record = replay(agent, sample, ids, categories, products)
        if sample["scenario_type"] == "boundary":
            sessions.append(record)

    tally: dict = {}
    for session in sessions:
        for record in session["turns"]:
            tally[record["lost_to"]] = tally.get(record["lost_to"], 0) + 1
    mrr = sum(s["reciprocal_rank"] for s in sessions) / len(sessions)
    pinned = pinned_width(sessions)
    first_question_declined = sum(
        1 for s in sessions
        if len(s["turns"]) > 1 and s["turns"][0]["asked"]
        and "don't have a preference" in s["turns"][1]["said_to_agent"])

    summary = {
        "sessions": len(sessions),
        "mrr": round(mrr, 4),
        "mttc": round(sum(len(s["turns"]) for s in sessions) / len(sessions), 3),
        "turn_outcomes": tally,
        "sessions_whose_first_question_was_declined": first_question_declined,
        "if_width_had_been_pinned_at_ten": pinned,
        "mean_width_before_the_hit": round(
            sum(t["width"] for s in sessions for t in s["turns"][:-1])
            / max(sum(len(s["turns"]) - 1 for s in sessions), 1), 3),
        "answerability_when_finished": agent.answers.snapshot(),
    }
    print(json.dumps(summary, indent=2))
    for session in sessions:
        print("  %-14s rr=%.3f  %s" % (
            session["sample_id"], session["reciprocal_rank"],
            " -> ".join("t%d:%s%s" % (t["turn"], t["lost_to"],
                                      "" if t["internal_rank"] is None
                                      else "@%d" % t["internal_rank"])
                        for t in session["turns"])))
    Path(OUTPUT).write_text(json.dumps(
        {"summary": summary, "sessions": sessions}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
