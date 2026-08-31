"""Where sessions are lost: retrieval, or ordering?

The score is one number and it hides which of two very different failures produced
it. A target that never entered the candidate pool cannot be recovered by any
reranker, weight or extra turn -- that is a retrieval failure, and the fix is
recall. A target sitting in the pool at rank eleven is a ranking failure, and the
fix is the scoring function. Reporting only HitRate@10 conflates them, and a
roadmap built on the conflated number spends its effort in the wrong place.

So this replays the official session loop and records, at every turn:

    in_pool     was the target among the candidates the scorer ranked over
    rank        where the untruncated internal ranking put it
    pool        how many candidates there were

and reports, per split and per adversarial axis:

    pool recall      the ceiling -- no ordering change can beat this
    rank | in pool   how much of that ceiling the ranking actually converts
    lost to recall   sessions where the target was never in the pool
    lost to ranking  sessions where it was in the pool and never in the top ten

What it found when it was first run is worth recording, because it redirected the
plan it was written to support: on the official sessions pool recall is 1.000 at
turn one. There is no retrieval headroom on the public harness at all, and every
point of the shadow MRR gap is an ordering problem. Under category-synonym
paraphrase recall does break -- 0.885 -- but even there it accounts for under half
of the axis loss, and the scaffold axis loses 0.078 with recall still at 1.000.

    python3 -m tools.recall                     # official sessions, per split
    python3 -m tools.recall --axis category     # under an adversarial rewrite
    python3 -m tools.recall --v6-baseline       # the frozen V6 Phase 1 baseline:
                                                # dev + category axis, full pinning,
                                                # aggregate miss taxonomy, STOP-3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import uuid
from pathlib import Path

from evaluator.local_evaluator import (MAX_TURNS, TOP_K, catalog_index,
                                       coarse_category, customer_reply, evaluate,
                                       initial_message, load_jsonl,
                                       materialize_hidden_fields)

CATALOG = "data/catalog.jsonl"
SPLITS = {"dev": "analysis/dev.jsonl",
          "holdout": "analysis/holdout.jsonl",
          "full": "data/public_set.jsonl"}


def _rewriter(axes, seed: int):
    if not axes:
        return None
    from tools.adversarial import Rewriter
    return Rewriter(axes, seed)


def trace(agent, samples, catalog_ids, categories, products, axes=()) -> list[dict]:
    """One record per session: where the target was, turn by turn."""
    agent.trace_pool = True
    agent.trace_route = True
    out: list[dict] = []
    for index, sample in enumerate(samples):
        rewriter = _rewriter(axes, index + 1)
        session_id = f"recall_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])),
                                  disclosed)

        turns: list[dict] = []
        turn1_route: dict = {}
        for turn in range(1, MAX_TURNS + 1):
            asked = rewriter.rewrite(message) if rewriter else message
            try:
                response = agent.respond(session_id, asked, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if turn == 1:
                route = agent.route_trace(session_id)
                turn1_route = {k: route.get(k)
                               for k in ("exact", "hedged", "fallback",
                                         "prefilter_pool_sha256",
                                         "prefilter_pool_size",
                                         "baseline_prefix_sha256",
                                         "baseline_prefix_size",
                                         "transform_lookups",
                                         "transform_activations")}
            pool = agent.candidate_pool(session_id)
            ranking = agent.internal_ranking(session_id)
            rank = ranking.index(target) + 1 if target in ranking else None
            turns.append({"turn": turn, "in_pool": target in pool,
                          "pool": len(pool), "rank": rank})
            if rank is not None and rank <= TOP_K:
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                if str(override.get("new_value", "")):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message",
                                           "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used)

        ever_pool = any(t["in_pool"] for t in turns)
        best = [t["rank"] for t in turns if t["rank"] is not None]
        state = agent.sessions.get(session_id)
        out.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "turn1_in_pool": turns[0]["in_pool"] if turns else False,
            "ever_in_pool": ever_pool,
            "first_pool_turn": next((t["turn"] for t in turns if t["in_pool"]), None),
            "best_rank": min(best) if best else None,
            "turn1_pool": turns[0]["pool"] if turns else 0,
            "turn1_route": turn1_route,
            # Diagnostic-only fields: they carry the label-derived true phrase and
            # the observed category so the miss taxonomy can classify a failure.
            # They may live in the ignored sidecar, never in a committed artifact.
            "true_category": coarse_category(categories.get(target, [])),
            "observed_category": (state.dialog.category if state is not None else None),
            "turns": turns,
        })
    agent.trace_pool = False
    agent.trace_route = False
    return out


def summarise(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    hit = [r for r in rows if r["best_rank"] is not None and r["best_rank"] <= TOP_K]
    in_pool = [r for r in rows if r["ever_in_pool"]]
    pools = sorted(r["turn1_pool"] for r in rows)
    first_turns = [r["first_pool_turn"] if r.get("first_pool_turn") is not None else 11
                   for r in rows]
    conditional_mrr = (statistics.mean(1.0 / r["best_rank"] for r in in_pool
                                       if r["best_rank"]) if in_pool else 0.0)
    return {
        "sessions": len(rows),
        "pool_recall_turn1": round(sum(r["turn1_in_pool"] for r in rows) / n, 4),
        "pool_recall_ever": round(len(in_pool) / n, 4),
        # First turn the target entered the pool; a miss counts as 11, matching
        # the MTTC convention.
        "first_pool_turn_mean": round(statistics.mean(first_turns), 4),
        "hit_rate_at_10": round(len(hit) / n, 4),
        # The share of the retrieval ceiling the ranking actually converts. This
        # is a conditional HIT RATE: the historical label `rank_given_pool`
        # named a rank it never measured, and V6 P3 retires it.
        "hr_at_10_given_pool": round(len(hit) / max(len(in_pool), 1), 4),
        # The actual conditional ordering quality among pooled sessions.
        "conditional_mrr_given_pool": round(conditional_mrr, 4),
        "lost_to_recall": round(sum(1 for r in rows if not r["ever_in_pool"]) / n, 4),
        "lost_to_ranking": round(
            sum(1 for r in rows
                if r["ever_in_pool"] and (r["best_rank"] is None or r["best_rank"] > TOP_K)) / n, 4),
        "median_turn1_pool": int(statistics.median([r["turn1_pool"] for r in rows] or [0])),
        # Nearest-rank percentile, matching the resource probe's convention.
        "turn1_pool_p95": pools[int(0.95 * (len(pools) - 1))] if pools else 0,
        "turn1_fallback_rate": round(
            sum(1 for r in rows if (r.get("turn1_route") or {}).get("fallback")) / n, 4),
    }


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def miss_taxonomy(rows: list[dict]) -> dict:
    """Aggregate classification of the retrieval misses (V6 P4, addendum B3/B4).

    Labels are read here and only here: the returned report carries counts,
    never sample ids, phrases, or target identities.
    """
    from src.semantic import stem
    from src.text import tokens

    def component(true_phrase: str, observed_phrase: str) -> str:
        true_toks = [stem(t) for t in tokens(true_phrase or "")]
        obs_toks = [stem(t) for t in tokens(observed_phrase or "")]
        if not true_toks or not obs_toks:
            return "both"
        head_ok = true_toks[-1] == obs_toks[-1]
        mods_ok = set(true_toks[:-1]) == set(obs_toks[:-1])
        if head_ok and mods_ok:
            return "neither"
        return {  # head_ok, mods_ok
            (False, True): "head_only",
            (True, False): "modifier_only",
            (False, False): "both",
        }[(head_ok, mods_ok)]

    buckets = ("wrong_or_missing_shelf_resolution", "no_usable_category_language",
               "later_state_failure", "other")
    counts = {bucket: 0 for bucket in buckets}
    by_scenario = {bucket: {} for bucket in buckets}
    by_component = {bucket: {} for bucket in buckets}
    misses = [r for r in rows if not r["ever_in_pool"]]
    for row in misses:
        observed = (row.get("observed_category") or "").strip()
        if not observed:
            bucket = "no_usable_category_language"
        elif row["turn1_in_pool"]:
            # Pooled at turn 1 yet never again; kept for shape, expected zero.
            bucket = "later_state_failure"
        else:
            bucket = "wrong_or_missing_shelf_resolution"
        comp = component(row.get("true_category") or "", observed)
        counts[bucket] += 1
        scenario = row.get("scenario_type") or "unknown"
        by_scenario[bucket][scenario] = by_scenario[bucket].get(scenario, 0) + 1
        by_component[bucket][comp] = by_component[bucket].get(comp, 0) + 1

    # STOP-3, amended by B3: stop if fewer than three misses are wrong/missing
    # shelf resolution, OR fewer than three of those attach to the head/final
    # token (alone or as part of "both").
    shelf = counts["wrong_or_missing_shelf_resolution"]
    head_components = by_component["wrong_or_missing_shelf_resolution"]
    qualifying = head_components.get("head_only", 0) + head_components.get("both", 0)
    base = len(misses)
    return {
        "base_misses": base,
        "counts": counts,
        "by_scenario": {k: v for k, v in by_scenario.items() if v},
        "by_component": {k: v for k, v in by_component.items() if v},
        "stop3": {
            "shelf_resolution_misses": shelf,
            "head_attributable": qualifying,
            "threshold": 3,
            "proportion_wilson95": list(wilson(qualifying, base)),
            "decision": "stop" if min(shelf, qualifying) < 3 else "proceed",
            "limitation": ("the base is six misses; the check cannot distinguish a "
                           "mechanism that addresses half the loss from one that "
                           "addresses a quarter (addendum B4)"),
        },
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def v6_baseline(seed: int = 0) -> tuple[dict, dict]:
    """The frozen V6 Phase 1 baseline: dev split, category axis, both widths.

    Returns (baseline_report, miss_taxonomy_report). Per-session rows -- which
    carry sample ids and label-derived phrases -- go only to the ignored
    sidecar; the committed reports are aggregate.
    """
    from unittest import mock

    from src.agent import Agent
    from tools.adversarial import Adversarial

    ids, categories, products = catalog_index(CATALOG)
    samples = load_jsonl(SPLITS["dev"])

    base = Agent(CATALOG)
    rows = trace(Agent.sharing_index(base), samples, ids, categories, products,
                 ("category",))
    summary = summarise(rows)
    taxonomy = miss_taxonomy(rows)

    def category_score(fixed_width: bool) -> dict:
        if fixed_width:
            with mock.patch.dict(os.environ, {"P_PROBE": "10", "P_WIDEN": "1"}):
                agent = Agent(CATALOG)
        else:
            agent = Agent(CATALOG)
        wrapped = Adversarial(Agent.sharing_index(agent), ("category",), seed)
        result = evaluate(wrapped, samples, ids, categories, products)
        return {"technical_score": round(result["recommended_technical_score"], 5),
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": round(result["mrr"], 5), "mttc": result["mttc"]}

    sidecar = Path("analysis/_v6_baseline_rows.json")
    sidecar.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    hashes = {"catalog_sha256": _sha256(CATALOG),
              "dev_split_sha256": _sha256(SPLITS["dev"]),
              "evaluator_sha256": _sha256("evaluator/local_evaluator.py"),
              "recall_sha256": _sha256("tools/recall.py"),
              "agent_sha256": _sha256("src/agent.py")}
    baseline = {
        "schema": "techjam-v6-current-baseline-v1",
        "scope": "V6 Phase 1 frozen baseline; development sessions 1-100; "
                 "category paraphrase axis",
        "command": "python3 -m tools.recall --v6-baseline",
        "seed": seed,
        "split": "dev",
        "axis": "category",
        "hashes": hashes,
        "recall": summary,
        "category_technical_score": {"normal_width": category_score(False),
                                     "fixed_width_10": category_score(True)},
        "miss_taxonomy": taxonomy,
        "limitations": [
            "Sessions 101-200 are a previously exposed confirmation split, not a "
            "pristine holdout; nothing here reads them.",
            "Per-session rows are analysis/_v6_baseline_rows.json, an ignored "
            "sidecar; the committed report is aggregate only.",
        ],
    }
    return baseline, taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage retrieval recall")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--splits", default="dev,holdout,full")
    parser.add_argument("--axis", default="",
                        help="comma-separated adversarial axes, e.g. category,scaffold")
    parser.add_argument("--output", default="analysis/recall.json")
    parser.add_argument("--v6-baseline", action="store_true",
                        help="freeze the V6 Phase 1 dev/category baseline and miss "
                             "taxonomy into analysis/v6_current_baseline.json and "
                             "analysis/v6_miss_taxonomy.json")
    args = parser.parse_args()

    if args.v6_baseline:
        baseline, taxonomy = v6_baseline()
        Path("analysis/v6_current_baseline.json").write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        Path("analysis/v6_miss_taxonomy.json").write_text(
            json.dumps(taxonomy, indent=2) + "\n", encoding="utf-8")
        row = baseline["recall"]
        print("  dev/category pool@t1=%.3f  pool_ever=%.3f  first_pool_turn=%.2f  "
              "HR@10=%.3f  hr|pool=%.3f  cond_mrr|pool=%.3f  fallback=%.3f"
              % (row["pool_recall_turn1"], row["pool_recall_ever"],
                 row["first_pool_turn_mean"], row["hit_rate_at_10"],
                 row["hr_at_10_given_pool"], row["conditional_mrr_given_pool"],
                 row["turn1_fallback_rate"]))
        print("  scores  normal=%.5f  fixed_width=%.5f"
              % (baseline["category_technical_score"]["normal_width"]["technical_score"],
                 baseline["category_technical_score"]["fixed_width_10"]["technical_score"]))
        stop = taxonomy["stop3"]
        print("  STOP-3  shelf=%d  head-attributable=%d  base=%d  -> %s"
              % (stop["shelf_resolution_misses"], stop["head_attributable"],
                 taxonomy["base_misses"], stop["decision"]))
        print("\nwrote analysis/v6_current_baseline.json")
        print("wrote analysis/v6_miss_taxonomy.json")
        return

    from src.agent import Agent
    ids, categories, products = catalog_index(args.catalog)
    axes = tuple(a for a in args.axis.split(",") if a)
    base = Agent(args.catalog)

    report: dict[str, dict] = {}
    for split in args.splits.split(","):
        path = SPLITS.get(split)
        if not path or not Path(path).exists():
            continue
        rows = trace(Agent.sharing_index(base), load_jsonl(path), ids, categories,
                     products, axes)
        report[split] = summarise(rows)
        row = report[split]
        print("  %-8s pool@t1=%.3f  pool_ever=%.3f  HR@10=%.3f  hr|pool=%.3f  "
              "lost:recall=%.3f ranking=%.3f  median_pool=%d"
              % (split, row["pool_recall_turn1"], row["pool_recall_ever"],
                 row["hit_rate_at_10"], row["hr_at_10_given_pool"], row["lost_to_recall"],
                 row["lost_to_ranking"], row["median_turn1_pool"]))

    Path(args.output).write_text(json.dumps(
        {"axes": list(axes), "splits": report}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
