"""Build the reranker's training set by replaying recorded sessions.

One row per (turn snapshot, candidate) pair, with the label "is this the target".
The features come from `src/features.py` -- the same function the live path calls --
and the context comes from `Agent.last_context`, which is the context the live path
actually built. Nothing here recomputes a feature its own way, because a training
set that disagrees with the server by one normalisation produces a model that looks
fine in the report and ranks badly in production.

## Snapshots

A session is not one training example. The state after "Women Dresses" and the
state after "Women Dresses; cotton; black" are different problems, and the second
is the one the ranking has to get right. Every turn of every replayed session
becomes a snapshot, so the set naturally covers category-only, one constraint, two,
the turns either side of an intent override, and the boundary "no preference" reply.

## Negatives

Random negatives teach nothing: almost any product drawn from fifty thousand is
trivially separable from the target, and a model trained on those learns the
popularity prior and stops. So the negatives are the candidates the *existing
ranking already put at the top* -- the products that actually beat or nearly beat
the target under the weighted sum. Those are the comparisons the model has to win
to be worth shipping, and they arrive pre-mined by construction: they are the head
of the pool the agent itself produced.

## Splitting

Grouped by session, and the groups follow the repository's existing dev/holdout
split (sessions 1-100 / 101-200) rather than a fresh random one, so a model
validated here is validated on sessions no weight in this repo was fitted on.
Sessions whose target appears in both splits would leak; the organizer's public and
private sets already use separate users and targets, and within the public set the
two files are disjoint by construction.

    python3 -m tools.rerank_data --split dev     --output analysis/rerank_dev.jsonl
    python3 -m tools.rerank_data --split holdout --output analysis/rerank_holdout.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from evaluator.local_evaluator import (MAX_TURNS, TOP_K, catalog_index,
                                       classify_constraint,
                                       coarse_category, customer_reply,
                                       initial_message, load_jsonl,
                                       materialize_hidden_fields)
from tools.rerank_provenance import (CacheError, build_contract, load_cache,
                                     write_cache)

CATALOG = "data/catalog.jsonl"
SPLITS = {"dev": "analysis/dev.jsonl",
          "holdout": "analysis/holdout.jsonl",
          "full": "data/public_set.jsonl"}

# How many candidates per snapshot to record. The target plus the head of the
# existing ranking: deep enough that the model sees the comparisons it has to win,
# shallow enough that the file stays a few tens of megabytes.
NEGATIVES = 40
V4_LAYOUT = "raw_live_head_plus_six_decimal_compatibility"


def training_contract(catalog_path: str | Path, split_path: str | Path,
                      negatives: int = NEGATIVES) -> dict:
    """Provenance for raw serving rows plus the legacy training projection."""
    from src.features import FEATURES
    from src.rerank import DEPTH

    return build_contract(
        catalog_path=catalog_path,
        split_path=split_path,
        features=FEATURES,
        depth=DEPTH,
        generator_options={
            "compatibility_depth": negatives,
            "kind": "training",
            "layout": V4_LAYOUT,
            "live_numeric_precision": "raw_serving_float",
            "max_turns": MAX_TURNS,
            "projection": "dedupe(full_ranked_ids[:compatibility_depth] + [target])",
            "reference_ranker": "weighted_sum_reranker_disabled",
            "stop_policy": "first_visible_target",
            "top_k": TOP_K,
            "transcript": "visible_user_message_per_snapshot",
            "with_scores": False,
        },
    )


def load_training_cache(path: str | Path, catalog_path: str | Path,
                        split_path: str | Path,
                        negatives: int = NEGATIVES) -> list[dict]:
    """V4-aware loader; legacy JSONL without provenance is deliberately stale."""
    return load_cache(path, training_contract(catalog_path, split_path, negatives))


def base_agent(catalog_path: str):
    """An Agent with no reranker -- constructible even in the middle of a retrain.

    Both this tool and tools/snapshot_mrr.py want the ordering the weighted sum
    produces, never a half-applied model: training against a model's own output is
    a feedback loop, and measuring one against itself is not a measurement.

    Setting `agent.reranker = None` after construction is not enough. Appending a
    feature to `src.features.FEATURES` invalidates the committed asset by design,
    and `rerank.Model.from_dict` refuses it loudly -- which happens inside
    `Agent.__init__`, before anything here can turn the model off. That refusal is
    correct for the serving path and it must stay loud there; here it would mean
    the only way to rebuild the training set is to move the asset out of the way
    by hand and remember to put it back. So the asset path is pointed at nothing
    for the duration of the construction, which `rerank.load` already treats as a
    supported state.
    """
    from src import rerank
    from src.agent import Agent
    saved, rerank.ASSET = rerank.ASSET, "analysis/__retraining_in_progress__.json"
    try:
        agent = Agent(catalog_path)
    finally:
        rerank.ASSET = saved
    agent.reranker = None
    return agent


def snapshot_record(*, sample_id: str, scenario_type: str, turn: int,
                    target: str, message: str, features, ranked, vector_for,
                    catalog_ids, candidate_pool, compatibility_depth: int,
                    compatibility_scores: bool, diagnostics=None) -> dict:
    """Build one V4 replay record without inventing an actionable target row.

    ``live_rows`` is the exact bounded serving head: raw feature values, raw base
    scores, and true live ranks. ``rows`` is the deliberately flawed A0-compatible
    view, rounded to six decimals and projected as
    ``dedupe(full_ranked_ids[:compatibility_depth] + [target])``.  Keeping the two
    views explicit lets V4 reproduce the old tool without ever confusing an
    appended, unreachable target with something the deployed reranker could move.
    """
    if isinstance(compatibility_depth, bool) \
            or not isinstance(compatibility_depth, int) \
            or compatibility_depth <= 0:
        raise ValueError("compatibility_depth must be a positive integer")
    identifiers = set(catalog_ids)
    if target not in identifiers:
        raise ValueError(f"target {target!r} is absent from catalog_ids")

    ordered = [(float(score), str(pid)) for score, pid in ranked
               if str(pid) in identifiers]
    ranked_ids = [pid for _, pid in ordered]
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked head contains duplicate product ids")
    base = {pid: score for score, pid in ordered}
    target_rank = (ranked_ids.index(target) + 1) if target in base else None
    in_pool = target in set(candidate_pool)
    if target_rank is not None and not in_pool:
        raise ValueError("live target is absent from candidate_pool")
    reachability = ("rerankable" if target_rank is not None else
                    "rerank_depth_miss" if in_pool else "route_pool_miss")

    raw_vectors: dict[str, list[float]] = {}

    def raw_vector(pid: str) -> list[float]:
        if pid not in raw_vectors:
            raw_vectors[pid] = [float(value) for value in vector_for(pid)]
        return list(raw_vectors[pid])

    live_rows = [
        {
            "pid": pid,
            "y": int(pid == target),
            "x": raw_vector(pid),
            "s": score,
            "live_rank": rank,
            "in_rerank_head": True,
        }
        for rank, (score, pid) in enumerate(ordered, start=1)
    ]
    compatibility_ids = list(dict.fromkeys(
        ranked_ids[:compatibility_depth] + [target]
    ))
    rows = []
    for pid in compatibility_ids:
        row = {
            "pid": pid,
            "y": int(pid == target),
            "x": [round(value, 6) for value in raw_vector(pid)],
        }
        if compatibility_scores:
            score = base.get(pid)
            row["s"] = None if score is None else round(score, 6)
        rows.append(row)

    diagnostic_values = {
        "active_constraints": 0,
        "candidate_pool_size": len(set(candidate_pool)),
        "disclosed_constraint_types": [],
        "facets": [],
        "has_budget": False,
        "has_refusal": False,
        "route": "unknown",
    }
    if diagnostics is not None:
        diagnostic_values.update(dict(diagnostics))

    return {
        "schema": "techjam-rerank-snapshot-v4",
        "sample_id": str(sample_id),
        "scenario_type": str(scenario_type),
        "turn": int(turn),
        "target": target,
        "features": [str(name) for name in features],
        "reference_message": str(message),
        "reference_message_sha256": hashlib.sha256(
            str(message).encode("utf-8")
        ).hexdigest(),
        "target_in_pool": in_pool,
        "target_in_rerank_head": target_rank is not None,
        "target_live_rank": target_rank,
        "reachability": reachability,
        "diagnostics": diagnostic_values,
        "compatibility": {
            "depth": int(compatibility_depth),
            "numeric_precision": "six_decimal",
            "projection": "dedupe(full_ranked_ids[:depth] + [target])",
            "with_scores": bool(compatibility_scores),
        },
        "rows": rows,
        "live_rows": live_rows,
    }


def snapshots(agent, samples, catalog_ids, categories, products,
              negatives: int = NEGATIVES, with_scores: bool = False):
    """Replay the official loop, yielding one row per turn.

    `negatives` and `with_scores` exist for tools/snapshot_mrr.py, which needs the
    same replay this trainer uses but over the whole reranking depth and with the
    weighted sum's own score kept alongside each vector -- a reordering cannot be
    measured without the ordering it started from. The defaults reproduce the
    training set byte for byte, so the two callers cannot drift apart.
    """
    from src.features import FEATURES, vector

    agent.trace_features = True
    agent.trace_pool = True
    try:
        for sample in samples:
            # The session id is deliberately deterministic and never serialized.
            # Random UUIDs bought no isolation and made byte-rebuild reasoning harder.
            session_id = f"train_{sample['sample_id']}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            message = initial_message(
                effective, coarse_category(categories.get(target, [])), disclosed
            )

            for turn in range(1, MAX_TURNS + 1):
                failures_before = agent.failures
                response = agent.respond(session_id, message, turn, TOP_K)
                if agent.failures != failures_before:
                    raise RuntimeError(
                        "snapshot replay encountered a swallowed Agent.respond "
                        f"failure for {sample['sample_id']} turn {turn}; "
                        "refusing to serialize possibly stale trace state"
                    )
                context, head = agent.last_context(session_id)
                if context is not None and head:
                    candidate_pool = agent.candidate_pool(session_id)
                    record = snapshot_record(
                        sample_id=sample["sample_id"],
                        scenario_type=sample["scenario_type"], turn=turn,
                        target=target, message=message, features=FEATURES,
                        ranked=head, vector_for=lambda pid: vector(pid, context),
                        catalog_ids=catalog_ids,
                        candidate_pool=candidate_pool,
                        compatibility_depth=negatives,
                        compatibility_scores=with_scores,
                        diagnostics={
                            "active_constraints": int(context.constraints),
                            "candidate_pool_size": len(candidate_pool),
                            "disclosed_constraint_types": sorted({
                                classify_constraint(str(value))
                                for value in disclosed
                            }),
                            "facets": sorted(str(name) for name in context.facets),
                            "has_budget": context.budget is not None,
                            "has_refusal": bool(context.refused),
                            "route": (
                                "exact" if context.exact else
                                "inferred_hedge" if context.primary else
                                "global_fallback"
                            ),
                        },
                    )
                    if any(row["y"] for row in record["rows"]) \
                            and len(record["rows"]) > 1:
                        yield record
                recommended = [
                    item["parent_asin"]
                    for item in response.get("recommendations") or []
                ]
                if target in recommended:
                    break
                if turn == MAX_TURNS:
                    break
                override = effective.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    if str(override.get("new_value", "")):
                        disclosed.add(str(override["new_value"]))
                    message = str(override.get(
                        "message", "Actually, please ignore my earlier preference."
                    ))
                else:
                    message, boundary_used = customer_reply(
                        effective, response.get("ask_attribute"), disclosed,
                        boundary_used,
                    )
    finally:
        agent.trace_features = False
        agent.trace_pool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reranker training set")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--split", default="dev", choices=sorted(SPLITS))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    path = SPLITS[args.split]
    if not Path(path).exists():
        raise SystemExit(f"{path} is not here; see tools/setup_check.py")

    try:
        contract = training_contract(args.catalog, path, NEGATIVES)
    except CacheError as exc:
        raise SystemExit(str(exc)) from exc

    ids, categories, products = catalog_index(args.catalog)
    # Train against the shipped weighted sum, never against a half-applied model.
    agent = base_agent(args.catalog)

    output = Path(args.output or f"analysis/rerank_{args.split}.jsonl")
    groups = list(snapshots(agent, load_jsonl(path), ids, categories, products))
    written = sum(len(group["rows"]) for group in groups)
    try:
        write_cache(output, groups, contract)
    except CacheError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{args.split}: {len(groups)} snapshots, {written} rows -> {output}")


if __name__ == "__main__":
    main()
