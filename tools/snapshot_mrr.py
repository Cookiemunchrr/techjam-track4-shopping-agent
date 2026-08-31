"""Snapshot MRR: the metric a reordering change has to be measured with.

The composite score is 0.50*HR@10 + 0.30*MRR + 0.20*efficiency, and on this
harness its retrieval half is saturated -- internal HR@10 is 1.000 on every
official session. A change that only reorders candidates can therefore move at
most three tenths of it, diluted further by turn dynamics it does not touch. That
is how a real +0.03 improvement in ordering came back "within noise" and was
declined once already; the write-up is in analysis/reranker_experiment.json under
"the_measurement_error".

So this measures the mechanism directly. Replay the official loop once, with the
reranker off, and record every turn as a *snapshot*: the candidates the weighted
sum put in its head, in that order, with the score that produced it and the
feature vector the live path computed. Then ask one question of each ordering
under test -- where does the target land -- with the candidate set, the dialogue
state, the commit width and the turn dynamics all held fixed, because none of them
is what a reranker changes.

    snapshot MRR = mean over sessions of (mean over that session's snapshots
                                          of 1 / rank(target))

Sessions are the unit, not turns: a five-turn conversation is one observation of
one target, and averaging over turns instead would let the sessions that took
longest decide the number. The same grouping runs through the paired bootstrap,
because two snapshots of one conversation are not independent draws.

    python3 -m tools.snapshot_mrr --split dev
    python3 -m tools.snapshot_mrr --split dev --model analysis/reranker_candidate.json
    python3 -m tools.snapshot_mrr --splits dev,holdout --against analysis/reranker.json

`--against` is the comparison GATES-R asks for: a candidate model is not measured
against the weighted sum, which it was always going to beat, but against the model
that is already shipped.

`--blends` sweeps the one free parameter the asset has:

    python3 -m tools.snapshot_mrr --split dev --model cand.json --blends 0.01,0.05,0.1

Blend is how far the model may move things relative to the ordering the weighted
sum already produced, and it is not a knob to turn until the number goes up. The
shipped 0.05 was chosen on dev alone and validated on holdout once, and the reason
it is not the dev argmax is written into the asset: a larger blend widens the
model's score range, `CommitPolicy.width` reads a wider rank-1-to-rank-2 margin as
confidence, and the slate narrows -- which raises the official score without
ranking anything better. Sweep on dev, choose, then look at holdout once.

The replay is cached (analysis/rerank_snapshots_<split>.jsonl, regenerable and
gitignored) because it costs a catalog load and a hundred sessions. Its V4
manifest binds every group to the inputs, source, effective configuration, depth,
and feature schema that produced it. A missing or stale cache is never rebuilt as
a side effect of measurement; pass `--rebuild` explicitly.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluator.local_evaluator import MAX_TURNS, TOP_K
from src.features import FEATURES
from src.rerank import DEPTH
from tools.rerank_data import V4_LAYOUT
from tools.rerank_provenance import (CacheError, build_contract, load_cache,
                                     write_cache)

CATALOG = "data/catalog.jsonl"
SPLITS = {"dev": "analysis/dev.jsonl",
          "holdout": "analysis/holdout.jsonl",
          "full": "data/public_set.jsonl"}
CACHE = "analysis/rerank_snapshots_%s.jsonl"
BOOTSTRAP = 2000


def snapshot_contract(split: str, catalog: str | Path = CATALOG,
                      depth: int = DEPTH) -> dict:
    """The fixed-state replay contract for one split and candidate depth."""
    if split not in SPLITS:
        raise CacheError(f"unknown split {split!r}")
    return build_contract(
        catalog_path=catalog,
        split_path=SPLITS[split],
        features=FEATURES,
        depth=depth,
        generator_options={
            "compatibility_depth": depth,
            "kind": "snapshot_mrr",
            "layout": V4_LAYOUT,
            "live_numeric_precision": "raw_serving_float",
            "max_turns": MAX_TURNS,
            "projection": "dedupe(full_ranked_ids[:compatibility_depth] + [target])",
            "reference_ranker": "weighted_sum_reranker_disabled",
            "stop_policy": "first_visible_target",
            "top_k": TOP_K,
            "transcript": "visible_user_message_per_snapshot",
            "with_scores": True,
        },
    )


def collect(split: str, catalog: str = CATALOG, depth: int = DEPTH) -> list[dict]:
    """Replay one split and record its snapshots. Reranker off, by construction."""
    from evaluator.local_evaluator import catalog_index, load_jsonl
    from tools.rerank_data import base_agent, snapshots

    path = SPLITS[split]
    if not Path(path).exists():
        raise SystemExit(f"{path} is not here; see tools/setup_check.py")
    ids, categories, products = catalog_index(catalog)
    # The snapshots are the ordering the reranker is asked to improve on. Recording
    # them from an agent that is already reranking would measure a model against
    # its own output.
    agent = base_agent(catalog)
    return list(snapshots(agent, load_jsonl(path), ids, categories, products,
                          negatives=depth, with_scores=True))


def cached(split: str, rebuild: bool = False, catalog: str = CATALOG,
           depth: int = DEPTH) -> list[dict]:
    path = Path(CACHE % split)
    try:
        contract = snapshot_contract(split, catalog, depth)
        if not rebuild:
            return load_cache(path, contract)
        groups = collect(split, catalog, depth)
        write_cache(path, groups, contract)
        return groups
    except CacheError as exc:
        action = "" if rebuild else "; pass --rebuild to regenerate explicitly"
        raise SystemExit(f"{exc}{action}") from exc


def reciprocal_rank(group: dict, model=None) -> float:
    """Where the target lands in this snapshot under one ordering.

    Rows arrive in the order the weighted sum produced, so the base ranking needs
    no re-sorting -- and must not get one, because `profile.break_ties` has already
    settled ties inside it and re-sorting on score alone would undo that. A model
    reorders exactly the head the live path would reorder, by exactly the
    arithmetic `rerank.Model.apply` uses.

    A target the ranking never had in its head (`s` is None) scores zero: the
    reranker cannot reach it, so no ordering under test can be credited for it.
    """
    # V4 records exact raw serving values separately from the rounded A0
    # compatibility projection. Older synthetic fixtures and historical caches
    # remain readable through the fallback.
    head = [row for row in group.get("live_rows", group["rows"])
            if row.get("s") is not None]
    if model is None:
        for position, row in enumerate(head, start=1):
            if row["y"]:
                return 1.0 / position
        return 0.0
    rescored = [(row["s"] + model.blend * model.score_vector(row["x"]), row["pid"], row["y"])
                for row in head]
    rescored.sort(key=lambda triple: (-triple[0], triple[1]))
    for position, (_, _, is_target) in enumerate(rescored, start=1):
        if is_target:
            return 1.0 / position
    return 0.0


def by_session(groups) -> dict:
    out: dict[str, list[dict]] = {}
    for group in groups:
        out.setdefault(str(group["sample_id"]), []).append(group)
    return out


def session_scores(groups, model=None) -> dict:
    """One number per conversation: the mean reciprocal rank across its turns."""
    return {sid: sum(reciprocal_rank(g, model) for g in rows) / len(rows)
            for sid, rows in by_session(groups).items()}


def mrr(groups, model=None) -> float:
    scores = session_scores(groups, model)
    return sum(scores.values()) / len(scores) if scores else 0.0


def paired(groups, before, after, seed: int = 0, resamples: int = BOOTSTRAP):
    """95% interval for the difference, resampling whole sessions.

    Grouped because the unit of independence is the conversation, not the turn:
    every snapshot of one session shares a target, and resampling snapshots would
    treat five looks at the same answer as five observations. Both orderings see
    the same resample, so the session-to-session variance that dominates a
    marginal interval cancels and what is left is the difference.
    """
    first, second = session_scores(groups, before), session_scores(groups, after)
    keys = sorted(first)
    if not keys:
        return (0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(keys)
    deltas = []
    for _ in range(resamples):
        draw = [keys[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(second[k] - first[k] for k in draw) / n)
    deltas.sort()
    return (deltas[int(0.025 * resamples)], deltas[int(0.500 * resamples)],
            deltas[int(0.975 * resamples)], n)


def load_model(path: str | None, blend: float | None = None):
    """An ordering under test, including one trained before a feature was appended.

    `rerank.Model.from_dict` refuses an asset whose feature list is not the
    current one, and that refusal stays exactly as strict on the serving path. It
    would make this tool useless, though: the comparison GATES-R asks for is a
    candidate against *the model that is already shipped*, and appending a feature
    is precisely the change that puts those two on different vectors.

    An older asset is admitted here on one condition -- its feature list is a
    prefix of the current one, which is the only change `features.FEATURES`
    permits. A model that never saw the appended features is exactly the model
    that scores them zero, so padding is not an approximation of the old ordering,
    it is the old ordering. Anything else is still refused.
    """
    if not path or path.lower() == "none":
        return None
    from src.rerank import Model
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    names = list(payload.get("features") or FEATURES)
    if names != list(FEATURES):
        if names != list(FEATURES[:len(names)]):
            raise SystemExit(
                f"{path} lists features that are not a prefix of "
                f"features.FEATURES; it was trained against a different vector "
                f"and padding it would compare two different models.")
        print(f"  {path}: trained on {len(names)} features, padding "
              f"{len(FEATURES) - len(names)} appended ones with zero weight")
        payload = dict(payload, features=list(FEATURES),
                       weights=list(payload["weights"]) + [0.0] * (len(FEATURES) - len(names)))
    model = Model.from_dict(payload)
    if blend is not None:
        model.blend = float(blend)
    return model


def report(groups, base_model, model, seed: int = 0) -> dict:
    low, mid, high, n = paired(groups, base_model, model, seed)
    before = mrr(groups, base_model)
    after = mrr(groups, model)
    return {
        "snapshots": len(groups),
        "sessions": n,
        "before": round(before, 5),
        "after": round(after, 5),
        # The point estimate is the observed paired difference. The median of a
        # finite bootstrap sample is useful as a resampling diagnostic, but it is
        # not the empirical effect and can differ enough to make report arithmetic
        # appear inconsistent.
        "delta": round(after - before, 5),
        "bootstrap_median_delta": round(mid, 5),
        "ci95": [round(low, 5), round(high, 5)],
        "verdict": "significant" if low > 0 or high < 0 else "within noise",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot MRR for a reordering change")
    parser.add_argument("--splits", default="dev,holdout")
    parser.add_argument("--split", default="", help="shorthand for --splits")
    parser.add_argument("--model", default="analysis/reranker.json",
                        help="the ordering under test; 'none' for the weighted sum")
    parser.add_argument("--against", default="none",
                        help="the ordering it has to beat. Default 'none' (the "
                             "weighted sum); pass analysis/reranker.json to hold a "
                             "candidate to the standing ship bar")
    parser.add_argument("--blend", type=float, default=None,
                        help="override the blend in the asset under test")
    parser.add_argument("--blends", default="",
                        help="comma-separated blends to sweep instead of one run; "
                             "choose on dev, then read holdout once")
    parser.add_argument("--against-blend", type=float, default=None)
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--rebuild", action="store_true", help="re-replay the sessions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="analysis/snapshot_mrr.json")
    args = parser.parse_args()

    splits = [args.split] if args.split else args.splits.split(",")
    base_model = load_model(args.against, args.against_blend)
    blends = [float(b) for b in args.blends.split(",") if b.strip()] or [None]
    rows = []
    for split in splits:
        groups = cached(split.strip(), args.rebuild, args.catalog)
        for blend in blends:
            model = load_model(args.model, args.blend if blend is None else blend)
            row = {"split": split.strip(), "against": args.against,
                   "model": args.model,
                   "blend": None if model is None else model.blend,
                   **report(groups, base_model, model, args.seed)}
            rows.append(row)
            print("  %-8s blend %-6s %d snapshots / %d sessions   %.4f -> %.4f   "
                  "%+.4f [%+.4f, %+.4f]  %s" % (
                      row["split"], row["blend"], row["snapshots"], row["sessions"],
                      row["before"], row["after"], row["delta"], row["ci95"][0],
                      row["ci95"][1], row["verdict"]))
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
