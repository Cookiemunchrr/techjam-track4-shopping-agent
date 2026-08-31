"""LLM semantic ranking probe (V7 P1): measured evidence for or against an
LLM reranking stage, on the same instrument the linear reranker cleared.

Pillar I names "Multi-Route Retrieval -> LLM Semantic Ranking". The shipped
ranking stage is a linear model over interpretable features (src/rerank.py);
no LLM ordering has ever been built or measured here. This tool is the
measurement. It lives in tools/ and never in src/: the scored path keeps its
zero-socket property by construction, and any tokens spent are BUILD-TIME
tokens, reported separately from the "0 prompt / 0 completion" scoring claim.

Method: reuse the snapshot-MRR harness (tools/snapshot_mrr.py) bit-for-bit --
same frozen snapshots, same base order, same session grouping, same paired
bootstrap, same blend discipline (sweep on dev, read holdout once). The only
thing swapped is where the reordering signal comes from: a position credit
assigned by an LLM asked to rank the top-N candidates, instead of the linear
model's score_vector. Two self-checks prove the swap is the only difference:

  1. arithmetic_equivalence -- this file's rescoring arithmetic, fed the
     shipped linear model's own contributions, reproduces
     snapshot_mrr.reciprocal_rank exactly (max abs difference 0.0).
  2. bootstrap_parity -- this file's paired bootstrap over precomputed score
     dictionaries returns the identical interval to snapshot_mrr.paired on
     the same input.

Usage:
    set AIAND_API_KEY=...   (or export; never in a file, never committed)
    python3 -m tools.llm_rerank_probe --splits dev --limit 5     # smoke
    python3 -m tools.llm_rerank_probe --splits dev,holdout       # full run

Env: AIAND_API_KEY (required), AIAND_BASE_URL (default
https://api.aiand.com/v1), LLM_RERANK_MODEL (default moonshotai/kimi-k3).
Requests and responses are cached whole in analysis/_llm_rerank_cache.jsonl
(gitignored) so a re-run re-checks the measurement without re-spending
tokens. The RESULT is committed (analysis/llm_rerank_probe.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools import snapshot_mrr
from tools.snapshot_mrr import BOOTSTRAP
# The HTTP call and ballot parsing live in the shared client (W2 extracts them
# for the opt-in runtime stage); the probe's public names are kept as aliases
# so its tests and cache format are undisturbed.
from tools.llm_client import chat_once as _shared_chat_once, parse_order

TOP_N = 20
BLENDS = [0.05, 0.1, 0.25, 0.5, 1.0]
CACHE = Path("analysis/_llm_rerank_cache.jsonl")
CATALOG = "data/catalog.jsonl"
OUTPUT = "analysis/llm_rerank_probe.json"

SYSTEM = (
    "You are ranking candidate products for a shopper in a multi-turn shopping "
    "conversation. Rank the candidates by how well they match everything the "
    "shopper has asked for so far, most relevant first. Reply with JSON only: "
    '{"order": ["<product id>", ...]} containing every given id exactly once.'
)


# --------------------------------------------------------------------------
# Catalog join: the LLM ranks titles, the snapshots carry only product ids.
# --------------------------------------------------------------------------

def load_titles(catalog_path: str = CATALOG) -> dict[str, str]:
    titles = {}
    with open(catalog_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pid = row.get("parent_asin") or row.get("asin")
            if pid:
                titles[pid] = (row.get("title") or "").strip()
    return titles


# --------------------------------------------------------------------------
# The LLM call, with a whole-transcript cache so re-runs spend nothing.
# --------------------------------------------------------------------------

def _cache_key(body: dict) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                cache[entry["key"]] = entry
    return cache


def chat_once(body: dict, cache: dict[str, dict], cache_path: Path,
              retries: int = 3) -> dict:
    """One chat-completion call, cached by exact request bytes."""
    return _shared_chat_once(body, cache, cache_path, retries=retries)


def llm_order(group: dict, history: list[str], titles: dict[str, str],
              cache: dict, cache_path: Path, model: str,
              top_n: int = TOP_N) -> tuple[list[str], dict]:
    """Rank this snapshot's top-N head rows. Returns (order, usage)."""
    head = [row for row in group.get("live_rows", group["rows"])
            if row.get("s") is not None][:top_n]
    pids = [row["pid"] for row in head]
    if not any(row["y"] for row in head):
        return pids, {"reachable": False}  # target outside top-N: no call
    lines = ["The shopper's messages so far:", ""]
    lines += [f"Shopper: {m}" for m in history]
    lines += ["", "Candidates (id -- title):"]
    lines += [f"{i + 1}. {pid} -- {titles.get(pid, '(untitled)')}"
              for i, pid in enumerate(pids)]
    lines += ["", f"Rank all {len(pids)} candidates, best first."]
    body = {"model": model, "temperature": 0, "max_tokens": 8192,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "\n".join(lines)}]}
    payload = chat_once(body, cache, cache_path)
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    # A reasoning model can exhaust max_tokens on reasoning and return
    # content null/empty: a declined ballot, scored as the base order and
    # counted -- a model that often cannot answer is itself a finding.
    content = message.get("content") or ""
    usage = dict(payload.get("usage") or {})
    usage["reachable"] = True
    usage["empty_content"] = not content.strip()
    usage["_cached"] = payload.get("_cached", False)
    return parse_order(content, pids), usage


# --------------------------------------------------------------------------
# Reorder arithmetic: snapshot_mrr.reciprocal_rank with the signal swapped.
# --------------------------------------------------------------------------

def rr_with_credits(group: dict, credits: dict[str, float], blend: float) -> float:
    """reciprocal_rank, verbatim but for the credit source (cited, tested).

    snapshot_mrr.reciprocal_rank computes ``s + blend * model.score_vector(x)``
    per head row and re-sorts on (-score, pid). Here the per-row contribution
    is ``blend * credits.get(pid, 0.0)`` and everything else -- the head
    filter, the sort key, the tie-break -- is identical by construction.
    """
    head = [row for row in group.get("live_rows", group["rows"])
            if row.get("s") is not None]
    rescored = [(row["s"] + blend * credits.get(row["pid"], 0.0), row["pid"], row["y"])
                for row in head]
    rescored.sort(key=lambda triple: (-triple[0], triple[1]))
    for position, (_, _, is_target) in enumerate(rescored, start=1):
        if is_target:
            return 1.0 / position
    return 0.0


def rr_raw_order(group: dict, order: list[str], top_n: int = TOP_N) -> float:
    """The LLM's order taken raw: base scores ignored entirely."""
    head = [row for row in group.get("live_rows", group["rows"])
            if row.get("s") is not None]
    position = {pid: i for i, pid in enumerate(order)}
    ranked = sorted(range(len(head)),
                    key=lambda i: (position.get(head[i]["pid"], top_n + i),))
    for rank, i in enumerate(ranked, start=1):
        if head[i]["y"]:
            return 1.0 / rank
    return 0.0


def session_scores_ordered(groups, credits_by_gid, blend) -> dict:
    out: dict[str, list[float]] = {}
    for group in groups:
        credits = credits_by_gid.get(id(group), {})
        out.setdefault(str(group["sample_id"]), []).append(
            rr_with_credits(group, credits, blend))
    return {sid: sum(vals) / len(vals) for sid, vals in out.items()}


def paired_dicts(first: dict, second: dict, seed: int = 0,
                 resamples: int = BOOTSTRAP):
    """snapshot_mrr.paired over precomputed per-session scores, verbatim loop."""
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


# --------------------------------------------------------------------------
# Self-checks: the swap is the only difference.
# --------------------------------------------------------------------------

def self_checks(groups_by_split: dict[str, list]) -> dict:
    from tools.snapshot_mrr import load_model, reciprocal_rank, paired, mrr
    model = load_model("analysis/reranker.json")
    worst = 0.0
    for split, groups in groups_by_split.items():
        for group in groups:
            head = [r for r in group.get("live_rows", group["rows"])
                    if r.get("s") is not None]
            credits = {r["pid"]: model.score_vector(r["x"]) for r in head}
            mine = rr_with_credits(group, credits, model.blend)
            theirs = reciprocal_rank(group, model)
            worst = max(worst, abs(mine - theirs))
    any_groups = next(iter(groups_by_split.values()))
    theirs_interval = paired(any_groups, None, None)
    base = snapshot_mrr.session_scores(any_groups, None)
    mine_interval = paired_dicts(base, base)
    return {
        "arithmetic_equivalence_max_abs_diff": worst,
        "bootstrap_parity": list(theirs_interval) == list(mine_interval),
        "harness": "tools/snapshot_mrr.py",
        "reference_reranker_recorded": {"dev": 0.02067, "holdout": 0.03544},
        "reference_reranker_here": {
            split: round(mrr(groups, model) - mrr(groups, None), 5)
            for split, groups in groups_by_split.items()},
    }


# --------------------------------------------------------------------------

def run_split(split: str, titles: dict, cache: dict, cache_path: Path,
              model: str, blends: list[float], limit: int = 0) -> dict:
    groups = snapshot_mrr.cached(split)
    if limit:
        keep = sorted({str(g["sample_id"]) for g in groups})[:limit]
        groups = [g for g in groups if str(g["sample_id"]) in keep]
    # The dialogue so far: this session's visible user messages up to this
    # turn, in turn order -- the state the live retriever ranked under.
    history: dict[str, list[str]] = {}
    orders: dict[int, list[str]] = {}
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0,
                   "calls": 0, "cached_calls": 0, "empty_content": 0}
    unreachable = 0
    for group in sorted(groups, key=lambda g: (str(g["sample_id"]), g["turn"])):
        sid = str(group["sample_id"])
        history.setdefault(sid, []).append(group["reference_message"])
        order, usage = llm_order(group, history[sid], titles, cache,
                                 cache_path, model)
        orders[id(group)] = order
        if not usage.get("reachable"):
            unreachable += 1
            continue
        usage_total["calls"] += 1
        usage_total["cached_calls"] += 1 if usage.get("_cached") else 0
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
        usage_total["empty_content"] += 1 if usage.get("empty_content") else 0

    credits_by_gid = {}
    raw_scores = {}
    for group in groups:
        order = orders[id(group)]
        n = len(order)
        credits = {pid: (n - 1 - i) / (n - 1) if n > 1 else 0.0
                   for i, pid in enumerate(order)}
        credits_by_gid[id(group)] = credits
        raw_scores[id(group)] = rr_raw_order(group, order)

    base = snapshot_mrr.session_scores(groups, None)
    results = {}
    for blend in blends:
        mine = session_scores_ordered(groups, credits_by_gid, blend)
        low, mid, high, n = paired_dicts(base, mine)
        results[f"blend_{blend}"] = {
            "before": round(sum(base.values()) / len(base), 5),
            "after": round(sum(mine.values()) / len(mine), 5),
            "delta": round(sum(mine.values()) / len(mine)
                           - sum(base.values()) / len(base), 5),
            "bootstrap_median_delta": round(mid, 5),
            "ci95": [round(low, 5), round(high, 5)],
            "sessions": n,
        }
    raw_by_session: dict[str, list[float]] = {}
    for group in groups:
        raw_by_session.setdefault(str(group["sample_id"]), []).append(
            raw_scores[id(group)])
    raw_mean = {sid: sum(v) / len(v) for sid, v in raw_by_session.items()}
    low, mid, high, n = paired_dicts(base, raw_mean)
    results["raw_llm_order"] = {
        "before": round(sum(base.values()) / len(base), 5),
        "after": round(sum(raw_mean.values()) / len(raw_mean), 5),
        "delta": round(sum(raw_mean.values()) / len(raw_mean)
                       - sum(base.values()) / len(base), 5),
        "bootstrap_median_delta": round(mid, 5),
        "ci95": [round(low, 5), round(high, 5)],
        "sessions": n,
    }
    return {"snapshots": len(groups), "sessions": len(base),
            "target_outside_top_n": unreachable, "usage": usage_total,
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM semantic ranking probe (V7 P1)")
    parser.add_argument("--splits", default="dev,holdout")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--blends", default=",".join(str(b) for b in BLENDS))
    parser.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke)")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    model = os.environ.get("LLM_RERANK_MODEL", "moonshotai/kimi-k3")
    blends = [float(b) for b in args.blends.split(",") if b.strip()]
    titles = load_titles()
    cache = _load_cache(CACHE)
    splits = [s.strip() for s in args.splits.split(",")]

    groups_by_split = {split: snapshot_mrr.cached(split) for split in splits}
    checks = self_checks(groups_by_split)
    if checks["arithmetic_equivalence_max_abs_diff"] != 0.0 \
            or not checks["bootstrap_parity"]:
        raise SystemExit(f"self-checks failed; refusing to measure: {checks}")

    per_split = {}
    for split in splits:
        print(f"  {split}: calling {model} ...", flush=True)
        per_split[split] = run_split(split, titles, cache, CACHE, model,
                                     blends, args.limit)
        for name, row in per_split[split]["results"].items():
            print("    %-14s %.4f -> %.4f  %+.5f [%+.5f, %+.5f]" % (
                name, row["before"], row["after"], row["delta"],
                row["ci95"][0], row["ci95"][1]), flush=True)

    totals = {"prompt_tokens": sum(s["usage"]["prompt_tokens"] for s in per_split.values()),
              "completion_tokens": sum(s["usage"]["completion_tokens"] for s in per_split.values()),
              "calls": sum(s["usage"]["calls"] for s in per_split.values()),
              "cached_calls": sum(s["usage"]["cached_calls"] for s in per_split.values())}
    report = {
        "schema": "techjam-llm-rerank-probe-v1",
        "plan": "IMPROVEMENT_PLAN_V7.txt P1",
        "model": model,
        "endpoint": os.environ.get("AIAND_BASE_URL", "https://api.aiand.com/v1"),
        "protocol": {
            "instrument": "tools/snapshot_mrr.py snapshots, base order, grouping and bootstrap reused; the reordering signal is an LLM position credit over the top-20 head rows",
            "top_n": args.top_n,
            "blends_swept": blends + ["raw"],
            "blend_discipline": "sweep on dev, choose on dev, read holdout once",
            "temperature": 0,
            "credit": "linear position credit over the top-N: best 1.0, Nth 0.0; rows beyond top-N keep their base score",
            "cache": str(CACHE) + " (gitignored; whole request/response per call)",
            "dialogue_state": "the session's visible user messages up to the snapshot's turn",
        },
        "self_checks": checks,
        "splits": per_split,
        "build_time_tokens": {**totals,
                              "note": "build-time measurement tokens; the scored path remains 0 prompt / 0 completion"},
        "cost_usd": None,
        "cost_note": "AI& does not expose per-call pricing to this probe; report from the AI& billing console against the token counts above.",
        "verdict": None,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
