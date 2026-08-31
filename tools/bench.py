"""Benchmark runner: score the agent across splits, paraphrase levels and axes.

    python3 -m tools.bench                  # clean run on every split
    python3 -m tools.bench --paraphrase     # the scaffolding-only matrix
    python3 -m tools.bench --adversarial    # the axes that actually move the score
    python3 -m tools.bench --longtail       # below-median-popularity targets only

The adversarial matrix used to ignore --splits entirely and always measure the
full 200-session set, which made a dev-only adversarial read impossible without
also opening the holdout. It now honors both, defaulting to the full set so a
plain `--adversarial` run is unchanged:

    python3 -m tools.bench --adversarial --splits dev --axes category

A requested split whose file is absent is an error, not a silently skipped row:
a run that measured nothing must not read as a run that passed. Rebuild the
splits with `python3 -m tools.setup_check --splits`.

Any mode can be compared against an earlier run of the same mode:

    python3 -m tools.bench --longtail --output analysis/longtail_after.json \
        --against analysis/longtail.json

which is the long-tail ship gate a reranker retrain has to clear (GATES-R item e
in the improvement plan). The point estimate on that slice is 25 sessions wide;
the paired interval is the only reading of it that means anything.
"""
from __future__ import annotations

import argparse
import json
import os
import time

CATALOG = "data/catalog.jsonl"
SPLITS = {"dev": "analysis/dev.jsonl",
          "holdout": "analysis/holdout.jsonl",
          "full": "data/public_set.jsonl"}


BOOTSTRAP = 2000       # resamples; enough for a stable 95% interval at n=200


def _split_path(split: str) -> str:
    """Resolve a split name to its file, failing closed when it is absent.

    An absent split used to be skipped, so a run over a missing file reported
    whatever remained as though it were the requested measurement. A split that
    was asked for and cannot be read is a stop, not a pass.
    """
    if split not in SPLITS:
        raise SystemExit(f"unknown split {split!r}; choose from {sorted(SPLITS)}")
    path = SPLITS[split]
    if not os.path.exists(path):
        raise SystemExit(
            f"split {split!r} is missing at {path}; refusing to report a run that "
            "measured nothing. Rebuild it with: python3 -m tools.setup_check --splits")
    return path


def _dataset_sha256(path: str) -> str:
    """Content hash of the measured split, carried on every output row.

    A paired comparison is only meaningful when both sides measured the same
    input; the hash is what lets `--against` refuse a mismatch instead of
    intersecting session ids and calling the overlap a comparison.
    """
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_interval(sessions, seed: int = 0, resamples: int = BOOTSTRAP):
    """A 95% confidence interval for TechnicalScore, by resampling sessions.

    With two hundred sessions a difference of a few thousandths is one session
    landing differently, and the repo has already spent effort on changes that
    turned out to be inside that band. Reporting the interval alongside the point
    estimate is what makes "this beat the baseline" a claim rather than a hope.

    Resampling is over whole sessions, which is the unit that is independent
    here -- the turns inside one session emphatically are not.
    """
    import random
    if not sessions:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(sessions)
    scores: list[float] = []
    for _ in range(resamples):
        draw = [sessions[rng.randrange(n)] for _ in range(n)]
        hits = sum(1 for s in draw if s["hit"])
        mrr = sum(s["reciprocal_rank"] for s in draw) / n
        mttc = sum(s["first_hit_turn"] if s["first_hit_turn"] is not None else 11
                   for s in draw) / n
        efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
        scores.append(0.50 * hits / n + 0.30 * mrr + 0.20 * efficiency)
    scores.sort()
    return (scores[int(0.025 * resamples)],
            scores[int(0.500 * resamples)],
            scores[int(0.975 * resamples)])


def _score(draw) -> float:
    n = len(draw)
    hits = sum(1 for s in draw if s["hit"])
    mrr = sum(s["reciprocal_rank"] for s in draw) / n
    mttc = sum(s["first_hit_turn"] if s["first_hit_turn"] is not None else 11
               for s in draw) / n
    return 0.50 * hits / n + 0.30 * mrr + 0.20 * max(0.0, min(1.0, (11.0 - mttc) / 10.0))


def paired_interval(before: dict, after: dict, seed: int = 0,
                    resamples: int = BOOTSTRAP):
    """95% interval for the *difference* between two runs on the same sessions.

    This is the test that matters and the marginal intervals above are not it.
    A single run's interval is roughly +/-0.017 on two hundred sessions, which
    would declare every change this repo has ever made insignificant; but the two
    runs see the same sessions, so almost all of that spread is shared and cancels.
    Resampling the *paired* difference removes it, and a change whose interval
    still straddles zero has genuinely not been shown to do anything.

    Both arguments map sample_id -> session record. Only ids present in both are
    used, so a run over a different split is compared on the overlap or not at all.
    """
    import random
    shared = sorted(set(before) & set(after))
    if not shared:
        return (0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(shared)
    deltas: list[float] = []
    for _ in range(resamples):
        draw = [shared[rng.randrange(n)] for _ in range(n)]
        deltas.append(_score([after[i] for i in draw]) - _score([before[i] for i in draw]))
    deltas.sort()
    return (deltas[int(0.025 * resamples)], deltas[int(0.500 * resamples)],
            deltas[int(0.975 * resamples)], n)


def _by_id(sessions) -> dict:
    return {str(s["sample_id"]): s for s in sessions or []}


def _harness():
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    return catalog_index, evaluate, load_jsonl


# Which reranker asset every agent built in this process should serve. Empty means
# whatever `src/rerank.py` loads, which is the shipped one.
MODEL = ""


def _agent(catalog: str = CATALOG):
    """A fresh Agent per measurement, optionally serving a candidate model.

    Fresh because the elicitation model learns across sessions on one instance, so
    reusing it would carry training from the previous row into this one. Optionally
    overridden because GATES-R holds a candidate to this whole battery -- splits,
    paraphrase axes, long-tail slice -- and copying a candidate over the shipped
    asset to measure it is how a half-tested model gets left in place.
    """
    from src.rerank import load as load_shipped
    from tools.rerank_data import base_agent
    from tools.snapshot_mrr import load_model
    agent = base_agent(catalog)
    agent.reranker = load_model(MODEL) if MODEL else load_shipped()
    return agent


def run(splits, levels, seed: int = 0) -> list[dict]:
    catalog_index, evaluate, load_jsonl = _harness()
    from tools.paraphrase import ParaphrasingAgent

    ids, categories, products = catalog_index(CATALOG)
    rows = []
    for split in splits:
        path = _split_path(split)
        dataset_sha256 = _dataset_sha256(path)
        samples = load_jsonl(path)
        for level in levels:
            # A fresh Agent per measurement. The elicitation model learns across
            # sessions, so reusing one instance would carry training from the
            # previous row into this one and quietly inflate it.
            agent = _agent()
            wrapped = agent if level == 0 else ParaphrasingAgent(agent, level, seed)
            start = time.perf_counter()
            result = evaluate(wrapped, samples, ids, categories, products)
            low, _, high = bootstrap_interval(result.get("sessions") or [], seed)
            rows.append({
                "split": split, "paraphrase_level": level,
                "dataset_sha256": dataset_sha256,
                "technical_score": round(result["recommended_technical_score"], 5),
                "ci95": [round(low, 5), round(high, 5)],
                # Kept so a later run can be compared against this one with
                # `--against`, which is the only comparison that clears the noise.
                "sessions": [{"sample_id": str(x["sample_id"]), "hit": x["hit"],
                              "reciprocal_rank": x["reciprocal_rank"],
                              "first_hit_turn": x["first_hit_turn"]}
                             for x in (result.get("sessions") or [])],
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": round(result["mrr"], 5), "mttc": result["mttc"],
                "seconds": round(time.perf_counter() - start, 1),
                "scenario_metrics": result["scenario_metrics"],
            })
            print("  %-8s L%d  score=%.5f [%.5f, %.5f]  HR=%.3f  MRR=%.3f  MTTC=%.2f  (%.0fs)"
                  % (split, level, rows[-1]["technical_score"], low, high,
                     rows[-1]["hit_rate_at_10"], rows[-1]["mrr"], rows[-1]["mttc"],
                     rows[-1]["seconds"]))
    return rows


def run_adversarial(seed: int = 0, splits=("full",), axes=None) -> list[dict]:
    """One row per axis, plus the combination. See tools/adversarial.py.

    `splits` defaults to the full public set, which is what this mode measured
    before it learned to see --splits at all; passing --splits dev keeps a
    development read away from the holdout file. `axes` restricts the matrix to
    the control row plus the named axes (granularity rows are named
    "granularity=1"/"granularity=3"); None runs the whole frozen matrix.
    """
    catalog_index, evaluate, load_jsonl = _harness()
    from tools.adversarial import AXES, Adversarial, drifted_categories

    ids, categories, products = catalog_index(CATALOG)
    selected = set(axes) if axes else None
    rows = []
    for split in splits:
        path = _split_path(split)
        dataset_sha256 = _dataset_sha256(path)
        samples = load_jsonl(path)
        plans = [("control", ()), *[(axis, (axis,)) for axis in AXES if axis != "granularity"],
                 ("all paraphrase axes", tuple(a for a in AXES if a != "granularity"))]
        for label, plan_axes in plans:
            if selected is not None and label != "control" and label not in selected:
                continue
            agent = _agent()
            wrapped = agent if not plan_axes else Adversarial(agent, plan_axes, seed)
            result = evaluate(wrapped, samples, ids, categories, products)
            rows.append({"axis": label, "split": split,
                         "dataset_sha256": dataset_sha256, **_metrics(result)})
            print("  %-22s %-8s score=%.5f  HR=%.3f  MRR=%.3f  MTTC=%.2f" % (
                label, split, rows[-1]["technical_score"], rows[-1]["hit_rate_at_10"],
                rows[-1]["mrr"], rows[-1]["mttc"]))

        # Simulator drift is not a paraphrase: the harness itself behaves differently.
        for components in (1, 3):
            label = f"granularity={components}"
            if selected is not None and label not in selected:
                continue
            agent = _agent()
            result = evaluate(agent, samples, ids,
                              drifted_categories(categories, components), products)
            rows.append({"axis": label, "split": split,
                         "dataset_sha256": dataset_sha256, **_metrics(result)})
            print("  %-22s %-8s score=%.5f  HR=%.3f  MRR=%.3f  MTTC=%.2f" % (
                label, split, rows[-1]["technical_score"], rows[-1]["hit_rate_at_10"],
                rows[-1]["mrr"], rows[-1]["mttc"]))
    return rows


LONGTAIL_PERCENTILE = 0.90


def run_longtail() -> list[dict]:
    """How the agent does when the target is *not* the obvious popular choice.

    Converts the popularity prior from a stated limitation into a number, and
    incidentally measures how little room there is to state it: the median target
    sits at the 99.3rd percentile of its own bucket by review count, and only 4
    of 200 fall below their bucket's median. The slice is therefore drawn at the
    90th percentile, which is still only 25 sessions. Report the sample size --
    a limitation that cannot be measured on the available data is a finding, not
    an excuse.
    """
    catalog_index, evaluate, load_jsonl = _harness()
    ids, categories, products = catalog_index(CATALOG)
    full_path = _split_path("full")
    dataset_sha256 = _dataset_sha256(full_path)
    samples = load_jsonl(full_path)
    agent = _agent()
    catalog = agent.catalog
    where = {pid: name for name, members in catalog.buckets.items() for pid in members}

    hard, easy, percentiles = [], [], []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        members = catalog.buckets.get(where.get(target, ""), [])
        if not members:
            continue
        pops = sorted(catalog.meta[pid]["pop"] for pid in members)
        below = sum(1 for value in pops if value < catalog.meta[target]["pop"])
        percentile = below / max(len(pops) - 1, 1)
        percentiles.append(percentile)
        (hard if percentile < LONGTAIL_PERCENTILE else easy).append(sample)

    percentiles.sort()
    median_percentile = percentiles[len(percentiles) // 2] if percentiles else 0.0
    print("  median target sits at the %.1fth percentile of its own bucket"
          % (median_percentile * 100))

    rows = [{"slice": "target popularity percentile (median)",
             "value": round(median_percentile, 4), "session_count": len(percentiles),
             "split": "full", "dataset_sha256": dataset_sha256}]
    for label, subset in ((f"below p{int(LONGTAIL_PERCENTILE * 100)} popularity", hard),
                          ("at or above", easy)):
        if not subset:
            continue
        result = evaluate(_agent(), subset, ids, categories, products)
        # `session_count`, not `sessions`: _metrics fills the latter with the
        # per-session detail --against needs, and the count is the number this
        # slice exists to report. 25 sessions is the whole finding here.
        rows.append({"slice": label, "session_count": len(subset),
                     "split": "full", "dataset_sha256": dataset_sha256,
                     **_metrics(result)})
        print("  %-24s n=%-4d score=%.5f  HR=%.3f  MRR=%.3f" % (
            label, rows[-1]["session_count"], rows[-1]["technical_score"],
            rows[-1]["hit_rate_at_10"], rows[-1]["mrr"]))
    return rows


def _metrics(result: dict) -> dict:
    """Row summary, with the per-session detail `--against` needs.

    The adversarial axes and the long-tail slice used to report a point estimate
    and nothing else, so the only mode that could be compared against an earlier
    run was the plain one. Those are precisely the rows a reranker retrain has to
    be held to -- the improvement plan's GATES-R makes the compound paraphrase axis
    and the long-tail slice ship gates -- and a point estimate on 25 sessions is
    not a comparison. Carrying the sessions costs a few hundred kilobytes of JSON
    and turns both into paired intervals.
    """
    return {
        "technical_score": round(result["recommended_technical_score"], 5),
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": round(result["mrr"], 5),
        "mttc": result["mttc"],
        "sessions": [{"sample_id": str(x["sample_id"]), "hit": x["hit"],
                      "reciprocal_rank": x["reciprocal_rank"],
                      "first_hit_turn": x["first_hit_turn"]}
                     for x in (result.get("sessions") or [])],
        "scenario_metrics": result["scenario_metrics"],
    }


def _row_key(row: dict):
    """What makes two rows the same measurement across runs.

    One key covers every mode: a split at a paraphrase level, an adversarial axis,
    a popularity slice. Comparing a row against a different row is worse than not
    comparing it at all, so the key is explicit rather than positional.
    """
    return (row.get("split"), row.get("paraphrase_level"),
            row.get("slice"), row.get("axis"))


def _row_label(row: dict) -> str:
    return str(row.get("axis") or row.get("slice")
               or "%s L%s" % (row.get("split"), row.get("paraphrase_level")))


def _require_comparable(previous: dict, row: dict) -> None:
    """Refuse a paired comparison that would silently intersect two runs.

    The paired interval is only meaningful when both sides measured the same
    sessions from the same input. Comparing the id-overlap of two different
    splits reads as a paired comparison and is not one, so a mismatched dataset
    hash or a mismatched session set is a stop, not a smaller n. An earlier
    artifact from before rows carried hashes simply cannot prove the input
    matched; the session-set check still applies to it.
    """
    label = _row_label(row)
    old_hash = previous.get("dataset_sha256")
    new_hash = row.get("dataset_sha256")
    if old_hash and new_hash and old_hash != new_hash:
        raise SystemExit(
            f"refusing to compare {label}: dataset hashes differ "
            f"({old_hash[:12]}... vs {new_hash[:12]}...); the two runs did not "
            "measure the same input")
    old_ids = {str(s["sample_id"]) for s in previous["sessions"]}
    new_ids = {str(s["sample_id"]) for s in row["sessions"]}
    if old_ids != new_ids:
        raise SystemExit(
            f"refusing to compare {label}: session sets differ "
            f"({len(old_ids)} vs {len(new_ids)} sessions); a paired interval over "
            "the intersection would silently compare different sessions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the agent across splits and axes")
    parser.add_argument("--paraphrase", action="store_true", help="also run levels 1-3")
    parser.add_argument("--adversarial", action="store_true",
                        help="per-axis adversarial matrix (tools/adversarial.py)")
    parser.add_argument("--longtail", action="store_true",
                        help="below-median-popularity targets only")
    parser.add_argument("--splits", default=None,
                        help="comma-separated splits (dev,holdout,full). Default: "
                             "all three for a clean run; the full public set for "
                             "--adversarial, matching its historical behavior")
    parser.add_argument("--axes", default=None,
                        help="comma-separated adversarial axes to run alongside the "
                             "control row (category,natural,scaffold,constraint,"
                             "granularity=1,granularity=3,all paraphrase axes). "
                             "Default: the whole frozen matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="",
                        help="a candidate reranker asset to serve instead of the "
                             "shipped one; 'none' for the weighted sum alone")
    parser.add_argument("--output", default="analysis/bench.json")
    parser.add_argument("--against", default="",
                        help="an earlier bench.json; report the paired 95%% interval "
                             "for the difference, which is the test that clears noise")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    if args.splits is None:
        splits = ["full"] if args.adversarial else ["dev", "holdout", "full"]
    else:
        splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    axes = None
    if args.axes is not None:
        from tools.adversarial import AXES
        axes = [value.strip() for value in args.axes.split(",") if value.strip()]
        valid = ({"control", "all paraphrase axes", "granularity=1", "granularity=3"}
                 | {axis for axis in AXES if axis != "granularity"})
        unknown = [axis for axis in axes if axis not in valid]
        if not axes or unknown:
            raise SystemExit(f"unknown/empty adversarial axes: {unknown}; "
                             f"choose from {sorted(valid)}")

    if args.adversarial:
        rows = run_adversarial(args.seed, splits=splits, axes=axes)
    elif args.longtail:
        rows = run_longtail()
    else:
        levels = [0, 1, 2, 3] if args.paraphrase else [0]
        rows = run(splits, levels, args.seed)
    if args.against:
        with open(args.against, encoding="utf-8") as fh:
            earlier = {_row_key(r): r for r in json.load(fh)}
        print("\npaired difference vs %s (95%% interval; straddling zero means "
              "not shown)" % args.against)
        for row in rows:
            previous = earlier.get(_row_key(row))
            if previous is None and (row.get("axis") or row.get("slice")):
                # Rows predating the split key: match on axis/slice alone, then
                # let _require_comparable prove the sessions actually agree.
                legacy = dict(row)
                legacy.pop("split", None)
                previous = earlier.get(_row_key(legacy))
            # A summary row -- the long-tail percentile line -- has no per-session
            # detail to pair on. Checked by type rather than truthiness, because an
            # earlier file may carry a count under the same name.
            if not previous or not isinstance(previous.get("sessions"), list) \
                    or not isinstance(row.get("sessions"), list):
                continue
            _require_comparable(previous, row)
            low, mid, high, n = paired_interval(_by_id(previous["sessions"]),
                                                _by_id(row["sessions"]), args.seed)
            verdict = "significant" if low > 0 or high < 0 else "within noise"
            row["paired_delta"] = {"median": round(mid, 5),
                                   "ci95": [round(low, 5), round(high, 5)],
                                   "sessions": n, "verdict": verdict}
            print("  %-24s delta=%+.5f [%+.5f, %+.5f]  n=%-4d %s" % (
                _row_label(row), mid, low, high, n, verdict))

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
