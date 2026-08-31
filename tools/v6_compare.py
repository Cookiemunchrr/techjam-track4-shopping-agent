"""V6 decision runner: development comparison, gates, selection, and the
consumption ledger (V6 P2, Section 10).

    python3 -m tools.v6_compare --run off        # one mode's evidence (child)
    python3 -m tools.v6_compare --run aliases
    python3 -m tools.v6_compare --compare        # gates + selection
    python3 -m tools.v6_compare --consume s1 <candidate payload sha>   # Phase 4

Each --run executes in its own process with its own P_SHELF_TRANSFORM and
PYTHONHASHSEED=0, and writes an identifier-bearing sidecar
(analysis/_v6_cmp_<mode>.json, gitignored). --compare refuses to pair runs
whose ordered session sets or registered input hashes differ -- comparing the
intersection of two different runs is how mismatched evidence reads as paired
(T14). The committed artifacts (analysis/v6_dev_comparison.json,
analysis/v6_selection.json) are aggregate only.

Candidate B (mnn) was disqualified at V6-D8 (mapping audit) before any score run;
the comparison covers baseline vs Candidate A only, and the registered Holm
family accordingly contains a single comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

MODES = ("off", "aliases")     # mnn declined at V6-D8; see analysis/v6_blind_mapping_audits.json
SIDECAR = "analysis/_v6_cmp_{mode}.json"
LEDGER = Path("analysis/v6_validation_ledger.json")
MANIFEST = Path("analysis/v6_validation_manifest.json")
AUDITS = Path("analysis/v6_blind_mapping_audits.json")
BOOTSTRAP = 20_000
SEED = 0

CATALOG = "data/catalog.jsonl"
DEV = "analysis/dev.jsonl"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------- child
def run_mode(mode: str) -> None:
    """Collect one mode's dev evidence. Identifier-bearing; sidecar only."""
    from src.agent import Agent
    from tools import bench
    from tools.recall import SPLITS, summarise, trace
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

    ids, categories, products = catalog_index(CATALOG)
    samples = load_jsonl(SPLITS["dev"])

    base = Agent(CATALOG)
    rows = trace(Agent.sharing_index(base), samples, ids, categories, products,
                 ("category",))
    recall_rows = [{
        "sample_id": r["sample_id"], "scenario_type": r["scenario_type"],
        "turn1_in_pool": r["turn1_in_pool"], "ever_in_pool": r["ever_in_pool"],
        "first_pool_turn": r["first_pool_turn"], "best_rank": r["best_rank"],
        "turn1_pool": r["turn1_pool"], "turn1_route": r["turn1_route"],
    } for r in rows]

    def category_score(fixed_width: bool):
        from unittest import mock
        from tools.adversarial import Adversarial
        if fixed_width:
            with mock.patch.dict(os.environ, {"P_PROBE": "10", "P_WIDEN": "1"}):
                agent = Agent(CATALOG)
        else:
            agent = Agent(CATALOG)
        wrapped = Adversarial(Agent.sharing_index(agent), ("category",), SEED)
        result = evaluate(wrapped, samples, ids, categories, products)
        return result

    normal = category_score(False)
    fixed = category_score(True)

    env = dict(os.environ)
    shadow_out = Path(f"analysis/_v6_cmp_{mode}_shadow.json")
    subprocess.run([sys.executable, "-m", "tools.shadow", "--axis", "clean",
                    "--dataset", DEV, "--output", str(shadow_out)],
                   check=True, env=env)
    shadow = json.loads(shadow_out.read_text(encoding="utf-8"))

    audit_out = Path(f"analysis/_v6_cmp_{mode}_audit.json")
    subprocess.run([sys.executable, "-m", "tools.audit",
                    "--output", str(audit_out)], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    audit = json.loads(audit_out.read_text(encoding="utf-8"))

    adv_rows = bench.run_adversarial(SEED, splits=["dev"])

    def score_rows(result):
        return [{"sample_id": str(s["sample_id"]), "hit": s["hit"],
                 "reciprocal_rank": s["reciprocal_rank"],
                 "first_hit_turn": s["first_hit_turn"]}
                for s in (result.get("sessions") or [])]

    payload = {
        "mode": mode,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "hashes": {
            "catalog_sha256": _sha256(CATALOG),
            "dev_split_sha256": _sha256(SPLITS["dev"]),
            "evaluator_sha256": _sha256("evaluator/local_evaluator.py"),
            "agent_sha256": _sha256("src/agent.py"),
            "shelf_transform_sha256": _sha256("src/shelf_transform.py"),
            "transform_mode": os.environ.get("P_SHELF_TRANSFORM", "off"),
            "transform_payload_sha256": base.shelf_transform.payload_sha256,
        },
        "recall_summary": summarise(rows),
        "recall_rows": recall_rows,
        "category_normal": {"metrics": {"technical_score": normal["recommended_technical_score"],
                                        "hit_rate_at_10": normal["hit_rate_at_10"],
                                        "mrr": normal["mrr"], "mttc": normal["mttc"]},
                            "sessions": score_rows(normal)},
        "category_fixed_width": {"metrics": {"technical_score": fixed["recommended_technical_score"],
                                             "hit_rate_at_10": fixed["hit_rate_at_10"],
                                             "mrr": fixed["mrr"], "mttc": fixed["mttc"]},
                                 "sessions": score_rows(fixed)},
        "shadow_clean": shadow.get("clean"),
        "public_transcript_sha256": audit.get("public_transcript_sha256"),
        "public_score": (audit.get("public_score") or {}).get("recommended_technical_score"),
        "adversarial_dev": [{"axis": row["axis"], "technical_score": row["technical_score"],
                             "sessions": row["sessions"]} for row in adv_rows],
    }
    Path(SIDECAR.format(mode=mode)).write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {SIDECAR.format(mode=mode)}")


# ------------------------------------------------------------------ compare
def _score(rows) -> float:
    n = max(len(rows), 1)
    hits = sum(1 for r in rows if r["hit"])
    mrr = sum(r["reciprocal_rank"] for r in rows) / n
    mttc = sum(r["first_hit_turn"] if r["first_hit_turn"] is not None else 11
               for r in rows) / n
    return 0.50 * hits / n + 0.30 * mrr + 0.20 * max(0.0, min(1.0, (11.0 - mttc) / 10.0))


def paired_bootstrap(before, after, resamples: int = BOOTSTRAP, seed: int = SEED):
    """Paired aggregate-score bootstrap over whole sessions."""
    assert [r["sample_id"] for r in before] == [r["sample_id"] for r in after]
    rng = random.Random(seed)
    n = len(before)
    deltas = []
    for _ in range(resamples):
        draw = [rng.randrange(n) for _ in range(n)]
        deltas.append(_score([after[i] for i in draw]) - _score([before[i] for i in draw]))
    deltas.sort()
    return {"observed": round(_score(after) - _score(before), 5),
            "median": round(deltas[resamples // 2], 5),
            "ci95": [round(deltas[int(0.025 * resamples)], 5),
                     round(deltas[int(0.975 * resamples)], 5)],
            "one_sided_lb95": round(deltas[int(0.05 * resamples)], 5)}


def recall_bootstrap(before, after, field: str, resamples: int = BOOTSTRAP, seed: int = SEED):
    """Paired bootstrap of a per-session binary/rate recall field."""
    assert [r["sample_id"] for r in before] == [r["sample_id"] for r in after]
    rng = random.Random(seed)
    n = len(before)
    deltas = []
    for _ in range(resamples):
        draw = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(after[i][field] for i in draw) / n
                      - sum(before[i][field] for i in draw) / n)
    deltas.sort()
    improved = sum(1 for i in range(n) if after[i][field] > before[i][field])
    worsened = sum(1 for i in range(n) if after[i][field] < before[i][field])
    return {"observed": round(sum(after[i][field] for i in range(n)) / n
                              - sum(before[i][field] for i in range(n)) / n, 5),
            "median": round(deltas[resamples // 2], 5),
            "ci95": [round(deltas[int(0.025 * resamples)], 5),
                     round(deltas[int(0.975 * resamples)], 5)],
            "one_sided_lb95": round(deltas[int(0.05 * resamples)], 5),
            "improved": improved, "worsened": worsened,
            "tied": n - improved - worsened}


def mcnemar_exact(before, after, field: str) -> dict:
    """Exact two-sided McNemar p for a binary per-session field."""
    b = sum(1 for x, y in zip(before, after) if not x[field] and y[field])
    c = sum(1 for x, y in zip(before, after) if x[field] and not y[field])
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_two_sided": round(min(1.0, 2 * tail), 6)}


# The registered V6-D8 thresholds. Read from the audit rather than asserted: a
# gate that is hardcoded to pass is not a gate, and a re-run must not be able
# to inherit a verdict the outcome-blind audit does not actually support.
V6D8_MIN_OBSERVED = 0.98
V6D8_MIN_LOWER_BOUND = 0.95

_V6D8_ENTRY = {"aliases": "aliases_reviewed_by_mnn_engineer",
               "mnn": "mnn_reviewed_by_alias_engineer"}


def v6d8_gate(mode: str) -> dict:
    """V6-D8 evaluated from analysis/v6_blind_mapping_audits.json.

    Fails closed and loudly: a missing, malformed, or absent audit entry stops
    the comparison instead of reporting a pass nobody measured.
    """
    key = _V6D8_ENTRY.get(mode)
    if key is None:
        raise SystemExit(f"refusing to compare: no registered V6-D8 audit for mode {mode!r}")
    if not AUDITS.is_file():
        raise SystemExit(f"refusing to compare: {AUDITS} is missing; V6-D8 cannot be evaluated")
    entry = (json.loads(AUDITS.read_text(encoding="utf-8")) or {}).get(key)
    if not isinstance(entry, dict):
        raise SystemExit(f"refusing to compare: {AUDITS} has no {key!r} entry")
    try:
        observed = float(entry["precision"])
        lower = float(entry["wilson95"][0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SystemExit(f"refusing to compare: malformed V6-D8 audit entry {key!r}: {exc}")
    return {"precision": observed, "lb95": lower,
            "pass": observed >= V6D8_MIN_OBSERVED and lower >= V6D8_MIN_LOWER_BOUND,
            "reference": str(AUDITS)}


def compare() -> dict:
    runs = {}
    for mode in MODES:
        path = Path(SIDECAR.format(mode=mode))
        if not path.exists():
            raise SystemExit(f"missing {path}; run --run {mode} first")
        runs[mode] = json.loads(path.read_text(encoding="utf-8"))
    off, cand = runs["off"], runs["aliases"]

    # T14: refuse mismatched runs rather than pair the intersection.
    shared_keys = ("catalog_sha256", "dev_split_sha256", "evaluator_sha256",
                   "agent_sha256", "shelf_transform_sha256")
    mismatched = [k for k in shared_keys if off["hashes"][k] != cand["hashes"][k]]
    if mismatched:
        raise SystemExit(f"refusing to compare: mismatched hashes {mismatched}")
    for family in ("recall_rows",):
        ids_off = [r["sample_id"] for r in off[family]]
        ids_cand = [r["sample_id"] for r in cand[family]]
        if ids_off != ids_cand:
            raise SystemExit(f"refusing to compare: {family} session sets/order differ")
    for family in ("category_normal", "category_fixed_width"):
        ids_off = [r["sample_id"] for r in off[family]["sessions"]]
        ids_cand = [r["sample_id"] for r in cand[family]["sessions"]]
        if ids_off != ids_cand:
            raise SystemExit(f"refusing to compare: {family} session sets/order differ")

    b_recall, c_recall = off["recall_rows"], cand["recall_rows"]
    turn1 = recall_bootstrap(b_recall, c_recall, "turn1_in_pool")
    ever = recall_bootstrap(b_recall, c_recall, "ever_in_pool")
    mcnemar = mcnemar_exact(b_recall, c_recall, "turn1_in_pool")

    normal = paired_bootstrap(off["category_normal"]["sessions"],
                              cand["category_normal"]["sessions"])
    fixed = paired_bootstrap(off["category_fixed_width"]["sessions"],
                             cand["category_fixed_width"]["sessions"])

    # B5: scenario-stratified, descriptive only.
    scenarios = sorted({r["scenario_type"] for r in b_recall})
    strata = {}
    for scenario in scenarios:
        idx = [i for i, r in enumerate(b_recall) if r["scenario_type"] == scenario]
        b_sub = [b_recall[i] for i in idx]
        c_sub = [c_recall[i] for i in idx]
        strata[scenario] = {
            "n": len(idx),
            "turn1_recall_baseline": round(sum(r["turn1_in_pool"] for r in b_sub) / len(idx), 4),
            "turn1_recall_candidate": round(sum(r["turn1_in_pool"] for r in c_sub) / len(idx), 4),
        }

    adv_off = {row["axis"]: row for row in off["adversarial_dev"]}
    adv_cand = {row["axis"]: row for row in cand["adversarial_dev"]}
    axes = {}
    for axis in adv_off:
        if axis not in adv_cand:
            raise SystemExit(f"refusing to compare: axis {axis!r} missing in one run")
        axes[axis] = paired_bootstrap(adv_off[axis]["sessions"],
                                      adv_cand[axis]["sessions"])

    b_sum, c_sum = off["recall_summary"], cand["recall_summary"]
    lost_rank_b = round(b_sum["lost_to_ranking"] * b_sum["sessions"])
    lost_rank_c = round(c_sum["lost_to_ranking"] * c_sum["sessions"])

    # V6-D1: on sessions the baseline resolved exactly, the candidate must have
    # zero transform lookups and an identical ordered pool digest. V6-D2: whenever
    # the candidate's transform fired, its pre-append pool digest must equal the
    # baseline's pre-filter pool digest (the prefix proof); when it did not
    # fire, the full pre-filter digests must match.
    d1_holds = True
    d2_holds = True
    for b_row, c_row in zip(b_recall, c_recall):
        b_route, c_route = b_row["turn1_route"], c_row["turn1_route"]
        if b_route.get("exact"):
            if c_route.get("transform_lookups") != 0 \
                    or b_route.get("prefilter_pool_sha256") != c_route.get("prefilter_pool_sha256"):
                d1_holds = False
        if c_route.get("transform_activations"):
            if c_route.get("baseline_prefix_sha256") != b_route.get("prefilter_pool_sha256"):
                d2_holds = False
        elif b_route.get("prefilter_pool_sha256") != c_route.get("prefilter_pool_sha256"):
            d2_holds = False

    gates = {
        "V6-D1_exact_route_digests_identical": (
            off["public_transcript_sha256"] == cand["public_transcript_sha256"]
            and d1_holds),
        "V6-D2_no_baseline_pool_member_lost": d2_holds,
        "V6-D3_turn1_recall": {"delta": turn1["observed"], "lb95": turn1["one_sided_lb95"],
                               "pass": turn1["observed"] >= 0.030 and turn1["one_sided_lb95"] > 0},
        "V6-D4_ever_recall": {"value": c_sum["pool_recall_ever"],
                              "pass": c_sum["pool_recall_ever"] >= 0.970},
        "V6-D5_lost_to_ranking": {"baseline": lost_rank_b, "candidate": lost_rank_c,
                                  "pass": lost_rank_c <= lost_rank_b},
        "V6-D6_fixed_width_score": {"delta": fixed["observed"], "lb95": fixed["one_sided_lb95"],
                                    "pass": fixed["one_sided_lb95"] > 0},
        "V6-D7_normal_width_score": {"delta": normal["observed"], "lb95": normal["ci95"][0],
                                     "pass": normal["observed"] > 0 and normal["ci95"][0] >= -0.005},
        "V6-D8_mapping_audit": v6d8_gate("aliases"),
        "N1_public_byte_identical": off["public_transcript_sha256"] == cand["public_transcript_sha256"],
        "N4_fallback_frequency": {"baseline": b_sum["turn1_fallback_rate"],
                                  "candidate": c_sum["turn1_fallback_rate"],
                                  "pass": c_sum["turn1_fallback_rate"] <= b_sum["turn1_fallback_rate"]},
        "N5_pool_p95": {"baseline": b_sum["turn1_pool_p95"], "candidate": c_sum["turn1_pool_p95"],
                        "pass": c_sum["turn1_pool_p95"] <= 1.25 * b_sum["turn1_pool_p95"]},
        "N6_shadow_clean": {"baseline": (off["shadow_clean"] or {}).get("shadow_score"),
                            "candidate": (cand["shadow_clean"] or {}).get("shadow_score")},
    }
    if gates["N6_shadow_clean"]["baseline"] is not None:
        delta = (gates["N6_shadow_clean"]["candidate"] or 0) - gates["N6_shadow_clean"]["baseline"]
        gates["N6_shadow_clean"]["delta"] = round(delta, 5)
        gates["N6_shadow_clean"]["pass"] = delta >= -0.005

    axes_lower = {axis: axes[axis]["ci95"][0] for axis in axes}
    gates["N2_axes"] = {axis: {"lb95": lb, "pass": lb >= -0.005}
                        for axis, lb in axes_lower.items()
                        if axis in ("natural", "scaffold", "constraint",
                                    "granularity=1", "granularity=3")}
    gates["N3_combined_axis"] = {"lb95": axes_lower.get("all paraphrase axes"),
                                 "pass": axes_lower.get("all paraphrase axes", -1) >= -0.010}

    report = {
        "schema": "techjam-v6-dev-comparison-v1",
        "scope": "V6 Phase 3 development comparison: baseline vs aliases, dev split, "
                 "category axis primary. Candidate B (mnn) declined at V6-D8; single-comparison "
                 "Holm family.",
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "unit": "session"},
        "hashes": off["hashes"],
        "recall": {
            "baseline": b_sum, "candidate": c_sum,
            "turn1_delta": turn1, "ever_delta": ever, "mcnemar_turn1": mcnemar,
        },
        "scores": {
            "category_normal_width": {"baseline": off["category_normal"]["metrics"],
                                      "candidate": cand["category_normal"]["metrics"],
                                      "paired": normal},
            "category_fixed_width_10": {"baseline": off["category_fixed_width"]["metrics"],
                                        "candidate": cand["category_fixed_width"]["metrics"],
                                        "paired": fixed},
        },
        "scenario_strata_descriptive_only": strata,
        "adversarial_axes_dev": {axis: {"baseline": adv_off[axis]["technical_score"],
                                        "candidate": adv_cand[axis]["technical_score"],
                                        "paired": axes[axis]} for axis in axes},
        "shadow_clean": gates["N6_shadow_clean"],
        "public_transcript": {"baseline": off["public_transcript_sha256"],
                              "candidate": cand["public_transcript_sha256"],
                              "identical": gates["N1_public_byte_identical"],
                              "baseline_score": off["public_score"],
                              "candidate_score": cand["public_score"]},
        "gates": gates,
    }
    Path("analysis/v6_dev_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote analysis/v6_dev_comparison.json")
    return report


# ------------------------------------------------------------------- ledger
def consume(set_name: str, candidate_hash: str) -> None:
    """Append a consumption record; refuse a second read of the same set."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    set_hash = manifest[set_name]["body_sha256" if set_name == "s1" else "phrases_sha256"]
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    for record in ledger["consumed"]:
        if record["set_hash"] == set_hash:
            raise SystemExit(
                f"refusing: {set_name} ({set_hash[:12]}...) was already consumed by "
                f"candidate {record['candidate_hash'][:12]}...; a confirmation set "
                "is read exactly once (STOP-8)")
    ledger["consumed"].append({
        "set": set_name, "set_hash": set_hash, "candidate_hash": candidate_hash,
    })
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(f"consumed {set_name} for candidate {candidate_hash[:12]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 decision runner")
    parser.add_argument("--run", choices=MODES, default="")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--consume", nargs=2, metavar=("SET", "CANDIDATE_HASH"))
    args = parser.parse_args()
    if args.run:
        run_mode(args.run)
    elif args.compare:
        compare()
    elif args.consume:
        consume(*args.consume)
    else:
        parser.error("one of --run/--compare/--consume is required")


if __name__ == "__main__":
    main()
