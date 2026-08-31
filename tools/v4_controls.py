"""Fresh, identifier-safe end-to-end controls for the V4-0 foundation.

The ranking snapshot report is intentionally fixed-state. This companion command
runs the behavior-level controls that snapshots cannot represent: official scoring
on dev/holdout/full, width pinned to ten, the shadow evaluator, adversarial axes,
the protected public transcript, and the separately measured route-resource report.

Raw evaluator outputs live only in a temporary directory because they contain public
sample identifiers. The committed aggregate contains no sessions or targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.rerank_provenance import (canonical_json, effective_runtime_config,
                                     sha256_file)
from tools.resource_probe import (BUDGETED, DEFAULT_SAMPLES, DEFAULT_WARMUPS,
                                  GATE_CONFIDENCE, LATENCY_SCOPE, MESSAGES,
                                  MIN_GATE_SAMPLES, P99_BUDGET_MS,
                                  P99_CONVENTION, REPORT_SCHEMA, ROUTES,
                                  TARGET_VIOLATION_RATE, WORKLOAD_SCHEMA,
                                  _message_sha256, _validate_report, acceptable,
                                  resource_source_sha256)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCE = REPO / "analysis" / "resource_probe_v4_0.json"
DEFAULT_OUTPUT = REPO / "analysis" / "v4_end_to_end_controls.json"
EXPECTED_PUBLIC_TRANSCRIPT = (
    "7c30023e3b8f951d35f8a449a066cfff45f8995d8d9994142e0e5929f8958d04"
)
EXPECTED_PUBLIC_SCORE = 0.916125
EXPECTED_BASELINE_CONTROLS_SHA256 = (
    "cfe866ef4d8556e5d85f2036672ae218047c0665cf68fdf0e757b011205c7e98"
)
RUNTIME_KEYS = (
    "P_ASK", "P_FUSE", "P_MARGIN", "P_OVERLOAD", "P_PROBE", "P_PRUNE",
    "P_SOFT", "P_UNSURE", "P_WIDEN", "W_BUDGET", "W_FACET", "W_PHRASE",
    "W_POP", "W_TXT",
)
CONTROL_SOURCE_PATHS = tuple(
    [
        "evaluator/local_evaluator.py",
        "tools/adversarial.py",
        "tools/audit.py",
        "tools/bench.py",
        "tools/paraphrase.py",
        "tools/rerank_data.py",
        "tools/rerank_provenance.py",
        "tools/resource_probe.py",
        "tools/shadow.py",
        "tools/snapshot_mrr.py",
        "tools/v4_controls.py",
    ]
    + [
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "src").rglob("*.py"))
    ]
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _environment(*, fixed_width: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    for key in RUNTIME_KEYS:
        env.pop(key, None)
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(REPO))
    if fixed_width:
        env.update(P_PROBE="10", P_WIDEN="1")
    return env


def current_control_inputs() -> dict:
    """Exact current identities needed to reproduce every control family."""
    return {
        "catalog_sha256": sha256_file(REPO / "data" / "catalog.jsonl"),
        "public_split_sha256": sha256_file(REPO / "data" / "public_set.jsonl"),
        "dev_split_sha256": sha256_file(REPO / "analysis" / "dev.jsonl"),
        "holdout_split_sha256": sha256_file(REPO / "analysis" / "holdout.jsonl"),
        "serving_model_sha256": sha256_file(REPO / "analysis" / "reranker.json"),
        "control_sources": {
            name: sha256_file(REPO / name) for name in CONTROL_SOURCE_PATHS
        },
    }


def _run_json(label: str, command: list[str], output: Path, *,
              env: dict[str, str], timeout: float = 1200.0):
    done = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True,
                          text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(
            f"{label} failed ({done.returncode}):\n"
            f"stdout={done.stdout}\nstderr={done.stderr}"
        )
    try:
        with output.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} did not write valid JSON: {output}") from exc


def compact_bench(rows) -> list[dict]:
    """Strip identifiers and wall-clock timing from a bench result."""
    allowed = (
        "axis", "ci95", "hit_rate_at_10", "mrr", "mttc", "paraphrase_level",
        "scenario_metrics", "slice", "split", "technical_score",
    )
    compact = []
    for row in rows:
        item = {key: row[key] for key in allowed if key in row}
        sessions = row.get("sessions")
        if isinstance(sessions, list):
            item["sample_count"] = len(sessions)
        elif isinstance(row.get("session_count"), int):
            item["sample_count"] = row["session_count"]
        compact.append(item)
    return compact


def compact_audit(payload: dict) -> dict:
    """Keep deterministic correctness fields; route timing has its own protocol."""
    return {key: payload.get(key) for key in (
        "python_hash_seed", "public_transcript_sha256", "evaluator_sha256",
        "forbidden_src_imports", "swallowed_turn_failures", "public_score",
    )}


def _finite_metric(row: object, *, shadow: bool = False) -> bool:
    if not isinstance(row, dict):
        return False
    names = (("shadow_score", "retrieval_hit_rate_at_10", "retrieval_mrr",
              "retrieval_mttc") if shadow else
             ("technical_score", "hit_rate_at_10", "mrr", "mttc"))
    return all(
        isinstance(row.get(name), (int, float))
        and not isinstance(row.get(name), bool)
        and math.isfinite(float(row[name]))
        for name in names
    )


def validate_control_completeness(official, fixed_width, adversarial,
                                  shadows) -> None:
    """Fail closed if a command silently omitted a split, axis, or session."""
    expected_splits = {"dev": 100, "holdout": 100, "full": 200}
    expected_scenarios = {"boundary", "browsing", "buying", "intent_override"}
    expected_scenario_counts = {
        "dev": {"boundary": 3, "browsing": 44, "buying": 35,
                "intent_override": 18},
        "holdout": {"boundary": 7, "browsing": 36, "buying": 45,
                    "intent_override": 12},
        "full": {"boundary": 10, "browsing": 80, "buying": 80,
                 "intent_override": 30},
    }

    def valid_scenarios(scenarios, split):
        if not isinstance(scenarios, dict) or set(scenarios) != expected_scenarios:
            return False
        for name, item in scenarios.items():
            if not isinstance(item, dict) or any(
                    not isinstance(item.get(metric), (int, float))
                    or isinstance(item.get(metric), bool)
                    or not math.isfinite(float(item[metric]))
                    for metric in ("hit_rate_at_10", "mrr", "mttc")) \
                    or isinstance(item.get("sample_count"), bool) \
                    or not isinstance(item.get("sample_count"), int) \
                    or item["sample_count"] < 0 \
                    or not 0.0 <= float(item["hit_rate_at_10"]) <= 1.0 \
                    or not 0.0 <= float(item["mrr"]) <= 1.0 \
                    or not 0.0 <= float(item["mttc"]) <= 11.0:
                return False
        return {
            name: item["sample_count"] for name, item in scenarios.items()
        } == expected_scenario_counts[split]

    for label, rows in (("official", official), ("fixed_width", fixed_width)):
        if not isinstance(rows, list):
            raise ValueError(f"{label} controls are not a list")
        indexed = {
            (row.get("split"), row.get("paraphrase_level")): row
            for row in rows if isinstance(row, dict)
        }
        if len(rows) != len(expected_splits) or len(indexed) != len(rows):
            raise ValueError(f"{label} controls contain duplicate or extra rows")
        if set(indexed) != {(split, 0) for split in expected_splits}:
            raise ValueError(f"{label} controls lack an exact dev/holdout/full matrix")
        for (split, _), row in indexed.items():
            if row.get("sample_count") != expected_splits[split] \
                    or not _finite_metric(row):
                raise ValueError(f"{label} {split} metrics/count are invalid")
            scenarios = row.get("scenario_metrics")
            if not valid_scenarios(scenarios, split):
                raise ValueError(f"{label} {split} scenario matrix is incomplete")

    expected_axes = {
        "control", "category", "natural", "scaffold", "constraint",
        "all paraphrase axes", "granularity=1", "granularity=3",
    }
    if not isinstance(adversarial, list):
        raise ValueError("adversarial controls are not a list")
    axes = {row.get("axis"): row for row in adversarial if isinstance(row, dict)}
    if len(adversarial) != len(expected_axes) or len(axes) != len(adversarial):
        raise ValueError("adversarial controls contain duplicate or extra rows")
    if set(axes) != expected_axes:
        raise ValueError("adversarial controls lack the exact frozen axis matrix")
    if any(row.get("sample_count") != 200 or not _finite_metric(row)
           for row in axes.values()):
        raise ValueError("adversarial controls have invalid metrics/counts")
    if any(not valid_scenarios(row.get("scenario_metrics"), "full")
           for row in axes.values()):
        raise ValueError("adversarial scenario matrices are incomplete")

    if not isinstance(shadows, dict):
        raise ValueError("shadow controls are missing")
    for split, count in expected_splits.items():
        row = shadows.get(split)
        if not _finite_metric(row, shadow=True) or row.get("sample_count") != count:
            raise ValueError(f"shadow {split} metrics/count are invalid")
    operational = shadows.get("full_operational_axes")
    expected_shadow_axes = {"clean", "fresh_agent", "shuffle_seed1", "shuffle_seed2"}
    if not isinstance(operational, dict) or set(operational) != expected_shadow_axes:
        raise ValueError("shadow operational axis matrix is incomplete")
    if any(not _finite_metric(row, shadow=True) or row.get("sample_count") != 200
           for row in operational.values()):
        raise ValueError("shadow operational axes have invalid metrics/counts")
    if shadows["full"] != operational["clean"]:
        raise ValueError("shadow full control disagrees with its clean operational axis")


def validate_v4_resource(resource: dict) -> None:
    _validate_report(resource, "V4-0")
    if set((resource.get("routes") or {})) != set(ROUTES):
        raise ValueError("V4-0 resource report must contain exact, hedged, and fallback")
    protocol = resource["protocol"]
    expected = {
        "fresh_process_per_route": True,
        "python_hash_seed": "0",
        "catalog": "data/catalog.jsonl",
        "catalog_sha256": sha256_file(REPO / "data" / "catalog.jsonl"),
        "serving_model_sha256": sha256_file(REPO / "analysis" / "reranker.json"),
        "source_sha256": resource_source_sha256(),
        "samples_per_route": DEFAULT_SAMPLES,
        "warmups_per_route": DEFAULT_WARMUPS,
        "latency_scope": LATENCY_SCOPE,
        "p99_convention": P99_CONVENTION,
        "workload_schema": WORKLOAD_SCHEMA,
        "budget_gate": "fixed_workload_zero_exceedance_canary",
        "budget_ms": P99_BUDGET_MS,
        "gate_confidence": GATE_CONFIDENCE,
        "target_violation_rate": TARGET_VIOLATION_RATE,
        "minimum_gate_samples": MIN_GATE_SAMPLES,
        "effective_runtime": effective_runtime_config(_environment()),
        "iid_assumption_verified": False,
        "serial_samples_per_fresh_process": True,
        "workloads_per_route": 1,
        "route_execution_order": list(ROUTES),
        "route_workload_sha256": {
            route: _message_sha256(route) for route in ROUTES
        },
    }
    mismatches = [name for name, value in expected.items()
                  if protocol.get(name) != value]
    if mismatches:
        raise ValueError(
            "V4-0 resource report is stale/non-default: " + ", ".join(mismatches)
        )
    if resource.get("schema") != REPORT_SCHEMA or resource.get("comparison") is not None:
        raise ValueError("V4-0 resource report is not a fresh absolute baseline")
    if any(route not in BUDGETED and route != "fallback" for route in resource["routes"]):
        raise ValueError("V4-0 resource report contains an unknown gate family")
    if not acceptable(resource):
        raise ValueError("V4-0 resource canary does not pass semantic gates")


def validate_controls_report(report: dict) -> None:
    """Reject stale, forged, incomplete, or red end-to-end control artifacts."""
    if report.get("schema") != "techjam-v4-end-to-end-controls-v1":
        raise ValueError("unsupported V4-0 controls schema")
    if report.get("inputs") != current_control_inputs():
        raise ValueError("V4-0 control inputs/source closure are stale")
    controls = report.get("controls")
    if not isinstance(controls, dict) or set(controls) != {
            "official", "fixed_width_10", "shadow", "adversarial"}:
        raise ValueError("V4-0 control families are incomplete")
    validate_control_completeness(
        controls["official"], controls["fixed_width_10"],
        controls["adversarial"], controls["shadow"],
    )
    if report.get("controls_sha256") != _sha256_json(controls) \
            or report.get("controls_sha256") != EXPECTED_BASELINE_CONTROLS_SHA256:
        raise ValueError("V4-0 controls digest does not match its body")
    protected = report.get("protected_baseline")
    if not isinstance(protected, dict):
        raise ValueError("V4-0 protected baseline is missing")
    score = ((protected.get("public_score") or {}).get(
        "recommended_technical_score"))
    protected_ok = (
        protected.get("python_hash_seed") == "0"
        and protected.get("public_transcript_sha256") == EXPECTED_PUBLIC_TRANSCRIPT
        and protected.get("expected_public_transcript_sha256")
        == EXPECTED_PUBLIC_TRANSCRIPT
        and protected.get("transcript_match") is True
        and isinstance(score, (int, float)) and not isinstance(score, bool)
        and abs(float(score) - EXPECTED_PUBLIC_SCORE) <= 1e-12
        and protected.get("expected_public_score") == EXPECTED_PUBLIC_SCORE
        and protected.get("score_match") is True
        and protected.get("forbidden_src_imports") == []
        and protected.get("swallowed_turn_failures") == 0
        and protected.get("evaluator_sha256")
        == report["inputs"]["control_sources"]["evaluator/local_evaluator.py"]
    )
    if not protected_ok:
        raise ValueError("V4-0 protected behavior evidence is inconsistent")
    resource = report.get("resource")
    if not isinstance(resource, dict):
        raise ValueError("V4-0 embedded resource report is missing")
    validate_v4_resource(resource)
    gates = report.get("gates")
    expected_gates = {
        "protected_behavior_parity": True,
        "end_to_end_matrix_complete": True,
        "resource_canary_passed": True,
        "all_v4_0_controls_passed": True,
    }
    # `1 == True` in Python, so dict equality alone accepts a non-boolean gate
    # value. These gates are fail-closed provenance evidence: require real bools.
    if gates != expected_gates or any(
            not isinstance(gates[name], bool) for name in expected_gates):
        raise ValueError("V4-0 control gates are missing, non-boolean, or red")


def _commands(root: Path) -> dict[str, tuple[list[str], Path, dict[str, str]]]:
    normal = _environment()
    fixed = _environment(fixed_width=True)
    report = {
        "audit": (
            [sys.executable, "-m", "tools.audit", "--output", str(root / "audit.json")],
            root / "audit.json", normal,
        ),
        "official": (
            [sys.executable, "-m", "tools.bench", "--splits", "dev,holdout,full",
             "--output", str(root / "official.json")],
            root / "official.json", normal,
        ),
        "fixed_width": (
            [sys.executable, "-m", "tools.bench", "--splits", "dev,holdout,full",
             "--output", str(root / "fixed_width.json")],
            root / "fixed_width.json", fixed,
        ),
        "adversarial": (
            [sys.executable, "-m", "tools.bench", "--adversarial",
             "--output", str(root / "adversarial.json")],
            root / "adversarial.json", normal,
        ),
        "shadow_full": (
            [sys.executable, "-m", "tools.shadow", "--axis", "all",
             "--output", str(root / "shadow_full.json")],
            root / "shadow_full.json", normal,
        ),
        "shadow_dev": (
            [sys.executable, "-m", "tools.shadow", "--axis", "clean",
             "--dataset", "analysis/dev.jsonl",
             "--output", str(root / "shadow_dev.json")],
            root / "shadow_dev.json", normal,
        ),
        "shadow_holdout": (
            [sys.executable, "-m", "tools.shadow", "--axis", "clean",
             "--dataset", "analysis/holdout.jsonl",
             "--output", str(root / "shadow_holdout.json")],
            root / "shadow_holdout.json", normal,
        ),
    }
    return report


def build_controls(resource_path: str | Path) -> dict:
    resource_file = Path(resource_path)
    with resource_file.open(encoding="utf-8") as handle:
        resource = json.load(handle)
    validate_v4_resource(resource)

    with tempfile.TemporaryDirectory(prefix="techjam-v4-controls-") as directory:
        root = Path(directory)
        raw = {
            label: _run_json(label, command, output, env=env)
            for label, (command, output, env) in _commands(root).items()
        }

    audit = compact_audit(raw["audit"])
    official = compact_bench(raw["official"])
    fixed_width = compact_bench(raw["fixed_width"])
    adversarial = compact_bench(raw["adversarial"])
    shadows = {
        "dev": raw["shadow_dev"].get("clean"),
        "holdout": raw["shadow_holdout"].get("clean"),
        "full": raw["shadow_full"].get("clean"),
        "full_operational_axes": raw["shadow_full"],
    }
    validate_control_completeness(official, fixed_width, adversarial, shadows)
    trace_match = audit.get("public_transcript_sha256") == EXPECTED_PUBLIC_TRANSCRIPT
    score = ((audit.get("public_score") or {}).get("recommended_technical_score"))
    score_match = isinstance(score, (int, float)) and abs(
        float(score) - EXPECTED_PUBLIC_SCORE) <= 1e-12
    compact = {
        "official": official,
        "fixed_width_10": fixed_width,
        "shadow": shadows,
        "adversarial": adversarial,
    }
    report = {
        "schema": "techjam-v4-end-to-end-controls-v1",
        "scope": "V4-0 controls; shipped asset only; no candidate model",
        "inputs": current_control_inputs(),
        "protected_baseline": {
            **audit,
            "expected_public_transcript_sha256": EXPECTED_PUBLIC_TRANSCRIPT,
            "transcript_match": trace_match,
            "expected_public_score": EXPECTED_PUBLIC_SCORE,
            "score_match": score_match,
        },
        "controls": compact,
        "controls_sha256": _sha256_json(compact),
        "resource": resource,
        "gates": {
            "protected_behavior_parity": trace_match and score_match,
            "end_to_end_matrix_complete": True,
            "resource_canary_passed": acceptable(resource),
            "all_v4_0_controls_passed": trace_match and score_match
            and acceptable(resource),
        },
        "limitations": [
            "Fixed-width, shadow, adversarial, and slice results are frozen baseline "
            "controls, not candidate deltas or candidate acceptance thresholds.",
            "The resource canary uses one fixed workload per route and serial turns; "
            "it is not an IID route-wide SLA estimate.",
            "Field-level title/structured/description provenance is unavailable in "
            "the current 15-feature snapshots and is not reconstructed post hoc.",
        ],
    }
    # Fail closed: the report does not exist for callers until it has survived
    # its own validation. Returning before this call is what let a malformed or
    # red report read as green.
    validate_controls_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fresh V4-0 end-to-end controls")
    parser.add_argument("--resource", type=Path, default=DEFAULT_RESOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_controls(args.resource)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0 if report["gates"]["all_v4_0_controls_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
