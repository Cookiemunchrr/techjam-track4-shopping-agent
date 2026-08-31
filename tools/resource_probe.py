"""Fresh-process latency and memory probes for the three serving routes.

V4 needs comparable resource evidence, not timings inherited from different
processes and workloads.  This tool launches one child process per route, fixes the
messages and hash seed, excludes construction from turn latency, and reports the
same median/p95/p99/max and peak-RSS fields for each route.

Exact and hedged turns retain the repository's 50 ms target. The controlled
engineering canary requires at least 299 serial turns and no latency at or above
50 ms. The accompanying one-sided zero-violation bound is reported only as an IID
sensitivity calculation: one fixed message and serial turns through one process do
not establish independent samples from a route-wide workload distribution. The
existing full-catalog fallback is deliberately *not* relabelled as a 50 ms path.
Its first V4 run is an absolute baseline. ``--baseline`` reports a descriptive
delta, but deliberately exits non-zero because no non-inferiority margin or
independent-block comparison has yet been registered.

Typical use::

    python3 -m tools.resource_probe --output analysis/resource_probe_v4_0.json
    python3 -m tools.resource_probe --include-fallback
    python3 -m tools.resource_probe --include-fallback \
        --baseline analysis/resource_probe_v4_0.json

The fallback is opt-in because a gate-capable run scores all 50,000 products 299
times and is intentionally slow.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO / "data" / "catalog.jsonl"
ROUTES = ("exact", "hedged", "fallback")
BUDGETED = frozenset(("exact", "hedged"))
P99_BUDGET_MS = 50.0
GATE_CONFIDENCE = 0.95
TARGET_VIOLATION_RATE = 0.01
DEFAULT_SAMPLES = 299
DEFAULT_WARMUPS = 3
MIN_GATE_SAMPLES = math.floor(
    math.log(1.0 - GATE_CONFIDENCE) / math.log(1.0 - TARGET_VIOLATION_RATE)
) + 1
REPORT_SCHEMA = 3
WORKLOAD_SCHEMA = 2
LATENCY_SCOPE = "Agent.respond after reset; Agent construction excluded"
P99_CONVENTION = "nearest-rank"
COMPATIBILITY_FIELDS = (
    "fresh_process_per_route",
    "python_hash_seed",
    "platform",
    "machine",
    "platform_release",
    "processor",
    "processor_identity_quality",
    "logical_cpu_count",
    "python_version",
    "catalog_sha256",
    "serving_model_sha256",
    "source_sha256",
    "samples_per_route",
    "warmups_per_route",
    "latency_scope",
    "p99_convention",
    "workload_schema",
    "budget_gate",
    "budget_ms",
    "gate_confidence",
    "target_violation_rate",
    "minimum_gate_samples",
    "effective_runtime",
    "iid_assumption_verified",
    "serial_samples_per_fresh_process",
    "workloads_per_route",
    "route_execution_order",
)

RESOURCE_SOURCE_PATHS = tuple(
    ["tools/resource_probe.py", "tools/rerank_provenance.py"]
    + [
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "src").rglob("*.py"))
    ]
)

# Fixed, target-independent route workloads.  The hedged message is the one already
# guarded in tests/test_pillar1_routing.py; the fallback token is deliberately outside
# both catalog vocabulary and shelf-name trigrams.
MESSAGES = {
    "exact": ("I'm looking for Tops & Tees T-Shirts. A key requirement is: "
              "100% Cotton."),
    "hedged": ("I'm looking for something to keep my neck warm. A key requirement "
                "is: merino wool, machine washable, charcoal."),
    "fallback": "qzxvplmnrtyk",
}


def file_sha256(path: str | Path) -> str:
    """Streaming content identity for the catalog used by every child."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _message_sha256(route: str) -> str:
    return hashlib.sha256(MESSAGES[route].encode("utf-8")).hexdigest()


def resource_source_sha256() -> dict[str, str]:
    """Current build/runtime source closure for a resource measurement."""
    return {
        relative: file_sha256(REPO / relative)
        for relative in RESOURCE_SOURCE_PATHS
    }


def _logical_path(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(REPO).as_posix()
    except ValueError:
        return str(candidate)


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def _processor_label() -> str:
    """Best-effort CPU label without spawning a command during measurement."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine()


def percentile(values, quantile: float) -> float:
    """Nearest-rank percentile, pinned so reports cannot drift by convention."""
    if not values:
        raise ValueError("a percentile needs at least one sample")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def zero_violation_u95(samples: int) -> float:
    """One-sided 95% binomial upper bound after observing zero violations."""
    if samples < 1:
        raise ValueError("the zero-violation bound needs at least one sample")
    alpha = 1.0 - GATE_CONFIDENCE
    return 1.0 - alpha ** (1.0 / samples)


def summarize(route: str, elapsed_ms, *, build_seconds: float,
              peak_rss_mb: float, warmups: int) -> dict:
    """Canonical result shape shared by real children and contract fixtures."""
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")
    samples = [float(value) for value in elapsed_ms]
    if not samples:
        raise ValueError("resource probe recorded no turns")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("resource latency samples must be finite and non-negative")
    if not math.isfinite(float(build_seconds)) or float(build_seconds) < 0.0:
        raise ValueError("resource build time must be finite and non-negative")
    if not math.isfinite(float(peak_rss_mb)) or float(peak_rss_mb) < 0.0:
        raise ValueError("resource peak RSS must be finite and non-negative")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("resource warmups must be a non-negative integer")
    ordered = sorted(samples)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2 else
              (ordered[middle - 1] + ordered[middle]) / 2.0)
    latency = {
        "median": round(median, 3),
        "p95": round(percentile(ordered, 0.95), 3),
        "p99": round(percentile(ordered, 0.99), 3),
        "max": round(max(ordered), 3),
    }
    if route in BUDGETED:
        violations = sum(value >= P99_BUDGET_MS for value in samples)
        u95 = zero_violation_u95(len(samples)) if violations == 0 else None
        enforced = len(samples) >= MIN_GATE_SAMPLES
        gate = {
            "kind": "fixed_workload_zero_exceedance_canary",
            "limit_ms": P99_BUDGET_MS,
            "iid_sensitivity_confidence": GATE_CONFIDENCE,
            "target_violation_rate": TARGET_VIOLATION_RATE,
            "minimum_samples": MIN_GATE_SAMPLES,
            "violations": violations,
            "observed_violation_rate": round(violations / len(samples), 6),
            "iid_zero_violation_u95_sensitivity": (
                None if u95 is None else round(u95, 6)
            ),
            "iid_assumption_verified": False,
            "statistical_claim": "none; fixed-message serial engineering canary",
            "enforced": enforced,
            "passed": (violations == 0 and u95 is not None and
                       u95 < TARGET_VIOLATION_RATE) if enforced else None,
        }
        if not enforced:
            gate["note"] = ("route-shape canary only; too few samples for the "
                            "predeclared 299-turn zero-exceedance canary")
    else:
        gate = {
            "kind": "absolute_descriptive_baseline",
            "limit_ms": None,
            "minimum_samples": MIN_GATE_SAMPLES,
            "enforced": False,
            "passed": None,
            "note": ("absolute baseline only; a same-protocol point delta is "
                     "descriptive until a margin and independent blocks are "
                     "predeclared; never apply the 50 ms exact/hedged gate"),
        }
    return {
        "route": route,
        "samples": len(samples),
        "warmups": int(warmups),
        "build_seconds": round(float(build_seconds), 3),
        "peak_rss_mb": round(float(peak_rss_mb), 1),
        "latency_ms": latency,
        "gate": gate,
    }


def _route_shape(agent, route: str) -> dict:
    """Prove the fixed workload still reaches the route its label claims."""
    from src.routing import (category_key, detect_intent, exact_bucket,
                             route_detail)
    from src.text import split_clauses

    message = MESSAGES[route]
    clauses = split_clauses(message)
    key = category_key(clauses)
    intent = detect_intent(message, clauses)
    pool, primary, hedge = route_detail(agent.catalog, agent.semantic, key, intent)
    exact = exact_bucket(agent.catalog, key) is not None
    shape = {
        "exact": exact,
        "hedged": bool(hedge),
        "pool_size": len(pool),
        "catalog_size": agent.catalog.size,
        "primary_size": len(primary),
    }
    valid = ((route == "exact" and exact and not hedge) or
             (route == "hedged" and not exact and bool(hedge)) or
             (route == "fallback" and not exact and not hedge and
              len(pool) == agent.catalog.size and not primary))
    if not valid:
        raise RuntimeError(f"{route} workload no longer reaches its declared path: {shape}")
    return shape


def _response_sha256(response: object) -> str:
    """Identifier-free proof that the fixed workload stayed deterministic."""
    try:
        payload = json.dumps(response, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"resource response is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def measure_child(route: str, catalog: str | Path, samples: int,
                  warmups: int) -> dict:
    """Measure one route in the current (already fresh) process."""
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative")
    from src.agent import Agent

    started = time.perf_counter()
    agent = Agent(catalog)
    # Integer-only diagnostics preserve the production allocation pattern. Retaining
    # every pool frozenset would add up to 50,000 ids to each of 256 live sessions.
    agent.trace_pool_size = True
    build_seconds = time.perf_counter() - started
    shape = _route_shape(agent, route)
    message = MESSAGES[route]

    response_digests: set[str] = set()
    scored_pool_sizes: set[int] = set()

    def observe(session: str, response: object) -> None:
        if not isinstance(response, dict) or not response.get("recommendations"):
            raise RuntimeError(f"{route} probe produced no recommendations")
        response_digests.add(_response_sha256(response))
        scored_pool_sizes.add(agent.candidate_pool_size(session))

    for index in range(warmups):
        session = f"resource_{route}_warm_{index}"
        agent.reset(session, {})
        response = agent.respond(session, message, 1, 10)
        observe(session, response)

    elapsed = []
    for index in range(samples):
        session = f"resource_{route}_{index}"
        agent.reset(session, {})
        turn_started = time.perf_counter_ns()
        response = agent.respond(session, message, 1, 10)
        elapsed.append((time.perf_counter_ns() - turn_started) / 1_000_000.0)
        observe(session, response)
    if agent.failures:
        raise RuntimeError(f"{route} probe swallowed {agent.failures} turn failures")
    if len(response_digests) != 1:
        raise RuntimeError(
            f"{route} fixed workload produced {len(response_digests)} responses")
    if len(scored_pool_sizes) != 1 or next(iter(scored_pool_sizes), 0) <= 0:
        raise RuntimeError(
            f"{route} fixed workload changed scored pool: {sorted(scored_pool_sizes)}")

    row = summarize(route, elapsed, build_seconds=build_seconds,
                    peak_rss_mb=_rss_mb(), warmups=warmups)
    shape["actual_scored_pool_size"] = next(iter(scored_pool_sizes))
    row["route_shape"] = shape
    row["measurement_trace"] = "integer_pool_size_only"
    row["response_sha256"] = next(iter(response_digests))
    row["response_deterministic"] = True
    return row


def _child_command(route: str, catalog: str | Path, samples: int,
                   warmups: int) -> list[str]:
    return [
        sys.executable, "-m", "tools.resource_probe",
        "--child", route,
        "--catalog", str(Path(catalog).resolve()),
        "--samples", str(samples),
        "--warmups", str(warmups),
    ]


def probe_routes(routes, catalog: str | Path = DEFAULT_CATALOG,
                 samples: int = DEFAULT_SAMPLES, warmups: int = DEFAULT_WARMUPS,
                 timeout: float = 900.0) -> dict:
    """Run each requested route in a separate, fixed-hash-seed subprocess."""
    names = tuple(routes)
    if not names or any(name not in ROUTES for name in names):
        raise ValueError(f"routes must be a non-empty subset of {ROUTES}")
    if len(set(names)) != len(names):
        raise ValueError("routes must not contain duplicates")
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative")

    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(REPO))
    from tools.rerank_provenance import effective_runtime_config

    runtime_config = effective_runtime_config(env)
    processor_label = _processor_label()
    catalog_digest = file_sha256(catalog)
    measured = {}
    for route in names:
        command = _child_command(route, catalog, samples, warmups)
        done = subprocess.run(command, cwd=str(REPO), env=env, capture_output=True,
                              text=True, timeout=timeout)
        if done.returncode:
            raise RuntimeError(
                f"{route} resource child failed ({done.returncode}):\n"
                f"stdout={done.stdout}\nstderr={done.stderr}")
        try:
            measured[route] = json.loads(done.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{route} resource child did not emit JSON: {done.stdout!r}") from exc
    return {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "fresh_process_per_route": True,
            "python_hash_seed": "0",
            "platform": sys.platform,
            "machine": platform.machine(),
            "platform_release": f"{platform.system()} {platform.release()}",
            "processor": processor_label,
            "processor_identity_quality": (
                "model_name" if processor_label != platform.machine()
                else "architecture_only"
            ),
            "logical_cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "catalog": _logical_path(catalog),
            "catalog_sha256": catalog_digest,
            "serving_model_sha256": file_sha256(
                REPO / "analysis" / "reranker.json"
            ),
            "source_sha256": resource_source_sha256(),
            "samples_per_route": int(samples),
            "warmups_per_route": int(warmups),
            "latency_scope": LATENCY_SCOPE,
            "p99_convention": P99_CONVENTION,
            "workload_schema": WORKLOAD_SCHEMA,
            "budget_gate": "fixed_workload_zero_exceedance_canary",
            "budget_ms": P99_BUDGET_MS,
            "gate_confidence": GATE_CONFIDENCE,
            "target_violation_rate": TARGET_VIOLATION_RATE,
            "minimum_gate_samples": MIN_GATE_SAMPLES,
            "effective_runtime": runtime_config,
            "iid_assumption_verified": False,
            "serial_samples_per_fresh_process": True,
            "workloads_per_route": 1,
            "route_execution_order": list(names),
            "route_workload_sha256": {
                route: _message_sha256(route) for route in names
            },
            "fallback_policy": ("absolute/descriptive only until a non-inferiority "
                                "margin and independent blocks are predeclared; never "
                                "subject to the 50 ms route gate"),
        },
        "routes": measured,
}


def _validate_report(report: dict, label: str) -> None:
    """Reject incomplete reports before any latency values are compared."""
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"{label} resource schema {report.get('schema')!r} is not {REPORT_SCHEMA}")
    protocol = report.get("protocol")
    routes = report.get("routes")
    if not isinstance(protocol, dict) or not isinstance(routes, dict) or not routes:
        raise ValueError(f"{label} resource report lacks protocol/routes")
    missing = [field for field in COMPATIBILITY_FIELDS if field not in protocol]
    if missing:
        raise ValueError(f"{label} resource protocol lacks {missing}")
    workloads = protocol.get("route_workload_sha256")
    if not isinstance(workloads, dict):
        raise ValueError(f"{label} resource protocol lacks route workload hashes")
    sources = protocol.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(
            not isinstance(name, str) or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in sources.items()):
        raise ValueError(f"{label} resource protocol lacks valid source hashes")
    route_order = protocol.get("route_execution_order")
    if not isinstance(route_order, list) or len(route_order) != len(routes) \
            or len(set(route_order)) != len(route_order) \
            or set(route_order) != set(routes):
        raise ValueError(f"{label} resource route order disagrees with route rows")
    for route, row in routes.items():
        if route not in ROUTES or not isinstance(row, dict) or row.get("route") != route:
            raise ValueError(f"{label} has an invalid {route!r} route row")
        if route not in workloads:
            raise ValueError(f"{label} lacks the {route!r} workload hash")
        if row.get("samples") != protocol["samples_per_route"]:
            raise ValueError(f"{label} {route} sample count disagrees with its protocol")
        if row.get("warmups") != protocol["warmups_per_route"]:
            raise ValueError(f"{label} {route} warmup count disagrees with its protocol")
        shape = row.get("route_shape")
        if not isinstance(shape, dict) or not isinstance(
                shape.get("actual_scored_pool_size"), int) \
                or shape["actual_scored_pool_size"] <= 0:
            raise ValueError(f"{label} {route} lacks the actual scored route shape")
        if route == "exact" and (shape.get("exact") is not True
                                 or shape.get("hedged") is not False):
            raise ValueError(f"{label} exact workload has the wrong route shape")
        if route == "hedged" and (shape.get("exact") is not False
                                  or shape.get("hedged") is not True):
            raise ValueError(f"{label} hedged workload has the wrong route shape")
        if route == "fallback" and (
                shape.get("exact") is not False
                or shape.get("hedged") is not False
                or shape.get("pool_size") != shape.get("catalog_size")
                or shape.get("actual_scored_pool_size") != shape.get("catalog_size")
                or shape.get("primary_size") != 0):
            raise ValueError(f"{label} fallback workload has the wrong route shape")
        digest = row.get("response_sha256")
        if row.get("response_deterministic") is not True \
                or not isinstance(digest, str) or len(digest) != 64 \
                or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{label} {route} lacks a deterministic response digest")
        if row.get("measurement_trace") != "integer_pool_size_only":
            raise ValueError(f"{label} {route} used a production-distorting trace")
        for name in ("build_seconds", "peak_rss_mb"):
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{label} {route} has invalid {name}")
        latency = row.get("latency_ms")
        if not isinstance(latency, dict) or any(
                not isinstance(latency.get(name), (int, float))
                or isinstance(latency.get(name), bool)
                or not math.isfinite(float(latency[name]))
                or float(latency[name]) < 0.0
                for name in ("median", "p95", "p99", "max")):
            raise ValueError(f"{label} {route} has invalid latency statistics")
        median, p95, p99, maximum = (
            float(latency[name]) for name in ("median", "p95", "p99", "max")
        )
        if not 0.0 <= median <= p95 <= p99 <= maximum:
            raise ValueError(f"{label} {route} latency order is impossible")

        gate = row.get("gate")
        if not isinstance(gate, dict):
            raise ValueError(f"{label} {route} lacks a gate contract")
        samples = row["samples"]
        minimum = protocol.get("minimum_gate_samples")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError(f"{label} resource minimum sample count is invalid")
        if route in BUDGETED:
            violations = gate.get("violations")
            if isinstance(violations, bool) or not isinstance(violations, int) \
                    or not 0 <= violations <= samples:
                raise ValueError(f"{label} {route} violation count is invalid")
            enforced = samples >= minimum
            expected_u95 = (
                round(zero_violation_u95(samples), 6)
                if violations == 0 else None
            )
            expected_pass = (
                violations == 0 and expected_u95 is not None
                and expected_u95 < float(protocol["target_violation_rate"])
            ) if enforced else None
            expected_rate = round(violations / samples, 6)
            expected = {
                "kind": protocol.get("budget_gate"),
                "limit_ms": protocol.get("budget_ms"),
                "iid_sensitivity_confidence": protocol.get("gate_confidence"),
                "target_violation_rate": protocol.get("target_violation_rate"),
                "minimum_samples": minimum,
                "violations": violations,
                "observed_violation_rate": expected_rate,
                "iid_zero_violation_u95_sensitivity": expected_u95,
                "iid_assumption_verified": False,
                "statistical_claim": "none; fixed-message serial engineering canary",
                "enforced": enforced,
                "passed": expected_pass,
            }
            if any(gate.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{label} {route} gate arithmetic is inconsistent")
            limit = float(protocol["budget_ms"])
            if (violations == 0 and maximum >= limit) \
                    or (violations > 0 and maximum < limit):
                raise ValueError(f"{label} {route} violations disagree with max latency")
        else:
            expected = {
                "kind": "absolute_descriptive_baseline",
                "limit_ms": None,
                "minimum_samples": minimum,
                "enforced": False,
                "passed": None,
            }
            if any(gate.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{label} fallback gate is not descriptive-only")


def validate_compatible(report: dict, baseline: dict) -> None:
    """Require identical measurement/catalog/workload protocols before comparison."""
    _validate_report(report, "current")
    _validate_report(baseline, "baseline")
    current_protocol = report["protocol"]
    baseline_protocol = baseline["protocol"]
    for field in COMPATIBILITY_FIELDS:
        if current_protocol[field] != baseline_protocol[field]:
            raise ValueError(
                f"resource protocol mismatch for {field}: "
                f"{current_protocol[field]!r} != {baseline_protocol[field]!r}")
    missing_routes = set(report["routes"]) - set(baseline["routes"])
    if missing_routes:
        raise ValueError(f"baseline lacks routes required for comparison: {sorted(missing_routes)}")
    for route in report["routes"]:
        current_hash = current_protocol["route_workload_sha256"].get(route)
        baseline_hash = baseline_protocol["route_workload_sha256"].get(route)
        if current_hash != baseline_hash:
            raise ValueError(
                f"resource workload mismatch for {route}: "
                f"{current_hash!r} != {baseline_hash!r}")
        if report["routes"][route].get("route_shape") \
                != baseline["routes"][route].get("route_shape"):
            raise ValueError(f"resource route shape mismatch for {route}")


def compare_with_baseline(report: dict, baseline: dict) -> dict:
    """Attach same-protocol deltas without manufacturing a fallback gate.

    Two independent empirical p99 point estimates do not define a statistically
    valid no-regression test. Until a non-inferiority margin and independent-block
    protocol are registered, the fallback comparison stays explicitly inconclusive.
    """
    validate_compatible(report, baseline)
    current = copy.deepcopy(report)
    comparisons = {}
    for route, row in current.get("routes", {}).items():
        old = (baseline.get("routes") or {}).get(route)
        if old is None:
            continue
        before = float(old["latency_ms"]["p99"])
        after = float(row["latency_ms"]["p99"])
        comparisons[route] = {
            "baseline_p99_ms": before,
            "current_p99_ms": after,
            "delta_p99_ms": round(after - before, 3),
            "response_digest_match": (
                row.get("response_sha256") == old.get("response_sha256")
            ),
        }
        if route == "fallback":
            row["gate"]["baseline_p99_ms"] = before
            row["gate"]["enforced"] = False
            row["gate"]["passed"] = None
            row["gate"]["comparison_status"] = "descriptive_only"
            row["gate"]["note"] = (
                "same-protocol point delta only; no registered non-inferiority "
                "margin or independent-block interval, so this cannot pass a gate"
            )
    current["comparison"] = comparisons
    return current


def acceptable(report: dict) -> bool:
    """True only when every requested budgeted route proves its gate.

    ``None`` is neutral only for the fallback's first absolute baseline.  An
    under-sampled exact/hedged canary is useful evidence but is not a passing
    operational audit and therefore gets a non-zero CLI status.
    """
    try:
        if report.get("schema") is not None:
            _validate_report(report, "candidate")
    except (TypeError, ValueError):
        return False
    for route, row in (report.get("routes") or {}).items():
        passed = row.get("gate", {}).get("passed")
        if route in BUDGETED:
            gate = row.get("gate") or {}
            latency = row.get("latency_ms") or {}
            if passed is not True or row.get("samples", 0) < MIN_GATE_SAMPLES \
                    or gate.get("violations") != 0 \
                    or float(latency.get("max", math.inf)) >= P99_BUDGET_MS:
                return False
        if route == "fallback" and row.get("gate", {}).get("baseline_p99_ms") \
                is not None:
            # A requested comparison is deliberately inconclusive until the
            # protocol has a preregistered margin and independent blocks.
            return False
        if passed is False:
            return False
    return True


def _parse_routes(value: str, include_fallback: bool) -> tuple[str, ...]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if include_fallback and "fallback" not in names:
        names.append("fallback")
    return tuple(names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-process route resource probe")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--routes", default="exact,hedged")
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=ROUTES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        row = measure_child(args.child, args.catalog, args.samples, args.warmups)
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return 0

    routes = _parse_routes(args.routes, args.include_fallback)
    report = probe_routes(routes, args.catalog, args.samples, args.warmups)
    if args.baseline:
        with args.baseline.open(encoding="utf-8") as handle:
            report = compare_with_baseline(report, json.load(handle))
    body = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")
    return 0 if acceptable(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
