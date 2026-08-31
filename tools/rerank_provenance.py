"""Content-addressed contracts for reranker training and snapshot caches.

The reranking tools are build-time experiments, but their cached rows are only
meaningful under the exact replay that produced them.  A matching feature count
is not enough: changing dialog parsing, routing, scoring, evaluator turns,
environment overrides, candidate depth, or even the ordered split can leave a
schema-compatible cache semantically stale.

This module keeps the JSONL payload compatible with the legacy trainer while
binding it to a canonical sidecar formed by appending ``.manifest.jsonl`` (for
example, ``rerank_dev.jsonl.manifest.jsonl``). Every group carries the contract
digest as well. V4-aware readers validate the complete payload and fail closed;
regeneration is always an explicit caller action.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
CACHE_FORMAT = "techjam-rerank-jsonl"
MANIFEST_VERSION = 1
SCHEMA_VERSION = 4
# Keep regenerable manifests under the repository's existing
# ``analysis/rerank_*.jsonl`` ignore rule alongside their payloads.
MANIFEST_SUFFIX = ".manifest.jsonl"


class CacheError(RuntimeError):
    """Base class for cache contract failures."""


class CacheUnavailable(CacheError):
    """A required input or cache file does not exist."""


class CacheStale(CacheError):
    """The cache is intact but belongs to a different replay contract."""


class CacheCorrupt(CacheError):
    """The manifest or payload contradicts itself."""


def canonical_json(value: object) -> str:
    """The one serialization used for contracts, manifests, and JSONL rows."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CacheCorrupt(f"value is not canonical JSON: {exc}") from exc


def _canonical_clone(value: object):
    """Reject non-JSON options and detach the contract from caller mutation."""
    return json.loads(canonical_json(value))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_identity(path: str | Path) -> dict:
    """Read only the identifiers needed to prove split isolation.

    The result is a diagnostic input, never a serving asset.  Callers that commit
    a report should store counts/hashes rather than the identifier lists.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise CacheUnavailable(f"ordered split unavailable: {candidate}")
    sample_ids: list[str] = []
    targets: list[str] = []
    records: list[dict[str, str]] = []
    try:
        with candidate.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row.get("sample_id") if isinstance(row, dict) else None
                truth = row.get("ground_truth") if isinstance(row, dict) else None
                target = truth.get("parent_asin") if isinstance(truth, dict) else None
                scenario = row.get("scenario_type") if isinstance(row, dict) else None
                if not isinstance(sample_id, str) or not sample_id \
                        or not isinstance(target, str) or not target \
                        or not isinstance(scenario, str) or not scenario:
                    raise CacheCorrupt(
                        f"{candidate} line {line_number} has no "
                        "sample_id/target/scenario_type")
                sample_ids.append(sample_id)
                targets.append(target)
                records.append({"sample_id": sample_id, "target": target,
                                "scenario_type": scenario})
    except json.JSONDecodeError as exc:
        raise CacheCorrupt(f"{candidate} contains invalid JSON: {exc}") from exc
    if not sample_ids:
        raise CacheCorrupt(f"{candidate} contains no sessions")
    duplicate_sessions = sorted(
        key for key, count in Counter(sample_ids).items() if count > 1
    )
    if duplicate_sessions:
        raise CacheCorrupt(
            f"{candidate} repeats {len(duplicate_sessions)} sample_id values")
    return {"sample_ids": sample_ids, "targets": targets, "records": records,
            "ordered_sha256": sha256_file(candidate)}


def catalog_identity(path: str | Path) -> dict:
    """Exact catalog membership used to reject foreign cache rows.

    The identifier list lives only in ignored, build-time cache manifests. Reports
    expose its count and digest, never the identifiers themselves. A digest alone
    could prove which catalog was used but could not prove that every cached row was
    actually a member of it, so the validation contract deliberately carries both.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise CacheUnavailable(f"catalog unavailable: {candidate}")
    identifiers: list[str] = []
    try:
        with candidate.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                pid = row.get("parent_asin") if isinstance(row, dict) else None
                if not isinstance(pid, str) or not pid:
                    raise CacheCorrupt(
                        f"{candidate} line {line_number} has no parent_asin")
                identifiers.append(pid)
    except json.JSONDecodeError as exc:
        raise CacheCorrupt(f"{candidate} contains invalid JSON: {exc}") from exc
    if not identifiers:
        raise CacheCorrupt(f"{candidate} contains no products")
    duplicates = sorted(
        key for key, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise CacheCorrupt(
            f"{candidate} repeats {len(duplicates)} parent_asin values")
    ordered = sorted(identifiers)
    return {
        "count": len(ordered),
        "identifiers": ordered,
        "identifiers_sha256": _sha256_json(ordered),
    }


def split_isolation(first_path: str | Path, second_path: str | Path) -> dict:
    """Return and enforce the session/target independence of two data splits."""
    first, second = split_identity(first_path), split_identity(second_path)
    session_overlap = sorted(set(first["sample_ids"]) & set(second["sample_ids"]))
    target_overlap = sorted(set(first["targets"]) & set(second["targets"]))
    if session_overlap or target_overlap:
        raise CacheStale(
            "split isolation failed: "
            f"{len(session_overlap)} session ids and {len(target_overlap)} targets overlap")
    return {
        "first": {"sessions": len(first["sample_ids"]),
                  "unique_targets": len(set(first["targets"])),
                  "ordered_sha256": first["ordered_sha256"]},
        "second": {"sessions": len(second["sample_ids"]),
                   "unique_targets": len(set(second["targets"])),
                   "ordered_sha256": second["ordered_sha256"]},
        "session_overlap": 0,
        "target_overlap": 0,
    }


def _required_file(path: str | Path, logical_name: str) -> dict:
    candidate = Path(path)
    if not candidate.is_file():
        raise CacheUnavailable(f"{logical_name} unavailable: {candidate}")
    return {"bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}


def _optional_file(path: str | Path, logical_name: str) -> dict:
    candidate = Path(path)
    if not candidate.exists():
        return {"bytes": 0, "present": False, "sha256": None}
    if not candidate.is_file():
        raise CacheUnavailable(f"{logical_name} is not a file: {candidate}")
    return {"bytes": candidate.stat().st_size, "present": True,
            "sha256": sha256_file(candidate)}


def default_source_paths(kind: str | None = None) -> dict[str, Path]:
    """The conservative source closure that can change a replayed snapshot."""
    paths = {
        path.relative_to(REPO).as_posix(): path
        for path in sorted((REPO / "src").rglob("*.py"))
    }
    tool_paths = ["tools/rerank_data.py", "tools/rerank_provenance.py"]
    if kind == "snapshot_mrr":
        tool_paths.append("tools/snapshot_mrr.py")
    # A row cache is bound to code that generates/replays rows. The trainer only
    # consumes those rows; hashing it here would invalidate a frozen V4-0 corpus
    # merely because V4-1's objective implementation changed. Model/experiment
    # provenance owns the trainer hash when a candidate is actually fitted.
    for relative in tool_paths:
        paths[relative] = REPO / relative
    return dict(sorted(paths.items()))


def effective_runtime_config(env: Mapping[str, str] | None = None) -> dict:
    """Resolve every P_/W_ knob exactly as ``Agent`` construction does.

    Raw environment spellings are deliberately not hashed.  For example,
    ``W_POP=1.40`` and the default value are the same effective configuration and
    should not invalidate one another.  Invalid numeric spellings likewise fall
    back through the production constructors before entering the contract.
    """
    from src.policy import CommitPolicy, RejectionModel
    from src.scoring import Weights

    values = os.environ if env is None else env
    weights = Weights.from_env(values)
    policy = CommitPolicy.from_env(values)
    rejection = RejectionModel(values.get("P_PRUNE", RejectionModel.RESET)).mode
    return {
        "agent": {
            "P_ASK": values.get("P_ASK", "infogain"),
            "P_FUSE": "hedged" if values.get("P_FUSE", "hedged") == "hedged"
            else "disabled",
            "P_PRUNE": rejection,
        },
        "commit_policy": asdict(policy),
        "scoring_weights": asdict(weights),
    }


def build_contract(*, catalog_path: str | Path, split_path: str | Path,
                   features: Sequence[str], depth: int,
                   generator_options: Mapping[str, object],
                   evaluator_path: str | Path | None = None,
                   model_path: str | Path | None = None,
                   source_paths: Mapping[str, str | Path] | None = None,
                   env: Mapping[str, str] | None = None) -> dict:
    """Build the deterministic contract a cache must match before it is read."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    names = [str(name) for name in features]
    if not names or len(set(names)) != len(names):
        raise ValueError("feature schema must contain unique names")

    evaluator = Path(evaluator_path) if evaluator_path is not None \
        else REPO / "evaluator" / "local_evaluator.py"
    model = Path(model_path) if model_path is not None \
        else REPO / "analysis" / "reranker.json"
    kind = str(generator_options.get("kind", ""))
    sources = default_source_paths(kind) if source_paths is None \
        else {str(name): Path(path) for name, path in source_paths.items()}
    if not sources:
        raise CacheUnavailable("source provenance unavailable: no source files declared")

    source_hashes = {
        name: _required_file(path, f"source {name}")
        for name, path in sorted(sources.items())
    }
    schema = {"names": names, "width": len(names)}
    schema["sha256"] = _sha256_json(schema)
    catalog = _required_file(catalog_path, "catalog")
    catalog["identity"] = catalog_identity(catalog_path)
    ordered_split = _required_file(split_path, "ordered split")
    split = split_identity(split_path)
    ordered_split["identity"] = {
        "records": split["records"],
        "records_sha256": _sha256_json(split["records"]),
        "sessions": len(split["records"]),
    }
    return {
        "cache_format": CACHE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "catalog": catalog,
        # Hashing raw bytes makes order part of the split contract.
        "ordered_split": ordered_split,
        "evaluator_protocol": _required_file(evaluator, "evaluator protocol"),
        "sources": source_hashes,
        "model_asset": _optional_file(model, "model asset"),
        "feature_schema": schema,
        "depth": depth,
        "generator_options": _canonical_clone(dict(generator_options)),
        "effective_runtime": effective_runtime_config(env),
        "serialization": {
            "canonical_json": "sort_keys,separators,ascii,no_nan",
            "python_implementation": sys.implementation.name,
            "python_version": [sys.version_info.major, sys.version_info.minor],
        },
    }


def contract_sha256(contract: Mapping[str, object]) -> str:
    return _sha256_json(contract)


def manifest_path(cache_path: str | Path) -> Path:
    return Path(str(Path(cache_path)) + MANIFEST_SUFFIX)


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(float(value))


def validate_groups(groups: Sequence[dict], contract: Mapping[str, object],
                    provenance: str) -> dict:
    """Validate every group and row, returning deterministic summary counts."""
    if not groups:
        raise CacheCorrupt("cache contains no groups")
    schema = contract.get("feature_schema")
    if not isinstance(schema, dict):
        raise CacheCorrupt("contract.feature_schema is missing")
    names = schema.get("names")
    width = schema.get("width")
    if not isinstance(names, list) or isinstance(width, bool) \
            or not isinstance(width, int) or width != len(names):
        raise CacheCorrupt("contract.feature_schema dimensions disagree")
    options = contract.get("generator_options")
    if not isinstance(options, dict):
        raise CacheCorrupt("contract.generator_options is missing")
    with_scores = options.get("with_scores") is True
    v4_layout = options.get("layout") == "raw_live_head_plus_six_decimal_compatibility"
    compatibility_depth = options.get("compatibility_depth", contract.get("depth"))
    if isinstance(compatibility_depth, bool) or not isinstance(compatibility_depth, int) \
            or compatibility_depth <= 0:
        raise CacheCorrupt("contract compatibility_depth is invalid")
    depth = contract.get("depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise CacheCorrupt("contract.depth is invalid")

    catalog = contract.get("catalog")
    catalog_identity_value = (
        catalog.get("identity") if isinstance(catalog, dict) else None
    )
    catalog_identifiers = (
        catalog_identity_value.get("identifiers")
        if isinstance(catalog_identity_value, dict) else None
    )
    if not isinstance(catalog_identifiers, list) or not catalog_identifiers \
            or any(not isinstance(pid, str) or not pid for pid in catalog_identifiers) \
            or len(catalog_identifiers) != len(set(catalog_identifiers)):
        raise CacheCorrupt("contract.catalog identity is missing or invalid")
    if catalog_identifiers != sorted(catalog_identifiers) \
            or catalog_identity_value.get("count") != len(catalog_identifiers) \
            or catalog_identity_value.get("identifiers_sha256") \
            != _sha256_json(catalog_identifiers):
        raise CacheCorrupt("contract.catalog identity contradicts itself")
    catalog_ids = set(catalog_identifiers)

    ordered_split = contract.get("ordered_split")
    split_identity_value = (
        ordered_split.get("identity") if isinstance(ordered_split, dict) else None
    )
    split_records = (
        split_identity_value.get("records")
        if isinstance(split_identity_value, dict) else None
    )
    if not isinstance(split_records, list) or not split_records:
        raise CacheCorrupt("contract.ordered_split identity is missing")
    if split_identity_value.get("sessions") != len(split_records) \
            or split_identity_value.get("records_sha256") != _sha256_json(split_records):
        raise CacheCorrupt("contract.ordered_split identity contradicts itself")
    expected_sessions: list[str] = []
    expected_by_session: dict[str, tuple[str, str]] = {}
    for record_index, record in enumerate(split_records, start=1):
        if not isinstance(record, dict):
            raise CacheCorrupt(
                f"contract.ordered_split record {record_index} is not an object")
        sample_id = record.get("sample_id")
        target = record.get("target")
        scenario = record.get("scenario_type")
        if not all(isinstance(value, str) and value
                   for value in (sample_id, target, scenario)):
            raise CacheCorrupt(
                f"contract.ordered_split record {record_index} is invalid")
        if sample_id in expected_by_session:
            raise CacheCorrupt(
                f"contract.ordered_split repeats sample_id {sample_id!r}")
        if target not in catalog_ids:
            raise CacheCorrupt(
                f"contract.ordered_split target {target!r} is outside catalog")
        expected_sessions.append(sample_id)
        expected_by_session[sample_id] = (target, scenario)

    total_rows = total_live_rows = unreachable = 0
    unreachable_sessions: set[str] = set()
    reachability_counts: Counter[str] = Counter()
    sessions: set[str] = set()
    group_keys: set[tuple[str, int]] = set()
    target_by_session: dict[str, str] = {}
    scenarios: Counter[str] = Counter()
    observed_session_order: list[str] = []
    turns_by_session: dict[str, list[int]] = {}
    for group_index, group in enumerate(groups, start=1):
        prefix = f"group {group_index}"
        if not isinstance(group, dict):
            raise CacheCorrupt(f"{prefix} is not an object")
        if group.get("_provenance") != provenance:
            raise CacheCorrupt(f"{prefix} provenance does not match its manifest")
        if group.get("features") != names:
            raise CacheCorrupt(f"{prefix} feature list does not match the contract")
        sample_id = group.get("sample_id")
        target = group.get("target")
        turn = group.get("turn")
        scenario = group.get("scenario_type")
        if not isinstance(sample_id, str) or not sample_id:
            raise CacheCorrupt(f"{prefix} has no valid sample_id")
        if not isinstance(target, str) or not target:
            raise CacheCorrupt(f"{prefix} has no valid target")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
            raise CacheCorrupt(f"{prefix} has no valid turn")
        if not isinstance(scenario, str) or not scenario:
            raise CacheCorrupt(f"{prefix} has no valid scenario_type")
        expected = expected_by_session.get(sample_id)
        if expected is None:
            raise CacheCorrupt(
                f"{prefix} sample_id {sample_id!r} is outside ordered split")
        if (target, scenario) != expected:
            raise CacheCorrupt(
                f"{prefix} target/scenario do not match ordered split")
        if target not in catalog_ids:
            raise CacheCorrupt(f"{prefix} target is outside catalog")
        key = (sample_id, turn)
        if key in group_keys:
            raise CacheCorrupt(f"{prefix} duplicates sample_id/turn {key!r}")
        group_keys.add(key)
        previous_target = target_by_session.setdefault(sample_id, target)
        if previous_target != target:
            raise CacheCorrupt(f"{prefix} changes target within session {sample_id}")
        if sample_id not in turns_by_session:
            observed_session_order.append(sample_id)
            turns_by_session[sample_id] = []
        turns_by_session[sample_id].append(turn)

        rows = group.get("rows")
        if not isinstance(rows, list) or len(rows) < 2:
            raise CacheCorrupt(f"{prefix} must contain at least two rows")
        row_limit = compatibility_depth + 1 if v4_layout else depth + 1
        if len(rows) > row_limit:
            raise CacheCorrupt(
                f"{prefix} has {len(rows)} rows beyond compatibility depth "
                f"{compatibility_depth}")
        seen: set[str] = set()
        positives: list[dict] = []
        for row_index, row in enumerate(rows, start=1):
            where = f"{prefix} row {row_index}"
            if not isinstance(row, dict):
                raise CacheCorrupt(f"{where} is not an object")
            pid = row.get("pid")
            if not isinstance(pid, str) or not pid:
                raise CacheCorrupt(f"{where} has no valid pid")
            if pid in seen:
                raise CacheCorrupt(f"{prefix} contains duplicate pid {pid}")
            if pid not in catalog_ids:
                raise CacheCorrupt(f"{where} pid is outside catalog")
            seen.add(pid)
            label = row.get("y")
            if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                raise CacheCorrupt(f"{where} label must be integer 0 or 1")
            vector = row.get("x")
            if not isinstance(vector, list) or len(vector) != width:
                got = len(vector) if isinstance(vector, list) else "non-list"
                raise CacheCorrupt(f"{where} vector width {got} != {width}")
            for feature_index, value in enumerate(vector):
                if not _is_finite_number(value):
                    raise CacheCorrupt(
                        f"{where} feature {names[feature_index]} is not finite")
            if with_scores:
                if "s" not in row:
                    raise CacheCorrupt(f"{where} has no base score")
                score = row["s"]
                if score is not None and not _is_finite_number(score):
                    raise CacheCorrupt(f"{where} base score is not finite")
                if score is None and label != 1:
                    raise CacheCorrupt(f"{where} non-target cannot be unreachable")
            elif "s" in row:
                raise CacheCorrupt(f"{where} has a score under with_scores=false")
            if label == 1:
                positives.append(row)
        if len(positives) != 1:
            raise CacheCorrupt(f"{prefix} contains {len(positives)} target rows")
        if positives[0]["pid"] != target:
            raise CacheCorrupt(f"{prefix} target label does not match group.target")
        if with_scores and positives[0]["s"] is None:
            unreachable += 1
            unreachable_sessions.add(sample_id)

        if v4_layout:
            if group.get("schema") != "techjam-rerank-snapshot-v4":
                raise CacheCorrupt(f"{prefix} has no supported V4 schema")
            message = group.get("reference_message")
            message_digest = group.get("reference_message_sha256")
            if not isinstance(message, str) or not isinstance(message_digest, str) \
                    or message_digest != hashlib.sha256(message.encode("utf-8")).hexdigest():
                raise CacheCorrupt(f"{prefix} reference message hash is invalid")
            compatibility = group.get("compatibility")
            expected_compatibility = {
                "depth": compatibility_depth,
                "numeric_precision": "six_decimal",
                "projection": "dedupe(full_ranked_ids[:depth] + [target])",
                "with_scores": with_scores,
            }
            if compatibility != expected_compatibility:
                raise CacheCorrupt(f"{prefix} compatibility declaration is stale")
            diagnostics = group.get("diagnostics")
            if not isinstance(diagnostics, dict):
                raise CacheCorrupt(f"{prefix} diagnostics are missing")
            active_constraints = diagnostics.get("active_constraints")
            pool_size = diagnostics.get("candidate_pool_size")
            route = diagnostics.get("route")
            facets = diagnostics.get("facets")
            constraint_types = diagnostics.get("disclosed_constraint_types")
            if isinstance(active_constraints, bool) \
                    or not isinstance(active_constraints, int) \
                    or active_constraints < 0:
                raise CacheCorrupt(f"{prefix} active constraint count is invalid")
            if isinstance(pool_size, bool) or not isinstance(pool_size, int) \
                    or pool_size < 0:
                raise CacheCorrupt(f"{prefix} candidate pool size is invalid")
            if route not in ("exact", "inferred_hedge", "global_fallback", "unknown"):
                raise CacheCorrupt(f"{prefix} route diagnostic is invalid")
            if not isinstance(facets, list) or any(
                    not isinstance(name, str) for name in facets):
                raise CacheCorrupt(f"{prefix} facet diagnostics are invalid")
            if not isinstance(constraint_types, list) or any(
                    not isinstance(name, str) for name in constraint_types):
                raise CacheCorrupt(
                    f"{prefix} disclosed constraint diagnostics are invalid")
            if not isinstance(diagnostics.get("has_budget"), bool) \
                    or not isinstance(diagnostics.get("has_refusal"), bool):
                raise CacheCorrupt(f"{prefix} evidence diagnostics are invalid")

            live_rows = group.get("live_rows")
            if not isinstance(live_rows, list) or not live_rows:
                raise CacheCorrupt(f"{prefix} has no live serving rows")
            if len(live_rows) > depth:
                raise CacheCorrupt(f"{prefix} has {len(live_rows)} live rows beyond depth {depth}")
            if pool_size < len(live_rows):
                raise CacheCorrupt(f"{prefix} live head exceeds candidate pool size")
            live_seen: set[str] = set()
            live_target_rows: list[dict] = []
            for live_index, live_row in enumerate(live_rows, start=1):
                where = f"{prefix} live row {live_index}"
                if not isinstance(live_row, dict):
                    raise CacheCorrupt(f"{where} is not an object")
                pid = live_row.get("pid")
                if not isinstance(pid, str) or not pid:
                    raise CacheCorrupt(f"{where} has no valid pid")
                if pid in live_seen:
                    raise CacheCorrupt(f"{prefix} live rows duplicate pid {pid}")
                if pid not in catalog_ids:
                    raise CacheCorrupt(f"{where} pid is outside catalog")
                live_seen.add(pid)
                label = live_row.get("y")
                if isinstance(label, bool) or not isinstance(label, int) \
                        or label not in (0, 1):
                    raise CacheCorrupt(f"{where} label must be integer 0 or 1")
                vector = live_row.get("x")
                if not isinstance(vector, list) or len(vector) != width:
                    got = len(vector) if isinstance(vector, list) else "non-list"
                    raise CacheCorrupt(f"{where} vector width {got} != {width}")
                for feature_index, value in enumerate(vector):
                    if not _is_finite_number(value):
                        raise CacheCorrupt(
                            f"{where} feature {names[feature_index]} is not finite")
                if not _is_finite_number(live_row.get("s")):
                    raise CacheCorrupt(f"{where} base score is not finite")
                if live_row.get("live_rank") != live_index:
                    raise CacheCorrupt(f"{where} live_rank does not match its order")
                if live_row.get("in_rerank_head") is not True:
                    raise CacheCorrupt(f"{where} is not marked in_rerank_head")
                if label:
                    live_target_rows.append(live_row)

            target_in_head = group.get("target_in_rerank_head")
            target_in_pool = group.get("target_in_pool")
            target_rank = group.get("target_live_rank")
            if not isinstance(target_in_head, bool) or not isinstance(target_in_pool, bool):
                raise CacheCorrupt(f"{prefix} target membership flags are invalid")
            if target_in_head:
                if not target_in_pool:
                    raise CacheCorrupt(f"{prefix} live target is absent from candidate pool")
                if len(live_target_rows) != 1 or live_target_rows[0]["pid"] != target:
                    raise CacheCorrupt(f"{prefix} live target row is inconsistent")
                if target_rank != live_target_rows[0]["live_rank"]:
                    raise CacheCorrupt(f"{prefix} target_live_rank is inconsistent")
                expected_reachability = "rerankable"
            else:
                if live_target_rows or target_rank is not None:
                    raise CacheCorrupt(f"{prefix} unreachable target appears in live rows")
                expected_reachability = (
                    "rerank_depth_miss" if target_in_pool else "route_pool_miss"
                )
                if not with_scores:
                    unreachable += 1
                    unreachable_sessions.add(sample_id)
            if group.get("reachability") != expected_reachability:
                raise CacheCorrupt(f"{prefix} reachability classification is inconsistent")

            expected_ids = list(dict.fromkeys(
                [row["pid"] for row in live_rows[:compatibility_depth]] + [target]
            ))
            if [row["pid"] for row in rows] != expected_ids:
                raise CacheCorrupt(f"{prefix} compatibility projection is not literal")
            live_by_pid = {row["pid"]: row for row in live_rows}
            for row_index, row in enumerate(rows, start=1):
                live = live_by_pid.get(row["pid"])
                if live is None:
                    if row["pid"] != target or target_in_head:
                        raise CacheCorrupt(
                            f"{prefix} row {row_index} is not in the live head")
                    if with_scores and row["s"] is not None:
                        raise CacheCorrupt(
                            f"{prefix} unreachable compatibility target has a score")
                    continue
                rounded = [round(float(value), 6) for value in live["x"]]
                if row["x"] != rounded:
                    raise CacheCorrupt(
                        f"{prefix} row {row_index} is not the six-decimal live vector")
                if with_scores and row["s"] != round(float(live["s"]), 6):
                    raise CacheCorrupt(
                        f"{prefix} row {row_index} score is not the rounded live score")

            total_live_rows += len(live_rows)
            reachability_counts[expected_reachability] += 1
        total_rows += len(rows)
        sessions.add(sample_id)
        scenarios[scenario] += 1

    if observed_session_order != expected_sessions:
        missing = sorted(set(expected_sessions) - set(observed_session_order))
        raise CacheCorrupt(
            "cache sessions do not exactly match ordered split "
            f"(order mismatch or {len(missing)} missing)")
    max_turns = options.get("max_turns")
    for sample_id, turns in turns_by_session.items():
        if turns != list(range(1, len(turns) + 1)):
            raise CacheCorrupt(
                f"session {sample_id!r} turns are not contiguous from one")
        if isinstance(max_turns, int) and not isinstance(max_turns, bool) \
                and turns[-1] > max_turns:
            raise CacheCorrupt(
                f"session {sample_id!r} exceeds declared max_turns")

    return {
        "groups": len(groups),
        "live_rows": total_live_rows,
        "reachability": dict(sorted(reachability_counts.items())),
        "rows": total_rows,
        "sessions": len(sessions),
        "targets": len(set(target_by_session.values())),
        "unreachable_snapshots": unreachable,
        "unreachable_sessions": len(unreachable_sessions),
        "scenario_groups": dict(sorted(scenarios.items())),
    }


def canonical_payload(groups: Sequence[dict]) -> bytes:
    return b"".join((canonical_json(group) + "\n").encode("utf-8")
                    for group in groups)


def payload_sha256(groups: Sequence[dict]) -> str:
    return _sha256_bytes(canonical_payload(groups))


def _decorated(groups: Iterable[dict], provenance: str) -> list[dict]:
    out: list[dict] = []
    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise CacheCorrupt(f"group {index} is not an object")
        existing = group.get("_provenance")
        if existing is not None and existing != provenance:
            raise CacheStale(f"group {index} belongs to different provenance")
        out.append({**group, "_provenance": provenance})
    return out


def write_cache(cache_path: str | Path, groups: Iterable[dict],
                contract: Mapping[str, object]) -> dict:
    """Atomically write a canonical payload and its canonical manifest."""
    path = Path(cache_path)
    provenance = contract_sha256(contract)
    stored = _decorated(groups, provenance)
    counts = validate_groups(stored, contract, provenance)
    payload = canonical_payload(stored)
    manifest = {
        "cache_format": CACHE_FORMAT,
        "manifest_version": MANIFEST_VERSION,
        "provenance": provenance,
        "contract": _canonical_clone(dict(contract)),
        "counts": counts,
        "payload_sha256": _sha256_bytes(payload),
    }
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = manifest_path(path)
    payload_partial = Path(str(path) + ".partial")
    manifest_partial = Path(str(sidecar) + ".partial")
    try:
        payload_partial.write_bytes(payload)
        manifest_partial.write_bytes(manifest_bytes)
        payload_partial.replace(path)
        manifest_partial.replace(sidecar)
    finally:
        if payload_partial.exists():
            payload_partial.unlink()
        if manifest_partial.exists():
            manifest_partial.unlink()
    return manifest


_MISSING = object()


def _differences(expected: object, actual: object, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            here = f"{prefix}.{key}" if prefix else str(key)
            left, right = expected.get(key, _MISSING), actual.get(key, _MISSING)
            if left is _MISSING or right is _MISSING:
                out.append(here)
            else:
                out.extend(_differences(left, right, here))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [prefix]
        out = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            out.extend(_differences(left, right, f"{prefix}[{index}]"))
        return out
    return [] if expected == actual else [prefix]


def _read_manifest(path: Path) -> dict:
    sidecar = manifest_path(path)
    if not sidecar.exists():
        if path.exists():
            raise CacheStale(
                f"{path} is a legacy/unprovenanced cache; rebuild it explicitly")
        raise CacheUnavailable(f"cache unavailable: {path}")
    if not path.exists():
        raise CacheUnavailable(f"cache payload unavailable: {path}")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheCorrupt(f"cannot read cache manifest {sidecar}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CacheCorrupt(f"cache manifest {sidecar} is not an object")
    if manifest.get("cache_format") != CACHE_FORMAT \
            or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise CacheStale(f"cache manifest {sidecar} has an unsupported format")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise CacheCorrupt(f"cache manifest {sidecar} has no contract")
    if manifest.get("provenance") != contract_sha256(contract):
        raise CacheCorrupt(f"cache manifest {sidecar} has an invalid contract digest")
    return manifest


def _read_payload(path: Path) -> tuple[list[dict], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CacheUnavailable(f"cache payload unavailable: {path}: {exc}") from exc
    groups: list[dict] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            group = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CacheCorrupt(f"{path} line {line_number} is invalid JSON: {exc}") from exc
        groups.append(group)
    return groups, payload


def load_cache(cache_path: str | Path,
               expected_contract: Mapping[str, object]) -> list[dict]:
    """Load only a complete cache produced under ``expected_contract``."""
    path = Path(cache_path)
    manifest = _read_manifest(path)
    actual_contract = manifest["contract"]
    differences = _differences(expected_contract, actual_contract)
    if differences:
        summary = ", ".join(differences[:8])
        if len(differences) > 8:
            summary += f", and {len(differences) - 8} more"
        raise CacheStale(f"cache provenance mismatch: {summary}")

    groups, raw_payload = _read_payload(path)
    if manifest.get("payload_sha256") != _sha256_bytes(raw_payload):
        raise CacheCorrupt(f"cache payload digest does not match {manifest_path(path)}")
    counts = validate_groups(groups, actual_contract, manifest["provenance"])
    if manifest.get("counts") != counts:
        raise CacheCorrupt("cache manifest counts do not match the complete payload")
    # Provenance is a storage contract, not a model feature.  Existing metric and
    # trainer helpers therefore receive the legacy group shape after validation.
    return [{key: value for key, value in group.items() if key != "_provenance"}
            for group in groups]
