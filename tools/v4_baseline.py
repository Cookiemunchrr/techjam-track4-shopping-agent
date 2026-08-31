"""Build the deterministic V4-0 ranking/provenance diagnostic report.

This command never rebuilds data and never fits a model.  It reads only caches
that already pass the V4 provenance contract, reports aggregate diagnostics, and
keeps all sample/target identifiers out of the committed result.

    python3 -m tools.v4_baseline \
        --output analysis/improvement_plan_v4_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.features import FEATURES
from tools import snapshot_mrr
from tools.rerank_data import (CATALOG, NEGATIVES, SPLITS,
                               load_training_cache)
from tools.rerank_provenance import (CacheError, canonical_json, manifest_path,
                                     sha256_file, split_isolation)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "analysis" / "reranker.json"
DEFAULT_OUTPUT = REPO / "analysis" / "improvement_plan_v4_results.json"
DEFAULT_CONTROLS = REPO / "analysis" / "v4_end_to_end_controls.json"
TRAINING_SPLITS = frozenset(("dev", "holdout"))

# These are the freshly replayed, content-addressed snapshots for protected
# baseline f9b1357.  They deliberately pin both the opaque cache bytes and the
# semantically relevant base/shipped candidate orders.  A source or catalog
# change must therefore be re-baselined explicitly; a self-consistent manifest
# alone cannot silently bless changed ranking behaviour.
PROTECTED_F9_SNAPSHOT_BASELINES = {
    "dev": {
        "payload_sha256": "46efbf197c3ccd85077773b2f9504713cbde251ed98c98a6840290fadfb682bc",
        "catalog_sha256": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
        "split_sha256": "857b435be5cdb6fc53c5a7c40ef044a6a766249016953666db25e406bb354723",
        "model_sha256": "e108939c1cf87da8df0be844965a9612ed2b540a74764e76647e361749478ed7",
        "transcript_sha256": "74d82b4d0162193d19712e2e539775e6c86fb94e1567e5d40b2a5c1020138ec8",
        "order_pair_sha256": "981ade293ec8147b979dbcea2e5bcfd1ac299d430c70581d5073dd99b23379b0",
        "comparison": {
            "after": 0.68569,
            "before": 0.66502,
            "bootstrap_median_delta": 0.02015,
            "ci95": [0.00238, 0.04196],
            "delta": 0.02067,
            "sessions": 100,
            "snapshots": 148,
            "verdict": "significant",
        },
    },
    "holdout": {
        "payload_sha256": "8d732d969fb0a9eec007568a3a71855edf935f5ef8f2241df69bbd9e389eea6e",
        "catalog_sha256": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
        "split_sha256": "9eba73617b5fad91ac98b6dcdcdc4cadfc54a1298cdb0ab5762253ffd66a09a6",
        "model_sha256": "e108939c1cf87da8df0be844965a9612ed2b540a74764e76647e361749478ed7",
        "transcript_sha256": "41ed44bfe0a8467ff68fa3465272e97eaccff7354e3f3eece43742095312c281",
        "order_pair_sha256": "d12778dabb6ed2050ed35b440e1e83105992a82db028865e55ffb215bb01bd31",
        "comparison": {
            "after": 0.68006,
            "before": 0.64461,
            "bootstrap_median_delta": 0.03471,
            "ci95": [0.01444, 0.06086],
            "delta": 0.03544,
            "sessions": 100,
            "snapshots": 144,
            "verdict": "significant",
        },
    },
}


def serving_rows(group: dict) -> list[dict]:
    return [row for row in group.get("live_rows", group.get("rows", ()))
            if row.get("s") is not None]


def ordered_rows(group: dict, model=None) -> list[dict]:
    rows = serving_rows(group)
    if model is None:
        return rows
    rescored = [
        (row["s"] + model.blend * model.score_vector(row["x"]), row["pid"], row)
        for row in rows
    ]
    rescored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in rescored]


def target_rank(group: dict, model=None) -> int | None:
    for rank, row in enumerate(ordered_rows(group, model), start=1):
        if row.get("y") == 1:
            return rank
    return None


def rank_histogram(groups, model=None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for group in groups:
        rank = target_rank(group, model)
        counts["miss" if rank is None else str(rank)] += 1
    return dict(sorted(counts.items(), key=lambda item: (
        item[0] == "miss", int(item[0]) if item[0] != "miss" else 10**9
    )))


def session_changes(groups, before, after) -> dict[str, int]:
    first = snapshot_mrr.session_scores(groups, before)
    second = snapshot_mrr.session_scores(groups, after)
    counts = Counter()
    for sample_id in sorted(first):
        delta = second[sample_id] - first[sample_id]
        counts["improved" if delta > 1e-15 else
               "worsened" if delta < -1e-15 else "tied"] += 1
    return {name: counts.get(name, 0) for name in ("improved", "worsened", "tied")}


def _mrr_slice(groups, model) -> dict:
    sessions = snapshot_mrr.by_session(groups)
    return {
        "snapshots": len(groups),
        "sessions": len(sessions),
        "base_mrr": round(snapshot_mrr.mrr(groups), 6),
        "shipped_mrr": round(snapshot_mrr.mrr(groups, model), 6),
    }


def _bands(value: int | None) -> str:
    if value is None:
        return "unreachable"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 10:
        return "4-10"
    if value <= 40:
        return "11-40"
    return "41-200"


def _pool_band(value: int) -> str:
    if value <= 50:
        return "1-50"
    if value <= 200:
        return "51-200"
    if value <= 2000:
        return "201-2000"
    return "2001+"


def _decile(value: float) -> str:
    bounded = min(1.0, max(0.0, float(value)))
    index = min(9, int(bounded * 10.0))
    return f"d{index + 1}"


def slice_report(groups, model) -> dict:
    dimensions: dict[str, dict[str, list[dict]]] = {
        "scenario": defaultdict(list),
        "turn": defaultdict(list),
        "route": defaultdict(list),
        "candidate_pool_size": defaultdict(list),
        "active_constraints": defaultdict(list),
        "base_rank": defaultdict(list),
        "disclosed_evidence": defaultdict(list),
        "target_global_popularity": defaultdict(list),
        "target_shelf_popularity": defaultdict(list),
    }
    popularity_index = FEATURES.index("popularity")
    shelf_popularity_index = FEATURES.index("shelf_popularity")
    for group in groups:
        diagnostics = group.get("diagnostics") or {}
        constraints = int(diagnostics.get("active_constraints", 0))
        dimensions["scenario"][str(group.get("scenario_type", "unknown"))].append(group)
        dimensions["turn"]["turn_1" if group.get("turn") == 1 else "later"].append(group)
        dimensions["route"][str(diagnostics.get("route", "unknown"))].append(group)
        dimensions["candidate_pool_size"][_pool_band(
            int(diagnostics.get("candidate_pool_size", 0))
        )].append(group)
        dimensions["active_constraints"][
            "0" if constraints == 0 else "1" if constraints == 1 else "2+"
        ].append(group)
        dimensions["base_rank"][_bands(target_rank(group))].append(group)
        evidence = list(diagnostics.get("disclosed_constraint_types") or [])
        if diagnostics.get("has_budget") and "budget" not in evidence:
            evidence.append("budget")
        if diagnostics.get("has_refusal"):
            evidence.append("refusal")
        for name in sorted(set(evidence)) or ["none"]:
            dimensions["disclosed_evidence"][str(name)].append(group)
        target_row = next(
            (row for row in group.get("rows", ()) if row.get("y") == 1), None
        )
        if target_row is not None:
            dimensions["target_global_popularity"][_decile(
                target_row["x"][popularity_index]
            )].append(group)
            dimensions["target_shelf_popularity"][_decile(
                target_row["x"][shelf_popularity_index]
            )].append(group)
    return {
        dimension: {
            name: _mrr_slice(rows, model)
            for name, rows in sorted(values.items())
        }
        for dimension, values in dimensions.items()
    }


def transcript_sha256(groups) -> str:
    records = [
        {
            "sample_id": group["sample_id"],
            "turn": group["turn"],
            "message": group.get("reference_message", ""),
        }
        for group in groups
    ]
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def internal_order_sha256(groups, model=None) -> str:
    """Hash ordered candidate identities without publishing benchmark labels."""
    records = [
        {
            "sample_id": group["sample_id"],
            "turn": group["turn"],
            "candidate_order": [row["pid"] for row in ordered_rows(group, model)],
        }
        for group in groups
    ]
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def order_pair_sha256(groups, model) -> str:
    """Hash base/shipped orders while excluding labels and benchmark IDs."""
    payload = {
        "schema": "techjam-v4-f9-order-pair-v1",
        "snapshots": [
            {
                "turn": group["turn"],
                "base_order": [row["pid"] for row in ordered_rows(group)],
                "shipped_order": [
                    row["pid"] for row in ordered_rows(group, model)
                ],
            }
            for group in groups
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_snapshot_pin(actual: dict, expected: dict) -> None:
    """Require exact protected-cache, internal-order, and metric agreement."""
    if actual != expected:
        mismatches = sorted(
            key for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise CacheError(
            "protected f9 snapshot baseline mismatch: " + ", ".join(mismatches)
        )


def protected_snapshot_control(split: str, groups, model,
                               cache_path: str | Path, *,
                               catalog_path: str | Path = CATALOG,
                               model_path: str | Path = DEFAULT_MODEL) -> dict:
    expected = PROTECTED_F9_SNAPSHOT_BASELINES.get(split)
    if expected is None:
        return {
            "status": "not_applicable",
            "reason": "only dev and holdout are protected selection baselines",
        }
    actual = {
        "payload_sha256": sha256_file(cache_path),
        "catalog_sha256": sha256_file(catalog_path),
        "split_sha256": sha256_file(SPLITS[split]),
        "model_sha256": sha256_file(model_path),
        "transcript_sha256": transcript_sha256(groups),
        "order_pair_sha256": order_pair_sha256(groups, model),
        "comparison": snapshot_mrr.report(groups, None, model),
    }
    validate_snapshot_pin(actual, expected)
    return {"status": "passed", **actual}


def snapshot_report(groups, model) -> dict:
    comparison = snapshot_mrr.report(groups, None, model)
    base_ranks = [target_rank(group) for group in groups]
    shipped_ranks = [target_rank(group, model) for group in groups]
    return {
        "comparison": comparison,
        "transcript_sha256": transcript_sha256(groups),
        "base_target_rank_histogram": rank_histogram(groups),
        "shipped_target_rank_histogram": rank_histogram(groups, model),
        "session_changes": session_changes(groups, None, model),
        "base_snapshot_target_within_top10_rate": round(
            sum(rank is not None and rank <= 10 for rank in base_ranks)
            / len(base_ranks), 6
        ) if base_ranks else 0.0,
        "snapshot_target_within_top10_rate": round(
            sum(rank is not None and rank <= 10 for rank in shipped_ranks)
            / len(shipped_ranks), 6
        ) if shipped_ranks else 0.0,
        "hindsight_target_in_live_head_rate": round(
            sum(rank is not None for rank in base_ranks) / len(base_ranks), 6
        ) if base_ranks else 0.0,
        "reachability": dict(sorted(Counter(
            group.get("reachability", "legacy_unknown") for group in groups
        ).items())),
        "slices": slice_report(groups, model),
        "unavailable_slice_families": {
            "evidence_field_provenance": (
                "current 15-feature snapshots do not retain title/structured/"
                "description attribution; V4-0 does not infer it after the fact"
            ),
            "paraphrase_axes": (
                "reported by the separate frozen end-to-end controls, not by "
                "clean reference snapshots"
            ),
        },
    }


def pair_mass_report(groups) -> dict:
    a0_by_session: defaultdict[str, int] = defaultdict(int)
    actionable_by_session: defaultdict[str, int] = defaultdict(int)
    reachability = Counter()
    for group in groups:
        negatives = sum(row.get("y") == 0 for row in group.get("rows", ()))
        sample_id = str(group["sample_id"])
        a0_by_session[sample_id] += negatives
        # Preserve sessions with zero actionable comparisons in the denominator.
        # Omitting them would make the distribution look healthier precisely when
        # the reranker cannot act on a target.
        actionable_by_session[sample_id] += 0
        state = group.get("reachability", "legacy_unknown")
        reachability[state] += 1
        if state in ("rerankable", "legacy_unknown"):
            actionable_by_session[sample_id] += negatives

    def summary(values) -> dict:
        ordered = sorted(values)
        return {
            "sessions": len(ordered),
            "min": min(ordered) if ordered else 0,
            "median": statistics.median(ordered) if ordered else 0,
            "max": max(ordered) if ordered else 0,
            "mean": round(statistics.fmean(ordered), 6) if ordered else 0.0,
        }

    return {
        "a0_compatibility_pairs_per_session": summary(a0_by_session.values()),
        "actionable_pairs_per_session": summary(actionable_by_session.values()),
        "reachability": dict(sorted(reachability.items())),
    }


def _parse_catalog_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for pattern in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), pattern)
        except ValueError:
            continue
    return None


def catalog_prior_probe(catalog_path: str | Path, groups, model, *,
                        include_recency: bool = True) -> dict:
    losses = []
    wanted: set[str] = set()
    for group in groups:
        if group.get("turn") != 1:
            continue
        order = ordered_rows(group, model)
        rank = target_rank(group, model)
        if rank is None or rank <= 1 or not order:
            continue
        target = str(group["target"])
        winner = str(order[0]["pid"])
        target_row = next(row for row in serving_rows(group) if row["pid"] == target)
        winner_row = next(row for row in serving_rows(group) if row["pid"] == winner)
        losses.append((target, winner, target_row, winner_row))
        wanted.update((target, winner))

    popularity_index = FEATURES.index("popularity")
    target_less_popular = sum(
        target_row["x"][popularity_index] < winner_row["x"][popularity_index]
        for _, _, target_row, winner_row in losses
    )
    report = {
        "turn1_shipped_losses": len(losses),
        "target_less_popular_than_winner": target_less_popular,
    }
    if not include_recency:
        report["recency_probe"] = {
            "status": "withheld",
            "reason": ("dev-only hypothesis diagnostic; holdout is not read for "
                       "V4-6 selection before registration"),
        }
        return report

    metadata: dict[str, tuple[datetime | None, float]] = {}
    parseable_catalog_dates = total_products = 0
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            total_products += 1
            details = product.get("details") or {}
            date = _parse_catalog_date(
                details.get("Date First Available") if isinstance(details, dict) else None
            )
            if date is not None:
                parseable_catalog_dates += 1
            pid = str(product.get("parent_asin", ""))
            if pid in wanted:
                try:
                    reviews = float(product.get("rating_number") or 0.0)
                except (TypeError, ValueError):
                    reviews = 0.0
                metadata[pid] = (date, reviews)

    dated = [
        (metadata[target], metadata[winner])
        for target, winner, _, _ in losses
        if target in metadata and winner in metadata
        and metadata[target][0] is not None and metadata[winner][0] is not None
    ]
    return {**report,
        "catalog_dates": {
            "products": total_products,
            "parseable": parseable_catalog_dates,
        },
        "loss_pairs_with_both_dates": len(dated),
        "target_newer_than_winner": sum(
            target_meta[0] > winner_meta[0] for target_meta, winner_meta in dated
        ),
        "target_has_fewer_reviews": sum(
            target_meta[1] < winner_meta[1] for target_meta, winner_meta in dated
        ),
    }


def cache_summary(path: str | Path, *, validated: bool,
                  reason: str | None = None) -> dict:
    sidecar = manifest_path(path)
    payload = Path(path)
    try:
        display_path = payload.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        display_path = str(payload)
    if not payload.exists() or not sidecar.exists():
        return {"status": "unavailable", "path": display_path}
    if not validated:
        return {"status": "invalid", "path": display_path,
                "reason": reason or "cache was not validated"}
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    contract = manifest.get("contract") or {}
    sources = contract.get("sources") or {}
    return {
        "status": "validated",
        "path": display_path,
        "payload_sha256": manifest.get("payload_sha256"),
        "provenance": manifest.get("provenance"),
        "counts": manifest.get("counts"),
        "contract": {
            "catalog_sha256": (contract.get("catalog") or {}).get("sha256"),
            "depth": contract.get("depth"),
            "evaluator_sha256": (
                contract.get("evaluator_protocol") or {}
            ).get("sha256"),
            "feature_schema_sha256": (
                contract.get("feature_schema") or {}
            ).get("sha256"),
            "ordered_split_sha256": (
                contract.get("ordered_split") or {}
            ).get("sha256"),
            "source_bundle_sha256": hashlib.sha256(
                canonical_json(sources).encode("utf-8")
            ).hexdigest(),
        },
    }


def build_report(*, splits, catalog: str | Path, model_path: str | Path,
                 controls_path: str | Path = DEFAULT_CONTROLS) -> dict:
    model = snapshot_mrr.load_model(str(model_path))
    if model is None:
        raise SystemExit(f"serving model unavailable: {model_path}")
    controls_file = Path(controls_path)
    try:
        controls = json.loads(controls_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"V4-0 end-to-end controls unavailable: {controls_file}") from exc
    if controls.get("schema") != "techjam-v4-end-to-end-controls-v1":
        raise SystemExit(f"unsupported V4-0 controls: {controls_file}")
    report = {
        "schema": "techjam-v4-foundation-report-v1",
        "scope": "V4-0 measurement/provenance; no candidate model",
        "report_builder_sha256": sha256_file(__file__),
        "model_sha256": sha256_file(model_path),
        "catalog_sha256": sha256_file(catalog),
        "split_isolation": split_isolation(SPLITS["dev"], SPLITS["holdout"]),
        "end_to_end_controls_sha256": sha256_file(controls_file),
        "end_to_end_controls": controls,
        "splits": {},
    }
    for split in splits:
        groups = snapshot_mrr.cached(split, rebuild=False, catalog=str(catalog))
        snapshot_path = REPO / f"analysis/rerank_snapshots_{split}.jsonl"
        protected_control = protected_snapshot_control(
            split, groups, model, snapshot_path,
            catalog_path=catalog, model_path=model_path,
        )
        training_path = REPO / f"analysis/rerank_{split}.jsonl"
        training_valid = False
        training_error = None
        if split in TRAINING_SPLITS:
            try:
                training = load_training_cache(
                    training_path, catalog, SPLITS[split], NEGATIVES
                )
                pair_mass = {"status": "validated", **pair_mass_report(training)}
                training_valid = True
            except CacheError as exc:
                training_error = str(exc)
                pair_mass = {"status": "unavailable", "reason": training_error}
            training_cache = cache_summary(
                training_path, validated=training_valid, reason=training_error
            )
        else:
            pair_mass = {
                "status": "not_applicable",
                "reason": "full split is evaluation-only; no training cache is built",
            }
            training_cache = {
                "status": "not_applicable",
                "path": training_path.relative_to(REPO).as_posix(),
            }
        report["splits"][split] = {
            "snapshot_cache": cache_summary(
                snapshot_path, validated=True
            ),
            "protected_f9_snapshot_control": protected_control,
            "training_cache": training_cache,
            "snapshot": snapshot_report(groups, model),
            "training_pair_mass": pair_mass,
            "prior_probe": catalog_prior_probe(
                catalog, groups, model, include_recency=(split == "dev")
            ),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic V4-0 report")
    parser.add_argument("--splits", default="dev,holdout,full")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--controls", default=str(DEFAULT_CONTROLS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    unknown = [value for value in splits if value not in snapshot_mrr.SPLITS]
    if not splits or unknown:
        raise SystemExit(f"unknown/empty splits: {unknown}")
    report = build_report(splits=splits, catalog=args.catalog,
                          model_path=args.model, controls_path=args.controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
