"""Operational evidence for an offline, deterministic submission.

The test suite pins each invariant independently. This command produces the
compact, reproducible report used by the submission write-up::

    python3 -m tools.audit
    PYTHONHASHSEED=1 python3 -m tools.audit --hash-only

Runtime measurements are descriptive rather than score-tuned. The response hash
contains every public-set customer turn and agent response but excludes random
session identifiers and timing data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "data" / "catalog.jsonl"
PUBLIC = REPO / "data" / "public_set.jsonl"
EVALUATOR = REPO / "evaluator" / "local_evaluator.py"
FORBIDDEN = frozenset({"socket", "subprocess", "urllib", "requests", "evaluator"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forbidden_imports(root: Path = REPO / "src") -> list[str]:
    """AST-level runtime dependency audit; comments and strings do not count."""
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN:
                    found.append(f"{path.relative_to(REPO)}:{node.lineno}:{name}")
    return found


class TranscriptAgent:
    """Transparent recorder excluding evaluator-generated random session ids."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.records: list[dict] = []

    def reset(self, session_id: str, profile: dict) -> None:
        self.records.append({"reset": profile})
        self.agent.reset(session_id, profile)

    def respond(self, session_id: str, message: str, turn: int, top_k: int) -> dict:
        response = self.agent.respond(session_id, message, turn, top_k)
        self.records.append({"message": message, "turn": turn, "top_k": top_k,
                             "response": response})
        return response


def public_transcript(catalog: Path = CATALOG,
                      dataset: Path = PUBLIC) -> tuple[str, dict, object]:
    """Return canonical transcript digest, evaluator result, and built agent."""
    from src.agent import Agent

    samples = load_jsonl(dataset)
    identifiers, categories, products = catalog_index(catalog)
    traced = TranscriptAgent(Agent(catalog))
    result = evaluate(traced, samples, identifiers, categories, products)
    body = json.dumps(traced.records, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest(), result, traced.agent


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def main() -> int:
    parser = argparse.ArgumentParser(description="Operational submission audit")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--dataset", type=Path, default=PUBLIC)
    parser.add_argument("--output", type=Path,
                        default=REPO / "analysis" / "operational_audit.json")
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    digest, result, agent = public_transcript(args.catalog, args.dataset)
    elapsed = time.perf_counter() - start
    if args.hash_only:
        print(digest)
        return 0

    agent.reset("audit_warm", {})
    message = "I'm looking for Tops & Tees T-Shirts. A key requirement is: 100% Cotton."
    agent.respond("audit_warm", message, 1, 10)
    samples: list[float] = []
    for _ in range(25):
        turn_start = time.perf_counter()
        agent.respond("audit_warm", message, 2, 10)
        samples.append((time.perf_counter() - turn_start) * 1000)
    ordered = sorted(samples)
    report = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
        "public_transcript_sha256": digest,
        "evaluator_sha256": sha256(EVALUATOR),
        "forbidden_src_imports": forbidden_imports(),
        "swallowed_turn_failures": agent.failures,
        "public_score": {key: result[key] for key in (
            "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
            "recommended_technical_score", "reported_token_usage")},
        "resources": {
            "full_public_audit_seconds": round(elapsed, 3),
            "audit_process_peak_rss_mb": round(_rss_mb(), 1),
            "warm_turn_p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
            "warm_turn_max_ms": round(max(samples), 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")
    return int(bool(report["forbidden_src_imports"] or report["swallowed_turn_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
