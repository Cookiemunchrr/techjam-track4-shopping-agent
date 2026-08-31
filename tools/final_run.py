"""Run the official final evaluation and write the retention bundle the FAQ requires.

`docs/final_evaluation_faq.md` section 1 puts four obligations on us, and this tool
exists so none of them depends on remembering something on the day:

    - run the UNMODIFIED official evaluator
    - run it against the frozen SUBMITTED COMMIT
    - retain the generated `results.json`, INCLUDING PER-SESSION RESULTS
    - retain it together with the commit hash and relevant environment and
      execution details, because "the organizer may request logs or other
      supporting evidence"

`results.json` is gitignored in this repository (it is a generated artifact, and the
public tree should not carry a stale copy). That is correct for development and
actively dangerous on final-evaluation day, so this tool copies it into a bundle
directory that is NOT ignored, next to a manifest recording everything above.

Usage, once the organizer releases the final package::

    python3 -m tools.final_run --dataset path/to/final_set.jsonl

Add --allow-dirty only if you understand that the FAQ freezes the submitted commit
and that a dirty tree means the thing you ran is not the thing you submitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALUATOR = REPO / "evaluator" / "local_evaluator.py"
DEFAULT_CATALOG = REPO / "data" / "catalog.jsonl"
UPSTREAM_REMOTE = "upstream"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return ""


def rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(raw / 1e6 if sys.platform == "darwin" else raw / 1e3, 1)


def evaluator_is_unmodified() -> dict:
    """Compare evaluator/ against the organizer remote, if it is configured.

    The FAQ requires the UNMODIFIED official evaluator. We cannot prove that
    offline against a package we have not fetched, so we report what we can:
    the local digest, and whether it matches the organizer remote's tip.
    """
    local = sha256(EVALUATOR)
    remotes = git("remote").split()
    if UPSTREAM_REMOTE not in remotes:
        return {"local_sha256": local, "compared_to_upstream": False,
                "note": f"no '{UPSTREAM_REMOTE}' remote configured; digest recorded "
                        "but not compared. Add the organizer repo as a remote and "
                        "re-run to get a real comparison."}
    git("fetch", "--quiet", UPSTREAM_REMOTE)
    diff = git("diff", "--stat", f"{UPSTREAM_REMOTE}/main", "HEAD", "--", "evaluator/")
    return {"local_sha256": local, "compared_to_upstream": True,
            "identical_to_upstream_main": diff == "",
            "diff_stat": diff or None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Official final evaluation + retention bundle")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="the organizer's released final session file")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--bundle", type=Path, default=REPO / "final_evaluation",
                    help="directory to write the retention bundle into")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even though the working tree differs from the commit")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        print("The final package is released AFTER the Devpost deadline; there is "
              "nothing to run before then.", file=sys.stderr)
        return 2
    if not args.catalog.exists():
        print(f"catalog not found: {args.catalog}\n"
              "run: python3 -m tools.setup_check --unpack --splits", file=sys.stderr)
        return 2

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print("REFUSING TO RUN: the working tree has uncommitted changes.\n"
              "The FAQ freezes the submitted commit; a dirty tree means the thing you "
              "ran is not the thing you submitted. Commit, stash, or pass "
              "--allow-dirty deliberately.\n\n" + dirty, file=sys.stderr)
        return 1

    if args.bundle.exists():
        print(f"REFUSING TO OVERWRITE an existing bundle: {args.bundle}\n"
              "Final-evaluation evidence is written once. Move it aside if you "
              "genuinely need to re-run.", file=sys.stderr)
        return 1

    results_path = REPO / "results.json"
    previous = results_path.read_bytes() if results_path.exists() else None

    print(f"running the official evaluator on {args.dataset} ...")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "evaluator.local_evaluator",
         "--catalog", str(args.catalog), "--dataset", str(args.dataset),
         "--output", str(results_path)],
        cwd=str(REPO), capture_output=True, text=True)
    elapsed = round(time.time() - started, 3)

    if proc.returncode != 0:
        if previous is not None:
            results_path.write_bytes(previous)
        sys.stderr.write(proc.stdout + proc.stderr)
        print(f"\nevaluator exited {proc.returncode}; nothing retained.", file=sys.stderr)
        return proc.returncode

    results = json.loads(results_path.read_text())
    sessions = results.get("sessions") or []

    args.bundle.mkdir(parents=True)
    (args.bundle / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    (args.bundle / "evaluator_stdout.log").write_text(proc.stdout + proc.stderr)

    manifest = {
        "schema": "techjam-final-evaluation-retention-v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "obligation": "docs/final_evaluation_faq.md section 1",
        "submitted_commit": {
            "sha": git("rev-parse", "HEAD"),
            "subject": git("log", "-1", "--format=%s"),
            "committed_at": git("log", "-1", "--format=%cI"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean": not dirty,
            "dirty_files": dirty.splitlines() if dirty else [],
        },
        "official_evaluator": evaluator_is_unmodified(),
        "inputs": {
            "dataset": str(args.dataset),
            "dataset_sha256": sha256(args.dataset),
            "dataset_sessions": len(sessions),
            "catalog": str(args.catalog),
            "catalog_sha256": sha256(args.catalog),
        },
        "headline": {k: results.get(k) for k in
                     ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
                      "recommended_technical_score", "reported_token_usage")},
        "scenario_metrics": results.get("scenario_metrics"),
        "per_session_results_retained": len(sessions),
        "execution": {
            "wall_seconds": elapsed,
            "peak_child_rss_mb": rss_mb(),
            "seconds_per_session": round(elapsed / len(sessions), 4) if sessions else None,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": sys.platform,
            "platform_release": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "third_party_dependencies": "none; Python standard library only "
                                        "(requirements.txt is empty)",
        },
        "network_and_credentials": {
            "llm_stage_enabled": bool(__import__("os").environ.get("TECHJAM_LLM_RERANK") == "1"
                                      and __import__("os").environ.get("AIAND_API_KEY")),
            "note": "The default path opens no socket and spends no tokens. If the "
                    "LLM stage was enabled for this run, reported token usage above "
                    "is non-zero and the run is NOT the default configuration.",
            "reported_token_usage": results.get("reported_token_usage"),
        },
    }
    (args.bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nscore    : {results.get('recommended_technical_score')}")
    print(f"sessions : {len(sessions)} (per-session results retained)")
    print(f"commit   : {manifest['submitted_commit']['sha']}")
    print(f"bundle   : {args.bundle}")
    print("\nRetain this directory. The organizer may request it as supporting evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
