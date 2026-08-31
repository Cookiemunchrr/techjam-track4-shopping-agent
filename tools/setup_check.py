"""Verify (and, on request, produce) the working catalog a fresh checkout needs.

`data/catalog.jsonl` is the organizer's frozen catalog, decompressed. It is
gitignored -- it is not ours to redistribute and it is 60 MB -- so a clean clone
has the archive and not the file, and 29 tests skip without saying why.

Deliberately a separate command rather than something `Agent()` does on demand:

  * decompressing inside the constructor would fold a one-off 12-second unpack
    into the cold-start number the README publishes, which is a measurement of
    the agent and should stay one;
  * the scoring environment may mount the checkout read-only, and an agent whose
    first act is to write 60 MB into its own repository fails there for a reason
    that has nothing to do with shopping.

    python3 -m tools.setup_check              # report what is present
    python3 -m tools.setup_check --unpack     # decompress the archive if needed
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ARCHIVE = Path("catalog.jsonl.gz")
CATALOG = Path("data/catalog.jsonl")
# Published in the README Quick Start, and checked here so the two cannot drift.
ARCHIVE_SHA = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"
CATALOG_SHA = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
EXPECTED_ROWS = 50000


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(archive: Path, target: Path) -> None:
    """Decompress to a sibling temporary file, then move it into place.

    Writing straight to the target leaves a half-written catalog behind if this is
    interrupted, and a half-written catalog is worse than no catalog: it loads,
    it is short, and every number measured against it is quietly wrong.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".partial")
    try:
        with gzip.open(archive, "rb") as source, staging.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        staging.replace(target)
    finally:
        if staging.exists():
            staging.unlink()


PUBLIC_SET = Path("data/public_set.jsonl")
DEV_SPLIT = Path("analysis/dev.jsonl")
HOLDOUT_SPLIT = Path("analysis/holdout.jsonl")
SPLIT_AT = 100                  # sessions 1-100 fit, 101-200 validate


def split_status(fix: bool) -> tuple[str, list[str]]:
    """Report -- and on request rebuild -- the dev/holdout slices of the public set.

    Both files are gitignored, so a clean clone does not have them. That is not a
    cosmetic gap: `tools/bench.py` passes over a split whose file is missing
    rather than failing, so an absent split reads as a run that passed and
    measured nothing, and working rule 1 ("never tune on all 200; fit on dev,
    validate on holdout") becomes unenforceable without saying so.

    Regenerating is cheap and exact -- they are the first and second hundred
    lines of `data/public_set.jsonl` in file order, verified here rather than
    assumed.
    """
    if not PUBLIC_SET.exists():
        return "missing", [f"{PUBLIC_SET} is absent; it ships with the repository"]

    lines = PUBLIC_SET.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 * SPLIT_AT:
        return "missing", [f"{PUBLIC_SET} has {len(lines)} sessions, expected {2 * SPLIT_AT}"]

    problems: list[str] = []
    state = "ok"
    for path, wanted in ((DEV_SPLIT, lines[:SPLIT_AT]), (HOLDOUT_SPLIT, lines[SPLIT_AT:])):
        body = "\n".join(wanted) + "\n"
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == body:
            print(f"sessions {path}  {len(wanted)} sessions")
            continue
        # The two failures are different and the second is worse. An absent split
        # makes bench pass over it -- a run that measured nothing, reported as a
        # run that passed. A stale one is measured, and reports a number for
        # sessions that are not the split it names.
        label, consequence = (("absent", "bench will silently skip this split")
                              if current is None else
                              ("stale", "bench will measure the wrong sessions"))
        if not fix:
            print(f"sessions {path}  {label} ({consequence})")
            problems.append(f"{path} is {label}; run: python3 -m tools.setup_check --splits")
            state = "problem"
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        print(f"sessions {path}  rebuilt, {len(wanted)} sessions")
        state = "rebuilt" if state != "problem" else state
    return state, problems


def rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the catalog a fresh checkout needs")
    parser.add_argument("--unpack", action="store_true",
                        help="decompress the archive when the catalog is missing")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--verify-checksums", action="store_true",
                        help="hash the files as well as counting them (slow)")
    parser.add_argument("--splits", action="store_true",
                        help="rebuild analysis/dev.jsonl and analysis/holdout.jsonl "
                             "from data/public_set.jsonl when absent or stale")
    args = parser.parse_args()

    problems: list[str] = []

    if not args.catalog.exists():
        if not args.unpack:
            print(f"missing  {args.catalog}")
            if args.archive.exists():
                print(f"         run: python3 -m tools.setup_check --unpack")
            else:
                print(f"         and {args.archive} is not here either -- see the README "
                      f"Quick Start for the download URL")
            return 1
        if not args.archive.exists():
            print(f"cannot unpack: {args.archive} is not here. See the README Quick Start.")
            return 1
        print(f"unpacking {args.archive} -> {args.catalog} ...")
        unpack(args.archive, args.catalog)

    size_mb = args.catalog.stat().st_size / 1048576
    count = rows(args.catalog)
    print(f"catalog  {args.catalog}  {size_mb:.0f} MB  {count} rows")
    if count != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, found {count}")

    with args.catalog.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    for field in ("parent_asin", "title", "categories"):
        if field not in first:
            problems.append(f"first row has no {field!r}")

    if args.verify_checksums:
        for path, expected in ((args.catalog, CATALOG_SHA), (args.archive, ARCHIVE_SHA)):
            if not path.exists():
                continue
            actual = sha256(path)
            mark = "ok" if actual == expected else "MISMATCH"
            print(f"sha256   {path}  {actual[:8]}…  {mark}")
            if actual != expected:
                problems.append(f"{path} checksum does not match the published one")

    state = "present" if PUBLIC_SET.exists() else "absent (some benchmarks will skip)"
    print(f"sessions {PUBLIC_SET}  {state}")
    _, split_problems = split_status(args.splits)
    problems.extend(split_problems)

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nready: python3 -m evaluator.local_evaluator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
