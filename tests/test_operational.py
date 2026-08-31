"""Group G — operational limits.

docs/submission_rules.md: "The organizer reserves the right to run your submission
under CPU, memory, timeout, and network restrictions." Each of these is therefore
a scoring risk, and each one needs a stated budget rather than a vague assurance.
"""
from __future__ import annotations

import gc
import os
import re
import resource
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from src.agent import Agent
from src.policy import TOP_K
from tests.fixtures import PROFILE, RichCatalog

REPO = Path(__file__).resolve().parents[1]
CATALOG = str(REPO / "data" / "catalog.jsonl")

# Budgets. Set above the measured value with headroom, not at it.
# Environment note (V6 Phase 0): these absolute values are machine-local by
# construction. The original machine measured the build at 6.2 s; the V6
# execution machine (WSL2, Python 3.10) measures 21 s at full clock and 33-36 s
# throttled, so the build budget is re-derived here at 60 s -- 1.7x above the
# worst observed, same convention as the original 4x-above-measured. The
# registered V6 resource canary (tools/resource_probe.py, zero >=50 ms on fresh
# serial workloads) is NOT re-derived: it runs on a quiet, full-clock machine,
# and its gate values are unchanged.
BUILD_SECONDS = 60.0      # original 25.0 (measured 6.2 s); here 21-36 s
TURN_MILLIS = 50.0        # measured ~1 ms
PEAK_RSS_MB = 1200.0      # measured 632 MB
F9_TRANSCRIPT_SHA256 = (
    "7c30023e3b8f951d35f8a449a066cfff45f8995d8d9994142e0e5929f8958d04"
)


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class ResourceTest(unittest.TestCase):
    """G1, G2, G3 — cold start, per-turn latency, peak memory."""

    @classmethod
    def setUpClass(cls):
        gc.collect()
        start = time.perf_counter()
        cls.agent = Agent(CATALOG)
        cls.build_seconds = time.perf_counter() - start
        cls.peak_mb = _rss_mb()

    def test_g1_cold_start_within_budget(self):
        self.assertLess(self.build_seconds, BUILD_SECONDS,
                        f"index build took {self.build_seconds:.1f}s")

    def test_g2_per_turn_latency_within_budget(self):
        self.agent.reset("lat", PROFILE)
        message = "I'm looking for Tops & Tees T-Shirts. A key requirement is: 100% Cotton."
        self.agent.respond("lat", message, 1, TOP_K)      # warm
        start = time.perf_counter()
        for _ in range(20):
            self.agent.respond("lat", message, 2, TOP_K)
        millis = (time.perf_counter() - start) / 20 * 1000
        self.assertLess(millis, TURN_MILLIS, f"per-turn latency {millis:.1f}ms")

    def test_g3_peak_rss_within_ceiling(self):
        """In its own process: other tests in this run have already built catalogs,
        and a shared-process reading measures the suite, not the agent."""
        import subprocess
        # VmHWM, not getrusage: under Linux's vfork-style spawn the child's
        # inherited mm can report the *parent's* high-water as its own maxrss
        # (observed: 1731 MB reported for a process whose VmHWM was 660 MB).
        # /proc/self/status VmHWM is per-mm and immune to it. getrusage stays
        # as the fallback where /proc does not exist (macOS).
        body = (
            "import sys\n"
            "from src.agent import Agent\n"
            "Agent(%r)\n"
            "peak = None\n"
            "try:\n"
            "    with open('/proc/self/status') as fh:\n"
            "        for line in fh:\n"
            "            if line.startswith('VmHWM'):\n"
            "                peak = int(line.split()[1]) / 1e3\n"
            "except OSError:\n"
            "    pass\n"
            "if peak is None:\n"
            "    import resource\n"
            "    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            "    peak = raw / 1e6 if sys.platform == 'darwin' else raw / 1e3\n"
            "print(peak)\n"
        ) % CATALOG
        done = subprocess.run([sys.executable, "-c", body], cwd=str(REPO),
                              env=dict(os.environ, PYTHONPATH=str(REPO)),
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        peak = float(done.stdout.strip().splitlines()[-1])
        self.assertLess(peak, PEAK_RSS_MB, f"peak RSS {peak:.0f}MB for one agent")


class MemoryGrowthTest(unittest.TestCase):
    """G4 — the private set is 800 sessions and nothing evicts session state."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.agent = Agent(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_g4_session_store_stays_bounded_over_many_sessions(self):
        for index in range(1000):
            session = f"bulk{index}"
            self.agent.reset(session, PROFILE)
            self.agent.respond(session, "I'm looking for Accessories Belts.", 1, TOP_K)
        self.assertLessEqual(
            len(self.agent.sessions), 512,
            f"{len(self.agent.sessions)} sessions retained; nothing evicts them")

    def test_g4b_a_long_session_does_not_grow_state_without_bound(self):
        self.agent.reset("long", PROFILE)
        for turn in range(1, 11):
            self.agent.respond("long", f"For that, what matters is: detail number {turn}.", turn, TOP_K)
        state = self.agent.sessions["long"]
        held = getattr(state, "clauses", None)
        if held is None:
            held = state.dialog.active()
        self.assertLessEqual(len(held), 24, "unbounded constraint accumulation")


class PythonVersionTest(unittest.TestCase):
    """G6 — the README declares a minimum; it has to be true.

    Read the floor out of the README rather than hard-coding it, so the test keeps
    testing something when the declared minimum moves. The earlier form pinned
    "3.10+" literally and skipped itself the day the claim became 3.9+, which is a
    test reporting green for having stopped looking.
    """

    def test_g6_declared_minimum_matches_reality(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        declared = re.findall(r"Python (\d+)\.(\d+)\+", readme)
        self.assertTrue(declared, "the README no longer declares a minimum Python version")
        floor = min((int(major), int(minor)) for major, minor in declared)
        self.assertGreaterEqual(
            sys.version_info[:2], floor,
            f"README declares Python {floor[0]}.{floor[1]}+ but the suite is passing on "
            f"{sys.version_info.major}.{sys.version_info.minor}; fix the claim or the floor")


class SplitReproducibilityTest(unittest.TestCase):
    """The dev/holdout split has to be rebuildable, or working rule 1 is unenforceable.

    Both files are gitignored, so a clean clone does not have them, and
    `tools/bench.py` passes over a split whose file is missing rather than
    failing -- an absent split reads as a run that passed and measured nothing.
    "Fit on dev, validate on holdout" then quietly stops being checked.
    """

    def test_the_splits_are_the_first_and_second_hundred_sessions(self):
        """Pins the definition `tools.setup_check.split_status` rebuilds from.

        Skipped rather than failed when the files are absent: that is the state a
        clean checkout is legitimately in, and it is what --splits is for.
        """
        from tools.setup_check import DEV_SPLIT, HOLDOUT_SPLIT, PUBLIC_SET, SPLIT_AT
        if not (PUBLIC_SET.exists() and DEV_SPLIT.exists() and HOLDOUT_SPLIT.exists()):
            self.skipTest("splits absent; run python3 -m tools.setup_check --splits")
        public = PUBLIC_SET.read_text(encoding="utf-8").splitlines()
        self.assertEqual(DEV_SPLIT.read_text(encoding="utf-8").splitlines(),
                         public[:SPLIT_AT])
        self.assertEqual(HOLDOUT_SPLIT.read_text(encoding="utf-8").splitlines(),
                         public[SPLIT_AT:])

    def test_the_splits_do_not_overlap_and_cover_the_public_set(self):
        """Fitting and validating on the same session would invalidate every holdout number."""
        from tools.setup_check import DEV_SPLIT, HOLDOUT_SPLIT
        import json
        if not (DEV_SPLIT.exists() and HOLDOUT_SPLIT.exists()):
            self.skipTest("splits absent; run python3 -m tools.setup_check --splits")
        def ids(path):
            return {json.loads(line)["sample_id"]
                    for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        dev, holdout = ids(DEV_SPLIT), ids(HOLDOUT_SPLIT)
        self.assertEqual(dev & holdout, set(), "a session is in both dev and holdout")
        self.assertEqual(len(dev), 100)
        self.assertEqual(len(holdout), 100)


class OfflineIsolationTest(unittest.TestCase):
    """NEW-B12.2..4 — runtime isolation is demonstrated, not assumed."""

    def test_multi_session_evaluation_never_opens_a_socket(self):
        from evaluator.local_evaluator import catalog_index, evaluate

        with RichCatalog() as path:
            identifiers, categories, products = catalog_index(path)
            agent = Agent(path)
            samples = [
                {"sample_id": "offline_buy", "scenario_type": "buying",
                 "user_profile": PROFILE,
                 "ground_truth": {"parent_asin": "R_BELT_LEATHER"}},
                {"sample_id": "offline_browse", "scenario_type": "browsing",
                 "user_profile": PROFILE,
                 "ground_truth": {"parent_asin": "R_SCARF_SILK"}},
            ]
            with mock.patch("socket.socket", side_effect=AssertionError("network access")) as sock:
                result = evaluate(agent, samples, identifiers, categories, products)
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(agent.failures, 0)
            sock.assert_not_called()

    def test_full_session_needs_no_file_access_after_construction(self):
        with RichCatalog() as path:
            agent = Agent(path)
        agent.reset("sealed", PROFILE)
        blocked = AssertionError("post-construction file access")
        with mock.patch("builtins.open", side_effect=blocked) as builtins_open, \
                mock.patch("io.open", side_effect=blocked) as io_open:
            for turn in range(1, 11):
                response = agent.respond(
                    "sealed", "I'm looking for Accessories Belts. "
                    "A key requirement is: 100% Leather.", turn, TOP_K)
                self.assertTrue(response["recommendations"])
        builtins_open.assert_not_called()
        io_open.assert_not_called()
        self.assertEqual(agent.failures, 0)

    def test_src_has_no_network_process_or_evaluator_imports(self):
        from tools.audit import forbidden_imports

        self.assertEqual(forbidden_imports(), [])


@unittest.skipUnless(os.path.exists(CATALOG), "full catalog not present")
class HashSeedAuditTest(unittest.TestCase):
    """NEW-B12.1 — the full public transcript is stable across hash seeds."""

    def test_full_public_transcript_hash_matches_under_three_seeds(self):
        import subprocess

        hashes = set()
        env = dict(os.environ, PYTHONPATH=str(REPO))
        for seed in ("1", "2", "777"):
            done = subprocess.run(
                [sys.executable, "-m", "tools.audit", "--hash-only"],
                cwd=str(REPO), env=dict(env, PYTHONHASHSEED=seed),
                capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0,
                             f"seed {seed} failed:\nstdout={done.stdout}\nstderr={done.stderr}")
            hashes.add(done.stdout.strip().splitlines()[-1])
        self.assertEqual(len(hashes), 1, f"full transcript varies by hash seed: {hashes}")
        self.assertEqual(
            hashes, {F9_TRANSCRIPT_SHA256},
            "V4-0 confidence plumbing changed the protected f9b1357 public trace",
        )


if __name__ == "__main__":
    unittest.main()
