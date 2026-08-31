"""Group A — contract conformance the existing suite does not cover.

A1-A10 and A14 live in tests/test_agent_contract.py. The cases here are the ones
that were missing, and A11/A12 are the two that would actually have fired: both
organizer documents publish the interface as a class with `reset` and `respond`
and no constructor argument, so a private harness may well call `Agent()` from
its own working directory.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.agent import Agent
from src.policy import TOP_K
from tests.fixtures import PROFILE, RichCatalog

REPO = Path(__file__).resolve().parents[1]


class ConstructionTest(unittest.TestCase):
    """A11, A12 — the agent must be loadable the way the contract shows it."""

    def _run(self, body: str, cwd: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONPATH=str(REPO))
        env.update(env_extra or {})
        return subprocess.run([sys.executable, "-c", body], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=180)

    @unittest.skipUnless((REPO / "data" / "catalog.jsonl").exists(), "full catalog not present")
    def test_a11_constructs_with_no_arguments_from_an_arbitrary_cwd(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            done = self._run("from starter.agent import Agent; Agent(); print('ok')", elsewhere)
        self.assertIn("ok", done.stdout, f"Agent() failed from another cwd:\n{done.stderr}")

    @unittest.skipUnless((REPO / "data" / "catalog.jsonl").exists(), "full catalog not present")
    def test_a12_full_turn_from_an_arbitrary_cwd(self):
        body = (
            "from starter.agent import Agent\n"
            "a = Agent()\n"
            "a.reset('s', {})\n"
            "r = a.respond('s', \"I'm looking for Accessories Belts.\", 1, 10)\n"
            "assert isinstance(r['recommendations'], list) and r['recommendations']\n"
            "print('ok')\n"
        )
        with tempfile.TemporaryDirectory() as elsewhere:
            done = self._run(body, elsewhere)
        self.assertIn("ok", done.stdout, f"a full turn failed from another cwd:\n{done.stderr}")

    @unittest.skipUnless((REPO / "data" / "catalog.jsonl").exists(), "full catalog not present")
    def test_env_var_overrides_the_catalog_location(self):
        body = "from starter.agent import Agent; print(Agent().catalog.size)"
        with tempfile.TemporaryDirectory() as elsewhere:
            done = self._run(body, elsewhere,
                             {"TECHJAM_CATALOG": str(REPO / "data" / "catalog.jsonl")})
        self.assertTrue(done.stdout.strip().isdigit(), done.stderr)

    def test_a_missing_catalog_names_every_path_it_tried(self):
        """A silent FileNotFoundError on a relative path is unhelpful under a harness."""
        with tempfile.TemporaryDirectory() as elsewhere:
            done = self._run(
                "from src.agent import Agent\n"
                "try:\n"
                "    Agent('definitely-not-here.jsonl')\n"
                "except Exception as exc:\n"
                "    print(type(exc).__name__, str(exc))\n",
                elsewhere)
        self.assertIn("definitely-not-here.jsonl", done.stdout + done.stderr)


class SessionOrderingTest(unittest.TestCase):
    """A13 — the harness owns the turn counter; we must not trust it blindly."""

    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.agent = Agent(cls._ctx.__enter__())

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_a13_out_of_order_turn_numbers_do_not_corrupt_state(self):
        self.agent.reset("ooo", PROFILE)
        for turn in (3, 1, 7, 2, 10):
            response = self.agent.respond("ooo", "I'm looking for Accessories Belts.", turn, TOP_K)
            self.assertIsInstance(response["recommendations"], list)

    def test_a13b_turn_zero_and_negative_turns_are_survivable(self):
        self.agent.reset("neg", PROFILE)
        for turn in (0, -1, 99):
            response = self.agent.respond("neg", "Something about belts.", turn, TOP_K)
            self.assertIsInstance(response["message"], str)

    def test_a8_deterministic_across_two_fresh_processes(self):
        """A8 — the existing suite checks determinism inside one process only."""
        catalog = getattr(self, "path", None)
        body = (
            "from src.agent import Agent\n"
            "from tests.fixtures import RichCatalog\n"
            "with RichCatalog() as p:\n"
            "    a = Agent(p)\n"
            "    a.reset('d', {})\n"
            "    r = a.respond('d', \"I'm looking for Accessories Belts. A key requirement is: Suede.\", 1, 10)\n"
            "    print([x['parent_asin'] for x in r['recommendations']])\n"
        )
        env = dict(os.environ, PYTHONPATH=str(REPO))
        runs = set()
        for seed in ("0", "1"):
            done = subprocess.run([sys.executable, "-c", body], cwd=str(REPO),
                                  env=dict(env, PYTHONHASHSEED=seed),
                                  capture_output=True, text=True, timeout=180)
            self.assertEqual(done.returncode, 0, done.stderr)
            runs.add(done.stdout.strip())
        self.assertEqual(len(runs), 1, f"output varies with PYTHONHASHSEED: {runs}")
        del catalog


if __name__ == "__main__":
    unittest.main()
