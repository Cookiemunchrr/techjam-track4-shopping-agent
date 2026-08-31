"""The V6 trace-only route record: shape, flags, and serving invariance.

The record is the evidence channel a shelf-transform candidate is later held to
(T08: the baseline ordered pool must be an exact prefix of the candidate pool).
These tests pin that the record exists only when asked for, that its frozen
field shape is what the protocol registers, and that enabling it changes nothing
about what the agent serves.
"""
from __future__ import annotations

import os
import unittest

from src.agent import Agent
from src.routing import TRACK_BUCKETS  # noqa: F401  (import smoke for the route module)
from src.shelf_transform import load as load_transform
from tests.fixtures import PROFILE, RichCatalog


MESSAGE = "I'm looking for Tops & Tees T-Shirts. A key requirement is: 100% Cotton."

FROZEN_FIELDS = {
    "exact", "hedged", "fallback", "prefilter_pool_size", "prefilter_pool_sha256",
    "scored_pool_size", "transform_lookups", "transform_activations",
    "added_shelves", "baseline_prefix_sha256", "baseline_prefix_size",
    # W4 additions: the pre-retrieval prediction and whether it fired.
    "estimated_pool_size", "pre_cutoff",
}


class RouteTraceTest(unittest.TestCase):
    def setUp(self):
        self.ctx = RichCatalog()
        self.agent = Agent(self.ctx.__enter__())
        self.addCleanup(self.ctx.__exit__, None, None, None)

    def _respond(self, agent):
        agent.reset("s", PROFILE)
        return agent.respond("s", MESSAGE, 1, 10)

    def test_off_by_default_records_nothing(self):
        self.assertFalse(self.agent.trace_route)
        self._respond(self.agent)
        self.assertEqual(self.agent.route_trace("s"), {})

    def test_record_shape_and_baseline_zeroes(self):
        self.agent.trace_route = True
        self._respond(self.agent)
        trace = self.agent.route_trace("s")
        self.assertEqual(set(trace), FROZEN_FIELDS)
        self.assertIsInstance(trace["prefilter_pool_sha256"], str)
        self.assertEqual(len(trace["prefilter_pool_sha256"]), 64)
        self.assertGreater(trace["prefilter_pool_size"], 0)
        self.assertGreater(trace["scored_pool_size"], 0)
        # No transform exists yet: the counters the candidate must populate are
        # zero in the baseline, and the baseline prefix digest is null.
        self.assertEqual(trace["transform_lookups"], 0)
        self.assertEqual(trace["transform_activations"], 0)
        self.assertEqual(trace["added_shelves"], [])
        self.assertIsNone(trace["baseline_prefix_sha256"])
        self.assertIsNone(trace["baseline_prefix_size"])
        # An exactly-named shelf in the fixture catalog resolves outright.
        self.assertTrue(trace["exact"])
        self.assertFalse(trace["hedged"])
        self.assertFalse(trace["fallback"])

    def test_tracing_does_not_change_what_is_served(self):
        plain = self._respond(self.agent)
        traced_agent = Agent.sharing_index(self.agent)
        traced_agent.trace_route = True
        traced = self._respond(traced_agent)
        self.assertEqual(plain, traced)

    def test_sharing_index_preserves_the_flag(self):
        self.agent.trace_route = True
        clone = Agent.sharing_index(self.agent)
        self.assertTrue(clone.trace_route)
        clone.trace_route = False
        self._respond(clone)
        self.assertEqual(clone.route_trace("s"), {})


@unittest.skipUnless(os.path.exists("data/catalog.jsonl"), "full catalog not present")
class CandidateIntegrationTest(unittest.TestCase):
    """The shared hook, exercised end to end on the real catalog."""

    CATALOG = "data/catalog.jsonl"

    @classmethod
    def setUpClass(cls):
        cls.base = Agent(cls.CATALOG)

    def _agent(self, mode):
        from unittest import mock
        with mock.patch.dict(os.environ, {"P_SHELF_TRANSFORM": mode}):
            agent = Agent(self.CATALOG)
        agent.trace_route = True
        agent.trace_pool = True
        return agent

    def _turn(self, agent, sid, message):
        agent.reset(sid, PROFILE)
        return agent.respond(sid, message, 1, 10)

    def test_invalid_mode_fails_closed_to_off(self):
        agent = self._agent("nonsense")
        self.assertEqual(agent.shelf_transform.mode, "off")

    def test_exact_route_never_consults_the_transform(self):
        """G4/T03: an exact shelf resolves with zero transform lookups and a
        byte-identical response to the baseline."""
        baseline = self._agent("off")
        candidate = self._agent("aliases")
        self.assertEqual(candidate.shelf_transform.mode, "aliases")
        message = "I'm looking for Tops & Tees T-Shirts. A key requirement is: 100% Cotton."
        b = self._turn(baseline, "b", message)
        c = self._turn(candidate, "c", message)
        self.assertEqual(b, c)
        trace = candidate.route_trace("c")
        self.assertTrue(trace["exact"])
        self.assertEqual(trace["transform_lookups"], 0)
        self.assertEqual(trace["transform_activations"], 0)

    def test_candidate_pool_extends_the_baseline_prefix(self):
        """T08: the baseline ordered pool is an exact prefix of the candidate
        pre-filter pool, and the trace digests prove it."""
        baseline = self._agent("off")
        candidate = self._agent("aliases")
        message = "I'm looking for runners."   # "runners" aliases to sneakers
        self._turn(baseline, "b", message)
        self._turn(candidate, "c", message)
        b, c = baseline.route_trace("b"), candidate.route_trace("c")
        self.assertFalse(b["exact"])
        self.assertEqual(c["transform_lookups"], 1)
        if c["transform_activations"]:
            # The candidate's pre-append pool IS the baseline pool...
            self.assertEqual(c["baseline_prefix_sha256"], b["prefilter_pool_sha256"])
            self.assertEqual(c["baseline_prefix_size"], b["prefilter_pool_size"])
            # ...and the append grew it by exactly the added shelves' members.
            added = sum(len(candidate.catalog.buckets[name])
                        for name in c["added_shelves"])
            self.assertEqual(c["prefilter_pool_size"],
                             c["baseline_prefix_size"] + added)
            self.assertLessEqual(len(c["added_shelves"]), 2)

    def test_sharing_index_carries_the_same_immutable_transform(self):
        """T21: the benchmark agent tests the mode it claims to test."""
        agent = self._agent("aliases")
        clone = Agent.sharing_index(agent)
        self.assertIs(clone.shelf_transform, agent.shelf_transform)
        self.assertEqual(clone.shelf_transform.mode, "aliases")


if __name__ == "__main__":
    unittest.main()
