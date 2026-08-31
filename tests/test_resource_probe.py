"""Fast contract tests for the opt-in V4 route resource probe."""
from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from tools import resource_probe as probe


class PercentileTest(unittest.TestCase):
    def test_nearest_rank_percentiles_are_pinned(self):
        values = list(range(1, 101))
        self.assertEqual(probe.percentile(values, 0.50), 50.0)
        self.assertEqual(probe.percentile(values, 0.95), 95.0)
        self.assertEqual(probe.percentile(values, 0.99), 99.0)

    def test_empty_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            probe.percentile([], 0.99)


class RouteGateTest(unittest.TestCase):
    def summary(self, route, values):
        return probe.summarize(
            route,
            values,
            build_seconds=1.25,
            peak_rss_mb=700.0,
            warmups=3,
        )

    @staticmethod
    def report(route, row):
        row = json.loads(json.dumps(row))
        row["route_shape"] = {
            "exact": route == "exact", "hedged": route == "hedged",
            "pool_size": 100 if route == "fallback" else 10,
            "catalog_size": 100,
            "primary_size": 0 if route == "fallback" else 10,
            "actual_scored_pool_size": 100 if route == "fallback" else 10,
        }
        row["response_sha256"] = "a" * 64
        row["response_deterministic"] = True
        row["measurement_trace"] = "integer_pool_size_only"
        return {
            "schema": probe.REPORT_SCHEMA,
            "protocol": {
                "fresh_process_per_route": True,
                "python_hash_seed": "0",
                "platform": "test-platform",
                "machine": "test-machine",
                "platform_release": "test-release",
                "processor": "test-processor",
                "processor_identity_quality": "model_name",
                "logical_cpu_count": 8,
                "python_version": "test-python",
                "catalog_sha256": "catalog-a",
                "serving_model_sha256": "model-a",
                "source_sha256": {"src/agent.py": "a" * 64},
                "samples_per_route": row["samples"],
                "warmups_per_route": row["warmups"],
                "latency_scope": "respond-only",
                "p99_convention": "nearest-rank",
                "workload_schema": probe.WORKLOAD_SCHEMA,
                "budget_gate": "fixed_workload_zero_exceedance_canary",
                "budget_ms": 50.0,
                "gate_confidence": 0.95,
                "target_violation_rate": 0.01,
                "minimum_gate_samples": 299,
                "effective_runtime": {"fixed": True},
                "iid_assumption_verified": False,
                "serial_samples_per_fresh_process": True,
                "workloads_per_route": 1,
                "route_execution_order": [route],
                "route_workload_sha256": {route: "workload-a"},
            },
            "routes": {route: row},
        }

    def test_exact_and_hedged_pass_only_with_299_zero_violation_samples(self):
        for route in ("exact", "hedged"):
            row = self.summary(route, [1.0] * 299)
            self.assertEqual(
                row["gate"]["kind"], "fixed_workload_zero_exceedance_canary")
            self.assertEqual(row["gate"]["limit_ms"], 50.0)
            self.assertTrue(row["gate"]["enforced"])
            self.assertEqual(row["gate"]["violations"], 0)
            self.assertLess(
                row["gate"]["iid_zero_violation_u95_sensitivity"], 0.01)
            self.assertFalse(row["gate"]["iid_assumption_verified"])
            self.assertTrue(row["gate"]["passed"])

    def test_the_299_sample_minimum_is_the_first_to_put_u95_below_one_percent(self):
        self.assertGreaterEqual(probe.zero_violation_u95(298), 0.01)
        self.assertLess(probe.zero_violation_u95(299), 0.01)
        self.assertAlmostEqual(probe.zero_violation_u95(100), 0.029513, places=6)
        self.assertEqual(probe.DEFAULT_SAMPLES, 299)

    def test_99_fast_and_one_slow_is_not_hidden_by_descriptive_p99(self):
        row = self.summary("hedged", [1.0] * 99 + [1000.0])
        self.assertEqual(row["latency_ms"]["p99"], 1.0)
        self.assertEqual(row["latency_ms"]["max"], 1000.0)
        self.assertEqual(row["gate"]["violations"], 1)
        self.assertFalse(row["gate"]["enforced"])
        self.assertIsNone(row["gate"]["passed"])

    def test_one_violation_fails_an_enforced_budget_gate(self):
        row = self.summary("exact", [1.0] * 298 + [50.0])
        self.assertTrue(row["gate"]["enforced"])
        self.assertEqual(row["gate"]["violations"], 1)
        self.assertFalse(row["gate"]["passed"])

    def test_a_five_sample_route_canary_does_not_claim_a_p99_gate(self):
        row = self.summary("hedged", [10.0, 11.0, 12.0, 13.0, 70.0])
        self.assertEqual(row["latency_ms"]["p99"], 70.0)
        self.assertFalse(row["gate"]["enforced"])
        self.assertIsNone(row["gate"]["passed"])
        self.assertFalse(probe.acceptable({"routes": {"hedged": row}}))

    def test_fallback_is_absolute_only_without_a_same_protocol_baseline(self):
        row = self.summary("fallback", [300.0] * 100)
        self.assertEqual(row["latency_ms"]["p99"], 300.0)
        self.assertEqual(row["gate"]["kind"], "absolute_descriptive_baseline")
        self.assertIsNone(row["gate"]["limit_ms"])
        self.assertIsNone(row["gate"]["passed"])
        self.assertTrue(probe.acceptable({"routes": {"fallback": row}}))

    def test_fallback_comparison_is_descriptive_and_cannot_exit_green(self):
        current = self.report("fallback", self.summary("fallback", [310.0] * 299))
        baseline = self.report("fallback", self.summary("fallback", [300.0] * 299))
        compared = probe.compare_with_baseline(current, baseline)
        gate = compared["routes"]["fallback"]["gate"]
        self.assertEqual(gate["baseline_p99_ms"], 300.0)
        self.assertIsNone(gate["passed"])
        self.assertEqual(gate["comparison_status"], "descriptive_only")
        self.assertNotIn("budget_ms", gate)
        self.assertTrue(compared["comparison"]["fallback"]["response_digest_match"])
        self.assertFalse(probe.acceptable(compared))

    def test_under_sampled_fallback_comparison_cannot_exit_green(self):
        current = self.report("fallback", self.summary("fallback", [310.0] * 5))
        baseline = self.report("fallback", self.summary("fallback", [300.0] * 5))
        self.assertFalse(probe.acceptable(
            probe.compare_with_baseline(current, baseline)))

    def test_non_finite_and_negative_measurements_are_rejected(self):
        for values in ([float("nan")] * 299, [-1.0] * 299,
                       [float("inf")] * 299):
            with self.subTest(values=values[:1]), self.assertRaises(ValueError):
                self.summary("exact", values)

    def test_baseline_comparison_rejects_every_protocol_mismatch(self):
        current = self.report("fallback", self.summary("fallback", [300.0] * 100))
        for key, different in (
            ("fresh_process_per_route", False),
            ("python_hash_seed", "1"),
            ("platform", "different-platform"),
            ("machine", "different-machine"),
            ("platform_release", "different-release"),
            ("processor", "different-processor"),
            ("processor_identity_quality", "architecture_only"),
            ("logical_cpu_count", 16),
            ("python_version", "different-python"),
            ("catalog_sha256", "catalog-b"),
            ("serving_model_sha256", "model-b"),
            ("source_sha256", {"src/agent.py": "b" * 64}),
            ("samples_per_route", 99),
            ("warmups_per_route", 4),
            ("latency_scope", "includes-reset"),
            ("p99_convention", "interpolated"),
            ("workload_schema", 999),
            ("budget_gate", "empirical_p99"),
            ("budget_ms", 51.0),
            ("gate_confidence", 0.90),
            ("target_violation_rate", 0.02),
            ("minimum_gate_samples", 100),
            ("effective_runtime", {"fixed": False}),
            ("iid_assumption_verified", True),
            ("serial_samples_per_fresh_process", False),
            ("workloads_per_route", 2),
            ("route_execution_order", []),
        ):
            baseline = json.loads(json.dumps(current))
            baseline["protocol"][key] = different
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.compare_with_baseline(current, baseline)

        baseline = json.loads(json.dumps(current))
        baseline["schema"] = 999
        with self.assertRaises(ValueError):
            probe.compare_with_baseline(current, baseline)

        baseline = json.loads(json.dumps(current))
        baseline["protocol"]["route_workload_sha256"]["fallback"] = "workload-b"
        with self.assertRaises(ValueError):
            probe.compare_with_baseline(current, baseline)

        baseline = json.loads(json.dumps(current))
        baseline["routes"]["fallback"]["route_shape"][
            "actual_scored_pool_size"] = 11
        with self.assertRaises(ValueError):
            probe.compare_with_baseline(current, baseline)

        baseline = json.loads(json.dumps(current))
        baseline["routes"]["fallback"]["peak_rss_mb"] = -1.0
        with self.assertRaises(ValueError):
            probe.compare_with_baseline(current, baseline)

    def test_forged_green_gate_is_rejected_from_raw_summary_invariants(self):
        report = self.report("exact", self.summary("exact", [1.0] * 299))
        mutations = []
        forged_latency = json.loads(json.dumps(report))
        forged_latency["routes"]["exact"]["latency_ms"].update(
            p99=1000.0, max=1000.0
        )
        mutations.append(forged_latency)
        forged_violations = json.loads(json.dumps(report))
        forged_violations["routes"]["exact"]["gate"].update(
            violations=299, observed_violation_rate=1.0, passed=True
        )
        mutations.append(forged_violations)
        forged_fallback = self.report(
            "fallback", self.summary("fallback", [1.0] * 299)
        )
        forged_fallback["routes"]["fallback"]["gate"].update(
            limit_ms=50.0, enforced=True, passed=True
        )
        mutations.append(forged_fallback)
        for mutation in mutations:
            with self.subTest(route=list(mutation["routes"])), self.assertRaises(ValueError):
                probe._validate_report(mutation, "forged")
            self.assertFalse(probe.acceptable(mutation))


class FreshProcessTest(unittest.TestCase):
    @mock.patch("tools.resource_probe._rss_mb", return_value=100.0)
    @mock.patch("tools.resource_probe._route_shape", return_value={
        "exact": True, "hedged": False, "pool_size": 10,
        "catalog_size": 100, "primary_size": 10,
    })
    def test_timed_probe_records_only_integer_pool_size(self, shape, rss):
        from src import agent as agent_module

        instances = []

        class FakeAgent:
            def __init__(self, catalog):
                self.failures = 0
                self.trace_pool = False
                self.trace_pool_size = False
                instances.append(self)

            def reset(self, session, profile):
                pass

            def respond(self, session, message, turn, top_k):
                return {"message": "ok", "ask_attribute": None,
                        "recommendations": [{"parent_asin": "p1"}]}

            def candidate_pool_size(self, session):
                return 10

        with mock.patch.object(agent_module, "Agent", FakeAgent):
            row = probe.measure_child("exact", "unused.jsonl", samples=3, warmups=1)
        self.assertFalse(instances[0].trace_pool)
        self.assertTrue(instances[0].trace_pool_size)
        self.assertEqual(row["measurement_trace"], "integer_pool_size_only")
        self.assertEqual(row["route_shape"]["actual_scored_pool_size"], 10)

    def test_response_digest_is_canonical_and_rejects_nan(self):
        self.assertEqual(
            probe._response_sha256({"b": 2, "a": [1]}),
            probe._response_sha256({"a": [1], "b": 2}),
        )
        with self.assertRaises(RuntimeError):
            probe._response_sha256({"latency": float("nan")})

    @mock.patch("tools.resource_probe.file_sha256", return_value="catalog-digest")
    @mock.patch("tools.resource_probe.subprocess.run")
    def test_every_route_is_measured_in_its_own_child(self, run, digest):
        def completed(command, **kwargs):
            route = command[command.index("--child") + 1]
            row = probe.summarize(
                route, [1.0] * 100, build_seconds=1.0,
                peak_rss_mb=600.0, warmups=2,
            )
            row["route_shape"] = {
                "exact": route == "exact", "hedged": route == "hedged",
                "pool_size": 100 if route == "fallback" else 10,
                "catalog_size": 100,
                "primary_size": 0 if route == "fallback" else 10,
                "actual_scored_pool_size": 100 if route == "fallback" else 10,
            }
            row["response_sha256"] = "a" * 64
            row["response_deterministic"] = True
            row["measurement_trace"] = "integer_pool_size_only"
            return subprocess.CompletedProcess(command, 0, json.dumps(row), "")

        run.side_effect = completed
        report = probe.probe_routes(
            ("exact", "hedged", "fallback"),
            catalog="data/catalog.jsonl",
            samples=100,
            warmups=2,
        )
        self.assertEqual(run.call_count, 3)
        self.assertEqual(set(report["routes"]), {"exact", "hedged", "fallback"})
        self.assertEqual(report["protocol"]["catalog_sha256"], "catalog-digest")
        self.assertEqual(set(report["protocol"]["route_workload_sha256"]),
                         {"exact", "hedged", "fallback"})
        self.assertEqual(
            set(report["protocol"]["source_sha256"]),
            set(probe.RESOURCE_SOURCE_PATHS),
        )
        self.assertEqual(report["protocol"]["serving_model_sha256"],
                         "catalog-digest")
        self.assertEqual(digest.call_count, 2 + len(probe.RESOURCE_SOURCE_PATHS))
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIn("--child", command)
            self.assertEqual(call.kwargs["env"]["PYTHONHASHSEED"], "0")

    def test_unknown_or_duplicate_routes_are_rejected(self):
        with self.assertRaises(ValueError):
            probe.probe_routes(("unknown",), "data/catalog.jsonl", 1, 0)
        with self.assertRaises(ValueError):
            probe.probe_routes(("exact", "exact"), "data/catalog.jsonl", 1, 0)
if __name__ == "__main__":
    unittest.main()
