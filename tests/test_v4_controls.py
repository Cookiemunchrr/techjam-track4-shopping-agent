"""V4-0 end-to-end control reports are deterministic and identifier-safe."""
from __future__ import annotations

import os
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import v4_controls


class CompactReportTest(unittest.TestCase):
    def test_bench_compaction_removes_sessions_targets_and_timing(self):
        rows = [{
            "split": "dev", "technical_score": 0.9, "mrr": 0.7,
            "seconds": 12.3,
            "sessions": [{"sample_id": "public_secret", "target": "secret_asin"}],
        }]
        compact = v4_controls.compact_bench(rows)
        self.assertEqual(compact, [{
            "mrr": 0.7, "sample_count": 1, "split": "dev",
            "technical_score": 0.9,
        }])
        self.assertNotIn("public_secret", str(compact))
        self.assertNotIn("secret_asin", str(compact))

    def test_audit_compaction_drops_machine_dependent_resource_timing(self):
        compact = v4_controls.compact_audit({
            "public_transcript_sha256": "a" * 64,
            "public_score": {"recommended_technical_score": 0.916125},
            "resources": {"seconds": 99.0},
        })
        self.assertNotIn("resources", compact)
        self.assertEqual(compact["public_transcript_sha256"], "a" * 64)


class EnvironmentTest(unittest.TestCase):
    def test_control_environment_clears_inherited_knobs_and_pins_width(self):
        with mock.patch.dict(os.environ, {"W_POP": "999", "P_ASK": "none"}):
            normal = v4_controls._environment()
            fixed = v4_controls._environment(fixed_width=True)
        self.assertNotIn("W_POP", normal)
        self.assertNotIn("P_ASK", normal)
        self.assertEqual(normal["PYTHONHASHSEED"], "0")
        self.assertEqual((fixed["P_PROBE"], fixed["P_WIDEN"]), ("10", "1"))


class CompletenessGateTest(unittest.TestCase):
    @staticmethod
    def metric(split, count):
        counts = {
            "dev": {"boundary": 3, "browsing": 44, "buying": 35,
                    "intent_override": 18},
            "holdout": {"boundary": 7, "browsing": 36, "buying": 45,
                        "intent_override": 12},
            "full": {"boundary": 10, "browsing": 80, "buying": 80,
                     "intent_override": 30},
        }[split]
        scenarios = {
            name: {"sample_count": size, "hit_rate_at_10": 1.0,
                   "mrr": 0.7, "mttc": 2.0}
            for name, size in counts.items()
        }
        return {
            "split": split, "paraphrase_level": 0, "sample_count": count,
            "technical_score": 0.9, "hit_rate_at_10": 1.0,
            "mrr": 0.7, "mttc": 2.0, "scenario_metrics": scenarios,
        }

    @staticmethod
    def adversarial(axis):
        row = CompletenessGateTest.metric("full", 200)
        return {"axis": axis, "sample_count": 200, "technical_score": 0.8,
                "hit_rate_at_10": 0.9, "mrr": 0.6, "mttc": 3.0,
                "scenario_metrics": row["scenario_metrics"]}

    @staticmethod
    def shadow(count):
        return {"sample_count": count, "shadow_score": 0.8,
                "retrieval_hit_rate_at_10": 1.0, "retrieval_mrr": 0.6,
                "retrieval_mttc": 2.0}

    def fixture(self):
        official = [self.metric("dev", 100), self.metric("holdout", 100),
                    self.metric("full", 200)]
        axes = [self.adversarial(name) for name in (
            "control", "category", "natural", "scaffold", "constraint",
            "all paraphrase axes", "granularity=1", "granularity=3",
        )]
        clean = self.shadow(200)
        shadows = {
            "dev": self.shadow(100), "holdout": self.shadow(100), "full": clean,
            "full_operational_axes": {
                "clean": clean, "fresh_agent": self.shadow(200),
                "shuffle_seed1": self.shadow(200),
                "shuffle_seed2": self.shadow(200),
            },
        }
        return official, axes, shadows

    def test_complete_frozen_matrix_is_accepted(self):
        official, axes, shadows = self.fixture()
        v4_controls.validate_control_completeness(
            official, list(official), axes, shadows)

    def test_missing_split_axis_or_shadow_fails_closed(self):
        official, axes, shadows = self.fixture()
        mutations = (
            (official[:-1], list(official), axes, shadows),
            (official, list(official), axes[:-1], shadows),
            (official, list(official), axes,
             {**shadows, "holdout": None}),
        )
        for values in mutations:
            with self.subTest(values=str(values)[:80]), self.assertRaises(ValueError):
                v4_controls.validate_control_completeness(*values)

    def test_duplicate_official_or_adversarial_row_fails_closed(self):
        official, axes, shadows = self.fixture()
        mutations = (
            (official + [official[0]], list(official), axes, shadows),
            (official, list(official), axes + [axes[0]], shadows),
        )
        for values in mutations:
            with self.subTest(values=str(values)[:80]), self.assertRaises(ValueError):
                v4_controls.validate_control_completeness(*values)

    def test_malformed_or_compensated_scenario_rows_fail_closed(self):
        official, axes, shadows = self.fixture()
        malformed = [dict(row) for row in official]
        malformed[0] = dict(malformed[0])
        malformed[0]["scenario_metrics"] = dict(
            malformed[0]["scenario_metrics"])
        malformed[0]["scenario_metrics"]["boundary"] = None
        compensated = [dict(row) for row in official]
        compensated[0] = dict(compensated[0])
        compensated[0]["scenario_metrics"] = {
            name: dict(item)
            for name, item in compensated[0]["scenario_metrics"].items()
        }
        compensated[0]["scenario_metrics"]["boundary"]["sample_count"] -= 1
        compensated[0]["scenario_metrics"]["browsing"]["sample_count"] += 1
        for rows in (malformed, compensated):
            with self.subTest(rows=rows[0]["scenario_metrics"]), self.assertRaises(ValueError):
                v4_controls.validate_control_completeness(
                    rows, list(official), axes, shadows
                )

    @mock.patch("tools.v4_controls._validate_report")
    def test_v4_resource_requires_all_three_routes(self, validate):
        with self.assertRaisesRegex(ValueError, "exact, hedged, and fallback"):
            v4_controls.validate_v4_resource({"routes": {"exact": {}, "hedged": {}}})
        validate.assert_called_once()


class ControlArtifactGateTest(unittest.TestCase):
    @staticmethod
    def report():
        official, axes, shadows = CompletenessGateTest().fixture()
        controls = {
            "official": official,
            "fixed_width_10": copy.deepcopy(official),
            "adversarial": axes,
            "shadow": shadows,
        }
        inputs = {
            "control_sources": {
                "evaluator/local_evaluator.py": "e" * 64,
            }
        }
        return {
            "schema": "techjam-v4-end-to-end-controls-v1",
            "inputs": inputs,
            "protected_baseline": {
                "python_hash_seed": "0",
                "public_transcript_sha256": v4_controls.EXPECTED_PUBLIC_TRANSCRIPT,
                "expected_public_transcript_sha256":
                    v4_controls.EXPECTED_PUBLIC_TRANSCRIPT,
                "transcript_match": True,
                "public_score": {
                    "recommended_technical_score": v4_controls.EXPECTED_PUBLIC_SCORE,
                },
                "expected_public_score": v4_controls.EXPECTED_PUBLIC_SCORE,
                "score_match": True,
                "forbidden_src_imports": [],
                "swallowed_turn_failures": 0,
                "evaluator_sha256": "e" * 64,
            },
            "controls": controls,
            "controls_sha256": v4_controls._sha256_json(controls),
            "resource": {},
            "gates": {
                "protected_behavior_parity": True,
                "end_to_end_matrix_complete": True,
                "resource_canary_passed": True,
                "all_v4_0_controls_passed": True,
            },
        }

    def validate(self, report):
        with mock.patch("tools.v4_controls.current_control_inputs",
                        return_value=report["inputs"]), \
                mock.patch("tools.v4_controls.validate_v4_resource"), \
                mock.patch.object(
                    v4_controls, "EXPECTED_BASELINE_CONTROLS_SHA256",
                    report["controls_sha256"],
                ):
            v4_controls.validate_controls_report(report)

    def test_complete_current_green_report_is_accepted(self):
        self.validate(self.report())

    def test_false_missing_or_non_boolean_gate_is_rejected(self):
        for name in list(self.report()["gates"]):
            for value in (False, None, 1):
                report = self.report()
                if value is None:
                    report["gates"].pop(name)
                else:
                    report["gates"][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.validate(report)

    def test_body_input_or_protected_evidence_mutation_is_rejected(self):
        reports = []
        body = self.report()
        body["controls"]["official"][0]["mrr"] = 0.123
        reports.append(body)
        inputs = self.report()
        inputs["inputs"] = {"stale": True}
        reports.append(inputs)
        protected = self.report()
        protected["protected_baseline"]["swallowed_turn_failures"] = 1
        reports.append(protected)
        for report in reports:
            with self.subTest(keys=sorted(report)), self.assertRaises((KeyError, ValueError)):
                self.validate(report)


class BuildValidationTest(unittest.TestCase):
    """build_controls validates its report before the report exists for callers.

    The defect being pinned: the report was returned one statement before
    validate_controls_report(report), so the validation was unreachable and a
    malformed gate was emitted as green.
    """

    def _build(self, validate):
        with tempfile.TemporaryDirectory() as directory:
            resource_path = Path(directory) / "resource.json"
            resource_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(v4_controls, "validate_v4_resource"), \
                    mock.patch.object(v4_controls, "_run_json", return_value={}), \
                    mock.patch.object(v4_controls, "compact_audit", return_value={}), \
                    mock.patch.object(v4_controls, "compact_bench", return_value=[]), \
                    mock.patch.object(v4_controls, "validate_control_completeness"), \
                    mock.patch.object(v4_controls, "acceptable", return_value=True), \
                    mock.patch.object(v4_controls, "current_control_inputs",
                                      return_value={}), \
                    mock.patch.object(v4_controls, "validate_controls_report", validate):
                return v4_controls.build_controls(resource_path)

    def test_a_malformed_report_cannot_be_emitted_as_green(self):
        validate = mock.Mock(side_effect=ValueError("malformed report"))
        with self.assertRaisesRegex(ValueError, "malformed report"):
            self._build(validate)
        validate.assert_called_once()

    def test_the_returned_report_is_the_validated_one(self):
        validate = mock.Mock()
        report = self._build(validate)
        validate.assert_called_once()
        self.assertIs(validate.call_args.args[0], report)
        self.assertEqual(report["schema"], "techjam-v4-end-to-end-controls-v1")


if __name__ == "__main__":
    unittest.main()
