"""Offline guards for tools/llm_rerank_probe.py.

The probe's network half is exercised once, at measurement time, with the whole
transcript cached. What the suite can guard forever is the half that must be
exact: the order parsing, and the two places the probe re-implements
snapshot_mrr arithmetic (cited in the probe's docstring). If either drifts,
the LLM result stops being comparable to the linear reranker's record, which
is the entire point of the instrument.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import snapshot_mrr  # noqa: E402
from tools.llm_rerank_probe import parse_order, paired_dicts, rr_with_credits  # noqa: E402


def _group(rows):
    return {"sample_id": "s1", "rows": rows,
            "live_rows": rows, "turn": 1, "reference_message": "m"}


ROWS = [
    {"pid": "A", "s": 3.0, "y": 0, "x": [0.1]},
    {"pid": "B", "s": 2.0, "y": 1, "x": [0.9]},
    {"pid": "C", "s": 1.0, "y": 0, "x": [0.5]},
]


class _Shim:
    """A model in the snapshot_mrr sense: blend plus a per-feature-vector score."""

    def __init__(self, blend, weights):
        self.blend = blend
        self.weights = weights

    def score_vector(self, x):
        return self.weights[x[0]]


class ParseOrderTest(unittest.TestCase):
    def test_valid_order_passes_through(self):
        self.assertEqual(parse_order('{"order": ["B", "A", "C"]}', ["A", "B", "C"]),
                         ["B", "A", "C"])

    def test_unknown_ids_dropped_and_missing_appended_in_base_order(self):
        self.assertEqual(parse_order('{"order": ["C", "ZZZ"]}', ["A", "B", "C"]),
                         ["C", "A", "B"])

    def test_fenced_json_accepted(self):
        self.assertEqual(parse_order('```json\n{"order": ["C", "A", "B"]}\n```',
                                     ["A", "B", "C"]), ["C", "A", "B"])

    def test_malformed_reply_is_a_declined_ballot(self):
        self.assertEqual(parse_order("I cannot rank these.", ["A", "B", "C"]),
                         ["A", "B", "C"])


class ArithmeticEquivalenceTest(unittest.TestCase):
    """rr_with_credits fed a model's own contributions is reciprocal_rank."""

    def test_bit_exact_against_the_harness(self):
        shim = _Shim(blend=0.05, weights={0.1: 4.0, 0.9: -2.0, 0.5: 0.25})
        group = _group(ROWS)
        credits = {row["pid"]: shim.score_vector(row["x"]) for row in ROWS}
        self.assertEqual(rr_with_credits(group, credits, shim.blend),
                         snapshot_mrr.reciprocal_rank(group, shim))

    def test_base_order_needs_no_resorting(self):
        group = _group(ROWS)
        self.assertEqual(rr_with_credits(group, {}, 0.5),
                         snapshot_mrr.reciprocal_rank(group, None))

    def test_promoting_the_target_to_first_scores_one(self):
        group = _group(ROWS)
        self.assertEqual(rr_with_credits(group, {"B": 100.0}, 1.0), 1.0)


class BootstrapParityTest(unittest.TestCase):
    def test_paired_dicts_matches_snapshot_mrr_paired(self):
        groups = [_group(ROWS), _group(list(reversed(ROWS)))]
        theirs = snapshot_mrr.paired(groups, None, None)
        base = snapshot_mrr.session_scores(groups, None)
        mine = paired_dicts(base, base)
        self.assertEqual(list(theirs), list(mine))


if __name__ == "__main__":
    unittest.main()
