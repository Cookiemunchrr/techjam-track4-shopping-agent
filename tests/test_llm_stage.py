"""W2 — the opt-in LLM ranking stage: guard, reorder, and the fallback matrix.

The stage is off unless TECHJAM_LLM_RERANK=1 AND a key is present; when active,
every failure mode returns the incoming linear order untouched. The transport
is stubbed through sys.modules, so these tests are offline. The live demo run
is recorded separately in analysis/v8_w2_llm_stage.json.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from src.agent import Agent
from src.catalog import Catalog
from src.policy import TOP_K
from tests.fixtures import PROFILE, RichCatalog

FLAGS = {"TECHJAM_LLM_RERANK": "1", "AIAND_API_KEY": "test-key-not-real"}


def fake_client(payload=None, raises=None):
    module = types.ModuleType("tools.llm_client")
    module.DEFAULT_MODEL = "test-model"
    module.parse_order = lambda content, valid: list(valid)

    def chat_once(body, cache, cache_path, timeout=90, retries=1):
        chat_once.calls.append({"body": body, "timeout": timeout})
        if raises is not None:
            raise raises
        return payload if payload is not None else {
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
    chat_once.calls = []
    module.chat_once = chat_once
    return module


class StageBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = RichCatalog()
        cls.path = cls._ctx.__enter__()
        cls.catalog = Catalog(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def slate(self, agent, session="s"):
        out = agent.respond(session, "I'm looking for Accessories Belts. "
                                     "A key requirement is: Leather.", 1, TOP_K)
        return [item["parent_asin"] for item in out["recommendations"]]


class GuardTest(StageBase):
    def test_w2_default_path_never_imports_the_stage_or_the_client(self):
        sys.modules.pop("src.llm_rank", None)
        sys.modules.pop("tools.llm_client", None)
        # The environment is controlled in full: the key may legitimately exist
        # in a developer shell, and the guard must hold anyway.
        for env in ({}, {"TECHJAM_LLM_RERANK": "1"}, {"AIAND_API_KEY": "x"},
                    {"TECHJAM_LLM_RERANK": "0", "AIAND_API_KEY": "x"}):
            with mock.patch.dict(os.environ, env, clear=True):
                agent = Agent(self.path)
                agent.reset("g", PROFILE)
                self.slate(agent, "g")
                self.assertNotIn("tools.llm_client", sys.modules,
                                 f"client imported under {env}")
                self.assertNotIn("src.llm_rank", sys.modules,
                                 f"stage imported under {env}")

    def test_w2_missing_key_means_inactive_even_with_the_flag(self):
        from src import llm_rank
        with mock.patch.dict(os.environ, {"TECHJAM_LLM_RERANK": "1"}, clear=True):
            self.assertFalse(llm_rank.active())


class ReorderTest(StageBase):
    def test_w2_with_flags_and_a_stubbed_transport_the_stage_runs(self):
        client = fake_client(payload={"choices": [{"message": {"content":
                                      '{"order": ["R_BELT_LEATHER"]}'}}]})
        with mock.patch.dict(os.environ, FLAGS), \
                mock.patch.dict(sys.modules, {"tools.llm_client": client}):
            agent = Agent(self.path)
            agent.reset("l", PROFILE)
            self.slate(agent, "l")
        self.assertTrue(client.chat_once.calls, "the stage never called the model")
        body = client.chat_once.calls[0]["body"]
        self.assertIn("R_BELT_LEATHER", str(body),
                      "the model was not shown the session's candidates")
        self.assertEqual(body["temperature"], 0)

    def test_w2_the_reorder_is_real_when_scores_are_within_blend(self):
        from src import llm_rank
        # Base scores 0.03 apart: inside the 0.05 blend, so the model's order
        # decides. (The D11 finding in miniature: it cannot overcome real gaps.)
        ranked = [(1.00, "R_BELT_LEATHER"), (0.97, "R_BELT_CANVAS")]
        client = fake_client()
        client.parse_order = lambda content, valid: ["R_BELT_CANVAS", "R_BELT_LEATHER"]
        client.chat_once = lambda body, cache, cache_path, timeout=90, retries=1: {
            "choices": [{"message": {"content": '{"order": []}'}}]}
        with mock.patch.dict(os.environ, FLAGS), \
                mock.patch.dict(sys.modules, {"tools.llm_client": client}):
            out = llm_rank.apply(list(ranked), category="Accessories Belts",
                                 phrases=["leather"], catalog=self.catalog, turn=1)
        self.assertEqual(out[0][1], "R_BELT_CANVAS",
                         "the model's order did not move a within-blend pair")

    def test_w2_the_timeout_is_set_on_every_call(self):
        from src import llm_rank
        ranked = [(1.0, "R_BELT_LEATHER"), (0.9, "R_BELT_CANVAS")]
        client = fake_client(payload={"choices": [{"message": {"content":
                                      '{"order": ["R_BELT_CANVAS", "R_BELT_LEATHER"]}'}}]})
        client.parse_order = lambda content, valid: ["R_BELT_CANVAS", "R_BELT_LEATHER"]
        with mock.patch.dict(os.environ, FLAGS), \
                mock.patch.dict(sys.modules, {"tools.llm_client": client}):
            llm_rank.apply(ranked, category="Accessories Belts", phrases=["leather"],
                           catalog=self.catalog, turn=1)
        self.assertEqual(client.chat_once.calls[0]["timeout"], llm_rank.TIMEOUT_S)


class FallbackMatrixTest(StageBase):
    """Every failure mode returns the incoming linear order, exactly."""

    def _run(self, client):
        from src import llm_rank
        ranked = [(1.0, "R_BELT_LEATHER"), (0.9, "R_BELT_CANVAS"),
                  (0.8, "R_BELT_SUEDE")]
        with mock.patch.dict(os.environ, FLAGS), \
                mock.patch.dict(sys.modules, {"tools.llm_client": client}):
            return llm_rank.apply(list(ranked), category="Accessories Belts",
                                  phrases=["leather"], catalog=self.catalog, turn=1)

    def test_w2_network_error_falls_back(self):
        self.assertEqual(self._run(fake_client(raises=OSError("down"))),
                         [(1.0, "R_BELT_LEATHER"), (0.9, "R_BELT_CANVAS"),
                          (0.8, "R_BELT_SUEDE")])

    def test_w2_timeout_falls_back(self):
        self.assertEqual(self._run(fake_client(raises=TimeoutError())),
                         [(1.0, "R_BELT_LEATHER"), (0.9, "R_BELT_CANVAS"),
                          (0.8, "R_BELT_SUEDE")])

    def test_w2_malformed_reply_falls_back(self):
        client = fake_client(payload={"choices": [{"message": {"content": "no json"}}]})
        out = self._run(client)
        self.assertEqual([pid for _, pid in out],
                         ["R_BELT_LEATHER", "R_BELT_CANVAS", "R_BELT_SUEDE"])

    def test_w2_empty_content_falls_back(self):
        client = fake_client(payload={"choices": [{"message": {"content": None}}]})
        out = self._run(client)
        self.assertEqual([pid for _, pid in out],
                         ["R_BELT_LEATHER", "R_BELT_CANVAS", "R_BELT_SUEDE"])

    def test_w2_late_turns_are_not_risked(self):
        from src import llm_rank
        ranked = [(1.0, "R_BELT_LEATHER"), (0.9, "R_BELT_CANVAS")]
        client = fake_client(payload={"choices": [{"message": {"content":
                                      '{"order": ["R_BELT_CANVAS", "R_BELT_LEATHER"]}'}}]})
        client.parse_order = lambda content, valid: ["R_BELT_CANVAS", "R_BELT_LEATHER"]
        with mock.patch.dict(os.environ, FLAGS), \
                mock.patch.dict(sys.modules, {"tools.llm_client": client}):
            out = llm_rank.apply(list(ranked), category="Accessories Belts",
                                 phrases=["leather"], catalog=self.catalog, turn=7)
        self.assertFalse(client.chat_once.calls, "a late turn called the model")
        self.assertEqual([pid for _, pid in out], ["R_BELT_LEATHER", "R_BELT_CANVAS"])


if __name__ == "__main__":
    unittest.main()
