"""W2: an optional LLM semantic ranking stage. OFF BY DEFAULT.

Pillar I's pipeline base names "Multi-Route Retrieval -> LLM Semantic Ranking".
This module is that stage, and it is deliberately not the default: the ordering
was measured against the shipped linear reranker and lost (D11,
analysis/llm_rerank_probe.json), so the linear stage is the declared offline
fallback and the default. The stage exists so the architecture is real and
reachable, with the measurement beside it -- not to chase the score, which is
why nothing here is tunable.

Activation requires BOTH environment variables:
    TECHJAM_LLM_RERANK=1
    AIAND_API_KEY=<key>            (read by the shared client; never stored)

Absent either, agent.py never imports this module and nothing here runs. When
active, any failure -- missing key, network error, timeout, malformed reply,
empty content -- returns the incoming linear order unchanged. The scored path
is byte-identical to the no-flag run by construction: the harness sets neither
variable.

This file imports os and nothing else. The network client lives in
tools/llm_client.py and is imported inside the active branch only, so the
operational AST scan over src/ never sees it (V8 C1/C4).
"""
from __future__ import annotations

import os

TOP_N = 20          # the window the model may reorder; same as the D11 probe
BLEND = 0.05        # the sweep's least-bad weight (D11); declared, not tunable
TIMEOUT_S = 30.0      # explicit per-turn ceiling; this path is an opt-in demo
                      # surface, not the scored one -- a reasoning model's real
                      # latency is 10-60 s and the turn must survive it exactly once
MAX_TURNS_ACTIVE = 4   # early turns only: late turns decide, and are not risked


def active(env=None) -> bool:
    """Both flags present. Read at call time, not import time."""
    env = os.environ if env is None else env
    return env.get("TECHJAM_LLM_RERANK") == "1" and bool(env.get("AIAND_API_KEY"))


def apply(ranked, *, category, phrases, catalog, turn: int = 1, env=None):
    """Reorder the top-N of an already-ranked list, or return it untouched.

    `ranked` is the live path's own [(score, pid), ...] after the linear
    reranker. The blend is the probe's arithmetic exactly: the LLM's position
    credit over the top-N window, weighted 0.05, added to the existing score,
    re-sorted with the same (-score, pid) key. Anything at all going wrong
    returns `ranked` as received -- the fallback is total by construction.
    """
    if not active(env) or turn > MAX_TURNS_ACTIVE or len(ranked) < 2:
        return ranked
    try:
        from tools.llm_client import DEFAULT_MODEL, chat_once, parse_order
    except Exception:
        return ranked
    window = ranked[:TOP_N]
    pids = [pid for _, pid in window]
    lines = ["You are ranking candidate products for a shopper in a "
             "multi-turn shopping conversation. Rank the candidates by how "
             "well they match everything the shopper has asked for so far, "
             "most relevant first. Reply with JSON only: "
             '{"order": ["<product id>", ...]} containing every given id '
             "exactly once.", ""]
    if category:
        lines.append(f"The shopper is looking for: {category}.")
    if phrases:
        lines.append("Requirements stated so far: " + "; ".join(phrases))
    lines += ["", "Candidates (id -- title):"]
    lines += [f"{i + 1}. {pid} -- "
              f"{catalog.meta[pid].get('title') or '(untitled)'}"
              for i, pid in enumerate(pids)]
    lines += ["", f"Rank all {len(pids)} candidates, best first."]
    body = {"model": os.environ.get("LLM_RERANK_MODEL", DEFAULT_MODEL),
            "temperature": 0, "max_tokens": 8192,
            "messages": [{"role": "user", "content": "\n".join(lines)}]}
    try:
        payload = chat_once(body, None, None, timeout=TIMEOUT_S, retries=1)
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
            return ranked
        order = parse_order(content, pids)
    except (Exception, SystemExit):
        # The client fails loudly by design (a measurement tool must); the
        # stage's contract is the opposite -- any failure at all is the
        # linear order, unchanged.
        return ranked
    credits = {pid: (len(pids) - 1 - i) / (len(pids) - 1) if len(pids) > 1 else 0.0
               for i, pid in enumerate(order)}
    rescored = [(score + BLEND * credits.get(pid, 0.0), pid)
                for score, pid in ranked]
    rescored.sort(key=lambda pair: (-pair[0], pair[1]))
    return rescored
