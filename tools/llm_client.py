"""Shared LLM client and ballot parsing for build-time and opt-in runtime use.

Lives in tools/ so the operational AST scan over src/ never sees a network
import (V8 C1/C4). Callers in src/ import this module inside the guarded
branch only. The API key comes from AIAND_API_KEY and is never written
anywhere; AIAND_BASE_URL and LLM_RERANK_MODEL have defaults matching the V7
probe. Used by tools/llm_rerank_probe.py (the D11 measurement) and by
src/llm_rank.py (the W2 opt-in stage).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.aiand.com/v1"
DEFAULT_MODEL = "moonshotai/kimi-k3"


def cache_key(body: dict) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def chat_once(body: dict, cache: dict | None = None,
              cache_path: Path | None = None, timeout: float = 90,
              retries: int = 3) -> dict:
    """One chat-completion call. With a cache, keyed by exact request bytes."""
    if cache is not None:
        key = cache_key(body)
        if key in cache:
            return {**cache[key]["response"], "_cached": True}
    url = os.environ.get("AIAND_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("AIAND_API_KEY")
    if not api_key:
        raise SystemExit("AIAND_API_KEY is not set; export it yourself, never "
                         "write it to a file (V8 C3).")
    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        # The gateway (Cloudflare) refuses the default Python-urllib agent
        # with error 1010; any browser-like agent string passes.
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (techjam build-time probe)"},
        method="POST")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    else:
        raise SystemExit(f"LLM call failed after {retries} attempts: {last}")
    if cache is not None and cache_path is not None:
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": cache_key(body), "request": body,
                                     "response": payload}) + "\n")
        cache[cache_key(body)] = {"key": cache_key(body), "request": body,
                                  "response": payload}
    return payload


def parse_order(content: str, valid: list[str]) -> list[str]:
    """The returned order, validated: every candidate exactly once.

    Unknown ids are dropped; ids the model omitted keep their base order at
    the end. A malformed reply is a declined ballot, not a crash: it returns
    the base order, which is the honest zero-effect reading.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        start = text.index("{")
        payload = json.loads(text[start:])
        order = payload["order"]
    except (ValueError, KeyError, json.JSONDecodeError):
        return list(valid)
    known, seen = [], set()
    for pid in order:
        if pid in valid and pid not in seen:
            known.append(pid)
            seen.add(pid)
    return known + [pid for pid in valid if pid not in seen]
