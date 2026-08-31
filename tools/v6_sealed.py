"""V6 sealed lexical-shift confirmation instrument (S1).

Wraps an agent for the one-shot confirmation over sessions 101-200. The rewrite
is online -- it transforms whatever message arrives, so a candidate that changes
the dialogue does not get a stale transcript -- and catalog-only: every category
shift comes from the sealed body built by tools/build_v6_validation.py, which
derives variants from the shelf labels' own tokens. The repository's adversarial
dictionary is never consulted, which is what makes this a *confirmation* of
vocabulary generalization rather than a rerun of the development axis.

The body (analysis/_v6_s1_shifts.json) is an ignored sealed sidecar; the public
manifest (analysis/v6_validation_manifest.json) carries its hash. Consumption is
recorded in analysis/v6_validation_ledger.json and enforced by tools/v6_compare.py.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

BODY = Path("analysis/_v6_s1_shifts.json")
MANIFEST = Path("analysis/v6_validation_manifest.json")

_LOOKING = re.compile(r"^I'm looking for (.+?)\.(\s.*)?$", re.S)
_MATTERS = re.compile(r"For that, what matters is:\s*(.+?)\.?$")
_CORRECTION = re.compile(r"Actually, ignore my earlier preference\.\s*"
                         r"What I need is:\s*(.+?)\.?$")
_DECLINE = re.compile(r"I don.t have (?:an? |an additional )?preference for (\w+)")

# Generic follow-up phrasings. Constraint values pass through verbatim; only the
# scaffold moves, so the disclosed evidence is unchanged.
_MATTERS_TEMPLATES = (
    "What matters for it is: {}",
    "The things that matter are: {}",
    "For this one, what counts is: {}",
)
_CORRECTION_LEADS = (
    "Forget what I said earlier",
    "Disregard my earlier preference",
    "Let me correct myself",
)
_DECLINE_TEMPLATES = (
    "No particular preference for {}",
    "I have no strong feelings about {}",
    "{} is not something I have a preference for",
)


def _sha256_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_body(path: Path = BODY) -> dict:
    """Load the sealed shift body, fail closed, and prove it is the frozen one."""
    if not path.exists():
        raise RuntimeError(
            f"sealed S1 body {path} is absent; confirmation cannot run without it")
    body = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["s1"]["body_sha256"]
    actual = _sha256_json(body)
    if actual != expected:
        raise RuntimeError(
            f"sealed S1 body hash {actual[:12]}... does not match the frozen "
            f"manifest {expected[:12]}...; refusing to run a drifted confirmation")
    return body


class SealedShiftAgent:
    """The confirmation wrapper: lexical shift of category language, online."""

    def __init__(self, agent, session_ordinal: int, body: dict | None = None) -> None:
        self._agent = agent
        self._body = body if body is not None else load_body()
        # Fixed seed schedule: the session ordinal against the frozen base.
        self._rng = random.Random(self._body["seed_base"] + session_ordinal)

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def _shift_category(self, phrase: str) -> str:
        from src.text import normalise
        shifts = self._body["shifts"]
        variants = shifts.get(phrase) or shifts.get(normalise(phrase))
        if not variants:
            for label, options in shifts.items():
                if normalise(label) == normalise(phrase):
                    variants = options
                    break
        if not variants:
            return phrase
        return self._rng.choice(variants)

    def rewrite(self, message: str) -> str:
        if not message:
            return message
        rng = self._rng

        match = _CORRECTION.search(message)
        if match:
            lead = rng.choice(_CORRECTION_LEADS)
            return f"{lead}. What I need is: {match.group(1)}."

        match = _MATTERS.search(message)
        if match:
            return rng.choice(_MATTERS_TEMPLATES).format(match.group(1)) + "."

        match = _DECLINE.search(message)
        if match:
            return rng.choice(_DECLINE_TEMPLATES).format(match.group(1)) + "."

        match = _LOOKING.match(message)
        if match:
            category, rest = match.group(1), match.group(2) or ""
            template = rng.choice(self._body["opening_templates"])
            return template.format(category=self._shift_category(category)) + rest

        return message

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int):
        return self._agent.respond(session_id, self.rewrite(user_message), turn, top_k)
