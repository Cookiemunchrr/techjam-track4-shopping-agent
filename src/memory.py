"""Long-term, cross-session memory keyed by an anonymized profile signature.

Pillar III asks for "continuously updating short-term session states and long-term
user profiles". The short-term half is src/state.py. This is the long-term half.

Two caveats, both belonging in the write-up rather than hidden:

  * The competition simulates every session as an isolated single user, so there is
    limited opportunity for this layer to pay off on the public or private sets. It
    is architecturally real and demonstrable; its measured effect here is small, and
    saying so is a better result than overclaiming it.
  * The signature comes only from the aggregate fields the contract exposes. There
    is no identifier to key on and we do not construct one.

What it carries is genuinely useful in a deployed system and measurable here: which
shelves a profile of this shape converges on, and which questions it answers. The
elicitation model in src/answerability.py is the same idea applied globally.
"""
from __future__ import annotations

import collections
import hashlib

MAX_PROFILES = 4096          # bounded: 800 private sessions must not grow forever
TOP_BUCKETS = 16


class LongTermMemory:
    """Aggregate, per-profile-shape memory. Never keyed by a person."""

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    @staticmethod
    def signature(user_profile: dict | None) -> str:
        """A stable key for profiles of the same shape.

        Same profile gives the same signature; different preferences give a
        different one. Hashed so nothing reconstructible is retained.
        """
        profile = user_profile if isinstance(user_profile, dict) else {}
        tags = profile.get("preference_tags")
        parts = [
            ",".join(sorted(str(tag).lower() for tag in tags)) if isinstance(tags, list) else "",
            str(profile.get("purchase_frequency") or ""),
            str(profile.get("rating_style") or ""),
            str(profile.get("average_prior_rating") or ""),
        ]
        return hashlib.sha256(" ".join(parts).encode("utf-8")).hexdigest()[:16]

    def _entry(self, signature: str) -> dict:
        entry = self._store.get(signature)
        if entry is None:
            if len(self._store) >= MAX_PROFILES:
                self._store.pop(next(iter(self._store)))
            entry = self._store[signature] = {
                "buckets": collections.Counter(), "asked": collections.Counter(),
                "answered": collections.Counter(), "sessions": 0,
            }
        return entry

    def observe(self, signature: str, bucket: str | None = None,
                attribute: str | None = None, answered: bool = True) -> None:
        entry = self._entry(signature)
        if bucket:
            entry["buckets"][bucket] += 1
            if len(entry["buckets"]) > TOP_BUCKETS:
                entry["buckets"] = collections.Counter(
                    dict(entry["buckets"].most_common(TOP_BUCKETS)))
        if attribute:
            entry["asked"][attribute] += 1
            if answered:
                entry["answered"][attribute] += 1

    def start_session(self, signature: str) -> None:
        self._entry(signature)["sessions"] += 1

    def recall(self, signature: str) -> dict:
        entry = self._store.get(signature)
        if not entry:
            return {}
        return {
            "buckets": dict(entry["buckets"]),
            "asked": dict(entry["asked"]),
            "answered": dict(entry["answered"]),
            "sessions": entry["sessions"],
        }

    def __len__(self) -> int:
        return len(self._store)
