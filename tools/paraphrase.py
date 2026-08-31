"""Paraphrasing robustness harness.

docs/competition_specification.md warns: "If natural-language paraphrasing is added
by the organizer, it cannot decide correctness." The private set may therefore say
the same things in different words. This harness rewrites the customer's utterances
before the agent sees them, leaving ground truth untouched, so we can measure how
much of our score survives.

Three levels, increasingly unkind:

  L1 scaffolding   the meta-phrasing changes; the quoted constraint text does not
  L2 surface       L1 plus casing, punctuation, contractions and filler
  L3 lossy         L2 plus perturbation of the constraint text itself, which is
                   the case that breaks any agent relying on verbatim matching

Deterministic: seeded per session with crc32, so a run is reproducible and
diffable. hash() on str is PYTHONHASHSEED-salted and must never be used here --
it made every reported paraphrase delta vary by up to 0.010 between runs.
"""
from __future__ import annotations

import random
import re

OPENINGS = [
    "I'm looking for {}", "I need {}", "I'm after {}", "Looking for {}",
    "I'd like {}", "Can you find me {}", "I want to buy {}", "Show me {}",
]
REQUIREMENT = [
    "A key requirement is: {}", "It has to be {}", "The important thing is {}",
    "Must have {}", "I specifically need {}", "One thing though - {}",
]
MATTERS = [
    "For that, what matters is: {}", "What matters is {}", "Mainly {}",
    "The things I care about are {}", "Well, {}", "I'd say {}",
]
DECLINE = [
    "I don't have a preference for {}", "No strong opinion on {}",
    "{} doesn't really matter to me", "I'm easy on {}", "No preference on {}",
]
EXPLORING = [
    "but I'm still exploring", "though I'm just browsing", "but I haven't decided yet",
    "I'm still looking around", "but I'm open to ideas",
]
CORRECTION = [
    "Actually, ignore my earlier preference", "Actually, forget that",
    "Never mind what I said", "I changed my mind", "On second thought",
    "Scratch that", "Actually no",
]
FILLER = ["hmm, ", "ok so ", "right, ", "well, ", "let me think - ", ""]

# The organizer's templates, as regexes with the payload captured.
_CATEGORY = re.compile(r"^I'm looking for (.+?)(?:, but I'm still exploring\.|\.|$)")
_REQUIREMENT = re.compile(r"A key requirement is:\s*(.+?)\.?$")
_MATTERS = re.compile(r"For that, what matters is:\s*(.+?)\.?$")
_DECLINE = re.compile(r"I don.t have (?:an? |an additional )?preference for (\w+)")
_CORRECTION = re.compile(r"Actually, ignore my earlier preference\.\s*What I need is:\s*(.+?)\.?$")
_NUDGE = re.compile(r"Those options are not quite right yet")


class Paraphraser:
    """Rewrites a customer utterance without changing what was asked for."""

    def __init__(self, level: int = 1, seed: int = 0) -> None:
        self.level = max(0, min(3, level))
        self.rng = random.Random(seed)

    # -- constraint payloads ------------------------------------------------
    def _payload(self, text: str) -> str:
        """At L3 the quoted product text is itself disturbed."""
        if self.level < 3:
            return text
        words = text.split()
        if len(words) > 6 and self.rng.random() < 0.7:
            cut = self.rng.randint(1, max(1, len(words) // 3))
            words = words[:-cut]                       # shopper paraphrases, does not recite
        if len(words) > 3 and self.rng.random() < 0.4:
            words.pop(self.rng.randrange(len(words)))  # drops a word
        return " ".join(words)

    def _payloads(self, text: str) -> str:
        parts = [self._payload(p.strip()) for p in text.split(";")]
        joiner = " and " if self.level >= 2 and self.rng.random() < 0.5 else "; "
        return joiner.join(p for p in parts if p)

    # -- surface ------------------------------------------------------------
    def _surface(self, text: str) -> str:
        if self.level < 2:
            return text
        if self.rng.random() < 0.3:
            text = self.rng.choice(FILLER) + text
        if self.rng.random() < 0.25:
            text = text.lower()
        if self.rng.random() < 0.2:
            text = text.replace(" - ", ", ").replace(";", ",")
        if self.rng.random() < 0.15:
            text = text.rstrip(".")
        return text

    # -- entry point --------------------------------------------------------
    def rewrite(self, message: str) -> str:
        if not message or self.level == 0:
            return message
        pick = self.rng.choice

        match = _CORRECTION.search(message)
        if match:
            return self._surface(
                f"{pick(CORRECTION)}. {pick(REQUIREMENT).format(self._payload(match.group(1)))}.")

        match = _MATTERS.search(message)
        if match:
            return self._surface(pick(MATTERS).format(self._payloads(match.group(1))) + ".")

        match = _DECLINE.search(message)
        if match:
            return self._surface(pick(DECLINE).format(match.group(1)) + ", use your judgement.")

        if _NUDGE.search(message):
            return self._surface(pick(["Not quite - ask me something specific.",
                                       "Those aren't right. What else do you need to know?",
                                       "Hmm, no. Ask me about one thing."]))

        match = _CATEGORY.search(message)
        if match:
            out = pick(OPENINGS).format(match.group(1))
            if "still exploring" in message:
                out += ", " + pick(EXPLORING)
            requirement = _REQUIREMENT.search(message)
            if requirement:
                out += ". " + pick(REQUIREMENT).format(self._payload(requirement.group(1)))
            return self._surface(out + ".")

        return self._surface(message)


class ParaphrasingAgent:
    """Wraps a real Agent so it only ever sees paraphrased customer turns.

    Ground truth, scoring and the simulator's policy are untouched -- only the
    words reaching the agent change.
    """

    def __init__(self, agent, level: int = 1, seed: int = 0) -> None:
        self._agent = agent
        self._level = level
        self._seed = seed
        self._index = 0
        self._rewriters: dict[str, Paraphraser] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Seeded from the arrival order, not the session_id: the evaluator mints
        # ids with uuid4(), so anything derived from the id is random per run and
        # every reported paraphrase delta becomes noise.
        self._index += 1
        self._rewriters[str(session_id)] = Paraphraser(self._level, self._seed + self._index)
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        rewriter = self._rewriters.get(str(session_id)) or Paraphraser(self._level, self._seed)
        return self._agent.respond(session_id, rewriter.rewrite(user_message), turn, top_k)
