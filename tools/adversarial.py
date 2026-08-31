"""Adversarial rewriting: the axes the paraphrase harness holds constant.

`tools/paraphrase.py` varies the scaffolding a customer wraps around their request.
It has two problems as a robustness measure, and both were found by reading it
rather than by running it:

  1. It re-inserts the category phrase verbatim at every level, including L3. The
     category is the single most load-bearing input in the system, so the "most
     unkind" setting never touches the thing that actually matters.
  2. Its scaffolding vocabulary -- OPENINGS, REQUIREMENT, MATTERS -- is the same
     list of phrases as FILLER_PREFIX in src/text.py. It only ever generates
     paraphrases the parser was written to strip. A closed loop.

Measured on the pre-rebuild agent, that mattered a great deal:

    control                              0.94365   HR 0.990
    case / word order / articles         0.94365   HR 0.990    +/- 0.000
    constraint text only (the old L3)    0.89422   HR 0.965    -0.049
    category head noun synonymised       0.64072   HR 0.680    -0.303

So this harness varies five axes independently, each switchable, and every word in
its vocabulary is deliberately absent from both src/text.py and the mined lexicon
in src/semantic.py. Nothing here is a word the agent has been taught.

    category      head noun replaced by a synonym or hypernym
    natural       taxonomy path rewritten as something a person would say
    scaffold      openings drawn from outside FILLER_PREFIX
    constraint    quoted product text synonymised, not truncated
    granularity   simulator drift: coarse_category taking 1 or 3 components

The last one is not a paraphrase at all. It models a plausible difference between
the public and private harnesses, which is worth knowing about before it happens
rather than afterwards.

Deterministic, seeded from arrival order. The evaluator mints session ids with
uuid4(), so anything derived from the id is random per run.
"""
from __future__ import annotations

import random
import re

AXES = ("category", "natural", "scaffold", "constraint", "granularity")

# Openings absent from src/text.py:FILLER_PREFIX. If the parser has been taught
# these, the harness has stopped being a test.
OPENINGS = [
    "Hoping to pick up {}", "In the market for {}", "On the hunt for {}",
    "Trying to track down {}", "Any recommendations on {}", "Do you carry {}",
    "Been meaning to replace my {}", "Wondering if you stock {}",
]
REQUIREMENTS = [
    "Non-negotiable: {}", "Whatever else, it has got to have {}",
    "The bit I actually care about is {}", "Dealbreaker if it is not {}",
    "Top of my list: {}",
]
FOLLOWUPS = [
    "The bit I care about: {}", "Sticking point is {}", "What I keep coming back to: {}",
    "If it helps, {}", "Narrowing it down: {}",
]

# Head-noun substitutions. Chosen to be genuinely different words rather than
# spelling variants, because a stemmer already handles spelling variants.
HEAD_NOUNS = {
    "shirts": "tops", "shirt": "top", "t-shirts": "tees", "tees": "t shirts",
    "socks": "hosiery", "sweaters": "jumpers", "pants": "trousers",
    "sneakers": "trainers", "shoes": "footwear", "jewelry": "jewellery",
    "necklaces": "chains", "earrings": "ear studs", "rings": "bands",
    "bracelets": "wristbands", "hoodies": "pullovers", "sweatshirts": "pullovers",
    "jackets": "coats", "coats": "jackets", "boots": "booties", "sandals": "slides",
    "dresses": "frocks", "watches": "timepieces", "wallets": "billfolds",
    "belts": "waist straps", "hats": "caps", "gloves": "mittens",
    "scarves": "wraps", "underwear": "undergarments", "bras": "bralettes",
    "leggings": "tights", "shorts": "short pants", "ties": "neckties",
    "sunglasses": "shades", "eyewear": "glasses", "tops": "blouses",
    "clothing": "apparel", "accessories": "extras", "athletic": "sports",
    "active": "activewear", "novelty": "quirky", "women": "ladies", "men": "gents",
}

# Constraint-text substitutions. Truncation is not enough: a truncated phrase is
# still a prefix, and a prefix is still a substring, so phrase containment
# survives it almost intact. These change the words.
CONSTRAINT_WORDS = {
    "cotton": "pure cotton weave", "polyester": "poly blend", "imported": "shipped in",
    "machine wash": "washable in a machine", "closure": "fastening",
    "lightweight": "barely there", "comfortable": "comfy", "stainless steel": "steel",
    "hypoallergenic": "kind to skin", "adjustable": "adjusts",
    "breathable": "airy", "durable": "long lasting", "soft": "gentle",
    "elastic": "stretchy", "waterproof": "keeps water out", "leather": "hide",
    "handmade": "made by hand", "genuine": "the real thing",
}

_CATEGORY = re.compile(r"^I'm looking for (.+?)(?=(?:, but I'm still exploring\.)|\.|$)")
_REQUIREMENT = re.compile(r"A key requirement is:\s*(.+?)\.?$")
_MATTERS = re.compile(r"For that, what matters is:\s*(.+?)\.?$")
_DECLINE = re.compile(r"I don.t have (?:an? |an additional )?preference for (\w+)")
_CORRECTION = re.compile(r"Actually, ignore my earlier preference\.\s*"
                         r"What I need is:\s*(.+?)\.?$")
_NUDGE = re.compile(r"Those options are not quite right yet")


class Rewriter:
    """Rewrites one session's utterances along the enabled axes."""

    def __init__(self, axes=AXES, seed: int = 0) -> None:
        self.axes = set(axes)
        self.rng = random.Random(seed)

    # -- category --------------------------------------------------------------
    def _category(self, phrase: str) -> str:
        words = phrase.split()
        if "category" in self.axes:
            words = [HEAD_NOUNS.get(word.lower().strip("&,"), word) for word in words]
        text = " ".join(words)
        if "natural" in self.axes:
            parts = text.split()
            if len(parts) > 1:
                parts = parts[::-1]          # taxonomy order is not speech order
            text = self.rng.choice(["a ", "some ", "a pair of ", "a decent ", ""]) \
                + " ".join(parts).lower()
        return text

    # -- constraints -----------------------------------------------------------
    def _constraint(self, text: str) -> str:
        if "constraint" not in self.axes:
            return text
        out = text
        for word, replacement in CONSTRAINT_WORDS.items():
            out = re.sub(re.escape(word), replacement, out, flags=re.I)
        words = out.split()
        if len(words) > 5 and self.rng.random() < 0.6:
            keep = sorted(self.rng.sample(range(len(words)), max(3, int(len(words) * 0.6))))
            words = [words[index] for index in keep]
        return " ".join(words)

    def _constraints(self, text: str) -> str:
        parts = [self._constraint(part.strip()) for part in text.split(";")]
        return "; ".join(part for part in parts if part)

    # -- entry point -----------------------------------------------------------
    def rewrite(self, message: str) -> str:
        if not message:
            return message
        pick = self.rng.choice
        scaffold = "scaffold" in self.axes

        match = _CORRECTION.search(message)
        if match:
            lead = pick(["Hold on, forget what I said before",
                         "Scratch all that", "Change of plan"]) if scaffold \
                else "Actually, ignore my earlier preference"
            return f"{lead}. {pick(REQUIREMENTS).format(self._constraint(match.group(1)))}."

        match = _MATTERS.search(message)
        if match:
            template = pick(FOLLOWUPS) if scaffold else "For that, what matters is: {}"
            return template.format(self._constraints(match.group(1))) + "."

        match = _DECLINE.search(message)
        if match:
            if not scaffold:
                return message
            return pick([f"No strong feelings on {match.group(1)}",
                         f"{match.group(1)} is up to you",
                         f"Genuinely do not mind about {match.group(1)}"]) + "."

        if _NUDGE.search(message):
            if not scaffold:
                return message
            return pick(["Not quite. Ask me something specific.",
                         "Those are not right. What else do you need to know?",
                         "Hmm, no. Ask me about one thing."])

        match = _CATEGORY.search(message)
        if match:
            category = self._category(match.group(1))
            opening = pick(OPENINGS).format(category) if scaffold \
                else f"I'm looking for {category}"
            if "still exploring" in message:
                opening += pick([", nothing settled yet", ", still weighing it up",
                                 ", just seeing what is out there"]) if scaffold \
                    else ", but I'm still exploring"
            requirement = _REQUIREMENT.search(message)
            if requirement:
                body = self._constraint(requirement.group(1))
                opening += ". " + (pick(REQUIREMENTS).format(body) if scaffold
                                   else f"A key requirement is: {body}")
            return opening + "."

        return message


class Adversarial:
    """Wraps an Agent so it only ever sees rewritten customer turns.

    Ground truth, scoring and the simulator's policy are untouched. Only the words
    reaching the agent change.
    """

    def __init__(self, agent, axes=AXES, seed: int = 0) -> None:
        self._agent = agent
        self._axes = tuple(axes)
        self._seed = seed
        self._index = 0
        self._rewriters: dict[str, Rewriter] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._index += 1
        self._rewriters[str(session_id)] = Rewriter(self._axes, self._seed + self._index)
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        rewriter = self._rewriters.get(str(session_id)) or Rewriter(self._axes, self._seed)
        return self._agent.respond(session_id, rewriter.rewrite(user_message), turn, top_k)


def drifted_categories(catalog_categories: dict, components: int) -> dict:
    """Simulator drift: coarse_category keeping a different number of path parts.

    The public harness keeps the last two. Nothing promises the private one does,
    and the difference is worth measuring before it is discovered.
    """
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    drifted: dict = {}
    for parent_asin, values in catalog_categories.items():
        cleaned: list = []
        for value in values:
            for part in str(value).split(","):
                part = part.strip()
                if part and part.lower() not in excluded:
                    cleaned.append(part)
        drifted[parent_asin] = ([" ".join(cleaned[-components:])] if cleaned
                                else ["clothing item"])
    return drifted
