"""The dialog state machine.

Pillar II asks for information accumulation, intent override with slot erasure and
rewriting, and a retrieval cutoff when the candidate pool is overloaded.

The pre-rebuild agent kept a flat list of clause strings and never removed anything
from it. This transcript was real:

    T1  "I'm looking for Men Hoodies. A key requirement is: cotton."   -> a hoodie
    T3  "Actually, forget that. What I need is: a leather belt."       -> the same hoodie

    state after T4: ['cotton', 'navy blue', 'Actually, forget that',
                     'a leather belt', 'full grain leather']

Three separate defects in six lines: the retracted constraints were still scoring,
the correction phrase itself had become a constraint and was being fed to BM25, and
the category was frozen because it was only ever assigned on turn 1.

What replaces it is a typed slot table. Each observation records which attribute it
constrains, when it was said, and whether it was stated positively or refused.
Superseded slots are held at near-zero weight rather than deleted, because "the
customer changed their mind" is not the same as "the customer never said it" -- and
because the organizer's own override sessions derive the old and the new value from
the same product, so deletion is measurably wrong here as well as conceptually
coarse.
"""
from __future__ import annotations

import re

from .catalog import COLOR_RE, MATERIAL_RE
from .text import (DECLINE, NO_PREFERENCE, RESTART, is_committed, is_correction,
                   is_exploring, is_meta, is_restart, normalise, split_clauses,
                   tokens)

POSITIVE = 1
NEGATIVE = -1

# Weight retained by a constraint the customer has replaced. Not zero: they did say
# it, and on this simulator the replaced value often still describes the target.
SUPERSEDED_WEIGHT = 0.08
# Per-turn recency decay. A constraint from six turns ago still counts, but the one
# they just said counts more.
DECAY = 0.93
# ...except for the ones a shopper pins down rather than muses about. "Leather" said
# on turn 1 still binds on turn 8; a feature bullet quoted in passing does not carry
# the same standing. Without this, decay alone pushes the material out of the window
# during distillation and a suede belt outranks the leather one that was asked for.
STANDING = frozenset({"material", "color", "brand", "budget", "size"})
MIN_WEIGHT = 0.05
MAX_SLOTS = 24              # hard ceiling; `distil` keeps the live set smaller

# Cues that mark the *rest* of the clause as a retraction rather than a constraint.
# Stripped before the clause is stored -- otherwise "Actually, forget that" becomes
# search evidence, which is what used to happen.
CORRECTION_LEAD = re.compile(
    r"^(?:actually|instead|never ?mind|scratch that|forget (?:that|it)|"
    r"changed my mind|on second thought|no wait|hold on)\b[\s,.:;-]*"
    r"(?:(?:ignore|forget)\s+(?:my\s+)?(?:earlier\s+)?(?:preference|request|that|it)\b[\s,.:;-]*)?",
    re.I,
)

# Explicit refusal of a value: "not polyester", "I don't want leather", "no wool",
# "nothing in suede". Captures what is being refused, not the whole sentence --
# storing "It must not be polyester" as evidence would score *for* polyester,
# which is exactly what used to happen.
NEGATION = re.compile(
    r"\b(?:(?:do\s?n[o']?t|does\s?n[o']?t|dont|don't)\s+(?:want|like|need)|"
    r"not|no|without|avoid|nothing|never)\b"
    # "no more than $30" is a ceiling, not a refusal of the word "more". Without
    # this guard the comparison itself was captured as the refused value, the
    # budget was lost entirely, and the agent went on to penalise every product
    # whose text contains "more".
    r"(?!\s+(?:more|less|fewer|greater|higher|lower)\s+than\b)"
    r"\s+(?:be\s+|any\s+|a\s+|an\s+|the\s+|in\s+|made\s+of\s+)*"
    r"(?P<value>[a-z][\w-]{2,}(?:\s+[a-z][\w-]{2,}){0,2})", re.I)

# Words that can follow a negation without naming anything a product could be.
# The evaluator's own nudge is "Those options are not quite right yet", and the
# parser read it as a refusal of "quite right yet" -- then penalised every product
# whose text happened to contain those words, which is 3,512 of the 50,000 here.
# A shopper saying the list is wrong has told us the list is wrong; they have not
# told us about a material.
_HOLLOW = frozenset({
    "quite", "right", "yet", "really", "very", "exactly", "still", "sure",
    "certain", "close", "correct", "perfect", "ideal", "suitable", "option",
    "options", "one", "ones", "thing", "things", "what", "working", "there",
    "them", "these", "those", "it", "that", "much", "far", "enough", "all",
    "please", "thanks", "thank",
})


def substantive(value: str) -> str | None:
    """The part of a refused value that names something, or None if none of it does.

    Trimmed from both ends rather than filtered throughout, because the hollow
    words are scaffolding around the value and not inside it: "quite bright" is a
    refusal of bright, while a hollow word between two real ones is part of the
    phrase.
    """
    words = value.split()
    while words and words[0].lower().strip(".,;:") in _HOLLOW:
        words.pop(0)
    while words and words[-1].lower().strip(".,;:") in _HOLLOW:
        words.pop()
    return " ".join(words) if words else None


# A negation under a negative consequence is a requirement, not a refusal.
# "Dealbreaker if it is not Water Resistant" asks *for* water resistance; recorded
# as an exclusion it inverts the polarity of the hardest constraint the shopper
# has, so the slot votes against exactly the products they insisted on.
# Deliberately narrow: "unless" reads the other way ("anything unless it is
# polyester" really is a refusal), so it is not in here.
DOUBLE_NEGATIVE = re.compile(
    r"\b(?:deal\s?-?breakers?|no\s+good|no\s+use|useless|pointless|"
    r"won'?t\s+work|not\s+worth\s+it)\b[^.]{0,40}?\bif\b[^.]{0,40}?\bnot\b", re.I)

# A number that is not part of a percentage. "100% Leather" is a composition, and
# reading its 100 as a price is the most available way for this parser to invent a
# budget nobody stated.
PRICE = re.compile(r"(?:\$|usd\s*)?(?<![\d.])(\d+(?:\.\d+)?)(?![\d.]*\s*%)"
                   r"\s*(?:dollars?|bucks?)?", re.I)
BUDGET_CUE = re.compile(r"\b(?:budget|under|below|less than|cheaper than|around|about|"
                        r"no more than|up to|price|at least|more than|over|above|"
                        r"starting at|no less than|upwards of)\b", re.I)
# Words that mean money on their own. Everything else in BUDGET_CUE is a comparison
# that only means money when there is money next to it -- and in this catalog the
# difference is not academic. "Under" is a brand here (Under Armour) and "about"
# turns up in any natural phrasing of a requirement ("the bit I care about is 100%
# Leather"). Both were being filed as budgets, and because `budget` is not `feature`,
# each new pseudo-budget superseded the last real constraint filed under it.
BUDGET_WORD = re.compile(r"\b(?:budget|price|pricing|afford|spend|cost)\b", re.I)
CURRENCY = re.compile(r"[$\u20ac\u00a3]|\b(?:usd|dollars?|bucks?)\b", re.I)
# The subset of those cues that state a ceiling rather than a target. "Under $30"
# is satisfied by anything cheaper; "around $30" is not. See DialogState.budget.
# The lookbehinds matter: "no more than" is a ceiling and "no less than" is a
# floor, and each contains the other's keyword. Without them a single phrase
# satisfies both patterns and the shopper's meaning is decided by pattern order.
BUDGET_CEILING = re.compile(r"\b(?:under|below|(?<!no )less than|cheaper than|"
                            r"no more than|up to|max|maximum|at most|within)\b", re.I)
# ...and the other direction. "At least $50" is a shopper telling us the cheap end
# of the shelf is not what they are looking for -- a real thing to say about
# clothing, where price carries quality, and previously read as "around $50", which
# penalises the £200 coat they would have been happy with.
BUDGET_FLOOR = re.compile(r"\b(?:at least|(?<!no )more than|over|above|starting at|"
                          r"minimum|no less than|upwards of|north of)\b", re.I)
# The money span alone: an optional comparison cue, the amount, and an optional
# unit. Bounded on both sides so what lands in a slot is the budget and not the
# sentence it was mentioned in. See DialogState._observe_opening.
BUDGET_SPAN = re.compile(
    r"(?:\b(?:budget|price|under|below|less than|cheaper than|around|about|"
    r"no more than|up to|max|maximum|at most|within|at least|more than|over|"
    r"above|starting at|no less than|upwards of|north of)\s+)?"
    r"(?:\$|usd\s*)?(?<![\d.])\d+(?:\.\d+)?(?![\d.]*\s*%)"
    r"(?:\s*(?:dollars?|bucks?))?", re.I)
# What is left joining two halves of a sentence once the middle is removed.
# "Forget everything, let's start over and I need a silk scarf" leaves "let's" and
# "and" behind, which are not requirements.
CONNECTIVE = re.compile(r"^(?:and|so|then|now|ok(?:ay)?|well|let'?s|let us|"
                        r"i'?ll|we'?ll|i'?d|please)\b[\s,.:;-]*", re.I)
SIZE_CUE = re.compile(r"\b(?:size|sizing|fit|width|wide|narrow|petite|plus|tall)\b", re.I)
USE_CUE = re.compile(r"\b(?:hiking|running|gym|workout|winter|summer|outdoor|work|"
                     r"office|wedding|party|travel|everyday|casual)\b", re.I)
STYLE_CUE = re.compile(r"\b(?:style|sleeve|neck|collar|department|cut|fit|slim|"
                       r"loose|relaxed|vintage|modern)\b", re.I)


def _information(text: str, catalog) -> float:
    """Mean inverse document frequency of the words in a constraint.

    A cheap proxy for how much a phrase narrows the catalog. Used only to decide
    what survives distillation, never to score a product.

    Words absent from the catalog score zero rather than the maximum. IDF peaks at
    df == 0, which is exactly backwards here: a word matching no product cannot
    separate two products, however rare it is. Left uncorrected, unmatchable
    phrasing ("under thirty dollars", scoring 3.689) displaces the material the
    shopper actually stated ("Leather", 2.079) during distillation, and a suede
    belt outranks the leather one that was asked for.
    """
    if catalog is None:
        return 1.0
    words = [word for word in tokens(text) if len(word) > 2]
    if not words:
        return 1.0
    try:
        return sum(catalog.idf(word) if catalog.df.get(word) else 0.0
                   for word in words) / len(words)
    except Exception:
        return 1.0


def alternatives(text: str, attribute: str) -> tuple:
    """Facet values a clause offers as interchangeable, or ().

    "Blue or green would work" is one constraint with two acceptable answers.
    Returns () unless every branch of the "or" resolves to a value of the same
    attribute -- "leather or something cheap" is not a list of alternatives, and
    guessing that it is would invent a constraint the shopper did not state.
    """
    if attribute not in ("material", "color"):
        return ()
    parts = ALTERNATIVE.split(text)
    if len(parts) < 2:
        return ()
    pattern = MATERIAL_RE if attribute == "material" else COLOR_RE
    found: list[str] = []
    for part in parts:
        hit = pattern.search(part)
        if hit is None:
            return ()
        value = hit.group(1).lower()
        if value not in found:
            found.append(value)
    return tuple(found) if len(found) > 1 else ()


def classify(text: str) -> str:
    """Which attribute a phrase constrains.

    Mirrors the granularity of the contract's `ask_attribute` enum so a slot can be
    matched against the question that produced it. Deliberately not a copy of the
    evaluator's own `classify_constraint`: we need this to work on paraphrased text,
    not on the organizer's templates.
    """
    lowered = text.lower()
    if BUDGET_WORD.search(lowered) or CURRENCY.search(lowered) \
            or (BUDGET_CUE.search(lowered) and PRICE.search(lowered)):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered):
        return "color"
    if SIZE_CUE.search(lowered):
        return "size"
    if STYLE_CUE.search(lowered):
        return "style"
    if USE_CUE.search(lowered):
        return "use_case"
    return "feature"


class Budget:
    """A price the customer named, and which side of it they meant.

    Three readings, and they are genuinely different instructions:

        cap    "under $30"    anything cheaper is fine
        floor  "at least $50" anything dearer is fine
        tier   "around $30"   far below is as wrong as far above
    """

    __slots__ = ("amount", "cap", "floor")

    def __init__(self, amount: float, cap: bool = False, floor: bool = False) -> None:
        self.amount = float(amount)
        self.cap = bool(cap)
        self.floor = bool(floor)

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        mark = "<=" if self.cap else (">=" if self.floor else "~")
        return f"<Budget {mark}{self.amount:g}>"


# How the values in a slot combine. "Blue or green would work" is one constraint
# with two acceptable answers, and storing it as the single string "blue or green"
# makes it unsatisfiable: no product's text contains that phrase, the facet reader
# takes whichever colour it saw first, and the next colour the shopper mentions
# supersedes the pair wholesale.
EQUALS = "equals"
ONE_OF = "one_of"

# Splits a clause into alternatives. Only " or " -- "and" is left alone because in
# product language it usually joins two different attributes ("cotton and machine
# washable"), which the clause splitter already handles as separate constraints.
ALTERNATIVE = re.compile(r"\s+or\s+", re.I)


class Slot:
    """One thing the customer said, and what we make of it."""

    __slots__ = ("text", "attribute", "turn", "polarity", "superseded", "information",
                 "operator", "values")

    def __init__(self, text: str, attribute: str, turn: int, polarity: int = POSITIVE,
                 information: float = 1.0, values=None) -> None:
        self.text = text
        self.attribute = attribute
        self.turn = turn
        self.polarity = polarity
        self.superseded = False
        # The acceptable values, when the catalog can resolve them. One value is
        # `equals`; several are `one_of`, satisfied by any of them.
        self.values: tuple = tuple(values or ())
        self.operator = ONE_OF if len(self.values) > 1 else EQUALS
        # How discriminative this constraint is against the catalog. Distillation
        # ranks on weight * information, so "Full Grain Leather" survives a
        # compression that "and comfortable" does not.
        self.information = information

    def weight(self, now: int) -> float:
        base = SUPERSEDED_WEIGHT if self.superseded else 1.0
        if self.attribute in STANDING:
            return base                      # pinned down; it does not fade
        return max(base * (DECAY ** max(0, now - self.turn)), MIN_WEIGHT * base)

    def __repr__(self) -> str:       # pragma: no cover - debugging aid
        mark = "-" if self.superseded else ("!" if self.polarity < 0 else "")
        return f"<{mark}{self.attribute}@{self.turn} {self.text!r}>"


class DialogState:
    """Typed, decaying, supersedable constraint state for one conversation.

    Supersession is a selectable mode (W3): "decay" (default) holds a replaced
    slot at SUPERSEDED_WEIGHT; "erase" removes it from the structure entirely --
    the literal slot erasure Pillar II names. The mode is chosen once per agent
    (P_SUPERSEDE); which one is the default is a measurement, not a preference
    (analysis/v8_w3_erasure.json).
    """

    __slots__ = ("slots", "turn", "category", "category_turn", "intent",
                 "buying_evidence", "browsing_evidence", "supersede_mode")


    def __init__(self, supersede_mode: str = "decay") -> None:
        self.slots: list[Slot] = []
        self.turn = 0
        self.category: str | None = None
        self.category_turn = 0
        self.intent: str | None = None
        # Cumulative, not a turn-1 verdict. See `read_intent`.
        self.buying_evidence = 0.0
        self.browsing_evidence = 0.0
        self.supersede_mode = supersede_mode

    def _retire(self, marked: list) -> None:
        """Supersede the marked slots, in whichever mode this state runs.

        decay: mark them; they keep a SUPERSEDED_WEIGHT trace. erase: they leave
        the structure, so active(), weighted_phrases(), budget() and facets()
        cannot see them and their text cannot reach BM25 or the phrase channel.
        """
        if not marked:
            return
        if self.supersede_mode == "erase":
            gone = {id(slot) for slot in marked}
            self.slots = [slot for slot in self.slots if id(slot) not in gone]
        else:
            for slot in marked:
                slot.superseded = True

    # ----------------------------------------------------------------- intent --
    # How much evidence must separate the two tracks before the pool changes shape.
    # Below this the shopper has said nothing decisive and `open` is the honest
    # reading, which is also the widest of the three.
    INTENT_MARGIN = 1.0
    # A hard constraint is the strongest buying signal a shopper gives, and it is
    # the one the simulator actually produces. Cues are worth less because they are
    # a manner of speaking; a stated material is a decision.
    PER_CONSTRAINT = 1.0
    PER_CUE = 1.5

    def read_intent(self, message: str, clauses) -> str:
        """Buying, browsing or open, from everything said so far.

        The old rule looked at the opening turn once: an exploring cue meant
        browsing, more than one clause meant buying, and anything else meant open.
        It was never revisited unless the shopper changed category, so somebody who
        opened vaguely and then named a material, a colour and a budget stayed
        `open` for the rest of the session -- and `open` retrieves from two shelves
        where `buying` retrieves from one, so the track never narrowed around the
        constraints they had just given.

        Two kinds of evidence, and they answer different questions:

          a cue on this turn   how the shopper wants to be *served*. "I'm still
                               comparing" asks for a space to look at even from
                               somebody who has already named three requirements,
                               so a cue on the current turn settles the workflow.
          pinned constraints   how much the shopper has *decided*. Accumulated
                               across turns, and counted fresh each time so a
                               retraction lowers the score instead of leaving it
                               stranded at its high-water mark.

        Constraints are never erased by the track: "I need sneakers but I'm still
        comparing" is a browsing workflow that still ranks on every constraint
        given. Only the shape of the pool changes.
        """
        # A restart is the shopper saying they have told us nothing. Whatever the
        # evidence was, it described a search they have abandoned.
        if is_restart(message):
            self.buying_evidence = self.browsing_evidence = 0.0
            return "open"

        exploring, committed = is_exploring(message), is_committed(message)
        if exploring:
            self.browsing_evidence += self.PER_CUE
        if committed:
            self.buying_evidence += self.PER_CUE
        if exploring != committed:
            return "browsing" if exploring else "buying"

        pinned = {slot.attribute for slot in self.slots
                  if not slot.superseded and slot.polarity > 0
                  and slot.attribute in STANDING}
        buying = self.buying_evidence + self.PER_CONSTRAINT * len(pinned)
        browsing = self.browsing_evidence
        if buying - browsing >= self.INTENT_MARGIN:
            return "buying"
        if browsing - buying >= self.INTENT_MARGIN:
            return "browsing"
        # Nothing separates them. Several clauses is somebody who has said a good
        # deal even if none of it pinned a standing attribute; one is somebody who
        # has not. The old rule, kept as the tie-break it always should have been.
        return "buying" if len(clauses or ()) > 1 else "open"

    # ---------------------------------------------------------------- observe --
    def observe(self, message: str, turn: int, catalog=None, skip_first: bool = False,
                opening_index: int = 0) -> bool:
        """Absorb a turn; ``opening_index`` identifies a routed category clause."""
        self.turn = max(self.turn, turn)
        if not message or DECLINE.search(message):
            return False

        # Correction scope. A cue means "I am changing something", not "I am
        # changing everything": which slots it retires depends on what follows it.
        #
        #   attribute  the default. "Actually, brown" supersedes the colour and
        #              leaves the material, the budget and the size standing --
        #              handled by the same-attribute rule in `_add`, which needs no
        #              cue at all because real shoppers rarely give one.
        #   product    a different kind of product entirely. Not decided here; see
        #              `product_reset`, driven by Agent._is_new_category.
        #   global     an explicit restart, and only that.
        #
        # The old code took every cue as global and superseded every earlier slot.
        # That is wrong on this simulator too, not just in principle: the override
        # sessions derive the old and the new value from the same target product,
        # so the constraints it was retiring still described the thing being
        # searched for.
        restarting = is_restart(message)
        correcting = is_correction(message) or restarting
        if restarting:
            self.restart(turn)
        clauses = split_clauses(message)
        opening = None
        if skip_first and clauses:
            # Product corrections often lead with "forget that" and name the new
            # shelf in the following clause. Remove the clause that actually drove
            # routing, not blindly the first, or the shelf name becomes a feature.
            index = opening_index if 0 <= opening_index < len(clauses) else 0
            opening = clauses.pop(index)

        learned = False
        if opening is not None:
            learned = self._observe_opening(opening, turn, catalog) or learned
        for clause in clauses:
            cleaned = CORRECTION_LEAD.sub("", clause).strip(" ,:;.—-")
            if restarting:
                # "Forget everything and start over" is an instruction, not a
                # requirement. Left in, it becomes search evidence -- the same
                # defect as the correction lead, which is why both are stripped.
                cleaned = RESTART.sub("", cleaned).strip(" ,:;.—-")
                cleaned = CONNECTIVE.sub("", cleaned).strip(" ,:;.—-")
            if len(cleaned) <= 2:
                continue
            if is_meta(cleaned):
                # An instruction about the conversation, not about a product.
                # "Ask me about one specific attribute" is the evaluator telling
                # us how to behave; stored as a constraint it ranks products by
                # whether their text says "specific" or "option".
                continue
            if NO_PREFERENCE.search(cleaned):
                # The shopper handed this attribute back to us. That is a real
                # answer and it carries no constraint, so the clause contributes
                # nothing -- and must not reach NEGATION, which would read "do not
                # mind about brand" as a refusal of the words "mind about brand".
                continue
            polarity = POSITIVE
            negated = NEGATION.search(cleaned)
            if negated:
                # Keep only the refused value. The surrounding words are scaffolding
                # and would otherwise be scored as though the customer asked for it.
                value = substantive(negated.group("value").strip())
                if value is None:
                    continue                     # a complaint, not a constraint
                # Two negatives make a requirement, and getting this backwards is
                # worse than missing it: the slot is filed against the products the
                # shopper insisted on.
                polarity = POSITIVE if DOUBLE_NEGATIVE.search(cleaned) else NEGATIVE
                cleaned = value
                if len(cleaned) <= 2:
                    continue
            attribute = classify(cleaned)
            if self._add(cleaned, attribute, turn, polarity, replace=restarting,
                         information=_information(cleaned, catalog),
                         values=alternatives(cleaned, attribute)):
                learned = True
        return learned

    # Attributes that can be stated in the same breath as the category and still be
    # about the product rather than about the category. Deliberately excludes
    # `feature`, which is what `classify` falls through to: whatever is left of an
    # opening clause after the shelf name is removed is overwhelmingly the category
    # noun itself ("boots"), and re-entering that as a constraint scores the
    # routing decision twice.
    OPENING_ATTRIBUTES = ("material", "color", "size", "budget")

    def _observe_opening(self, clause: str, turn: int, catalog=None) -> bool:
        """Constraints the customer attached to the category in one phrase.

        "I'm looking for black leather boots" names a shelf and two facets at once.
        The opening clause used to be discarded whole, on the reasoning that it was
        the category and nothing else. That is true of this simulator -- turn 1 is
        `coarse_category(target.categories)` verbatim -- and false of anybody
        speaking naturally, which is why the loss showed up on the `natural` axis
        and never on the official score.

        The shelf name is removed first when it can be located, so a clause that
        *is* only a category ends up empty and this method changes nothing. What
        survives is read for concrete facets only; see OPENING_ATTRIBUTES.
        """
        from .routing import locate_bucket

        remainder = clause
        if catalog is not None:
            located = locate_bucket(catalog, normalise(clause))
            if located is not None:
                _, start, size = located
                words = normalise(clause).split()
                remainder = " ".join(words[:start] + words[start + size:])
        if not remainder.strip():
            return False

        learned = False
        for attribute, pattern in (("material", MATERIAL_RE), ("color", COLOR_RE)):
            either = alternatives(remainder, attribute)
            if either:
                # "blue or green boots" is one constraint, not two competing ones.
                if self._add(" or ".join(either), attribute, turn, POSITIVE,
                             replace=False,
                             information=_information(either[0], catalog),
                             values=either):
                    learned = True
                continue
            seen: set[str] = set()
            for match in pattern.finditer(remainder):
                value = match.group(1).lower()
                if value in seen:
                    continue
                seen.add(value)
                if self._add(value, attribute, turn, POSITIVE, replace=False,
                             information=_information(value, catalog),
                             values=(value,)):
                    learned = True
        # A price stated alongside the category is about the product too. Only the
        # money span is kept, never the surrounding clause: slot text is scored
        # verbatim by the phrase term, so storing "a black cotton dress under $40"
        # would score the category words a second time, on top of the routing that
        # already acted on them.
        money = BUDGET_SPAN.search(remainder)
        if money and classify(money.group(0)) == "budget":
            text = money.group(0).strip(" ,:;.—-")
            if len(text) > 2 and self._add(text, "budget", turn, POSITIVE, replace=False,
                                           information=_information(text, catalog)):
                learned = True
        return learned

    def _add(self, text: str, attribute: str, turn: int, polarity: int, replace: bool,
             information: float = 1.0, values=None) -> bool:
        for slot in self.slots:
            if slot.text.lower() == text.lower() and slot.polarity == polarity:
                return False                     # already known; do not double-count

        # Information accumulation is the default; rewriting happens when the
        # customer either signals a correction or simply states a different value
        # for a slot they have already filled. Real customers do the second more
        # often than the first, and never say "actually" when they do.
        if attribute != "feature":
            self._retire([slot for slot in self.slots
                          # Only earlier turns. Two constraints stated in the same breath
                          # ("what matters is: A; B") are both current, not a correction.
                          if slot.attribute == attribute and slot.turn < turn
                          and not slot.superseded and slot.polarity == polarity])
        if replace:                          # global scope only; see `observe`
            self._retire([slot for slot in self.slots if slot.turn < turn])

        self.slots.append(Slot(text, attribute, turn, polarity, information, values))
        if len(self.slots) > MAX_SLOTS:
            self.distil(MAX_SLOTS)
        return True

    # Constraints that describe the *kind of product* rather than the shopper.
    # A change of product retires these; budget and brand outlive it, because
    # "under $40" is a fact about the person and survives switching shelves.
    PRODUCT_BOUND = ("material", "color", "size", "style", "use_case", "feature")

    def restart(self, turn: int) -> None:
        """Retire the complete brief, even when no replacement is stated yet."""
        self._retire([slot for slot in self.slots if slot.turn < turn])

    def product_reset(self, turn: int) -> None:
        """The customer switched to a different kind of product.

        Distinct from slot override, which rewrites one value while the product
        intent holds. Here the earlier constraints described something they have
        stopped shopping for: "red cotton dress" then "actually, leather boots"
        should not keep scoring for red and cotton. Superseded rather than
        deleted, on the same reasoning as everywhere else in this file.
        """
        self._retire([slot for slot in self.slots
                      if slot.turn < turn and slot.attribute in self.PRODUCT_BOUND])

    def supersede(self, attribute: str) -> None:
        self._retire([slot for slot in self.slots if slot.attribute == attribute])

    # ------------------------------------------------------------------ query --
    def active(self) -> list[Slot]:
        """Slots still carrying meaningful weight, most recent first."""
        live = [slot for slot in self.slots if slot.weight(self.turn) > MIN_WEIGHT]
        return sorted(live, key=lambda slot: -slot.turn)

    def phrases(self) -> list[str]:
        """The active, current, positive constraint text."""
        return [slot.text for slot in self.active()
                if slot.polarity > 0 and not slot.superseded]

    def weighted_phrases(self) -> list[tuple[str, float]]:
        """Positive constraints with their decayed weights, for lexical scoring.

        In decay mode, superseded slots appear here at SUPERSEDED_WEIGHT rather
        than vanishing. Erasure is a dial, not a switch: "the customer changed
        their mind" is not the same claim as "the customer never said it", and
        the strength of the distinction is an empirical question -- see the sweep
        in the README. In erase mode (P_SUPERSEDE=erase) they vanish outright:
        measured in analysis/v8_w3_erasure.json.
        """
        return [(slot.text, slot.weight(self.turn))
                for slot in self.active() if slot.polarity > 0]

    def budget(self) -> "Budget | None":
        """The price the customer said they were working with, if they said one.

        Read from the most recent active, positive `budget` slot only -- numbers
        appear in feature text constantly ("7 x 3 x 0 inches", "8 Ounces") and
        reading those as prices would be worse than having no budget at all. A
        range collapses to its midpoint, so "between $20 and $40" and "around $30"
        mean the same thing, which for a shopper they do.

        "Under $30" and "around $30" do not mean the same thing, though, and the
        difference decides what a $12 belt is. Under is a ceiling: anything below
        it is fine. Around is a tier: far below it is as wrong as far above,
        because the shopper is describing the kind of thing they want and not
        only what they will pay. `cap` carries that distinction to the scorer.

        Note for anyone tuning against the simulator: `intent_card` sets the
        budget to the target's *exact* price, so a tolerance fitted on generated
        sessions collapses into an answer-key detector. See src/scoring.py for
        where the tolerance actually comes from.
        """
        for slot in self.active():
            if slot.attribute != "budget" or slot.polarity < 0 or slot.superseded:
                continue
            found = [float(match.group(1)) for match in PRICE.finditer(slot.text)]
            found = [value for value in found if value > 0]
            if found:
                # A range ("between $20 and $40") collapses to its midpoint and is
                # neither a cap nor a floor; a single figure takes whichever cue
                # the shopper attached to it.
                ranged = len(found) > 1
                cap = not ranged and bool(BUDGET_CEILING.search(slot.text))
                floor = not ranged and not cap and bool(BUDGET_FLOOR.search(slot.text))
                return Budget(sum(found) / len(found), cap=cap, floor=floor)
        return None

    def facets(self) -> dict:
        """Stated material and colour, as catalog facet values.

        Distinct from the constraint text these are drawn from. The phrase term
        already rewards "leather" appearing anywhere in a product's blob, which a
        canvas bag with a leather trim satisfies; this asks the narrower question
        of whether the product *is* the thing they named. Only material and colour,
        because they are the only free-text attributes the catalog resolves to a
        single value per product (src/catalog.py: Catalog.facet).
        """
        found: dict = {}
        for slot in self.active():
            if slot.polarity < 0 or slot.superseded or slot.attribute not in ("material", "color"):
                continue
            if slot.attribute in found:
                continue                      # most recent wins; active() is newest first
            if slot.values:
                # An alternative set is satisfied by any member; the scorer takes
                # the best rather than the sum. See scoring.Scorer._facet.
                found[slot.attribute] = slot.values if len(slot.values) > 1 else slot.values[0]
                continue
            pattern = MATERIAL_RE if slot.attribute == "material" else COLOR_RE
            hit = pattern.search(slot.text)
            if hit:
                found[slot.attribute] = hit.group(1).lower()
        return found

    def rejected(self) -> list[str]:
        """Text the customer explicitly refused."""
        return [slot.text for slot in self.slots
                if slot.polarity < 0 and not slot.superseded]

    def weights(self) -> dict[str, float]:
        return {slot.text: slot.weight(self.turn) for slot in self.slots}

    # Attributes we can be confident the customer has actually pinned down.
    # `feature` is the catch-all `classify` falls through to, so treating it as
    # answered would silence the single most productive question in the whole
    # elicitation set from turn one of every session.
    SETTLED = ("material", "color", "brand", "budget")

    def answered(self) -> set[str]:
        return {slot.attribute for slot in self.slots
                if not slot.superseded and slot.attribute in self.SETTLED}

    # ------------------------------------------------------------- distillation --
    def distil(self, cap: int = 12) -> None:
        """Compress the history to its most informative `cap` slots.

        Pillar III's "Personalized Context Distillation". Keeps the highest-weight
        slots, which after decay and supersession means the recent and the
        un-retracted, and guarantees at least one slot per attribute so distilling
        cannot silently drop a whole constraint dimension.
        """
        if len(self.slots) <= cap:
            return
        ranked = sorted(self.slots,
                        key=lambda slot: (-(slot.weight(self.turn) * slot.information),
                                          -slot.turn))
        keep: list[Slot] = []
        seen_attributes: set[str] = set()
        for slot in ranked:
            if slot.attribute not in seen_attributes:
                keep.append(slot)
                seen_attributes.add(slot.attribute)
        for slot in ranked:
            if len(keep) >= cap:
                break
            if slot not in keep:
                keep.append(slot)
        self.slots = sorted(keep[:cap], key=lambda slot: slot.turn)
