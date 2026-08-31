"""Asking which shelf, when the shelf cannot be inferred.

analysis/shelf_election_finding.json closes the inference question: near-duplicate
shelves stock genuinely similar products, so no amount of constraint evidence
separates them. Across five turns of accumulating constraints the shelf posterior
never sharpened -- top-1 stayed between 0.15 and 0.23, maximum probability stayed
at 0.33. And the same measurement says what it is worth: with the shelf known, hit
rate goes from 0.554 to 0.771 against a pool ceiling of 0.787.

So the information exists, it is worth a great deal, and it is not in the catalog.
It is in the customer, and the way to get something from a customer is to ask.

That is Pillar II's proactive guidance, arriving at the case it was written for. The
existing over-generality cutoff fires on *pool size* -- too many candidates to rank
meaningfully. This fires on a different kind of overload: the pool is a perfectly
reasonable size and we cannot tell which half of it the customer meant. Both end
the same way, with a structured question instead of a shrug.

What makes the question answerable is that it is closed. "What category?" is a worse
question than "did you mean baseball caps, beanies, or sun hats?" -- the second
names the actual shelves in contention, so the customer recognises rather than
recalls, and the answer maps back onto a shelf without parsing.

## What this costs on the official harness

Nothing, and that is the problem. `local_evaluator.classify_constraint` never
returns "category", so `customer_reply` answers every category question with "I
don't have an additional preference for category" -- the simulated shopper cannot
tell us which shelf they meant, which no real shopper would find difficult. And
0/200 public sessions reach this code at all, because the opening message is
`coarse_category(target.categories)` verbatim and always resolves outright.

So this is shipped on the second door: product-justified, measured against a
shopper who can answer, and disclosed as a cost where the harness cannot. See
tools/shelfbench.py --realistic for the measurement, and the guard in `should_ask`
for what keeps the cost bounded.
"""
from __future__ import annotations

from .text import normalise, tokens

# How many shelves to offer. Three is the most a person can hold in a sentence and
# still answer from recognition; past that the question is its own cognitive load,
# which is the thing the efficiency metric exists to penalise.
MAX_OPTIONS = 3
# Below this many candidate shelves there is no real ambiguity to resolve and the
# question is a wasted turn.
MIN_CANDIDATES = 2


def options(catalog, hedge, limit: int = MAX_OPTIONS) -> list[str]:
    """The shelves worth naming in a clarification, best first.

    Ordered by retrieval confidence rather than by constraint evidence, on the
    finding that the category phrase is the better-calibrated signal -- the whole
    reason evidence-based re-election was declined.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in hedge:
        if name in seen or not catalog.buckets.get(name):
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out


def should_ask(catalog, hedge, already_asked: bool) -> bool:
    """Is the shelf ambiguous enough to be worth a turn?

    Asked at most once per session. A customer who has answered this has told us
    everything they can about which shelf they meant, and asking again spends a
    turn to learn nothing -- the same reasoning that stops the elicitation model
    re-asking a declined attribute.
    """
    if already_asked or not hedge:
        return False
    return len(options(catalog, hedge)) >= MIN_CANDIDATES


def phrasing(offered: list[str]) -> str:
    """The structured clarification prompt itself."""
    names = [_readable(name) for name in offered]
    if len(names) == 1:
        return f"Just to check -- is it {names[0]} you're after?"
    listed = ", ".join(names[:-1]) + f", or {names[-1]}"
    return f"A couple of these could fit -- are you after {listed}?"


def match(message: str, offered: list[str]) -> str | None:
    """Which offered shelf the customer picked, if any.

    Matches on the distinctive words of each option rather than on the whole name:
    a customer offered "Hats & Caps Baseball Caps" answers "the baseball ones", and
    the words that identify their choice are the ones the options do *not* share.
    Requiring the full name back would make the question unanswerable in practice.
    """
    if not message or not offered:
        return None
    spoken = set(tokens(message))
    if not spoken:
        return None

    per_option = [set(tokens(name)) for name in offered]
    shared = set.intersection(*per_option) if len(per_option) > 1 else set()

    best, best_score = None, 0.0
    for name, words in zip(offered, per_option):
        distinctive = words - shared
        if not distinctive:
            continue
        hits = len(distinctive & spoken)
        if not hits:
            continue
        # Fraction of what makes this option distinctive that the customer said.
        score = hits / len(distinctive)
        if score > best_score:
            best, best_score = name, score
    if best_score <= 0.0:
        return None
    return best


def _readable(name: str) -> str:
    """A shelf name as a person would say it.

    Taxonomy names repeat their parent ("Hats & Caps Baseball Caps"), which reads
    as a stutter out loud. Keep the tail -- in this taxonomy the specific noun
    comes last -- and drop the leading component when it echoes the tail.
    """
    parts = [part for part in name.split() if part]
    if len(parts) <= 2:
        return normalise(name) or name.lower()
    tail = " ".join(parts[-2:])
    head = " ".join(parts[:-2])
    if set(tokens(head)) & set(tokens(tail)):
        return normalise(tail) or tail.lower()
    return normalise(name) or name.lower()
