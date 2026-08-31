"""Near-duplicate shelf micro-benchmark: does the *target item* survive an ambiguous shelf?

The failure analysis behind this file: on the category adversarial axis the agent
identifies the right *kind* of product almost every time and then picks the wrong
*shelf*. This catalog carries many near-duplicate shelves -- `Jewelry Necklaces`
and `Necklaces Chains`, `Tops & Tees Tanks & Camis` and `Tees & Blouses Tanks &
Camis` -- and a target lives on exactly one of them. Routing to the twin puts the
target outside the pool entirely, and no amount of ranking recovers it.

Two design decisions distinguish this from the axes in tools/bench.py, and both
follow from that diagnosis:

  it grades the item, not the shelf
      A bench that scores "did we resolve to shelf X" grades a proxy. The customer
      does not care which shelf we searched; they care whether their product is on
      screen. Earlier attempts at this problem were tuned against shelf top-1 and
      looked like noise, because shelf top-1 was never the thing that was broken.

  the ambiguous phrase is the shared words, not a synonym
      When two shelf names overlap, a shopper naming the product naturally utters
      the overlap: someone shopping either necklace shelf says "necklaces". That
      is a real utterance, not an adversarial rewrite, so this bench measures a
      failure the agent will meet in production and not only under tools/adversarial.

Rank is read from `Agent.internal_ranking` for the same reason tools/shadow.py does:
recommendation width must not be able to move the number.

One thing this bench cannot inherit from the evaluator is a shopper who can answer
a question about *which* shelf. `local_evaluator.classify_constraint` never returns
"category", so its simulated customer replies "I don't have an additional preference
for category" to a question any real person answers instantly. `--realistic` swaps
in a customer who names their own shelf when asked. That is a more faithful model of
a person, not a more generous one: they answer in their own words, and the agent only
benefits if the shelf it offered was among the ones it put to them.

Both numbers are reported, because they measure different things. The default is
what the official harness would score. `--realistic` is what the feature is worth.

    python3 -m tools.shelfbench                  # build pairs, run, report
    python3 -m tools.shelfbench --limit 120      # fewer sessions, faster loop
    python3 -m tools.shelfbench --realistic      # a customer who can say which shelf
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import uuid
from pathlib import Path

from evaluator.local_evaluator import (MAX_TURNS, TOP_K, catalog_index,
                                       classify_constraint, intent_card)

CATALOG = "data/catalog.jsonl"

# What counts as "near-duplicate". Two shelves qualify when their name stems
# overlap heavily and each is big enough that landing on the wrong one is a real
# loss rather than a rounding error.
MIN_JACCARD = 0.34
MIN_SHARED = 1
MIN_MEMBERS = 40
# Popularity strata. The public targets are 5-core purchases and skew popular, so
# a bench drawn only from shelf heads would flatter any popularity-led ranker.
STRATA = 3

PROFILE = {
    "average_prior_rating": 4.0,
    "preference_tags": ["fit", "quality"],
    "purchase_frequency": "3-4 prior purchases",
    "rating_style": "usually positive",
    "summary": "Prior purchases emphasize fit, quality; ratings are usually positive.",
}


def _stems(name: str) -> frozenset:
    from src.semantic import stem
    from src.text import tokens
    return frozenset(stem(word) for word in tokens(name))


def _surface(names, shared_stems) -> tuple:
    """Real words for a set of shared stems, longest surface form first.

    The stemmer truncates -- "shoes" stems to "sho" -- so handing the stems
    straight to the agent would open the session with a phrase no person has ever
    said, and a benchmark whose input is a stemmer artifact measures the stemmer.
    Map each shared stem back to the longest word either shelf name spells it
    with, and keep taxonomy order so the phrase reads the way the shelves do.
    """
    from src.semantic import stem
    from src.text import tokens

    best: dict[str, str] = {}
    order: dict[str, int] = {}
    for name in names:
        for position, word in enumerate(tokens(name)):
            root = stem(word)
            if root not in shared_stems:
                continue
            if len(word) > len(best.get(root, "")):
                best[root] = word
            order.setdefault(root, position)
    return tuple(best[root] for root in sorted(best, key=lambda r: order[r]))


def near_duplicate_pairs(catalog, min_members: int = MIN_MEMBERS) -> list[tuple[str, str, tuple]]:
    """Shelf pairs whose names overlap enough to be confusable, with the shared words.

    Returns (shelf_a, shelf_b, shared_words) sorted for determinism. The shared
    words are the benchmark's input: they are what a shopper says when the word
    they reach for belongs to both shelves. Surface words, not stems -- see
    `_surface` for why that distinction is not cosmetic.
    """
    big = {name: _stems(name) for name, members in catalog.buckets.items()
           if len(members) >= min_members and _stems(name)}
    by_stem: dict[str, list[str]] = {}
    for name, stems in big.items():
        for token in stems:
            by_stem.setdefault(token, []).append(name)

    pairs: dict[tuple[str, str], tuple] = {}
    for names in by_stem.values():
        if len(names) < 2:
            continue
        for i, left in enumerate(sorted(names)):
            for right in sorted(names)[i + 1:]:
                a, b = big[left], big[right]
                shared = a & b
                if len(shared) < MIN_SHARED:
                    continue
                if len(shared) / len(a | b) < MIN_JACCARD:
                    continue
                pairs[(left, right)] = _surface((left, right), shared)
    return [(left, right, shared) for (left, right), shared in sorted(pairs.items())]


def build_sessions(catalog, products: dict, limit: int, seed: int = 0) -> list[dict]:
    """One session per (pair, shelf, popularity stratum), capped at `limit`.

    Both shelves of a pair are used as the target's home, so the bench cannot be
    passed by always preferring the alphabetically-first or the larger shelf.
    """
    rng = random.Random(seed)
    pairs = near_duplicate_pairs(catalog)
    rows: list[dict] = []
    for left, right, shared in pairs:
        phrase = " ".join(shared)
        for home, twin in ((left, right), (right, left)):
            members = sorted(catalog.buckets[home],
                             key=lambda pid: (-catalog.meta[pid]["pop"], pid))
            if len(members) < STRATA:
                continue
            size = len(members) // STRATA
            for index in range(STRATA):
                band = members[index * size:(index + 1) * size] or members
                target = band[rng.randrange(len(band))]
                if target not in products:
                    continue
                rows.append({
                    "sample_id": f"shelf_{len(rows):05d}",
                    "scenario_type": "buying",
                    "ground_truth": {"parent_asin": target},
                    "user_profile": dict(PROFILE),
                    "shelf": home,
                    "twin": twin,
                    "ambiguous_phrase": phrase,
                    "stratum": index,
                })
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def _shelf_reply(shelf: str) -> str:
    """What a person says when asked which of several categories they meant.

    Their own words for their own shelf, not a selection from a menu -- a customer
    answers "I'm after baseball caps", they do not read an option back verbatim.
    Whether that lands is then the agent's problem, which is the point.
    """
    from src.clarify import _readable
    return f"I'm after {_readable(shelf)}."


def _reply(card: dict, attribute: object, disclosed: set) -> str:
    """The evaluator's disclosure rule with the `other` wildcard removed.

    Same reasoning as tools/shadow.py: the wildcard is a simulator artifact and a
    bench that leans on it would reward exploiting it.
    """
    attribute = attribute if isinstance(attribute, str) else None
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute."
    constraints = [*[str(v) for v in card.get("hard_constraints", [])],
                   *[str(v) for v in card.get("soft_preferences", [])]]
    matches = [v for v in constraints
               if v not in disclosed and classify_constraint(v) == attribute][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}."
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + "."


def _ranking(agent, session_id: str, response: dict, catalog_ids: set) -> list[str]:
    getter = getattr(agent, "internal_ranking", None)
    ranked = list(getter(session_id)) if callable(getter) else []
    if not ranked:
        ranked = [item.get("parent_asin", "") if isinstance(item, dict) else item
                  for item in response.get("recommendations") or []]
    out, seen = [], set()
    for pid in map(str, ranked):
        if pid and pid not in seen and pid in catalog_ids:
            seen.add(pid)
            out.append(pid)
        if len(out) >= TOP_K:
            break
    return out


def run(agent_factory, sessions: list[dict], catalog_ids: set, products: dict,
        realistic: bool = False, clarify: bool = True) -> dict:
    """One pass over the sessions.

    `clarify=False` is the ablation: it silences src/clarify.py so the same agent
    can be measured with and without the question. Nothing else changes, so the
    difference between the two rows is attributable to asking alone.
    """
    if not clarify:
        import src.agent as agent_module
        original = agent_module.shelf_ambiguous
        agent_module.shelf_ambiguous = lambda *args, **kwargs: False
        try:
            return run(agent_factory, sessions, catalog_ids, products, realistic, clarify=True)
        finally:
            agent_module.shelf_ambiguous = original
    agent = agent_factory()
    results: list[dict] = []
    for row in sessions:
        session_id = f"shelf_{uuid.uuid4().hex}"
        agent.reset(session_id, row["user_profile"])
        target = str(row["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        disclosed: set[str] = set()
        hard = [str(v) for v in card.get("hard_constraints", [])]
        opener = hard[0] if hard else ""
        if opener:
            disclosed.add(opener)
        message = (f"I'm looking for {row['ambiguous_phrase']}. A key requirement is: {opener}."
                   if opener else f"I'm looking for {row['ambiguous_phrase']}.")
        hit_turn = best_rank = None
        elected_first = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if elected_first is None:
                elected_first = _elected(agent, session_id)
            ranked = _ranking(agent, session_id, response, catalog_ids)
            if target in ranked:
                best_rank, hit_turn = ranked.index(target) + 1, turn
                break
            if turn == MAX_TURNS:
                break
            asked = response.get("ask_attribute")
            if realistic and asked == "category":
                message = _shelf_reply(row["shelf"])
            else:
                message = _reply(card, asked, disclosed)
        results.append({
            "sample_id": row["sample_id"], "shelf": row["shelf"], "twin": row["twin"],
            "stratum": row["stratum"], "hit": hit_turn is not None,
            "first_hit_turn": hit_turn, "rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            "elected_correct": None if elected_first is None else elected_first == row["shelf"],
            "elected_at_hit": _elected(agent, session_id) == row["shelf"],
        })
    return summarise(results)


def _elected(agent, session_id: str):
    """Which shelf the agent is currently searching, if it exposes one."""
    getter = getattr(agent, "elected_shelf", None)
    return getter(session_id) if callable(getter) else None


def summarise(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {"sample_count": 0}
    hit = sum(int(r["hit"]) for r in results) / n
    mrr = statistics.fmean(r["reciprocal_rank"] for r in results)
    mttc = statistics.fmean(r["first_hit_turn"] if r["first_hit_turn"] else MAX_TURNS + 1
                            for r in results)
    known = [r for r in results if r["elected_correct"] is not None]
    strata = {}
    for index in range(STRATA):
        band = [r for r in results if r["stratum"] == index]
        if band:
            strata[f"stratum_{index}"] = {
                "sample_count": len(band),
                "hit_rate_at_10": round(sum(int(r["hit"]) for r in band) / len(band), 4),
                "mrr": round(statistics.fmean(r["reciprocal_rank"] for r in band), 4),
            }
    return {
        "sample_count": n,
        "hit_rate_at_10": round(hit, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        # Diagnostics. Reported, never optimised against: shelf accuracy is the
        # proxy this bench exists to stop grading.
        "elected_correct_turn1": round(
            sum(int(bool(r["elected_correct"])) for r in known) / len(known), 4) if known else None,
        "elected_correct_at_hit": round(
            sum(int(bool(r["elected_at_hit"])) for r in results) / n, 4),
        "strata": strata,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Near-duplicate shelf micro-benchmark")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realistic", action="store_true",
                        help="a customer who can say which shelf they meant")
    parser.add_argument("--no-clarify", action="store_true",
                        help="ablation: silence the clarification question")
    parser.add_argument("--output", default="analysis/shelfbench.json")
    args = parser.parse_args()

    from src.agent import Agent
    ids, _categories, products = catalog_index(args.catalog)
    base = Agent(args.catalog)
    sessions = build_sessions(base.catalog, products, args.limit, args.seed)
    row = run(lambda: Agent.sharing_index(base), sessions, ids, products,
              args.realistic, clarify=not args.no_clarify)
    print("  pairs=%d  sessions=%d  HR=%.3f  MRR=%.4f  MTTC=%.2f  shelf@1=%s" % (
        len(near_duplicate_pairs(base.catalog)), row["sample_count"],
        row["hit_rate_at_10"], row["mrr"], row["mttc"], row["elected_correct_turn1"]))
    for name, band in row["strata"].items():
        print("    %-12s n=%-4d HR=%.3f  MRR=%.4f" % (name, band["sample_count"],
                                                      band["hit_rate_at_10"], band["mrr"]))
    Path(args.output).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
