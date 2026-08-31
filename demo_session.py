#!/usr/bin/env python3
"""One annotated multi-turn session, end to end.

    python3 demo_session.py                 # a buying session
    python3 demo_session.py --scenario intent_override
    python3 demo_session.py --sample public_0002
    python3 demo_session.py --all           # one of each scenario
    python3 demo_session.py --ambiguous     # a shelf the phrase cannot resolve

Required deliverable ("one demonstrated multi-turn session") and the script the
demo video follows. Drives the organizer's own simulator, so the customer's turns
are exactly what the scored evaluation would send -- nothing here is scripted.

`--ambiguous` is the exception, and the reason it exists is worth reading. The
public set never produces an ambiguous shelf: the opening message is
`coarse_category(target.categories)` verbatim, so all 200 sessions resolve a shelf
outright and the clarification path in src/clarify.py runs zero times. That path
is worth +0.167 hit rate and 1.2 turns against a customer who can answer it
(tools/shelfbench.py --realistic), so a demo that only ever shows the resolved case
would leave out the most consequential behaviour in the agent. The customer here
is still simulated
from the organizer's `intent_card`; the only change is that they open with the
words two shelves share, which is what a person says when the word they reach for
belongs to both, and that they can say which one they meant.
"""
from __future__ import annotations

import argparse
import json
import sys

from evaluator.local_evaluator import (MAX_TURNS, TOP_K, catalog_index, coarse_category,
                                       customer_reply, initial_message, load_jsonl,
                                       materialize_hidden_fields, normalize_recommendations)
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
PUBLIC = "data/public_set.jsonl"

BAR = "-" * 78


def _title(agent: Agent, parent_asin: str, width: int = 58) -> str:
    text = agent.catalog.corpus.get(parent_asin, "")
    return (text[:width] + "...") if len(text) > width else text


def run(agent: Agent, sample: dict, ids, categories, products, verbose: bool = True) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behaviour = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behaviour}
    disclosed: set = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    session = f"demo_{sample['sample_id']}"
    agent.reset(session, sample["user_profile"])
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    if verbose:
        print(BAR)
        print(f"{sample['sample_id']}   scenario={sample['scenario_type']}   "
              f"hidden target={target}")
        print(f"   the customer wants : {_title(agent, target)}")
        print(f"   profile            : {sample['user_profile']['summary']}")
        print(BAR)

    hit_turn = rank = None
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session, message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), ids)
        state = agent.sessions[session]

        if verbose:
            print(f"\nTURN {turn}")
            print(f"  customer : {message}")
            print(f"  agent    : {response['message']}")
            print(f"  ask      : {response['ask_attribute']}")
            print(f"  shows    : {len(ranked)} item(s)")
            for position, parent_asin in enumerate(ranked[:3], start=1):
                mark = "  <-- the target" if parent_asin == target else ""
                print(f"     {position}. {parent_asin}  {_title(agent, parent_asin, 44)}{mark}")
            print(f"  state    : category={state.dialog.category!r} "
                  f"intent={state.dialog.intent} strategy={state.orchestrator.strategy()}")
            for slot in state.dialog.slots:
                flag = "retracted" if slot.superseded else ("refused" if slot.polarity < 0 else "")
                print(f"             slot {slot.attribute:<9} {slot.text[:44]!r} {flag}")

        if override_applied and target in ranked:
            hit_turn, rank = turn, ranked.index(target) + 1
            if verbose:
                print(f"\n  HIT at turn {turn}, rank {rank}.")
            break
        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore that."))
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used)

    if verbose and hit_turn is None:
        print("\n  MISS: ten turns without surfacing the target.")
    return {"hit": hit_turn is not None, "turn": hit_turn, "rank": rank}


def _first_clarifying(agent: Agent, rows, products, limit: int = 400):
    """The first session in which turn 1 produces a clarification question."""
    from evaluator.local_evaluator import intent_card

    for row in rows[:limit]:
        target = str(row["ground_truth"]["parent_asin"])
        if len(agent.catalog.buckets[row["shelf"]]) < 60:
            continue
        hard = [str(v) for v in intent_card(products[target]).get("hard_constraints", [])]
        opener = (f"I'm looking for {row['ambiguous_phrase']}. A key requirement is: {hard[0]}."
                  if hard else f"I'm looking for {row['ambiguous_phrase']}.")
        probe = f"probe_{row['sample_id']}"
        agent.reset(probe, row["user_profile"])
        if agent.respond(probe, opener, 1, TOP_K).get("ask_attribute") == "category":
            agent.sessions.pop(probe, None)
            return row
        agent.sessions.pop(probe, None)
    return rows[0]


def run_ambiguous(agent: Agent, ids, products, seed: int = 0) -> dict:
    """One session where the opening phrase names two shelves at once."""
    from evaluator.local_evaluator import intent_card
    from tools.shelfbench import _reply, _shelf_reply, build_sessions

    rows = build_sessions(agent.catalog, products, limit=0, seed=seed)
    # Scan for a session where the clarification actually fires. Ambiguity is a
    # property of the phrase, not of every near-duplicate pair -- some shared
    # phrases still resolve outright -- so picking the first row would as often
    # demonstrate the ordinary path. Deterministic, and stated rather than hidden:
    # this is a chosen example of a behaviour, not a sampled one.
    row = _first_clarifying(agent, rows, products)
    target = str(row["ground_truth"]["parent_asin"])
    card = intent_card(products[target])
    hard = [str(v) for v in card.get("hard_constraints", [])]
    disclosed = set(hard[:1])

    session = "demo_ambiguous"
    agent.reset(session, row["user_profile"])
    message = (f"I'm looking for {row['ambiguous_phrase']}. A key requirement is: {hard[0]}."
               if hard else f"I'm looking for {row['ambiguous_phrase']}.")

    print(BAR)
    print(f"ambiguous shelf   the phrase {row['ambiguous_phrase']!r} names both")
    print(f"   {row['shelf']}  ({len(agent.catalog.buckets[row['shelf']])} items)  <-- the target lives here")
    print(f"   {row['twin']}  ({len(agent.catalog.buckets[row['twin']])} items)")
    print(f"   hidden target  : {target}  {_title(agent, target)}")
    print(BAR)

    hit_turn = rank = None
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session, message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), ids)
        state = agent.sessions[session]
        print(f"\nTURN {turn}")
        print(f"  customer : {message}")
        print(f"  agent    : {response['message']}")
        print(f"  ask      : {response['ask_attribute']}")
        print(f"  shows    : {len(ranked)} item(s)")
        for position, parent_asin in enumerate(ranked[:3], start=1):
            mark = "  <-- the target" if parent_asin == target else ""
            print(f"     {position}. {parent_asin}  {_title(agent, parent_asin, 44)}{mark}")
        print(f"  state    : category={state.dialog.category!r} "
              f"offered={state.offered or None}")
        if target in ranked:
            hit_turn, rank = turn, ranked.index(target) + 1
            print(f"\n  HIT at turn {turn}, rank {rank}.")
            break
        if turn == MAX_TURNS:
            break
        asked = response.get("ask_attribute")
        message = (_shelf_reply(row["shelf"]) if asked == "category"
                   else _reply(card, asked, disclosed))

    if hit_turn is None:
        print("\n  MISS: ten turns without surfacing the target.")
    return {"hit": hit_turn is not None, "turn": hit_turn, "rank": rank}


def run_showcase(agent: Agent) -> None:
    """A deterministic live behavior demo: clarification, accumulation, switch.

    This is deliberately chosen for pedagogical clarity. It is not an evaluator
    sample and makes no performance claim. Every response and state value is
    produced by the shipped Agent; assertions make the command fail if the
    demonstrated behavior drifts.
    """
    from src.routing import exact_bucket

    session = "demo_showcase"
    agent.reset(session, {})
    messages = [
        "I am looking for outdoor work. Synthetic is fine, but I am "
        "not sure which kind.",
        "Rain gear \u2014 I need something waterproof.",
        "Actually, I need Accessories Belts instead. Leather, buckle closure.",
        "Brown would be ideal, and keep it under $50.",
    ]

    print(BAR)
    print("CHOSEN BEHAVIOR SHOWCASE (not an evaluation sample or score estimate)")
    print("closed shelf clarification -> accumulated evidence -> product switch")
    print(BAR)

    old_product_slots = []
    asked_before_switch: set[str] = set()
    final_response = None
    for turn, message in enumerate(messages, 1):
        response = agent.respond(session, message, turn, TOP_K)
        state = agent.sessions[session]
        ranked = [str(item["parent_asin"]) for item in response["recommendations"]]
        resolved = exact_bucket(agent.catalog, state.dialog.category or "")

        print(f"\nTURN {turn}")
        print(f"  customer : {message}")
        print(f"  agent    : {response['message']}")
        print(f"  ask      : {response['ask_attribute']}")
        print(f"  shelf    : {resolved or state.dialog.category}")
        print(f"  shows    : {len(ranked)} item(s)")
        for position, parent_asin in enumerate(ranked[:3], 1):
            print(f"     {position}. {parent_asin}  {_title(agent, parent_asin, 48)}")
        for slot in state.dialog.slots:
            flag = "retracted" if slot.superseded else "active"
            print(f"  slot     : {slot.attribute:<9} {slot.text[:44]!r} {flag}")

        if turn == 1:
            if response["ask_attribute"] != "category" \
                    or "Outdoor & Work Rain" not in state.offered:
                raise RuntimeError("showcase drift: closed rain-shelf question was not offered")
        elif turn == 2:
            if resolved != "Outdoor & Work Rain":
                raise RuntimeError(f"showcase drift: rain shelf did not resolve ({resolved!r})")
            old_product_slots = [slot for slot in state.dialog.slots if not slot.superseded]
            asked_before_switch = set(state.asked)
        elif turn == 3:
            if resolved != "Accessories Belts":
                raise RuntimeError(f"showcase drift: belt switch did not route ({resolved!r})")
            if old_product_slots and not all(slot.superseded for slot in old_product_slots):
                raise RuntimeError("showcase drift: old product constraints remained active")
            if "brand" in asked_before_switch and "brand" not in state.asked:
                raise RuntimeError("showcase drift: person-level brand history was erased")
            if any("accessories belts" in slot.text.lower()
                   for slot in state.dialog.slots if not slot.superseded):
                raise RuntimeError("showcase drift: category leaked into feature evidence")
        final_response = response

    state = agent.sessions[session]
    if agent.failures:
        raise RuntimeError(f"showcase swallowed {agent.failures} internal failure(s)")
    top = str(final_response["recommendations"][0]["parent_asin"])
    explanation = agent.scorer.explain(top, state.dialog.weighted_phrases(), state.dialog.budget())
    print("\nBASE-SCORE COMPONENTS FOR THE FINAL TOP ITEM")
    print("  (routing/facet boosts and the learned reranker are separate stages)")
    print(json.dumps(explanation, indent=2, sort_keys=True))
    print("\nshowcase assertions: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one annotated demo session")
    parser.add_argument("--scenario", default="buying",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--sample", help="a specific sample_id, e.g. public_0002")
    parser.add_argument("--all", action="store_true", help="one session per scenario")
    parser.add_argument("--ambiguous", action="store_true",
                        help="a near-duplicate shelf the opening phrase cannot resolve")
    parser.add_argument("--showcase", action="store_true",
                        help="chosen live clarification + product-switch demonstration")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dataset", default=PUBLIC)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    ids, categories, products = catalog_index(args.catalog)
    print(f"indexing {args.catalog} ...", file=sys.stderr)
    agent = Agent(args.catalog)

    if args.showcase:
        run_showcase(agent)
        print("\ntoken usage: 0 prompt, 0 completion. No network, no model, no API key.")
        return

    if args.ambiguous:
        outcome = run_ambiguous(agent, ids, products)
        print(f"\nresult: {outcome}")
        print("\ntoken usage: 0 prompt, 0 completion. No network, no model, no API key.")
        return

    if args.sample:
        chosen = [s for s in samples if s["sample_id"] == args.sample]
        if not chosen:
            raise SystemExit(f"no sample {args.sample!r} in {args.dataset}")
    elif args.all:
        chosen = [next(s for s in samples if s["scenario_type"] == scenario)
                  for scenario in ("buying", "browsing", "intent_override", "boundary")]
    else:
        chosen = [next(s for s in samples if s["scenario_type"] == args.scenario)]

    for sample in chosen:
        outcome = run(agent, sample, ids, categories, products)
        print(f"\n{BAR}\nresult: {outcome}\n")

    print("token usage: 0 prompt, 0 completion. No network, no model, no API key.")


if __name__ == "__main__":
    main()
