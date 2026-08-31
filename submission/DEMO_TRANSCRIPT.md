# Demonstrated multi-turn sessions

Two sessions, recorded live from the submission tree. The first is a chosen
behavior showcase; the second is an unmodified evaluator session that converts,
so the bundle contains a real scored conversion and not only a demonstration.

- Recorded: 2026-08-31 (Asia/Singapore)
- Source state: run `git rev-parse HEAD` in the frozen repository used for submission;
  the live command below contains assertions and exits nonzero if its behavior drifts.
- Catalog SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- Evaluator SHA-256: `79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564`
- Both sessions: 0 prompt tokens, 0 completion tokens, no network, no model, no API key.

---

## Session A — behavior showcase

- Command: `python3 demo_session.py --showcase`

This is a deliberately chosen product-behavior demonstration, not an evaluator
sample or performance estimate. The command executes the shipped `Agent` live and
fails nonzero if its clarification or state-transition assertions drift. The transcript
below was re-recorded from the current submission tree after those assertions passed.

## Transcript

### Turn 1 — recognize an ambiguous shelf

Customer:

> I am looking for outdoor work. Synthetic is fine, but I am not
> sure which kind.

Agent:

> A couple of these could fit -- are you after outdoor & work hunting, outdoor &
> work rain, or outdoor & work work & safety?

Structured question: `category`. Eight live recommendations were shown.

### Turn 2 — answer and accumulate

Customer:

> Rain gear — I need something waterproof.

The live state resolves to `Outdoor & Work Rain`; the agent asks for `brand` and
shows eight rain products. `something waterproof` is retained as active evidence.

### Turn 3 — switch products

Customer:

> Actually, I need Accessories Belts instead. Leather, buckle closure.

The live state resolves to `Accessories Belts`. Both old outdoor-work slots are
marked retracted, `Leather, buckle closure` remains active as `material`, and the
routed category is not duplicated as feature evidence. Five belt products are
shown; the first is `B08FLYBP2J`.

### Turn 4 — add color and budget

Customer:

> Brown would be ideal, and keep it under $50.

The final state retains the active material and adds `color=brown` plus a budget
ceiling. Eight products are shown. The final top product is `B0CHZYH847`; its
base-score component breakdown is:

```json
{
  "budget": 0.0,
  "lexical": 0.2933894323931303,
  "phrase": 0.0,
  "popularity": 0.539277317190251,
  "total": 0.8326667495833813
}
```

Routing/facet boosts and the learned reranker are separate stages from this base
breakdown. The live command ended with `showcase assertions: PASS`, zero swallowed
failures, zero tokens, no network, and no model/API key.

---

## Session B — a real evaluator session that converts

- Command: `python3 demo_session.py --scenario intent_override`
- Sample: `public_0002`, the first `intent_override` session in the public set
- Hidden target: `B071X54486` — *hide & drink, rustic handmade full grain leather men's belt*

Unlike Session A, nothing here is chosen for effect. This is the organizer's own
simulator driving the shipped agent through a scored session, and it ends in a hit.
The customer's profile reads *"Prior purchases emphasize fit, comfort, style;
ratings are critical."*

### Turn 1 — a buying opening with one disclosed constraint

Customer:

> I'm looking for Accessories Belts. Buckle closure

Agent asks `brand` and shows 5 items. The target is already at **rank 2**, but an
intent-override session cannot convert before the override is sent (evaluator rule:
`override_applied` only flips on turn 3 or 4), so this does not score. State:
`category='accessories belts'`, `intent=buying`, one active `feature` slot.

### Turn 2 — the customer declines to answer

Customer:

> I don't have an additional preference for brand.

This is a *no-preference* reply, not a negative constraint. The agent does not add
`brand` as evidence against anything; it records that brand was asked and not
answered, drops `intent` to `open`, and widens to 8 items. This is the
`NO_PREFERENCE` path described in the README — filing it as a refusal would have
penalised 22.1% of the catalog.

### Turn 3 — intent override, and the hit

Customer:

> Actually, ignore my earlier preference. What I need is: leather.

Agent asks `style` and shows 5 items:

```
1. B08L13H7SY  bulliant men's belt, reversible belt 1.25" for men
2. B071X54486  hide & drink, rustic handmade full grain leather men's belt   <-- target
3. B01N44118B  maikun mens dress leather belt plaque buckle
```

**HIT at turn 3, rank 2.** Evaluator result: `{'hit': True, 'turn': 3, 'rank': 2}`,
contributing reciprocal rank 0.5 and first-hit turn 3 to the official aggregate.

What the state shows at that moment is the honest version of intent override:

```
slot feature   'Buckle closure'
slot material  'leather'
```

The earlier `feature` slot is **not deleted**. It is retained at
`SUPERSEDED_WEIGHT = 0.08` (`src/state.py`), so a retracted preference keeps a decayed
trace rather than vanishing. See README *Compliance scope* — the specification's phrase
is "slot erasure", and this is a rewrite plus a decayed trace, which is not the same
thing. It is also why the target held rank 2 across the override instead of being
re-retrieved from scratch.

Across all 30 public `intent_override` sessions the agent reaches HR@10 1.000, MRR
0.864444, MTTC 4.000 (`analysis/final_numbers.json`).
