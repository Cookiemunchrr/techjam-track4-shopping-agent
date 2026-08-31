# Three-minute demo script

Every number shown below is reproduced by a checked-in command or JSON artifact.
Do not replace the live terminal segments with canned output.

## 0:00–0:25 — The failure mode

Show a shopper changing from a vague use case to a precise product. Explain that a
single static query cannot represent accumulation, clarification, and intent
override in the same conversation.

## 0:25–0:55 — Architecture

Show the pipeline diagram in `README.md`: three-way intent routing, in-memory
multi-route retrieval, interpretable scoring plus the bounded linear reranker,
confidence-based commit, and online question answerability.

## 0:55–1:45 — Live multi-turn behavior

Run:

```bash
python3 demo_session.py --showcase
```

Call out the closed shelf question, the selected shelf, accumulated constraints,
the later product switch, and superseded product-bound slots. State on screen that
this is a deliberately chosen behavior demonstration, not an evaluation sample or
performance estimate.

## 1:45–2:15 — Explainability

Use the live session's top recommendation and `Scorer.explain`. Label the output
accurately as the base-score component breakdown; facet boosts, routing boosts, and
the learned reranker are additional stages.

## 2:15–2:45 — Evidence, including failed ideas

Show `results.json`, `analysis/bench.json`, and
`analysis/improvement_plan_v3_results.json`. Contrast official, shadow, snapshot
MRR, and adversarial measurements. Mention that category-query expansion and hedge
slate reservation were implemented but removed when their predeclared gates failed.

## 2:45–3:00 — Operations and close

Run or show the committed output of:

```bash
python3 -m tools.audit
```

Close on: standard library only, no network, zero runtime tokens, $0 scoring-model
cost, deterministic across three hash seeds, and a read-only catalog.
