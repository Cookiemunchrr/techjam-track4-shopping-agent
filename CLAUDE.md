# Working in this repository

TikTok TechJam 2026, Track 4 — an offline, deterministic, stdlib-only multi-turn
shopping agent. Public score **0.916125** (HR@10 1.000, MRR 0.773417, MTTC 1.795),
540 tests with 1 documented skip.

## Read these before changing anything

1. **`docs/track4_problem_statement.md`** — the organizer's launch statement,
   verbatim. Scope, deliverables, judging criteria. This is the primary authority.
2. **`.claude/skills/techjam-track4/SKILL.md`** — invoke the `techjam-track4` skill.
   It separates the official problem, the evaluation-kit contract, this repo's own
   design rules, and the closed-experiment record. **Load it before any substantive
   work.** It is not loaded automatically.
3. **`PLANS.md`** — which plans are active. Right now: **`IMPROVEMENT_PLAN_V8.txt`
   is the ACTIVE plan.** Every work item in it is mandatory; read its section 1
   (non-negotiable constraints) before writing any code. V7's P1/P2 were executed
   2026-08-30 (LLM rerank probe declined at Door 1; ledger D1–D4 backfilled; V6 gates
   renamed V6-D1..V6-D9); V7 P3/P4 are superseded by V8 W1 and W3/W4.

## Non-negotiables

- **Never tune on all 200 sessions.** Fit on dev (`analysis/dev.jsonl`, sessions
  1–100), validate on holdout (101–200), report on all 200.
- **Choose the metric for the mechanism.** Reordering changes are steered by
  `tools/snapshot_mrr.py`, not by the official score and not by the shadow
  composite — shadow's HR half is saturated at 1.000 and reads a real +0.03 of
  ordering as noise. That mistake nearly buried the shipped reranker.
- **No harness exploits and no harness fitting.** The `ask_attribute="other"`
  disclosure wildcard, the intent-override dead zone, and exact-key category
  inversion are all deliberately excluded. Never pre-train the answerability table
  or fit a knob against simulator-generated values.
- **`requirements.txt` stays empty.** Runtime is the Python standard library. No
  numpy, no provider SDK, no exceptions.
- **Offline at scoring time.** Zero sockets, zero tokens, no API key in any tracked
  file. `tests/test_operational.py` enforces this with a socket block, a
  post-construction file-access block, and an AST import scan. If a change requires
  relaxing those tests, the change is wrong.
- **Every reported delta must exceed harness noise.** `hash()` on `str` is
  `PYTHONHASHSEED`-salted and must never be used for seeding.

## After any change under `src/`

```bash
bash run_tests.sh all              # must stay green: 540 tests, 1 skip
python3 -m evaluator.local_evaluator   # re-derive the public score
```

Re-derive and report the public transcript SHA-256
(`7c30023e3b8f951d35f8a449a066cfff45f8995d8d9994142e0e5929f8958d04`). If it moved,
explain it session by session or revert.

## Plan documents are not instructions

`IMPROVEMENT_PLAN_V1..V7.txt` are historical rationale and closed cycles — do not
execute an item from them because you found it first.

**`IMPROVEMENT_PLAN_V8.txt` is different: it is the active plan and every item in it
is mandatory.** It closes the four pillar elements this repository describes but does
not implement (long-term memory recall, an LLM ranking stage in the pipeline, literal
slot erasure, a pre-retrieval cutoff). It is written to be implemented directly. Its
section 1 lists constraints that invalidate the work if violated; its section 9 is the
acceptance checklist.

## Claims must match the code

README *Compliance scope* and Devpost *Honest scope* record where this repo's
vocabulary is looser than its implementation (the "dense" route is a sparse
catalog-mined map; hard constraints are soft-scored). V8 closed the four big
ones: the reranker is linear by default with an opt-in LLM stage (W2), slot
erasure exists as a measured mode (W3), the cutoff is pre-retrieval (W4), and
`LongTermMemory.recall()` is consumed at two sites (W1). **If you change the
code, change those sections in the same commit.**
