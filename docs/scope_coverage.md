# Scope coverage — every 4.3 bullet and pillar element, with code, test and number

One row per obligation in `docs/track4_problem_statement.md` (the organizer's launch
statement) and the four pillars. The rule this file answers: a judge greps a phrase and
finds a function, a test, and a measurement — not a paragraph. Measurements named D* are
in `analysis/declined_experiments.json`; W* are the V8 work items with their artifacts
under `analysis/v8_*.json`.

## Pillar I — Core architecture: intent routing & hybrid pipeline

| Element | Code | Test | Measurement |
|---|---|---|---|
| Intent detection, Buying/Browsing split | `src/routing.py` `detect_intent`, `src/state.py` `DialogState.read_intent` (cumulative evidence, not a turn-1 verdict) | `tests/test_pillar1_routing.py` | category axis +0.043 from the global route (`analysis/global_route.json`); dual-track behavior under paraphrase in `analysis/v4_end_to_end_controls.json` |
| High-precision filter track for "Buying" | exact shelf resolution: `src/routing.py` `exact_bucket` (longest-rightmost window scan) | `tests/test_pillar1_routing.py` | turn-1 pool recall 1.000 on the public set (`analysis/recall_decomposition.json`) |
| Diverse dense retrieval track for "Browsing" | `src/semantic.py` mined word→shelf vocabulary + browsing neighbours in `route_detail` | `tests/test_pillar1_routing.py` | "dense" is catalog-mined, not vector similarity; a real GloVe-50 route was built and declined (D1) |
| Multi-Route Retrieval → LLM Semantic Ranking | routes: `src/routing.py` `routes_for` (category, dense, lexical); LLM stage: `src/llm_rank.py` (opt-in, `TECHJAM_LLM_RERANK=1` + key, off by default, total fallback) | `tests/test_llm_stage.py` (guard, reorder, fallback matrix) | LLM ordering measured and declined as default: D11 (`analysis/llm_rerank_probe.json`); live demo `analysis/v8_w2_llm_stage.json` |
| keyword / category / vector similarity | keyword: BM25 in `src/scoring.py`; category: bucket routes in `src/routing.py`; vector: declined with evidence (D1) | `tests/test_scoring_policy.py`, `tests/test_pillar1_routing.py` | D1: 0.94401 → 0.94401 official, −0.04 top-1 shelf resolution — vector similarity earned nothing here |

## Pillar II — Dialog strategy: multi-turn scenario evolution

| Element | Code | Test | Measurement |
|---|---|---|---|
| Information Accumulation (incremental slots) | `src/state.py` typed slots with turn, polarity, weight | `tests/test_pillar2_dialog.py` `AccumulationTest` | distillation invariance: `tests/test_pillar3_evolution.py` `DistillationTest` |
| Intent Override — slot erasure and rewriting | rewriting: `src/state.py` `_retire` in decay mode (0.08 trace); **literal erasure**: `P_SUPERSEDE=erase` removes the slot from the structure (W3) | `tests/test_pillar2_dialog.py` `OverrideTest`, `ErasureModeTest` | erase measured on the shipped agent: −0.00275 official, one intent-override hit — decay stays default, erasure is a tested mode (`analysis/v8_w3_erasure.json`) |
| Over-Generality — immediate retrieval cutoff | pre-retrieval prediction: `src/routing.py` `estimated_pool_size` + `src/policy.py` `CommitPolicy.pre_cutoff`; post-retrieval confirmation: `CommitPolicy.cutoff` (W4) | `tests/test_pillar2_dialog.py` `PreRetrievalCutoffTest` (spy-proven scoring skip) | fires 58 turns / 42 sessions; score and transcript byte-identical; exact p99 10.76→9.32 ms (`analysis/v8_w4_cutoff.json`) |
| Structured proactive clarification | `src/clarify.py` (shelf disambiguation question) | `tests/test_clarify.py` | +0.167 hit rate and −1.2 turns when the shopper can answer (`tools/shelfbench.py --realistic`) |

## Pillar III — Self-evolution: dynamic context programming

| Element | Code | Test | Measurement |
|---|---|---|---|
| Personalized Context Distillation, short-term | `src/state.py` `distil` (information-weighted, standing slots pinned) | `tests/test_pillar3_evolution.py` `DistillationTest` | distilled state ranks within epsilon of full history |
| Long-term user profiles, continuously updated | `src/memory.py` `LongTermMemory` (write) **and its consumers**: `src/profile.py` recalled-shelf tie-break, `src/answerability.py` recalled-prior tie-settle (W1) | `tests/test_pillar3_evolution.py` `LongTermConsumptionTest` (two-session loop closure, cross-profile isolation) | zero effect on this harness by construction — isolated single-user sessions; transcript byte-identical (`analysis/v8_w1_longterm.json`) |
| Adaptive orchestration / re-orchestration | `src/orchestrator.py` `observe`/`strategy` (stall → broaden → cover) | `tests/test_pillar3_evolution.py` `ReOrchestrationTest` | no-op on healthy sessions is a tested invariant (D8-family) |

## Pillar IV — Evaluation matrix

| Element | Code | Test | Measurement |
|---|---|---|---|
| Coverage (HitRate@K) | `evaluator/local_evaluator.py` | `tests/test_metrics_integrity.py` | 1.000 on the 200-session public set (`analysis/final_numbers.json`) |
| Precision (MRR / Top-K) | same | same | MRR 0.773417; the learned reranker's +0.0207 dev / +0.0354 holdout (`analysis/snapshot_mrr.json`) |
| Efficiency (MTTC) | same | same | MTTC 1.795; composite 0.916125 |

## 4.3 limits

| Limit | Where enforced | Test |
|---|---|---|
| Max 10 turns | `src/policy.py` `MAX_TURNS`; the evaluator terminates the session | `tests/test_agent_contract.py` (turns 1..MAX_TURNS) |
| Read-only catalog, no mutation, no mock ASINs | catalog hash pinned (`da979b05…` in `analysis/v6_cycle_protocol.json` inputs); the agent builds an in-memory index and never writes | `tests/test_operational.py` (post-construction file-access block), evaluator checksum pinned in `tests/test_ranking_provenance.py` |

## 4.3 out of scope — the negatives, stated explicitly

- **No UI/UX** — backend API only (`src/agent.py` implements the published contract).
- **No training or fine-tuning of base LLMs** — no base model is trained anywhere in
  this repository; the reranker is a small linear model over interpretable features,
  exported as JSON (`analysis/reranker.json`).
- **No external vector DB clusters** — everything runs in memory
  (`tests/test_operational.py` resource guards; peak RSS ≈ 766 MB).
- **Text only** — no multimodal processing anywhere.

## The one sentence version

Every named element is a function with a test; the two that measured worse than the
default (LLM ordering, literal erasure) ship as real, tested, opt-in mechanisms with
the cost recorded — not as prose.
