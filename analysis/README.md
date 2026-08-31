# TechJam 2026 — Track 4 analysis bundle

The current authority is the repository root `README.md`, the binding
`.claude/skills/techjam-track4/SKILL.md`, and `../IMPROVEMENT_PLAN_V4.txt`. The three
committed V4-0 aggregate artifacts are:

- `resource_probe_v4_0.json` — separate-process exact/hedged/fallback measurements;
- `v4_end_to_end_controls.json` — complete identifier-stripped official, fixed-width,
  adversarial, shadow, operational, protected-trace, and resource controls;
- `improvement_plan_v4_results.json` — validated cache identities, protected internal
  order/metric pins, reachability, pair mass, rank changes, and slices.

Reproduce them from the repository root after fetching the catalog and generating the
ignored caches/manifests:

```bash
python3 -m tools.rerank_data --split dev
python3 -m tools.rerank_data --split holdout
python3 -m tools.snapshot_mrr --splits dev,holdout,full --rebuild
python3 -m tools.resource_probe --include-fallback \
  --output analysis/resource_probe_v4_0.json
python3 -m tools.v4_controls \
  --resource analysis/resource_probe_v4_0.json \
  --output analysis/v4_end_to_end_controls.json
python3 -m tools.v4_baseline \
  --output analysis/improvement_plan_v4_results.json
```

The build caches and `.manifest.jsonl` sidecars are deliberately ignored. Manifests
contain exact catalog IDs and ordered split target/session records so validators can fail
closed; committed aggregate reports do not publish those identifiers. V4-0 changes no
serving model or evaluator behavior.

## Archived V1/V2 material

The material below documents the early `agent_v1_probe.py`/`agent_v2.py` investigation.
Its 0.89431 headline and setup commands are historical and must not be mistaken for the
current 0.916125 agent or the V4 reproduction protocol. It is retained as an audit trail.

### Historical headline

| Agent | TechnicalScore | Held-out (sessions 101–200) |
|---|---|---|
| Provided `weak_bm25` starter | 0.10671 | — |
| Popularity + category only, zero NLP | 0.7912 | 0.7925 |
| **`agent_v2.py` — honest reference agent** | **0.89431** | **0.8852** |
| `agent_v1_probe.py` — exploit-maximising probe | 0.88298 | — |

The honest agent **beats** the exploit version. Use `agent_v2.py`. `agent_v1_probe.py` exists only
to produce the audit numbers in section 7 of the brief — do not build on it.

## Reproduce

```bash
git clone https://github.com/TechJam2026/techjam-conversational-search.git kit
cd kit
curl -L -o catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
gzip -dc catalog.jsonl.gz > data/catalog.jsonl

python3 -m evaluator.local_evaluator                     # 0.10671  provided starter
cp ../agent_v2.py starter/agent.py
python3 -m evaluator.local_evaluator                     # 0.89431  honest v2

python3 ../analyze_priors.py                             # -> priors.json   (popularity + profile)
python3 ../analyze_kit.py                                # -> findings.json (catalog + constraints)

# ablations need the 100/100 split
head -100 data/public_set.jsonl > /tmp/dev.jsonl
tail -100 data/public_set.jsonl > /tmp/holdout.jsonl
python3 ../run_ablations_v2.py                           # -> ablations_v2.json / .csv
```

## Policy knobs on `agent_v2.py`

All are environment variables so the ablation script can sweep them without editing code.

| Variable | Default | Meaning |
|---|---|---|
| `W_POP` | `1.00` | weight on the popularity prior — the single biggest legitimate signal |
| `W_TXT` | `0.60` | weight on BM25 over accumulated dialogue |
| `W_PHRASE` | `1.00` | weight on long-phrase containment |
| `W_TAG` | `0.00` | profile `preference_tags` — measured, found harmful, left off |
| `W_STAR` | `0.00` | profile `average_prior_rating` — same |
| `P_ASK` | `infogain` | elicitation policy: `infogain`, `fixed:<attr>`, or `none` |
| `P_PROBE` | `1` | items committed per turn before widening |
| `P_WIDEN` | `8` | turn at which the list widens to the full 10 |
| `P_PRUNE` | `reset` | rejection model: `hard`, `soft`, or `reset` (soft + clear on self-correction) |
| `P_SOFT` | `0.55` | score penalty applied to already-shown items |

## What is deliberately NOT in `agent_v2.py`

Four harness behaviours that would raise the score but do not correspond to anything real, all
documented in section 7 of the brief and all excluded from the implementation:

1. `ask_attribute="other"` as a disclosure wildcard
2. staying silent during the intent-override dead zone
3. regexing the organizer's exact message templates
4. hard-filtering on 170-character verbatim feature strings

`agent_v1_probe.py` uses all four and still scores lower. That comparison is the point.

## Caveats

- `agent_v2.py` is a **reference implementation**, not a finished submission. It has no dense
  retrieval layer and no paraphrase-robustness testing — both are on the Sunday plan in section 10.
- Weights were tuned on sessions 1–100 only. Do not re-tune on all 200 and report that number.
- The popularity prior is a real property of 5-core sampling, but leaning on it is the classic
  long-tail pathology. Report the `W_POP=0` score (0.8202) alongside it.
