# Submission bundle

The organizer's recommended layout asks for `agent.py`, `requirements.txt`,
`README.md` and `src/` under a `submission/` directory. This repository *is* that
bundle, laid out at the top level so the official harness command works unchanged:

| Recommended | Here |
|---|---|
| `submission/agent.py` | `starter/agent.py` — the entry point the harness imports |
| `submission/src/` | `src/` — the implementation |
| `submission/requirements.txt` | `requirements.txt` — empty by design, stdlib only |
| `submission/README.md` | `README.md` — overview, setup, reproduction, limitations |

**This is not a preference, it is what the official evaluator requires.**
`evaluator/local_evaluator.py` line 12 reads:

```python
from starter.agent import Agent
```

and line 306 constructs it as `Agent(args.catalog)`. The import path is hard-coded in
the organizer's own scorer. Moving the implementation under `submission/` would break
`python3 -m evaluator.local_evaluator`, which is the one command the rules require to
work. `docs/submission_rules.md` calls its tree a **"Recommended File Layout"**, and
`docs/final_evaluation_faq.md` §7 confirms teams "may replace the starter Agent as long
as the required `Agent` interface is preserved" and asks for "a clear command for running
the Agent with the official evaluator" — which is exactly what the layout below
preserves.

`starter/agent.py` re-exports `src.agent.Agent`, and `Agent.__init__` takes an optional
`catalog_path`, so it satisfies both the evaluator's `Agent(args.catalog)` call and
zero-argument construction. Therefore:

```bash
python3 -m evaluator.local_evaluator
```

runs the submitted agent with no path edits, exactly as the participant kit ships.

## Disclosure

- **Network access:** not required or used by the default path. The optional LLM
  reranker is off by default and falls back completely to the local linear order on
  any network, API, timeout, or parsing failure.
- **Model / API:** deterministic local linear reranker; no runtime LLM or external API.
  Reported token usage is 0 prompt, 0 completion across all 200 public sessions —
  `results.json` carries `total_tokens: 0`. `results.json` is the evaluator's own
  output and is not committed — regenerate it with `python3 -m evaluator.local_evaluator`;
  the committed copies are `analysis/final_numbers.json` and
  `analysis/operational_audit.json`. An LLM
  was used at development time to write and review code and to author test material;
  everything it produced is a committed asset or a measurement, and none of it is
  called during scoring.
- **Estimated model cost for scoring:** $0.00.
- **Latency / memory (current HEAD, 2026-08-31):** quiet arm64 macOS host, Python 3.9.6,
  299 serial turns per route in a fresh process. Exact route p99 5.178 ms, max 5.460 ms.
  Hedged route p99 33.566 ms, max 34.647 ms. 0/299 observations at or above the 50 ms
  engineering canary on either route. Peak RSS 781.2 / 788.9 MB. Agent construction
  17.069-17.678 s, one-off, excluded from turn latency. Artifact:
  `analysis/resource_probe_v6_baseline.json`, whose recorded source hashes match this
  tree exactly. A run taken on a loaded machine produced one 56.8 ms hedged observation:
  50 ms is a canary, not an SLA.
- **Resource profile (V4-0, historical protocol):** on the recorded arm64 macOS host, separate fresh-process V4-0
  fixed-message canaries measured exact p99 3.788 ms and hedged p99 29.291 ms, each
  with 0/299 observations at or above 50 ms; peak RSS was 765.5 and 768.2 MB.
  Full-catalog fallback p99 was 52.355 ms and is an absolute descriptive baseline,
  not an SLA/comparative pass. See `analysis/resource_probe_v4_0.json`. Older
  14.7 s/773 MB/41 ms measurements used a different protocol and are historical.
- **Python:** 3.9+. Verified on 3.9.6.
- **Catalog:** read only. Never mutated, never extended, no synthetic identifiers.

## Publication artifacts

- `DEVPOST.md` — ready-to-publish submission copy, pending only the public video URL.
- `VIDEO_SCRIPT.md` — three-minute evidence-backed storyboard.
- `DEMO_TRANSCRIPT.md` — two recorded live sessions: the clarification and
  product-switch showcase, and an unmodified evaluator `intent_override` session
  (`public_0002`) that converts at turn 3, rank 2.

The owner still needs to make the repository public, publish the Devpost entry and
YouTube video, and replace the remaining video URL placeholder. The repository URL and
team attribution are already filled.
