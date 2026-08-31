# Files referenced here but not published

Three artifacts named in `analysis/README.md` and in `TECHJAM-TRACK4-BRIEF.md`
are deliberately absent from the public submission tree:

| File | Why it is not published |
|---|---|
| `agent_v1_probe.py` | An exploit-maximising probe agent built only to measure the four evaluator findings in the brief's section 7. Its *measurements* are published; a runnable harness exploit is not. |
| `ablations_v1.json` / `.csv`, `run_ablations_v1.py` | The ablation set for that probe agent. Same reason. |
| `agent_v2.py` | The 211-line honest reference agent from the initial analysis. Superseded by `src/`, and retained privately only as a historical baseline. Its scores are in the brief. |

Every number these files produced is reported in `TECHJAM-TRACK4-BRIEF.md` and
reproduces from the committed evaluator against the frozen catalog. Nothing in
`src/` depends on them.
