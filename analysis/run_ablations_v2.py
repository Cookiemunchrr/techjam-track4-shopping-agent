"""Ablate the honest agent (agent_v2.py). Writes ablations_v2.json + .csv.
Run from inside the participant kit with agent_v2.py copied to starter/agent.py."""
import json, os, subprocess, sys, csv, time

DEV, HOLD, FULL = "/tmp/dev.jsonl", "/tmp/holdout.jsonl", "data/public_set.jsonl"
CONFIGS = [
 ("full",            "Honest v2 — everything on",                              {}),
 ("no_pop",          "No popularity prior",                                    {"W_POP":"0"}),
 ("pop_only",        "Popularity + category only (no NLP at all)",             {"W_TXT":"0","W_PHRASE":"0"}),
 ("no_bm25",         "No BM25 lexical signal",                                 {"W_TXT":"0"}),
 ("no_phrase",       "No phrase containment",                                  {"W_PHRASE":"0"}),
 ("with_profile",    "Profile signals re-enabled (tags + rating level)",       {"W_TAG":"0.35","W_STAR":"0.15"}),
 ("ask_none",        "No elicitation at all",                                  {"P_ASK":"none"}),
 ("ask_feature",     "Fixed question: always ask 'feature'",                   {"P_ASK":"fixed:feature"}),
 ("ask_material",    "Fixed question: always ask 'material'",                  {"P_ASK":"fixed:material"}),
 ("commit10",        "Show 10 every turn instead of committing narrow",        {"P_PROBE":"10"}),
 ("commit2",         "Commit width 2",                                         {"P_PROBE":"2"}),
 ("prune_hard",      "Hard-delete shown items instead of soft penalty",        {"P_PRUNE":"hard"}),
 ("prune_soft",      "Soft penalty, no correction-triggered reset",            {"P_PRUNE":"soft"}),
]

rows = []
for key, label, env in CONFIGS:
    r = {"key": key, "label": label, "env": env}
    for split, path in (("dev", DEV), ("holdout", HOLD), ("full", FULL)):
        if not os.path.exists(path): continue
        e = dict(os.environ); e.update(env)
        subprocess.run([sys.executable, "-m", "evaluator.local_evaluator",
                        "--dataset", path, "--output", "/tmp/_a2.json"],
                       env=e, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = json.load(open("/tmp/_a2.json"))
        r[split] = {"technical_score": d["recommended_technical_score"],
                    "hit_rate_at_10": d["hit_rate_at_10"], "mrr": d["mrr"], "mttc": d["mttc"]}
        if split == "full": r["scenario_metrics"] = d["scenario_metrics"]
    print("%-14s %-52s full=%.4f  holdout=%.4f" % (key, label,
          r["full"]["technical_score"], r.get("holdout", {}).get("technical_score", float("nan"))))
    rows.append(r)

rows.append({"key":"baseline_bm25","label":"Provided weak BM25 starter","env":{},
             "full":{"technical_score":0.10671,"hit_rate_at_10":0.125,"mrr":0.068034,"mttc":9.81}})

json.dump(rows, open("/tmp/export/ablations_v2.json","w"), indent=2)
with open("/tmp/export/ablations_v2.csv","w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["key","label","full_score","full_hr","full_mrr","full_mttc","dev_score","holdout_score"])
    for r in rows:
        w.writerow([r["key"], r["label"], r["full"]["technical_score"], r["full"]["hit_rate_at_10"],
                    r["full"]["mrr"], r["full"]["mttc"],
                    r.get("dev",{}).get("technical_score",""), r.get("holdout",{}).get("technical_score","")])
print("\nwrote ablations_v2.json / .csv")
