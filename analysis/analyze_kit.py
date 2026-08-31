"""Characterise the TechJam Track 4 evaluator + catalog. Writes findings.json."""
import json, collections, statistics, sys
sys.path.insert(0, '.')
from evaluator.local_evaluator import (coarse_category, intent_card, catalog_index,
                                       classify_constraint, searchable_text)

ids, cats, prods = catalog_index('data/catalog.jsonl')
samples = [json.loads(l) for l in open('data/public_set.jsonl')]
buckets = collections.defaultdict(list)
for pa, c in cats.items():
    buckets[coarse_category(c)].append(pa)
corpus = {pa: " ".join(searchable_text(p).split()).lower() for pa, p in prods.items()}

F = {}
F['catalog'] = {'products': len(ids), 'distinct_coarse_category_buckets': len(buckets)}
sizes = sorted((len(v) for v in buckets.values()), reverse=True)
F['catalog']['bucket_size'] = {'max': sizes[0], 'median': statistics.median(sizes),
                               'mean': round(statistics.fmean(sizes), 1)}
F['catalog']['largest_buckets'] = [[k, len(v)] for k, v in
                                   sorted(buckets.items(), key=lambda x: -len(x[1]))[:10]]

F['public_set'] = {'sessions': len(samples),
                   'scenario_mix': dict(collections.Counter(s['scenario_type'] for s in samples)),
                   'difficulty_mix': dict(collections.Counter(s['difficulty_bucket'] for s in samples))}

bs = sorted(len(buckets[coarse_category(cats[s['ground_truth']['parent_asin']])]) for s in samples)
F['turn1_category_filter'] = {
    'note': 'size of the exact coarse_category bucket that contains the target',
    'min': bs[0], 'p25': bs[len(bs)//4], 'median': bs[len(bs)//2],
    'p75': bs[3*len(bs)//4], 'p90': bs[int(.9*len(bs))], 'max': bs[-1],
    'mean': round(statistics.fmean(bs)),
    'frac_le_10': round(sum(x <= 10 for x in bs)/len(bs), 3),
    'frac_le_50': round(sum(x <= 50 for x in bs)/len(bs), 3),
    'frac_le_200': round(sum(x <= 200 for x in bs)/len(bs), 3)}

labels, ncon, surv_cat, surv_all, tgt_ok = collections.Counter(), [], [], [], 0
for s in samples:
    t = s['ground_truth']['parent_asin']
    card = intent_card(prods[t])
    cons_raw = list(dict.fromkeys(card['hard_constraints'] + card['soft_preferences']))
    ncon.append(len(cons_raw))
    for v in cons_raw: labels[classify_constraint(v)] += 1
    cons = [c.lower() for c in cons_raw]
    sc = [pa for pa in buckets[coarse_category(cats[t])] if all(c in corpus[pa] for c in cons)]
    surv_cat.append(len(sc)); tgt_ok += (t in sc)
    surv_all.append(sum(1 for pa in ids if all(c in corpus[pa] for c in cons)))

sc, sa = sorted(surv_cat), sorted(surv_all)
F['constraints'] = {
    'per_session': dict(collections.Counter(ncon)),
    'attribute_label_distribution': dict(labels.most_common()),
    'exact_match_within_category_bucket': {
        'target_survives_strict_AND': f"{tgt_ok}/{len(samples)}",
        'sessions_reduced_to_1': round(sum(x == 1 for x in sc)/len(sc), 3),
        'sessions_reduced_to_le_10': round(sum(x <= 10 for x in sc)/len(sc), 3),
        'median_survivors': sc[len(sc)//2], 'p90': sc[int(.9*len(sc))], 'max': sc[-1]},
    'exact_match_over_full_catalog': {
        'sessions_reduced_to_1': round(sum(x == 1 for x in sa)/len(sa), 3),
        'sessions_reduced_to_le_10': round(sum(x <= 10 for x in sa)/len(sa), 3),
        'median_survivors': sa[len(sa)//2], 'max': sa[-1]}}

F['scoring_arithmetic'] = {
    'formula': '0.50*HitRate@10 + 0.30*MRR + 0.20*clip((11-MTTC)/10, 0, 1)',
    'cost_of_one_extra_turn': 0.02,
    'gain_rank5_to_rank1': round(0.30 * (1.0 - 0.2), 4),
    'turns_of_delay_worth_paying_for_rank1_over_rank5': round(0.30*0.8/0.02, 1)}

json.dump(F, open('/tmp/export/findings.json', 'w'), indent=2)
print(json.dumps(F, indent=2))
