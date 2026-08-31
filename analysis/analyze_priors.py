"""Measure the legitimate, generalizable signals in the Track 4 data. Writes priors.json."""
import json, statistics, collections, sys, math, random
sys.path.insert(0, '.')
from evaluator.local_evaluator import catalog_index, coarse_category, searchable_text

ids, cats, prods = catalog_index('data/catalog.jsonl')
samples = [json.loads(l) for l in open('data/public_set.jsonl')]
tg = [s['ground_truth']['parent_asin'] for s in samples]
buckets = collections.defaultdict(list)
for pa, c in cats.items(): buckets[coarse_category(c)].append(pa)
corpus = {pa: " ".join(searchable_text(p).split()).lower() for pa, p in prods.items()}

def q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(int(p * len(v)), len(v) - 1)] if v else None

F = {}

# --- popularity prior -------------------------------------------------------
cat_rn = [p.get('rating_number') for p in prods.values()]
tgt_rn = [prods[t].get('rating_number') for t in tg]
pct = []
for t in tg:
    b = buckets[coarse_category(cats[t])]
    v = prods[t].get('rating_number') or 0
    vals = [prods[x].get('rating_number') or 0 for x in b]
    pct.append(sum(1 for x in vals if x < v) / max(len(vals) - 1, 1))
F['popularity_prior'] = {
    'why_it_is_legitimate': 'sessions are sampled from the Amazon Clothing 5-core leave-last-out '
                            'split, so every target is a real purchase; popularity bias in '
                            'interaction data is a documented property, not a harness artifact',
    'rating_number_median': {'catalog': q(cat_rn, .5), 'target': q(tgt_rn, .5)},
    'rating_number_p25_p75': {'catalog': [q(cat_rn, .25), q(cat_rn, .75)],
                              'target':  [q(tgt_rn, .25), q(tgt_rn, .75)]},
    'average_rating_median': {'catalog': q([p.get('average_rating') for p in prods.values()], .5),
                              'target':  q([prods[t].get('average_rating') for t in tg], .5)},
    'target_popularity_percentile_within_own_bucket': {
        'mean': round(statistics.fmean(pct), 3), 'median': round(sorted(pct)[len(pct)//2], 3),
        'frac_top_quartile': round(sum(1 for x in pct if x >= .75)/len(pct), 3),
        'frac_bottom_quartile': round(sum(1 for x in pct if x <= .25)/len(pct), 3)}}

# --- popularity-only retrieval ---------------------------------------------
ranks = []
for t in tg:
    b = sorted(buckets[coarse_category(cats[t])], key=lambda x: -(prods[x].get('rating_number') or 0))
    ranks.append(b.index(t) + 1)
F['popularity_only_retrieval'] = {
    'method': 'take the category named in turn 1, sort that bucket by review count, stop. no NLP.',
    **{f'target_in_top_{k}': round(sum(1 for x in ranks if x <= k)/len(ranks), 3) for k in (1,3,5,10,20,50)},
    'median_rank': sorted(ranks)[len(ranks)//2], 'mean_rank': round(statistics.fmean(ranks), 1)}

# --- profile signal ---------------------------------------------------------
xs = [(s['user_profile'].get('average_prior_rating'),
       prods[s['ground_truth']['parent_asin']].get('average_rating')) for s in samples]
xs = [(a, b) for a, b in xs if a is not None and b is not None]
mx, my = statistics.fmean(a for a,_ in xs), statistics.fmean(b for _,b in xs)
sx, sy = statistics.pstdev([a for a,_ in xs]), statistics.pstdev([b for _,b in xs])
corr = statistics.fmean((a-mx)*(b-my) for a,b in xs) / (sx*sy) if sx*sy else 0.0

rnd = random.Random(0); pool = list(ids)
ht = hr = hb = tot = 0
for s in samples:
    t = s['ground_truth']['parent_asin']
    tgt, ctl = corpus[t], corpus[rnd.choice(pool)]
    ctb = corpus[rnd.choice(buckets[coarse_category(cats[t])])]
    for tag in s['user_profile'].get('preference_tags', []):
        tot += 1; ht += tag.lower() in tgt; hr += tag.lower() in ctl; hb += tag.lower() in ctb
F['profile_signal'] = {
    'corr_prior_rating_vs_target_rating': round(corr, 3),
    'preference_tag_appears_in': {'target': round(ht/tot, 3), 'random_same_bucket': round(hb/tot, 3),
                                  'random_any_product': round(hr/tot, 3)},
    'lift_vs_same_bucket': round((ht/tot)/(hb/tot), 3),
    'verdict': 'real but weak; wiring it into the score measurably HURT (-0.005). reported as a '
               'negative result rather than used.'}

# --- difficulty_bucket ------------------------------------------------------
F['difficulty_bucket'] = {d: dict(collections.Counter(
    s['scenario_type'] for s in samples if s['difficulty_bucket'] == d)) for d in ('easy','medium','hard')}
F['difficulty_bucket']['note'] = ('difficulty is a relabelling of scenario_type, not an independent '
                                  'property of the product: easy=buying, medium=browsing+boundary, hard=override')

json.dump(F, open('/tmp/export/priors.json', 'w'), indent=2)
print(json.dumps(F, indent=2))
