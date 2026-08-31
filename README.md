# Conversational Shopping Agent — TikTok TechJam 2026, Track 4

Finds a shopper's hidden target product inside ten turns, over a frozen 50,000-item
Amazon clothing catalog.

**No LLM. No embeddings. No GPU. No network. Zero tokens. Standard library only.**

| | TechnicalScore | HR@10 | MRR | MTTC |
|---|---|---|---|---|
| Provided `weak_bm25` starter | 0.10671 | 0.125 | 0.068 | 9.81 |
| **This agent** | **0.91612** | **1.000** | **0.773** | **1.79** |

Tuned on public sessions 1–100, which score **0.92340**; held-out sessions 101–200
score **0.90555**. A gap of 0.018, so the number is not an artefact of fitting.

Behavior A/B deltas below use a **paired** bootstrap over sessions when both systems see
the same conversations (`python3 -m tools.bench --against <earlier.json>`). A single
run's marginal interval is roughly ±0.017 on two hundred sessions; pairing cancels the
shared spread. Resource `delta_p99_ms` values, when present, are descriptive point
differences unless a margin and independent-block protocol is explicitly preregistered.
A paired behavior interval that straddles zero has not shown a change and is labelled
that way.

## Where to look

This README is long because it is the evidence record as well as the overview. If you
are reading it to judge the submission rather than to work on it, these six sections
are the ones that answer the questions you are likely to have:

| If you want | Read |
|---|---|
| Run it in one command | [Quick start](#quick-start) |
| What it is and how it works | [Architecture](#architecture) |
| Whether the labels match the code | [Compliance scope](#compliance-scope-what-we-call-things-and-what-they-actually-are) |
| What it does not do | [Limitations](#limitations) · [What we would improve with more time](#what-we-would-improve-with-more-time) |
| Whether the run is reproducible | [Cross-session state](#cross-session-state-and-what-it-costs) |
| Latency, memory, tokens, cost | [Disclosure](#disclosure) |
| The demonstrated sessions | [`submission/DEMO_TRANSCRIPT.md`](submission/DEMO_TRANSCRIPT.md) |

Everything between Architecture and Limitations is the experiment record: what we
built, what we measured, and what we declined to ship because it did not clear its
gate. It is there so no claim above rests on an assertion.

## The number is not the headline

An earlier version of this agent scored **0.95876**. This one scores 0.91612, and the
difference is the point.

The brief asks for four things. That earlier version implemented one and a half of
them: `detect_intent` returned buying/browsing and the value was consumed in exactly
one place — choosing between two question openers — while both tracks retrieved from
an identical pool, and nothing at all addressed self-evolution. Building the rest cost
**0.022**, itemised and measured:

| Behaviour the brief asks for | Cost |
|---|---|
| Correct slot erasure on intent override | −0.013 |
| Never re-offering an item the customer just rejected | −0.012 |
| Not asking about an attribute already settled | −0.003 |
| Runtime re-orchestration on a stalled session | −0.003 |
| Negation handling (`"not polyester"`) | −0.000 |
| Over-generality retrieval cutoff | +0.002 |
| Profile tie-break | +0.001 |
| Recommendation width driven by confidence, not by first-sight termination | −0.048 |

We report the delta rather than reverting it. Judging weights Innovation and Impact at
20% each and both are written to discount work that only holds on the hackathon prompt;
the leaderboard number is not itself a listed criterion.

Erasure is a dial rather than a switch, and the shape of the trade-off is worth seeing:

| Weight retained by a retracted constraint | 0.00 | **0.08** | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| TechnicalScore | 0.9361 | **0.9370** | 0.9405 | 0.9485 | 0.9487 |
| Intent-override MRR | 0.763 | **0.774** | 0.826 | 0.922 | 0.922 |

Not erasing scores better only because the organizer's `behavior_for` derives the old
and the new value from the same target product, so on this harness the retracted
preference still describes the answer. We ship 0.08, a decayed trace rather than a
deletion. Genuine erasure is no longer hypothetical: `P_SUPERSEDE=erase` removes the
replaced slot from the state entirely, and on the *current* agent it costs 0.00275 of
official score and one intent-override hit (HR@10 1.000 → 0.995), with every adversarial
axis within ±0.003 — so literal erasure exists as a tested, documented mode and stays
off by default on the measurement (`analysis/v8_w3_erasure.json`). The table above was
measured on an earlier agent revision and is kept as the shape of the trade-off; the
shipped-agent numbers are in the artifact.

## What the data actually looks like

Sessions come from the Amazon Clothing **5-core leave-last-out** split, so every target
is a real purchase, and real purchases are overwhelmingly popular items.

| `rating_number` | catalog median | target median |
|---|---|---|
| | 12 | **7,078** |

The median target sits at the **99.3rd percentile** of its own category bucket by review
count. Only **4 of 200** fall below their bucket's median.

But popularity is not the channel doing the work, and it is worth being precise about
this because it is easy to get backwards:

| | TechnicalScore |
|---|---|
| Global popularity alone, no category | 0.02728 |
| Static keyword matching (the provided starter) | 0.10671 |
| Category alone, arbitrary order within the bucket | 0.23712 |
| **Category + popularity, zero language understanding** | **0.83395** |

Either channel alone is near-useless; together they are 8–30× either one. Popularity is
a query-independent *prior* — PageRank, in the web-search analogy — and BM25 is the
query-dependent match. They multiply.

The uncomfortable consequence: the turn-1 message is literally
`coarse_category(target.categories)`, the target's own taxonomy path emitted verbatim.
So the largest single input is a simulator artefact with no real-world analogue — no
shopper says "Earrings Hoop" — and an agent built only on inverting it has nothing
underneath. Synonymise the head noun and the earlier version fell to **0.641**. Most of
this rebuild is about putting something underneath it.

## Architecture

```
turn N ──> observe        typed slots: attribute, turn, polarity, weight
                          accumulate; supersede on correction or contradiction;
                          decay by recency; strip the correction cue itself
       ──> route          buying   -> the named shelf, high precision
                          browsing -> the shelf plus its vocabulary neighbours
                          resolution fuses two routes with RRF, and hedges
                          across more shelves the less certain it is
       ──> score          popularity prior  log1p(rating_number)      w 1.40
                          BM25              k1 2.5, b 0.75            w 0.40
                          phrase            verbatim containment      w 1.00
                          primary shelf, refusals, profile tie-break
       ──> rerank         linear model over 15 interpretable features (default);
                          an LLM semantic ranking stage exists behind
                          TECHJAM_LLM_RERANK=1 + key, off by default, measured
                          and declined as the default (D11), with a total
                          fallback to the linear order on any failure
       ──> commit         width 1 while confident, 10 near the limit,
                          1 when the pool is over-general, 10 when stalled
       ──> elicit         value(a) = expected_reduction(a) x P(answerable | a)
       ──> adapt          update P(answerable), long-term memory, strategy
```

| Module | Responsibility |
|---|---|
| `src/text.py` | tokenisation, clause splitting, dialogue-act lexicons |
| `src/catalog.py` | catalog index: BM25 stats, buckets, popularity, facets |
| `src/semantic.py` | mined word→shelf vocabulary; the semantic ("dense") route |
| `src/routing.py` | intent detection, dual-track pools, diversification |
| `src/fusion.py` | reciprocal rank fusion |
| `src/state.py` | typed slots, supersession, decay, distillation |
| `src/scoring.py` | the scoring terms, and `explain` |
| `src/elicitation.py` | answerability-weighted question selection |
| `src/answerability.py` | what the agent learns about which questions land |
| `src/orchestrator.py` | stall detection and strategy switching |
| `src/profile.py` | the bounded profile tie-break |
| `src/memory.py` | long-term, per-profile-shape memory |
| `src/rerank.py` | the linear reranker (default ranking stage) |
| `src/llm_rank.py` | the opt-in LLM semantic ranking stage (off by default) |
| `src/policy.py` | commit width, over-generality cutoff, rejection model |
| `src/paths.py` | catalog location, so `Agent()` works from anywhere |
| `src/agent.py` | the harness entry point |

### The "dense" route is catalog-mined, not a vector index

The catalog is already a synonym dictionary. Sellers use the words the taxonomy does
not, and those words identify the shelf without sharing a token with its name:

| Word | Hits | Top shelves |
|---|---|---|
| `footwear` | 2,659 | Shoes Fashion Sneakers / Pumps / Flats |
| `timepiece` | 382 | Watches Wrist Watches |
| `shades` | 180 | Sunglasses & Eyewear Accessories Sunglasses |
| `billfold` | 22 | Card Cases & Money Organizers Wallets |

So `src/semantic.py` mines `P(shelf | word)` from product text, prunes words spread over
more than a quarter of the shelves, and adds a suffix stem plus character trigrams.

We call this the *dense* route because of where it sits in the pipeline — it is the
non-lexical arm that RRF fuses against BM25 — but it is a **sparse, offline association
map with a trigram fallback, not an embedding index and not vector similarity**. Read
*Compliance scope* below before quoting the word "dense" anywhere it could be mistaken
for a learned vector space. We built and measured a real one; it is the GloVe-50 row in
*Limitations*, and it moved nothing.
Built from the catalog rather than a generated word list on purpose: a lexicon written
by a language model and then tested against synonyms written by the same model is a
closed loop, which is the failure mode described under **Robustness** below.

### Elicitation is where self-evolution shows up

The earlier version asked `brand` 130 times across 200 sessions for a yield of exactly
zero, and no catalog-derived criterion can avoid that. Take the honest
information-theoretic objective, expected surviving pool size `n · Σp²`: a
high-cardinality facet like `store` minimises it. Gain ratio, normalised entropy and
expected pool size all agree that `brand` is the best question, and all three are
useless, because every one of them conditions on the answer arriving.

`P(the customer can answer)` is a fact about the person, not the products, so the agent
learns it by asking. It ships with a uniform prior and no pre-trained table — a table
fitted to the public simulator would score better from turn one and would be exactly the
harness fitting this rebuild exists to remove. After 200 sessions:

| Attribute | Asked | Answered | P(answerable) |
|---|---|---|---|
| `feature` | 126 | 115 | 0.906 |
| `material` | 47 | 39 | 0.816 |
| `color` | 27 | 8 | 0.310 |
| `brand` | 45 | 1 | 0.043 |

Question yield went from **97/368 (26%)** to **164/315 (52%)**, and MTTC from 2.88 to 2.65.

## Retrieval or ordering? Instrument before you build

The score is one number and it hides which of two very different failures produced it.
A target that never entered the candidate pool cannot be recovered by any reranker,
weight or extra turn. A target sitting in the pool at rank eleven is a ranking failure.
`tools/recall.py` replays the official loop and separates them, using a diagnostic hook
(`Agent.candidate_pool`, off unless asked for) that records what the scorer actually
ranked over.

It was written to support a plan whose headline item was product-level retrieval
fusion, on the reasoning that paraphrase-driven recall loss was the largest measured
weakness. The first run redirected the plan:

| | pool recall @ turn 1 | HR@10 | lost to recall | lost to ranking |
|---|---|---|---|---|
| official sessions (all 200) | **1.000** | 1.000 | **0.000** | **0.000** |
| `category` axis | 0.885 | 0.895 | 0.090 | 0.015 |
| `scaffold` axis | 1.000 | 0.985 | 0.000 | 0.015 |
| `natural` axis | 1.000 | 1.000 | 0.000 | 0.000 |
| `constraint` axis | 1.000 | 0.980 | 0.000 | 0.020 |

On every official session the target is in the pool from turn one, and reachable in
the top ten within ten turns in every session. **There is no retrieval headroom on the
public harness at all**; every remaining point of the official score is ordering and
turn count. The one axis where retrieval genuinely breaks is category-head-noun
synonym, and there it dominates ranking six to one.

That is what pointed at the reranker below — and it is also why the reranker's first
evaluation went wrong. When the retrieval half of a metric is pinned at 1.000, the
metric stops being sensitive to the half that is not.

That killed the plan's largest item and promoted a smaller one. `scaffold` loses
nothing to recall — its loss was entirely that the shelf name was not being found, so
the primary boost fell back to its unsure strength. The fix was a bug fix in
`exact_bucket`, not a new retrieval route:

`exact_bucket` scanned for the shelf name **anchored at the end** of the phrase, which
encodes the assumption that a shopper stops talking once they have named the thing.
They do not. *"do you carry Underwear Briefs, just seeing what is out there"* put the
shelf in the middle of the sentence, and the anchored scan missed it on 47% of
scaffolded openings. Trailing conversational tail is exactly as ordinary as leading
scaffolding. Scanning every contiguous window — latest-ending first, longest at each
ending — took shelf resolution under `scaffold` from 0.530 to **1.000**.

Both halves of that ordering are load-bearing, and getting one backwards costs real
score. Preferring the longest window outright reads *"shoes & jewelry women dresses"*
as the **Shoes & Jewelry Women** shelf rather than **Women Dresses**, because the
taxonomy's leaf noun comes last — that cost 0.084 under granularity drift, and is now
pinned by a test.

### The global lexical route: declined, then reopened on its own terms

The global product-level route was **declined** (`analysis/declined_experiments.json`,
D7): the official sessions have no recall failure for it to fix, and a global BM25 pass
costs 205–288 ms against a published 50 ms per-turn budget. Widening the existing routes
instead (D6) was free on control exactly as predicted, bought 0.885 → 0.905 recall on
the category axis, and still lost 0.017 on the compound axis, because the candidate the
pool gained was not one the ranking could find.

The decline named its own reopening condition — *"revisit only with an inverted index"* —
and both halves of the objection have since changed. `Catalog.index_postings` builds a
term → products index once at construction (**+0.3 s of build, +30 MB of RSS**), so a
query touches only the products containing a query term instead of all fifty thousand.
And D6's sentence about the ranking not finding what the pool gained was written before
the learned reranker existed. `analysis/global_route.json` is the re-test.

It fires **only when the shelf did not resolve outright**, which is the only case where
the pool can be wrong about which shelf was meant — and it *appends*, never merges, so
nothing already in the pool moves or is lost. Recall-only widening, pinned by a test.

| | official (all 3 splits) | category axis | compound axis | every other axis |
|---|---|---|---|---|
| paired Δ | **+0.00000** (zero width) | **+0.04298 [+0.01911, +0.07209]** | +0.02297 (no regression) | **+0.00000** |

The control column is exactly zero because the route is *unreachable* on the public
set — the opening message is the shelf name verbatim, so it always resolves — and a
test asserts pool equality on a real official session rather than trusting that.
Sessions lost to retrieval on the category axis fall from **9.0% to 3.5%**;
rank-given-pool holds at 0.990, so the candidates the pool gains do convert. That is
D6's objection answered rather than argued with. `granularity=1` gains +0.02297,
significant, for the same reason: simulator drift breaks shelf resolution the way a
synonym does.

The one free parameter, how many postings a turn may walk, was set from **latency, not
score**: across 4k/8k/12k/20k the axis scores 0.8377/0.8363/0.8361/0.8430 — a spread of
0.007 on 200 sessions, which is noise — while the 99th-percentile turn runs
41/45/45/**52** ms. The largest budget is the only one that breaks the 50 ms budget and
it buys a difference that cannot be measured, so 4,000 it is.

**Zero public-set value by construction, and that is the point.** The private set is 800
sessions of unknown paraphrase level, and the category axis is where this agent is most
exposed. Insurance on the one measured retrieval break, for +0.3 s and +30 MB.

One thing found while measuring this and worth disclosing rather than burying: on the
category axis, one turn in 548 takes **~347 ms — with the route on or off**. It is the
fallback pool. When nothing resolves at all, `route_detail` returns the whole catalog
and the scorer ranks fifty thousand products. Not caused by this change, not fixed by
it, and the one place the agent can exceed its per-turn budget. This is a historical
adversarial-category workload, not the V4-0 fixed-message fallback baseline; the two
latencies are not compared as a regression.

## Robustness

The specification used to warn that natural-language paraphrasing might be added, and
this section was built against that risk. **The 2026-08-31 FAQ closes it**: final
evaluation "messages follow the templates and deterministic response policy in the
released official evaluator" and "no undisclosed natural-language paraphrases are
introduced". Everything below therefore tests *beyond* what the final evaluation will
do. We are keeping it — a shopping agent that only works on templated input is not a
shopping agent — but it should be read as robustness we were not required to have, not
as risk mitigation. The kit's own `tools/paraphrase.py` is not a sufficient test of that, for two
reasons found by reading it: it re-inserts the category phrase **verbatim** at every
level including L3, so its most aggressive setting never varies the most load-bearing
input; and its scaffolding vocabulary is the same list of phrases as `FILLER_PREFIX` in
`src/text.py`, so it only generates paraphrases the parser was written to strip.

`tools/adversarial.py` varies five axes independently, and every word in its vocabulary
is deliberately absent from both the parser lexicon and the mined one.

| Axis | Baseline | State fidelity | + reranker | + global route | + parser | |
|---|---|---|---|---|---|---|
| control | 0.90089 | 0.90624 | 0.91612 | 0.91612 | **0.91612** | |
| `natural` | 0.86157 | 0.88635 | 0.91144 | 0.91144 | **0.91154** | taxonomy path rewritten as speech |
| `constraint` | 0.85085 | 0.86485 | 0.87330 | 0.87330 | **0.88575** | quoted text synonymised, not truncated |
| `scaffold` | 0.82337 | 0.80880 | 0.83431 | 0.83431 | **0.86491** | openings outside `FILLER_PREFIX` |
| `category` | 0.74768 | 0.78745 | 0.79350 | 0.83774 | **0.84141** | head noun replaced by a synonym |
| all four | 0.63029 | 0.63759 | 0.66953 | 0.69330 | **0.72166** | |
| `granularity=3` | 0.89892 | 0.90589 | 0.91548 | 0.91548 | **0.91548** | simulator drift, not a paraphrase |
| `granularity=1` | 0.85199 | 0.86681 | 0.86669 | 0.89039 | **0.89039** | |

Every axis is above its baseline, and the compound axis — four paraphrases at once, the
closest thing here to a private-set stress test — gains **0.091** end to end. The last
column moves only where shelf resolution breaks, because that is the only case the
global route can reach; everywhere else it is bit-identical, control included.

The middle column is worth one note. `scaffold` fell 0.015 at the state-fidelity step while its **artifact-free** score rose
(0.84982 → 0.85278) with hit rate up and MTTC down 1.84 → 1.69. Resolving the shelf
correctly finds the target *sooner*, and a target found sooner is exposed on a turn
when fewer constraints have been disclosed and the slate is wider — so official MRR
falls while retrieval quality improves. This is the same width artifact the shadow
harness exists to catch, arriving from the other direction. Steer by the shadow score;
report the official one.

For scale, the earlier version scored **0.641** on the `category` axis alone.

**Rule update, 2026-08-31.** The organizer's final evaluation FAQ
(`docs/final_evaluation_faq.md`) supersedes the earlier "organizer policy may disable
network access": teams run the final evaluation in their own environments, so *network
access and external API calls are allowed* and *an offline fallback is not mandatory*.
This agent's default path still has no network path at all — that is now a deliberate
engineering choice with no rule behind it, and the reasons it is still the right one are
determinism, zero credentials, $0 runtime cost and no vendor availability risk, not
compliance. The optional LLM stage (`src/llm_rank.py`) is the network path we do have,
and the FAQ makes it explicitly permitted.

## Testing

The full suite is **541 tests**, run with `bash run_tests.sh all`. From a bare clone it
is green with four skips, all of them naming the setup step they want; after
`python3 -m tools.setup_check --splits` rebuilds the two gitignored split files it is
green with **none**. Nothing errors in either state — a test that needs an artifact a
clean checkout legitimately lacks says so and skips. It is organised by the pillar each
test defends:

- **A · contract** — schema, attribute enum, dedupe, top-k cap, ordering, session
  isolation, determinism across processes, and `Agent()` constructing from an arbitrary
  working directory
- **B · Pillar I** — buying and browsing must produce *different* pools; browsing must
  span shelves; synonyms and hypernyms must route; the exact-match branch must be
  reachable *through `category_key`*
- **C · Pillar II** — erasure, category re-routing, the correction cue never becoming
  evidence, no re-offering a rejected item, the over-generality cutoff, question yield
- **D · Pillar III** — distillation fidelity and boundedness, a profile prior that
  cannot outrank a stated constraint, memory that survives `reset` and never leaks
  across profiles, re-orchestration that is a no-op on healthy sessions
- **E · metrics** — the exception guard counts rather than hides; MRR floored at *fixed
  width 10* so ranking quality is separable from the commit policy; recall floored
  before ranking; the harness must be seed-stable
- **F · adversarial** — the five axes above, plus negation, contradiction without a cue,
  and the long-tail slice
- **G · operational** — build time, per-turn latency, peak RSS in its own process,
  bounded session store, declared Python version

## Reading the sentence, not the template

The official turn-1 message is `coarse_category(target.categories)` **verbatim**,
followed by one constraint in a fixed frame. That single fact hides three defects that
any real shopper triggers immediately, and none of which the leaderboard can see.

**The opening clause was discarded whole.** `observe(skip_first=True)` dropped
`clauses[0]` on the reasoning that it was the category and nothing else — true of this
simulator, false of *"I'm looking for black leather boots"*, where the shopper named a
shelf and two facets in one breath. The shelf span is now located and removed, and
what survives is read for concrete facets only. A clause that *is* only a category
ends up empty, so the official path is unchanged by construction; the gain shows up on
the `natural` axis (+0.025).

**A correction retired everything.** Any correction cue set `replace=True` for every
clause, so *"Black leather boots under $100"* followed by *"Actually, brown"* retired
the leather and the budget along with the black. There are now three scopes: attribute
(the default, and the one that needs no cue because real shoppers rarely give one),
product (`product_reset`, on a genuine change of shelf), and global — reachable only
by an explicit restart, *"forget everything, let's start over"*. The restart phrasing
is itself stripped before storage, because otherwise the instruction becomes search
evidence, which is the defect `src/state.py` opens by describing.

Scoping this narrowly is right on this harness too, not only in principle: the
organizer's `behavior_for` derives the override's old and new value from the **same
target product**, so the constraint the blanket rule was retiring still described the
answer.

**"No more than $30" was parsed as refusing the word "more".** The negation pattern
matched `no` + the comparison, the budget vanished entirely, and every product whose
text contains "more" was penalised for it. Budgets now read three ways —
ceiling, floor (*"at least $50"*, previously unmodelled and treated as a tier) and
tier — with lookbehinds so that "no more than" and "no less than", each of which
contains the other's keyword, cannot both match.

Refusals are also resolved against facets rather than substrings. *"Nothing in
leather"* used to scan for the substring anywhere in the blob, so a cotton canvas bag
whose description said *"pairs well with leather boots"* was penalised exactly as hard
as a leather bag. Three tiers now: the product states it, its prose merely mentions
it, or neither — and "neither" is a penalty the substring scan could not express.

The asymmetry with **positive** facet scoring is deliberate and measured. Multi-valued
facet sets were built for positive scoring and **declined** (D5): matching any of a
product's materials is *less* discriminative than matching its primary one, because a
listing reading "polyester, cotton lining" then answers a cotton request as strongly
as a cotton shirt does, and there are far more of those than there are trims. It cost
0.007 on holdout and got monotonically worse as the credit widened. The sets are kept
and used only by the refusal logic, where wrongly demoting a product over a word in
its marketing copy is the worse error.

| Change | dev | holdout | full | paired 95% interval (full) |
|---|---|---|---|---|
| baseline | 0.90620 | 0.90112 | 0.90089 | |
| **all of the above** | **0.91435** | 0.90052 | **0.90624** | **+0.00526 [+0.00048, +0.01116]** — significant |

Holdout is flat (−0.0006, interval straddles zero), dev and the full set are
significant improvements, and seven of eight adversarial axes rose.

**Disclosed cost.** The dual-track intent is now re-read every turn from accumulated
evidence rather than assigned once at turn 1 — the old rule left a shopper who opened
vaguely and then named three requirements retrieving from the wide `open` pool for the
rest of the session. Cues on the current turn settle the *workflow* ("I need sneakers
but I'm still comparing" is a browsing workflow that still ranks on every constraint),
while pinned constraints accumulate as evidence of how much has been *decided*. It
changes the track mid-session on 15 of 200 official sessions, ten of them
browsing → buying, and it costs **−0.0004** on the full set (paired interval
[−0.0026, +0.0018], within noise on every split) and 0.009 on the compound adversarial
axis. Shipped on the second door: it is what Pillar I asks for, and the cost is stated
rather than hidden.

## Two numbers, not one

`evaluator/local_evaluator.py` is never modified and stays the only source of the
competition score. Beside it, `tools/shadow.py` replays the same sessions with three
simulator artifacts removed — the `other` disclosure wildcard, the intent-override
dead zone, and the interaction between recommendation width and rank.

That third one is the point. The official session ends when the target appears in the
**shown** slate, so narrowing the slate suppresses low-rank exposure and inflates MRR.
The shadow harness instead scores `Agent.internal_ranking`: the untruncated top ten,
recorded before the dialogue policy decides how much of it to show. Width therefore
cannot move the rank. Across widths 1 to 10 the official score swings **0.031** and
the shadow score **0.004**.

|  | Official | Shadow |
|---|---|---|
| this agent | 0.91612 | 0.89854 |
| HR@10 | 1.000 | 1.000 |
| MRR | 0.773 | 0.677 |
| MTTC | 1.79 | 1.23 |

Read that table before optimising anything. Retrieval puts the target in the top ten
by turn **1.32**, in every session.

The two MRR figures are the diagnostic. They once read 0.934 official against 0.672
shadow — a quarter of the headline rank quality was the width policy waiting for rank
one before showing anything, not search. They now read 0.773 against 0.677, and that
gap closed because the policy changed, not because the measurement did. The number
worth improving was always the shadow one.

Shadow HR@10 is 1.000 and shadow MTTC is 1.23: retrieval finds the target, early, in
every session. Combined with the recall decomposition above, that locates all of the
remaining official headroom precisely — **73% of it is MRR**, which is ordering, and
the rest is turn count. Not recall.

Rank is deliberately *not* removed — the brief asks for the target early **and**
highly ranked, and weights MRR at 30%, so the shadow score uses the official
weighting on the internal ranking. Dropping the rank term would answer a question the
challenge is not asking.

The harness also runs two operational axes, because both are ways a private harness
could differ from this one:

| Axis | Shadow | |
|---|---|---|
| clean | 0.89854 | |
| fresh agent per session | 0.89436 | cross-session learning is worth 0.004, not load-bearing |
| shuffled order, 2 seeds | 0.89630 / 0.89654 | ordering is worth ~0.002, and the shipped order is the high end |

Those are the numbers in `analysis/shadow.json`; `python3 -m tools.shadow` reprints them.
The shuffle row is a bound, not a null result, and the same axis on the *official* metric
is stated plainly under [Cross-session state](#cross-session-state-and-what-it-costs)
below: session order is worth about 0.001 there. Small, measured, and disclosed rather
than assumed away.

**What the gap is for.** A change that lifts the official score and leaves the shadow
score flat is buying score from the simulator. Re-adding the `other` wildcard scores
**+0.017 official and −0.005 shadow** — flagged, and declined. The phrase-floor fix in
this repo scores positive on both, which is why it shipped.

It also settled the largest open question in the design. A flat `base_width = 1`
committed a single item on **85% of turns**, and the official evaluator rewards that
because a session ends the moment the target appears in the *shown* slate. Measured
across widths 1 to 10, the official score moves **0.031** and the shadow score
**0.004** — the benefit is almost entirely the artifact. Replacing it with
confidence-driven width costs **0.048 official** and leaves the shadow score flat,
while official MTTC *improves* from 2.56 to 1.93 because the agent stops hiding what
it has already found. The agent no longer answers with a single item on any turn.
We took the trade.

`P_PROBE=1` narrows the whole ladder to 1/2/4/7 and scores **0.90784**, recovering
about a quarter of the cost while still never answering with a single item. Restoring
the flat width of one means reverting the policy, not setting the variable — worth
saying plainly, because the environment override reads like a full undo and is not.

This is a diagnostic, not a second leaderboard, and not a claim about real users. It
runs the same 200 sessions against the same simulated shopper. It answers exactly one
question — how much of the score depends on the three behaviours it removes — and
should not be read as measuring anything else.

## The learned reranker, and the metric that nearly buried it

`tools/recall.py` located all the remaining official headroom in ordering: recall is
1.000 on every session, so what is left is MRR. That is what a reranker is for, so
one was built — pairwise logistic regression over the same evidence the weighted sum
already uses, trained on dev, pure standard library, exported as JSON coefficients
with a dot product at serving time.

**It was measured, declined, and the decline was wrong.** The mistake is kept here
because it is more useful than the result.

### The blunt metric

The first evaluation used the composite shadow score — 0.50·HR + 0.30·MRR +
0.20·efficiency, read off `Agent.internal_ranking`. Every configuration came back
"within noise" and the model was declined.

That metric could not have found the effect. Internal HR@10 is **already 1.000** on
every official session, so half the composite cannot move; the efficiency fifth is
driven by turn dynamics, not ordering. A real +0.03 MRR improvement enters that
composite as +0.009 — indistinguishable from zero at n=200. *Most of the metric was
measuring the parts the change could not affect.*

The fix is to measure the mechanism directly: hold the candidate set and the dialogue
state fixed, and ask only where the target lands under each ordering. No dialogue
policy, no turn dynamics, no commit width. **Snapshot MRR**, paired bootstrap grouped
by session:

| | weighted sum | + model | delta |
|---|---|---|---|
| dev | 0.6650 | 0.6857 | **+0.0205 [+0.0022, +0.0415]** significant |
| holdout | 0.6465 | 0.6820 | **+0.0346 [+0.0137, +0.0600]** significant |

Significant on both, including the held-out half. The model ranks better.

### The artifact is still real, and it is why the blend is small

At blend 1.0 the model's wider score range inflates rank-1-to-rank-2 margins,
`CommitPolicy.width` reads margin as confidence, mean shown width collapses 4.61 →
3.28, and the official score rises 0.025 **without ranking anything better** — the
slate narrows, low-rank exposure is suppressed, MRR inflates. The first analysis was
right about that trap and wrong to conclude the model was nothing but the trap.

The blend is set to **0.05**, where the width distortion is 4.61 → 4.44 rather than
4.61 → 3.28. To measure what survives with the artifact removed entirely, the whole
thing was re-run with the slate pinned at ten items on every turn, so width cannot
vary at all:

| fixed width = 10 | delta at blend 0.05 |
|---|---|
| dev | +0.0052 [−0.0050, +0.0151] |
| holdout | +0.0050 [−0.0063, +0.0163] |
| full | +0.0036 [−0.0039, +0.0110] |

Positive on all three, none individually significant at n=200. So roughly a third of
the official gain is ranking and the rest is width interaction — stated plainly
rather than claimed as all real. `model_only`, with the weighted sum removed
entirely, is not significant on either split: this is a **correction to a good
ordering, not a replacement for it**, which is the other reason the blend is small.

### Shipped

| | before | after |
|---|---|---|
| dev | 0.90620 | **0.92340** |
| holdout | 0.90112 | **0.90555** |
| full | 0.90089 | **0.91612** — paired +0.0098 [+0.0017, +0.0193], significant |
| HR@10 (full) | 0.995 | **1.000** |

Seven of eight adversarial axes improve and the eighth is flat: `natural` +0.025,
`scaffold` +0.026, `constraint` +0.008, `category` +0.006, `granularity=3` +0.010,
`granularity=1` −0.0001, and **all four paraphrase axes together +0.032** — the axis
the previous change had regressed. Per-turn latency is 1.7 ms against a 50 ms budget;
runtime stays stdlib and offline, because the asset is a list of coefficients.

### Would more data help? No — and that is the useful part

Before blaming the thin training set (148 snapshots), the same model was fitted on
all 200 sessions and evaluated on those same 200 — deliberately leaky, an upper bound
rather than a result. On snapshot MRR over the full set: the shipped dev-trained model
gives **+0.0279 [+0.0135, +0.0443]**, and the model fitted on the evaluation sessions
themselves gives **+0.0280 [+0.0121, +0.0458]**.

Identical to three decimals. Fitting on the answers buys nothing over fitting on dev,
so the training set is not the binding constraint and more sessions would not move
it. **The gain is capped by what these fifteen features can express.** The next lever
is features that say something the current ones do not — or a model class that can
use interactions a linear one cannot, which remains untested.

One diagnostic worth recording: the model puts popularity at +8.30 against phrase at
+1.09 — 7.6∶1, where the shipped weights are 1.4∶1 and `test_scoring_policy` pins the
floor at 3∶1 on the grounds that below it the conversation has stopped mattering. At
blend 0.05 it shifts the combined ranking only slightly, but the direction is worth
watching, and it is now watched by a test rather than by this paragraph:
`PopularityPathologyTest` fails if the *combined* ranking — weighted sum plus the
model — ever puts a popular non-match above the long-tail product the shopper
described, at the shipped blend or at four times it.

### Would better features help? Also no — and that closes the second of three levers

The sentence above says the next lever is features. We built the batch and it does not
work either. Six new features, ranked by expected value before any of them was
measured, with the first designed specifically to express something no reweighting of
the fifteen can fake — that a product **disagrees**. A listing that states polyester
scores zero against a shopper who asked for cotton, and so does a listing that states
nothing at all, and those are not the same situation.

Full ablation in `analysis/reranker_features.json`. The control — the fifteen shipped
features, retrained by the ablation tool — reproduces the committed asset exactly, a
zero-width interval on both splits, so every other row is measured against a pipeline
known to be the shipped one. What the table says:

| feature added alone | grouped-CV | dev snapshot MRR | holdout |
|---|---|---|---|
| *(control: the shipped fifteen)* | 0.9223 | +0.0000 | +0.0000 |
| `constraint_coverage` | **0.9254** | +0.0124 [−0.0010, +0.0322] | +0.0027 [−0.0141, +0.0230] |
| `facet_disagreement_material` | 0.9226 | +0.0000 | +0.0000 |
| `facet_disagreement_color` | 0.9223 | +0.0000 | +0.0000 |
| `shelf_lexical_rank` | 0.9188 | +0.0044 | −0.0008 |
| `phrase_in_title` | 0.9192 | −0.0000 | −0.0037 |
| `price_percentile_in_shelf` | 0.9150 | +0.0003 | −0.0004 |

**Not one row is significant.** Only `constraint_coverage` carries anything at all, and
three of the six lower cross-validated accuracy below the control. End to end at blend
0.05 the best of them scores dev −0.0008, holdout +0.0006, full −0.0007, and five of
the eight adversarial axes move the wrong way — worst of all the compound paraphrase
axis at −0.0089, which is the axis that stands in for the private set. Declined in
full; the definitions live in the commit that carried the experiment.

The interesting failure is `facet_disagreement`, because we can say exactly why it
could not be learned. `tools/violations.py` counts how often the agent shows a product
contradicting a stated constraint: **3 of 678 shown items on the clean public set —
0.44% — against 13.0% under constraint paraphrase.** The feature targets a failure the
training distribution barely contains, so a pairwise objective has almost nothing to
fit it on; the weight came out at −0.555, which at blend 0.05 moves a score by 0.028
against a spread near 1.0, and the measured effect on the violation rate was two items
in six hundred. Training on the paraphrase axes would supply the missing examples and
was refused: `tools/adversarial.py` is *our* paraphraser, and a model fitted to it
would be fitted to our guess about the private set rather than to the shopper.

One of the six was re-tested after the decline was challenged, and the challenge was
half right. `constraint_coverage` counted every stated constraint equally — but
`weighted_phrases` hands superseded constraints over at weight 0.08 and decays older
ones, because erasure here is a dial and not a switch. The feature was therefore
counting a constraint the shopper had just *retracted* as fully as the one they had
just stated, on exactly the intent-override sessions where it should have been
sharpest. On a two-clause example the unweighted version scores a leather belt and a
suede belt identically at 0.5 where a weighted one separates them 0.93 to 0.07.

A real defect, and it changes nothing. The weighted feature differs from the unweighted
one on 4.4% of rows but only **13 of 148 snapshots**, and the metric is session-weighted
over a hundred sessions, so the two come back identical to four decimal places. Fitted
together with a third partial-credit variant, the model splits one signal almost evenly
across the three columns — 2.03, 2.02, 2.18 — which is what having one feature three
times looks like. The best of them is official dev −0.0018 and compound axis −0.0100,
so the same bar fails the same way. The first decline was right on evidence that had a
bug in it; this one is right on evidence that does not.

So two of the three levers on ranking quality are now measured and closed — not the
data, not these features. The third is below.

### Would a richer model class help? No — and that closes the third

A linear model cannot say "popularity counts for less once the shopper has named a
material", however its weights are set; only something with interactions can. So:
boosted regression trees, depth 1 and depth 2, under the same pairwise logistic
objective, with the number of rounds chosen by the same grouped cross-validation
inside dev. `analysis/reranker_stumps.json`.

| model | grouped-CV | dev pairwise | holdout pairwise | dev − holdout |
|---|---|---|---|---|
| **linear (shipped)** | **0.9223** | 0.9289 | **0.9032** | **0.026** |
| depth-1 stumps, 460 rounds | 0.8786 | 0.9268 | 0.8810 | 0.046 |
| depth-2 trees, 300 rounds | 0.8921 | **0.9617** | 0.8895 | 0.072 |

Depth 1 is not undertrained — offered 1,500 rounds, cross-validation picked 460 and
the curve plateaued below the linear model. Depth 2 is the overfit in plain sight: it
fits dev *better* than the linear model and holdout *worse*, with a dev-to-holdout gap
nearly three times as wide.

That gap is what the snapshot-MRR sweep reports. Depth 2 at blend 0.2 gains
**+0.0339 [+0.0080, +0.0630]** on dev — significant, and the only significant positive
on the table — against **+0.0042 [−0.0220, +0.0328]** on holdout. Choosing the blend on
dev is the shipped procedure, and it picks exactly that blend and buys nothing. Several
blends come back significantly *negative* on holdout. Declined; the linear model stays,
because it is simpler, its fifteen weights are readable, and explainability is a judged
criterion rather than a preference.

**All three levers are now closed.** Not more data, not better features, not a richer
model class. The honest conclusion is a statement about this evaluation set as much as
about this model: the targets are real 5-core purchases sitting at the 99.3rd
popularity percentile of their own shelves, and a ranker that knows the shelf and the
prior is already close to what the data can distinguish. The remaining official
headroom is not in ordering, whatever `MRR 0.773` looks like from the outside.

### What is a question worth? Less than the turn it costs

The elicitation criterion ranks attributes by how much of the pool an answer removes.
The standing objection — written into the specification a roadmap ago and never
tested — is that pool reduction is the wrong quantity: a question that splits the
*bottom* of the candidate set scores well on it and cannot move the head at all, and
what the score pays for is where the target lands.

The objection is right. `src/elicitation.expected_gain` computes the right quantity:
read a belief over which candidate is the target from the scores already computed,
apply the facet term the scorer *would* apply for each likely answer, re-sort, and
read the change in expected reciprocal rank. The threshold it is compared against is
not fitted to anything — a turn is 0.02 of TechnicalScore and a unit of MRR is 0.30,
so a question has to buy **0.0667** of reciprocal rank to pay for itself, straight off
the published formula.

Measured over every turn of both splits (`analysis/question_value.json`):

| attribute | pool reduction | expected gain (median / max) | P(answered) | pays for the turn |
|---|---|---|---|---|
| `brand` | **0.978** | 0.0027 / 0.0624 | 0.17 | 0% |
| `material` | 0.389 | **0.0057** / 0.0426 | 0.83 | 0% |
| `budget` | 0.185 | 0.0061 / 0.0511 | 0.50 | 0% |
| `color` | 0.272 | 0.0007 / 0.0317 | 0.33 | 0% |

Two things fall out. **Not one question, on any turn of either split, is expected to
buy enough rank to pay for the turn it spends** — the largest gain observed anywhere is
0.0624 against a cost of 0.0667. That is what a harness looks like when HR@10 is
already 1.000 and the median session hits on turn one. And the criterion independently
rediscovers the failure that `src/answerability.py` exists for: pool reduction ranks
`brand` first because store is high-cardinality, the counterfactual ranks it near last
because knowing the brand barely moves the head — the same 130-questions-for-zero-yield
finding, from the other side, with no rule saying "do not ask about brand".

Which is also why it is worth nothing: the learned answerability term already fixes
that, so there is nothing left to repair. Switching the criterion over costs **dev
−0.0032, holdout −0.0029, full −0.0055**, and HR@10 falls off 1.000 to 0.995 — a
different question order is a different conversation, and one session stops reaching
its target at all. Against a customer who *can* answer (`shelfbench --realistic`) it is
flat on hit rate and time-to-contact and slightly worse on MRR. Declined on both doors,
and left reachable under `P_ASK=counterfactual` beside the other ablation controls,
because it costs nothing on any default path and it is how the finding stays re-runnable.

### The weakest scenario slice, and why it is weak

Boundary is the worst scenario in the evaluator output — MRR 0.671 against 0.749 buying
and 0.776 browsing — and nobody had looked at it. `analysis/boundary_diagnosis.py`
replays all two hundred sessions in order (the elicitation model learns *across*
sessions, so replaying the ten boundary ones alone measures a cold-start agent the
official run never has — the first draft of the script did that and produced a much
worse-looking answer) and classifies every turn by which layer lost the target.

**Zero retrieval failures.** Nine turns lost to ordering, six to width, and the two
readings are different in kind:

- The nine ordering turns are two sessions, and both are the same story: a boundary
  session opens with `"I'm looking for X, but I'm still exploring"` and states nothing,
  so the first ranking *is* the popularity prior, and the target is not popular. It
  enters the top ten the turn a real constraint finally arrives — rank 8 the moment
  `leather` lands, rank 1 the moment `color: pink` does. There is no reordering to fix
  here; there is nothing to order on yet.
- The six width turns are the interesting ones, because the agent already had the
  answer and chose to show fewer than ten. That looks like a defect and measures as the
  opposite. Pinning the slate at ten items would have scored this slice **MRR 0.538,
  MTTC 1.9** against the actual **0.671 / 2.5**: waiting costs 0.6 turns and buys 0.133
  of rank. Every one of the three delayed sessions ended at a better rank than it would
  have hit at earlier — rank 9→3, rank 7→1, rank 4→2.

So the commit policy is not what makes boundary weak; it is what keeps boundary from
being weaker, and the slice is weak for a reason the ranking cannot fix. One structural
fact is worth recording alongside it: `customer_reply` refuses the *first* question of
every boundary session by construction, whatever it asks. Five of the ten sessions ran
long enough to spend that turn.

## Evaluation audit — findings we did not build on

Reading `evaluator/local_evaluator.py` turned up five behaviours that raise the score or
suppress it and correspond to nothing real. All are excluded, and the last two are places
where the harness is silent about behaviour a real shopping agent needs.

1. **`ask_attribute="other"` is a disclosure wildcard.** `(attribute == "other" or
   classify_constraint(v) == attribute)` unlocks every undisclosed constraint at once.
   Measured worth on this agent: **+0.016**. We would have declined it at +0.16.
2. **Intent-override sessions have a dead zone.** The hit check is
   `if override_applied and target in ranked`, and the flag only flips at turn 3 or 4,
   so earlier recommendations cannot register. Staying deliberately silent through it
   is worth roughly +0.08 and exploits a scoring bug. The demo session shows the agent
   surfacing the target at turn 2 of an override session and receiving nothing for it.
3. **The specification and the ground truth disagree.** The brief asks for slot erasure;
   `behavior_for` derives both the old and the new value from the same target product,
   so implementing erasure correctly is penalised. We implemented it and reported the
   cost (−0.013) rather than choosing the score.
4. **`brand` is unanswerable by construction.** `classify_constraint` can only return
   `budget`, `color`, `feature`, `material`, `size`, `style`, `use_case`. We learn this
   from evidence rather than hard-coding it, so the same mechanism works on a simulator
   that behaves differently. `category` is unanswerable for the same reason, and `size`
   (0/19), `style` (0/19) and `budget` (0/5) are never answered in practice either.
5. **A budget is never disclosed at all.** `intent_card` appends `budget around $price`
   *last*, then truncates to `cleaned[:4]`; features and details always fill those four
   slots first. Verified: **0 of 200** sessions. The agent's own learned table had
   recorded `budget: asked 5, answered 0` before anyone read it as structural. The term
   is implemented anyway — a shopping agent that ignores a stated budget is not finished
   — and it measures 0.91612 → 0.91612, identical to five decimals. Note the trap it
   sits next to: the simulator sets the budget to the target's *exact* price, so a
   tolerance fitted on generated sessions collapses into an answer-key detector. Ours
   comes from what a budget means to a shopper: proportional (log space), bounded at
   log(4), and always outvotable by a better match.

One trap for anyone hard-filtering: under a strict AND over all four constraints the
target survives in only **159/200** sessions, because `_flatten_values` renders `details`
as `"key: value"` while `searchable_text` renders them `"key value"`. Score softly.

## Asking the question the catalog cannot answer

Shelf identity is worth +0.167 hit rate and 1.2 turns, is not inferable from any amount
of product evidence, and is known to exactly one party in the conversation. `src/clarify.py` asks
them. When retrieval leaves near-duplicate shelves standing, the agent puts a closed
question naming the shelves actually in contention — *"are you after baseball caps, sun
hats, or accessories hats & caps?"* — because "what category?" asks the customer to
recall while this asks them to recognise, and because a named option maps back onto a
shelf without parsing. Their answer sets the category outright and does **not** trigger
a product reset: they answered a question, they did not switch products.

| | HR | MRR | MTTC |
|---|---|---|---|
| no clarification (`--no-clarify`) | 0.629 | 0.389 | 5.81 |
| clarification, a customer who cannot answer (the evaluator's) | 0.629 | 0.386 | 5.81 |
| clarification, a customer who can (`--realistic`) | **0.796** | **0.424** | **4.63** |

The middle row is what this harness scores, and it is flat, because
`local_evaluator.classify_constraint` never returns `category` — so its simulated shopper
answers a question every real person answers instantly with *"I don't have an additional
preference for category"*. On the public set the row does not exist at all: the opening
message is `coarse_category(target.categories)` verbatim, so **0 of 200** public sessions
ever reach an ambiguous shelf. Official score before and after: 0.91612 and 0.91612. What the ablation costs against a
customer who cannot answer is 0.003 of MRR and nothing else.

The same instrumentation that measured this also found a parser bug worth naming, because
it is the shape mistakes take in this codebase. `BUDGET_CUE` contains "under" and "about" —
"Under" is a brand in an apparel catalog and "the bit I care about is 100% Leather" is
ordinary speech — so on the scaffold axis 44% of all stored slots were being filed as
budgets. Since `_add` supersedes an earlier slot of the same non-`feature` attribute, each
pseudo-budget retired the last real constraint filed under it: **138 of 646 slots superseded,
against 1 of 656 once fixed**. The old behaviour scored 0.014 *higher* on the combined
paraphrase axis, by discarding 137 things the customer had said. That is erasure by accident,
and no one would write it down as a policy.

Nothing hard-codes when to ask. `category` competes in `elicitation.choose` on the same
terms as every other question — a reduction the caller measured times an answerability
the model learned — so a question nobody answers loses that comparison by itself. Over
120 ambiguous sessions the model converges to **p(category) = 0.014** against the
simulator and **0.990** against a customer who answers. Same code, opposite policies,
decided entirely by who is on the other end. That is the clearest demonstration in the
repo of what Pillar III is asking for.

## Two doors for shipping a change

Three of the things in this repo — the commit-width policy, the shelf clarification, the
budget term — are worth less on this simulator than the thing they replaced, or worth
exactly nothing. A rule of "ship only what improves dev and holdout" would have blocked
all three, so there are two doors:

1. it improves dev **and** holdout without regressing the shadow score or the relevant
   adversarial axis; or
2. it is justified as product behaviour, **measured against a model of a real customer**,
   and its cost on this harness is disclosed rather than absorbed.

The second door is not a licence to ship unmeasured work. A second-door change still has
to be measured — just against something other than the leaderboard. Every one of them
carries its number above.

## The V6 cycle: a measured decline, and why the public number stays 0.91612

The question "can this score 0.95?" has a measured answer, and it is no — not honestly.
The score formula allows it (perfect rank at current timing is 0.984; a label-aware oracle
is 0.992), but the remaining public gap is **unidentifiable ordering**, not modelling:
in 53/55 dev and 50/51 holdout losing turn-1 cases the target is *less* popular than the
winner while sharing every disclosed constraint, so the visible evidence genuinely ties
them — and 48/100 dev sessions disclose no lexical constraint at all. Every capacity
probe is negative (fitting on the answers buys +0.0003; six features and boosted trees
were declined). The only measured way above ~0.94 is the pre-rebuild behavior of
terminating on first sight, which this repo deliberately abandoned: it exploits the
harness, no shopper does it, and the judging criteria (Technical Execution 35%, Impact
20%, Feasibility 15%) explicitly discount it. The honest ceiling of this architecture is
~0.935. (The headroom derivation lives in an internal working document that is not part
of this public tree; every number it rests on is in `analysis/`.)

What looked winnable at the time was the **private paraphrase tail**: the category axis
drops the score to 0.84066 and the all-axes paraphrase to 0.72979, and the specification
then warned that natural-language paraphrasing might be added. (The 2026-08-31 FAQ has
since ruled that out for the final evaluation, which retires the risk this cycle was
hedging — the cycle's finding stands, its motivation does not.) The V6 cycle attacked
exactly that with a bounded, preregistered cycle over two non-exact shelf-resolution candidates
(blind aliases vs catalog-only sparse mutual nearest neighbours), the exact/public route
held byte-identical. **It terminated in DECLINE.** The alias candidate passed every
safety guard — public transcript byte-identical, zero transform lookups on exact
sessions, every noninferiority axis clean — but moved dev category turn-1 pool recall
+0.010 against the registered +0.030 gate, recovering 1 of 13 turn-1 misses and 0 of 6
ever-misses: the vocabulary the harness substitutes is not general-language head
synonyms. The MNN candidate never reached scoring: its outcome-blind audit found 32.9%
supported precision against a 98% bar — catalog co-occurrence at that threshold is
listing exhaust (SKU fragments, bare measurements, marketing coincidences), not shopper
shelf vocabulary. Both routes are closed in `analysis/declined_experiments.json`; the
full evidence chain is `analysis/improvement_plan_v6_results.json`. That is the pipeline
working as intended: a bounded question, a frozen protocol, and a negative result that
is cheap to trust because nothing was tuned after the fact.

### The LLM semantic ranking probe: tested, declined, on the record

The pipeline base names *LLM Semantic Ranking* by name, and this repository's ranking
stage is a linear model — so the obvious question is whether an LLM would order better.
It was measured, on the same frozen snapshot harness and the same paired bootstrap the
linear reranker cleared, with the probe living in `tools/` only so the scored path keeps
its zero-socket property (`tools/llm_rerank_probe.py`; 254 snapshots ranked by
`moonshotai/kimi-k3` at temperature 0, whole transcripts cached for re-checking).

The result is not close. The best blend of the LLM's ordering is **+0.0025 dev
[−0.0044, +0.0100], −0.0061 holdout** — within noise on dev, negative on holdout — and
every stronger blend degrades monotonically, down to **−0.172 dev / −0.249 holdout**
when the LLM's order is taken raw. Against the shipped reranker's +0.0207 / +0.0354
(reproduced bit-exactly by the probe's own self-check), semantic judgement adds nothing:
the harness quotes constraint strings verbatim from target metadata, so the ranking
signal is lexical containment plus a popularity prior, both already exploited — and taken
raw, the LLM *demotes* the popularity prior the targets are drawn from. Declined as D11
in `analysis/declined_experiments.json`; full record `analysis/llm_rerank_probe.json`.
The tokens spent were build-time measurement tokens (200k prompt / 431k completion); the
scored path remains 0 prompt / 0 completion.

Read the shape rather than the sign: the degradation is *monotone* in how much the LLM
is trusted, which is not a tuning miss but the signal being wrong — taken raw, the model
demotes the popularity prior the targets are actually drawn from. And the limits of the
claim, because they matter: this is **one model, one prompt strategy, and one placement**
— reordering an already-retrieved head. It licenses "an LLM ranking stage was built,
measured against the standing bar, and declined." It does not license "an LLM adds
nothing here." In particular the adversarial category axis (0.84066) is a *parsing and
routing* weakness, not a ranking one, and this probe says nothing about it.

### The decision pipeline

Nothing ships on a point estimate. A change passes through, in order:

1. **Registration.** Gates, formulas, filters, seeds, and prohibited inputs are hashed
   into `analysis/v6_cycle_protocol.json` *before* any candidate asset exists. Changing
   any of them after results is a new experiment, not a continuation.
2. **Development gates (V6-D1—V6-D9).** Dev split only (sessions 1—100): exact-route
   byte-parity, turn-1 pool recall +0.030 with a paired one-sided bound above zero,
   fixed-width score gain, outcome-blind mapping-audit precision ≥0.98, and every
   noninferiority guard (other axes, shadow, fallback rate, pool growth) as one
   intersection — a failed guard is never averaged away.
3. **Selection.** The fixed rule in the protocol chooses the winner (larger fixed-width
   lower bound; ties break to smaller asset, then faster activation). The loser is never
   combined or rescued.
4. **One-shot confirmation (C1—C6).** The winner alone is read once against the sealed
   lexical-shift confirmation set and the target-free shelf-language audit, plus the full
   public transcript, adversarial, shadow, resource, and test guards. Failure declines the
   whole cycle; a measured decline is a committed result, not a restart.
5. **Record.** Every outcome lands in `analysis/` with hashes, intervals, and an explicit
   pass/fail/blocked disposition; declined experiments go to
   `analysis/declined_experiments.json` so nobody re-litigates them.

### Phase 0 repairs (foundation, behaviour-neutral)

These changed no scoring behavior — the clean public run and the full adversarial matrix
reproduce the committed control values exactly (full 0.91612; category 0.84066) — and each
exists because a guard was measuring the wrong thing:

- **Split isolation (`tools/bench.py`).** `--adversarial` used to ignore `--splits` and
  always read the full 200-session set; it now honours `--splits` and `--axes`, a
  requested-but-absent split fails closed instead of silently passing, every row carries
  its dataset hash, and `--against` *refuses* mismatched inputs rather than comparing the
  intersecting session ids and calling it paired. (A literal `%` in the `--against` help
  string also crashed `--help` outright.)
- **Fail-closed controls (`tools/v4_controls.py`).** `_commands` had lost its `return`,
  and `build_controls` returned its report one statement *before* validating it — the
  validation was unreachable, so a malformed report could be emitted as green. Both are
  repaired and pinned by regression tests.
- **The RSS meter (`tests/test_operational.py`).** G3's child process read its peak RSS
  via `getrusage`, which under Linux's vfork-style spawn can inherit the *spawning
  parent's* memory high-water — it reported 1731 MB for a process whose true peak
  (`/proc/self/status` VmHWM) was 660 MB. The child now reads VmHWM; the 1200 MB budget
  is unchanged.
- **The hedged-latency guard (`tests/test_pillar1_routing.py`).** The in-process 10-turn
  average measured the suite's GC state as much as the agent. It now measures in a fresh
  process and asserts a 150 ms coarse tripwire: the V6 execution machine (WSL2, Python
  3.10) runs this workload at ~55-60 ms at full clock and ~107-119 ms throttled, so the
  tripwire clears machine power state while still catching a real route regression. The
  registered 50 ms zero-exceedance canary in `tools/resource_probe.py` is unchanged and
  remains the ship gate, measured on a quiet full-clock machine (hedged max 48.6 ms over
  299 turns). The cold-start budget in `tests/test_operational.py` is re-derived the same
  way (60 s; 21 s full-clock, 33-36 s throttled, against the original machine's 6.2 s).
- **Fresh baselines.** `analysis/dev.jsonl` / `holdout.jsonl` rebuilt and checksum-verified;
  the suite was 463 tests at that point, green, on the repaired foundation; it is 541 today.
- **Resource baseline, regenerated at the current HEAD (2026-08-31).**
  `analysis/resource_probe_v6_baseline.json` is now measured on this repository's exact
  source — its recorded `source_sha256` map matches every file in `src/` and `tools/`
  byte for byte, which the previous artifact no longer did. Quiet arm64 macOS host,
  Python 3.9.6, 299 serial turns per route in a fresh process:

  | Route | median | p95 | p99 | max | peak RSS | build |
  |---|---:|---:|---:|---:|---:|---:|
  | exact | 3.997 ms | 4.772 ms | 5.178 ms | 5.460 ms | 781.2 MB | 17.069 s |
  | hedged | 30.914 ms | 32.550 ms | 33.566 ms | 34.647 ms | 788.9 MB | 17.678 s |

  Zero observations at or above the 50 ms canary on either route. The earlier V6 figures
  (exact max 7.1 ms, hedged max 48.6 ms, peak RSS 662 MB) were measured on the WSL2/x86_64
  execution machine and are retained only as history; they are not comparable to the row
  above and were never regenerated after the source changed.

  **One contended run, reported rather than discarded.** The first regeneration attempt ran
  immediately after the 567-second test suite on the same host and produced a single hedged
  observation at 56.8 ms — 1 violation in 299, gate failed. Re-run on a quiet machine, as the
  protocol requires, the same workload produced 0/299. We are reporting the quiet run as the
  baseline and this paragraph as its caveat: the hedged route has roughly 15 ms of headroom
  against a 50 ms budget, and that headroom does not survive a loaded machine. Treat 50 ms as
  an engineering canary, which is what it is called, and not as a latency SLA.


## Compliance scope: what we call things, and what they actually are

The innovation directions in `docs/competition_specification.md` name vector similarity
and LLM-based semantic ranking. We did not ship either, and this repository should not
be read as claiming otherwise. Five places where a fast reading of our own vocabulary
would overstate what the code does:

| We say | The code actually does | Where |
|---|---|---|
| "dense route" | sparse catalog-mined `P(shelf \| word)` map with suffix-stem and character-trigram fallback; **no embeddings, no vector similarity** | `src/semantic.py` |
| "reranker" | two stages exist: the default is a small **linear model** over interpretable features, and an **LLM semantic ranking stage** (`src/llm_rank.py`) exists behind `TECHJAM_LLM_RERANK=1` + `AIAND_API_KEY`, off by default, with a total fallback to the linear order on any failure. It is off because it was measured and lost (D11); the live demo run is in `analysis/v8_w2_llm_stage.json` | `src/rerank.py`, `src/llm_rank.py`, `analysis/llm_rerank_probe.json` |
| "hard constraint" | **soft-scored with a large weight**, never a filter and never a lock. Deliberate: a hard filter on a mis-parsed constraint empties the pool. But it is a ranking statement, not a guarantee | `src/scoring.py` |
| "slot erasure" on intent override | both modes exist: the default **down-weights the superseded slot to `SUPERSEDED_WEIGHT = 0.08`**; `P_SUPERSEDE=erase` deletes it from the state entirely. Erase measured **−0.00275 official and one intent-override hit**, so decay stays the default and the deviation is quantified, not asserted | `src/state.py`, `analysis/v8_w3_erasure.json` |
| "retrieval cutoff" on over-general turns | **both halves exist**: a pre-retrieval prediction (`CommitPolicy.pre_cutoff` — pool breadth estimated from bucket sizes before any product is scored; fires only when no discriminating constraint is filed, then scores a bounded 240-item sample instead of the full pool) and the post-retrieval confirmation (`CommitPolicy.cutoff`). Measured at zero cost: score and transcript byte-identical | `src/routing.py`, `src/policy.py`, `analysis/v8_w4_cutoff.json` |

And the loop that used to be open is now closed:

- **`LongTermMemory.recall()` is consumed at two production sites** (W1). A profile
  shape's recalled shelves enter the capped profile tie-break (`src/profile.py`), and its
  recalled asked/answered history seeds the answerability model as a bounded prior that
  settles exact-value ties between *measured* question values (`src/answerability.py`,
  `src/elicitation.py`). The additive form of that prior was measured first and moved the
  public transcript; the shipped tie-break form is the reduction the plan's own gate
  demanded. On this harness the effect is exactly zero — sessions are isolated
  single-user interactions, so a profile history can only ever settle a coin flip, and
  the public transcript is byte-identical at the shipped session ordering. Measurement:
  `analysis/v8_w1_longterm.json`. The *shared learned state* around it is not zero-effect,
  and is quantified under [Cross-session state](#cross-session-state-and-what-it-costs).

The evaluation consequence of each of these is measured, not assumed: the vector route
we did build is the GloVe-50 row under *Limitations*, and it changed the official score
by 0.00000 while costing 0.04 of top-1 shelf resolution on the very axis it targeted.
We would rather ship the thing that measured better under an honest label than ship the
thing with the expected label.


## Cross-session state, and what it costs

Two structures outlive a session by design — the answerability model
(`src/answerability.py`) and long-term memory (`src/memory.py`), both constructed once in
`Agent.__init__` and named there as "shared across sessions: this is what makes the agent
adaptive rather than merely stateful within one conversation". That is Pillar III, and it
is the mechanism the elicitation table above is built from: `P(the customer can answer)`
is learned by asking, across sessions, from a uniform prior.

`docs/final_evaluation_faq.md` §5 requires that "conversational state must remain isolated
between sessions" while allowing shared immutable indexes. This agent satisfies it in the
sense the clause is about — every constraint, slot, rejection, question and slate lives in
a per-`session_id` `Session` object (`src/agent.py`), and `tests/test_agent_contract.py`
pins that no dialogue state leaks between two open sessions. What crosses the boundary is
not conversation, it is an aggregate over questions asked and answered, with no session
content and no identifier in it. But it is mutable and it accumulates, so the run is
order-conditional, and that deserves a number rather than a reassurance:

| Official metric, 200 public sessions | Score | Sessions whose turn or rank changed |
|---|---|---|
| shipped order | **0.916125** | — |
| shuffled, seed 7 | 0.914950 | 10 / 200 |
| shuffled, seed 13 | 0.912225 | 7 / 200 |
| a fresh learned state for every session | 0.904675 | 41 / 200 |

Three things follow, and we would rather state them than be asked.

- **Cross-session learning is worth 0.0115 of official score**, not zero. The shadow
  harness reports 0.004 for the same axis, but shadow scores internal ranking with the
  commit policy removed, so the two measure different quantities and the official figure
  is the one to quote. The last row is the honest floor: what this agent scores if a
  harness constructs one instance per session.
- **The exact score is conditioned on session order**, by about 0.001-0.004 on 200
  sessions, and both sampled shuffles landed *below* the shipped order rather than
  straddling it. Any single reported number, including 0.916125, is the number *at the
  shipped ordering of the public set*. The swing is well inside the ±0.017 marginal
  interval a 200-session run carries anyway, and it shrinks on 800.
- **It is bounded.** `MAX_SESSIONS = 256` evicts old sessions, `MAX_PROFILES = 4096` caps
  the memory, and the answerability table is keyed by attribute, of which there are ten.
  An 800-session run accumulates the table, not the transcripts.

One honest caveat on top of that: every ordering we have sampled, on both metrics,
scores at or below the shipped one. Two seeds is not enough to call 0.916125 a
high-water mark, but it is enough that we will not call it order-independent either.
The claim we will not make is that ordering is irrelevant.

## Limitations

- **Popularity bias, measured and now guarded.** The prior is a large lever and it is
  the classic long-tail pathology (Steck 2011; Abdollahpouri et al.). We wanted to
  report performance on niche intent and largely cannot: only 4 of 200 targets sit
  below their bucket's median, and the median target sits at the **99.3rd percentile**
  of its own shelf. Drawn at the 90th percentile instead, n=25 scores **0.853** against
  **0.923** for the rest — a real gap, on a sample too small to lean on. The honest
  statement is that this evaluation set has almost no long tail to bury.

  Two things about that number are worth stating plainly. It moved: before the learned
  reranker the same slice scored 0.839 against 0.915, so the model *narrowed* the gap
  slightly rather than widening it, which is not what we expected from a model that
  wants a popularity-to-phrase ratio of 7.6:1 where the shipped weights use 1.4:1. And
  a point estimate on 25 sessions is not a finding, so `tools/bench --longtail` now
  records its sessions and takes `--against`, which turns the slice into a paired
  interval and a ship gate for any future retrain. The unit test that goes with it
  (`tests/test_rerank.py::PopularityPathologyTest`) puts the same invariant where it
  fails fastest: a long-tail leather belt the shopper described, a popular canvas one
  they did not, and the requirement that the *combined* ranking — weighted sum plus
  the model — keeps the first one on top at four times the shipped blend.
- **Category resolution is still the weakest axis, and we now know why it is not a
  vocabulary problem.** 0.746 under synonym substitution against 0.944 clean. An
  earlier draft of this README guessed that a local sentence encoder over the 1,115
  shelf names would close much of it. That guess was wrong, and measuring it was
  worth more than the feature would have been.

  Classifying every category-axis failure: 65.5% resolve correctly, **25.5% land on a
  near-duplicate of the right shelf** (Jewelry Necklaces for Necklaces Chains, Women
  Bodysuits for Shapewear Bodysuits), 8.5% reach the wrong kind of product, and 0.5%
  resolve nothing. The lexical and mined routes together come up empty in **1 session
  out of 200** even with every head noun substituted, because the axis replaces some
  words and leaves the rest, and the rest carry the signal.

  So the gap is shelf *disambiguation*, not shelf *identification* — and a
  general-language vector space is precisely the wrong instrument for it, because
  "Jewelry Necklaces" and "Necklaces Chains" are semantically identical. Three
  attempts, all reverted, all measured on the public set and the shadow harness:

  | Attempt | Official | Shadow |
  |---|---|---|
  | baseline | 0.94401 | 0.89397 |
  | GloVe-50 vector route fused in (200k words, 19.5 MB, numpy) | 0.94401 | — |
  | near-duplicate shelves merged into the primary boost | 0.93399 | 0.88328 |
  | near-duplicate shelves given a graded partial boost | 0.94475 | 0.89407 |

  The vector route also cost 0.04 of top-1 shelf resolution on the very axis it was
  built for. Nothing here earned a 19.5 MB asset and a numpy dependency, so nothing
  here shipped.

  Two further attempts, and then the answer. The first three all chose the shelf from
  the *opening phrase*. The obvious remaining idea was to choose it from the
  conversation instead — the evidence that separates twins arrives on turns 2 and 3,
  as constraints, so a shelf picked at turn 1 is picked blind. That needed a benchmark
  that grades the right thing, because shelf top-1 is a proxy and it is not the broken
  part: `tools/shelfbench.py` mines 170 near-duplicate pairs from the catalog, puts a
  target on one of them, opens with the words the two shelves share (what a shopper
  actually says when the word they reach for belongs to both), and scores where the
  *item* lands.

  | Attempt | shelfbench HR | shelfbench MRR | shelf top-1 | category axis |
  |---|---|---|---|---|
  | static boost on the retrieval leader (incumbent) | 0.558 | 0.343 | 0.176 | 0.74494 |
  | argmax re-election, boost scaled by margin | 0.525 | 0.325 | 0.205 | — |
  | posterior over shelves, belief spread not committed | 0.558 | 0.341 | 0.254 | 0.73598 |

  The first row of that table is the finding. Re-election **worked on its own terms** —
  shelf accuracy rose from 0.176 to 0.205, and to 0.254 for the posterior — and the item
  metric got worse anyway. The true shelf is among the candidates only 75% of the time
  and there are about nine of them, so concentrating belief on a leader that is right one
  time in four loses more than it wins. Improving an argmax is the wrong move when the
  argmax is usually wrong.

  And no version of it can work, because the posterior never sharpens. Across five turns
  of accumulating constraints, top-1 accuracy stays between 0.15 and 0.23 and the maximum
  probability stays at about 0.33. Near-duplicate shelves stock genuinely similar
  products; the evidence that would separate them is the same evidence on both sides.
  This is the wall the GloVe route hit, reached from the other direction — there we could
  not separate two labels that mean the same thing, here we cannot separate two shelves
  that stock the same thing.

  What the same measurement also shows is that the information is worth a great deal and
  simply is not ours: with the shelf supplied rather than inferred, hit rate rises by more
  than a sixth against a pool ceiling of 0.787. So the agent asks — see *Asking the
  question the catalog cannot answer* below. Retrieval on this catalog is at a local
  optimum; the way past it was not a better retriever.

  (The three rows above were measured at the revision of `shelfbench` that opened with the
  two shelves' shared *stems*. "Shoes" stems to "sho", which is not a thing a person says,
  so the bench now opens with their shared surface words and every absolute number rose by
  about 0.07. The comparison between the variants is unaffected — same bench, same day —
  and the numbers below are all from the corrected version.)
- **Long-term memory is read, but the harness gives it almost nothing to work with.**
  `LongTermMemory.recall()` is consumed by the profile tie-break and by question
  selection (see *Compliance scope* above). The limitation is the interlocutor, not the
  loop: every session is an isolated single user by design, so a profile's history can
  only ever settle an exact tie between two equally-measured questions — which does not
  arise on the public set, leaving the transcript byte-identical. See *Compliance scope*
  above, and item 1 of *What we would improve with more time* below.
- **`preference_tags` are close to unusable.** They are category-level abstractions —
  the literal strings `"material"`, `"fit"`, `"comfort"` — and every clothing listing
  contains those words. Used as an additive bonus this cost **0.20**. It is now a
  tie-breaker that only reorders candidates already within 0.02 of each other.
- **The simulator is not a user.** Simulator-based CRS evaluation correlates weakly with
  human satisfaction (Kendall's τ 0.14–0.37; SIGIR-AP 2025). The adversarial harness
  narrows that gap; it does not close it.

### What we would improve with more time

Ranked by how much we think each would move a real shopper, not the leaderboard:

1. **Validate against humans.** Every number here is simulator-bound. The adversarial
   and shadow harnesses narrow the gap between simulator and person; they do not close
   it, and the published τ 0.14–0.37 correlation says how far that gap can be.
2. **Attack shelf disambiguation with signals the catalog does not expose.** We proved
   the ceiling from two directions and it is not a vocabulary or a model-class problem:
   near-duplicate shelves stock the same products, so the separating evidence is
   identical on both sides. Supplying the shelf raises hit rate by more than a sixth.
   That information has to come from somewhere outside the product text.
3. **Price coverage.** 78.9% of products carry no price, which makes budget the
   constraint we can least often honour. A price-imputation model with an abstention
   rule is the obvious next asset.

## Quick start

```bash
curl -L -o catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
# catalog.jsonl.gz sha256: 07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
# data/catalog.jsonl sha256: da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67
gzip -dc catalog.jsonl.gz > data/catalog.jsonl

python3 -m tools.setup_check --unpack --splits --verify-checksums
                                                   # verify both published hashes,
                                                   # decompress the catalog, and rebuild
                                                   # the dev/holdout splits a clean clone omits
python3 -m evaluator.local_evaluator               # 0.91612
python3 demo_session.py --all                      # one annotated session per scenario
python3 demo_session.py --showcase                 # live clarification + product switch
bash run_tests.sh all                              # 541 tests, no skips after setup
python3 -m tools.bench --adversarial               # the robustness matrix
python3 -m tools.shadow                            # the artifact-sensitivity matrix
python3 -m tools.bench --longtail                  # the popularity slice
python3 -m tools.shelfbench --realistic            # near-duplicate shelves, a customer
                                                   # who can say which one they meant
python3 -m tools.recall                            # retrieval vs ranking, per stage
python3 -m tools.audit                             # hash-seed/offline/resource evidence
python3 -m tools.bench --against analysis/bench.json   # paired interval for a change

# the reranker and V4-0 foundation, reproducible end to end
python3 -m tools.rerank_data --split dev           # writes ignored cache + manifest
python3 -m tools.rerank_data --split holdout
python3 -m tools.snapshot_mrr --splits dev,holdout,full --rebuild
python3 -m tools.rerank_train                      # legacy A0 compatibility control;
                                                   # writes an inert candidate, not V4-1
python3 -m tools.snapshot_mrr                      # paired session snapshot metric
python3 -m tools.resource_probe --include-fallback \
  --output analysis/resource_probe_v4_0.json       # 299-turn fixed-workload canary
python3 -m tools.v4_controls \
  --resource analysis/resource_probe_v4_0.json \
  --output analysis/v4_end_to_end_controls.json    # complete frozen control matrix
python3 -m tools.v4_baseline \
  --output analysis/improvement_plan_v4_results.json
python3 -m tools.rerank_ablate                     # add-one / drop-one over any
                                                   # features appended to FEATURES
python3 -m tools.rerank_stumps --depth 2           # boosted trees, declined
python3 -m tools.violations --axis constraint      # contradictions in the slate
python3 analysis/boundary_diagnosis.py             # the weakest scenario slice
python3 analysis/question_value.py                 # what a question is worth
```

Python 3.9+, verified on 3.9.6. No `pip install` step — `requirements.txt` is empty by
design. Nothing to configure; `Agent()` locates the catalog itself.

## Final evaluation

Per `docs/final_evaluation_faq.md` §1, the 800 final sessions are released **after** the
Devpost deadline, and each team runs the unmodified official evaluator itself, against
its own frozen submitted commit, then retains the results.

`results.json` is gitignored here — it is a generated artifact and the public tree should
not carry a stale copy. That is right for development and dangerous on evaluation day, so
the retention step is a tool rather than a thing to remember:

```bash
# One-time setup: lets the retention manifest compare the evaluator to the organizer.
git remote add upstream https://github.com/TechJam2026/techjam-conversational-search.git
git fetch upstream

python3 -m tools.final_run --dataset path/to/the/released/final_set.jsonl
```

It writes `final_evaluation/` containing `results.json` **with per-session results**,
the evaluator's stdout, and a `manifest.json` recording the submitted commit SHA and
cleanliness, the evaluator's digest and whether it still matches the organizer remote,
dataset and catalog checksums, headline and per-scenario metrics, wall time and peak RSS,
the Python version and hardware, and whether the optional LLM stage was enabled. That is
the "commit hash and relevant environment and execution details" the FAQ asks for, and
the evidence to hand over if the organizer requests it.

Three refusals are deliberate:

- **a dirty working tree** — the FAQ freezes the submitted commit, so a dirty tree means
  the thing you ran is not the thing you submitted (override with `--allow-dirty` only
  if you know why);
- **an existing bundle** — final-evaluation evidence is written once;
- **a missing dataset** — with a reminder that the package does not exist before the
  deadline.

Dry-run verified against the public 200: 0.916125, 200 per-session rows retained.

### On the day

1. `git status` must be clean and on the submitted commit — `tools.final_run` refuses a
   dirty tree for exactly this reason. Untracked files are ignored, so the released
   dataset can sit in the tree.
2. `data/catalog.jsonl` must be the frozen catalog: sha256 `da979b05…`, 50,000 rows.
   `python3 -m tools.setup_check --verify-checksums` confirms it.
3. If the released package ships its own `evaluator/`, **use theirs verbatim** — that is
   what "the unmodified official evaluator" means, and the manifest records the digest
   either way. Nothing under `src/` changes, whatever the package contains.
4. `python3 -m tools.final_run --dataset <the released file>`, then keep
   `final_evaluation/` intact. It is the retention obligation in FAQ §1, commit hash and
   environment included.
5. Sanity-check before reporting: `reported_token_usage.total_tokens` is 0 and
   `swallowed turn failures` is absent from stderr. A non-zero token count means the
   optional LLM stage was somehow enabled and the run is not the default configuration.

**Rehearsed at that scale.** We do not have the final sessions, but we do have the shape
of the run: 800 sequential sessions (the public 200 replayed as four shuffled blocks under
distinct `sample_id`s) through the unmodified evaluator path, on the recorded arm64 macOS
host with Python 3.9.6.

| | |
|---|---|
| Agent construction | 17.6 s, once |
| 800 sessions | 7.8 s total, 9.7 ms/session |
| Peak RSS | 961 MB |
| Session store at the end | 256, the `MAX_SESSIONS` bound |
| Profile signatures retained | 75, against the `MAX_PROFILES` cap of 4096 |
| Swallowed turn failures | 0 |
| Reported tokens | 0 |

Nothing grows without a bound and nothing degrades over four times the public workload,
so budget about half a minute of compute and a gigabyte of RAM, not an afternoon. That
run's score is not a forecast — it re-uses public targets — but the operational shape is
the one evaluation day will have. Expect a headline number in the same region as 0.916125
rather than equal to it: different sessions, and the score is order-conditional by about
0.001-0.004 (see [Cross-session state](#cross-session-state-and-what-it-costs)).

## Environment variables

Required by the FAQ (`docs/final_evaluation_faq.md` §7) to be documented. **None is
required to run the agent** — every one has a default, and the official evaluator sets
none of them, so `python3 -m evaluator.local_evaluator` reproduces 0.916125 in a clean
shell with an empty environment. They exist so declined and alternative mechanisms stay
reachable and testable rather than deleted.

| Variable | Default | Effect |
|---|---|---|
| `TECHJAM_CATALOG` | `data/catalog.jsonl` | catalog location, so `Agent()` works from any cwd |
| `TECHJAM_LLM_RERANK` | unset | `=1` **and** a key present enables the optional LLM ranking stage (`src/llm_rank.py`) |
| `AIAND_API_KEY` | unset | credential for that stage. **Never commit a value.** Both variables are required together; either alone is inert |
| `AIAND_BASE_URL` | `https://api.aiand.com/v1` | endpoint for the LLM stage |
| `LLM_RERANK_MODEL` | `moonshotai/kimi-k3` | model id for the LLM stage |
| `P_SUPERSEDE` | `decay` | `=erase` selects literal slot erasure instead of the 0.08 decayed trace (`analysis/v8_w3_erasure.json`) |
| `P_ASK` | `infogain` | question-selection strategy; `counterfactual` reaches the declined P8 mechanism |
| `P_FUSE`, `P_PRUNE`, `P_PROBE`, `P_SHELF_TRANSFORM` | shipped values | route fusion, rejection model, probe width, and the dormant V6 shelf transform (`off`) |
| `P_K`, `P_N`, `P_LIMIT`, `P_MARGIN`, `P_SOFT`, `P_UNSURE`, `P_WIDEN`, `P_OVERLOAD`, `P_BUCKETS` | shipped values | scoring and commit-policy knobs, exposed for ablation |
| `W_POP`, `W_TXT`, `W_PHRASE` | `1.40`, `0.40`, `1.00` | popularity, lexical/BM25, and exact-phrase scoring weights |
| `W_BUDGET`, `W_FACET` | `0.60`, `0.50` | budget-preference and structured-facet scoring weights |

No credential is needed for the default path, and no secret value appears anywhere in
this repository.


## Disclosure

Network: never used. Runtime model: a deterministic local linear reranker; no LLM/API is
called. Tokens: 0 prompt, 0 completion across all 200 sessions — verified in
`results.json`, which reports `total_tokens: 0`. `results.json` is the evaluator's own
output file and is deliberately **not committed** (it is in `.gitignore`, like every
other regenerable artifact): reproduce it in one command with
`python3 -m evaluator.local_evaluator`. The committed copies of the same evidence are
`analysis/final_numbers.json` (score, scenario slices, token counts) and
`analysis/operational_audit.json` (evaluator checksum, forbidden-import scan, resources).
Estimated scoring cost: **$0.00**.
Current-HEAD resource evidence (2026-08-31, quiet arm64 macOS host, Python 3.9.6, 299
serial turns per route in a fresh process, `analysis/resource_probe_v6_baseline.json`,
whose recorded source hashes match this tree byte for byte): exact p99 5.178 ms / max
5.460 ms, hedged p99 33.566 ms / max 34.647 ms, 0/299 observations at or above 50 ms on
either route; peak RSS 781.2 and 788.9 MB; agent construction 17.069-17.678 s, excluded from
turn latency. A run taken while the machine was loaded produced one 56.8 ms hedged
observation; the 50 ms figure is an engineering canary, not an SLA.

On the recorded arm64 macOS host, V4-0 fresh-process fixed-message canaries measured
exact p99 3.788 ms and hedged p99 29.291 ms, each with 0/299 observations at or above
50 ms; peak RSS was 765.5 and 768.2 MB. Full-catalog fallback p99 was 52.355 ms and is
descriptive only, not an SLA or comparative pass. Older 14.7 s/773 MB/41 ms figures used
a different protocol and are historical. An LLM was used at development time to write
and review code, to run the P1 ranking probe above (200k prompt / 431k completion
tokens, build-time only, `analysis/llm_rerank_probe.json`), and to demo the W2 opt-in
stage (8k prompt / 18k completion, `analysis/v8_w2_llm_stage.json`); none is called at
runtime or during scoring. Network access is **not required**: the default path opens
no socket, and the only network-capable code is the opt-in stage, whose fallback is the
default — the submission runs identically with the network disabled.

## Repository layout

```
src/                        the agent
tests/                      541 offline tests, grouped by pillar
tools/                      adversarial harness, paired behavior benchmark runner,
                            V4 provenance/resource/control/report builders, paraphraser,
                            near-duplicate shelf micro-benchmark,
                            retrieval-vs-ranking decomposition (recall.py),
                            constraint-contradiction counter (violations.py),
                            catalog setup check, operational audit, and reranker bench:
                            rerank_data / legacy-A0 rerank_train / rerank_ablate /
                            rerank_stumps, with snapshot_mrr.py as the ranking metric
analysis/                   measurements and the technical brief
demo_session.py             one annotated multi-turn session
starter/agent.py            harness shim re-exporting src.agent.Agent
starter/agent_baseline.py   the provided weak_bm25 starter, preserved
submission/                 bundle disclosure, Devpost draft, video script,
                            and recorded live demo transcript
run_tests.sh                test runner
```

## Team contributions

Deliverable 4.5(2) requires this section **in the README**, not only on Devpost. The
same attribution is included in the Devpost *Team* section.

| Member | Contribution |
|---|---|
| Song Yao Zhu | Dialogue state and constraint parsing (`src/state.py`, `src/text.py`); the learned reranker and the snapshot-MRR metric it was judged by (`src/rerank.py`, `tools/snapshot_mrr.py`); the dev/holdout split protocol and the declined-experiment ledger; submission packaging. |
| Keegan Gan | Agent refinement, model-quality verification, and comparative evaluation against the optional LLM reranking approach. |
| Keith Chia | Agent evaluation and continued research. Agent evaluation tests. |
| Isaac Choong | Model-quality verification and production of the submission demo video. |
| Lee Ren Kai | Defensive recommendation-limit handling for zero, negative, oversized, and malformed `top_k` values; regression coverage for non-positive limits. |
