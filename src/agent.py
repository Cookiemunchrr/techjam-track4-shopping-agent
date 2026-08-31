"""The Agent the harness imports.

Pipeline per turn:

    observe -> route -> score -> commit -> elicit -> adapt

Every layer degrades into the next rather than depending on exact strings, because
docs/competition_specification.md warns the customer utterances may be paraphrased.
Runs offline on CPU with no model calls, so `usage` is always zero.

Where each pillar of the brief lives:

    I   dual-track routing        src/routing.py, src/semantic.py, src/fusion.py
    II  dialog state machine      src/state.py, CommitPolicy.cutoff
    III self-evolution            src/answerability.py, src/memory.py,
                                  src/profile.py, src/orchestrator.py
    IV  evaluation                tools/bench.py, tools/adversarial.py, tests/
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from .answerability import AnswerModel
from .catalog import COLOR_RE, MATERIAL_RE, Catalog
from .clarify import match as match_shelf
from .clarify import options as shelf_options
from .clarify import phrasing as shelf_phrasing
from .clarify import should_ask as shelf_ambiguous
from .elicitation import ALLOWED, choose
from .features import Context
from .memory import LongTermMemory
from .orchestrator import BROADEN, COVER, Orchestrator
from .paths import resolve as resolve_catalog
from .rerank import load as load_reranker
from .policy import TOP_K, CommitPolicy, RejectionModel
from .profile import ProfilePrior, break_ties
from .routing import (BROWSING, PRIMARY_BOOST, UNSURE_BOOST_SCALE, category_key,
                      detect_intent, diversify, estimated_pool_size, exact_bucket,
                      fuse_global, route_detail, _overlap_ranked)
from .scoring import Scorer, Weights
from .semantic import Semantic, stem
from .shelf_transform import (MAX_ADDED_SHELVES, OVERLAP_LIMIT,
                              load as load_transform, rrf_merge)
from .state import DialogState, classify
from .text import (RESTART, fold_unicode, is_correction, is_restart, normalise,
                   split_clauses, tokens)

DEFAULT_CATALOG = None          # see src/paths.py -- resolved, never assumed

MAX_SESSIONS = 256              # bounded; the private set is 800 sequential sessions
# How deep the learned reranker is allowed to reorder. It reorders; it never
# retrieves, so nothing below this line can be promoted and nothing above it can be
# lost. See src/rerank.py.
RERANK_DEPTH = 200
DISTIL_CAP = 12                 # slots kept after Personalized Context Distillation
# W4: when the pre-retrieval cutoff fires, scoring is bounded to a sample twice
# the size of the elicitation window (elicitation.POOL = 120), so the question
# choice sees every value it would have seen. Declared structure, not fitted.
PRE_CUTOFF_SAMPLE = 240
NEGATIVE_PENALTY = 0.9          # weight of an explicitly refused attribute value
# How much of that a product earns for merely mentioning the refused value in its
# prose rather than claiming it. The asymmetry with positive scoring is deliberate
# and measured: for a positive facet, membership in the value set is *less*
# discriminative than equality on the primary value and costs score (see
# scoring.Scorer._facet). For a refusal it is the other way round -- wrongly
# demoting a product over a word in its marketing copy is the worse error, so the
# refusal reads the wider set and grades it instead of ignoring it.
MENTION_SCALE = 0.4
PRODUCT_BOUND_QUESTIONS = frozenset((*DialogState.PRODUCT_BOUND, "category"))


class Session:
    """Per-conversation state. One instance per session_id."""

    __slots__ = ("dialog", "asked", "rejection", "pending", "orchestrator",
                 "profile", "signature", "internal", "offered", "pool",
                 "pool_size", "context", "ranked", "route_trace")

    def __init__(self, rejection_mode: str, profile: ProfilePrior,
                 signature: str, supersede_mode: str = "decay") -> None:
        self.dialog = DialogState(supersede_mode)
        # Diagnostics only. See Agent.internal_ranking.
        self.internal: list[str] = []
        # Diagnostics only, and only when the matching Agent.trace_* flag is on.
        self.pool: frozenset[str] = frozenset()
        self.pool_size = 0
        self.context = None
        self.ranked: list = []
        self.route_trace: dict | None = None
        # Shelves put to the customer by a clarification, awaiting their pick.
        self.offered: list[str] = []
        self.asked: set[str] = set()
        self.rejection = RejectionModel(rejection_mode)
        self.pending: str | None = None      # the question awaiting an answer
        self.orchestrator = Orchestrator()
        self.profile = profile
        self.signature = signature

    # Kept so anything reading the old flat-clause interface still works.
    @property
    def clauses(self) -> list[str]:
        return self.dialog.phrases()

    @property
    def category(self) -> str | None:
        return self.dialog.category

    @property
    def intent(self) -> str | None:
        return self.dialog.intent


class Agent:
    """Conversational shopping agent. See README for the design rationale."""

    def __init__(self, catalog_path: str | Path | None = DEFAULT_CATALOG) -> None:
        env = os.environ
        self.catalog_source = str(resolve_catalog(catalog_path))
        self.catalog = Catalog(self.catalog_source)
        self.semantic = Semantic(self.catalog)
        self.scorer = Scorer(self.catalog, Weights.from_env(env))
        self.commit = CommitPolicy.from_env(env)
        self.rejection_mode = env.get("P_PRUNE", RejectionModel.RESET)
        # W3: literal slot erasure as a selectable, measured mode. Anything
        # unrecognised fails closed to the default.
        self.supersede_mode = env.get("P_SUPERSEDE", "decay")
        if self.supersede_mode not in ("decay", "erase"):
            self.supersede_mode = "decay"
        self.ask_mode = env.get("P_ASK", "infogain")
        # The V6 shelf transform. Off by default; one immutable instance, loaded
        # once, checksum-validated, failing closed to off. See src/shelf_transform.py.
        self.shelf_transform = load_transform(env.get("P_SHELF_TRANSFORM", "off"),
                                              self.catalog_source)
        # Whether an unresolved shelf may pull candidates from the whole catalog.
        # On, and unreachable on every official session by construction -- the
        # opening message always resolves its shelf outright there, so the control
        # score moves by exactly zero. It is insurance for a private set whose
        # category vocabulary is not the catalog's. See analysis/global_route.json.
        self.fuse_global = env.get("P_FUSE", "hedged") == "hedged"
        if self.fuse_global:
            self.catalog.index_postings()
        self.sessions: "collections.OrderedDict[str, Session]" = collections.OrderedDict()
        self.failures = 0                    # see the guard in `respond`
        # Both shared across sessions: this is what makes the agent adaptive
        # rather than merely stateful within one conversation.
        self.answers = AnswerModel()
        self.memory = LongTermMemory()
        # Diagnostics are off by default. Recall/rerank tooling retains a pool or
        # feature head; resource tooling records only an integer pool size.
        self.trace_pool = False
        self.trace_pool_size = False
        self.trace_features = False
        self.trace_route = False
        self.reranker = load_reranker()

    @classmethod
    def sharing_index(cls, other: "Agent") -> "Agent":
        """A new agent over the same frozen index, with its own learned state.

        The catalog and the mined vocabulary are read-only and cost ten seconds to
        build; only `answers` and `memory` accumulate. Splitting the two lets
        tools/shadow.py measure cold start -- what this agent would score if the
        private harness constructed one instance per session -- without paying the
        index cost eight hundred times.
        """
        agent = cls.__new__(cls)
        agent.catalog, agent.semantic, agent.scorer = other.catalog, other.semantic, other.scorer
        agent.catalog_source = other.catalog_source
        agent.commit, agent.rejection_mode = other.commit, other.rejection_mode
        agent.supersede_mode = other.supersede_mode
        agent.ask_mode = other.ask_mode
        agent.sessions = collections.OrderedDict()
        agent.failures = 0
        agent.answers = AnswerModel()          # fresh: this is the point
        agent.memory = LongTermMemory()
        agent.fuse_global = other.fuse_global
        # The same immutable transform object and mode (T21): never reloaded,
        # never silently replaced.
        agent.shelf_transform = other.shelf_transform
        agent.trace_pool = other.trace_pool
        agent.trace_pool_size = other.trace_pool_size
        agent.trace_features = other.trace_features
        agent.trace_route = other.trace_route
        agent.reranker = other.reranker
        return agent

    # ------------------------------------------------------------------ API --
    def reset(self, session_id: str, user_profile: dict) -> None:
        """Begin a session.

        The aggregate profile becomes a capped tie-breaker (src/profile.py) and a
        key into long-term memory (src/memory.py). It is never allowed to outrank
        something the customer actually said. What earlier sessions with this
        profile shape learned is recalled here and consumed in both places:
        shelves it converged on enter the tie-break (W1a), and its asked/answered
        history seeds the answerability model as a bounded prior (W1b).
        """
        signature = self.memory.signature(user_profile)
        self.memory.start_session(signature)
        recalled = self.memory.recall(signature)
        self.answers.set_prior(recalled.get("asked"), recalled.get("answered"),
                               recalled.get("sessions", 0))
        self.sessions[str(session_id)] = Session(
            self.rejection_mode,
            ProfilePrior(user_profile, recalled.get("buckets")), signature,
            self.supersede_mode)
        # Bounded: 800 sequential private sessions must not grow without limit.
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)

    def candidate_pool(self, session_id: str) -> frozenset:
        """Every candidate the last turn ranked over. Diagnostics only.

        Recorded only while `trace_pool` is set, because retaining a pool of up to
        fifty thousand ids per live session buys nothing at scoring time.

        This is the hook that separates the two ways a session can fail. If the
        target is not in here, no reranker, no weight and no extra turn can
        recover it, and the honest fix is retrieval; if it is in here and ranked
        eleventh, retrieval did its job and the fix is ordering. Reporting one
        number for both is how a ranking problem gets mistaken for a recall
        problem -- see tools/recall.py, which found exactly that.
        """
        state = self.sessions.get(str(session_id))
        return state.pool if state is not None else frozenset()

    def candidate_pool_size(self, session_id: str) -> int:
        """Size of the last scored pool without retaining its identifiers.

        Resource probes use this integer-only hook. Turning on ``trace_pool`` for
        a full-catalog timing run would retain a 50,000-entry frozenset in every
        live session and measure diagnostic allocation rather than production.
        """
        state = self.sessions.get(str(session_id))
        return state.pool_size if state is not None else 0

    def route_trace(self, session_id: str) -> dict:
        """Route evidence for the last turn. Diagnostics only.

        Recorded only while `trace_route` is set. Carries the exact/hedged/
        fallback flags, the ordered pre-filter pool digest and size (the baseline
        a V6 shelf transform must provably extend, not reorder), the final scored
        pool size, and the transform counters a candidate populates -- all zero
        in the baseline. Contains no target or session identifiers beyond what
        the caller already holds.
        """
        state = self.sessions.get(str(session_id))
        if state is None or not state.route_trace:
            return {}
        return dict(state.route_trace)

    def last_context(self, session_id: str):
        """(feature context, pre-rerank head) for the last turn. Diagnostics only.

        Recorded only while `trace_features` is on. tools/rerank_data.py uses it to
        build a training set out of exactly what the live path computed, which is
        the only way to be sure the model is trained on the features it is served.
        """
        state = self.sessions.get(str(session_id))
        return (state.context, list(state.ranked)) if state is not None else (None, [])

    def internal_ranking(self, session_id: str) -> list[str]:
        """The untruncated head of the last ranking. Diagnostics only.

        `respond` never reads this and it is never returned to the harness. It
        exists so tools/shadow.py can measure retrieval quality independently of
        how much of that ranking the dialogue policy chose to show -- otherwise
        recommendation width and ranking quality are confounded, and a policy that
        merely hides ranks 2-10 is indistinguishable from one that ranks better.
        """
        state = self.sessions.get(str(session_id))
        return list(state.internal) if state is not None else []

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(str(session_id))
        if state is None:                    # defensive: harness may skip reset
            state = Session(self.rejection_mode, ProfilePrior(None),
                            self.memory.signature(None), self.supersede_mode)
            self.sessions[str(session_id)] = state

        try:
            return self._respond(state, user_message or "", turn,
                                 self._normalise_top_k(top_k))
        except Exception:                    # never forfeit a turn to a crash
            self._record_failure()
            return self._empty()

    @staticmethod
    def _normalise_top_k(top_k: int | None) -> int:
        """Keep the harness-controlled recommendation limit within the API bounds.

        A negative value would be interpreted as a Python slice from the end and
        could expose almost the entire ranking. Zero is meaningful to callers that
        only want the conversational response, while invalid values fall back to
        the contract's standard maximum.
        """
        if top_k is None:
            return TOP_K
        try:
            requested = int(top_k)
        except (TypeError, ValueError, OverflowError):
            return TOP_K
        return max(0, min(requested, TOP_K))

    # -------------------------------------------------------------- internal --
    def _respond(self, state: Session, message: str, turn: int, top_k: int) -> dict:
        dialog = state.dialog
        restarting = is_restart(message)
        if restarting:
            # A restart abandons the shelf as well as its product constraints. The
            # same turn may name a replacement shelf, so compute `first` afterwards.
            dialog.restart(turn)
            dialog.category = None
            dialog.category_turn = turn
            state.asked.clear()
            state.offered = []
        first = turn <= 1 or dialog.category is None
        switched = False

        if restarting:
            # The restart words are instructions, not a category or constraint.
            # Anything before the global instruction is abandoned context too;
            # only a replacement request stated after it belongs to the new epoch.
            # Learned answerability and long-term memory remain cross-epoch.
            folded_message = fold_unicode(message)
            restart_match = RESTART.search(folded_message)
            content_message = RESTART.sub("", folded_message[restart_match.end():]) \
                if restart_match is not None else folded_message
        else:
            content_message = message
        if turn > 1 and is_correction(message):
            state.rejection.on_correction()

        # -- observe ---------------------------------------------------------
        clauses = split_clauses(content_message)
        picked = match_shelf(content_message, state.offered) if state.offered else None
        if picked is not None:
            # They answered the clarification. This is the shelf, stated outright,
            # so it goes in without passing the `_is_new_category` bar -- that bar
            # exists to stop a constraint masquerading as a category, and nothing
            # is masquerading when we asked the question. It is emphatically not a
            # product switch either: the constraints they have already given still
            # describe the thing they are looking for, so no product_reset here.
            dialog.category = picked
            dialog.category_turn = turn
            state.offered = []
            dialog.observe(content_message, turn, self.catalog, skip_first=True)
            learned = True
            first = False          # they named a shelf; this is not an opening turn
        else:
            learned = None         # nothing observed yet -- see below
        named_index = 0
        named = category_key(clauses)
        shelf_match = self._named_shelf(clauses) if not first or restarting else None
        if shelf_match is not None:
            named, named_index = shelf_match
        if first:
            dialog.category = named
        if picked is None and not first and named \
                and self._is_new_category(dialog.category, named, message):
            # The category is re-derived on every turn, not frozen at turn 1. A
            # customer who says "actually I need a leather belt" must stop being
            # shown hoodies, which is exactly what used to happen.
            dialog.category = named
            dialog.category_turn = turn
            # A different kind of product retires the constraints that described
            # the old one. See DialogState.product_reset.
            dialog.product_reset(turn)
            state.asked.difference_update(PRODUCT_BOUND_QUESTIONS)
            switched = True
            # Only a genuine change of *category* clears what we have shown: those
            # items were answers to a different question. A change of constraint
            # does not, because the customer still passed on them -- clearing there
            # re-offers the exact product they just rejected.
            state.rejection.on_correction()

        if learned is None:      # None, not False: a clarification answer was
            # already observed above, and observing it twice would double-count it
            learned = dialog.observe(content_message, turn, self.catalog,
                                     skip_first=first or switched,
                                     opening_index=named_index
                                     if switched or restarting else 0)

        # -- adapt -----------------------------------------------------------
        if state.pending is not None:
            self.answers.observe(state.pending, learned)
            self.memory.observe(state.signature, attribute=state.pending, answered=learned)
            state.pending = None
            state.orchestrator.observe(learned)
        strategy = state.orchestrator.strategy()

        if len(dialog.slots) > DISTIL_CAP:
            dialog.distil(DISTIL_CAP)

        # Re-read every turn, and after `observe`, so the constraints this turn
        # supplied count towards it. Assigning the track once at turn 1 left a
        # shopper who opened vaguely and then named three requirements retrieving
        # from an `open` two-shelf pool forever. See DialogState.read_intent.
        dialog.intent = dialog.read_intent(message, clauses)

        # -- route -----------------------------------------------------------
        intent = BROWSING if strategy == BROADEN else dialog.intent
        # W4: a genuinely pre-retrieval cutoff. Breadth is knowable from bucket
        # sizes alone, before any product is scored; when the predicted pool
        # exceeds the overload threshold and the shopper has filed no
        # discriminating constraint, score only a bounded representative sample
        # and go straight to the question. CommitPolicy.cutoff stays as the
        # post-retrieval confirmation below -- one predicts, one confirms.
        estimated = estimated_pool_size(self.catalog, self.semantic,
                                        dialog.category, intent)
        pre_cut = self.commit.pre_cutoff(estimated,
                                         has_constraints=bool(dialog.phrases()))
        pool, primary, hedge = route_detail(self.catalog, self.semantic,
                                            dialog.category, intent)
        if pre_cut:
            # A representative slate at bounded scoring cost. With no constraint
            # filed the full pass leads with the boosted primary shelf, so the
            # sample must too: the primary shelf's most popular members first,
            # then the rest by popularity. A raw popularity cut over the whole
            # pool mixes shelves and drops items the boost would have surfaced --
            # measured at -0.0044 before this ordering was restored.
            by_pop = sorted(pool, key=lambda pid: (-self.catalog.meta[pid]["pop"],
                                                   pid))
            lead = [pid for pid in by_pop if pid in primary]
            pool = (lead + [pid for pid in by_pop if pid not in primary])[
                :PRE_CUTOFF_SAMPLE]
            primary &= set(pool)
        if dialog.category:            self.memory.observe(state.signature, bucket=dialog.category)
        phrases = dialog.weighted_phrases()
        if not pre_cut and self.fuse_global and hedge:
            # Only when the shelf did not resolve outright. On every official
            # session it does, so this is unreachable there by construction -- it
            # is insurance against a private set whose category vocabulary is not
            # the catalog's. Appended, so the fusion can only add candidates.
            query: collections.Counter = collections.Counter()
            for text, weight in self.scorer._weighted(phrases):
                for term in tokens(text):
                    query[term] += weight
            pool = fuse_global(self.catalog, pool, query)
        # -- V6 shelf transform: pool-only additional non-exact shelf evidence.
        # The exact route never consults a transform (G4); the transform reads
        # the category phrase only (T22). When it fires, the complete baseline
        # pre-filter pool stays an ordered prefix: candidate products are
        # appended, nothing is removed or reordered, and no score or feature
        # bonus attaches to them (G8).
        exact = exact_bucket(self.catalog, dialog.category) is not None
        added_shelves: list[str] = []
        transform_lookups = 0
        transform_activations = 0
        baseline_prefix_digest = None
        baseline_prefix_size = None
        if not pre_cut and not exact and self.shelf_transform.mode != "off":
            transform_lookups = 1
            replacements = self.shelf_transform.transform(dialog.category or "")
            if replacements:
                fused = rrf_merge([_overlap_ranked(self.catalog, phrase,
                                                   limit=OVERLAP_LIMIT)
                                   for phrase in replacements])
                pool_set = set(pool)
                for name in fused:
                    members = self.catalog.buckets.get(name, ())
                    if not members or any(pid in pool_set for pid in members):
                        continue      # already represented in the baseline pool
                    added_shelves.append(name)
                    pool_set.update(members)
                    if len(added_shelves) == MAX_ADDED_SHELVES:
                        break
                if added_shelves:
                    transform_activations = 1
                    if self.trace_route:
                        baseline_prefix_size = len(pool)
                        baseline_prefix_digest = hashlib.sha256(
                            json.dumps(pool, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                    for name in added_shelves:
                        pool.extend(self.catalog.buckets[name])

        overloaded = pre_cut or self.commit.cutoff(len(pool))
        if self.trace_route:
            # The ordered pre-filter pool is the baseline a candidate transform
            # must provably extend rather than reorder (V6 T08). Recorded as a
            # digest, not a list: order-bearing evidence without retaining ids.
            prefilter_size = len(pool)
            prefilter_digest = hashlib.sha256(
                json.dumps(pool, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        pool = state.rejection.filter(pool)
        if self.trace_pool:
            state.pool = frozenset(pool)
        if self.trace_pool_size:
            state.pool_size = len(pool)

        # -- score -----------------------------------------------------------
        # Computed once and shared with the reranker's feature context below, so
        # what a model is served is literally what the weighted sum was scored on.
        if self.trace_route:
            state.route_trace = {
                "exact": exact,
                "hedged": bool(hedge),
                "fallback": not exact and not hedge,
                "prefilter_pool_size": prefilter_size,
                "prefilter_pool_sha256": prefilter_digest,
                "scored_pool_size": len(pool),
                "estimated_pool_size": estimated,
                "pre_cutoff": pre_cut,
                "transform_lookups": transform_lookups,
                "transform_activations": transform_activations,
                "added_shelves": added_shelves,
                "baseline_prefix_sha256": baseline_prefix_digest,
                "baseline_prefix_size": baseline_prefix_size,
            }
        refusals = dict(self._refusals(dialog.rejected(), pool))
        facets, budget = dialog.facets(), dialog.budget()
        ranked = self.scorer.rank(pool, phrases, state.rejection.penalised,
                                  self._boost(state, pool, primary, refusals, exact),
                                  budget, facets)
        # Personalization decides between candidates the conversation has already
        # tied, and never more than that. See src/profile.py for what happens when
        # it is allowed to do more.
        ranked = break_ties(ranked, state.profile, self.catalog)

        if self.reranker is not None or self.trace_features:
            context = Context(
                self.catalog, self.scorer, phrases, facets, budget, primary, exact,
                state.rejection.penalised, refusals,
                constraints=len({slot.attribute for slot in dialog.slots
                                 if not slot.superseded and slot.polarity > 0}))
            if self.trace_features:
                state.context, state.ranked = context, list(ranked[:RERANK_DEPTH])
            if self.reranker is not None:
                ranked = self.reranker.apply(ranked, context)

        # W2: the optional LLM semantic ranking stage. Off unless BOTH env vars
        # are set, and imported only inside this branch, so the default path
        # never touches the module (C1/C4). The linear order above is the
        # declared offline fallback and the default; the ordering was measured
        # and declined (D11), which is why this stage is opt-in.
        if os.environ.get("TECHJAM_LLM_RERANK") == "1" \
                and os.environ.get("AIAND_API_KEY"):
            from . import llm_rank
            ranked = llm_rank.apply(
                ranked, category=dialog.category,
                phrases=[text for text, _ in phrases], catalog=self.catalog,
                turn=turn)

        # The untruncated head, recorded before the dialogue policy decides how
        # much of it to show. Never read back by `respond`. See internal_ranking.
        state.internal = [pid for _, pid in ranked[:TOP_K]]

        # V4-0 names the existing confidence channel without changing it.  This is
        # exactly the post-rerank top-two margin CommitPolicy.width used to read for
        # itself.  Keeping it explicit prevents a future residual-score rescaling
        # from silently changing presentation policy; migrating to a different
        # statistic is V4-1Q and is deliberately not part of this behavior-parity
        # patch.
        legacy_confidence = self.commit.legacy_confidence(ranked)

        # -- elicit ----------------------------------------------------------
        # Decided before the slate, because whether there is a question left to
        # ask is what determines whether holding candidates back can still pay.
        attribute = self._ask(ranked, state, pool, hedge)

        # -- commit ----------------------------------------------------------
        width = min(self.commit.width(
            turn, ranked, intent == BROWSING, confidence=legacy_confidence
        ), top_k)
        if strategy == COVER:
            width = top_k                    # probing has stopped working; cover
        elif attribute is None:
            # No question left means no later turn can sharpen this ranking, so
            # there is nothing to gain by showing less than we have. Withholding
            # is only ever justified by a question that might improve the order.
            width = top_k
        elif overloaded and turn < self.commit.widen_turn:
            # Pillar II's over-generality cutoff: ten items drawn from thousands is
            # a shrug, not an answer. Show a sample of what we have and ask.
            # Narrow here is not the withholding the width policy exists to avoid --
            # the pool is too general to rank meaningfully, so there is nothing to
            # withhold. But a single item is a strange answer to any question, so
            # this floors at the confident width rather than at one.
            width = self.commit.base_width
        if intent == BROWSING and width > 1:
            picks = diversify(self.catalog, ranked[:width * 6], width, primary)
        else:
            picks = [pid for _, pid in ranked[:width]]
        state.rejection.record(picks)

        return {
            "message": self._say(attribute, state, overloaded),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": pid} for pid in picks],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _named_shelf(self, clauses) -> tuple[str, int] | None:
        """The first shelf named outright, together with its clause index.

        A correction usually leads with the retraction and names the new category
        after it, so scanning only the opening clause finds "Actually, forget that"
        and misses the part that matters. Returning the index also lets observation
        strip that routed shelf while retaining facets attached to the same clause.
        """
        from .state import CORRECTION_LEAD
        for index, clause in enumerate(clauses):
            cleaned = normalise(CORRECTION_LEAD.sub("", clause).strip(" ,:;.-"))
            if not cleaned:
                continue
            base = category_key([cleaned]) or cleaned
            # "Accessories Belts, leather" names a shelf and then qualifies it, so
            # try the whole phrase and then each comma-delimited prefix.
            parts = base.split(",")
            for size in range(len(parts), 0, -1):
                candidate = ",".join(parts[:size]).strip()
                if candidate and exact_bucket(self.catalog, candidate) is not None:
                    return candidate, index
        return None

    def _is_new_category(self, current: str | None, named: str, message: str) -> bool:
        """Does this turn name a different shelf from the one we are searching?

        Deliberately hard to trigger. An earlier version accepted any phrase the
        semantic route could resolve, which is every phrase -- `Semantic.resolve`
        normalises so its best hit always scores 1.0, so the confidence test never
        rejected anything and ordinary constraints ("Material:alloy") were read as
        category changes. That threw away the conversation on most turns and cost
        0.21: 0.960 down to 0.749.

        So the bar is a shelf name the customer actually said. Either the phrase
        names a bucket outright, or they signalled a correction *and* the phrase
        shares a word with the shelf it resolves to. A constraint can never
        masquerade as a category.
        """
        if not named or named == current:
            return False
        here = exact_bucket(self.catalog, current or "")
        resolved = exact_bucket(self.catalog, named)
        if resolved is None:
            # A shelf is a kind of product, not a property of one. "cotton" is a
            # material and "navy" is a colour, however confidently either resolves
            # to some bucket -- reading them as categories is what took intent
            # override from HR 1.000 to 0.633.
            if not is_correction(message) or classify(named) != "feature":
                return False
            hits = self.semantic.resolve(named, limit=1)
            if not hits:
                return False
            candidate = hits[0][0]
            spoken = {stem(word) for word in tokens(named)}
            if not (spoken & {stem(word) for word in tokens(candidate)}):
                return False
            resolved = candidate
        return resolved != here

    def _boost(self, state: Session, pool, primary, refusals=None,
               exact: bool | None = None) -> dict | None:
        """Per-item additive bonuses: the named shelf, the profile, refusals."""
        boost: dict[str, float] = {}

        if primary and len(primary) < len(pool):
            # Full strength only when the shelf name matched outright; when it was
            # inferred, "primary" is a guess and must not dominate.
            if exact is None:
                exact = exact_bucket(self.catalog, state.dialog.category) is not None
            strength = PRIMARY_BOOST if exact else PRIMARY_BOOST * UNSURE_BOOST_SCALE
            for pid in primary:
                boost[pid] = strength

        if refusals is None:
            refusals = dict(self._refusals(state.dialog.rejected(), pool))
        for pid, penalty in refusals.items():
            boost[pid] = boost.get(pid, 0.0) - penalty

        return boost or None

    def _refusals(self, rejected, pool):
        """Per-item penalty for values the customer refused.

        A refusal that names a material or a colour is resolved against the
        product's facets rather than against its text. The difference is not
        academic: "nothing in leather" used to scan for the substring "leather"
        anywhere in the blob, so a cotton canvas bag whose description says
        "pairs well with leather boots" was penalised exactly as hard as a leather
        bag, and a leather bag that happened not to use the word escaped.

        Three tiers, and the middle one is the point:

            states it     the product's title, features or details say leather
            mentions it   only its prose does -- weaker evidence, weaker penalty
            neither       no penalty at all, which the substring scan could not do

        A refusal the catalog cannot resolve to a facet ("no zippers") keeps the
        text scan, because for those the text is the only evidence there is.
        """
        wanted: dict[str, set[str]] = {"material": set(), "color": set()}
        lexical: set[str] = set()
        for text in rejected:
            lowered = text.lower()
            resolved = False
            for attribute, pattern in (("material", MATERIAL_RE), ("color", COLOR_RE)):
                for match in pattern.finditer(lowered):
                    wanted[attribute].add(match.group(1).lower())
                    resolved = True
            if not resolved:
                # Tokenised the same way the catalog was. A raw split leaves
                # punctuation attached, so a refusal ending a sentence produced
                # "polyester." -- which matches no index term and, under the old
                # substring test, matched only products whose text happened to
                # punctuate it the same way.
                lexical.update(word for word in tokens(lowered) if len(word) > 3)
        if not lexical and not any(wanted.values()):
            return

        catalog = self.catalog
        for pid in pool:
            penalty = 0.0
            for attribute, values in wanted.items():
                if not values:
                    continue
                if values & set(catalog.facet_values(pid, attribute)):
                    penalty = NEGATIVE_PENALTY
                    break
                if values & set(catalog.facet_values(pid, attribute, loose=True)):
                    penalty = max(penalty, NEGATIVE_PENALTY * MENTION_SCALE)
            if lexical and penalty < NEGATIVE_PENALTY:
                # Whole tokens, not substrings. `corpus` is one normalised string,
                # so `"silk" in blob` also matches "silky" -- 802 products against
                # 478 that actually contain the word -- and "right" matches
                # "bright" and "upright", 3,397 against 2,135. A refusal that
                # cannot be resolved to a facet is the case where the text is the
                # only evidence there is, which is exactly why that evidence has
                # to be read at the word boundary rather than the letter.
                if not lexical.isdisjoint(catalog.tf[pid]):
                    penalty = NEGATIVE_PENALTY
            if penalty:
                yield pid, penalty

    def _ask(self, ranked, state: Session, pool=(), hedge=()) -> str | None:
        if self.ask_mode == "none":
            return None
        if self.ask_mode.startswith("fixed:"):
            attribute = self.ask_mode.split(":", 1)[1]
            return attribute if attribute in ALLOWED else "feature"
        # Never ask about something the customer has already told us.
        asked = state.asked | state.dialog.answered()
        extra = None
        offer: list[str] = []
        if shelf_ambiguous(self.catalog, hedge, "category" in asked):
            # Which shelf they meant is not inferable -- see the finding in
            # src/clarify.py -- so it competes as a question like any other. Its
            # reduction is the share of the pool the shelves they did *not* mean
            # account for, which is exactly what the answer removes.
            offer = shelf_options(self.catalog, hedge)
            kept = len(self.catalog.buckets.get(offer[0], ()))
            extra = {"category": max(0.0, 1.0 - kept / max(len(pool), 1))}
        attribute = choose(self.catalog, ranked, asked, self.answers, extra,
                           self.scorer if self.ask_mode == "counterfactual" else None)
        # An offer the customer did not take expires with the turn that made it.
        # Left standing it would match a later, unrelated sentence against shelves
        # nobody has mentioned since -- and the customer has, in effect, declined.
        state.offered = []
        if attribute:
            state.asked.add(attribute)
            state.pending = attribute
            if attribute == "category":
                state.offered = offer
        return attribute

    @staticmethod
    def _say(attribute: str | None, state: Session, overloaded: bool = False) -> str:
        if not attribute:
            return "Here are the closest matches I found."
        if attribute == "category" and state.offered:
            # A closed question naming the shelves actually in contention. See
            # src/clarify.py for why an open one ("what category?") is worse.
            return shelf_phrasing(state.offered)
        if overloaded:
            opener = "That could be a lot of things"
        elif state.orchestrator.stalled:
            opener = "Let me try a different angle"
        elif state.intent == BROWSING:
            opener = "To narrow this down"
        else:
            opener = "So I can be precise"
        phrasing = {
            "material": "what material are you after?",
            "color": "any color you have in mind?",
            "brand": "is there a brand you prefer?",
            "budget": "roughly what budget are you working with?",
            "size": "what size do you need?",
            "style": "what style are you going for?",
            "use_case": "what will you mainly use it for?",
            "feature": "is there a feature that matters most?",
        }.get(attribute, "anything else that matters?")
        return f"{opener} -- {phrasing}"

    def _record_failure(self) -> None:
        """A swallowed exception must still leave evidence.

        A guard that hides a total failure reports 0.000 with a clean exit, which is
        indistinguishable from "the private set was harder". Count every one, and
        print the first so a CI log carries the traceback.
        """
        self.failures += 1
        if self.failures == 1:
            print("src.agent: first swallowed turn failure:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    @staticmethod
    def _empty() -> dict:
        return {"message": "Let me try that again -- could you tell me more?",
                "ask_attribute": "feature", "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
