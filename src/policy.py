"""How many candidates to commit, and how to treat the ones already shown.

The scoring function is 0.50*HitRate@10 + 0.30*MRR + 0.20*clip((11-MTTC)/10, 0, 1).
One extra turn costs 0.02; lifting a hit from rank 5 to rank 1 is worth 0.24.
Rank is worth about twelve turns of delay, so confidence should gate width.
"""
from __future__ import annotations

from dataclasses import dataclass

TOP_K = 10
MAX_TURNS = 10


@dataclass(frozen=True)
class CommitPolicy:
    """Decides how many recommendations to return on a given turn.

    Breadth comes from uncertainty. A confident ranking is a verdict and deserves a
    short answer; a flat one is a shrug and deserves options; browsing wants a space
    rather than a winner.

    This replaces a flat `base_width = 1`, which committed a single item on 85% of
    turns. That policy scored better -- the official session ends the moment the
    target appears in the *shown* slate, so withholding suppresses low-rank exposure
    and lifts MRR -- and tests/test_metrics_integrity.py has always recorded what it
    was worth: headline MRR 0.954 against 0.656 for the same retrieval at width ten.
    tools/shadow.py makes the same point continuously, by scoring the untruncated
    internal ranking: across widths 1 to 10 the official score moves 0.031 and the
    artifact-free score moves 0.004. A benefit that vanishes when one line of the
    simulator changes is not a product decision, so it is not the default here.

    The cost is real and is stated in the README rather than hidden: 0.048 of
    TechnicalScore. `P_PROBE=1` narrows the whole ladder to 1/2/4/7 and recovers
    about a quarter of that (0.90784); it does not restore the flat width of one,
    which needs the policy itself reverted.
    """
    base_width: int = 2        # width when one candidate clearly leads
    widen_turn: int = 8        # from here on, stop probing and cover
    margin_floor: float = 0.10 # below this the top candidate is not distinctive
    unsure_width: int = 2      # retained for the env contract; see `width`
    overload: int = 2000       # pool size above which we ask before we recommend
    high_margin: float = 0.60  # a leader this far clear is a verdict
    mid_margin: float = 0.30
    browse_extra: int = 4      # exploration is shown a space, not a winner

    @classmethod
    def from_env(cls, env) -> "CommitPolicy":
        def i(name, default):
            try:
                return int(env.get(name, default))
            except (TypeError, ValueError):
                return default
        def f(name, default):
            try:
                return float(env.get(name, default))
            except (TypeError, ValueError):
                return default
        return cls(i("P_PROBE", 2), i("P_WIDEN", 8), f("P_MARGIN", 0.10), i("P_UNSURE", 2),
                   i("P_OVERLOAD", 2000))

    def cutoff(self, pool_size: int) -> bool:
        """Is the candidate set too general to recommend from confidently?

        Pillar II asks for an "immediate retrieval cutoff" on candidate pool
        overload. Ten items drawn from thousands is not an answer, it is a shrug;
        the useful move is to ask first. Never fires on a pool already narrow
        enough to rank meaningfully.

        This is the post-retrieval safety net. The pre-retrieval half is
        `pre_cutoff`; one predicts, one confirms.
        """
        return pool_size > self.overload

    def pre_cutoff(self, estimated_size: int, has_constraints: bool) -> bool:
        """W4: predict overload before the scoring pass, from bucket sizes alone.

        Fires only when the predicted pool exceeds the overload threshold AND the
        shopper has filed no discriminating constraint yet. A stated constraint is
        exactly what the full scoring pass can exploit, so its presence keeps the
        pass; without one, scoring thousands of candidates to show a shrug is the
        waste Pillar II's "immediate retrieval cutoff" names.
        """
        return estimated_size > self.overload and not has_constraints

    @staticmethod
    def legacy_confidence(ranked) -> float:
        """The confidence statistic used by the shipped pre-V4 policy.

        It is intentionally named rather than presented as calibrated probability:
        the value is only the top-two margin of the final numeric score list.  V4-0
        keeps that exact behavior while making the channel explicit, so a future
        ranker cannot change presentation width merely because nobody noticed that
        its residual has a different scale.

        Degenerate rankings never consult confidence in :meth:`width`; returning
        zero here gives callers a finite, explicit value without changing that path.
        """
        if len(ranked) < 2:
            return 0.0
        return ranked[0][0] - ranked[1][0]

    def width(self, turn: int, ranked, browsing: bool = False, *,
              confidence: float | None = None) -> int:
        """Commit narrow when the leader is clear, wider when it is not.

        Every tier is expressed as `base_width + extra`, so the single knob still
        moves the whole ladder and `P_PROBE=1` still means "as narrow as it gets".

        ``confidence=None`` is the backwards-compatible interface: derive the same
        post-rerank top-two margin this method read before V4.  The agent passes that
        value explicitly.  Accepting an explicit value is plumbing only; switching
        to a different confidence source is V4-1Q and requires its own measurement.
        """
        if turn >= self.widen_turn:
            return TOP_K
        if len(ranked) < 2:
            return max(self.base_width, 1)
        margin = self.legacy_confidence(ranked) if confidence is None else confidence
        if browsing:
            extra = self.browse_extra
        elif margin >= self.high_margin:
            extra = 0                        # a clear verdict
        elif margin >= self.mid_margin:
            extra = 1
        elif margin >= self.margin_floor:
            extra = 3
        else:
            extra = 6                        # nothing separates the head; cover it
        return max(1, min(self.base_width + extra, TOP_K))


class RejectionModel:
    """Tracks what has been shown.

    A shopper not picking an item is weak evidence, not proof -- so items are
    penalised rather than deleted, and a stated change of intent clears the slate
    entirely. (Hard deletion cost 0.101 on the v2 reference agent; on the shipped
    agent the same question, asked of slot erasure, measured -0.00275 -- see
    analysis/v8_w3_erasure.json. The 0.101 figure is a historical reference-agent
    measurement, not a current-agent one.)
    """

    HARD, SOFT, RESET = "hard", "soft", "reset"

    def __init__(self, mode: str = RESET) -> None:
        self.mode = mode if mode in (self.HARD, self.SOFT, self.RESET) else self.RESET
        self.shown: set[str] = set()
        self.last: set[str] = set()      # what we put in front of them most recently

    def record(self, product_ids) -> None:
        self.last = set(product_ids)
        self.shown.update(product_ids)

    def on_correction(self) -> None:
        """A change of intent retires most rejections, but not the newest one.

        Items shown under the old constraint were judged against the old
        constraint, so under a new one they deserve another look -- clearing is
        right. What is not right is bouncing the product they just turned down
        straight back to rank one, so the most recent slate stays penalised.
        """
        if self.mode == self.RESET:
            self.shown = set(self.last)

    def filter(self, candidates):
        """Only the hard mode removes candidates outright."""
        if self.mode == self.HARD:
            return [pid for pid in candidates if pid not in self.shown]
        return candidates

    @property
    def penalised(self) -> frozenset:
        return frozenset() if self.mode == self.HARD else frozenset(self.shown)
