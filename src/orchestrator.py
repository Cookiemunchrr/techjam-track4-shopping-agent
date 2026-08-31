"""Runtime strategy switching.

Pillar III asks the agent to "achieve runtime workflow re-orchestration and strategy
alignment, ensuring the agent iteratively refines its own guidance logic".

The honest minimum version of that is: notice when the conversation has stopped
producing information, and stop doing the thing that is not working. Three states,
driven entirely by whether the customer's answers are carrying anything.

    default    ask a narrow question, commit narrow
    broaden    the questions are not landing, so widen the pool and change tack
    cover      stop probing, fill the slate

D8 is the test that matters more than the feature: a healthy session must come out
of this untouched. Adaptive machinery that degrades the conversations which already
work is worse than no adaptive machinery.
"""
from __future__ import annotations

DEFAULT = "default"
BROADEN = "broaden"
COVER = "cover"

BROADEN_AFTER = 2      # consecutive turns that taught us nothing
COVER_AFTER = 4


class Orchestrator:
    """Per-session stall detector and strategy selector."""

    __slots__ = ("dead_turns", "worst")

    def __init__(self) -> None:
        self.dead_turns = 0
        self.worst = 0        # high-water mark, for reporting only

    def observe(self, informative: bool) -> None:
        if informative:
            self.dead_turns = 0
        else:
            self.dead_turns += 1
            self.worst = max(self.worst, self.dead_turns)

    def strategy(self) -> str:
        if self.dead_turns >= COVER_AFTER:
            return COVER
        if self.dead_turns >= BROADEN_AFTER:
            return BROADEN
        return DEFAULT

    @property
    def stalled(self) -> bool:
        return self.strategy() != DEFAULT
