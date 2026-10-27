"""The interface every extractor implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .models import Candidate, Exclusion, ExtractionResult, Mention
from .normalize import name_tokens


class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(
        self,
        answer: str,
        candidates: Sequence[Candidate],
        target: str | None = None,
        location: str | None = None,
    ) -> ExtractionResult:
        """Find which candidates `answer` names, their positions, and whether `target` is one."""


def resolve_target(target: str | None, candidates: Sequence[Candidate]) -> str | None:
    """Map a free-text target onto the candidate it refers to."""
    if target is None:
        return None
    want = name_tokens(target)
    for c in candidates:
        if name_tokens(c.name) == want or any(name_tokens(a) == want for a in c.aliases):
            return c.name
    return target


def build_result(
    mentions: list[Mention],
    candidates: Sequence[Candidate],
    target: str | None,
    excluded: list[Exclusion],
) -> ExtractionResult:
    mentions = sorted(
        mentions,
        key=lambda m: (m.position is None, m.position or 0, m.evidence_span[0]),
    )
    tgt = resolve_target(target, candidates)
    hit = next((m for m in mentions if tgt is not None and m.matched_candidate == tgt), None)
    return ExtractionResult(
        mentions=mentions,
        target=tgt,
        target_named=hit is not None,
        target_position=hit.position if hit else None,
        excluded=excluded,
    )
