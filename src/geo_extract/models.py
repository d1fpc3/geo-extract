"""Data types shared by every extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Via = Literal["name", "alias", "domain", "citation", "llm"]


@dataclass(frozen=True)
class Candidate:
    """A business we want to look for in an answer."""

    name: str
    domain: str | None = None
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_obj(cls, obj: str | dict[str, Any]) -> Candidate:
        if isinstance(obj, str):
            return cls(name=obj)
        return cls(
            name=obj["name"],
            domain=obj.get("domain"),
            aliases=tuple(obj.get("aliases", ())),
        )

    @classmethod
    def from_line(cls, line: str) -> Candidate:
        """Parse `Name | domain.com | alias one; alias two` (domain and aliases optional)."""
        parts = [p.strip() for p in line.split("|")]
        name = parts[0]
        domain = parts[1] if len(parts) > 1 and parts[1] else None
        aliases = (
            tuple(a.strip() for a in parts[2].split(";") if a.strip()) if len(parts) > 2 else ()
        )
        return cls(name=name, domain=domain, aliases=aliases)


@dataclass(frozen=True)
class Mention:
    """One business the answer named.

    `matched_candidate` is None for businesses that were named but are not in the
    candidate list (useful for discovering competitors you are not tracking yet).
    `position` is the business's rank in the answer (see `structure.py` for the rule),
    or None when it was only cited as a source.
    """

    name: str
    matched_candidate: str | None
    position: int | None
    confidence: float
    evidence_span: tuple[int, int]
    evidence: str
    via: Via


@dataclass(frozen=True)
class Exclusion:
    """A candidate that appears in the answer only in a negative context (closed, avoid)."""

    candidate: str
    reason: str
    evidence: str


@dataclass
class ExtractionResult:
    mentions: list[Mention]
    target: str | None
    target_named: bool
    target_position: int | None
    excluded: list[Exclusion] = field(default_factory=list)

    @property
    def named_candidates(self) -> dict[str, int | None]:
        """Candidate name to position, for every candidate the answer named."""
        return {
            m.matched_candidate: m.position
            for m in self.mentions
            if m.matched_candidate is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
