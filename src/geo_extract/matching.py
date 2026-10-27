"""Fuzzy matching of candidate names against token windows of an answer."""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .normalize import GENERIC, Token, distinctive_tokens, name_tokens

CONNECTORS = frozenset({"and", "the", "of"})
FULL_THRESHOLD = 90.0
CORE_THRESHOLD = 92.0
MIN_CORE_CHARS = 4


@dataclass(frozen=True)
class Hit:
    candidate: str
    start: int
    end: int
    score: float  # 0 to 1
    via: str  # "name" | "alias" | "core"
    n_tokens: int


def _similarity(window: list[str], target: list[str]) -> float:
    """Best of spaced and compact comparison, so "EcoCool" matches "Eco Cool"."""
    spaced = fuzz.ratio(" ".join(window), " ".join(target))
    compact = fuzz.ratio("".join(window), "".join(target))
    return max(spaced, compact)


def _windows(tokens: list[Token], size: int) -> list[tuple[int, int]]:
    return [(i, i + size) for i in range(0, max(0, len(tokens) - size + 1))]


def match_name(
    tokens: list[Token],
    candidate: str,
    variant: str,
    via: str,
    location_generic: frozenset[str],
) -> list[Hit]:
    """All places in the answer where `variant` (the name or an alias) appears.

    Two strategies:
    * full: the whole name (minus legal suffixes) fuzzily matches a window.
    * core: only the distinctive words match ("Polar Bear" for "Polar Bear HVAC"). Core
      hits need a capitalized first word, so "arctic temperatures" never hits
      "Arctic Edge Cooling" mid-sentence in lowercase prose, and they are scored lower.
    """
    hits: list[Hit] = []
    full = name_tokens(variant)
    if not full:
        return hits
    norms = [t.norm for t in tokens]
    for size in range(max(1, len(full) - 1), len(full) + 2):
        for i, j in _windows(tokens, size):
            window = norms[i:j]
            # strip legal suffixes inside the window so "Ironclad Roofing Co." == "Ironclad Roofing"
            trimmed = window[:]
            while len(trimmed) > 1 and trimmed[-1] in {"llc", "inc", "co", "company", "corp"}:
                trimmed.pop()
            score = _similarity(trimmed, full)
            if score >= FULL_THRESHOLD and window[0][:1] == full[0][:1]:
                hits.append(
                    Hit(candidate, tokens[i].start, tokens[j - 1].end, score / 100, via, size)
                )

    # truncated: the answer drops trailing trade words ("Tamarac Termite & Pest" for
    # "Tamarac Termite and Pest Control"). The kept part must still identify the business:
    # either it has a distinctive word, or it is 3+ words all written as a proper noun.
    # That stops "Bel Air" in "plumbers in Bel Air" from matching "Bel Air Plumbing".
    weak = GENERIC | location_generic
    for k in range(max(2, len(full) - 2), len(full)):
        tail = full[k:]
        if not all(t in GENERIC for t in tail):
            continue
        prefix = full[:k]
        has_distinctive = any(t not in weak for t in prefix)
        for i, j in _windows(tokens, k):
            if norms[i:j] != prefix or not tokens[i].capitalized:
                continue
            proper = all(t.capitalized or t.norm in CONNECTORS for t in tokens[i:j])
            if has_distinctive or (k >= 3 and proper):
                hits.append(Hit(candidate, tokens[i].start, tokens[j - 1].end, 0.8, via, k))

    core = distinctive_tokens(variant, location_generic)
    if core and len("".join(core)) >= MIN_CORE_CHARS and core != full:
        for size in range(max(1, len(core) - 1), len(core) + 2):
            for i, j in _windows(tokens, size):
                if not tokens[i].capitalized:
                    continue
                window = [w for w in norms[i:j] if w not in CONNECTORS]
                if not window or window[0][:1] != core[0][:1]:
                    continue
                score = fuzz.ratio("".join(window), "".join(core))
                if score >= CORE_THRESHOLD:
                    hits.append(
                        Hit(
                            candidate,
                            tokens[i].start,
                            tokens[j - 1].end,
                            0.6 + 0.2 * score / 100,
                            "core",
                            size,
                        )
                    )
    # one hit per span: several strategies can find the same text
    best: dict[tuple[int, int], Hit] = {}
    for h in hits:
        key = (h.start, h.end)
        if key not in best or h.score > best[key].score:
            best[key] = h
    return sorted(best.values(), key=lambda h: (h.start, h.end))


def resolve_overlaps(hits: list[Hit]) -> list[Hit]:
    """Keep one hit per stretch of text.

    When two candidates claim overlapping text, the higher score wins, and on a tie the
    hit covering more tokens wins. That is what separates "Summit Roofing" from
    "Summit Roofing & Gutters" when the answer says the longer name.
    """
    ordered = sorted(hits, key=lambda h: (-round(h.score, 3), -(h.end - h.start)))
    kept: list[Hit] = []
    for h in ordered:
        if any(h.start < k.end and k.start < h.end for k in kept):
            continue
        kept.append(h)
    return sorted(kept, key=lambda h: h.start)
