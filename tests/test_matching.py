from geo_extract.matching import Hit, match_name, resolve_overlaps
from geo_extract.normalize import location_words, tokenize

NOLOC: frozenset[str] = frozenset()


def hits_for(text: str, name: str, loc: frozenset[str] = NOLOC) -> list[str]:
    return [text[h.start : h.end] for h in match_name(tokenize(text), name, name, "name", loc)]


def test_exact_and_suffix_variants() -> None:
    assert "Ironclad Roofing Co" in hits_for(
        "Try Ironclad Roofing Co. today", "Ironclad Roofing Company"
    )


def test_compact_spelling_variant() -> None:
    assert hits_for("Clear Flow Plumbing is great", "ClearFlow Plumbing LLC")


def test_core_match_on_distinctive_words() -> None:
    got = hits_for("If Arctic Edge is booked", "Arctic Edge Cooling")
    assert got == ["Arctic Edge"]  # one hit per span even when two strategies agree


def test_core_match_needs_capitalization() -> None:
    assert hits_for("arctic edge of the storm", "Arctic Edge Cooling") == []


def test_location_words_cannot_match_alone() -> None:
    loc = location_words("Bel Air, MD")
    assert hits_for("Plumbers in Bel Air are busy", "Bel Air Plumbing", loc) == []


def test_truncated_trade_words() -> None:
    loc = location_words("Tamarac, FL")
    assert hits_for("Tamarac Termite & Pest is good", "Tamarac Termite and Pest Control", loc)


def test_longer_name_wins_overlap() -> None:
    text = "Summit Roofing & Gutters does gutters"
    toks = tokenize(text)
    hits = match_name(toks, "Summit Roofing", "Summit Roofing", "name", NOLOC)
    hits += match_name(toks, "Summit Roofing & Gutters", "Summit Roofing & Gutters", "name", NOLOC)
    kept = resolve_overlaps(hits)
    assert [h.candidate for h in kept] == ["Summit Roofing & Gutters"]


def test_resolve_overlaps_prefers_score() -> None:
    a = Hit("A", 0, 10, 0.9, "name", 2)
    b = Hit("B", 5, 12, 0.95, "name", 2)
    assert [h.candidate for h in resolve_overlaps([a, b])] == ["B"]
