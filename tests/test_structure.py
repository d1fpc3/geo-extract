from geo_extract.structure import assign_positions, find_items, negative_reason

LIST = """Intro line.

1. **Alpha HVAC** - great
   - sub detail about Beta
2. **Beta HVAC** - also good

Sources:
1. https://alpha.com
"""


def test_items_skip_sub_bullets_and_citations() -> None:
    items = find_items(LIST)
    assert [i.slot for i in items] == [1, 2]
    assert LIST[items[0].head_start : items[0].head_end] == "Alpha HVAC"


def test_table_rows_are_items_and_header_is_not() -> None:
    table = "| # | Name |\n|---|---|\n| 1 | Alpha |\n| 2 | Beta |\n"
    items = find_items(table)
    assert [table[i.head_start : i.head_end] for i in items] == ["Alpha", "Beta"]


def test_numbered_markdown_headings_are_items() -> None:
    text = "### 1. Alpha\nDescription.\n\n### 2. Beta\nMore."
    assert len(find_items(text)) == 2


def test_prose_mentions_interleave_with_items() -> None:
    text = "Gamma is popular.\n\n1. Alpha - x\n2. Beta - y\n"
    a = text.index("Alpha")
    b = text.index("Beta")
    pos = assign_positions(
        text,
        {"Gamma": [(0, 5, "name")], "Alpha": [(a, a + 5, "name")], "Beta": [(b, b + 4, "name")]},
    )
    assert pos == {"Gamma": 1, "Alpha": 2, "Beta": 3}


def test_head_mention_beats_incidental_mention() -> None:
    text = "1. Alpha - founded by an ex-Beta tech\n2. Beta - big\n"
    b1 = text.index("Beta")
    b2 = text.index("Beta", b1 + 1)
    pos = assign_positions(text, {"Beta": [(b1, b1 + 4, "name"), (b2, b2 + 4, "name")]})
    assert pos == {"Beta": 2}


def test_citation_only_has_no_position() -> None:
    assert assign_positions("x", {"A": [(0, 1, "citation")]}) == {"A": None}


def test_negative_cue_scoped_between_mentions() -> None:
    text = "Alpha closed down last year, but Beta is great."
    a, b = (0, 5), (text.index("Beta"), text.index("Beta") + 4)
    assert negative_reason(text, *a, [b]) is not None
    assert negative_reason(text, *b, [a]) is None


def test_abbreviation_does_not_end_sentence() -> None:
    text = "Ironclad Roofing Co. is good. Avoid Zeta."
    assert negative_reason(text, 0, 19, []) is None
