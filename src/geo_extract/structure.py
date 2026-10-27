"""Answer structure: list items, ranking slots, sentences and negative context.

Position rule (used by every extractor and by the eval's gold labels):

* Walk the answer top to bottom. Every top-level list item is one slot (numbered lists,
  bullets, numbered markdown headings and table rows all count). Sub-bullets and indented
  detail lines belong to the item above them. Lines that are only a citation are skipped.
* A business's position is the slot of the first item whose *head* names it (the bold
  name, or the text before the first dash or colon).
* A business named only in prose, outside any item head, takes a slot of its own at its
  first mention, in reading order.
* A business named only inside another item's description takes that item's slot.
* A business that is only cited as a source URL has no position (None).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ITEM = re.compile(
    r"""^(?P<indent>[ \t]*)
        (?:\#{1,6}[ \t]*)?          # optional markdown heading
        (?:\*\*|__)?                # optional bold opener before the marker
        (?:(?P<num>\d{1,2})[.)]|[-*•‣◦+])
        [ \t]+(?P<body>.+)$""",
    re.VERBOSE,
)
_TABLE_ROW = re.compile(r"^\s*\|(?P<body>.+)\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_HEAD_SPLIT = re.compile(r"\s(?:-|\u2013|\u2014)\s|:\s|\s\(|,\s")
_CITATION_ONLY = re.compile(r"^[\s\[\]\d.():]*(?:https?://\S+|\S+\.\w{2,})\s*$")

NEGATIVE_CUES = re.compile(
    r"""\b(
        (?:permanently|temporarily)\s+closed
      | (?:has|have|had|was|is|recently)\s+(?:been\s+)?closed
      | closed\s+(?:its|their|down|for\s+good|in\s+\d{4}|last\s+year|permanently)
      | shut\s+down | out\s+of\s+business | no\s+longer\s+(?:operat\w*|in\s+business|open|servic\w*)
      | went\s+out\s+of
      | avoid | steer\s+clear | stay\s+away | be\s+cautious | would\s*n[o']?t\s+recommend
      | do\s+not\s+recommend | don'?t\s+recommend | not\s+recommended
      | complaints?\s+(?:about|against) | scam
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


_CLAUSE_BREAK = re.compile(
    r";\s|,\s*(?:but|while|whereas|though|although|yet|however)\b|\s(?:but|whereas)\s", re.I
)


@dataclass(frozen=True)
class Item:
    slot: int
    start: int  # offset of the item's first character
    end: int  # offset after the item's last line (including its sub-lines)
    head_start: int
    head_end: int


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _head_span(text: str, body_start: int, body_end: int) -> tuple[int, int]:
    body = text[body_start:body_end]
    bold = _BOLD.search(body)
    if bold and bold.start() <= 3:
        group = 1 if bold.group(1) is not None else 2
        return body_start + bold.start(group), body_start + bold.end(group)
    cut = _HEAD_SPLIT.search(body)
    end = cut.start() if cut else len(body)
    return body_start, body_start + min(end, 90)


def find_items(text: str) -> list[Item]:
    """Top-level ranking items in reading order, with running slot numbers."""
    items: list[Item] = []
    top_indent: int | None = None
    current: dict[str, int] | None = None
    table_header_seen = False

    def close(end: int) -> None:
        nonlocal current
        if current is not None:
            items.append(Item(len(items) + 1, current["start"], end, current["hs"], current["he"]))
            current = None

    for start, end in _line_spans(text):
        line = text[start:end]
        if not line.strip():
            continue
        m = _ITEM.match(line)
        t = _TABLE_ROW.match(line)
        if t and not _TABLE_SEP.match(line):
            close(start)
            if not table_header_seen:
                table_header_seen = True  # first row of a table is its header
                continue
            cells = [c for c in t.group("body").split("|")]
            # the name is the first cell that is not just a rank number
            offset = start + line.index("|") + 1
            for cell in cells:
                if cell.strip() and not cell.strip().strip("#.").isdigit():
                    cs = offset + len(cell) - len(cell.lstrip())
                    current = {"start": start, "hs": cs, "he": cs + len(cell.strip())}
                    break
                offset += len(cell) + 1
            close(end)
            continue
        if _TABLE_SEP.match(line) and "|" in line:
            continue
        table_header_seen = False if not line.lstrip().startswith("|") else table_header_seen
        if m:
            indent = len(m.group("indent").expandtabs(4))
            body_start = start + m.start("body")
            body = m.group("body")
            if _CITATION_ONLY.match(body):
                close(start)
                continue
            if top_indent is None or indent <= top_indent:
                close(start)
                top_indent = indent
                hs, he = _head_span(text, body_start, end)
                current = {"start": start, "hs": hs, "he": he}
            # deeper indent: a sub-bullet, it stays part of the current item
            continue
        indented = line[:1] in (" ", "\t")
        if current is not None and indented:
            continue  # detail line under an item
        if current is not None and not indented:
            # an unindented paragraph line right under a heading-style item is its description
            prev_heading = text[current["start"] : current["start"] + 1] == "#"
            if prev_heading:
                continue
        close(start)
        top_indent = None
    close(len(text))
    return items


def item_at(items: list[Item], offset: int) -> Item | None:
    for it in items:
        if it.start <= offset < it.end:
            return it
    return None


def in_head(item: Item, start: int, end: int) -> bool:
    return start >= item.head_start and start < item.head_end


def sentence_bounds(text: str, offset: int) -> tuple[int, int]:
    """Start and end of the sentence (or line) containing `offset`."""
    left = max(text.rfind(c, 0, offset) for c in ".!?\n")
    # don't split on abbreviations like "Co." or "Dr." directly before the offset
    while left > 0 and re.search(r"\b(?:Co|Inc|Dr|Mr|Ms|St|Jr|Elec|Corp|Ltd)$", text[:left]):
        left = max(text.rfind(c, 0, left) for c in ".!?\n")
    start = left + 1
    end = len(text)
    for m in re.finditer(r"[.!?](?=\s)|\n", text[offset:]):
        cand = offset + m.start()
        if re.search(r"\b(?:Co|Inc|Dr|Mr|Ms|St|Jr|Elec|Corp|Ltd)$", text[:cand]):
            continue
        end = cand
        break
    return start, end


def negative_reason(text: str, start: int, end: int, others: list[tuple[int, int]]) -> str | None:
    """A negative cue that applies to the mention at [start, end), or None.

    The cue must sit in the same clause of the same sentence, and be closer to this mention
    than to any neighbouring mention, so "X closed, but Y is great" only hits X.
    """
    s_start, s_end = sentence_bounds(text, start)
    for sep in _CLAUSE_BREAK.finditer(text, s_start, s_end):
        if sep.end() <= start:
            s_start = sep.end()
        elif sep.start() >= end:
            s_end = sep.start()
            break
    lo, hi = s_start, s_end
    prev_end, next_start = None, None
    for o_start, o_end in others:
        if lo < o_end <= start:
            lo, prev_end = o_end, o_end
        if end <= o_start < hi:
            hi, next_start = o_start, o_start
    for cue in NEGATIVE_CUES.finditer(text, lo, hi):
        if (
            cue.end() <= start
            and prev_end is not None
            and cue.start() - prev_end < start - cue.end()
        ):
            continue  # sits right after the previous business, so it describes that one
        if (
            cue.start() >= end
            and next_start is not None
            and next_start - cue.end() < cue.start() - end
        ):
            continue  # sits right before the next business
        return cue.group(0)
    return None


def assign_positions(
    text: str,
    mentions: dict[str, list[tuple[int, int, str]]],
) -> dict[str, int | None]:
    """Apply the position rule above.

    `mentions` maps a business key to its recommended mentions as (start, end, via), where
    via is "citation" for mentions that are only a cited URL.
    """
    items = find_items(text)
    head_slot: dict[str, int] = {}
    tail_slot: dict[str, int] = {}
    prose_first: dict[str, int] = {}
    for key, spans in mentions.items():
        for start, end, via in spans:
            if via == "citation":
                continue
            it = item_at(items, start)
            if it is None:
                prose_first[key] = min(prose_first.get(key, start), start)
            elif in_head(it, start, end):
                head_slot[key] = min(head_slot.get(key, it.slot), it.slot)
            else:
                tail_slot[key] = min(tail_slot.get(key, it.slot), it.slot)

    # prose-only businesses get their own slots, interleaved with items in reading order
    prose_only = {k: off for k, off in prose_first.items() if k not in head_slot}
    events: list[tuple[int, str, str | None]] = [(it.start, "item", None) for it in items]
    events += [(off, "prose", k) for k, off in prose_only.items()]
    events.sort(key=lambda e: (e[0], e[1] == "prose"))
    item_renumber: dict[int, int] = {}
    prose_slot: dict[str, int] = {}
    item_idx = 0
    for slot, (_, kind, ev_key) in enumerate(events, start=1):
        if kind == "item":
            item_idx += 1
            item_renumber[item_idx] = slot
        elif ev_key is not None:
            prose_slot[ev_key] = slot

    out: dict[str, int | None] = {}
    for key in mentions:
        if key in head_slot:
            out[key] = item_renumber[head_slot[key]]
        elif key in prose_slot:
            out[key] = prose_slot[key]
        elif key in tail_slot:
            out[key] = item_renumber[tail_slot[key]]
        else:
            out[key] = None
    return out
