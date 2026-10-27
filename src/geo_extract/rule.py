"""Deterministic extractor: no network, no model, fully explainable."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from .base import Extractor, build_result
from .matching import Hit, match_name, resolve_overlaps
from .models import Candidate, Exclusion, ExtractionResult, Mention, Via
from .normalize import find_domains, location_words, registered_domain, tokenize
from .structure import _CITATION_ONLY, assign_positions, find_items, negative_reason

_CITATION_LINE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)]|[-*])?\s*(?:\[[^\]]*\]\()?https?://", re.I)
_SOURCES_HEADER = re.compile(r"^\s*(?:\*\*)?(?:sources?|references|citations)\b", re.I | re.M)


def _line_of(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start : end if end != -1 else len(text)]


class RuleExtractor(Extractor):
    """Normalization + fuzzy matching + answer structure. See README for the design."""

    name = "rule"

    def extract(
        self,
        answer: str,
        candidates: Sequence[Candidate],
        target: str | None = None,
        location: str | None = None,
    ) -> ExtractionResult:
        loc = location_words(location)
        domains = find_domains(answer)
        url_spans = [(s, e) for _, s, e in domains]
        tokens = [t for t in tokenize(answer) if not any(s <= t.start < e for s, e in url_spans)]

        hits: list[Hit] = []
        for cand in candidates:
            hits += match_name(tokens, cand.name, cand.name, "name", loc)
            for alias in cand.aliases:
                hits += match_name(tokens, cand.name, alias, "alias", loc)
        kept = resolve_overlaps(hits)

        sources_at = _SOURCES_HEADER.search(answer)
        sources_start = sources_at.start() if sources_at else len(answer)
        by_domain = {registered_domain(c.domain): c.name for c in candidates if c.domain}

        spans: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        best: dict[str, Hit] = {}
        excluded: dict[str, Exclusion] = {}
        all_spans = [(h.start, h.end) for h in kept]

        for h in kept:
            others = [s for s in all_spans if s != (h.start, h.end)]
            reason = negative_reason(answer, h.start, h.end, others)
            if reason:
                excluded.setdefault(
                    h.candidate, Exclusion(h.candidate, reason, answer[h.start : h.end])
                )
                continue
            spans[h.candidate].append((h.start, h.end, h.via))
            if h.candidate not in best or h.score > best[h.candidate].score:
                best[h.candidate] = h

        for dom, start, end in domains:
            cand_name = by_domain.get(dom)
            if cand_name is None:
                continue
            line = _line_of(answer, start)
            is_citation = (
                start >= sources_start
                or bool(_CITATION_LINE.match(line))
                or bool(_CITATION_ONLY.match(line.strip()))
            )
            dom_via = "citation" if is_citation else "domain"
            spans[cand_name].append((start, end, dom_via))
            if cand_name not in best:
                best[cand_name] = Hit(cand_name, start, end, 0.9, dom_via, 1)

        # A candidate that was excluded but also recommended elsewhere counts as named.
        for cand_name in list(excluded):
            if cand_name in spans:
                del excluded[cand_name]

        # Businesses named in item heads that are not in the candidate list.
        unknown: dict[str, tuple[int, int]] = {}
        for it in find_items(answer):
            if any(it.head_start <= s < it.head_end for s, _ in all_spans):
                continue
            if any(it.head_start <= s < it.head_end for _, s, _ in domains):
                continue
            head = answer[it.head_start : it.head_end].strip(" *_#[]()")
            words = head.split()
            if 0 < len(words) <= 7 and words[0][:1].isupper():
                unknown[f"?{it.slot}:{head}"] = (it.head_start, it.head_end)

        keyed = dict(spans)
        for k, (s, e) in unknown.items():
            keyed[k] = [(s, e, "name")]
        positions = assign_positions(answer, keyed)

        mentions: list[Mention] = []
        for cand_name, hit in best.items():
            vias = {v for _, _, v in spans[cand_name]}
            via: Via
            if vias == {"citation"}:
                via = "citation"
            elif hit.via in ("alias", "domain"):
                via = "alias" if hit.via == "alias" else "domain"
            else:
                via = "name"
            mentions.append(
                Mention(
                    name=answer[hit.start : hit.end],
                    matched_candidate=cand_name,
                    position=positions.get(cand_name),
                    confidence=round(hit.score, 3),
                    evidence_span=(hit.start, hit.end),
                    evidence=answer[hit.start : hit.end],
                    via=via,
                )
            )
        for k, (s, e) in unknown.items():
            label = k.split(":", 1)[1]
            mentions.append(
                Mention(label, None, positions.get(k), 0.5, (s, e), answer[s:e], "name")
            )
        return build_result(mentions, candidates, target, list(excluded.values()))
