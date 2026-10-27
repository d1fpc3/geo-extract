"""LLM extractor: Claude reads the answer and reports mentions through a forced tool call.

The model only decides *which* candidates are recommended and quotes the exact text it
saw. Positions are then computed by the same deterministic rule the RuleExtractor uses,
so the two extractors can be compared on identical terms.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .base import Extractor, build_result
from .models import Candidate, Exclusion, ExtractionResult, Mention
from .structure import assign_positions

DEFAULT_MODEL = "claude-haiku-5-5"

SYSTEM = (
    "You extract business mentions from an AI assistant's answer to a local-business question. "
    "Report every candidate business the answer names, including spelling variants, "
    "abbreviations and URL-only citations. Mark a mention recommended=false when the answer "
    "names the business only to say it closed, or to warn the reader away from it. "
    "Quote the exact characters from the answer as evidence. Do not invent mentions."
)

TOOL: dict[str, Any] = {
    "name": "report_mentions",
    "description": "Report which candidate businesses the answer names.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mentions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {"type": "string", "description": "Exact candidate name"},
                        "evidence": {"type": "string", "description": "Exact quote from answer"},
                        "recommended": {"type": "boolean"},
                        "cited_only": {
                            "type": "boolean",
                            "description": "True if the business appears only as a source URL",
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate", "evidence", "recommended", "cited_only"],
                },
            }
        },
        "required": ["mentions"],
    },
}


class LLMExtractor(Extractor):
    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self.model = model
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError("Install the llm extra: pip install 'geo-extract[llm]'") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY to use the LLM extractor")
            client = anthropic.Anthropic()
        self.client = client

    def _call(self, answer: str, candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
        listing = "\n".join(
            f"- {c.name}"
            + (f" (site: {c.domain})" if c.domain else "")
            + (f" (also known as: {', '.join(c.aliases)})" if c.aliases else "")
            for c in candidates
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=SYSTEM,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "report_mentions"},
            messages=[
                {
                    "role": "user",
                    "content": f"Candidates:\n{listing}\n\n<answer>\n{answer}\n</answer>",
                }
            ],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return list(block.input.get("mentions", []))
        return []

    def extract(
        self,
        answer: str,
        candidates: Sequence[Candidate],
        target: str | None = None,
        location: str | None = None,
    ) -> ExtractionResult:
        names = {c.name for c in candidates}
        raw = self._call(answer, candidates)
        spans: dict[str, list[tuple[int, int, str]]] = {}
        evidence: dict[str, tuple[int, int]] = {}
        excluded: list[Exclusion] = []
        for m in raw:
            cand = m.get("candidate")
            if cand not in names:
                continue  # the model named something outside the list; ignore it
            quote = str(m.get("evidence", ""))
            start = answer.find(quote) if quote else -1
            if start == -1:
                start = answer.lower().find(quote.lower()) if quote else -1
            if start == -1:
                continue  # unverifiable quote: refuse to trust it
            end = start + len(quote)
            if not m.get("recommended", True):
                excluded.append(Exclusion(cand, str(m.get("reason", "negative context")), quote))
                continue
            via = "citation" if m.get("cited_only") else "llm"
            spans.setdefault(cand, []).append((start, end, via))
            evidence.setdefault(cand, (start, end))
        excluded = [e for e in excluded if e.candidate not in spans]
        positions = assign_positions(answer, spans)
        mentions = [
            Mention(
                name=answer[s:e],
                matched_candidate=cand,
                position=positions[cand],
                confidence=0.9,
                evidence_span=(s, e),
                evidence=answer[s:e],
                via="citation" if all(v == "citation" for *_, v in spans[cand]) else "llm",
            )
            for cand, (s, e) in evidence.items()
        ]
        return build_result(mentions, candidates, target, excluded)
