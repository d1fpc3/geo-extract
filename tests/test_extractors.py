from types import SimpleNamespace
from typing import Any

from geo_extract import Candidate, RuleExtractor
from geo_extract.llm import LLMExtractor

ANSWER = """Top HVAC in Bel Air:

1. **Polar Bear Heating & Air** - fast
2. **Harford Comfort Systems** - heat pumps
3. **Bayside HVAC** - permanently closed, skip it

Source: https://www.ridgelinemech.com/reviews
"""

CANDS = [
    Candidate("Polar Bear Heating & Air"),
    Candidate("Harford Comfort Systems"),
    Candidate("Bayside HVAC"),
    Candidate("Ridgeline Mechanical", domain="ridgelinemech.com"),
]


def test_rule_extractor_end_to_end() -> None:
    r = RuleExtractor().extract(
        ANSWER, CANDS, target="polar bear heating and air", location="Bel Air"
    )
    assert r.named_candidates == {
        "Polar Bear Heating & Air": 1,
        "Harford Comfort Systems": 2,
        "Ridgeline Mechanical": None,
    }
    assert r.target == "Polar Bear Heating & Air"
    assert r.target_named and r.target_position == 1
    assert [e.candidate for e in r.excluded] == ["Bayside HVAC"]
    ridgeline = next(m for m in r.mentions if m.matched_candidate == "Ridgeline Mechanical")
    assert ridgeline.via == "citation"


def test_rule_extractor_reports_untracked_list_names() -> None:
    text = "1. Alpha Air - good\n2. Zeta Cooling - good\n"
    r = RuleExtractor().extract(text, [Candidate("Alpha Air")])
    untracked = [m.name for m in r.mentions if m.matched_candidate is None]
    assert untracked == ["Zeta Cooling"]


def test_evidence_spans_point_at_original_text() -> None:
    r = RuleExtractor().extract(ANSWER, CANDS)
    for m in r.mentions:
        s, e = m.evidence_span
        assert ANSWER[s:e] == m.evidence


class FakeClient:
    """Stands in for anthropic.Anthropic so the LLM path is tested without a key."""

    def __init__(self, mentions: list[dict[str, Any]]) -> None:
        self.calls: list[dict[str, Any]] = []
        block = SimpleNamespace(type="tool_use", input={"mentions": mentions})
        self.messages = SimpleNamespace(create=self._create)
        self._response = SimpleNamespace(content=[block])

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


def test_llm_extractor_with_mocked_client() -> None:
    client = FakeClient(
        [
            {
                "candidate": "Polar Bear Heating & Air",
                "evidence": "Polar Bear Heating & Air",
                "recommended": True,
                "cited_only": False,
            },
            {
                "candidate": "Harford Comfort Systems",
                "evidence": "Harford Comfort Systems",
                "recommended": True,
                "cited_only": False,
            },
            {
                "candidate": "Bayside HVAC",
                "evidence": "Bayside HVAC",
                "recommended": False,
                "cited_only": False,
                "reason": "closed",
            },
            {
                "candidate": "Ridgeline Mechanical",
                "evidence": "https://www.ridgelinemech.com/reviews",
                "recommended": True,
                "cited_only": True,
            },
            {
                "candidate": "Made Up Co",
                "evidence": "nowhere",
                "recommended": True,
                "cited_only": False,
            },
        ]
    )
    r = LLMExtractor(client=client).extract(ANSWER, CANDS, target="Harford Comfort Systems")
    assert r.named_candidates == {
        "Polar Bear Heating & Air": 1,
        "Harford Comfort Systems": 2,
        "Ridgeline Mechanical": None,
    }
    assert r.target_position == 2
    assert [e.candidate for e in r.excluded] == ["Bayside HVAC"]
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["tool_choice"] == {"type": "tool", "name": "report_mentions"}


def test_llm_extractor_rejects_unverifiable_quotes() -> None:
    client = FakeClient(
        [
            {
                "candidate": "Bayside HVAC",
                "evidence": "Bayside is the best",
                "recommended": True,
                "cited_only": False,
            },
        ]
    )
    r = LLMExtractor(client=client).extract(ANSWER, CANDS)
    assert r.named_candidates == {}
