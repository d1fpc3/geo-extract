# geo-extract

**Given an AI assistant's answer to a local-business question, find which businesses it named, in what order, and whether yours was one of them.**

This is the scoring core behind AI search visibility tracking. Ask ChatGPT, Claude, Perplexity or Gemini "who's the best HVAC company in Bel Air?" and you get prose, lists, tables and citations back. `geo-extract` turns that text into a ranked, verifiable list of businesses.

```
$ geo-extract check --answer examples/answer.txt --candidates examples/candidates.txt \
    --target "Polar Bear Heating & Air" --location "Bel Air, MD"

| Pos | Business                    | As written                            | Via      | Conf |
|   1 | Harford Comfort Systems     | Harford Comfort Systems               | name     | 1.00 |
|   2 | Polar Bear Heating & Air    | Polar Bear Heating & Air              | name     | 1.00 |
|   3 | Chesapeake Climate Company  | Chesapeake Climate Co                 | name     | 1.00 |
|   5 | Northwind Air (not tracked) | Northwind Air                         | name     | 0.50 |
|   - | Ridgeline Mechanical LLC    | https://www.ridgelinemech.com/reviews | citation | 0.90 |
Excluded Bayside HVAC: "permanently closed" (Bayside HVAC)
Polar Bear Heating & Air was named (position 2).
```

## Why it's harder than a string search

- **Names drift.** "ClearFlow Plumbing LLC" shows up as "Clear Flow Plumbing", "Joe's" as "Joes", "Tamarac Termite and Pest Control" as "Tamarac Termite & Pest", and a town name inside a business name ("Bel Air Plumbing") must never match the town on its own.
- **Being mentioned is not being recommended.** "Towson Wire Works appears to have permanently closed" or "I'd steer clear of Quickfix" names a business without recommending it. The cue has to be scoped to the right business in a sentence that names two.
- **Position depends on structure.** Rank comes from numbered lists, bullets, markdown headings and table rows, while sub-bullets, incidental mentions ("started by a former Polar Bear tech") and citation footers must not shift it. Near-duplicates ("Summit Roofing" vs "Summit Roofing & Gutters") have to resolve to the right company.

## Results

From [`evals/RESULTS.md`](evals/RESULTS.md), regenerated with `geo-extract eval`:

| Extractor | Split | Cases | Precision | Recall | F1 | Position acc. | Target acc. |
|---|---|---|---|---|---|---|---|
| rule | dev | 22 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rule | held-out | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rule | stress | 7 | 0.667 | 0.500 | 0.571 | 0.500 | 0.429 |

Read these honestly:

- **The dataset is synthetic.** Every answer was written by hand to mimic how the engines format local recommendations, using fictional businesses. None of it is captured engine output. It is small, so treat it as a regression suite with teeth, not a benchmark.
- **The held-out split was never used for tuning.** The rule extractor was developed against `dev` only; `heldout` was run once at the end.
- **The stress split is where rules break, on purpose.** It holds the cases a deterministic matcher can't solve: all-lowercase voice answers, closures without a keyword ("they're gone now"), acronyms with no alias ("HCS"), an untracked near-duplicate, and "avoid" used in a positive sentence. Those are exactly the cases the `LLMExtractor` exists for.

## Quickstart

```bash
uv sync                       # or: pip install -e .
uv run geo-extract check --answer examples/answer.txt --candidates examples/candidates.txt \
    --target "Polar Bear Heating & Air" --location "Bel Air, MD"
uv run geo-extract check --answer - --target "Your Business" --json < answer.txt
uv run geo-extract eval       # writes evals/RESULTS.md
uv run pytest
```

Candidates file, one per line: `Name | domain.com | alias one; alias two` (domain and aliases optional).

As a library:

```python
from geo_extract import Candidate, RuleExtractor

result = RuleExtractor().extract(
    answer_text,
    [Candidate("Polar Bear Heating & Air", domain="polarbearair.com"), Candidate("Bayside HVAC")],
    target="Polar Bear Heating & Air",
    location="Bel Air, MD",
)
result.target_named, result.target_position, result.named_candidates, result.excluded
```

### LLM extractor

```bash
uv sync --extra llm
export ANTHROPIC_API_KEY=...
uv run geo-extract check --extractor llm --answer answer.txt --candidates candidates.txt
uv run geo-extract eval --llm      # scores both extractors side by side
```

Claude (`claude-haiku-5-5`) is forced through a single tool call, `report_mentions`, and must quote the exact text it saw for every mention. Quotes that can't be found in the answer are thrown away, so the model can't invent a mention. Positions are then computed by the same deterministic rule as the rule extractor, so the two are compared on identical terms. The LLM path is covered by tests with a mocked client; no live numbers are published here yet.

## How the eval works

Each case in `evals/dataset/*.jsonl` has the query, location, answer text, candidate list, target business and gold labels: which candidates were recommended, and at what position. Scoring is per candidate, micro-averaged across cases:

- **Precision / recall / F1:** did we name the right candidates?
- **Position accuracy:** for correctly named candidates, is the rank exactly right?
- **Target accuracy:** did we get "was my business named?" right? This is the number a business owner sees.

The position rule is written down once in [`structure.py`](src/geo_extract/structure.py) and used by both the extractors and the gold labels. Every top-level list item, numbered heading or table row is a slot. A business takes the slot whose head names it, or a slot of its own if it's only named in prose. Citation-only businesses count as named, with no position.

## Design notes

- **Evidence or it didn't happen.** Every mention carries a character span into the original answer, and tests assert those spans are exact. That makes results auditable and lets a UI highlight what the engine actually said.
- **Three matching strategies, scored differently.** Full-name fuzzy match (spaced and compact, so "EcoCool" == "Eco-Cool"), truncated match (trailing trade words dropped), and core match (just the distinctive words, capitalized). Overlaps resolve by score, then by span length, which is what separates near-duplicate names.
- **Location-aware.** Words from the query's location are treated as generic, so they can't identify a business alone.
- **Negative cues are clause-scoped.** "Alpha closed down, but Beta is great" excludes only Alpha.
- **Untracked competitors are reported too.** List items that name a business outside your candidate list come back with `matched_candidate=None`, so you find out who else the engine recommends.

## Where this came from

This is a standalone, open version of the extraction step inside [SeenRank](https://seenrank.com), which asks AI engines the questions real customers ask every week and tracks which local businesses get named.

Built by Frankie Claridge. MIT licensed.
