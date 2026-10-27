# Eval dataset

`dataset/dev.jsonl`, `dataset/heldout.jsonl` and `dataset/stress.jsonl` hold 39 labeled cases.

**These are synthetic fixtures.** Each answer was hand-written to mimic how ChatGPT, Claude, Perplexity, Gemini and voice assistants format local-business recommendations: numbered lists, bold bullets with sub-bullets, markdown headings, tables, prose, citation footers. Every business name is fictional. None of these are captured engine outputs.

| Split | Cases | Purpose |
|---|---|---|
| dev | 22 | Used while building the rule extractor |
| heldout | 10 | Written alongside dev, run once at the end, never used for tuning |
| stress | 7 | Known failure modes for rules: lowercase voice answers, closures without a keyword, acronyms without an alias, untracked near-duplicates, "avoid" in a positive sentence |

Hard cases covered: spelling and suffix variants, possessives, a business mentioned only to say it closed, "steer clear" warnings, near-duplicate names, URL-only citations, a top-5 list in markdown headings, a markdown table, prose with no list, multiple locations in one answer, incidental mentions inside another business's description, aliases, and an answer that names nobody.

## Case format

```json
{
  "id": "d04",
  "split": "dev",
  "engine_style": "chatgpt",
  "query": "electrician towson",
  "location": "Towson, MD",
  "target": "Towson Wire Works",
  "candidates": [{"name": "Bright Spark Electric"}, {"name": "Towson Wire Works"}],
  "answer": "...",
  "gold": {
    "named": [{"candidate": "Bright Spark Electric", "position": 1}],
    "target_named": false,
    "target_position": null
  }
}
```

`position` follows the rule in `src/geo_extract/structure.py`; `null` means cited as a source only.

Run `geo-extract eval` (add `--llm` with `ANTHROPIC_API_KEY` set to score the LLM extractor too). Results land in `RESULTS.md`.
