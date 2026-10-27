"""Command line: `geo-extract check` and `geo-extract eval`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .base import Extractor
from .evaluation import evaluate, load_cases, to_markdown
from .models import Candidate
from .rule import RuleExtractor


def _extractor(kind: str) -> Extractor:
    if kind == "llm":
        from .llm import LLMExtractor

        return LLMExtractor()
    return RuleExtractor()


def _read_candidates(path: str) -> list[Candidate]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [Candidate.from_line(x) for x in lines if x.strip() and not x.lstrip().startswith("#")]


def cmd_check(args: argparse.Namespace) -> int:
    answer = (
        sys.stdin.read() if args.answer == "-" else Path(args.answer).read_text(encoding="utf8")
    )
    candidates = _read_candidates(args.candidates) if args.candidates else []
    if args.target and not any(c.name == args.target for c in candidates):
        candidates.append(Candidate(args.target))
    result = _extractor(args.extractor).extract(answer, candidates, args.target, args.location)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    console = Console()
    table = Table(title="Businesses named in the answer", header_style="bold")
    table.add_column("Pos", justify="right")
    table.add_column("Business")
    table.add_column("As written")
    table.add_column("Via")
    table.add_column("Conf", justify="right")
    for m in result.mentions:
        name = m.matched_candidate or f"[dim]{m.name} (not tracked)[/dim]"
        if m.matched_candidate and m.matched_candidate == result.target:
            name = f"[bold green]{name}[/bold green]"
        table.add_row(str(m.position or "-"), name, m.evidence, m.via, f"{m.confidence:.2f}")
    console.print(table)
    for e in result.excluded:
        console.print(f'[yellow]Excluded[/yellow] {e.candidate}: "{e.reason}" ({e.evidence})')
    if result.target:
        if result.target_named:
            pos = f"position {result.target_position}" if result.target_position else "cited only"
            console.print(f"[bold green]{result.target} was named[/bold green] ({pos}).")
        else:
            console.print(f"[bold red]{result.target} was not named.[/bold red]")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    kinds = ["rule"]
    if args.llm or os.environ.get("ANTHROPIC_API_KEY"):
        kinds.append("llm")
    reports = []
    for kind in kinds:
        ex = _extractor(kind)
        for split in args.splits:
            path = Path(args.dataset) / f"{split}.jsonl"
            reports.append(evaluate(ex, load_cases(path), split))
    md = to_markdown(reports)
    if args.out:
        Path(args.out).write_text(md, encoding="utf8", newline="\n")
    console = Console()
    for r in reports:
        console.print(r.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="geo-extract", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="Which businesses does this AI answer name?")
    c.add_argument("--answer", required=True, help="Path to the answer text, or - for stdin")
    c.add_argument("--target", help="Your business")
    c.add_argument("--candidates", help="File with one candidate per line: Name | domain | aliases")
    c.add_argument("--location", help='Query location, e.g. "Bel Air, MD"')
    c.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    c.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("eval", help="Score extractors on the labeled dataset")
    e.add_argument("--dataset", default="evals/dataset")
    e.add_argument("--splits", nargs="+", default=["dev", "heldout", "stress"])
    e.add_argument("--out", default="evals/RESULTS.md")
    e.add_argument("--llm", action="store_true", help="Also run the LLM extractor")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
