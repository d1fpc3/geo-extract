import json
from pathlib import Path

import pytest

from geo_extract import RuleExtractor
from geo_extract.cli import main
from geo_extract.evaluation import evaluate, load_cases, to_markdown

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset"


def test_dataset_is_well_formed() -> None:
    total = 0
    for split in ("dev", "heldout", "stress"):
        for case in load_cases(DATASET / f"{split}.jsonl"):
            total += 1
            names = {c["name"] for c in case["candidates"]}
            assert case["target"] in names
            assert all(g["candidate"] in names for g in case["gold"]["named"])
            assert case["split"] == split
    assert total >= 30


def test_eval_harness_runs_and_dev_stays_green() -> None:
    report = evaluate(RuleExtractor(), load_cases(DATASET / "dev.jsonl"), "dev")
    assert len(report.cases) >= 20
    # regression guard: the rule extractor should not slip on the dev split
    assert report.f1 >= 0.95
    assert report.position_accuracy >= 0.95
    md = to_markdown([report])
    assert "| rule | dev |" in md


def test_cli_check_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    answer = tmp_path / "a.txt"
    answer.write_text("1. Alpha Air - good\n2. Beta Heat - fine\n", encoding="utf8")
    cands = tmp_path / "c.txt"
    cands.write_text("Alpha Air\nBeta Heat | betaheat.com\n# comment\n", encoding="utf8")
    assert (
        main(
            [
                "check",
                "--answer",
                str(answer),
                "--candidates",
                str(cands),
                "--target",
                "Beta Heat",
                "--json",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["target_named"] is True
    assert out["target_position"] == 2


def test_cli_check_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    answer = tmp_path / "a.txt"
    answer.write_text("Gamma Roofing is the best roofer around.", encoding="utf8")
    assert main(["check", "--answer", str(answer), "--target", "Delta Roofing"]) == 0
    assert "was not named" in capsys.readouterr().out


def test_cli_eval_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "RESULTS.md"
    assert main(["eval", "--dataset", str(DATASET), "--splits", "dev", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf8").startswith("# Eval results")
