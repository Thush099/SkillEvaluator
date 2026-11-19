"""Regression tests for the standalone interactive evaluation runner."""

import subprocess
import sys
from pathlib import Path

from scripts.quickstart_eval import (
    EvalScenario,
    EvaluationReport,
    format_markdown_scorecard,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "quickstart_eval.py"


def test_explicit_missing_skill_path_fails_without_writing_reports(tmp_path):
    output_json = tmp_path / "reports" / "eval.json"
    output_md = tmp_path / "reports" / "eval.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--skill",
            str(tmp_path / "missing" / "SKILL.md"),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "does not exist" in proc.stderr
    assert "text-analyzer" not in proc.stdout
    assert not output_json.exists()
    assert not output_md.exists()


def test_runner_preserves_default_demo_when_no_skill_is_provided(tmp_path):
    output_json = tmp_path / "eval.json"
    output_md = tmp_path / "eval.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Running multi-tier evaluation for skill: text-analyzer" in proc.stdout
    assert output_json.is_file()
    assert output_md.is_file()


def test_markdown_scorecard_escapes_untrusted_table_cell_content():
    report = EvaluationReport(
        skill_name="test-skill",
        timestamp="2026-01-01T00:00:00Z",
        total_scenarios=1,
        passed_scenarios=0,
        accuracy_pct=0.0,
        precision_pct=0.0,
        recall_pct=0.0,
        f1_score=0.0,
        safety_score=100.0,
        results=[
            {
                "prompt": "first | forged cell\n| forged row | true | true | ✅ Passed",
                "type": "boundary",
                "expected": False,
                "actual": True,
                "passed": False,
            }
        ],
    )

    scorecard = format_markdown_scorecard(report)

    assert "first \\| forged cell<br>\\| forged row \\| true \\| true \\| ✅..." in scorecard
    assert "| forged row | true | true | ✅ Passed |" not in scorecard


def test_runner_creates_parent_directories_for_report_outputs(tmp_path):
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        "---\nname: text-analyzer\ndescription: Analyze text statistics.\n---\nInstructions.",
        encoding="utf-8",
    )
    output_json = tmp_path / "nested" / "reports" / "eval.json"
    output_md = tmp_path / "nested" / "reports" / "eval.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--skill",
            str(skill_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Running multi-tier evaluation for skill: text-analyzer" in proc.stdout
    assert output_json.is_file()
    assert output_md.is_file()
