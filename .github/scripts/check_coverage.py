"""Enforce AI-DataFlux overall and critical-module coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

OVERALL_LINE_MIN = 75.0
OVERALL_BRANCH_MIN = 65.0
CRITICAL_GROUP_LINE_MIN = 85.0
CRITICAL_FILE_LINE_MIN = 70.0


def _percent(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered * 100.0 / total


def _line_coverage(summary: dict[str, Any]) -> float:
    return _percent(int(summary["covered_lines"]), int(summary["num_statements"]))


def _branch_coverage(summary: dict[str, Any]) -> float:
    return _percent(int(summary["covered_branches"]), int(summary["num_branches"]))


def _group_summary(
    files: dict[str, Any],
    predicate: Callable[[str], bool],
) -> tuple[list[tuple[str, dict[str, Any]]], int, int]:
    selected = [
        (path, data["summary"]) for path, data in files.items() if predicate(path)
    ]
    covered = sum(int(summary["covered_lines"]) for _, summary in selected)
    statements = sum(int(summary["num_statements"]) for _, summary in selected)
    return selected, covered, statements


def check_coverage(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    totals = report["totals"]
    overall_line = _line_coverage(totals)
    overall_branch = _branch_coverage(totals)
    print(f"overall line: {overall_line:.2f}% (required {OVERALL_LINE_MIN:.2f}%)")
    print(
        f"overall branch: {overall_branch:.2f}% "
        f"(required {OVERALL_BRANCH_MIN:.2f}%)"
    )
    if overall_line < OVERALL_LINE_MIN:
        failures.append(
            f"overall line coverage {overall_line:.2f}% < {OVERALL_LINE_MIN:.2f}%"
        )
    if overall_branch < OVERALL_BRANCH_MIN:
        failures.append(
            f"overall branch coverage {overall_branch:.2f}% < {OVERALL_BRANCH_MIN:.2f}%"
        )

    files = report["files"]
    groups: dict[str, Callable[[str], bool]] = {
        "jobs": lambda path: path.startswith("src/jobs/"),
        "core-runner": lambda path: path
        in {"src/core/processor.py", "src/core/job_runner.py"},
        "gateway": lambda path: path.startswith("src/gateway/"),
    }
    for name, predicate in groups.items():
        selected, covered, statements = _group_summary(files, predicate)
        if not selected:
            failures.append(f"critical group {name} matched no files")
            continue
        group_line = _percent(covered, statements)
        print(
            f"critical group {name} line: {group_line:.2f}% "
            f"(required {CRITICAL_GROUP_LINE_MIN:.2f}%)"
        )
        if group_line < CRITICAL_GROUP_LINE_MIN:
            failures.append(
                f"critical group {name} line coverage {group_line:.2f}% "
                f"< {CRITICAL_GROUP_LINE_MIN:.2f}%"
            )
        for path, summary in selected:
            file_line = _line_coverage(summary)
            if file_line < CRITICAL_FILE_LINE_MIN:
                failures.append(
                    f"critical file {path} line coverage {file_line:.2f}% "
                    f"< {CRITICAL_FILE_LINE_MIN:.2f}%"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="coverage.py JSON report")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = check_coverage(report)
    if failures:
        print("coverage gate failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("coverage gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
