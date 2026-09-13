"""CSV + category x template matrix reporting (spec §10's "탈락자 제거" goal)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from mvp_harness.runner import HarnessRecord

RUN_CSV_FIELDS = [
    "image_id",
    "category",
    "template_id",
    "validator_status",
    "reasons",
    "retry_attempts_used",
    "processing_time_ms",
]


def write_run_csv(records: list[HarnessRecord], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RUN_CSV_FIELDS)
        for r in records:
            writer.writerow(
                [
                    r.image_id,
                    r.category,
                    r.template_id,
                    r.validator_status,
                    ";".join(r.reasons),
                    r.retry_attempts_used,
                    f"{r.processing_time_ms:.1f}",
                ]
            )


MATRIX_CSV_FIELDS = ["category", "template_id", "total", "pass", "review", "reject", "error", "pass_rate"]


def write_matrix_report(records: list[HarnessRecord], path: Path) -> None:
    """
    One row per (category, template_id) actually seen, with PASS/REVIEW/
    REJECT/ERROR counts and a pass_rate - meant to make a template that's
    disproportionately bad for one category visible at a glance (§10).
    Interpreting these numbers (which template "wins") is explicitly out
    of scope for this build step - this only produces them.
    """
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        cells[(r.category, r.template_id)][r.validator_status] += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MATRIX_CSV_FIELDS)
        for (category, template_id) in sorted(cells):
            counts = cells[(category, template_id)]
            total = sum(counts.values())
            p, rv, rj, er = counts.get("PASS", 0), counts.get("REVIEW", 0), counts.get("REJECT", 0), counts.get("ERROR", 0)
            pass_rate = p / total if total else 0.0
            writer.writerow([category, template_id, total, p, rv, rj, er, f"{pass_rate:.2f}"])
