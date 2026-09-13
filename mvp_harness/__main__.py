"""CLI entry point: `python -m mvp_harness <input_dir> <output_dir> [options]`."""
from __future__ import annotations

import argparse
from pathlib import Path

from mvp_harness.cli import run_harness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Standard pipeline's 360-image MVP test harness (spec §10) over a folder of photos."
    )
    parser.add_argument("input_dir", type=Path, help="Folder of category-prefixed photos, e.g. ramen_001.jpg")
    parser.add_argument("output_dir", type=Path, help="Folder to write rendered result images and reports into")
    parser.add_argument(
        "--run-csv", type=Path, default=None, help="Per-(image,template) CSV path (default: <output_dir>/run_results.csv)"
    )
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=None,
        help="Category x template summary CSV path (default: <output_dir>/matrix_report.csv)",
    )
    parser.add_argument(
        "--error-log", type=Path, default=None, help="Traceback log path for failed pairs (default: <output_dir>/errors.log)"
    )
    args = parser.parse_args()

    run_csv = args.run_csv or args.output_dir / "run_results.csv"
    matrix_csv = args.matrix_csv or args.output_dir / "matrix_report.csv"
    error_log = args.error_log or args.output_dir / "errors.log"

    records = run_harness(args.input_dir, args.output_dir, run_csv, matrix_csv, error_log)

    counts: dict[str, int] = {}
    for record in records:
        counts[record.validator_status] = counts.get(record.validator_status, 0) + 1

    print(f"Processed {len(records)} (image, template) pairs.")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"Run CSV:       {run_csv}")
    print(f"Matrix report: {matrix_csv}")
    print(f"Error log:     {error_log}")


if __name__ == "__main__":
    main()
