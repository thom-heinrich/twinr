"""Merge multiple Pi tool-matrix slice artifacts into one combined result.

Use this after bounded real-Pi slice runs when the full live matrix would be
too large or too expensive for one SSH invocation. The script keeps the merge
policy deterministic so acceptance artifacts remain comparable across runs.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src ./.venv/bin/python test/merge_pi_tool_matrix_results.py \
        --input-json artifacts/reports/tool_matrix_core.json \
        --input-json artifacts/reports/tool_matrix_self_coding.json \
        --output-json artifacts/reports/tool_matrix_merged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from test.pi_tool_matrix_catalog import merge_tool_matrix_results


def main() -> int:
    """Load slice JSON payloads, merge them, and print the combined artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.input_json]
    merged = merge_tool_matrix_results(results)
    payload = json.dumps(merged, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
