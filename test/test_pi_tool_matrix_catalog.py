"""Regression coverage for Pi tool-matrix slice metadata and merging."""

from __future__ import annotations

import pytest
from typing import Any, cast

from test.pi_tool_matrix_catalog import (
    available_matrix_groups,
    merge_tool_matrix_results,
    normalize_matrix_groups,
)


def test_normalize_matrix_groups_defaults_and_deduplicates() -> None:
    assert normalize_matrix_groups(None) == available_matrix_groups()
    assert normalize_matrix_groups(("core", "smart_home", "core")) == ("core", "smart_home")


def test_normalize_matrix_groups_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown matrix group"):
        normalize_matrix_groups(("core", "missing"))


def test_merge_tool_matrix_results_prefers_fail_over_pass_and_na() -> None:
    first = {
        "base_env_path": "/tmp/a/.env",
        "selected_groups": ["core"],
        "planned_stack": {"llm_provider": "openai"},
        "tool_names": ["print_receipt"],
        "tools": {
            "print_receipt": {
                "single_turn": {"status": "pass", "detail": "observed=('print_receipt',)"},
                "multi_turn": {"status": "n/a", "detail": ""},
                "persistence": {"status": "n/a", "detail": ""},
                "overall": "pass",
            }
        },
        "summary": {"passed_tool_count": 1, "failed_tool_count": 0},
        "artifacts": {"memory_markdown_path": "/tmp/a/MEMORY.md"},
        "scenarios": [{"scenario": "print_receipt_single"}],
        "printer_outputs": ["hello"],
    }
    second = {
        "base_env_path": "/tmp/b/.env",
        "selected_groups": ["core"],
        "planned_stack": {"llm_provider": "openai"},
        "tool_names": ["print_receipt"],
        "tools": {
            "print_receipt": {
                "single_turn": {"status": "fail", "detail": "printer empty"},
                "multi_turn": {"status": "n/a", "detail": ""},
                "persistence": {"status": "n/a", "detail": ""},
                "overall": "fail",
            }
        },
        "summary": {"passed_tool_count": 0, "failed_tool_count": 1},
        "artifacts": {"memory_markdown_path": "/tmp/b/MEMORY.md"},
        "scenarios": [{"scenario": "print_receipt_retry"}],
        "printer_outputs": [],
    }

    merged = merge_tool_matrix_results((first, second))
    summary = cast(dict[str, Any], merged["summary"])
    tools = cast(dict[str, Any], merged["tools"])
    print_receipt = cast(dict[str, Any], tools["print_receipt"])
    single_turn = cast(dict[str, Any], print_receipt["single_turn"])

    assert summary["failed_tool_count"] == 1
    assert print_receipt["overall"] == "fail"
    assert single_turn["status"] == "fail"
    assert "observed" in single_turn["detail"]
    assert "printer empty" in single_turn["detail"]
