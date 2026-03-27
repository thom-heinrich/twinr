"""Shared datatypes for git guard scan inputs and findings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathChange:
    """Describe one path-level change collected from git."""

    path: str
    status: str
    previous_path: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class AddedLine:
    """Describe one added line collected from a git patch."""

    path: str
    line_number: int | None
    text: str
    commit: str | None = None


@dataclass(frozen=True)
class ScanIssue:
    """Describe one policy violation found by the git guard."""

    rule_id: str
    message: str
    path: str
    line_number: int | None = None
    excerpt: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Wrap the findings produced by a guard run."""

    issues: tuple[ScanIssue, ...]

    @property
    def ok(self) -> bool:
        """Return True when the scan produced no policy violations."""

        return not self.issues
