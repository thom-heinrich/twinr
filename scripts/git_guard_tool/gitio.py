"""Collect staged and outgoing git changes for git guard scanning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from git_guard_tool.types import AddedLine, PathChange

_ZERO_OID = "0" * 40


class GitGuardGitError(RuntimeError):
    """Raise when git data needed for scanning cannot be collected."""


@dataclass(frozen=True)
class PrePushUpdate:
    """Describe one ref update delivered to the `pre-push` hook."""

    local_ref: str
    local_oid: str
    remote_ref: str
    remote_oid: str


def _run_git(repo_root: Path, args: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        input=input_text,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise GitGuardGitError(f"`git {' '.join(args)}` failed: {detail}")
    return completed.stdout


def _git_object_exists(repo_root: Path, oid: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{oid}^{{commit}}"],
        cwd=repo_root,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return completed.returncode == 0


def resolve_repo_root(explicit_repo_root: str | None = None) -> Path:
    """Resolve the git toplevel that should be scanned."""

    if explicit_repo_root:
        return Path(explicit_repo_root).resolve()
    discovered = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        capture_output=True,
    )
    if discovered.returncode != 0:
        detail = discovered.stderr.strip() or discovered.stdout.strip() or "not inside a git repository"
        raise GitGuardGitError(detail)
    return Path(discovered.stdout.strip()).resolve()


def parse_pre_push_updates(text: str) -> tuple[PrePushUpdate, ...]:
    """Parse the stdin payload that git passes to the `pre-push` hook."""

    updates: list[PrePushUpdate] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise GitGuardGitError(f"invalid pre-push update line: {raw_line!r}")
        updates.append(
            PrePushUpdate(
                local_ref=parts[0],
                local_oid=parts[1],
                remote_ref=parts[2],
                remote_oid=parts[3],
            )
        )
    return tuple(updates)


def _parse_name_status_z(output: str, *, commit: str | None = None) -> tuple[PathChange, ...]:
    entries = [entry for entry in output.split("\x00") if entry]
    changes: list[PathChange] = []
    index = 0
    while index < len(entries):
        status_token = entries[index]
        index += 1
        status = status_token[:1]
        if status in {"R", "C"}:
            if index + 1 >= len(entries):
                raise GitGuardGitError("malformed git name-status output")
            previous_path = entries[index]
            path = entries[index + 1]
            index += 2
        else:
            if index >= len(entries):
                raise GitGuardGitError("malformed git name-status output")
            previous_path = None
            path = entries[index]
            index += 1
        changes.append(PathChange(path=path, status=status, previous_path=previous_path, commit=commit))
    return tuple(changes)


def _parse_hunk_start(header: str) -> int | None:
    if "@@" not in header:
        return None
    middle = header.split("@@", maxsplit=2)[1].strip()
    for token in middle.split():
        if not token.startswith("+"):
            continue
        line_spec = token[1:]
        start_text = line_spec.split(",", maxsplit=1)[0]
        try:
            return int(start_text)
        except ValueError:
            return None
    return None


def parse_added_lines(patch_text: str, *, commit: str | None = None) -> tuple[AddedLine, ...]:
    """Parse added lines plus line numbers from unified diff text."""

    additions: list[AddedLine] = []
    current_path: str | None = None
    current_line_number: int | None = None

    for raw_line in patch_text.splitlines():
        if raw_line.startswith("diff --git "):
            current_path = None
            current_line_number = None
            continue
        if raw_line.startswith("rename to "):
            current_path = raw_line[len("rename to ") :].strip()
            continue
        if raw_line.startswith("+++ "):
            rendered = raw_line[4:].strip()
            if rendered == "/dev/null":
                current_path = None
            elif rendered.startswith("b/"):
                current_path = rendered[2:]
            else:
                current_path = rendered
            continue
        if raw_line.startswith("@@"):
            current_line_number = _parse_hunk_start(raw_line)
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            additions.append(
                AddedLine(
                    path=current_path or "<unknown>",
                    line_number=current_line_number,
                    text=raw_line[1:],
                    commit=commit,
                )
            )
            if current_line_number is not None:
                current_line_number += 1
            continue
        if raw_line.startswith(" ") and current_line_number is not None:
            current_line_number += 1
    return tuple(additions)


def collect_staged_changes(repo_root: Path) -> tuple[tuple[PathChange, ...], tuple[AddedLine, ...]]:
    """Collect staged path changes and added lines."""

    path_output = _run_git(repo_root, ["diff", "--cached", "--name-status", "-z", "--diff-filter=ACMR"])
    patch_output = _run_git(
        repo_root,
        ["diff", "--cached", "--unified=0", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/"],
    )
    return _parse_name_status_z(path_output), parse_added_lines(patch_output)


def _collect_outgoing_commits(repo_root: Path, remote: str, updates: tuple[PrePushUpdate, ...]) -> tuple[str, ...]:
    commits: list[str] = []
    seen: set[str] = set()

    for update in updates:
        if update.local_oid == _ZERO_OID:
            continue
        if update.remote_oid != _ZERO_OID:
            if not _git_object_exists(repo_root, update.remote_oid):
                raise GitGuardGitError(
                    "remote ref "
                    f"{update.remote_ref} points to {update.remote_oid}, which is not present locally; "
                    f"run `git fetch {remote} {update.remote_ref}` and integrate that remote commit before pushing"
                )
            rev_list = _run_git(repo_root, ["rev-list", "--reverse", f"{update.remote_oid}..{update.local_oid}"])
        else:
            rev_list = _run_git(
                repo_root,
                ["rev-list", "--reverse", update.local_oid, "--not", f"--remotes={remote}"],
            )
            if not rev_list.strip():
                rev_list = _run_git(repo_root, ["rev-list", "--reverse", f"{update.local_oid}^!"])
        for commit in rev_list.splitlines():
            rendered = commit.strip()
            if not rendered or rendered in seen:
                continue
            seen.add(rendered)
            commits.append(rendered)
    return tuple(commits)


def collect_push_changes(
    repo_root: Path,
    *,
    remote: str,
    updates: tuple[PrePushUpdate, ...],
) -> tuple[tuple[PathChange, ...], tuple[AddedLine, ...]]:
    """Collect path changes and added lines for commits that are about to be pushed."""

    commits = _collect_outgoing_commits(repo_root, remote, updates)
    path_changes: list[PathChange] = []
    additions: list[AddedLine] = []
    for commit in commits:
        path_output = _run_git(repo_root, ["diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-z", commit])
        patch_output = _run_git(
            repo_root,
            ["show", "--format=", "--unified=0", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/", commit],
        )
        path_changes.extend(_parse_name_status_z(path_output, commit=commit))
        additions.extend(parse_added_lines(patch_output, commit=commit))
    return tuple(path_changes), tuple(additions)
