"""Install and run the repo-local git content/secrets guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from git_guard_tool.gitio import (
    GitGuardGitError,
    collect_push_changes,
    collect_staged_changes,
    parse_pre_push_updates,
    resolve_repo_root,
)
from git_guard_tool.policy import load_policy
from git_guard_tool.scanner import scan_changes
from git_guard_tool.types import ScanResult


class _CliError(RuntimeError):
    """Raise when CLI arguments or execution preconditions are invalid."""

    def __init__(self, message: str, *, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="git_guard", description=__doc__)
    parser.add_argument("--repo-root", help="Override the repository root to scan.")
    parser.add_argument("--policy", help="Override the policy file path.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--json-pretty", action="store_true", help="Emit pretty JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("schema", help="Emit JSON schema-like command metadata.")

    install_parser = subparsers.add_parser("install", help="Install the versioned git hooks into this repository.")
    install_parser.add_argument(
        "--hooks-path",
        default="scripts/git_hooks",
        help="Repo-relative path that should become git's core.hooksPath.",
    )

    subparsers.add_parser("scan-staged", help="Scan staged changes for guard violations.")

    push_parser = subparsers.add_parser("scan-push", help="Scan outgoing commits for guard violations.")
    push_parser.add_argument("--remote", default="origin", help="Remote name passed by git's pre-push hook.")
    push_parser.add_argument(
        "--stdin-file",
        help="Read pre-push ref updates from this file instead of stdin. Useful for tests.",
    )
    return parser


def _json_enabled(args: argparse.Namespace) -> bool:
    return bool(args.json or args.json_pretty or args.command == "schema")


def _print_json(payload: dict[str, object], *, pretty: bool) -> None:
    rendered = json.dumps(payload, indent=2 if pretty else None, sort_keys=True)
    sys.stdout.write(rendered)
    sys.stdout.write("\n")


def _render_result_text(result: ScanResult, *, policy_path: Path, mode: str) -> str:
    if result.ok:
        return f"Twinr git guard: {mode} scan passed with policy {policy_path}."
    lines = [
        f"Twinr git guard blocked the {mode} action.",
        f"Policy: {policy_path}",
        "",
    ]
    for issue in result.issues:
        location = issue.path
        if issue.line_number is not None:
            location = f"{location}:{issue.line_number}"
        prefix = f"[{issue.rule_id}] {location}"
        detail = issue.message
        if issue.commit:
            detail = f"{detail} (commit {issue.commit[:12]})"
        lines.append(f"- {prefix}: {detail}")
        if issue.excerpt:
            lines.append(f"  {issue.excerpt}")
    return "\n".join(lines)


def _result_payload(result: ScanResult, *, policy_path: Path, mode: str) -> dict[str, object]:
    return {
        "ok": result.ok,
        "mode": mode,
        "policy_path": str(policy_path),
        "issues": [
            {
                "rule_id": issue.rule_id,
                "message": issue.message,
                "path": issue.path,
                "line_number": issue.line_number,
                "excerpt": issue.excerpt,
                "commit": issue.commit,
            }
            for issue in result.issues
        ],
    }


def _read_push_updates(args: argparse.Namespace) -> str:
    if args.stdin_file:
        return Path(args.stdin_file).read_text(encoding="utf-8")
    return sys.stdin.read()


def _install_hooks(repo_root: Path, hooks_path: str) -> dict[str, object]:
    resolved_hooks_path = repo_root / hooks_path
    if not resolved_hooks_path.exists():
        raise _CliError(f"hooks path does not exist: {resolved_hooks_path}")
    for hook_name in ("pre-commit", "pre-push"):
        if not (resolved_hooks_path / hook_name).exists():
            raise _CliError(f"missing required hook wrapper: {resolved_hooks_path / hook_name}")
    completed = subprocess.run(
        ["git", "config", "--local", "core.hooksPath", hooks_path],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git config failed"
        raise _CliError(detail, exit_code=1)
    return {"ok": True, "hooks_path": hooks_path, "repo_root": str(repo_root)}


def _schema() -> dict[str, object]:
    return {
        "ok": True,
        "schema": {
            "summary": "Repo-local git hook guard for blocked terms, phone-like numbers, and secret material.",
            "commands": {
                "schema": "Emit JSON metadata for tool discovery.",
                "install": "Configure git to use the versioned hook wrappers under scripts/git_hooks.",
                "scan-staged": "Scan staged changes. Intended for pre-commit hooks and manual checks.",
                "scan-push": "Scan outgoing commits from pre-push hook stdin updates.",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    json_enabled = _json_enabled(args)
    pretty_json = bool(args.json_pretty)

    try:
        if args.command == "schema":
            _print_json(_schema(), pretty=True)
            return 0

        repo_root = resolve_repo_root(args.repo_root)
        policy = load_policy(repo_root, policy_path=Path(args.policy).resolve() if args.policy else None)

        if args.command == "install":
            payload = _install_hooks(repo_root, args.hooks_path)
            if json_enabled:
                _print_json(payload, pretty=pretty_json)
            else:
                sys.stdout.write(
                    f"Configured git core.hooksPath={payload['hooks_path']} for {payload['repo_root']}\n"
                )
            return 0

        if args.command == "scan-staged":
            path_changes, added_lines = collect_staged_changes(repo_root)
            result = scan_changes(path_changes=path_changes, added_lines=added_lines, policy=policy)
            payload = _result_payload(result, policy_path=policy.policy_path, mode="staged")
            if json_enabled:
                _print_json(payload, pretty=pretty_json)
            else:
                sys.stdout.write(_render_result_text(result, policy_path=policy.policy_path, mode="staged"))
                sys.stdout.write("\n")
            return 0 if result.ok else 1

        if args.command == "scan-push":
            updates_text = _read_push_updates(args)
            updates = parse_pre_push_updates(updates_text)
            path_changes, added_lines = collect_push_changes(repo_root, remote=args.remote, updates=updates)
            result = scan_changes(path_changes=path_changes, added_lines=added_lines, policy=policy)
            payload = _result_payload(result, policy_path=policy.policy_path, mode="push")
            if json_enabled:
                _print_json(payload, pretty=pretty_json)
            else:
                sys.stdout.write(_render_result_text(result, policy_path=policy.policy_path, mode="push"))
                sys.stdout.write("\n")
            return 0 if result.ok else 1

        raise _CliError(f"unknown command: {args.command}")
    except _CliError as exc:
        payload = {"ok": False, "error": "cli_error", "detail": str(exc)}
        if json_enabled:
            _print_json(payload, pretty=pretty_json)
        else:
            sys.stderr.write(f"git_guard: {exc}\n")
        return exc.exit_code
    except GitGuardGitError as exc:
        payload = {"ok": False, "error": "git_error", "detail": str(exc)}
        if json_enabled:
            _print_json(payload, pretty=pretty_json)
        else:
            sys.stderr.write(f"git_guard: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
