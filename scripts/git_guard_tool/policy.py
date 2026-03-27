"""Load the configurable policy used by the repo-local git guard."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


DEFAULT_POLICY_FILE = ".git_guard.json"


def _normalize_lowered(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized.append(text.casefold())
    return tuple(dict.fromkeys(normalized))


def _normalize_exact(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized.append(text)
    return tuple(dict.fromkeys(normalized))


def _normalize_positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return default
    else:
        return default
    return parsed if parsed > 0 else default


def _as_table(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return value
    return {}


@dataclass(frozen=True)
class FilePolicy:
    """Configure file-level blocks and ignore prefixes."""

    ignore_path_prefixes: tuple[str, ...]
    blocked_exact_names: tuple[str, ...]
    blocked_suffixes: tuple[str, ...]


@dataclass(frozen=True)
class ContentPolicy:
    """Configure text-level blocks and secret-like token checks."""

    blocked_terms: tuple[str, ...]
    secret_prefixes: tuple[str, ...]
    sensitive_key_fragments: tuple[str, ...]
    placeholder_values: tuple[str, ...]
    secret_min_length: int


@dataclass(frozen=True)
class PhonePolicy:
    """Configure conservative phone-like token detection."""

    min_digits: int
    max_digits: int


@dataclass(frozen=True)
class GuardPolicy:
    """Bundle all git guard policy sections."""

    policy_path: Path
    max_issues: int
    files: FilePolicy
    content: ContentPolicy
    phones: PhonePolicy

    def ignores_path(self, path: str) -> bool:
        """Return True when the policy should skip this repo-relative path."""

        lowered = path.casefold()
        return any(lowered.startswith(prefix) for prefix in self.files.ignore_path_prefixes)


def load_policy(repo_root: Path, *, policy_path: Path | None = None) -> GuardPolicy:
    """Load a guard policy from disk, falling back to safe defaults."""

    resolved_policy_path = (policy_path or (repo_root / DEFAULT_POLICY_FILE)).resolve()
    raw: dict[str, object] = {}
    if resolved_policy_path.exists():
        with resolved_policy_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            raw = loaded

    guard_table = _as_table(raw.get("guard"))
    path_table = _as_table(raw.get("paths"))
    content_table = _as_table(raw.get("content"))
    phone_table = _as_table(raw.get("phones"))

    return GuardPolicy(
        policy_path=resolved_policy_path,
        max_issues=_normalize_positive_int(guard_table.get("max_issues"), default=50),
        files=FilePolicy(
            ignore_path_prefixes=_normalize_lowered(path_table.get("ignore_path_prefixes"))
            or (".git/", ".venv/", "artifacts/", "state/"),
            blocked_exact_names=_normalize_lowered(path_table.get("blocked_exact_names"))
            or (".env", ".env.pi", ".env.chonkydb", ".env.twinr-proxy", ".voice_gateway_alias.env"),
            blocked_suffixes=_normalize_lowered(path_table.get("blocked_suffixes"))
            or (".pem", ".key", ".p12", ".pfx", ".kdbx", ".mobileprovision"),
        ),
        content=ContentPolicy(
            blocked_terms=_normalize_lowered(content_table.get("blocked_terms")) or ("chaos", "warhammer"),
            secret_prefixes=_normalize_exact(content_table.get("secret_prefixes"))
            or ("sk-proj-", "sk-", "ghp_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xapp-", "AIza"),
            sensitive_key_fragments=_normalize_lowered(content_table.get("sensitive_key_fragments"))
            or (
                "api_key",
                "apikey",
                "secret",
                "token",
                "password",
                "passwd",
                "pwd",
                "private_key",
                "access_key",
                "session_key",
                "auth_key",
            ),
            placeholder_values=_normalize_lowered(content_table.get("placeholder_values"))
            or (
                "",
                "example",
                "example-value",
                "placeholder",
                "placeholder-value",
                "dummy",
                "test",
                "fake",
                "secret-key",
                "token",
                "redacted",
                "<redacted>",
                "your-value-here",
                "changeme",
                "replace-me",
                "...",
            ),
            secret_min_length=_normalize_positive_int(content_table.get("secret_min_length"), default=12),
        ),
        phones=PhonePolicy(
            min_digits=_normalize_positive_int(phone_table.get("min_digits"), default=7),
            max_digits=_normalize_positive_int(phone_table.get("max_digits"), default=15),
        ),
    )
