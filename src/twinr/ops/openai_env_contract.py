"""Validate Twinr's OpenAI environment contract for Pi acceptance runs.

This module owns fail-closed checks for the local `.env` contract that Pi-side
operator probes and acceptance scripts rely on. It does not execute provider
requests itself; instead it proves that the env file is present, parseable,
contains the required OpenAI credentials, and can initialize the shared OpenAI
client without one-off shell injection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.providers.openai.core.client import _default_client_factory
from twinr.web.support.store import read_env_values


@dataclass(frozen=True, slots=True)
class OpenAIEnvContractStatus:
    """Describe whether one Twinr `.env` file satisfies the OpenAI contract.

    Attributes:
        env_path: Canonical env-file path that was checked.
        ok: Whether the env file satisfied every required contract step.
        env_exists: Whether the env file exists on disk.
        env_line_count: Count of physical newline-delimited lines in the file.
        parsed_key_count: Count of parsed env assignments returned by Twinr's
            hardened env reader.
        literal_newline_collapse_detected: Whether the file appears to contain
            a collapsed single-line payload with literal ``\\n`` sequences.
        openai_api_key_present: Whether ``OPENAI_API_KEY`` is available after
            parsing the env file.
        openai_project_id_present: Whether ``OPENAI_PROJ_ID`` is available
            after parsing the env file.
        config_loaded: Whether ``TwinrConfig.from_env`` succeeded.
        client_initialized: Whether the default OpenAI client could be
            constructed directly from the env-backed config.
        issues: Stable machine-readable issue identifiers for failed checks.
        detail: Plain-language summary of the first blocking failure or the
            success state.
    """

    env_path: str
    ok: bool
    env_exists: bool
    env_line_count: int
    parsed_key_count: int
    literal_newline_collapse_detected: bool
    openai_api_key_present: bool
    openai_project_id_present: bool
    config_loaded: bool
    client_initialized: bool
    issues: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the status."""

        return asdict(self)


def check_openai_env_contract(env_path: str | Path) -> OpenAIEnvContractStatus:
    """Validate one Twinr `.env` file for direct OpenAI-provider use.

    Args:
        env_path: Path to the env file that should back Pi acceptance or
            provider probes.

    Returns:
        A normalized contract status. The check is fail-closed: any parse,
        credential, or client-initialization failure turns ``ok`` false and is
        recorded in ``issues``.
    """

    normalized_path = Path(env_path).expanduser().resolve(strict=False)
    issues: list[str] = []
    env_exists = normalized_path.is_file()
    raw_text = ""
    env_values: dict[str, str] = {}
    config_loaded = False
    client_initialized = False
    openai_api_key_present = False
    openai_project_id_present = False
    config: TwinrConfig | None = None

    if not env_exists:
        issues.append("env_file_missing")
    else:
        raw_text = normalized_path.read_text(encoding="utf-8")
        try:
            env_values = read_env_values(normalized_path)
        except Exception:
            issues.append("env_parse_failed")
            env_values = {}

    env_line_count = raw_text.count("\n")
    literal_newline_collapse_detected = _looks_like_literal_newline_collapse(raw_text)
    if literal_newline_collapse_detected:
        issues.append("env_literal_newline_collapse")

    openai_api_key_present = bool((env_values.get("OPENAI_API_KEY") or "").strip())
    openai_project_id_present = bool((env_values.get("OPENAI_PROJ_ID") or "").strip())
    if env_exists and not openai_api_key_present:
        issues.append("openai_api_key_missing")
    if env_exists and not openai_project_id_present:
        issues.append("openai_project_id_missing")

    if env_exists:
        try:
            config = TwinrConfig.from_env(normalized_path)
            config_loaded = True
        except Exception:
            issues.append("twinr_config_from_env_failed")
            config = None

    if config is not None:
        try:
            _default_client_factory(config)
            client_initialized = True
        except Exception:
            issues.append("openai_client_init_failed")

    deduped_issues = tuple(dict.fromkeys(issues))
    detail = _status_detail(deduped_issues)
    return OpenAIEnvContractStatus(
        env_path=str(normalized_path),
        ok=not deduped_issues,
        env_exists=env_exists,
        env_line_count=env_line_count,
        parsed_key_count=len(env_values),
        literal_newline_collapse_detected=literal_newline_collapse_detected,
        openai_api_key_present=openai_api_key_present,
        openai_project_id_present=openai_project_id_present,
        config_loaded=config_loaded,
        client_initialized=client_initialized,
        issues=deduped_issues,
        detail=detail,
    )


def _looks_like_literal_newline_collapse(raw_text: str) -> bool:
    """Return whether text resembles a one-line `.env` collapsed via ``\\n``.

    The Pi backup artifact observed during acceptance held an entire env file in
    one physical line with many literal ``\\n`` escape sequences. This check
    flags that specific corruption shape without treating ordinary multiline
    values as broken.
    """

    if not raw_text:
        return False
    physical_lines = raw_text.count("\n")
    literal_newlines = raw_text.count("\\n")
    return physical_lines <= 2 and literal_newlines >= 2


def _status_detail(issues: tuple[str, ...]) -> str:
    """Render a concise operator-facing detail string for one status."""

    if not issues:
        return "Env file is parseable, carries OpenAI credentials, and initializes the OpenAI client directly."
    first_issue = issues[0]
    messages = {
        "env_file_missing": "Env file is missing.",
        "env_parse_failed": "Env file could not be parsed safely.",
        "env_literal_newline_collapse": "Env file appears to contain collapsed literal newline sequences.",
        "openai_api_key_missing": "OPENAI_API_KEY is missing from the env file.",
        "openai_project_id_missing": "OPENAI_PROJ_ID is missing from the env file.",
        "twinr_config_from_env_failed": "TwinrConfig could not be loaded from the env file.",
        "openai_client_init_failed": "OpenAI client initialization failed from the env-backed config.",
    }
    return messages.get(first_issue, "OpenAI env contract validation failed.")
