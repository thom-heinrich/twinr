from __future__ import annotations

from typing import Any

from twinr.agent.base_agent.config import TwinrConfig


def _default_client_factory(config: TwinrConfig) -> Any:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI backend")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` in /twinr first."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
    if _should_send_project_header(config):
        kwargs["project"] = config.openai_project_id
    return OpenAI(**kwargs)


def _should_send_project_header(config: TwinrConfig) -> bool:
    if not config.openai_project_id:
        return False
    if config.openai_send_project_header is not None:
        return config.openai_send_project_header
    api_key = config.openai_api_key or ""
    return not api_key.startswith("sk-proj-")
