from __future__ import annotations

from typing import Any

from twinr.agent.base_agent.config import TwinrConfig


def default_groq_client(config: TwinrConfig) -> Any:
    api_key = (config.groq_api_key or "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to use the Groq provider")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` in /twinr first."
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=_default_groq_base_url(config),
        timeout=config.groq_timeout_s,
    )


def _default_groq_base_url(config: TwinrConfig) -> str:
    return (config.groq_base_url or "https://api.groq.com/openai/v1").rstrip("/")
