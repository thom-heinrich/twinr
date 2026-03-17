"""Run the hardened child-side loader for one sandboxed skill handler."""

from __future__ import annotations

import asyncio
import inspect
import os
import pickle
from pathlib import Path
from typing import Any

from twinr.agent.self_coding.sandbox.broker import SandboxSkillContextProxy
from twinr.agent.self_coding.sandbox.os_hardening import (
    SandboxHardeningLimits,
    SandboxHardeningReport,
    apply_baseline_os_hardening,
    apply_post_load_landlock,
)
from twinr.agent.self_coding.sandbox.trusted_loader import load_trusted_skill_module


_MAX_IPC_MESSAGE_BYTES = 1 * 1024 * 1024


def sandbox_loader_child_main(
    *,
    connection: Any,
    source_text: str,
    entry_module: str,
    handler_name: str,
    event_name: str | None,
    materialized_root: str,
    limits: Any,
) -> None:
    """Execute one trusted skill handler inside the hardened child process."""

    hardening = SandboxHardeningReport()
    try:
        root = Path(materialized_root).resolve(strict=True)
        hardening = apply_baseline_os_hardening(
            limits=SandboxHardeningLimits(
                cpu_seconds=int(getattr(limits, "cpu_seconds")),
                address_space_bytes=int(getattr(limits, "address_space_bytes")),
                max_open_files=int(getattr(limits, "max_open_files")),
            ),
            working_directory=root,
            keep_fds=_connection_keep_fds(connection),
        )
        proxy = SandboxSkillContextProxy(connection=connection)
        module = load_trusted_skill_module(source_text=source_text, filename=entry_module)
        hardening = apply_post_load_landlock(report=hardening, readable_root=root)
        _send_message(connection, {"kind": "loader_ready", "hardening": hardening.to_payload()})
        handler = module.get_handler(handler_name)
        call_args, call_kwargs = _build_handler_call(handler, proxy, event_name)
        result = handler(*call_args, **call_kwargs)
        if inspect.isawaitable(result):
            asyncio.run(result)
        _send_message(connection, {"kind": "completed", "hardening": hardening.to_payload()})
    except BaseException as exc:
        _try_send_message(
            connection,
            {
                "kind": "failed",
                "error": _normalize_error_text(exc),
                "hardening": hardening.to_payload(),
            },
        )
    finally:
        _close_connection(connection)


def _connection_keep_fds(connection: Any) -> tuple[int, ...]:
    fileno = getattr(connection, "fileno", None)
    if not callable(fileno):
        return ()
    try:
        return (int(fileno()),)
    except (TypeError, ValueError, OSError):
        return ()


def _build_handler_call(handler: Any, context: SandboxSkillContextProxy, event_name: str | None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    signature = inspect.signature(handler)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        return (), {}
    first = parameters[0]
    if first.kind not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
        raise RuntimeError("sandbox skill handler must accept ctx as the first positional parameter")
    args: list[Any] = [context]
    kwargs: dict[str, Any] = {}
    if len(parameters) >= 2:
        second = parameters[1]
        if second.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            args.append(event_name)
        elif second.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[second.name] = event_name
    return tuple(args), kwargs


def _send_message(connection: Any, message: dict[str, Any]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_IPC_MESSAGE_BYTES:
        raise RuntimeError("sandbox IPC payload is too large")
    connection.send_bytes(payload)


def _try_send_message(connection: Any, message: dict[str, Any]) -> None:
    try:
        _send_message(connection, message)
    except Exception:
        pass


def _close_connection(connection: Any) -> None:
    close = getattr(connection, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _normalize_error_text(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:512]
