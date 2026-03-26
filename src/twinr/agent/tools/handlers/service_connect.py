"""Handle bounded service-connect requests for runtime tool calls."""

from __future__ import annotations

from typing import Any

from twinr.channels.service_connect import start_service_connect_flow

from .handler_telemetry import emit_best_effort, record_event_best_effort


def handle_connect_service_integration(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Start or summarize a supported service-connect flow from a tool payload."""

    service_name = str(arguments.get("service", "")).strip()
    if not service_name:
        raise RuntimeError("connect_service_integration requires `service`")

    result = start_service_connect_flow(owner.config, service=service_name)
    emit_best_effort(owner, "service_connect_tool_call=true")
    emit_best_effort(owner, f"service_connect_status={result.status}")
    emit_best_effort(owner, f"service_connect_service={result.service_id}")
    record_event_best_effort(
        owner,
        "service_connect_requested",
        "Twinr handled a bounded service-connect request.",
        {
            "service": result.service_id,
            "status": result.status,
            "phase": result.phase,
            "started": result.started,
            "running": result.running,
            "paired": result.paired,
            "qr_visible": result.qr_visible,
        },
    )
    return result.to_dict()
