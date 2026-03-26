"""Start and surface bounded service-connect flows from Twinr runtime code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.channels.onboarding import ChannelPairingSnapshot, FileBackedChannelOnboardingStore, InProcessChannelPairingRegistry
from twinr.display.service_connect_cues import DisplayServiceConnectCue, DisplayServiceConnectCueStore


_SERVICE_LABELS: dict[str, str] = {
    "whatsapp": "WhatsApp",
}
_SERVICE_ALIASES: dict[str, str] = {
    "wa": "whatsapp",
    "whatsapp": "whatsapp",
}
_PAIRING_REGISTRY = InProcessChannelPairingRegistry()
_WHATSAPP_CHANNEL_ID = "whatsapp"


@dataclass(frozen=True, slots=True)
class ServiceConnectRequestResult:
    """Describe the outcome of one runtime service-connect request."""

    status: str
    service_id: str
    service_label: str
    supported: bool
    started: bool
    running: bool
    paired: bool
    phase: str
    summary: str
    detail: str
    qr_visible: bool
    display_surface: str = "right_info_panel"

    def to_dict(self) -> dict[str, object]:
        """Serialize the result into one JSON-safe tool payload."""

        return {
            "status": self.status,
            "service": self.service_id,
            "service_label": self.service_label,
            "supported": self.supported,
            "supported_services": sorted(_SERVICE_LABELS),
            "started": self.started,
            "running": self.running,
            "paired": self.paired,
            "phase": self.phase,
            "summary": self.summary,
            "detail": self.detail,
            "qr_visible": self.qr_visible,
            "display_surface": self.display_surface,
        }


def start_service_connect_flow(config: TwinrConfig, *, service: str) -> ServiceConnectRequestResult:
    """Start or summarize one bounded service-connect flow for a named service."""

    raw_service = str(service or "").strip()
    if not raw_service:
        raise ValueError("Service name is required.")

    resolved_service_id = _resolve_service_id(raw_service)
    if resolved_service_id is None:
        service_label = raw_service or "Service"
        result = ServiceConnectRequestResult(
            status="unsupported",
            service_id=_service_slug(raw_service) or "unsupported",
            service_label=service_label,
            supported=False,
            started=False,
            running=False,
            paired=False,
            phase="unsupported",
            summary="Not supported yet",
            detail="That service is not available yet. Current supported service: WhatsApp.",
            qr_visible=False,
        )
        _show_result_on_display(config, result)
        return result

    if resolved_service_id == _WHATSAPP_CHANNEL_ID:
        return _start_whatsapp_service_connect(config)

    service_label = _service_label(resolved_service_id)
    result = ServiceConnectRequestResult(
        status="unsupported",
        service_id=resolved_service_id,
        service_label=service_label,
        supported=False,
        started=False,
        running=False,
        paired=False,
        phase="unsupported",
        summary="Not supported yet",
        detail="That service is not available yet.",
        qr_visible=False,
    )
    _show_result_on_display(config, result)
    return result


def _start_whatsapp_service_connect(config: TwinrConfig) -> ServiceConnectRequestResult:
    from twinr.channels.whatsapp.pairing import WhatsAppPairingCoordinator, probe_whatsapp_runtime

    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    store = FileBackedChannelOnboardingStore.from_project_root(project_root, channel_id=_WHATSAPP_CHANNEL_ID)
    cue_store = DisplayServiceConnectCueStore.from_config(config)
    pairing_window_s = 90.0

    def observer(snapshot: ChannelPairingSnapshot) -> None:
        _sync_whatsapp_snapshot_to_display(
            cue_store=cue_store,
            snapshot=snapshot,
            pairing_window_s=pairing_window_s,
        )

    coordinator = WhatsAppPairingCoordinator(
        store=store,
        registry=_PAIRING_REGISTRY,
        snapshot_observer=observer,
    )
    coordinator.load_snapshot()
    probe = probe_whatsapp_runtime(config, env_path=project_root / ".env")

    if not probe.node_ready:
        result = ServiceConnectRequestResult(
            status="unavailable",
            service_id=_WHATSAPP_CHANNEL_ID,
            service_label=_service_label(_WHATSAPP_CHANNEL_ID),
            supported=True,
            started=False,
            running=False,
            paired=False,
            phase="unavailable",
            summary="Node runtime missing",
            detail=probe.node_detail,
            qr_visible=False,
        )
        _show_result_on_display(config, result)
        return result

    if not probe.worker_ready:
        result = ServiceConnectRequestResult(
            status="unavailable",
            service_id=_WHATSAPP_CHANNEL_ID,
            service_label=_service_label(_WHATSAPP_CHANNEL_ID),
            supported=True,
            started=False,
            running=False,
            paired=False,
            phase="unavailable",
            summary="Worker not ready",
            detail=probe.worker_detail,
            qr_visible=False,
        )
        _show_result_on_display(config, result)
        return result

    if probe.paired:
        result = ServiceConnectRequestResult(
            status="already_connected",
            service_id=_WHATSAPP_CHANNEL_ID,
            service_label=_service_label(_WHATSAPP_CHANNEL_ID),
            supported=True,
            started=False,
            running=False,
            paired=True,
            phase="paired",
            summary="Already connected",
            detail=probe.pair_detail,
            qr_visible=False,
        )
        _show_result_on_display(config, result)
        return result

    started = coordinator.start_pairing(config)
    if started:
        cue_store.save(
            DisplayServiceConnectCue(
                source="service_connect",
                service_id=_WHATSAPP_CHANNEL_ID,
                service_label=_service_label(_WHATSAPP_CHANNEL_ID),
                phase="starting",
                summary="Starting pairing",
                detail="Twinr is opening the WhatsApp pairing flow. The QR will appear here on the right as soon as it is ready.",
                accent="info",
            ),
            hold_seconds=max(30.0, float(coordinator.pairing_window_s) + 10.0),
        )
        return ServiceConnectRequestResult(
            status="started",
            service_id=_WHATSAPP_CHANNEL_ID,
            service_label=_service_label(_WHATSAPP_CHANNEL_ID),
            supported=True,
            started=True,
            running=True,
            paired=False,
            phase="starting",
            summary="Starting pairing",
            detail="Twinr is opening the WhatsApp pairing flow. The QR will appear on the right info panel as soon as it is ready.",
            qr_visible=False,
        )

    latest = coordinator.load_snapshot()
    return ServiceConnectRequestResult(
        status="already_running",
        service_id=_WHATSAPP_CHANNEL_ID,
        service_label=_service_label(_WHATSAPP_CHANNEL_ID),
        supported=True,
        started=False,
        running=latest.running,
        paired=latest.paired or probe.paired,
        phase=latest.phase,
        summary=latest.summary if latest.summary != "Not started" else "Pairing already running",
        detail=(
            latest.detail
            if latest.detail != "No pairing window has been started yet."
            else "Twinr already has an active WhatsApp pairing flow."
        ),
        qr_visible=bool(latest.qr_needed and latest.qr_image_data_url),
    )


def _resolve_service_id(raw_service: str) -> str | None:
    slug = _service_slug(raw_service)
    if not slug:
        return None
    return _SERVICE_ALIASES.get(slug)


def _service_slug(raw_service: str) -> str:
    return "".join(character for character in str(raw_service or "").strip().lower() if character.isalnum())


def _service_label(service_id: str) -> str:
    return _SERVICE_LABELS.get(service_id, service_id.title())


def _show_result_on_display(config: TwinrConfig, result: ServiceConnectRequestResult) -> None:
    cue_store = DisplayServiceConnectCueStore.from_config(config)
    cue_store.save(
        DisplayServiceConnectCue(
            source="service_connect",
            service_id=result.service_id,
            service_label=result.service_label,
            phase=result.phase,
            summary=result.summary,
            detail=result.detail,
            accent=_result_accent(result),
        ),
        hold_seconds=25.0,
    )


def _sync_whatsapp_snapshot_to_display(
    *,
    cue_store: DisplayServiceConnectCueStore,
    snapshot: ChannelPairingSnapshot,
    pairing_window_s: float,
) -> None:
    if (
        not snapshot.running
        and not snapshot.qr_needed
        and not snapshot.fatal
        and not snapshot.paired
        and snapshot.phase in {"idle", "", "timed_out"}
    ):
        cue_store.clear()
        return

    cue_store.save(
        DisplayServiceConnectCue(
            source="service_connect",
            service_id=_WHATSAPP_CHANNEL_ID,
            service_label=_service_label(_WHATSAPP_CHANNEL_ID),
            phase=snapshot.phase,
            summary=snapshot.summary,
            detail=snapshot.detail,
            qr_image_data_url=snapshot.qr_image_data_url if snapshot.qr_needed else None,
            accent=_snapshot_accent(snapshot),
        ),
        hold_seconds=max(25.0, float(pairing_window_s) + 10.0) if snapshot.running else 25.0,
    )


def _snapshot_accent(snapshot: ChannelPairingSnapshot) -> str:
    if snapshot.fatal or snapshot.auth_repair_needed:
        return "alert"
    if snapshot.paired:
        return "success"
    if snapshot.qr_needed:
        return "warm"
    return "info"


def _result_accent(result: ServiceConnectRequestResult) -> str:
    if result.paired:
        return "success"
    if result.status in {"unsupported", "unavailable"}:
        return "alert"
    return "info"


__all__ = [
    "ServiceConnectRequestResult",
    "start_service_connect_flow",
]
