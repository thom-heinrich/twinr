"""Support helpers for the server-rendered WhatsApp setup wizard."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shlex
import subprocess
import time

from twinr.agent.base_agent import TwinrConfig
from twinr.channels import ChannelTransportError
from twinr.channels.whatsapp.config import WhatsAppChannelConfig, normalize_whatsapp_digits
from twinr.channels.whatsapp.node_runtime import resolve_whatsapp_node_binary
from twinr.channels.whatsapp.worker_bridge import (
    WhatsAppWorkerBridge,
    WhatsAppWorkerExitedError,
    WhatsAppWorkerStatusEvent,
)
from twinr.web.support.channel_onboarding import (
    ChannelPairingSnapshot,
    FileBackedChannelOnboardingStore,
    InProcessChannelPairingRegistry,
    _now_iso,
)

_WHATSAPP_CHANNEL_ID = "whatsapp"
_PAIRING_WINDOW_SECONDS = 90.0


@dataclass(frozen=True, slots=True)
class WhatsAppRuntimeProbe:
    """Summarize the local WhatsApp worker prerequisites for the web wizard."""

    node_binary: str
    node_version: str
    node_ready: bool
    node_detail: str
    worker_root: Path
    worker_ready: bool
    worker_detail: str
    auth_dir: Path
    auth_dir_exists: bool
    creds_path: Path
    paired: bool
    pair_detail: str
    launch_command: str


@dataclass(slots=True)
class WhatsAppPairingCoordinator:
    """Run one bounded WhatsApp pairing window behind the web wizard."""

    store: FileBackedChannelOnboardingStore
    registry: InProcessChannelPairingRegistry
    pairing_window_s: float = _PAIRING_WINDOW_SECONDS

    def load_snapshot(self) -> ChannelPairingSnapshot:
        """Load and heal the persisted WhatsApp pairing snapshot."""

        snapshot = self.store.load()
        if not snapshot.running or self.registry.is_running(_WHATSAPP_CHANNEL_ID):
            return snapshot

        finished_at = _now_iso()
        if snapshot.paired:
            healed = replace(
                snapshot,
                phase="paired",
                summary="Paired",
                detail="A stored linked-device session is ready. Open a new pairing window only if you need another QR.",
                running=False,
                qr_needed=False,
                qr_svg=None,
                updated_at=finished_at,
                finished_at=finished_at,
            )
        else:
            healed = replace(
                snapshot,
                phase="idle",
                summary="Pairing stopped",
                detail="The previous pairing window is no longer running. Start a new pairing window if you still need a QR.",
                running=False,
                qr_needed=False,
                qr_svg=None,
                updated_at=finished_at,
                finished_at=finished_at,
            )
        return self.store.save(healed)

    def start_pairing(self, config: TwinrConfig) -> bool:
        """Start one bounded pairing job unless another one is already running."""

        return self.registry.start(_WHATSAPP_CHANNEL_ID, lambda: self._run_pairing(config))

    def _run_pairing(self, config: TwinrConfig) -> None:
        transport: WhatsAppWorkerBridge | None = None
        snapshot = self.store.save(
            ChannelPairingSnapshot(
                channel_id=_WHATSAPP_CHANNEL_ID,
                phase="starting",
                summary="Starting pairing",
                detail="Twinr is opening a temporary WhatsApp pairing window.",
                running=True,
                updated_at=_now_iso(),
                started_at=_now_iso(),
            )
        )

        try:
            whatsapp_config = WhatsAppChannelConfig.from_twinr_config(config)
            transport = WhatsAppWorkerBridge(whatsapp_config)
            transport.start()
        except Exception as exc:
            self.store.save(
                _failed_snapshot(
                    snapshot,
                    detail=_startup_failure_detail(exc),
                    exit_code=_extract_exit_code(exc),
                    auth_repair_needed=False,
                )
            )
            return

        deadline = time.monotonic() + max(5.0, float(self.pairing_window_s))
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self.store.save(_timeout_snapshot(snapshot))
                    return

                try:
                    event = transport.next_event(timeout_s=min(1.0, max(0.05, remaining)))
                except WhatsAppWorkerExitedError as exc:
                    self.store.save(
                        _failed_snapshot(
                            snapshot,
                            detail=_worker_exit_detail(snapshot, exc),
                            exit_code=exc.exit_code,
                            auth_repair_needed=snapshot.auth_repair_needed,
                        )
                    )
                    return

                if event is None:
                    continue

                snapshot = self.store.save(_snapshot_for_status_event(snapshot, event))
                if snapshot.paired or snapshot.fatal:
                    return
        finally:
            if transport is not None:
                transport.stop()


def canonicalize_whatsapp_allow_from(raw_value: str) -> str:
    """Normalize one WhatsApp phone number into ``+<digits>`` form."""

    normalized_digits = normalize_whatsapp_digits(str(raw_value).strip())
    return f"+{normalized_digits}"


def normalize_project_relative_directory(project_root: Path, raw_value: str, *, label: str) -> str:
    """Validate one project-rooted directory path and return it relative to the project."""

    normalized = str(raw_value or "").strip()
    if not normalized:
        raise ValueError(f"Please choose {label}.")

    resolved_root = Path(project_root).expanduser().resolve(strict=False)
    candidate = _resolve_project_path(resolved_root, normalized)
    try:
        relative_candidate = candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the Twinr project folder.") from exc
    return relative_candidate.as_posix()


def probe_whatsapp_runtime(config: TwinrConfig, *, env_path: Path) -> WhatsAppRuntimeProbe:
    """Collect bounded local runtime facts for the WhatsApp wizard."""

    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    node_binary = resolve_whatsapp_node_binary(
        project_root,
        str(getattr(config, "whatsapp_node_binary", "node") or "node"),
    )
    auth_dir = _resolve_project_path(project_root, str(getattr(config, "whatsapp_auth_dir", "")))
    worker_root = _resolve_project_path(project_root, str(getattr(config, "whatsapp_worker_root", "")))

    node_ready, node_version, node_detail = _probe_node_binary(node_binary)

    package_json = worker_root / "package.json"
    worker_entry = worker_root / "index.mjs"
    worker_ready = worker_root.is_dir() and package_json.is_file() and worker_entry.is_file()
    if worker_ready:
        worker_detail = f"Worker package found at {worker_root}."
    elif worker_root.is_dir() and not package_json.is_file():
        worker_detail = f"Worker package.json is missing under {worker_root}."
    elif worker_root.is_dir():
        worker_detail = f"Worker entrypoint is missing under {worker_root}."
    else:
        worker_detail = f"Worker folder is missing: {worker_root}."

    creds_path = auth_dir / "creds.json"
    auth_dir_exists = auth_dir.exists()
    paired = creds_path.is_file()
    if paired:
        pair_detail = f"Stored linked-device session found at {creds_path}."
    elif auth_dir_exists:
        pair_detail = f"Auth folder exists at {auth_dir}, but no linked-device session is stored yet."
    else:
        pair_detail = f"Auth folder does not exist yet. Twinr will create {auth_dir} during the first pairing run."

    return WhatsAppRuntimeProbe(
        node_binary=node_binary,
        node_version=node_version,
        node_ready=node_ready,
        node_detail=node_detail,
        worker_root=worker_root,
        worker_ready=worker_ready,
        worker_detail=worker_detail,
        auth_dir=auth_dir,
        auth_dir_exists=auth_dir_exists,
        creds_path=creds_path,
        paired=paired,
        pair_detail=pair_detail,
        launch_command=_build_launch_command(project_root=project_root, env_path=Path(env_path).resolve(strict=False)),
    )


def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve one project-relative or absolute path without requiring it to exist."""

    candidate = Path(configured_path or "").expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _probe_node_binary(node_binary: str) -> tuple[bool, str, str]:
    """Return readiness, display version, and detail for the configured Node.js binary."""

    try:
        completed = subprocess.run(
            [node_binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (
            False,
            "Unavailable",
            f"Could not run {node_binary!r}: {exc}",
        )

    version_text = (completed.stdout or completed.stderr or "").strip() or "Unavailable"
    if completed.returncode != 0:
        return (
            False,
            version_text,
            f"Could not query {node_binary!r}: {version_text or completed.returncode}",
        )

    major_version = _parse_node_major(version_text)
    if major_version is None:
        return False, version_text, "Could not read the Node.js major version."
    if major_version < 20:
        return False, version_text, f"Baileys needs Node.js 20+. Current runtime is {version_text}."
    return True, version_text, f"Node.js {version_text} is ready for the Baileys worker."


def _parse_node_major(version_text: str) -> int | None:
    """Parse the major version from one ``node --version`` style string."""

    cleaned = str(version_text or "").strip()
    if not cleaned:
        return None
    cleaned = cleaned.lstrip("vV")
    head = cleaned.split(".", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def _build_launch_command(*, project_root: Path, env_path: Path) -> str:
    """Build the operator-facing command used to start the WhatsApp channel loop."""

    venv_python = project_root / ".venv" / "bin" / "python"
    python_command = "./.venv/bin/python" if venv_python.is_file() else "python3"
    return (
        f"cd {shlex.quote(str(project_root))} && "
        f"PYTHONPATH=src {shlex.quote(python_command)} -m twinr "
        f"--env-file {shlex.quote(str(env_path))} --run-whatsapp-channel"
    )


def _snapshot_for_status_event(
    snapshot: ChannelPairingSnapshot,
    event: WhatsAppWorkerStatusEvent,
) -> ChannelPairingSnapshot:
    """Map one worker status event to one persisted operator snapshot."""

    updated_at = _now_iso()
    phase, summary, detail = _status_copy_for_event(event)
    auth_repair_needed = event.fatal and _requires_auth_repair(event.detail)
    paired = event.connection == "open" or snapshot.paired
    running = not paired and not event.fatal
    qr_cleared = event.fatal or paired or event.connection in {"close", "reconnecting"}
    if event.qr_available and running:
        qr_needed = True
        qr_svg = event.qr_svg or snapshot.qr_svg
    elif qr_cleared or not running:
        qr_needed = False
        qr_svg = None
    else:
        qr_needed = snapshot.qr_needed and running
        qr_svg = snapshot.qr_svg if qr_needed else None
    worker_ready = snapshot.worker_ready or event.qr_available or event.connection in {"booting", "connecting", "reconnecting", "open", "close"}
    last_reconnect_reason = event.detail if event.connection == "reconnecting" and event.detail else snapshot.last_reconnect_reason
    reconnect_in_ms = event.reconnect_in_ms if event.connection == "reconnecting" else None

    return replace(
        snapshot,
        phase=phase,
        summary=summary,
        detail=detail,
        running=running,
        qr_needed=qr_needed,
        qr_svg=qr_svg,
        paired=paired,
        fatal=event.fatal,
        auth_repair_needed=auth_repair_needed,
        worker_ready=worker_ready,
        account_id=event.account_jid or snapshot.account_id,
        last_worker_detail=event.detail or summary,
        last_reconnect_reason=last_reconnect_reason,
        reconnect_in_ms=reconnect_in_ms,
        status_code=event.status_code,
        exit_code=None,
        updated_at=updated_at,
        finished_at=updated_at if paired or event.fatal else None,
    )


def _status_copy_for_event(event: WhatsAppWorkerStatusEvent) -> tuple[str, str, str]:
    """Translate one worker event into plain-language phase, summary, and detail."""

    if event.fatal:
        if _requires_auth_repair(event.detail):
            return (
                "failed",
                "Auth repair needed",
                "The stored WhatsApp session is no longer valid. Start a fresh pairing window and scan the next QR.",
            )
        return (
            "failed",
            "Pairing failed",
            f"The pairing worker stopped with: {_event_detail_label(event.detail)}.",
        )

    if event.qr_available:
        return (
            "qr_needed",
            "QR needed",
            "A fresh QR is ready below. In WhatsApp, open Linked devices and scan it now.",
        )
    if event.connection == "booting":
        return (
            "worker_ready",
            "Worker ready",
            "Twinr started the temporary WhatsApp worker and is waiting for WhatsApp to answer.",
        )
    if event.connection == "connecting":
        return (
            "connecting",
            "Connecting",
            "Twinr is opening the WhatsApp linked-device connection.",
        )
    if event.connection == "reconnecting":
        wait_suffix = _reconnect_suffix(event.reconnect_in_ms)
        return (
            "reconnecting",
            "Reconnecting",
            f"WhatsApp closed the connection ({_event_detail_label(event.detail)}) and Twinr is retrying{wait_suffix}.",
        )
    if event.connection == "open":
        account_suffix = f" as {event.account_jid}." if event.account_jid else "."
        return (
            "paired",
            "Paired",
            f"The linked-device session is open{account_suffix} You can move on to the self-chat test.",
        )
    if event.connection == "close":
        return (
            "closed",
            "Connection closed",
            f"The worker closed the current connection with {_event_detail_label(event.detail)}.",
        )
    return (
        event.connection or "running",
        "Waiting",
        "Twinr is waiting for the next WhatsApp worker update.",
    )


def _timeout_snapshot(snapshot: ChannelPairingSnapshot) -> ChannelPairingSnapshot:
    """Close one bounded pairing window when it reaches its deadline."""

    updated_at = _now_iso()
    if snapshot.paired:
        return replace(
            snapshot,
            running=False,
            qr_needed=False,
            qr_svg=None,
            updated_at=updated_at,
            finished_at=updated_at,
        )

    detail = (
        "The QR wait window closed before WhatsApp finished pairing. Start a new pairing window if you still need a QR."
        if snapshot.qr_needed
        else "The temporary pairing window ended without a stored linked-device session."
    )
    return replace(
        snapshot,
        phase="timed_out",
        summary="Pairing window closed",
        detail=detail,
        running=False,
        qr_needed=False,
        qr_svg=None,
        fatal=False,
        updated_at=updated_at,
        finished_at=updated_at,
    )


def _failed_snapshot(
    snapshot: ChannelPairingSnapshot,
    *,
    detail: str,
    exit_code: int | None,
    auth_repair_needed: bool,
) -> ChannelPairingSnapshot:
    """Build one terminal failed snapshot from a startup or worker error."""

    updated_at = _now_iso()
    summary = "Auth repair needed" if auth_repair_needed else "Pairing failed"
    return replace(
        snapshot,
        phase="failed",
        summary=summary,
        detail=detail,
        running=False,
        qr_needed=False,
        qr_svg=None,
        fatal=True,
        auth_repair_needed=auth_repair_needed,
        exit_code=exit_code,
        updated_at=updated_at,
        finished_at=updated_at,
    )


def _startup_failure_detail(exc: Exception) -> str:
    """Convert one startup failure into operator-facing wizard copy."""

    if isinstance(exc, ChannelTransportError):
        return str(exc)
    return f"Twinr could not start the temporary WhatsApp worker: {exc}"


def _worker_exit_detail(snapshot: ChannelPairingSnapshot, exc: WhatsAppWorkerExitedError) -> str:
    """Convert one unexpected worker exit into operator-facing copy."""

    if snapshot.auth_repair_needed:
        return "The stored WhatsApp session needs repair. Start a fresh pairing window and scan a new QR."
    if snapshot.qr_needed:
        return "The pairing worker stopped before WhatsApp completed the QR scan."
    return f"The pairing worker exited unexpectedly with code {exc.exit_code}."


def _extract_exit_code(exc: Exception) -> int | None:
    """Extract an optional worker exit code from one startup exception."""

    if isinstance(exc, WhatsAppWorkerExitedError):
        return exc.exit_code
    return None


def _requires_auth_repair(detail: str | None) -> bool:
    """Return whether one worker detail means the saved auth session is broken."""

    normalized = str(detail or "").strip().lower()
    return normalized in {"badsession", "loggedout"}


def _event_detail_label(detail: str | None) -> str:
    """Render one optional worker detail in plain language."""

    text = str(detail or "").strip()
    return text or "an unknown reason"


def _reconnect_suffix(reconnect_in_ms: int | None) -> str:
    """Format the reconnect countdown suffix for operator copy."""

    if reconnect_in_ms is None or reconnect_in_ms <= 0:
        return ""
    seconds = max(1, round(reconnect_in_ms / 1000))
    return f" in about {seconds} seconds"
