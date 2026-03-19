"""Build the template-ready WhatsApp setup wizard context."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twinr.agent.base_agent import TwinrConfig
from twinr.web.support.channel_onboarding import ChannelPairingSnapshot
from twinr.web.support.contracts import DetailMetric, WizardCheckRow, WizardStep
from twinr.web.support.forms import _text_field
from twinr.web.support.whatsapp import canonicalize_whatsapp_allow_from, probe_whatsapp_runtime

_WIZARD_STEP_KEYS = ("chat", "runtime", "pairing", "test")


def build_whatsapp_wizard_page_context(
    config: TwinrConfig,
    env_values: Mapping[str, object] | None,
    *,
    env_path: str | Path,
    pairing_snapshot: ChannelPairingSnapshot | None = None,
    requested_step: str | None = None,
) -> dict[str, object]:
    """Build the full page context for the WhatsApp self-chat wizard."""

    normalized_env_values = _normalized_env_values(env_values)
    probe = probe_whatsapp_runtime(config, env_path=Path(env_path))
    snapshot = pairing_snapshot if isinstance(pairing_snapshot, ChannelPairingSnapshot) else ChannelPairingSnapshot.initial("whatsapp")

    raw_allow_from = normalized_env_values.get("TWINR_WHATSAPP_ALLOW_FROM", "")
    allow_from_label = _allow_from_label(raw_allow_from)
    allow_from_ready = allow_from_label != "Not set yet"

    guardrails_ready = bool(config.whatsapp_self_chat_mode) and not bool(config.whatsapp_groups_enabled)
    runtime_ready = probe.node_ready and probe.worker_ready
    auth_repair_needed = bool(snapshot.auth_repair_needed)
    pairing_ready = (probe.paired or snapshot.paired) and not auth_repair_needed
    test_ready = allow_from_ready and guardrails_ready and runtime_ready and pairing_ready

    pairing_window_summary, pairing_window_detail, pairing_window_status = _pairing_window_display(
        snapshot,
        pairing_ready=pairing_ready,
    )
    linked_device_summary, linked_device_detail, linked_device_status = _linked_device_display(
        probe,
        snapshot,
        pairing_ready=pairing_ready,
    )
    worker_process_summary, worker_process_detail, worker_process_status = _worker_process_display(snapshot)
    reconnect_summary, reconnect_detail, reconnect_status = _reconnect_display(snapshot)

    default_step = _default_step(
        allow_from_ready=allow_from_ready,
        runtime_ready=runtime_ready,
        pairing_ready=pairing_ready,
        pairing_running=snapshot.running,
        auth_repair_needed=auth_repair_needed,
        test_ready=test_ready,
    )
    current_step = _sanitize_step(requested_step, fallback=default_step)

    if snapshot.running and snapshot.qr_needed:
        overall_status = "warn"
        overall_status_label = "QR needed"
        overall_detail = (
            "The pairing window is open right now. Scan the QR shown on this page with WhatsApp Linked devices. "
            "This page refreshes automatically while Twinr waits."
        )
    elif snapshot.running:
        overall_status = "warn"
        overall_status_label = "Pairing live"
        overall_detail = (
            "Twinr is opening a temporary pairing window and waiting for WhatsApp to answer. "
            "Stay on this page until the status changes to QR needed or Paired."
        )
    elif auth_repair_needed:
        overall_status = "fail"
        overall_status_label = "Needs repair"
        overall_detail = (
            "The stored WhatsApp linked-device session is no longer valid. "
            "Start a fresh pairing window in step 3 and scan the next QR."
        )
    elif test_ready:
        overall_status = "ok"
        overall_status_label = "Ready"
        overall_detail = (
            "Twinr is configured for one self-chat with your own WhatsApp number. "
            "Open your 'message yourself' chat and send a short test message."
        )
    elif allow_from_ready and runtime_ready:
        overall_status = "warn"
        overall_status_label = "Needs pairing"
        overall_detail = (
            "The allowlist and worker runtime are ready. "
            "Open a temporary pairing window in step 3, scan the QR, and then return to the self-chat test."
        )
    else:
        overall_status = "blocked"
        overall_status_label = "Needs setup"
        overall_detail = (
            "Finish the saved number and the worker runtime first. "
            "This flow stays limited to one internal self-chat and keeps group chats blocked."
        )

    summary_metrics = (
        DetailMetric(
            label="Allowed chat",
            value=allow_from_label,
            detail="Twinr accepts only one direct chat for this internal self-chat test.",
        ),
        DetailMetric(
            label="Worker runtime",
            value="Ready" if runtime_ready else "Needs fix",
            detail=f"{probe.node_detail} {probe.worker_detail}".strip(),
        ),
        DetailMetric(
            label="Pairing window",
            value=pairing_window_summary,
            detail=pairing_window_detail,
        ),
        DetailMetric(
            label="Linked device",
            value=linked_device_summary,
            detail=linked_device_detail,
        ),
    )

    chat_step_status = "ok" if allow_from_ready and guardrails_ready else ("warn" if not allow_from_ready else "fail")
    runtime_step_status = "ok" if runtime_ready else ("warn" if probe.node_ready or probe.worker_ready else "fail")
    pairing_step_status = (
        "fail"
        if auth_repair_needed
        else "ok"
        if pairing_ready
        else "warn"
        if runtime_ready and allow_from_ready and guardrails_ready
        else "blocked"
    )
    pairing_action_enabled = allow_from_ready and guardrails_ready and runtime_ready and not snapshot.running
    pairing_action_hint = _pairing_action_hint(
        runtime_ready=runtime_ready,
        allow_from_ready=allow_from_ready,
        guardrails_ready=guardrails_ready,
        snapshot=snapshot,
    )
    pairing_qr_src, pairing_qr_alt, pairing_qr_detail = _pairing_qr_media(snapshot)

    pairing_checks: list[WizardCheckRow] = [
        WizardCheckRow(
            label="Pairing window",
            summary=pairing_window_summary,
            detail=pairing_window_detail,
            status=pairing_window_status,
        ),
        WizardCheckRow(
            label="Linked-device session",
            summary=linked_device_summary,
            detail=linked_device_detail,
            status=linked_device_status,
        ),
        WizardCheckRow(
            label="Worker process",
            summary=worker_process_summary,
            detail=worker_process_detail,
            status=worker_process_status,
        ),
        WizardCheckRow(
            label="Last reconnect",
            summary=reconnect_summary,
            detail=reconnect_detail,
            status=reconnect_status,
        ),
    ]
    if snapshot.account_id:
        pairing_checks.append(
            WizardCheckRow(
                label="Linked account",
                summary=snapshot.account_id,
                detail="This is the WhatsApp account currently linked to the temporary worker session.",
                status="ok" if pairing_ready else "muted",
            )
        )

    steps = (
        WizardStep(
            key="chat",
            index=1,
            title="Choose your own WhatsApp chat",
            description="Save the one phone number that Twinr may listen to.",
            status=chat_step_status,
            status_label="Saved" if chat_step_status == "ok" else ("Needed" if chat_step_status == "warn" else "Fix guardrails"),
            detail=(
                "For this flow, the allowlisted number should be your own mobile number. "
                "Saving this step also forces self-chat mode on and keeps groups blocked."
            ),
            fields=(
                _text_field(
                    "TWINR_WHATSAPP_ALLOW_FROM",
                    "Your WhatsApp number",
                    normalized_env_values,
                    "",
                    help_text="Use international format such as +491711234567.",
                    tooltip_text="Twinr will accept only this one direct-message sender.",
                    placeholder="+491711234567",
                ),
            ),
            checks=(
                WizardCheckRow(
                    label="Allowed number",
                    summary=allow_from_label,
                    detail="Twinr listens only to this one direct chat.",
                    status="ok" if allow_from_ready else "warn",
                ),
                WizardCheckRow(
                    label="Self-chat mode",
                    summary="Enabled" if config.whatsapp_self_chat_mode else "Still off",
                    detail="Twinr may answer only inside your own chat with yourself.",
                    status="ok" if config.whatsapp_self_chat_mode else ("warn" if not allow_from_ready else "fail"),
                ),
                WizardCheckRow(
                    label="Groups",
                    summary="Blocked" if not config.whatsapp_groups_enabled else "Still enabled",
                    detail="Groups stay off for this internal test setup.",
                    status="ok" if not config.whatsapp_groups_enabled else "fail",
                ),
            ),
            action="save_chat",
            action_label="Save chat step",
            current=current_step == "chat",
        ),
        WizardStep(
            key="runtime",
            index=2,
            title="Prepare the worker runtime",
            description="Point Twinr to a working Node.js 20+ runtime and a stable auth folder.",
            status=runtime_step_status,
            status_label="Ready" if runtime_step_status == "ok" else "Needs check",
            detail=(
                "Use the default auth folder unless you have a specific reason to move it. "
                "On the Pi, the important part is a Node.js 20+ binary for the Baileys worker."
            ),
            fields=(
                _text_field(
                    "TWINR_WHATSAPP_NODE_BINARY",
                    "Node.js binary",
                    normalized_env_values,
                    str(getattr(config, "whatsapp_node_binary", "node") or "node"),
                    help_text="Baileys currently needs Node.js 20 or newer.",
                    tooltip_text="This binary starts the dedicated WhatsApp worker process.",
                    placeholder="node",
                ),
                _text_field(
                    "TWINR_WHATSAPP_AUTH_DIR",
                    "Auth folder",
                    normalized_env_values,
                    str(getattr(config, "whatsapp_auth_dir", "state/channels/whatsapp/auth") or "state/channels/whatsapp/auth"),
                    help_text="Keep this folder stable so the QR pairing survives restarts.",
                    tooltip_text="The folder must stay inside the Twinr project root.",
                    placeholder="state/channels/whatsapp/auth",
                ),
            ),
            checks=(
                WizardCheckRow(
                    label="Node.js runtime",
                    summary=probe.node_version,
                    detail=probe.node_detail,
                    status="ok" if probe.node_ready else "fail",
                ),
                WizardCheckRow(
                    label="Worker package",
                    summary="Found" if probe.worker_ready else "Missing",
                    detail=probe.worker_detail,
                    status="ok" if probe.worker_ready else "fail",
                ),
                WizardCheckRow(
                    label="Auth folder",
                    summary=str(probe.auth_dir),
                    detail="Twinr stores the WhatsApp linked-device session in this folder.",
                    status="ok" if probe.auth_dir_exists else "muted",
                ),
            ),
            action="save_runtime",
            action_label="Save runtime step",
            current=current_step == "runtime",
        ),
        WizardStep(
            key="pairing",
            index=3,
            title="Pair WhatsApp once",
            description="Open a bounded pairing window from the dashboard and scan the QR with Linked devices.",
            status=pairing_step_status,
            status_label=_pairing_status_label(snapshot, pairing_ready=pairing_ready, auth_repair_needed=auth_repair_needed),
            detail=_pairing_step_detail(
                runtime_ready=runtime_ready,
                allow_from_ready=allow_from_ready,
                guardrails_ready=guardrails_ready,
                snapshot=snapshot,
                pairing_ready=pairing_ready,
                auth_repair_needed=auth_repair_needed,
            ),
            checks=tuple(pairing_checks),
            action="start_pairing",
            action_label=_pairing_action_label(snapshot, pairing_ready=pairing_ready, auth_repair_needed=auth_repair_needed),
            action_enabled=pairing_action_enabled,
            action_hint=pairing_action_hint,
            media_src=pairing_qr_src,
            media_alt=pairing_qr_alt,
            media_title="Scan this QR with WhatsApp",
            media_detail=pairing_qr_detail,
            code_block=probe.launch_command,
            code_title="Manual fallback on the Pi",
            current=current_step == "pairing",
        ),
        WizardStep(
            key="test",
            index=4,
            title="Run the self-chat test",
            description="Use your own 'message yourself' chat as the Twinr conversation.",
            status="ok" if test_ready else "muted",
            status_label="Ready" if test_ready else "Wait",
            detail=(
                "Open WhatsApp, choose your chat with yourself, and send a short text such as 'Hallo Twinr'. "
                "Twinr should answer in the same chat."
                if test_ready
                else "Do not test yet. Finish the saved number, runtime, and pairing steps first."
            ),
            checks=(
                WizardCheckRow(
                    label="Allowed chat",
                    summary=allow_from_label,
                    detail="This should be the same number that owns the WhatsApp account.",
                    status="ok" if allow_from_ready else "warn",
                ),
                WizardCheckRow(
                    label="Self-chat guard",
                    summary="On" if config.whatsapp_self_chat_mode else "Off",
                    detail="Prevents Twinr from treating arbitrary self-messages as normal inbound traffic.",
                    status="ok" if config.whatsapp_self_chat_mode else "fail",
                ),
                WizardCheckRow(
                    label="Group chats",
                    summary="Blocked" if not config.whatsapp_groups_enabled else "Allowed",
                    detail="Keep this off while the channel is still internal-test only.",
                    status="ok" if not config.whatsapp_groups_enabled else "fail",
                ),
                WizardCheckRow(
                    label="Pairing state",
                    summary=linked_device_summary,
                    detail=linked_device_detail,
                    status=linked_device_status,
                ),
            ),
            current=current_step == "test",
        ),
    )

    return {
        "overall_status": overall_status,
        "overall_status_label": overall_status_label,
        "overall_detail": overall_detail,
        "page_refresh_seconds": 3 if snapshot.running else None,
        "summary_metrics": summary_metrics,
        "steps": steps,
    }


def _normalized_env_values(env_values: Mapping[str, object] | None) -> dict[str, str]:
    """Normalize mapping-like env values into a stable string dictionary."""

    if not isinstance(env_values, Mapping):
        return {}

    normalized: dict[str, str] = {}
    for key, raw_value in env_values.items():
        if not isinstance(key, str):
            continue
        if raw_value is None:
            normalized[key] = ""
            continue
        normalized[key] = raw_value if isinstance(raw_value, str) else str(raw_value)
    return normalized


def _allow_from_label(raw_allow_from: str) -> str:
    """Return a safe operator-facing label for the saved WhatsApp number."""

    normalized = str(raw_allow_from or "").strip()
    if not normalized:
        return "Not set yet"
    try:
        return canonicalize_whatsapp_allow_from(normalized)
    except (TypeError, ValueError):
        return normalized


def _default_step(
    *,
    allow_from_ready: bool,
    runtime_ready: bool,
    pairing_ready: bool,
    pairing_running: bool,
    auth_repair_needed: bool,
    test_ready: bool,
) -> str:
    """Pick the most useful wizard step for the current setup state."""

    if not allow_from_ready:
        return "chat"
    if not runtime_ready:
        return "runtime"
    if pairing_running or auth_repair_needed or not pairing_ready:
        return "pairing"
    if not test_ready:
        return "test"
    return "test"


def _sanitize_step(requested_step: str | None, *, fallback: str) -> str:
    """Normalize one untrusted step query parameter."""

    candidate = str(requested_step or "").strip().lower()
    return candidate if candidate in _WIZARD_STEP_KEYS else fallback


def _pairing_window_display(
    snapshot: ChannelPairingSnapshot,
    *,
    pairing_ready: bool,
) -> tuple[str, str, str]:
    """Summarize the bounded pairing window state for operators."""

    if snapshot.running:
        return snapshot.summary, snapshot.detail, "warn"
    if snapshot.auth_repair_needed:
        return snapshot.summary, snapshot.detail, "fail"
    if snapshot.phase == "failed":
        return snapshot.summary, snapshot.detail, "fail"
    if snapshot.phase == "timed_out":
        return snapshot.summary, snapshot.detail, "warn"
    if pairing_ready:
        return (
            "Closed",
            "No live pairing window is open right now because a linked-device session is already stored.",
            "ok",
        )
    if snapshot.summary != "Not started":
        return snapshot.summary, snapshot.detail, "muted"
    return (
        "Not started",
        "Use the button below to open a temporary pairing window directly from the dashboard.",
        "muted",
    )


def _pairing_qr_media(snapshot: ChannelPairingSnapshot) -> tuple[str, str, str]:
    """Build the optional portal-renderable QR block for the pairing step."""

    qr_svg = str(snapshot.qr_svg or "").strip()
    if not qr_svg or not snapshot.qr_needed:
        return ("", "", "")

    encoded_svg = base64.b64encode(qr_svg.encode("utf-8")).decode("ascii")
    return (
        f"data:image/svg+xml;base64,{encoded_svg}",
        "WhatsApp linked-device pairing QR code",
        "If scanning fails, keep the page open and use the terminal QR fallback below.",
    )


def _linked_device_display(
    probe: Any,
    snapshot: ChannelPairingSnapshot,
    *,
    pairing_ready: bool,
) -> tuple[str, str, str]:
    """Summarize the stored linked-device session state."""

    if snapshot.auth_repair_needed:
        return "Repair needed", snapshot.detail, "fail"
    if pairing_ready and probe.paired:
        return "Stored", probe.pair_detail, "ok"
    if pairing_ready:
        return "Paired", snapshot.detail, "ok"
    if probe.auth_dir_exists:
        return "Missing", probe.pair_detail, "warn"
    return "Missing", probe.pair_detail, "muted"


def _worker_process_display(snapshot: ChannelPairingSnapshot) -> tuple[str, str, str]:
    """Summarize the temporary pairing worker state."""

    if snapshot.running:
        if snapshot.worker_ready:
            return "Running", snapshot.last_worker_detail or snapshot.detail, "ok"
        return "Starting", snapshot.detail, "warn"
    if snapshot.last_worker_detail:
        return "Stopped", f"Last worker signal: {snapshot.last_worker_detail}.", "muted"
    return (
        "Idle",
        "The temporary pairing worker starts only while a bounded pairing window is open.",
        "muted",
    )


def _reconnect_display(snapshot: ChannelPairingSnapshot) -> tuple[str, str, str]:
    """Summarize the last reconnect reason emitted by the worker."""

    if snapshot.last_reconnect_reason:
        if snapshot.reconnect_in_ms:
            seconds = max(1, round(snapshot.reconnect_in_ms / 1000))
            detail = f"WhatsApp closed the socket before. Twinr plans the next reconnect in about {seconds} seconds."
        else:
            detail = "The worker reported at least one reconnect cycle during the last pairing window."
        return snapshot.last_reconnect_reason, detail, "warn" if snapshot.running else "muted"
    return (
        "None yet",
        "The worker has not reported any reconnect cycle in this web session yet.",
        "muted",
    )


def _pairing_status_label(
    snapshot: ChannelPairingSnapshot,
    *,
    pairing_ready: bool,
    auth_repair_needed: bool,
) -> str:
    """Return the short status-pill label for the pairing step."""

    if auth_repair_needed:
        return "Repair"
    if pairing_ready:
        return "Paired"
    if snapshot.running:
        return "Live"
    return "Open pairing"


def _pairing_action_label(
    snapshot: ChannelPairingSnapshot,
    *,
    pairing_ready: bool,
    auth_repair_needed: bool,
) -> str:
    """Return the operator-facing action label for the pairing button."""

    if auth_repair_needed:
        return "Start fresh pairing window"
    if snapshot.running:
        return "Pairing window is running"
    if pairing_ready:
        return "Open pairing window again"
    return "Start pairing window"


def _pairing_action_hint(
    *,
    runtime_ready: bool,
    allow_from_ready: bool,
    guardrails_ready: bool,
    snapshot: ChannelPairingSnapshot,
) -> str:
    """Return one short action hint for the pairing step."""

    if not allow_from_ready:
        return "Save your own WhatsApp number first so Twinr knows which self-chat to allow."
    if not guardrails_ready:
        return "Step 1 must keep self-chat mode on and groups blocked before pairing starts."
    if not runtime_ready:
        return "Finish step 2 first. The pairing window needs a working Node.js 20+ runtime and the Baileys worker package."
    if snapshot.running:
        return "A pairing window is already active. This page refreshes automatically while Twinr waits for WhatsApp."
    return "Once WhatsApp asks for a QR, Twinr shows it directly in this wizard and keeps the Pi command below as a fallback."


def _pairing_step_detail(
    *,
    runtime_ready: bool,
    allow_from_ready: bool,
    guardrails_ready: bool,
    snapshot: ChannelPairingSnapshot,
    pairing_ready: bool,
    auth_repair_needed: bool,
) -> str:
    """Return the supporting copy shown under the pairing step header."""

    if not allow_from_ready:
        return "Finish the saved-number step first so Twinr knows which self-chat should be allowlisted."
    if not guardrails_ready:
        return "Self-chat mode must stay on and group chats must stay blocked before the pairing worker starts."
    if not runtime_ready:
        return "Finish the runtime step first so the bounded pairing worker can start cleanly."
    if auth_repair_needed:
        return "The stored session needs repair. Start a new bounded pairing window and scan the next QR from Linked devices."
    if snapshot.running and snapshot.qr_needed:
        return "The pairing window is already open. Scan the QR shown below now and stay on this page until the status switches to Paired."
    if snapshot.running:
        return "Twinr is already trying to open the bounded pairing window. This page refreshes automatically while it waits."
    if pairing_ready:
        return "A linked-device session is already stored. You only need to open another pairing window if you want to replace that session."
    return "Use the button below to start a short pairing window directly from the dashboard. The manual Pi command stays available as a fallback."
