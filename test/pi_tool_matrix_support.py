"""Provide deterministic local backends for the Pi tool-matrix harness.

The broad Pi tool matrix exercises the real spoken/tool loop with live LLM
calls. Some tools still need bounded local backends, though, because they
depend on optional device integrations or local identity stores that would make
the acceptance run flaky or environment-specific.

This module keeps those deterministic matrix-only backends in one place:

- local portrait-profile enrollment and status
- shared household identity state
- smart-home entity, control, and event-stream state
- optional browser-automation workspace bridge
- bounded service-connect pairing state
- bounded WhatsApp outbound dispatch state
- self-coding compile artifacts
- world-intelligence remote-state snapshots

The goal is not to fake tool results. The live model still chooses tools and
the real handlers still execute. These helpers only supply stable backend state
behind those handlers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
import json
import textwrap
from typing import cast

from twinr.agent.personality.intelligence.models import WorldFeedItem
from twinr.agent.self_coding.codex_driver import (
    CodexCompileArtifact,
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileResult,
)
from twinr.agent.self_coding.status import ArtifactKind
from twinr.channels.onboarding import ChannelPairingSnapshot
from twinr.channels.whatsapp import WhatsAppOutboundResult
from twinr.hardware.household_identity import (
    HouseholdIdentityFeedbackEvent,
    HouseholdIdentityMemberStatus,
    HouseholdIdentityObservation,
    HouseholdIdentityQuality,
    HouseholdIdentityStatus,
)
from twinr.hardware.household_voice_identity import HouseholdVoiceSummary
from twinr.hardware.portrait_match import PortraitEnrollmentResult
from twinr.integrations import manifest_for_id
from twinr.integrations.smarthome.adapter import SmartHomeAdapterSettings, SmartHomeIntegrationAdapter
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventBatch,
    SmartHomeEventKind,
)


def _utc_iso() -> str:
    """Return one current UTC ISO timestamp for matrix fixture payloads."""

    return datetime.now(UTC).isoformat()


_MATRIX_BROWSER_ADAPTER = textwrap.dedent(
    """
    from twinr.browser_automation import (
        BrowserAutomationAvailability,
        BrowserAutomationArtifact,
        BrowserAutomationRequest,
        BrowserAutomationResult,
    )


    class MatrixBrowserDriver:
        def availability(self) -> BrowserAutomationAvailability:
            return BrowserAutomationAvailability(
                enabled=True,
                available=True,
                reason="matrix-ready",
            )

        def execute(self, request: BrowserAutomationRequest) -> BrowserAutomationResult:
            domain = request.allowed_domains[0] if request.allowed_domains else "example.org"
            final_url = request.start_url or f"https://{domain}/"
            return BrowserAutomationResult(
                ok=True,
                status="completed",
                summary=f"Checked visible site state on {domain}.",
                final_url=final_url,
                artifacts=(
                    BrowserAutomationArtifact(
                        kind="screenshot",
                        path="artifacts/browser_matrix.png",
                        content_type="image/png",
                    ),
                ),
                data={
                    "task_id": request.task_id,
                    "goal": request.goal,
                    "allowed_domains": list(request.allowed_domains),
                },
            )


    def create_browser_automation_driver(*, config, project_root):
        del config, project_root
        return MatrixBrowserDriver()
    """
).lstrip()


def write_matrix_browser_workspace(root: Path) -> Path:
    """Create one deterministic local browser workspace for matrix runs."""

    workspace = root / "browser_automation"
    workspace.mkdir(parents=True, exist_ok=True)
    adapter_path = workspace / "adapter.py"
    adapter_path.write_text(_MATRIX_BROWSER_ADAPTER, encoding="utf-8")
    return adapter_path


def install_matrix_whatsapp_runtime(root: Path) -> tuple[Path, Path]:
    """Create the minimal WhatsApp files needed to expose the tool locally."""

    auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
    worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
    auth_dir.mkdir(parents=True, exist_ok=True)
    worker_root.mkdir(parents=True, exist_ok=True)
    (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
    (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
    return auth_dir, worker_root


def matrix_service_connect_probe() -> SimpleNamespace:
    """Return one ready WhatsApp runtime probe for service-connect tests."""

    return SimpleNamespace(
        node_ready=True,
        node_detail="Node ok",
        worker_ready=True,
        worker_detail="Worker ok",
        paired=False,
        pair_detail="No linked session yet.",
    )


class MatrixServiceConnectCoordinator:
    """Keep one deterministic pairing coordinator for service-connect runs."""

    def __init__(self, *, store, registry, snapshot_observer=None, pairing_window_s: float = 90.0) -> None:
        del store, registry
        self._snapshot_observer = snapshot_observer
        self.pairing_window_s = pairing_window_s

    def load_snapshot(self) -> ChannelPairingSnapshot:
        """Return the initial unpaired onboarding snapshot."""

        return ChannelPairingSnapshot.initial("whatsapp")

    def start_pairing(self, _config) -> bool:
        """Pretend pairing started and optionally emit one starting snapshot."""

        if callable(self._snapshot_observer):
            self._snapshot_observer(
                ChannelPairingSnapshot(
                    channel_id="whatsapp",
                    phase="starting",
                    summary="Starting pairing",
                    detail="Twinr is preparing the WhatsApp QR on the right info panel.",
                    running=True,
                    qr_needed=True,
                    worker_ready=True,
                    updated_at=_utc_iso(),
                )
            )
        return True


@dataclass(slots=True)
class MatrixWhatsAppDispatch:
    """Capture one bounded WhatsApp send surface for matrix scenarios."""

    sent_messages: list[dict[str, str | None]]

    def __init__(self) -> None:
        self.sent_messages = []

    def dispatch(
        self,
        *,
        chat_jid: str,
        text: str,
        recipient_label: str,
        reply_to_message_id: str | None = None,
    ) -> WhatsAppOutboundResult:
        """Record one sent WhatsApp and return a successful outbound result."""

        self.sent_messages.append(
            {
                "chat_jid": chat_jid,
                "text": text,
                "recipient_label": recipient_label,
                "reply_to_message_id": reply_to_message_id,
            }
        )
        index = len(self.sent_messages)
        return WhatsAppOutboundResult.sent(
            request_id=f"matrix-wa-{index}",
            message_id=f"wa-msg-{index}",
        )


@dataclass(slots=True)
class MatrixWorldRemoteState:
    """Persist matrix-only world-intelligence snapshots in memory."""

    snapshots: dict[str, dict[str, object]]

    def __init__(self) -> None:
        self.snapshots = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        """Return one stored world snapshot payload if it exists."""

        del local_path
        payload = self.snapshots.get(snapshot_kind)
        if payload is None:
            return None
        return dict(payload)

    def save_snapshot(self, *, snapshot_kind: str, payload) -> None:
        """Persist one world snapshot payload for later loop restarts."""

        self.snapshots[snapshot_kind] = dict(payload)


@dataclass(slots=True)
class MatrixPortraitProvider:
    """Keep one deterministic local face-profile state for matrix scenarios."""

    user_id: str = "main_user"
    display_name: str | None = None
    reference_image_count: int = 0

    def capture_and_enroll_reference(
        self,
        *,
        user_id: str,
        display_name: str | None,
        source: str,
    ) -> SimpleNamespace:
        """Add one local portrait reference and return an enrollment payload."""

        del source
        self.user_id = user_id
        self.display_name = display_name or self.display_name or "Thom"
        self.reference_image_count += 1
        return SimpleNamespace(
            status="enrolled",
            user_id=self.user_id,
            display_name=self.display_name,
            reference_id=f"ref_{self.reference_image_count}",
            reference_image_count=self.reference_image_count,
        )

    def summary(self, *, user_id: str) -> SimpleNamespace:
        """Return the current bounded portrait-profile summary."""

        return SimpleNamespace(
            enrolled=self.reference_image_count > 0 and user_id == self.user_id,
            user_id=self.user_id,
            display_name=self.display_name,
            reference_image_count=self.reference_image_count,
            store_path="matrix://portrait",
        )

    def observe(self) -> SimpleNamespace:
        """Return one stable live portrait-match observation."""

        return SimpleNamespace(
            state="likely_reference_user",
            confidence=0.82,
            fused_confidence=0.88,
            temporal_state="stable_match",
            temporal_observation_count=2,
            matched_user_id=self.user_id if self.reference_image_count > 0 else None,
            matched_user_display_name=self.display_name if self.reference_image_count > 0 else None,
            candidate_user_count=1 if self.reference_image_count > 0 else 0,
            capture_source_device="matrix-static-camera",
        )

    def clear_identity_profile(self, *, user_id: str) -> SimpleNamespace:
        """Delete the matrix portrait-profile state."""

        self.user_id = user_id
        self.reference_image_count = 0
        return SimpleNamespace(
            status="cleared",
            user_id=self.user_id,
            reference_image_count=0,
        )


@dataclass(slots=True)
class _MatrixHouseholdMember:
    """Track one bounded household-identity member during the matrix run."""

    user_id: str
    display_name: str | None
    portrait_reference_count: int = 0
    voice_sample_count: int = 0
    confirm_count: int = 0
    deny_count: int = 0


class MatrixHouseholdIdentityManager:
    """Keep one deterministic shared household-identity state."""

    def __init__(self) -> None:
        self.primary_user_id = "main_user"
        self._members: dict[str, _MatrixHouseholdMember] = {}

    def status(
        self,
        *,
        audio_pcm: bytes | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> HouseholdIdentityStatus:
        """Return the current matrix household-identity status."""

        del audio_pcm, sample_rate, channels
        members = tuple(self._member_payload(member) for member in self._members.values())
        current_observation = None
        if members:
            first = members[0]
            modalities = []
            if first.portrait_reference_count > 0:
                modalities.append("face")
            if first.voice_sample_count > 0:
                modalities.append("voice")
            if not modalities:
                modalities.append("session")
            current_observation = HouseholdIdentityObservation(
                state="known_household_member",
                matched_user_id=first.user_id,
                matched_user_display_name=first.display_name,
                confidence=0.91,
                modalities=tuple(modalities),
                temporal_state="stable_match",
                session_support_ratio=0.92,
                session_observation_count=3,
                policy_recommendation="allow_sensitive_confirmation",
                block_reason=None,
                voice_assessment=None,
                portrait_observation=None,
            )
        return HouseholdIdentityStatus(
            primary_user_id=self.primary_user_id,
            members=members,
            current_observation=current_observation,
        )

    def enroll_face(
        self,
        *,
        user_id: str,
        display_name: str | None,
    ) -> tuple[PortraitEnrollmentResult, HouseholdIdentityMemberStatus]:
        """Add one face enrollment to the shared matrix household state."""

        member = self._ensure_member(user_id=user_id, display_name=display_name)
        member.portrait_reference_count += 1
        return (
            PortraitEnrollmentResult(
                status="enrolled",
                user_id=member.user_id,
                display_name=member.display_name,
                reference_id=f"face_ref_{member.portrait_reference_count}",
                reference_image_count=member.portrait_reference_count,
            ),
            self._member_payload(member),
        )

    def enroll_voice(
        self,
        audio_pcm: bytes,
        *,
        sample_rate: int,
        channels: int,
        user_id: str,
        display_name: str | None,
    ) -> tuple[HouseholdVoiceSummary, HouseholdIdentityMemberStatus]:
        """Add one voice enrollment to the shared matrix household state."""

        del channels
        member = self._ensure_member(user_id=user_id, display_name=display_name)
        member.voice_sample_count += 1
        duration_ms = int((len(audio_pcm) / 2.0) / float(sample_rate) * 1000.0) if sample_rate > 0 else 0
        return (
            HouseholdVoiceSummary(
                user_id=member.user_id,
                display_name=member.display_name,
                primary_user=member.user_id == self.primary_user_id,
                enrolled=True,
                sample_count=member.voice_sample_count,
                average_duration_ms=duration_ms,
                updated_at=_utc_iso(),
                store_path="matrix://household-voice",
            ),
            self._member_payload(member),
        )

    def record_feedback(
        self,
        *,
        outcome: str,
        user_id: str | None,
        display_name: str | None,
    ) -> tuple[HouseholdIdentityFeedbackEvent, HouseholdIdentityMemberStatus]:
        """Store one explicit household-identity confirmation or denial."""

        normalized_user_id = user_id or self.primary_user_id
        member = self._ensure_member(user_id=normalized_user_id, display_name=display_name)
        if outcome == "confirm":
            member.confirm_count += 1
        else:
            member.deny_count += 1
        event = HouseholdIdentityFeedbackEvent(
            user_id=member.user_id,
            display_name=member.display_name,
            outcome=outcome,
            modalities=("session",),
            created_at=_utc_iso(),
            source="matrix",
        )
        return event, self._member_payload(member)

    def _ensure_member(self, *, user_id: str, display_name: str | None) -> _MatrixHouseholdMember:
        normalized_user_id = user_id or self.primary_user_id
        member = self._members.get(normalized_user_id)
        if member is None:
            member = _MatrixHouseholdMember(
                user_id=normalized_user_id,
                display_name=display_name or "Thom",
            )
            self._members[normalized_user_id] = member
        elif display_name:
            member.display_name = display_name
        return member

    def _member_payload(self, member: _MatrixHouseholdMember) -> HouseholdIdentityMemberStatus:
        quality = HouseholdIdentityQuality(
            score=min(
                1.0,
                0.38
                + (member.portrait_reference_count * 0.14)
                + (member.voice_sample_count * 0.16)
                + (member.confirm_count * 0.08)
                - (member.deny_count * 0.1),
            ),
            state="usable"
            if member.portrait_reference_count > 0 or member.voice_sample_count > 0
            else "needs_enrollment",
            face_reference_count=member.portrait_reference_count,
            voice_sample_count=member.voice_sample_count,
            confirm_count=member.confirm_count,
            deny_count=member.deny_count,
            recommended_next_step=(
                "capture_more_face_angles"
                if member.portrait_reference_count == 1 and member.voice_sample_count == 0
                else "done"
            ),
            guidance_hints=(
                ("capture_more_face_angles",)
                if member.portrait_reference_count == 1 and member.voice_sample_count == 0
                else ("speak_clear_sentence", "quiet_room")
            ),
        )
        return HouseholdIdentityMemberStatus(
            user_id=member.user_id,
            display_name=member.display_name,
            primary_user=member.user_id == self.primary_user_id,
            portrait_reference_count=member.portrait_reference_count,
            voice_sample_count=member.voice_sample_count,
            confirm_count=member.confirm_count,
            deny_count=member.deny_count,
            quality=quality,
        )


class MatrixSmartHomeProvider:
    """Keep one bounded smart-home state behind the real integration adapter."""

    def __init__(self) -> None:
        self._entities: dict[str, dict[str, object]] = {
            "light.living_room": {
                "provider": "matrix_smart_home",
                "label": "Wohnzimmerlampe",
                "entity_class": SmartHomeEntityClass.LIGHT,
                "area": "Wohnzimmer",
                "controllable": True,
                "supported_commands": (
                    SmartHomeCommand.TURN_ON,
                    SmartHomeCommand.TURN_OFF,
                    SmartHomeCommand.SET_BRIGHTNESS,
                ),
                "state": {"power": "off", "brightness": 10},
            },
            "light.hallway": {
                "provider": "matrix_smart_home",
                "label": "Flurlicht",
                "entity_class": SmartHomeEntityClass.LIGHT,
                "area": "Flur",
                "controllable": True,
                "supported_commands": (
                    SmartHomeCommand.TURN_ON,
                    SmartHomeCommand.TURN_OFF,
                    SmartHomeCommand.SET_BRIGHTNESS,
                ),
                "state": {"power": "off", "brightness": 25},
            },
            "sensor.motion_hallway": {
                "provider": "matrix_smart_home",
                "label": "Flur Bewegungsmelder",
                "entity_class": SmartHomeEntityClass.MOTION_SENSOR,
                "area": "Flur",
                "controllable": False,
                "supported_commands": (),
                "state": {"motion": False},
            },
        }
        self.last_read_entity_ids: tuple[str, ...] = ()
        self.last_control_command: str | None = None
        self.last_control_entity_ids: tuple[str, ...] = ()
        self.last_sensor_limit: int = 0

    def build_adapter(self) -> SmartHomeIntegrationAdapter:
        """Build the real generic smart-home adapter around the matrix state."""

        manifest = manifest_for_id("smart_home_hub")
        if manifest is None:
            raise RuntimeError("smart_home_hub manifest is unavailable.")
        return SmartHomeIntegrationAdapter(
            manifest=manifest,
            entity_provider=self,
            controller=self,
            sensor_stream=self,
            settings=SmartHomeAdapterSettings(),
        )

    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        """Return the current bounded matrix entity list."""

        del include_unavailable
        self.last_read_entity_ids = entity_ids
        requested = set(entity_ids)
        entities: list[SmartHomeEntity] = []
        for entity_id, record in self._entities.items():
            if requested and entity_id not in requested:
                continue
            if entity_class is not None and record["entity_class"] is not entity_class:
                continue
            entities.append(
                SmartHomeEntity(
                    entity_id=entity_id,
                    provider=str(record["provider"]),
                    label=str(record["label"]),
                    entity_class=record["entity_class"],
                    area=str(record["area"]),
                    readable=True,
                    controllable=bool(record["controllable"]),
                    online=True,
                    supported_commands=cast(tuple[SmartHomeCommand, ...], record["supported_commands"]),
                    state=dict(cast(dict[str, object], record["state"])),
                    attributes={},
                )
            )
        return sorted(entities, key=lambda entity: entity.label.casefold())

    def control(
        self,
        *,
        command: SmartHomeCommand,
        entity_ids: tuple[str, ...],
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Mutate the bounded matrix smart-home state."""

        del parameters
        self.last_control_command = command.value
        self.last_control_entity_ids = entity_ids
        for entity_id in entity_ids:
            state = cast(dict[str, object], self._entities[entity_id]["state"])
            if command is SmartHomeCommand.TURN_ON:
                state["power"] = "on"
            elif command is SmartHomeCommand.TURN_OFF:
                state["power"] = "off"
            elif command is SmartHomeCommand.SET_BRIGHTNESS:
                state["brightness"] = 60
                state["power"] = "on"
        return {
            "targets": [
                {
                    "entity_id": entity_id,
                    "command": command.value,
                    "state": dict(cast(dict[str, object], self._entities[entity_id]["state"])),
                }
                for entity_id in entity_ids
            ]
        }

    def read_sensor_stream(
        self,
        *,
        cursor: str | None = None,
        limit: int,
    ) -> SmartHomeEventBatch:
        """Return one bounded synthetic smart-home event batch."""

        del cursor
        self.last_sensor_limit = limit
        events = (
            SmartHomeEvent(
                event_id="evt_motion_1",
                provider="matrix_smart_home",
                entity_id="sensor.motion_hallway",
                event_kind=SmartHomeEventKind.MOTION_DETECTED,
                observed_at=_utc_iso(),
                label="Flur Bewegungsmelder",
                area="Flur",
                details={"motion": True},
            ),
            SmartHomeEvent(
                event_id="evt_light_1",
                provider="matrix_smart_home",
                entity_id="light.hallway",
                event_kind=SmartHomeEventKind.STATE_CHANGED,
                observed_at=_utc_iso(),
                label="Flurlicht",
                area="Flur",
                details={"power": cast(dict[str, object], self._entities["light.hallway"]["state"]).get("power", "off")},
            ),
        )
        return SmartHomeEventBatch(events=events[:limit], next_cursor="matrix_cursor_1", stream_live=True)

    def state_for(self, entity_id: str) -> dict[str, object]:
        """Return one entity state snapshot for assertions."""

        return dict(cast(dict[str, object], self._entities[entity_id]["state"]))


class MatrixSelfCodingCompileDriver:
    """Return one deterministic self-coding compile artifact for live scenarios."""

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        """Produce one compile result that activation can enable safely."""

        del request
        if event_sink is not None:
            event_sink(
                CodexCompileEvent(kind="turn_started"),
                CodexCompileProgress(
                    driver_name="MatrixSelfCodingCompileDriver",
                    event_count=1,
                    last_event_kind="turn_started",
                ),
            )
        return CodexCompileResult(
            status="ok",
            summary="Compiled",
            artifacts=(
                CodexCompileArtifact(
                    kind=ArtifactKind.AUTOMATION_MANIFEST,
                    artifact_name="automation_manifest.json",
                    media_type="application/json",
                    content=json.dumps(
                        {
                            "automation": {
                                "name": "Announce Family Updates",
                                "trigger": {
                                    "kind": "if_then",
                                    "event_name": "new_message",
                                    "all_conditions": [],
                                    "any_conditions": [],
                                    "cooldown_seconds": 45,
                                },
                                "actions": [
                                    {
                                        "kind": "say",
                                        "text": "Family update arrived.",
                                    }
                                ],
                            }
                        }
                    ),
                    summary="Matrix automation manifest",
                ),
            ),
        )


def matrix_world_feed_items() -> tuple[WorldFeedItem, ...]:
    """Return one bounded deterministic RSS refresh result for matrix runs."""

    return (
        WorldFeedItem(
            feed_url="https://example.com/feeds/hamburg-local.xml",
            source="Hamburg News",
            title="District council approves quieter night buses",
            link="https://example.com/hamburg/night-bus",
            published_at="2026-03-20T07:00:00+00:00",
        ),
        WorldFeedItem(
            feed_url="https://example.com/feeds/hamburg-local.xml",
            source="Hamburg News",
            title="Committee debates district heating plan",
            link="https://example.com/hamburg/heating",
            published_at="2026-03-20T08:30:00+00:00",
        ),
    )


__all__ = [
    "MatrixHouseholdIdentityManager",
    "MatrixPortraitProvider",
    "MatrixServiceConnectCoordinator",
    "MatrixSelfCodingCompileDriver",
    "MatrixSmartHomeProvider",
    "MatrixWhatsAppDispatch",
    "MatrixWorldRemoteState",
    "install_matrix_whatsapp_runtime",
    "matrix_service_connect_probe",
    "matrix_world_feed_items",
    "write_matrix_browser_workspace",
]
