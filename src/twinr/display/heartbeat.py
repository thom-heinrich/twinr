"""Persist display-loop forward-progress heartbeats for health/supervision.

The productive Pi runtime cannot infer display liveness from lock ownership
alone because a hung display companion thread can still keep the lock while the
panel stops receiving updates. This module stores a small authoritative
heartbeat file that records the display loop's last idle/render progress.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.locks import loop_lock_owner


_DISPLAY_HEARTBEAT_RELATIVE_PATH = Path("artifacts") / "stores" / "ops" / "display_heartbeat.json"
_DEFAULT_STALE_AFTER_S = 15.0
_DEFAULT_RENDERING_STALE_AFTER_S = 30.0
_MIN_STALE_AFTER_S = 5.0

LoopOwnerFn = Callable[[TwinrConfig, str], int | None]


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _normalize_timestamp(value: str | None) -> datetime | None:
    """Parse one persisted ISO-8601 timestamp into an aware UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def display_heartbeat_stale_after_s(config: TwinrConfig) -> float:
    """Return the allowed heartbeat age before display liveness is stale."""

    try:
        poll_interval_s = float(getattr(config, "display_poll_interval_s", 0.5) or 0.5)
    except (TypeError, ValueError):
        poll_interval_s = 0.5
    return max(_MIN_STALE_AFTER_S, max(_DEFAULT_STALE_AFTER_S, poll_interval_s * 20.0))


def display_rendering_heartbeat_stale_after_s(config: TwinrConfig) -> float:
    """Return the longer timeout tolerated for an in-flight panel render."""

    return max(
        display_heartbeat_stale_after_s(config),
        _DEFAULT_RENDERING_STALE_AFTER_S,
    )


@dataclass(frozen=True, slots=True)
class DisplayHeartbeat:
    """Represent one persisted display-loop forward-progress sample."""

    pid: int
    updated_at: str
    runtime_status: str
    phase: str
    seq: int
    last_render_started_at: str | None = None
    last_render_completed_at: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the heartbeat for JSON persistence."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DisplayHeartbeat":
        """Build one heartbeat from JSON payload data."""

        return cls(
            pid=int(payload.get("pid", 0) or 0),
            updated_at=str(payload.get("updated_at", "") or ""),
            runtime_status=str(payload.get("runtime_status", "") or ""),
            phase=str(payload.get("phase", "") or ""),
            seq=int(payload.get("seq", 0) or 0),
            last_render_started_at=_optional_text(payload.get("last_render_started_at")),
            last_render_completed_at=_optional_text(payload.get("last_render_completed_at")),
            detail=_optional_text(payload.get("detail")),
        )


@dataclass(slots=True)
class DisplayHeartbeatStore:
    """Read and write the display-loop heartbeat artifact."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayHeartbeatStore":
        """Resolve the canonical heartbeat path from Twinr config."""

        project_root = Path(config.project_root).expanduser().resolve()
        return cls(project_root / _DISPLAY_HEARTBEAT_RELATIVE_PATH)

    def load(self) -> DisplayHeartbeat | None:
        """Load the current display heartbeat if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return DisplayHeartbeat.from_dict(payload)
        except Exception:
            return None

    def save(self, heartbeat: DisplayHeartbeat) -> None:
        """Persist one heartbeat atomically."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(heartbeat.to_dict(), ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


@dataclass(frozen=True, slots=True)
class DisplayCompanionHealth:
    """Describe whether the display companion currently proves forward progress."""

    owner_pid: int | None
    running: bool
    count: int
    detail: str
    heartbeat: DisplayHeartbeat | None = None


def display_heartbeat_age_s(
    heartbeat: DisplayHeartbeat,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return the heartbeat age in seconds, or ``None`` for invalid timestamps."""

    updated_at = _normalize_timestamp(heartbeat.updated_at)
    if updated_at is None:
        return None
    effective_now = (now or _utc_now()).astimezone(timezone.utc)
    return max(0.0, (effective_now - updated_at).total_seconds())


def is_display_heartbeat_fresh(
    heartbeat: DisplayHeartbeat,
    *,
    config: TwinrConfig,
    expected_pid: int | None = None,
    now: datetime | None = None,
) -> bool:
    """Return whether the persisted heartbeat proves current display progress."""

    if expected_pid is not None and heartbeat.pid != expected_pid:
        return False
    age_s = display_heartbeat_age_s(heartbeat, now=now)
    if age_s is None:
        return False
    if age_s <= display_heartbeat_stale_after_s(config):
        return True
    rendering_age_s = _display_rendering_age_s(heartbeat, now=now)
    if rendering_age_s is None:
        return False
    return rendering_age_s <= display_rendering_heartbeat_stale_after_s(config)


def build_display_heartbeat(
    *,
    runtime_status: str,
    phase: str,
    seq: int,
    detail: str | None = None,
    pid: int | None = None,
    updated_at: datetime | None = None,
    last_render_started_at: datetime | None = None,
    last_render_completed_at: datetime | None = None,
) -> DisplayHeartbeat:
    """Build one normalized display heartbeat snapshot."""

    effective_updated_at = (updated_at or _utc_now()).astimezone(timezone.utc)
    return DisplayHeartbeat(
        pid=int(os.getpid() if pid is None else pid),
        updated_at=effective_updated_at.isoformat(),
        runtime_status=str(runtime_status or "").strip().lower() or "error",
        phase=str(phase or "").strip().lower() or "idle",
        seq=max(0, int(seq)),
        last_render_started_at=_format_optional_datetime(last_render_started_at),
        last_render_completed_at=_format_optional_datetime(last_render_completed_at),
        detail=_optional_text(detail),
    )


def save_display_heartbeat(
    store: DisplayHeartbeatStore,
    *,
    runtime_status: str,
    phase: str,
    seq: int,
    detail: str | None = None,
    pid: int | None = None,
    updated_at: datetime | None = None,
    last_render_started_at: datetime | None = None,
    last_render_completed_at: datetime | None = None,
) -> None:
    """Persist one normalized display heartbeat through the canonical contract."""

    store.save(
        build_display_heartbeat(
            runtime_status=runtime_status,
            phase=phase,
            seq=seq,
            detail=detail,
            pid=pid,
            updated_at=updated_at,
            last_render_started_at=last_render_started_at,
            last_render_completed_at=last_render_completed_at,
        )
    )


def assess_display_companion_health(
    config: TwinrConfig,
    *,
    owner_pid: int | None = None,
    loop_name: str = "display-loop",
    heartbeat_store: DisplayHeartbeatStore | None = None,
    loop_owner_fn: LoopOwnerFn | None = None,
    now: datetime | None = None,
) -> DisplayCompanionHealth:
    """Assess whether a display companion lock owner is also making heartbeat progress."""

    resolved_owner_pid = owner_pid
    if resolved_owner_pid is None:
        resolver = loop_lock_owner if loop_owner_fn is None else loop_owner_fn
        resolved_owner_pid = resolver(config, loop_name)
    if resolved_owner_pid is None:
        return DisplayCompanionHealth(
            owner_pid=None,
            running=False,
            count=0,
            detail="display-companion not detected",
            heartbeat=None,
        )

    store = heartbeat_store if heartbeat_store is not None else DisplayHeartbeatStore.from_config(config)
    heartbeat = store.load()
    if heartbeat is None:
        return DisplayCompanionHealth(
            owner_pid=resolved_owner_pid,
            running=False,
            count=0,
            detail=f"pid={resolved_owner_pid} display-companion heartbeat missing",
            heartbeat=None,
        )
    if is_display_heartbeat_fresh(
        heartbeat,
        config=config,
        expected_pid=resolved_owner_pid,
        now=now,
    ):
        return DisplayCompanionHealth(
            owner_pid=resolved_owner_pid,
            running=True,
            count=1,
            detail=f"pid={resolved_owner_pid} display-companion",
            heartbeat=heartbeat,
        )
    heartbeat_age_s = display_heartbeat_age_s(heartbeat, now=now)
    age_text = f" stale={heartbeat_age_s:.1f}s" if heartbeat_age_s is not None else " stale"
    return DisplayCompanionHealth(
        owner_pid=resolved_owner_pid,
        running=False,
        count=0,
        detail=f"pid={resolved_owner_pid} display-companion heartbeat {heartbeat.phase or 'unknown'}{age_text}",
        heartbeat=heartbeat,
    )


def display_heartbeat_has_progress(
    config: TwinrConfig,
    *,
    expected_pid: int,
    started_at_utc: datetime | None,
    heartbeat_store: DisplayHeartbeatStore | None = None,
    now: datetime | None = None,
) -> bool:
    """Return whether the heartbeat proves progress for the current runtime child."""

    if started_at_utc is None:
        return False
    store = heartbeat_store if heartbeat_store is not None else DisplayHeartbeatStore.from_config(config)
    heartbeat = store.load()
    if heartbeat is None:
        return False
    if not is_display_heartbeat_fresh(heartbeat, config=config, expected_pid=expected_pid, now=now):
        return False
    updated_at = _normalize_timestamp(heartbeat.updated_at)
    if updated_at is None:
        return False
    normalized_started_at = (
        started_at_utc.replace(tzinfo=timezone.utc)
        if started_at_utc.tzinfo is None
        else started_at_utc.astimezone(timezone.utc)
    )
    return updated_at >= normalized_started_at


def _format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _display_rendering_age_s(
    heartbeat: DisplayHeartbeat,
    *,
    now: datetime | None = None,
) -> float | None:
    if str(heartbeat.phase or "").strip().lower() != "rendering":
        return None
    started_at = _normalize_timestamp(heartbeat.last_render_started_at)
    if started_at is None:
        return None
    completed_at = _normalize_timestamp(heartbeat.last_render_completed_at)
    if completed_at is not None and completed_at >= started_at:
        return None
    effective_now = (now or _utc_now()).astimezone(timezone.utc)
    return max(0.0, (effective_now - started_at).total_seconds())


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
