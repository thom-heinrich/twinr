"""Expose runtime-facing XVF3800 signal observations for Twinr."""

from __future__ import annotations

import logging
import math
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable
import time
from dataclasses import replace

from twinr.hardware.respeaker.claim_contract import build_respeaker_signal_claim_contract
from twinr.hardware.respeaker.derived_signals import derive_respeaker_signal_state
from twinr.hardware.respeaker.models import ReSpeakerPrimitiveSnapshot, ReSpeakerSignalSnapshot
from twinr.hardware.respeaker.snapshot_service import capture_respeaker_primitive_snapshot
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport


_DEFAULT_LOCK_TIMEOUT_S = 0.25
_DEFAULT_SNAPSHOT_TIMEOUT_S = 0.75
_SOURCE = "respeaker_xvf3800"
_SOURCE_TYPE = "observed"

_LOG = logging.getLogger(__name__)


class _SnapshotTimeoutError(TimeoutError):
    """Internal timeout used when one primitive snapshot blocks too long."""


class _TransportInitError(RuntimeError):
    """Internal transport initialization failure."""


class ReSpeakerSignalProvider:
    """Read conservative runtime-facing signals from the XVF3800.

    The provider wraps one bounded primitive snapshot read and adds tiny
    runtime-only state such as ``recent_speech_age_s``. It deliberately emits
    ``None`` for unknown values instead of guessing across host-control or
    permission failures.
    """

    def __init__(
        self,
        *,
        sensor_window_ms: int = 0,
        lock_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
        snapshot_timeout_s: float = _DEFAULT_SNAPSHOT_TIMEOUT_S,
        transport: ReSpeakerLibusbTransport | None = None,
        snapshot_factory: Callable[..., ReSpeakerPrimitiveSnapshot] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        assistant_output_active_predicate: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize one provider with bounded lock and snapshot settings."""

        # AUDIT-FIX(#2): Coerce potentially malformed config values safely instead of crashing
        # service startup on bad env/input types.
        self.sensor_window_ms = self._coerce_non_negative_int(sensor_window_ms, default=0)
        # AUDIT-FIX(#2): Reject NaN/inf/invalid lock timeout values and keep a bounded minimum.
        self.lock_timeout_s = self._coerce_min_float(
            lock_timeout_s,
            minimum=0.01,
            default=_DEFAULT_LOCK_TIMEOUT_S,
        )
        # AUDIT-FIX(#1): Add a real bound for the primitive snapshot read so a wedged transport
        # cannot block the caller indefinitely.
        self.snapshot_timeout_s = self._coerce_min_float(
            snapshot_timeout_s,
            minimum=0.05,
            default=max(self.lock_timeout_s, _DEFAULT_SNAPSHOT_TIMEOUT_S),
        )
        # AUDIT-FIX(#3): Defer transport construction to observation time so provider startup
        # cannot fail or hang immediately on missing/bad hardware.
        self.transport: ReSpeakerLibusbTransport | None = transport
        # AUDIT-FIX(#8): Use explicit None checks for injectable collaborators so falsy test
        # doubles/custom callables are not discarded accidentally.
        self._snapshot_factory = (
            snapshot_factory if snapshot_factory is not None else capture_respeaker_primitive_snapshot
        )
        self._monotonic_clock = monotonic_clock if monotonic_clock is not None else time.monotonic
        self._assistant_output_active_predicate = assistant_output_active_predicate
        self._lock = Lock()
        self._last_speech_monotonic: float | None = None
        self._stuck_snapshot_thread: Thread | None = None
        self._closed = False

    def observe(self) -> ReSpeakerSignalSnapshot:
        """Capture one conservative runtime-facing XVF3800 signal snapshot."""

        if self._closed:
            # AUDIT-FIX(#7): After close(), return a deterministic degraded snapshot instead of
            # continuing to probe hardware from a logically closed provider.
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="provider_closed",
                transport_reason="provider_closed",
            )

        if not self._lock.acquire(timeout=self.lock_timeout_s):
            # AUDIT-FIX(#4): Invalidate cached speech age when continuity of observation is lost.
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="provider_lock_timeout",
                transport_reason="provider_lock_timeout",
            )
        try:
            if self._stuck_snapshot_thread is not None:
                if self._stuck_snapshot_thread.is_alive():
                    # AUDIT-FIX(#1): Do not start additional reads while a previous bounded snapshot
                    # worker is still stuck in transport code.
                    self._invalidate_recent_speech()
                    return self._degraded_snapshot(
                        device_runtime_mode="provider_snapshot_timeout",
                        transport_reason="provider_snapshot_timeout",
                    )
                self._stuck_snapshot_thread = None

            primitive = self._capture_snapshot_bounded()
            now_monotonic = self._read_monotonic_clock()
            assistant_output_active = self._assistant_output_active()
            speech_detected = primitive.direction.speech_detected
            derived = derive_respeaker_signal_state(
                primitive.direction,
                assistant_output_active=assistant_output_active,
            )
            recent_speech_age_s = self._compute_recent_speech_age_s(
                host_control_ready=primitive.host_control_ready,
                now_monotonic=now_monotonic,
                speech_detected=speech_detected,
                assistant_output_active=assistant_output_active,
                speech_overlap_likely=derived.speech_overlap_likely,
            )
            snapshot = ReSpeakerSignalSnapshot(
                captured_at=primitive.captured_at,
                source=_SOURCE,
                source_type=_SOURCE_TYPE,
                sensor_window_ms=self.sensor_window_ms,
                device_runtime_mode=primitive.device_runtime_mode,
                host_control_ready=primitive.host_control_ready,
                transport_reason=primitive.transport.reason,
                requires_elevated_permissions=primitive.transport.requires_elevated_permissions,
                firmware_version=primitive.firmware_version,
                speech_detected=speech_detected,
                room_quiet=primitive.direction.room_quiet,
                recent_speech_age_s=recent_speech_age_s,
                assistant_output_active=assistant_output_active,
                azimuth_deg=primitive.direction.doa_degrees,
                direction_confidence=derived.direction_confidence,
                beam_activity=primitive.direction.beam_speech_energies,
                speech_overlap_likely=derived.speech_overlap_likely,
                barge_in_detected=derived.barge_in_detected,
                mute_active=primitive.mute.mute_active,
                gpo_logic_levels=primitive.mute.gpo_logic_levels,
            )
            return replace(
                snapshot,
                claim_contract=build_respeaker_signal_claim_contract(snapshot),
            )
        except _TransportInitError:
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="transport_init_error",
                transport_reason="transport_init_error",
            )
        except _SnapshotTimeoutError:
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="provider_snapshot_timeout",
                transport_reason="provider_snapshot_timeout",
            )
        except Exception:
            # AUDIT-FIX(#6): Keep runtime-facing error reasons generic and log full detail server-side
            # instead of leaking exception class names in public signal payloads.
            _LOG.exception("ReSpeaker signal observation failed")
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="signal_provider_error",
                transport_reason="signal_provider_error",
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        """Close the provider.

        The current libusb snapshot path is per-call and does not hold open
        device resources between observations, so close is intentionally a
        best-effort logical shutdown only.
        """

        # AUDIT-FIX(#7): Mark the provider closed so callers stop probing hardware after shutdown.
        self._closed = True
        return None

    def _assistant_output_active(self) -> bool | None:
        """Return whether Twinr is currently speaking when a callback exists."""

        # AUDIT-FIX(#5): Absence of a predicate means "unknown", not False.
        if self._assistant_output_active_predicate is None:
            return None
        try:
            return bool(self._assistant_output_active_predicate())
        except Exception:
            return None

    def _capture_snapshot_bounded(self) -> ReSpeakerPrimitiveSnapshot:
        """Capture one primitive snapshot with a hard caller-side timeout."""

        result_queue: Queue[tuple[str, Any]] = Queue(maxsize=1)

        def _worker() -> None:
            try:
                transport = self._get_or_create_transport()
                primitive = self._snapshot_factory(transport=transport)
            except Exception as exc:
                result_queue.put(("error", exc))
            else:
                result_queue.put(("ok", primitive))

        worker = Thread(
            target=_worker,
            name="respeaker-snapshot",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=self.snapshot_timeout_s)
        if worker.is_alive():
            # AUDIT-FIX(#1): Keep exactly one timed-out daemon worker reference so we can avoid
            # piling up more blocked readers while still allowing automatic recovery once it exits.
            self._stuck_snapshot_thread = worker
            raise _SnapshotTimeoutError("bounded snapshot read timed out")

        self._stuck_snapshot_thread = None
        try:
            outcome, value = result_queue.get_nowait()
        except Empty as exc:
            raise RuntimeError("snapshot worker exited without publishing a result") from exc

        if outcome == "error":
            raise value
        return value

    def _compute_recent_speech_age_s(
        self,
        *,
        host_control_ready: bool,
        now_monotonic: float,
        speech_detected: bool | None,
        assistant_output_active: bool | None,
        speech_overlap_likely: bool | None,
    ) -> float | None:
        """Compute recent speech age without guessing across blind intervals."""

        if host_control_ready is not True:
            # AUDIT-FIX(#4): Unknown host-control state breaks continuity, so any previous speech
            # timestamp becomes untrustworthy and must be discarded.
            self._invalidate_recent_speech()
            return None

        recent_speech_detected = (
            speech_overlap_likely if assistant_output_active is True else speech_detected
        )
        if recent_speech_detected is True:
            self._last_speech_monotonic = now_monotonic
            return 0.0

        if self._last_speech_monotonic is None:
            return None

        return max(0.0, now_monotonic - self._last_speech_monotonic)

    def _degraded_snapshot(
        self,
        *,
        device_runtime_mode: str,
        transport_reason: str,
    ) -> ReSpeakerSignalSnapshot:
        """Build one conservative degraded snapshot."""

        return ReSpeakerSignalSnapshot(
            captured_at=time.time(),
            source=_SOURCE,
            source_type=_SOURCE_TYPE,
            sensor_window_ms=self.sensor_window_ms,
            device_runtime_mode=device_runtime_mode,
            host_control_ready=False,
            transport_reason=transport_reason,
            recent_speech_age_s=None,
            assistant_output_active=self._assistant_output_active(),
        )

    def _get_or_create_transport(self) -> ReSpeakerLibusbTransport:
        """Return the injected transport or construct one lazily."""

        if self.transport is not None:
            return self.transport
        try:
            self.transport = ReSpeakerLibusbTransport()
        except Exception as exc:
            _LOG.exception("Failed to initialize ReSpeaker libusb transport")
            raise _TransportInitError("failed to initialize ReSpeaker libusb transport") from exc
        return self.transport

    def _invalidate_recent_speech(self) -> None:
        """Forget cached speech timing when observation continuity is lost."""

        self._last_speech_monotonic = None

    def _read_monotonic_clock(self) -> float:
        """Read and validate the injected monotonic clock."""

        value = float(self._monotonic_clock())
        if not math.isfinite(value):
            raise ValueError("monotonic clock returned non-finite value")
        return value

    @staticmethod
    def _coerce_non_negative_int(value: object, *, default: int) -> int:
        """Convert to a non-negative int without propagating config parse errors."""

        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return max(0, parsed)

    @staticmethod
    def _coerce_min_float(value: object, *, minimum: float, default: float) -> float:
        """Convert to a finite float with a lower bound and safe fallback."""

        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(parsed):
            return default
        return max(minimum, parsed)
