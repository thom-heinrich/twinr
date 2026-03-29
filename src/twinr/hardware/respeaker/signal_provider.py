# CHANGELOG: 2026-03-28
# BUG-1: `sensor_window_ms` now has real effect; recent primitive snapshots are reused within the
#        requested window instead of hitting the USB control path on every `observe()`.
# BUG-2: Default transport lifecycle is now truly per-observation; stale libusb handles are no
#        longer retained across disconnect/reconnect cycles or leaked past `close()`.
# BUG-3: Degraded snapshots now also carry a claim contract, matching the successful path.
# SEC-1: Added failure backoff/circuit-breaker behavior so a hostile or faulting USB peripheral
#        cannot force unbounded high-rate snapshot thread churn and repeated transport hammering.
# IMP-1: Timeout hints are now pushed down into the transport/snapshot layer when supported
#        (`timeout_ms`/`timeout_s`, transport timeout setters/defaults), instead of relying only
#        on caller-side thread joins.
# IMP-2: Transport cleanup is deterministic for owned transports, with best-effort PyUSB resource
#        disposal when available.
"""Expose runtime-facing XVF3800 signal observations for Twinr."""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import replace
from queue import Empty, Queue
from threading import Lock, Thread
import time
from typing import Any, Callable

from twinr.hardware.respeaker.claim_contract import build_respeaker_signal_claim_contract
from twinr.hardware.respeaker.derived_signals import derive_respeaker_signal_state
from twinr.hardware.respeaker.models import ReSpeakerPrimitiveSnapshot, ReSpeakerSignalSnapshot
from twinr.hardware.respeaker.snapshot_service import capture_respeaker_primitive_snapshot
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport


_DEFAULT_LOCK_TIMEOUT_S = 0.25
_DEFAULT_SNAPSHOT_TIMEOUT_S = 0.75
_DEFAULT_FAILURE_BACKOFF_BASE_S = 0.25
_DEFAULT_FAILURE_BACKOFF_MAX_S = 5.0
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

    2026 edge/runtime notes:
    - A bounded caller-side join is still kept as a last-resort containment
      layer for wedged native USB calls.
    - By default, transports are now short-lived per observation rather than
      sticky across the lifetime of the provider. This aligns better with
      hotplug/re-enumeration reality for USB control paths.
    - When the deeper transport or snapshot layer exposes native timeout knobs,
      this provider passes timeout hints down so the stack can fail fast before
      the outer thread guard is needed.
    """

    def __init__(
        self,
        *,
        sensor_window_ms: int = 0,
        lock_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
        snapshot_timeout_s: float = _DEFAULT_SNAPSHOT_TIMEOUT_S,
        transport: ReSpeakerLibusbTransport | None = None,
        transport_factory: Callable[[], ReSpeakerLibusbTransport] | None = None,
        close_injected_transport_on_close: bool = False,
        failure_backoff_base_s: float = _DEFAULT_FAILURE_BACKOFF_BASE_S,
        failure_backoff_max_s: float = _DEFAULT_FAILURE_BACKOFF_MAX_S,
        snapshot_factory: Callable[..., ReSpeakerPrimitiveSnapshot] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        assistant_output_active_predicate: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize one provider with bounded lock and snapshot settings."""

        self.sensor_window_ms = self._coerce_non_negative_int(sensor_window_ms, default=0)
        self.lock_timeout_s = self._coerce_min_float(
            lock_timeout_s,
            minimum=0.01,
            default=_DEFAULT_LOCK_TIMEOUT_S,
        )
        self.snapshot_timeout_s = self._coerce_min_float(
            snapshot_timeout_s,
            minimum=0.05,
            default=max(self.lock_timeout_s, _DEFAULT_SNAPSHOT_TIMEOUT_S),
        )
        self.transport = transport
        self._transport_factory = (
            transport_factory if transport_factory is not None else ReSpeakerLibusbTransport
        )
        self._close_injected_transport_on_close = bool(close_injected_transport_on_close)
        self._failure_backoff_base_s = self._coerce_min_float(
            failure_backoff_base_s,
            minimum=0.01,
            default=_DEFAULT_FAILURE_BACKOFF_BASE_S,
        )
        self._failure_backoff_max_s = self._coerce_min_float(
            failure_backoff_max_s,
            minimum=self._failure_backoff_base_s,
            default=max(_DEFAULT_FAILURE_BACKOFF_MAX_S, self._failure_backoff_base_s),
        )
        self._snapshot_factory = (
            snapshot_factory if snapshot_factory is not None else capture_respeaker_primitive_snapshot
        )
        self._snapshot_factory_accepts_timeout_ms = self._callable_accepts_kwarg(
            self._snapshot_factory,
            "timeout_ms",
        )
        self._snapshot_factory_accepts_timeout_s = self._callable_accepts_kwarg(
            self._snapshot_factory,
            "timeout_s",
        )
        self._monotonic_clock = monotonic_clock if monotonic_clock is not None else time.monotonic
        self._assistant_output_active_predicate = assistant_output_active_predicate

        self._lock = Lock()
        self._last_speech_monotonic: float | None = None
        self._last_primitive_snapshot: ReSpeakerPrimitiveSnapshot | None = None
        self._last_primitive_monotonic: float | None = None
        self._stuck_snapshot_thread: Thread | None = None
        self._closed = False

        self._consecutive_failures = 0
        self._cooldown_until_monotonic = 0.0
        self._last_failure_mode = "signal_provider_error"
        self._last_failure_reason = "signal_provider_error"

    def observe(self) -> ReSpeakerSignalSnapshot:
        """Capture one conservative runtime-facing XVF3800 signal snapshot."""

        if self._closed:
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="provider_closed",
                transport_reason="provider_closed",
            )

        if not self._lock.acquire(timeout=self.lock_timeout_s):
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="provider_lock_timeout",
                transport_reason="provider_lock_timeout",
            )

        try:
            pre_observe_monotonic = self._read_monotonic_clock()

            if self._stuck_snapshot_thread is not None:
                if self._stuck_snapshot_thread.is_alive():
                    self._register_transport_failure(
                        now_monotonic=pre_observe_monotonic,
                        device_runtime_mode="provider_snapshot_timeout",
                        transport_reason="provider_snapshot_timeout",
                    )
                    self._invalidate_recent_speech()
                    return self._degraded_snapshot(
                        device_runtime_mode="provider_snapshot_timeout",
                        transport_reason="provider_snapshot_timeout",
                    )
                self._stuck_snapshot_thread = None

            if self._cooldown_is_open(pre_observe_monotonic):
                self._invalidate_recent_speech()
                return self._degraded_snapshot(
                    device_runtime_mode=self._last_failure_mode,
                    transport_reason=self._last_failure_reason,
                )

            assistant_output_active = self._assistant_output_active()
            primitive, observation_monotonic, live_capture = self._get_primitive_for_observation(
                pre_observe_monotonic,
            )
            now_monotonic = self._read_monotonic_clock()

            snapshot = self._build_snapshot_from_primitive(
                primitive,
                now_monotonic=now_monotonic,
                observation_monotonic=observation_monotonic,
                assistant_output_active=assistant_output_active,
            )

            if live_capture:
                self._record_live_capture_success(
                    primitive=primitive,
                    observation_monotonic=observation_monotonic,
                )

            return snapshot

        except _TransportInitError:
            self._register_transport_failure(
                now_monotonic=self._safe_read_monotonic_clock(),
                device_runtime_mode="transport_init_error",
                transport_reason="transport_init_error",
            )
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="transport_init_error",
                transport_reason="transport_init_error",
            )
        except _SnapshotTimeoutError:
            self._register_transport_failure(
                now_monotonic=self._safe_read_monotonic_clock(),
                device_runtime_mode="provider_snapshot_timeout",
                transport_reason="provider_snapshot_timeout",
            )
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="provider_snapshot_timeout",
                transport_reason="provider_snapshot_timeout",
            )
        except Exception:
            _LOG.exception("ReSpeaker signal observation failed")
            self._register_transport_failure(
                now_monotonic=self._safe_read_monotonic_clock(),
                device_runtime_mode="signal_provider_error",
                transport_reason="signal_provider_error",
            )
            self._invalidate_recent_speech()
            return self._degraded_snapshot(
                device_runtime_mode="signal_provider_error",
                transport_reason="signal_provider_error",
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        """Close the provider."""

        self._closed = True
        self._invalidate_recent_speech()
        self._last_primitive_snapshot = None
        self._last_primitive_monotonic = None

        if self._close_injected_transport_on_close and self.transport is not None:
            worker = self._stuck_snapshot_thread
            if worker is None or not worker.is_alive():
                self._safe_close_transport_instance(self.transport)
                self.transport = None

        return None

    def _assistant_output_active(self) -> bool | None:
        """Return whether Twinr is currently speaking when a callback exists."""

        if self._assistant_output_active_predicate is None:
            return None
        try:
            return bool(self._assistant_output_active_predicate())
        except Exception:
            return None

    def _get_primitive_for_observation(
        self,
        now_monotonic: float,
    ) -> tuple[ReSpeakerPrimitiveSnapshot, float, bool]:
        """Return a primitive snapshot, reusing a fresh cached one when allowed."""

        if self._can_reuse_last_primitive(now_monotonic):
            return self._last_primitive_snapshot, self._last_primitive_monotonic, False  # type: ignore[return-value]

        primitive = self._capture_snapshot_bounded()
        observation_monotonic = self._read_monotonic_clock()
        return primitive, observation_monotonic, True

    def _capture_snapshot_bounded(self) -> ReSpeakerPrimitiveSnapshot:
        """Capture one primitive snapshot with a hard caller-side timeout."""

        result_queue: Queue[tuple[str, Any]] = Queue(maxsize=1)

        def _worker() -> None:
            transport: ReSpeakerLibusbTransport | None = None
            owns_transport = False
            try:
                transport, owns_transport = self._get_transport_for_snapshot()
                primitive = self._call_snapshot_factory(transport)
            except Exception as exc:
                try:
                    result_queue.put(("error", exc))
                except Exception:
                    _LOG.exception("ReSpeaker snapshot worker failed to publish exception")
            else:
                try:
                    result_queue.put(("ok", primitive))
                except Exception:
                    _LOG.exception("ReSpeaker snapshot worker failed to publish result")
            finally:
                if owns_transport and transport is not None:
                    self._safe_close_transport_instance(transport)

        worker = Thread(
            target=_worker,
            name="respeaker-snapshot",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=self.snapshot_timeout_s)
        if worker.is_alive():
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

    def _build_snapshot_from_primitive(
        self,
        primitive: ReSpeakerPrimitiveSnapshot,
        *,
        now_monotonic: float,
        observation_monotonic: float,
        assistant_output_active: bool | None,
    ) -> ReSpeakerSignalSnapshot:
        """Build one runtime-facing snapshot from a primitive observation."""

        speech_detected = primitive.direction.speech_detected
        derived = derive_respeaker_signal_state(
            primitive.direction,
            assistant_output_active=assistant_output_active,
        )
        recent_speech_age_s = self._compute_recent_speech_age_s(
            host_control_ready=primitive.host_control_ready,
            now_monotonic=now_monotonic,
            observation_monotonic=observation_monotonic,
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
        return self._with_claim_contract(snapshot)

    def _compute_recent_speech_age_s(
        self,
        *,
        host_control_ready: bool,
        now_monotonic: float,
        observation_monotonic: float,
        speech_detected: bool | None,
        assistant_output_active: bool | None,
        speech_overlap_likely: bool | None,
    ) -> float | None:
        """Compute recent speech age without guessing across blind intervals."""

        if host_control_ready is not True:
            self._invalidate_recent_speech()
            return None

        recent_speech_detected = (
            speech_overlap_likely if assistant_output_active is True else speech_detected
        )
        if recent_speech_detected is True:
            self._last_speech_monotonic = observation_monotonic
            return max(0.0, now_monotonic - observation_monotonic)

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

        snapshot = ReSpeakerSignalSnapshot(
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
        return self._with_claim_contract(snapshot)

    def _get_transport_for_snapshot(
        self,
    ) -> tuple[ReSpeakerLibusbTransport, bool]:
        """Return an injected transport or construct a short-lived transport."""

        if self.transport is not None:
            return self.transport, False

        try:
            transport = self._transport_factory()
        except Exception as exc:
            _LOG.exception("Failed to initialize ReSpeaker libusb transport")
            raise _TransportInitError("failed to initialize ReSpeaker libusb transport") from exc

        self._configure_transport_timeout(transport)
        return transport, True

    def _call_snapshot_factory(
        self,
        transport: ReSpeakerLibusbTransport,
    ) -> ReSpeakerPrimitiveSnapshot:
        """Invoke the snapshot factory, passing timeout hints when supported."""

        kwargs: dict[str, Any] = {"transport": transport}
        timeout_ms = self._snapshot_timeout_ms()

        if self._snapshot_factory_accepts_timeout_ms:
            kwargs["timeout_ms"] = timeout_ms
        if self._snapshot_factory_accepts_timeout_s:
            kwargs["timeout_s"] = self.snapshot_timeout_s

        return self._snapshot_factory(**kwargs)

    def _configure_transport_timeout(self, transport: ReSpeakerLibusbTransport) -> None:
        """Best-effort timeout configuration for transports that expose it."""

        timeout_ms = self._snapshot_timeout_ms()

        for attr_name, value in (
            ("default_timeout_ms", timeout_ms),
            ("default_timeout", timeout_ms),
            ("timeout_ms", timeout_ms),
            ("timeout_s", self.snapshot_timeout_s),
        ):
            if hasattr(transport, attr_name):
                try:
                    setattr(transport, attr_name, value)
                except Exception:
                    _LOG.debug(
                        "Failed to set %s on ReSpeaker transport",
                        attr_name,
                        exc_info=True,
                    )

        for method_name, argument in (
            ("set_timeout_ms", timeout_ms),
            ("configure_timeout_ms", timeout_ms),
        ):
            method = getattr(transport, method_name, None)
            if callable(method):
                try:
                    method(argument)
                except Exception:
                    _LOG.debug(
                        "Failed to call %s on ReSpeaker transport",
                        method_name,
                        exc_info=True,
                    )

    def _record_live_capture_success(
        self,
        *,
        primitive: ReSpeakerPrimitiveSnapshot,
        observation_monotonic: float,
    ) -> None:
        """Remember a successful live capture and clear transport failure state."""

        self._last_primitive_snapshot = primitive
        self._last_primitive_monotonic = observation_monotonic
        self._consecutive_failures = 0
        self._cooldown_until_monotonic = 0.0

    def _register_transport_failure(
        self,
        *,
        now_monotonic: float,
        device_runtime_mode: str,
        transport_reason: str,
    ) -> None:
        """Open a small backoff window after transport-facing failures."""

        self._last_failure_mode = device_runtime_mode
        self._last_failure_reason = transport_reason
        self._consecutive_failures += 1

        exponent = max(0, self._consecutive_failures - 1)
        cooldown_s = min(
            self._failure_backoff_max_s,
            self._failure_backoff_base_s * (2**exponent),
        )
        self._cooldown_until_monotonic = max(
            self._cooldown_until_monotonic,
            now_monotonic + cooldown_s,
        )

    def _cooldown_is_open(self, now_monotonic: float) -> bool:
        """Return whether the provider is in temporary fault backoff."""

        return (
            self._consecutive_failures > 0
            and now_monotonic < self._cooldown_until_monotonic
        )

    def _can_reuse_last_primitive(self, now_monotonic: float) -> bool:
        """Return whether a recent primitive snapshot can be reused safely."""

        if self.sensor_window_ms <= 0:
            return False
        if self._consecutive_failures > 0:
            return False
        if self._last_primitive_snapshot is None or self._last_primitive_monotonic is None:
            return False

        age_s = now_monotonic - self._last_primitive_monotonic
        if age_s < 0.0:
            return False

        return age_s <= (self.sensor_window_ms / 1000.0)

    def _with_claim_contract(
        self,
        snapshot: ReSpeakerSignalSnapshot,
    ) -> ReSpeakerSignalSnapshot:
        """Attach the runtime claim contract to a snapshot."""

        try:
            claim_contract = build_respeaker_signal_claim_contract(snapshot)
        except Exception:
            _LOG.exception("Failed to build ReSpeaker signal claim contract")
            return snapshot

        return replace(snapshot, claim_contract=claim_contract)

    def _safe_close_transport_instance(self, transport: object) -> None:
        """Best-effort deterministic cleanup for a transport instance."""

        close_method = getattr(transport, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                _LOG.debug("Failed to close ReSpeaker transport", exc_info=True)

        dispose_method = getattr(transport, "dispose", None)
        if callable(dispose_method):
            try:
                dispose_method()
            except Exception:
                _LOG.debug("Failed to dispose ReSpeaker transport", exc_info=True)

        device = getattr(transport, "device", None)
        if device is not None:
            try:
                import usb.util  # type: ignore[import-not-found]
            except Exception:
                return None
            try:
                usb.util.dispose_resources(device)
            except Exception:
                _LOG.debug("Failed to dispose PyUSB device resources", exc_info=True)
        return None

    def _invalidate_recent_speech(self) -> None:
        """Forget cached speech timing when observation continuity is lost."""

        self._last_speech_monotonic = None

    def _read_monotonic_clock(self) -> float:
        """Read and validate the injected monotonic clock."""

        value = float(self._monotonic_clock())
        if not math.isfinite(value):
            raise ValueError("monotonic clock returned non-finite value")
        return value

    def _safe_read_monotonic_clock(self) -> float:
        """Read monotonic time with a conservative fallback for failure handling."""

        try:
            return self._read_monotonic_clock()
        except Exception:
            return time.monotonic()

    def _snapshot_timeout_ms(self) -> int:
        """Return the configured snapshot timeout in milliseconds."""

        return max(1, int(round(self.snapshot_timeout_s * 1000.0)))

    @staticmethod
    def _callable_accepts_kwarg(func: object, kwarg_name: str) -> bool:
        """Return whether a callable accepts a named keyword argument."""

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True

        return kwarg_name in signature.parameters

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