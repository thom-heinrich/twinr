"""Dependency-light libusb transport for XVF3800 host-control I/O."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_ubyte, c_uint16, c_void_p
import ctypes
import ctypes.util
import math  # AUDIT-FIX(#2): Needed for finite-range validation of timing configuration.
import struct
from threading import RLock
import time

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerProbeResult,
    ReSpeakerTransportAvailability,
)


_CONTROL_READ_REQUEST_TYPE = 0xC0
_CONTROL_WRITE_REQUEST_TYPE = 0x40
_CONTROL_REQUEST = 0
_CONTROL_SUCCESS = 0
_SERVICER_COMMAND_RETRY = 64
_DEFAULT_READ_TIMEOUT_MS = 1000
_DEFAULT_RETRY_SLEEP_S = 0.01
_DEFAULT_MAX_RETRY_ATTEMPTS = 100
_DEFAULT_MAX_SINGLE_READ_DURATION_S = 5.0  # AUDIT-FIX(#1): Hard-cap one blocking read so the transport cannot stall the app for ~100s/spec.
_MAX_REASON_LENGTH = 96  # AUDIT-FIX(#5): Keep upstream reason strings short and stable.
_MAX_SAFE_UINT16 = 0xFFFF  # AUDIT-FIX(#2): libusb control-transfer fields are uint16_t.
_MAX_SAFE_READ_TIMEOUT_MS = int(_DEFAULT_MAX_SINGLE_READ_DURATION_S * 1000)  # AUDIT-FIX(#1): A single libusb call must stay within the overall per-read budget.
_MAX_SAFE_RETRY_SLEEP_S = 1.0  # AUDIT-FIX(#1): Prevent configuration from injecting multi-second sleeps between retries.
_MAX_SAFE_RETRY_ATTEMPTS = 1000  # AUDIT-FIX(#2): Defend against pathological configuration values.
_XVF3800_VENDOR_ID = 0x2886
_XVF3800_PRODUCT_ID = 0x001A
_TRANSPORT_IO_LOCK = RLock()


def _sanitize_reason(value: object, *, fallback: str) -> str:
    """Return a short single-line reason string safe to propagate upstream."""

    # AUDIT-FIX(#5): Normalize low-level exception/error text so callers do not receive unstable multiline internals.
    text = " ".join(str(value).split())
    if not text:
        return fallback
    return text[:_MAX_REASON_LENGTH]


def _parse_uint16(value: object, *, default: int) -> tuple[int, bool]:
    """Parse a uint16 configuration value."""

    # AUDIT-FIX(#2): Reject invalid or out-of-range USB IDs instead of letting ctypes wrap them implicitly.
    if isinstance(value, bool):
        return default, False
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default, False
    if 0 <= parsed <= _MAX_SAFE_UINT16:
        return parsed, True
    return default, False


def _parse_bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Parse an integer with a safe fallback and clamp."""

    # AUDIT-FIX(#2): Keep timing/retry configuration inside safe bounds for the blocking C transport.
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Parse a float with a safe fallback and clamp."""

    # AUDIT-FIX(#2): Defend against NaN/inf/negative sleep budgets that would break retry timing.
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(maximum, parsed))


def _expected_payload_length(spec: ReSpeakerParameterSpec) -> int | None:
    """Return the fixed payload length for the spec, or None for variable-width CHAR values."""

    try:
        value_count = int(spec.value_count)
    except (AttributeError, TypeError, ValueError):
        return -1
    if value_count < 0:
        return -1
    if spec.value_type is ReSpeakerParameterType.CHAR:
        return None
    if spec.value_type is ReSpeakerParameterType.UINT8:
        return value_count
    if spec.value_type is ReSpeakerParameterType.UINT16:
        return value_count * 2
    if spec.value_type in (
        ReSpeakerParameterType.UINT32,
        ReSpeakerParameterType.INT32,
        ReSpeakerParameterType.FLOAT,
        ReSpeakerParameterType.RADIANS,
    ):
        return value_count * 4
    return -1


def _validate_parameter_spec(spec: ReSpeakerParameterSpec) -> str | None:
    """Validate a read spec before it is passed to ctypes/libusb."""

    # AUDIT-FIX(#2): Fail malformed specs early instead of allocating unbounded buffers or wrapping uint16 fields in ctypes.
    try:
        read_length = int(spec.read_length)
        request_value = int(spec.request_value)
        resid = int(spec.resid)
        value_count = int(spec.value_count)
    except (AttributeError, TypeError, ValueError) as exc:
        return f"invalid_spec:{exc.__class__.__name__}"

    if read_length < 1:
        return "invalid_spec:read_length_must_include_status_byte"
    if read_length > _MAX_SAFE_UINT16:
        return "invalid_spec:read_length_exceeds_usb_control_limit"
    if request_value < 0 or request_value > _MAX_SAFE_UINT16:
        return "invalid_spec:request_value_out_of_range"
    if resid < 0 or resid > _MAX_SAFE_UINT16:
        return "invalid_spec:resid_out_of_range"
    if value_count < 0:
        return "invalid_spec:value_count_out_of_range"

    payload_length = _expected_payload_length(spec)
    if payload_length == -1:
        return "invalid_spec:unsupported_value_type"
    if payload_length is not None and payload_length > (read_length - 1):
        return "invalid_spec:payload_larger_than_buffer"
    return None


def _normalize_write_values(
    spec: ReSpeakerParameterSpec,
    values: Sequence[object] | str | bytes | bytearray,
) -> bytes:
    """Encode one write payload according to the XVF3800 host-control schema."""

    access_mode = str(getattr(spec, "access_mode", "") or "").strip().lower()
    if access_mode == "ro":
        raise ValueError("write_not_allowed_for_read_only_spec")

    value_count = int(spec.value_count)
    if value_count < 0:
        raise ValueError("invalid_value_count")

    if spec.value_type is ReSpeakerParameterType.CHAR:
        if isinstance(values, str):
            encoded = values.encode("utf-8")
        elif isinstance(values, (bytes, bytearray)):
            encoded = bytes(values)
        else:
            encoded = bytes(values)
        if len(encoded) > value_count:
            raise ValueError("char_payload_exceeds_spec_length")
        return encoded

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("write_values_must_be_a_numeric_sequence")

    normalized = tuple(values)
    if len(normalized) != value_count:
        raise ValueError("write_value_count_mismatch")

    if spec.value_type is ReSpeakerParameterType.UINT8:
        payload = bytearray()
        for index, value in enumerate(normalized):
            if isinstance(value, bool):
                raise TypeError(f"write_value[{index}] must be an int")
            integer = int(value)
            if integer < 0 or integer > 0xFF:
                raise ValueError(f"write_value[{index}] out_of_range_uint8")
            payload.extend(integer.to_bytes(1, byteorder="little", signed=False))
        return bytes(payload)

    if spec.value_type is ReSpeakerParameterType.UINT16:
        return struct.pack(
            "<" + ("H" * value_count),
            *(
                _coerce_bounded_write_int(value, minimum=0, maximum=0xFFFF, index=index)
                for index, value in enumerate(normalized)
            ),
        )

    if spec.value_type is ReSpeakerParameterType.UINT32:
        return struct.pack(
            "<" + ("I" * value_count),
            *(
                _coerce_bounded_write_int(value, minimum=0, maximum=0xFFFFFFFF, index=index)
                for index, value in enumerate(normalized)
            ),
        )

    if spec.value_type is ReSpeakerParameterType.INT32:
        return struct.pack(
            "<" + ("i" * value_count),
            *(
                _coerce_bounded_write_int(
                    value,
                    minimum=-(2**31),
                    maximum=(2**31) - 1,
                    index=index,
                )
                for index, value in enumerate(normalized)
            ),
        )

    if spec.value_type in (ReSpeakerParameterType.FLOAT, ReSpeakerParameterType.RADIANS):
        return struct.pack(
            "<" + ("f" * value_count),
            *(
                _coerce_finite_write_float(value, index=index)
                for index, value in enumerate(normalized)
            ),
        )

    raise ValueError("unsupported_write_value_type")


def _coerce_bounded_write_int(value: object, *, minimum: int, maximum: int, index: int) -> int:
    """Validate one integer write value before it reaches ctypes/libusb."""

    if isinstance(value, bool):
        raise TypeError(f"write_value[{index}] must be an int")
    integer = int(value)
    if integer < minimum or integer > maximum:
        raise ValueError(f"write_value[{index}] out_of_range")
    return integer


def _coerce_finite_write_float(value: object, *, index: int) -> float:
    """Validate one float-style write value before USB encoding."""

    if isinstance(value, bool):
        raise TypeError(f"write_value[{index}] must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"write_value[{index}] must be finite")
    return normalized


class _LibusbBindings:
    """Wrap the tiny libusb surface needed for XVF3800 host-control reads."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._configure_signatures()

    @classmethod
    def from_system(cls) -> "_LibusbBindings | None":
        library_name = ctypes.util.find_library("usb-1.0")
        if not library_name:
            return None
        try:
            return cls(ctypes.CDLL(library_name))
        except (AttributeError, OSError, TypeError):  # AUDIT-FIX(#3): Treat missing symbols / bad shared-library surfaces as transport-unavailable, not startup crashes.
            return None

    def _configure_signatures(self) -> None:
        self._library.libusb_init.argtypes = [POINTER(c_void_p)]
        self._library.libusb_init.restype = c_int
        self._library.libusb_exit.argtypes = [c_void_p]
        self._library.libusb_open_device_with_vid_pid.argtypes = [c_void_p, c_uint16, c_uint16]
        self._library.libusb_open_device_with_vid_pid.restype = c_void_p
        self._library.libusb_close.argtypes = [c_void_p]
        self._library.libusb_control_transfer.argtypes = [
            c_void_p,
            c_ubyte,
            c_ubyte,
            c_uint16,
            c_uint16,
            POINTER(c_ubyte),
            c_uint16,
            c_uint,
        ]
        self._library.libusb_control_transfer.restype = c_int
        self._library.libusb_error_name.argtypes = [c_int]
        self._library.libusb_error_name.restype = c_char_p

    def init_context(self) -> c_void_p:
        context = c_void_p()
        result = self._library.libusb_init(byref(context))
        if result != 0:
            raise OSError(f"libusb_init failed with code {result}")
        return context

    def open_device(self, context: c_void_p, vendor_id: int, product_id: int) -> c_void_p | None:
        handle = self._library.libusb_open_device_with_vid_pid(context, vendor_id, product_id)
        return handle or None

    def close(self, handle: c_void_p | None) -> None:
        if handle:
            self._library.libusb_close(handle)

    def exit(self, context: c_void_p | None) -> None:
        if context:
            self._library.libusb_exit(context)

    def control_transfer(
        self,
        handle: c_void_p,
        *,
        request_type: int,
        request: int,
        value: int,
        index: int,
        buffer: object,
        timeout_ms: int,
    ) -> int:
        buffer_length = len(buffer)
        if buffer_length < 0 or buffer_length > _MAX_SAFE_UINT16:  # AUDIT-FIX(#2): libusb_control_transfer takes uint16_t wLength; reject impossible buffers before ctypes wraps them.
            raise ValueError("buffer_length_out_of_range")
        return int(
            self._library.libusb_control_transfer(
                handle,
                request_type,
                request,
                value,
                index,
                buffer,
                buffer_length,
                timeout_ms,
            )
        )

    def error_name(self, code: int) -> str:
        try:
            return str(self._library.libusb_error_name(int(code)).decode("utf-8", errors="replace"))
        except Exception:
            return f"libusb_error_{code}"


class ReSpeakerLibusbTransport:
    """Read and write bounded XVF3800 host-control parameters via system libusb."""

    def __init__(
        self,
        *,
        vendor_id: int = _XVF3800_VENDOR_ID,
        product_id: int = _XVF3800_PRODUCT_ID,
        read_timeout_ms: int = _DEFAULT_READ_TIMEOUT_MS,
        max_retry_attempts: int = _DEFAULT_MAX_RETRY_ATTEMPTS,
        retry_sleep_s: float = _DEFAULT_RETRY_SLEEP_S,
        max_single_read_duration_s: float = _DEFAULT_MAX_SINGLE_READ_DURATION_S,
        bindings: object | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.vendor_id, vendor_valid = _parse_uint16(vendor_id, default=_XVF3800_VENDOR_ID)  # AUDIT-FIX(#2): Validate public USB IDs instead of allowing implicit ctypes wrapping.
        self.product_id, product_valid = _parse_uint16(product_id, default=_XVF3800_PRODUCT_ID)  # AUDIT-FIX(#2): Same for the product ID.
        self.max_single_read_duration_s = _parse_bounded_float(
            max_single_read_duration_s,
            default=_DEFAULT_MAX_SINGLE_READ_DURATION_S,
            minimum=0.25,
            maximum=_DEFAULT_MAX_SINGLE_READ_DURATION_S,
        )  # AUDIT-FIX(#1): Keep a hard upper bound for one blocking parameter read.
        self.read_timeout_ms = min(
            _parse_bounded_int(
                read_timeout_ms,
                default=_DEFAULT_READ_TIMEOUT_MS,
                minimum=1,
                maximum=_MAX_SAFE_READ_TIMEOUT_MS,
            ),
            max(1, int(self.max_single_read_duration_s * 1000)),
        )  # AUDIT-FIX(#1): Ensure one libusb call cannot outlive the per-read deadline.
        self.max_retry_attempts = _parse_bounded_int(
            max_retry_attempts,
            default=_DEFAULT_MAX_RETRY_ATTEMPTS,
            minimum=1,
            maximum=_MAX_SAFE_RETRY_ATTEMPTS,
        )  # AUDIT-FIX(#2): Prevent pathological retry counts from stretching a failure indefinitely.
        self.retry_sleep_s = _parse_bounded_float(
            retry_sleep_s,
            default=_DEFAULT_RETRY_SLEEP_S,
            minimum=0.0,
            maximum=_MAX_SAFE_RETRY_SLEEP_S,
        )  # AUDIT-FIX(#1): Bound backoff sleeps so they cannot freeze the caller between retries.
        self._config_error: str | None = None
        if not vendor_valid:
            self._config_error = "vendor_id_out_of_range"  # AUDIT-FIX(#2): Fail closed on bad USB IDs instead of talking to the wrong device.
        elif not product_valid:
            self._config_error = "product_id_out_of_range"  # AUDIT-FIX(#2): Fail closed on bad USB IDs instead of talking to the wrong device.
        self._bindings = bindings if bindings is not None else _LibusbBindings.from_system()
        self._sleep = sleep_fn or time.sleep
        self._time = time_fn or time.time
        self._monotonic = time.monotonic  # AUDIT-FIX(#1): Retry budgets must use a clock that cannot move backwards.

    def capture_reads(
        self,
        specs: Iterable[ReSpeakerParameterSpec],
        *,
        probe: ReSpeakerProbeResult | None = None,
    ) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
        """Read one bounded batch of host-control parameters."""

        if self._config_error is not None:
            return (
                ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason=f"invalid_config:{self._config_error}",
                ),
                {},
            )

        if self._bindings is None:
            return (
                ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason="libusb_unavailable",
                ),
                {},
            )

        with _TRANSPORT_IO_LOCK:
            context = None
            handle = None
            reads: dict[str, ReSpeakerParameterRead] = {}
            try:
                context = self._bindings.init_context()
                handle = self._bindings.open_device(context, self.vendor_id, self.product_id)
                if handle is None:
                    return (
                        self._availability_for_open_failure(probe),
                        {},
                    )
                for index, spec in enumerate(specs):  # AUDIT-FIX(#3): Iterate inside the transport guard so generator failures become clean transport errors with any earlier reads preserved.
                    spec_name = self._unique_spec_name(reads, self._spec_name(spec, index))  # AUDIT-FIX(#4): Preserve all reads even when spec names collide.
                    try:
                        reads[spec_name] = self._read_parameter(handle, spec)
                    except Exception as exc:  # AUDIT-FIX(#3): One broken spec or ctypes edge case must not abort the entire batch.
                        reads[spec_name] = self._failed_read(
                            spec,
                            attempt_count=0,
                            error=f"read_exception:{exc.__class__.__name__}",
                        )
                return (
                    ReSpeakerTransportAvailability(backend="libusb", available=True),
                    reads,
                )
            except Exception as exc:  # AUDIT-FIX(#3): Narrow OSError-only handling missed ctypes/signature/generator failures and could crash the caller.
                return (
                    ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason=self._transport_error_reason(exc),
                    ),
                    reads,
                )
            finally:
                self._safe_release(handle=handle, context=context)  # AUDIT-FIX(#6): Cleanup must never mask the original transport outcome.

    def write_parameter(
        self,
        spec: ReSpeakerParameterSpec,
        values: Sequence[object] | str | bytes | bytearray,
        *,
        probe: ReSpeakerProbeResult | None = None,
    ) -> ReSpeakerTransportAvailability:
        """Write one bounded XVF3800 host-control parameter."""

        if self._config_error is not None:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason=f"invalid_config:{self._config_error}",
            )

        if self._bindings is None:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason="libusb_unavailable",
            )

        try:
            payload = _normalize_write_values(spec, values)
        except Exception as exc:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason=f"invalid_write:{_sanitize_reason(exc, fallback=exc.__class__.__name__)}",
            )

        context = None
        handle = None
        with _TRANSPORT_IO_LOCK:
            try:
                context = self._bindings.init_context()
                handle = self._bindings.open_device(context, self.vendor_id, self.product_id)
                if handle is None:
                    return self._availability_for_open_failure(probe)

                buffer = (c_ubyte * len(payload)).from_buffer_copy(payload)
                transfer_size = self._bindings.control_transfer(
                    handle,
                    request_type=_CONTROL_WRITE_REQUEST_TYPE,
                    request=_CONTROL_REQUEST,
                    value=int(spec.cmdid),
                    index=int(spec.resid),
                    buffer=buffer,
                    timeout_ms=self.read_timeout_ms,
                )
                if transfer_size < 0:
                    return ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason=self._binding_error_name(transfer_size),
                    )
                if transfer_size != len(payload):
                    return ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason=f"short_write:{transfer_size}/{len(payload)}",
                    )
                return ReSpeakerTransportAvailability(backend="libusb", available=True)
            except Exception as exc:
                return ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason=self._transport_error_reason(exc),
                )
            finally:
                self._safe_release(handle=handle, context=context)

    def _availability_for_open_failure(self, probe: ReSpeakerProbeResult | None) -> ReSpeakerTransportAvailability:
        if probe is not None and not probe.usb_visible:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason="device_not_visible",
            )
        return ReSpeakerTransportAvailability(
            backend="libusb",
            available=False,
            reason="permission_denied_or_transport_blocked",
            requires_elevated_permissions=True,
        )

    def _transport_error_reason(self, exc: Exception) -> str:
        # AUDIT-FIX(#5): Surface a stable class-based reason and only include sanitized detail for known libusb-style OSError messages.
        error_type = exc.__class__.__name__
        if isinstance(exc, OSError):
            detail = _sanitize_reason(exc, fallback=error_type)
            if detail.startswith("libusb_") or "failed with code" in detail:
                return f"transport_error:{error_type}:{detail}"
        return f"transport_error:{error_type}"

    def _binding_error_name(self, code: int) -> str:
        # AUDIT-FIX(#5): Keep C-library error strings single-line and bounded before attaching them to reads.
        try:
            if self._bindings is not None:
                return _sanitize_reason(
                    self._bindings.error_name(code),
                    fallback=f"libusb_error_{code}",
                )
        except Exception:
            pass
        return f"libusb_error_{code}"

    def _spec_name(self, spec: object, index: int) -> str:
        # AUDIT-FIX(#4): Give malformed/unnamed specs deterministic keys so one bad entry does not erase another result.
        name = getattr(spec, "name", None)
        if isinstance(name, str) and name:
            return name
        return f"unnamed_spec_{index}"

    def _unique_spec_name(self, reads: dict[str, ReSpeakerParameterRead], base_name: str) -> str:
        # AUDIT-FIX(#4): Avoid silent overwrites when the batch contains duplicate spec names.
        if base_name not in reads:
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name}__duplicate_{suffix}"
            if candidate not in reads:
                return candidate
            suffix += 1

    def _safe_release(self, *, handle: c_void_p | None, context: c_void_p | None) -> None:
        # AUDIT-FIX(#6): Never let best-effort cleanup hide the actual device/transport result.
        if self._bindings is None:
            return
        try:
            self._bindings.close(handle)
        except Exception:
            pass
        try:
            self._bindings.exit(context)
        except Exception:
            pass

    def _failed_read(
        self,
        spec: ReSpeakerParameterSpec,
        *,
        attempt_count: int,
        error: str,
        captured_at: float | None = None,
        status_code: int | None = None,
    ) -> ReSpeakerParameterRead:
        kwargs = {
            "spec": spec,
            "captured_at": self._time() if captured_at is None else captured_at,
            "ok": False,
            "attempt_count": max(0, int(attempt_count)),
            "error": _sanitize_reason(error, fallback="transport_error"),  # AUDIT-FIX(#5): Keep per-read error strings bounded and automation-friendly.
        }
        if status_code is not None:
            kwargs["status_code"] = status_code
        return ReSpeakerParameterRead(**kwargs)

    def _read_parameter(self, handle: object, spec: ReSpeakerParameterSpec) -> ReSpeakerParameterRead:
        validation_error = _validate_parameter_spec(spec)
        if validation_error is not None:
            return self._failed_read(
                spec,
                attempt_count=0,
                error=validation_error,
            )  # AUDIT-FIX(#2): Refuse malformed specs before they reach ctypes/libusb.

        attempts = 0
        deadline = self._monotonic() + self.max_single_read_duration_s  # AUDIT-FIX(#1): Bound total retry time with a monotonic deadline.
        deadline_exhausted = False
        while attempts < self.max_retry_attempts:
            if attempts > 0 and self._monotonic() >= deadline:
                deadline_exhausted = True
                break

            attempts += 1
            buffer = (c_ubyte * int(spec.read_length))()  # AUDIT-FIX(#2): Safe because the spec was range-validated above.
            transfer_size = self._bindings.control_transfer(
                handle,
                request_type=_CONTROL_READ_REQUEST_TYPE,
                request=_CONTROL_REQUEST,
                value=int(spec.request_value),
                index=int(spec.resid),
                buffer=buffer,
                timeout_ms=self.read_timeout_ms,
            )
            captured_at = self._time()
            if transfer_size < 0:
                return self._failed_read(
                    spec,
                    captured_at=captured_at,
                    attempt_count=attempts,
                    error=self._binding_error_name(transfer_size),
                )
            if transfer_size > len(buffer):
                return self._failed_read(
                    spec,
                    captured_at=captured_at,
                    attempt_count=attempts,
                    error="transfer_size_exceeded_buffer",
                )  # AUDIT-FIX(#2): Guard against impossible bindings/mocks before slicing raw bytes.
            response = bytes(buffer[:transfer_size])
            if not response:
                return self._failed_read(
                    spec,
                    captured_at=captured_at,
                    attempt_count=attempts,
                    error="empty_response",
                )
            status_code = int(response[0])
            if status_code == _CONTROL_SUCCESS:
                decoded = _decode_parameter_bytes(spec, response[1:])
                if decoded is None:
                    return self._failed_read(
                        spec,
                        captured_at=captured_at,
                        attempt_count=attempts,
                        status_code=status_code,
                        error="decode_error",
                    )
                return ReSpeakerParameterRead(
                    spec=spec,
                    captured_at=captured_at,
                    ok=True,
                    attempt_count=attempts,
                    status_code=status_code,
                    decoded_value=decoded,
                )
            if status_code != _SERVICER_COMMAND_RETRY:
                return self._failed_read(
                    spec,
                    captured_at=captured_at,
                    attempt_count=attempts,
                    status_code=status_code,
                    error=f"device_status_{status_code}",
                )
            if attempts < self.max_retry_attempts and self.retry_sleep_s > 0.0:
                remaining_sleep = deadline - self._monotonic()
                if remaining_sleep <= 0.0:
                    deadline_exhausted = True
                    break
                self._sleep(min(self.retry_sleep_s, remaining_sleep))  # AUDIT-FIX(#1): Never sleep longer than the remaining per-read budget.
        return self._failed_read(
            spec,
            captured_at=self._time(),
            attempt_count=attempts,
            status_code=_SERVICER_COMMAND_RETRY,
            error="servicer_retry_deadline_exhausted" if deadline_exhausted else "servicer_retry_exhausted",
        )


def _decode_parameter_bytes(
    spec: ReSpeakerParameterSpec,
    payload: bytes,
) -> tuple[int | float, ...] | str | None:
    try:
        value_count = int(spec.value_count)
        if value_count < 0:
            return None
        if spec.value_type is ReSpeakerParameterType.CHAR:
            char_payload = payload[:value_count] if value_count > 0 else payload
            return char_payload.split(b"\x00", 1)[0].decode("utf-8", errors="replace")  # AUDIT-FIX(#7): Stop at the first NUL terminator instead of leaking garbage suffix bytes.
        if spec.value_type is ReSpeakerParameterType.UINT8:
            if len(payload) < value_count:
                return None
            return tuple(int(value) for value in payload[:value_count])
        if spec.value_type is ReSpeakerParameterType.UINT16:
            required = value_count * 2
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("H" * value_count), payload[:required])
        if spec.value_type is ReSpeakerParameterType.UINT32:
            required = value_count * 4
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("I" * value_count), payload[:required])
        if spec.value_type is ReSpeakerParameterType.INT32:
            required = value_count * 4
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("i" * value_count), payload[:required])
        if spec.value_type in (ReSpeakerParameterType.FLOAT, ReSpeakerParameterType.RADIANS):
            required = value_count * 4
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("f" * value_count), payload[:required])
    except (AttributeError, OverflowError, struct.error, UnicodeDecodeError, ValueError):
        return None
    return None
