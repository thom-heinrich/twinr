"""Dependency-light libusb transport for XVF3800 host-control reads."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from ctypes import POINTER, byref, c_char_p, c_int, c_uint, c_ubyte, c_uint16, c_void_p
import ctypes
import ctypes.util
import struct
import time

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerProbeResult,
    ReSpeakerTransportAvailability,
)


_CONTROL_READ_REQUEST_TYPE = 0xC0
_CONTROL_REQUEST = 0
_CONTROL_SUCCESS = 0
_SERVICER_COMMAND_RETRY = 64
_DEFAULT_READ_TIMEOUT_MS = 1000
_DEFAULT_RETRY_SLEEP_S = 0.01
_DEFAULT_MAX_RETRY_ATTEMPTS = 100
_XVF3800_VENDOR_ID = 0x2886
_XVF3800_PRODUCT_ID = 0x001A


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
        except OSError:
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
        return int(
            self._library.libusb_control_transfer(
                handle,
                request_type,
                request,
                value,
                index,
                buffer,
                len(buffer),
                timeout_ms,
            )
        )

    def error_name(self, code: int) -> str:
        try:
            return str(self._library.libusb_error_name(int(code)).decode("utf-8", errors="replace"))
        except Exception:
            return f"libusb_error_{code}"


class ReSpeakerLibusbTransport:
    """Read bounded XVF3800 host-control parameters via system libusb."""

    def __init__(
        self,
        *,
        vendor_id: int = _XVF3800_VENDOR_ID,
        product_id: int = _XVF3800_PRODUCT_ID,
        read_timeout_ms: int = _DEFAULT_READ_TIMEOUT_MS,
        max_retry_attempts: int = _DEFAULT_MAX_RETRY_ATTEMPTS,
        retry_sleep_s: float = _DEFAULT_RETRY_SLEEP_S,
        bindings: object | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.vendor_id = int(vendor_id)
        self.product_id = int(product_id)
        self.read_timeout_ms = max(1, int(read_timeout_ms))
        self.max_retry_attempts = max(1, int(max_retry_attempts))
        self.retry_sleep_s = max(0.0, float(retry_sleep_s))
        self._bindings = bindings if bindings is not None else _LibusbBindings.from_system()
        self._sleep = sleep_fn or time.sleep
        self._time = time_fn or time.time

    def capture_reads(
        self,
        specs: Iterable[ReSpeakerParameterSpec],
        *,
        probe: ReSpeakerProbeResult | None = None,
    ) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
        """Read one bounded batch of host-control parameters."""

        spec_list = tuple(specs)
        if self._bindings is None:
            return (
                ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason="libusb_unavailable",
                ),
                {},
            )

        context = None
        handle = None
        try:
            context = self._bindings.init_context()
            handle = self._bindings.open_device(context, self.vendor_id, self.product_id)
            if handle is None:
                return (
                    self._availability_for_open_failure(probe),
                    {},
                )
            reads = {
                spec.name: self._read_parameter(handle, spec)
                for spec in spec_list
            }
            return (
                ReSpeakerTransportAvailability(backend="libusb", available=True),
                reads,
            )
        except OSError as exc:
            return (
                ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason=f"transport_error:{exc}",
                ),
                {},
            )
        finally:
            if self._bindings is not None:
                self._bindings.close(handle)
                self._bindings.exit(context)

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

    def _read_parameter(self, handle: object, spec: ReSpeakerParameterSpec) -> ReSpeakerParameterRead:
        attempts = 0
        while attempts < self.max_retry_attempts:
            attempts += 1
            buffer = (c_ubyte * spec.read_length)()
            transfer_size = self._bindings.control_transfer(
                handle,
                request_type=_CONTROL_READ_REQUEST_TYPE,
                request=_CONTROL_REQUEST,
                value=spec.request_value,
                index=spec.resid,
                buffer=buffer,
                timeout_ms=self.read_timeout_ms,
            )
            captured_at = self._time()
            if transfer_size < 0:
                return ReSpeakerParameterRead(
                    spec=spec,
                    captured_at=captured_at,
                    ok=False,
                    attempt_count=attempts,
                    error=self._bindings.error_name(transfer_size),
                )
            response = bytes(buffer[:transfer_size])
            if not response:
                return ReSpeakerParameterRead(
                    spec=spec,
                    captured_at=captured_at,
                    ok=False,
                    attempt_count=attempts,
                    error="empty_response",
                )
            status_code = int(response[0])
            if status_code == _CONTROL_SUCCESS:
                decoded = _decode_parameter_bytes(spec, response[1:])
                if decoded is None:
                    return ReSpeakerParameterRead(
                        spec=spec,
                        captured_at=captured_at,
                        ok=False,
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
                return ReSpeakerParameterRead(
                    spec=spec,
                    captured_at=captured_at,
                    ok=False,
                    attempt_count=attempts,
                    status_code=status_code,
                    error=f"device_status_{status_code}",
                )
            if attempts < self.max_retry_attempts and self.retry_sleep_s > 0.0:
                self._sleep(self.retry_sleep_s)
        return ReSpeakerParameterRead(
            spec=spec,
            captured_at=self._time(),
            ok=False,
            attempt_count=attempts,
            status_code=_SERVICER_COMMAND_RETRY,
            error="servicer_retry_exhausted",
        )


def _decode_parameter_bytes(
    spec: ReSpeakerParameterSpec,
    payload: bytes,
) -> tuple[int | float, ...] | str | None:
    try:
        if spec.value_type is ReSpeakerParameterType.CHAR:
            return payload.rstrip(b"\x00").decode("utf-8", errors="replace")
        if spec.value_type is ReSpeakerParameterType.UINT8:
            if len(payload) < spec.value_count:
                return None
            return tuple(int(value) for value in payload[: spec.value_count])
        if spec.value_type is ReSpeakerParameterType.UINT16:
            required = spec.value_count * 2
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("H" * spec.value_count), payload[:required])
        if spec.value_type is ReSpeakerParameterType.UINT32:
            required = spec.value_count * 4
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("I" * spec.value_count), payload[:required])
        if spec.value_type is ReSpeakerParameterType.INT32:
            required = spec.value_count * 4
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("i" * spec.value_count), payload[:required])
        if spec.value_type in (ReSpeakerParameterType.FLOAT, ReSpeakerParameterType.RADIANS):
            required = spec.value_count * 4
            if len(payload) < required:
                return None
            return struct.unpack("<" + ("f" * spec.value_count), payload[:required])
    except (struct.error, UnicodeDecodeError, ValueError):
        return None
    return None
