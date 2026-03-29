# CHANGELOG: 2026-03-28
# BUG-1: Enforce the per-read deadline on every libusb call; the old code could overshoot the "5s hard cap" by one full transfer timeout.
# BUG-2: Reject lossy integer writes (for example 1.9 -> 1) and validate write cmd/resid fields so malformed specs cannot silently hit the wrong USB command.
# BUG-3: Validate fixed-width CHAR read specs and zero-pad fixed-width CHAR writes so string parameters cannot be silently truncated or short-written.
# BUG-4: Distinguish device_not_found, multiple_matching_devices, and permission_denied instead of reporting most open failures as permission problems.
# SEC-1: Fail closed when multiple identical VID/PID devices are present unless the caller selects one explicitly (serial / bus / address / port path).
# IMP-1: Replace libusb_open_device_with_vid_pid()-only discovery with explicit enumeration and stable device selectors, matching libusb guidance for "real applications".
# IMP-2: Reuse a bounded session/handle across calls, add explicit close()/context-manager support, and auto-drop stale sessions after transport-level disconnect errors.
# IMP-3: Prefer libusb_init_context() when available, while keeping a libusb_init() fallback for older/shared-library surfaces.
"""Dependency-light libusb transport for XVF3800 host-control I/O."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_ssize_t,
    c_uint,
    c_uint16,
    c_ubyte,
    c_void_p,
)
import ctypes
import ctypes.util
import math
import operator
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
_DEFAULT_MAX_SINGLE_READ_DURATION_S = 5.0
_DEFAULT_MAX_SESSION_IDLE_S = 30.0

_MAX_REASON_LENGTH = 96
_MAX_SAFE_UINT16 = 0xFFFF
_MAX_SAFE_READ_TIMEOUT_MS = int(_DEFAULT_MAX_SINGLE_READ_DURATION_S * 1000)
_MAX_SAFE_RETRY_SLEEP_S = 1.0
_MAX_SAFE_RETRY_ATTEMPTS = 1000
_MAX_SAFE_PORT_DEPTH = 7

_XVF3800_VENDOR_ID = 0x2886
_XVF3800_PRODUCT_ID = 0x001A


class _LibusbDeviceDescriptor(Structure):
    _fields_ = [
        ("bLength", c_ubyte),
        ("bDescriptorType", c_ubyte),
        ("bcdUSB", c_uint16),
        ("bDeviceClass", c_ubyte),
        ("bDeviceSubClass", c_ubyte),
        ("bDeviceProtocol", c_ubyte),
        ("bMaxPacketSize0", c_ubyte),
        ("idVendor", c_uint16),
        ("idProduct", c_uint16),
        ("bcdDevice", c_uint16),
        ("iManufacturer", c_ubyte),
        ("iProduct", c_ubyte),
        ("iSerialNumber", c_ubyte),
        ("bNumConfigurations", c_ubyte),
    ]


def _sanitize_reason(value: object, *, fallback: str) -> str:
    """Return a short single-line reason string safe to propagate upstream."""

    text = " ".join(str(value).split())
    if not text:
        return fallback
    return text[:_MAX_REASON_LENGTH]


def _parse_uint16(value: object, *, default: int) -> tuple[int, bool]:
    """Parse a uint16 configuration value."""

    if isinstance(value, bool):
        return default, False
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default, False
    if 0 <= parsed <= _MAX_SAFE_UINT16:
        return parsed, True
    return default, False


def _parse_uint8(value: object, *, default: int) -> tuple[int, bool]:
    """Parse a uint8 configuration value."""

    if isinstance(value, bool):
        return default, False
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default, False
    if 0 <= parsed <= 0xFF:
        return parsed, True
    return default, False


def _parse_bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Parse an integer with a safe fallback and clamp."""

    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Parse a float with a safe fallback and clamp."""

    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_optional_selector_str(value: object) -> str | None:
    """Normalize an optional selector string."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_port_numbers(value: object) -> tuple[int, ...] | None:
    """Parse a USB port-path selector such as '1-4-3' or [1, 4, 3]."""

    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip().replace(".", "-").replace("/", "-").replace(",", "-").replace(":", "-")
        if not normalized:
            return None
        parts = [part for part in normalized.split("-") if part]
    else:
        try:
            parts = [str(operator.index(part)) for part in tuple(value)]
        except Exception:
            return None

    if not parts or len(parts) > _MAX_SAFE_PORT_DEPTH:
        return None

    parsed: list[int] = []
    for part in parts:
        try:
            number = int(part)
        except (TypeError, ValueError):
            return None
        if number < 0 or number > 255:
            return None
        parsed.append(number)
    return tuple(parsed)


def _port_numbers_to_reason_fragment(port_numbers: Sequence[int] | None) -> str | None:
    """Return a stable short fragment for a port-path selector."""

    if not port_numbers:
        return None
    return "-".join(str(int(number)) for number in port_numbers)


def _expected_payload_length(spec: ReSpeakerParameterSpec) -> int | None:
    """Return the fixed payload length for the spec, or None for variable-width CHAR values."""

    try:
        value_count = int(spec.value_count)
    except (AttributeError, TypeError, ValueError):
        return -1
    if value_count < 0:
        return -1
    if spec.value_type is ReSpeakerParameterType.CHAR:
        return None if value_count == 0 else value_count
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


def _validate_write_spec(spec: ReSpeakerParameterSpec, *, payload_length: int) -> str | None:
    """Validate a write spec before it is passed to ctypes/libusb."""

    try:
        cmdid = int(spec.cmdid)
        resid = int(spec.resid)
        value_count = int(spec.value_count)
    except (AttributeError, TypeError, ValueError) as exc:
        return f"invalid_spec:{exc.__class__.__name__}"

    if cmdid < 0 or cmdid > _MAX_SAFE_UINT16:
        return "invalid_spec:cmdid_out_of_range"
    if resid < 0 or resid > _MAX_SAFE_UINT16:
        return "invalid_spec:resid_out_of_range"
    if value_count < 0:
        return "invalid_spec:value_count_out_of_range"
    if payload_length < 0 or payload_length > _MAX_SAFE_UINT16:
        return "invalid_spec:payload_length_out_of_range"

    expected = _expected_payload_length(spec)
    if expected == -1:
        return "invalid_spec:unsupported_value_type"
    if expected is not None and payload_length != expected:
        return "invalid_spec:payload_length_mismatch"
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
        if value_count == 0:
            if len(encoded) > _MAX_SAFE_UINT16:
                raise ValueError("char_payload_exceeds_usb_control_limit")
            return encoded
        if len(encoded) > value_count:
            raise ValueError("char_payload_exceeds_spec_length")
        return encoded.ljust(value_count, b"\x00")

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("write_values_must_be_a_numeric_sequence")

    normalized = tuple(values)
    if len(normalized) != value_count:
        raise ValueError("write_value_count_mismatch")

    if spec.value_type is ReSpeakerParameterType.UINT8:
        payload = bytearray()
        for index, value in enumerate(normalized):
            integer = _coerce_bounded_write_int(value, minimum=0, maximum=0xFF, index=index)
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
    try:
        integer = operator.index(value)
    except Exception as exc:
        raise TypeError(f"write_value[{index}] must be an int") from exc
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

    _FALLBACK_LIBRARY_NAMES = (
        "libusb-1.0.so.0",
        "libusb-1.0.so",
        "usb-1.0",
    )

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._configure_signatures()

    @classmethod
    def from_system(cls) -> "_LibusbBindings | None":
        candidates: list[str] = []
        discovered = ctypes.util.find_library("usb-1.0")
        if discovered:
            candidates.append(discovered)
        candidates.extend(name for name in cls._FALLBACK_LIBRARY_NAMES if name not in candidates)

        last_error: Exception | None = None
        for library_name in candidates:
            try:
                return cls(ctypes.CDLL(library_name))
            except (AttributeError, OSError, TypeError) as exc:
                last_error = exc
                continue
        if last_error is not None:
            return None
        return None

    def _configure_signatures(self) -> None:
        if hasattr(self._library, "libusb_init_context"):
            self._library.libusb_init_context.argtypes = [POINTER(c_void_p), c_void_p, c_int]
            self._library.libusb_init_context.restype = c_int

        self._library.libusb_init.argtypes = [POINTER(c_void_p)]
        self._library.libusb_init.restype = c_int

        self._library.libusb_exit.argtypes = [c_void_p]
        self._library.libusb_exit.restype = None

        self._library.libusb_open_device_with_vid_pid.argtypes = [c_void_p, c_uint16, c_uint16]
        self._library.libusb_open_device_with_vid_pid.restype = c_void_p

        self._library.libusb_open.argtypes = [c_void_p, POINTER(c_void_p)]
        self._library.libusb_open.restype = c_int

        self._library.libusb_close.argtypes = [c_void_p]
        self._library.libusb_close.restype = None

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

        if hasattr(self._library, "libusb_strerror"):
            self._library.libusb_strerror.argtypes = [c_int]
            self._library.libusb_strerror.restype = c_char_p

        if hasattr(self._library, "libusb_get_device_list"):
            self._library.libusb_get_device_list.argtypes = [c_void_p, POINTER(POINTER(c_void_p))]
            self._library.libusb_get_device_list.restype = c_ssize_t

        if hasattr(self._library, "libusb_free_device_list"):
            self._library.libusb_free_device_list.argtypes = [POINTER(c_void_p), c_int]
            self._library.libusb_free_device_list.restype = None

        if hasattr(self._library, "libusb_get_device_descriptor"):
            self._library.libusb_get_device_descriptor.argtypes = [c_void_p, POINTER(_LibusbDeviceDescriptor)]
            self._library.libusb_get_device_descriptor.restype = c_int

        if hasattr(self._library, "libusb_get_bus_number"):
            self._library.libusb_get_bus_number.argtypes = [c_void_p]
            self._library.libusb_get_bus_number.restype = c_ubyte

        if hasattr(self._library, "libusb_get_device_address"):
            self._library.libusb_get_device_address.argtypes = [c_void_p]
            self._library.libusb_get_device_address.restype = c_ubyte

        if hasattr(self._library, "libusb_get_port_numbers"):
            self._library.libusb_get_port_numbers.argtypes = [c_void_p, POINTER(c_ubyte), c_int]
            self._library.libusb_get_port_numbers.restype = c_int

        if hasattr(self._library, "libusb_get_string_descriptor_ascii"):
            self._library.libusb_get_string_descriptor_ascii.argtypes = [
                c_void_p,
                c_ubyte,
                POINTER(c_ubyte),
                c_int,
            ]
            self._library.libusb_get_string_descriptor_ascii.restype = c_int

        if hasattr(self._library, "libusb_claim_interface"):
            self._library.libusb_claim_interface.argtypes = [c_void_p, c_int]
            self._library.libusb_claim_interface.restype = c_int

        if hasattr(self._library, "libusb_release_interface"):
            self._library.libusb_release_interface.argtypes = [c_void_p, c_int]
            self._library.libusb_release_interface.restype = c_int

        if hasattr(self._library, "libusb_set_auto_detach_kernel_driver"):
            self._library.libusb_set_auto_detach_kernel_driver.argtypes = [c_void_p, c_int]
            self._library.libusb_set_auto_detach_kernel_driver.restype = c_int

        if hasattr(self._library, "libusb_reset_device"):
            self._library.libusb_reset_device.argtypes = [c_void_p]
            self._library.libusb_reset_device.restype = c_int

    def init_context(self) -> c_void_p:
        context = c_void_p()
        if hasattr(self._library, "libusb_init_context"):
            result = self._library.libusb_init_context(byref(context), None, 0)
        else:
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
        if buffer_length < 0 or buffer_length > _MAX_SAFE_UINT16:
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

    def error_detail(self, code: int) -> str:
        if hasattr(self._library, "libusb_strerror"):
            try:
                raw = self._library.libusb_strerror(int(code))
                if raw:
                    return raw.decode("utf-8", errors="replace")
            except Exception:
                pass
        return self.error_name(code)

    def supports_enumeration(self) -> bool:
        required = (
            "libusb_get_device_list",
            "libusb_free_device_list",
            "libusb_get_device_descriptor",
            "libusb_open",
            "libusb_get_bus_number",
            "libusb_get_device_address",
            "libusb_get_port_numbers",
        )
        return all(hasattr(self._library, symbol) for symbol in required)

    def claim_interface(self, handle: c_void_p, interface_number: int, *, auto_detach: bool) -> int:
        if auto_detach and hasattr(self._library, "libusb_set_auto_detach_kernel_driver"):
            self._library.libusb_set_auto_detach_kernel_driver(handle, 1)
        if not hasattr(self._library, "libusb_claim_interface"):
            return 0
        return int(self._library.libusb_claim_interface(handle, int(interface_number)))

    def release_interface(self, handle: c_void_p, interface_number: int) -> int:
        if not hasattr(self._library, "libusb_release_interface"):
            return 0
        return int(self._library.libusb_release_interface(handle, int(interface_number)))

    def reset_device(self, handle: c_void_p) -> int:
        if not hasattr(self._library, "libusb_reset_device"):
            return 0
        return int(self._library.libusb_reset_device(handle))

    def _get_string_descriptor_ascii(self, handle: c_void_p, index: int) -> str | None:
        if not index or not hasattr(self._library, "libusb_get_string_descriptor_ascii"):
            return None
        buffer = (c_ubyte * 256)()
        result = int(self._library.libusb_get_string_descriptor_ascii(handle, int(index), buffer, len(buffer)))
        if result < 0:
            return None
        return bytes(buffer[:result]).decode("utf-8", errors="replace")

    def find_open_device(
        self,
        context: c_void_p,
        *,
        vendor_id: int,
        product_id: int,
        serial_number: str | None = None,
        bus_number: int | None = None,
        device_address: int | None = None,
        port_numbers: Sequence[int] | None = None,
        require_unique: bool = True,
    ) -> tuple[c_void_p | None, dict[str, object] | None, str | None]:
        """
        Enumerate matching devices and open a stable target.

        Returns (handle, identity, reason). When handle is None, reason contains a stable failure code.
        """

        if not self.supports_enumeration():
            handle = self.open_device(context, vendor_id, product_id)
            if handle is None:
                return None, None, "device_not_found_or_open_failed"
            return handle, None, None

        device_list = POINTER(c_void_p)()
        count = int(self._library.libusb_get_device_list(context, byref(device_list)))
        if count < 0:
            raise OSError(f"libusb_get_device_list failed with code {count}")

        matches: list[tuple[c_void_p, dict[str, object]]] = []
        access_denied = False
        open_fail_reason: str | None = None
        descriptor_candidate_count = 0

        try:
            for index in range(count):
                device = device_list[index]
                if not device:
                    break

                descriptor = _LibusbDeviceDescriptor()
                descriptor_result = int(self._library.libusb_get_device_descriptor(device, byref(descriptor)))
                if descriptor_result != 0:
                    open_fail_reason = f"descriptor_failed:{self.error_name(descriptor_result)}"
                    continue

                if int(descriptor.idVendor) != vendor_id or int(descriptor.idProduct) != product_id:
                    continue

                current_bus = int(self._library.libusb_get_bus_number(device))
                current_address = int(self._library.libusb_get_device_address(device))

                port_buffer = (c_ubyte * _MAX_SAFE_PORT_DEPTH)()
                port_count = int(self._library.libusb_get_port_numbers(device, port_buffer, len(port_buffer)))
                current_port_numbers = tuple(int(port_buffer[i]) for i in range(port_count)) if port_count > 0 else tuple()

                if bus_number is not None and current_bus != bus_number:
                    continue
                if device_address is not None and current_address != device_address:
                    continue
                if port_numbers is not None and tuple(port_numbers) != current_port_numbers:
                    continue

                descriptor_candidate_count += 1

                handle = c_void_p()
                open_result = int(self._library.libusb_open(device, byref(handle)))
                if open_result != 0:
                    if "access" in self.error_name(open_result).lower():
                        access_denied = True
                    else:
                        open_fail_reason = f"open_failed:{self.error_name(open_result)}"
                    continue

                keep_handle = False
                try:
                    current_serial = self._get_string_descriptor_ascii(handle, int(descriptor.iSerialNumber))
                    if serial_number is not None and current_serial != serial_number:
                        continue

                    identity = {
                        "bus_number": current_bus,
                        "device_address": current_address,
                        "port_numbers": current_port_numbers,
                        "serial_number": current_serial,
                        "manufacturer": self._get_string_descriptor_ascii(handle, int(descriptor.iManufacturer)),
                        "product": self._get_string_descriptor_ascii(handle, int(descriptor.iProduct)),
                    }
                    matches.append((handle, identity))
                    keep_handle = True
                finally:
                    if not keep_handle:
                        self.close(handle)

            if require_unique and serial_number is None and descriptor_candidate_count > 1:
                for handle, _identity in matches:
                    self.close(handle)
                return None, None, "multiple_matching_devices"

            if not matches:
                if access_denied:
                    return None, None, "access_denied"
                if open_fail_reason is not None:
                    return None, None, open_fail_reason
                return None, None, "device_not_found"

            matches.sort(
                key=lambda item: (
                    int(item[1].get("bus_number") or 0),
                    tuple(int(part) for part in (item[1].get("port_numbers") or ())),
                    int(item[1].get("device_address") or 0),
                    str(item[1].get("serial_number") or ""),
                )
            )

            if require_unique and len(matches) > 1:
                for handle, _identity in matches:
                    self.close(handle)
                return None, None, "multiple_matching_devices"

            selected_handle, selected_identity = matches[0]
            for extra_handle, _identity in matches[1:]:
                self.close(extra_handle)
            return selected_handle, selected_identity, None
        finally:
            self._library.libusb_free_device_list(device_list, 1)


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
        max_session_idle_s: float = _DEFAULT_MAX_SESSION_IDLE_S,
        serial_number: str | None = None,
        bus_number: int | None = None,
        device_address: int | None = None,
        port_numbers: Sequence[int] | str | None = None,
        allow_first_match_if_multiple: bool = False,  # BREAKING: multi-device ambiguity now fails closed unless this is set or a selector is supplied.
        control_interface_number: int | None = None,
        auto_detach_kernel_driver: bool = False,
        bindings: object | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.vendor_id, vendor_valid = _parse_uint16(vendor_id, default=_XVF3800_VENDOR_ID)
        self.product_id, product_valid = _parse_uint16(product_id, default=_XVF3800_PRODUCT_ID)

        parsed_bus_number = None
        parsed_device_address = None
        bus_valid = True
        address_valid = True
        if bus_number is not None:
            parsed_bus_number, bus_valid = _parse_uint8(bus_number, default=0)
            if not bus_valid:
                parsed_bus_number = None
        if device_address is not None:
            parsed_device_address, address_valid = _parse_uint8(device_address, default=0)
            if not address_valid:
                parsed_device_address = None

        parsed_port_numbers = _parse_port_numbers(port_numbers)
        port_valid = (port_numbers is None) or (parsed_port_numbers is not None)

        self.serial_number = _parse_optional_selector_str(serial_number)
        self.bus_number = parsed_bus_number
        self.device_address = parsed_device_address
        self.port_numbers = parsed_port_numbers
        self.allow_first_match_if_multiple = bool(allow_first_match_if_multiple)

        interface_valid = True
        parsed_interface_number = None
        if control_interface_number is not None:
            parsed_interface_number, interface_valid = _parse_uint8(control_interface_number, default=0)
            if not interface_valid:
                parsed_interface_number = None
        self.control_interface_number = parsed_interface_number
        self.auto_detach_kernel_driver = bool(auto_detach_kernel_driver)

        self.max_single_read_duration_s = _parse_bounded_float(
            max_single_read_duration_s,
            default=_DEFAULT_MAX_SINGLE_READ_DURATION_S,
            minimum=0.25,
            maximum=_DEFAULT_MAX_SINGLE_READ_DURATION_S,
        )
        self.read_timeout_ms = min(
            _parse_bounded_int(
                read_timeout_ms,
                default=_DEFAULT_READ_TIMEOUT_MS,
                minimum=1,
                maximum=_MAX_SAFE_READ_TIMEOUT_MS,
            ),
            max(1, int(self.max_single_read_duration_s * 1000)),
        )
        self.max_retry_attempts = _parse_bounded_int(
            max_retry_attempts,
            default=_DEFAULT_MAX_RETRY_ATTEMPTS,
            minimum=1,
            maximum=_MAX_SAFE_RETRY_ATTEMPTS,
        )
        self.retry_sleep_s = _parse_bounded_float(
            retry_sleep_s,
            default=_DEFAULT_RETRY_SLEEP_S,
            minimum=0.0,
            maximum=_MAX_SAFE_RETRY_SLEEP_S,
        )
        self.max_session_idle_s = _parse_bounded_float(
            max_session_idle_s,
            default=_DEFAULT_MAX_SESSION_IDLE_S,
            minimum=0.0,
            maximum=3600.0,
        )

        self._config_error: str | None = None
        if not vendor_valid:
            self._config_error = "vendor_id_out_of_range"
        elif not product_valid:
            self._config_error = "product_id_out_of_range"
        elif not bus_valid:
            self._config_error = "bus_number_out_of_range"
        elif not address_valid:
            self._config_error = "device_address_out_of_range"
        elif not port_valid:
            self._config_error = "port_numbers_out_of_range"
        elif not interface_valid:
            self._config_error = "control_interface_number_out_of_range"

        self._bindings = bindings if bindings is not None else _LibusbBindings.from_system()
        self._sleep = sleep_fn or time.sleep
        self._time = time_fn or time.time
        self._monotonic = time.monotonic

        self._io_lock = RLock()
        self._session_context: c_void_p | None = None
        self._session_handle: c_void_p | None = None
        self._session_claimed_interface: int | None = None
        self._session_identity: dict[str, object] | None = None
        self._last_used_monotonic: float | None = None

    def __enter__(self) -> "ReSpeakerLibusbTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def device_identity(self) -> dict[str, object] | None:
        """Return the most recently opened device identity, if known."""

        with self._io_lock:
            if self._session_identity is None:
                return None
            return dict(self._session_identity)

    def close(self) -> None:
        """Release any cached libusb session/handle."""

        with self._io_lock:
            self._drop_session_locked()

    def reset_device(self) -> ReSpeakerTransportAvailability:
        """Issue a libusb port reset against the currently-open device, if supported."""

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

        with self._io_lock:
            availability = self._ensure_session_locked(None)
            if not availability.available:
                return availability
            if self._session_handle is None:
                return ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason="device_not_open",
                )
            try:
                if not hasattr(self._bindings, "reset_device"):
                    return ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason="reset_unsupported",
                    )
                result = int(self._bindings.reset_device(self._session_handle))
                if result != 0:
                    self._drop_session_locked()
                    return ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason=self._binding_error_name(result),
                    )
                self._drop_session_locked()
                return ReSpeakerTransportAvailability(backend="libusb", available=True)
            except Exception as exc:
                self._drop_session_locked()
                return ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason=self._transport_error_reason(exc),
                )

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

        with self._io_lock:
            reads: dict[str, ReSpeakerParameterRead] = {}
            try:
                availability = self._ensure_session_locked(probe)
                if not availability.available:
                    return availability, {}

                if self._session_handle is None:
                    return (
                        ReSpeakerTransportAvailability(
                            backend="libusb",
                            available=False,
                            reason="device_not_open",
                        ),
                        {},
                    )

                for index, spec in enumerate(specs):
                    spec_name = self._unique_spec_name(reads, self._spec_name(spec, index))
                    if self._session_handle is None:
                        reads[spec_name] = self._failed_read(
                            spec,
                            attempt_count=0,
                            error="device_disconnected",
                        )
                        continue
                    try:
                        reads[spec_name] = self._read_parameter(self._session_handle, spec)
                    except Exception as exc:
                        reads[spec_name] = self._failed_read(
                            spec,
                            attempt_count=0,
                            error=f"read_exception:{exc.__class__.__name__}",
                        )
                self._mark_session_used_locked()
                return ReSpeakerTransportAvailability(backend="libusb", available=True), reads
            except Exception as exc:
                self._drop_session_locked()
                return (
                    ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason=self._transport_error_reason(exc),
                    ),
                    reads,
                )

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
            validation_error = _validate_write_spec(spec, payload_length=len(payload))
            if validation_error is not None:
                raise ValueError(validation_error)
        except Exception as exc:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason=f"invalid_write:{_sanitize_reason(exc, fallback=exc.__class__.__name__)}",
            )

        with self._io_lock:
            availability = self._ensure_session_locked(probe)
            if not availability.available:
                return availability

            if self._session_handle is None:
                return ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason="device_not_open",
                )

            try:
                buffer = (c_ubyte * len(payload)).from_buffer_copy(payload)
                transfer_size = self._bindings.control_transfer(
                    self._session_handle,
                    request_type=_CONTROL_WRITE_REQUEST_TYPE,
                    request=_CONTROL_REQUEST,
                    value=int(spec.cmdid),
                    index=int(spec.resid),
                    buffer=buffer,
                    timeout_ms=self.read_timeout_ms,
                )
                if transfer_size < 0:
                    if self._should_drop_session_from_error(transfer_size):
                        self._drop_session_locked()
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
                self._mark_session_used_locked()
                return ReSpeakerTransportAvailability(backend="libusb", available=True)
            except Exception as exc:
                self._drop_session_locked()
                return ReSpeakerTransportAvailability(
                    backend="libusb",
                    available=False,
                    reason=self._transport_error_reason(exc),
                )

    def _ensure_session_locked(
        self,
        probe: ReSpeakerProbeResult | None,
    ) -> ReSpeakerTransportAvailability:
        if self._session_handle is not None and self._session_context is not None and not self._session_expired_locked():
            self._mark_session_used_locked()
            return ReSpeakerTransportAvailability(backend="libusb", available=True)

        self._drop_session_locked()

        context = None
        handle = None
        claimed_interface = None
        identity = None
        try:
            context = self._bindings.init_context()

            if hasattr(self._bindings, "find_open_device"):
                try:
                    handle, identity, open_reason = self._bindings.find_open_device(
                        context,
                        vendor_id=self.vendor_id,
                        product_id=self.product_id,
                        serial_number=self.serial_number,
                        bus_number=self.bus_number,
                        device_address=self.device_address,
                        port_numbers=self.port_numbers,
                        require_unique=(not self.allow_first_match_if_multiple),
                    )
                except TypeError:
                    handle = self._bindings.open_device(context, self.vendor_id, self.product_id)
                    identity = None
                    open_reason = None if handle is not None else "device_not_found_or_open_failed"
                if handle is None:
                    self._safe_release(handle=None, context=context, claimed_interface=None)
                    return self._availability_for_open_failure(probe, open_reason)
            else:
                handle = self._bindings.open_device(context, self.vendor_id, self.product_id)
                if handle is None:
                    self._safe_release(handle=None, context=context, claimed_interface=None)
                    return self._availability_for_open_failure(probe, None)

            if self.control_interface_number is not None and hasattr(self._bindings, "claim_interface"):
                claim_result = int(
                    self._bindings.claim_interface(
                        handle,
                        self.control_interface_number,
                        auto_detach=self.auto_detach_kernel_driver,
                    )
                )
                if claim_result != 0:
                    self._safe_release(
                        handle=handle,
                        context=context,
                        claimed_interface=None,
                    )
                    return ReSpeakerTransportAvailability(
                        backend="libusb",
                        available=False,
                        reason=f"claim_interface_failed:{self._binding_error_name(claim_result)}",
                    )
                claimed_interface = int(self.control_interface_number)

            self._session_context = context
            self._session_handle = handle
            self._session_claimed_interface = claimed_interface
            self._session_identity = identity
            self._mark_session_used_locked()
            return ReSpeakerTransportAvailability(backend="libusb", available=True)
        except Exception as exc:
            self._safe_release(handle=handle, context=context, claimed_interface=claimed_interface)
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason=self._transport_error_reason(exc),
            )

    def _session_expired_locked(self) -> bool:
        if self.max_session_idle_s <= 0.0:
            return False
        if self._last_used_monotonic is None:
            return False
        return (self._monotonic() - self._last_used_monotonic) > self.max_session_idle_s

    def _mark_session_used_locked(self) -> None:
        self._last_used_monotonic = self._monotonic()

    def _drop_session_locked(self) -> None:
        self._safe_release(
            handle=self._session_handle,
            context=self._session_context,
            claimed_interface=self._session_claimed_interface,
        )
        self._session_context = None
        self._session_handle = None
        self._session_claimed_interface = None
        self._session_identity = None
        self._last_used_monotonic = None

    def _availability_for_open_failure(
        self,
        probe: ReSpeakerProbeResult | None,
        open_reason: str | None,
    ) -> ReSpeakerTransportAvailability:
        if probe is not None and not probe.usb_visible:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason="device_not_visible",
            )

        if open_reason == "device_not_found":
            selector_parts = []
            if self.serial_number is not None:
                selector_parts.append(f"serial={self.serial_number}")
            if self.bus_number is not None:
                selector_parts.append(f"bus={self.bus_number}")
            if self.device_address is not None:
                selector_parts.append(f"address={self.device_address}")
            port_fragment = _port_numbers_to_reason_fragment(self.port_numbers)
            if port_fragment is not None:
                selector_parts.append(f"port={port_fragment}")
            suffix = ",".join(selector_parts)
            reason = "device_not_found" if not suffix else f"device_not_found:{suffix}"
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason=reason,
            )

        if open_reason == "multiple_matching_devices":
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason="multiple_matching_devices",
            )

        if open_reason == "access_denied":
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason="permission_denied_or_transport_blocked",
                requires_elevated_permissions=True,
            )

        if open_reason:
            return ReSpeakerTransportAvailability(
                backend="libusb",
                available=False,
                reason=open_reason,
            )

        return ReSpeakerTransportAvailability(
            backend="libusb",
            available=False,
            reason="device_not_found_or_open_failed",
        )

    def _transport_error_reason(self, exc: Exception) -> str:
        error_type = exc.__class__.__name__
        if isinstance(exc, OSError):
            detail = _sanitize_reason(exc, fallback=error_type)
            if detail.startswith("libusb_") or "failed with code" in detail:
                return f"transport_error:{error_type}:{detail}"
        return f"transport_error:{error_type}"

    def _binding_error_name(self, code: int) -> str:
        try:
            if self._bindings is not None:
                return _sanitize_reason(
                    self._bindings.error_name(code),
                    fallback=f"libusb_error_{code}",
                )
        except Exception:
            pass
        return f"libusb_error_{code}"

    def _should_drop_session_from_error(self, code: int) -> bool:
        name = self._binding_error_name(code).lower()
        return any(
            fragment in name
            for fragment in (
                "no_device",
                "pipe",
                "overflow",
                "io",
                "not_found",
            )
        )

    def _spec_name(self, spec: object, index: int) -> str:
        name = getattr(spec, "name", None)
        if isinstance(name, str) and name:
            return name
        return f"unnamed_spec_{index}"

    def _unique_spec_name(self, reads: dict[str, ReSpeakerParameterRead], base_name: str) -> str:
        if base_name not in reads:
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name}__duplicate_{suffix}"
            if candidate not in reads:
                return candidate
            suffix += 1

    def _safe_release(
        self,
        *,
        handle: c_void_p | None,
        context: c_void_p | None,
        claimed_interface: int | None,
    ) -> None:
        if self._bindings is None:
            return
        try:
            if handle is not None and claimed_interface is not None and hasattr(self._bindings, "release_interface"):
                self._bindings.release_interface(handle, claimed_interface)
        except Exception:
            pass
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
            "error": _sanitize_reason(error, fallback="transport_error"),
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
            )

        attempts = 0
        deadline = self._monotonic() + self.max_single_read_duration_s
        deadline_exhausted = False

        while attempts < self.max_retry_attempts:
            remaining_s = deadline - self._monotonic()
            if remaining_s <= 0.0:
                deadline_exhausted = True
                break

            attempts += 1
            timeout_ms = min(self.read_timeout_ms, max(1, int(remaining_s * 1000)))
            buffer = (c_ubyte * int(spec.read_length))()
            transfer_size = self._bindings.control_transfer(
                handle,
                request_type=_CONTROL_READ_REQUEST_TYPE,
                request=_CONTROL_REQUEST,
                value=int(spec.request_value),
                index=int(spec.resid),
                buffer=buffer,
                timeout_ms=timeout_ms,
            )
            captured_at = self._time()
            if transfer_size < 0:
                if self._should_drop_session_from_error(transfer_size):
                    self._drop_session_locked()
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
                )
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
                self._sleep(min(self.retry_sleep_s, remaining_sleep))

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
            return char_payload.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
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