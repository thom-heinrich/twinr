# CHANGELOG: 2026-03-28
# BUG-1: Cooldown no longer depends on wall-clock/observed_at timestamps; a monotonic clock gates capture attempts, avoiding NTP/RTC jumps and invalid observed_at values from causing capture storms or permanent cooldown lockout.
# BUG-2: JPEG encoding and filesystem writes are no longer performed while holding the store lock, removing cross-thread backpressure in the live vision lane and preventing repeated write-failure hammering.
# BUG-3: Capture filenames/retention no longer depend on lexicographically sorted observed_at timestamps; retention now prunes by actual artifact freshness and filenames fall back safely when observed_at is invalid or non-epoch.
# BUG-4: Metadata serialization is now cycle-safe and bounded; recursive/self-referential or huge debug payloads no longer explode CPU/disk usage or fail with recursion errors.
# SEC-1: Artifact writes are now atomic (secure temp file + os.replace) and private by default (0700 directory / 0600 files), preventing partial-file reads and reducing practical local disclosure risk for senior-care imagery.
# SEC-2: The capture path no longer directly writes to attacker-replaceable destination files; secure temp creation plus atomic replacement mitigates symlink/file-clobber attacks in writable capture directories.
# IMP-1: The encoder path now prefers libjpeg-turbo/PyTurboJPEG when available on ARM and falls back to Pillow, matching 2026 edge-vision practice for faster bounded JPEG persistence on Raspberry Pi class hardware.
# BUG-5: Candidate frames are now snapshotted into private contiguous memory before encoding, so saved artifacts cannot drift when upstream camera buffers are reused.
# IMP-2: The helper now emits richer bounded metadata (encoder, bytes, timings, resize info, observed_at_utc when valid) while remaining drop-in compatible.
# BREAKING: Capture artifacts are now created with private permissions by default (directory 0700, files 0600). Override via config fields if operators intentionally require broader local access.

"""Persist bounded gesture-candidate frames for manual optical QA.

The live gesture lane can produce structured evidence without immediately
publishing an emoji. When that happens, operators need one visual artifact to
answer the concrete question: "What exactly was in frame when Twinr thought a
gesture might be present?" This helper keeps that capture path bounded with a
cooldown, finite retention, and compact JSON sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
import io
import json
import math
import os
import tempfile
import time

import numpy as np
from PIL import Image

from .config import AICameraAdapterConfig


_NEGATIVE_LABELS = frozenset({"", "none", "unknown", "unavailable"})

_DEFAULT_JPEG_QUALITY = 85
_DEFAULT_JPEG_SUBSAMPLING = "4:2:0"
_DEFAULT_FILE_MODE = 0o600
_DEFAULT_DIR_MODE = 0o700
_DEFAULT_SAFE_JSON_MAX_DEPTH = 6
_DEFAULT_SAFE_JSON_MAX_ITEMS = 64
_DEFAULT_SAFE_JSON_MAX_STRING_CHARS = 512
_DEFAULT_SAFE_JSON_MAX_REPR_CHARS = 256
_DEFAULT_MAX_EDGE_PX = 0
_DEFAULT_DURABLE_WRITES = False
_DEFAULT_PREFER_TURBOJPEG = True

_UTC = timezone.utc
_PLAUSIBLE_EPOCH_MIN = 946684800.0   # 2000-01-01T00:00:00Z
_PLAUSIBLE_EPOCH_MAX = 4102444800.0  # 2100-01-01T00:00:00Z

try:  # Optional frontier encoder path on ARM/x86.
    from turbojpeg import (
        TJPF_RGB,
        TJSAMP_411,
        TJSAMP_420,
        TJSAMP_422,
        TJSAMP_440,
        TJSAMP_444,
        TJSAMP_GRAY,
        TurboJPEG,
    )
except Exception:  # pragma: no cover - optional dependency
    TJPF_RGB = None
    TJSAMP_411 = None
    TJSAMP_420 = None
    TJSAMP_422 = None
    TJSAMP_440 = None
    TJSAMP_444 = None
    TJSAMP_GRAY = None
    TurboJPEG = None


_TURBOJPEG_LOCK = Lock()
_TURBOJPEG_INSTANCE: Any = None
_TURBOJPEG_INIT_FAILED = False
_PIL_RESAMPLING = getattr(Image, "Resampling", Image)

_TURBOJPEG_SUBSAMPLING = {
    "4:4:4": TJSAMP_444,
    "444": TJSAMP_444,
    "4:2:2": TJSAMP_422,
    "422": TJSAMP_422,
    "4:2:0": TJSAMP_420,
    "420": TJSAMP_420,
    "4:4:0": TJSAMP_440,
    "440": TJSAMP_440,
    "4:1:1": TJSAMP_411,
    "411": TJSAMP_411,
    "gray": TJSAMP_GRAY,
    "grey": TJSAMP_GRAY,
}
_PILLOW_SUBSAMPLING = {
    "4:4:4": 0,
    "444": 0,
    "4:2:2": 1,
    "422": 1,
    "4:2:0": 2,
    "420": 2,
}


@dataclass(frozen=True, slots=True)
class GestureCandidateCaptureResult:
    """Describe one bounded gesture-candidate capture attempt."""

    saved: bool
    reasons: tuple[str, ...] = ()
    image_path: str | None = None
    metadata_path: str | None = None
    skipped_reason: str | None = None
    error: str | None = None

    def debug_fields(self) -> dict[str, object]:
        """Return one JSON-safe summary for the gesture debug payload."""

        return {
            "candidate_capture_saved": self.saved,
            "candidate_capture_reasons": list(self.reasons),
            "candidate_capture_image_path": self.image_path,
            "candidate_capture_metadata_path": self.metadata_path,
            "candidate_capture_skipped_reason": self.skipped_reason,
            "candidate_capture_error": self.error,
        }


class GestureCandidateCaptureStore:
    """Save one bounded JPEG plus compact metadata when gesture evidence appears."""

    def __init__(
        self,
        *,
        capture_dir: str,
        cooldown_s: float,
        max_images: int,
        clock: Any = time.time,
        cooldown_clock: Any = time.monotonic,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        jpeg_subsampling: str = _DEFAULT_JPEG_SUBSAMPLING,
        file_mode: int = _DEFAULT_FILE_MODE,
        dir_mode: int = _DEFAULT_DIR_MODE,
        durable_writes: bool = _DEFAULT_DURABLE_WRITES,
        prefer_turbojpeg: bool = _DEFAULT_PREFER_TURBOJPEG,
        max_metadata_depth: int = _DEFAULT_SAFE_JSON_MAX_DEPTH,
        max_metadata_items: int = _DEFAULT_SAFE_JSON_MAX_ITEMS,
        max_metadata_string_chars: int = _DEFAULT_SAFE_JSON_MAX_STRING_CHARS,
        max_metadata_repr_chars: int = _DEFAULT_SAFE_JSON_MAX_REPR_CHARS,
        max_edge_px: int = _DEFAULT_MAX_EDGE_PX,
    ) -> None:
        self.capture_dir = Path(capture_dir)
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.max_images = max(1, int(max_images))
        self._clock = clock
        self._cooldown_clock = cooldown_clock
        self._jpeg_quality = max(1, min(95, int(jpeg_quality)))
        self._jpeg_subsampling = _normalized_jpeg_subsampling(jpeg_subsampling)
        self._file_mode = _normalize_mode(file_mode, default=_DEFAULT_FILE_MODE)
        self._dir_mode = _normalize_mode(dir_mode, default=_DEFAULT_DIR_MODE)
        self._durable_writes = bool(durable_writes)
        self._prefer_turbojpeg = bool(prefer_turbojpeg)
        self._max_metadata_depth = max(1, int(max_metadata_depth))
        self._max_metadata_items = max(1, int(max_metadata_items))
        self._max_metadata_string_chars = max(32, int(max_metadata_string_chars))
        self._max_metadata_repr_chars = max(32, int(max_metadata_repr_chars))
        self._max_edge_px = max(0, int(max_edge_px))
        self._lock = Lock()
        self._last_attempt_monotonic: float | None = None
        self._capture_sequence = 0
        self._instance_token = f"{os.getpid():05d}_{time.monotonic_ns() & 0xFFFF_FFFF:08x}"

    @classmethod
    def from_config(
        cls,
        config: AICameraAdapterConfig,
        *,
        clock: Any = time.time,
        cooldown_clock: Any = time.monotonic,
    ) -> "GestureCandidateCaptureStore":
        """Build one bounded candidate-capture store from camera config."""

        return cls(
            capture_dir=config.gesture_candidate_capture_dir,
            cooldown_s=config.gesture_candidate_capture_cooldown_s,
            max_images=config.gesture_candidate_capture_max_images,
            clock=clock,
            cooldown_clock=cooldown_clock,
            jpeg_quality=getattr(config, "gesture_candidate_capture_jpeg_quality", _DEFAULT_JPEG_QUALITY),
            jpeg_subsampling=getattr(
                config,
                "gesture_candidate_capture_jpeg_subsampling",
                _DEFAULT_JPEG_SUBSAMPLING,
            ),
            file_mode=getattr(config, "gesture_candidate_capture_file_mode", _DEFAULT_FILE_MODE),
            dir_mode=getattr(config, "gesture_candidate_capture_dir_mode", _DEFAULT_DIR_MODE),
            durable_writes=getattr(config, "gesture_candidate_capture_durable_writes", _DEFAULT_DURABLE_WRITES),
            prefer_turbojpeg=getattr(
                config,
                "gesture_candidate_capture_prefer_turbojpeg",
                _DEFAULT_PREFER_TURBOJPEG,
            ),
            max_metadata_depth=getattr(
                config,
                "gesture_candidate_capture_metadata_max_depth",
                _DEFAULT_SAFE_JSON_MAX_DEPTH,
            ),
            max_metadata_items=getattr(
                config,
                "gesture_candidate_capture_metadata_max_items",
                _DEFAULT_SAFE_JSON_MAX_ITEMS,
            ),
            max_metadata_string_chars=getattr(
                config,
                "gesture_candidate_capture_metadata_max_string_chars",
                _DEFAULT_SAFE_JSON_MAX_STRING_CHARS,
            ),
            max_metadata_repr_chars=getattr(
                config,
                "gesture_candidate_capture_metadata_max_repr_chars",
                _DEFAULT_SAFE_JSON_MAX_REPR_CHARS,
            ),
            max_edge_px=getattr(config, "gesture_candidate_capture_max_edge_px", _DEFAULT_MAX_EDGE_PX),
        )

    def maybe_capture(
        self,
        *,
        observed_at: float,
        frame_rgb: Any,
        debug_details: Mapping[str, object] | None,
    ) -> GestureCandidateCaptureResult:
        """Save one candidate frame when the debug evidence suggests gesture intent."""

        reasons = _candidate_reasons(debug_details)
        if not reasons:
            return GestureCandidateCaptureResult(saved=False, skipped_reason="no_candidate_signal")

        array = _normalize_rgb_array(frame_rgb)
        if array is None:
            return GestureCandidateCaptureResult(
                saved=False,
                reasons=reasons,
                skipped_reason="frame_unavailable",
            )

        fallback_epoch_now = _finite_float_or_default(self._clock(), default=time.time()) or time.time()
        event_epoch = _plausible_epoch_seconds(observed_at)
        stem_epoch = event_epoch if event_epoch is not None else fallback_epoch_now
        cooldown_now = _finite_float_or_default(self._cooldown_clock(), default=time.monotonic()) or time.monotonic()

        original_shape = tuple(int(dimension) for dimension in array.shape)
        array, resize_info = _downscale_rgb_array(array, max_edge_px=self._max_edge_px)

        with self._lock:
            last_attempt = self._last_attempt_monotonic
            if (
                last_attempt is not None
                and self.cooldown_s > 0.0
                and (cooldown_now - last_attempt) < self.cooldown_s
            ):
                return GestureCandidateCaptureResult(
                    saved=False,
                    reasons=reasons,
                    skipped_reason="cooldown_active",
                )
            self._last_attempt_monotonic = cooldown_now
            self._capture_sequence += 1
            stem = _capture_stem(
                observed_at=stem_epoch,
                sequence=self._capture_sequence,
                reasons=reasons,
                instance_token=self._instance_token,
            )

        image_path = self.capture_dir / f"{stem}.jpg"
        metadata_path = self.capture_dir / f"{stem}.json"

        encode_started_ns = time.perf_counter_ns()
        try:
            jpeg_bytes, encoder_name = _encode_jpeg_bytes(
                array,
                quality=self._jpeg_quality,
                subsampling=self._jpeg_subsampling,
                prefer_turbojpeg=self._prefer_turbojpeg,
            )
        except Exception as exc:
            return GestureCandidateCaptureResult(
                saved=False,
                reasons=reasons,
                skipped_reason="encode_failed",
                error=_error_code(exc),
            )
        encode_ms = round((time.perf_counter_ns() - encode_started_ns) / 1_000_000.0, 3)

        safe_debug = _json_safe_value(
            dict(debug_details or {}),
            max_depth=self._max_metadata_depth,
            max_items=self._max_metadata_items,
            max_string_chars=self._max_metadata_string_chars,
            max_repr_chars=self._max_metadata_repr_chars,
        )
        metadata = {
            "store_version": 2,
            "captured_at_utc": _isoformat_utc(fallback_epoch_now),
            "observed_at": round(float(observed_at), 6) if _finite_float_or_default(observed_at, default=None) is not None else None,
            "observed_at_utc": _isoformat_utc(event_epoch) if event_epoch is not None else None,
            "cooldown_s": round(self.cooldown_s, 6),
            "reasons": list(reasons),
            "frame_shape": [int(dimension) for dimension in array.shape],
            "original_frame_shape": [int(dimension) for dimension in original_shape],
            "resize_applied": resize_info is not None,
            "resize": resize_info,
            "gesture_debug": safe_debug,
            "image": {
                "format": "jpeg",
                "quality": self._jpeg_quality,
                "subsampling": self._jpeg_subsampling,
                "encoder": encoder_name,
                "bytes": len(jpeg_bytes),
                "encode_ms": encode_ms,
            },
        }

        try:
            self._ensure_capture_dir()
            _atomic_write_bytes(
                image_path,
                jpeg_bytes,
                file_mode=self._file_mode,
                durable=self._durable_writes,
            )
            metadata_bytes = json.dumps(
                metadata,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            _atomic_write_bytes(
                metadata_path,
                metadata_bytes,
                file_mode=self._file_mode,
                durable=self._durable_writes,
            )
        except Exception as exc:
            _best_effort_unlink(image_path)
            _best_effort_unlink(metadata_path)
            return GestureCandidateCaptureResult(
                saved=False,
                reasons=reasons,
                skipped_reason="write_failed",
                error=_error_code(exc),
            )

        with self._lock:
            self._prune_locked()

        return GestureCandidateCaptureResult(
            saved=True,
            reasons=reasons,
            image_path=str(image_path.resolve()),
            metadata_path=str(metadata_path.resolve()),
        )

    def _ensure_capture_dir(self) -> None:
        self.capture_dir.mkdir(mode=self._dir_mode, parents=True, exist_ok=True)
        if not self.capture_dir.is_dir():
            raise NotADirectoryError(str(self.capture_dir))
        _best_effort_chmod(self.capture_dir, self._dir_mode)

    def _prune_locked(self) -> None:
        """Keep only the newest bounded set of capture artifacts."""

        grouped: dict[str, dict[str, object]] = {}
        for suffix in (".jpg", ".json"):
            for path in self.capture_dir.glob(f"gesture_candidate_*{suffix}"):
                stem_entry = grouped.setdefault(path.stem, {"paths": [], "freshness": float("-inf")})
                stem_entry["paths"].append(path)
                freshness = _best_effort_artifact_freshness(path)
                if freshness > stem_entry["freshness"]:
                    stem_entry["freshness"] = freshness

        ordered = sorted(
            grouped.items(),
            key=lambda item: (float(item[1]["freshness"]), str(item[0])),
            reverse=True,
        )
        for stem, entry in ordered[self.max_images:]:
            for path in entry["paths"]:
                _best_effort_unlink(path)


def _candidate_reasons(debug_details: Mapping[str, object] | None) -> tuple[str, ...]:
    """Return one stable ordered list of reasons why a frame is worth saving."""

    details = debug_details or {}
    reasons: list[str] = []
    if bool(details.get("forensics_zero_signal_capture_requested")):
        reasons.append("forensics_zero_signal")
    if _coerce_non_negative_int(details.get("live_hand_count")) > 0:
        reasons.append("live_hand")
    if _normalized_label(details.get("live_fine_hand_gesture")) not in _NEGATIVE_LABELS:
        reasons.append("live_fine_gesture")
    if _normalized_label(details.get("live_gesture_event")) not in _NEGATIVE_LABELS:
        reasons.append("live_gesture_event")
    if _normalized_label(details.get("pose_fallback_fine_hand_gesture")) not in _NEGATIVE_LABELS:
        reasons.append("pose_fallback_fine_gesture")
    if _normalized_label(details.get("pose_fallback_gesture_event")) not in _NEGATIVE_LABELS:
        reasons.append("pose_fallback_gesture_event")
    if _normalized_label(details.get("pose_hint_source")) not in _NEGATIVE_LABELS:
        if _finite_float_or_default(details.get("pose_hint_confidence"), default=None) is not None:
            reasons.append("pose_hint")
    if (
        _coerce_non_negative_int(details.get("person_roi_detection_count")) > 0
        and _normalized_label(details.get("person_roi_combined_gesture")) in _NEGATIVE_LABELS
        and _normalized_label(details.get("final_resolved_source")) in _NEGATIVE_LABELS
    ):
        reasons.append("person_roi_hand_without_symbol")
    if _normalized_label(details.get("final_resolved_source")) not in _NEGATIVE_LABELS:
        reasons.append("resolved_candidate")
    return tuple(dict.fromkeys(reasons))


def _capture_stem(
    *,
    observed_at: float,
    sequence: int,
    reasons: tuple[str, ...],
    instance_token: str,
) -> str:
    """Build one stable bounded capture stem for image and metadata files."""

    timestamp_text, millis = _safe_timestamp_text(observed_at)
    reason_suffix = "_".join(_safe_slug(reason, limit=24) for reason in reasons[:3]) or "candidate"
    return (
        f"gesture_candidate_{timestamp_text}{millis:03d}Z_"
        f"{sequence:05d}_{instance_token}_{reason_suffix}"
    )


def _normalize_rgb_array(frame_rgb: Any) -> np.ndarray | None:
    """Coerce one frame-like object into a contiguous RGB uint8 array."""

    try:
        array = np.asarray(frame_rgb)
    except Exception:
        return None
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        return None
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        return None
    return np.array(array, dtype=np.uint8, copy=True, order="C")


def _downscale_rgb_array(
    array: np.ndarray,
    *,
    max_edge_px: int,
) -> tuple[np.ndarray, dict[str, object] | None]:
    """Optionally downscale one RGB array to bound encode cost."""

    if max_edge_px <= 0:
        return array, None

    height, width = int(array.shape[0]), int(array.shape[1])
    longest_edge = max(height, width)
    if longest_edge <= max_edge_px:
        return array, None

    scale = max_edge_px / float(longest_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return array, None

    resized = Image.fromarray(array).resize((new_width, new_height), resample=_PIL_RESAMPLING.LANCZOS)
    resized_array = np.asarray(resized)
    if not resized_array.flags.c_contiguous:
        resized_array = np.ascontiguousarray(resized_array)
    return resized_array, {
        "algorithm": "lanczos",
        "max_edge_px": max_edge_px,
        "scale": round(scale, 6),
        "from": [height, width],
        "to": [int(resized_array.shape[0]), int(resized_array.shape[1])],
    }


def _encode_jpeg_bytes(
    array: np.ndarray,
    *,
    quality: int,
    subsampling: str,
    prefer_turbojpeg: bool,
) -> tuple[bytes, str]:
    """Encode one RGB uint8 frame into JPEG bytes."""

    if prefer_turbojpeg:
        turbo = _get_turbojpeg()
        if turbo is not None and TJPF_RGB is not None:
            turbo_subsample = _TURBOJPEG_SUBSAMPLING.get(subsampling)
            if turbo_subsample is not None:
                encoded = turbo.encode(
                    array,
                    quality=quality,
                    pixel_format=TJPF_RGB,
                    jpeg_subsample=turbo_subsample,
                )
                return bytes(encoded), "turbojpeg"

    buffer = io.BytesIO()
    save_kwargs: dict[str, object] = {
        "format": "JPEG",
        "quality": quality,
    }
    pillow_subsampling = _PILLOW_SUBSAMPLING.get(subsampling)
    if pillow_subsampling is not None:
        save_kwargs["subsampling"] = pillow_subsampling
    Image.fromarray(array).save(buffer, **save_kwargs)
    return buffer.getvalue(), "pillow"


def _get_turbojpeg() -> Any | None:
    """Return one cached TurboJPEG instance when available."""

    global _TURBOJPEG_INSTANCE, _TURBOJPEG_INIT_FAILED
    if _TURBOJPEG_INIT_FAILED or TurboJPEG is None:
        return None
    if _TURBOJPEG_INSTANCE is not None:
        return _TURBOJPEG_INSTANCE
    with _TURBOJPEG_LOCK:
        if _TURBOJPEG_INSTANCE is not None:
            return _TURBOJPEG_INSTANCE
        if _TURBOJPEG_INIT_FAILED:
            return None
        try:
            _TURBOJPEG_INSTANCE = TurboJPEG()
        except Exception:
            _TURBOJPEG_INIT_FAILED = True
            _TURBOJPEG_INSTANCE = None
        return _TURBOJPEG_INSTANCE


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    file_mode: int,
    durable: bool,
) -> None:
    """Atomically replace one file with the provided bytes."""

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp_",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(tmp_fd, file_mode)
        with os.fdopen(tmp_fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _best_effort_chmod(path, file_mode)
        if durable:
            _best_effort_fsync_directory(path.parent)
    except Exception:
        _best_effort_unlink(tmp_path)
        raise


def _json_safe_value(
    value: object,
    *,
    max_depth: int,
    max_items: int,
    max_string_chars: int,
    max_repr_chars: int,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> object:
    """Convert one object tree into bounded JSON-safe primitives."""

    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, str):
        return _truncate_text(value, max_string_chars)
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, bytearray):
        return {"type": "bytearray", "len": len(value)}
    if isinstance(value, memoryview):
        return {"type": "memoryview", "len": len(value)}
    if isinstance(value, Path):
        return _truncate_text(str(value), max_string_chars)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return _json_safe_value(
            value.item(),
            max_depth=max_depth,
            max_items=max_items,
            max_string_chars=max_string_chars,
            max_repr_chars=max_repr_chars,
            _depth=_depth,
            _seen=_seen,
        )
    if isinstance(value, np.ndarray):
        summary: dict[str, object] = {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
            "size": int(value.size),
        }
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            try:
                summary["min"] = _json_safe_value(
                    value.min().item(),
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    max_repr_chars=max_repr_chars,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
                summary["max"] = _json_safe_value(
                    value.max().item(),
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    max_repr_chars=max_repr_chars,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            except Exception:
                pass
        return summary

    object_id = id(value)
    if object_id in _seen:
        return "<cycle>"
    if _depth >= max_depth:
        return _safe_repr(value, max_repr_chars)

    if isinstance(value, Mapping):
        _seen.add(object_id)
        try:
            items: dict[str, object] = {}
            iterator = iter(value.items())
            for index, (key, item) in enumerate(iterator):
                if index >= max_items:
                    items["<truncated>"] = f"{len(value) - max_items} more items" if hasattr(value, "__len__") else "more items"
                    break
                key_text = _truncate_text(str(key), max_string_chars)
                items[key_text] = _json_safe_value(
                    item,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    max_repr_chars=max_repr_chars,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            return items
        finally:
            _seen.discard(object_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(object_id)
        try:
            result: list[object] = []
            sequence = list(value)
            for item in sequence[:max_items]:
                result.append(
                    _json_safe_value(
                        item,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_string_chars=max_string_chars,
                        max_repr_chars=max_repr_chars,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                )
            if len(sequence) > max_items:
                result.append(f"<truncated {len(sequence) - max_items} items>")
            return result
        finally:
            _seen.discard(object_id)

    return _safe_repr(value, max_repr_chars)


def _normalized_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return text or ""


def _finite_float_or_default(value: object, *, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        number = float(value)
    elif isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
    else:
        return default
    if not math.isfinite(number):
        return default
    return number


def _plausible_epoch_seconds(value: object) -> float | None:
    number = _finite_float_or_default(value, default=None)
    if number is None:
        return None
    if number < _PLAUSIBLE_EPOCH_MIN or number > _PLAUSIBLE_EPOCH_MAX:
        return None
    return number


def _coerce_non_negative_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        number = int(value)
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            return 0
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
    else:
        return 0
    return max(0, number)


def _safe_timestamp_text(epoch_seconds: float) -> tuple[str, int]:
    try:
        dt = datetime.fromtimestamp(float(epoch_seconds), tz=_UTC)
    except (OverflowError, OSError, ValueError, TypeError):
        dt = datetime.now(_UTC)
        epoch_seconds = dt.timestamp()
    millis = int((float(epoch_seconds) - math.floor(float(epoch_seconds))) * 1000.0)
    millis = min(999, max(0, millis))
    return dt.strftime("%Y%m%dT%H%M%S"), millis


def _isoformat_utc(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch_seconds), tz=_UTC).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def _normalize_mode(value: object, *, default: int) -> int:
    if isinstance(value, int):
        mode = value
    elif isinstance(value, str):
        try:
            mode = int(value, 8)
        except (TypeError, ValueError, OverflowError):
            return default
    else:
        return default
    return max(0, min(0o777, mode))


def _normalized_jpeg_subsampling(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in _TURBOJPEG_SUBSAMPLING:
        return text
    if text in _PILLOW_SUBSAMPLING:
        return text
    return _DEFAULT_JPEG_SUBSAMPLING


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _safe_repr(value: object, limit: int) -> str:
    try:
        return _truncate_text(repr(value), limit)
    except Exception:
        return f"<unrepresentable:{value.__class__.__name__}>"


def _safe_slug(value: str, *, limit: int) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "candidate"
    return cleaned[:limit]


def _best_effort_artifact_freshness(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return float("-inf")


def _best_effort_chmod(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except Exception:
        return


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def _best_effort_fsync_directory(path: Path) -> None:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        return
    finally:
        try:
            os.close(fd)
        except Exception:
            return


def _error_code(exc: Exception) -> str:
    return exc.__class__.__name__.strip().lower() or "capture_write_failed"


__all__ = [
    "GestureCandidateCaptureResult",
    "GestureCandidateCaptureStore",
]
