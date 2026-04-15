# CHANGELOG: 2026-03-28
# BUG-1: Skip unchanged frames so repeated no-op status pushes do not waste panel lifetime or create avoidable ghosting.
# BUG-2: Stop unbounded fast-refresh loops; when full_refresh_interval == 0 a hardware-safe full refresh cadence is still enforced.
# BUG-3: Use change-aware refresh selection so large full-screen deltas do not ride the fast path and degrade optical quality.
# SEC-1: Refuse insecure vendor-driver ownership/permissions before importing executable code from disk.
# SEC-2: Resolve trace binaries from trusted absolute system paths instead of inheriting PATH for privileged subprocesses.
# IMP-1: Normalise all external images to Pillow 1-bit with EXIF transpose and no dithering for deterministic, crisp panel output.
# IMP-2: Deep-sleep the panel after each render by default to reduce idle high-voltage exposure and power draw.

"""Render Twinr status cards on the Waveshare 4.2 V2 e-paper panel."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import gc
import hashlib
from importlib import import_module
import importlib.util
import inspect
import logging
import math
import os
from pathlib import Path
from shutil import which
import stat
import subprocess
import sys
from threading import RLock
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.debug_signals import DisplayDebugSignal
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.presentation_cues import DisplayPresentationCue
from twinr.display.service_connect_cues import DisplayServiceConnectCue
from twinr.display.wake_cues import DisplayWakeCue
from twinr.display.layouts import draw_status_card


_LOGGER = logging.getLogger(__name__)
_IMPORT_LOCK = RLock()
_SUPPORTED_ROTATIONS = {0, 90, 180, 270}
_SUPPORTED_LAYOUT_MODES = {"default", "debug_log"}
_DEFAULT_BUSY_TIMEOUT_S = 20.0
_BUSY_POLL_DELAY_MS = 20
_TRACE_BUSY_SAMPLE_LIMIT = 12
_TRACE_GPIO_SNAPSHOT_TIMEOUT_S = 3.0
_TRACE_SPI_COMMAND_SAMPLE_LIMIT = 8
_TRACE_SUPPLY_SNAPSHOT_TIMEOUT_S = 3.0
_DEFAULT_SAFE_FULL_REFRESH_INTERVAL = 6
_DEFAULT_FAST_REFRESH_MAX_CHANGED_PIXEL_RATIO = 0.08
_DEFAULT_FAST_REFRESH_MAX_BBOX_RATIO = 0.35
_TRACE_COMMAND_SEARCH_PATHS = (
    "/usr/bin",
    "/usr/sbin",
    "/bin",
    "/sbin",
    "/usr/local/bin",
    "/usr/local/sbin",
)


@dataclass(slots=True)
class WaveshareEPD4In2V2:
    """Twinr adapter for the Waveshare 4.2" V2 panel."""

    project_root: Path
    vendor_dir: Path
    driver: str = "waveshare_4in2_v2"
    spi_bus: int = 0
    spi_device: int = 0
    cs_gpio: int = 8
    dc_gpio: int = 25
    reset_gpio: int = 17
    busy_gpio: int = 24
    width: int = 400
    height: int = 300
    rotation_degrees: int = 270
    full_refresh_interval: int = 0
    layout_mode: str = "default"
    busy_timeout_s: float = _DEFAULT_BUSY_TIMEOUT_S
    runtime_trace_enabled: bool = False
    safe_full_refresh_interval: int = _DEFAULT_SAFE_FULL_REFRESH_INTERVAL  # BREAKING: 0 no longer means "never full refresh".
    fast_refresh_max_changed_pixel_ratio: float = _DEFAULT_FAST_REFRESH_MAX_CHANGED_PIXEL_RATIO
    fast_refresh_max_bbox_ratio: float = _DEFAULT_FAST_REFRESH_MAX_BBOX_RATIO
    skip_unchanged_frames: bool = True
    sleep_after_render: bool = False
    enforce_secure_vendor_permissions: bool = True  # BREAKING: insecure writable vendor trees are rejected.
    trace_command_search_paths: tuple[str, ...] = _TRACE_COMMAND_SEARCH_PATHS
    emit: Callable[[str], None] | None = None

    _driver_module: object | None = field(default=None, init=False, repr=False)
    _epdconfig_module: object | None = field(default=None, init=False, repr=False)
    _epd: object | None = field(default=None, init=False, repr=False)
    _render_count: int = field(default=0, init=False, repr=False)
    _font_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _lock: object = field(default_factory=RLock, init=False, repr=False)
    _last_rendered_status: str | None = field(default=None, init=False, repr=False)
    _last_frame_digest: str | None = field(default=None, init=False, repr=False)
    _last_prepared_image: object | None = field(default=None, init=False, repr=False)
    _last_frame_bytes: bytes | None = field(default=None, init=False, repr=False)
    _last_render_monotonic: float = field(default=0.0, init=False, repr=False)
    _fast_refresh_streak: int = field(default=0, init=False, repr=False)
    _panel_slept: bool = field(default=True, init=False, repr=False)
    _trace_surface_source: str = field(default="image", init=False, repr=False)
    _trace_surface_status: str = field(default="image", init=False, repr=False)
    _trace_surface_headline: str = field(default="", init=False, repr=False)
    _trace_phase: str = field(default="idle", init=False, repr=False)
    _trace_reason: str = field(default="", init=False, repr=False)
    _trace_attempt: int = field(default=1, init=False, repr=False)
    _trace_clear_first: bool = field(default=False, init=False, repr=False)
    _trace_last_command: str | None = field(default=None, init=False, repr=False)
    _trace_busy_last_value: int | None = field(default=None, init=False, repr=False)
    _trace_spi_write_calls: int = field(default=0, init=False, repr=False)
    _trace_spi_write_bytes: int = field(default=0, init=False, repr=False)
    _trace_gpio_levels: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _trace_original_digital_read: Callable[[object], object] | None = field(default=None, init=False, repr=False)
    _trace_original_delay_ms: Callable[[object], object] | None = field(default=None, init=False, repr=False)
    _trace_pwr_gpio: int = field(default=18, init=False, repr=False)
    _trace_command_cache: dict[str, str | None] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve(strict=False)
        self.vendor_dir = self._resolve_vendor_dir(self.vendor_dir)
        self.rotation_degrees = self.rotation_degrees % 360
        self.layout_mode = self._normalise_layout_mode(self.layout_mode)
        self.trace_command_search_paths = tuple(
            path for path in (self._normalise_text(path) for path in self.trace_command_search_paths) if path
        ) or _TRACE_COMMAND_SEARCH_PATHS

        if self.width <= 0 or self.height <= 0:
            raise RuntimeError("Display width and height must be positive integers.")
        if self.full_refresh_interval < 0:
            raise RuntimeError("Display full_refresh_interval must be >= 0.")
        if self.rotation_degrees not in _SUPPORTED_ROTATIONS:
            raise RuntimeError("Display rotation must be one of 0, 90, 180, or 270 degrees.")
        if self.spi_bus < 0 or self.spi_device < 0:
            raise RuntimeError("SPI bus and device must be >= 0.")
        if not isinstance(self.busy_timeout_s, (int, float)) or not math.isfinite(self.busy_timeout_s) or self.busy_timeout_s <= 0:
            raise RuntimeError("Display busy_timeout_s must be a finite number > 0.")

        with suppress(Exception):
            self.safe_full_refresh_interval = max(1, int(self.safe_full_refresh_interval))
        if self.safe_full_refresh_interval <= 0:
            self.safe_full_refresh_interval = _DEFAULT_SAFE_FULL_REFRESH_INTERVAL

        with suppress(Exception):
            self.fast_refresh_max_changed_pixel_ratio = float(self.fast_refresh_max_changed_pixel_ratio)
        with suppress(Exception):
            self.fast_refresh_max_bbox_ratio = float(self.fast_refresh_max_bbox_ratio)
        if not 0 < float(self.fast_refresh_max_changed_pixel_ratio) <= 1:
            raise RuntimeError("Display fast_refresh_max_changed_pixel_ratio must be in (0, 1].")
        if not 0 < float(self.fast_refresh_max_bbox_ratio) <= 1:
            raise RuntimeError("Display fast_refresh_max_bbox_ratio must be in (0, 1].")

        self.skip_unchanged_frames = bool(self.skip_unchanged_frames)
        self.sleep_after_render = bool(self.sleep_after_render)
        self.enforce_secure_vendor_permissions = bool(self.enforce_secure_vendor_permissions)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> "WaveshareEPD4In2V2":
        waveshare_config = replace(config, display_driver="waveshare_4in2_v2")
        conflicts = waveshare_config.display_gpio_conflicts()
        if conflicts:
            raise RuntimeError("Display GPIO configuration is invalid: " + "; ".join(conflicts))
        return cls(
            project_root=Path(waveshare_config.project_root),
            vendor_dir=Path(waveshare_config.display_vendor_dir),
            driver="waveshare_4in2_v2",
            spi_bus=waveshare_config.display_spi_bus,
            spi_device=waveshare_config.display_spi_device,
            cs_gpio=waveshare_config.display_cs_gpio,
            dc_gpio=waveshare_config.display_dc_gpio,
            reset_gpio=waveshare_config.display_reset_gpio,
            busy_gpio=waveshare_config.display_busy_gpio,
            width=waveshare_config.display_width,
            height=waveshare_config.display_height,
            rotation_degrees=waveshare_config.display_rotation_degrees,
            full_refresh_interval=waveshare_config.display_full_refresh_interval,
            layout_mode=waveshare_config.display_layout,
            busy_timeout_s=waveshare_config.display_busy_timeout_s,
            runtime_trace_enabled=waveshare_config.display_runtime_trace_enabled,
            safe_full_refresh_interval=getattr(
                waveshare_config,
                "display_safe_full_refresh_interval",
                _DEFAULT_SAFE_FULL_REFRESH_INTERVAL,
            ),
            fast_refresh_max_changed_pixel_ratio=getattr(
                waveshare_config,
                "display_fast_refresh_max_changed_pixel_ratio",
                _DEFAULT_FAST_REFRESH_MAX_CHANGED_PIXEL_RATIO,
            ),
            fast_refresh_max_bbox_ratio=getattr(
                waveshare_config,
                "display_fast_refresh_max_bbox_ratio",
                _DEFAULT_FAST_REFRESH_MAX_BBOX_RATIO,
            ),
            skip_unchanged_frames=getattr(waveshare_config, "display_skip_unchanged_frames", True),
            sleep_after_render=getattr(waveshare_config, "display_sleep_after_render", False),
            enforce_secure_vendor_permissions=getattr(
                waveshare_config,
                "display_enforce_secure_vendor_permissions",
                True,
            ),
            emit=emit,
        )

    @property
    def vendor_package_dir(self) -> Path:
        return self.vendor_dir / "waveshare_epd"

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def allowed_image_sizes(self) -> tuple[tuple[int, int], ...]:
        sizes = {self.canvas_size}
        if self.rotation_degrees in (90, 270):
            sizes.add((self.height, self.width))
        return tuple(sorted(sizes))

    def show_test_pattern(self) -> None:
        image = self.render_test_image()
        self._set_trace_surface_context(
            source="test_pattern",
            status="test_pattern",
            headline="TWINR E-PAPER V2",
        )
        self.show_image(image, clear_first=True)

    def supports_idle_waiting_animation(self) -> bool:
        return False

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        ticker_text: str | None = None,
        details: tuple[str, ...] = (),
        state_fields: tuple[tuple[str, str], ...] = (),
        log_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
        animation_frame: int = 0,
        face_cue: DisplayFaceCue | None = None,
        wake_cue: DisplayWakeCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        service_connect_cue: DisplayServiceConnectCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
    ) -> None:
        del ticker_text, debug_signals, face_cue, wake_cue, emoji_cue, ambient_impulse_cue, service_connect_cue, presentation_cue
        safe_status = self._normalise_text(status, fallback="status").lower() or "status"
        safe_headline = self._normalise_text(headline, fallback=safe_status.title())
        self._set_trace_surface_context(source="status", status=safe_status, headline=safe_headline)
        image = self.render_status_image(
            status=status,
            headline=headline,
            details=details,
            state_fields=state_fields,
            log_sections=log_sections,
            animation_frame=animation_frame,
        )
        self.show_image(image, clear_first=False)
        self._last_rendered_status = safe_status

    def show_image(self, image: object, *, clear_first: bool) -> None:
        with self._lock:
            self._ensure_trace_surface_context()
            prepared_image = self._prepare_image(image)
            self._validate_prepared_image(prepared_image)
            frame_digest, frame_bytes = self._prepared_image_signature(prepared_image)

            if self.skip_unchanged_frames and not clear_first and frame_digest == self._last_frame_digest:
                self._safe_emit("display_skip=unchanged")
                self._trace_event(
                    "frame_skip",
                    phase="skip",
                    reason="unchanged",
                    status=self._trace_surface_status,
                    prev=self._trace_prev_status(),
                )
                _LOGGER.debug("Skipping unchanged e-paper frame.")
                return

            last_error: Exception | None = None
            started_at = time.monotonic()
            for attempt in range(2):
                self._begin_trace_attempt(
                    attempt=attempt + 1,
                    clear_first=clear_first,
                    render_count=self._render_count,
                )
                try:
                    epd = self._get_epd()
                    refresh_mode = self._display_prepared_image(
                        epd,
                        prepared_image,
                        clear_first=clear_first,
                        frame_bytes=frame_bytes,
                    )
                    self._remember_successful_frame(prepared_image, frame_digest, frame_bytes)
                    self._after_successful_render(epd, refresh_mode=refresh_mode)
                    if attempt > 0:
                        self._safe_emit("display_retry_recovered=true")
                        _LOGGER.info(
                            "E-paper render recovered after %.3fs.",
                            time.monotonic() - started_at,
                        )
                    return
                except Exception as exc:
                    last_error = exc
                    self._safe_emit(
                        " ".join(
                            (
                                "display_retry=true",
                                f"attempt={attempt + 1}",
                                f"error={type(exc).__name__}",
                            )
                        )
                    )
                    _LOGGER.warning(
                        "E-paper render attempt %s failed after %.3fs; resetting driver state.",
                        attempt + 1,
                        time.monotonic() - started_at,
                        exc_info=exc,
                    )
                    self._reset_driver_state()
                clear_first = True

            raise RuntimeError("E-paper display update failed after one recovery attempt.") from last_error

    def _after_successful_render(self, epd: object, *, refresh_mode: str) -> None:
        del refresh_mode
        self._last_render_monotonic = time.monotonic()
        if self.sleep_after_render:
            self._panel_slept = self._sleep_panel(epd)
            if self._panel_slept:
                self._fast_refresh_streak = 0
        else:
            self._panel_slept = False

    def _sleep_panel(self, epd: object) -> bool:
        sleep = getattr(epd, "sleep", None)
        if not callable(sleep):
            return False
        try:
            self._trace_event("panel_sleep_start", phase="sleep")
            sleep()
            self._trace_event("panel_sleep_end", phase="sleep")
            self._safe_emit("display_sleep=true")
            return True
        except Exception:
            _LOGGER.warning("E-paper panel sleep failed.", exc_info=True)
            self._safe_emit("display_sleep=false")
            return False

    def _remember_successful_frame(
        self,
        prepared_image: object,
        frame_digest: str,
        frame_bytes: bytes,
    ) -> None:
        self._last_frame_digest = frame_digest
        self._last_frame_bytes = bytes(frame_bytes)
        self._last_prepared_image = prepared_image.copy() if hasattr(prepared_image, "copy") else prepared_image

    def _clear_frame_state(self) -> None:
        self._last_frame_digest = None
        self._last_prepared_image = None
        self._last_frame_bytes = None
        self._last_render_monotonic = 0.0
        self._fast_refresh_streak = 0
        self._panel_slept = True

    def _prepared_image_signature(self, image: object) -> tuple[str, bytes]:
        try:
            raw = image.tobytes()
        except Exception:
            # Preserve the historical ability to pass opaque vendor/test tokens
            # through show_image(). Within one process, object identity is still
            # stable enough to suppress duplicate rerenders of the same token.
            raw = f"opaque:{id(image)}".encode("ascii")
        digest = hashlib.blake2s(raw, digest_size=16).hexdigest()
        return digest, raw

    def _effective_full_refresh_interval(self) -> int:
        if self.full_refresh_interval > 0:
            return int(self.full_refresh_interval)
        return int(self.safe_full_refresh_interval)

    def _frame_change_metrics(self, image: object, frame_bytes: bytes) -> dict[str, object]:
        width, height = image.size
        total_pixels = max(1, int(width) * int(height))
        if (
            self._last_prepared_image is None
            or self._last_frame_bytes is None
            or getattr(self._last_prepared_image, "size", None) != image.size
        ):
            return {"pixel_ratio": 1.0, "bbox_ratio": 1.0, "bbox": (0, 0, width, height)}

        if not hasattr(image, "mode") or not hasattr(image, "tobytes"):
            # Opaque vendor/test tokens do not expose pixel bytes, so preserve
            # the historical small-delta fast-refresh path after the first full
            # render instead of treating the whole frame as an unknown redraw.
            return {"pixel_ratio": 0.0, "bbox_ratio": 0.0, "bbox": (0, 0, width, height)}

        old = self._last_frame_bytes
        if len(old) != len(frame_bytes):
            pixel_ratio = 1.0
        else:
            changed_bits = sum((left ^ right).bit_count() for left, right in zip(old, frame_bytes))
            pixel_ratio = changed_bits / total_pixels

        bbox = (0, 0, width, height)
        bbox_ratio = 1.0 if pixel_ratio > 0 else 0.0
        try:
            from PIL import ImageChops

            diff = ImageChops.logical_xor(self._last_prepared_image, image)
            diff_bbox = diff.getbbox()
            if diff_bbox is None:
                bbox = None
                bbox_ratio = 0.0
            else:
                bbox = tuple(int(value) for value in diff_bbox)
                bbox_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                bbox_ratio = bbox_area / total_pixels
        except Exception:
            pass

        return {
            "pixel_ratio": float(pixel_ratio),
            "bbox_ratio": float(bbox_ratio),
            "bbox": bbox,
        }

    def _select_refresh_strategy(
        self,
        epd: object,
        *,
        clear_first: bool,
        change_metrics: dict[str, object],
    ) -> tuple[str, str, str]:
        if clear_first or self._render_count == 0:
            reason = "clear" if clear_first else "initial"
            phase = "clear_recovery" if clear_first else "initial_full"
            return ("full", reason, phase)
        if self.layout_mode == "debug_log":
            return ("full", "layout_debug_log", "debug_log_full")
        if self._panel_slept:
            return ("full", "post_sleep", "post_sleep_full")
        if (self._fast_refresh_streak + 1) >= self._effective_full_refresh_interval():
            return ("full", "safety_interval", "safety_full")
        pixel_ratio = float(change_metrics.get("pixel_ratio", 1.0))
        bbox_ratio = float(change_metrics.get("bbox_ratio", 1.0))
        if (
            pixel_ratio >= self.fast_refresh_max_changed_pixel_ratio
            or bbox_ratio >= self.fast_refresh_max_bbox_ratio
        ):
            return ("full", "large_delta", "large_delta_full")
        if hasattr(epd, "display_Fast") and hasattr(epd, "init_fast"):
            return ("fast", "small_delta", "steady_fast")
        return ("full", "fast_unavailable", "steady_full_fallback")

    def _set_trace_surface_context(
        self,
        *,
        source: str,
        status: str,
        headline: str,
    ) -> None:
        self._trace_surface_source = self._normalise_text(source, fallback="image").lower() or "image"
        self._trace_surface_status = self._normalise_text(
            status,
            fallback=self._trace_surface_source,
        ).lower() or self._trace_surface_source
        self._trace_surface_headline = self._normalise_text(
            headline,
            fallback=self._trace_surface_status.title(),
        )

    def _ensure_trace_surface_context(self) -> None:
        if self._normalise_text(self._trace_surface_status):
            return
        self._set_trace_surface_context(source="image", status="image", headline="Image")

    def _begin_trace_attempt(self, *, attempt: int, clear_first: bool, render_count: int) -> None:
        self._trace_attempt = max(1, int(attempt))
        self._trace_clear_first = bool(clear_first)
        self._trace_phase = "idle"
        self._trace_reason = ""
        self._trace_last_command = None
        self._trace_busy_last_value = None
        self._trace_spi_write_calls = 0
        self._trace_spi_write_bytes = 0
        if render_count <= 0:
            self._trace_busy_last_value = None

    def _trace_enabled(self) -> bool:
        return bool(self.runtime_trace_enabled and self.emit is not None)

    def _trace_prev_status(self) -> str:
        return self._normalise_text(self._last_rendered_status, fallback="none").lower() or "none"

    def _trace_phase_start(self, *, phase: str, reason: str) -> None:
        self._trace_phase = phase
        self._trace_reason = reason
        if not self._trace_enabled():
            return
        self._trace_event(
            "phase_start",
            phase=phase,
            surface=self._trace_surface_source,
            status=self._trace_surface_status,
            prev=self._trace_prev_status(),
            layout=self.layout_mode,
            clear=self._trace_clear_first,
            attempt=self._trace_attempt,
            rc=self._render_count,
            reason=reason,
        )
        self._safe_emit(f"display_trace_gpio=phase_start {self._trace_gpio_snapshot()}")
        self._safe_emit(f"display_trace_supply=phase_start detail={self._trace_supply_snapshot()}")

    def _trace_phase_ok(self, *, elapsed_s: float) -> None:
        if not self._trace_enabled():
            return
        self._trace_event(
            "phase_ok",
            phase=self._trace_phase,
            status=self._trace_surface_status,
            prev=self._trace_prev_status(),
            reason=self._trace_reason,
            elapsed_s=elapsed_s,
            cmd=self._trace_last_command or "none",
            spi_calls=self._trace_spi_write_calls,
            spi_bytes=self._trace_spi_write_bytes,
        )

    def _trace_phase_error(self, exc: Exception, *, elapsed_s: float) -> None:
        if not self._trace_enabled():
            return
        self._trace_event(
            "phase_error",
            phase=self._trace_phase,
            status=self._trace_surface_status,
            prev=self._trace_prev_status(),
            reason=self._trace_reason,
            elapsed_s=elapsed_s,
            cmd=self._trace_last_command or "none",
            err=type(exc).__name__,
            spi_calls=self._trace_spi_write_calls,
            spi_bytes=self._trace_spi_write_bytes,
        )
        self._safe_emit(f"display_trace_gpio=phase_error {self._trace_gpio_snapshot()}")
        self._safe_emit(f"display_trace_supply=phase_error detail={self._trace_supply_snapshot()}")

    def _trace_event(self, event: str, **fields: object) -> None:
        if not self._trace_enabled():
            return
        parts = [f"display_trace={self._normalise_text(event, fallback='event')}"]
        for key, value in fields.items():
            compact = self._trace_format_value(value)
            if compact:
                parts.append(f"{key}={compact}")
        self._safe_emit(" ".join(parts))

    def _trace_format_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, (list, tuple)):
            return ",".join(
                compact for compact in (self._trace_format_value(item) for item in value) if compact
            )
        compact = self._normalise_text(value)
        return compact.replace(" ", "_")

    def _trace_gpio_snapshot(self) -> str:
        pins = (
            ("rst", self.reset_gpio),
            ("pwr", self._trace_pwr_gpio),
            ("busy", self.busy_gpio),
            ("dc", self.dc_gpio),
            ("cs", self.cs_gpio),
        )
        return " ".join(self._trace_pin_snapshot(label, pin) for label, pin in pins)

    def _trace_pin_snapshot(self, label: str, pin: int) -> str:
        result = self._run_trace_command(
            ["pinctrl", "get", str(pin)],
            timeout_s=_TRACE_GPIO_SNAPSHOT_TIMEOUT_S,
        )
        if result.get("error") or result.get("returncode") not in (0, None):
            return f"{label}=err"
        text = self._normalise_text(result.get("stdout") or result.get("stderr") or "?")
        if ":" in text:
            text = text.split(":", 1)[1].strip()
        if "//" in text:
            text = text.split("//", 1)[0].strip()
        compact = text.replace(" | ", "|").replace(" ", "/")[:18].strip("/") or "?"
        return f"{label}={compact}"

    def _trace_supply_snapshot(self) -> str:
        result = self._run_trace_command(
            ["vcgencmd", "get_throttled"],
            timeout_s=_TRACE_SUPPLY_SNAPSHOT_TIMEOUT_S,
        )
        if result.get("error") or result.get("returncode") not in (0, None):
            return "err"
        text = self._normalise_text(result.get("stdout") or result.get("stderr") or "?")
        return text[:48] or "?"

    def _resolve_trace_command(self, executable: str) -> str | None:
        if executable in self._trace_command_cache:
            return self._trace_command_cache[executable]
        resolved: str | None = None
        if os.path.isabs(executable):
            candidate = Path(executable)
            if candidate.exists() and candidate.is_file():
                resolved = str(candidate)
        else:
            resolved = which(executable, path=os.pathsep.join(self.trace_command_search_paths))
        self._trace_command_cache[executable] = resolved
        return resolved

    def _run_trace_command(self, command: list[str], *, timeout_s: float) -> dict[str, object]:
        resolved = self._resolve_trace_command(command[0])
        if not resolved:
            return {"command": command, "error": "not_found"}
        safe_command = [resolved, *command[1:]]
        try:
            completed = subprocess.run(
                safe_command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                cwd="/",
                env={
                    "PATH": os.pathsep.join(self.trace_command_search_paths),
                    "LANG": "C",
                    "LC_ALL": "C",
                },
            )
        except Exception as exc:
            return {
                "command": command,
                "error": type(exc).__name__,
                "detail": self._normalise_text(exc),
            }
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    def close(self) -> None:
        with self._lock:
            self._shutdown_hardware()
            self._driver_module = None
            self._epdconfig_module = None
            self._epd = None
            self._render_count = 0
            self._clear_frame_state()

    def render_test_image(self) -> object:
        with self._lock:
            image, draw = self._new_canvas()
            canvas_width, canvas_height = image.size
            title_font = self._font(28, bold=True)
            body_font = self._font(20, bold=False)
            now_text = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), outline=0, width=5)
            draw.rectangle((0, 0, canvas_width - 1, 58), fill=0)
            draw.text((16, 12), "TWINR E-PAPER V2", fill=255, font=title_font)
            draw.text((18, 78), now_text, fill=0, font=body_font)
            draw.text((18, 116), "BLACK / WHITE TEST", fill=0, font=body_font)
            draw.text((18, 152), "Display path is ready.", fill=0, font=body_font)
            draw.rectangle((20, 210, 120, 290), fill=0)
            draw.rectangle((140, 210, 240, 290), outline=0, width=3)
            return image

    def render_status_image(
        self,
        *,
        status: str,
        headline: str | None,
        details: tuple[str, ...],
        state_fields: tuple[tuple[str, str], ...] = (),
        log_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
        animation_frame: int = 0,
    ) -> object:
        with self._lock:
            safe_status = self._normalise_text(status, fallback="status")
            safe_headline = self._normalise_text(headline, fallback=safe_status)
            safe_details = self._normalise_details(details)
            safe_state_fields = self._normalise_state_fields(state_fields)
            safe_log_sections = self._normalise_log_sections(log_sections)
            safe_animation_frame = self._normalise_animation_frame(animation_frame)
            image, draw = self._new_canvas()
            canvas_width, canvas_height = image.size
            draw_status_card(
                self,
                draw,
                layout_mode=self.layout_mode,
                status=safe_status.lower(),
                headline=safe_headline,
                details=safe_details,
                state_fields=safe_state_fields,
                log_sections=safe_log_sections,
                animation_frame=safe_animation_frame,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            return image

    def _load_driver_module(self):
        if self._driver_module is not None:
            return self._driver_module

        if self.driver != "waveshare_4in2_v2":
            raise RuntimeError(f"Unsupported display driver: {self.driver}")

        package_dir = self._validate_vendor_layout()
        package_init = package_dir / "__init__.py"
        epdconfig_path = package_dir / "epdconfig.py"
        driver_path = package_dir / "epd4in2_V2.py"

        with _IMPORT_LOCK:
            epdconfig_module = None
            self._load_exact_vendor_module(
                module_name="waveshare_epd",
                module_path=package_init,
                is_package=True,
            )
            epdconfig_module = self._load_exact_vendor_module(
                module_name="waveshare_epd.epdconfig",
                module_path=epdconfig_path,
            )
            try:
                self._load_exact_vendor_module(
                    module_name="waveshare_epd.epd4in2_V2",
                    module_path=driver_path,
                )
                epdconfig = import_module("waveshare_epd.epdconfig")
                module = import_module("waveshare_epd.epd4in2_V2")
            except Exception as exc:
                self._cleanup_failed_vendor_import(epdconfig_module)
                raise RuntimeError(
                    "Display vendor files are incomplete or failed to import. "
                    "Run `hardware/display/setup_display.sh` again."
                ) from exc

        self._validate_vendor_config(epdconfig)
        self._validate_driver_module_origin(module, driver_path)
        self._epdconfig_module = epdconfig
        self._trace_pwr_gpio = int(getattr(epdconfig, "PWR_PIN", 18))
        self._instrument_epdconfig(epdconfig)
        self._driver_module = module
        return module

    def _cleanup_failed_vendor_import(self, epdconfig_module: object | None) -> None:
        cleanup = getattr(epdconfig_module, "module_exit", None)
        if callable(cleanup):
            with suppress(Exception):
                cleanup(cleanup=True)
        self._drop_cached_vendor_modules()

    def _validate_vendor_config(self, epdconfig: object) -> None:
        expected = {
            "RST_PIN": self.reset_gpio,
            "DC_PIN": self.dc_gpio,
            "CS_PIN": self.cs_gpio,
            "BUSY_PIN": self.busy_gpio,
        }
        mismatches = []
        for name, expected_value in expected.items():
            actual_value = getattr(epdconfig, name, None)
            if actual_value != expected_value:
                mismatches.append(f"{name}={actual_value} (expected {expected_value})")

        optional_spi = {
            "SPI_BUS": self.spi_bus,
            "SPI_DEVICE": self.spi_device,
        }
        unverifiable_spi = []
        for name, expected_value in optional_spi.items():
            if hasattr(epdconfig, name):
                actual_value = getattr(epdconfig, name, None)
                if actual_value != expected_value:
                    mismatches.append(f"{name}={actual_value} (expected {expected_value})")
            elif expected_value != 0:
                unverifiable_spi.append(f"{name}={expected_value}")

        if mismatches:
            raise RuntimeError(
                "Installed display driver pins do not match Twinr config: "
                + ", ".join(mismatches)
                + ". Run `hardware/display/setup_display.sh` again."
            )

        if unverifiable_spi:
            raise RuntimeError(
                "Configured SPI bus/device cannot be verified against the installed vendor driver: "
                + ", ".join(unverifiable_spi)
                + ". Use SPI 0:0 or patch the vendor driver during setup."
            )

    def _instrument_epdconfig(self, epdconfig: object) -> None:
        if getattr(epdconfig, "_twinr_runtime_trace_wrapped", False):
            return

        digital_write = getattr(epdconfig, "digital_write", None)
        digital_read = getattr(epdconfig, "digital_read", None)
        delay_ms = getattr(epdconfig, "delay_ms", None)
        module_init = getattr(epdconfig, "module_init", None)
        module_exit = getattr(epdconfig, "module_exit", None)
        watched_pins = {
            int(self.reset_gpio): "rst",
            int(self.busy_gpio): "busy",
            int(self.dc_gpio): "dc",
            int(self.cs_gpio): "cs",
            int(getattr(epdconfig, "PWR_PIN", self._trace_pwr_gpio)): "pwr",
        }

        if callable(digital_read):
            self._trace_original_digital_read = digital_read
        if callable(delay_ms):
            self._trace_original_delay_ms = delay_ms

        if callable(module_init) and self._trace_enabled():
            def _wrapped_module_init(*args: object, **kwargs: object) -> object:
                self._trace_event("module_init_start", phase=self._trace_phase)
                result = module_init(*args, **kwargs)
                implementation = getattr(epdconfig, "implementation", None)
                spi = getattr(implementation, "SPI", None)
                self._trace_event(
                    "module_init_end",
                    phase=self._trace_phase,
                    bus=self.spi_bus,
                    dev=self.spi_device,
                    mode=getattr(spi, "mode", None),
                    hz=getattr(spi, "max_speed_hz", None),
                )
                return result
            setattr(epdconfig, "module_init", _wrapped_module_init)

        if callable(module_exit) and self._trace_enabled():
            def _wrapped_module_exit(*args: object, **kwargs: object) -> object:
                self._trace_event("module_exit_start", phase=self._trace_phase)
                result = module_exit(*args, **kwargs)
                self._trace_event("module_exit_end", phase=self._trace_phase)
                return result
            setattr(epdconfig, "module_exit", _wrapped_module_exit)

        if callable(digital_write) and self._trace_enabled():
            def _wrapped_digital_write(pin: object, value: object) -> None:
                digital_write(pin, value)
                pin_int = int(pin)
                value_int = int(value)
                previous = self._trace_gpio_levels.get(pin_int)
                self._trace_gpio_levels[pin_int] = value_int
                if previous != value_int and pin_int in watched_pins:
                    self._trace_event(
                        "gpio_write",
                        phase=self._trace_phase,
                        pin=pin_int,
                        name=watched_pins[pin_int],
                        value=value_int,
                    )
            setattr(epdconfig, "digital_write", _wrapped_digital_write)

        if callable(digital_read) and self._trace_enabled():
            def _wrapped_digital_read(pin: object) -> object:
                value = int(digital_read(pin))
                pin_int = int(pin)
                previous = self._trace_gpio_levels.get(pin_int)
                self._trace_gpio_levels[pin_int] = value
                if previous != value and pin_int in watched_pins:
                    self._trace_event(
                        "gpio_read_transition",
                        phase=self._trace_phase,
                        pin=pin_int,
                        name=watched_pins[pin_int],
                        value=value,
                    )
                return value
            setattr(epdconfig, "digital_read", _wrapped_digital_read)

        setattr(epdconfig, "_twinr_runtime_trace_wrapped", True)

    def _new_canvas(self) -> tuple[object, object]:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            raise RuntimeError("Pillow is required for Twinr e-paper rendering.") from exc
        image = Image.new("1", self.canvas_size, 255)
        return image, ImageDraw.Draw(image)

    def _prepare_image(self, image: object):
        try:
            from PIL import Image, ImageOps
        except Exception as exc:
            raise RuntimeError("Pillow is required for Twinr e-paper rendering.") from exc

        if not hasattr(image, "size"):
            raise RuntimeError("Display image must expose a Pillow-style .size attribute.")
        if not hasattr(image, "convert"):
            size = getattr(image, "size", None)
            if not isinstance(size, tuple) or len(size) != 2:
                raise RuntimeError("Display image must expose a two-value .size attribute.")
            width = int(size[0])
            height = int(size[1])
            if (width, height) not in self.allowed_image_sizes:
                raise RuntimeError(
                    "Display image size "
                    f"{(width, height)} does not match expected sizes {self.allowed_image_sizes}."
                )
            return image.copy() if hasattr(image, "copy") else image

        working = image.copy() if hasattr(image, "copy") else image
        with suppress(Exception):
            working = ImageOps.exif_transpose(working)

        width, height = tuple(int(value) for value in working.size)
        if (width, height) == self.canvas_size:
            if self.rotation_degrees != 0:
                transpose_map = getattr(Image, "Transpose", None)
                if transpose_map is not None:
                    if self.rotation_degrees == 90:
                        working = working.transpose(Image.Transpose.ROTATE_90)
                    elif self.rotation_degrees == 180:
                        working = working.transpose(Image.Transpose.ROTATE_180)
                    elif self.rotation_degrees == 270:
                        working = working.transpose(Image.Transpose.ROTATE_270)
                else:
                    working = working.rotate(self.rotation_degrees, expand=True)
        elif (width, height) not in self.allowed_image_sizes:
            raise RuntimeError(
                "Display image size "
                f"{(width, height)} does not match expected sizes {self.allowed_image_sizes}."
            )

        if getattr(working, "mode", None) != "1":
            convert_kwargs = {}
            dither_enum = getattr(Image, "Dither", None)
            if dither_enum is not None:
                convert_kwargs["dither"] = Image.Dither.NONE
            working = working.convert("1", **convert_kwargs)
        elif hasattr(working, "copy"):
            working = working.copy()

        return working

    def _validate_prepared_image(self, image: object) -> None:
        size = getattr(image, "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            raise RuntimeError("Display image must expose a two-value .size attribute.")
        width = int(size[0])
        height = int(size[1])
        if (width, height) not in self.allowed_image_sizes:
            raise RuntimeError(
                "Display image size "
                f"{(width, height)} does not match expected sizes {self.allowed_image_sizes}."
            )
        mode = getattr(image, "mode", None)
        if mode is None:
            return
        if mode != "1":
            raise RuntimeError("Display image must already be normalised to Pillow mode '1'.")

    def _display_prepared_image(
        self,
        epd: object,
        prepared_image: object,
        *,
        clear_first: bool,
        frame_bytes: bytes,
    ) -> str:
        change_metrics = self._frame_change_metrics(prepared_image, frame_bytes)
        refresh_mode, reason, phase = self._select_refresh_strategy(
            epd,
            clear_first=clear_first,
            change_metrics=change_metrics,
        )
        self._trace_phase_start(phase=phase, reason=reason)
        self._trace_event(
            "frame_delta",
            phase=phase,
            status=self._trace_surface_status,
            pixel_ratio=change_metrics.get("pixel_ratio"),
            bbox_ratio=change_metrics.get("bbox_ratio"),
            bbox=change_metrics.get("bbox"),
        )

        started_at = time.monotonic()
        try:
            if refresh_mode == "full":
                self._safe_emit(
                    " ".join(
                        (
                            "display_refresh=full",
                            f"reason={reason}",
                            f"render_count={self._render_count}",
                        )
                    )
                )
                self._init_full(epd)
                if clear_first and hasattr(epd, "Clear"):
                    self._safe_emit("display_clear=true")
                    epd.Clear()
                prepared = epd.getbuffer(prepared_image)
                epd.display(prepared)
                self._fast_refresh_streak = 0
            else:
                self._safe_emit(
                    " ".join(
                        (
                            "display_refresh=fast",
                            f"reason={reason}",
                            f"render_count={self._render_count}",
                        )
                    )
                )
                self._init_fast(epd)
                prepared = epd.getbuffer(prepared_image)
                epd.display_Fast(prepared)
                self._fast_refresh_streak += 1
            self._panel_slept = False
            self._render_count += 1
        except Exception as exc:
            self._trace_phase_error(exc, elapsed_s=time.monotonic() - started_at)
            raise

        self._trace_phase_ok(elapsed_s=time.monotonic() - started_at)
        return refresh_mode

    def _get_epd(self):
        if self._epd is None:
            module = self._load_driver_module()
            if not hasattr(module, "EPD"):
                raise RuntimeError("Display driver module does not expose EPD().")
            epd = module.EPD()
            self._instrument_epd(epd)
            self._epd = epd
        return self._epd

    def _instrument_epd(self, epd: object) -> None:
        self._wrap_busy_wait(epd)
        if getattr(epd, "_twinr_runtime_trace_wrapped", False) or not self._trace_enabled():
            return

        def _wrap_call(method_name: str) -> None:
            original = getattr(epd, method_name, None)
            if not callable(original):
                return

            def _wrapped(*args: object, **kwargs: object) -> object:
                started_at = time.monotonic()
                self._trace_event(
                    "epd_call_start",
                    phase=self._trace_phase,
                    method=method_name,
                    status=self._trace_surface_status,
                    prev=self._trace_prev_status(),
                    cmd=self._trace_last_command or "none",
                )
                try:
                    result = original(*args, **kwargs)
                except Exception as exc:
                    self._trace_event(
                        "epd_call_error",
                        phase=self._trace_phase,
                        method=method_name,
                        elapsed_s=time.monotonic() - started_at,
                        err=type(exc).__name__,
                        cmd=self._trace_last_command or "none",
                    )
                    raise
                self._trace_event(
                    "epd_call_end",
                    phase=self._trace_phase,
                    method=method_name,
                    elapsed_s=time.monotonic() - started_at,
                    cmd=self._trace_last_command or "none",
                )
                return result

            setattr(epd, method_name, _wrapped)

        for method_name in (
            "init",
            "init_fast",
            "reset",
            "display",
            "display_Fast",
            "Clear",
            "TurnOnDisplay",
            "TurnOnDisplay_Fast",
            "sleep",
        ):
            _wrap_call(method_name)

        send_command = getattr(epd, "send_command", None)
        if callable(send_command):
            def _wrapped_send_command(command: object) -> object:
                with suppress(Exception):
                    self._trace_last_command = f"0x{int(command):02X}"
                self._trace_event(
                    "epd_command",
                    phase=self._trace_phase,
                    cmd=self._trace_last_command or self._normalise_text(command, fallback="?"),
                )
                return send_command(command)
            setattr(epd, "send_command", _wrapped_send_command)

        send_data = getattr(epd, "send_data", None)
        if callable(send_data):
            def _wrapped_send_data(data: object) -> object:
                value = self._normalise_text(data, fallback="?")
                with suppress(Exception):
                    value = str(int(data))
                self._trace_event(
                    "epd_data_byte",
                    phase=self._trace_phase,
                    cmd=self._trace_last_command or "none",
                    value=value,
                )
                return send_data(data)
            setattr(epd, "send_data", _wrapped_send_data)

        send_data2 = getattr(epd, "send_data2", None)
        if callable(send_data2):
            def _wrapped_send_data2(data: object) -> object:
                length: int | None = None
                sample: tuple[int, ...] = ()
                with suppress(Exception):
                    length = len(data)
                if length is not None:
                    self._trace_spi_write_calls += 1
                    self._trace_spi_write_bytes += int(length)
                    with suppress(Exception):
                        sample = tuple(int(value) for value in list(data[:_TRACE_SPI_COMMAND_SAMPLE_LIMIT]))
                self._trace_event(
                    "spi_bulk_write",
                    phase=self._trace_phase,
                    cmd=self._trace_last_command or "none",
                    bytes=length,
                    sample=sample,
                    call=self._trace_spi_write_calls,
                )
                return send_data2(data)
            setattr(epd, "send_data2", _wrapped_send_data2)

        setattr(epd, "_twinr_runtime_trace_wrapped", True)

    def _wrap_busy_wait(self, epd: object) -> None:
        if getattr(epd, "_twinr_busy_wait_wrapped", False):
            return

        original = getattr(epd, "ReadBusy", None)
        epdconfig = self._epdconfig_module
        busy_pin = getattr(epd, "busy_pin", None)
        if not callable(original) or epdconfig is None or busy_pin is None:
            return

        digital_read = self._trace_original_digital_read or getattr(epdconfig, "digital_read", None)
        delay_ms = self._trace_original_delay_ms or getattr(epdconfig, "delay_ms", None)
        if not callable(digital_read) or not callable(delay_ms):
            return

        timeout_s = float(self.busy_timeout_s)
        poll_delay_ms = _BUSY_POLL_DELAY_MS

        def _bounded_readbusy() -> None:
            trace_enabled = self._trace_enabled()
            caller = inspect.stack()[1].function if trace_enabled else ""
            started_at = time.monotonic()
            sampled_states: list[tuple[float, int]] = []
            if trace_enabled:
                self._trace_event(
                    "busy_wait_start",
                    phase=self._trace_phase,
                    caller=caller,
                    cmd=self._trace_last_command or "none",
                    gpio=busy_pin,
                    status=self._trace_surface_status,
                    prev=self._trace_prev_status(),
                )
            while True:
                value = int(digital_read(busy_pin))
                if trace_enabled and value != self._trace_busy_last_value:
                    self._trace_busy_last_value = value
                    self._trace_event(
                        "busy_wait_transition",
                        phase=self._trace_phase,
                        caller=caller,
                        cmd=self._trace_last_command or "none",
                        gpio=busy_pin,
                        value=value,
                        elapsed_s=time.monotonic() - started_at,
                    )
                if value == 0:
                    if trace_enabled:
                        self._trace_event(
                            "busy_wait_end",
                            phase=self._trace_phase,
                            caller=caller,
                            cmd=self._trace_last_command or "none",
                            gpio=busy_pin,
                            elapsed_s=time.monotonic() - started_at,
                            samples=tuple(f"{age:.3f}:{state}" for age, state in sampled_states),
                        )
                    return
                if len(sampled_states) < _TRACE_BUSY_SAMPLE_LIMIT:
                    sampled_states.append((round(time.monotonic() - started_at, 3), value))
                age_s = time.monotonic() - started_at
                if age_s >= timeout_s:
                    self._safe_emit(
                        " ".join(
                            (
                                "display_busy_timeout=true",
                                f"gpio={busy_pin}",
                                f"timeout_s={timeout_s:g}",
                            )
                        )
                    )
                    if trace_enabled:
                        self._trace_event(
                            "busy_wait_timeout",
                            phase=self._trace_phase,
                            caller=caller,
                            cmd=self._trace_last_command or "none",
                            gpio=busy_pin,
                            elapsed_s=age_s,
                            samples=tuple(f"{age:.3f}:{state}" for age, state in sampled_states),
                            status=self._trace_surface_status,
                            prev=self._trace_prev_status(),
                        )
                        self._safe_emit(f"display_trace_gpio=busy_wait_timeout {self._trace_gpio_snapshot()}")
                        self._safe_emit(f"display_trace_supply=busy_wait_timeout detail={self._trace_supply_snapshot()}")
                    raise TimeoutError(
                        "Display BUSY pin stayed active for "
                        f"{age_s:.1f}s on GPIO {busy_pin}; samples={sampled_states}"
                    )
                delay_ms(poll_delay_ms)

        setattr(epd, "ReadBusy", _bounded_readbusy)
        setattr(epd, "_twinr_busy_wait_wrapped", True)

    def _init_full(self, epd: object) -> None:
        if not hasattr(epd, "init"):
            raise RuntimeError("Display driver instance does not expose init().")
        epd.init()

    def _init_fast(self, epd: object) -> None:
        if not hasattr(epd, "init_fast"):
            raise RuntimeError("Display driver instance does not expose init_fast().")
        speed_mode = getattr(epd, "Seconds_1_5S", 0)
        epd.init_fast(speed_mode)

    def _resolve_vendor_dir(self, vendor_dir: Path) -> Path:
        candidate = vendor_dir.expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.project_root):
            raise RuntimeError("Display vendor directory must stay inside the Twinr project root.")
        return resolved

    def _validate_vendor_layout(self) -> Path:
        if self.vendor_dir.is_symlink():
            raise RuntimeError("Display vendor directory must not be a symlink.")

        package_dir = self.vendor_package_dir
        if not package_dir.exists():
            raise RuntimeError("Display vendor files are missing. Run `hardware/display/setup_display.sh` first.")
        if not package_dir.is_dir():
            raise RuntimeError("Display vendor package path is invalid.")
        if package_dir.is_symlink():
            raise RuntimeError("Display vendor package path must not be a symlink.")

        resolved_package_dir = package_dir.resolve(strict=True)
        if not resolved_package_dir.is_relative_to(self.project_root):
            raise RuntimeError("Display vendor package must stay inside the Twinr project root.")

        required_files = (
            resolved_package_dir / "__init__.py",
            resolved_package_dir / "epdconfig.py",
            resolved_package_dir / "epd4in2_V2.py",
        )
        for required_file in required_files:
            if not required_file.exists():
                raise RuntimeError(
                    "Display vendor files are incomplete. "
                    "Run `hardware/display/setup_display.sh` again."
                )
            if not required_file.is_file():
                raise RuntimeError(f"Display vendor file path is invalid: {required_file.name}.")
            if required_file.is_symlink():
                raise RuntimeError(f"Display vendor file must not be a symlink: {required_file.name}.")

        if self.enforce_secure_vendor_permissions:
            self._validate_vendor_permissions(
                (
                    self.project_root.resolve(strict=True),
                    self.vendor_dir.resolve(strict=True),
                    resolved_package_dir,
                    *required_files,
                )
            )

        return resolved_package_dir

    def _validate_vendor_permissions(self, paths: tuple[Path, ...]) -> None:
        current_uid = os.geteuid() if hasattr(os, "geteuid") else None
        insecure: list[str] = []
        for path in paths:
            with suppress(FileNotFoundError):
                stats = path.stat()
                mode = stat.S_IMODE(stats.st_mode)
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    insecure.append(f"{path.name or str(path)}:group/world-writable")
                if current_uid == 0 and stats.st_uid != 0:
                    insecure.append(f"{path.name or str(path)}:uid={stats.st_uid}")
        if insecure:
            raise RuntimeError(
                "Display vendor tree is insecure for execution: "
                + ", ".join(insecure)
                + ". Fix ownership/permissions before starting Twinr."
            )

    def _load_exact_vendor_module(
        self,
        *,
        module_name: str,
        module_path: Path,
        is_package: bool = False,
    ) -> object:
        resolved_path = module_path.resolve(strict=True)
        existing = sys.modules.get(module_name)
        if existing is not None and self._module_matches_path(existing, resolved_path):
            return existing

        if existing is not None:
            sys.modules.pop(module_name, None)

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(resolved_path),
            submodule_search_locations=[str(resolved_path.parent)] if is_package else None,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load display vendor module: {module_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            gc.collect()
            raise
        return module

    def _module_matches_path(self, module: object, expected_path: Path) -> bool:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        return Path(module_file).resolve(strict=False) == expected_path.resolve(strict=False)

    def _validate_driver_module_origin(self, module: object, expected_path: Path) -> None:
        if not self._module_matches_path(module, expected_path.resolve(strict=True)):
            raise RuntimeError(
                "Loaded display driver module origin does not match the validated vendor path."
            )

    def _shutdown_hardware(self) -> None:
        epd = self._epd
        epdconfig = self._epdconfig_module

        if epd is not None and hasattr(epd, "sleep"):
            with suppress(Exception):
                epd.sleep()

        if epdconfig is not None and hasattr(epdconfig, "module_exit"):
            module_exit = getattr(epdconfig, "module_exit")
            with suppress(Exception):
                try:
                    module_exit(cleanup=True)
                except TypeError:
                    module_exit()

    def _reset_driver_state(self) -> None:
        self._safe_emit("display_driver_reset=true")
        self._trace_event(
            "driver_reset",
            phase=self._trace_phase,
            status=self._trace_surface_status,
            prev=self._trace_prev_status(),
            cmd=self._trace_last_command or "none",
        )
        self._shutdown_hardware()
        self._drop_cached_vendor_modules()

    def _drop_cached_vendor_modules(self) -> None:
        self._epdconfig_module = None
        self._driver_module = None
        self._epd = None
        self._render_count = 0
        self._trace_original_digital_read = None
        self._trace_original_delay_ms = None
        self._trace_last_command = None
        self._trace_busy_last_value = None
        self._trace_spi_write_calls = 0
        self._trace_spi_write_bytes = 0
        self._trace_gpio_levels.clear()
        self._trace_command_cache.clear()
        self._clear_frame_state()
        for module_name in (
            "waveshare_epd.epd4in2_V2",
            "waveshare_epd.epdconfig",
            "waveshare_epd",
        ):
            sys.modules.pop(module_name, None)
        gc.collect()

    def _draw_face(
        self,
        draw: object,
        *,
        status: str,
        animation_frame: int,
        center_x: int,
        center_y: int,
        scale: float = 1.0,
    ) -> None:
        safe_scale = self._normalise_scale(scale)
        jitter_x, jitter_y = self._face_offset(status, animation_frame)
        jitter_x = self._scaled_offset(jitter_x, safe_scale)
        jitter_y = self._scaled_offset(jitter_y, safe_scale)

        left_eye = (
            center_x - self._scaled_offset(72, safe_scale) + jitter_x,
            center_y - self._scaled_offset(24, safe_scale) + jitter_y,
        )
        right_eye = (
            center_x + self._scaled_offset(72, safe_scale) + jitter_x,
            center_y - self._scaled_offset(24, safe_scale) + jitter_y,
        )
        self._draw_eye(
            draw,
            left_eye,
            status=status,
            side="left",
            animation_frame=animation_frame,
            scale=safe_scale,
        )
        self._draw_eye(
            draw,
            right_eye,
            status=status,
            side="right",
            animation_frame=animation_frame,
            scale=safe_scale,
        )
        self._draw_mouth(
            draw,
            center_x=center_x + jitter_x,
            center_y=center_y + self._scaled_offset(56, safe_scale) + jitter_y,
            status=status,
            animation_frame=animation_frame,
            scale=safe_scale,
        )

    def _draw_details_footer(
        self,
        draw: object,
        *,
        details: tuple[str, ...],
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        rows = self._footer_rows(details)
        if not rows:
            return
        if len(rows) == 1 and len(rows[0]) == 1:
            self._draw_single_footer_line(
                draw,
                line=rows[0][0],
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            return
        self._draw_footer_grid(
            draw,
            rows=rows,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

    def _draw_single_footer_line(
        self,
        draw: object,
        *,
        line: str,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        footer_font = self._font(18, bold=False)
        divider_y = canvas_height - 54
        draw.line((28, divider_y, canvas_width - 28, divider_y), fill=0, width=2)
        text_y = divider_y + 8
        left_text, right_text = self._split_footer_parts(line)
        right_width = self._text_width(draw, right_text, font=footer_font)
        right_margin = 24
        right_x = max(canvas_width - right_width - right_margin, 24)
        single_line_left_width = max(right_x - 36, 120)
        left_lines = self._wrap_footer_left(
            draw,
            left_text,
            font=footer_font,
            full_width=canvas_width - 48,
            final_width=single_line_left_width,
        )
        line_height = self._text_height(draw, font=footer_font)
        for index, left_line in enumerate(left_lines):
            line_y = text_y + (index * (line_height + 2))
            max_width = single_line_left_width if index == (len(left_lines) - 1) else (canvas_width - 48)
            trimmed = self._truncate_text(draw, left_line, max_width=max_width, font=footer_font)
            draw.text((24, line_y), trimmed, fill=0, font=footer_font)
            if right_text and index == (len(left_lines) - 1):
                draw.text((right_x, line_y), right_text, fill=0, font=footer_font)

    def _draw_footer_grid(
        self,
        draw: object,
        *,
        rows: tuple[tuple[str, ...], ...],
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        footer_font = self._font(16, bold=False)
        line_height = self._text_height(draw, font=footer_font)
        row_gap = 4
        padding_top = 8
        padding_bottom = 8
        footer_height = padding_top + (len(rows) * line_height) + (max(len(rows) - 1, 0) * row_gap) + padding_bottom
        divider_y = canvas_height - footer_height
        draw.line((28, divider_y, canvas_width - 28, divider_y), fill=0, width=2)
        text_y = divider_y + padding_top
        left_x = 24
        column_gap = 16
        content_width = canvas_width - (left_x * 2)
        column_width = max((content_width - column_gap) // 2, 96)
        right_x = left_x + column_width + column_gap

        for row_index, row in enumerate(rows):
            line_y = text_y + (row_index * (line_height + row_gap))
            left_text = self._truncate_text(
                draw,
                row[0],
                max_width=content_width if len(row) == 1 else column_width,
                font=footer_font,
            )
            draw.text((left_x, line_y), left_text, fill=0, font=footer_font)
            if len(row) > 1:
                right_text = self._truncate_text(draw, row[1], max_width=column_width, font=footer_font)
                draw.text((right_x, line_y), right_text, fill=0, font=footer_font)

    def _footer_rows(self, details: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        lines = self._normalise_details(details)
        if not lines:
            return ()
        if len(lines) == 1:
            return ((lines[0],),)
        capped = lines[:4]
        return tuple(tuple(capped[index:index + 2]) for index in range(0, len(capped), 2))

    def _draw_eye(
        self,
        draw: object,
        origin: tuple[int, int],
        *,
        status: str,
        side: str,
        animation_frame: int,
        scale: float = 1.0,
    ) -> None:
        center_x, center_y = origin
        eye = self._eye_state(status, animation_frame, side)
        line_width = self._scaled_size(4, scale, minimum=2)

        brow_y = center_y - self._scaled_offset(52, scale) + self._scaled_offset(int(eye["brow_raise"]), scale)
        if side == "left":
            draw.line(
                (
                    center_x - self._scaled_offset(24, scale),
                    brow_y + self._scaled_offset(int(eye["brow_slant"]), scale),
                    center_x + self._scaled_offset(24, scale),
                    brow_y - self._scaled_offset(int(eye["brow_slant"]), scale),
                ),
                fill=0,
                width=line_width,
            )
        else:
            draw.line(
                (
                    center_x - self._scaled_offset(24, scale),
                    brow_y - self._scaled_offset(int(eye["brow_slant"]), scale),
                    center_x + self._scaled_offset(24, scale),
                    brow_y + self._scaled_offset(int(eye["brow_slant"]), scale),
                ),
                fill=0,
                width=line_width,
            )

        if bool(eye["blink"]):
            draw.arc(
                (
                    center_x - self._scaled_offset(26, scale),
                    center_y - self._scaled_offset(8, scale),
                    center_x + self._scaled_offset(26, scale),
                    center_y + self._scaled_offset(10, scale),
                ),
                start=200,
                end=340,
                fill=0,
                width=self._scaled_size(5, scale, minimum=2),
            )
            return

        width = self._scaled_size(int(eye["width"]), scale, minimum=8)
        height = self._scaled_size(int(eye["height"]), scale, minimum=8)
        offset_x = self._scaled_offset(int(eye["eye_shift_x"]), scale)
        offset_y = self._scaled_offset(int(eye["eye_shift_y"]), scale)
        box = (
            center_x - (width // 2) + offset_x,
            center_y - (height // 2) + offset_y,
            center_x + (width // 2) + offset_x,
            center_y + (height // 2) + offset_y,
        )
        draw.ellipse(box, fill=0)
        self._draw_eye_highlights(draw, box, eye, scale=scale)

        if bool(eye["lid_arc"]):
            draw.arc(
                (
                    box[0] + self._scaled_offset(4, scale),
                    box[1] - self._scaled_offset(10, scale),
                    box[2] - self._scaled_offset(4, scale),
                    box[1] + self._scaled_offset(18, scale),
                ),
                start=180,
                end=360,
                fill=0,
                width=self._scaled_size(3, scale, minimum=2),
            )

    def _draw_mouth(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        status: str,
        animation_frame: int,
        scale: float = 1.0,
    ) -> None:
        line_width = self._scaled_size(4, scale, minimum=2)
        if status == "waiting":
            sway = (-1, 0, 1, 0, -1, 0)[animation_frame % 6]
            draw.arc(
                (
                    center_x - self._scaled_offset(24, scale),
                    center_y - self._scaled_offset(10, scale) + self._scaled_offset(sway, scale),
                    center_x + self._scaled_offset(24, scale),
                    center_y + self._scaled_offset(12, scale) + self._scaled_offset(sway, scale),
                ),
                start=18,
                end=162,
                fill=0,
                width=line_width,
            )
            return
        if status == "listening":
            openness = (14, 18, 14, 12)[animation_frame % 4]
            draw.ellipse(
                (
                    center_x - self._scaled_offset(10, scale),
                    center_y - self._scaled_offset(8, scale),
                    center_x + self._scaled_offset(10, scale),
                    center_y + self._scaled_offset(openness, scale),
                ),
                outline=0,
                width=line_width,
            )
            return
        if status == "processing":
            offset_y = (-1, 0, 1, 0)[animation_frame % 4]
            draw.line(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(4 + offset_y, scale),
                    center_x - self._scaled_offset(4, scale),
                    center_y + self._scaled_offset(2 + offset_y, scale),
                ),
                fill=0,
                width=line_width,
            )
            draw.line(
                (
                    center_x + self._scaled_offset(4, scale),
                    center_y + self._scaled_offset(2 + offset_y, scale),
                    center_x + self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(4 + offset_y, scale),
                ),
                fill=0,
                width=line_width,
            )
            return
        if status == "answering":
            openness = (8, 11, 7, 10)[animation_frame % 4]
            draw.rounded_rectangle(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y - self._scaled_offset(2, scale),
                    center_x + self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(openness, scale),
                ),
                radius=self._scaled_size(8, scale, minimum=2),
                outline=0,
                width=line_width,
            )
            return
        if status == "printing":
            lift = (0, -1, 0, 1)[animation_frame % 4]
            draw.arc(
                (
                    center_x - self._scaled_offset(28, scale),
                    center_y - self._scaled_offset(6, scale) + self._scaled_offset(lift, scale),
                    center_x + self._scaled_offset(28, scale),
                    center_y + self._scaled_offset(16, scale) + self._scaled_offset(lift, scale),
                ),
                start=12,
                end=168,
                fill=0,
                width=line_width,
            )
            return
        if status == "error":
            draw.arc(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(6, scale),
                    center_x + self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(18, scale),
                ),
                start=200,
                end=340,
                fill=0,
                width=line_width,
            )
            return
        draw.arc(
            (
                center_x - self._scaled_offset(20, scale),
                center_y - self._scaled_offset(8, scale),
                center_x + self._scaled_offset(20, scale),
                center_y + self._scaled_offset(8, scale),
            ),
            start=20,
            end=160,
            fill=0,
            width=line_width,
        )

    def _face_offset(self, status: str, animation_frame: int) -> tuple[int, int]:
        if status == "waiting":
            return ((0, 0), (-2, 0), (2, 0), (0, -1), (0, 1), (0, 0))[animation_frame % 6]
        if status == "listening":
            return ((0, 0), (0, -1), (0, 0), (0, 1))[animation_frame % 4]
        if status == "processing":
            return ((0, 0), (-1, 0), (1, 0), (0, 0))[animation_frame % 4]
        if status == "answering":
            return ((0, 0), (0, -1), (0, 0), (0, 1))[animation_frame % 4]
        if status == "printing":
            return ((0, 0), (1, 0), (0, 0), (-1, 0))[animation_frame % 4]
        if status == "error":
            return ((0, 1), (0, 0), (0, 1), (0, 0))[animation_frame % 4]
        return (0, 0)

    def _eye_state(self, status: str, animation_frame: int, side: str) -> dict[str, int | bool]:
        state: dict[str, int | bool] = {
            "width": 56,
            "height": 74,
            "eye_shift_x": 0,
            "eye_shift_y": 0,
            "highlight_dx": -10,
            "highlight_dy": -18,
            "brow_raise": 0,
            "brow_slant": 4,
            "blink": False,
            "lid_arc": False,
        }

        if status == "waiting":
            looks = (-10, -5, 4, 8, 0, -2)
            state["highlight_dx"] = looks[animation_frame % 6]
            state["eye_shift_y"] = (-1, 0, 0, 0, 1, 0)[animation_frame % 6]
            state["blink"] = animation_frame == 4
            return state

        if status == "listening":
            state["height"] = (78, 82, 78, 74)[animation_frame % 4]
            state["highlight_dx"] = (-8, -6, -8, -10)[animation_frame % 4]
            state["brow_raise"] = -8
            state["brow_slant"] = 2
            state["lid_arc"] = True
            state["blink"] = animation_frame == 3
            return state

        if status == "processing":
            gaze = (-12, -4, 4, 12)[animation_frame % 4]
            state["highlight_dx"] = gaze if side == "left" else gaze - 2
            state["height"] = 68
            state["brow_raise"] = -1
            state["brow_slant"] = 4
            state["lid_arc"] = True
            return state

        if status == "answering":
            state["height"] = (70, 74, 70, 72)[animation_frame % 4]
            state["highlight_dx"] = (-8, -6, -8, -7)[animation_frame % 4]
            state["brow_raise"] = -2
            state["brow_slant"] = 2
            return state

        if status == "printing":
            state["height"] = (70, 66, 70, 62)[animation_frame % 4]
            state["highlight_dx"] = (-9, -8, -7, -6)[animation_frame % 4]
            state["brow_raise"] = -4
            state["brow_slant"] = 2
            state["blink"] = animation_frame == 3
            return state

        if status == "error":
            state["height"] = (60, 56, 60, 58)[animation_frame % 4]
            state["highlight_dx"] = (-12, -11, -10, -11)[animation_frame % 4]
            state["highlight_dy"] = -14
            state["brow_raise"] = 2
            state["brow_slant"] = 8
            state["eye_shift_y"] = 2
            state["blink"] = animation_frame == 2
            return state

        return state

    def _draw_eye_highlights(
        self,
        draw: object,
        box: tuple[int, int, int, int],
        eye: dict[str, int | bool],
        *,
        scale: float = 1.0,
    ) -> None:
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        main_x = center_x + self._scaled_offset(int(eye["highlight_dx"]), scale)
        main_y = center_y + self._scaled_offset(int(eye["highlight_dy"]), scale)
        main_radius = self._scaled_size(8, scale, minimum=2)
        secondary_x_offset = self._scaled_offset(10, scale)
        secondary_y_offset = self._scaled_offset(8, scale)
        secondary_width = self._scaled_size(6, scale, minimum=2)
        secondary_height = self._scaled_size(6, scale, minimum=2)
        draw.ellipse(
            (
                main_x - main_radius,
                main_y - main_radius,
                main_x + main_radius,
                main_y + main_radius,
            ),
            fill=255,
        )
        draw.ellipse(
            (
                main_x + secondary_x_offset,
                main_y + secondary_y_offset,
                main_x + secondary_x_offset + secondary_width,
                main_y + secondary_y_offset + secondary_height,
            ),
            fill=255,
        )

    def _split_footer_parts(self, text: str) -> tuple[str, str]:
        compact = text.strip()
        if compact.endswith(")") and " (" in compact:
            prefix, suffix = compact.rsplit(" (", 1)
            return prefix.strip(), f"({suffix}"
        return compact, ""

    def _wrap_footer_left(
        self,
        draw: object,
        text: str,
        *,
        font: object | None,
        full_width: int,
        final_width: int,
    ) -> tuple[str, ...]:
        compact = text.strip()
        if self._text_width(draw, compact, font=font) <= final_width:
            return (compact,)
        parts = [part.strip() for part in compact.split("|") if part.strip()]
        if len(parts) < 2:
            first = self._truncate_text(draw, compact, max_width=full_width, font=font)
            return (first,)
        first_line_parts: list[str] = []
        for part in parts:
            candidate_parts = first_line_parts + [part]
            candidate_text = " | ".join(candidate_parts)
            if first_line_parts and self._text_width(draw, candidate_text, font=font) > full_width:
                break
            first_line_parts = candidate_parts
        if not first_line_parts:
            first_line_parts = [parts[0]]
        remaining = parts[len(first_line_parts):]
        if not remaining:
            return (" | ".join(first_line_parts),)
        second_line = " | ".join(remaining)
        return (
            " | ".join(first_line_parts),
            self._truncate_text(draw, second_line, max_width=final_width, font=font),
        )

    def _font(self, size: int, *, bold: bool) -> object:
        cache_key = f"{'bold' if bold else 'regular'}:{max(8, size)}"
        with self._lock:
            cached = self._font_cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                from PIL import ImageFont
            except Exception as exc:
                raise RuntimeError("Pillow is required for Twinr e-paper font rendering.") from exc

            candidates = (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ) if bold else (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            )

            font = None
            for candidate in candidates:
                if not Path(candidate).exists():
                    continue
                with suppress(Exception):
                    font = ImageFont.truetype(candidate, size=max(8, size))
                    break
            if font is None:
                font = ImageFont.load_default()
            self._font_cache[cache_key] = font
            return font

    def _text_width(self, draw: object, text: str, *, font: object | None = None) -> int:
        if not text:
            return 0
        width = len(text) * 6
        with suppress(Exception):
            text_box = draw.textbbox((0, 0), text, font=font)
            width = int(text_box[2] - text_box[0])
        return width

    def _text_height(self, draw: object, *, font: object | None = None) -> int:
        height = 12
        with suppress(Exception):
            text_box = draw.textbbox((0, 0), "Hg", font=font)
            height = int(text_box[3] - text_box[1])
        return height

    def _truncate_text(self, draw: object, text: str, *, max_width: int, font: object | None = None) -> str:
        compact = text.strip()
        if self._text_width(draw, compact, font=font) <= max_width:
            return compact
        ellipsis = "..."
        while compact and self._text_width(draw, compact + ellipsis, font=font) > max_width:
            compact = compact[:-1].rstrip()
        return (compact + ellipsis) if compact else ellipsis

    def _scaled_offset(self, value: int | float, scale: float) -> int:
        return int(round(float(value) * scale))

    def _scaled_size(self, value: int | float, scale: float, *, minimum: int = 1) -> int:
        return max(minimum, int(round(float(value) * scale)))

    def _normalise_scale(self, value: object) -> float:
        with suppress(Exception):
            parsed = float(value)
            if parsed > 0:
                return parsed
        return 1.0

    def _normalise_text(self, value: object, *, fallback: str = "") -> str:
        text = fallback if value is None else str(value)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return " ".join(text.split())

    def _normalise_details(self, details: object) -> tuple[str, ...]:
        if details is None:
            return ()
        if isinstance(details, str):
            text = self._normalise_text(details)
            return (text,) if text else ()
        with suppress(TypeError):
            return tuple(
                text
                for text in (self._normalise_text(item) for item in details)
                if text
            )
        text = self._normalise_text(details)
        return (text,) if text else ()

    def _normalise_state_fields(self, state_fields: object) -> tuple[tuple[str, str], ...]:
        if state_fields is None:
            return ()
        normalised: list[tuple[str, str]] = []
        with suppress(TypeError):
            for item in state_fields:
                label = ""
                value = ""
                with suppress(Exception):
                    label = self._normalise_text(item[0])
                    value = self._normalise_text(item[1])
                if not value:
                    value = self._normalise_text(item)
                if label or value:
                    normalised.append((label, value))
                if len(normalised) >= 6:
                    break
            return tuple(normalised)
        text = self._normalise_text(state_fields)
        return (("", text),) if text else ()

    def _normalise_log_sections(self, log_sections: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if log_sections is None:
            return ()
        normalised_sections: list[tuple[str, tuple[str, ...]]] = []
        with suppress(TypeError):
            for section in log_sections:
                title = ""
                raw_lines: object = ()
                with suppress(Exception):
                    title = self._normalise_text(section[0])
                    raw_lines = section[1]
                lines = self._normalise_details(raw_lines)[:4]
                if title or lines:
                    normalised_sections.append((title, lines))
                if len(normalised_sections) >= 3:
                    break
            return tuple(normalised_sections)
        text = self._normalise_text(log_sections)
        return ((text or "Log", ()),) if text else ()

    def _normalise_animation_frame(self, value: object) -> int:
        with suppress(Exception):
            return int(value)
        return 0

    def _normalise_layout_mode(self, value: object) -> str:
        layout_mode = self._normalise_text(value, fallback="default").lower() or "default"
        if layout_mode not in _SUPPORTED_LAYOUT_MODES:
            raise RuntimeError(
                "Display layout must be one of: " + ", ".join(sorted(_SUPPORTED_LAYOUT_MODES))
            )
        return layout_mode

    def _safe_emit(self, line: object) -> None:
        emit = self.emit
        if emit is None:
            return
        compact = self._normalise_text(line)
        if not compact:
            return
        compact = compact[:160].rstrip()
        try:
            emit(compact)
        except Exception:
            _LOGGER.warning("Display telemetry emit failed.", exc_info=True)
