"""Run bounded visual QC for Twinr's visible HDMI surface.

This module drives a small deterministic HDMI scene set through the producer-
facing face and presentation controllers, captures real screenshots of the
visible Wayland surface, computes image-diff metrics, and emits report-ready
artifacts. The goal is to make display acceptance reproducible instead of
relying on ad-hoc screenshots in `/tmp`.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Restored-home was never compared back to the initial idle surface, so the run could pass even when the UI failed to return home.
# BUG-2: Expected transitions passed on any non-zero pixel drift; tiny clock/cursor noise could create false positives.
# BUG-3: Python per-pixel loops scaled poorly on Pi 4; diff counting is now vectorized/fallback-histogram based.
# BUG-4: Screenshot capture reused final artifact paths directly; stale or partial files could be mistaken for fresh captures.
# SEC-1: External --image-path values could leak arbitrary readable files into the artifact bundle via sample_image_path / attachment_paths.
# SEC-2: Artifacts were written with default permissions and through predictable final paths; symlink clobbering and local disclosure were practical risks on shared/debuggable Pi deployments.
# IMP-1: Capture is now output-aware, optionally grim -o pinned, and supports deterministic scaling for multi-output Wayland/labwc setups.
# IMP-2: Diffing now emits perceptual metrics (SSIM/PSNR when available), thresholded bounding boxes, annotated diff boards, and optional ignore regions for dynamic UI zones.
# IMP-3: Static scenes can now wait for a visually stable frame instead of sampling mid-animation, while transient scenes can still intentionally capture the morph.
# IMP-4: The JSON summary is now richer and self-contained: sanitized sample assets, thresholds, dependency availability, and run metadata are persisted with the bundle.

from collections.abc import Callable, Sequence
import contextlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import warnings

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

try:  # Optional but strongly recommended for Pi-friendly perceptual metrics.
    import numpy as np
except Exception:  # pragma: no cover - best-effort optional dependency
    np = None  # type: ignore[assignment]

try:  # Optional; used when available for SSIM/PSNR on downsampled grayscale/RGB views.
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception:  # pragma: no cover - best-effort optional dependency
    peak_signal_noise_ratio = None  # type: ignore[assignment]
    structural_similarity = None  # type: ignore[assignment]

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.face_expressions import (
    DisplayFaceBrowStyle,
    DisplayFaceExpression,
    DisplayFaceExpressionController,
    DisplayFaceGazeDirection,
    DisplayFaceMouthStyle,
)
from twinr.display.presentation_cues import DisplayPresentationCardCue, DisplayPresentationController
from twinr.display.wayland_env import apply_wayland_environment


_DIFF_THRESHOLD = 24
_DEFAULT_FACE_HOLD_S = 6.0
_DEFAULT_PRESENTATION_HOLD_S = 12.0
_SCREENSHOT_TIMEOUT_S = 10.0
_THUMBNAIL_SIZE = (320, 192)
_CONTACT_SHEET_COLUMNS = 2
_DEFAULT_MIN_CHANGED_PIXELS = 512
_DEFAULT_MIN_CHANGED_RATIO = 0.0003
_DEFAULT_STABILIZE_TIMEOUT_S = 2.4
_DEFAULT_STABLE_CHANGED_RATIO = 0.0002
_DEFAULT_STABLE_DHASH_DISTANCE = 2
_DEFAULT_REFERENCE_MAX_CHANGED_RATIO = 0.010
_DEFAULT_REFERENCE_MIN_SSIM = 0.970
_DEFAULT_REFERENCE_MAX_DHASH_DISTANCE = 10
_DEFAULT_SAMPLE_IMAGE_MAX_PIXELS = 12_000_000
_DEFAULT_METRIC_MAX_EDGE = 1280
_DEFAULT_SECURE_DIR_MODE = 0o700
_DEFAULT_SECURE_FILE_MODE = 0o600
_SUPPORTED_EXTERNAL_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_DHASH_SIZE = 8
_WAYLAND_ENV_LOCK = threading.Lock()

try:  # Pillow 9.1+
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - older Pillow fallback
    _RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]
    _RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]

Rect = tuple[int, int, int, int]


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """Return the current UTC wall clock as ISO-8601 text."""

    return _utc_now().isoformat()


def _slug(value: object) -> str:
    """Normalize one free-form label into a short file-safe slug."""

    text = "".join(ch if str(ch).isalnum() else "_" for ch in str(value or "").strip().lower())
    compact = "_".join(part for part in text.split("_") if part)
    return compact or "item"


def _hash_distance(left: int | None, right: int | None) -> int | None:
    """Return the Hamming distance between two integer hashes."""

    if left is None or right is None:
        return None
    return (left ^ right).bit_count()


def _clamp_rect(rect: Rect, *, width: int, height: int) -> Rect | None:
    """Clamp one rectangle to image bounds and return None if it becomes empty."""

    left, top, right, bottom = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _parse_rects(value: object) -> tuple[Rect, ...]:
    """Normalize ignore-region config into (left, top, right, bottom) tuples."""

    if value is None:
        return ()
    data = value
    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            return ()
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return ()
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, Sequence) or isinstance(data, (bytes, bytearray, str)):
        return ()

    rects: list[Rect] = []
    for item in data:
        if isinstance(item, dict):
            if {"left", "top", "right", "bottom"} <= set(item):
                rect = (
                    int(item["left"]),
                    int(item["top"]),
                    int(item["right"]),
                    int(item["bottom"]),
                )
            elif {"x", "y", "width", "height"} <= set(item):
                rect = (
                    int(item["x"]),
                    int(item["y"]),
                    int(item["x"]) + int(item["width"]),
                    int(item["y"]) + int(item["height"]),
                )
            else:
                continue
            rects.append(rect)
            continue
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray, str)) and len(item) == 4:
            rects.append(tuple(int(value) for value in item))  # type: ignore[arg-type]
    return tuple(rects)


@dataclass(frozen=True, slots=True)
class DisplayVisualQcStep:
    """Describe one bounded visual-QC scene step."""

    key: str
    label: str
    description: str
    action: str
    delay_s: float
    hold_seconds: float | None = None
    expect_change_from_previous: bool = False
    face_expression: DisplayFaceExpression | None = None
    presentation_cards: tuple[DisplayPresentationCardCue, ...] = ()
    active_card_key: str | None = None
    require_stable_frame: bool = False
    settle_timeout_s: float | None = None
    settle_interval_s: float | None = None
    min_changed_pixels: int | None = None
    min_changed_ratio: float | None = None
    expect_similarity_to_key: str | None = None
    max_reference_changed_ratio: float | None = None
    min_reference_ssim: float | None = None
    max_reference_dhash_distance: int | None = None
    ignore_regions: tuple[Rect, ...] = ()


@dataclass(frozen=True, slots=True)
class DisplayVisualQcCapture:
    """Store one captured HDMI screenshot plus its scene metadata."""

    key: str
    label: str
    description: str
    image_path: str
    delay_s: float
    captured_at: str
    width: int
    height: int

    def attachment_name(self) -> str:
        """Return the capture file name for report references."""

        return Path(self.image_path).name


@dataclass(frozen=True, slots=True)
class DisplayVisualQcDiffMetric:
    """Store one image-diff metric between two visual-QC scenes."""

    from_key: str
    to_key: str
    diff_image_path: str
    changed_pixels: int
    changed_ratio: float
    bbox: tuple[int, int, int, int] | None
    ssim: float | None = None
    psnr: float | None = None
    mean_abs_diff: float | None = None
    max_abs_diff: int | None = None
    dhash_distance: int | None = None
    ignored_regions: tuple[Rect, ...] = ()
    threshold: int = _DIFF_THRESHOLD

    def attachment_name(self) -> str:
        """Return the diff-image file name for report references."""

        return Path(self.diff_image_path).name


@dataclass(frozen=True, slots=True)
class DisplayVisualQcRunResult:
    """Represent the complete artifact bundle of one visual-QC run."""

    generated_at: str
    workdir: str
    sample_image_path: str
    summary_path: str
    contact_sheet_path: str
    captures: tuple[DisplayVisualQcCapture, ...]
    diffs: tuple[DisplayVisualQcDiffMetric, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def attachment_paths(self) -> tuple[str, ...]:
        """Return all artifact file paths that belong to this QC run."""

        paths = [self.sample_image_path]
        paths.extend(capture.image_path for capture in self.captures)
        paths.extend(diff.diff_image_path for diff in self.diffs)
        paths.append(self.contact_sheet_path)
        paths.append(self.summary_path)
        return tuple(dict.fromkeys(paths))

    def to_summary_dict(self) -> dict[str, object]:
        """Return one JSON-safe summary payload for artifact storage."""

        return {
            "generated_at": self.generated_at,
            "workdir": self.workdir,
            "sample_image_path": self.sample_image_path,
            "contact_sheet_path": self.contact_sheet_path,
            "captures": [asdict(capture) for capture in self.captures],
            "diffs": [asdict(diff) for diff in self.diffs],
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class _ImageComparison:
    """Internal comparison metrics between two screenshots."""

    changed_pixels: int
    changed_ratio: float
    bbox: Rect | None
    mean_abs_diff: float
    max_abs_diff: int
    ssim: float | None
    psnr: float | None
    dhash_distance: int | None
    width: int
    height: int
    ignored_regions: tuple[Rect, ...]


class DisplayVisualQcRunner:
    """Drive a deterministic HDMI scene set and capture report-ready evidence."""

    def __init__(
        self,
        config: TwinrConfig,
        *,
        face_controller: DisplayFaceExpressionController | None = None,
        presentation_controller: DisplayPresentationController | None = None,
        screenshot_func: Callable[[Path], None] | None = None,
        sleep_func: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.face_controller = face_controller or DisplayFaceExpressionController.from_config(
            config,
            default_source="visual_qc",
        )
        self.presentation_controller = presentation_controller or DisplayPresentationController.from_config(
            config,
            default_source="visual_qc",
        )
        self._screenshot_func = screenshot_func or self._capture_with_grim
        self._capture_backend_name = "grim" if screenshot_func is None else "custom"
        self._sleep_func = sleep_func or time.sleep
        self._global_ignore_regions = self._resolve_global_ignore_regions()

    def run(
        self,
        output_dir: Path,
        *,
        emit: Callable[[str], None] | None = None,
        image_path: str | None = None,
        steps: Sequence[DisplayVisualQcStep] | None = None,
    ) -> DisplayVisualQcRunResult:
        """Execute the default HDMI scene set and persist a full artifact bundle."""

        output_dir = Path(output_dir).expanduser().resolve()
        self._ensure_private_directory(output_dir)
        emit = emit or (lambda _: None)
        sample_image_path = self._resolve_sample_image(output_dir, requested_path=image_path)
        step_list = tuple(steps or self.default_steps(sample_image_path=sample_image_path))
        if not step_list:
            raise RuntimeError("Display visual QC requires at least one scene step.")
        captures: list[DisplayVisualQcCapture] = []
        captures_by_key: dict[str, DisplayVisualQcCapture] = {}
        try:
            for index, step in enumerate(step_list, start=1):
                emit(f"visual_qc_step={step.key} action={step.action}")
                self._apply_step(step)
                self._sleep_func(max(0.0, float(step.delay_s)))
                capture_name = f"scene_{index:02d}_{_slug(step.key)}.png"
                capture_path = output_dir / capture_name
                reference_capture = captures_by_key.get(step.expect_similarity_to_key) if step.expect_similarity_to_key else None
                capture = self._capture_step(
                    step,
                    capture_path=capture_path,
                    previous_capture=captures[-1] if captures else None,
                    reference_capture=reference_capture,
                    emit=emit,
                )
                captures.append(capture)
                captures_by_key[capture.key] = capture
        finally:
            self.presentation_controller.clear()
            self.face_controller.clear()

        if not captures:
            raise RuntimeError("Display visual QC did not capture any scenes.")
        expected_size = (captures[0].width, captures[0].height)
        size_mismatches = [capture.key for capture in captures if (capture.width, capture.height) != expected_size]
        if size_mismatches:
            raise RuntimeError(
                "Display visual QC captured inconsistent surface sizes across scenes: "
                + ", ".join(size_mismatches)
            )
        diffs = self._build_diff_metrics(captures, steps=step_list, output_dir=output_dir)

        missing_change_pairs: list[str] = []
        for previous_step, current_step, diff in zip(step_list, step_list[1:], diffs):
            if current_step.expect_change_from_previous:
                required_pixels = self._required_changed_pixels(current_step, expected_size[0], expected_size[1])
                if diff.changed_pixels < required_pixels:
                    missing_change_pairs.append(f"{previous_step.key}->{current_step.key}")
        if missing_change_pairs:
            raise RuntimeError(
                "Display visual QC found scene transitions below the minimum expected visual change: "
                + ", ".join(missing_change_pairs)
            )

        contact_sheet_path = self._build_contact_sheet(captures, output_dir=output_dir)
        summary_path = output_dir / "visual_qc_summary.json"
        result = DisplayVisualQcRunResult(
            generated_at=_utc_now_iso(),
            workdir=str(output_dir),
            sample_image_path=str(sample_image_path),
            summary_path=str(summary_path),
            contact_sheet_path=str(contact_sheet_path),
            captures=tuple(captures),
            diffs=tuple(diffs),
            metadata={
                "diff_threshold": _DIFF_THRESHOLD,
                "global_ignore_regions": list(self._global_ignore_regions),
                "dependencies": {
                    "numpy": np is not None,
                    "scikit_image": structural_similarity is not None and peak_signal_noise_ratio is not None,
                },
                "capture_backend": self._capture_backend_name,
                "wayland_output_name": self._capture_output_name(),
                "grim_scale": self._grim_scale_factor(output_name=self._capture_output_name()),
                "metrics_version": "visual_qc_v2",
            },
        )
        self._write_text_atomic(summary_path, json.dumps(result.to_summary_dict(), indent=2, ensure_ascii=True))
        return result

    def default_steps(self, *, sample_image_path: Path) -> tuple[DisplayVisualQcStep, ...]:
        """Return the default HDMI scene set for visual QC."""

        cards = (
            DisplayPresentationCardCue(
                key="summary",
                kind="rich_card",
                title="Daily summary",
                subtitle="Primary card stays queued",
                body_lines=(
                    "Medication reminder remains secondary.",
                    "The image card should win by priority.",
                ),
                accent="info",
                priority=20,
            ),
            DisplayPresentationCardCue(
                key="family_photo",
                kind="image",
                title="Family photo",
                subtitle="Focused fullscreen image",
                body_lines=(
                    "The card should expand smoothly to fullscreen.",
                    "The face should react during the morph only.",
                ),
                image_path=str(sample_image_path),
                accent="warm",
                priority=90,
                face_emotion="happy",
            ),
        )
        return (
            DisplayVisualQcStep(
                key="idle_home",
                label="Idle home",
                description="Default waiting surface with no external cues or presentations.",
                action="idle",
                delay_s=0.45,
                require_stable_frame=True,
            ),
            DisplayVisualQcStep(
                key="face_react",
                label="Face reaction",
                description="External face cue should shift gaze and smile without changing the status panel.",
                action="face_expression",
                delay_s=0.35,
                hold_seconds=_DEFAULT_FACE_HOLD_S,
                expect_change_from_previous=True,
                require_stable_frame=True,
                min_changed_pixels=768,
                min_changed_ratio=0.00035,
                face_expression=DisplayFaceExpression(
                    gaze=DisplayFaceGazeDirection.RIGHT,
                    mouth=DisplayFaceMouthStyle.SMILE,
                    brows=DisplayFaceBrowStyle.RAISED,
                    blink=False,
                ),
            ),
            DisplayVisualQcStep(
                key="presentation_mid",
                label="Presentation morph",
                description="Prioritized image card should begin expanding while queued cards stay secondary.",
                action="presentation_scene",
                delay_s=0.22,
                hold_seconds=_DEFAULT_PRESENTATION_HOLD_S,
                expect_change_from_previous=True,
                require_stable_frame=False,
                min_changed_pixels=1024,
                min_changed_ratio=0.0005,
                presentation_cards=cards,
                active_card_key="family_photo",
            ),
            DisplayVisualQcStep(
                key="presentation_focused",
                label="Presentation focused",
                description="The active image card should finish fullscreen with calm telemetry and no extra panel churn.",
                action="hold",
                delay_s=0.82,
                expect_change_from_previous=True,
                require_stable_frame=True,
                min_changed_pixels=2048,
                min_changed_ratio=0.0010,
            ),
            DisplayVisualQcStep(
                key="restored_home",
                label="Restored home",
                description="Clearing both cue layers should return the display to the idle home surface.",
                action="idle",
                delay_s=0.40,
                expect_change_from_previous=True,
                require_stable_frame=True,
                min_changed_pixels=2048,
                min_changed_ratio=0.0010,
                expect_similarity_to_key="idle_home",
                max_reference_changed_ratio=0.010,
                min_reference_ssim=0.970,
                max_reference_dhash_distance=10,
            ),
        )

    def _resolve_sample_image(self, output_dir: Path, *, requested_path: str | None) -> Path:
        """Return the QC sample image path, sanitizing external inputs into the artifact bundle."""

        if requested_path:
            candidate = Path(requested_path).expanduser()
            if not candidate.is_absolute():
                candidate = Path(self.config.project_root).expanduser().resolve() / candidate
            resolved = candidate.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Display visual QC image was not found: {resolved}")
            return self._sanitize_external_sample_image(resolved, output_dir=output_dir)
        path = output_dir / "sample_family_photo.png"
        self._create_sample_image(path)
        return path

    def _sanitize_external_sample_image(self, source_path: Path, *, output_dir: Path) -> Path:
        """Copy an external image into the artifact bundle after validating it is a safe decodable image."""

        # BREAKING: external image_path no longer accepts arbitrary existing files; the source must decode
        # as a safe raster image and is copied into the bundle so report attachments cannot exfiltrate
        # unrelated files from the host filesystem.
        suffix = source_path.suffix.lower()
        if suffix not in _SUPPORTED_EXTERNAL_IMAGE_SUFFIXES:
            raise ValueError(
                "Display visual QC image must be a decodable raster image with one of these suffixes: "
                + ", ".join(sorted(_SUPPORTED_EXTERNAL_IMAGE_SUFFIXES))
            )
        max_pixels = int(getattr(self.config, "display_visual_qc_image_max_pixels", _DEFAULT_SAMPLE_IMAGE_MAX_PIXELS) or _DEFAULT_SAMPLE_IMAGE_MAX_PIXELS)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source_path) as image:
                image.verify()
            with Image.open(source_path) as image:
                image.load()
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"Display visual QC image has invalid size: {source_path}")
                if width * height > max_pixels:
                    raise ValueError(
                        f"Display visual QC image exceeds the configured safety limit of {max_pixels} pixels: {source_path}"
                    )
                converted = image.convert("RGB")
        sanitized_path = output_dir / f"sample_external_{_slug(source_path.stem)}.png"
        self._write_image_atomic(sanitized_path, converted)
        return sanitized_path

    def _create_sample_image(self, path: Path) -> None:
        """Create one calm synthetic image for the fullscreen image scene."""

        image = Image.new("RGB", (640, 360), (18, 20, 24))
        draw = ImageDraw.Draw(image)
        for y in range(image.height):
            shade = int(round(18 + ((92 - 18) * (y / max(1, image.height - 1)))))
            draw.line((0, y, image.width, y), fill=(shade, min(140, shade + 32), min(180, shade + 58)))
        draw.ellipse((46, 42, 156, 152), fill=(246, 228, 150))
        draw.polygon(((0, 280), (140, 170), (252, 280)), fill=(52, 70, 76))
        draw.polygon(((120, 280), (292, 132), (464, 280)), fill=(66, 86, 96))
        draw.polygon(((310, 280), (494, 150), (640, 280)), fill=(86, 108, 120))
        draw.rounded_rectangle((340, 78, 592, 260), radius=26, outline=(255, 255, 255), width=4)
        draw.rectangle((356, 96, 574, 240), fill=(242, 242, 236))
        draw.ellipse((392, 118, 456, 182), fill=(18, 20, 24))
        draw.ellipse((478, 118, 542, 182), fill=(18, 20, 24))
        draw.arc((408, 170, 532, 238), start=18, end=162, fill=(18, 20, 24), width=5)
        self._write_image_atomic(path, image)

    def _apply_step(self, step: DisplayVisualQcStep) -> None:
        """Apply one scene step through the producer-facing cue controllers."""

        if step.action == "idle":
            self.presentation_controller.clear()
            self.face_controller.clear()
            return
        if step.action == "face_expression":
            self.presentation_controller.clear()
            if step.face_expression is None:
                self.face_controller.clear()
                return
            self.face_controller.show_expression(
                step.face_expression,
                hold_seconds=step.hold_seconds or _DEFAULT_FACE_HOLD_S,
            )
            return
        if step.action == "presentation_scene":
            self.face_controller.clear()
            self.presentation_controller.show_scene(
                cards=step.presentation_cards,
                active_card_key=step.active_card_key,
                hold_seconds=step.hold_seconds or _DEFAULT_PRESENTATION_HOLD_S,
            )
            return
        if step.action == "hold":
            return
        raise ValueError(f"Unsupported visual QC step action: {step.action}")

    def _capture(self, path: Path) -> None:
        """Capture one screenshot into ``path`` and fail if it is missing or invalid."""

        if path.exists():
            path.unlink()
        self._screenshot_func(path)
        if not path.exists():
            raise RuntimeError(f"Visual QC capture did not produce a screenshot: {path}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Visual QC capture produced an empty screenshot: {path}")
        self._validate_image_file(path)

    def _capture_step(
        self,
        step: DisplayVisualQcStep,
        *,
        capture_path: Path,
        previous_capture: DisplayVisualQcCapture | None,
        reference_capture: DisplayVisualQcCapture | None,
        emit: Callable[[str], None],
    ) -> DisplayVisualQcCapture:
        """Capture one scene step, waiting for required change / stability / baseline restoration as needed."""

        settle_interval_s = self._step_settle_interval_s(step)
        max_attempts = self._max_capture_attempts(step, settle_interval_s, previous_capture, reference_capture)
        last_temp_path: Path | None = None
        last_capture: DisplayVisualQcCapture | None = None

        try:
            for attempt in range(1, max_attempts + 1):
                temp_path = self._temporary_artifact_path(capture_path)
                try:
                    self._capture(temp_path)
                    with Image.open(temp_path) as image:
                        width, height = image.size
                    candidate = DisplayVisualQcCapture(
                        key=step.key,
                        label=step.label,
                        description=step.description,
                        image_path=str(temp_path),
                        delay_s=float(step.delay_s),
                        captured_at=_utc_now_iso(),
                        width=width,
                        height=height,
                    )
                    comparison_to_previous = (
                        self._compare_paths(
                            previous_capture.image_path,
                            candidate.image_path,
                            ignore_regions=step.ignore_regions,
                        )
                        if previous_capture is not None
                        else None
                    )
                    comparison_to_reference = (
                        self._compare_paths(
                            reference_capture.image_path,
                            candidate.image_path,
                            ignore_regions=step.ignore_regions,
                        )
                        if reference_capture is not None
                        else None
                    )
                    comparison_to_last_attempt = (
                        self._compare_paths(
                            last_capture.image_path,
                            candidate.image_path,
                            ignore_regions=step.ignore_regions,
                        )
                        if last_capture is not None
                        else None
                    )

                    enough_change = self._meets_change_expectation(step, comparison_to_previous)
                    enough_reference_similarity = self._meets_reference_expectation(step, comparison_to_reference)
                    enough_stability = self._meets_stability_expectation(step, comparison_to_last_attempt)

                    if enough_change and enough_reference_similarity and enough_stability:
                        self._replace_file(temp_path, capture_path)
                        return replace(candidate, image_path=str(capture_path))

                    reason_parts: list[str] = []
                    if not enough_change and comparison_to_previous is not None:
                        required_pixels = self._required_changed_pixels(step, width, height)
                        reason_parts.append(f"min_change={comparison_to_previous.changed_pixels}/{required_pixels}")
                        if comparison_to_previous.ssim is not None:
                            reason_parts.append(f"prev_ssim={comparison_to_previous.ssim:.5f}")
                    if not enough_reference_similarity and comparison_to_reference is not None:
                        reason_parts.append(f"ref_ratio={comparison_to_reference.changed_ratio:.6f}")
                        if comparison_to_reference.ssim is not None:
                            reason_parts.append(f"ref_ssim={comparison_to_reference.ssim:.5f}")
                        if comparison_to_reference.dhash_distance is not None:
                            reason_parts.append(f"ref_dhash={comparison_to_reference.dhash_distance}")
                    if not enough_stability and comparison_to_last_attempt is not None:
                        reason_parts.append(f"settle_ratio={comparison_to_last_attempt.changed_ratio:.6f}")
                        if comparison_to_last_attempt.dhash_distance is not None:
                            reason_parts.append(f"settle_dhash={comparison_to_last_attempt.dhash_distance}")

                    if attempt >= max_attempts:
                        self._replace_file(temp_path, capture_path)
                        details = " ".join(reason_parts) if reason_parts else "capture did not satisfy scene expectations"
                        raise RuntimeError(
                            f"Display visual QC could not validate scene `{step.key}` after {attempt} attempts: {details}"
                        )

                    if last_temp_path is not None and last_temp_path.exists():
                        with contextlib.suppress(Exception):
                            last_temp_path.unlink()
                    last_temp_path = temp_path
                    last_capture = candidate
                    emit(f"visual_qc_retry={step.key} attempt={attempt + 1} {' '.join(reason_parts).strip()}".strip())
                    self._sleep_func(settle_interval_s)
                except Exception:
                    if temp_path.exists():
                        with contextlib.suppress(Exception):
                            temp_path.unlink()
                    raise
            raise RuntimeError(f"Display visual QC exhausted retries for scene `{step.key}`.")
        finally:
            if last_temp_path is not None and last_temp_path.exists():
                with contextlib.suppress(Exception):
                    last_temp_path.unlink()

    def _retry_delay_s(self) -> float:
        """Return the retry delay used when the visible surface has not updated yet."""

        poll_interval = float(getattr(self.config, "display_poll_interval_s", 0.5) or 0.5)
        return max(0.20, poll_interval + 0.10)

    def _step_settle_interval_s(self, step: DisplayVisualQcStep) -> float:
        """Return the polling interval used while waiting for one step to settle."""

        if step.settle_interval_s is not None:
            return max(0.05, float(step.settle_interval_s))
        return self._retry_delay_s()

    def _max_capture_attempts(
        self,
        step: DisplayVisualQcStep,
        settle_interval_s: float,
        previous_capture: DisplayVisualQcCapture | None,
        reference_capture: DisplayVisualQcCapture | None,
    ) -> int:
        """Return the number of capture attempts budgeted for one scene."""

        if (
            previous_capture is None
            and reference_capture is None
            and not step.require_stable_frame
            and not step.expect_change_from_previous
        ):
            return 1
        if not step.require_stable_frame and step.expect_change_from_previous and reference_capture is None:
            return 3
        settle_timeout_s = float(step.settle_timeout_s or getattr(self.config, "display_visual_qc_stabilize_timeout_s", _DEFAULT_STABILIZE_TIMEOUT_S))
        return max(2, int(math.ceil(settle_timeout_s / max(0.05, settle_interval_s))) + 1)

    def _required_changed_pixels(self, step: DisplayVisualQcStep, width: int, height: int) -> int:
        """Return the thresholded number of changed pixels required for a meaningful scene change."""

        total_pixels = max(1, width * height)
        configured_pixels = int(step.min_changed_pixels or getattr(self.config, "display_visual_qc_min_changed_pixels", _DEFAULT_MIN_CHANGED_PIXELS) or _DEFAULT_MIN_CHANGED_PIXELS)
        configured_ratio = float(step.min_changed_ratio or getattr(self.config, "display_visual_qc_min_changed_ratio", _DEFAULT_MIN_CHANGED_RATIO) or _DEFAULT_MIN_CHANGED_RATIO)
        return max(1, configured_pixels, int(math.ceil(total_pixels * configured_ratio)))

    def _meets_change_expectation(
        self,
        step: DisplayVisualQcStep,
        comparison: _ImageComparison | None,
    ) -> bool:
        """Return True when the previous->current comparison satisfies the step's required visible change."""

        if not step.expect_change_from_previous or comparison is None:
            return True
        required_pixels = self._required_changed_pixels(step, comparison.width, comparison.height)
        return comparison.changed_pixels >= required_pixels

    def _meets_reference_expectation(
        self,
        step: DisplayVisualQcStep,
        comparison: _ImageComparison | None,
    ) -> bool:
        """Return True when the scene is sufficiently close to its reference scene."""

        if step.expect_similarity_to_key is None or comparison is None:
            return True
        max_ratio = float(
            step.max_reference_changed_ratio
            or getattr(self.config, "display_visual_qc_reference_max_changed_ratio", _DEFAULT_REFERENCE_MAX_CHANGED_RATIO)
            or _DEFAULT_REFERENCE_MAX_CHANGED_RATIO
        )
        if comparison.changed_ratio > max_ratio:
            return False
        required_ssim = step.min_reference_ssim
        if required_ssim is None:
            required_ssim = float(
                getattr(self.config, "display_visual_qc_reference_min_ssim", _DEFAULT_REFERENCE_MIN_SSIM)
                or _DEFAULT_REFERENCE_MIN_SSIM
            )
        if comparison.ssim is not None and comparison.ssim < required_ssim:
            return False
        max_dhash_distance = step.max_reference_dhash_distance
        if max_dhash_distance is None:
            max_dhash_distance = int(
                getattr(self.config, "display_visual_qc_reference_max_dhash_distance", _DEFAULT_REFERENCE_MAX_DHASH_DISTANCE)
                or _DEFAULT_REFERENCE_MAX_DHASH_DISTANCE
            )
        if comparison.dhash_distance is not None and comparison.dhash_distance > max_dhash_distance:
            return False
        return True

    def _meets_stability_expectation(
        self,
        step: DisplayVisualQcStep,
        comparison: _ImageComparison | None,
    ) -> bool:
        """Return True when two consecutive candidate captures are stable enough to treat as the settled frame."""

        if not step.require_stable_frame:
            return True
        if comparison is None:
            return True
        max_ratio = float(
            getattr(self.config, "display_visual_qc_stable_changed_ratio", _DEFAULT_STABLE_CHANGED_RATIO)
            or _DEFAULT_STABLE_CHANGED_RATIO
        )
        if comparison.changed_ratio <= max_ratio:
            return True
        max_dhash_distance = int(
            getattr(self.config, "display_visual_qc_stable_max_dhash_distance", _DEFAULT_STABLE_DHASH_DISTANCE)
            or _DEFAULT_STABLE_DHASH_DISTANCE
        )
        return comparison.dhash_distance is not None and comparison.dhash_distance <= max_dhash_distance

    def _capture_with_grim(self, path: Path) -> None:
        """Capture one Wayland screenshot with ``grim`` via an atomic temp file."""

        if self.config.display_driver != "hdmi_wayland":
            raise RuntimeError(
                "Display visual QC currently requires the hdmi_wayland backend because it captures the visible compositor surface."
            )
        grim_path = shutil.which("grim")
        if not grim_path:
            raise RuntimeError("`grim` is required for display visual QC but was not found in PATH.")
        path.parent.mkdir(parents=True, exist_ok=True)
        output_name = self._capture_output_name()
        scale_factor = self._grim_scale_factor(output_name=output_name)

        command = [grim_path]
        if scale_factor is not None:
            command.extend(["-s", str(scale_factor)])
        if output_name:
            command.extend(["-o", output_name])

        temp_path = self._temporary_artifact_path(path)
        command.append(str(temp_path))

        env = self._build_wayland_subprocess_env()
        try:
            subprocess.run(
                command,
                check=True,
                timeout=_SCREENSHOT_TIMEOUT_S,
                env=env,
            )
            self._validate_image_file(temp_path)
            self._replace_file(temp_path, path)
        finally:
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()

    def _capture_output_name(self) -> str | None:
        """Return the preferred Wayland output name, if configured."""

        for attr in (
            "display_visual_qc_output_name",
            "display_wayland_output_name",
            "display_wayland_output",
            "display_output_name",
        ):
            value = getattr(self.config, attr, None)
            if value:
                return str(value)
        return None

    def _grim_scale_factor(self, *, output_name: str | None) -> int | None:
        """Return the grim scale factor for deterministic captures."""

        value = getattr(self.config, "display_visual_qc_grim_scale", None)
        # BREAKING: when an output is explicitly pinned and no scale is configured, captures default to 1x
        # instead of grim's "highest output scale" behavior so artifact dimensions stay stable on multi-output desks.
        if value in (None, ""):
            return 1 if output_name else None
        scale = int(value)
        return max(1, scale)

    def _build_wayland_subprocess_env(self) -> dict[str, str]:
        """Build a subprocess environment for grim without leaving global os.environ mutated."""

        with _WAYLAND_ENV_LOCK:
            original_env = os.environ.copy()
            try:
                apply_wayland_environment(
                    self.config.display_wayland_display,
                    configured_runtime_dir=self.config.display_wayland_runtime_dir,
                )
                return os.environ.copy()
            finally:
                os.environ.clear()
                os.environ.update(original_env)

    def _build_diff_metrics(
        self,
        captures: Sequence[DisplayVisualQcCapture],
        *,
        steps: Sequence[DisplayVisualQcStep],
        output_dir: Path,
    ) -> tuple[DisplayVisualQcDiffMetric, ...]:
        """Compute image-diff metrics and save one annotated diff board per transition."""

        step_by_key = {step.key: step for step in steps}
        metrics: list[DisplayVisualQcDiffMetric] = []
        for index, (left, right) in enumerate(zip(captures, captures[1:]), start=1):
            step = step_by_key.get(right.key)
            ignore_regions = step.ignore_regions if step is not None else ()
            comparison = self._compare_paths(left.image_path, right.image_path, ignore_regions=ignore_regions)
            diff_path = output_dir / f"diff_{index:02d}_{_slug(left.key)}__{_slug(right.key)}.png"
            self._build_diff_board(
                left.image_path,
                right.image_path,
                comparison=comparison,
                output_path=diff_path,
            )
            metrics.append(
                DisplayVisualQcDiffMetric(
                    from_key=left.key,
                    to_key=right.key,
                    diff_image_path=str(diff_path),
                    changed_pixels=comparison.changed_pixels,
                    changed_ratio=comparison.changed_ratio,
                    bbox=comparison.bbox,
                    ssim=comparison.ssim,
                    psnr=comparison.psnr,
                    mean_abs_diff=comparison.mean_abs_diff,
                    max_abs_diff=comparison.max_abs_diff,
                    dhash_distance=comparison.dhash_distance,
                    ignored_regions=comparison.ignored_regions,
                    threshold=_DIFF_THRESHOLD,
                )
            )
        return tuple(metrics)

    def _difference_hash(self, image: Image.Image, *, hash_size: int = _DHASH_SIZE) -> int:
        """Return a compact difference hash for one image."""

        grayscale = image.convert("L").resize((hash_size + 1, hash_size), _RESAMPLE_BILINEAR)
        pixels = list(grayscale.getdata())
        value = 0
        bit_index = 0
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for column in range(hash_size):
                if pixels[row_start + column] > pixels[row_start + column + 1]:
                    value |= 1 << bit_index
                bit_index += 1
        return value

    def _metric_image(self, image: Image.Image) -> Image.Image:
        """Return one normalized RGB image view used for comparisons."""

        return image.convert("RGB")

    def _downsample_for_metrics(self, image: Image.Image) -> Image.Image:
        """Downsample large images before perceptual metrics to keep Pi 4 runtime bounded."""

        max_edge = int(
            getattr(self.config, "display_visual_qc_metric_max_edge", _DEFAULT_METRIC_MAX_EDGE)
            or _DEFAULT_METRIC_MAX_EDGE
        )
        if max_edge <= 0:
            return image.copy()
        width, height = image.size
        longest = max(width, height)
        if longest <= max_edge:
            return image.copy()
        scale = max_edge / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return image.resize(new_size, _RESAMPLE_BILINEAR)

    def _apply_ignore_regions(self, image: Image.Image, regions: Sequence[Rect]) -> Image.Image:
        """Return a copy of the image with dynamic ignore regions blanked out."""

        if not regions:
            return image
        masked = image.copy()
        draw = ImageDraw.Draw(masked)
        for region in regions:
            clamped = _clamp_rect(region, width=masked.width, height=masked.height)
            if clamped is None:
                continue
            draw.rectangle(clamped, fill=(0, 0, 0))
        return masked

    def _compare_paths(
        self,
        left_path: str | Path,
        right_path: str | Path,
        *,
        ignore_regions: Sequence[Rect] = (),
    ) -> _ImageComparison:
        """Return thresholded image-diff metrics between two screenshots."""

        with Image.open(left_path) as left_original, Image.open(right_path) as right_original:
            left = self._metric_image(left_original)
            right = self._metric_image(right_original)

        if left.size != right.size:
            raise RuntimeError(
                f"Display visual QC cannot compare images with different sizes: {left.size} vs {right.size}"
            )

        combined_ignore_regions = tuple(self._global_ignore_regions) + tuple(ignore_regions)
        if combined_ignore_regions:
            left = self._apply_ignore_regions(left, combined_ignore_regions)
            right = self._apply_ignore_regions(right, combined_ignore_regions)

        diff = ImageChops.difference(left, right)

        ssim_value: float | None = None
        psnr_value: float | None = None
        mean_abs_diff = 0.0
        max_abs_diff = 0
        threshold_bbox: Rect | None = None
        changed_pixels = 0
        dhash_distance: int | None = None

        if np is not None:
            left_np = np.asarray(left, dtype=np.uint8)
            right_np = np.asarray(right, dtype=np.uint8)
            abs_diff = np.abs(left_np.astype(np.int16) - right_np.astype(np.int16))
            per_pixel_change = abs_diff.max(axis=2)
            changed_mask = per_pixel_change >= _DIFF_THRESHOLD
            changed_pixels = int(np.count_nonzero(changed_mask))
            total_pixels = max(1, left.width * left.height)
            changed_ratio = changed_pixels / float(total_pixels)
            if changed_pixels > 0:
                rows, columns = np.where(changed_mask)
                threshold_bbox = (
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                )
            mean_abs_diff = float(abs_diff.mean())
            max_abs_diff = int(abs_diff.max())
            left_hash = self._difference_hash(left)
            right_hash = self._difference_hash(right)
            dhash_distance = _hash_distance(left_hash, right_hash)

            if peak_signal_noise_ratio is not None:
                try:
                    psnr_value = float(peak_signal_noise_ratio(left_np, right_np, data_range=255))
                except Exception:
                    psnr_value = None
            if structural_similarity is not None:
                try:
                    left_small = self._downsample_for_metrics(left).convert("L")
                    right_small = self._downsample_for_metrics(right).convert("L")
                    left_small_np = np.asarray(left_small, dtype=np.uint8)
                    right_small_np = np.asarray(right_small, dtype=np.uint8)
                    ssim_value = float(
                        structural_similarity(
                            left_small_np,
                            right_small_np,
                            data_range=255,
                            full=False,
                        )
                    )
                except Exception:
                    ssim_value = None
            return _ImageComparison(
                changed_pixels=changed_pixels,
                changed_ratio=changed_ratio,
                bbox=threshold_bbox,
                mean_abs_diff=mean_abs_diff,
                max_abs_diff=max_abs_diff,
                ssim=ssim_value,
                psnr=psnr_value,
                dhash_distance=dhash_distance,
                width=left.width,
                height=left.height,
                ignored_regions=tuple(
                    rect
                    for rect in (_clamp_rect(region, width=left.width, height=left.height) for region in combined_ignore_regions)
                    if rect is not None
                ),
            )

        thresholded = diff.convert("L").point(lambda value: 255 if value >= _DIFF_THRESHOLD else 0, mode="L")
        hist = thresholded.histogram()
        changed_pixels = int(hist[255]) if len(hist) > 255 else 0
        total_pixels = max(1, left.width * left.height)
        changed_ratio = changed_pixels / float(total_pixels)
        threshold_bbox = thresholded.getbbox()
        mean_abs_diff = float(sum(index * count for index, count in enumerate(diff.convert("L").histogram())) / total_pixels)
        max_abs_diff = int(diff.convert("L").getextrema()[1])
        dhash_distance = _hash_distance(self._difference_hash(left), self._difference_hash(right))
        return _ImageComparison(
            changed_pixels=changed_pixels,
            changed_ratio=changed_ratio,
            bbox=tuple(int(value) for value in threshold_bbox) if threshold_bbox is not None else None,
            mean_abs_diff=mean_abs_diff,
            max_abs_diff=max_abs_diff,
            ssim=None,
            psnr=None,
            dhash_distance=dhash_distance,
            width=left.width,
            height=left.height,
            ignored_regions=tuple(
                rect
                for rect in (_clamp_rect(region, width=left.width, height=left.height) for region in combined_ignore_regions)
                if rect is not None
            ),
        )

    def _count_changed_pixels(self, left_path: str | Path, right_path: str | Path) -> int:
        """Return the count of visually changed pixels between two screenshots."""

        return self._compare_paths(left_path, right_path).changed_pixels

    def _build_diff_board(
        self,
        left_path: str | Path,
        right_path: str | Path,
        *,
        comparison: _ImageComparison,
        output_path: Path,
    ) -> Path:
        """Render an annotated three-panel diff board (before / after / heatmap)."""

        font = ImageFont.load_default()
        with Image.open(left_path) as left_original, Image.open(right_path) as right_original:
            left = self._metric_image(left_original)
            right = self._metric_image(right_original)

        if comparison.ignored_regions:
            left = self._apply_ignore_regions(left, comparison.ignored_regions)
            right = self._apply_ignore_regions(right, comparison.ignored_regions)

        diff = ImageChops.difference(left, right)
        heatmap = ImageOps.autocontrast(diff)

        left_marked = left.copy()
        right_marked = right.copy()
        heatmap_marked = heatmap.copy()
        left_draw = ImageDraw.Draw(left_marked)
        right_draw = ImageDraw.Draw(right_marked)
        heatmap_draw = ImageDraw.Draw(heatmap_marked)

        if comparison.bbox is not None:
            left_draw.rectangle(comparison.bbox, outline=(255, 196, 0), width=3)
            right_draw.rectangle(comparison.bbox, outline=(255, 196, 0), width=3)
            heatmap_draw.rectangle(comparison.bbox, outline=(255, 255, 255), width=3)
        for region in comparison.ignored_regions:
            left_draw.rectangle(region, outline=(120, 120, 120), width=2)
            right_draw.rectangle(region, outline=(120, 120, 120), width=2)
            heatmap_draw.rectangle(region, outline=(120, 120, 120), width=2)

        tile_size = (360, 216)
        tiles = [
            ("Before", ImageOps.contain(left_marked, tile_size)),
            ("After", ImageOps.contain(right_marked, tile_size)),
            ("Heatmap", ImageOps.contain(heatmap_marked, tile_size)),
        ]
        margin = 18
        header_h = 74
        footer_h = 70
        board_width = (len(tiles) * tile_size[0]) + ((len(tiles) + 1) * margin)
        board_height = header_h + footer_h + tile_size[1] + (2 * margin)
        board = Image.new("RGB", (board_width, board_height), (14, 14, 16))
        draw = ImageDraw.Draw(board)

        metrics_line = [
            f"changed={comparison.changed_pixels} ({comparison.changed_ratio:.4%})",
            f"mad={comparison.mean_abs_diff:.2f}",
            f"max={comparison.max_abs_diff}",
        ]
        if comparison.ssim is not None:
            metrics_line.append(f"ssim={comparison.ssim:.5f}")
        if comparison.psnr is not None:
            metrics_line.append(f"psnr={comparison.psnr:.2f}")
        if comparison.dhash_distance is not None:
            metrics_line.append(f"dhash={comparison.dhash_distance}")
        draw.text((margin, 18), "Visual diff", fill=(255, 255, 255), font=font)
        draw.text((margin, 38), " | ".join(metrics_line), fill=(180, 180, 180), font=font)

        for index, (label, tile) in enumerate(tiles):
            tile_x = margin + (index * (tile_size[0] + margin))
            tile_y = header_h
            paste_x = tile_x + ((tile_size[0] - tile.width) // 2)
            paste_y = tile_y + ((tile_size[1] - tile.height) // 2)
            board.paste(tile, (paste_x, paste_y))
            draw.text((tile_x, tile_y + tile_size[1] + 12), label, fill=(255, 255, 255), font=font)

        footer = [f"threshold={_DIFF_THRESHOLD}"]
        if comparison.bbox is not None:
            footer.append(f"bbox={comparison.bbox}")
        if comparison.ignored_regions:
            footer.append(f"ignored_regions={len(comparison.ignored_regions)}")
        draw.text((margin, board_height - footer_h + 14), " | ".join(footer), fill=(140, 140, 140), font=font)
        self._write_image_atomic(output_path, board)
        return output_path

    def _build_contact_sheet(
        self,
        captures: Sequence[DisplayVisualQcCapture],
        *,
        output_dir: Path,
    ) -> Path:
        """Render one compact contact sheet over the captured scene sequence."""

        font = ImageFont.load_default()
        columns = _CONTACT_SHEET_COLUMNS
        rows = max(1, (len(captures) + columns - 1) // columns)
        margin = 20
        caption_h = 54
        cell_w = _THUMBNAIL_SIZE[0]
        cell_h = _THUMBNAIL_SIZE[1] + caption_h
        sheet = Image.new(
            "RGB",
            (
                (columns * cell_w) + ((columns + 1) * margin),
                (rows * cell_h) + ((rows + 1) * margin),
            ),
            (12, 12, 12),
        )
        draw = ImageDraw.Draw(sheet)
        for index, capture in enumerate(captures):
            column = index % columns
            row = index // columns
            x = margin + (column * (cell_w + margin))
            y = margin + (row * (cell_h + margin))
            with Image.open(capture.image_path) as original:
                thumbnail = ImageOps.contain(original.convert("RGB"), _THUMBNAIL_SIZE)
            thumb_x = x + ((cell_w - thumbnail.width) // 2)
            thumb_y = y
            sheet.paste(thumbnail, (thumb_x, thumb_y))
            text_y = y + _THUMBNAIL_SIZE[1] + 8
            draw.text((x, text_y), f"{index + 1}. {capture.label}", fill=(255, 255, 255), font=font)
            draw.text((x, text_y + 16), capture.key.replace("_", " "), fill=(178, 178, 178), font=font)
            draw.text((x, text_y + 32), f"{capture.width}x{capture.height}", fill=(150, 150, 150), font=font)
        path = output_dir / "visual_qc_contact_sheet.png"
        self._write_image_atomic(path, sheet)
        return path

    def _resolve_global_ignore_regions(self) -> tuple[Rect, ...]:
        """Return the global ignore regions configured for dynamic UI zones."""

        return _parse_rects(getattr(self.config, "display_visual_qc_ignore_regions", None))

    def _temporary_artifact_path(self, final_path: Path) -> Path:
        """Return a temp path in the same directory as the final artifact."""

        final_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=str(final_path.parent),
            prefix=f".{final_path.stem}.",
            suffix=final_path.suffix or ".tmp",
        )
        os.close(fd)
        temp_path = Path(temp_name)
        self._chmod_private_file(temp_path)
        return temp_path

    def _replace_file(self, source_path: Path, destination_path: Path) -> None:
        """Atomically replace the destination file with the source file."""

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_path, destination_path)
        self._chmod_private_file(destination_path)

    def _write_text_atomic(self, path: Path, content: str) -> None:
        """Write UTF-8 text atomically and with private permissions."""

        temp_path = self._temporary_artifact_path(path)
        try:
            temp_path.write_text(content, encoding="utf-8")
            self._replace_file(temp_path, path)
        finally:
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()

    def _write_image_atomic(self, path: Path, image: Image.Image) -> None:
        """Write an image atomically and with private permissions."""

        temp_path = self._temporary_artifact_path(path)
        try:
            image.save(temp_path)
            self._replace_file(temp_path, path)
        finally:
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()

    def _validate_image_file(self, path: Path) -> None:
        """Fail if the image cannot be decoded as a valid raster image."""

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise RuntimeError(f"Visual QC capture has invalid dimensions: {path}")

    def _ensure_private_directory(self, path: Path) -> None:
        """Create the artifact directory and tighten its permissions."""

        path.mkdir(parents=True, exist_ok=True, mode=_DEFAULT_SECURE_DIR_MODE)
        self._chmod_private_directory(path)

    def _chmod_private_directory(self, path: Path) -> None:
        """Best-effort chmod for the artifact directory."""

        with contextlib.suppress(PermissionError, NotImplementedError, OSError):
            os.chmod(path, int(getattr(self.config, "display_visual_qc_directory_mode", _DEFAULT_SECURE_DIR_MODE) or _DEFAULT_SECURE_DIR_MODE))

    def _chmod_private_file(self, path: Path) -> None:
        """Best-effort chmod for artifact files."""

        with contextlib.suppress(PermissionError, NotImplementedError, OSError):
            os.chmod(path, int(getattr(self.config, "display_visual_qc_file_mode", _DEFAULT_SECURE_FILE_MODE) or _DEFAULT_SECURE_FILE_MODE))


def build_visual_qc_report_payload(
    result: DisplayVisualQcRunResult,
    *,
    title: str | None = None,
    task_id: str | None = None,
) -> dict[str, object]:
    """Build one report-tool payload for a completed visual-QC run."""

    captures = list(result.captures)
    diffs = list(result.diffs)
    max_changed_ratio = max((diff.changed_ratio for diff in diffs), default=0.0)
    min_ssim = min((diff.ssim for diff in diffs if diff.ssim is not None), default=None)
    nonzero_diffs = sum(1 for diff in diffs if diff.changed_pixels > 0)
    capture_refs = [capture.attachment_name() for capture in captures]
    diff_refs = [diff.attachment_name() for diff in diffs]
    links = [
        "script:src/twinr/display/visual_qc.py",
        "script:hardware/display/run_visual_qc.py",
        "script:src/twinr/display/presentation_cues.py",
        "script:src/twinr/display/hdmi_presentation_graph.py",
    ]
    if task_id:
        links.append(f"task:{task_id}")

    dependency_metadata = result.metadata.get("dependencies") if isinstance(result.metadata, dict) else None
    dependency_line = None
    if isinstance(dependency_metadata, dict):
        dependency_line = ", ".join(f"{key}={value}" for key, value in sorted(dependency_metadata.items()))

    summary_lines = [
        "The HDMI visual QC runner captured the live Twinr surface across idle, external face reaction, presentation morph, fullscreen focus, and restored-home scenes.",
        "This report stores the screenshots, annotated diff boards, contact sheet, and machine-readable metrics so display acceptance is reproducible on the real Pi instead of relying on ad-hoc screenshots.",
    ]
    if dependency_line:
        summary_lines.append(f"Perceptual-metric dependency status for this run: {dependency_line}.")

    insights = [
        "The scene set is producer-driven: face cues and presentation cards are applied through the same public controllers that future capabilities will use, so the QC path validates real integration contracts rather than a test-only renderer hook.",
        "Transition evidence is now both thresholded and perceptual. Each scene pair stores changed-pixel metrics, annotated bounding boxes, and perceptual similarity metrics when available, which cuts false positives from tiny background drift.",
        "The restored-home step is explicitly compared back to the original idle capture, closing a blind spot where the old runner could pass even when the surface failed to reset to baseline.",
    ]
    if result.metadata.get("wayland_output_name"):
        insights.append(
            f"The capture backend was pinned to the configured Wayland output `{result.metadata['wayland_output_name']}`, improving determinism on multi-output Raspberry Pi setups."
        )

    kpis = [
        {
            "name": "captures_total",
            "value": len(captures),
            "unit": "captures",
            "baseline": len(captures),
            "delta": "+0",
            "explanation": "A full scene set requires all planned captures to succeed so the contact sheet and diff chain cover the visible lifecycle from idle through fullscreen presentation and back.",
        },
        {
            "name": "transition_pairs_with_change",
            "value": nonzero_diffs,
            "unit": "pairs",
            "baseline": len(diffs),
            "delta": f"{nonzero_diffs}/{len(diffs)}",
            "explanation": "Every expected change pair should show materially visible movement; this KPI guards against frozen surfaces or missing morph/face reactions.",
        },
        {
            "name": "max_changed_ratio",
            "value": round(max_changed_ratio, 6),
            "unit": "ratio",
            "baseline": 0.0,
            "delta": f"+{max_changed_ratio:.6f}",
            "explanation": "The largest changed-pixel ratio across transitions gives a compact signal for how much of the screen moved during the most substantial morph without depending on log output alone.",
        },
    ]
    if min_ssim is not None:
        kpis.append(
            {
                "name": "min_ssim",
                "value": round(float(min_ssim), 6),
                "unit": "score",
                "baseline": 1.0,
                "delta": f"{float(min_ssim) - 1.0:+.6f}",
                "explanation": "The minimum SSIM observed across transitions is a perceptual similarity floor, useful when pixel counts alone are too brittle for visually rich scenes.",
            }
        )

    return {
        "title": title or "HDMI display visual QC run",
        "kind": "ops",
        "tags": ["display", "visual-qc", "hdmi", "ops"],
        "summary": summary_lines,
        "insights": insights,
        "context": {
            "objective": "Prove that the visible HDMI Twinr surface can be validated reproducibly through screenshot-backed visual QC instead of manual one-off screenshot checks.",
            "scope": "This run covers the default idle home screen, one external face-expression reaction, one prioritized multi-card presentation morph, the fullscreen focused card, and the restored idle state on the live HDMI surface.",
            "methodology": "Drive the public face and presentation controllers through a bounded deterministic scene plan, capture the visible Wayland frame with grim, compute thresholded and perceptual diff metrics, and persist screenshots plus summaries as durable report assets.",
            "data_window": f"Visual QC run generated at {result.generated_at}.",
        },
        "evidence": [
            {
                "kind": "attachment",
                "ref": f"attachment:{Path(result.contact_sheet_path).name}",
                "note": "Contact sheet of the captured scene sequence for quick human review.",
            },
            {
                "kind": "attachment",
                "ref": f"attachment:{Path(result.summary_path).name}",
                "note": "Machine-readable JSON summary with all captures, dimensions, thresholds, and diff metrics.",
            },
            {
                "kind": "attachment",
                "ref": f"attachment:{capture_refs[0] if capture_refs else Path(result.contact_sheet_path).name}",
                "note": "Representative source screenshot from the captured HDMI scene set.",
            },
            {
                "kind": "attachment",
                "ref": f"attachment:{diff_refs[0] if diff_refs else Path(result.summary_path).name}",
                "note": "Representative annotated diff board showing visible change between consecutive captures.",
            },
        ],
        "links": links,
        "results": {
            "narrative": "The visual QC runner completed the HDMI scene set, stored screenshots plus annotated diff boards, and produced stable metrics showing real visual changes at every expected transition while verifying that restored-home converged back to the initial idle baseline.",
            "kpis": kpis,
            "tests": [
                {
                    "name": "HDMI display visual QC scene set",
                    "dataset_or_suite": "Twinr live display visual QC",
                    "status": "pass",
                    "sample_size_n": len(captures),
                    "kpi_refs": [kpi["name"] for kpi in kpis],
                    "notes": "The suite uses the public face-expression and presentation controllers, then captures the visible Wayland surface with grim.",
                }
            ],
        },
        "limitations": [
            "The current runner validates the visible HDMI scene set only; it does not yet exercise future webview or multi-surface content types.",
        ],
        "risks": [
            "A compositor or session change on the Pi can still invalidate screenshot capture even when the Twinr renderer itself is healthy, so the runner depends on the same Wayland environment assumptions as the live HDMI backend.",
        ],
        "next_actions": [
            "Add capability-specific scene sets once approved webview or richer content surfaces land on HDMI.",
            "Wire the visual QC runner into a bounded ops self-test so the screenshot pass becomes part of post-deploy acceptance.",
            "Add app-specific ignore regions or golden baselines for the highest-risk dynamic areas once the design language stabilizes further.",
        ],
        "repro": [
            "python3 hardware/display/run_visual_qc.py --env-file .env",
        ],
    }


def build_visual_qc_report_markdown(result: DisplayVisualQcRunResult) -> str:
    """Render one Markdown report body with inline screenshots and diff images."""

    lines = [
        "# HDMI Display Visual QC",
        "",
        "This report captures the bounded Twinr HDMI scene set from the live visible surface.",
        "",
        f"![Contact sheet](./assets/{Path(result.contact_sheet_path).name})",
        "",
        "## Captures",
        "",
    ]
    for capture in result.captures:
        lines.extend(
            [
                f"### {capture.label}",
                "",
                capture.description,
                "",
                f"- Scene key: `{capture.key}`",
                f"- Delay before capture: `{capture.delay_s:.2f}s`",
                f"- Captured size: `{capture.width}x{capture.height}`",
                "",
                f"![{capture.label}](./assets/{capture.attachment_name()})",
                "",
            ]
        )
    if result.diffs:
        lines.extend(["## Transition Diffs", ""])
    for diff in result.diffs:
        bbox = "none" if diff.bbox is None else f"`{diff.bbox}`"
        lines.extend(
            [
                f"### {diff.from_key} -> {diff.to_key}",
                "",
                f"- Changed pixels: `{diff.changed_pixels}`",
                f"- Changed ratio: `{diff.changed_ratio:.6f}`",
                f"- Bounding box: {bbox}",
            ]
        )
        if diff.ssim is not None:
            lines.append(f"- SSIM: `{diff.ssim:.6f}`")
        if diff.psnr is not None:
            lines.append(f"- PSNR: `{diff.psnr:.3f}`")
        if diff.mean_abs_diff is not None:
            lines.append(f"- Mean abs diff: `{diff.mean_abs_diff:.3f}`")
        if diff.max_abs_diff is not None:
            lines.append(f"- Max abs diff: `{diff.max_abs_diff}`")
        if diff.dhash_distance is not None:
            lines.append(f"- dHash distance: `{diff.dhash_distance}`")
        if diff.ignored_regions:
            lines.append(f"- Ignored regions: `{diff.ignored_regions}`")
        lines.extend(
            [
                "",
                f"![{diff.from_key} to {diff.to_key}](./assets/{diff.attachment_name()})",
                "",
            ]
        )
    lines.extend(
        [
            "## Machine Summary",
            "",
            f"- Summary JSON: `./assets/{Path(result.summary_path).name}`",
            f"- Sample image: `./assets/{Path(result.sample_image_path).name}`",
            "",
        ]
    )
    return "\n".join(lines)
