"""Run bounded visual QC for Twinr's visible HDMI surface.

This module drives a small deterministic HDMI scene set through the producer-
facing face and presentation controllers, captures real screenshots of the
visible Wayland surface, computes image-diff metrics, and emits report-ready
artifacts. The goal is to make display acceptance reproducible instead of
relying on ad-hoc screenshots in `/tmp`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

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
        }


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
        self._sleep_func = sleep_func or time.sleep

    def run(
        self,
        output_dir: Path,
        *,
        emit: Callable[[str], None] | None = None,
        image_path: str | None = None,
    ) -> DisplayVisualQcRunResult:
        """Execute the default HDMI scene set and persist a full artifact bundle."""

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        emit = emit or (lambda _: None)
        sample_image_path = self._resolve_sample_image(output_dir, requested_path=image_path)
        steps = self.default_steps(sample_image_path=sample_image_path)
        captures: list[DisplayVisualQcCapture] = []
        try:
            for index, step in enumerate(steps, start=1):
                emit(f"visual_qc_step={step.key} action={step.action}")
                self._apply_step(step)
                self._sleep_func(max(0.0, float(step.delay_s)))
                capture_name = f"scene_{index:02d}_{_slug(step.key)}.png"
                capture_path = output_dir / capture_name
                capture = self._capture_step(
                    step,
                    capture_path=capture_path,
                    previous_capture=captures[-1] if captures else None,
                    emit=emit,
                )
                captures.append(capture)
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
        diffs = self._build_diff_metrics(captures, output_dir=output_dir)
        no_change_pairs = [f"{diff.from_key}->{diff.to_key}" for diff in diffs if diff.changed_pixels <= 0]
        if no_change_pairs:
            raise RuntimeError(
                "Display visual QC found scene transitions without visible change: "
                + ", ".join(no_change_pairs)
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
        )
        summary_path.write_text(json.dumps(result.to_summary_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
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
            ),
            DisplayVisualQcStep(
                key="face_react",
                label="Face reaction",
                description="External face cue should shift gaze and smile without changing the status panel.",
                action="face_expression",
                delay_s=0.35,
                hold_seconds=_DEFAULT_FACE_HOLD_S,
                expect_change_from_previous=True,
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
            ),
            DisplayVisualQcStep(
                key="restored_home",
                label="Restored home",
                description="Clearing both cue layers should return the display to the idle home surface.",
                action="idle",
                delay_s=0.40,
                expect_change_from_previous=True,
            ),
        )

    def _resolve_sample_image(self, output_dir: Path, *, requested_path: str | None) -> Path:
        """Return the QC sample image path, generating one when needed."""

        if requested_path:
            candidate = Path(requested_path).expanduser()
            if not candidate.is_absolute():
                candidate = Path(self.config.project_root).expanduser().resolve() / candidate
            resolved = candidate.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Display visual QC image was not found: {resolved}")
            return resolved
        path = output_dir / "sample_family_photo.png"
        self._create_sample_image(path)
        return path

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
        image.save(path)

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
        """Capture one screenshot into ``path`` and fail if it is missing."""

        self._screenshot_func(path)
        if not path.exists():
            raise RuntimeError(f"Visual QC capture did not produce a screenshot: {path}")

    def _capture_step(
        self,
        step: DisplayVisualQcStep,
        *,
        capture_path: Path,
        previous_capture: DisplayVisualQcCapture | None,
        emit: Callable[[str], None],
    ) -> DisplayVisualQcCapture:
        """Capture one scene step and retry when a visible change is expected but absent."""

        max_attempts = 3 if step.expect_change_from_previous and previous_capture is not None else 1
        for attempt in range(1, max_attempts + 1):
            self._capture(capture_path)
            with Image.open(capture_path) as image:
                width, height = image.size
            capture = DisplayVisualQcCapture(
                key=step.key,
                label=step.label,
                description=step.description,
                image_path=str(capture_path),
                delay_s=float(step.delay_s),
                captured_at=_utc_now_iso(),
                width=width,
                height=height,
            )
            if previous_capture is None or not step.expect_change_from_previous:
                return capture
            if self._count_changed_pixels(previous_capture.image_path, capture.image_path) > 0:
                return capture
            if attempt >= max_attempts:
                return capture
            emit(f"visual_qc_retry={step.key} attempt={attempt + 1}")
            self._sleep_func(self._retry_delay_s())
        raise RuntimeError(f"Display visual QC exhausted retries for scene `{step.key}`.")

    def _retry_delay_s(self) -> float:
        """Return the retry delay used when the visible surface has not updated yet."""

        poll_interval = float(getattr(self.config, "display_poll_interval_s", 0.5) or 0.5)
        return max(0.25, poll_interval + 0.15)

    def _capture_with_grim(self, path: Path) -> None:
        """Capture one Wayland screenshot with ``grim``."""

        if self.config.display_driver != "hdmi_wayland":
            raise RuntimeError(
                "Display visual QC currently requires the hdmi_wayland backend because it captures the visible compositor surface."
            )
        grim_path = shutil.which("grim")
        if not grim_path:
            raise RuntimeError("`grim` is required for display visual QC but was not found in PATH.")
        apply_wayland_environment(
            self.config.display_wayland_display,
            configured_runtime_dir=self.config.display_wayland_runtime_dir,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [grim_path, str(path)],
            check=True,
            timeout=_SCREENSHOT_TIMEOUT_S,
        )

    def _build_diff_metrics(
        self,
        captures: Sequence[DisplayVisualQcCapture],
        *,
        output_dir: Path,
    ) -> tuple[DisplayVisualQcDiffMetric, ...]:
        """Compute image-diff metrics and save one diff image per transition."""

        metrics: list[DisplayVisualQcDiffMetric] = []
        for index, (left, right) in enumerate(zip(captures, captures[1:]), start=1):
            with Image.open(left.image_path) as first, Image.open(right.image_path) as second:
                diff = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
                bbox = diff.getbbox()
                highlighted = ImageOps.autocontrast(diff)
                diff_path = output_dir / f"diff_{index:02d}_{_slug(left.key)}__{_slug(right.key)}.png"
                highlighted.save(diff_path)
                changed_pixels = sum(1 for pixel in diff.getdata() if max(pixel) >= _DIFF_THRESHOLD)
                total_pixels = max(1, diff.width * diff.height)
            metrics.append(
                DisplayVisualQcDiffMetric(
                    from_key=left.key,
                    to_key=right.key,
                    diff_image_path=str(diff_path),
                    changed_pixels=changed_pixels,
                    changed_ratio=changed_pixels / total_pixels,
                    bbox=tuple(int(value) for value in bbox) if bbox is not None else None,
                )
            )
        return tuple(metrics)

    def _count_changed_pixels(self, left_path: str | Path, right_path: str | Path) -> int:
        """Return the count of visually changed pixels between two screenshots."""

        with Image.open(left_path) as left, Image.open(right_path) as right:
            diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
            return sum(1 for pixel in diff.getdata() if max(pixel) >= _DIFF_THRESHOLD)

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
        sheet.save(path)
        return path


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
    return {
        "title": title or "HDMI display visual QC run",
        "kind": "ops",
        "tags": ["display", "visual-qc", "hdmi", "ops"],
        "summary": [
            "The HDMI visual QC runner captured the live Twinr surface across idle, external face reaction, presentation morph, fullscreen focus, and restored-home scenes.",
            "This report stores the screenshots, diff heatmaps, contact sheet, and machine-readable metrics so display acceptance is reproducible on the real Pi instead of relying on ad-hoc screenshots.",
        ],
        "insights": [
            "The scene set is producer-driven: face cues and presentation cards are applied through the same public controllers that future capabilities will use, so the QC path validates real integration contracts rather than a test-only renderer hook.",
            "Transition evidence is image-based and sequential. Each scene pair produces a diff heatmap plus changed-pixel metrics, which makes regressions visible even when telemetry alone still says the display loop is healthy.",
            "The artifact bundle now lands in the report system with a contact sheet and JSON summary, so operators and future agents can compare runs without hunting through `/tmp` or the system journal.",
        ],
        "context": {
            "objective": "Prove that the visible HDMI Twinr surface can be validated reproducibly through screenshot-backed visual QC instead of manual one-off screenshot checks.",
            "scope": "This run covers the default idle home screen, one external face-expression reaction, one prioritized multi-card presentation morph, the fullscreen focused card, and the restored idle state on the live HDMI surface.",
            "methodology": "Drive the public face and presentation controllers through a bounded deterministic scene plan, capture each visible Wayland frame with grim, compute per-transition diff metrics, and persist screenshots plus summaries as durable report assets.",
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
                "note": "Machine-readable JSON summary with all captures, dimensions, and diff metrics.",
            },
            {
                "kind": "attachment",
                "ref": f"attachment:{capture_refs[0] if capture_refs else Path(result.contact_sheet_path).name}",
                "note": "Representative source screenshot from the captured HDMI scene set.",
            },
            {
                "kind": "attachment",
                "ref": f"attachment:{diff_refs[0] if diff_refs else Path(result.summary_path).name}",
                "note": "Representative diff heatmap showing visible scene change between consecutive captures.",
            },
        ],
        "links": links,
        "results": {
            "narrative": "The visual QC runner completed the full HDMI scene set, stored screenshots plus diff heatmaps, and produced stable metrics showing real visual changes at every expected transition while keeping all captures at one consistent panel size.",
            "kpis": [
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
                    "explanation": "Every sequential scene pair should show visible pixel change; this KPI guards against frozen surfaces or missing morph/face reactions.",
                },
                {
                    "name": "max_changed_ratio",
                    "value": round(max_changed_ratio, 6),
                    "unit": "ratio",
                    "baseline": 0.0,
                    "delta": f"+{max_changed_ratio:.6f}",
                    "explanation": "The largest changed-pixel ratio across transitions gives a compact signal for how much of the screen moved during the most substantial morph without depending on log output alone.",
                },
            ],
            "tests": [
                {
                    "name": "HDMI display visual QC scene set",
                    "dataset_or_suite": "Twinr live display visual QC",
                    "status": "pass",
                    "sample_size_n": len(captures),
                    "kpi_refs": [
                        "captures_total",
                        "transition_pairs_with_change",
                        "max_changed_ratio",
                    ],
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
            "Add goldens or structural assertions for the highest-risk areas once the design language stabilizes further.",
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
