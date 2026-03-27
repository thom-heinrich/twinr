from datetime import datetime
from io import BytesIO
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.debug_signals import DisplayDebugSignal
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.face_expressions import (
    DisplayFaceBrowStyle,
    DisplayFaceGazeDirection,
    DisplayFaceMouthStyle,
)
from twinr.display.factory import create_display_adapter
from twinr.display.hdmi_fbdev import (
    FramebufferBitfield,
    FramebufferGeometry,
    HdmiFramebufferDisplay,
)
from twinr.display.hdmi_default_scene import HdmiDefaultSceneRenderer
from twinr.display.hdmi_default_scene import HdmiStatusPanelModel
from twinr.display.hdmi_wayland import HdmiWaylandDisplay
from twinr.display.presentation_cues import DisplayPresentationCardCue, DisplayPresentationCue
from twinr.display.service_connect_cues import DisplayServiceConnectCue
from twinr.display.wayland_env import apply_wayland_environment, resolve_wayland_socket


_QR_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aW7cAAAAASUVORK5CYII="


class _FakeFont:
    def __init__(self, size: int, *, bold: bool) -> None:
        self.size = size
        self.bold = bold


class _FakeSceneTools:
    def _font(self, size: int, *, bold: bool) -> _FakeFont:
        return _FakeFont(size, bold=bold)

    def _text_width(self, draw, text: str, *, font=None) -> int:
        del draw
        size = getattr(font, "size", 16)
        return len(text) * max(8, size // 2)

    def _text_height(self, draw, *, font=None) -> int:
        del draw
        return getattr(font, "size", 16)

    def _truncate_text(self, draw, text: str, *, max_width: int, font=None) -> str:
        del draw, max_width, font
        return text

    def _wrapped_lines(self, draw, lines, *, max_width: int, font, max_lines: int):
        del draw, max_width, font
        wrapped: list[str] = []
        for line in lines:
            if not line:
                continue
            parts = [part.strip() for part in str(line).split("|") if part.strip()]
            if not parts:
                parts = [str(line)]
            for part in parts:
                wrapped.append(part)
                if len(wrapped) >= max_lines:
                    return tuple(wrapped[:max_lines])
        return tuple(wrapped[:max_lines])

    def _normalise_text(self, value, *, fallback: str) -> str:
        compact = " ".join(str(value or "").split()).strip()
        return compact or fallback

    def _render_emoji_glyph(self, emoji: str, *, target_size: int):
        del emoji, target_size
        return None


class _RecordingDraw:
    def __init__(self) -> None:
        self.rounded_rectangles: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []

    def rounded_rectangle(self, box, *, radius, fill, outline=None, width=1) -> None:
        self.rounded_rectangles.append(
            {
                "box": box,
                "radius": radius,
                "fill": fill,
                "outline": outline,
                "width": width,
            }
        )

    def text(self, position, text, *, fill, font) -> None:
        self.text_calls.append(
            {
                "position": position,
                "text": text,
                "fill": fill,
                "font": font,
            }
        )


def _rgb565_geometry(*, width: int = 800, height: int = 480) -> FramebufferGeometry:
    return FramebufferGeometry(
        width=width,
        height=height,
        bits_per_pixel=16,
        line_length=width * 2,
        red=FramebufferBitfield(offset=11, length=5, msb_right=0),
        green=FramebufferBitfield(offset=5, length=6, msb_right=0),
        blue=FramebufferBitfield(offset=0, length=5, msb_right=0),
        transp=FramebufferBitfield(offset=0, length=0, msb_right=0),
    )


class HdmiFramebufferDisplayTests(unittest.TestCase):
    def make_display(
        self,
        *,
        layout_mode: str = "default",
        width: int = 800,
        height: int = 480,
    ) -> HdmiFramebufferDisplay:
        display = HdmiFramebufferDisplay(
            framebuffer_path=Path("/dev/null"),
            layout_mode=layout_mode,
        )
        display._geometry = _rgb565_geometry(width=width, height=height)
        return display

    def test_factory_selects_hdmi_backend(self) -> None:
        config = TwinrConfig(display_driver="hdmi_fbdev")
        sentinel = object()

        with mock.patch("twinr.display.factory.HdmiFramebufferDisplay.from_config", return_value=sentinel):
            adapter = create_display_adapter(config)

        self.assertIs(adapter, sentinel)

    def test_pack_framebuffer_bytes_encodes_rgb565_pixels(self) -> None:
        display = self.make_display()
        geometry = _rgb565_geometry(width=1, height=1)
        image = Image.new("RGBA", (1, 1), (255, 0, 0, 255))

        payload = display._pack_framebuffer_bytes(image, geometry)

        self.assertEqual(payload, b"\x00\xf8")

    def test_show_status_writes_bytes_into_framebuffer(self) -> None:
        display = self.make_display()
        framebuffer = BytesIO(b"\x00" * display.geometry.frame_size_bytes)

        with mock.patch.object(HdmiFramebufferDisplay, "_open_framebuffer", return_value=framebuffer):
            display.show_status(
                "waiting",
                headline="Bereit",
                state_fields=(
                    ("Status", "Waiting"),
                    ("Internet", "ok"),
                    ("AI", "ok"),
                    ("System", "ok"),
                    ("Zeit", "12:34"),
                ),
            )

        framebuffer.seek(0)
        self.assertNotEqual(framebuffer.read(), b"\x00" * display.geometry.frame_size_bytes)

    def test_render_status_image_draws_default_screen(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
        )

        image = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )

        self.assertEqual(image.size, display.canvas_size)
        dark_pixels = sum(1 for pixel in image.getdata() if max(pixel) <= 20)
        face_pixels = image.crop(scene.layout.face_box)
        panel_pixels = image.crop(scene.layout.panel_box)
        face_bright = sum(1 for pixel in face_pixels.getdata() if min(pixel) >= 220)
        panel_bright = sum(1 for pixel in panel_pixels.getdata() if min(pixel) >= 220)

        self.assertEqual(image.getpixel((10, 10)), (0, 0, 0))
        self.assertLessEqual(max(image.getpixel((500, 180))), 18)
        self.assertGreater(dark_pixels, 300000)
        self.assertGreater(face_bright, 9000)
        self.assertLess(panel_bright, 2000)

    def test_prompt_mode_panel_has_no_left_accent_bar_and_uses_one_large_text_scale(self) -> None:
        renderer = HdmiDefaultSceneRenderer(tools=_FakeSceneTools())
        draw = _RecordingDraw()
        panel = renderer._build_panel_model(
            reserve_bus=types.SimpleNamespace(
                service_connect_cue=None,
                ambient_impulse_cue=DisplayAmbientImpulseCue(
                    topic_key="ai companions",
                    headline="Denkst du, dass das heute kippt?",
                    body="Ich wuerde da gern kurz mit dir draufschauen.",
                    eyebrow="",
                    symbol="question",
                    accent="warm",
                    action="invite_follow_up",
                    attention_state="shared_thread",
                    source="test",
                    updated_at="2026-03-22T10:00:00+00:00",
                    expires_at="2026-03-22T10:20:00+00:00",
                ),
                owner="ambient_impulse",
            )
        )

        renderer._draw_status_panel(
            draw,
            box=(400, 80, 780, 430),
            panel=panel,
            compact=False,
        )

        self.assertTrue(panel.prompt_mode)
        self.assertEqual(len(draw.rounded_rectangles), 1)
        text_sizes = {call["font"].size for call in draw.text_calls}
        self.assertEqual(text_sizes, {31})
        self.assertTrue(any("Denkst du" in call["text"] for call in draw.text_calls))
        self.assertTrue(any("draufschauen" in call["text"] for call in draw.text_calls))

    def test_service_connect_panel_model_uses_qr_payload_and_service_copy(self) -> None:
        renderer = HdmiDefaultSceneRenderer(tools=_FakeSceneTools())

        panel = renderer._build_panel_model(
            reserve_bus=types.SimpleNamespace(
                service_connect_cue=DisplayServiceConnectCue(
                    service_id="whatsapp",
                    service_label="WhatsApp",
                    phase="qr",
                    summary="Scan the QR",
                    detail="Open Linked Devices on your phone.",
                    qr_image_data_url=_QR_DATA_URL,
                    accent="warm",
                ),
                ambient_impulse_cue=None,
                owner="service_connect",
            )
        )

        self.assertEqual(panel.eyebrow, "WHATSAPP")
        self.assertEqual(panel.headline, "Scan the QR")
        self.assertEqual(panel.helper_text, "Open Linked Devices on your phone.")
        self.assertEqual(panel.image_data_url, _QR_DATA_URL)
        self.assertEqual(panel.accent, "warm")

    def test_prompt_mode_panel_uses_available_height_for_long_copy(self) -> None:
        renderer = HdmiDefaultSceneRenderer(tools=_FakeSceneTools())
        draw = _RecordingDraw()
        panel = HdmiStatusPanelModel(
            eyebrow="",
            headline="H1|H2|H3|H4",
            helper_text="B1|B2|B3|B4",
            cards=(),
            prompt_mode=True,
        )

        renderer._draw_status_panel(
            draw,
            box=(400, 80, 780, 430),
            panel=panel,
            compact=False,
        )

        rendered_lines = tuple(call["text"] for call in draw.text_calls)
        self.assertEqual(rendered_lines, ("H1", "H2", "H3", "H4", "B1", "B2", "B3", "B4"))
        self.assertGreater(len(rendered_lines), 5)

    def test_prompt_mode_panel_shrinks_text_to_keep_copy_inside_shorter_box(self) -> None:
        renderer = HdmiDefaultSceneRenderer(tools=_FakeSceneTools())
        draw = _RecordingDraw()
        panel = HdmiStatusPanelModel(
            eyebrow="",
            headline="H1|H2",
            helper_text="B1|B2",
            cards=(),
            prompt_mode=True,
        )

        renderer._draw_status_panel(
            draw,
            box=(400, 80, 780, 250),
            panel=panel,
            compact=False,
        )

        rendered_lines = tuple(call["text"] for call in draw.text_calls)
        text_sizes = {call["font"].size for call in draw.text_calls}
        self.assertEqual(rendered_lines, ("H1", "H2", "B1", "B2"))
        self.assertEqual(text_sizes, {28})

    def test_prompt_mode_panel_clips_on_min_size_before_overflowing_box(self) -> None:
        renderer = HdmiDefaultSceneRenderer(tools=_FakeSceneTools())
        draw = _RecordingDraw()
        panel = HdmiStatusPanelModel(
            eyebrow="",
            headline="H1|H2",
            helper_text="B1|B2",
            cards=(),
            prompt_mode=True,
        )

        renderer._draw_status_panel(
            draw,
            box=(400, 80, 780, 190),
            panel=panel,
            compact=False,
        )

        rendered_lines = tuple(call["text"] for call in draw.text_calls)
        text_sizes = {call["font"].size for call in draw.text_calls}
        self.assertEqual(rendered_lines, ("H1", "H2", "B1"))
        self.assertEqual(text_sizes, {20})

    def test_render_status_image_hides_bottom_news_ticker_in_extended_view(self) -> None:
        display = self.make_display()
        base = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )
        ticked = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
            ticker_text="Tagesschau · Calm readable headline for seniors",
        )

        self.assertIsNone(ImageChops.difference(base, ticked).getbbox())

    def test_default_scene_does_not_reserve_bottom_band_for_hidden_ticker_text(self) -> None:
        display = self.make_display()
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )

        without_ticker = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=0,
        )
        with_ticker = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=0,
            ticker_text="Tagesschau · Headline",
        )

        self.assertFalse(without_ticker.layout.ticker_reserved)
        self.assertFalse(with_ticker.layout.ticker_reserved)
        self.assertEqual(without_ticker.layout.panel_box[3], with_ticker.layout.panel_box[3])
        self.assertEqual(without_ticker.layout.face_box[3], with_ticker.layout.face_box[3])

    def test_default_scene_keeps_noncompact_panel_when_ticker_text_is_hidden(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            ticker_text="Tagesschau · Calm readable headline for seniors",
        )

        self.assertFalse(scene.layout.compact_panel)

    def test_default_scene_waiting_face_animates_between_idle_frames(self) -> None:
        display = self.make_display()
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=0,
        )
        first = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=0,
        )
        second = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=5,
        )

        diff = ImageChops.difference(first.crop(scene.layout.face_box), second.crop(scene.layout.face_box))
        changed_pixels = sum(1 for pixel in diff.getdata() if max(pixel) >= 24)

        self.assertIsNotNone(diff.getbbox())
        self.assertGreater(changed_pixels, 600)

    def test_waiting_eye_geometry_stays_calm_across_idle_frames(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        states = [renderer._eye_state("waiting", frame, "left") for frame in range(12)]

        self.assertEqual({int(state["width"]) for state in states}, {56})
        self.assertEqual({int(state["height"]) for state in states}, {74})
        self.assertEqual({int(state["eye_shift_x"]) for state in states}, {0})
        self.assertEqual({bool(state["lid_arc"]) for state in states}, {False})
        self.assertLessEqual(max(abs(int(state["eye_shift_y"])) for state in states), 1)
        self.assertLessEqual(max(abs(int(state["highlight_dx"])) for state in states), 1)
        self.assertLessEqual(max(abs(int(state["highlight_dy"])) for state in states), 1)
        self.assertEqual(sum(1 for state in states if bool(state["blink"])), 1)

    def test_listening_and_processing_eyes_keep_stable_hdmi_geometry(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()

        listening = [renderer._eye_state("listening", frame, "left") for frame in range(6)]
        processing = [renderer._eye_state("processing", frame, "left") for frame in range(6)]

        self.assertEqual({int(state["width"]) for state in listening}, {56})
        self.assertEqual({int(state["eye_shift_x"]) for state in listening}, {0})
        self.assertEqual({bool(state["lid_arc"]) for state in listening}, {False})
        self.assertLessEqual(max(int(state["height"]) for state in listening) - min(int(state["height"]) for state in listening), 6)

        self.assertEqual({int(state["width"]) for state in processing}, {56})
        self.assertEqual({int(state["height"]) for state in processing}, {68})
        self.assertEqual({int(state["eye_shift_x"]) for state in processing}, {0})
        self.assertEqual({bool(state["lid_arc"]) for state in processing}, {False})

    def test_external_face_cue_overrides_gaze_brows_and_blink_without_eye_resize(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        base = renderer._eye_state("waiting", 0, "left")
        cue = DisplayFaceCue(gaze_x=2, gaze_y=-2, brows="inward_tilt", blink=True)

        cued = renderer._eye_state("waiting", 0, "left", face_cue=cue)

        self.assertEqual(int(base["width"]), int(cued["width"]))
        self.assertEqual(int(base["height"]), int(cued["height"]))
        self.assertEqual(int(cued["eye_shift_x"]), int(base["eye_shift_x"]) + 8)
        self.assertEqual(int(cued["eye_shift_y"]), -6)
        self.assertEqual(int(cued["highlight_dx"]), int(base["highlight_dx"]) + 12)
        self.assertEqual(int(cued["highlight_dy"]), int(base["highlight_dy"]) - 10)
        self.assertEqual(int(cued["brow_raise"]), 0)
        self.assertEqual(str(cued["brow_style"]), "inward_tilt")
        self.assertTrue(bool(cued["blink"]))

    def test_runtime_voice_quiet_face_cue_closes_waiting_eyes_with_soft_brows(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        quiet_cue = DisplayFaceCue(
            source="runtime_voice_quiet",
            head_dy=1,
            mouth="neutral",
            brows="soft",
            blink=True,
        )

        eye = renderer._eye_state("waiting", 0, "left", face_cue=quiet_cue)
        face_offset = renderer._face_offset("waiting", 0, face_cue=quiet_cue)

        self.assertTrue(bool(eye["blink"]))
        self.assertEqual(str(eye["brow_style"]), "soft")
        self.assertEqual(int(eye["brow_raise"]), -3)
        self.assertEqual(face_offset, (0, 2))

    def test_error_eye_baseline_no_longer_stares_off_to_the_side(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()

        states = [renderer._eye_state("error", frame, "left") for frame in range(6)]

        self.assertLessEqual(max(abs(int(state["highlight_dx"])) for state in states), 2)
        self.assertLessEqual(max(abs(int(state["highlight_dy"])) for state in states), 1)

    def test_error_scene_keeps_sad_expression_while_preserving_attention_gaze(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        scene = renderer.build_scene(
            width=800,
            height=480,
            status="error",
            headline="Check system",
            helper_text="Twinr prueft das System.",
            state_fields=(
                ("Status", "Check system"),
                ("Internet", "warn"),
            ),
            animation_frame=0,
            face_cue=DisplayFaceCue(gaze_x=2, mouth="speak", brows="raised", blink=True, head_dx=1),
        )

        self.assertIsNotNone(scene.face_cue)
        assert scene.face_cue is not None
        self.assertEqual(scene.face_cue.gaze_x, 2)
        self.assertEqual(scene.face_cue.head_dx, 1)
        self.assertTrue(scene.face_cue.blink)
        self.assertIsNone(scene.face_cue.mouth)
        self.assertIsNone(scene.face_cue.brows)

        eye = renderer._eye_state("error", 0, "left", face_cue=scene.face_cue)

        self.assertEqual(int(eye["highlight_dx"]), 16)
        self.assertEqual(int(eye["eye_shift_x"]), 12)

    def test_error_scene_freezes_random_eye_churn_when_directional_cue_is_active(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        cue = DisplayFaceCue(gaze_x=2, head_dx=1)

        first = renderer._eye_state("error", 0, "left", face_cue=cue)
        second = renderer._eye_state("error", 4, "left", face_cue=cue)

        self.assertEqual(int(first["highlight_dx"]), int(second["highlight_dx"]))
        self.assertEqual(int(first["eye_shift_x"]), int(second["eye_shift_x"]))
        self.assertEqual(bool(first["blink"]), bool(second["blink"]))

    def test_error_face_offset_uses_scaled_head_turn_and_no_error_bob_when_cued(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        cue = DisplayFaceCue(gaze_x=2, head_dx=2)

        first = renderer._face_offset("error", 0, face_cue=cue)
        second = renderer._face_offset("error", 4, face_cue=cue)

        self.assertEqual(first, (16, 0))
        self.assertEqual(second, (16, 0))

    def test_waiting_scene_freezes_idle_eye_churn_when_directional_cue_is_active(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        cue = DisplayFaceCue(gaze_x=2, head_dx=1)

        first = renderer._eye_state("waiting", 0, "left", face_cue=cue)
        second = renderer._eye_state("waiting", 8, "left", face_cue=cue)

        self.assertEqual(int(first["highlight_dx"]), int(second["highlight_dx"]))
        self.assertEqual(int(first["eye_shift_y"]), int(second["eye_shift_y"]))
        self.assertEqual(bool(first["blink"]), bool(second["blink"]))

    def test_waiting_face_offset_freezes_idle_bob_when_directional_cue_is_active(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        cue = DisplayFaceCue(gaze_x=2, head_dx=1)

        first = renderer._face_offset("waiting", 0, face_cue=cue)
        second = renderer._face_offset("waiting", 8, face_cue=cue)

        self.assertEqual(first, (5, 0))
        self.assertEqual(second, (5, 0))

    def test_waiting_face_region_changes_less_between_frames_when_directional_cue_is_active(self) -> None:
        display = self.make_display()
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )
        cue = DisplayFaceCue(gaze_x=2, head_dx=1)
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=0,
            face_cue=cue,
        )
        first = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=0,
            face_cue=cue,
        )
        second = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=8,
            face_cue=cue,
        )

        diff = ImageChops.difference(first.crop(scene.layout.face_box), second.crop(scene.layout.face_box))
        changed_pixels = sum(1 for pixel in diff.getdata() if max(pixel) >= 24)

        self.assertLess(changed_pixels, 350)

    def test_external_face_cue_supports_six_brow_styles(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        styles = (
            DisplayFaceBrowStyle.STRAIGHT,
            DisplayFaceBrowStyle.INWARD_TILT,
            DisplayFaceBrowStyle.OUTWARD_TILT,
            DisplayFaceBrowStyle.ROOF,
            DisplayFaceBrowStyle.RAISED,
            DisplayFaceBrowStyle.SOFT,
        )

        seen = {
            style.value: renderer._eye_state("waiting", 0, "left", face_cue=DisplayFaceCue(brows=style.value))
            for style in styles
        }

        self.assertEqual({str(state["brow_style"]) for state in seen.values()}, {style.value for style in styles})
        self.assertEqual(int(seen["raised"]["brow_raise"]), -11)
        self.assertEqual(int(seen["roof"]["brow_raise"]), -1)
        self.assertEqual(int(seen["soft"]["brow_raise"]), -3)

    def test_default_scene_face_cue_changes_only_face_region(self) -> None:
        display = self.make_display()
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )
        cue = DisplayFaceCue(gaze_x=2, mouth="scrunched", brows="roof", head_dx=1)
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=0,
            face_cue=cue,
        )
        first = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=0,
        )
        second = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=0,
            face_cue=cue,
        )

        face_diff = ImageChops.difference(first.crop(scene.layout.face_box), second.crop(scene.layout.face_box))
        panel_diff = ImageChops.difference(first.crop(scene.layout.panel_box), second.crop(scene.layout.panel_box))
        changed_face_pixels = sum(1 for pixel in face_diff.getdata() if max(pixel) >= 24)
        changed_panel_pixels = sum(1 for pixel in panel_diff.getdata() if max(pixel) >= 24)

        self.assertIsNotNone(face_diff.getbbox())
        self.assertGreater(changed_face_pixels, 600)
        self.assertEqual(changed_panel_pixels, 0)

    def test_expression_cue_mouth_and_gaze_values_render_without_panel_churn(self) -> None:
        display = self.make_display()
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )
        cue = DisplayFaceCue(
            gaze_x=DisplayFaceGazeDirection.DOWN_LEFT.axes()[0],
            gaze_y=DisplayFaceGazeDirection.DOWN_LEFT.axes()[1],
            mouth=DisplayFaceMouthStyle.THINKING.value,
            brows=DisplayFaceBrowStyle.OUTWARD_TILT.value,
        )
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=0,
            face_cue=cue,
        )
        base = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=0,
        )
        cued = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=0,
            face_cue=cue,
        )

        face_diff = ImageChops.difference(base.crop(scene.layout.face_box), cued.crop(scene.layout.face_box))
        panel_diff = ImageChops.difference(base.crop(scene.layout.panel_box), cued.crop(scene.layout.panel_box))

        self.assertIsNotNone(face_diff.getbbox())
        self.assertGreater(sum(1 for pixel in face_diff.getdata() if max(pixel) >= 24), 500)
        self.assertEqual(sum(1 for pixel in panel_diff.getdata() if max(pixel) >= 24), 0)

    def test_default_surface_uses_english_copy_and_translated_state_values(self) -> None:
        display = self.make_display()

        self.assertEqual(display._status_headline("waiting", fallback="Waiting"), "Waiting")
        self.assertEqual(display._status_helper_text("listening"), "Listening now. Speak at your own pace.")
        self.assertEqual(display._display_state_value("Internet", "ok"), "Online")
        self.assertEqual(display._display_state_value("AI", "ok"), "Ready")
        self.assertEqual(display._display_state_value("System", "Achtung"), "Warning")
        self.assertEqual(display._display_state_value("System", "warm"), "Warm")

    def test_default_scene_renderer_moves_runtime_state_into_header_and_leaves_panel_reserved(self) -> None:
        display = self.make_display()
        renderer = display._default_scene_renderer

        self.assertIsInstance(renderer, HdmiDefaultSceneRenderer)
        scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "warm"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
        )

        self.assertEqual(scene.header.brand, "TWINR")
        self.assertEqual(scene.header.state, "WAITING")
        self.assertEqual(scene.header.time_value, "12:34")
        self.assertEqual(scene.header.system_value, "ERROR")
        self.assertEqual(scene.panel.cards, ())
        self.assertGreater(scene.layout.panel_box[0], scene.layout.face_box[2])

    def test_default_scene_header_only_surface_hides_network_and_ai_status(self) -> None:
        display = self.make_display()

        first = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )
        second = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "fehlt"),
                ("AI", "fehler"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )

        self.assertIsNone(ImageChops.difference(first, second).getbbox())

    def test_default_scene_renderer_keeps_extended_header_height_without_waiting_for_debug_signals(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()

        base_scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            debug_signals=(),
            animation_frame=0,
        )
        debug_scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            debug_signals=(
                DisplayDebugSignal(
                    key="motion_state",
                    label="MOTION_STILL",
                    accent="neutral",
                    priority=90,
                ),
            ),
            animation_frame=0,
        )

        self.assertEqual(
            debug_scene.layout.header_box[3] - debug_scene.layout.header_box[1],
            base_scene.layout.header_box[3] - base_scene.layout.header_box[1],
        )

    def test_draw_twinr_header_renders_debug_signal_pills_and_overflow(self) -> None:
        renderer = HdmiDefaultSceneRenderer(tools=_FakeSceneTools())
        draw = _RecordingDraw()

        renderer._draw_twinr_header(
            draw,
            box=(20, 18, 260, 86),
            header=types.SimpleNamespace(
                brand="TWINR",
                state="WAITING",
                time_value="12:34",
                system_value="OK",
                system_accent=(116, 242, 170),
                debug_signals=(
                    DisplayDebugSignal(key="possible_fall", label="POSSIBLE_FALL", accent="alert", priority=10),
                    DisplayDebugSignal(key="attention_window", label="ATTENTION_WINDOW", accent="info", priority=30),
                    DisplayDebugSignal(key="motion_state", label="MOTION_STILL", accent="neutral", priority=90),
                    DisplayDebugSignal(key="pose", label="POSE_UPRIGHT", accent="neutral", priority=80),
                ),
            ),
        )

        pill_texts = [call["text"] for call in draw.text_calls]
        pill_positions = {
            call["position"][1]
            for call in draw.text_calls
            if call["text"] in {"POSSIBLE_FALL", "ATTENTION_WINDOW", "MOTION_STILL", "POSE_UPRIGHT"}
        }

        self.assertIn("POSSIBLE_FALL", pill_texts)
        self.assertTrue(any(text.startswith("+") for text in pill_texts))
        self.assertTrue(any(entry["outline"] == (255, 134, 110) for entry in draw.rounded_rectangles))
        self.assertGreaterEqual(len(pill_positions), 2)

    def test_default_scene_keeps_right_hand_area_visibly_reserved_when_idle(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
        )
        image = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )

        panel_crop = image.crop(scene.layout.panel_box)
        self.assertIsNotNone(panel_crop.getbbox())
        dim_pixels = sum(1 for pixel in panel_crop.getdata() if max(pixel) >= 72)
        bright_pixels = sum(1 for pixel in panel_crop.getdata() if min(pixel) >= 220)
        self.assertGreater(dim_pixels, 200)
        self.assertLess(bright_pixels, 200)

    def test_default_scene_renders_real_emoji_into_reserved_right_hand_area(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            emoji_cue=DisplayEmojiCue(symbol="thumbs_up", accent="success"),
        )
        base = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )
        with_emoji = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
            emoji_cue=DisplayEmojiCue(symbol="thumbs_up", accent="success"),
        )

        panel_diff = ImageChops.difference(base.crop(scene.layout.panel_box), with_emoji.crop(scene.layout.panel_box))

        self.assertIsNotNone(panel_diff.getbbox())
        self.assertGreater(sum(1 for pixel in panel_diff.getdata() if max(pixel) >= 24), 1200)

    def test_default_scene_renders_ambient_impulse_card_into_reserved_area(self) -> None:
        display = self.make_display()
        base_scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
        )
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            ambient_impulse_cue=DisplayAmbientImpulseCue(
                topic_key="ai companions",
                headline="Ich habe dazu heute etwas gelesen. Was meinst du?",
                body="Es geht um AI companions. Dann weiss ich besser, ob ich dranbleiben soll.",
                symbol="question",
                accent="warm",
                action="ask_one",
            ),
        )
        base = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )
        with_impulse = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
            ambient_impulse_cue=DisplayAmbientImpulseCue(
                topic_key="ai companions",
                headline="Ich habe dazu heute etwas gelesen. Was meinst du?",
                body="Es geht um AI companions. Dann weiss ich besser, ob ich dranbleiben soll.",
                symbol="question",
                accent="warm",
                action="ask_one",
            ),
        )

        panel_diff = ImageChops.difference(base.crop(scene.layout.panel_box), with_impulse.crop(scene.layout.panel_box))

        self.assertIsNotNone(panel_diff.getbbox())
        self.assertGreater(sum(1 for pixel in panel_diff.getdata() if max(pixel) >= 24), 2000)
        self.assertGreater(
            scene.layout.panel_box[2] - scene.layout.panel_box[0],
            base_scene.layout.panel_box[2] - base_scene.layout.panel_box[0],
        )

    def test_default_scene_renders_service_connect_qr_card_into_reserved_area(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            service_connect_cue=DisplayServiceConnectCue(
                service_id="whatsapp",
                service_label="WhatsApp",
                phase="qr",
                summary="Scan the QR",
                detail="Open Linked Devices on your phone.",
                qr_image_data_url=_QR_DATA_URL,
                accent="warm",
            ),
        )
        base = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
        )
        with_service_connect = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(),
            animation_frame=0,
            service_connect_cue=DisplayServiceConnectCue(
                service_id="whatsapp",
                service_label="WhatsApp",
                phase="qr",
                summary="Scan the QR",
                detail="Open Linked Devices on your phone.",
                qr_image_data_url=_QR_DATA_URL,
                accent="warm",
            ),
        )

        panel_diff = ImageChops.difference(base.crop(scene.layout.panel_box), with_service_connect.crop(scene.layout.panel_box))

        self.assertEqual(scene.panel.image_data_url, _QR_DATA_URL)
        self.assertIsNotNone(panel_diff.getbbox())
        self.assertGreater(sum(1 for pixel in panel_diff.getdata() if max(pixel) >= 24), 3000)

    def test_render_emoji_glyph_emits_once_when_font_is_missing(self) -> None:
        emitted: list[str] = []
        display = HdmiFramebufferDisplay(
            framebuffer_path=Path("/dev/null"),
            layout_mode="default",
            emit=emitted.append,
        )

        with mock.patch.object(
            HdmiFramebufferDisplay,
            "_emoji_font_candidates",
            return_value=(Path("/missing/NotoColorEmoji.ttf"),),
        ):
            self.assertIsNone(display._render_emoji_glyph("👍", target_size=96))
            self.assertIsNone(display._render_emoji_glyph("👍", target_size=96))

        self.assertEqual(sum("display_emoji_font=missing" in line for line in emitted), 1)

    def test_default_scene_uses_extended_header_and_keeps_face_first_layout(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "warm"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
        )

        header_height = scene.layout.header_box[3] - scene.layout.header_box[1]
        face_width = scene.layout.face_box[2] - scene.layout.face_box[0]
        panel_width = scene.layout.panel_box[2] - scene.layout.panel_box[0]

        self.assertGreater(header_height, 80)
        self.assertGreater(face_width, 380)
        self.assertLess(panel_width, 340)

    def test_default_scene_with_ticker_text_still_keeps_extended_face_first_layout(self) -> None:
        display = self.make_display()
        scene = display._scene_renderer().build_scene(
            width=800,
            height=480,
            status="error",
            headline="Check system",
            helper_text="Please check the system in debug view.",
            state_fields=(
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "fehler"),
                ("Zeit", "10:41"),
            ),
            animation_frame=0,
            ticker_text="Europaeische Zentralbank: Wann steigen die Zinsen?",
        )

        header_height = scene.layout.header_box[3] - scene.layout.header_box[1]
        face_width = scene.layout.face_box[2] - scene.layout.face_box[0]
        panel_width = scene.layout.panel_box[2] - scene.layout.panel_box[0]

        self.assertFalse(scene.layout.ticker_reserved)
        self.assertFalse(scene.layout.compact_panel)
        self.assertGreater(header_height, 80)
        self.assertGreater(face_width, panel_width)
        self.assertLess(panel_width, 330)

    def test_default_scene_widescreen_caps_panel_width_and_scales_face_region(self) -> None:
        display = self.make_display(width=1920, height=1080)
        scene = display._scene_renderer().build_scene(
            width=1920,
            height=1080,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "warm"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            ticker_text="Tagesschau · Calm readable headline for seniors",
        )

        header_height = scene.layout.header_box[3] - scene.layout.header_box[1]
        face_width = scene.layout.face_box[2] - scene.layout.face_box[0]
        panel_width = scene.layout.panel_box[2] - scene.layout.panel_box[0]

        self.assertGreater(header_height, 90)
        self.assertGreater(face_width, panel_width)
        self.assertLess(panel_width, 640)

    def test_face_scale_grows_on_widescreen_hdmi(self) -> None:
        display = self.make_display(width=1920, height=1080)
        renderer = display._scene_renderer()

        widescreen_scale = renderer._face_scale_for_box((248, 78, 1070, 990))
        compact_scale = renderer._face_scale_for_box((38, 90, 350, 390))

        self.assertGreater(widescreen_scale, 2.0)
        self.assertGreater(widescreen_scale, compact_scale)

    def test_default_scene_builds_morphing_presentation_overlay(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        cue = DisplayPresentationCue(
            kind="rich_card",
            title="Family Call",
            subtitle="Marta is waiting",
            body_lines=("Tap green to answer",),
            accent="warm",
            updated_at="2026-03-18T16:00:00+00:00",
            expires_at="2026-03-18T16:00:20+00:00",
        )

        scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "warm"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            presentation_cue=cue,
            presentation_now=datetime.fromisoformat("2026-03-18T16:00:00.400000+00:00"),
        )

        assert scene.presentation_graph is not None
        assert scene.presentation_graph.active_node is not None
        active = scene.presentation_graph.active_node
        self.assertGreater(active.progress, 0.4)
        self.assertLess(active.progress, 0.6)
        self.assertLess(active.box[0], scene.layout.panel_box[0])
        self.assertEqual(active.box[1], scene.layout.panel_box[1])
        self.assertEqual(active.box[2], scene.layout.panel_box[2])
        self.assertEqual(active.box[3], scene.layout.panel_box[3])

    def test_presentation_graph_prefers_highest_priority_card_and_face_sync(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()
        cue = DisplayPresentationCue(
            cards=(
                DisplayPresentationCardCue(key="summary", title="Summary", priority=20, accent="info"),
                DisplayPresentationCardCue(
                    key="family_photo",
                    kind="image",
                    title="Family Photo",
                    priority=90,
                    accent="warm",
                    face_emotion="happy",
                ),
            ),
            updated_at="2026-03-18T16:00:00+00:00",
            expires_at="2026-03-18T16:00:20+00:00",
        )

        scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "warm"),
                ("Zeit", "12:34"),
            ),
            animation_frame=0,
            presentation_cue=cue,
            presentation_now=datetime.fromisoformat("2026-03-18T16:00:00.400000+00:00"),
        )

        assert scene.presentation_graph is not None
        assert scene.presentation_graph.active_node is not None
        assert scene.face_cue is not None
        self.assertEqual(scene.presentation_graph.active_node.key, "family_photo")
        self.assertEqual(len(scene.presentation_graph.queued_cards), 1)
        self.assertEqual(scene.face_cue.gaze_x, 2)
        self.assertEqual(scene.face_cue.mouth, "smile")
        self.assertEqual(scene.face_cue.brows, "raised")

    def test_render_status_image_draws_image_presentation_surface(self) -> None:
        display = self.make_display()
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "family.png"
            Image.new("RGB", (120, 120), (240, 220, 180)).save(image_path)
            cue = DisplayPresentationCue(
                kind="image",
                title="Marta",
                subtitle="Photo received",
                body_lines=("Tap green to answer",),
                image_path=str(image_path),
                accent="success",
                updated_at="2026-03-18T15:59:00+00:00",
                expires_at="2026-03-18T16:01:00+00:00",
            )

            base = display.render_status_image(
                status="waiting",
                headline="Waiting",
                details=("Internet ok", "AI ok"),
                state_fields=state_fields,
                log_sections=(),
                animation_frame=0,
            )
            presented = display.render_status_image(
                status="waiting",
                headline="Waiting",
                details=("Internet ok", "AI ok"),
                state_fields=state_fields,
                log_sections=(),
                animation_frame=0,
                presentation_cue=cue,
            )

        diff = ImageChops.difference(base, presented)
        changed_pixels = sum(1 for pixel in diff.getdata() if max(pixel) >= 24)

        self.assertIsNotNone(diff.getbbox())
        self.assertGreater(changed_pixels, 25000)

    def test_wrapped_lines_truncate_oversized_first_token(self) -> None:
        display = self.make_display()
        draw = display._new_canvas()[1]
        font = display._font(18, bold=False)

        wrapped = display._wrapped_lines(
            draw,
            ("SUPERCALIFRAGILISTICEXPIALIDOCIOUS STATUS",),
            max_width=120,
            font=font,
            max_lines=2,
        )

        self.assertLessEqual(len(wrapped), 2)
        self.assertTrue(all(display._text_width(draw, line, font=font) <= 120 for line in wrapped))

    def test_render_status_image_draws_debug_log_screen(self) -> None:
        display = self.make_display(layout_mode="debug_log")

        image = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=(),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(
                ("System Log", ("remote ok", "service healthy")),
                ("Hardware Log", ("hdmi ok", "buttons ok")),
                ("Sprachlog", ("user Hallo", "ai Guten Tag")),
            ),
            animation_frame=0,
        )

        self.assertEqual(image.size, display.canvas_size)
        self.assertGreater(sum(1 for pixel in image.getdata() if pixel != (255, 255, 255)), 2000)


class FakeQtConstants:
    Window = 0x01
    FramelessWindowHint = 0x02
    WindowStaysOnTopHint = 0x04
    AlignCenter = 0x08


class FakeQApplication:
    _instance = None

    def __init__(self, _args: list[object]) -> None:
        type(self)._instance = self
        self.process_events_calls = 0
        self.quit_on_last_window_closed: bool | None = None

    @classmethod
    def instance(cls):
        return cls._instance

    def setQuitOnLastWindowClosed(self, value: bool) -> None:
        self.quit_on_last_window_closed = value

    def processEvents(self) -> None:
        self.process_events_calls += 1


class FakeQWidget:
    def __init__(self) -> None:
        self.title = ""
        self.flags = 0
        self.stylesheet = ""
        self.resize_calls: list[tuple[int, int]] = []
        self.fullscreen_calls = 0
        self.raise_calls = 0
        self.activate_calls = 0
        self.close_calls = 0

    def setWindowTitle(self, value: str) -> None:
        self.title = value

    def setWindowFlags(self, flags: int) -> None:
        self.flags = flags

    def setStyleSheet(self, value: str) -> None:
        self.stylesheet = value

    def resize(self, width: int, height: int) -> None:
        self.resize_calls.append((width, height))

    def showFullScreen(self) -> None:
        self.fullscreen_calls += 1

    def raise_(self) -> None:
        self.raise_calls += 1

    def activateWindow(self) -> None:
        self.activate_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FakeQLabel:
    def __init__(self, _parent: object = None) -> None:
        self.scaled_contents_values: list[bool] = []
        self.alignments: list[int] = []
        self.geometry_calls: list[tuple[int, int, int, int]] = []
        self.pixmap_calls: list[object] = []
        self.resize_calls: list[tuple[int, int]] = []
        self.show_calls = 0

    def setScaledContents(self, value: bool) -> None:
        self.scaled_contents_values.append(bool(value))

    def setAlignment(self, value: int) -> None:
        self.alignments.append(value)

    def setGeometry(self, left: int, top: int, width: int, height: int) -> None:
        self.geometry_calls.append((left, top, width, height))

    def setPixmap(self, pixmap: object) -> None:
        self.pixmap_calls.append(pixmap)

    def resize(self, width: int, height: int) -> None:
        self.resize_calls.append((width, height))

    def show(self) -> None:
        self.show_calls += 1


class FakeQImage:
    Format_RGBA8888 = 24

    def __init__(self, payload: bytes, width: int, height: int, stride: int, image_format: int) -> None:
        self.payload = payload
        self.width = width
        self.height = height
        self.stride = stride
        self.image_format = image_format


class FakeQPixmap:
    from_image_calls: list[FakeQImage] = []

    @classmethod
    def fromImage(cls, image: FakeQImage) -> dict[str, object]:
        cls.from_image_calls.append(image)
        return {
            "width": image.width,
            "height": image.height,
            "stride": image.stride,
            "format": image.image_format,
        }


def _fake_pyqt_modules() -> dict[str, object]:
    FakeQApplication._instance = None
    FakeQPixmap.from_image_calls = []
    qt_core = types.SimpleNamespace(Qt=FakeQtConstants)
    qt_gui = types.SimpleNamespace(QImage=FakeQImage, QPixmap=FakeQPixmap)
    qt_widgets = types.SimpleNamespace(
        QApplication=FakeQApplication,
        QWidget=FakeQWidget,
        QLabel=FakeQLabel,
    )
    pyqt5 = types.SimpleNamespace(QtCore=qt_core, QtGui=qt_gui, QtWidgets=qt_widgets)
    return {
        "PyQt5": pyqt5,
        "PyQt5.QtCore": qt_core,
        "PyQt5.QtGui": qt_gui,
        "PyQt5.QtWidgets": qt_widgets,
    }


class HdmiWaylandDisplayTests(unittest.TestCase):
    def test_factory_selects_hdmi_wayland_backend(self) -> None:
        config = TwinrConfig(display_driver="hdmi_wayland")
        sentinel = object()

        with mock.patch("twinr.display.factory.HdmiWaylandDisplay.from_config", return_value=sentinel):
            adapter = create_display_adapter(config)

        self.assertIs(adapter, sentinel)

    def test_show_status_renders_into_fullscreen_wayland_surface(self) -> None:
        emitted: list[str] = []
        display = HdmiWaylandDisplay(
            width=320,
            height=240,
            wayland_display="wayland-0",
            wayland_runtime_dir="/run/user/1000",
            emit=emitted.append,
        )

        with mock.patch.dict(sys.modules, _fake_pyqt_modules()):
            with mock.patch(
                "twinr.display.hdmi_wayland.apply_wayland_environment",
                return_value=Path("/run/user/1000/wayland-0"),
            ):
                display.show_status(
                    "waiting",
                    headline="Bereit",
                    state_fields=(
                        ("Status", "Waiting"),
                        ("Internet", "ok"),
                        ("AI", "ok"),
                        ("System", "ok"),
                        ("Zeit", "12:34"),
                    ),
                )
                window = display._qt_window
                image_label = display._qt_image_label
                display.tick()
                display.close()

        app = FakeQApplication.instance()
        assert app is not None
        assert window is not None
        assert image_label is not None
        self.assertIsNone(display._qt_app)
        self.assertEqual(window.title, "TWINR")
        self.assertEqual(window.flags, 0x07)
        self.assertEqual(window.fullscreen_calls, 1)
        self.assertEqual(window.raise_calls, 1)
        self.assertEqual(window.activate_calls, 1)
        self.assertEqual(window.close_calls, 1)
        self.assertEqual(image_label.geometry_calls, [(0, 0, 320, 240)])
        self.assertEqual(image_label.resize_calls, [(320, 240)])
        self.assertEqual(image_label.show_calls, 1)
        self.assertEqual(len(image_label.pixmap_calls), 1)
        self.assertGreaterEqual(app.process_events_calls, 2)
        first_image = FakeQPixmap.from_image_calls[0]
        self.assertEqual((first_image.width, first_image.height), (320, 240))
        self.assertEqual(first_image.stride, 1280)
        self.assertTrue(any("display_wayland=ready" in line for line in emitted))
        self.assertTrue(any("toolkit=pyqt5" in line for line in emitted))

    def test_resolve_wayland_socket_prefers_configured_runtime_dir(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("pathlib.Path.glob", side_effect=AssertionError("glob should not run")):
                with tempfile.TemporaryDirectory() as temp_dir:
                    runtime_dir = Path(temp_dir)
                    socket_path = runtime_dir / "wayland-0"
                    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        server.bind(str(socket_path))
                        resolved = resolve_wayland_socket(
                            "wayland-0",
                            configured_runtime_dir=str(runtime_dir),
                        )
                    finally:
                        server.close()
                    self.assertEqual(resolved, socket_path)

    def test_apply_wayland_environment_sets_toolkit_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            socket_path = runtime_dir / "wayland-0"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(socket_path))
                with mock.patch.dict("os.environ", {}, clear=True):
                    resolved = apply_wayland_environment(
                        "wayland-0",
                        configured_runtime_dir=str(runtime_dir),
                    )
                    self.assertEqual(resolved, socket_path)
                    self.assertEqual(os.environ["XDG_RUNTIME_DIR"], str(runtime_dir))
                    self.assertEqual(os.environ["WAYLAND_DISPLAY"], "wayland-0")
                    self.assertEqual(os.environ["SDL_VIDEODRIVER"], "wayland")
                    self.assertEqual(os.environ["QT_QPA_PLATFORM"], "wayland")
            finally:
                server.close()
