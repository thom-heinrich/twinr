from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

from PIL import ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display.face_cues import DisplayFaceCue
from twinr.display.hdmi_ambient_moments import HdmiAmbientMomentDirector
from twinr.display.hdmi_default_scene import HdmiDefaultSceneRenderer
from twinr.display.hdmi_fbdev import FramebufferBitfield, FramebufferGeometry, HdmiFramebufferDisplay
from twinr.display.presentation_cues import DisplayPresentationCue


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


class HdmiAmbientMomentTests(unittest.TestCase):
    def make_display(self) -> HdmiFramebufferDisplay:
        display = HdmiFramebufferDisplay(framebuffer_path=Path("/dev/null"), layout_mode="default")
        display._geometry = _rgb565_geometry()
        display._default_scene_renderer = HdmiDefaultSceneRenderer(
            display,
            ambient_director=HdmiAmbientMomentDirector(
                bucket_seconds=60.0,
                active_window_s=8.0,
                trigger_divisor=1,
            ),
        )
        return display

    def test_director_only_resolves_idle_moments_for_waiting_without_other_overrides(self) -> None:
        director = HdmiAmbientMomentDirector(bucket_seconds=60.0, active_window_s=8.0, trigger_divisor=1)
        active_now = datetime.fromisoformat("2026-03-18T18:00:04+00:00")

        active = director.resolve(
            status="waiting",
            now=active_now,
            face_cue_active=False,
            presentation_active=False,
        )
        blocked_by_face = director.resolve(
            status="waiting",
            now=active_now,
            face_cue_active=True,
            presentation_active=False,
        )
        blocked_by_presentation = director.resolve(
            status="waiting",
            now=active_now,
            face_cue_active=False,
            presentation_active=True,
        )
        blocked_by_status = director.resolve(
            status="listening",
            now=active_now,
            face_cue_active=False,
            presentation_active=False,
        )

        self.assertIsNotNone(active)
        self.assertIsNone(blocked_by_face)
        self.assertIsNone(blocked_by_presentation)
        self.assertIsNone(blocked_by_status)

    def test_director_cycles_through_extended_ambient_moment_catalog(self) -> None:
        director = HdmiAmbientMomentDirector(bucket_seconds=60.0, active_window_s=8.0, trigger_divisor=1)
        seen: set[str] = set()
        start = datetime.fromisoformat("2026-03-18T00:00:04+00:00")

        for minute in range(0, 6 * 60, 1):
            moment = director.resolve(
                status="waiting",
                now=start + timedelta(minutes=minute),
                face_cue_active=False,
                presentation_active=False,
            )
            if moment is not None:
                seen.add(moment.key)
            if {"sparkle", "heart", "curious", "sleepy", "wave", "crown"} <= seen:
                break

        self.assertTrue({"sparkle", "heart", "curious", "sleepy", "wave", "crown"} <= seen)

    def test_default_trigger_divisor_still_allows_multiple_moment_keys(self) -> None:
        director = HdmiAmbientMomentDirector()
        seen: set[str] = set()
        start = datetime.fromisoformat("2026-03-18T00:00:04+00:00")

        for slot in range(0, 30 * 24 * 12):
            moment = director.resolve(
                status="waiting",
                now=start + timedelta(minutes=slot * 5),
                face_cue_active=False,
                presentation_active=False,
            )
            if moment is None:
                continue
            seen.add(moment.key)
            if len(seen) >= 3:
                break

        self.assertGreaterEqual(len(seen), 3)

    def test_scene_build_injects_ambient_moment_and_face_cue_during_active_idle_window(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()

        scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Ready",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=3,
            ambient_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
        )

        self.assertIsNotNone(scene.ambient_moment)
        self.assertIsNotNone(scene.face_cue)
        assert scene.ambient_moment is not None
        assert scene.face_cue is not None
        self.assertEqual(scene.face_cue.source, "ambient_moment")
        self.assertIn(scene.ambient_moment.key, {"sparkle", "heart", "curious", "sleepy", "wave", "crown"})

    def test_ambient_moment_yields_to_external_face_cue_and_presentation(self) -> None:
        display = self.make_display()
        renderer = display._scene_renderer()

        scene_with_face = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Ready",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=3,
            face_cue=DisplayFaceCue(mouth="smile", brows="raised"),
            ambient_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
        )
        scene_with_presentation = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Ready",
            helper_text="Press the green button and speak naturally.",
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            animation_frame=3,
            presentation_cue=DisplayPresentationCue(
                kind="rich_card",
                title="Family Call",
                updated_at="2026-03-18T18:00:00+00:00",
                expires_at="2026-03-18T18:00:20+00:00",
            ),
            presentation_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
            ambient_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
        )

        self.assertIsNone(scene_with_face.ambient_moment)
        self.assertIsNone(scene_with_presentation.ambient_moment)

    def test_ambient_moment_changes_face_region_without_panel_churn(self) -> None:
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
            headline="Ready",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=3,
            ambient_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
        )
        base = display.render_status_image(
            status="waiting",
            headline="Ready",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=3,
            render_now=datetime.fromisoformat("2026-03-18T18:00:12+00:00"),
        )
        ambient = display.render_status_image(
            status="waiting",
            headline="Ready",
            details=("Internet ok", "AI ok"),
            state_fields=state_fields,
            log_sections=(),
            animation_frame=3,
            render_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
        )

        face_diff = ImageChops.difference(base.crop(scene.layout.face_box), ambient.crop(scene.layout.face_box))
        panel_diff = ImageChops.difference(base.crop(scene.layout.panel_box), ambient.crop(scene.layout.panel_box))
        changed_face_pixels = sum(1 for pixel in face_diff.getdata() if max(pixel) >= 24)
        changed_panel_pixels = sum(1 for pixel in panel_diff.getdata() if max(pixel) >= 24)

        self.assertIsNotNone(face_diff.getbbox())
        self.assertGreater(changed_face_pixels, 350)
        self.assertEqual(changed_panel_pixels, 0)


if __name__ == "__main__":
    unittest.main()
