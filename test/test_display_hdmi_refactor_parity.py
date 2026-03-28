from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.debug_signals import DisplayDebugSignal
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.hdmi_default_scene import HdmiDefaultSceneRenderer
from twinr.display.hdmi_default_scene_impl.renderer import HdmiDefaultSceneRenderer as HdmiDefaultSceneRendererImpl
from twinr.display.presentation_cues import DisplayPresentationCue
from twinr.display.reserve_bus import DisplayReserveBusState
from twinr.display.service_connect_cues import DisplayServiceConnectCue
from test._display_hdmi_test_support import _FakeSceneTools, _RecordingDraw, _QR_DATA_URL

_EXPECTED_GOLDEN_DIGESTS = {
    "waiting_scene": "e93e24a105250bcb374221dce01cd3923c865861d3dc657a1968a9afbdaff832",
    "service_scene": "c170e437b88341d861a4d7018f5a0823f37accdcaf68628681caa7983f1a2f40",
    "presentation_scene": "a9d456ead2ee22d5dedc485b3b1331c41d93f37cb9956d24df7e8c5edfbc43f1",
    "face_metrics": "71a426ef5b07d3223d2f5131f9cee16daa67865878c0ec1e4b72edf1b80f8784",
    "prompt_draw": "69b1daecf35436c762c77ccaa5c8a594b0be5402e59dfc3f553be2965fa6c6b7",
}


def _normalize_payload(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _normalize_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__") and not isinstance(value, (int, float, bool, str, type(None))):
        return {
            key: _normalize_payload(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return value


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


class HdmiDefaultSceneRefactorParityTests(unittest.TestCase):
    def _build_cases(
        self,
        *,
        renderer_cls: type[HdmiDefaultSceneRenderer] | type[HdmiDefaultSceneRendererImpl],
    ) -> dict[str, object]:
        renderer = renderer_cls(tools=_FakeSceneTools())
        state_fields = (
            ("Status", "Waiting"),
            ("Internet", "ok"),
            ("AI", "ok"),
            ("System", "ok"),
            ("Zeit", "12:34"),
        )

        waiting_scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            debug_signals=(
                DisplayDebugSignal(
                    key="motion_state",
                    label="MOTION_STILL",
                    accent="neutral",
                    priority=90,
                ),
                DisplayDebugSignal(
                    key="pose",
                    label="POSE_UPRIGHT",
                    accent="info",
                    priority=50,
                ),
            ),
            animation_frame=5,
        )
        service_scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
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
        presentation_scene = renderer.build_scene(
            width=800,
            height=480,
            status="waiting",
            headline="Waiting",
            helper_text="Press the green button and speak naturally.",
            state_fields=state_fields,
            animation_frame=2,
            presentation_cue=DisplayPresentationCue(
                kind="rich_card",
                title="Family Call",
                subtitle="In 10 minutes",
                body_lines=("Anna is ready when you are.",),
                updated_at="2026-03-18T18:00:00+00:00",
                expires_at="2026-03-18T18:00:20+00:00",
            ),
            presentation_now=datetime.fromisoformat("2026-03-18T18:00:04+00:00"),
        )
        cue = DisplayFaceCue(
            gaze_x=2,
            gaze_y=-2,
            mouth="scrunched",
            brows="roof",
            blink=True,
            head_dx=1,
        )
        face_metrics = {
            "waiting_eye": renderer._eye_state("waiting", 0, "left", face_cue=cue),
            "error_eye": renderer._eye_state("error", 0, "left", face_cue=cue),
            "waiting_offset": renderer._face_offset("waiting", 8, face_cue=cue),
            "error_offset": renderer._face_offset("error", 4, face_cue=cue),
        }
        prompt_panel = renderer._build_panel_model(
            reserve_bus=DisplayReserveBusState(
                owner="ambient_impulse",
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
            )
        )
        draw = _RecordingDraw()
        renderer._draw_status_panel(draw, box=(400, 80, 780, 250), panel=prompt_panel, compact=False)
        prompt_draw = {
            "rounded_rectangles": draw.rounded_rectangles,
            "text_calls": [
                {
                    "position": list(call["position"]),
                    "text": call["text"],
                    "fill": list(call["fill"]) if isinstance(call["fill"], tuple) else call["fill"],
                    "font_size": call["font"].size,
                    "font_bold": call["font"].bold,
                }
                for call in draw.text_calls
            ],
        }
        return {
            "waiting_scene": waiting_scene,
            "service_scene": service_scene,
            "presentation_scene": presentation_scene,
            "face_metrics": face_metrics,
            "prompt_draw": prompt_draw,
        }

    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            HdmiDefaultSceneRenderer.__module__,
            "twinr.display.hdmi_default_scene",
        )

    def test_golden_master_hashes_remain_stable(self) -> None:
        for name, payload in self._build_cases(renderer_cls=HdmiDefaultSceneRenderer).items():
            with self.subTest(case=name):
                self.assertEqual(_payload_digest(payload), _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        wrapped = self._build_cases(renderer_cls=HdmiDefaultSceneRenderer)
        internal = self._build_cases(renderer_cls=HdmiDefaultSceneRendererImpl)
        self.assertEqual(_normalize_payload(wrapped), _normalize_payload(internal))
