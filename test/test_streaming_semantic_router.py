from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
from typing import Callable
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.streaming_capture import StreamingCaptureController
from twinr.agent.workflows.streaming_semantic_router import StreamingSemanticRouterRuntime


def _fake_bundle():
    return SimpleNamespace(
        metadata=SimpleNamespace(
            model_id="router-v1",
            authoritative_labels=("web",),
        )
    )


class _FakeRouter:
    def __init__(self) -> None:
        self.warmup_calls: list[str] = []
        self.classify_calls: list[str] = []

    def warmup(self, probe_text: str = "warmup") -> None:
        self.warmup_calls.append(probe_text)

    def classify(self, transcript: str):
        self.classify_calls.append(transcript)
        return SimpleNamespace(
            label="web",
            confidence=0.91,
            margin=0.33,
            authoritative=False,
            fallback_reason=None,
            model_id="router-v1",
            latency_ms=1.2,
            scores={"web": 0.91},
        )


class _LoopStub:
    def __init__(self, *, model_dir: str) -> None:
        self.config = SimpleNamespace(
            local_semantic_router_mode="gated",
            local_semantic_router_model_dir=model_dir,
            local_semantic_router_user_intent_model_dir=None,
            local_semantic_router_trace=True,
            local_semantic_router_warmup_enabled=True,
            local_semantic_router_warmup_probe="wie ist die lage",
            local_semantic_router_supported_labels=("web", "memory", "tool"),
            local_semantic_router_max_chars=1024,
            local_semantic_router_cache_size=128,
            local_semantic_router_cache_ttl_s=10.0,
            local_semantic_router_min_confidence=None,
            local_semantic_router_min_margin=None,
            local_semantic_router_allowed_model_root=None,
            streaming_first_word_enabled=False,
        )
        self.emitted: list[str] = []
        self.trace_events: list[tuple[str, dict[str, object]]] = []
        self.scheduled_warmups: list[tuple[str, Callable[[], None]]] = []
        self.waited_warmups: list[tuple[str, int | None]] = []

    def emit(self, message: str) -> None:
        self.emitted.append(message)

    def _trace_event(self, name: str, **payload) -> None:
        self.trace_events.append((name, payload))

    def _schedule_speculative_warmup(self, name: str, runner) -> None:
        self.scheduled_warmups.append((name, runner))

    def _wait_for_speculative_warmup(self, name: str, *, wait_ms: int | None = None) -> None:
        self.waited_warmups.append((name, wait_ms))


class StreamingSemanticRouterRuntimeTests(unittest.TestCase):
    def test_runtime_defers_user_intent_ort_validation_during_idle_startup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            user_intent_dir = Path(temp_dir) / "user-intent"
            model_dir.mkdir()
            user_intent_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            loop.config.local_semantic_router_user_intent_model_dir = str(user_intent_dir)
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                return_value=_fake_bundle(),
            ) as load_route_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.load_user_intent_bundle",
                    return_value=_fake_bundle(),
                ) as load_user_bundle:
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.TwoStageLocalSemanticRouter",
                        return_value=router,
                    ):
                        with mock.patch(
                            "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                        ):
                            runtime = StreamingSemanticRouterRuntime(loop)
                            load_route_bundle.assert_not_called()
                            load_user_bundle.assert_not_called()
                            runtime.maybe_start_warmup("Wie ist die Lage heute?")
                            self.assertEqual(len(loop.scheduled_warmups), 1)
                            _, runner = loop.scheduled_warmups[0]
                            runner()

        load_route_bundle.assert_called_once_with(model_dir.resolve())
        load_user_bundle.assert_called_once_with(
            user_intent_dir.resolve(),
            eager_runtime_validation=False,
        )

    def test_runtime_does_not_eagerly_warm_router_during_initial_build(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            model_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                return_value=_fake_bundle(),
            ) as load_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.LocalSemanticRouter",
                    return_value=router,
                ):
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                    ):
                        StreamingSemanticRouterRuntime(loop)
                        load_bundle.assert_not_called()

        self.assertEqual(router.warmup_calls, [])
        self.assertEqual(loop.scheduled_warmups, [])

    def test_runtime_schedules_turn_near_warmup_after_transcript_activity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            model_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                return_value=_fake_bundle(),
            ) as load_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.LocalSemanticRouter",
                    return_value=router,
                ):
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                    ) as record_phase:
                        runtime = StreamingSemanticRouterRuntime(loop)
                        load_bundle.assert_not_called()
                        runtime.maybe_start_warmup("Wie ist die Lage heute?")
                        self.assertEqual(len(loop.scheduled_warmups), 1)
                        name, runner = loop.scheduled_warmups[0]
                        self.assertEqual(name, "local_semantic_router")
                        runner()
                        runtime.maybe_start_warmup("Noch mal warm machen")

        load_bundle.assert_called_once_with(model_dir.resolve())
        self.assertEqual(router.warmup_calls, ["Wie ist die Lage heute?"])
        self.assertEqual(len(loop.scheduled_warmups), 1)
        self.assertEqual(
            [call.kwargs["label"] for call in record_phase.call_args_list],
            [
                "streaming_loop.semantic_router.bundle_ready",
                "streaming_loop.semantic_router.lazy_warmup",
            ],
        )

    def test_runtime_waits_for_pending_warmup_before_resolving_transcript(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            model_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                return_value=_fake_bundle(),
            ) as load_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.LocalSemanticRouter",
                    return_value=router,
                ):
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                    ):
                        runtime = StreamingSemanticRouterRuntime(loop)
                        resolution = runtime.resolve_transcript("Bitte such nach dem Wetter.")

        self.assertIsNone(resolution)
        load_bundle.assert_not_called()
        self.assertEqual(len(loop.scheduled_warmups), 1)
        self.assertEqual(loop.waited_warmups, [("local_semantic_router", None)])
        self.assertEqual(router.warmup_calls, [])
        self.assertEqual(router.classify_calls, [])

    def test_runtime_builds_synchronously_without_warmup_scheduler(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            model_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            loop._schedule_speculative_warmup = None  # type: ignore[assignment]
            loop._wait_for_speculative_warmup = None  # type: ignore[assignment]
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                return_value=_fake_bundle(),
            ) as load_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.LocalSemanticRouter",
                    return_value=router,
                ):
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                    ) as record_phase:
                        runtime = StreamingSemanticRouterRuntime(loop)
                        resolution = runtime.resolve_transcript("Bitte such nach dem Wetter.")

        self.assertIsNotNone(resolution)
        load_bundle.assert_called_once_with(model_dir.resolve())
        self.assertEqual(router.warmup_calls, ["Bitte such nach dem Wetter."])
        self.assertEqual(router.classify_calls, ["Bitte such nach dem Wetter."])
        self.assertEqual(
            [call.kwargs["label"] for call in record_phase.call_args_list],
            [
                "streaming_loop.semantic_router.bundle_ready",
                "streaming_loop.semantic_router.lazy_warmup",
            ],
        )

    def test_reload_defers_rebuild_and_keeps_previous_router_on_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            model_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            loop._schedule_speculative_warmup = None  # type: ignore[assignment]
            loop._wait_for_speculative_warmup = None  # type: ignore[assignment]
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                side_effect=[_fake_bundle(), RuntimeError("broken_bundle")],
            ) as load_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.LocalSemanticRouter",
                    return_value=router,
                ):
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                    ):
                        runtime = StreamingSemanticRouterRuntime(loop)
                        first_resolution = runtime.resolve_transcript("Bitte such nach dem Wetter.")
                        self.assertIsNotNone(first_resolution)
                        self.assertEqual(load_bundle.call_count, 1)
                        runtime.reload()
                        self.assertEqual(load_bundle.call_count, 1)
                        second_resolution = runtime.resolve_transcript("Noch mal Wetter bitte.")

        self.assertIsNotNone(second_resolution)
        self.assertEqual(load_bundle.call_count, 2)
        self.assertEqual(
            router.warmup_calls,
            ["Bitte such nach dem Wetter.", "Noch mal Wetter bitte."],
        )
        self.assertEqual(
            router.classify_calls,
            ["Bitte such nach dem Wetter.", "Noch mal Wetter bitte."],
        )
        self.assertIn("local_semantic_router_unavailable=RuntimeError", loop.emitted)
        self.assertIn(
            "streaming_local_semantic_router_reload_kept_previous",
            [name for name, _payload in loop.trace_events],
        )

    def test_reload_keeps_previous_router_when_model_dir_disappears(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "router"
            model_dir.mkdir()
            loop = _LoopStub(model_dir=str(model_dir))
            loop._schedule_speculative_warmup = None  # type: ignore[assignment]
            loop._wait_for_speculative_warmup = None  # type: ignore[assignment]
            router = _FakeRouter()
            with mock.patch(
                "twinr.agent.workflows.streaming_semantic_router.load_semantic_router_bundle",
                return_value=_fake_bundle(),
            ) as load_bundle:
                with mock.patch(
                    "twinr.agent.workflows.streaming_semantic_router.LocalSemanticRouter",
                    return_value=router,
                ):
                    with mock.patch(
                        "twinr.agent.workflows.streaming_semantic_router.record_streaming_memory_phase_best_effort"
                    ):
                        runtime = StreamingSemanticRouterRuntime(loop)
                        first_resolution = runtime.resolve_transcript("Bitte such nach dem Wetter.")
                        self.assertIsNotNone(first_resolution)
                        self.assertEqual(load_bundle.call_count, 1)
                        loop.config.local_semantic_router_model_dir = None
                        runtime.reload()
                        second_resolution = runtime.resolve_transcript("Bitte such noch mal nach dem Wetter.")

        self.assertIsNotNone(second_resolution)
        self.assertEqual(load_bundle.call_count, 1)
        self.assertEqual(
            router.classify_calls,
            ["Bitte such nach dem Wetter.", "Bitte such noch mal nach dem Wetter."],
        )
        self.assertIn(
            "streaming_local_semantic_router_reload_kept_previous",
            [name for name, _payload in loop.trace_events],
        )


class StreamingCaptureControllerWarmupTests(unittest.TestCase):
    def test_interim_transcript_starts_router_warmup_before_other_speculation(self) -> None:
        calls: list[tuple[object, ...]] = []
        loop = SimpleNamespace(
            _trace_event=lambda *args, **kwargs: None,
            _maybe_start_local_semantic_router_warmup=lambda text: calls.append(("warmup", text)),
            _maybe_start_speculative_first_word=lambda text: calls.append(("first_word", text)),
            _maybe_start_speculative_supervisor_decision=lambda text: calls.append(("supervisor", text)),
            _maybe_start_speculative_long_term_context=lambda text, final_transcript=False: calls.append(
                ("long_term", final_transcript, text)
            ),
        )

        StreamingCaptureController(loop).handle_stt_interim("  Wie ist die Lage?  ")

        self.assertEqual(
            calls,
            [
                ("warmup", "Wie ist die Lage?"),
                ("first_word", "Wie ist die Lage?"),
                ("supervisor", "Wie ist die Lage?"),
                ("long_term", False, "Wie ist die Lage?"),
            ],
        )

    def test_endpoint_transcript_starts_router_warmup_with_final_flag(self) -> None:
        calls: list[tuple[object, ...]] = []
        loop = SimpleNamespace(
            _trace_event=lambda *args, **kwargs: None,
            _maybe_start_local_semantic_router_warmup=lambda text: calls.append(("warmup", text)),
            _maybe_start_speculative_first_word=lambda text: calls.append(("first_word", text)),
            _maybe_start_speculative_supervisor_decision=lambda text: calls.append(("supervisor", text)),
            _maybe_start_speculative_long_term_context=lambda text, final_transcript=False: calls.append(
                ("long_term", final_transcript, text)
            ),
        )
        event = SimpleNamespace(transcript="  Druck das bitte aus.  ", speech_final=True, utterance_end=True)

        StreamingCaptureController(loop).handle_stt_endpoint(event)

        self.assertEqual(
            calls,
            [
                ("warmup", "Druck das bitte aus."),
                ("first_word", "Druck das bitte aus."),
                ("supervisor", "Druck das bitte aus."),
                ("long_term", True, "Druck das bitte aus."),
            ],
        )


if __name__ == "__main__":
    unittest.main()
