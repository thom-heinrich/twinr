"""Browser automation loader, bridge, and regression coverage for Twinr."""

import json
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
import importlib.util
import sys
import tempfile
import textwrap
from typing import Any, cast
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.tools.handlers.browser import handle_browser_automation
from twinr.agent.tools.prompting.instructions import (
    build_supervisor_decision_instructions,
    build_tool_agent_instructions,
)
from twinr.agent.tools.runtime.availability import available_realtime_tool_names
from twinr.agent.tools.schemas import build_agent_tool_schemas
from twinr.browser_automation import (
    BrowserAutomationArtifact,
    BrowserAutomationRequest,
    BrowserAutomationResult,
    BrowserAutomationUnavailableError,
    load_browser_automation_driver,
    probe_browser_automation,
)

from browser_automation.api_discovery import collect_inline_api_packets
from browser_automation.completion_verifier import SameUrlCompletionVerifier, derive_trace_navigation_state
from browser_automation.dense_reader import DensePageReader, DensePageReaderResult, _content_block_packets_from_evidence
from browser_automation.tessairact_bridge import (
    TessairactVendoredDriver,
    _SessionState,
    _TessairactToolRegistry,
    _VendoredLoopRuntime,
    _local_host_safety_flags,
    _restore_modules,
    _temporary_modules,
)
from twinr.agent.tools.runtime.browser_follow_up import (
    peek_pending_browser_follow_up_hint,
    remember_pending_browser_follow_up_hint,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_SPECIALIST = (
    _REPO_ROOT
    / "browser_automation"
    / "vendor"
    / "src"
    / "tessairact"
    / "capabilities"
    / "capabilities"
    / "browser_use"
    / "agent"
    / "specialist.py"
)


def _load_vendored_specialist_for_test() -> Any:
    backend_module = ModuleType("tessairact.infrastructure.llm.backends.backend")
    setattr(backend_module, "StrictJSONSchemaCall", dict)
    setattr(backend_module, "call_strict_json_schema", lambda *_args, **_kwargs: {})

    types_module = ModuleType("tessairact.infrastructure.llm.backends.types")
    setattr(types_module, "ChatMessage", dict)

    shared_types_module = ModuleType("tessairact.capabilities.shared.types")
    setattr(shared_types_module, "ToolCall", dict)
    setattr(shared_types_module, "ToolCallBatch", dict)

    validation_module = ModuleType("tessairact.capabilities.shared.helpers.validation")
    setattr(validation_module, "validate_json_schema", lambda **_kwargs: None)

    backup, installed = _temporary_modules(
        {
            "tessairact.infrastructure.llm.backends.backend": backend_module,
            "tessairact.infrastructure.llm.backends.types": types_module,
            "tessairact.capabilities.shared.types": shared_types_module,
            "tessairact.capabilities.shared.helpers.validation": validation_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("twinr_test_vendor_specialist", _VENDORED_SPECIALIST)
        if spec is None or spec.loader is None:
            raise AssertionError("vendored specialist import spec missing")
        module = importlib.util.module_from_spec(spec)
        previous_module = sys.modules.get(spec.name)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            if previous_module is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous_module
        return module
    finally:
        _restore_modules(backup, installed)


class BrowserAutomationLoaderTests(unittest.TestCase):
    def test_probe_reports_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)
            availability = probe_browser_automation(config=config, project_root=root)

        self.assertFalse(availability.enabled)
        self.assertFalse(availability.available)
        self.assertEqual(availability.reason, "Browser automation is disabled by config.")

    def test_probe_reports_missing_entry_module_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_BROWSER_AUTOMATION_ENABLED=true\n", encoding="utf-8")
            (root / "browser_automation").mkdir()

            config = TwinrConfig.from_env(env_path)
            availability = probe_browser_automation(config=config, project_root=root)

        self.assertTrue(availability.enabled)
        self.assertFalse(availability.available)
        self.assertEqual(availability.reason, "Browser automation entry module does not exist.")

    def test_loader_imports_local_driver_from_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_BROWSER_AUTOMATION_ENABLED=true\n", encoding="utf-8")
            workspace = root / "browser_automation"
            workspace.mkdir()
            (workspace / "adapter.py").write_text(
                textwrap.dedent(
                    """
                    from twinr.browser_automation import (
                        BrowserAutomationArtifact,
                        BrowserAutomationAvailability,
                        BrowserAutomationRequest,
                        BrowserAutomationResult,
                    )


                    class DemoDriver:
                        def availability(self) -> BrowserAutomationAvailability:
                            return BrowserAutomationAvailability(
                                enabled=True,
                                available=True,
                                reason="ready",
                            )

                        def execute(self, request: BrowserAutomationRequest) -> BrowserAutomationResult:
                            return BrowserAutomationResult(
                                ok=True,
                                status="completed",
                                summary=f"executed: {request.goal}",
                                final_url=request.start_url,
                                artifacts=(
                                    BrowserAutomationArtifact(
                                        kind="screenshot",
                                        path="artifacts/browser.png",
                                        content_type="image/png",
                                    ),
                                ),
                                data={"task_id": request.task_id},
                            )


                    def create_browser_automation_driver(*, config, project_root):
                        return DemoDriver()
                    """
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)
            driver = load_browser_automation_driver(config=config, project_root=root)
            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="check-hours",
                    goal="Check the opening hours.",
                    start_url="https://example.org",
                    allowed_domains=("example.org",),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "https://example.org")
        self.assertEqual(result.data["task_id"], "check-hours")
        self.assertEqual(
            result.artifacts[0],
            BrowserAutomationArtifact(
                kind="screenshot",
                path="artifacts/browser.png",
                content_type="image/png",
            ),
        )

    def test_loader_fails_closed_when_workspace_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            workspace = root / "browser_automation"
            workspace.mkdir()
            (workspace / "adapter.py").write_text(
                "def create_browser_automation_driver(*, config, project_root):\n    return object()\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)
            with self.assertRaises(BrowserAutomationUnavailableError):
                load_browser_automation_driver(config=config, project_root=root)


class BrowserAutomationRuntimeTests(unittest.TestCase):
    def test_runtime_availability_hides_browser_tool_until_workspace_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "TWINR_BROWSER_AUTOMATION_ENABLED=true\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)

            hidden = available_realtime_tool_names(
                config,
                tool_names=("search_live_info", "browser_automation"),
            )

            workspace = root / "browser_automation"
            workspace.mkdir()
            (workspace / "adapter.py").write_text(
                "def create_browser_automation_driver(*, config, project_root):\n    raise RuntimeError('not loaded in probe')\n",
                encoding="utf-8",
            )

            shown = available_realtime_tool_names(
                config,
                tool_names=("search_live_info", "browser_automation"),
            )

        self.assertEqual(hidden, ("search_live_info",))
        self.assertEqual(shown, ("search_live_info", "browser_automation"))

    def test_browser_schema_describes_specific_site_interaction(self) -> None:
        schema = build_agent_tool_schemas(("browser_automation",))[0]

        self.assertEqual(schema["name"], "browser_automation")
        self.assertIn("specific website", schema["description"])
        self.assertIn("search_live_info", schema["description"])
        self.assertIn("different method", schema["description"])
        self.assertIn("take a little longer", schema["description"])
        self.assertIn("could not be verified", schema["description"])
        self.assertEqual(set(schema["parameters"]["required"]), {"goal", "allowed_domains"})
        self.assertIn("allowed_domains", schema["parameters"]["properties"])

    def test_search_schema_marks_live_site_interaction_as_browser_work(self) -> None:
        schemas = build_agent_tool_schemas(("search_live_info", "browser_automation"))
        search_schema = next(schema for schema in schemas if schema["name"] == "search_live_info")

        self.assertIn("booking flows", search_schema["description"])
        self.assertIn("forms", search_schema["description"])
        self.assertIn("social profile/post/story", search_schema["description"])

    def test_browser_handler_executes_local_driver_and_returns_json_safe_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_BROWSER_AUTOMATION_ENABLED=true\n", encoding="utf-8")
            workspace = root / "browser_automation"
            workspace.mkdir()
            (workspace / "adapter.py").write_text(
                textwrap.dedent(
                    """
                    from twinr.browser_automation import (
                        BrowserAutomationAvailability,
                        BrowserAutomationArtifact,
                        BrowserAutomationRequest,
                        BrowserAutomationResult,
                    )


                    class DemoDriver:
                        def availability(self) -> BrowserAutomationAvailability:
                            return BrowserAutomationAvailability(
                                enabled=True,
                                available=True,
                                reason="ready",
                            )

                        def execute(self, request: BrowserAutomationRequest) -> BrowserAutomationResult:
                            return BrowserAutomationResult(
                                ok=True,
                                status="completed",
                                summary=f"checked: {request.goal}",
                                final_url=request.start_url,
                                artifacts=(
                                    BrowserAutomationArtifact(
                                        kind="screenshot",
                                        path="artifacts/browser.png",
                                        content_type="image/png",
                                    ),
                                ),
                                data={"task_id": request.task_id, "domains": list(request.allowed_domains)},
                            )


                    def create_browser_automation_driver(*, config, project_root):
                        return DemoDriver()
                    """
                ),
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            emitted: list[str] = []
            events: list[tuple[tuple[object, ...], dict[str, object]]] = []
            usages: list[dict[str, object]] = []
            owner = SimpleNamespace(
                config=config,
                emit=emitted.append,
                _record_event=lambda *args, **kwargs: events.append((args, kwargs)),
                _record_usage=lambda **kwargs: usages.append(kwargs),
            )

            result = handle_browser_automation(
                owner,
                {
                    "goal": "Check the holiday opening hours.",
                    "start_url": "https://example.org/hours",
                    "allowed_domains": ["example.org"],
                    "max_steps": 8,
                    "capture_html": True,
                },
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_url"], "https://example.org/hours")
        artifacts = cast(list[dict[str, object]], result["artifacts"])
        self.assertEqual(artifacts[0]["kind"], "screenshot")
        self.assertTrue(result["used_web_search"])
        self.assertIn("browser_automation_status=starting", emitted)
        self.assertIn("browser_automation_status=completed", emitted)
        self.assertTrue(any(call[0][0] == "browser_automation_started" for call in events))
        self.assertTrue(any(call[0][0] == "browser_automation_completed" for call in events))
        self.assertEqual(usages[0]["request_kind"], "browser_automation")

    def test_browser_handler_returns_failed_payload_when_driver_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_BROWSER_AUTOMATION_ENABLED=true\n", encoding="utf-8")
            config = TwinrConfig.from_env(env_path)
            owner = SimpleNamespace(config=config)

            result = handle_browser_automation(
                owner,
                {
                    "goal": "Check the opening hours.",
                    "allowed_domains": ["example.org"],
                },
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "unavailable")

    def test_browser_handler_repairs_bad_handoff_host_from_pending_search_hint(self) -> None:
        runtime = SimpleNamespace()
        remember_pending_browser_follow_up_hint(
            runtime,
            question="Hat Café Luise heute ein Mittagsmenü veröffentlicht?",
            follow_up_url="https://www.cafe-luise-baeckerei.de/",
            follow_up_domain="cafe-luise-baeckerei.de",
            site_follow_up_recommended=True,
            question_resolved=False,
            verification_status="partial",
            reason="Die offizielle Website sollte geprüft werden.",
            sources=("https://www.cafe-luise-baeckerei.de/",),
        )
        owner = SimpleNamespace(
            config=SimpleNamespace(project_root=str(_REPO_ROOT)),
            runtime=runtime,
            emit=lambda _message: None,
            _record_event=lambda *_args, **_kwargs: None,
            _record_usage=lambda **_kwargs: None,
        )
        captured: dict[str, Any] = {}

        class _Driver:
            def execute(self, request: BrowserAutomationRequest) -> BrowserAutomationResult:
                captured["request"] = request
                return BrowserAutomationResult(
                    ok=True,
                    status="completed",
                    summary="checked",
                    final_url=request.start_url,
                    data={"task_id": request.task_id},
                )

        with (
            mock.patch("twinr.agent.tools.handlers.browser.load_browser_automation_driver", return_value=_Driver()),
            mock.patch("twinr.agent.tools.runtime.browser_follow_up._host_resolves", return_value=False),
        ):
            result = handle_browser_automation(
                owner,
                {
                    "goal": "Prüfe auf der Website von Café Luise, ob für heute ein Mittagsmenü veröffentlicht wurde.",
                    "start_url": "https://www.cafe-luise.de/",
                    "allowed_domains": ["cafe-luise.de"],
                },
            )

        request = cast(BrowserAutomationRequest, captured["request"])
        self.assertEqual(request.start_url, "https://www.cafe-luise-baeckerei.de/")
        self.assertEqual(
            request.allowed_domains,
            ("cafe-luise-baeckerei.de", "www.cafe-luise-baeckerei.de"),
        )
        self.assertIn("Wenn dort kein aktueller Nachweis sichtbar ist", request.goal)
        repair = cast(dict[str, object], cast(dict[str, object], result["data"])["request_repair"])
        self.assertEqual(repair["effective_start_url"], "https://www.cafe-luise-baeckerei.de/")
        self.assertEqual(repair["original_start_url"], "https://www.cafe-luise.de/")
        self.assertIsNone(peek_pending_browser_follow_up_hint(runtime))

    def test_instruction_builders_only_add_browser_guidance_when_tool_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            disabled_env = root / ".env.disabled"
            disabled_env.write_text("", encoding="utf-8")
            disabled_config = TwinrConfig.from_env(disabled_env)

            workspace = root / "browser_automation"
            workspace.mkdir()
            (workspace / "adapter.py").write_text(
                "def create_browser_automation_driver(*, config, project_root):\n    return object()\n",
                encoding="utf-8",
            )
            enabled_env = root / ".env.enabled"
            enabled_env.write_text("TWINR_BROWSER_AUTOMATION_ENABLED=true\n", encoding="utf-8")
            enabled_config = TwinrConfig.from_env(enabled_env)

            disabled_tool = build_tool_agent_instructions(disabled_config)
            disabled_supervisor = build_supervisor_decision_instructions(disabled_config)
            enabled_tool = build_tool_agent_instructions(enabled_config)
            enabled_supervisor = build_supervisor_decision_instructions(enabled_config)

        self.assertNotIn("browser_automation", disabled_tool)
        self.assertNotIn("browser interaction on a specific website", disabled_supervisor)
        self.assertIn("browser_automation", enabled_tool)
        self.assertIn("Use search_live_info first", enabled_tool)
        self.assertIn("offer the user one short model-authored follow-up", enabled_tool)
        self.assertIn("different method", enabled_tool)
        self.assertIn("Do not turn that offer into a fixed stock sentence", enabled_tool)
        self.assertIn("treat that as unresolved rather than as the final answer", enabled_tool)
        self.assertIn("short assent or go-ahead", enabled_tool)
        self.assertIn("already explicitly asked Twinr to check", enabled_tool)
        self.assertIn("does not by itself count as explicit browser authorization", enabled_tool)
        self.assertIn("Do not tack on an extra website-check offer", enabled_tool)
        self.assertIn("prefer the short proactive alternate-method offer", enabled_tool)
        self.assertIn("may take a little longer", enabled_tool)
        self.assertIn("reuse that exact site_follow_up_url or site_follow_up_domain", enabled_tool)
        self.assertIn("Prefer a normal web-search handoff", enabled_supervisor)
        self.assertIn("proactively offer a different method", enabled_supervisor)
        self.assertIn("rather than a fixed canned permission sentence", enabled_supervisor)
        self.assertIn("should stay unresolved rather than being treated as a finished answer", enabled_supervisor)
        self.assertIn("short assent or go-ahead", enabled_supervisor)
        self.assertIn("does not by itself make browser work explicit", enabled_supervisor)
        self.assertIn("Do not turn a search answer", enabled_supervisor)
        self.assertIn("browser interaction on a specific website", enabled_supervisor)
        self.assertIn("report back shortly", enabled_supervisor)
        self.assertIn("reuse that exact site_follow_up_url or site_follow_up_domain", enabled_supervisor)


class VendoredBrowserLoopContractTests(unittest.TestCase):
    def test_vendored_tool_schema_exposes_inspect_and_tab_controls(self) -> None:
        registry = _TessairactToolRegistry()
        session_act = next(tool for tool in registry.list_tools() if tool["name"] == "browser.session_act")
        action_schema = session_act["input_schema"]["properties"]["actions"]["items"]
        action_enum = set(action_schema["properties"]["kind"]["enum"])

        self.assertIn("inspect_page", action_enum)
        self.assertIn("use_tab", action_enum)
        self.assertIn("tab_index", action_schema["properties"])
        self.assertIn("include_content_blocks", action_schema["properties"])
        self.assertIn("snapshot_include_content_blocks", session_act["input_schema"]["properties"])
        self.assertIn("snapshot_max_content_blocks", session_act["input_schema"]["properties"])

    def test_vendored_snapshot_trace_evidence_keeps_dense_page_tail_and_browser_state(self) -> None:
        specialist = _load_vendored_specialist_for_test()
        snapshot = {
            "title": "Dense Quotes Page",
            "text": "Lead evidence " + ("x" * 980) + " Andre Gide author evidence",
            "text_truncated": True,
            "elements": [
                {
                    "ref": "r7",
                    "role": "link",
                    "tag": "a",
                    "text": "Next",
                    "href": "https://quotes.toscrape.com/js/page/3/",
                }
            ],
            "content_blocks": [
                {
                    "source": "div",
                    "tag": "div",
                    "heading": "",
                    "secondary_text": "Andre Gide",
                    "text": "\"There is only one thing that makes a dream impossible to achieve\" Andre Gide",
                    "links": [{"text": "Andre Gide", "href": "https://quotes.toscrape.com/author/Andre-Gide"}],
                }
            ],
            "tabs": [
                {"tab_index": 0, "active": False, "title": "Quotes Start", "url": "https://quotes.toscrape.com/js/"},
                {"tab_index": 1, "active": True, "title": "Quotes Page 2", "url": "https://quotes.toscrape.com/js/page/2/"},
            ],
            "frames": [
                {"frame_index": 0, "active": True, "name": "main", "url": "https://quotes.toscrape.com/js/page/2/"}
            ],
            "tab_index": 1,
            "frame_index": 0,
        }

        evidence = specialist._snapshot_trace_evidence(snapshot)

        self.assertTrue(evidence["text_truncated"])
        self.assertIn("Andre Gide", evidence["text_excerpt"])
        self.assertEqual(evidence["content_blocks_excerpt"][0]["secondary_text"], "Andre Gide")
        self.assertEqual(evidence["tabs_excerpt"][1]["tab_index"], 1)
        self.assertTrue(evidence["tabs_excerpt"][1]["active"])
        self.assertEqual(evidence["frames_excerpt"][0]["frame_index"], 0)
        self.assertIn("Andre Gide", evidence["content_blocks_excerpt"][0]["text"])
        self.assertEqual(evidence["tab_index"], 1)
        self.assertEqual(evidence["frame_index"], 0)


class SameTabNavigationRegressionTests(unittest.TestCase):
    def test_same_tab_actions_refresh_final_url_after_settle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = _VendoredLoopRuntime(
                workspace_root=root / "browser_automation",
                task_token="same-tab-test",
                allowed_domains=("example.test",),
                default_capture_screenshot=False,
                default_capture_html=False,
                max_runtime_s=15.0,
            )

            session = SimpleNamespace(
                session_id="sess_demo",
                context=SimpleNamespace(pages=[]),
                page=SimpleNamespace(
                    url="https://example.test/docs/start",
                    keyboard=SimpleNamespace(press=lambda _key: None),
                ),
                trace_enabled=False,
                current_tab_index=0,
                current_frame_index=0,
                trace_zip_path=None,
            )

            class _FakeLocator:
                def scroll_into_view_if_needed(self, timeout: int | None = None) -> None:
                    del timeout

                def click(self, timeout: int | None = None) -> None:
                    del timeout

                def fill(self, _text: str, timeout: int | None = None) -> None:
                    del timeout

                def press(self, _key: str, timeout: int | None = None) -> None:
                    del timeout

            def _settle(*, page: object, timeout_ms: int) -> None:
                del timeout_ms
                cast(Any, page).url = "https://example.test/docs/dark-mode"

            stale_action_result = {
                "opened_new_tab": False,
                "tab_index": 0,
                "final_url": "https://example.test/docs/start",
            }

            for action in (
                {"kind": "click_ref", "ref": "r1"},
                {"kind": "type_ref", "ref": "r1", "text": "dark mode", "submit": True},
                {"kind": "press_key", "key": "Enter"},
            ):
                with self.subTest(kind=action["kind"]):
                    session.page.url = "https://example.test/docs/start"
                    with (
                        mock.patch.object(runtime, "_sync_current_page", return_value=session.page),
                        mock.patch.object(runtime, "_resolve_frame", return_value=object()),
                        mock.patch.object(runtime, "_frame_locator", return_value=_FakeLocator()),
                        mock.patch.object(runtime, "_context_pages", return_value=[session.page]),
                        mock.patch.object(runtime, "_promote_new_page_if_any", return_value=dict(stale_action_result)),
                        mock.patch.object(runtime, "_settle_page", side_effect=_settle),
                    ):
                        result = cast(dict[str, Any], runtime._execute_action(session=cast(Any, session), action=action))

                    self.assertEqual(result["final_url"], "https://example.test/docs/dark-mode")


class BrowserDecisionModuleTests(unittest.TestCase):
    def test_local_host_safety_flags_allow_explicit_loopback(self) -> None:
        self.assertEqual(
            _local_host_safety_flags(allowed_domains=("127.0.0.1", "example.org")),
            (True, True),
        )

    def test_trace_navigation_state_prefers_active_browser_state_urls(self) -> None:
        raw_result = {
            "visited_urls": ["https://tailwindcss.com/docs/installation/using-vite"],
            "steps": [
                {
                    "final_url": "https://tailwindcss.com/docs/installation/using-vite",
                    "url": "https://tailwindcss.com/docs/installation/using-vite",
                    "tabs_excerpt": [
                        {
                            "tab_index": 0,
                            "active": True,
                            "url": "https://tailwindcss.com/docs/dark-mode",
                        }
                    ],
                    "frames_excerpt": [
                        {
                            "frame_index": 0,
                            "active": True,
                            "url": "https://tailwindcss.com/docs/dark-mode",
                        }
                    ],
                }
            ],
        }

        trace_state = derive_trace_navigation_state(
            raw_result=raw_result,
            start_url="https://tailwindcss.com/docs/installation/using-vite",
        )

        self.assertEqual(trace_state.final_url, "https://tailwindcss.com/docs/dark-mode")
        self.assertEqual(
            trace_state.visited_urls,
            (
                "https://tailwindcss.com/docs/installation/using-vite",
                "https://tailwindcss.com/docs/dark-mode",
            ),
        )
        self.assertEqual(trace_state.latest_url_source, "active_frame_url")

    def test_trace_navigation_state_ignores_truncated_active_urls(self) -> None:
        full_url = (
            "http://localhost:7770/"
            "6s-wireless-headphones-over-ear-noise-canceling-hi-fi-bass-foldable-stereo-"
            "wireless-kid-headsets-earbuds-with-built-in-mic-micro-sd-tf-fm-for-iphone-"
            "samsung-ipad-pc-black-gold.html"
        )
        raw_result = {
            "visited_urls": [full_url],
            "steps": [
                {
                    "final_url": full_url,
                    "url": full_url,
                    "tabs_excerpt": [
                        {
                            "tab_index": 0,
                            "active": True,
                            "url": "http://localhost:7770/6s-wireless-headphones-over-ear-noise-canceling…",
                        }
                    ],
                    "frames_excerpt": [
                        {
                            "frame_index": 0,
                            "active": True,
                            "url": "http://localhost:7770/6s-wireless-headphones-over-ear-noise-canceling%E2%80%A6",
                        }
                    ],
                }
            ],
        }

        trace_state = derive_trace_navigation_state(
            raw_result=raw_result,
            start_url=full_url,
        )

        self.assertEqual(trace_state.final_url, full_url)
        self.assertEqual(trace_state.latest_url_source, "final_url")

    def test_content_block_packets_from_evidence_prioritizes_and_dedupes_matches(self) -> None:
        packets = _content_block_packets_from_evidence(
            evidence_packets=[
                {
                    "kind": "section_activation",
                    "url": "http://localhost:7770/product",
                    "content_blocks": [
                        {
                            "heading": "Reviews",
                            "secondary_text": "Review by Catso",
                            "text": "The ear cups are way too small for adult sized ears.",
                            "links": [{"text": "Page 2", "href": "http://localhost:7770/product?p=2"}],
                        },
                        {
                            "heading": "Reviews",
                            "secondary_text": "Review by Catso",
                            "text": "The ear cups are way too small for adult sized ears.",
                            "links": [{"text": "Page 2", "href": "http://localhost:7770/product?p=2"}],
                        },
                        {
                            "heading": "Reviews",
                            "secondary_text": "Review by Anglebert Dinkherhump",
                            "text": "They feel too small around the ears during long workouts.",
                            "links": [],
                        },
                    ],
                },
                {
                    "kind": "scroll_window",
                    "url": "http://localhost:7770/product",
                    "content_blocks": [
                        {
                            "heading": "Specs",
                            "secondary_text": "",
                            "text": "Battery life is 18 hours.",
                            "links": [],
                        }
                    ],
                },
            ],
            priority_terms=("ear cups", "small", "reviews"),
            limit=3,
        )

        self.assertEqual(len(packets), 3)
        self.assertEqual(packets[0]["kind"], "content_block")
        self.assertEqual(packets[0]["source_packet_kind"], "section_activation")
        self.assertIn("Catso", packets[0]["secondary_text"])
        self.assertIn("Anglebert", packets[1]["secondary_text"])
        self.assertEqual(packets[2]["heading"], "Specs")

    def test_inline_api_packets_collect_framework_payloads(self) -> None:
        fake_page = SimpleNamespace(
            evaluate=lambda _script: {
                "description": "Modern docs page",
                "json_ld": ['{"@type":"TechArticle","headline":"Dark mode"}'],
                "next_data_script": '{"page":"/docs/dark-mode"}',
                "next_data_window": '{"props":{"pageProps":{"variant":"dark:"}}}',
                "nuxt_data": "",
            }
        )

        packets, used_sources = collect_inline_api_packets(
            page=cast(Any, fake_page),
            url="https://example.test/docs/dark-mode",
        )

        self.assertEqual(
            [packet["kind"] for packet in packets],
            ["meta_description", "structured_data", "next_data_script", "next_data_window"],
        )
        self.assertEqual(
            used_sources,
            ["meta_description", "json_ld", "next_data_script", "next_data_window"],
        )

    def test_same_url_completion_verifier_supports_negative_on_page_observation_tasks(self) -> None:
        fake_response = SimpleNamespace(
            output_text=json.dumps(
                {
                    "requires_other_page": False,
                    "observational_page_check": True,
                    "answer_scoped_to_current_page": True,
                    "reason": "The task is a visible-state check on the inspected page, and the answer stays limited to that page.",
                }
            )
        )
        fake_client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: fake_response))
        raw_result = {
            "ok": True,
            "done_reason": "Seite geprüft; keine Termin- oder Slot-Anzeige sichtbar",
            "answer_markdown": (
                "Auf der offiziellen Seite sind **keine freien Termine oder Buchungsslots sichtbar**. "
                "Es ist nur eine Kontakt-/Informationsseite mit Formular zu sehen.\n\n"
                "## Sources\n- https://www.beim-schlump.de/psychotherapie/\n"
            ),
            "key_points": [
                "Kein sichtbarer Kalender, keine freien Termine und keine Buchungsslots auf der Seite",
                "Nur Kontaktformular und Praxisinformationen vorhanden",
            ],
            "visited_urls": ["https://www.beim-schlump.de/psychotherapie/"],
            "steps": [
                {
                    "title": "Praxis am Schlump",
                    "text_excerpt": "Kontaktformular und Praxisinformationen ohne sichtbaren Terminbuchungskalender.",
                }
            ],
        }
        trace_state = derive_trace_navigation_state(
            raw_result=raw_result,
            start_url="https://www.beim-schlump.de/psychotherapie/",
        )
        with mock.patch("browser_automation.completion_verifier._default_client_factory", return_value=fake_client):
            verifier = SameUrlCompletionVerifier(config=SimpleNamespace(), model="gpt-5.4-mini")
            verdict = verifier.verify(
                goal=(
                    "Auf der offiziellen Termin- bzw. Kontaktseite der Praxis am Schlump prüfen, "
                    "ob heute freie Termine, Kalendertage oder Buchungsslots sichtbar sind. "
                    "Wenn keine Slots sichtbar sind, sage das klar."
                ),
                start_url="https://www.beim-schlump.de/psychotherapie/",
                raw_result=raw_result,
                trace_state=trace_state,
            )

        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertTrue(verdict["supported"])
        self.assertTrue(verdict["observational_page_check"])
        self.assertTrue(verdict["answer_scoped_to_current_page"])
        self.assertFalse(verdict["requires_other_page"])


class DenseReaderRescueTests(unittest.TestCase):
    def test_dense_reader_keeps_observational_negative_unresolved_until_classifier_runs(self) -> None:
        normalized = DensePageReader._normalize_freeform_verification(
            {
                "supported": False,
                "not_found": True,
                "observational_page_check": True,
                "answer_scoped_to_current_page": True,
                "answer_markdown": "Auf der aktuell sichtbaren Seite ist kein Mittagsmenü für heute erkennbar.",
                "key_points": ["kein Mittagsmenü für heute sichtbar"],
                "reason": "The inspected current page only shows general site information.",
            }
        )

        self.assertFalse(normalized["supported"])
        self.assertTrue(normalized["not_found"])
        self.assertTrue(normalized["observational_page_check"])
        self.assertTrue(normalized["answer_scoped_to_current_page"])

    def test_dense_reader_does_not_auto_promote_positive_scoped_observation(self) -> None:
        normalized = DensePageReader._normalize_freeform_verification(
            {
                "supported": False,
                "not_found": False,
                "observational_page_check": True,
                "answer_scoped_to_current_page": True,
                "answer_markdown": "Three Stars",
                "key_points": ["One review title is Three Stars."],
                "reason": "The answer stays scoped to the inspected current page, but the verifier did not prove a supported match.",
            }
        )

        self.assertFalse(normalized["supported"])
        self.assertFalse(normalized["not_found"])

    def test_dense_reader_observation_classifier_returns_structured_flags(self) -> None:
        reader = DensePageReader.__new__(DensePageReader)
        with mock.patch.object(
            reader,
            "_call_json",
            return_value={
                "observational_page_check": True,
                "supported_negative_observation": True,
                "reason": "The goal is a visible-state check and the inspected page supports a scoped negative answer.",
            },
        ):
            verdict = reader._classify_negative_observational_support(
                goal="Prüfe auf der sichtbaren Seite, ob ein Mittagsmenü sichtbar ist.",
                evidence_packets=(
                    {
                        "kind": "page_snapshot",
                        "url": "https://example.test/",
                        "text_excerpt": "Allgemeine Inhalte ohne Mittagsmenü.",
                    },
                ),
                verification={
                    "supported": False,
                    "not_found": True,
                    "observational_page_check": False,
                    "answer_scoped_to_current_page": True,
                    "answer_markdown": "",
                    "key_points": [],
                    "reason": "Die Seite zeigt kein Mittagsmenü.",
                },
            )

        self.assertTrue(verdict["observational_page_check"])
        self.assertTrue(verdict["supported_negative_observation"])

    def test_session_close_preserves_session_for_late_follow_up_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = _VendoredLoopRuntime(
                workspace_root=root,
                task_token="late_follow_up",
                allowed_domains=("example.com",),
                default_capture_screenshot=False,
                default_capture_html=False,
                max_runtime_s=30.0,
            )

            class _FakePage:
                def __init__(self) -> None:
                    self.closed = False

                def close(self) -> None:
                    self.closed = True

                def is_closed(self) -> bool:
                    return self.closed

            class _FakeContext:
                def __init__(self, page: _FakePage) -> None:
                    self._page = page
                    self.closed = False

                @property
                def pages(self) -> list[_FakePage]:
                    return [self._page]

                def storage_state(self, *, path: str) -> None:
                    Path(path).write_text('{"cookies":[],"origins":[]}', encoding="utf-8")

                def close(self) -> None:
                    self.closed = True

            page = _FakePage()
            context = _FakeContext(page)
            runtime._sessions["sess_test"] = _SessionState(
                session_id="sess_test",
                context=cast(Any, context),
                page=cast(Any, page),
                trace_enabled=False,
            )

            payload = runtime._browser_session_close(arguments={"session_id": "sess_test"})

            self.assertTrue(payload["ok"])
            self.assertIn("sess_test", runtime._sessions)
            self.assertIsNotNone(payload["storage_state_path"])
            self.assertTrue(Path(str(payload["storage_state_path"])).is_file())
            self.assertEqual(runtime.latest_storage_state_path, payload["storage_state_path"])

            runtime.close_all()

            self.assertTrue(page.closed)
            self.assertTrue(context.closed)

    def test_driver_passes_browser_context_metadata_to_dense_reader_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bootstrap_state = root / "shopping_state.json"
            bootstrap_state.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": False,
                        "error": "insufficient_evidence",
                        "visited_urls": [
                            "http://localhost:7770/",
                            "http://localhost:7770/sales/guest/form/",
                        ],
                        "steps": [{"final_url": "http://localhost:7770/sales/guest/form/"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            captured: dict[str, object] = {}

            class _UnsupportedReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=False,
                        answer_markdown="",
                        key_points=(),
                        reason="Missing order evidence.",
                        evidence_packets=(),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _UnsupportedReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="shopping_order_state_bootstrap",
                    goal="Get the status of my latest order.",
                    start_url="http://localhost:7770/",
                    allowed_domains=("localhost", "127.0.0.1"),
                    metadata={
                        "browser_context_storage_state_path": str(bootstrap_state),
                        "browser_context_extra_http_headers": {"X-Test-Login": "benchmark-user"},
                        "source_intent": "Get the status of my latest order.",
                        "results_schema": {"type": "array", "items": {"type": "string"}},
                    },
                )
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "failed")
        self.assertEqual(captured["storage_state_path"], str(bootstrap_state))
        self.assertEqual(captured["extra_http_headers"], {"X-Test-Login": "benchmark-user"})
        self.assertEqual(captured["goal"], "Get the status of my latest order.")
        self.assertEqual(captured["results_schema"], {"type": "array", "items": {"type": "string"}})

    def test_driver_completed_structured_answer_is_refined_by_dense_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bootstrap_state = root / "shopping_state.json"
            bootstrap_state.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": True,
                        "done_reason": "completed",
                        "answer_markdown": '{"results":["Catso","Dibbins"]}',
                        "key_points": ["Partial list from the page."],
                        "visited_urls": [
                            "http://localhost:7770/product.html",
                            "http://localhost:7770/product-reviews.html",
                        ],
                        "steps": [{"final_url": "http://localhost:7770/product-reviews.html"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            captured: dict[str, object] = {}

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight skipped in this test so completed-answer refinement can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown='{"results":["Catso","Dibbins","Anglebert Dinkherhump","Michelle Davis"]}',
                        key_points=("All four matching reviewers were visible on the review page.",),
                        reason="Dense reader extracted the full supported reviewer set.",
                        evidence_packets=(
                            {
                                "kind": "find_text",
                                "url": "http://localhost:7770/product-reviews.html",
                                "query": "small",
                                "text": "Catso ... Dibbins ... Anglebert Dinkherhump ... Michelle Davis",
                            },
                        ),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _FakeReader)
            driver._task_intent_classifier = cast(
                Any,
                SimpleNamespace(
                    classify=lambda **_kwargs: {
                        "auth_navigation_needed": False,
                        "task_profile": "generic_read",
                        "reason": "test_override",
                    }
                ),
            )

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="shopping_structured_refine",
                    goal="Return the reviewer names who mention that the ear cups are too small as JSON only.",
                    start_url="http://localhost:7770/product.html",
                    allowed_domains=("localhost", "127.0.0.1"),
                    metadata={
                        "task_kind": "auth_read",
                        "source_intent": "Return the reviewer names who mention that the ear cups are too small.",
                        "results_schema": {"type": "array", "items": {"type": "string"}},
                        "browser_context_storage_state_path": str(bootstrap_state),
                    },
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "http://localhost:7770/product-reviews.html")
        self.assertEqual(result.data["mode"], "tessairact_completed_plus_dense_reader")
        self.assertEqual(
            result.data["answer_markdown"],
            '{"results":["Catso","Dibbins","Anglebert Dinkherhump","Michelle Davis"]}',
        )
        self.assertEqual(
            captured["goal"],
            "Return the reviewer names who mention that the ear cups are too small.",
        )
        self.assertEqual(captured["storage_state_path"], str(bootstrap_state))
        self.assertEqual(captured["results_schema"], {"type": "array", "items": {"type": "string"}})
        self.assertTrue(result.data["verifier_decision"]["supported"])

    def test_dense_reader_finalize_structured_answer_uses_correction_pass(self) -> None:
        reader = object.__new__(DensePageReader)
        extract_mock = mock.Mock(
            return_value={
                "results": ["Catso", "Dibbins", "Michelle Davis"],
                "key_points": ["Initial partial candidate."],
                "reason": "Initial extraction missed one supported reviewer.",
            }
        )
        correct_mock = mock.Mock(
            return_value={
                "results": ["Catso", "Dibbins", "Anglebert Dinkherhump", "Michelle Davis"],
                "key_points": ["Corrected to the full supported reviewer set."],
                "reason": "Validation added the missing supported reviewer.",
            }
        )
        cast(Any, reader)._extract_structured_answer = extract_mock
        cast(Any, reader)._correct_structured_answer = correct_mock

        finalized = DensePageReader._finalize_structured_answer(
            reader,
            goal="Get name(s) of reviewer(s) who mention ear cups being small.",
            navigator_data={"answer_markdown": '{"results":["Catso","Dibbins","Michelle Davis"]}'},
            evidence_packets=(),
            results_schema={"type": "array", "items": {"type": "string"}},
        )

        self.assertEqual(
            finalized["results"],
            ["Catso", "Dibbins", "Anglebert Dinkherhump", "Michelle Davis"],
        )
        self.assertEqual(finalized["reason"], "Validation added the missing supported reviewer.")
        extract_mock.assert_called_once()
        correct_mock.assert_called_once()

    def test_driver_recovers_auth_task_with_auth_navigation_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bootstrap_state = root / "shopping_state.json"
            bootstrap_state.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": True,
                        "done_reason": "No supported evidence of the latest order status or arrival date was available.",
                        "answer_markdown": '{"results":[{"status":"unavailable","arrival_date":null}]}',
                        "visited_urls": [
                            "http://localhost:7770/",
                            "http://localhost:7770/customer/account/login/",
                        ],
                        "steps": [{"final_url": "http://localhost:7770/customer/account/login/"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            captured: dict[str, object] = {}

            class _FakeAuthRescue:
                def __init__(self, **_: object) -> None:
                    pass

                def run(self, **kwargs: object) -> object:
                    captured.update(kwargs)
                    return SimpleNamespace(
                        supported=True,
                        not_found=False,
                        answer_markdown='{"results":[{"status":"canceled","arrival_date":null}]}',
                        key_points=("Latest order is canceled and has no arrival date.",),
                        reason="Authenticated navigation rescue reached the order detail page.",
                        evidence_packets=(
                            {
                                "kind": "auth_navigation_snapshot",
                                "url": "http://localhost:7770/sales/order/view/order_id/123/",
                            },
                        ),
                        artifacts=(),
                        used_api_sources=(),
                        visited_urls=(
                            "http://localhost:7770/",
                            "http://localhost:7770/customer/account/",
                            "http://localhost:7770/sales/order/view/order_id/123/",
                        ),
                    )

            class _UnexpectedReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **_: object) -> DensePageReaderResult:
                    raise AssertionError("dense reader should not run before auth navigation rescue succeeds")

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._auth_navigation_rescue_factory = cast(Any, _FakeAuthRescue)
            driver._dense_reader_factory = cast(Any, _UnexpectedReader)
            driver._task_intent_classifier = cast(
                Any,
                SimpleNamespace(
                    classify=lambda **_kwargs: {
                        "auth_navigation_needed": True,
                        "task_profile": "account_read",
                        "reason": "test_override",
                    }
                ),
            )

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="auth_navigation_rescue",
                    goal="Get the status of my latest order and when will it arrive.",
                    start_url="http://localhost:7770/",
                    allowed_domains=("localhost", "127.0.0.1"),
                    metadata={
                        "task_kind": "auth_read",
                        "source_intent": "Get the status of my latest order and when will it arrive.",
                        "results_schema": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "arrival_date": {"type": "string"},
                                },
                            },
                        },
                        "browser_context_storage_state_path": str(bootstrap_state),
                    },
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["mode"], "tessairact_auth_navigation_rescue")
        self.assertEqual(result.final_url, "http://localhost:7770/sales/order/view/order_id/123/")
        self.assertEqual(captured["storage_state_path"], str(bootstrap_state))
        self.assertEqual(
            captured["goal"],
            "Get the status of my latest order and when will it arrive.",
        )
        self.assertTrue(result.data["verifier_decision"]["supported"])

    def test_driver_reader_storage_state_prefers_bootstrap_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bootstrap_state = root / "bootstrap_state.json"
            latest_state = root / "latest_state.json"
            bootstrap_state.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            latest_state.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")
            request = BrowserAutomationRequest(
                task_id="state-preference",
                goal="Read the latest order.",
                start_url="http://localhost:7770/",
                allowed_domains=("localhost",),
                metadata={"browser_context_storage_state_path": str(bootstrap_state)},
            )

            preferred = TessairactVendoredDriver._reader_storage_state_path(
                request=request,
                latest_storage_state_path=str(latest_state),
            )

        self.assertEqual(preferred, str(bootstrap_state))

    def test_driver_recovers_same_page_completion_with_dense_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": True,
                        "done_reason": "The start page did not fully prove the lunch menu claim.",
                        "answer_markdown": (
                            "Ich konnte auf der Startseite keinen aktuellen Hinweis auf ein Mittagsmenü "
                            "finden, war mir aber noch nicht sicher."
                        ),
                        "visited_urls": ["https://www.cafe-luise-baeckerei.de/"],
                        "steps": [{"final_url": "https://www.cafe-luise-baeckerei.de/"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            class _UnsupportedSameUrlVerifier:
                def verify(self, **_: object) -> dict[str, object]:
                    return {
                        "supported": False,
                        "requires_other_page": True,
                        "observational_page_check": True,
                        "answer_scoped_to_current_page": True,
                        "reason": "The completion needs one more targeted read on the current page before it can be trusted.",
                    }

            captured: dict[str, object] = {}

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight skipped in this test so same-page rescue can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown=(
                            "Auf der aktuell sichtbaren offiziellen Website ist kein sichtbares Mittagsmenü "
                            "oder Mittagstisch für heute erkennbar."
                        ),
                        key_points=("kein sichtbares Mittagsmenü auf der aktuell sichtbaren offiziellen Website",),
                        reason="The dense reader re-checked the current official page and supported a scoped negative observational answer.",
                        evidence_packets=(
                            {
                                "kind": "page_snapshot",
                                "url": "https://www.cafe-luise-baeckerei.de/",
                                "text_excerpt": "Startseite mit Öffnungszeiten und Neuigkeiten, aber ohne Mittagsmenü.",
                            },
                        ),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._same_url_completion_verifier = cast(Any, _UnsupportedSameUrlVerifier())
            driver._dense_reader_factory = cast(Any, _FakeReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="cafe_luise_same_page_rescue",
                    goal=(
                        "Prüfe auf der offiziellen Website von Café Luise in Hamburg, ob für heute ein "
                        "Mittagsmenü oder Mittagstisch veröffentlicht wurde. Wenn nichts Aktuelles sichtbar "
                        "ist, sage das klar und kurz."
                    ),
                    start_url="https://www.cafe-luise-baeckerei.de/",
                    allowed_domains=("cafe-luise-baeckerei.de", "www.cafe-luise-baeckerei.de"),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "https://www.cafe-luise-baeckerei.de/")
        self.assertEqual(captured["url"], "https://www.cafe-luise-baeckerei.de/")
        self.assertEqual(result.data["mode"], "tessairact_same_page_plus_dense_reader")
        self.assertTrue(result.data["verifier_decision"]["supported"])
        self.assertIn("kein sichtbares Mittagsmenü", result.data["answer_markdown"])

    def test_driver_recovers_fail_closed_navigation_with_dense_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": False,
                        "error": "insufficient_evidence",
                        "visited_urls": [
                            "https://react.dev/",
                            "https://react.dev/reference/react/useEffectEvent",
                        ],
                        "steps": [{"final_url": "https://react.dev/reference/react/useEffectEvent"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight skipped in this test so navigation rescue can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown="useEffectEvent lets you separate events from effects.",
                        key_points=("separate events from effects",),
                        reason="Reader captured the requested summary directly from the destination page.",
                        evidence_packets=(
                            {
                                "kind": "find_text",
                                "url": "https://react.dev/reference/react/useEffectEvent",
                                "query": "useEffectEvent",
                                "text": "useEffectEvent lets you separate events from effects.",
                            },
                        ),
                        artifacts=(
                            BrowserAutomationArtifact(
                                kind="reader_evidence",
                                path="artifacts/react-reader-evidence.json",
                                content_type="application/json",
                            ),
                        ),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _FakeReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="react_useeffectevent",
                    goal="Read the useEffectEvent summary.",
                    start_url="https://react.dev/",
                    allowed_domains=("react.dev",),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "https://react.dev/reference/react/useEffectEvent")
        self.assertEqual(result.data["mode"], "tessairact_navigation_plus_dense_reader")
        self.assertIn("separate events from effects", result.data["answer_markdown"])
        self.assertTrue(result.data["verifier_decision"]["supported"])
        self.assertTrue(any(artifact.kind == "reader_evidence" for artifact in result.artifacts))

    def test_driver_recovers_fail_closed_same_page_read_with_dense_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": False,
                        "error": "insufficient_evidence",
                        "visited_urls": [
                            "http://localhost:7770/product.html",
                        ],
                        "steps": [
                            {
                                "final_url": "http://localhost:7770/product.html",
                                "title": "Product details",
                                "text_excerpt": "Customer reviews mention multiple ear-cup sizes.",
                            }
                        ],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            captured: dict[str, object] = {}

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight skipped in this test so same-page fail-closed rescue can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown='{"results":["Catso","Dibbins"]}',
                        key_points=("Catso and Dibbins mention the ear cups being small.",),
                        reason="Dense reader confirmed the reviewer names directly from the current product page.",
                        evidence_packets=(
                            {
                                "kind": "find_text",
                                "url": "http://localhost:7770/product.html",
                                "query": "ear cups small",
                                "text": "Review by Catso ... very small ears ... Review by Dibbins ... ear cups MAY be ok for children ... way too small",
                            },
                        ),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _FakeReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="same_page_read_rescue",
                    goal="Get the reviewer names who mention the ear cups being small.",
                    start_url="http://localhost:7770/product.html",
                    allowed_domains=("localhost", "127.0.0.1"),
                    metadata={"task_kind": "read"},
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "http://localhost:7770/product.html")
        self.assertEqual(captured["url"], "http://localhost:7770/product.html")
        self.assertEqual(result.data["mode"], "tessairact_navigation_plus_dense_reader")
        self.assertEqual(result.data["answer_markdown"], '{"results":["Catso","Dibbins"]}')
        self.assertTrue(result.data["verifier_decision"]["supported"])

    def test_driver_keeps_fail_closed_result_when_dense_reader_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": False,
                        "error": "insufficient_evidence",
                        "visited_urls": ["https://tailwindcss.com/docs/dark-mode"],
                        "steps": [{"final_url": "https://tailwindcss.com/docs/dark-mode"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            class _UnsupportedReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **_: object) -> DensePageReaderResult:
                    return DensePageReaderResult(
                        supported=False,
                        answer_markdown="",
                        key_points=(),
                        reason="The targeted dark-mode class did not appear in the evidence packets.",
                        evidence_packets=(),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _UnsupportedReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="tailwind_darkmode",
                    goal="Read the manual dark mode class.",
                    start_url="https://tailwindcss.com/docs/installation/using-vite",
                    allowed_domains=("tailwindcss.com",),
                )
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "browser_loop_failed")
        self.assertEqual(result.final_url, "https://tailwindcss.com/docs/dark-mode")

    def test_driver_maps_dense_reader_not_found_to_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": False,
                        "error": "insufficient_evidence",
                        "visited_urls": ["http://localhost:7770/product.html"],
                        "steps": [{"final_url": "http://localhost:7770/product.html"}],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            class _NotFoundReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight skipped in this test so not-found rescue can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    return DensePageReaderResult(
                        supported=False,
                        not_found=True,
                        answer_markdown="",
                        key_points=(),
                        reason="The inspected review list does not contain any 2-star-or-below review titles.",
                        evidence_packets=(),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _NotFoundReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="review_not_found",
                    goal="Get all review titles with 2 stars or below for the product on the current page.",
                    start_url="http://localhost:7770/product.html",
                    allowed_domains=("localhost", "127.0.0.1"),
                    metadata={"task_kind": "read"},
                )
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "not_found")
        self.assertIn("not_found:", result.summary)
        self.assertTrue(result.data["reader_rescue"]["not_found"])
        self.assertTrue(result.data["verifier_decision"]["not_found"])

    def test_driver_allows_dense_reader_rescue_after_returning_to_start_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_BROWSER_AUTOMATION_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            driver = TessairactVendoredDriver(config=config, workspace_root=root / "browser_automation")

            class _FakeSpecialist:
                BrowserLoopConfig = staticmethod(lambda **kwargs: dict(kwargs))

                @staticmethod
                def run_browser_loop(**_: object) -> dict[str, object]:
                    return {
                        "ok": False,
                        "error": "insufficient_evidence",
                        "visited_urls": [
                            "https://www.cafe-luise-baeckerei.de/",
                            "https://www.cafe-luise-baeckerei.de/neuigkeiten",
                            "https://www.cafe-luise-baeckerei.de/neuigkeiten/die-nordreportage-einblicke-backstube",
                        ],
                        "steps": [
                            {"final_url": "https://www.cafe-luise-baeckerei.de/neuigkeiten"},
                            {"final_url": "https://www.cafe-luise-baeckerei.de/"},
                        ],
                    }

            class _FakeLoader:
                def load(self) -> object:
                    return _FakeSpecialist()

            captured: dict[str, object] = {}

            class _FakeReader:
                def __init__(self, **_: object) -> None:
                    pass

                def read(self, **kwargs: object) -> DensePageReaderResult:
                    if bool(kwargs.get("document_fast_path_only")):
                        return DensePageReaderResult(
                            supported=False,
                            answer_markdown="",
                            key_points=(),
                            reason="Preflight skipped in this test so return-to-start rescue can be exercised.",
                            evidence_packets=(),
                            artifacts=(),
                        )
                    captured.update(kwargs)
                    return DensePageReaderResult(
                        supported=True,
                        answer_markdown=(
                            "Auf der aktuell sichtbaren offiziellen Website ist kein Mittagsmenü oder Mittagstisch "
                            "für heute erkennbar."
                        ),
                        key_points=("kein aktueller Hinweis auf ein heutiges Mittagsangebot sichtbar",),
                        reason="The inspected official page supports a negative observational answer scoped to the current page.",
                        evidence_packets=(
                            {
                                "kind": "page_snapshot",
                                "url": "https://www.cafe-luise-baeckerei.de/",
                                "text_excerpt": "Startseite mit Öffnungszeiten und Neuigkeiten, aber ohne Mittagsmenü.",
                            },
                        ),
                        artifacts=(),
                    )

            driver._specialist_loader = cast(Any, _FakeLoader())
            driver._dense_reader_factory = cast(Any, _FakeReader)

            result = driver.execute(
                BrowserAutomationRequest(
                    task_id="cafe_luise_startpage_observation",
                    goal=(
                        "Prüfe auf der offiziellen Website von Café Luise in Hamburg, ob für heute ein "
                        "Mittagsmenü oder Mittagstisch veröffentlicht wurde. Wenn nichts Aktuelles sichtbar "
                        "ist, sage das klar und kurz."
                    ),
                    start_url="https://www.cafe-luise-baeckerei.de/",
                    allowed_domains=("cafe-luise-baeckerei.de", "www.cafe-luise-baeckerei.de"),
                )
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_url, "https://www.cafe-luise-baeckerei.de/")
        self.assertEqual(captured["url"], "https://www.cafe-luise-baeckerei.de/")
        self.assertEqual(result.data["mode"], "tessairact_navigation_plus_dense_reader")
        self.assertTrue(result.data["verifier_decision"]["supported"])
        self.assertIn("kein Mittagsmenü", result.data["answer_markdown"])


if __name__ == "__main__":
    unittest.main()
