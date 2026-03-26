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
from browser_automation.dense_reader import DensePageReaderResult
from browser_automation.tessairact_bridge import (
    TessairactVendoredDriver,
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
        self.assertIn("asking the user", schema["description"])
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
        self.assertIn("ask the user one short follow-up question", enabled_tool)
        self.assertIn("treat that as unresolved rather than as the final answer", enabled_tool)
        self.assertIn("short assent or go-ahead", enabled_tool)
        self.assertIn("already explicitly asked Twinr to check", enabled_tool)
        self.assertIn("does not by itself count as explicit browser authorization", enabled_tool)
        self.assertIn("Do not tack on an extra website-check offer", enabled_tool)
        self.assertIn("prefer the short permission question", enabled_tool)
        self.assertIn("may take a little longer", enabled_tool)
        self.assertIn("reuse that exact site_follow_up_url or site_follow_up_domain", enabled_tool)
        self.assertIn("Prefer a normal web-search handoff", enabled_supervisor)
        self.assertIn("ask the user for permission", enabled_supervisor)
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

                def read(self, **_: object) -> DensePageReaderResult:
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
