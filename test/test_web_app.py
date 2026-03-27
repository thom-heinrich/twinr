from array import array
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from typing import cast
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings

from fastapi.testclient import TestClient
from test.self_coding_test_utils import stable_sha256

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveTimingStore
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore
from twinr.automations import (
    AutomationAction,
    AutomationDefinition,
    AutomationStore,
    IfThenAutomationTrigger,
    TimeAutomationTrigger,
    build_sensor_trigger,
)
from twinr.agent.self_coding.contracts import (
    ActivationRecord,
    CompileJobRecord,
    CompileRunStatusRecord,
    LiveE2EStatusRecord,
    SkillHealthRecord,
)
from twinr.agent.self_coding.status import CompileJobStatus, CompileTarget, LearnedSkillStatus
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.memory.longterm.evaluation.live_midterm_attest import (
    LiveMidtermAttestResult,
    LiveMidtermSeedTurn,
    write_live_midterm_attest_artifacts,
)
from twinr.memory.longterm.retrieval.operator_search import LongTermOperatorSearchResult
from twinr.memory.longterm.core.models import (
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.memory.reminders import ReminderStore
from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry
from twinr.integrations import (
    HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY,
    ManagedIntegrationConfig,
    TwinrIntegrationStore,
    hue_application_key_env_key_for_host,
)
from twinr.integrations.email.connectivity import EmailConnectionTestResult, EmailTransportProbe
from twinr.ops import DeviceFact, DeviceOverview, DeviceStatus, TwinrOpsEventStore, resolve_ops_paths
from twinr.ops.remote_memory_watchdog import (
    RemoteMemoryWatchdogSample,
    RemoteMemoryWatchdogSnapshot,
    RemoteMemoryWatchdogStore,
)
from twinr.web.app import create_app
from twinr.web.support.channel_onboarding import ChannelPairingSnapshot

_TEST_WHATSAPP_ALLOW_FROM = "+15555554567"
_TEST_WHATSAPP_ALLOW_FROM_DISPLAY = "+1 555 555 4567"
_TEST_WEB_HOST = "192.0.2.10"


class _WarningQuietTestClient(TestClient):
    """Suppress the known Task/Future cancel deprecation noise during requests."""

    def request(self, *args, **kwargs):  # type: ignore[override]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing 'msg' argument to Task\.cancel\(\) is deprecated since Python 3\.11, and scheduled for removal in Python 3\.14\.",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Passing 'msg' argument to Future\.cancel\(\) is deprecated since Python 3\.11, and scheduled for removal in Python 3\.14\.",
                category=DeprecationWarning,
            )
            return super().request(*args, **kwargs)


def _voice_sample_wav_bytes(*, frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 16000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    buffer = io.BytesIO()
    # Pylint resolves this stdlib writer handle as ``Wave_read`` even in ``wb`` mode.
    # pylint: disable=no-member
    with cast(wave.Wave_write, wave.open(buffer, "wb")) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())
    # pylint: enable=no-member
    return buffer.getvalue()


def _write_whatsapp_worker_package(project_root: Path) -> Path:
    worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
    worker_root.mkdir(parents=True, exist_ok=True)
    (worker_root / "package.json").write_text("{\"name\": \"worker\"}\n", encoding="utf-8")
    (worker_root / "index.mjs").write_text("console.log('worker');\n", encoding="utf-8")
    return worker_root


class WebAppTests(unittest.TestCase):
    def make_client(
        self,
        *,
        extra_env: dict[str, str] | None = None,
        base_url: str = "http://localhost",
        client_host: str = "127.0.0.1",
    ) -> tuple[TestClient, Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        env_path = root / ".env"
        personality_dir = root / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        env_lines = [
            "OPENAI_MODEL=gpt-5.4-mini",
            "OPENAI_API_KEY=sk-test-1234",
            "TWINR_WEB_HOST=0.0.0.0",
            "TWINR_WEB_PORT=1337",
            f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
        ]
        if extra_env:
            env_lines.extend(f"{key}={value}" for key, value in extra_env.items())
        env_path.write_text(
            "\n".join(env_lines) + "\n",
            encoding="utf-8",
        )
        (personality_dir / "SYSTEM.md").write_text("System text\n", encoding="utf-8")
        (personality_dir / "PERSONALITY.md").write_text("Personality text\n", encoding="utf-8")
        (personality_dir / "USER.md").write_text("User text\n", encoding="utf-8")
        return _WarningQuietTestClient(create_app(env_path), base_url=base_url, client=(client_host, 50000)), env_path

    def _mock_whatsapp_node_ready(self, version: str = "v20.20.1"):
        return patch(
            "twinr.web.support.whatsapp.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=f"{version}\n", stderr=""),
        )

    def test_dashboard_renders_summary(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Twinr", response.text)
        self.assertIn("Dashboard", response.text)
        self.assertIn("Reminders", response.text)
        self.assertIn("****1234", response.text)
        self.assertIn("Status and failures", response.text)

    def test_managed_web_auth_redirects_unauthenticated_requests_to_login(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WEB_REQUIRE_AUTH": "1",
                "TWINR_WEB_ALLOW_REMOTE": "1",
            }
        )

        response = client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/auth/login")
        self.assertTrue((env_path.parent / "state" / "web_auth.json").exists())

    def test_managed_web_auth_forces_password_change_after_bootstrap_login(self) -> None:
        client, _env_path = self.make_client(
            extra_env={
                "TWINR_WEB_REQUIRE_AUTH": "1",
                "TWINR_WEB_ALLOW_REMOTE": "1",
            }
        )

        response = client.post(
            "/auth/login",
            data={"username": "admin", "password": "admin", "next": "/ops/debug"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/auth/password")

        password_page = client.get("/auth/password")

        self.assertEqual(password_page.status_code, 200)
        self.assertIn("Set a new password", password_page.text)
        self.assertIn("at least 8 characters", password_page.text)

    def test_managed_web_auth_password_change_replaces_bootstrap_password(self) -> None:
        client, _env_path = self.make_client(
            extra_env={
                "TWINR_WEB_REQUIRE_AUTH": "1",
                "TWINR_WEB_ALLOW_REMOTE": "1",
            }
        )

        client.post(
            "/auth/login",
            data={"username": "admin", "password": "admin", "next": "/"},
            follow_redirects=False,
        )

        response = client.post(
            "/auth/password",
            data={
                "current_password": "admin",
                "new_password": "fresh-password-123",
                "confirm_password": "fresh-password-123",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/?saved=1")

        client.post("/auth/logout", follow_redirects=False)
        failed_login = client.post(
            "/auth/login",
            data={"username": "admin", "password": "admin", "next": "/"},
        )
        self.assertEqual(failed_login.status_code, 200)
        self.assertIn("not correct", failed_login.text)

        new_login = client.post(
            "/auth/login",
            data={"username": "admin", "password": "fresh-password-123", "next": "/"},
            follow_redirects=False,
        )
        self.assertEqual(new_login.status_code, 303)
        self.assertEqual(new_login.headers["location"], "/")

    def test_managed_web_auth_allows_remote_bootstrap_login_and_password_change(self) -> None:
        client, _env_path = self.make_client(
            extra_env={
                "TWINR_WEB_REQUIRE_AUTH": "1",
                "TWINR_WEB_ALLOW_REMOTE": "1",
                "TWINR_WEB_ALLOWED_HOSTS": "portal",
            },
            base_url="http://portal",
            client_host="192.168.1.44",
        )

        response = client.get("/auth/login")

        self.assertEqual(response.status_code, 200)
        self.assertIn("admin</strong> / <strong>admin", response.text)

        login = client.post(
            "/auth/login",
            data={"username": "admin", "password": "admin", "next": "/ops/debug"},
            headers={"origin": "http://portal"},
            follow_redirects=False,
        )

        self.assertEqual(login.status_code, 303)
        self.assertEqual(login.headers["location"], "/auth/password")

        password_change = client.post(
            "/auth/password",
            data={
                "current_password": "admin",
                "new_password": "remote-password-123",
                "confirm_password": "remote-password-123",
            },
            headers={"origin": "http://portal"},
            follow_redirects=False,
        )

        self.assertEqual(password_change.status_code, 303)
        self.assertEqual(password_change.headers["location"], "/?saved=1")

        dashboard = client.get("/")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Dashboard", dashboard.text)

    def test_whatsapp_runtime_step_accepts_https_proxy_same_origin_headers(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WEB_ALLOWED_HOSTS": _TEST_WEB_HOST,
            },
            base_url=f"http://{_TEST_WEB_HOST}",
            client_host="127.0.0.1",
        )

        response = client.post(
            "/connect/whatsapp",
            data={
                "_action": "save_runtime",
                "TWINR_WHATSAPP_NODE_BINARY": "/twinr/state/tools/node-v20.20.1-linux-arm64/bin/node",
                "TWINR_WHATSAPP_AUTH_DIR": "state/channels/whatsapp/auth",
            },
            headers={
                "host": _TEST_WEB_HOST,
                "origin": f"https://{_TEST_WEB_HOST}",
                "referer": f"https://{_TEST_WEB_HOST}/connect/whatsapp?step=runtime",
                "x-forwarded-proto": "https",
                "x-forwarded-host": _TEST_WEB_HOST,
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/connect/whatsapp?saved=1&step=pairing")
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn(
            "TWINR_WHATSAPP_NODE_BINARY=/twinr/state/tools/node-v20.20.1-linux-arm64/bin/node",
            env_text,
        )

    def test_dashboard_renders_self_coding_status_summary(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        store.save_compile_status(
            CompileRunStatusRecord(
                job_id="job_dashboard123",
                phase="streaming",
                driver_name="CodexSdkDriver",
                event_count=12,
                last_event_kind="assistant_delta",
                diagnostics={
                    "model": "gpt-5-codex",
                    "reasoning_effort": "high",
                    "duration_seconds": 67.4,
                    "fallback_reason": "CodexSdkDriver: timeout",
                },
            )
        )
        store.save_activation(
            ActivationRecord(
                skill_id="read_emails",
                skill_name="Read Emails",
                version=1,
                status=LearnedSkillStatus.ACTIVE,
                job_id="job_dashboard123",
                artifact_id="artifact_dashboard123",
                metadata={"automation_id": "ase_read_emails_v1"},
            )
        )
        store.save_activation(
            ActivationRecord(
                skill_id="announce_updates",
                skill_name="Announce Updates",
                version=1,
                status=LearnedSkillStatus.SOFT_LAUNCH_READY,
                job_id="job_dashboard124",
                artifact_id="artifact_dashboard124",
                metadata={"automation_id": "ase_announce_updates_v1"},
            )
        )
        store.save_live_e2e_status(
            LiveE2EStatusRecord(
                suite_id="morning_briefing",
                environment="local",
                status="passed",
                duration_seconds=67.4,
                model="gpt-5-codex",
                reasoning_effort="high",
                details="26 passed",
            )
        )

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Self-coding", response.text)
        self.assertIn("1 active", response.text)
        self.assertIn("1 soft launch", response.text)
        self.assertIn("assistant_delta", response.text)
        self.assertIn("gpt-5-codex", response.text)
        self.assertIn("live e2e passed", response.text.lower())

    def test_ops_self_coding_page_renders_telemetry_and_controls(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        store.save_compile_status(
            CompileRunStatusRecord(
                job_id="job_self_coding_ops",
                phase="completed",
                driver_name="CodexSdkDriver",
                event_count=42,
                last_event_kind="turn_completed",
                diagnostics={
                    "model": "gpt-5-codex",
                    "reasoning_effort": "high",
                    "duration_seconds": 67.4,
                    "fallback_reason": "CodexSdkDriver: timeout",
                },
            )
        )
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=1,
                status=LearnedSkillStatus.PAUSED,
                job_id="job_self_coding_ops_v1",
                artifact_id="artifact_self_coding_ops_v1",
                metadata={
                    "artifact_kind": "skill_package",
                    "automation_ids": ["ase_morning_briefing_v1_schedule", "ase_morning_briefing_v1_sensor"],
                },
            )
        )
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=2,
                status=LearnedSkillStatus.ACTIVE,
                job_id="job_self_coding_ops_v2",
                artifact_id="artifact_self_coding_ops_v2",
                metadata={
                    "artifact_kind": "skill_package",
                    "automation_ids": ["ase_morning_briefing_v2_schedule", "ase_morning_briefing_v2_sensor"],
                },
            )
        )
        store.save_skill_health(
            SkillHealthRecord(
                skill_id="morning_briefing",
                version=1,
                status="healthy",
                trigger_count=5,
                delivered_count=2,
                error_count=1,
            )
        )
        store.save_live_e2e_status(
            LiveE2EStatusRecord(
                suite_id="morning_briefing",
                environment="local",
                status="passed",
                duration_seconds=67.4,
                model="gpt-5-codex",
                reasoning_effort="high",
                details="26 passed",
            )
        )

        response = client.get("/ops/self-coding")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Self-coding operations", response.text)
        self.assertIn("gpt-5-codex", response.text)
        self.assertIn("67.4", response.text)
        self.assertIn("CodexSdkDriver: timeout", response.text)
        self.assertIn("Pause skill", response.text)
        self.assertIn("Reactivate skill", response.text)
        self.assertIn("Rollback skill", response.text)
        self.assertIn("Retest skill", response.text)
        self.assertIn("Clean up version", response.text)
        self.assertIn("Morning Briefing", response.text)

    def test_ops_self_coding_pause_and_reactivate_update_activation_status(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=1,
                status=LearnedSkillStatus.ACTIVE,
                job_id="job_self_coding_toggle",
                artifact_id="artifact_self_coding_toggle",
                metadata={"automation_ids": ["ase_morning_briefing_v1"]},
            )
        )
        automation_store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        automation_store.upsert(
            AutomationDefinition(
                automation_id="ase_morning_briefing_v1",
                name="Morning Briefing",
                description="Read the briefing aloud.",
                enabled=True,
                trigger=IfThenAutomationTrigger(event_name="camera.person_visible"),
                actions=(AutomationAction(kind="say", text="Hallo"),),
                source="test",
                tags=("self_coding",),
            )
        )

        pause_response = client.post(
            "/ops/self-coding/pause",
            data={"skill_id": "morning_briefing", "version": "1"},
            follow_redirects=False,
        )
        paused = store.load_activation("morning_briefing", version=1)

        reactivate_response = client.post(
            "/ops/self-coding/reactivate",
            data={"skill_id": "morning_briefing", "version": "1"},
            follow_redirects=False,
        )
        active = store.load_activation("morning_briefing", version=1)

        self.assertEqual(pause_response.status_code, 303)
        self.assertEqual(paused.status, LearnedSkillStatus.PAUSED)
        self.assertEqual(reactivate_response.status_code, 303)
        self.assertEqual(active.status, LearnedSkillStatus.ACTIVE)

    def test_ops_self_coding_rollback_restores_previous_version(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=1,
                status=LearnedSkillStatus.PAUSED,
                job_id="job_self_coding_v1",
                artifact_id="artifact_self_coding_v1",
                metadata={"automation_ids": ["ase_morning_briefing_v1"]},
            )
        )
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=2,
                status=LearnedSkillStatus.ACTIVE,
                job_id="job_self_coding_v2",
                artifact_id="artifact_self_coding_v2",
                metadata={"automation_ids": ["ase_morning_briefing_v2"]},
            )
        )
        automation_store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        automation_store.upsert(
            AutomationDefinition(
                automation_id="ase_morning_briefing_v1",
                name="Morning Briefing v1",
                description="Old version.",
                enabled=False,
                trigger=IfThenAutomationTrigger(event_name="camera.person_visible"),
                actions=(AutomationAction(kind="say", text="Hallo v1"),),
                source="test",
                tags=("self_coding",),
            )
        )
        automation_store.upsert(
            AutomationDefinition(
                automation_id="ase_morning_briefing_v2",
                name="Morning Briefing v2",
                description="New version.",
                enabled=True,
                trigger=IfThenAutomationTrigger(event_name="camera.person_visible"),
                actions=(AutomationAction(kind="say", text="Hallo v2"),),
                source="test",
                tags=("self_coding",),
            )
        )

        response = client.post(
            "/ops/self-coding/rollback",
            data={"skill_id": "morning_briefing"},
            follow_redirects=False,
        )

        restored = store.load_activation("morning_briefing", version=1)
        paused = store.load_activation("morning_briefing", version=2)
        restored_automation = automation_store.get("ase_morning_briefing_v1")
        paused_automation = automation_store.get("ase_morning_briefing_v2")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(restored.status, LearnedSkillStatus.ACTIVE)
        self.assertEqual(paused.status, LearnedSkillStatus.PAUSED)
        self.assertIsNotNone(restored_automation)
        self.assertIsNotNone(paused_automation)
        assert restored_automation is not None
        assert paused_automation is not None
        self.assertTrue(restored_automation.enabled)
        self.assertFalse(paused_automation.enabled)

    def test_ops_self_coding_cleanup_retires_version(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=1,
                status=LearnedSkillStatus.PAUSED,
                job_id="job_self_coding_cleanup",
                artifact_id="artifact_self_coding_cleanup",
                metadata={"automation_ids": ["ase_morning_briefing_v1"]},
            )
        )
        automation_store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        automation_store.upsert(
            AutomationDefinition(
                automation_id="ase_morning_briefing_v1",
                name="Morning Briefing v1",
                description="Old version.",
                enabled=False,
                trigger=IfThenAutomationTrigger(event_name="camera.person_visible"),
                actions=(AutomationAction(kind="say", text="Hallo"),),
                source="test",
                tags=("self_coding",),
            )
        )

        response = client.post(
            "/ops/self-coding/cleanup",
            data={"skill_id": "morning_briefing", "version": "1"},
            follow_redirects=False,
        )

        cleaned = store.load_activation("morning_briefing", version=1)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(cleaned.status, LearnedSkillStatus.RETIRED)
        self.assertIsNone(automation_store.get("ase_morning_briefing_v1"))

    def test_ops_self_coding_retest_records_capture_result(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        store.save_activation(
            ActivationRecord(
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                version=2,
                status=LearnedSkillStatus.ACTIVE,
                job_id="job_self_coding_retest",
                artifact_id="artifact_self_coding_retest",
                metadata={"artifact_kind": "skill_package", "automation_ids": ["ase_morning_briefing_v2"]},
            )
        )

        with patch("twinr.web.app.run_self_coding_skill_retest") as retest:
            response = client.post(
                "/ops/self-coding/retest",
                data={"skill_id": "morning_briefing", "version": "2"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        retest.assert_called_once()

    def test_ops_self_coding_page_renders_watchdog_rows(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        stale_at = datetime(2026, 3, 17, 7, 0, tzinfo=timezone.utc)
        store.save_job(
            CompileJobRecord(
                job_id="job_self_coding_stale",
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                status=CompileJobStatus.COMPILING,
                requested_target=CompileTarget.SKILL_PACKAGE,
                spec_hash=stable_sha256("spec_self_coding_stale"),
                created_at=stale_at,
                updated_at=stale_at,
            )
        )
        store.save_compile_status(
            CompileRunStatusRecord(
                job_id="job_self_coding_stale",
                phase="streaming",
                driver_name="CodexSdkDriver",
                event_count=8,
                last_event_kind="assistant_delta",
                started_at=stale_at,
                updated_at=stale_at,
                diagnostics={"timeout_reason": "compile stalled"},
            )
        )
        store.save_execution_run(
            run_id="run_self_coding_stale",
            run_kind="retest",
            skill_id="morning_briefing",
            version=2,
            status="running",
            started_at=stale_at,
            updated_at=stale_at,
            metadata={"environment": "web", "timeout_reason": "retest stalled"},
        )

        response = client.get("/ops/self-coding")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Watchdog", response.text)
        self.assertIn("job_self_coding_stale", response.text)
        self.assertIn("run_self_coding_stale", response.text)
        self.assertIn("Clean up stale compile", response.text)
        self.assertIn("Clean up stale run", response.text)

    def test_ops_self_coding_cleanup_watchdog_rows_updates_store(self) -> None:
        client, env_path = self.make_client()
        store = SelfCodingStore.from_project_root(env_path.parent)
        stale_at = datetime(2026, 3, 17, 7, 0, tzinfo=timezone.utc)
        store.save_job(
            CompileJobRecord(
                job_id="job_self_coding_cleanup_stale",
                skill_id="morning_briefing",
                skill_name="Morning Briefing",
                status=CompileJobStatus.COMPILING,
                requested_target=CompileTarget.SKILL_PACKAGE,
                spec_hash=stable_sha256("spec_self_coding_cleanup_stale"),
                created_at=stale_at,
                updated_at=stale_at,
            )
        )
        store.save_compile_status(
            CompileRunStatusRecord(
                job_id="job_self_coding_cleanup_stale",
                phase="streaming",
                driver_name="CodexSdkDriver",
                event_count=2,
                started_at=stale_at,
                updated_at=stale_at,
            )
        )
        store.save_execution_run(
            run_id="run_self_coding_cleanup_stale",
            run_kind="retest",
            skill_id="morning_briefing",
            version=2,
            status="running",
            started_at=stale_at,
            updated_at=stale_at,
            metadata={"environment": "web"},
        )

        compile_response = client.post(
            "/ops/self-coding/cleanup-compile",
            data={"job_id": "job_self_coding_cleanup_stale"},
            follow_redirects=False,
        )
        run_response = client.post(
            "/ops/self-coding/cleanup-run",
            data={"run_id": "run_self_coding_cleanup_stale"},
            follow_redirects=False,
        )

        compile_status = store.load_compile_status("job_self_coding_cleanup_stale")
        compile_job = store.load_job("job_self_coding_cleanup_stale")
        execution_run = store.load_execution_run("run_self_coding_cleanup_stale")

        self.assertEqual(compile_response.status_code, 303)
        self.assertEqual(run_response.status_code, 303)
        self.assertEqual(compile_status.phase, "aborted")
        self.assertEqual(compile_job.status, CompileJobStatus.FAILED)
        self.assertEqual(execution_run.status, "cleaned")
        self.assertEqual(execution_run.reason, "operator_cleanup")

    def test_connect_page_renders_inline_help(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/connect")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Open WhatsApp wizard", response.text)
        self.assertIn("field-tooltip", response.text)
        self.assertIn("The main OpenAI secret used for chat, speech, vision, and realtime requests.", response.text)
        self.assertIn("Controls which backend answers normal text questions.", response.text)
        self.assertIn("(?)", response.text)

    def test_whatsapp_wizard_renders_setup_flow(self) -> None:
        client, _env_path = self.make_client()

        with self._mock_whatsapp_node_ready():
            response = client.get("/connect/whatsapp")

        self.assertEqual(response.status_code, 200)
        self.assertIn("WhatsApp self-chat wizard", response.text)
        self.assertIn("Choose your own WhatsApp chat", response.text)
        self.assertIn("Prepare the worker runtime", response.text)
        self.assertIn("Pair WhatsApp once", response.text)
        self.assertIn("Manual fallback on the Pi", response.text)
        self.assertEqual(response.text.count("WhatsApp self-chat wizard"), 1)

    def test_whatsapp_wizard_post_saves_chat_step_and_guardrails(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/connect/whatsapp",
            data={
                "_action": "save_chat",
                "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM_DISPLAY,
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/connect/whatsapp?saved=1&step=runtime")
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn(f"TWINR_WHATSAPP_ALLOW_FROM={_TEST_WHATSAPP_ALLOW_FROM}", env_text)
        self.assertIn("TWINR_WHATSAPP_SELF_CHAT_MODE=true", env_text)
        self.assertIn("TWINR_WHATSAPP_GROUPS_ENABLED=false", env_text)

    def test_whatsapp_wizard_post_saves_runtime_step(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/connect/whatsapp",
            data={
                "_action": "save_runtime",
                "TWINR_WHATSAPP_NODE_BINARY": "/opt/node-v20/bin/node",
                "TWINR_WHATSAPP_AUTH_DIR": "state/channels/whatsapp/auth",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/connect/whatsapp?saved=1&step=pairing")
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("TWINR_WHATSAPP_NODE_BINARY=/opt/node-v20/bin/node", env_text)
        self.assertIn("TWINR_WHATSAPP_AUTH_DIR=state/channels/whatsapp/auth", env_text)

    def test_whatsapp_wizard_shows_paired_state_when_creds_exist(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM,
                "TWINR_WHATSAPP_SELF_CHAT_MODE": "1",
                "TWINR_WHATSAPP_GROUPS_ENABLED": "0",
                "TWINR_WHATSAPP_AUTH_DIR": "state/channels/whatsapp/auth",
            }
        )
        auth_dir = env_path.parent / "state" / "channels" / "whatsapp" / "auth"
        auth_dir.mkdir(parents=True, exist_ok=True)
        (auth_dir / "creds.json").write_text("{}", encoding="utf-8")
        _write_whatsapp_worker_package(env_path.parent)

        with self._mock_whatsapp_node_ready():
            response = client.get("/connect/whatsapp")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Stored linked-device session found", response.text)
        self.assertIn("Twinr is configured for one self-chat", response.text)
        self.assertIn("Paired", response.text)

    def test_whatsapp_wizard_post_starts_pairing_window_from_ui(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM,
                "TWINR_WHATSAPP_SELF_CHAT_MODE": "1",
                "TWINR_WHATSAPP_GROUPS_ENABLED": "0",
            }
        )
        _write_whatsapp_worker_package(env_path.parent)

        with self._mock_whatsapp_node_ready():
            with patch("twinr.web.app.WhatsAppPairingCoordinator.start_pairing", return_value=True) as start_mock:
                response = client.post(
                    "/connect/whatsapp",
                    data={"_action": "start_pairing"},
                    follow_redirects=False,
                )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/connect/whatsapp?step=pairing")
        start_mock.assert_called_once()

    def test_whatsapp_wizard_shows_live_qr_status_and_auto_refresh(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM,
                "TWINR_WHATSAPP_SELF_CHAT_MODE": "1",
                "TWINR_WHATSAPP_GROUPS_ENABLED": "0",
            }
        )
        _write_whatsapp_worker_package(env_path.parent)

        live_snapshot = ChannelPairingSnapshot(
            channel_id="whatsapp",
            phase="qr_needed",
            summary="QR needed",
            detail="A fresh QR is ready below.",
            running=True,
            qr_needed=True,
            qr_svg="<svg viewBox='0 0 10 10'></svg>",
            worker_ready=True,
            last_worker_detail="worker_ready",
            updated_at="2026-03-19T12:00:00+00:00",
            started_at="2026-03-19T11:59:00+00:00",
        )

        with self._mock_whatsapp_node_ready():
            with patch("twinr.web.app.WhatsAppPairingCoordinator.load_snapshot", return_value=live_snapshot):
                response = client.get("/connect/whatsapp")

        self.assertEqual(response.status_code, 200)
        self.assertIn('http-equiv="refresh"', response.text)
        self.assertIn('content="3"', response.text)
        self.assertIn("QR needed", response.text)
        self.assertIn("A fresh QR is ready below", response.text)
        self.assertIn("Scan this QR with WhatsApp", response.text)
        self.assertIn("data:image/svg+xml;base64,", response.text)
        self.assertIn("A pairing window is already active", response.text)

    def test_whatsapp_wizard_surfaces_auth_repair_needed(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM,
                "TWINR_WHATSAPP_SELF_CHAT_MODE": "1",
                "TWINR_WHATSAPP_GROUPS_ENABLED": "0",
                "TWINR_WHATSAPP_AUTH_DIR": "state/channels/whatsapp/auth",
            }
        )
        auth_dir = env_path.parent / "state" / "channels" / "whatsapp" / "auth"
        auth_dir.mkdir(parents=True, exist_ok=True)
        (auth_dir / "creds.json").write_text("{}", encoding="utf-8")
        _write_whatsapp_worker_package(env_path.parent)

        repair_snapshot = ChannelPairingSnapshot(
            channel_id="whatsapp",
            phase="failed",
            summary="Auth repair needed",
            detail="The stored WhatsApp session is no longer valid. Start a fresh pairing window and scan the next QR.",
            fatal=True,
            auth_repair_needed=True,
            worker_ready=True,
            last_worker_detail="badSession",
            last_reconnect_reason="badSession",
            updated_at="2026-03-19T12:00:00+00:00",
            finished_at="2026-03-19T12:00:00+00:00",
        )

        with self._mock_whatsapp_node_ready():
            with patch("twinr.web.app.WhatsAppPairingCoordinator.load_snapshot", return_value=repair_snapshot):
                response = client.get("/connect/whatsapp")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Needs repair", response.text)
        self.assertIn("Auth repair needed", response.text)
        self.assertIn("badSession", response.text)
        self.assertIn("Start fresh pairing window", response.text)

    def test_email_wizard_renders_setup_flow(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/integrations/email")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Email setup wizard", response.text)
        self.assertIn("Choose the mail provider", response.text)
        self.assertIn("Save the mailbox login", response.text)
        self.assertIn("Review the server settings", response.text)
        self.assertIn("Enable mail with clear guardrails", response.text)
        self.assertIn("United Domains", response.text)
        self.assertIn("iCloud Mail", response.text)
        self.assertIn("Outlook.com / Microsoft mail", response.text)
        self.assertIn("Run connection test", response.text)

    def test_email_wizard_post_saves_profile_step(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/integrations/email",
            data={
                "_action": "save_profile",
                "profile": "united_domains",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/integrations/email?saved=1&step=account")
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("email_mailbox")
        self.assertEqual(record.value("profile"), "united_domains")
        self.assertFalse(record.enabled)

    def test_email_wizard_post_saves_account_step_and_secret(self) -> None:
        client, env_path = self.make_client()
        store = TwinrIntegrationStore.from_project_root(env_path.parent)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=False,
                settings={"profile": "gmail"},
            )
        )

        response = client.post(
            "/integrations/email",
            data={
                "_action": "save_account",
                "account_email": "anna@gmail.com",
                "from_address": "anna@gmail.com",
                "TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/integrations/email?saved=1&step=transport")
        record = store.get("email_mailbox")
        self.assertEqual(record.value("account_email"), "anna@gmail.com")
        self.assertEqual(record.value("from_address"), "anna@gmail.com")
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("TWINR_INTEGRATION_EMAIL_APP_PASSWORD=abcdefghijklmnop", env_text)

    def test_email_wizard_post_saves_transport_step(self) -> None:
        client, env_path = self.make_client()
        store = TwinrIntegrationStore.from_project_root(env_path.parent)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=False,
                settings={"profile": "united_domains"},
            )
        )

        response = client.post(
            "/integrations/email",
            data={
                "_action": "save_transport",
                "imap_host": "",
                "imap_port": "",
                "imap_mailbox": "INBOX",
                "smtp_host": "",
                "smtp_port": "",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/integrations/email?saved=1&step=guardrails")
        record = store.get("email_mailbox")
        self.assertEqual(record.value("imap_host"), "imaps.udag.de")
        self.assertEqual(record.value("smtp_host"), "smtps.udag.de")
        self.assertEqual(record.value("imap_port"), "993")
        self.assertEqual(record.value("smtp_port"), "587")

    def test_email_wizard_post_saves_guardrails_and_enables_integration(self) -> None:
        client, env_path = self.make_client(
            extra_env={"TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcdefghijklmnop"}
        )
        store = TwinrIntegrationStore.from_project_root(env_path.parent)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=False,
                settings={
                    "profile": "gmail",
                    "account_email": "anna@gmail.com",
                    "from_address": "anna@gmail.com",
                    "imap_host": "imap.gmail.com",
                    "imap_port": "993",
                    "imap_mailbox": "INBOX",
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": "587",
                    "connection_test_status": "ok",
                    "connection_test_summary": "Connection test passed",
                    "connection_test_detail": "Twinr reached both servers.",
                    "connection_test_imap_status": "ok",
                    "connection_test_imap_summary": "Connected",
                    "connection_test_imap_detail": "IMAP worked.",
                    "connection_test_smtp_status": "ok",
                    "connection_test_smtp_summary": "Connected",
                    "connection_test_smtp_detail": "SMTP worked.",
                    "connection_test_tested_at": "2026-03-26T17:00:00+00:00",
                },
            )
        )

        response = client.post(
            "/integrations/email",
            data={
                "_action": "save_guardrails",
                "enabled": "true",
                "unread_only_default": "true",
                "restrict_reads_to_known_senders": "false",
                "restrict_recipients_to_known_contacts": "true",
                "known_contacts_text": "Anna <anna@gmail.com>",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/integrations/email?saved=1&step=guardrails")
        record = store.get("email_mailbox")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("restrict_recipients_to_known_contacts"), "true")
        self.assertEqual(record.value("known_contacts_text"), "Anna <anna@gmail.com>")

    def test_email_wizard_post_runs_connection_test_and_saves_redacted_result(self) -> None:
        client, env_path = self.make_client(
            extra_env={"TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcdefghijklmnop"}
        )
        store = TwinrIntegrationStore.from_project_root(env_path.parent)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=False,
                settings={
                    "profile": "gmail",
                    "account_email": "anna@gmail.com",
                    "from_address": "anna@gmail.com",
                    "imap_host": "imap.gmail.com",
                    "imap_port": "993",
                    "imap_mailbox": "INBOX",
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": "587",
                },
            )
        )

        with patch(
            "twinr.web.app.run_email_connectivity_test",
            return_value=EmailConnectionTestResult(
                status="ok",
                summary="Connection test passed",
                detail="Twinr reached both servers.",
                imap=EmailTransportProbe(status="ok", summary="Connected", detail="IMAP worked."),
                smtp=EmailTransportProbe(status="ok", summary="Connected", detail="SMTP worked."),
                tested_at="2026-03-26T18:00:00+00:00",
            ),
        ):
            response = client.post(
                "/integrations/email",
                data={
                    "_action": "run_connection_test",
                    "enabled": "false",
                    "unread_only_default": "true",
                    "restrict_reads_to_known_senders": "false",
                    "restrict_recipients_to_known_contacts": "true",
                    "known_contacts_text": "Anna <anna@gmail.com>",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/integrations/email?saved=1&step=guardrails")
        record = store.get("email_mailbox")
        self.assertEqual(record.value("connection_test_status"), "ok")
        self.assertEqual(record.value("connection_test_imap_summary"), "Connected")
        self.assertEqual(record.value("connection_test_smtp_summary"), "Connected")
        self.assertFalse(record.enabled)
        self.assertEqual(record.value("restrict_recipients_to_known_contacts"), "true")

    def test_email_wizard_requires_connection_test_before_enabling(self) -> None:
        client, env_path = self.make_client(
            extra_env={"TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcdefghijklmnop"}
        )
        store = TwinrIntegrationStore.from_project_root(env_path.parent)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=False,
                settings={
                    "profile": "gmail",
                    "account_email": "anna@gmail.com",
                    "from_address": "anna@gmail.com",
                    "imap_host": "imap.gmail.com",
                    "imap_port": "993",
                    "imap_mailbox": "INBOX",
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": "587",
                },
            )
        )

        response = client.post(
            "/integrations/email",
            data={
                "_action": "save_guardrails",
                "enabled": "true",
                "unread_only_default": "true",
                "restrict_reads_to_known_senders": "false",
                "restrict_recipients_to_known_contacts": "false",
                "known_contacts_text": "",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/integrations/email?error=", response.headers["location"])
        self.assertIn("Run+the+connection+test+successfully+before+enabling+email.", response.headers["location"])
        self.assertFalse(store.get("email_mailbox").enabled)

    def test_email_wizard_transport_change_disables_mail_until_retested(self) -> None:
        client, env_path = self.make_client()
        store = TwinrIntegrationStore.from_project_root(env_path.parent)
        store.save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=True,
                settings={
                    "profile": "gmail",
                    "account_email": "anna@gmail.com",
                    "from_address": "anna@gmail.com",
                    "imap_host": "imap.gmail.com",
                    "imap_port": "993",
                    "imap_mailbox": "INBOX",
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": "587",
                    "connection_test_status": "ok",
                    "connection_test_summary": "Connection test passed",
                    "connection_test_detail": "Twinr reached both servers.",
                    "connection_test_imap_status": "ok",
                    "connection_test_imap_summary": "Connected",
                    "connection_test_imap_detail": "IMAP worked.",
                    "connection_test_smtp_status": "ok",
                    "connection_test_smtp_summary": "Connected",
                    "connection_test_smtp_detail": "SMTP worked.",
                    "connection_test_tested_at": "2026-03-26T17:00:00+00:00",
                },
            )
        )

        response = client.post(
            "/integrations/email",
            data={
                "_action": "save_transport",
                "imap_host": "imap2.gmail.com",
                "imap_port": "993",
                "imap_mailbox": "INBOX",
                "smtp_host": "smtp.gmail.com",
                "smtp_port": "587",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/integrations/email?saved=1&step=guardrails")
        record = store.get("email_mailbox")
        self.assertFalse(record.enabled)
        self.assertEqual(record.value("imap_host"), "imap2.gmail.com")
        self.assertEqual(record.value("connection_test_status"), "")

    def test_integrations_page_renders_mail_and_calendar_flows(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/integrations")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Integration overview", response.text)
        self.assertIn("Email mailbox", response.text)
        self.assertIn("Open email wizard", response.text)
        self.assertIn("/integrations/email", response.text)
        self.assertIn("WhatsApp self-chat", response.text)
        self.assertIn("Open WhatsApp wizard", response.text)
        self.assertIn("/connect/whatsapp", response.text)
        self.assertIn("Save calendar integration", response.text)
        self.assertIn("Save smart-home integration", response.text)
        self.assertIn("Social-history learning", response.text)
        self.assertIn("Learn from my social media history", response.text)
        self.assertIn("Save and import now", response.text)
        self.assertIn("ICS file", response.text)
        self.assertIn("Philips Hue", response.text)

    def test_integrations_post_saves_social_history_config_and_queues_import(self) -> None:
        client, env_path = self.make_client(
            extra_env={
                "TWINR_WHATSAPP_ALLOW_FROM": _TEST_WHATSAPP_ALLOW_FROM,
                "TWINR_WHATSAPP_AUTH_DIR": "state/channels/whatsapp/auth",
                "TWINR_WHATSAPP_WORKER_ROOT": "src/twinr/channels/whatsapp/worker",
            }
        )
        auth_dir = env_path.parent / "state" / "channels" / "whatsapp" / "auth"
        worker_root = env_path.parent / "src" / "twinr" / "channels" / "whatsapp" / "worker"
        auth_dir.mkdir(parents=True, exist_ok=True)
        worker_root.mkdir(parents=True, exist_ok=True)
        (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
        (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")

        fake_queue = SimpleNamespace(
            submit_request=lambda **_kwargs: SimpleNamespace(request_id="hist-req-1")
        )

        with patch(
            "twinr.web.app.probe_whatsapp_runtime",
            return_value=SimpleNamespace(paired=True),
        ), patch(
            "twinr.web.app.WhatsAppHistoryImportQueue.from_twinr_config",
            return_value=fake_queue,
        ):
            response = client.post(
                "/integrations",
                data={
                    "_integration_id": "social_history_learning",
                    "_integration_action": "save_and_import_social_history",
                    "enabled": "true",
                    "source": "whatsapp",
                    "lookback_key": "6m",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("social_history_learning")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("source"), "whatsapp")
        self.assertEqual(record.value("lookback_key"), "6m")
        self.assertEqual(record.value("last_import_status"), "queued")
        self.assertEqual(record.value("last_import_request_id"), "hist-req-1")

    def test_integrations_post_saves_email_config_and_secret(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "email_mailbox",
                "enabled": "true",
                "profile": "gmail",
                "account_email": "anna@gmail.com",
                "from_address": "anna@gmail.com",
                "TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
                "imap_host": "",
                "imap_port": "",
                "imap_mailbox": "INBOX",
                "smtp_host": "",
                "smtp_port": "",
                "unread_only_default": "true",
                "restrict_reads_to_known_senders": "false",
                "restrict_recipients_to_known_contacts": "false",
                "known_contacts_text": "Anna <anna@gmail.com>",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("TWINR_INTEGRATION_EMAIL_APP_PASSWORD=abcdefghijklmnop", env_text)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("email_mailbox")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("imap_host"), "imap.gmail.com")
        self.assertEqual(record.value("smtp_host"), "smtp.gmail.com")
        self.assertEqual(record.value("account_email"), "anna@gmail.com")
        store_text = TwinrIntegrationStore.from_project_root(env_path.parent).path.read_text(encoding="utf-8")
        self.assertNotIn("abcd efgh ijkl mnop", store_text)

        response = client.get("/integrations")
        self.assertNotIn("abcd", response.text)
        self.assertNotIn("mnop", response.text)
        self.assertIn("Configured", response.text)
        self.assertIn("Open email wizard", response.text)
        self.assertIn("Stored separately in .env as the app password.", response.text)

    def test_integrations_post_saves_calendar_config(self) -> None:
        client, env_path = self.make_client()
        calendar_path = env_path.parent / "state" / "calendar.ics"
        calendar_path.parent.mkdir(parents=True, exist_ok=True)
        calendar_path.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "calendar_agenda",
                "enabled": "true",
                "source_kind": "ics_file",
                "source_value": "state/calendar.ics",
                "timezone": "Europe/Berlin",
                "default_upcoming_days": "5",
                "max_events": "10",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("calendar_agenda")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("source_value"), "state/calendar.ics")
        response = client.get("/integrations")
        self.assertIn("Ready", response.text)
        self.assertIn("state/calendar.ics", response.text)

    def test_integrations_post_saves_smart_home_hue_config_and_secret(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "smart_home_hub",
                "enabled": "true",
                "provider": "hue",
                "bridge_host": "192.168.1.20",
                "TWINR_INTEGRATION_HUE_APPLICATION_KEY": "local-hue-key",
                "verify_tls": "false",
                "request_timeout_s": "10",
                "event_timeout_s": "2",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("TWINR_INTEGRATION_HUE_APPLICATION_KEY=local-hue-key", env_text)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("smart_home_hub")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("provider"), "hue")
        self.assertEqual(record.value("bridge_host"), "192.168.1.20")
        self.assertEqual(record.value("event_timeout_s"), "2")
        store_text = TwinrIntegrationStore.from_project_root(env_path.parent).path.read_text(encoding="utf-8")
        self.assertNotIn("local-hue-key", store_text)

        response = client.get("/integrations")
        self.assertIn("Smart Home", response.text)
        self.assertIn("192.168.1.20", response.text)
        self.assertIn("Hue application key", response.text)

    def test_integrations_post_saves_multi_bridge_hue_config_and_host_specific_secrets(self) -> None:
        client, env_path = self.make_client()
        secondary_env_key = hue_application_key_env_key_for_host("192.168.1.21")
        primary_host_env_key = hue_application_key_env_key_for_host("192.168.1.20")

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "smart_home_hub",
                "enabled": "true",
                "provider": "hue",
                "bridge_host": "192.168.1.20",
                HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY: "192.168.1.21",
                "TWINR_INTEGRATION_HUE_APPLICATION_KEY": "primary-hue-key",
                secondary_env_key: "secondary-hue-key",
                "verify_tls": "false",
                "request_timeout_s": "10",
                "event_timeout_s": "2",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("TWINR_INTEGRATION_HUE_APPLICATION_KEY=primary-hue-key", env_text)
        self.assertIn(f"{primary_host_env_key}=primary-hue-key", env_text)
        self.assertIn(f"{secondary_env_key}=secondary-hue-key", env_text)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("smart_home_hub")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("provider"), "hue")
        self.assertEqual(record.value("bridge_host"), "192.168.1.20")
        self.assertEqual(record.value(HUE_ADDITIONAL_BRIDGE_HOSTS_SETTING_KEY), "192.168.1.21")
        store_text = TwinrIntegrationStore.from_project_root(env_path.parent).path.read_text(encoding="utf-8")
        self.assertNotIn("primary-hue-key", store_text)
        self.assertNotIn("secondary-hue-key", store_text)

        response = client.get("/integrations")
        self.assertIn("Additional bridge hosts", response.text)
        self.assertIn("192.168.1.21", response.text)
        self.assertIn("Hue application key for 192.168.1.21", response.text)

    def test_integrations_post_rejects_calendar_url_with_query_token(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "calendar_agenda",
                "enabled": "true",
                "source_kind": "ics_url",
                "source_value": "https://calendar.example.com/feed.ics?token=super-secret",
                "timezone": "Europe/Berlin",
                "default_upcoming_days": "5",
                "max_events": "10",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("error=", response.headers["location"])
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("calendar_agenda")
        self.assertFalse(record.enabled)
        if TwinrIntegrationStore.from_project_root(env_path.parent).path.exists():
            store_text = TwinrIntegrationStore.from_project_root(env_path.parent).path.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", store_text)

    def test_voice_profile_page_renders_status_and_actions(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/voice-profile")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live voice status", response.text)
        self.assertIn("Capture and enroll sample", response.text)
        self.assertIn("Verify now", response.text)
        self.assertIn("Reset profile", response.text)
        self.assertIn("No raw enrollment audio is stored", response.text)

    def test_voice_profile_post_enroll_verify_and_reset(self) -> None:
        client, env_path = self.make_client()
        sample = _voice_sample_wav_bytes()
        voice_store_path = env_path.parent / "state" / "voice_profile.json"

        with patch("twinr.web.app._capture_voice_profile_sample", return_value=sample):
            enroll_response = client.post("/voice-profile", data={"_action": "enroll"})
        self.assertTrue(voice_store_path.exists())

        with patch("twinr.web.app._capture_voice_profile_sample", return_value=sample):
            verify_response = client.post("/voice-profile", data={"_action": "verify"})
        reset_response = client.post("/voice-profile", data={"_action": "reset"})

        self.assertEqual(enroll_response.status_code, 200)
        self.assertIn("Profile updated", enroll_response.text)
        self.assertIn("No raw audio was kept.", enroll_response.text)

        self.assertEqual(verify_response.status_code, 200)
        self.assertIn("Likely user", verify_response.text)
        self.assertIn("Confidence", verify_response.text)

        self.assertEqual(reset_response.status_code, 200)
        self.assertIn("Profile reset", reset_response.text)
        self.assertFalse(voice_store_path.exists())

    def test_automations_page_renders_family_sections_and_forms(self) -> None:
        client, env_path = self.make_client()
        store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        store.create_time_automation(
            name="Morning weather",
            description="Speak the weather every morning.",
            schedule="daily",
            time_of_day="08:00",
            timezone_name="Europe/Berlin",
            actions=(
                AutomationAction(
                    kind="llm_prompt",
                    text="Tell Thom the morning weather in Schwarzenbek.",
                    payload={"delivery": "spoken", "allow_web_search": True},
                ),
            ),
        )
        sensor_trigger = build_sensor_trigger("vad_quiet", hold_seconds=30, cooldown_seconds=180)
        store.create_if_then_automation(
            name="Quiet room check-in",
            description="Offer help if the room stays quiet.",
            event_name=sensor_trigger.event_name,
            all_conditions=sensor_trigger.all_conditions,
            any_conditions=sensor_trigger.any_conditions,
            cooldown_seconds=sensor_trigger.cooldown_seconds,
            actions=(AutomationAction(kind="say", text="Ich bin weiter hier, falls du etwas brauchst."),),
        )

        response = client.get("/automations")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Automation families", response.text)
        self.assertIn("Scheduled", response.text)
        self.assertIn("Sensor-triggered", response.text)
        self.assertIn("Email Mailbox automations", response.text)
        self.assertIn("Calendar Agenda automations", response.text)
        self.assertIn("Integration not configured", response.text)
        self.assertIn("Morning weather", response.text)
        self.assertIn("Quiet room check-in", response.text)
        self.assertIn("Add scheduled automation", response.text)
        self.assertIn("Add sensor automation", response.text)
        self.assertIn("Tell Thom the morning weather in Schwarzenbek.", response.text)
        self.assertIn("room microphone has been quiet", response.text)

    def test_automations_page_shows_configured_integration_family_state(self) -> None:
        client, env_path = self.make_client()
        TwinrIntegrationStore.from_project_root(env_path.parent).save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=True,
                settings={"account_email": "anna@example.com"},
            )
        )

        response = client.get("/automations")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Email Mailbox automations", response.text)
        self.assertIn("Integration configured", response.text)

    def test_automations_post_creates_time_automation(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/automations",
            data={
                "_action": "save_time_automation",
                "automation_id": "",
                "name": "Daily headlines",
                "description": "Print the top headlines each morning.",
                "enabled": "true",
                "schedule": "daily",
                "due_at": "",
                "time_of_day": "08:00",
                "weekdays_text": "",
                "timezone_name": "Europe/Berlin",
                "tags_text": "news, morning",
                "delivery": "printed",
                "content_mode": "llm_prompt",
                "allow_web_search": "true",
                "content": "Print the top headlines of the day in short German bullet points.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").load_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        assert isinstance(entry.trigger, TimeAutomationTrigger)
        self.assertEqual(entry.name, "Daily headlines")
        self.assertEqual(entry.trigger.schedule, "daily")
        self.assertEqual(entry.trigger.time_of_day, "08:00")
        self.assertEqual(entry.actions[0].kind, "llm_prompt")
        self.assertEqual(entry.actions[0].payload["delivery"], "printed")
        self.assertTrue(entry.actions[0].payload["allow_web_search"])
        self.assertEqual(entry.tags, ("news", "morning"))

    def test_automations_post_creates_sensor_automation(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/automations",
            data={
                "_action": "save_sensor_automation",
                "automation_id": "",
                "name": "Welcome after motion",
                "description": "Greet after motion is seen.",
                "enabled": "true",
                "trigger_kind": "pir_motion_detected",
                "hold_seconds": "",
                "cooldown_seconds": "120",
                "tags_text": "sensor, welcome",
                "delivery": "spoken",
                "content_mode": "static_text",
                "allow_web_search": "false",
                "content": "Hallo, ich bin bereit.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").load_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        assert isinstance(entry.trigger, IfThenAutomationTrigger)
        self.assertEqual(entry.name, "Welcome after motion")
        self.assertEqual(entry.actions[0].kind, "say")
        self.assertEqual(entry.actions[0].text, "Hallo, ich bin bereit.")
        self.assertEqual(entry.trigger.event_name, "pir.motion_detected")
        self.assertEqual(entry.tags, ("sensor", "welcome"))

    def test_automations_post_toggles_and_deletes_automation(self) -> None:
        client, env_path = self.make_client()
        store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        entry = store.create_time_automation(
            name="Morning greeting",
            schedule="daily",
            time_of_day="09:00",
            timezone_name="Europe/Berlin",
            actions=(AutomationAction(kind="say", text="Guten Morgen."),),
            source="web_ui",
        )

        toggle_response = client.post(
            "/automations",
            data={"_action": "toggle_automation", "automation_id": entry.automation_id},
            follow_redirects=False,
        )
        self.assertEqual(toggle_response.status_code, 303)
        toggled = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").get(entry.automation_id)
        self.assertIsNotNone(toggled)
        assert toggled is not None
        self.assertFalse(toggled.enabled)

        delete_response = client.post(
            "/automations",
            data={"_action": "delete_automation", "automation_id": entry.automation_id},
            follow_redirects=False,
        )
        self.assertEqual(delete_response.status_code, 303)
        remaining = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").load_entries()
        self.assertEqual(remaining, ())

    def test_settings_post_updates_env(self) -> None:
        client, env_path = self.make_client()
        settings_response = client.get("/settings")
        self.assertEqual(settings_response.status_code, 200)

        response = client.post(
            "/settings",
            data={
                "OPENAI_MODEL": "gpt-4o-mini",
                "OPENAI_STT_MODEL": "whisper-1",
                "OPENAI_TTS_MODEL": "gpt-4o-mini-tts",
                "OPENAI_TTS_VOICE": "marin",
                "OPENAI_TTS_SPEED": "0.90",
                "OPENAI_REALTIME_MODEL": "gpt-4o-realtime-preview",
                "OPENAI_REALTIME_VOICE": "sage",
                "OPENAI_REALTIME_SPEED": "1.05",
                "TWINR_WEB_HOST": "127.0.0.1",
                "TWINR_WEB_PORT": "1440",
                "TWINR_SPEECH_PAUSE_MS": "900",
                "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S": "3.0",
                "TWINR_CONVERSATION_FOLLOW_UP_ENABLED": "true",
                "TWINR_AUDIO_SPEECH_THRESHOLD": "800",
                "TWINR_AUDIO_BEEP_FREQUENCY_HZ": "1200",
                "TWINR_AUDIO_BEEP_DURATION_MS": "180",
                "TWINR_PRINTER_HEADER_TEXT": "TWINR.com",
                "TWINR_PRINTER_LINE_WIDTH": "28",
                "TWINR_PRINTER_FEED_LINES": "3",
                "TWINR_PRINTER_QUEUE": "Thermal_GP58",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("OPENAI_MODEL=gpt-4o-mini", env_text)
        self.assertIn("OPENAI_TTS_SPEED=0.90", env_text)
        self.assertIn("OPENAI_REALTIME_SPEED=1.05", env_text)
        self.assertIn("TWINR_WEB_PORT=1440", env_text)
        self.assertIn("TWINR_PRINTER_LINE_WIDTH=28", env_text)

    def test_settings_page_renders_extended_sections_and_hover_help(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/settings")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Models and voices", response.text)
        self.assertIn("Search", response.text)
        self.assertIn("Camera and vision", response.text)
        self.assertIn("Proactive behavior", response.text)
        self.assertIn("Proactive timing", response.text)
        self.assertIn("Proactive sensitivity", response.text)
        self.assertIn("Voice gateway and audio capture", response.text)
        self.assertIn("Remote voice gateway", response.text)
        self.assertIn("Voice gateway websocket", response.text)
        self.assertIn("Remote ASR URL", response.text)
        self.assertIn("Remote ASR timeout (s)", response.text)
        self.assertIn("Buttons and motion sensor", response.text)
        self.assertIn("Display and printer", response.text)
        self.assertIn("Adaptive timing", response.text)
        self.assertIn("Observed patterns", response.text)
        self.assertIn("Configured baselines", response.text)
        self.assertIn("Reset learned timing", response.text)
        self.assertIn("field-tooltip", response.text)
        self.assertIn("How much image detail Twinr asks OpenAI to inspect.", response.text)
        self.assertIn("Optional speaking instructions sent with text-to-speech requests.", response.text)
        self.assertIn("TTS speed", response.text)
        self.assertIn("Realtime speed", response.text)
        self.assertIn(
            "Keep the live transcript-first remote voice gateway enabled for hands-free speech. Twinr uses only this remote path for live voice turns.",
            response.text,
        )
        self.assertIn("After this many quiet seconds without motion, the scene is treated as idle / low-motion.", response.text)

    def test_settings_page_renders_current_adaptive_timing_profile(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        store = AdaptiveTimingStore(config.adaptive_timing_store_path, config=config)
        store.record_no_speech_timeout(initial_source="button", follow_up=False)
        store.record_capture(
            initial_source="button",
            follow_up=False,
            speech_started_after_ms=1800,
            resumed_after_pause_count=1,
        )

        response = client.get("/settings")

        self.assertEqual(response.status_code, 200)
        self.assertIn("8.75 s", response.text)
        self.assertIn("1230 ms", response.text)
        self.assertIn("470 ms", response.text)
        self.assertIn("1 ok / 1 timeout", response.text)
        self.assertIn("Persistent learned timing profile on disk.", response.text)

    def test_settings_post_updates_extended_env_values(self) -> None:
        client, env_path = self.make_client()
        settings_response = client.get("/settings")
        self.assertEqual(settings_response.status_code, 200)

        response = client.post(
            "/settings",
            data={
                "OPENAI_REASONING_EFFORT": "high",
                "OPENAI_TTS_FORMAT": "mp3",
                "OPENAI_TTS_INSTRUCTIONS": "Speak warm, clear German.",
                "OPENAI_REALTIME_TRANSCRIPTION_MODEL": "whisper-1",
                "OPENAI_REALTIME_LANGUAGE": "de",
                "OPENAI_REALTIME_INPUT_SAMPLE_RATE": "16000",
                "OPENAI_REALTIME_INSTRUCTIONS": "Stay calm and concise.",
                "TWINR_OPENAI_ENABLE_WEB_SEARCH": "true",
                "TWINR_CONVERSATION_WEB_SEARCH": "always",
                "OPENAI_SEARCH_MODEL": "gpt-5.2-chat-latest",
                "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE": "high",
                "TWINR_OPENAI_WEB_SEARCH_COUNTRY": "DE",
                "TWINR_AUDIO_INPUT_DEVICE": "plughw:2,0",
                "TWINR_AUDIO_OUTPUT_DEVICE": "default",
                "TWINR_AUDIO_SAMPLE_RATE": "22050",
                "TWINR_AUDIO_CHANNELS": "1",
                "TWINR_AUDIO_CHUNK_MS": "120",
                "TWINR_AUDIO_PREROLL_MS": "450",
                "TWINR_AUDIO_SPEECH_START_CHUNKS": "2",
                "TWINR_AUDIO_START_TIMEOUT_S": "6.5",
                "TWINR_AUDIO_MAX_RECORD_SECONDS": "25",
                "TWINR_AUDIO_BEEP_VOLUME": "0.65",
                "TWINR_SEARCH_FEEDBACK_TONES_ENABLED": "false",
                "TWINR_CAMERA_DEVICE": "/dev/video2",
                "TWINR_CAMERA_WIDTH": "800",
                "TWINR_CAMERA_HEIGHT": "600",
                "TWINR_CAMERA_FRAMERATE": "25",
                "TWINR_CAMERA_INPUT_FORMAT": "mjpeg",
                "TWINR_CAMERA_FFMPEG_PATH": "/usr/bin/ffmpeg",
                "OPENAI_VISION_DETAIL": "high",
                "TWINR_VISION_REFERENCE_IMAGE": "/home/thh/reference-user.jpg",
                "TWINR_USER_DISPLAY_NAME": "Thom",
                "TWINR_PROACTIVE_ENABLED": "true",
                "TWINR_PROACTIVE_POLL_INTERVAL_S": "3.5",
                "TWINR_PROACTIVE_CAPTURE_INTERVAL_S": "7.0",
                "TWINR_PROACTIVE_LOW_MOTION_AFTER_S": "14.0",
                "TWINR_PROACTIVE_AUDIO_ENABLED": "true",
                "TWINR_PROACTIVE_AUDIO_DEVICE": "plughw:CARD=CameraB409241,DEV=0",
                "TWINR_PROACTIVE_AUDIO_SAMPLE_MS": "900",
                "TWINR_VOICE_ORCHESTRATOR_ENABLED": "true",
                "TWINR_VOICE_ORCHESTRATOR_WS_URL": "ws://voice-gateway.example/ws/orchestrator/voice",
                "TWINR_VOICE_ORCHESTRATOR_SHARED_SECRET": "voice-secret",
                "TWINR_VOICE_ORCHESTRATOR_WAKE_CANDIDATE_WINDOW_MS": "1700",
                "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL": "http://asr.example:18090",
                "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_TIMEOUT_S": "5.5",
                "TWINR_PROACTIVE_ATTENTION_WINDOW_S": "8.5",
                "TWINR_PROACTIVE_FLOOR_STILLNESS_S": "26.0",
                "TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD": "0.76",
                "TWINR_WEB_HOST": "127.0.0.1",
                "TWINR_WEB_PORT": "1441",
                "TWINR_PERSONALITY_DIR": "personality",
                "TWINR_RUNTIME_STATE_PATH": "/tmp/twinr-runtime-state.json",
                "TWINR_MEMORY_MARKDOWN_PATH": "/tmp/MEMORY.md",
                "TWINR_REMINDER_STORE_PATH": "/tmp/reminders.json",
                "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP": "true",
                "TWINR_GPIO_CHIP": "gpiochip4",
                "TWINR_GREEN_BUTTON_GPIO": "23",
                "TWINR_YELLOW_BUTTON_GPIO": "22",
                "TWINR_BUTTON_ACTIVE_LOW": "true",
                "TWINR_BUTTON_BIAS": "pull-up",
                "TWINR_BUTTON_DEBOUNCE_MS": "90",
                "TWINR_BUTTON_PROBE_LINES": "23,22,24",
                "TWINR_PIR_MOTION_GPIO": "17",
                "TWINR_PIR_ACTIVE_HIGH": "true",
                "TWINR_PIR_BIAS": "pull-down",
                "TWINR_DISPLAY_DRIVER": "waveshare_4in2_v2",
                "TWINR_DISPLAY_VENDOR_DIR": "hardware/display/vendor",
                "TWINR_DISPLAY_SPI_BUS": "0",
                "TWINR_DISPLAY_SPI_DEVICE": "1",
                "TWINR_DISPLAY_CS_GPIO": "8",
                "TWINR_DISPLAY_DC_GPIO": "25",
                "TWINR_DISPLAY_RESET_GPIO": "17",
                "TWINR_DISPLAY_BUSY_GPIO": "24",
                "TWINR_DISPLAY_WIDTH": "400",
                "TWINR_DISPLAY_HEIGHT": "300",
                "TWINR_DISPLAY_ROTATION_DEGREES": "270",
                "TWINR_DISPLAY_FULL_REFRESH_INTERVAL": "2",
                "TWINR_DISPLAY_POLL_INTERVAL_S": "0.8",
                "TWINR_PRINTER_QUEUE": "Thermal_GP58",
                "TWINR_PRINTER_DEVICE_URI": "usb://Acme/Printer",
                "TWINR_PRINT_BUTTON_COOLDOWN_S": "3.5",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("OPENAI_REASONING_EFFORT=high", env_text)
        self.assertIn('OPENAI_TTS_INSTRUCTIONS="Speak warm, clear German."', env_text)
        self.assertIn("TWINR_CONVERSATION_WEB_SEARCH=always", env_text)
        self.assertIn("TWINR_CAMERA_DEVICE=/dev/video2", env_text)
        self.assertIn("OPENAI_VISION_DETAIL=high", env_text)
        self.assertIn("TWINR_PROACTIVE_AUDIO_DEVICE=plughw:CARD=CameraB409241,DEV=0", env_text)
        self.assertIn("TWINR_PROACTIVE_LOW_MOTION_AFTER_S=14.0", env_text)
        self.assertIn("TWINR_VOICE_ORCHESTRATOR_ENABLED=true", env_text)
        self.assertIn(
            "TWINR_VOICE_ORCHESTRATOR_WS_URL=ws://voice-gateway.example/ws/orchestrator/voice",
            env_text,
        )
        self.assertIn("TWINR_VOICE_ORCHESTRATOR_WAKE_CANDIDATE_WINDOW_MS=1700", env_text)
        self.assertIn("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL=http://asr.example:18090", env_text)
        self.assertIn("TWINR_PROACTIVE_ATTENTION_WINDOW_S=8.5", env_text)
        self.assertIn("TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD=0.76", env_text)
        self.assertIn("TWINR_RUNTIME_STATE_PATH=/tmp/twinr-runtime-state.json", env_text)
        self.assertIn("TWINR_BUTTON_PROBE_LINES=23,22,24", env_text)
        self.assertIn("TWINR_PRINTER_DEVICE_URI=usb://Acme/Printer", env_text)

    def test_settings_post_resets_adaptive_timing_profile(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        store = AdaptiveTimingStore(config.adaptive_timing_store_path, config=config)
        store.record_no_speech_timeout(initial_source="button", follow_up=False)
        store.record_capture(
            initial_source="button",
            follow_up=False,
            speech_started_after_ms=2200,
            resumed_after_pause_count=1,
        )

        response = client.post(
            "/settings",
            data={"_action": "reset_adaptive_timing"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/settings?saved=1")
        reset_profile = store.current()
        default_profile = store.default_profile()
        self.assertEqual(reset_profile.button_start_timeout_s, default_profile.button_start_timeout_s)
        self.assertEqual(reset_profile.follow_up_start_timeout_s, default_profile.follow_up_start_timeout_s)
        self.assertEqual(reset_profile.speech_pause_ms, default_profile.speech_pause_ms)
        self.assertEqual(reset_profile.pause_grace_ms, default_profile.pause_grace_ms)
        self.assertEqual(reset_profile.button_success_count, 0)
        self.assertEqual(reset_profile.button_timeout_count, 0)
        self.assertEqual(reset_profile.pause_resume_count, 0)

    def test_memory_page_renders_live_snapshot(self) -> None:
        client, env_path = self.make_client()
        store = RuntimeSnapshotStore(env_path.parent / "runtime-state.json")
        PersistentMemoryMarkdownStore(env_path.parent / "state" / "MEMORY.md").remember(
            kind="appointment",
            summary="Arzttermin am Montag um 14 Uhr.",
            details="Bei Dr. Meyer in Hamburg.",
        )
        reminder_store = ReminderStore(env_path.parent / "state" / "reminders.json")
        reminder_store.schedule(
            due_at="2030-03-15T09:00",
            kind="medication",
            summary="An die Tabletten erinnern.",
            details="Nach dem Fruehstueck.",
        )
        delivered = reminder_store.schedule(
            due_at="2030-03-15T12:00",
            kind="appointment",
            summary="An den Arzttermin erinnern.",
            details="Dr. Meyer in Hamburg.",
        )
        reminder_store.mark_delivered(delivered.reminder_id, delivered_at=datetime(2026, 3, 13, 12, 5, tzinfo=timezone.utc))
        store.save(
            status="waiting",
            memory_turns=(
                ConversationTurn(
                    "system",
                    "Twinr memory summary:\n- Verified web lookup: Bus 24 -> 07:30 Uhr",
                    datetime(2026, 3, 12, 17, 59, tzinfo=timezone.utc),
                ),
                ConversationTurn(
                    "user",
                    "Erinnere mich an den Termin",
                    datetime(2026, 3, 12, 18, 0, tzinfo=timezone.utc),
                ),
                ConversationTurn(
                    "assistant",
                    "Der Termin ist um 14 Uhr.",
                    datetime(2026, 3, 12, 18, 0, 3, tzinfo=timezone.utc),
                ),
            ),
            memory_raw_tail=(
                ConversationTurn(
                    "user",
                    "Erinnere mich an den Termin",
                    datetime(2026, 3, 12, 18, 0, tzinfo=timezone.utc),
                ),
                ConversationTurn(
                    "assistant",
                    "Der Termin ist um 14 Uhr.",
                    datetime(2026, 3, 12, 18, 0, 3, tzinfo=timezone.utc),
                ),
            ),
            memory_ledger=(
                MemoryLedgerItem(
                    kind="conversation_summary",
                    content="User asked about an appointment. Twinr answered with the time.",
                    created_at=datetime(2026, 3, 12, 17, 58, tzinfo=timezone.utc),
                    source="compactor",
                ),
            ),
            memory_search_results=(
                SearchMemoryEntry(
                    question="Wann faehrt der Bus?",
                    answer="Bus 24 faehrt um 07:30 Uhr.",
                    sources=("https://example.com/fahrplan",),
                    created_at=datetime(2026, 3, 12, 17, 57, tzinfo=timezone.utc),
                    location_hint="Schwarzenbek",
                    date_context="2026-03-13",
                ),
            ),
            memory_state=MemoryState(
                active_topic="Termin",
                last_user_goal="Termindetails behalten",
                pending_printable="Der Termin ist um 14 Uhr.",
                last_search_summary="Bus 24 faehrt um 07:30 Uhr.",
                open_loops=("Termin bestaetigen",),
            ),
            last_transcript="Erinnere mich an den Termin",
            last_response="Der Termin ist um 14 Uhr.",
        )

        response = client.get("/memory")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live memory snapshot", response.text)
        self.assertIn("Durable memories", response.text)
        self.assertIn("Memory state", response.text)
        self.assertIn("Raw tail", response.text)
        self.assertIn("Recent search results", response.text)
        self.assertIn("Scheduled reminders", response.text)
        self.assertIn("Pending reminders", response.text)
        self.assertIn("Delivered reminders", response.text)
        self.assertIn("An die Tabletten erinnern.", response.text)
        self.assertIn("An den Arzttermin erinnern.", response.text)
        self.assertIn("Arzttermin am Montag um 14 Uhr.", response.text)
        self.assertIn("Erinnere mich an den Termin", response.text)
        self.assertIn("Der Termin ist um 14 Uhr.", response.text)
        self.assertIn("Bus 24 faehrt um 07:30 Uhr.", response.text)
        self.assertIn("Termindetails behalten", response.text)

    def test_memory_post_adds_durable_memory_entry(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/memory",
            data={
                "_action": "add_memory",
                "memory_kind": "appointment",
                "memory_summary": "Arzttermin am Montag um 14 Uhr.",
                "memory_details": "Bei Dr. Meyer in Hamburg.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        memory_entries = PersistentMemoryMarkdownStore(env_path.parent / "state" / "MEMORY.md").load_entries()
        self.assertEqual(len(memory_entries), 1)
        self.assertEqual(memory_entries[0].kind, "appointment")
        self.assertEqual(memory_entries[0].summary, "Arzttermin am Montag um 14 Uhr.")

    def test_memory_post_adds_reminder_entry(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/memory",
            data={
                "_action": "add_reminder",
                "reminder_due_at": "2030-03-15T12:00",
                "reminder_kind": "appointment",
                "reminder_summary": "An den Arzttermin erinnern.",
                "reminder_details": "Unterlagen mitnehmen.",
                "reminder_original_request": "Erinnere mich morgen um 12 Uhr an den Arzttermin.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        reminder_entries = ReminderStore(env_path.parent / "state" / "reminders.json").load_entries()
        self.assertEqual(len(reminder_entries), 1)
        self.assertEqual(reminder_entries[0].kind, "appointment")
        self.assertEqual(reminder_entries[0].summary, "An den Arzttermin erinnern.")

    def test_memory_post_marks_reminder_delivered(self) -> None:
        client, env_path = self.make_client()
        reminder_store = ReminderStore(env_path.parent / "state" / "reminders.json")
        reminder = reminder_store.schedule(
            due_at="2030-03-15T12:00",
            kind="appointment",
            summary="An den Arzttermin erinnern.",
        )

        response = client.post(
            "/memory",
            data={
                "_action": "mark_reminder_delivered",
                "reminder_id": reminder.reminder_id,
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ReminderStore(env_path.parent / "state" / "reminders.json").load_entries()
        self.assertTrue(entries[0].delivered)

    def test_memory_post_deletes_reminder(self) -> None:
        client, env_path = self.make_client()
        reminder_store = ReminderStore(env_path.parent / "state" / "reminders.json")
        reminder = reminder_store.schedule(
            due_at="2030-03-15T12:00",
            kind="appointment",
            summary="An den Arzttermin erinnern.",
        )

        response = client.post(
            "/memory",
            data={
                "_action": "delete_reminder",
                "reminder_id": reminder.reminder_id,
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ReminderStore(env_path.parent / "state" / "reminders.json").load_entries()
        self.assertEqual(entries, ())

    def test_personality_post_updates_base_files_and_preserves_managed_section(self) -> None:
        client, env_path = self.make_client()
        personality_dir = env_path.parent / "personality"
        ManagedContextFileStore(
            personality_dir / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        ).upsert(category="humor", instruction="Use only light humor.")

        response = client.post(
            "/personality",
            data={
                "SYSTEM": "Updated system",
                "PERSONALITY_BASE": "Updated personality",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual((personality_dir / "SYSTEM.md").read_text(encoding="utf-8"), "Updated system\n")
        personality_text = (personality_dir / "PERSONALITY.md").read_text(encoding="utf-8")
        self.assertIn("Updated personality", personality_text)
        self.assertIn("humor: Use only light humor.", personality_text)

    def test_personality_post_adds_managed_update(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/personality",
            data={
                "_action": "upsert_managed",
                "category": "response_style",
                "instruction": "Keep answers short and calm.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ManagedContextFileStore(
            env_path.parent / "personality" / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        ).load_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "response_style")
        self.assertEqual(entries[0].instruction, "Keep answers short and calm.")

    def test_user_post_adds_managed_profile_update(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/user",
            data={
                "_action": "upsert_managed",
                "category": "pets",
                "instruction": "Thom has two dogs.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ManagedContextFileStore(
            env_path.parent / "personality" / "USER.md",
            section_title="Twinr managed user updates",
        ).load_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "pets")
        self.assertEqual(entries[0].instruction, "Thom has two dogs.")

    def test_ops_logs_page_renders_structured_events(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        TwinrOpsEventStore.from_config(config).append(
            event="turn_started",
            message="Green button started a conversation turn.",
            data={"button": "green"},
        )

        response = client.get("/ops/logs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Latest structured events", response.text)
        self.assertIn("turn_started", response.text)
        self.assertIn("Green button started a conversation turn.", response.text)

    def test_ops_usage_page_renders_usage_summary(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        from twinr.ops import TokenUsage, TwinrUsageStore

        TwinrUsageStore.from_config(config).append(
            source="hardware_loop",
            request_kind="conversation",
            model="gpt-5.2",
            response_id="resp_usage_1",
            token_usage=TokenUsage(input_tokens=90, output_tokens=30, total_tokens=120),
        )

        response = client.get("/ops/usage")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Usage summary", response.text)
        self.assertIn("resp_usage_1", response.text)
        self.assertIn("gpt-5.2", response.text)

    def test_ops_health_page_renders_system_health(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/health")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live system health", response.text)
        self.assertIn("Services", response.text)

    def test_ops_health_page_renders_remote_memory_watchdog_status(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        store = RemoteMemoryWatchdogStore.from_config(config)
        store.save(
            RemoteMemoryWatchdogSnapshot(
                schema_version=1,
                started_at="2026-03-16T18:30:00Z",
                updated_at="2026-03-16T18:31:00Z",
                hostname="picarx",
                pid=321,
                interval_s=1.0,
                history_limit=3600,
                sample_count=12,
                failure_count=1,
                last_ok_at="2026-03-16T18:31:00Z",
                last_failure_at="2026-03-16T18:25:00Z",
                artifact_path=str(store.path),
                current=RemoteMemoryWatchdogSample(
                    seq=12,
                    captured_at="2026-03-16T18:31:00Z",
                    status="ok",
                    ready=True,
                    mode="remote_primary",
                    required=True,
                    latency_ms=18250.0,
                    consecutive_ok=4,
                    consecutive_fail=0,
                    detail=None,
                ),
                recent_samples=(),
            )
        )

        response = client.get("/ops/health")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Remote memory watchdog", response.text)
        self.assertIn("remote_primary", response.text)
        self.assertIn("18250.0 ms", response.text)
        self.assertIn("Consecutive ok", response.text)
        self.assertIn("2026-03-16T18:25:00Z", response.text)

    def test_ops_debug_page_renders_overview_tabs_and_summary(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        RuntimeSnapshotStore(config.runtime_state_path).save(
            status="waiting",
            memory_turns=(),
            last_transcript="Wie geht es dir heute?",
            last_response="Mir geht es gut.",
        )
        TwinrOpsEventStore.from_config(config).append(
            event="turn_started",
            message="Green button started a conversation turn.",
            data={"request_source": "button"},
        )
        from twinr.ops import TokenUsage, TwinrUsageStore

        TwinrUsageStore.from_config(config).append(
            source="hardware_loop",
            request_kind="conversation",
            model="gpt-5.2",
            response_id="resp_debug_overview",
            token_usage=TokenUsage(input_tokens=100, output_tokens=40, total_tokens=140),
        )

        response = client.get("/ops/debug")

        self.assertEqual(response.status_code, 200)
        self.assertIn("page-ops_debug", response.text)
        self.assertIn("ops-debug-page", response.text)
        self.assertIn("ops-debug-tabs", response.text)
        self.assertIn("Debug categories", response.text)
        self.assertIn("/ops/debug?tab=runtime", response.text)
        self.assertIn("ChonkyDB", response.text)
        self.assertIn("Current state at a glance", response.text)
        self.assertIn("LLM requests 24h", response.text)

    def test_ops_debug_page_renders_memory_attest_block(self) -> None:
        client, env_path = self.make_client()
        write_live_midterm_attest_artifacts(
            LiveMidtermAttestResult(
                probe_id="midterm_live_20260318T100000Z",
                status="ok",
                started_at="2026-03-18T10:00:00Z",
                finished_at="2026-03-18T10:00:09Z",
                env_path=str(env_path),
                base_project_root=str(env_path.parent),
                runtime_namespace="twinr_midterm_attest_midterm_live_20260318t100000z",
                writer_root=str(env_path.parent / "writer"),
                fresh_reader_root=str(env_path.parent / "reader"),
                flush_ok=True,
                midterm_context_present=True,
                follow_up_query="Was bringt mir Lea heute Abend um 19 Uhr vorbei?",
                follow_up_answer_text="Lea bringt dir eine Thermoskanne mit selbstgemachter Linsensuppe vorbei.",
                follow_up_model="gpt-5.2",
                expected_answer_terms=("thermoskanne", "linsensuppe"),
                matched_answer_terms=("thermoskanne", "linsensuppe"),
                writer_packet_ids=(
                    "midterm:20260318_lea_visit_thermos_lentil_soup_1900",
                    "midterm:user_has_daughter_lea",
                ),
                remote_packet_ids=(
                    "midterm:20260318_lea_visit_thermos_lentil_soup_1900",
                    "midterm:user_has_daughter_lea",
                ),
                fresh_reader_packet_ids=(
                    "midterm:20260318_lea_visit_thermos_lentil_soup_1900",
                    "midterm:user_has_daughter_lea",
                ),
                seed_turns=(
                    LiveMidtermSeedTurn(
                        prompt="Meine Tochter Lea bringt mir heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                        response_text="Ich merke mir das für später.",
                    ),
                ),
                last_path_warning_class="outside_root_local_fallback_skipped",
                last_path_warning_message="Skipped local snapshot fallback because the path is outside the configured Twinr memory root.",
            ),
            project_root=env_path.parent,
        )

        response = client.get("/ops/debug")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Memory attestation", response.text)
        self.assertIn("outside_root_local_fallback_skipped", response.text)
        self.assertIn("Writer packet ids", response.text)
        self.assertIn("Fresh-reader packet ids", response.text)
        self.assertIn("midterm:user_has_daughter_lea", response.text)

    def test_ops_debug_page_renders_memory_search_tab(self) -> None:
        client, _env_path = self.make_client()
        fake_search = LongTermOperatorSearchResult(
            query_text="Lea Linsensuppe",
            query_profile=LongTermQueryProfile.from_text(
                "Lea Linsensuppe",
                canonical_english_text="lea lentil soup",
            ),
            durable_objects=(
                LongTermMemoryObjectV1(
                    memory_id="fact:lea_visit",
                    kind="fact",
                    summary="Lea bringt heute Abend Linsensuppe vorbei.",
                    details="Um 19 Uhr mit einer Thermoskanne.",
                    status="active",
                    confidence=0.93,
                    source=LongTermSourceRefV1(source_type="conversation"),
                ),
            ),
            episodic_entries=(),
            midterm_packets=(
                LongTermMidtermPacketV1(
                    packet_id="midterm:lea_soup",
                    kind="visit",
                    summary="Lea bringt heute Abend Suppe vorbei.",
                    details="Thermoskanne, 19 Uhr.",
                    query_hints=("lea", "linsensuppe"),
                ),
            ),
            conflict_queue=(
                LongTermConflictQueueItemV1(
                    slot_key="favorite_drink",
                    question="Welches Getraenk stimmt jetzt?",
                    reason="Zwei widerspruechliche Erinnerungen sind offen.",
                    candidate_memory_id="fact:oolong",
                    options=(
                        LongTermConflictOptionV1(
                            memory_id="fact:oolong",
                            summary="Oolong-Tee",
                            status="candidate",
                        ),
                        LongTermConflictOptionV1(
                            memory_id="fact:coffee",
                            summary="Kaffee",
                            status="active",
                        ),
                    ),
                ),
            ),
            graph_context="Graph says Lea is the daughter and the visit is tonight.",
        )

        with patch("twinr.web.app.run_long_term_operator_search", return_value=fake_search):
            response = client.get("/ops/debug?tab=memory_search&memory_query=Lea+Linsensuppe")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Memory Search", response.text)
        self.assertIn("Canonical English", response.text)
        self.assertIn("lea lentil soup", response.text)
        self.assertIn("Lea bringt heute Abend Linsensuppe vorbei.", response.text)
        self.assertIn("midterm:lea_soup", response.text)
        self.assertIn("Graph context preview", response.text)

    def test_ops_debug_page_renders_conversation_lab_tab(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/debug?tab=conversation_lab")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Conversation Lab", response.text)
        self.assertIn("New Session", response.text)
        self.assertIn("Send Turn", response.text)
        self.assertIn("No portal conversation turns are stored yet.", response.text)
        self.assertIn("/ops/debug/conversation-lab/send", response.text)

    def test_ops_debug_page_renders_conversation_lab_session_traces(self) -> None:
        client, _env_path = self.make_client()
        fake_search_snapshot = {
            "query": "Lea Linsensuppe",
            "status": {"label": "2 hits", "detail": "durable 1 · episodic 0 · midterm 1 · conflicts 0", "status": "ok"},
            "summary_rows": (
                {"label": "Query", "value": "Lea Linsensuppe", "detail": None, "status": "muted", "copy": True, "wide": True},
            ),
            "sections": (
                {
                    "title": "Durable memory",
                    "description": "Stable facts and structured long-term objects that matched the query.",
                    "empty_message": "No durable facts matched this query.",
                    "items": (
                        {
                            "badge": "fact",
                            "level": "ok",
                            "title": "Lea bringt morgen Suppe vorbei.",
                            "time": "2026-03-19T11:30:00Z",
                            "body": "Um 19 Uhr mit einer Thermoskanne.",
                            "meta_lines": ("memory_id=fact:lea_visit",),
                        },
                    ),
                },
            ),
            "context_blocks": (),
        }
        fake_state = {
            "sessions": (
                {
                    "session_id": "session_20260319",
                    "title": "Lea soup",
                    "updated_at": "2026-03-19T11:30:00Z",
                    "turn_count": 1,
                    "status": "ok",
                },
            ),
            "active_session": {
                "session_id": "session_20260319",
                "title": "Lea soup",
                "created_at": "2026-03-19T11:29:00Z",
                "updated_at": "2026-03-19T11:30:00Z",
                "turns": (
                    {
                        "created_at": "2026-03-19T11:30:00Z",
                        "status": "ok",
                        "status_badge": {"label": "Completed", "status": "ok"},
                        "prompt": "Lea kommt morgen um 19 Uhr vorbei.",
                        "response": "Ich merke mir das und erinnere dich gern daran.",
                        "result_rows": (
                            {"label": "Model", "value": "gpt-5.2", "detail": None, "status": "muted", "copy": False, "wide": False},
                        ),
                        "route_items": (
                            {
                                "badge": "decision",
                                "level": "ok",
                                "title": "Dual Lane Final Path Selected",
                                "time": None,
                                "body": "Run resolved direct supervisor path",
                                "meta_lines": ("action: direct",),
                            },
                        ),
                        "tool_items": (
                            {
                                "badge": "remember_memory",
                                "level": "ok",
                                "title": "Remember Memory",
                                "time": None,
                                "body": "Stored a new durable memory entry.",
                                "meta_lines": ("status: stored",),
                            },
                        ),
                        "telemetry_items": (),
                        "memory_rows": (
                            {"label": "Flush result", "value": "ok", "detail": None, "status": "ok", "copy": False, "wide": False},
                        ),
                        "retrieval_before": fake_search_snapshot,
                        "retrieval_after": fake_search_snapshot,
                    },
                ),
            },
            "missing_session": False,
        }

        with patch("twinr.web.app.load_conversation_lab_state", return_value=fake_state):
            response = client.get("/ops/debug?tab=conversation_lab&lab_session=session_20260319")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Routing And Model Path", response.text)
        self.assertIn("Tool Activity", response.text)
        self.assertIn("Memory Write", response.text)
        self.assertIn("Retrieval Before Turn", response.text)
        self.assertIn("Retrieval After Turn", response.text)
        self.assertIn("Lea kommt morgen um 19 Uhr vorbei.", response.text)
        self.assertIn("Ich merke mir das und erinnere dich gern daran.", response.text)
        self.assertIn("Lea soup", response.text)

    def test_ops_debug_conversation_lab_routes_redirect_to_session(self) -> None:
        client, _env_path = self.make_client()

        with patch("twinr.web.app.create_conversation_lab_session", return_value="session_new"):
            new_response = client.post(
                "/ops/debug/conversation-lab/new",
                data={},
                follow_redirects=False,
            )

        self.assertEqual(new_response.status_code, 303)
        self.assertEqual(new_response.headers["location"], "/ops/debug?tab=conversation_lab&lab_session=session_new")

        with patch("twinr.web.app.run_conversation_lab_turn", return_value="session_updated"):
            send_response = client.post(
                "/ops/debug/conversation-lab/send",
                data={"session_id": "session_old", "prompt": "Bitte teste die neue Portal-Konversation."},
                follow_redirects=False,
            )

        self.assertEqual(send_response.status_code, 303)
        self.assertEqual(send_response.headers["location"], "/ops/debug?tab=conversation_lab&lab_session=session_updated")

    def test_ops_debug_page_renders_chonkydb_tab(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        store = RemoteMemoryWatchdogStore.from_config(config)
        sample = RemoteMemoryWatchdogSample(
            seq=12,
            captured_at="2026-03-16T18:31:00Z",
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=18250.0,
            consecutive_ok=4,
            consecutive_fail=0,
            detail=None,
        )
        store.save(
            RemoteMemoryWatchdogSnapshot(
                schema_version=1,
                started_at="2026-03-16T18:30:00Z",
                updated_at="2026-03-16T18:31:00Z",
                hostname="picarx",
                pid=os.getpid(),
                interval_s=1.0,
                history_limit=3600,
                sample_count=12,
                failure_count=1,
                last_ok_at="2026-03-16T18:31:00Z",
                last_failure_at="2026-03-16T18:25:00Z",
                artifact_path=str(store.path),
                current=sample,
                recent_samples=(sample,),
            )
        )

        response = client.get("/ops/debug?tab=chonkydb")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Required remote assessment", response.text)
        self.assertIn("Watchdog snapshot", response.text)
        self.assertIn("remote_primary", response.text)
        self.assertIn("18250.0 ms", response.text)
        self.assertIn("PID alive", response.text)

    def test_ops_debug_page_renders_hardware_tab(self) -> None:
        client, _env_path = self.make_client()
        fake_overview = DeviceOverview(
            captured_at="2026-03-13T16:10:00+00:00",
            devices=(
                DeviceStatus(
                    key="printer",
                    label="Printer",
                    status="warn",
                    summary="Queue is visible, but paper output must be confirmed on the device.",
                    facts=(
                        DeviceFact("Queue", "Thermal_GP58"),
                        DeviceFact("Paper status", "unknown on the current raw USB/CUPS path"),
                    ),
                    notes=("Twinr cannot prove real paper output from this printer path.",),
                ),
            ),
        )

        with patch("twinr.web.app.collect_device_overview", return_value=fake_overview):
            response = client.get("/ops/debug?tab=hardware")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live hardware overview", response.text)
        self.assertIn("Printer", response.text)
        self.assertIn("Thermal_GP58", response.text)
        self.assertIn("Queue is visible, but paper output must be confirmed on the device.", response.text)

    def test_ops_debug_page_raw_tab_redacts_env(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/debug?tab=raw")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Redacted env", response.text)
        self.assertIn("OPENAI_API_KEY", response.text)
        self.assertNotIn("sk-test-1234", response.text)
        self.assertIn("Config check summary", response.text)

    def test_ops_devices_page_renders_device_status(self) -> None:
        client, _env_path = self.make_client()
        fake_overview = DeviceOverview(
            captured_at="2026-03-13T16:10:00+00:00",
            devices=(
                DeviceStatus(
                    key="printer",
                    label="Printer",
                    status="warn",
                    summary="Queue is visible, but paper output must be confirmed on the device.",
                    facts=(
                        DeviceFact("Queue", "Thermal_GP58"),
                        DeviceFact("Paper status", "unknown on the current raw USB/CUPS path"),
                    ),
                    notes=("Twinr cannot prove real paper output from this printer path.",),
                ),
                DeviceStatus(
                    key="camera",
                    label="Camera",
                    status="ok",
                    summary="Camera device `/dev/video0` is present.",
                    facts=(DeviceFact("Device", "/dev/video0"),),
                ),
            ),
        )

        with patch("twinr.web.app.collect_device_overview", return_value=fake_overview):
            response = client.get("/ops/devices")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Hardware devices", response.text)
        self.assertIn("Queue is visible, but paper output must be confirmed on the device.", response.text)
        self.assertIn("Paper status", response.text)
        self.assertIn("unknown on the current raw USB/CUPS path", response.text)
        self.assertIn("Camera device `/dev/video0` is present.", response.text)

    def test_ops_self_test_page_lists_pir_motion_and_proactive_mic(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/self-test")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Printer-Testdruck", response.text)
        self.assertIn("confirm the paper output on the device", response.text)
        self.assertIn("PIR motion", response.text)
        self.assertIn("Wait for a motion trigger on the configured PIR input.", response.text)
        self.assertIn("Proaktives Mikrofon", response.text)
        self.assertIn("proactive background microphone", response.text)

    def test_ops_support_post_builds_bundle(self) -> None:
        client, env_path = self.make_client()

        response = client.post("/ops/support")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Latest bundle", response.text)
        bundle_files = list(resolve_ops_paths(env_path.parent).bundles_root.glob("*.zip"))
        self.assertEqual(len(bundle_files), 1)

    def test_ops_self_test_route_renders_runner_result(self) -> None:
        client, _env_path = self.make_client()
        fake_result = SimpleNamespace(
            status="ok",
            summary="Confirmation beep played.",
            details=("Output device: default",),
            artifact_name="speaker-test.txt",
            finished_at="2026-03-13T08:00:00+00:00",
        )

        with patch("twinr.web.app.TwinrSelfTestRunner") as runner_cls:
            runner_cls.available_tests.return_value = (
                ("speaker", "Speaker-Beep", "Play a local confirmation beep."),
            )
            runner_cls.return_value.run.return_value = fake_result

            response = client.post("/ops/self-test", data={"test_name": "speaker"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Last result", response.text)
        self.assertIn("Confirmation beep played.", response.text)
        self.assertIn("speaker-test.txt", response.text)


if __name__ == "__main__":
    unittest.main()
