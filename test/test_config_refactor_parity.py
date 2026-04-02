from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent._config.loading import load_twinr_config
from twinr.agent.base_agent.config import TwinrConfig

_FEATURE_ENV_TEXT = (
    "\n".join(
        [
            "OPENAI_MODEL=gpt-5.4-mini",
            "TWINR_LOCAL_SEMANTIC_ROUTER_MODE=gated",
            "TWINR_LOCAL_SEMANTIC_ROUTER_MODEL_DIR=artifacts/router/bundle",
            "TWINR_LOCAL_SEMANTIC_ROUTER_USER_INTENT_MODEL_DIR=artifacts/router/user_intent_bundle",
            "TWINR_LOCAL_SEMANTIC_ROUTER_TRACE=false",
            "TWINR_BROWSER_AUTOMATION_ENABLED=true",
            "TWINR_BROWSER_AUTOMATION_WORKSPACE_PATH=browser_automation",
            "TWINR_BROWSER_AUTOMATION_ENTRY_MODULE=adapter.py",
            "TWINR_CAMERA_HOST_MODE=onboard",
            "TWINR_CAMERA_DEVICE=/dev/video1",
            "TWINR_DISPLAY_LAYOUT=DEBUG_FACE",
            "TWINR_DISPLAY_BUSY_TIMEOUT_S=12.5",
            "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_MIN_WAKE_DURATION_MS=450",
            "TWINR_VOICE_ORCHESTRATOR_INTENT_STAGE1_WINDOW_BONUS_MS=555",
        ]
    )
    + "\n"
)
_ENV_OVERRIDE_TEXT = "OPENAI_MODEL=legacy\nTWINR_DISPLAY_DRIVER=waveshare_4in2_v2\n"
_NORMALIZED_ROOT_KEYS = (
    "project_root",
    "whatsapp_auth_dir",
    "whatsapp_worker_root",
    "runtime_state_path",
    "memory_markdown_path",
    "reminder_store_path",
    "automation_store_path",
    "voice_profile_store_path",
    "adaptive_timing_store_path",
    "long_term_memory_path",
    "attention_servo_state_path",
)
_EXPECTED_GOLDEN_DIGESTS = {
    "empty": "8aa5e6905d0efce91628cb72a7c6585a47147a05c3d97c4fb33c8a0efc2171e2",
    "feature_env": "a46701dc818e866c070dce8d10915d7a13392eab9157c3f2d3e3042435e61a74",
    "env_override": "86c22423f4586925a30737a4e58d04168e336d843cd54e93936403cc9af8f015",
}


def _normalized_config_payload(config: TwinrConfig) -> str:
    """Return a stable JSON payload for full-config parity hashing."""

    data = asdict(config)
    project_root = data["project_root"]
    for key in _NORMALIZED_ROOT_KEYS:
        value = data[key]
        if isinstance(value, str):
            data[key] = value.replace(project_root, "<PROJECT_ROOT>")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class TwinrConfigRefactorParityTests(unittest.TestCase):
    def _load_config(
        self, env_text: str, overrides: dict[str, str] | None = None
    ) -> TwinrConfig:
        overrides = overrides or {}
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(env_text, encoding="utf-8")
            old_values = {key: os.environ.get(key) for key in overrides}
            try:
                for key, value in overrides.items():
                    os.environ[key] = value
                return TwinrConfig.from_env(env_path)
            finally:
                for key, old_value in old_values.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(TwinrConfig.__module__, "twinr.agent.base_agent.config")

    def test_wrapper_matches_internal_loader_for_same_fixture_path(self) -> None:
        cases: tuple[tuple[str, str, dict[str, str]], ...] = (
            ("empty", "", {}),
            ("feature_env", _FEATURE_ENV_TEXT, {}),
            (
                "env_override",
                _ENV_OVERRIDE_TEXT,
                {
                    "OPENAI_MODEL": "override-model",
                    "TWINR_DISPLAY_DRIVER": "hdmi_wayland",
                },
            ),
        )
        for name, env_text, overrides in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    env_path = Path(temp_dir) / ".env"
                    env_path.write_text(env_text, encoding="utf-8")
                    old_values = {key: os.environ.get(key) for key in overrides}
                    try:
                        for key, value in overrides.items():
                            os.environ[key] = value
                        wrapped = TwinrConfig.from_env(env_path)
                        internal = load_twinr_config(TwinrConfig, env_path)
                    finally:
                        for key, old_value in old_values.items():
                            if old_value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = old_value
                self.assertEqual(asdict(wrapped), asdict(internal))

    def test_golden_master_hashes_remain_stable(self) -> None:
        cases: tuple[tuple[str, str, dict[str, str]], ...] = (
            ("empty", "", {}),
            ("feature_env", _FEATURE_ENV_TEXT, {}),
            (
                "env_override",
                _ENV_OVERRIDE_TEXT,
                {
                    "OPENAI_MODEL": "override-model",
                    "TWINR_DISPLAY_DRIVER": "hdmi_wayland",
                },
            ),
        )
        for name, env_text, overrides in cases:
            with self.subTest(case=name):
                payload = _normalized_config_payload(
                    self._load_config(env_text, overrides)
                )
                digest = sha256(payload.encode("utf-8")).hexdigest()
                self.assertEqual(digest, _EXPECTED_GOLDEN_DIGESTS[name])
