from __future__ import annotations

from twinr.web.contracts import SettingsSection
from twinr.web.forms import _select_field, _text_field
from twinr.web.store import FileBackedSetting, mask_secret
from twinr.web.viewmodels_common import _PROVIDER_OPTIONS, _TRISTATE_BOOL_OPTIONS


def _connect_sections(env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    return (
        SettingsSection(
            title="Provider routing",
            description="Choose which backend handles each pipeline stage.",
            fields=(
                _select_field(
                    "TWINR_PROVIDER_LLM",
                    "LLM provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls which backend answers normal text questions.",
                ),
                _select_field(
                    "TWINR_PROVIDER_STT",
                    "STT provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls which backend turns speech into text.",
                ),
                _select_field(
                    "TWINR_PROVIDER_TTS",
                    "TTS provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls which backend turns replies into spoken audio.",
                ),
                _select_field(
                    "TWINR_PROVIDER_REALTIME",
                    "Realtime provider",
                    env_values,
                    _PROVIDER_OPTIONS,
                    "openai",
                    tooltip_text="Controls the low-latency voice session backend.",
                ),
            ),
        ),
        SettingsSection(
            title="OpenAI",
            description="Main account and auth settings for the currently active runtime.",
            fields=(
                FileBackedSetting(
                    key="OPENAI_API_KEY",
                    label="API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('OPENAI_API_KEY'))}. Leave blank to keep it unchanged.",
                    tooltip_text="The main OpenAI secret used for chat, speech, vision, and realtime requests.",
                    input_type="password",
                    placeholder="sk-...",
                    secret=True,
                ),
                _text_field(
                    "OPENAI_PROJ_ID",
                    "Project ID",
                    env_values,
                    "",
                    placeholder="proj_...",
                    tooltip_text="Optional project id. Only set this when your account setup requires an explicit OpenAI project.",
                ),
                _select_field(
                    "OPENAI_SEND_PROJECT_HEADER",
                    "Project header",
                    env_values,
                    _TRISTATE_BOOL_OPTIONS,
                    "",
                    help_text="Use auto unless you explicitly need to force the header on or off.",
                    tooltip_text="Auto is usually correct. Force this only if your OpenAI key/project setup needs it.",
                ),
            ),
        ),
        SettingsSection(
            title="Other providers",
            description="Stored here now so later provider adapters can use them without editing files by hand.",
            fields=(
                FileBackedSetting(
                    key="DEEPINFRA_API_KEY",
                    label="DeepInfra API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('DEEPINFRA_API_KEY'))}. Leave blank to keep it unchanged.",
                    tooltip_text="Credential for a future DeepInfra provider integration.",
                    input_type="password",
                    placeholder="DeepInfra key",
                    secret=True,
                ),
                FileBackedSetting(
                    key="OPENROUTER_API_KEY",
                    label="OpenRouter API key",
                    value="",
                    help_text=f"Current value: {mask_secret(env_values.get('OPENROUTER_API_KEY'))}. Leave blank to keep it unchanged.",
                    tooltip_text="Credential for a future OpenRouter provider integration.",
                    input_type="password",
                    placeholder="OpenRouter key",
                    secret=True,
                ),
            ),
        ),
    )
