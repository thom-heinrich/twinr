"""Build the Twinr web settings presenter models.

This module shapes ``TwinrConfig`` values into template-ready settings sections
and adaptive-timing cards while keeping malformed learned-state data from
breaking the dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent import AdaptiveTimingStore, TwinrConfig
from twinr.web.support.contracts import AdaptiveTimingView, DetailMetric, SettingsSection
from twinr.web.support.forms import _select_field, _text_field, _textarea_field
from twinr.web.presenters.common import (
    _BOOL_OPTIONS,
    _CONVERSATION_WEB_SEARCH_OPTIONS,
    _GPIO_BIAS_OPTIONS,
    _REASONING_EFFORT_OPTIONS,
    _SEARCH_CONTEXT_OPTIONS,
    _VISION_DETAIL_OPTIONS,
    _WAKEWORD_FALLBACK_BACKEND_OPTIONS,
    _WAKEWORD_PRIMARY_BACKEND_OPTIONS,
    _WAKEWORD_VERIFIER_MODE_OPTIONS,
    _YES_NO_OPTIONS,
    _format_millis_label,
    _format_seconds_label,
)

def _csv_display(values: object, *, separator: str = ", ") -> str:
    """Render optional iterable config values as one display string."""

    # AUDIT-FIX(#4): Normalize optional iterables so missing list-based config does not crash the settings page.
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    try:
        return separator.join(str(value) for value in values if value is not None)
    except TypeError:
        return str(values)


def _int_or_default(value: object, default: int = 0) -> int:
    """Parse one integer or return the provided fallback."""

    # AUDIT-FIX(#1): Build a safe fallback profile when adaptive timing state cannot be loaded.
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: object, default: float = 0.0) -> float:
    """Parse one float or return the provided fallback."""

    # AUDIT-FIX(#1): Build a safe fallback profile when adaptive timing state cannot be loaded.
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fallback_pause_grace_ms(config: TwinrConfig) -> int:
    """Infer a baseline pause-grace value from whichever config field exists."""

    # AUDIT-FIX(#1): Infer a baseline pause-grace value from config without assuming one exact TwinrConfig field name.
    for attr_name in (
        "pause_grace_ms",
        "conversation_pause_grace_ms",
        "audio_pause_grace_ms",
        "adaptive_timing_pause_grace_ms",
    ):
        value = getattr(config, attr_name, None)
        if value is not None:
            return _int_or_default(value, 0)
    return 0


def _fallback_timing_profile(config: TwinrConfig) -> SimpleNamespace:
    """Build a conservative adaptive-timing snapshot from config defaults."""

    # AUDIT-FIX(#1): Keep the dashboard renderable even when the learned timing store is missing or unreadable.
    return SimpleNamespace(
        button_start_timeout_s=_float_or_default(getattr(config, "audio_start_timeout_s", 0.0), 0.0),
        follow_up_start_timeout_s=_float_or_default(getattr(config, "conversation_follow_up_timeout_s", 0.0), 0.0),
        speech_pause_ms=_int_or_default(getattr(config, "speech_pause_ms", 0), 0),
        pause_grace_ms=_fallback_pause_grace_ms(config),
        button_success_count=0,
        button_timeout_count=0,
        follow_up_success_count=0,
        follow_up_timeout_count=0,
        pause_resume_count=0,
        clean_pause_streak=0,
    )


def _coerce_timing_profile(profile: object, fallback: object) -> SimpleNamespace:
    """Normalize a timing-profile-like object into a complete snapshot."""

    # AUDIT-FIX(#1): Normalize partially populated timing snapshots so attribute drift does not crash the dashboard.
    return SimpleNamespace(
        button_start_timeout_s=_float_or_default(
            getattr(profile, "button_start_timeout_s", getattr(fallback, "button_start_timeout_s", 0.0)),
            _float_or_default(getattr(fallback, "button_start_timeout_s", 0.0), 0.0),
        ),
        follow_up_start_timeout_s=_float_or_default(
            getattr(profile, "follow_up_start_timeout_s", getattr(fallback, "follow_up_start_timeout_s", 0.0)),
            _float_or_default(getattr(fallback, "follow_up_start_timeout_s", 0.0), 0.0),
        ),
        speech_pause_ms=_int_or_default(
            getattr(profile, "speech_pause_ms", getattr(fallback, "speech_pause_ms", 0)),
            _int_or_default(getattr(fallback, "speech_pause_ms", 0), 0),
        ),
        pause_grace_ms=_int_or_default(
            getattr(profile, "pause_grace_ms", getattr(fallback, "pause_grace_ms", 0)),
            _int_or_default(getattr(fallback, "pause_grace_ms", 0), 0),
        ),
        button_success_count=_int_or_default(
            getattr(profile, "button_success_count", getattr(fallback, "button_success_count", 0)),
            _int_or_default(getattr(fallback, "button_success_count", 0), 0),
        ),
        button_timeout_count=_int_or_default(
            getattr(profile, "button_timeout_count", getattr(fallback, "button_timeout_count", 0)),
            _int_or_default(getattr(fallback, "button_timeout_count", 0), 0),
        ),
        follow_up_success_count=_int_or_default(
            getattr(profile, "follow_up_success_count", getattr(fallback, "follow_up_success_count", 0)),
            _int_or_default(getattr(fallback, "follow_up_success_count", 0), 0),
        ),
        follow_up_timeout_count=_int_or_default(
            getattr(profile, "follow_up_timeout_count", getattr(fallback, "follow_up_timeout_count", 0)),
            _int_or_default(getattr(fallback, "follow_up_timeout_count", 0), 0),
        ),
        pause_resume_count=_int_or_default(
            getattr(profile, "pause_resume_count", getattr(fallback, "pause_resume_count", 0)),
            _int_or_default(getattr(fallback, "pause_resume_count", 0), 0),
        ),
        clean_pause_streak=_int_or_default(
            getattr(profile, "clean_pause_streak", getattr(fallback, "clean_pause_streak", 0)),
            _int_or_default(getattr(fallback, "clean_pause_streak", 0), 0),
        ),
    )


def _safe_store_path(path_value: object) -> Path | None:
    """Return a ``Path`` for configured store values or ``None`` when unset."""

    # AUDIT-FIX(#1): Avoid crashing the view model when the adaptive timing store path is unset.
    if path_value in (None, ""):
        return None
    return Path(str(path_value))


def _safe_local_timezone(timezone_name: object) -> tuple[object, str | None]:
    """Return a timezone object plus an optional operator-facing warning note."""

    # AUDIT-FIX(#2): Fall back to UTC when the configured IANA timezone is missing or invalid.
    if not timezone_name:
        return timezone.utc, "Local timezone is not configured; displaying UTC"
    try:
        return ZoneInfo(str(timezone_name)), None
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc, "Local timezone is invalid; displaying UTC"


def _adaptive_timing_last_updated_label(
    store_path: Path | None,
    timezone_name: object,
) -> tuple[str, str | None]:
    """Describe when the adaptive timing store was last updated for the UI."""

    # AUDIT-FIX(#2): Replace exists()+stat() TOCTOU logic with one stat() call and handle filesystem/time conversion failures.
    if store_path is None:
        return "Adaptive timing store path is not configured", None
    try:
        stat_result = store_path.stat()
    except FileNotFoundError:
        return "No learned timing saved yet", None
    except OSError:
        return "Learned timing file exists, but its metadata is unavailable", None

    tzinfo, timezone_note = _safe_local_timezone(timezone_name)
    try:
        updated = datetime.fromtimestamp(stat_result.st_mtime, tz=tzinfo)
    except (OverflowError, OSError, ValueError):
        return "Learned timing file exists, but its last-updated time is unavailable", timezone_note

    return updated.strftime("%Y-%m-%d %H:%M:%S %Z"), timezone_note


def _load_adaptive_timing_profiles(config: TwinrConfig) -> tuple[object, object, str | None]:
    """Load current and baseline adaptive-timing profiles with safe fallbacks.

    Returns:
        Tuple of current profile, baseline profile, and an optional status note
        that explains degraded-state fallbacks to the operator.
    """

    # AUDIT-FIX(#1): Guard all adaptive timing store reads so a corrupt/partial file does not 500 the dashboard.
    fallback_profile = _fallback_timing_profile(config)
    store_path_value = getattr(config, "adaptive_timing_store_path", None)
    if not store_path_value:
        return fallback_profile, fallback_profile, "Showing configured defaults because no adaptive timing store path is configured"

    try:
        store = AdaptiveTimingStore(store_path_value, config=config)
    except Exception:
        return fallback_profile, fallback_profile, "Showing configured defaults because the adaptive timing store is unavailable"

    status_notes: list[str] = []
    try:
        baseline = store.default_profile()
    except Exception:
        baseline = fallback_profile
        status_notes.append("Baseline defaults were inferred from config")
    baseline = _coerce_timing_profile(baseline, fallback_profile)  # AUDIT-FIX(#1): Guarantee a complete baseline snapshot before rendering.

    try:
        current = store.current()
    except Exception:
        current = baseline
        status_notes.append("Learned timing could not be read; showing configured defaults")
    current = _coerce_timing_profile(current, baseline)  # AUDIT-FIX(#1): Guarantee a complete current snapshot before rendering.

    return current, baseline, "; ".join(status_notes) or None



def _adaptive_timing_view(config: TwinrConfig) -> AdaptiveTimingView:
    """Build the adaptive-timing card view model for the settings page."""

    current, baseline, status_note = _load_adaptive_timing_profiles(config)  # AUDIT-FIX(#1): Fail closed to defaults instead of crashing on store read errors.
    store_path = _safe_store_path(getattr(config, "adaptive_timing_store_path", None))  # AUDIT-FIX(#1): Handle unset store paths without raising TypeError.
    last_updated_label, timezone_note = _adaptive_timing_last_updated_label(
        store_path,
        getattr(config, "local_timezone_name", None),
    )  # AUDIT-FIX(#2): Make file metadata and timezone conversion resilient.
    status_notes = [note for note in (timezone_note, status_note) if note]
    if status_notes:
        last_updated_label = f"{last_updated_label}; {'; '.join(status_notes)}"  # AUDIT-FIX(#3): Surface degraded-state information to the operator instead of failing silently.
    return AdaptiveTimingView(
        enabled=config.adaptive_timing_enabled,
        path="" if store_path is None else str(store_path),  # AUDIT-FIX(#1): Keep the view model valid when the store path is unset.
        last_updated_label=last_updated_label,
        current_metrics=(
            DetailMetric(
                label="Button start timeout",
                value=_format_seconds_label(current.button_start_timeout_s),
                detail="How long Twinr currently waits after the green button before giving up on the first spoken word.",
            ),
            DetailMetric(
                label="Follow-up start timeout",
                value=_format_seconds_label(current.follow_up_start_timeout_s),
                detail="How long Twinr waits after its answer when it re-opens the brief follow-up listening window.",
            ),
            DetailMetric(
                label="Speech pause cutoff",
                value=_format_millis_label(current.speech_pause_ms),
                detail="Current silence threshold before Twinr thinks a spoken turn may be finished.",
            ),
            DetailMetric(
                label="Pause grace window",
                value=_format_millis_label(current.pause_grace_ms),
                detail="Extra grace period that keeps the same turn alive when someone pauses and then continues speaking.",
            ),
        ),
        counter_metrics=(
            DetailMetric(
                label="Green-button results",
                value=f"{current.button_success_count} ok / {current.button_timeout_count} timeout",
                detail="A few early timeouts push the listening window up quickly; steady fast starts pull it back down slowly.",
            ),
            DetailMetric(
                label="Follow-up results",
                value=f"{current.follow_up_success_count} ok / {current.follow_up_timeout_count} timeout",
                detail="Tracks how often the short second-listen window was actually used versus timing out empty.",
            ),
            DetailMetric(
                label="Mid-question resumes",
                value=str(current.pause_resume_count),
                detail="Counts how often Twinr detected more speech during the pause grace window and kept the same turn running.",
            ),
            DetailMetric(
                label="Clean pause streak",
                value=str(current.clean_pause_streak),
                detail="After several clean turn endings in a row, Twinr slowly tightens the learned pause timings again.",
            ),
        ),
        baseline_metrics=(
            DetailMetric(
                label="Configured button baseline",
                value=_format_seconds_label(baseline.button_start_timeout_s),
                detail="Fixed minimum from Settings before any learning is applied.",
            ),
            DetailMetric(
                label="Configured follow-up baseline",
                value=_format_seconds_label(baseline.follow_up_start_timeout_s),
                detail="Fixed minimum follow-up window from Settings.",
            ),
            DetailMetric(
                label="Configured speech pause baseline",
                value=_format_millis_label(baseline.speech_pause_ms),
                detail="Minimum pause cutoff Twinr will not tune below.",
            ),
            DetailMetric(
                label="Configured grace baseline",
                value=_format_millis_label(baseline.pause_grace_ms),
                detail="Minimum grace window Twinr keeps available for slower, hesitant speech.",
            ),
        ),
    )


def _settings_sections(config: TwinrConfig, env_values: dict[str, str]) -> tuple[SettingsSection, ...]:
    """Build the full settings page section list in display order."""

    green_button_gpio = "" if config.green_button_gpio is None else str(config.green_button_gpio)
    yellow_button_gpio = "" if config.yellow_button_gpio is None else str(config.yellow_button_gpio)
    pir_motion_gpio = "" if config.pir_motion_gpio is None else str(config.pir_motion_gpio)
    button_probe_lines = _csv_display(getattr(config, "button_probe_lines", None), separator=",")  # AUDIT-FIX(#4): Optional GPIO probe lists must not crash the settings UI.
    return (
        SettingsSection(
            title="Models and voices",
            description="Main OpenAI model choices for chat, speech, and realtime audio.",
            fields=(
                _text_field(
                    "OPENAI_MODEL",
                    "LLM model",
                    env_values,
                    config.default_model,
                    tooltip_text="Main chat and reasoning model for the standard Twinr flow.",
                ),
                _select_field(
                    "OPENAI_REASONING_EFFORT",
                    "Reasoning effort",
                    env_values,
                    _REASONING_EFFORT_OPTIONS,
                    config.openai_reasoning_effort,
                    tooltip_text="Higher effort can improve harder answers but usually costs more time and tokens.",
                ),
                _text_field(
                    "OPENAI_STT_MODEL",
                    "STT model",
                    env_values,
                    config.openai_stt_model,
                    tooltip_text="Model used when Twinr converts recorded speech into text.",
                ),
                _text_field(
                    "OPENAI_TTS_MODEL",
                    "TTS model",
                    env_values,
                    config.openai_tts_model,
                    tooltip_text="Model used to synthesize spoken replies.",
                ),
                _text_field(
                    "OPENAI_TTS_VOICE",
                    "TTS voice",
                    env_values,
                    config.openai_tts_voice,
                    tooltip_text="Voice name used for normal spoken replies.",
                ),
                _text_field(
                    "OPENAI_TTS_SPEED",
                    "TTS speed",
                    env_values,
                    f"{config.openai_tts_speed:.2f}",
                    tooltip_text="Speaking speed factor for normal spoken replies. Keep this close to 1.0 for a calm, natural result.",
                ),
                _text_field(
                    "OPENAI_TTS_FORMAT",
                    "TTS format",
                    env_values,
                    config.openai_tts_format,
                    tooltip_text="Audio format for generated speech, for example wav or mp3.",
                ),
                _textarea_field(
                    "OPENAI_TTS_INSTRUCTIONS",
                    "TTS instructions",
                    env_values,
                    config.openai_tts_instructions or "",
                    placeholder="Speak in clear, warm, natural standard German...",
                    tooltip_text="Optional speaking instructions sent with text-to-speech requests.",
                    rows=4,
                ),
                _text_field(
                    "OPENAI_REALTIME_MODEL",
                    "Realtime model",
                    env_values,
                    config.openai_realtime_model,
                    tooltip_text="Model used by the low-latency live voice session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_VOICE",
                    "Realtime voice",
                    env_values,
                    config.openai_realtime_voice,
                    tooltip_text="Voice name used inside the realtime session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_SPEED",
                    "Realtime speed",
                    env_values,
                    f"{config.openai_realtime_speed:.2f}",
                    tooltip_text="Speaking speed factor for the low-latency live voice session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_TRANSCRIPTION_MODEL",
                    "Realtime transcription model",
                    env_values,
                    config.openai_realtime_transcription_model,
                    tooltip_text="Speech-to-text model used inside the realtime session.",
                ),
                _text_field(
                    "OPENAI_REALTIME_LANGUAGE",
                    "Realtime language",
                    env_values,
                    config.openai_realtime_language or "",
                    placeholder="de",
                    tooltip_text="Hint for the main spoken language during realtime audio.",
                ),
                _text_field(
                    "OPENAI_REALTIME_INPUT_SAMPLE_RATE",
                    "Realtime input sample rate",
                    env_values,
                    str(config.openai_realtime_input_sample_rate),
                    tooltip_text="Sample rate sent to the realtime backend after local resampling.",
                ),
                _textarea_field(
                    "OPENAI_REALTIME_INSTRUCTIONS",
                    "Realtime instructions",
                    env_values,
                    config.openai_realtime_instructions or "",
                    placeholder="Speak in clear, warm, natural standard German...",
                    tooltip_text="Optional system-style instructions used only in realtime mode.",
                    rows=4,
                ),
            ),
        ),
        SettingsSection(
            title="Search",
            description="Controls when Twinr is allowed to browse and how broad that search context should be.",
            fields=(
                _select_field(
                    "TWINR_OPENAI_ENABLE_WEB_SEARCH",
                    "OpenAI web search",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.openai_enable_web_search else "false",
                    tooltip_text="Master switch for OpenAI-backed web search in the standard conversation loop.",
                ),
                _select_field(
                    "TWINR_CONVERSATION_WEB_SEARCH",
                    "Conversation web search",
                    env_values,
                    _CONVERSATION_WEB_SEARCH_OPTIONS,
                    config.conversation_web_search,
                    tooltip_text="Auto searches only freshness-sensitive questions. Always forces search. Never disables it for normal chats.",
                ),
                _text_field(
                    "OPENAI_SEARCH_MODEL",
                    "Search model",
                    env_values,
                    config.openai_search_model,
                    tooltip_text="Model used when Twinr performs a web-backed response.",
                ),
                _select_field(
                    "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE",
                    "Search context size",
                    env_values,
                    _SEARCH_CONTEXT_OPTIONS,
                    config.openai_web_search_context_size,
                    tooltip_text="How much retrieved web context to include with the answer.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_COUNTRY",
                    "Search country",
                    env_values,
                    config.openai_web_search_country or "",
                    placeholder="DE",
                    tooltip_text="Optional country hint for localized search results.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_REGION",
                    "Search region",
                    env_values,
                    config.openai_web_search_region or "",
                    placeholder="HH",
                    tooltip_text="Optional region hint for localized search results.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_CITY",
                    "Search city",
                    env_values,
                    config.openai_web_search_city or "",
                    placeholder="Hamburg",
                    tooltip_text="Optional city hint for localized search results.",
                ),
                _text_field(
                    "TWINR_OPENAI_WEB_SEARCH_TIMEZONE",
                    "Search timezone",
                    env_values,
                    config.openai_web_search_timezone or "",
                    placeholder="Europe/Berlin",
                    tooltip_text="Timezone hint used when time-sensitive search answers need local context.",
                ),
            ),
        ),
        SettingsSection(
            title="Conversation flow",
            description="Short-turn timing around button conversations and automatic follow-up listening.",
            fields=(
                _text_field(
                    "TWINR_SPEECH_PAUSE_MS",
                    "Speech pause (ms)",
                    env_values,
                    str(config.speech_pause_ms),
                    tooltip_text="How long Twinr waits after silence before it stops recording a turn.",
                ),
                _select_field(
                    "TWINR_CONVERSATION_FOLLOW_UP_ENABLED",
                    "Follow-up listening",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.conversation_follow_up_enabled else "false",
                    tooltip_text="If enabled, Twinr briefly listens again after each spoken reply.",
                ),
                _text_field(
                    "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S",
                    "Follow-up timeout (s)",
                    env_values,
                    str(config.conversation_follow_up_timeout_s),
                    tooltip_text="Length of the extra listening window after Twinr has answered.",
                ),
                _text_field(
                    "TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS",
                    "Follow-up speech start chunks",
                    env_values,
                    str(config.audio_follow_up_speech_start_chunks),
                    tooltip_text="How many active audio chunks are needed before follow-up recording starts.",
                ),
                _text_field(
                    "TWINR_AUDIO_FOLLOW_UP_IGNORE_MS",
                    "Follow-up ignore (ms)",
                    env_values,
                    str(config.audio_follow_up_ignore_ms),
                    tooltip_text="Short ignore window right after playback so Twinr does not immediately hear itself.",
                ),
            ),
        ),
        SettingsSection(
            title="Audio capture",
            description="Microphone device selection and thresholds for recorded speech turns.",
            fields=(
                _text_field(
                    "TWINR_AUDIO_INPUT_DEVICE",
                    "Input device",
                    env_values,
                    config.audio_input_device,
                    tooltip_text="ALSA input device used for normal button-based recordings.",
                ),
                _text_field(
                    "TWINR_AUDIO_OUTPUT_DEVICE",
                    "Output device",
                    env_values,
                    config.audio_output_device,
                    tooltip_text="ALSA output device used for Twinr speech playback.",
                ),
                _text_field(
                    "TWINR_AUDIO_SAMPLE_RATE",
                    "Sample rate",
                    env_values,
                    str(config.audio_sample_rate),
                    tooltip_text="Capture sample rate for the normal audio recording path.",
                ),
                _text_field(
                    "TWINR_AUDIO_CHANNELS",
                    "Channels",
                    env_values,
                    str(config.audio_channels),
                    tooltip_text="Number of microphone channels used for normal recording.",
                ),
                _text_field(
                    "TWINR_AUDIO_CHUNK_MS",
                    "Chunk size (ms)",
                    env_values,
                    str(config.audio_chunk_ms),
                    tooltip_text="Length of each captured audio chunk while listening.",
                ),
                _text_field(
                    "TWINR_AUDIO_PREROLL_MS",
                    "Preroll (ms)",
                    env_values,
                    str(config.audio_preroll_ms),
                    tooltip_text="Keeps a small amount of audio from just before speech starts.",
                ),
                _text_field(
                    "TWINR_AUDIO_SPEECH_THRESHOLD",
                    "Speech threshold",
                    env_values,
                    str(config.audio_speech_threshold),
                    tooltip_text="Volume threshold used to decide when speech has started.",
                ),
                _text_field(
                    "TWINR_AUDIO_SPEECH_START_CHUNKS",
                    "Speech start chunks",
                    env_values,
                    str(config.audio_speech_start_chunks),
                    tooltip_text="How many active chunks are needed before a new recording officially starts.",
                ),
                _text_field(
                    "TWINR_AUDIO_START_TIMEOUT_S",
                    "Start timeout (s)",
                    env_values,
                    str(config.audio_start_timeout_s),
                    tooltip_text="How long Twinr waits for speech before abandoning a listen attempt.",
                ),
                _text_field(
                    "TWINR_AUDIO_MAX_RECORD_SECONDS",
                    "Max record seconds",
                    env_values,
                    str(config.audio_max_record_seconds),
                    tooltip_text="Upper limit for one captured turn so recordings do not run forever.",
                ),
            ),
        ),
        SettingsSection(
            title="Audio feedback",
            description="Confirmation beeps and optional search progress tones.",
            fields=(
                _text_field(
                    "TWINR_AUDIO_BEEP_FREQUENCY_HZ",
                    "Beep frequency (Hz)",
                    env_values,
                    str(config.audio_beep_frequency_hz),
                    tooltip_text="Pitch of the short confirmation beep.",
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_DURATION_MS",
                    "Beep duration (ms)",
                    env_values,
                    str(config.audio_beep_duration_ms),
                    tooltip_text="Length of the normal confirmation beep.",
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_VOLUME",
                    "Beep volume",
                    env_values,
                    str(config.audio_beep_volume),
                    tooltip_text="Volume multiplier for local beeps.",
                ),
                _text_field(
                    "TWINR_AUDIO_BEEP_SETTLE_MS",
                    "Beep settle (ms)",
                    env_values,
                    str(config.audio_beep_settle_ms),
                    tooltip_text="Small pause after a beep so the next step does not clip.",
                ),
                _select_field(
                    "TWINR_SEARCH_FEEDBACK_TONES_ENABLED",
                    "Search feedback tones",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.search_feedback_tones_enabled else "false",
                    tooltip_text="If enabled, Twinr plays quiet tones while a web search answer is still in progress.",
                ),
                _text_field(
                    "TWINR_SEARCH_FEEDBACK_DELAY_MS",
                    "Search tone delay (ms)",
                    env_values,
                    str(config.search_feedback_delay_ms),
                    tooltip_text="How long Twinr waits before starting search progress tones.",
                ),
                _text_field(
                    "TWINR_SEARCH_FEEDBACK_PAUSE_MS",
                    "Search tone pause (ms)",
                    env_values,
                    str(config.search_feedback_pause_ms),
                    tooltip_text="Pause between search progress tone bursts.",
                ),
                _text_field(
                    "TWINR_SEARCH_FEEDBACK_VOLUME",
                    "Search tone volume",
                    env_values,
                    str(config.search_feedback_volume),
                    tooltip_text="Volume multiplier for the quieter search progress tones.",
                ),
            ),
        ),
        SettingsSection(
            title="Camera and vision",
            description="Camera capture and image understanding settings.",
            fields=(
                _text_field(
                    "TWINR_CAMERA_DEVICE",
                    "Camera device",
                    env_values,
                    config.camera_device,
                    tooltip_text="Video device Twinr uses for still captures and proactive observation.",
                ),
                _text_field(
                    "TWINR_CAMERA_WIDTH",
                    "Camera width",
                    env_values,
                    str(config.camera_width),
                    tooltip_text="Capture width for still images.",
                ),
                _text_field(
                    "TWINR_CAMERA_HEIGHT",
                    "Camera height",
                    env_values,
                    str(config.camera_height),
                    tooltip_text="Capture height for still images.",
                ),
                _text_field(
                    "TWINR_CAMERA_FRAMERATE",
                    "Camera framerate",
                    env_values,
                    str(config.camera_framerate),
                    tooltip_text="Requested frame rate used when ffmpeg grabs from the camera.",
                ),
                _text_field(
                    "TWINR_CAMERA_INPUT_FORMAT",
                    "Camera input format",
                    env_values,
                    config.camera_input_format or "",
                    placeholder="bayer_grbg8",
                    tooltip_text="Optional raw camera pixel format. Leave blank unless the camera needs a specific format.",
                ),
                _text_field(
                    "TWINR_CAMERA_FFMPEG_PATH",
                    "ffmpeg path",
                    env_values,
                    config.camera_ffmpeg_path,
                    tooltip_text="Command path for ffmpeg, which Twinr uses to capture still images.",
                ),
                _select_field(
                    "OPENAI_VISION_DETAIL",
                    "Vision detail",
                    env_values,
                    _VISION_DETAIL_OPTIONS,
                    config.openai_vision_detail,
                    tooltip_text="How much image detail Twinr asks OpenAI to inspect.",
                ),
                _text_field(
                    "TWINR_VISION_REFERENCE_IMAGE",
                    "Reference image path",
                    env_values,
                    config.vision_reference_image_path or "",
                    placeholder="/home/thh/reference-user.jpg",
                    tooltip_text="Optional stored portrait that Twinr can send alongside a live camera frame.",
                ),
            ),
        ),
        SettingsSection(
            title="Proactive behavior",
            description="Bounded idle-time prompts based on PIR, camera, and optional background audio.",
            fields=(
                _text_field(
                    "TWINR_USER_DISPLAY_NAME",
                    "Display name",
                    env_values,
                    config.user_display_name or "",
                    placeholder="Thom",
                    tooltip_text="Name Twinr may use in gentle proactive prompts.",
                ),
                _select_field(
                    "TWINR_PROACTIVE_ENABLED",
                    "Proactive mode",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.proactive_enabled else "false",
                    tooltip_text="Turns the proactive observation service on while Twinr is idle.",
                ),
                _select_field(
                    "TWINR_PROACTIVE_AUDIO_ENABLED",
                    "Background audio",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.proactive_audio_enabled else "false",
                    tooltip_text="Lets the proactive watcher sample ambient audio while Twinr is waiting.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_AUDIO_DEVICE",
                    "Background audio device",
                    env_values,
                    config.proactive_audio_input_device or "",
                    placeholder="plughw:CARD=CameraB409241,DEV=0",
                    tooltip_text="Optional dedicated ALSA input for the background watcher, for example the PS-Eye microphone.",
                ),
                _select_field(
                    "TWINR_PROACTIVE_AUDIO_DISTRESS_ENABLED",
                    "Distress detector",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.proactive_audio_distress_enabled else "false",
                    tooltip_text="Experimental ambient audio detector for stronger distress-like sounds.",
                ),
                _select_field(
                    "TWINR_WAKEWORD_ENABLED",
                    "Wakeword mode",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.wakeword_enabled else "false",
                    tooltip_text="Arms a presence-gated wakeword path so Twinr can listen after it senses someone nearby.",
                ),
                _select_field(
                    "TWINR_WAKEWORD_PRIMARY_BACKEND",
                    "Primary wakeword detector",
                    env_values,
                    _WAKEWORD_PRIMARY_BACKEND_OPTIONS,
                    config.wakeword_primary_backend,
                    help_text=(
                        "openWakeWord is the professional default for passive listening. "
                        "STT stays available as an explicit degraded fallback path."
                    ),
                    tooltip_text="Choose the first-stage detector Twinr uses before any optional lexical verification.",
                ),
                _select_field(
                    "TWINR_WAKEWORD_FALLBACK_BACKEND",
                    "Fallback detector",
                    env_values,
                    _WAKEWORD_FALLBACK_BACKEND_OPTIONS,
                    config.wakeword_fallback_backend,
                    help_text="If the preferred detector cannot run, Twinr can either fall back to STT or disable wakeword activation entirely.",
                    tooltip_text="Fallback only matters when the primary local detector is unavailable or not configured.",
                ),
                _select_field(
                    "TWINR_WAKEWORD_VERIFIER_MODE",
                    "STT verifier",
                    env_values,
                    _WAKEWORD_VERIFIER_MODE_OPTIONS,
                    config.wakeword_verifier_mode,
                    help_text="The verifier is a second-stage text check behind the acoustic detector, not the main wakeword engine.",
                    tooltip_text="Use ambiguity-only to verify only borderline local hits; always is stricter but adds latency.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_VERIFIER_MARGIN",
                    "Verifier margin",
                    env_values,
                    str(config.wakeword_verifier_margin),
                    tooltip_text="Extra score band above the local threshold where Twinr still asks the verifier before opening a turn.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_PHRASES",
                    "Wakeword phrases",
                    env_values,
                    _csv_display(getattr(config, "wakeword_phrases", None)),  # AUDIT-FIX(#4): Optional wakeword phrases must render as an empty field, not raise TypeError.
                    placeholder="hey twinr, hey twinna, twinr",
                    wide=True,
                    tooltip_text="Comma-separated wakeword aliases that Twinr treats as valid starts for a hands-free turn.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_OPENWAKEWORD_MODELS",
                    "openWakeWord models",
                    env_values,
                    _csv_display(getattr(config, "wakeword_openwakeword_models", None)),  # AUDIT-FIX(#4): Optional wakeword model lists must render as an empty field, not raise TypeError.
                    placeholder="state/wakeword_models/hey_twinr.tflite",
                    wide=True,
                    tooltip_text="Comma-separated local model paths used by openWakeWord.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_CALIBRATION_PROFILE_PATH",
                    "Calibration profile",
                    env_values,
                    config.wakeword_calibration_profile_path,
                    placeholder="state/wakeword_calibration.json",
                    wide=True,
                    tooltip_text="Active per-device wakeword calibration file that tunes thresholds for the real room.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_CALIBRATION_RECOMMENDED_PATH",
                    "Recommended profile",
                    env_values,
                    config.wakeword_calibration_recommended_path,
                    placeholder="state/wakeword_calibration.recommended.json",
                    wide=True,
                    tooltip_text="Where Twinr writes autotune recommendations before an operator decides whether to apply them.",
                ),
            ),
        ),
        SettingsSection(
            title="Proactive timing",
            description="Idle timing, capture spacing, and per-trigger hold durations. Lower values react faster; higher values reduce chatter.",
            fields=(
                _text_field(
                    "TWINR_PROACTIVE_POLL_INTERVAL_S",
                    "Wake pause (s)",
                    env_values,
                    str(config.proactive_poll_interval_s),
                    tooltip_text="How long Twinr sleeps between proactive monitor wakeups while idle.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_CAPTURE_INTERVAL_S",
                    "Camera pause (s)",
                    env_values,
                    str(config.proactive_capture_interval_s),
                    tooltip_text="Minimum pause between proactive camera inspections.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_MOTION_WINDOW_S",
                    "Motion window (s)",
                    env_values,
                    str(config.proactive_motion_window_s),
                    tooltip_text="How long recent PIR motion keeps proactive inspection active before Twinr goes quiet again.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_LOW_MOTION_AFTER_S",
                    "Idle after (s)",
                    env_values,
                    str(config.proactive_low_motion_after_s),
                    tooltip_text="After this many quiet seconds without motion, the scene is treated as idle / low-motion.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_AUDIO_SAMPLE_MS",
                    "Background sample (ms)",
                    env_values,
                    str(config.proactive_audio_sample_ms),
                    tooltip_text="Length of each ambient audio sample window.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_SAMPLE_MS",
                    "Wakeword sample (ms)",
                    env_values,
                    str(config.wakeword_sample_ms),
                    tooltip_text="Length of the ambient audio window Twinr sends through phrase spotting when wakeword mode is armed.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_PRESENCE_GRACE_S",
                    "Wake presence grace (s)",
                    env_values,
                    str(config.wakeword_presence_grace_s),
                    tooltip_text="How long recent visible presence keeps wakeword mode armed after Twinr last saw someone.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_MOTION_GRACE_S",
                    "Wake motion grace (s)",
                    env_values,
                    str(config.wakeword_motion_grace_s),
                    tooltip_text="How long recent PIR motion alone keeps wakeword mode armed.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_SPEECH_GRACE_S",
                    "Wake speech grace (s)",
                    env_values,
                    str(config.wakeword_speech_grace_s),
                    tooltip_text="How long recent room speech can extend an already present wakeword session.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_ATTEMPT_COOLDOWN_S",
                    "Wake retry cooldown (s)",
                    env_values,
                    str(config.wakeword_attempt_cooldown_s),
                    tooltip_text="Minimum pause between wakeword phrase-spotting attempts, to avoid repeated triggers from one burst of room speech.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_MIN_ACTIVE_RATIO",
                    "Wake audio ratio",
                    env_values,
                    str(config.wakeword_min_active_ratio),
                    tooltip_text="Minimum active-chunk ratio before Twinr treats ambient audio as a wakeword candidate.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_MIN_ACTIVE_CHUNKS",
                    "Wake active chunks",
                    env_values,
                    str(config.wakeword_min_active_chunks),
                    tooltip_text="Minimum number of active background-audio chunks before Twinr even tries wakeword phrase spotting.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_OPENWAKEWORD_PATIENCE_FRAMES",
                    "Wake patience frames",
                    env_values,
                    str(config.wakeword_openwakeword_patience_frames),
                    tooltip_text="How many adjacent openWakeWord frames may stay below threshold before a detection candidate is dropped.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_PERSON_RETURNED_ABSENCE_S",
                    "Return absence (s)",
                    env_values,
                    str(config.proactive_person_returned_absence_s),
                    tooltip_text="How long someone must be absent before Twinr may greet them as returned.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_PERSON_RETURNED_RECENT_MOTION_S",
                    "Return motion recency (s)",
                    env_values,
                    str(config.proactive_person_returned_recent_motion_s),
                    tooltip_text="How recent PIR motion must be to count as a real return instead of stale presence.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_ATTENTION_WINDOW_S",
                    "Attention hold (s)",
                    env_values,
                    str(config.proactive_attention_window_s),
                    tooltip_text="How long someone must look toward Twinr while staying quiet before it offers help.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SLUMPED_QUIET_S",
                    "Slumped quiet hold (s)",
                    env_values,
                    str(config.proactive_slumped_quiet_s),
                    tooltip_text="How long a slumped, quiet, low-motion posture must persist before Twinr checks in.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSSIBLE_FALL_STILLNESS_S",
                    "Fall stillness hold (s)",
                    env_values,
                    str(config.proactive_possible_fall_stillness_s),
                    tooltip_text="How long the post-fall floor state must stay still before Twinr asks about help.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_FLOOR_STILLNESS_S",
                    "Floor stillness hold (s)",
                    env_values,
                    str(config.proactive_floor_stillness_s),
                    tooltip_text="How long someone must remain low, quiet, and still before Twinr asks for a response.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SHOWING_INTENT_HOLD_S",
                    "Showing hold (s)",
                    env_values,
                    str(config.proactive_showing_intent_hold_s),
                    tooltip_text="How long a hand or object near the camera should persist before Twinr asks whether you want to show something.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSITIVE_CONTACT_HOLD_S",
                    "Smile hold (s)",
                    env_values,
                    str(config.proactive_positive_contact_hold_s),
                    tooltip_text="How long a visible smile toward the device should persist before Twinr opens a positive greeting.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_DISTRESS_HOLD_S",
                    "Distress hold (s)",
                    env_values,
                    str(config.proactive_distress_hold_s),
                    tooltip_text="How long the distress-like audio pattern must last before Twinr gently checks in.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_FALL_TRANSITION_WINDOW_S",
                    "Fall transition window (s)",
                    env_values,
                    str(config.proactive_fall_transition_window_s),
                    tooltip_text="Maximum time between an upright posture and a floor posture for the event to count as a possible fall.",
                ),
            ),
        ),
        SettingsSection(
            title="Proactive sensitivity",
            description="Minimum normalized evidence score per proactive trigger. Range 0.0 to 1.0. Lower values trigger more easily.",
            fields=(
                _text_field(
                    "TWINR_PROACTIVE_PERSON_RETURNED_SCORE_THRESHOLD",
                    "Person returned min score",
                    env_values,
                    str(config.proactive_person_returned_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `person_returned` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_ATTENTION_WINDOW_SCORE_THRESHOLD",
                    "Attention min score",
                    env_values,
                    str(config.proactive_attention_window_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `attention_window` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SLUMPED_QUIET_SCORE_THRESHOLD",
                    "Slumped quiet min score",
                    env_values,
                    str(config.proactive_slumped_quiet_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `slumped_quiet` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSSIBLE_FALL_SCORE_THRESHOLD",
                    "Possible fall min score",
                    env_values,
                    str(config.proactive_possible_fall_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `possible_fall` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_FLOOR_STILLNESS_SCORE_THRESHOLD",
                    "Floor stillness min score",
                    env_values,
                    str(config.proactive_floor_stillness_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `floor_stillness` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD",
                    "Showing intent min score",
                    env_values,
                    str(config.proactive_showing_intent_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `showing_intent` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_POSITIVE_CONTACT_SCORE_THRESHOLD",
                    "Positive contact min score",
                    env_values,
                    str(config.proactive_positive_contact_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `positive_contact` trigger.",
                ),
                _text_field(
                    "TWINR_PROACTIVE_DISTRESS_POSSIBLE_SCORE_THRESHOLD",
                    "Distress min score",
                    env_values,
                    str(config.proactive_distress_possible_score_threshold),
                    tooltip_text="Minimum combined evidence score for the `distress_possible` trigger.",
                ),
                _text_field(
                    "TWINR_WAKEWORD_OPENWAKEWORD_THRESHOLD",
                    "Wakeword min score",
                    env_values,
                    str(config.wakeword_openwakeword_threshold),
                    tooltip_text="Minimum openWakeWord detection score before Twinr accepts a wakeword match.",
                ),
            ),
        ),
        SettingsSection(
            title="Web UI and files",
            description="Operator UI binding plus the main local file paths Twinr writes to.",
            fields=(
                _text_field(
                    "TWINR_WEB_HOST",
                    "Web host",
                    env_values,
                    config.web_host,
                    tooltip_text="Host interface for the local settings dashboard.",
                ),
                _text_field(
                    "TWINR_WEB_PORT",
                    "Web port",
                    env_values,
                    str(config.web_port),
                    tooltip_text="Port for the local settings dashboard.",
                ),
                _text_field(
                    "TWINR_PERSONALITY_DIR",
                    "Personality directory",
                    env_values,
                    config.personality_dir,
                    tooltip_text="Folder containing SYSTEM.md, PERSONALITY.md, and USER.md.",
                ),
                _text_field(
                    "TWINR_RUNTIME_STATE_PATH",
                    "Runtime state path",
                    env_values,
                    config.runtime_state_path,
                    tooltip_text="JSON snapshot file the running Twinr process updates for the dashboard.",
                ),
                _text_field(
                    "TWINR_MEMORY_MARKDOWN_PATH",
                    "Memory markdown path",
                    env_values,
                    config.memory_markdown_path,
                    tooltip_text="Markdown file where durable memories are stored.",
                ),
                _text_field(
                    "TWINR_REMINDER_STORE_PATH",
                    "Reminder store path",
                    env_values,
                    config.reminder_store_path,
                    tooltip_text="JSON file used for saved reminders.",
                ),
                _select_field(
                    "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP",
                    "Restore runtime state",
                    env_values,
                    _BOOL_OPTIONS,
                    "true" if config.restore_runtime_state_on_startup else "false",
                    tooltip_text="If enabled, Twinr restores the last saved runtime snapshot at startup.",
                ),
                _text_field(
                    "TWINR_REMINDER_POLL_INTERVAL_S",
                    "Reminder poll interval (s)",
                    env_values,
                    str(config.reminder_poll_interval_s),
                    tooltip_text="How often Twinr checks whether a reminder is due.",
                ),
                _text_field(
                    "TWINR_REMINDER_RETRY_DELAY_S",
                    "Reminder retry delay (s)",
                    env_values,
                    str(config.reminder_retry_delay_s),
                    tooltip_text="Delay before a reminder is retried after a failed attempt.",
                ),
                _text_field(
                    "TWINR_REMINDER_MAX_ENTRIES",
                    "Reminder max entries",
                    env_values,
                    str(config.reminder_max_entries),
                    tooltip_text="Upper bound for stored reminder entries.",
                ),
            ),
        ),
        SettingsSection(
            title="Buttons and motion sensor",
            description="GPIO wiring and debounce settings for buttons and the PIR sensor.",
            fields=(
                _text_field(
                    "TWINR_GPIO_CHIP",
                    "GPIO chip",
                    env_values,
                    config.gpio_chip,
                    tooltip_text="GPIO chip device name used for all button, PIR, and display lines.",
                ),
                _text_field(
                    "TWINR_GREEN_BUTTON_GPIO",
                    "Green button GPIO",
                    env_values,
                    green_button_gpio,
                    tooltip_text="GPIO line number for the green conversation button.",
                ),
                _text_field(
                    "TWINR_YELLOW_BUTTON_GPIO",
                    "Yellow button GPIO",
                    env_values,
                    yellow_button_gpio,
                    tooltip_text="GPIO line number for the yellow print button.",
                ),
                _select_field(
                    "TWINR_BUTTON_ACTIVE_LOW",
                    "Buttons active low",
                    env_values,
                    _YES_NO_OPTIONS,
                    "true" if config.button_active_low else "false",
                    tooltip_text="Set this to Yes when the button pulls the line low when pressed.",
                ),
                _select_field(
                    "TWINR_BUTTON_BIAS",
                    "Button bias",
                    env_values,
                    _GPIO_BIAS_OPTIONS,
                    config.button_bias,
                    tooltip_text="Internal pull configuration for the button lines.",
                ),
                _text_field(
                    "TWINR_BUTTON_DEBOUNCE_MS",
                    "Button debounce (ms)",
                    env_values,
                    str(config.button_debounce_ms),
                    tooltip_text="Debounce time applied to button presses.",
                ),
                _text_field(
                    "TWINR_BUTTON_PROBE_LINES",
                    "Button probe lines",
                    env_values,
                    button_probe_lines,
                    help_text="Comma-separated GPIO line numbers.",
                    tooltip_text="Fallback GPIO lines the button setup tools probe during hardware discovery.",
                ),
                _text_field(
                    "TWINR_PIR_MOTION_GPIO",
                    "PIR motion GPIO",
                    env_values,
                    pir_motion_gpio,
                    tooltip_text="GPIO line number connected to the PIR motion sensor.",
                ),
                _select_field(
                    "TWINR_PIR_ACTIVE_HIGH",
                    "PIR active high",
                    env_values,
                    _YES_NO_OPTIONS,
                    "true" if config.pir_active_high else "false",
                    tooltip_text="Set this to Yes when the PIR sensor drives the line high on motion.",
                ),
                _select_field(
                    "TWINR_PIR_BIAS",
                    "PIR bias",
                    env_values,
                    _GPIO_BIAS_OPTIONS,
                    config.pir_bias,
                    tooltip_text="Internal pull configuration for the PIR input line.",
                ),
                _text_field(
                    "TWINR_PIR_DEBOUNCE_MS",
                    "PIR debounce (ms)",
                    env_values,
                    str(config.pir_debounce_ms),
                    tooltip_text="Debounce time applied to PIR motion edges.",
                ),
            ),
        ),
        SettingsSection(
            title="Display and printer",
            description="E-paper wiring plus printer queue and receipt layout.",
            fields=(
                _text_field(
                    "TWINR_DISPLAY_DRIVER",
                    "Display driver",
                    env_values,
                    config.display_driver,
                    tooltip_text="Driver id for the configured e-paper display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_VENDOR_DIR",
                    "Display vendor dir",
                    env_values,
                    config.display_vendor_dir,
                    tooltip_text="Path to the vendor display driver files.",
                ),
                _text_field(
                    "TWINR_DISPLAY_SPI_BUS",
                    "Display SPI bus",
                    env_values,
                    str(config.display_spi_bus),
                    tooltip_text="SPI bus number used by the e-paper display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_SPI_DEVICE",
                    "Display SPI device",
                    env_values,
                    str(config.display_spi_device),
                    tooltip_text="SPI device number used by the e-paper display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_CS_GPIO",
                    "Display CS GPIO",
                    env_values,
                    str(config.display_cs_gpio),
                    tooltip_text="GPIO line used for display chip select.",
                ),
                _text_field(
                    "TWINR_DISPLAY_DC_GPIO",
                    "Display DC GPIO",
                    env_values,
                    str(config.display_dc_gpio),
                    tooltip_text="GPIO line used for display data/command switching.",
                ),
                _text_field(
                    "TWINR_DISPLAY_RESET_GPIO",
                    "Display reset GPIO",
                    env_values,
                    str(config.display_reset_gpio),
                    tooltip_text="GPIO line used to reset the display panel.",
                ),
                _text_field(
                    "TWINR_DISPLAY_BUSY_GPIO",
                    "Display busy GPIO",
                    env_values,
                    str(config.display_busy_gpio),
                    tooltip_text="GPIO line that reports when the display is still busy refreshing.",
                ),
                _text_field(
                    "TWINR_DISPLAY_WIDTH",
                    "Display width",
                    env_values,
                    str(config.display_width),
                    tooltip_text="Logical display width in pixels.",
                ),
                _text_field(
                    "TWINR_DISPLAY_HEIGHT",
                    "Display height",
                    env_values,
                    str(config.display_height),
                    tooltip_text="Logical display height in pixels.",
                ),
                _text_field(
                    "TWINR_DISPLAY_ROTATION_DEGREES",
                    "Display rotation",
                    env_values,
                    str(config.display_rotation_degrees),
                    tooltip_text="Rotation applied before content is drawn to the display.",
                ),
                _text_field(
                    "TWINR_DISPLAY_FULL_REFRESH_INTERVAL",
                    "Full refresh interval",
                    env_values,
                    str(config.display_full_refresh_interval),
                    tooltip_text="How many partial updates happen before forcing a full display refresh.",
                ),
                _text_field(
                    "TWINR_DISPLAY_POLL_INTERVAL_S",
                    "Display poll interval (s)",
                    env_values,
                    str(config.display_poll_interval_s),
                    tooltip_text="How often the display service checks for new runtime state.",
                ),
                _text_field(
                    "TWINR_PRINTER_QUEUE",
                    "Printer queue",
                    env_values,
                    config.printer_queue,
                    tooltip_text="CUPS queue name for the receipt printer.",
                ),
                _text_field(
                    "TWINR_PRINTER_DEVICE_URI",
                    "Printer device URI",
                    env_values,
                    config.printer_device_uri or "",
                    placeholder="usb://...",
                    tooltip_text="Optional direct CUPS device URI for the printer.",
                ),
                _text_field(
                    "TWINR_PRINTER_HEADER_TEXT",
                    "Printer header",
                    env_values,
                    config.printer_header_text,
                    tooltip_text="Header text printed at the top of each receipt.",
                ),
                _text_field(
                    "TWINR_PRINTER_LINE_WIDTH",
                    "Printer line width",
                    env_values,
                    str(config.printer_line_width),
                    help_text="Lower values make the receipt narrower.",
                    tooltip_text="Maximum characters per printed line.",
                ),
                _text_field(
                    "TWINR_PRINTER_FEED_LINES",
                    "Printer feed lines",
                    env_values,
                    str(config.printer_feed_lines),
                    tooltip_text="Blank lines fed after a print job completes.",
                ),
                _text_field(
                    "TWINR_PRINT_BUTTON_COOLDOWN_S",
                    "Print button cooldown (s)",
                    env_values,
                    str(config.print_button_cooldown_s),
                    tooltip_text="Minimum delay between yellow-button print actions.",
                ),
            ),
        ),
    )
