"""Run bounded live LLM acceptance for guided user-discovery.

Purpose
-------
Exercise Twinr's real tool-calling provider path for guided user-discovery
against ``gpt-5.4-mini``. The script runs a small fixed discovery dialogue,
verifies that the live model triggers discovery review/correction/delete tool
calls, and then attests the resulting managed-context and structured-memory
state from an isolated runtime workspace.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_user_discovery_live_acceptance.py --env-file .env
    PYTHONPATH=src python3 test/run_user_discovery_live_acceptance.py --env-file /twinr/.env --output artifacts/reports/user_discovery_live.json

Inputs
------
- ``--env-file``: Twinr env file used for live provider credentials and remote
  memory access.
- ``--run-id``: Optional suffix for the isolated remote namespace.
- ``--output``: Optional JSON artifact path.

Outputs
-------
- JSON summary written to stdout.
- Optional JSON artifact written to ``--output``.
- Exit code 0 when all acceptance checks pass, 1 otherwise.

Notes
-----
The script intentionally uses an isolated temporary workspace and a unique
remote namespace so the live acceptance run does not mutate Twinr's productive
managed context, graph memory, or durable prompt-memory namespace.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.tools import (
    ToolCallingStreamingLoop,
    build_agent_tool_schemas,
    build_tool_agent_instructions,
)
from twinr.agent.tools.handlers.user_discovery import handle_manage_user_discovery
from twinr.providers.openai import OpenAIBackend, OpenAIToolCallingAgentProvider


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    """Describe one bounded pass/fail assertion for the live acceptance run."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class TurnArtifact:
    """Capture one live provider turn plus the observed tool calls."""

    prompt: str
    answer: str
    tool_calls: tuple[str, ...]
    raw_tool_calls: tuple[dict[str, object], ...]
    raw_tool_results: tuple[dict[str, object], ...]
    model: str | None
    emitted: tuple[str, ...]
    status: str


def _normalize_project_root(env_path: Path, config: TwinrConfig) -> Path:
    """Resolve the authoritative project root for one env-backed config."""

    configured = Path(str(getattr(config, "project_root", ".") or ".")).expanduser()
    if configured.is_absolute():
        return configured.resolve(strict=False)
    return (env_path.parent.resolve(strict=False) / configured).resolve(strict=False)


def _resolve_personality_dir(base_project_root: Path, config: TwinrConfig) -> Path:
    """Resolve the personality directory that live acceptance should clone."""

    raw_personality_dir = Path(str(getattr(config, "personality_dir", "personality") or "personality")).expanduser()
    candidate = raw_personality_dir if raw_personality_dir.is_absolute() else (base_project_root / raw_personality_dir)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Personality directory not found for live discovery acceptance: {resolved}")
    return resolved


def _safe_namespace_suffix(value: str | None) -> str:
    """Normalize a free-form id into a remote-namespace-safe suffix."""

    safe_chars: list[str] = []
    for char in str(value or "").lower():
        if char.isalnum():
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    normalized = "".join(safe_chars).strip("_")
    return normalized or "discovery_live"


def _json_safe(value: object) -> object:
    """Convert a nested object into a JSON-safe structure."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Return one whitespace-normalized short string for diagnostics."""

    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."


class LiveDiscoveryContext:
    """Own one isolated live-discovery acceptance workspace."""

    def __init__(self, *, base_env_path: Path, run_id: str | None) -> None:
        self.base_env_path = base_env_path.resolve()
        self.base_config = TwinrConfig.from_env(self.base_env_path)
        self.base_project_root = _normalize_project_root(self.base_env_path, self.base_config)
        self.temp_dir = TemporaryDirectory(prefix="twinr-user-discovery-live-")
        self.root = Path(self.temp_dir.name)
        self.state_dir = self.root / "state"
        self.personality_dir = self.root / "personality"
        self.env_path = self.root / ".env"
        self.namespace = f"user_discovery_live_{_safe_namespace_suffix(run_id or uuid4().hex[:12])}"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            _resolve_personality_dir(self.base_project_root, self.base_config),
            self.personality_dir,
            dirs_exist_ok=True,
        )
        user_path = self.personality_dir / "USER.md"
        user_path.write_text("", encoding="utf-8")
        base_env_text = self.base_env_path.read_text(encoding="utf-8")
        overrides = [
            "OPENAI_MODEL=gpt-5.4-mini",
            "TWINR_LLM_PROVIDER=openai",
            "TWINR_PROACTIVE_ENABLED=false",
            "TWINR_WAKEWORD_ENABLED=false",
            "TWINR_CONVERSATION_FOLLOW_UP_ENABLED=false",
            "TWINR_SEARCH_FEEDBACK_TONES_ENABLED=false",
            "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP=false",
            "TWINR_LONG_TERM_MEMORY_ENABLED=true",
            "TWINR_LONG_TERM_MEMORY_MODE=remote_primary",
            "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED=true",
            "TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS=false",
            "TWINR_LONG_TERM_MEMORY_REMOTE_RUNTIME_CHECK_MODE=direct",
            f"TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE={self.namespace}",
            "TWINR_PERSONALITY_DIR=personality",
            f"TWINR_RUNTIME_STATE_PATH={self.state_dir / 'runtime-state.json'}",
            f"TWINR_MEMORY_MARKDOWN_PATH={self.state_dir / 'MEMORY.md'}",
            f"TWINR_REMINDER_STORE_PATH={self.state_dir / 'reminders.json'}",
            f"TWINR_AUTOMATION_STORE_PATH={self.state_dir / 'automations.json'}",
            f"TWINR_VOICE_PROFILE_STORE_PATH={self.state_dir / 'voice_profile.json'}",
            f"TWINR_ADAPTIVE_TIMING_STORE_PATH={self.state_dir / 'adaptive_timing.json'}",
            f"TWINR_LONG_TERM_MEMORY_PATH={self.state_dir / 'chonkydb'}",
        ]
        self.env_path.write_text(base_env_text.rstrip() + "\n" + "\n".join(overrides) + "\n", encoding="utf-8")

    def close(self) -> None:
        """Release the temporary acceptance workspace."""

        self.temp_dir.cleanup()

    def make_harness(self, *, emitted: list[str]) -> SimpleNamespace:
        """Build one direct live tool-loop harness in the isolated workspace."""

        config = TwinrConfig.from_env(self.env_path)
        runtime = TwinrRuntime(config=config)
        runtime.user_voice_status = "identified"
        runtime.user_voice_confidence = 0.99
        runtime.user_voice_user_id = "live-acceptance"
        runtime.user_voice_user_display_name = "Live Acceptance"
        owner = SimpleNamespace(
            config=config,
            runtime=runtime,
            emit=emitted.append,
            _record_event=lambda *args, **kwargs: None,
            _current_turn_audio_pcm=b"synthetic",
            _current_turn_audio_sample_rate=config.openai_realtime_input_sample_rate,
        )
        tool_name = "manage_user_discovery"
        provider = OpenAIToolCallingAgentProvider(OpenAIBackend(config=config))
        tool_loop = ToolCallingStreamingLoop(
            provider,
            tool_handlers={tool_name: lambda arguments: handle_manage_user_discovery(owner, arguments)},
            tool_schemas=build_agent_tool_schemas((tool_name,)),
        )
        return SimpleNamespace(
            config=config,
            runtime=runtime,
            owner=owner,
            tool_loop=tool_loop,
        )


def run_text_turn(
    harness: SimpleNamespace,
    prompt: str,
) -> TurnArtifact:
    """Run one live tool-capable turn and capture the observed tool calls."""

    emitted: list[str] = []
    harness.owner.emit = emitted.append
    now_iso = datetime.now().astimezone().isoformat()
    harness.runtime.user_voice_status = "identified"
    harness.runtime.user_voice_confidence = 0.99
    harness.runtime.user_voice_checked_at = now_iso
    harness.runtime.user_voice_user_id = "live-acceptance"
    harness.runtime.user_voice_user_display_name = "Live Acceptance"
    harness.runtime.user_voice_match_source = "acceptance_harness"
    harness.runtime.begin_listening(request_source="user_discovery_live_acceptance")
    harness.runtime.submit_transcript(prompt)
    response = harness.tool_loop.run(
        prompt,
        conversation=harness.runtime.tool_provider_conversation_context(),
        instructions=build_tool_agent_instructions(
            harness.config,
            extra_instructions=harness.config.openai_realtime_instructions,
        ),
        allow_web_search=False,
    )
    harness.runtime.begin_answering()
    answer = harness.runtime.finalize_agent_turn(response.text)
    harness.runtime.record_personality_tool_history(
        tool_calls=response.tool_calls,
        tool_results=response.tool_results,
    )
    harness.runtime.finish_speaking()
    return TurnArtifact(
        prompt=prompt,
        answer=answer,
        tool_calls=tuple(call.name for call in response.tool_calls),
        raw_tool_calls=tuple(
            {
                "name": call.name,
                "arguments": _json_safe(call.arguments),
            }
            for call in response.tool_calls
        ),
        raw_tool_results=tuple(
            {
                "name": result.name,
                "output": _json_safe(result.output),
            }
            for result in response.tool_results
        ),
        model=_compact_text(response.model, max_len=80) or None,
        emitted=tuple(emitted),
        status=harness.runtime.status.value,
    )


def _tool_actions(artifact: TurnArtifact, *, tool_name: str) -> tuple[str, ...]:
    """Extract ordered action names for one tool from a turn artifact."""

    actions: list[str] = []
    for item in artifact.raw_tool_calls:
        if item.get("name") != tool_name:
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            continue
        action = _compact_text(arguments.get("action"), max_len=48)
        if action:
            actions.append(action)
    return tuple(actions)


def _route_kinds(artifact: TurnArtifact) -> tuple[str, ...]:
    """Collect all discovery route/storage kinds seen in one turn."""

    kinds: list[str] = []
    for item in artifact.raw_tool_calls:
        if item.get("name") != "manage_user_discovery":
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            continue
        learned_facts = arguments.get("learned_facts")
        if isinstance(learned_facts, list):
            for fact in learned_facts:
                if isinstance(fact, dict):
                    storage = _compact_text(fact.get("storage"), max_len=32)
                    if storage:
                        kinds.append(storage)
        memory_routes = arguments.get("memory_routes")
        if isinstance(memory_routes, list):
            for route in memory_routes:
                if isinstance(route, dict):
                    route_kind = _compact_text(route.get("route_kind"), max_len=32)
                    if route_kind:
                        kinds.append(route_kind)
    return tuple(kinds)


def _contains_folded(haystack: str, needle: str) -> bool:
    """Return whether one compact case-folded string contains another."""

    return " ".join(needle.split()).casefold() in " ".join(haystack.split()).casefold()


def _discovery_lines(text: str) -> tuple[str, ...]:
    """Return only discovery-managed context lines from a markdown profile file."""

    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        compact = " ".join(raw_line.split()).strip()
        if "user_discovery_" in compact:
            lines.append(compact)
    return tuple(lines)


def run_acceptance(*, env_file: Path, run_id: str | None) -> dict[str, object]:
    """Execute the fixed live discovery scenario and return one JSON-safe report."""

    context = LiveDiscoveryContext(base_env_path=env_file, run_id=run_id)
    harness: SimpleNamespace | None = None
    try:
        harness = context.make_harness(emitted=[])
        turns = (
            run_text_turn(
                harness,
                "Ich moechte jetzt den Kennenlern-Dialog fuer mein Profil starten. Bitte beginne mit deinen Kennenlernfragen.",
            ),
            run_text_turn(
                harness,
                (
                    "Bitte erfasse im Kennenlernen Folgendes: Ich moechte als Thom angesprochen werden. "
                    "Du kannst mich duzen. Bitte speichere Anna als meine Tochter. "
                    "Bitte speichere Melitta als meine bevorzugte Kaffeemarke. "
                    "Fuer morgen habe ich vor, Anna anzurufen. Ja, bitte jetzt speichern."
                ),
            ),
            run_text_turn(
                harness,
                (
                    "Zeig mir bitte, was du ueber mich gelernt hast, und korrigiere es direkt: "
                    "Nenn mich kuenftig Tom statt Thom, und den Plan Anna morgen anzurufen "
                    "kannst du wieder loeschen. Ja, bitte jetzt aendern und den Plan loeschen."
                ),
            ),
        )

        lookup_before = harness.runtime.lookup_contact(name="Anna")
        graph_context_after = harness.runtime.graph_memory.build_prompt_context("Anna Melitta tomorrow Tom")
        final_review = harness.runtime.manage_user_discovery(action="review_profile")
        user_text = (context.personality_dir / "USER.md").read_text(encoding="utf-8")
        personality_path = context.personality_dir / "PERSONALITY.md"
        personality_text = personality_path.read_text(encoding="utf-8") if personality_path.exists() else ""
        memory_path = context.state_dir / "MEMORY.md"
        memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
        discovery_state_path = context.state_dir / "user_discovery.json"
        discovery_state = json.loads(discovery_state_path.read_text(encoding="utf-8")) if discovery_state_path.exists() else {}
        review_summaries = tuple(item.summary for item in final_review.review_items)
        user_discovery_lines = _discovery_lines(user_text)
        personality_discovery_lines = _discovery_lines(personality_text)

        model_names = tuple(turn.model for turn in turns if turn.model)
        start_actions = _tool_actions(turns[0], tool_name="manage_user_discovery")
        learning_actions = _tool_actions(turns[1], tool_name="manage_user_discovery")
        correction_actions = _tool_actions(turns[2], tool_name="manage_user_discovery")
        learning_route_kinds = _route_kinds(turns[1])

        checks = (
            AcceptanceCheck(
                name="live_model_is_gpt_5_4_mini",
                passed=bool(model_names) and all(str(name).startswith("gpt-5.4-mini") for name in model_names),
                detail=f"models={model_names}",
            ),
            AcceptanceCheck(
                name="start_turn_called_discovery",
                passed="start_or_resume" in start_actions,
                detail=f"actions={start_actions}",
            ),
            AcceptanceCheck(
                name="learning_turn_used_structured_routes",
                passed=(
                    "answer" in learning_actions
                    and {"user_profile", "contact", "preference", "plan"}.issubset(set(learning_route_kinds))
                ),
                detail=f"actions={learning_actions}; route_kinds={learning_route_kinds}",
            ),
            AcceptanceCheck(
                name="correction_turn_reviewed_and_mutated",
                passed={"review_profile", "replace_fact", "delete_fact"}.issubset(set(correction_actions)),
                detail=f"actions={correction_actions}",
            ),
            AcceptanceCheck(
                name="managed_context_updated_after_replace",
                passed=(
                    any(_contains_folded(summary, "Tom") for summary in review_summaries)
                    and not any(_contains_folded(summary, "Thom") for summary in review_summaries)
                ),
                detail=f"review_summaries={review_summaries}",
            ),
            AcceptanceCheck(
                name="structured_memory_used_route_beyond_managed_context",
                passed=(
                    any(kind in {"contact", "preference", "plan"} for kind in learning_route_kinds)
                    and not any(_contains_folded(summary, "Call Anna") for summary in review_summaries)
                ),
                detail=f"route_kinds={learning_route_kinds}; review_summaries={review_summaries}",
            ),
            AcceptanceCheck(
                name="deleted_plan_missing_from_review",
                passed=(
                    any(_contains_folded(summary, "Tom") for summary in review_summaries)
                    and not any(_contains_folded(summary, "Call Anna") for summary in review_summaries)
                ),
                detail=f"review_summaries={review_summaries}",
            ),
        )

        return {
            "passed": all(check.passed for check in checks),
            "requested_model": harness.config.default_model,
            "actual_models": model_names,
            "remote_namespace": context.namespace,
            "workspace_root": str(context.root),
            "turns": [asdict(turn) for turn in turns],
            "checks": [asdict(check) for check in checks],
            "user_profile_excerpt": _compact_text(user_text, max_len=280),
            "personality_excerpt": _compact_text(personality_text, max_len=280),
            "user_discovery_lines": user_discovery_lines,
            "personality_discovery_lines": personality_discovery_lines,
            "memory_excerpt": _compact_text(memory_text, max_len=280),
            "graph_context_excerpt": _compact_text(graph_context_after, max_len=280),
            "final_review_summaries": review_summaries,
            "discovery_state": _json_safe(discovery_state),
        }
    finally:
        if harness is not None:
            try:
                harness.runtime.shutdown(timeout_s=1.0)
            except Exception:
                pass
        context.close()


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run acceptance, and emit JSON output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Twinr env file used for live provider credentials.")
    parser.add_argument("--run-id", default=None, help="Optional suffix for the isolated remote namespace.")
    parser.add_argument("--output", default=None, help="Optional JSON artifact path.")
    args = parser.parse_args(argv)

    result = run_acceptance(
        env_file=Path(args.env_file),
        run_id=args.run_id,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if bool(result.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
