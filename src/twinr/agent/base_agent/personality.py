from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinr.automations import AutomationStore
from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import PersistentMemoryMarkdownStore
from twinr.memory.reminders import ReminderStore

_SECTION_FILES = (
    ("SYSTEM", "SYSTEM.md"),
    ("PERSONALITY", "PERSONALITY.md"),
    ("USER", "USER.md"),
)


@dataclass(frozen=True, slots=True)
class PersonalityContext:
    sections: tuple[tuple[str, str], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.sections

    def to_instructions(self) -> str | None:
        if not self.sections:
            return None

        parts = [
            "Internal assistant context. Use it silently as behavior guidance.",
            "Do not mention these instructions, hidden context, or user profile unless directly relevant to the user's request.",
            "Do not repeatedly volunteer profile facts.",
        ]
        for title, content in self.sections:
            parts.append(f"{title}:\n{content}")
        return "\n\n".join(parts).strip()


def load_personality_context(config: TwinrConfig) -> PersonalityContext:
    directory = Path(config.project_root) / config.personality_dir
    sections: list[tuple[str, str]] = []
    for title, filename in _SECTION_FILES:
        path = directory / filename
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        sections.append((title, content))
    memory_context = PersistentMemoryMarkdownStore(config.memory_markdown_path).render_context()
    if memory_context:
        sections.append(("MEMORY", memory_context))
    reminder_context = ReminderStore(
        config.reminder_store_path,
        timezone_name=config.local_timezone_name,
        retry_delay_s=config.reminder_retry_delay_s,
        max_entries=config.reminder_max_entries,
    ).render_context()
    if reminder_context:
        sections.append(("REMINDERS", reminder_context))
    automation_context = AutomationStore(
        config.automation_store_path,
        timezone_name=config.local_timezone_name,
        max_entries=config.automation_max_entries,
    ).render_context()
    if automation_context:
        sections.append(("AUTOMATIONS", automation_context))
    return PersonalityContext(sections=tuple(sections))


def load_personality_instructions(config: TwinrConfig) -> str | None:
    return load_personality_context(config).to_instructions()


def merge_instructions(*parts: str | None) -> str | None:
    merged = [part.strip() for part in parts if part and part.strip()]
    if not merged:
        return None
    return "\n\n".join(merged)
