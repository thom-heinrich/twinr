from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re

from twinr.agent.base_agent.config import TwinrConfig

_MANAGED_START = "<!-- TWINR_MANAGED_CONTEXT_START -->"
_MANAGED_END = "<!-- TWINR_MANAGED_CONTEXT_END -->"
_MEMORY_ENTRY_HEADING_RE = re.compile(r"^###\s+(?P<entry_id>[A-Z0-9_-]+)\s*$")
_MEMORY_FIELD_RE = re.compile(r"^-\s+(?P<key>[a-z_]+):\s*(?P<value>.*)\s*$")
_MANAGED_ENTRY_RE = re.compile(r"^-\s+(?P<key>[a-z0-9_]+):\s*(?P<value>.+?)\s*$", re.IGNORECASE)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or fallback


def _parse_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return _utcnow()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return _utcnow()


@dataclass(frozen=True, slots=True)
class ManagedContextEntry:
    key: str
    instruction: str
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class PersistentMemoryEntry:
    entry_id: str
    kind: str
    summary: str
    details: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


class ManagedContextFileStore:
    def __init__(self, path: str | Path, *, section_title: str) -> None:
        self.path = Path(path)
        self.section_title = section_title

    def load_base_text(self) -> str:
        prefix, _managed_entries, _suffix = self._split_document()
        return prefix.strip()

    def load_entries(self) -> tuple[ManagedContextEntry, ...]:
        _prefix, managed_entries, _suffix = self._split_document()
        return managed_entries

    def upsert(self, *, category: str, instruction: str) -> ManagedContextEntry:
        key = _slugify(category, fallback="update")
        clean_instruction = _normalize_text(instruction, limit=220)
        if not clean_instruction:
            raise ValueError("instruction must not be empty")

        entries = list(self.load_entries())
        updated = ManagedContextEntry(key=key, instruction=clean_instruction, updated_at=_utcnow())
        for index, existing in enumerate(entries):
            if existing.key != key:
                continue
            entries[index] = updated
            self._write_entries(tuple(entries))
            return updated
        entries.append(updated)
        self._write_entries(tuple(entries))
        return updated

    def replace_base_text(self, content: str) -> None:
        _prefix, managed_entries, suffix = self._split_document()
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        self._write_document(prefix=normalized, entries=managed_entries, suffix=suffix)

    def _split_document(self) -> tuple[str, tuple[ManagedContextEntry, ...], str]:
        if not self.path.exists():
            return "", (), ""
        text = self.path.read_text(encoding="utf-8")
        if _MANAGED_START not in text or _MANAGED_END not in text:
            return text.rstrip(), (), ""

        prefix, remainder = text.split(_MANAGED_START, 1)
        managed_block, suffix = remainder.split(_MANAGED_END, 1)
        entries: list[ManagedContextEntry] = []
        for raw_line in managed_block.splitlines():
            line = raw_line.strip()
            match = _MANAGED_ENTRY_RE.match(line)
            if match is None:
                continue
            entries.append(
                ManagedContextEntry(
                    key=_slugify(match.group("key"), fallback="update"),
                    instruction=match.group("value").strip(),
                )
            )
        return prefix.rstrip(), tuple(entries), suffix.lstrip("\n")

    def _write_entries(self, entries: tuple[ManagedContextEntry, ...]) -> None:
        prefix, _existing_entries, suffix = self._split_document()
        self._write_document(prefix=prefix, entries=entries, suffix=suffix)

    def _write_document(
        self,
        *,
        prefix: str,
        entries: tuple[ManagedContextEntry, ...],
        suffix: str,
    ) -> None:
        body_parts: list[str] = []
        if prefix:
            body_parts.append(prefix.rstrip())
        if entries:
            managed_lines = [
                _MANAGED_START,
                f"## {self.section_title}",
                "_This section is managed by Twinr. Keep entries short and stable._",
            ]
            for entry in entries:
                managed_lines.append(f"- {entry.key}: {entry.instruction}")
            managed_lines.append(_MANAGED_END)
            body_parts.append("\n".join(managed_lines))
        if suffix:
            body_parts.append(suffix.rstrip())
        rendered = "\n\n".join(part for part in body_parts if part).rstrip() + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(rendered, encoding="utf-8")


class PersistentMemoryMarkdownStore:
    def __init__(self, path: str | Path, *, max_entries: int = 24) -> None:
        self.path = Path(path)
        self.max_entries = max_entries

    def load_entries(self) -> tuple[PersistentMemoryEntry, ...]:
        if not self.path.exists():
            return ()
        entries: list[PersistentMemoryEntry] = []
        current: dict[str, str] | None = None
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            heading = _MEMORY_ENTRY_HEADING_RE.match(line)
            if heading is not None:
                if current is not None:
                    entry = self._entry_from_mapping(current)
                    if entry is not None:
                        entries.append(entry)
                current = {"entry_id": heading.group("entry_id")}
                continue
            if current is None:
                continue
            field_match = _MEMORY_FIELD_RE.match(line)
            if field_match is None:
                continue
            current[field_match.group("key")] = field_match.group("value").strip()
        if current is not None:
            entry = self._entry_from_mapping(current)
            if entry is not None:
                entries.append(entry)
        return tuple(entries)

    def remember(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        clean_kind = _slugify(kind, fallback="memory")
        clean_summary = _normalize_text(summary, limit=220)
        clean_details = _normalize_text(details or "", limit=420) or None
        if not clean_summary:
            raise ValueError("summary must not be empty")

        entries = list(self.load_entries())
        now = _utcnow()
        normalized_key = (clean_kind, clean_summary.lower())
        for index, existing in enumerate(entries):
            if (existing.kind, existing.summary.lower()) != normalized_key:
                continue
            updated = PersistentMemoryEntry(
                entry_id=existing.entry_id,
                kind=clean_kind,
                summary=clean_summary,
                details=clean_details or existing.details,
                created_at=existing.created_at,
                updated_at=now,
            )
            entries[index] = updated
            self._write_entries(tuple(entries))
            return updated

        entry = PersistentMemoryEntry(
            entry_id=f"MEM-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
            kind=clean_kind,
            summary=clean_summary,
            details=clean_details,
            created_at=now,
            updated_at=now,
        )
        entries.insert(0, entry)
        if len(entries) > self.max_entries:
            entries = entries[: self.max_entries]
        self._write_entries(tuple(entries))
        return entry

    def render_context(self, *, limit: int = 12) -> str | None:
        entries = self.load_entries()
        if not entries:
            return None
        lines = ["Durable remembered items explicitly saved for future turns:"]
        for entry in entries[:limit]:
            line = f"- [{entry.kind}] {entry.summary}"
            if entry.details and entry.details.lower() != entry.summary.lower():
                line += f" Details: {entry.details}"
            lines.append(line)
        return "\n".join(lines).strip()

    def _write_entries(self, entries: tuple[PersistentMemoryEntry, ...]) -> None:
        lines = [
            "# Twinr Memory",
            "",
            "This file is managed by Twinr.",
            "It stores durable memories only when the user explicitly asks Twinr to remember something for later.",
            "",
            "## Entries",
        ]
        if not entries:
            lines.extend(["", "_No saved memories yet._"])
        else:
            for entry in entries:
                lines.extend(
                    [
                        "",
                        f"### {entry.entry_id}",
                        f"- kind: {entry.kind}",
                        f"- created_at: {entry.created_at.isoformat()}",
                        f"- updated_at: {entry.updated_at.isoformat()}",
                        f"- summary: {entry.summary}",
                    ]
                )
                if entry.details:
                    lines.append(f"- details: {entry.details}")
        rendered = "\n".join(lines).rstrip() + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(rendered, encoding="utf-8")

    def _entry_from_mapping(self, data: dict[str, str]) -> PersistentMemoryEntry | None:
        summary = _normalize_text(data.get("summary", ""), limit=220)
        if not summary:
            return None
        return PersistentMemoryEntry(
            entry_id=data.get("entry_id", "").strip() or f"MEM-{_utcnow().strftime('%Y%m%dT%H%M%S%fZ')}",
            kind=_slugify(data.get("kind", "memory"), fallback="memory"),
            summary=summary,
            details=_normalize_text(data.get("details", ""), limit=420) or None,
            created_at=_parse_datetime(data.get("created_at", "")),
            updated_at=_parse_datetime(data.get("updated_at", data.get("created_at", ""))),
        )


@dataclass(frozen=True, slots=True)
class PromptContextStore:
    memory_store: PersistentMemoryMarkdownStore
    user_store: ManagedContextFileStore
    personality_store: ManagedContextFileStore

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PromptContextStore":
        personality_dir = Path(config.project_root) / config.personality_dir
        return cls(
            memory_store=PersistentMemoryMarkdownStore(config.memory_markdown_path),
            user_store=ManagedContextFileStore(
                personality_dir / "USER.md",
                section_title="Twinr managed user updates",
            ),
            personality_store=ManagedContextFileStore(
                personality_dir / "PERSONALITY.md",
                section_title="Twinr managed personality updates",
            ),
        )
