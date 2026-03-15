from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from twinr.memory.longterm.ontology import kind_matches
from twinr.memory.longterm.models import LongTermMemoryObjectV1, LongTermRetentionResultV1


def _parse_iso_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(f"{value}T00:00:00")
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class LongTermRetentionPolicy:
    timezone_name: str = "Europe/Berlin"
    mode: str = "conservative"
    archive_enabled: bool = True
    ephemeral_episode_days: int = 7
    ephemeral_observation_days: int = 2
    stale_status_archive_days: int = 3
    summary_archive_days: int = 30

    def apply(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermRetentionResultV1:
        reference = now or datetime.now(ZoneInfo(self.timezone_name))
        kept: list[LongTermMemoryObjectV1] = []
        expired: list[LongTermMemoryObjectV1] = []
        archived: list[LongTermMemoryObjectV1] = []
        pruned_ids: list[str] = []

        for item in objects:
            action = self._classify(item=item, now=reference)
            if action == "keep":
                kept.append(item)
            elif action == "expire":
                updated = item if item.status == "expired" else item.with_updates(status="expired")
                kept.append(updated)
                expired.append(updated)
            elif action == "archive":
                archived.append(
                    item.with_updates(
                        archived_at=reference.astimezone(timezone.utc).isoformat(),
                        updated_at=reference,
                    )
                )
            elif action == "prune":
                pruned_ids.append(item.memory_id)

        return LongTermRetentionResultV1(
            kept_objects=tuple(kept),
            expired_objects=tuple(expired),
            pruned_memory_ids=tuple(pruned_ids),
            archived_objects=tuple(archived),
        )

    def _classify(self, *, item: LongTermMemoryObjectV1, now: datetime) -> str:
        item = item.canonicalized()
        if item.kind == "episode":
            age = now - item.updated_at.astimezone(now.tzinfo or ZoneInfo(self.timezone_name))
            if age > timedelta(days=max(1, self.ephemeral_episode_days)):
                return "archive" if self.archive_enabled else "prune"
            return "keep"
        if kind_matches(item.kind, "observation", item.attributes):
            age = now - item.updated_at.astimezone(now.tzinfo or ZoneInfo(self.timezone_name))
            if age > timedelta(days=max(1, self.ephemeral_observation_days)):
                return "prune"
            return "keep"
        if item.status in {"superseded", "invalid", "expired"}:
            age = now - item.updated_at.astimezone(now.tzinfo or ZoneInfo(self.timezone_name))
            if self.archive_enabled and age > timedelta(days=max(1, self.stale_status_archive_days)):
                return "archive"
            return "keep"
        if item.kind == "summary":
            age = now - item.updated_at.astimezone(now.tzinfo or ZoneInfo(self.timezone_name))
            if self.archive_enabled and age > timedelta(days=max(1, self.summary_archive_days)):
                return "archive"
            return "keep"
        if item.valid_to:
            valid_to = _parse_iso_date(item.valid_to)
            if valid_to is not None and valid_to.date() < now.date() and item.status in {"active", "candidate", "uncertain"}:
                return "expire"
        return "keep"


__all__ = ["LongTermRetentionPolicy"]
