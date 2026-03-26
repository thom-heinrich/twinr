"""Probe remote long-term snapshots before runtime uses them.

This module verifies that every required remote-primary snapshot and shard can
be loaded before Twinr enters runtime flows that depend on long-term memory.
It records per-snapshot pointer/origin evidence so fail-closed watchdogs can
prove exactly which remote read failed and how Twinr tried to resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.memory.context_store import PromptContextStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteFetchAttempt,
    LongTermRemoteSnapshotProbe,
    LongTermRemoteStateStore,
    LongTermRemoteUnavailableError,
)
from twinr.memory.longterm.storage.store import LongTermStructuredStore


@dataclass(frozen=True, slots=True)
class LongTermRemoteWarmCheck:
    """Capture one required remote snapshot check with pointer/origin evidence."""

    store: str
    snapshot_kind: str
    status: str
    latency_ms: float
    detail: str | None = None
    selected_source: str | None = None
    document_id: str | None = None
    pointer_document_id: str | None = None
    attempts: tuple[LongTermRemoteFetchAttempt, ...] = ()
    payload: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary without leaking full snapshot payloads."""

        return {
            "store": self.store,
            "snapshot_kind": self.snapshot_kind,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "selected_source": self.selected_source,
            "document_id": self.document_id,
            "pointer_document_id": self.pointer_document_id,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class LongTermRemoteWarmResult:
    """Capture the outcome of one runtime remote warm probe."""

    checked_snapshots: tuple[str, ...]
    ready: bool = True
    detail: str | None = None
    failed_store: str | None = None
    failed_snapshot_kind: str | None = None
    checks: tuple[LongTermRemoteWarmCheck, ...] = ()
    probe_mode: str = "archive_inclusive"
    archive_checked: bool = True
    archive_safe: bool = True
    health_tier: str = "ready"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary for ops artifacts."""

        return {
            "ready": self.ready,
            "detail": self.detail,
            "failed_store": self.failed_store,
            "failed_snapshot_kind": self.failed_snapshot_kind,
            "checked_snapshots": list(self.checked_snapshots),
            "checks": [check.to_dict() for check in self.checks],
            "probe_mode": self.probe_mode,
            "archive_checked": self.archive_checked,
            "archive_safe": self.archive_safe,
            "health_tier": self.health_tier,
        }


@dataclass(slots=True)
class LongTermRemoteHealthProbe:
    """Verify that required remote-primary long-term snapshots are readable.

    The runtime service uses this probe after store-level readiness checks to
    ensure every required snapshot kind and shard can actually be loaded.
    """

    prompt_context_store: PromptContextStore
    object_store: LongTermStructuredStore
    graph_store: TwinrPersonalGraphStore
    midterm_store: LongTermMidtermStore

    def probe_operational(self, *, include_archive: bool = True) -> LongTermRemoteWarmResult:
        """Return structured per-snapshot readiness evidence without raising.

        Args:
            include_archive: When false, skip the archival object snapshot
                during explicitly lighter current-state probes. External
                required-remote watchdogs must still treat archive-inclusive
                attestation as the only green/ready proof.
        """

        checked: list[str] = []
        checks: list[LongTermRemoteWarmCheck] = []
        try:
            prompt_remote_state = self._require_remote_state(self.prompt_context_store.memory_store.remote_state)
            result = self._probe_snapshot(
                store="prompt_context",
                remote_state=prompt_remote_state,
                snapshot_kind=self.prompt_context_store.memory_store.remote_snapshot_kind,
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result
            result = self._probe_snapshot(
                store="prompt_context",
                remote_state=prompt_remote_state,
                snapshot_kind=self.prompt_context_store.user_store.remote_snapshot_kind,
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result
            result = self._probe_snapshot(
                store="prompt_context",
                remote_state=prompt_remote_state,
                snapshot_kind=self.prompt_context_store.personality_store.remote_snapshot_kind,
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result

            object_remote_state = self._require_remote_state(self.object_store.remote_state)
            result = self._probe_snapshot(
                store="object_store",
                remote_state=object_remote_state,
                snapshot_kind="objects",
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result
            result = self._probe_snapshot(
                store="object_store",
                remote_state=object_remote_state,
                snapshot_kind="conflicts",
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result
            if include_archive:
                result = self._probe_snapshot(
                    store="object_store",
                    remote_state=object_remote_state,
                    snapshot_kind="archive",
                    checked=checked,
                    checks=checks,
                    include_archive=include_archive,
                )
                if not result.ready:
                    return result

            graph_remote_state = self._require_remote_state(self.graph_store.remote_state)
            result = self._probe_snapshot(
                store="graph_store",
                remote_state=graph_remote_state,
                snapshot_kind="graph",
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result

            midterm_remote_state = self._require_remote_state(self.midterm_store.remote_state)
            result = self._probe_snapshot(
                store="midterm_store",
                remote_state=midterm_remote_state,
                snapshot_kind="midterm",
                checked=checked,
                checks=checks,
                include_archive=include_archive,
            )
            if not result.ready:
                return result
        except LongTermRemoteUnavailableError as exc:
            return self._build_result(
                checked_snapshots=tuple(checked),
                ready=False,
                detail=str(exc),
                checks=tuple(checks),
                include_archive=include_archive,
            )

        return self._build_result(
            checked_snapshots=tuple(checked),
            ready=True,
            checks=tuple(checks),
            include_archive=include_archive,
        )

    def ensure_operational(self, *, include_archive: bool = True) -> LongTermRemoteWarmResult:
        """Load every required remote snapshot and shard once.

        Returns:
            A record of the snapshot kinds that were checked successfully.

        Raises:
            LongTermRemoteUnavailableError: If any required remote state,
                snapshot, or shard is missing or unreadable.
        """

        result = self.probe_operational(include_archive=include_archive)
        if result.ready:
            return result
        raise LongTermRemoteUnavailableError(
            str(result.detail or "Required remote long-term memory is unavailable.")
        )

    def _probe_snapshot(
        self,
        *,
        store: str,
        remote_state: LongTermRemoteStateStore,
        snapshot_kind: str | None,
        checked: list[str],
        checks: list[LongTermRemoteWarmCheck],
        include_archive: bool,
    ) -> LongTermRemoteWarmResult:
        """Probe one named snapshot and preserve pointer/origin evidence."""

        normalized_kind = str(snapshot_kind or "").strip()
        if not normalized_kind:
            return self._build_result(
                checked_snapshots=tuple(checked),
                ready=False,
                detail="Required remote long-term snapshot kind is missing.",
                failed_store=store,
                checks=tuple(checks),
                include_archive=include_archive,
            )
        probe_loader = getattr(remote_state, "probe_snapshot_load", None)
        if callable(probe_loader):
            # Ordinary warm health checks should reuse learned exact snapshot
            # document ids before re-walking current pointers. That keeps the
            # probe fail-closed while avoiding repeated multi-second origin
            # lookups for hot current snapshots such as objects/conflicts.
            probe = probe_loader(
                snapshot_kind=normalized_kind,
                prefer_cached_document_id=True,
            )
        else:
            payload = remote_state.load_snapshot(snapshot_kind=normalized_kind)
            probe = LongTermRemoteSnapshotProbe(
                snapshot_kind=normalized_kind,
                status="found" if isinstance(payload, dict) else "unavailable",
                latency_ms=0.0,
                detail=None if isinstance(payload, dict) else f"Required remote long-term snapshot {normalized_kind!r} is unavailable.",
                payload=payload if isinstance(payload, dict) else None,
            )
        check = self._warm_check_from_probe(store=store, probe=probe)
        checks.append(check)
        if not isinstance(probe.payload, dict):
            return self._build_result(
                checked_snapshots=tuple(checked),
                ready=False,
                detail=probe.detail or f"Required remote long-term snapshot {normalized_kind!r} is unavailable.",
                failed_store=store,
                failed_snapshot_kind=normalized_kind,
                checks=tuple(checks),
                include_archive=include_archive,
            )
        checked.append(normalized_kind)
        return self._build_result(
            checked_snapshots=tuple(checked),
            ready=True,
            checks=tuple(checks),
            include_archive=include_archive,
        )

    @staticmethod
    def _warm_check_from_probe(
        *,
        store: str,
        probe: LongTermRemoteSnapshotProbe,
    ) -> LongTermRemoteWarmCheck:
        return LongTermRemoteWarmCheck(
            store=store,
            snapshot_kind=probe.snapshot_kind,
            status=probe.status,
            latency_ms=probe.latency_ms,
            detail=probe.detail,
            selected_source=probe.selected_source,
            document_id=probe.document_id,
            pointer_document_id=probe.pointer_document_id,
            attempts=probe.attempts,
            payload=probe.payload,
        )

    @staticmethod
    def _build_result(
        *,
        checked_snapshots: tuple[str, ...],
        ready: bool,
        checks: tuple[LongTermRemoteWarmCheck, ...],
        include_archive: bool,
        detail: str | None = None,
        failed_store: str | None = None,
        failed_snapshot_kind: str | None = None,
    ) -> LongTermRemoteWarmResult:
        """Build one normalized warm-result payload with explicit attestation tier."""

        probe_mode = "archive_inclusive" if include_archive else "current_only"
        if not ready:
            health_tier = "hard_down"
        elif include_archive:
            health_tier = "ready"
        else:
            health_tier = "degraded"
        return LongTermRemoteWarmResult(
            checked_snapshots=checked_snapshots,
            ready=ready,
            detail=detail,
            failed_store=failed_store,
            failed_snapshot_kind=failed_snapshot_kind,
            checks=checks,
            probe_mode=probe_mode,
            archive_checked=include_archive,
            archive_safe=bool(ready and include_archive),
            health_tier=health_tier,
        )

    def _require_remote_state(self, remote_state: LongTermRemoteStateStore | None) -> LongTermRemoteStateStore:
        """Require an enabled remote state adapter before probing snapshots."""

        if remote_state is None or not remote_state.enabled:
            raise LongTermRemoteUnavailableError("Required remote long-term memory state is not configured.")
        return remote_state
