"""Force-repair prompt current heads on the remote ChonkyDB namespace.

This operator helper exists for one narrow failure mode: Twinr prompt-memory
and managed-context `.../catalog/current` heads can become unreadable blank
records on the remote backend. Once that happens, ordinary probe-first repair
flows may time out on the broken head itself before they can publish the known
canonical empty head.

The helper therefore builds a throwaway prompt-context store for one explicit
remote namespace and can publish the canonical empty current head directly for
`prompt_memory`, `user_context`, and `personality_context` without first
reading the previous head. It is intentionally operator-facing and explicit:
runtime behavior stays fail-closed, while repair work gets one bounded,
auditable escape hatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import argparse
import json
import tempfile
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore, PromptContextStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


_SUPPORTED_SNAPSHOT_KINDS = ("prompt_memory", "user_context", "personality_context")


@dataclass(frozen=True, slots=True)
class PromptCurrentHeadRepairItem:
    """Describe one per-snapshot repair attempt."""

    snapshot_kind: str
    forced: bool
    before_status: str
    after_status: str
    action: str
    ok: bool
    elapsed_s: float
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the repair item."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptCurrentHeadRepairResult:
    """Describe one complete remote prompt current-head repair run."""

    ok: bool
    namespace: str
    base_url: str
    snapshot_kinds: tuple[str, ...]
    items: tuple[PromptCurrentHeadRepairItem, ...]
    elapsed_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the repair result."""

        return {
            "ok": self.ok,
            "namespace": self.namespace,
            "base_url": self.base_url,
            "snapshot_kinds": list(self.snapshot_kinds),
            "items": [item.to_dict() for item in self.items],
            "elapsed_s": self.elapsed_s,
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for remote prompt current-head repair."""

    parser = argparse.ArgumentParser(
        description=(
            "Force-repair prompt-memory/user-context/personality-context "
            "catalog/current heads on one remote ChonkyDB namespace."
        )
    )
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--env-file",
        type=Path,
        default=project_root / ".env",
        help="Twinr env file that defines the ChonkyDB auth contract.",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Explicit remote namespace to repair, for example twinr_longterm_v1:twinr:a7f1ed265838.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Override the ChonkyDB base URL. Default: use the URL from --env-file.",
    )
    parser.add_argument(
        "--snapshot-kind",
        action="append",
        choices=_SUPPORTED_SNAPSHOT_KINDS,
        dest="snapshot_kinds",
        help="Repeat to choose specific prompt collections. Default: user_context + personality_context.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Publish the canonical empty current head directly without probing the old head first.",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help="When not using --force, also seed missing heads as canonical empty heads.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-write probe. Use only when the backend is too saturated for immediate verification.",
    )
    parser.add_argument(
        "--read-timeout-s",
        type=float,
        default=25.0,
        help="Remote read timeout in seconds for optional probe/verify steps.",
    )
    parser.add_argument(
        "--write-timeout-s",
        type=float,
        default=60.0,
        help="Remote write timeout in seconds for the direct current-head publish.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=0,
        help="Remote retry attempts for this bounded operator run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bounded remote prompt current-head repair helper."""

    args = build_parser().parse_args(argv)
    base_config = TwinrConfig.from_env(args.env_file)
    snapshot_kinds = tuple(args.snapshot_kinds or ("user_context", "personality_context"))
    result = repair_remote_prompt_current_heads(
        base_config=base_config,
        namespace=args.namespace,
        snapshot_kinds=snapshot_kinds,
        base_url=args.base_url or None,
        force=args.force,
        repair_missing=args.repair_missing,
        verify=not args.no_verify,
        read_timeout_s=args.read_timeout_s,
        write_timeout_s=args.write_timeout_s,
        retry_attempts=args.retry_attempts,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.ok else 1


def repair_remote_prompt_current_heads(
    *,
    base_config: TwinrConfig,
    namespace: str,
    snapshot_kinds: Sequence[str],
    base_url: str | None = None,
    force: bool = False,
    repair_missing: bool = False,
    verify: bool = True,
    read_timeout_s: float = 25.0,
    write_timeout_s: float = 60.0,
    retry_attempts: int = 0,
) -> PromptCurrentHeadRepairResult:
    """Repair one remote namespace by publishing canonical empty prompt heads."""

    started = time.monotonic()
    normalized_namespace = " ".join(str(namespace or "").split()).strip()
    if not normalized_namespace:
        raise ValueError("namespace must not be empty")
    normalized_snapshot_kinds = tuple(_normalize_snapshot_kinds(snapshot_kinds))
    effective_base_url = " ".join(str(base_url or base_config.chonkydb_base_url or "").split()).strip()
    if not effective_base_url:
        raise ValueError("base_url must not be empty")
    with ExitStack() as stack:
        temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="twinr_prompt_head_repair_")))
        config = _build_repair_config(
            base_config=base_config,
            runtime_root=temp_dir,
            namespace=normalized_namespace,
            base_url=effective_base_url,
            read_timeout_s=read_timeout_s,
            write_timeout_s=write_timeout_s,
            retry_attempts=retry_attempts,
        )
        prompt_store = _build_prompt_context_store(config=config, runtime_root=temp_dir)
        items = repair_prompt_current_heads_from_store(
            prompt_store=prompt_store,
            snapshot_kinds=normalized_snapshot_kinds,
            force=force,
            repair_missing=repair_missing,
            verify=verify,
        )
    elapsed_s = time.monotonic() - started
    return PromptCurrentHeadRepairResult(
        ok=all(item.ok for item in items),
        namespace=normalized_namespace,
        base_url=effective_base_url,
        snapshot_kinds=normalized_snapshot_kinds,
        items=items,
        elapsed_s=elapsed_s,
    )


def repair_prompt_current_heads_from_store(
    *,
    prompt_store: PromptContextStore,
    snapshot_kinds: Sequence[str],
    force: bool = False,
    repair_missing: bool = False,
    verify: bool = True,
) -> tuple[PromptCurrentHeadRepairItem, ...]:
    """Repair prompt heads using one already-built prompt-context store."""

    items: list[PromptCurrentHeadRepairItem] = []
    for snapshot_kind in _normalize_snapshot_kinds(snapshot_kinds):
        component = _component_for_snapshot_kind(prompt_store=prompt_store, snapshot_kind=snapshot_kind)
        remote_records = component._remote_records  # Operator-only repair path; keep runtime API unchanged.
        started = time.monotonic()
        before_status = "skipped_force" if force else "unknown"
        after_status = "not_checked"
        action = "none"
        ok = False
        detail: str | None = None
        try:
            if not force:
                before_status, _before_payload = remote_records.probe_current_head_result(snapshot_kind=snapshot_kind)
            if force:
                action = "force_publish_empty"
            elif before_status == "invalid":
                action = "repair_invalid"
            elif before_status == "missing" and repair_missing:
                action = "repair_missing"
            elif before_status == "found":
                action = "already_healthy"
            elif before_status == "missing":
                action = "missing_unrepaired"
                detail = "current head is missing; rerun with --repair-missing or --force to seed an empty head"
            else:
                action = "probe_unhealthy"
                detail = f"current head probe returned {before_status!r}"
            if action in {"force_publish_empty", "repair_invalid", "repair_missing"}:
                remote_records.save_empty_collection_head(
                    snapshot_kind=snapshot_kind,
                    attest_readback=False,
                )
            if verify:
                after_status, _after_payload = remote_records.probe_current_head_result(snapshot_kind=snapshot_kind)
            else:
                after_status = "not_checked"
            ok = action == "already_healthy" or action in {"force_publish_empty", "repair_invalid", "repair_missing"}
            if verify:
                ok = ok and after_status == "found"
                if ok and action != "already_healthy":
                    detail = "empty current head published and verified"
                elif not ok and action in {"force_publish_empty", "repair_invalid", "repair_missing"}:
                    detail = f"post-write probe returned {after_status!r}"
                elif ok and action == "already_healthy":
                    detail = "current head already healthy"
            elif ok and action != "already_healthy":
                detail = "empty current head published without post-write verification"
            elif ok:
                detail = "current head already healthy"
        except Exception as exc:  # pragma: no cover - covered by the result contract in tests via injected failures.
            ok = False
            if action == "none":
                action = "error"
            detail = f"{type(exc).__name__}: {exc}"
        items.append(
            PromptCurrentHeadRepairItem(
                snapshot_kind=snapshot_kind,
                forced=bool(force),
                before_status=before_status,
                after_status=after_status,
                action=action,
                ok=ok,
                elapsed_s=time.monotonic() - started,
                detail=detail,
            )
        )
    return tuple(items)


def _normalize_snapshot_kinds(snapshot_kinds: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_kind in snapshot_kinds:
        snapshot_kind = " ".join(str(raw_kind or "").split()).strip()
        if snapshot_kind not in _SUPPORTED_SNAPSHOT_KINDS:
            raise ValueError(f"Unsupported prompt current-head snapshot kind: {snapshot_kind!r}")
        if snapshot_kind in seen:
            continue
        seen.add(snapshot_kind)
        normalized.append(snapshot_kind)
    if not normalized:
        raise ValueError("At least one snapshot kind is required.")
    return tuple(normalized)


def _build_repair_config(
    *,
    base_config: TwinrConfig,
    runtime_root: Path,
    namespace: str,
    base_url: str,
    read_timeout_s: float,
    write_timeout_s: float,
    retry_attempts: int,
) -> TwinrConfig:
    """Return one throwaway remote-primary config for operator repair work."""

    state_dir = runtime_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        base_config,
        project_root=str(runtime_root),
        personality_dir=str(runtime_root / "personality"),
        memory_markdown_path=str(state_dir / "MEMORY.md"),
        runtime_state_path=str(runtime_root / "runtime-state.json"),
        long_term_memory_enabled=True,
        long_term_memory_mode="remote_primary",
        long_term_memory_remote_required=True,
        long_term_memory_path=str(state_dir / "chonkydb"),
        long_term_memory_remote_namespace=namespace,
        chonkydb_base_url=base_url,
        long_term_memory_remote_read_timeout_s=max(1.0, float(read_timeout_s)),
        long_term_memory_remote_write_timeout_s=max(1.0, float(write_timeout_s)),
        long_term_memory_remote_retry_attempts=max(0, int(retry_attempts)),
    )


def _build_prompt_context_store(*, config: TwinrConfig, runtime_root: Path) -> PromptContextStore:
    """Build a prompt-context store rooted entirely in one throwaway directory."""

    personality_dir = runtime_root / "personality"
    personality_dir.mkdir(parents=True, exist_ok=True)
    state_dir = runtime_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    remote_state = LongTermRemoteStateStore.from_config(config)
    return PromptContextStore(
        memory_store=PersistentMemoryMarkdownStore(
            state_dir / "MEMORY.md",
            remote_state=remote_state,
            allow_legacy_head_reads=False,
            root_dir=runtime_root,
        ),
        user_store=ManagedContextFileStore(
            personality_dir / "USER.md",
            section_title="Twinr managed user updates",
            remote_state=remote_state,
            remote_snapshot_kind="user_context",
            allow_legacy_head_reads=False,
            root_dir=runtime_root,
        ),
        personality_store=ManagedContextFileStore(
            personality_dir / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
            remote_state=remote_state,
            remote_snapshot_kind="personality_context",
            allow_legacy_head_reads=False,
            root_dir=runtime_root,
        ),
    )


def _component_for_snapshot_kind(
    *,
    prompt_store: PromptContextStore,
    snapshot_kind: str,
) -> ManagedContextFileStore | PersistentMemoryMarkdownStore:
    """Return the prompt-store component that owns one supported snapshot kind."""

    if snapshot_kind == "prompt_memory":
        return prompt_store.memory_store
    if snapshot_kind == "user_context":
        return prompt_store.user_store
    if snapshot_kind == "personality_context":
        return prompt_store.personality_store
    raise ValueError(f"Unsupported prompt current-head snapshot kind: {snapshot_kind!r}")


__all__ = [
    "PromptCurrentHeadRepairItem",
    "PromptCurrentHeadRepairResult",
    "build_parser",
    "main",
    "repair_prompt_current_heads_from_store",
    "repair_remote_prompt_current_heads",
]
