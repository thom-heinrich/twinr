"""Search and selector helpers for remote catalog storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import time
from typing import Any

from twinr.agent.workflows.forensics import workflow_decision, workflow_event, workflow_span
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.chonkydb.models import ChonkyDBRetrieveRequest, ChonkyDBTopKRecordsRequest
from twinr.memory.longterm.storage._remote_retry import (
    remote_read_retry_delay_s,
    should_retry_remote_read_error,
)
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    build_remote_read_failure_details,
    record_remote_read_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation
from twinr.memory.longterm.storage.remote_state import LongTermRemoteReadFailedError

from ._typing import RemoteCatalogMixinBase
from .shared import (
    LongTermRemoteCatalogEntry,
    _run_timed_workflow_step,
    _trace_search_details,
)

_FAST_TOPIC_TOPK_RETRY_ATTEMPTS_CAP = 2


def _emit_query_plan_workflow_event(
    *,
    snapshot_kind: str,
    operation: str,
    response: object,
    query_text: str,
    result_limit: int,
    allowed_doc_count: int,
    scope_ref: str | None,
    namespace: str | None,
    catalog_entry_count: int,
) -> None:
    """Surface server-reported query plans as retrieval trace events."""

    query_plan = getattr(response, "query_plan", None)
    if not isinstance(query_plan, Mapping):
        return
    workflow_event(
        kind="retrieval",
        msg="longterm_remote_catalog_query_plan",
        details={
            **_trace_search_details(
                snapshot_kind=snapshot_kind,
                query_text=query_text,
                result_limit=result_limit,
                allowed_doc_count=allowed_doc_count,
                scope_ref=scope_ref,
                namespace=namespace,
                catalog_entry_count=catalog_entry_count,
            ),
            "operation": operation,
            "query_plan": dict(query_plan),
        },
    )


class RemoteCatalogSearchMixin(RemoteCatalogMixinBase):
    def _fast_topic_topk_retry_attempts(self) -> tuple[int, int]:
        """Return the effective and configured retry budget for fast-topic reads.

        Fast-topic current-scope reads are latency-sensitive, so they should
        consume the same transient-error policy as other required remote reads
        but stop after one reattempt of the authoritative scope endpoint.
        """

        configured_attempts = self._remote_retry_attempts()
        return (
            min(_FAST_TOPIC_TOPK_RETRY_ATTEMPTS_CAP, configured_attempts),
            configured_attempts,
        )

    def search_current_item_payloads(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
        allow_catalog_fallback: bool = False,
    ) -> tuple[dict[str, object], ...] | None:
        """Search the current remote scope directly without hydrating the full catalog."""

        clean_query = self._normalize_text(query_text)
        if not clean_query:
            return ()
        cached_entries = self._cached_catalog_entries(snapshot_kind=snapshot_kind)
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        scope_ref, namespace = self._current_scope_request_context(snapshot_kind=snapshot_kind)
        can_scope_search = bool(
            supports_topk_records
            and callable(topk_records)
            and scope_ref
            and namespace
            and self._scope_search_supported(snapshot_kind=snapshot_kind)
        )
        workflow_decision(
            msg="longterm_remote_catalog_current_scope_strategy",
            question="Which remote catalog route should resolve the current snapshot search?",
            selected={
                "id": (
                    "scope_topk_records"
                    if can_scope_search
                    else "cached_catalog_selector"
                    if cached_entries is not None
                    else "return_none_for_caller_fallback"
                ),
                "summary": (
                    "Use current-scope one-shot top-k retrieval."
                    if can_scope_search
                    else "Use the cached catalog selector and hydrate only selected payloads."
                    if cached_entries is not None
                    else "Return None so the caller can choose a broader fallback path."
                ),
            },
            options=[
                {
                    "id": "scope_topk_records",
                    "summary": "Query the authoritative current-scope top-k endpoint.",
                    "score_components": {
                        "supports_topk_records": bool(supports_topk_records and callable(topk_records)),
                        "has_scope_ref": bool(scope_ref),
                        "has_namespace": bool(namespace),
                    },
                    "constraints_violated": [] if can_scope_search else ["scope_ref_or_topk_unavailable"],
                },
                {
                    "id": "cached_catalog_selector",
                    "summary": "Search the cached catalog projection locally and hydrate only selected payloads.",
                    "score_components": {
                        "cached_catalog_entries": 0 if cached_entries is None else len(cached_entries),
                    },
                    "constraints_violated": [] if cached_entries is not None else ["cached_catalog_missing"],
                },
                {
                    "id": "return_none_for_caller_fallback",
                    "summary": "Signal the caller to try a broader catalog-backed fallback path.",
                    "score_components": {
                        "cached_catalog_entries": 0 if cached_entries is None else len(cached_entries),
                    },
                    "constraints_violated": [] if cached_entries is None and not can_scope_search else ["better_route_available"],
                },
            ],
            context=_trace_search_details(
                snapshot_kind=snapshot_kind,
                query_text=clean_query,
                result_limit=limit,
                scope_ref=scope_ref,
                namespace=namespace,
                cached_catalog_entries=0 if cached_entries is None else len(cached_entries),
            ),
            confidence="high",
            guardrails=[
                "Prefer the authoritative current-scope top-k path when the backend supports it.",
                "Use cached catalog selection only as a bounded fallback, not the default hot path.",
            ],
            kpi_impact_estimate={
                "allow_catalog_fallback": bool(allow_catalog_fallback),
            },
        )
        if can_scope_search:
            definition = self._require_definition(snapshot_kind)
            bounded_limit = max(1, int(limit))
            max_request_limit = max(bounded_limit, self._retrieve_batch_size())
            scope_search_failed = False
            request_limit = self._initial_scope_search_limit(
                bounded_limit=bounded_limit,
                max_request_limit=max_request_limit,
                requires_client_filter=eligible is not None,
            )
            selected_payloads: list[dict[str, object]] = []
            selected_entries: list[LongTermRemoteCatalogEntry] = []
            seen_item_ids: set[str] = set()
            while True:
                try:
                    def run_scope_search() -> tuple[Mapping[str, object], ...]:
                        return self._search_remote_candidates(
                            snapshot_kind=snapshot_kind,
                            read_client=read_client,
                            query_text=clean_query,
                            result_limit=request_limit,
                            allowed_doc_ids=(),
                            scope_ref=scope_ref,
                            namespace=namespace,
                            catalog_entry_count=0,
                        )

                    candidates = _run_timed_workflow_step(
                        name="longterm_remote_catalog_scope_search_attempt",
                        kind="retrieval",
                        details=_trace_search_details(
                            snapshot_kind=snapshot_kind,
                            query_text=clean_query,
                            result_limit=bounded_limit,
                            candidate_limit=request_limit,
                            scope_ref=scope_ref,
                            namespace=namespace,
                        ),
                        operation=run_scope_search,
                    )
                except ChonkyDBError:
                    scope_search_failed = True
                    workflow_event(
                        kind="branch",
                        msg="longterm_remote_catalog_scope_search_failed",
                        level="WARNING",
                        details=_trace_search_details(
                            snapshot_kind=snapshot_kind,
                            query_text=clean_query,
                            result_limit=bounded_limit,
                            candidate_limit=request_limit,
                            scope_ref=scope_ref,
                            namespace=namespace,
                        ),
                        reason={
                            "selected": {"id": "cached_catalog_fallback", "summary": "Scope search failed; use the already cached catalog projection if present."},
                            "options": [
                                {"id": "cached_catalog_fallback", "summary": "Try the cached catalog projection.", "constraints_violated": []},
                                {"id": "abort_remote_lookup", "summary": "Return no payloads.", "constraints_violated": ["would_hide_recoverable_memory"]},
                            ],
                        },
                    )
                    break
                candidate_count = 0
                for candidate in candidates:
                    candidate_count += 1
                    entry = self._candidate_catalog_entry(definition=definition, payload=candidate)
                    if entry is None or entry.item_id in seen_item_ids:
                        continue
                    if eligible is not None and not eligible(entry):
                        continue
                    payload = self._extract_item_payload(
                        definition=definition,
                        item_id=entry.item_id,
                        payload=candidate,
                    )
                    if not isinstance(payload, Mapping):
                        continue
                    payload_dict = dict(payload)
                    self._store_item_payload(
                        snapshot_kind=snapshot_kind,
                        item_id=entry.item_id,
                        payload=payload_dict,
                    )
                    selected_payloads.append(payload_dict)
                    selected_entries.append(entry)
                    seen_item_ids.add(entry.item_id)
                    if len(selected_payloads) >= bounded_limit:
                        break
                if len(selected_payloads) >= bounded_limit:
                    break
                if candidate_count < request_limit or request_limit >= max_request_limit:
                    break
                next_limit = min(max_request_limit, request_limit * 2)
                if next_limit <= request_limit:
                    break
                workflow_event(
                    kind="branch",
                    msg="longterm_remote_catalog_scope_search_widen",
                    details=_trace_search_details(
                        snapshot_kind=snapshot_kind,
                        query_text=clean_query,
                        result_limit=bounded_limit,
                        candidate_limit=request_limit,
                        scope_ref=scope_ref,
                        namespace=namespace,
                    ),
                    reason={
                        "selected": {"id": "widen_request_limit", "summary": "The current scope-search window was too narrow; widen the candidate request."},
                        "options": [
                            {"id": "widen_request_limit", "summary": "Increase the request window.", "constraints_violated": []},
                            {"id": "stop_early", "summary": "Keep the smaller window.", "constraints_violated": ["risk_underfilled_results"]},
                        ],
                    },
                    kpi={
                        "previous_request_limit": request_limit,
                        "next_request_limit": next_limit,
                    },
                )
                request_limit = next_limit
            if not scope_search_failed:
                should_reconcile_scope_hits = bool(selected_payloads) or allow_catalog_fallback
                if should_reconcile_scope_hits:
                    with workflow_span(
                        name="longterm_remote_catalog_scope_rescue",
                        kind="retrieval",
                        details=_trace_search_details(
                            snapshot_kind=snapshot_kind,
                            query_text=clean_query,
                            result_limit=bounded_limit,
                            candidate_limit=request_limit,
                            scope_ref=scope_ref,
                            namespace=namespace,
                        ),
                    ):
                        rescued = self._rescue_scope_search_payloads_with_current_catalog(
                            snapshot_kind=snapshot_kind,
                            query_text=clean_query,
                            limit=bounded_limit,
                            eligible=eligible,
                            selected_entries=tuple(selected_entries),
                            selected_payloads=tuple(selected_payloads),
                            cached_only=not allow_catalog_fallback,
                        )
                    if rescued is not None:
                        return rescued
                return tuple(selected_payloads)

        if cached_entries is None:
            return None
        with workflow_span(
            name="longterm_remote_catalog_cached_selector_search",
            kind="retrieval",
            details=_trace_search_details(
                snapshot_kind=snapshot_kind,
                query_text=clean_query,
                result_limit=limit,
                cached_catalog_entries=len(cached_entries),
            ),
        ):
            cached_selected_entries = self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=cached_entries,
                query_text=clean_query,
                limit=limit,
                eligible=eligible,
            )
        if not cached_selected_entries:
            return ()
        with workflow_span(
            name="longterm_remote_catalog_cached_selector_hydrate",
            kind="retrieval",
            details=_trace_search_details(
                snapshot_kind=snapshot_kind,
                query_text=clean_query,
                result_limit=limit,
                cached_catalog_entries=len(cached_entries),
                candidate_limit=len(cached_selected_entries),
            ),
        ):
            return self.load_selection_item_payloads(
                snapshot_kind=snapshot_kind,
                item_ids=(entry.item_id for entry in cached_selected_entries),
            )

    def _rescue_fast_scope_search_with_current_catalog(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        failure_details: Mapping[str, object],
        allow_cold_catalog_load: bool = False,
    ) -> tuple[dict[str, object], ...] | None:
        """Recover fast scope reads from the current remote catalog when possible."""

        current_entries = self._cached_catalog_entries(snapshot_kind=snapshot_kind)
        if current_entries is None:
            current_entries = self._recent_catalog_entries(snapshot_kind=snapshot_kind)
        if current_entries is None:
            if allow_cold_catalog_load:
                try:
                    current_entries = self.load_catalog_entries(
                        snapshot_kind=snapshot_kind,
                        bypass_cache=True,
                    )
                except Exception as exc:
                    workflow_event(
                        kind="branch",
                        msg="longterm_remote_catalog_fast_scope_search_rescue_skipped",
                        details={
                            "snapshot_kind": snapshot_kind,
                            "result_limit": max(1, int(limit)),
                            "failure_classification": failure_details.get("classification"),
                            "failure_status_code": failure_details.get("status_code"),
                            "failure_timeout_reason": failure_details.get("timeout_reason"),
                            "reason": "current_catalog_cold_load_failed",
                            "cold_load_error_type": type(exc).__name__,
                        },
                    )
                    return None
            else:
                workflow_event(
                    kind="branch",
                    msg="longterm_remote_catalog_fast_scope_search_rescue_skipped",
                    details={
                        "snapshot_kind": snapshot_kind,
                        "result_limit": max(1, int(limit)),
                        "failure_classification": failure_details.get("classification"),
                        "failure_status_code": failure_details.get("status_code"),
                        "failure_timeout_reason": failure_details.get("timeout_reason"),
                        "reason": "current_catalog_projection_not_cached_in_process",
                    },
                )
                return None
        if current_entries is None:
            workflow_event(
                kind="branch",
                msg="longterm_remote_catalog_fast_scope_search_rescue_skipped",
                details={
                    "snapshot_kind": snapshot_kind,
                    "result_limit": max(1, int(limit)),
                    "failure_classification": failure_details.get("classification"),
                    "failure_status_code": failure_details.get("status_code"),
                    "failure_timeout_reason": failure_details.get("timeout_reason"),
                    "reason": "current_catalog_projection_unavailable",
                },
            )
            return None
        with workflow_span(
            name="longterm_remote_catalog_fast_scope_cached_selector_search",
            kind="retrieval",
            details={
                **_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    query_text=query_text,
                    result_limit=limit,
                    cached_catalog_entries=len(current_entries),
                ),
                "rescue_trigger": str(failure_details.get("classification") or "unknown"),
            },
        ):
            selected_entries = self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=current_entries,
                query_text=query_text,
                limit=limit,
            )
        if not selected_entries:
            return ()
        with workflow_span(
            name="longterm_remote_catalog_fast_scope_cached_selector_hydrate",
            kind="retrieval",
            details={
                **_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    query_text=query_text,
                    result_limit=limit,
                    cached_catalog_entries=len(current_entries),
                    candidate_limit=len(selected_entries),
                ),
                "rescue_trigger": str(failure_details.get("classification") or "unknown"),
            },
        ):
            payloads = self.load_selection_item_payloads(
                snapshot_kind=snapshot_kind,
                item_ids=(entry.item_id for entry in selected_entries),
            )
        workflow_event(
            kind="branch",
            msg="longterm_remote_catalog_fast_scope_search_rescued",
            details={
                "snapshot_kind": snapshot_kind,
                "result_limit": max(1, int(limit)),
                "cached_catalog_entries": len(current_entries),
                "rescued_payload_count": len(payloads),
                "failure_classification": failure_details.get("classification"),
                "failure_status_code": failure_details.get("status_code"),
                "failure_timeout_reason": failure_details.get("timeout_reason"),
            },
        )
        return payloads

    def search_current_item_payloads_fast(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        timeout_s: float | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Run one current-scope top-k read without widening or catalog rescue."""

        clean_query = self._normalize_text(query_text)
        if not clean_query:
            return ()
        remote_state = self._require_remote_state()
        base_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        read_client = self._client_with_timeout(base_client, timeout_s=timeout_s)
        scope_ref, namespace = self._current_scope_request_context(snapshot_kind=snapshot_kind)
        if not scope_ref or not namespace:
            raise ChonkyDBError("Current-scope topk_records is required for fast current-scope retrieval.")
        if not self._scope_search_supported(snapshot_kind=snapshot_kind):
            failure_details = {
                "classification": "client_contract_error",
                "status_code": None,
                "scope_ref": scope_ref,
                "reason": "scope_search_suppressed_after_prior_contract_failure",
            }
            rescued = self._rescue_fast_scope_search_with_current_catalog(
                snapshot_kind=snapshot_kind,
                query_text=clean_query,
                limit=max(1, int(limit)),
                failure_details=failure_details,
            )
            if rescued is not None:
                return rescued
            raise LongTermRemoteReadFailedError(
                "Required remote long-term fast-topic retrieval failed.",
                details=dict(failure_details),
            )
        bounded_limit = max(1, int(limit))
        definition = self._require_definition(snapshot_kind)
        fast_retry_attempts, configured_fast_retry_attempts = self._fast_topic_topk_retry_attempts()
        fast_retry_backoff_s = self._remote_retry_backoff_s() if fast_retry_attempts > 1 else 0.0
        fast_retry_mode = "bounded_transient_retry" if fast_retry_attempts > 1 else "single_attempt_configured"
        fast_read_context = LongTermRemoteReadContext(
            snapshot_kind=snapshot_kind,
            operation="fast_topic_topk_search",
            request_method="POST",
            request_payload_kind="topk_scope_query",
            query_text=clean_query,
            catalog_entry_count=0,
            allowed_doc_count=0,
            result_limit=bounded_limit,
            request_path="/v1/external/retrieve/topk_records",
            timeout_s=timeout_s,
            scope_ref=scope_ref,
            namespace=namespace,
            attempt_index=1,
            attempt_count=fast_retry_attempts,
            retry_attempts_configured=configured_fast_retry_attempts,
            retry_backoff_s=fast_retry_backoff_s,
            retry_mode=fast_retry_mode,
        )
        started_monotonic = time.monotonic()
        try:
            candidates = _run_timed_workflow_step(
                name="longterm_remote_catalog_fast_scope_search",
                kind="retrieval",
                details={
                    **_trace_search_details(
                        snapshot_kind=snapshot_kind,
                        query_text=clean_query,
                        result_limit=bounded_limit,
                        scope_ref=scope_ref,
                        namespace=namespace,
                        catalog_entry_count=0,
                    ),
                    "timeout_s": None if timeout_s is None else round(max(0.0, float(timeout_s)), 3),
                    "retry_attempts_configured": configured_fast_retry_attempts,
                    "retry_mode": fast_retry_mode,
                },
                operation=lambda: self._search_remote_candidates(
                    snapshot_kind=snapshot_kind,
                    read_client=read_client,
                    query_text=clean_query,
                    result_limit=bounded_limit,
                    allowed_doc_ids=(),
                    scope_ref=scope_ref,
                    namespace=namespace,
                    catalog_entry_count=0,
                    timeout_s=timeout_s,
                    remote_read_context=fast_read_context,
                    topk_failure_outcome="failed",
                    topk_retry_attempts=fast_retry_attempts,
                    topk_retry_attempts_configured=configured_fast_retry_attempts,
                    topk_retry_backoff_s=fast_retry_backoff_s,
                    allow_missing_current_head_empty=True,
                ),
            )
        except Exception as exc:
            if isinstance(exc, LongTermRemoteReadFailedError):
                failure_details = dict(exc.details)
            else:
                failure_details = build_remote_read_failure_details(
                    remote_state=remote_state,
                    context=fast_read_context,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                workflow_event(
                    kind="warning",
                    msg="longterm_remote_catalog_fast_scope_search_failed",
                    details=failure_details,
                )
            scope_failure_mode = self._scope_search_failure_mode(
                snapshot_kind=snapshot_kind,
                exc=exc,
            )
            rescued = self._rescue_fast_scope_search_with_current_catalog(
                snapshot_kind=snapshot_kind,
                query_text=clean_query,
                limit=bounded_limit,
                failure_details=failure_details,
                allow_cold_catalog_load=scope_failure_mode in {"unsupported_scope_search", "missing_current_head"},
            )
            if rescued is not None:
                return rescued
            if isinstance(exc, LongTermRemoteReadFailedError):
                raise
            raise LongTermRemoteReadFailedError(
                "Required remote long-term fast-topic retrieval failed.",
                details=failure_details,
            ) from exc
        selected_payloads: list[dict[str, object]] = []
        selected_entries: list[LongTermRemoteCatalogEntry] = []
        seen_item_ids: set[str] = set()
        for candidate in candidates:
            entry = self._candidate_catalog_entry(definition=definition, payload=candidate)
            if entry is None or entry.item_id in seen_item_ids:
                continue
            payload = self._extract_item_payload(
                definition=definition,
                item_id=entry.item_id,
                payload=candidate,
            )
            if not isinstance(payload, Mapping):
                continue
            payload_dict = dict(payload)
            self._store_item_payload(
                snapshot_kind=snapshot_kind,
                item_id=entry.item_id,
                payload=payload_dict,
            )
            selected_payloads.append(payload_dict)
            selected_entries.append(entry)
            seen_item_ids.add(entry.item_id)
            if len(selected_payloads) >= bounded_limit:
                break
        with workflow_span(
            name="longterm_remote_catalog_fast_scope_rescue",
            kind="retrieval",
            details={
                **_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    query_text=clean_query,
                    result_limit=bounded_limit,
                    scope_ref=scope_ref,
                    namespace=namespace,
                    catalog_entry_count=0,
                ),
                "candidate_limit": bounded_limit,
            },
        ):
            rescued = self._rescue_scope_search_payloads_with_current_catalog(
                snapshot_kind=snapshot_kind,
                query_text=clean_query,
                limit=bounded_limit,
                eligible=None,
                selected_entries=tuple(selected_entries),
                selected_payloads=tuple(selected_payloads),
                cached_only=True,
            )
        if rescued is not None:
            return rescued
        return tuple(selected_payloads)

    def _rescue_scope_search_payloads_with_current_catalog(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None,
        selected_entries: tuple[LongTermRemoteCatalogEntry, ...],
        selected_payloads: tuple[dict[str, object], ...],
        cached_only: bool = False,
    ) -> tuple[dict[str, object], ...] | None:
        """Reconcile empty/stale scope-search hits against the current catalog head."""

        current_entries = self._cached_catalog_entries(snapshot_kind=snapshot_kind)
        if current_entries is None:
            current_entries = self._recent_catalog_entries(snapshot_kind=snapshot_kind)
        if current_entries is None and not cached_only:
            current_entries = self.load_catalog_entries(snapshot_kind=snapshot_kind)
        if not current_entries:
            return ()
        current_by_id = {entry.item_id: entry for entry in current_entries}
        needs_rescue = not selected_payloads
        if not needs_rescue:
            for entry, payload in zip(selected_entries, selected_payloads, strict=False):
                current_entry = current_by_id.get(entry.item_id)
                if current_entry is None:
                    needs_rescue = True
                    break
                current_sha256 = self._normalize_text(current_entry.metadata.get("payload_sha256"))
                if not current_sha256:
                    continue
                if self._payload_sha256(payload) != current_sha256:
                    needs_rescue = True
                    break
        if not needs_rescue:
            return None
        selected_catalog_entries = self._local_search_catalog_entries(
            snapshot_kind=snapshot_kind,
            entries=current_entries,
            query_text=query_text,
            limit=limit,
            eligible=eligible,
        )
        if not selected_catalog_entries:
            return ()
        return self.load_selection_item_payloads(
            snapshot_kind=snapshot_kind,
            item_ids=(entry.item_id for entry in selected_catalog_entries),
        )

    def _initial_scope_search_limit(
        self,
        *,
        bounded_limit: int,
        max_request_limit: int,
        requires_client_filter: bool,
    ) -> int:
        """Choose the first one-shot top-k window for current-scope searches."""

        if not requires_client_filter:
            return bounded_limit
        return min(
            max_request_limit,
            max(bounded_limit * 8, min(max_request_limit, 16)),
        )

    def search_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Search one loaded catalog locally against the current remote projection."""

        all_entries = self.load_catalog_entries(snapshot_kind=snapshot_kind)
        if not all_entries:
            return ()
        entries = tuple(entry for entry in all_entries if eligible is None or eligible(entry))
        if not entries:
            return ()
        result_limit = max(1, int(limit))
        if len(entries) <= result_limit:
            return tuple(entries[:result_limit])
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        allowed_doc_ids = [
            entry.document_id
            for entry in entries
            if isinstance(entry.document_id, str) and entry.document_id
        ]
        if not allowed_doc_ids:
            return ()
        filtered_item_ids = tuple(entry.item_id for entry in entries)
        by_item_id = {entry.item_id: entry for entry in entries}
        topk_scope_ref, topk_namespace = self._topk_scope_request_context(
            snapshot_kind=snapshot_kind,
            entries=entries,
            all_entries=all_entries,
        )
        definition = self._require_definition(snapshot_kind)
        try:
            candidates = _run_timed_workflow_step(
                name="longterm_remote_catalog_search_catalog_remote",
                kind="retrieval",
                details=_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    query_text=query_text,
                    result_limit=result_limit,
                    allowed_doc_count=len(allowed_doc_ids),
                    scope_ref=topk_scope_ref,
                    namespace=topk_namespace,
                    catalog_entry_count=len(entries),
                ),
                operation=lambda: self._search_remote_candidates(
                    snapshot_kind=snapshot_kind,
                    read_client=read_client,
                    query_text=query_text,
                    result_limit=result_limit,
                    allowed_doc_ids=tuple(allowed_doc_ids),
                    scope_ref=topk_scope_ref,
                    namespace=topk_namespace,
                    catalog_entry_count=len(entries),
                ),
            )
        except Exception:
            return self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=all_entries,
                query_text=query_text,
                limit=result_limit,
                eligible=eligible,
            )
        if not candidates:
            return self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=all_entries,
                query_text=query_text,
                limit=result_limit,
                eligible=eligible,
            )

        selected: list[LongTermRemoteCatalogEntry] = []
        seen: set[str] = set()
        for candidate in candidates:
            metadata = candidate.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            item_id = self._normalize_item_id(metadata.get("twinr_memory_item_id"))
            if not item_id or item_id in seen:
                continue
            entry = by_item_id.get(item_id)
            if entry is None:
                continue
            cached_payload = self._extract_item_payload(
                definition=definition,
                item_id=item_id,
                payload=candidate,
            )
            if cached_payload is not None:
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=item_id,
                    payload=cached_payload,
                )
            selected.append(entry)
            seen.add(item_id)
            if len(selected) >= max(1, int(limit)):
                break
        if not selected:
            return self._local_search_catalog_entries(
                snapshot_kind=snapshot_kind,
                entries=all_entries,
                query_text=query_text,
                limit=result_limit,
                eligible=eligible,
            )
        self._store_search_result(
            snapshot_kind=snapshot_kind,
            query_text=query_text,
            limit=result_limit,
            filtered_item_ids=filtered_item_ids,
            selected_item_ids=tuple(entry.item_id for entry in selected),
        )
        return tuple(selected)

    def _search_remote_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: Any,
        query_text: str,
        result_limit: int,
        allowed_doc_ids: tuple[str, ...],
        scope_ref: str | None = None,
        namespace: str | None = None,
        catalog_entry_count: int,
        timeout_s: float | None = None,
        remote_read_context: LongTermRemoteReadContext | None = None,
        topk_failure_outcome: str = "fallback",
        topk_retry_attempts: int = 1,
        topk_retry_attempts_configured: int | None = None,
        topk_retry_backoff_s: float = 0.0,
        allow_missing_current_head_empty: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        """Run remote search, preferring one-shot structured top-k responses."""

        remote_state = self._require_remote_state()
        resolved_scope_ref = scope_ref if scope_ref and namespace and self._scope_search_supported(snapshot_kind=snapshot_kind) else None
        resolved_namespace = namespace if resolved_scope_ref else None
        scope_search = bool(resolved_scope_ref and resolved_namespace)
        retrieve_fallback_allowed = bool(allowed_doc_ids) or not scope_search
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        workflow_decision(
            msg="longterm_remote_catalog_search_api",
            question="Which remote API contract should serve this catalog candidate search?",
            selected={
                "id": "topk_records" if supports_topk_records and callable(topk_records) else "retrieve",
                "summary": (
                    "Use one-shot structured top-k retrieval."
                    if supports_topk_records and callable(topk_records)
                    else "Use the legacy retrieve endpoint."
                ),
            },
            options=[
                {
                    "id": "topk_records",
                    "summary": "Request structured top-k records directly from ChonkyDB.",
                    "score_components": {
                        "supports_topk_records": bool(supports_topk_records and callable(topk_records)),
                        "scope_search": scope_search,
                    },
                    "constraints_violated": [] if supports_topk_records and callable(topk_records) else ["topk_records_unavailable"],
                },
                {
                    "id": "retrieve",
                    "summary": "Use the legacy retrieve endpoint with ranked candidate handles.",
                    "score_components": {
                        "retrieve_fallback_allowed": retrieve_fallback_allowed,
                        "allowed_doc_count": len(allowed_doc_ids),
                    },
                    "constraints_violated": [] if retrieve_fallback_allowed else ["scope_search_requires_topk_records"],
                },
            ],
            context=_trace_search_details(
                snapshot_kind=snapshot_kind,
                query_text=query_text,
                result_limit=result_limit,
                allowed_doc_count=len(allowed_doc_ids),
                scope_ref=scope_ref,
                namespace=namespace,
                catalog_entry_count=catalog_entry_count,
            ),
            confidence="high",
            guardrails=[
                "Prefer topk_records whenever the backend supports structured one-shot responses.",
                "Use retrieve only when allowed_doc_ids or non-scope searches make fallback safe.",
            ],
            kpi_impact_estimate={
                "retrieve_fallback_allowed": retrieve_fallback_allowed,
            },
        )
        last_topk_error: Exception | None = None
        allowed_indexes = self._selection_search_allowed_indexes()
        resolved_topk_retry_attempts = max(1, int(topk_retry_attempts))
        resolved_topk_retry_attempts_configured = max(
            resolved_topk_retry_attempts,
            int(topk_retry_attempts_configured or resolved_topk_retry_attempts),
        )
        resolved_topk_retry_backoff_s = max(0.0, float(topk_retry_backoff_s))
        topk_context = remote_read_context or LongTermRemoteReadContext(
            snapshot_kind=snapshot_kind,
            operation="topk_search",
            request_method="POST",
            request_payload_kind="topk_scope_query" if scope_ref and namespace else "topk_allowed_doc_query",
            query_text=query_text,
            catalog_entry_count=catalog_entry_count,
            allowed_doc_count=len(allowed_doc_ids),
            result_limit=result_limit,
            request_path="/v1/external/retrieve/topk_records",
            timeout_s=timeout_s,
            scope_ref=scope_ref,
            namespace=namespace,
            attempt_index=1,
            attempt_count=resolved_topk_retry_attempts,
            retry_attempts_configured=resolved_topk_retry_attempts_configured,
            retry_backoff_s=resolved_topk_retry_backoff_s,
            retry_mode="bounded_transient_retry" if resolved_topk_retry_attempts > 1 else "single_attempt",
        )
        if supports_topk_records and callable(topk_records):
            last_topk_context = topk_context
            for attempt_index in range(1, resolved_topk_retry_attempts + 1):
                attempt_context = replace(
                    topk_context,
                    attempt_index=attempt_index,
                    attempt_count=resolved_topk_retry_attempts,
                    retry_attempts_configured=resolved_topk_retry_attempts_configured,
                    retry_backoff_s=resolved_topk_retry_backoff_s,
                )
                started_monotonic = time.monotonic()
                try:
                    response = _run_timed_workflow_step(
                        name="longterm_remote_catalog_topk_request",
                        kind="http",
                        details={
                            **_trace_search_details(
                                snapshot_kind=snapshot_kind,
                                query_text=query_text,
                                result_limit=result_limit,
                                allowed_doc_count=len(allowed_doc_ids),
                                scope_ref=scope_ref,
                                namespace=namespace,
                                catalog_entry_count=catalog_entry_count,
                            ),
                            "attempt_index": attempt_index,
                            "attempt_count": resolved_topk_retry_attempts,
                        },
                        operation=lambda: topk_records(
                            ChonkyDBTopKRecordsRequest(
                                query_text=query_text,
                                result_limit=result_limit,
                                include_content=False,
                                include_metadata=True,
                                allowed_indexes=allowed_indexes,
                                allowed_doc_ids=None if scope_search else allowed_doc_ids,
                                namespace=resolved_namespace,
                                scope_ref=resolved_scope_ref,
                                timeout_seconds=timeout_s,
                            )
                        ),
                    )
                    _emit_query_plan_workflow_event(
                        snapshot_kind=snapshot_kind,
                        operation=str(getattr(attempt_context, "operation", None) or "topk_search"),
                        response=response,
                        query_text=query_text,
                        result_limit=result_limit,
                        allowed_doc_count=len(allowed_doc_ids),
                        scope_ref=scope_ref,
                        namespace=namespace,
                        catalog_entry_count=catalog_entry_count,
                    )
                    record_remote_read_observation(
                        remote_state=remote_state,
                        context=attempt_context,
                        latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                        outcome="ok",
                        classification="ok",
                    )
                    return tuple(self._iter_retrieve_result_candidates(response))
                except Exception as exc:
                    failure_mode = (
                        self._scope_search_failure_mode(snapshot_kind=snapshot_kind, exc=exc)
                        if scope_search
                        else "other"
                    )
                    if failure_mode == "missing_current_head" and allow_missing_current_head_empty:
                        self._clear_unsupported_scope_search(snapshot_kind=snapshot_kind)
                        record_remote_read_observation(
                            remote_state=remote_state,
                            context=attempt_context,
                            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                            outcome="missing",
                            classification="not_found",
                        )
                        workflow_event(
                            kind="branch",
                            msg="longterm_remote_catalog_scope_search_missing_current_head",
                            details=_trace_search_details(
                                snapshot_kind=snapshot_kind,
                                query_text=query_text,
                                result_limit=result_limit,
                                allowed_doc_count=len(allowed_doc_ids),
                                scope_ref=scope_ref,
                                namespace=namespace,
                                catalog_entry_count=catalog_entry_count,
                            ),
                        )
                        return ()
                    retryable = should_retry_remote_read_error(exc)
                    if retryable and attempt_index < resolved_topk_retry_attempts:
                        delay_s = remote_read_retry_delay_s(
                            exc,
                            default_backoff_s=resolved_topk_retry_backoff_s,
                            attempt_index=attempt_index - 1,
                        )
                        workflow_event(
                            kind="retrieval",
                            msg="longterm_remote_catalog_topk_retry",
                            details={
                                **_trace_search_details(
                                    snapshot_kind=snapshot_kind,
                                    query_text=query_text,
                                    result_limit=result_limit,
                                    allowed_doc_count=len(allowed_doc_ids),
                                    scope_ref=scope_ref,
                                    namespace=namespace,
                                    catalog_entry_count=catalog_entry_count,
                                ),
                                "operation": str(getattr(attempt_context, "operation", None) or "topk_search"),
                                "attempt_index": attempt_index,
                                "attempt_count": resolved_topk_retry_attempts,
                                "retry_delay_s": delay_s,
                                "status_code": self._status_code_from_exception(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        if delay_s > 0.0:
                            time.sleep(delay_s)
                        continue
                    if failure_mode == "unsupported_scope_search":
                        self._remember_unsupported_scope_search(snapshot_kind=snapshot_kind)
                    last_topk_error = exc
                    last_topk_context = attempt_context
                    record_remote_read_diagnostic(
                        remote_state=remote_state,
                        context=attempt_context,
                        exc=exc,
                        started_monotonic=started_monotonic,
                        outcome=topk_failure_outcome,
                    )
                    break

        if not retrieve_fallback_allowed:
            if last_topk_error is not None:
                if (
                    remote_read_context is not None
                    and str(getattr(remote_read_context, "operation", "") or "") == "fast_topic_topk_search"
                ):
                    raise LongTermRemoteReadFailedError(
                        "Required remote long-term fast-topic retrieval failed.",
                        details=build_remote_read_failure_details(
                            remote_state=remote_state,
                            context=last_topk_context,
                            exc=last_topk_error,
                            started_monotonic=time.monotonic(),
                            outcome=topk_failure_outcome,
                        ),
                    ) from last_topk_error
                raise last_topk_error
            raise ChonkyDBError(
                "ChonkyDB topk_records is required for current-scope remote retrieval",
            )

        if last_topk_error is not None:
            workflow_event(
                kind="branch",
                msg="longterm_remote_catalog_fallback_to_retrieve",
                level="WARNING",
                details=_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    query_text=query_text,
                    result_limit=result_limit,
                    allowed_doc_count=len(allowed_doc_ids),
                    scope_ref=scope_ref,
                    namespace=namespace,
                    catalog_entry_count=catalog_entry_count,
                ),
                reason={
                    "selected": {"id": "retrieve", "summary": "Structured top-k failed; fall back to retrieve."},
                    "options": [
                        {"id": "retrieve", "summary": "Use the legacy retrieve endpoint.", "constraints_violated": []},
                        {"id": "raise_topk_error", "summary": "Abort immediately on top-k failure.", "constraints_violated": ["fallback_is_safe_for_this_search"]},
                    ],
                },
            )
        started_monotonic = time.monotonic()
        try:
            response = _run_timed_workflow_step(
                name="longterm_remote_catalog_retrieve_request",
                kind="http",
                details=_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    query_text=query_text,
                    result_limit=result_limit,
                    allowed_doc_count=len(allowed_doc_ids),
                    scope_ref=scope_ref,
                    namespace=namespace,
                    catalog_entry_count=catalog_entry_count,
                ),
                operation=lambda: read_client.retrieve(
                    ChonkyDBRetrieveRequest(
                        query_text=query_text,
                        result_limit=result_limit,
                        include_content=False,
                        include_metadata=True,
                        allowed_indexes=allowed_indexes,
                        allowed_doc_ids=allowed_doc_ids,
                    )
                ),
            )
        except Exception as exc:
            record_remote_read_diagnostic(
                remote_state=remote_state,
                context=LongTermRemoteReadContext(
                    snapshot_kind=snapshot_kind,
                    operation="retrieve_search",
                    request_method="POST",
                    request_payload_kind="retrieve_allowed_doc_query",
                    query_text=query_text,
                    catalog_entry_count=catalog_entry_count,
                    allowed_doc_count=len(allowed_doc_ids),
                    result_limit=result_limit,
                    request_path="/v1/external/retrieve",
                ),
                exc=exc,
                started_monotonic=started_monotonic,
                outcome="degraded",
            )
            raise
        record_remote_read_observation(
            remote_state=remote_state,
            context=LongTermRemoteReadContext(
                snapshot_kind=snapshot_kind,
                operation="retrieve_search",
                request_method="POST",
                request_payload_kind="retrieve_allowed_doc_query",
                query_text=query_text,
                catalog_entry_count=catalog_entry_count,
                allowed_doc_count=len(allowed_doc_ids),
                result_limit=result_limit,
                request_path="/v1/external/retrieve",
            ),
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )
        return tuple(self._iter_retrieve_result_candidates(response))

    def _topk_scope_request_context(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
        all_entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> tuple[str | None, str | None]:
        """Return the server-side scope context for full current-snapshot searches only."""

        if len(entries) != len(all_entries):
            return None, None
        return self._current_scope_request_context(snapshot_kind=snapshot_kind)

    def _local_search_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Rank already-loaded catalog entries locally through the cached selector."""

        bounded_limit = max(1, int(limit))
        cached_selector = self._local_search_selector(snapshot_kind=snapshot_kind, entries=entries)
        selector = self._build_local_search_selector(entries=entries) if cached_selector is None else cached_selector.selector
        by_item_id = {entry.item_id: entry for entry in entries} if cached_selector is None else cached_selector.by_item_id
        search_limit = bounded_limit if eligible is None else len(entries)
        selected_ids = selector.search(
            query_text,
            limit=max(1, search_limit),
            category="remote_catalog",
            allow_fallback=False,
        )
        selected: list[LongTermRemoteCatalogEntry] = []
        for item_id in selected_ids:
            entry = by_item_id.get(item_id)
            if entry is None:
                continue
            if eligible is not None and not eligible(entry):
                continue
            selected.append(entry)
            if len(selected) >= bounded_limit:
                break
        return tuple(selected)

    def _catalog_entry_search_text(self, entry: LongTermRemoteCatalogEntry) -> str:
        """Build one local fallback search document from catalog metadata."""

        parts: list[str] = [entry.item_id]
        for field_name in self._catalog_entry_text_fields():
            value = self._normalize_text(entry.metadata.get(field_name))
            if value:
                parts.append(value)
        for field_name in self._catalog_entry_list_fields():
            values = self._normalize_text_list(entry.metadata.get(field_name))
            if values:
                parts.extend(values)
        return " ".join(parts)

    def top_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
        preserve_order: bool = False,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Return current catalog entries ordered either by recency or catalog order."""

        entries = [
            entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
            if eligible is None or eligible(entry)
        ]
        if not preserve_order:
            entries.sort(key=lambda entry: (entry.updated_at(), entry.item_id), reverse=True)
        return tuple(entries[: max(1, int(limit))])


__all__ = [
    "RemoteCatalogSearchMixin",
]
