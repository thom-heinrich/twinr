# mypy: disable-error-code=attr-defined
"""Provider-context assembly for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

from twinr.agent.workflows.forensics import workflow_event, workflow_span
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.core.models import LongTermMemoryContext
from twinr.memory.longterm.core.ontology import kind_matches
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
    LongTermRemoteUnavailableError,
)
from twinr.memory.query_normalization import LongTermQueryProfile

from ._typing import ServiceMixinBase
from .compat import (
    _MAX_REVIEW_LIMIT,
    _coerce_positive_int,
    _context_details,
    _writer_state_details,
    logger,
)


class LongTermMemoryServiceContextMixin(ServiceMixinBase):
    """Build normal, fast, and tool-facing provider contexts."""

    def prewarm_foreground_read_cache(self) -> None:
        """Warm remote-backed read caches for the first foreground text turn."""

        with self._store_lock:
            with self._temporary_remote_probe_cache():
                remote_catalog = getattr(getattr(self, "object_store", None), "_remote_catalog", None)
                load_catalog_entries = cast(
                    Callable[..., object] | None,
                    getattr(remote_catalog, "load_catalog_entries", None),
                )
                if load_catalog_entries is not None:
                    with workflow_span(name="longterm_prewarm_object_catalog", kind="cache"):
                        load_catalog_entries(snapshot_kind="objects")
                    with workflow_span(name="longterm_prewarm_conflict_catalog", kind="cache"):
                        load_catalog_entries(snapshot_kind="conflicts")
                load_packets = getattr(self.midterm_store, "load_packets", None)
                if callable(load_packets):
                    load_packets()
                load_document = getattr(self.graph_store, "load_document", None)
                if callable(load_document):
                    load_document()

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build the normal long-term context injected into provider prompts."""

        try:
            query = self.query_rewriter.profile(query_text)
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                workflow_event(
                    kind="branch",
                    msg="longterm_provider_context_shared_lock_skipped",
                    details={
                        "lock_used": False,
                        "conversation_writer": writer_details,
                        "multimodal_writer": multimodal_writer_details,
                    },
                )
                try:
                    topic_context = self._build_quick_topic_context(
                        query_text=query_text,
                        workflow_event_name="longterm_provider_context_quick_memory_remote_read_failed",
                    )
                except LongTermRemoteReadFailedError as exc:
                    # Quick-topic hints are an auxiliary low-latency lane. When that
                    # bounded read times out, still let the main remote retriever build
                    # the authoritative provider context for the turn.
                    workflow_event(
                        kind="warning",
                        msg="longterm_provider_context_quick_memory_skipped",
                        details=dict(exc.details),
                    )
                    topic_context = None
                with self._temporary_remote_probe_cache():
                    with workflow_span(
                        name="longterm_service_provider_context_retrieval",
                        kind="retrieval",
                        details={"query_present": bool(str(query_text or "").strip())},
                    ):
                        context = self.retriever.build_context(
                            query=query,
                            original_query_text=query_text,
                        )
                if topic_context:
                    context = replace(context, topic_context=topic_context)
                workflow_event(
                    kind="retrieval",
                    msg="longterm_provider_context_built",
                    details=_context_details(context),
                )
                return context
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build long-term provider context.")
            return LongTermMemoryContext()

    def build_fast_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build a tiny topic-memory context for latency-sensitive answer paths."""

        if not self.config.long_term_memory_enabled or not self.config.long_term_memory_fast_topic_enabled:
            return LongTermMemoryContext()
        try:
            query = LongTermQueryProfile.from_text(query_text)
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_fast_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                topic_context = self._build_quick_topic_context(
                    query_text=query.original_text or query.retrieval_text,
                    workflow_event_name="longterm_fast_provider_context_remote_read_failed",
                )
                context = LongTermMemoryContext(topic_context=topic_context)
                workflow_event(
                    kind="retrieval",
                    msg="longterm_fast_provider_context_built",
                    details=_context_details(context),
                )
                return context
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build fast long-term provider context.")
            return LongTermMemoryContext()

    def _build_quick_topic_context(
        self,
        *,
        query_text: str | None,
        workflow_event_name: str,
    ) -> str | None:
        """Build the compact quick-memory topic block for one answer turn."""

        if (
            not self.config.long_term_memory_enabled
            or not self.config.long_term_memory_fast_topic_enabled
            or self.fast_topic_builder is None
        ):
            return None
        try:
            query = LongTermQueryProfile.from_text(query_text)
            return self.fast_topic_builder.build(query_profile=query)
        except LongTermRemoteReadFailedError as exc:
            workflow_event(
                kind="warning",
                msg=workflow_event_name,
                details=dict(exc.details),
            )
            raise
        except LongTermRemoteUnavailableError:
            raise
        except ChonkyDBError as exc:
            details: dict[str, object] = {
                "operation": "fast_provider_context",
                "request_kind": "read",
                "outcome": "failed",
                "classification": "unexpected_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            workflow_event(
                kind="warning",
                msg=workflow_event_name,
                details=details,
            )
            raise LongTermRemoteReadFailedError(
                "Required remote long-term fast-topic retrieval failed.",
                details=details,
            ) from exc
        except Exception as exc:
            logger.warning(
                "Failed to build fast-topic memory hints inside the quick-memory path. error_type=%s",
                type(exc).__name__,
            )
            return None

    def build_tool_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build a tool-facing context with sensitive details redacted."""

        try:
            query = self.query_rewriter.profile(query_text)
            writer_details = _writer_state_details(getattr(self, "writer", None))
            multimodal_writer_details = _writer_state_details(getattr(self, "multimodal_writer", None))
            with workflow_span(
                name="longterm_service_build_tool_provider_context",
                kind="retrieval",
                details={
                    "query_present": bool(str(query_text or "").strip()),
                    "conversation_writer": writer_details,
                    "multimodal_writer": multimodal_writer_details,
                },
            ):
                workflow_event(
                    kind="branch",
                    msg="longterm_tool_provider_context_shared_lock_skipped",
                    details={
                        "lock_used": False,
                        "conversation_writer": writer_details,
                        "multimodal_writer": multimodal_writer_details,
                    },
                )
                with self._temporary_remote_probe_cache():
                    context = self.retriever.build_context(
                        query=query,
                        original_query_text=query_text,
                    )
                    recall_limit = max(
                        1,
                        _coerce_positive_int(
                            getattr(self.config, "long_term_memory_recall_limit", 1),
                            default=1,
                            maximum=_MAX_REVIEW_LIMIT,
                        ),
                    )
                    conflict_queue = self.retriever.select_conflict_queue(
                        query=query,
                        limit=recall_limit,
                    )
                    conflicting_memory_ids = {
                        option.memory_id
                        for item in conflict_queue
                        for option in item.options
                    }
                    durable_objects = self.retriever.select_durable_objects(
                        query=query,
                        limit=recall_limit,
                    )
                    filtered_durable_objects = tuple(
                        item
                        for item in durable_objects
                        if not kind_matches(
                            item.kind,
                            "fact",
                            item.attributes,
                            attr_key="fact_type",
                            attr_value="contact_method",
                        )
                        and item.memory_id not in conflicting_memory_ids
                    )
                    tool_context = LongTermMemoryContext(
                        subtext_context=context.subtext_context,
                        midterm_context=context.midterm_context,
                        durable_context=self.retriever._render_durable_context(filtered_durable_objects),
                        episodic_context=context.episodic_context,
                        graph_context=self.graph_store.build_prompt_context(
                            query.retrieval_text or query.original_text,
                            include_contact_methods=False,
                        ),
                        conflict_context=None,
                    )
                workflow_event(
                    kind="retrieval",
                    msg="longterm_tool_provider_context_built",
                    details=_context_details(tool_context),
                )
                return tool_context
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build tool-facing long-term provider context.")
            return LongTermMemoryContext()
