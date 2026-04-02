"""Persist and query Twinr's local personal-memory graph.

This module owns the on-device graph store that backs contact lookup,
preferences, plans, and prompt-context extraction for long-term memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb._remote_graph_state import TwinrRemoteGraphState
from twinr.memory.chonkydb.client import ChonkyDBError, chonkydb_data_path
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.chonkydb.schema import (
    TwinrGraphDocumentV1,
    TwinrGraphEdgeV1,
    TwinrGraphNodeV1,
)
from twinr.temporal import parse_local_date_text
from twinr.text_utils import collapse_whitespace, is_valid_stable_identifier, retrieval_terms, slugify_identifier

if TYPE_CHECKING:
    from twinr.memory.longterm.core.models import LongTermGraphEdgeCandidateV1
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux/RPi provides fcntl, but keep the module import-safe.
    fcntl = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_CONTACT_METHOD_QUERY_TOKENS = frozenset(
    {
        "anrufen",
        "call",
        "contact",
        "dial",
        "email",
        "mail",
        "message",
        "nachricht",
        "nummer",
        "number",
        "phone",
        "reach",
        "rufen",
        "sms",
        "telefon",
        "telefonnummer",
        "text",
        "write",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(value: object, *, limit: int) -> str:
    # AUDIT-FIX(#9): Normalize non-string inputs defensively so malformed upstream payloads do not explode user flows.
    if limit <= 0:
        return ""
    text = collapse_whitespace(str(value or ""))
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


def _tokenize(value: object) -> tuple[str, ...]:
    return retrieval_terms(str(value or ""))


def _query_match_terms(query_terms: set[str]) -> set[str]:
    """Prefer content-bearing query terms over auxiliary-word-only overlap."""

    if not query_terms:
        return set()
    informative = {
        term
        for term in query_terms
        if isinstance(term, str) and term and (term.isdigit() or len(term) >= 4)
    }
    return informative or query_terms


def _has_query_overlap(*, query_terms: set[str], document_terms: set[str]) -> bool:
    """Return whether document terms overlap one query through exact or compound matches."""

    informative_query_terms = _query_match_terms(query_terms)
    informative_document_terms = _query_match_terms(document_terms)
    if not informative_query_terms or not informative_document_terms:
        return False
    if informative_query_terms.intersection(informative_document_terms):
        return True
    for query_term in informative_query_terms:
        for document_term in informative_document_terms:
            if query_term in document_term or document_term in query_term:
                return True
    return False


def _canonical_phone(value: object) -> str:
    # AUDIT-FIX(#3): Canonicalize phones deterministically so matching uses a stable representation.
    raw = str(value or "").strip()
    has_leading_plus = raw.startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return ""
    return f"+{digits}" if has_leading_plus else digits


def _canonical_email(value: object) -> str:
    return str(value or "").strip().lower()


def _infer_day_key(when_text: str | None, *, timezone_name: str) -> str | None:
    if not when_text:
        return None
    try:
        resolved = parse_local_date_text(when_text, timezone_name=timezone_name)
    except Exception:
        # AUDIT-FIX(#7): Bad date text must degrade gracefully instead of taking down plan persistence for senior users.
        logger.warning("Failed to parse local date text for graph memory.", exc_info=True)
        return None
    return resolved.isoformat() if resolved is not None else None


def _resolve_config_path(configured: str | Path, *, project_root: str | Path) -> Path:
    """Resolve one config-backed path against the Twinr project root."""

    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (Path(project_root).expanduser().resolve(strict=False) / path).resolve(strict=False)


def _remote_unavailable_error_type():
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

    return LongTermRemoteUnavailableError


def _is_remote_not_found_error(exc: Exception) -> bool:
    """Return whether one backend exception represents a legitimate missing doc."""

    if isinstance(exc, ChonkyDBError):
        return int(exc.status_code or 0) == 404
    return False


@dataclass(frozen=True, slots=True)
class TwinrGraphContactOption:
    """Describe one contact option shown during lookup clarification."""

    person_node_id: str
    label: str
    role: str | None = None
    phones: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()

    @property
    def detail(self) -> str:
        """Summarize the role and available contact methods for display."""

        parts: list[str] = []
        if self.role:
            parts.append(self.role)
        if self.phones:
            count = len(self.phones)
            parts.append(f"{count} phone number{'s' if count != 1 else ''}")
        if self.emails:
            count = len(self.emails)
            parts.append(f"{count} email{'s' if count != 1 else ''}")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class TwinrGraphLookupResult:
    """Represent the outcome of a contact lookup attempt."""

    status: str
    match: TwinrGraphContactOption | None = None
    options: tuple[TwinrGraphContactOption, ...] = ()
    question: str | None = None


@dataclass(frozen=True, slots=True)
class TwinrGraphWriteResult:
    """Represent the outcome of a graph write operation."""

    status: str
    label: str
    node_id: str
    edge_type: str | None = None
    question: str | None = None
    options: tuple[TwinrGraphContactOption, ...] = ()


@dataclass(frozen=True, slots=True)
class TwinrGraphContextSelection:
    """Hold one selected graph document plus the query plan that chose it."""

    document: TwinrGraphDocumentV1
    query_plan: dict[str, object] | None = None


class TwinrPersonalGraphStore:
    """Manage the persisted Twinr personal graph and prompt-memory extracts."""

    def __init__(
        self,
        path: str | Path,
        *,
        user_node_id: str = "user:main",
        user_label: str = "Main user",
        timezone_name: str = "Europe/Berlin",
        remote_state: "LongTermRemoteStateStore | None" = None,
        lock_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.user_node_id = user_node_id
        self.user_label = _normalize_text(user_label, limit=80) or "Main user"
        self.timezone_name = timezone_name
        self.remote_state = remote_state
        self._backup_path = self.path.with_name(f"{self.path.name}.bak")
        self._lock_path = (
            Path(lock_path).expanduser()
            if lock_path is not None
            else self.path.with_name(f"{self.path.name}.lock")
        )
        # AUDIT-FIX(#1): Protect the full read-modify-write cycle against concurrent callers in-process.
        self._document_lock_handle = threading.RLock()
        self._remote_graph = TwinrRemoteGraphState(remote_state)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrPersonalGraphStore":
        """Build a graph store from the active Twinr runtime configuration."""

        from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

        base = chonkydb_data_path(config)
        path = base / "twinr_graph_v1.json"
        runtime_state_path = _resolve_config_path(
            config.runtime_state_path,
            project_root=config.project_root,
        )
        return cls(
            path=path,
            user_label=config.user_display_name or "Main user",
            timezone_name=config.local_timezone_name,
            remote_state=LongTermRemoteStateStore.from_config(config),
            # Keep the inter-process graph lock outside the root-owned
            # ChonkyDB data directory so watchdog and runtime processes can
            # coordinate even when they do not share the same effective UID.
            lock_path=runtime_state_path.parent / "locks" / f"{path.name}.lock",
        )

    def load_document(self) -> TwinrGraphDocumentV1:
        """Load the current graph document from local or remote state."""

        with self._document_lock():
            return self._load_authoritative_document_locked()

    def ensure_remote_snapshot(self) -> bool:
        """Seed remote current-view state with the current graph if it is still missing."""

        if not self._remote_graph.enabled():
            return False
        with self._document_lock():
            document = self._load_local_document_locked()
            if document is None:
                current_view = self._remote_graph.current_view_summary()
                if isinstance(current_view, Mapping):
                    self._remote_graph.validate_current_view()
                    return False
                document = self._empty_document()
            return self._remote_graph.ensure_seeded(document=document)

    def ensure_remote_snapshot_for_readiness(self) -> bool:
        """Bootstrap graph readiness without forcing a seed for an empty namespace.

        Runtime readiness must stay read-only for a fresh namespace. When the
        local graph is still effectively empty and the remote graph has never
        been seeded, accept that state and let the readiness-specific probe/load
        helpers synthesize the canonical empty current-view summary.
        """

        if not self._remote_graph.enabled():
            return False
        with self._document_lock():
            document = self._load_local_document_locked()
            if document is None:
                current_view = self._remote_graph.current_view_summary()
                if isinstance(current_view, Mapping):
                    self._remote_graph.validate_current_view()
                return False
            if self._document_is_effectively_empty(document):
                current_view = self._remote_graph.current_view_summary()
                if isinstance(current_view, Mapping):
                    self._remote_graph.validate_current_view()
                    return False
                return False
            return self._remote_graph.ensure_seeded(document=document)

    def probe_remote_current_view(self) -> dict[str, object] | None:
        """Probe the remote graph current view without hydrating all node and edge payloads."""

        if not self._remote_graph.enabled():
            return None
        payload = self._remote_graph.probe_current_view()
        return dict(payload) if isinstance(payload, Mapping) else None

    def load_remote_current_view(self) -> dict[str, object] | None:
        """Load the authoritative remote graph current view via current-head records only."""

        if not self._remote_graph.enabled():
            return None
        payload = self._remote_graph.current_view_summary()
        return dict(payload) if isinstance(payload, Mapping) else None

    def probe_remote_current_view_for_readiness(self) -> dict[str, object] | None:
        """Probe the remote graph current view, accepting a fresh empty namespace.

        Runtime readiness should stay read-only. When the namespace has never
        carried any graph state and the local graph is still effectively empty,
        expose a synthetic empty current-view summary instead of forcing a seed
        write during `ensure_remote_ready()`.
        """

        graph_probe = getattr(self._remote_graph, "probe_current_view_for_readiness", None)
        try:
            if callable(graph_probe):
                payload = graph_probe()
            else:
                payload = self.probe_remote_current_view()
        except Exception as exc:
            if isinstance(exc, _remote_unavailable_error_type()) or _is_remote_not_found_error(exc):
                payload = None
            else:
                raise
        if isinstance(payload, Mapping):
            return dict(payload)
        return self._synthetic_empty_remote_current_view_summary()

    def load_remote_current_view_for_readiness(self) -> dict[str, object] | None:
        """Load the remote graph current view, accepting a fresh empty namespace."""

        try:
            payload = self.load_remote_current_view()
        except Exception as exc:
            if isinstance(exc, _remote_unavailable_error_type()) or _is_remote_not_found_error(exc):
                payload = None
            else:
                raise
        if isinstance(payload, Mapping):
            return dict(payload)
        return self._synthetic_empty_remote_current_view_summary()

    def query_remote_current_path(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        edge_types: tuple[str, ...] | None = None,
    ) -> dict[str, object] | None:
        """Query the active remote graph topology and map the path back to logical Twinr ids."""

        if not self._remote_graph.enabled():
            return None
        return self._remote_graph.query_current_path(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_types=edge_types,
        )

    def query_remote_current_neighbors(
        self,
        *,
        node_id: str,
        edge_types: tuple[str, ...] | None = None,
        limit: int = 10,
    ) -> dict[str, object] | None:
        """Query the active remote graph topology for one node's current neighbors."""

        if not self._remote_graph.enabled():
            return None
        return self._remote_graph.query_current_neighbors(
            node_id=node_id,
            edge_types=edge_types,
            limit=limit,
        )

    def apply_candidate_edges(
        self,
        graph_edges: tuple["LongTermGraphEdgeCandidateV1", ...] | list["LongTermGraphEdgeCandidateV1"],
    ) -> None:
        """Merge extracted long-term-memory edge candidates into the graph."""

        if not graph_edges:
            return
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            edges = list(document.edges)
            self._ensure_user_node(nodes)
            for candidate in graph_edges:
                try:
                    source_node_id = self._find_or_create_graph_ref_node(nodes, candidate.source_ref)
                    target_node_id = self._find_or_create_graph_ref_node(nodes, candidate.target_ref)
                    edges = self._upsert_edge(
                        edges,
                        TwinrGraphEdgeV1(
                            source_node_id=source_node_id,
                            edge_type=candidate.edge_type,
                            target_node_id=target_node_id,
                            confidence=candidate.confidence,
                            confirmed_by_user=candidate.confirmed_by_user,
                            origin="longterm_turn_extraction",
                            valid_from=candidate.valid_from,
                            valid_to=candidate.valid_to,
                            attributes=dict(candidate.attributes or {}),
                        ),
                    )
                except Exception:
                    # AUDIT-FIX(#9): Skip malformed extraction candidates instead of dropping the full memory write batch.
                    logger.warning("Skipping invalid long-term graph edge candidate.", exc_info=True)
            self._save_document_locked(nodes, edges, created_at=document.created_at)

    def remember_contact(
        self,
        *,
        given_name: str,
        family_name: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        role: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
        confirmed_by_user: bool = True,
    ) -> TwinrGraphWriteResult:
        """Create or update a person node plus its contact-method edges."""

        clean_given = _normalize_text(given_name, limit=80)
        clean_family = _normalize_text(family_name or "", limit=80) or None
        clean_role = _normalize_text(role or "", limit=80) or None
        clean_relation = _normalize_text(relation or "", limit=80) or None
        clean_phone = _canonical_phone(phone or "")
        clean_email = _canonical_email(email or "")
        clean_notes = _normalize_text(notes or "", limit=160) or None
        if not clean_given:
            raise ValueError("given_name is required.")

        match_role = clean_role or clean_relation
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            edges = list(document.edges)
            self._ensure_user_node(nodes)
            candidates = self._contact_candidates(
                document=document,
                given_name=clean_given,
                family_name=clean_family,
                role=match_role,
                contact_label=None,
                phone=clean_phone or None,
                email=clean_email or None,
            )
            if candidates:
                unique = self._resolve_contact_candidate(
                    document=document,
                    candidates=candidates,
                    family_name=clean_family,
                    role=match_role,
                    contact_label=None,
                    phone=clean_phone or None,
                    email=clean_email or None,
                )
                if unique is None:
                    options = self._contact_options(document, candidates)
                    return TwinrGraphWriteResult(
                        status="needs_clarification",
                        label=clean_given,
                        node_id="",
                        question=self._contact_conflict_question(clean_given, options),
                        options=options,
                    )
                person = unique
                label = self._merge_person_label(person.label, clean_given, clean_family)
                aliases = set(person.aliases)
                aliases.add(person.label)
                aliases.add(clean_given)
                attributes = dict(person.attributes or {})
                if clean_notes:
                    attributes["notes"] = clean_notes
                if clean_relation:
                    attributes["relation"] = clean_relation
                if clean_family:
                    attributes["family_name"] = clean_family
                attributes["given_name"] = clean_given
                nodes[person.node_id] = TwinrGraphNodeV1(
                    node_id=person.node_id,
                    node_type="person",
                    label=label,
                    aliases=tuple(sorted(alias for alias in aliases if alias.strip() and alias.strip() != label)),
                    attributes=attributes or None,
                    status=person.status,
                    graph_ref=person.graph_ref,
                )
                status = "updated"
                person_node_id = person.node_id
            else:
                label = self._merge_person_label("", clean_given, clean_family)
                person_node_id = self._unique_node_id(
                    node_type="person",
                    base_slug=_slugify(label, fallback=_slugify(clean_given, fallback="person")),
                    existing_ids=set(nodes),
                )
                attributes: dict[str, object] = {"given_name": clean_given}
                if clean_family:
                    attributes["family_name"] = clean_family
                if clean_relation:
                    attributes["relation"] = clean_relation
                if clean_notes:
                    attributes["notes"] = clean_notes
                aliases = tuple(alias for alias in sorted({clean_given}) if alias != label)
                nodes[person_node_id] = TwinrGraphNodeV1(
                    node_id=person_node_id,
                    node_type="person",
                    label=label,
                    aliases=aliases,
                    attributes=attributes,
                )
                status = "created"

            if clean_role or clean_relation:
                edge = TwinrGraphEdgeV1(
                    source_node_id=person_node_id,
                    edge_type="social_related_to_user",
                    target_node_id=self.user_node_id,
                    confirmed_by_user=confirmed_by_user,
                    attributes={
                        "role": clean_role or clean_relation or "known_contact",
                        "relation": clean_relation or "",
                    },
                )
                edges = self._upsert_edge(edges, edge)

            if clean_phone:
                phone_label = _normalize_text(phone or "", limit=80) or clean_phone
                phone_node_id = self._ensure_contact_method_node(
                    nodes,
                    node_type="phone",
                    label=phone_label,
                    canonical_value=clean_phone,
                )
                edges = self._upsert_edge(
                    edges,
                    TwinrGraphEdgeV1(
                        source_node_id=person_node_id,
                        edge_type="general_has_contact_method",
                        target_node_id=phone_node_id,
                        confirmed_by_user=confirmed_by_user,
                        attributes={"kind": "phone"},
                    ),
                )

            if clean_email:
                email_node_id = self._ensure_contact_method_node(
                    nodes,
                    node_type="email",
                    label=clean_email,
                    canonical_value=clean_email,
                )
                edges = self._upsert_edge(
                    edges,
                    TwinrGraphEdgeV1(
                        source_node_id=person_node_id,
                        edge_type="general_has_contact_method",
                        target_node_id=email_node_id,
                        confirmed_by_user=confirmed_by_user,
                        attributes={"kind": "email"},
                    ),
                )

            self._save_document_locked(nodes, edges, created_at=document.created_at)
        return TwinrGraphWriteResult(status=status, label=label, node_id=person_node_id, edge_type="social_related_to_user")

    def lookup_contact(
        self,
        *,
        name: str,
        family_name: str | None = None,
        role: str | None = None,
        contact_label: str | None = None,
    ) -> TwinrGraphLookupResult:
        """Find a remembered contact and request clarification when needed."""

        clean_name = _normalize_text(name, limit=80)
        clean_family = _normalize_text(family_name or "", limit=80) or None
        clean_role = _normalize_text(role or "", limit=80) or None
        clean_contact_label = _normalize_text(contact_label or "", limit=120) or None
        if not clean_name:
            raise ValueError("name is required.")
        document = self.load_document()
        candidates = self._contact_candidates(
            document=document,
            given_name=clean_name,
            family_name=clean_family,
            role=clean_role,
            contact_label=clean_contact_label,
            phone=None,
            email=None,
        )
        if not candidates:
            return TwinrGraphLookupResult(status="not_found")
        person = self._resolve_contact_candidate(
            document=document,
            candidates=candidates,
            family_name=clean_family,
            role=clean_role,
            contact_label=clean_contact_label,
            phone=None,
            email=None,
        )
        if person is None:
            options = self._contact_options(document, candidates)
            return TwinrGraphLookupResult(
                status="needs_clarification",
                options=options,
                question=self._contact_conflict_question(clean_name, options),
            )
        return TwinrGraphLookupResult(status="found", match=self._contact_option(document, person))

    def remember_preference(
        self,
        *,
        category: str,
        value: str,
        for_product: str | None = None,
        sentiment: str = "prefer",
        details: str | None = None,
        confirmed_by_user: bool = True,
    ) -> TwinrGraphWriteResult:
        """Store or update a user preference edge in the graph."""

        clean_category = _slugify(category, fallback="thing")
        clean_value = _normalize_text(value, limit=100)
        clean_product = _normalize_text(for_product or "", limit=80) or None
        clean_details = _normalize_text(details or "", limit=140) or None
        clean_sentiment = _normalize_text(sentiment, limit=20).lower()
        if not clean_value:
            raise ValueError("value is required.")
        if clean_sentiment in {"avoid", "dislike"}:
            edge_type = "user_avoids"
            preference_mode = "avoid"
        elif clean_sentiment in {"like", "prefer"}:
            edge_type = "user_prefers"
            preference_mode = "prefer"
        else:
            # AUDIT-FIX(#8): Reject unknown sentiment values instead of silently storing the wrong preference polarity.
            raise ValueError("sentiment must be one of: prefer, like, dislike, avoid.")
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            edges = list(document.edges)
            self._ensure_user_node(nodes)

            node_type = clean_category or "thing"
            node_id = self._find_or_create_named_node(nodes, node_type=node_type, label=clean_value)
            target = nodes[node_id]
            attributes = dict(target.attributes or {})
            attributes["category"] = clean_category
            if clean_details:
                attributes["details"] = clean_details
            nodes[node_id] = TwinrGraphNodeV1(
                node_id=target.node_id,
                node_type=target.node_type,
                label=target.label,
                aliases=target.aliases,
                attributes=attributes,
                status=target.status,
                graph_ref=target.graph_ref,
            )
            edge_attributes: dict[str, object] = {"category": clean_category}
            edge_attributes["preference_mode"] = preference_mode
            if clean_product:
                edge_attributes["for_product"] = clean_product
            if clean_details:
                edge_attributes["details"] = clean_details
            edges = self._upsert_edge(
                edges,
                TwinrGraphEdgeV1(
                    source_node_id=self.user_node_id,
                    edge_type=edge_type,
                    target_node_id=node_id,
                    confirmed_by_user=confirmed_by_user,
                    attributes=edge_attributes,
                ),
            )
            self._save_document_locked(nodes, edges, created_at=document.created_at)
        return TwinrGraphWriteResult(status="updated", label=clean_value, node_id=node_id, edge_type=edge_type)

    def remember_plan(
        self,
        *,
        summary: str,
        when_text: str | None = None,
        details: str | None = None,
        confirmed_by_user: bool = True,
    ) -> TwinrGraphWriteResult:
        """Store or update a user plan and its temporal edges."""

        clean_summary = _normalize_text(summary, limit=120)
        clean_when = _normalize_text(when_text or "", limit=80) or None
        clean_details = _normalize_text(details or "", limit=140) or None
        if not clean_summary:
            raise ValueError("summary is required.")
        day_key = _infer_day_key(clean_when, timezone_name=self.timezone_name)
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            edges = list(document.edges)
            self._ensure_user_node(nodes)

            base_slug = _slugify(f"{clean_summary}_{day_key or clean_when or 'plan'}", fallback="plan")
            existing_plan = self._find_plan_node(nodes.values(), summary=clean_summary, day_key=day_key, when_text=clean_when)
            if existing_plan is None:
                plan_node_id = self._unique_node_id("plan", base_slug=base_slug, existing_ids=set(nodes))
                attributes: dict[str, object] = {}
                if clean_when:
                    attributes["when_text"] = clean_when
                if day_key:
                    attributes["day_key"] = day_key
                if clean_details:
                    attributes["details"] = clean_details
                nodes[plan_node_id] = TwinrGraphNodeV1(
                    node_id=plan_node_id,
                    node_type="plan",
                    label=clean_summary,
                    attributes=attributes or None,
                )
                status = "created"
            else:
                plan_node_id = existing_plan.node_id
                attributes = dict(existing_plan.attributes or {})
                if clean_when:
                    attributes["when_text"] = clean_when
                if day_key:
                    attributes["day_key"] = day_key
                if clean_details:
                    attributes["details"] = clean_details
                nodes[plan_node_id] = TwinrGraphNodeV1(
                    node_id=existing_plan.node_id,
                    node_type=existing_plan.node_type,
                    label=clean_summary,
                    aliases=existing_plan.aliases,
                    attributes=attributes or None,
                    status=existing_plan.status,
                    graph_ref=existing_plan.graph_ref,
                )
                status = "updated"

            edges = self._upsert_edge(
                edges,
                TwinrGraphEdgeV1(
                    source_node_id=self.user_node_id,
                    edge_type="user_plans",
                    target_node_id=plan_node_id,
                    confirmed_by_user=confirmed_by_user,
                    attributes={"when_text": clean_when or "", "day_key": day_key or ""},
                ),
            )
            if day_key:
                day_node_id = self._find_or_create_named_node(nodes, node_type="day", label=day_key)
                edges = self._upsert_edge(
                    edges,
                    TwinrGraphEdgeV1(
                        source_node_id=plan_node_id,
                        edge_type="temporal_occurs_on",
                        target_node_id=day_node_id,
                        confirmed_by_user=confirmed_by_user,
                    ),
                )

            self._save_document_locked(nodes, edges, created_at=document.created_at)
        return TwinrGraphWriteResult(status=status, label=clean_summary, node_id=plan_node_id, edge_type="user_plans")

    def delete_contact(self, *, node_id: str) -> TwinrGraphWriteResult:
        """Delete one remembered contact and any now-orphaned contact-method nodes."""

        clean_node_id = _normalize_text(node_id, limit=120)
        if not clean_node_id:
            raise ValueError("node_id is required.")
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            contact = nodes.get(clean_node_id)
            if contact is None or contact.node_type != "person":
                return TwinrGraphWriteResult(status="not_found", label="", node_id=clean_node_id)
            label = contact.label
            del nodes[clean_node_id]
            edges = [
                edge
                for edge in document.edges
                if edge.source_node_id != clean_node_id and edge.target_node_id != clean_node_id
            ]
            self._prune_unreferenced_nodes(nodes, edges)
            self._save_document_locked(nodes, edges, created_at=document.created_at)
        return TwinrGraphWriteResult(status="deleted", label=label, node_id=clean_node_id)

    def delete_preference(
        self,
        *,
        node_id: str,
        edge_type: str | None = None,
    ) -> TwinrGraphWriteResult:
        """Delete one remembered preference edge and prune unused target nodes."""

        clean_node_id = _normalize_text(node_id, limit=120)
        clean_edge_type = _normalize_text(edge_type or "", limit=80) or None
        if not clean_node_id:
            raise ValueError("node_id is required.")
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            target = nodes.get(clean_node_id)
            label = target.label if target is not None else ""
            removed = False
            edges: list[TwinrGraphEdgeV1] = []
            for edge in document.edges:
                if edge.source_node_id == self.user_node_id and edge.target_node_id == clean_node_id:
                    if clean_edge_type is None or edge.edge_type == clean_edge_type:
                        removed = True
                        continue
                edges.append(edge)
            if not removed:
                return TwinrGraphWriteResult(status="not_found", label=label, node_id=clean_node_id, edge_type=clean_edge_type)
            self._prune_unreferenced_nodes(nodes, edges)
            self._save_document_locked(nodes, edges, created_at=document.created_at)
        return TwinrGraphWriteResult(status="deleted", label=label, node_id=clean_node_id, edge_type=clean_edge_type)

    def delete_plan(self, *, node_id: str) -> TwinrGraphWriteResult:
        """Delete one remembered user plan and any orphaned temporal helper nodes."""

        clean_node_id = _normalize_text(node_id, limit=120)
        if not clean_node_id:
            raise ValueError("node_id is required.")
        with self._document_lock():
            document = self._load_authoritative_document_locked()
            nodes = {node.node_id: node for node in document.nodes}
            plan = nodes.get(clean_node_id)
            if plan is None or plan.node_type != "plan":
                return TwinrGraphWriteResult(status="not_found", label="", node_id=clean_node_id, edge_type="user_plans")
            label = plan.label
            del nodes[clean_node_id]
            edges = [
                edge
                for edge in document.edges
                if edge.source_node_id != clean_node_id and edge.target_node_id != clean_node_id
            ]
            self._prune_unreferenced_nodes(nodes, edges)
            self._save_document_locked(nodes, edges, created_at=document.created_at)
        return TwinrGraphWriteResult(status="deleted", label=label, node_id=clean_node_id, edge_type="user_plans")

    def build_prompt_context(
        self,
        query_text: str | None,
        *,
        include_contact_methods: bool = True,
    ) -> str | None:
        """Build the serialized prompt-memory context for one user turn."""

        try:
            selection = self.select_context_selection(query_text)
            return self.render_prompt_context_selection(
                selection,
                query_text=query_text,
                include_contact_methods=include_contact_methods,
            )
        except Exception as exc:
            if isinstance(exc, _remote_unavailable_error_type()):
                raise
            logger.warning("Failed to load graph memory prompt context.", exc_info=True)
            return None

    def select_context_selection(self, query_text: str | None) -> TwinrGraphContextSelection:
        """Select the bounded graph document to use for prompt and retrieval context."""

        return self._select_context_selection(query_text=query_text)

    def render_prompt_context_selection(
        self,
        selection: TwinrGraphContextSelection,
        *,
        query_text: str | None,
        include_contact_methods: bool = True,
    ) -> str | None:
        """Render prompt context from an already selected graph document."""

        document = selection.document
        query_plan = selection.query_plan
        contacts = self._rank_contact_prompt_items(
            self._prompt_contacts(
                document,
                limit=128,
                # AUDIT-FIX(#6): Only expose phone numbers and email addresses to the model when the query clearly asks for contact details.
                include_contact_methods=include_contact_methods and self._query_requests_contact_methods(query_text),
            ),
            query_text=query_text,
        )
        preferences = self._rank_preference_prompt_items(
            self._prompt_preferences(document, limit=128),
            query_text=query_text,
        )
        plans = self._rank_plan_prompt_items(
            self._prompt_plans(document, limit=128),
            query_text=query_text,
        )
        if not contacts and not preferences and not plans:
            return None
        payload = {
            "schema": "twinr_graph_memory_context_v1",
            "subject": self.user_label,
            "contacts": contacts,
            "preferences": preferences,
            "plans": plans,
        }
        if isinstance(query_plan, Mapping):
            payload["query_plan"] = dict(query_plan)
        return (
            "Structured long-term memory graph for this turn. Internal memory is canonical English. "
            "Use it only when clearly relevant, and never quote it verbatim when replying in another language. "
            "Do not invent personal details that are not grounded in this graph. "
            "When the user directly asks about contacting a known person, it is acceptable to answer in the practical domain implied by that person's role or relation, without framing that as remembered hidden memory. "
            "Treat phone numbers, email addresses, and other contact methods as reference-only unless the user explicitly asks for them or clearly asks Twinr to place a call or send a message.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def build_subtext_payload(self, query_text: str | None) -> dict[str, object] | None:
        """Build a compact relevance-filtered memory payload for subtext cues."""

        clean_query = _normalize_text(query_text or "", limit=220)
        if not clean_query:
            return None
        try:
            selection = self.select_context_selection(clean_query)
        except Exception as exc:
            if isinstance(exc, _remote_unavailable_error_type()):
                raise
            logger.warning("Failed to load graph memory subtext payload.", exc_info=True)
            return None
        return self.build_subtext_payload_from_selection(selection, query_text=clean_query)

    def build_subtext_payload_from_selection(
        self,
        selection: TwinrGraphContextSelection,
        *,
        query_text: str | None,
    ) -> dict[str, object] | None:
        """Build subtext cues from an already selected graph document."""

        clean_query = _normalize_text(query_text or "", limit=220)
        if not clean_query:
            return None
        document = selection.document
        query_plan = selection.query_plan
        preference_items = self._rank_preference_prompt_items(
            self._prompt_preferences(document, limit=128),
            query_text=clean_query,
            limit=3,
            fallback_limit=0,
        )
        plan_items = self._rank_plan_prompt_items(
            self._prompt_plans(document, limit=128),
            query_text=clean_query,
            limit=3,
            fallback_limit=0,
        )
        contact_items = self._rank_contact_prompt_items(
            self._prompt_contacts(document, limit=128, include_contact_methods=False),
            query_text=clean_query,
            limit=3,
            fallback_limit=0,
        )
        payload: dict[str, object] = {}
        preference_biases = self._subtext_preferences(preference_items)
        if preference_biases:
            payload["preference_biases"] = preference_biases
        situational_threads = self._subtext_plans(plan_items)
        if situational_threads:
            payload["situational_threads"] = situational_threads
        social_context = self._subtext_contacts(contact_items)
        if social_context:
            payload["social_context"] = social_context
        if payload and isinstance(query_plan, Mapping):
            payload["query_plan"] = dict(query_plan)
        return payload or None

    def _select_context_selection(
        self,
        *,
        query_text: str | None,
    ) -> TwinrGraphContextSelection:
        """Prefer a bounded remote subgraph before hydrating the authoritative graph document."""

        clean_query = _normalize_text(query_text or "", limit=220)
        if clean_query and self._remote_graph.enabled():
            try:
                selection = self._remote_graph.select_current_subgraph(query_text=clean_query)
            except Exception as exc:
                if isinstance(exc, _remote_unavailable_error_type()):
                    raise
                logger.warning(
                    "Failed remote query-first graph selection; falling back to authoritative graph load.",
                    exc_info=True,
                )
            else:
                if selection is not None:
                    return TwinrGraphContextSelection(
                        document=selection.document,
                        query_plan=dict(selection.query_plan),
                    )
        return TwinrGraphContextSelection(
            document=self.load_document(),
            query_plan=None,
        )

    def _select_context_document(
        self,
        *,
        query_text: str | None,
    ) -> tuple[TwinrGraphDocumentV1, dict[str, object] | None]:
        """Compatibility wrapper around the selection object contract."""

        selection = self._select_context_selection(query_text=query_text)
        return selection.document, selection.query_plan

    def _rank_contact_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_text: str | None,
        limit: int = 8,
        fallback_limit: int = 3,
    ) -> list[dict[str, object]]:
        return self._rank_prompt_items(
            items,
            query_text=query_text,
            limit=limit,
            fallback_limit=fallback_limit,
        )

    def _rank_preference_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_text: str | None,
        limit: int = 8,
        fallback_limit: int = 3,
    ) -> list[dict[str, object]]:
        return self._rank_prompt_items(
            items,
            query_text=query_text,
            limit=limit,
            fallback_limit=fallback_limit,
        )

    def _rank_plan_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_text: str | None,
        limit: int = 8,
        fallback_limit: int = 3,
    ) -> list[dict[str, object]]:
        return self._rank_prompt_items(
            items,
            query_text=query_text,
            limit=limit,
            fallback_limit=fallback_limit,
        )

    def _rank_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_text: str | None,
        limit: int,
        fallback_limit: int,
    ) -> list[dict[str, object]]:
        if not items:
            return []
        clean_query = _normalize_text(query_text or "", limit=220)
        if not clean_query:
            return items[:limit]
        try:
            selector = FullTextSelector(
                tuple(
                    FullTextDocument(
                        doc_id=str(index),
                        category="prompt_item",
                        content=self._prompt_item_search_text(item),
                    )
                    for index, item in enumerate(items)
                )
            )
            selected_ids = selector.search(
                clean_query,
                limit=limit,
                category="prompt_item",
                allow_fallback=fallback_limit > 0,
            )
        except Exception:
            # AUDIT-FIX(#9): Prompt ranking is optional memory enrichment and must never crash the turn.
            logger.warning("Prompt item ranking failed; using deterministic fallback order.", exc_info=True)
            if fallback_limit <= 0:
                return []
            return items[: min(limit, fallback_limit)]
        if not selected_ids:
            if fallback_limit <= 0:
                return []
            return items[:fallback_limit]
        selected: list[dict[str, object]] = []
        seen_indices: set[int] = set()
        for item_id in selected_ids:
            if not str(item_id).isdigit():
                continue
            index = int(item_id)
            if 0 <= index < len(items) and index not in seen_indices:
                selected.append(items[index])
                seen_indices.add(index)
        if fallback_limit <= 0:
            query_terms = _query_match_terms(set(_tokenize(clean_query)))
            filtered = [
                item
                for item in selected
                if _has_query_overlap(
                    query_terms=query_terms,
                    document_terms=set(_tokenize(self._prompt_item_search_text(item))),
                )
            ]
            if not filtered:
                return []
            return filtered
        return selected

    def _subtext_preferences(self, items: list[dict[str, object]]) -> list[dict[str, str]]:
        cues: list[dict[str, str]] = []
        for item in items:
            value = _normalize_text(str(item.get("value", "")), limit=100)
            if not value:
                continue
            kind = str(item.get("type", "")).strip()
            product = _normalize_text(str(item.get("for_product", "")), limit=80)
            category = _normalize_text(str(item.get("category", "")), limit=40)
            if kind == "avoidance":
                guidance = f"Avoid steering the user toward {value} unless they explicitly ask for it."
            else:
                guidance = f"Let familiarity with {value} subtly influence suggestions when relevant."
            cue = {"kind": kind or "preference", "value": value, "guidance": guidance}
            if product:
                cue["topic"] = product
            elif category:
                cue["topic"] = category
            associations = item.get("associations")
            if isinstance(associations, list) and associations:
                relation_names = [
                    _normalize_text(str(entry.get("relation", "")), limit=40)
                    for entry in associations
                    if isinstance(entry, dict)
                ]
                compact_relations = [name for name in relation_names if name]
                if compact_relations:
                    cue["supporting_relations"] = ", ".join(compact_relations[:3])
            cues.append(cue)
        return cues

    def _subtext_plans(self, items: list[dict[str, object]]) -> list[dict[str, str]]:
        cues: list[dict[str, str]] = []
        for item in items:
            summary = _normalize_text(str(item.get("summary", "")), limit=120)
            if not summary:
                continue
            cue = {
                "topic": summary,
                "guidance": (
                    "If the current topic naturally connects to this situation, let it shape the answer's framing, "
                    "suggested plan, or priority without introducing it as an explicit memory fact."
                ),
            }
            when_value = _normalize_text(str(item.get("when") or item.get("date") or ""), limit=80)
            if when_value:
                cue["when"] = when_value
            details = _normalize_text(str(item.get("details", "")), limit=140)
            if details:
                cue["details"] = details
            cues.append(cue)
        return cues

    def _subtext_contacts(self, items: list[dict[str, object]]) -> list[dict[str, str]]:
        cues: list[dict[str, str]] = []
        for item in items:
            name = _normalize_text(str(item.get("name", "")), limit=100)
            if not name:
                continue
            role = _normalize_text(str(item.get("role") or item.get("relation") or ""), limit=100)
            if role:
                guidance = (
                    f"When this person is relevant, treat {name} as the user's familiar {role}. "
                    "Use that role as hidden planning context for urgency, practical reasons to contact them, and next steps. "
                    "If the user directly mentions this person, let the answer naturally speak in the practical domain implied by that role or relation instead of generic contact advice. "
                    "Prefer concrete reasons to contact them, practical domain language, or concrete follow-up questions over generic contact advice. "
                    "Do not frame the role as remembered hidden biography unless the user needs identity clarification or explicitly asks who this person is."
                )
            else:
                guidance = f"When this person is relevant, treat {name} as an already-known person in the user's life."
            cue = {"person": name, "guidance": guidance}
            if role:
                cue["role"] = role
            cues.append(cue)
        return cues

    def _prompt_item_search_text(self, value: object) -> str:
        return _normalize_text(" ".join(self._prompt_item_strings(value)), limit=480)

    def _prompt_item_strings(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for item in value:
                parts.extend(self._prompt_item_strings(item))
            return parts
        if isinstance(value, dict):
            parts: list[str] = []
            for item in value.values():
                parts.extend(self._prompt_item_strings(item))
            return parts
        return []

    def _prompt_contacts(
        self,
        document: TwinrGraphDocumentV1,
        *,
        limit: int = 12,
        include_contact_methods: bool = True,
    ) -> list[dict[str, object]]:
        contacts: list[dict[str, object]] = []
        for person in sorted(self._all_contact_nodes(document), key=lambda item: item.label.lower()):
            methods = self._contact_methods(document, person.node_id, canonical=False)
            role_info = self._contact_role_info(document, person.node_id)
            item: dict[str, object] = {
                "name": person.label,
            }
            if person.aliases:
                item["aliases"] = list(person.aliases)
            if role_info["role"]:
                item["role"] = role_info["role"]
            if role_info["relation"]:
                item["relation"] = role_info["relation"]
            if methods and include_contact_methods:
                phones = [method for kind, method in methods if kind == "phone"]
                emails = [method for kind, method in methods if kind == "email"]
                if phones:
                    item["phones"] = phones
                if emails:
                    item["emails"] = emails
            notes = _normalize_text(str((person.attributes or {}).get("notes", "")), limit=160)
            if notes:
                item["notes"] = notes
            contacts.append(item)
            if len(contacts) >= limit:
                break
        return contacts

    def _prompt_preferences(self, document: TwinrGraphDocumentV1, *, limit: int = 10) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for edge in sorted(
            self._outgoing_edges(
                document,
                self.user_node_id,
                edge_types={"user_prefers", "user_avoids", "user_engages_with"},
            ),
            key=lambda item: (item.edge_type, item.target_node_id),
        ):
            target = self._node_by_id(document, edge.target_node_id)
            if target is None:
                continue
            item: dict[str, object] = {
                "type": self._preference_kind(edge.edge_type),
                "value": target.label,
            }
            category = _normalize_text(str((edge.attributes or {}).get("category", "")), limit=40)
            if category:
                item["category"] = category
            for_product = _normalize_text(str((edge.attributes or {}).get("for_product", "")), limit=80)
            if for_product:
                item["for_product"] = for_product
            details = _normalize_text(
                str((edge.attributes or {}).get("details") or (target.attributes or {}).get("details", "")),
                limit=140,
            )
            if details:
                item["details"] = details
            if self._node_is_near_user(document, target.node_id):
                item["nearby"] = True
            associations = self._related_entities(document, target.node_id)
            if associations:
                item["associations"] = associations
            entries.append(item)
            if len(entries) >= limit:
                break
        return entries

    def _prompt_plans(self, document: TwinrGraphDocumentV1, *, limit: int = 10) -> list[dict[str, object]]:
        plans: list[tuple[tuple[str, str], dict[str, object]]] = []
        for edge in self._outgoing_edges(document, self.user_node_id, edge_types={"user_plans"}):
            plan = self._node_by_id(document, edge.target_node_id)
            if plan is None:
                continue
            item: dict[str, object] = {"summary": plan.label}
            when_text = _normalize_text(str((edge.attributes or {}).get("when_text", "")), limit=80)
            day_key = _normalize_text(str((edge.attributes or {}).get("day_key", "")), limit=40)
            details = _normalize_text(str((plan.attributes or {}).get("details", "")), limit=140)
            if when_text:
                item["when"] = when_text
            if day_key:
                item["date"] = day_key
            if details:
                item["details"] = details
            related_people = sorted(person.label for person in self._related_people_for_plan(document, plan.node_id))
            if related_people:
                item["related_people"] = related_people
            plans.append(((day_key or "9999-99-99", plan.label.lower()), item))
        plans.sort(key=lambda item: item[0])
        return [item for _sort_key, item in plans[:limit]]

    def _empty_document(self) -> TwinrGraphDocumentV1:
        user = TwinrGraphNodeV1(node_id=self.user_node_id, node_type="user", label=self.user_label)
        now = _utc_now_iso()  # AUDIT-FIX(#7): Emit timezone-aware UTC timestamps instead of naive datetimes with a manual suffix.
        return TwinrGraphDocumentV1(
            subject_node_id=self.user_node_id,
            graph_id="graph:user_main",
            created_at=now,
            updated_at=now,
            nodes=(user,),
            edges=(),
            metadata={"kind": "personal_graph"},
        )

    def _document_is_effectively_empty(self, document: TwinrGraphDocumentV1) -> bool:
        """Return whether one graph document only contains the canonical user node."""

        if document.edges:
            return False
        if len(document.nodes) != 1:
            return False
        user_node = document.nodes[0]
        return bool(
            user_node.node_id == self.user_node_id
            and user_node.node_type == "user"
            and document.subject_node_id == self.user_node_id
        )

    def _synthetic_empty_remote_current_view_summary(self) -> dict[str, object] | None:
        """Return a read-only synthetic current-view summary for a fresh namespace."""

        if not self._remote_graph.enabled():
            return None
        with self._document_lock():
            document = self._load_local_document_locked()
            if document is None:
                user = TwinrGraphNodeV1(node_id=self.user_node_id, node_type="user", label=self.user_label)
                document = TwinrGraphDocumentV1(
                    subject_node_id=self.user_node_id,
                    graph_id="graph:user_main",
                    created_at="1970-01-01T00:00:00Z",
                    updated_at="1970-01-01T00:00:00Z",
                    nodes=(user,),
                    edges=(),
                    metadata={"kind": "personal_graph"},
                )
            if not self._document_is_effectively_empty(document):
                return None
            serialized = json.dumps(
                document.to_payload(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            namespace = _normalize_text(getattr(self.remote_state, "namespace", None), limit=120) or "twinr_longterm_v1"
            topology_index_name = f"twinr_graph_{slugify_identifier(namespace, fallback='namespace')}_bootstrap_empty"
            return {
                "generation_id": f"gen_{hashlib.sha1(serialized).hexdigest()[:16]}",
                "topology_index_name": topology_index_name,
                "subject_node_id": document.subject_node_id,
                "graph_id": document.graph_id,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "topology_refs": {self.user_node_id: f"bootstrap_empty:{self.user_node_id}"},
                "synthetic_empty": True,
            }

    def _save_document(
        self,
        nodes: Mapping[str, TwinrGraphNodeV1],
        edges: list[TwinrGraphEdgeV1],
        *,
        created_at: str | None = None,
    ) -> None:
        with self._document_lock():
            # AUDIT-FIX(#1): Keep the public save path safe even if a new caller invokes it directly.
            effective_created_at = created_at or self._load_authoritative_document_locked().created_at
            self._save_document_locked(nodes, edges, created_at=effective_created_at)

    def _ensure_user_node(self, nodes: dict[str, TwinrGraphNodeV1]) -> None:
        if self.user_node_id not in nodes:
            nodes[self.user_node_id] = TwinrGraphNodeV1(
                node_id=self.user_node_id,
                node_type="user",
                label=self.user_label,
            )

    def _contact_candidates(
        self,
        *,
        document: TwinrGraphDocumentV1,
        given_name: str,
        family_name: str | None,
        role: str | None,
        contact_label: str | None,
        phone: str | None,
        email: str | None,
    ) -> list[TwinrGraphNodeV1]:
        full_label = self._merge_person_label("", given_name, family_name)
        given_tokens = set(_tokenize(given_name))
        family_tokens = set(_tokenize(family_name or ""))
        requested_role = (role or "").strip().lower()
        requested_contact_label = (contact_label or "").strip().lower()
        ranked: list[tuple[int, TwinrGraphNodeV1]] = []
        for node in document.nodes:
            if node.node_type != "person":
                continue
            stored_family = _normalize_text(str((node.attributes or {}).get("family_name", "")), limit=80) or None
            labels = {node.label, *node.aliases}
            normalized_labels = {label.lower() for label in labels if label}
            label_tokens: set[str] = set()
            for label in labels:
                label_tokens.update(_tokenize(label))
            option = self._contact_option(document, node)
            exact_contact_match = bool(
                (phone and phone in option.phones)
                or (email and email in option.emails)
            )
            if requested_contact_label and requested_contact_label not in normalized_labels:
                continue
            if family_name:
                family_matches = bool(
                    exact_contact_match
                    or
                    (stored_family and stored_family.lower() == family_name.lower())
                    or (family_tokens and family_tokens <= label_tokens)
                )
                if not family_matches:
                    # AUDIT-FIX(#4): Do not merge a partially specified person into a same-first-name contact when the family name does not match.
                    continue
            score = 0
            if full_label and full_label.lower() == node.label.lower():
                score += 8
            if requested_contact_label and requested_contact_label in normalized_labels:
                score += 8
            if given_tokens and given_tokens <= label_tokens:
                score += 3
            if family_tokens and family_tokens <= label_tokens:
                score += 3
            role_detail = (self._contact_role(document, node.node_id) or "").lower()
            if requested_role and requested_role not in role_detail and not exact_contact_match and role_detail:
                # AUDIT-FIX(#4): A requested role is a disambiguator, not a soft preference.
                continue
            if requested_role:
                score += 4
            if phone and phone in option.phones:
                score += 5
            if email and email in option.emails:
                score += 5
            if score > 0:
                ranked.append((score, node))
        ranked.sort(key=lambda item: (-item[0], item[1].label.lower()))
        return [node for _score, node in ranked]

    def _resolve_contact_candidate(
        self,
        *,
        document: TwinrGraphDocumentV1,
        candidates: list[TwinrGraphNodeV1],
        family_name: str | None,
        role: str | None,
        contact_label: str | None,
        phone: str | None,
        email: str | None,
    ) -> TwinrGraphNodeV1 | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            option = self._contact_option(document, candidates[0])
            candidate = candidates[0]
            requested_contact_label = (contact_label or "").strip().lower()
            exact_contact_match = bool(
                (phone and phone in option.phones)
                or (email and email in option.emails)
            )
            if requested_contact_label:
                normalized_labels = {
                    label.lower()
                    for label in {candidate.label, *candidate.aliases}
                    if label
                }
                if requested_contact_label not in normalized_labels:
                    return None
            if family_name:
                stored_family = _normalize_text(str((candidate.attributes or {}).get("family_name", "")), limit=80).lower()
                family_tokens = set(_tokenize(family_name))
                label_tokens = set(_tokenize(candidate.label))
                if not (
                    exact_contact_match
                    or (stored_family and stored_family == family_name.lower())
                    or (family_tokens and family_tokens <= label_tokens)
                ):
                    return None
            if role and option.role and role.lower() not in option.role.lower():
                return None
            if (phone and option.phones and phone not in option.phones) or (
                email and option.emails and email not in option.emails
            ):
                return None
            return candidate
        if family_name or role or phone or email:
            first = candidates[0]
            second = candidates[1]
            first_score = self._candidate_specificity(
                document,
                first,
                family_name=family_name,
                role=role,
                contact_label=contact_label,
                phone=phone,
                email=email,
            )
            second_score = self._candidate_specificity(
                document,
                second,
                family_name=family_name,
                role=role,
                contact_label=contact_label,
                phone=phone,
                email=email,
            )
            if first_score > second_score:
                return first
        return None

    def _candidate_specificity(
        self,
        document: TwinrGraphDocumentV1,
        node: TwinrGraphNodeV1,
        *,
        family_name: str | None,
        role: str | None,
        contact_label: str | None,
        phone: str | None,
        email: str | None,
    ) -> int:
        score = 0
        labels = {node.label, *node.aliases}
        normalized_labels = {label.lower() for label in labels if label}
        tokens: set[str] = set()
        for label in labels:
            tokens.update(_tokenize(label))
        if family_name and set(_tokenize(family_name)) <= tokens:
            score += 3
        if contact_label and contact_label.lower() in normalized_labels:
            score += 6
        role_detail = (self._contact_role(document, node.node_id) or "").lower()
        if role and role.lower() in role_detail:
            score += 3
        option = self._contact_option(document, node)
        if phone and phone in option.phones:
            score += 4
        if email and email in option.emails:
            score += 4
        return score

    def _contact_options(
        self,
        document: TwinrGraphDocumentV1,
        candidates: list[TwinrGraphNodeV1],
    ) -> tuple[TwinrGraphContactOption, ...]:
        return tuple(self._contact_option(document, node) for node in candidates)

    def _all_contact_nodes(self, document: TwinrGraphDocumentV1) -> list[TwinrGraphNodeV1]:
        return [node for node in document.nodes if node.node_type == "person"]

    def _contact_option(self, document: TwinrGraphDocumentV1, person: TwinrGraphNodeV1) -> TwinrGraphContactOption:
        methods = self._contact_methods(document, person.node_id, canonical=True)
        return TwinrGraphContactOption(
            person_node_id=person.node_id,
            label=person.label,
            role=self._contact_role(document, person.node_id),
            phones=tuple(sorted(method for kind, method in methods if kind == "phone")),
            emails=tuple(sorted(method for kind, method in methods if kind == "email")),
        )

    def _contact_role(self, document: TwinrGraphDocumentV1, person_node_id: str) -> str | None:
        role_info = self._contact_role_info(document, person_node_id)
        role = role_info["role"]
        relation = role_info["relation"]
        if role and relation and relation.lower() != role.lower():
            return f"{role}, {relation}"
        return role or relation or None

    def _contact_role_info(self, document: TwinrGraphDocumentV1, person_node_id: str) -> dict[str, str | None]:
        for edge in document.edges:
            if edge.source_node_id != person_node_id or edge.target_node_id != self.user_node_id:
                continue
            if edge.edge_type != "social_related_to_user":
                continue
            role = str((edge.attributes or {}).get("role", "")).strip()
            relation = str((edge.attributes or {}).get("relation", "")).strip()
            return {
                "role": role or None,
                "relation": relation or None,
            }
        return {"role": None, "relation": None}

    def _contact_methods(
        self,
        document: TwinrGraphDocumentV1,
        person_node_id: str,
        *,
        canonical: bool = False,
    ) -> tuple[tuple[str, str], ...]:
        methods: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for edge in document.edges:
            if edge.source_node_id != person_node_id or edge.edge_type != "general_has_contact_method":
                continue
            target = self._node_by_id(document, edge.target_node_id)
            if target is None:
                continue
            kind = str((edge.attributes or {}).get("kind", target.node_type or "")).strip() or target.node_type
            if canonical:
                value = _normalize_text(str((target.attributes or {}).get("canonical", target.label)), limit=320)
                if kind == "phone":
                    value = _canonical_phone(value)
                elif kind == "email":
                    value = _canonical_email(value)
            else:
                value = _normalize_text(target.label, limit=160)
            if not value:
                continue
            item = (kind, value)
            if item in seen:
                continue
            seen.add(item)
            methods.append(item)
        return tuple(sorted(methods))

    def _contact_conflict_question(self, name: str, options: tuple[TwinrGraphContactOption, ...]) -> str:
        labels: list[str] = []
        for option in options[:3]:
            if option.detail:
                labels.append(f"{option.label} ({option.detail})")
            else:
                labels.append(option.label)
        # AUDIT-FIX(#10): Keep clarification prompts plain and concrete for senior users.
        if not labels:
            return f"I know more than one person named {name}. Which one do you mean?"
        if len(labels) == 1:
            return f"Did you mean {labels[0]}?"
        if len(labels) == 2:
            return f"I know two people named {name}: {labels[0]} or {labels[1]}?"
        return f"I know more than one person named {name}: {', '.join(labels[:-1])}, or {labels[-1]}?"

    def _node_by_id(self, document: TwinrGraphDocumentV1, node_id: str) -> TwinrGraphNodeV1 | None:
        for node in document.nodes:
            if node.node_id == node_id:
                return node
        return None

    def _outgoing_edges(
        self,
        document: TwinrGraphDocumentV1,
        source_node_id: str,
        *,
        edge_types: set[str] | None = None,
    ) -> list[TwinrGraphEdgeV1]:
        edges: list[TwinrGraphEdgeV1] = []
        for edge in document.edges:
            if edge.source_node_id != source_node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            edges.append(edge)
        return edges

    def _incoming_edges(
        self,
        document: TwinrGraphDocumentV1,
        target_node_id: str,
        *,
        edge_types: set[str] | None = None,
    ) -> list[TwinrGraphEdgeV1]:
        edges: list[TwinrGraphEdgeV1] = []
        for edge in document.edges:
            if edge.target_node_id != target_node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            edges.append(edge)
        return edges

    def _related_people_for_plan(
        self,
        document: TwinrGraphDocumentV1,
        plan_node_id: str,
    ) -> list[TwinrGraphNodeV1]:
        people: list[TwinrGraphNodeV1] = []
        seen: set[str] = set()
        for edge in self._outgoing_edges(document, plan_node_id, edge_types={"general_related_to"}):
            person = self._node_by_id(document, edge.target_node_id)
            if person is None or person.node_type != "person" or person.node_id in seen:
                continue
            people.append(person)
            seen.add(person.node_id)
        for edge in self._incoming_edges(document, plan_node_id, edge_types={"general_related_to"}):
            person = self._node_by_id(document, edge.source_node_id)
            if person is None or person.node_type != "person" or person.node_id in seen:
                continue
            people.append(person)
            seen.add(person.node_id)
        return people

    def _related_entities(
        self,
        document: TwinrGraphDocumentV1,
        node_id: str,
    ) -> list[dict[str, str]]:
        related: list[dict[str, str]] = []
        for edge in self._outgoing_edges(document, node_id, edge_types={"general_related_to"}):
            target = self._node_by_id(document, edge.target_node_id)
            if target is None:
                continue
            relation = _normalize_text(str((edge.attributes or {}).get("relation", "")), limit=40) or "related_to"
            related.append(
                {
                    "relation": relation,
                    "label": target.label,
                    "type": target.node_type,
                }
            )
        return related

    def _node_is_near_user(self, document: TwinrGraphDocumentV1, node_id: str) -> bool:
        for edge in self._outgoing_edges(document, node_id, edge_types={"spatial_near"}):
            if edge.target_node_id == self.user_node_id:
                return True
        for edge in self._incoming_edges(document, node_id, edge_types={"spatial_near"}):
            if edge.source_node_id == self.user_node_id:
                return True
        return False

    def _unique_node_id(self, node_type: str, *, base_slug: str, existing_ids: set[str]) -> str:
        base = f"{node_type}:{base_slug}"
        if base not in existing_ids:
            return base
        counter = 2
        while f"{base}_{counter}" in existing_ids:
            counter += 1
        return f"{base}_{counter}"

    def _find_or_create_named_node(
        self,
        nodes: dict[str, TwinrGraphNodeV1],
        *,
        node_type: str,
        label: str,
    ) -> str:
        for node in nodes.values():
            if node.node_type == node_type and node.label.lower() == label.lower():
                return node.node_id
        node_id = self._unique_node_id(node_type, base_slug=_slugify(label, fallback=node_type), existing_ids=set(nodes))
        nodes[node_id] = TwinrGraphNodeV1(node_id=node_id, node_type=node_type, label=label)
        return node_id

    def _find_or_create_graph_ref_node(
        self,
        nodes: dict[str, TwinrGraphNodeV1],
        graph_ref: str,
    ) -> str:
        normalized_ref = collapse_whitespace(str(graph_ref or "")).strip()
        if not normalized_ref:
            raise ValueError("graph_ref is required.")
        if normalized_ref == self.user_node_id:
            self._ensure_user_node(nodes)
            return self.user_node_id
        for node in nodes.values():
            if node.graph_ref == normalized_ref or node.node_id == normalized_ref:
                return node.node_id
        raw_node_type, _, raw_stable_id = normalized_ref.partition(":")
        clean_node_type = _slugify(raw_node_type.strip().lower(), fallback="thing")
        stable_candidate = raw_stable_id.strip().lower()
        if stable_candidate and is_valid_stable_identifier(stable_candidate):
            stable_id = stable_candidate
        else:
            stable_id = _slugify(stable_candidate or raw_stable_id or clean_node_type, fallback=clean_node_type)
        candidate_node_id = f"{clean_node_type}:{stable_id}"
        if candidate_node_id in nodes:
            existing = nodes[candidate_node_id]
            if existing.graph_ref == normalized_ref:
                return candidate_node_id
            digest = hashlib.sha1(normalized_ref.encode("utf-8")).hexdigest()[:8]
            # AUDIT-FIX(#5): Avoid merging distinct graph refs that happen to slug to the same node id.
            candidate_node_id = self._unique_node_id(
                clean_node_type,
                base_slug=f"{stable_id}_{digest}",
                existing_ids=set(nodes),
            )
        label = raw_stable_id.strip().replace("_", " ") or stable_id
        nodes[candidate_node_id] = TwinrGraphNodeV1(
            node_id=candidate_node_id,
            node_type=clean_node_type,
            label=label,
            graph_ref=normalized_ref,
        )
        return candidate_node_id

    def _find_plan_node(
        self,
        nodes: Iterable[TwinrGraphNodeV1],
        *,
        summary: str,
        day_key: str | None,
        when_text: str | None,
    ) -> TwinrGraphNodeV1 | None:
        for node in nodes:
            if node.node_type != "plan":
                continue
            if node.label.lower() != summary.lower():
                continue
            attributes = node.attributes or {}
            if day_key and str(attributes.get("day_key", "")) == day_key:
                return node
            if when_text and str(attributes.get("when_text", "")).lower() == when_text.lower():
                return node
            if not day_key and not when_text:
                return node
        return None

    def _merge_person_label(self, current_label: str, given_name: str, family_name: str | None) -> str:
        if family_name:
            return f"{given_name} {family_name}".strip()
        return current_label.strip() or given_name

    def _contact_method_node_id(self, node_type: str, canonical_value: str, existing_ids: set[str]) -> str:
        base_slug = _slugify(canonical_value, fallback=node_type)
        candidate = f"{node_type}:{base_slug}"
        if candidate not in existing_ids:
            return candidate
        digest = hashlib.sha1(f"{node_type}:{canonical_value}".encode("utf-8")).hexdigest()[:8]
        # AUDIT-FIX(#5): Disambiguate slug collisions instead of forcing unrelated methods onto one node id.
        return self._unique_node_id(node_type, base_slug=f"{base_slug}_{digest}", existing_ids=existing_ids)

    def _upsert_edge(self, edges: list[TwinrGraphEdgeV1], new_edge: TwinrGraphEdgeV1) -> list[TwinrGraphEdgeV1]:
        for index, edge in enumerate(edges):
            if (
                edge.source_node_id == new_edge.source_node_id
                and edge.edge_type == new_edge.edge_type
                and edge.target_node_id == new_edge.target_node_id
            ):
                edges[index] = new_edge
                return edges
        edges.append(new_edge)
        return edges

    def _prune_unreferenced_nodes(
        self,
        nodes: dict[str, TwinrGraphNodeV1],
        edges: list[TwinrGraphEdgeV1],
    ) -> None:
        referenced_node_ids = {self.user_node_id}
        for edge in edges:
            referenced_node_ids.add(edge.source_node_id)
            referenced_node_ids.add(edge.target_node_id)
        removable_node_ids = [
            node_id
            for node_id, node in nodes.items()
            if node_id not in referenced_node_ids
            and node_id != self.user_node_id
            and node.graph_ref is None
        ]
        for node_id in removable_node_ids:
            nodes.pop(node_id, None)

    def _preference_prompt_line(self, edge_type: str, label: str, attributes: Mapping[str, object]) -> str:
        product = _normalize_text(str(attributes.get("for_product", "")), limit=60)
        if edge_type == "user_avoids":
            return f"User tends to avoid {label}."
        if edge_type == "user_engages_with":
            return f"User often engages with {label}."
        if product:
            return f"User prefers {label} for {product}."
        return f"User prefers {label}."

    def _preference_kind(self, edge_type: str) -> str:
        if edge_type == "user_avoids":
            return "avoidance"
        if edge_type == "user_engages_with":
            return "engagement"
        return "preference"

    @contextlib.contextmanager
    def _document_lock(self) -> Iterator[None]:
        with self._document_lock_handle:
            if fcntl is None:
                yield
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            if self._lock_path.parent != self.path.parent:
                with contextlib.suppress(OSError):
                    os.chmod(self._lock_path.parent, 0o1777)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            lock_fd = os.open(self._lock_path, flags, 0o666)
            try:
                with contextlib.suppress(OSError):
                    os.fchmod(lock_fd, 0o666)
                with os.fdopen(lock_fd, "a+b") as lock_handle:
                    lock_fd = -1
                    # AUDIT-FIX(#1): Add an advisory file lock so multiple worker threads/processes cannot interleave writes.
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                if lock_fd >= 0:
                    os.close(lock_fd)

    def _load_local_document_locked(self) -> TwinrGraphDocumentV1 | None:
        for candidate_path in (self.path, self._backup_path):
            payload = self._read_json_file_locked(candidate_path)
            if payload is None:
                continue
            document = self._document_from_payload(payload, source=str(candidate_path))
            if document is not None:
                return document
        return None

    def _load_authoritative_document_locked(self) -> TwinrGraphDocumentV1:
        """Load the graph from the remote current view first, then local compatibility state."""

        remote_document = self._load_remote_document()
        if remote_document is not None:
            self._purge_local_document_cache_locked()
            return remote_document
        local_document = self._load_local_document_locked()
        if local_document is not None:
            return local_document
        return self._empty_document()

    def _load_remote_document(self) -> TwinrGraphDocumentV1 | None:
        if not self._remote_graph.enabled():
            return None
        try:
            return self._remote_graph.load_document()
        except Exception as exc:
            if isinstance(exc, _remote_unavailable_error_type()) and self.remote_state is not None and self.remote_state.required:
                raise
            # AUDIT-FIX(#2): Optional remote graph state failures must not break local-only graph memory.
            logger.warning("Failed to load remote graph current view; continuing with local-only state.", exc_info=True)
            return None

    def _document_from_payload(self, payload: object, *, source: str) -> TwinrGraphDocumentV1 | None:
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            logger.warning("Ignoring non-mapping graph payload from %s.", source)
            return None
        try:
            return TwinrGraphDocumentV1.from_payload(payload)
        except ValueError as exc:
            if str(exc).startswith("Unsupported Twinr graph schema"):
                raise
            logger.warning("Ignoring invalid graph payload from %s.", source, exc_info=True)
            return None
        except Exception:
            # AUDIT-FIX(#2): Corrupt state should be quarantined and bypassed instead of crashing the assistant.
            logger.warning("Ignoring invalid graph payload from %s.", source, exc_info=True)
            return None

    def _read_json_file_locked(self, path: Path) -> Mapping[str, object] | None:
        try:
            raw_bytes = self._read_bytes_no_symlink(path)
        except FileNotFoundError:
            return None
        except OSError:
            # AUDIT-FIX(#1): Refuse unsafe or unreadable state files and fall back cleanly.
            logger.warning("Failed to read graph file %s safely.", path, exc_info=True)
            return None
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Failed to decode graph file %s.", path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            logger.warning("Ignoring graph file %s because it does not contain a JSON object.", path)
            return None
        return payload

    def _save_document_locked(
        self,
        nodes: Mapping[str, TwinrGraphNodeV1],
        edges: list[TwinrGraphEdgeV1],
        *,
        created_at: str,
    ) -> None:
        updated_at = _utc_now_iso()
        document = TwinrGraphDocumentV1(
            subject_node_id=self.user_node_id,
            graph_id="graph:user_main",
            created_at=created_at,
            updated_at=updated_at,
            nodes=tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
            edges=tuple(
                sorted(
                    edges,
                    key=lambda item: (item.source_node_id, item.edge_type, item.target_node_id),
                )
            ),
            metadata={"kind": "personal_graph"},
        )
        if self._remote_graph.enabled():
            try:
                self._remote_graph.persist_document(document=document)
                self._purge_local_document_cache_locked()
                return
            except Exception as exc:
                if isinstance(exc, _remote_unavailable_error_type()):
                    if self.remote_state is not None and self.remote_state.required:
                        raise
                    logger.warning("Failed to save remote graph current view; falling back to local graph cache.", exc_info=True)
                    self._write_document_locked(document)
                    return
                if self.remote_state is not None and self.remote_state.required:
                    raise _remote_unavailable_error_type()("Failed to save required remote graph current view.") from exc
                logger.warning("Failed to save remote graph current view; falling back to local graph cache.", exc_info=True)
        self._write_document_locked(document)

    def _write_document_locked(self, document: TwinrGraphDocumentV1) -> None:
        payload_text = json.dumps(document.to_payload(), ensure_ascii=False, indent=2) + "\n"
        # AUDIT-FIX(#1): Write through a temp file plus atomic replace so power loss cannot leave a torn JSON document behind.
        self._write_text_atomic_locked(self.path, payload_text)
        # AUDIT-FIX(#2): Keep a last-known-good backup for recovery from partial corruption on the primary file.
        self._write_text_atomic_locked(self._backup_path, payload_text)

    def _write_text_atomic_locked(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            self._fsync_directory(path.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()

    def _read_bytes_no_symlink(self, path: Path) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        handle = os.fdopen(fd, "rb")
        try:
            return handle.read()
        finally:
            handle.close()

    def _fsync_directory(self, directory: Path) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        try:
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _purge_local_document_cache_locked(self) -> None:
        """Remove stale local graph mirrors once the remote current view is authoritative."""

        removed_any = False
        for candidate in (self.path, self._backup_path):
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("Failed to remove stale local graph cache %s.", candidate, exc_info=True)
            else:
                removed_any = True
        if removed_any:
            self._fsync_directory(self.path.parent)

    def _query_requests_contact_methods(self, query_text: str | None) -> bool:
        tokens = set(_tokenize(query_text or ""))
        return bool(tokens & _CONTACT_METHOD_QUERY_TOKENS)

    def _ensure_contact_method_node(
        self,
        nodes: dict[str, TwinrGraphNodeV1],
        *,
        node_type: str,
        label: str,
        canonical_value: str,
    ) -> str:
        for node in nodes.values():
            if node.node_type != node_type:
                continue
            existing_canonical = _normalize_text(str((node.attributes or {}).get("canonical", "")), limit=320)
            if existing_canonical == canonical_value:
                return node.node_id
        node_id = self._contact_method_node_id(node_type, canonical_value, set(nodes))
        nodes[node_id] = TwinrGraphNodeV1(
            node_id=node_id,
            node_type=node_type,
            label=label,
            attributes={"canonical": canonical_value},
        )
        return node_id
