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
from typing import Iterable, Iterator, Mapping

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.chonkydb.schema import (
    TwinrGraphDocumentV1,
    TwinrGraphEdgeV1,
    TwinrGraphNodeV1,
)
from twinr.memory.longterm.core.models import LongTermGraphEdgeCandidateV1
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore
from twinr.temporal import parse_local_date_text
from twinr.text_utils import collapse_whitespace, is_valid_stable_identifier, retrieval_terms, slugify_identifier

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


class TwinrPersonalGraphStore:
    """Manage the persisted Twinr personal graph and prompt-memory extracts."""

    def __init__(
        self,
        path: str | Path,
        *,
        user_node_id: str = "user:main",
        user_label: str = "Main user",
        timezone_name: str = "Europe/Berlin",
        remote_state: LongTermRemoteStateStore | None = None,
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

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrPersonalGraphStore":
        """Build a graph store from the active Twinr runtime configuration."""

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
            local_document = self._load_local_document_locked()
            if local_document is not None:
                return local_document
            remote_document = self._load_remote_document()
            if remote_document is not None:
                # AUDIT-FIX(#2): Keep a local durable copy so state survives intermittent network or remote outages.
                self._write_document_locked(remote_document)
                return remote_document
            return self._empty_document()

    def ensure_remote_snapshot(self) -> bool:
        """Seed remote state with the current graph if no snapshot exists."""

        if self.remote_state is None or not self.remote_state.enabled:
            return False
        with self._document_lock():
            remote_document = self._load_remote_document()
            if remote_document is not None:
                return False
            document = self._load_local_document_locked() or self._empty_document()
            self.remote_state.save_snapshot(snapshot_kind="graph", payload=document.to_payload())
            return True

    def apply_candidate_edges(
        self,
        graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...] | list[LongTermGraphEdgeCandidateV1],
    ) -> None:
        """Merge extracted long-term-memory edge candidates into the graph."""

        if not graph_edges:
            return
        with self._document_lock():
            document = self._load_local_document_locked() or self._load_remote_document() or self._empty_document()
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
            document = self._load_local_document_locked() or self._load_remote_document() or self._empty_document()
            nodes = {node.node_id: node for node in document.nodes}
            edges = list(document.edges)
            self._ensure_user_node(nodes)
            candidates = self._contact_candidates(
                document=document,
                given_name=clean_given,
                family_name=clean_family,
                role=match_role,
                phone=clean_phone or None,
                email=clean_email or None,
            )
            if candidates:
                unique = self._resolve_contact_candidate(
                    document=document,
                    candidates=candidates,
                    family_name=clean_family,
                    role=match_role,
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
    ) -> TwinrGraphLookupResult:
        """Find a remembered contact and request clarification when needed."""

        clean_name = _normalize_text(name, limit=80)
        clean_family = _normalize_text(family_name or "", limit=80) or None
        clean_role = _normalize_text(role or "", limit=80) or None
        if not clean_name:
            raise ValueError("name is required.")
        document = self.load_document()
        candidates = self._contact_candidates(
            document=document,
            given_name=clean_name,
            family_name=clean_family,
            role=clean_role,
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
            document = self._load_local_document_locked() or self._load_remote_document() or self._empty_document()
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
            document = self._load_local_document_locked() or self._load_remote_document() or self._empty_document()
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

    def build_prompt_context(
        self,
        query_text: str | None,
        *,
        include_contact_methods: bool = True,
    ) -> str | None:
        """Build the serialized prompt-memory context for one user turn."""

        try:
            document = self.load_document()
        except Exception:
            logger.warning("Failed to load graph memory prompt context.", exc_info=True)
            return None
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
            document = self.load_document()
        except Exception:
            logger.warning("Failed to load graph memory subtext payload.", exc_info=True)
            return None
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
        return payload or None

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

    def _save_document(
        self,
        nodes: Mapping[str, TwinrGraphNodeV1],
        edges: list[TwinrGraphEdgeV1],
        *,
        created_at: str | None = None,
    ) -> None:
        with self._document_lock():
            # AUDIT-FIX(#1): Keep the public save path safe even if a new caller invokes it directly.
            effective_created_at = created_at or (
                self._load_local_document_locked() or self._load_remote_document() or self._empty_document()
            ).created_at
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
        phone: str | None,
        email: str | None,
    ) -> list[TwinrGraphNodeV1]:
        full_label = self._merge_person_label("", given_name, family_name)
        given_tokens = set(_tokenize(given_name))
        family_tokens = set(_tokenize(family_name or ""))
        requested_role = (role or "").strip().lower()
        ranked: list[tuple[int, TwinrGraphNodeV1]] = []
        for node in document.nodes:
            if node.node_type != "person":
                continue
            stored_family = _normalize_text(str((node.attributes or {}).get("family_name", "")), limit=80) or None
            labels = {node.label, *node.aliases}
            label_tokens: set[str] = set()
            for label in labels:
                label_tokens.update(_tokenize(label))
            option = self._contact_option(document, node)
            exact_contact_match = bool(
                (phone and phone in option.phones)
                or (email and email in option.emails)
            )
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
        phone: str | None,
        email: str | None,
    ) -> TwinrGraphNodeV1 | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            option = self._contact_option(document, candidates[0])
            candidate = candidates[0]
            exact_contact_match = bool(
                (phone and phone in option.phones)
                or (email and email in option.emails)
            )
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
            first_score = self._candidate_specificity(document, first, family_name=family_name, role=role, phone=phone, email=email)
            second_score = self._candidate_specificity(document, second, family_name=family_name, role=role, phone=phone, email=email)
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
        phone: str | None,
        email: str | None,
    ) -> int:
        score = 0
        tokens = set(_tokenize(node.label))
        if family_name and set(_tokenize(family_name)) <= tokens:
            score += 3
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

    def _load_remote_document(self) -> TwinrGraphDocumentV1 | None:
        if self.remote_state is None or not self.remote_state.enabled:
            return None
        try:
            payload = self.remote_state.load_snapshot(
                snapshot_kind="graph",
                local_path=self.path,
                prefer_cached_document_id=True,
            )
        except LongTermRemoteUnavailableError:
            if self.remote_state.required:
                raise
            logger.warning("Failed to load remote graph snapshot; continuing with local-only state.", exc_info=True)
            return None
        except Exception:
            # AUDIT-FIX(#2): Remote state is optional; network failures must not break local graph memory.
            logger.warning("Failed to load remote graph snapshot; continuing with local-only state.", exc_info=True)
            return None
        return self._document_from_payload(payload, source="remote_graph_snapshot")

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
        self._write_document_locked(document)
        payload = document.to_payload()
        if self.remote_state is not None and self.remote_state.enabled:
            try:
                self.remote_state.save_snapshot(snapshot_kind="graph", payload=payload)
            except Exception:
                # AUDIT-FIX(#2): Remote sync failures should be observable in logs but must not drop the local durable write.
                logger.warning("Failed to save remote graph snapshot; local graph state is still durable.", exc_info=True)

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
