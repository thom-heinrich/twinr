from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Mapping
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.query_normalization import tokenize_retrieval_text
from twinr.memory.chonkydb.schema import (
    TwinrGraphDocumentV1,
    TwinrGraphEdgeV1,
    TwinrGraphNodeV1,
)

_DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")
_CONTACT_QUERY_TOKENS = frozenset({"nummer", "telefon", "anrufen", "kontakt", "email", "mail"})
_SHOPPING_QUERY_TOKENS = frozenset({"kaufen", "kauf", "laden", "geschaeft", "geschäft", "shop", "wo", "bekomme"})
_PLAN_CONTEXT_TOKENS = frozenset({"wetter", "plan", "vorhaben", "machen", "wollte", "will", "noch", "heute", "today"})
_TODAY_QUERY_TOKENS = frozenset({"heute", "today"})
_TOMORROW_QUERY_TOKENS = frozenset({"morgen", "tomorrow"})


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


def _tokenize(value: str) -> tuple[str, ...]:
    return tokenize_retrieval_text(value)


def _canonical_phone(value: str) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit() or ch == "+")
    return digits


def _canonical_email(value: str) -> str:
    return str(value or "").strip().lower()


def _now_in_timezone(timezone_name: str) -> datetime:
    return datetime.now(ZoneInfo(timezone_name))


def _infer_day_key(when_text: str | None, *, timezone_name: str) -> str | None:
    text = (when_text or "").strip().lower()
    if not text:
        return None
    now = _now_in_timezone(timezone_name)
    if text in {"today", "heute"}:
        return now.date().isoformat()
    if text in {"tomorrow", "morgen"}:
        return (now.date() + timedelta(days=1)).isoformat()
    match = _DATE_RE.search(text)
    if match is not None:
        return match.group("date")
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class TwinrGraphContactOption:
    person_node_id: str
    label: str
    role: str | None = None
    phones: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()

    @property
    def detail(self) -> str:
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
    status: str
    match: TwinrGraphContactOption | None = None
    options: tuple[TwinrGraphContactOption, ...] = ()
    question: str | None = None


@dataclass(frozen=True, slots=True)
class TwinrGraphWriteResult:
    status: str
    label: str
    node_id: str
    edge_type: str | None = None
    question: str | None = None
    options: tuple[TwinrGraphContactOption, ...] = ()


class TwinrPersonalGraphStore:
    def __init__(
        self,
        path: str | Path,
        *,
        user_node_id: str = "user:main",
        user_label: str = "Main user",
        timezone_name: str = "Europe/Berlin",
    ) -> None:
        self.path = Path(path)
        self.user_node_id = user_node_id
        self.user_label = _normalize_text(user_label, limit=80) or "Main user"
        self.timezone_name = timezone_name

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrPersonalGraphStore":
        base = chonkydb_data_path(config)
        path = base / "twinr_graph_v1.json"
        return cls(
            path=path,
            user_label=config.user_display_name or "Main user",
            timezone_name=config.local_timezone_name,
        )

    def load_document(self) -> TwinrGraphDocumentV1:
        if not self.path.exists():
            return self._empty_document()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Personal graph file must contain a JSON object.")
        return TwinrGraphDocumentV1.from_payload(payload)

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
        clean_given = _normalize_text(given_name, limit=80)
        clean_family = _normalize_text(family_name or "", limit=80) or None
        clean_role = _normalize_text(role or "", limit=80) or None
        clean_relation = _normalize_text(relation or "", limit=80) or None
        clean_phone = _canonical_phone(phone or "")
        clean_email = _canonical_email(email or "")
        clean_notes = _normalize_text(notes or "", limit=160) or None
        if not clean_given:
            raise ValueError("given_name is required.")

        document = self.load_document()
        nodes = {node.node_id: node for node in document.nodes}
        edges = list(document.edges)
        self._ensure_user_node(nodes)
        candidates = self._contact_candidates(
            document=document,
            given_name=clean_given,
            family_name=clean_family,
            role=clean_role,
            phone=clean_phone or None,
            email=clean_email or None,
        )
        if candidates:
            unique = self._resolve_contact_candidate(
                document=document,
                candidates=candidates,
                family_name=clean_family,
                role=clean_role,
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
            aliases = tuple(alias for alias in {clean_given} if alias != label)
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
                edge_type="social_supports_user_as",
                target_node_id=self.user_node_id,
                confirmed_by_user=confirmed_by_user,
                attributes={
                    "role": clean_role or clean_relation or "known_contact",
                    "relation": clean_relation or "",
                },
            )
            edges = self._upsert_edge(edges, edge)

        if clean_phone:
            phone_label = phone.strip() if phone and phone.strip() else clean_phone
            phone_node_id = self._contact_method_node_id("phone", clean_phone, set(nodes))
            nodes.setdefault(
                phone_node_id,
                TwinrGraphNodeV1(
                    node_id=phone_node_id,
                    node_type="phone",
                    label=phone_label,
                    attributes={"canonical": clean_phone},
                ),
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
            email_node_id = self._contact_method_node_id("email", clean_email, set(nodes))
            nodes.setdefault(
                email_node_id,
                TwinrGraphNodeV1(
                    node_id=email_node_id,
                    node_type="email",
                    label=clean_email,
                    attributes={"canonical": clean_email},
                ),
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

        self._save_document(nodes, edges)
        return TwinrGraphWriteResult(status=status, label=label, node_id=person_node_id, edge_type="social_supports_user_as")

    def lookup_contact(
        self,
        *,
        name: str,
        family_name: str | None = None,
        role: str | None = None,
    ) -> TwinrGraphLookupResult:
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
        clean_category = _slugify(category, fallback="thing")
        clean_value = _normalize_text(value, limit=100)
        clean_product = _normalize_text(for_product or "", limit=80) or None
        clean_details = _normalize_text(details or "", limit=140) or None
        if not clean_value:
            raise ValueError("value is required.")
        document = self.load_document()
        nodes = {node.node_id: node for node in document.nodes}
        edges = list(document.edges)
        self._ensure_user_node(nodes)

        node_type = "thing"
        edge_type = "user_likes"
        if clean_category == "brand":
            node_type = "brand"
            edge_type = "user_prefers_brand" if sentiment != "dislike" else "user_dislikes"
        elif clean_category in {"store", "shop"}:
            node_type = "place"
            edge_type = "user_usually_buys_at" if sentiment != "dislike" else "user_dislikes"
        elif sentiment == "dislike":
            node_type = clean_category if clean_category in {"food", "drink", "activity", "music"} else "thing"
            edge_type = "user_dislikes"
        else:
            node_type = clean_category if clean_category in {"food", "drink", "activity", "music"} else "thing"

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
        self._save_document(nodes, edges)
        return TwinrGraphWriteResult(status="updated", label=clean_value, node_id=node_id, edge_type=edge_type)

    def remember_plan(
        self,
        *,
        summary: str,
        when_text: str | None = None,
        details: str | None = None,
        confirmed_by_user: bool = True,
    ) -> TwinrGraphWriteResult:
        clean_summary = _normalize_text(summary, limit=120)
        clean_when = _normalize_text(when_text or "", limit=80) or None
        clean_details = _normalize_text(details or "", limit=140) or None
        if not clean_summary:
            raise ValueError("summary is required.")
        day_key = _infer_day_key(clean_when, timezone_name=self.timezone_name)
        document = self.load_document()
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

        self._save_document(nodes, edges)
        return TwinrGraphWriteResult(status=status, label=clean_summary, node_id=plan_node_id, edge_type="user_plans")

    def build_prompt_context(self, query_text: str | None) -> str | None:
        document = self.load_document()
        query_tokens = set(_tokenize(query_text or ""))
        contacts = self._rank_contact_prompt_items(
            self._prompt_contacts(document, limit=128),
            query_tokens=query_tokens,
        )
        preferences = self._rank_preference_prompt_items(
            self._prompt_preferences(document, limit=128),
            query_tokens=query_tokens,
        )
        plans = self._rank_plan_prompt_items(
            self._prompt_plans(document, limit=128),
            query_tokens=query_tokens,
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
            "Treat phone numbers, email addresses, and other contact methods as reference-only unless the user explicitly asks for them or clearly asks Twinr to place a call or send a message.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def build_subtext_payload(self, query_text: str | None) -> dict[str, object] | None:
        clean_query = _normalize_text(query_text or "", limit=220)
        query_tokens = set(_tokenize(clean_query))
        if not query_tokens:
            return None
        document = self.load_document()
        preference_items = self._rank_preference_prompt_items(
            self._prompt_preferences(document, limit=128),
            query_tokens=query_tokens,
            limit=3,
            fallback_limit=0,
        )
        plan_items = self._rank_plan_prompt_items(
            self._prompt_plans(document, limit=128),
            query_tokens=query_tokens,
            limit=3,
            fallback_limit=0,
        )
        contact_items = self._rank_contact_prompt_items(
            self._prompt_contacts(document, limit=128),
            query_tokens=query_tokens,
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
        query_tokens: set[str],
        limit: int = 8,
        fallback_limit: int = 3,
    ) -> list[dict[str, object]]:
        return self._rank_prompt_items(
            items,
            query_tokens=query_tokens,
            weighted_fields=(
                ("name", 4),
                ("aliases", 3),
                ("role", 3),
                ("relation", 3),
                ("phones", 1),
                ("emails", 1),
                ("notes", 2),
            ),
            limit=limit,
            fallback_limit=fallback_limit,
        )

    def _rank_preference_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_tokens: set[str],
        limit: int = 8,
        fallback_limit: int = 3,
    ) -> list[dict[str, object]]:
        return self._rank_prompt_items(
            items,
            query_tokens=query_tokens,
            weighted_fields=(
                ("type", 2),
                ("value", 4),
                ("category", 2),
                ("for_product", 4),
                ("details", 2),
                ("carries_brands", 3),
            ),
            limit=limit,
            fallback_limit=fallback_limit,
        )

    def _rank_plan_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_tokens: set[str],
        limit: int = 8,
        fallback_limit: int = 3,
    ) -> list[dict[str, object]]:
        return self._rank_prompt_items(
            items,
            query_tokens=query_tokens,
            weighted_fields=(
                ("summary", 4),
                ("when", 3),
                ("date", 3),
                ("details", 2),
                ("related_people", 3),
            ),
            limit=limit,
            fallback_limit=fallback_limit,
        )

    def _rank_prompt_items(
        self,
        items: list[dict[str, object]],
        *,
        query_tokens: set[str],
        weighted_fields: tuple[tuple[str, int], ...],
        limit: int,
        fallback_limit: int,
    ) -> list[dict[str, object]]:
        if not items:
            return []
        if not query_tokens:
            return items[:limit]
        scored: list[tuple[int, int, dict[str, object]]] = []
        for index, item in enumerate(items):
            score = 0
            for field_name, weight in weighted_fields:
                value = item.get(field_name)
                tokens = self._prompt_item_tokens(value)
                overlap = len(query_tokens & tokens)
                if overlap > 0:
                    score += overlap * weight
            if score > 0:
                scored.append((score, index, item))
        if not scored:
            if fallback_limit <= 0:
                return []
            return items[:fallback_limit]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _score, _index, item in scored[:limit]]

    def _subtext_preferences(self, items: list[dict[str, object]]) -> list[dict[str, str]]:
        cues: list[dict[str, str]] = []
        for item in items:
            value = _normalize_text(str(item.get("value", "")), limit=100)
            if not value:
                continue
            kind = str(item.get("type", "")).strip()
            product = _normalize_text(str(item.get("for_product", "")), limit=80)
            category = _normalize_text(str(item.get("category", "")), limit=40)
            if kind == "preferred_brand":
                if product:
                    guidance = (
                        f"When helping with {product}, quietly bias suggestions toward {value} if it still fits the user's request."
                    )
                else:
                    guidance = f"Let familiarity with {value} gently bias recommendations when relevant."
            elif kind == "usual_store":
                guidance = f"When discussing where to buy something, treat {value} as a familiar option if it fits."
            elif kind == "disliked_item":
                guidance = f"Avoid steering the user toward {value} unless they explicitly ask for it."
            else:
                guidance = f"Let positive familiarity with {value} subtly influence suggestions when relevant."
            cue = {"kind": kind or "preference", "value": value, "guidance": guidance}
            if product:
                cue["topic"] = product
            elif category:
                cue["topic"] = category
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
                    "Use that role to shape urgency, practical reasons to contact them, and next steps. "
                    "If the advice depends on that role, it is fine to name the role naturally."
                )
            else:
                guidance = f"When this person is relevant, treat {name} as an already-known person in the user's life."
            cue = {"person": name, "guidance": guidance}
            if role:
                cue["role"] = role
            cues.append(cue)
        return cues

    def _prompt_item_tokens(self, value: object) -> set[str]:
        if isinstance(value, (list, tuple)):
            tokens: set[str] = set()
            for item in value:
                tokens.update(_tokenize(str(item)))
            return tokens
        return set(_tokenize(str(value or "")))

    def _prompt_contacts(self, document: TwinrGraphDocumentV1, *, limit: int = 12) -> list[dict[str, object]]:
        contacts: list[dict[str, object]] = []
        for person in sorted(self._all_contact_nodes(document), key=lambda item: item.label.lower()):
            methods = self._contact_methods(document, person.node_id)
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
            if methods:
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
                edge_types={"user_prefers_brand", "user_likes", "user_dislikes", "user_usually_buys_at"},
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
            if target.node_type == "place":
                item["nearby"] = self._node_is_near_user(document, target.node_id)
                brands = sorted(
                    target_brand.label
                    for target_brand in (
                        self._node_by_id(document, carry_edge.target_node_id)
                        for carry_edge in self._outgoing_edges(
                            document,
                            target.node_id,
                            edge_types={"general_carries_brand"},
                        )
                    )
                    if target_brand is not None
                )
                if brands:
                    item["carries_brands"] = brands
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

    def _collect_temporal_multihop_hints(
        self,
        *,
        document: TwinrGraphDocumentV1,
        query_text: str,
        query_tokens: set[str],
        today_key: str,
    ) -> tuple[list[tuple[int, str]], set[str]]:
        relevant_day_keys = self._query_day_keys(query_text, query_tokens=query_tokens, today_key=today_key)
        tomorrow_key = (_now_in_timezone(self.timezone_name).date() + timedelta(days=1)).isoformat()
        query_mentions_contact = bool(query_tokens & _CONTACT_QUERY_TOKENS)
        scored: list[tuple[int, str]] = []
        preferred_contact_ids: set[str] = set()

        for edge in self._outgoing_edges(document, self.user_node_id, edge_types={"user_plans"}):
            plan = self._node_by_id(document, edge.target_node_id)
            if plan is None:
                continue
            when_text = str((edge.attributes or {}).get("when_text", "") or "")
            day_key = str((edge.attributes or {}).get("day_key", "") or "")
            plan_tokens = set(_tokenize(plan.label))
            if when_text:
                plan_tokens.update(_tokenize(when_text))
            score = len(query_tokens & plan_tokens)
            if relevant_day_keys and day_key in relevant_day_keys:
                score += 4
            if query_tokens & {"termin", "treffen", "plan", "kalender", "physiotherapie", "arzt"} and (
                day_key in relevant_day_keys or score > 0
            ):
                score += 1
            if score <= 0:
                continue
            people = self._related_people_for_plan(document, plan.node_id)
            if not people:
                continue
            for person in people:
                option = self._contact_option(document, person)
                preferred_contact_ids.add(person.node_id)
                person_tokens = set(_tokenize(option.label))
                if option.role:
                    person_tokens.update(_tokenize(option.role))
                person_score = score + max(1, len(query_tokens & person_tokens))
                if query_mentions_contact and (option.phones or option.emails):
                    person_score += 2
                time_phrase = self._time_phrase(day_key=day_key, when_text=when_text, today_key=today_key, tomorrow_key=tomorrow_key)
                if query_mentions_contact:
                    line = f"Passender Kontakt zum Plan {time_phrase}: {option.label}"
                    if option.detail:
                        line += f" ({option.detail})"
                else:
                    line = f"Plan {time_phrase}: {plan.label} mit {option.label}"
                    if option.role:
                        line += f" ({option.role})"
                scored.append((person_score, line))
        return scored, preferred_contact_ids

    def _collect_store_multihop_hints(
        self,
        *,
        document: TwinrGraphDocumentV1,
        query_tokens: set[str],
    ) -> list[tuple[int, str]]:
        if not query_tokens & _SHOPPING_QUERY_TOKENS:
            return []
        preferred_brand_edges = {
            edge.target_node_id: edge
            for edge in self._outgoing_edges(document, self.user_node_id, edge_types={"user_prefers_brand"})
        }
        scored: list[tuple[int, str]] = []
        for edge in self._outgoing_edges(document, self.user_node_id, edge_types={"user_usually_buys_at"}):
            store = self._node_by_id(document, edge.target_node_id)
            if store is None or store.node_type != "place":
                continue
            product = _normalize_text(str((edge.attributes or {}).get("for_product", "")), limit=60)
            carry_labels: list[str] = []
            for carry_edge in self._outgoing_edges(document, store.node_id, edge_types={"general_carries_brand"}):
                if carry_edge.target_node_id not in preferred_brand_edges:
                    continue
                brand = self._node_by_id(document, carry_edge.target_node_id)
                if brand is not None:
                    carry_labels.append(brand.label)
            near_user = self._node_is_near_user(document, store.node_id)
            score = 2
            if product and set(_tokenize(product)) & query_tokens:
                score += 2
            if carry_labels:
                score += 3
            if near_user:
                score += 2
            if score <= 2:
                continue
            line = f"Persoenlicher Einkaufshinweis: {store.label}"
            if carry_labels:
                line += f" fuehrt wahrscheinlich {', '.join(sorted(set(carry_labels)))}"
                if product:
                    line += f" fuer {product}"
            elif product:
                line += f" ist ein passender Ort fuer {product}"
            if near_user:
                line += " und ist in der Naehe."
            else:
                line += "."
            scored.append((score, line))
        return scored

    def _collect_contact_hints(
        self,
        *,
        document: TwinrGraphDocumentV1,
        query_tokens: set[str],
        preferred_person_node_ids: set[str],
    ) -> list[tuple[int, str]]:
        scored: list[tuple[int, str]] = []
        for option in self._contact_options(document, self._all_contact_nodes(document)):
            if preferred_person_node_ids and option.person_node_id not in preferred_person_node_ids:
                continue
            relevance_tokens = set(_tokenize(option.label))
            if option.detail:
                relevance_tokens.update(_tokenize(option.detail))
            overlap = len(query_tokens & relevance_tokens)
            if overlap <= 0:
                continue
            score = overlap + 1
            if preferred_person_node_ids:
                score += 2
            line = f"Bekannter Kontakt: {option.label}"
            if option.detail:
                line += f" ({option.detail})"
            scored.append((score, line))
        return scored

    def _empty_document(self) -> TwinrGraphDocumentV1:
        user = TwinrGraphNodeV1(node_id=self.user_node_id, node_type="user", label=self.user_label)
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return TwinrGraphDocumentV1(
            subject_node_id=self.user_node_id,
            graph_id="graph:user_main",
            created_at=now,
            updated_at=now,
            nodes=(user,),
            edges=(),
            metadata={"kind": "personal_graph"},
        )

    def _save_document(self, nodes: Mapping[str, TwinrGraphNodeV1], edges: list[TwinrGraphEdgeV1]) -> None:
        existing = self.load_document() if self.path.exists() else self._empty_document()
        created_at = existing.created_at
        updated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        document = TwinrGraphDocumentV1(
            subject_node_id=self.user_node_id,
            graph_id="graph:user_main",
            created_at=created_at,
            updated_at=updated_at,
            nodes=tuple(nodes.values()),
            edges=tuple(edges),
            metadata={"kind": "personal_graph"},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(document.to_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
        ranked: list[tuple[int, TwinrGraphNodeV1]] = []
        for node in document.nodes:
            if node.node_type != "person":
                continue
            stored_family = _normalize_text(str((node.attributes or {}).get("family_name", "")), limit=80) or None
            if family_name and stored_family and stored_family.lower() != family_name.lower():
                continue
            score = 0
            labels = {node.label, *node.aliases}
            label_tokens = set()
            for label in labels:
                label_tokens.update(_tokenize(label))
            if full_label and full_label.lower() == node.label.lower():
                score += 8
            if given_tokens and given_tokens <= label_tokens:
                score += 3
            if family_tokens and family_tokens <= label_tokens:
                score += 3
            role_detail = (self._contact_role(document, node.node_id) or "").lower()
            if role and role_detail and role.lower() not in role_detail:
                continue
            if role and role.lower() and role.lower() in role_detail:
                score += 4
            option = self._contact_option(document, node)
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
            if not any((family_name, role)) and (option.phones or option.emails or option.role):
                if (phone and phone not in option.phones) or (email and email not in option.emails):
                    return None
            return candidates[0]
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
        methods = self._contact_methods(document, person.node_id)
        return TwinrGraphContactOption(
            person_node_id=person.node_id,
            label=person.label,
            role=self._contact_role(document, person.node_id),
            phones=tuple(method for kind, method in methods if kind == "phone"),
            emails=tuple(method for kind, method in methods if kind == "email"),
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
            if edge.edge_type != "social_supports_user_as":
                continue
            role = str((edge.attributes or {}).get("role", "")).strip()
            relation = str((edge.attributes or {}).get("relation", "")).strip()
            return {
                "role": role or None,
                "relation": relation or None,
            }
        return {"role": None, "relation": None}

    def _contact_methods(self, document: TwinrGraphDocumentV1, person_node_id: str) -> tuple[tuple[str, str], ...]:
        methods: list[tuple[str, str]] = []
        for edge in document.edges:
            if edge.source_node_id != person_node_id or edge.edge_type != "general_has_contact_method":
                continue
            target = self._node_by_id(document, edge.target_node_id)
            if target is None:
                continue
            kind = str((edge.attributes or {}).get("kind", target.node_type or "")).strip() or target.node_type
            methods.append((kind, target.label))
        return tuple(methods)

    def _contact_conflict_question(self, name: str, options: tuple[TwinrGraphContactOption, ...]) -> str:
        labels = []
        for option in options[:3]:
            if option.detail:
                labels.append(f"{option.label} ({option.detail})")
            else:
                labels.append(option.label)
        if not labels:
            return f"I know multiple entries for {name}. Which one do you mean?"
        if len(labels) == 1:
            return f"Do you mean {labels[0]}?"
        if len(labels) == 2:
            return f"I know two contacts named {name}: {labels[0]} or {labels[1]}?"
        return f"I know multiple contacts named {name}: {', '.join(labels[:-1])}, or {labels[-1]}?"

    def _query_day_keys(self, query_text: str, *, query_tokens: set[str], today_key: str) -> set[str]:
        day_keys: set[str] = set()
        if query_tokens & _TODAY_QUERY_TOKENS:
            day_keys.add(today_key)
        if query_tokens & _TOMORROW_QUERY_TOKENS:
            day_keys.add((_now_in_timezone(self.timezone_name).date() + timedelta(days=1)).isoformat())
        for match in _DATE_RE.finditer(query_text):
            day_keys.add(match.group("date"))
        return day_keys

    def _time_phrase(self, *, day_key: str, when_text: str, today_key: str, tomorrow_key: str) -> str:
        if day_key == today_key:
            return "heute"
        if day_key == tomorrow_key:
            return "morgen"
        if when_text:
            return when_text
        if day_key:
            return day_key
        return "zeitlich passend"

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

    def _find_plan_node(
        self,
        nodes,
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
        return candidate

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
        if edge_type == "user_prefers_brand":
            if product:
                return f"User strongly prefers {label} for {product}."
            return f"User strongly prefers {label}."
        if edge_type == "user_usually_buys_at":
            return f"User often buys at {label}."
        if edge_type == "user_dislikes":
            return f"User tends to dislike {label}."
        return f"User likes {label}."

    def _preference_kind(self, edge_type: str) -> str:
        if edge_type == "user_prefers_brand":
            return "preferred_brand"
        if edge_type == "user_usually_buys_at":
            return "usual_store"
        if edge_type == "user_dislikes":
            return "disliked_item"
        return "liked_item"
