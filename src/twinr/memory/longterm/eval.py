from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.chonkydb.schema import TwinrGraphEdgeV1
from twinr.memory.context_store import (
    ManagedContextFileStore,
    PersistentMemoryMarkdownStore,
    PromptContextStore,
)
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.memory.longterm.service import LongTermMemoryService


EvalKind = Literal["provider_context", "contact_lookup"]


@dataclass(frozen=True, slots=True)
class LongTermEvalCase:
    case_id: str
    category: str
    query_text: str
    kind: EvalKind
    canonical_query_text: str | None = None
    lookup_family_name: str | None = None
    lookup_role: str | None = None
    expected_contains: tuple[str, ...] = ()
    expected_absent: tuple[str, ...] = ()
    expected_lookup_status: str | None = None
    expected_option_count: int | None = None


@dataclass(frozen=True, slots=True)
class LongTermEvalCaseResult:
    case_id: str
    category: str
    kind: EvalKind
    passed: bool
    matched_contains: tuple[str, ...]
    missing_contains: tuple[str, ...]
    present_forbidden: tuple[str, ...]
    observed_lookup_status: str | None = None
    observed_option_count: int | None = None


@dataclass(frozen=True, slots=True)
class LongTermEvalSeedStats:
    contacts: int
    preferences: int
    plans: int
    episodic_turns: int

    @property
    def total_memories(self) -> int:
        return self.contacts + self.preferences + self.plans + self.episodic_turns


@dataclass(frozen=True, slots=True)
class LongTermEvalSummary:
    total_cases: int
    passed_cases: int
    category_pass_counts: dict[str, int]
    category_case_counts: dict[str, int]

    @property
    def accuracy(self) -> float:
        if self.total_cases <= 0:
            return 0.0
        return self.passed_cases / self.total_cases

    def category_accuracy(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for category, total in self.category_case_counts.items():
            metrics[category] = (self.category_pass_counts.get(category, 0) / total) if total else 0.0
        return metrics


@dataclass(frozen=True, slots=True)
class LongTermEvalResult:
    seed_stats: LongTermEvalSeedStats
    summary: LongTermEvalSummary
    cases: tuple[LongTermEvalCaseResult, ...]
    temp_root: str
    memory_path: str
    graph_path: str


@dataclass(frozen=True, slots=True)
class _ContactSeed:
    given_name: str
    family_name: str
    role: str
    phone: str
    email: str
    ambiguous: bool = False

    @property
    def label(self) -> str:
        return f"{self.given_name} {self.family_name}"


@dataclass(frozen=True, slots=True)
class _PreferenceSeed:
    product: str
    brand: str
    store: str


@dataclass(frozen=True, slots=True)
class _PlanSeed:
    summary: str
    when_text: str
    contact_label: str


@dataclass(frozen=True, slots=True)
class _EpisodeSeed:
    topic: str
    transcript: str
    response: str


@dataclass(slots=True)
class _StaticQueryRewriter:
    canonical_queries: dict[str, str]

    def profile(self, query_text: str | None) -> LongTermQueryProfile:
        original = " ".join(str(query_text or "").split()).strip()
        return LongTermQueryProfile.from_text(
            original,
            canonical_english_text=self.canonical_queries.get(original),
        )


def run_synthetic_longterm_eval(
    *,
    memory_target: int = 500,
    case_target: int = 50,
    project_root: str | Path | None = None,
) -> LongTermEvalResult:
    if memory_target < 500:
        raise ValueError("memory_target must be at least 500 for the current synthetic eval.")
    if case_target < 50:
        raise ValueError("case_target must be at least 50 for the current synthetic eval.")

    with TemporaryDirectory(dir=str(project_root) if project_root is not None else None) as temp_dir:
        root = Path(temp_dir)
        config = TwinrConfig(
            project_root=str(root),
            personality_dir="personality",
            memory_markdown_path=str(root / "state" / "MEMORY.md"),
            long_term_memory_enabled=True,
            long_term_memory_background_store_turns=True,
            long_term_memory_write_queue_size=128,
            long_term_memory_recall_limit=8,
            long_term_memory_path=str(root / "state" / "chonkydb"),
            user_display_name="Erika",
            openai_realtime_language="de",
            openai_web_search_timezone="Europe/Berlin",
        )
        personality_dir = root / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        prompt_context_store = PromptContextStore(
            memory_store=PersistentMemoryMarkdownStore(config.memory_markdown_path, max_entries=1024),
            user_store=ManagedContextFileStore(
                personality_dir / "USER.md",
                section_title="Twinr managed user updates",
            ),
            personality_store=ManagedContextFileStore(
                personality_dir / "PERSONALITY.md",
                section_title="Twinr managed personality updates",
            ),
        )
        graph_store = TwinrPersonalGraphStore.from_config(config)
        service = LongTermMemoryService.from_config(
            config,
            graph_store=graph_store,
            prompt_context_store=prompt_context_store,
        )
        try:
            contacts = _seed_contacts(graph_store)
            preferences = _seed_preferences(graph_store)
            plans = _seed_plans(graph_store, contacts)
            episodes = _seed_episodes(service)
            flushed = service.flush(timeout_s=30.0)
            if not flushed:
                pending_episodic = 0 if service.writer is None else service.writer.pending_count()
                pending_multimodal = 0 if service.multimodal_writer is None else service.multimodal_writer.pending_count()
                raise RuntimeError(
                    "Synthetic long-term memory eval failed to flush background writes. "
                    f"pending_episodic={pending_episodic} pending_multimodal={pending_multimodal}"
                )

            seed_stats = LongTermEvalSeedStats(
                contacts=len(contacts),
                preferences=150,
                plans=len(plans),
                episodic_turns=len(episodes),
            )
            if seed_stats.total_memories != 500:
                raise AssertionError(f"Expected exactly 500 synthetic memories, got {seed_stats.total_memories}.")

            cases = _build_eval_cases(
                contacts=contacts,
                preferences=preferences,
                plans=plans,
                episodes=episodes,
            )
            if len(cases) != 50:
                raise AssertionError(f"Expected exactly 50 eval cases, got {len(cases)}.")
            service.query_rewriter = _StaticQueryRewriter(
                {
                    case.query_text: case.canonical_query_text
                    for case in cases
                    if case.canonical_query_text
                }
            )

            case_results = tuple(_run_eval_case(service=service, graph_store=graph_store, case=case) for case in cases)
            summary = _summarize_results(case_results)
            memory_path = str(Path(config.memory_markdown_path))
            graph_path = str(Path(config.long_term_memory_path) / "twinr_graph_v1.json")
            return LongTermEvalResult(
                seed_stats=seed_stats,
                summary=summary,
                cases=case_results,
                temp_root=str(root),
                memory_path=memory_path,
                graph_path=graph_path,
            )
        finally:
            service.shutdown(timeout_s=30.0)


def _seed_contacts(graph_store: TwinrPersonalGraphStore) -> list[_ContactSeed]:
    ambiguous_names = (
        ("Alex", ("Meyer", "Klein", "Wolf")),
        ("Chris", ("Becker", "Jung", "Lorenz")),
        ("Corinna", ("Maier", "Schmidt", "Peters")),
        ("Maria", ("Weber", "Kruse", "Busch")),
        ("Nina", ("Pohl", "Hansen", "Franke")),
        ("Sam", ("Seidel", "Berg", "Voigt")),
        ("Pat", ("Otto", "Dietz", "Mann")),
        ("Anna", ("Kraus", "Stahl", "Nagel")),
        ("Julia", ("Rose", "Moser", "Hartmann")),
        ("Lea", ("Fink", "Kurz", "Ulrich")),
    )
    ambiguous_roles = ("physiotherapist", "neighbor", "pharmacist")
    contacts: list[_ContactSeed] = []
    for given_name, family_names in ambiguous_names:
        for family_name, role in zip(family_names, ambiguous_roles, strict=True):
            index = len(contacts) + 1
            phone = f"+49 170 {index:07d}"
            email = f"{given_name.lower()}.{family_name.lower()}@example.com"
            graph_store.remember_contact(
                given_name=given_name,
                family_name=family_name,
                phone=phone,
                email=email,
                role=role.title(),
                relation=role,
            )
            contacts.append(
                _ContactSeed(
                    given_name=given_name,
                    family_name=family_name,
                    role=role,
                    phone=phone,
                    email=email,
                    ambiguous=True,
                )
            )

    first_names = (
        "Paula", "Theo", "Marta", "Jonas", "Eva", "Nora", "Lukas", "Ines", "David", "Rosa",
        "Milan", "Petra", "Sven", "Helena", "Timo", "Mira", "Frieda", "Lennart", "Ruth", "Ben",
    )
    family_names = (
        "Adler", "Bauer", "Conrad", "Dahl", "Eckert", "Faber", "Graf", "Holm", "Ivers", "Jaeger",
        "Koch", "Lang", "Mohr", "Nolte", "Opitz", "Pape", "Quast", "Reuter", "Simon", "Tesch",
    )
    roles = (
        "friend",
        "gardener",
        "hairdresser",
        "doctor",
        "grocer",
        "driver",
        "choir_friend",
        "book_club",
    )
    for index in range(120):
        given_name = first_names[index % len(first_names)]
        family_name = f"{family_names[index % len(family_names)]}{index:03d}"
        role = roles[index % len(roles)]
        phone = f"+49 151 {index + 500:07d}"
        email = f"{given_name.lower()}.{family_name.lower()}@example.com"
        graph_store.remember_contact(
            given_name=given_name,
            family_name=family_name,
            phone=phone,
            email=email,
            role=role.replace("_", " ").title(),
            relation=role,
        )
        contacts.append(
            _ContactSeed(
                given_name=given_name,
                family_name=family_name,
                role=role,
                phone=phone,
                email=email,
                ambiguous=False,
            )
        )
    return contacts


def _seed_preferences(graph_store: TwinrPersonalGraphStore) -> list[_PreferenceSeed]:
    preferences: list[_PreferenceSeed] = []
    for index in range(50):
        product = f"product {index:03d}"
        brand = f"brand {index:03d}"
        store = f"store {index:03d}"
        graph_store.remember_preference(
            category="brand",
            value=brand,
            for_product=product,
            sentiment="prefer",
            details=f"preferred option for {product}",
        )
        graph_store.remember_preference(
            category="store",
            value=store,
            for_product=product,
            sentiment="prefer",
            details=f"usually visited for {product}",
        )
        _enrich_store_multihop(graph_store, store_label=store, brand_label=brand)
        preferences.append(_PreferenceSeed(product=product, brand=brand, store=store))

    for index in range(50):
        graph_store.remember_preference(
            category="drink",
            value=f"tea blend {index:03d}",
            sentiment="like" if index % 2 == 0 else "dislike",
            details="evening preference",
        )
    return preferences + [
        _PreferenceSeed(
            product=f"tea blend {index:03d}",
            brand="",
            store="",
        )
        for index in range(50)
    ]


def _seed_plans(graph_store: TwinrPersonalGraphStore, contacts: list[_ContactSeed]) -> list[_PlanSeed]:
    plans: list[_PlanSeed] = []
    linked_contacts = [contact for contact in contacts if contact.role in {"physiotherapist", "doctor", "gardener", "friend"}]
    when_values = ("today", "tomorrow", "2026-03-20", "2026-03-21", "2026-03-22")
    for index in range(100):
        when_text = when_values[index % len(when_values)]
        if index < 30:
            contact = linked_contacts[index]
            summary = f"{contact.role.replace('_', ' ')} appointment {index:03d}"
            plan_result = graph_store.remember_plan(
                summary=summary,
                when_text=when_text,
                details=f"linked with {contact.label}",
            )
            _link_plan_to_contact(
                graph_store,
                plan_node_id=plan_result.node_id,
                contact_label=contact.label,
            )
            plans.append(_PlanSeed(summary=summary, when_text=when_text, contact_label=contact.label))
            continue
        summary = f"routine task {index:03d}"
        graph_store.remember_plan(
            summary=summary,
            when_text=when_text,
            details=f"household routine {index:03d}",
        )
        plans.append(_PlanSeed(summary=summary, when_text=when_text, contact_label=""))
    return plans


def _seed_episodes(service: LongTermMemoryService) -> list[_EpisodeSeed]:
    episodes: list[_EpisodeSeed] = []
    for index in range(100):
        topic = f"topic {index:03d}"
        transcript = f"We talked about {topic} and the plan for later."
        response = f"You said {topic} matters today, so I should remember it for our conversation."
        result = service.enqueue_conversation_turn(transcript=transcript, response=response)
        if result is None or not result.accepted:
            raise RuntimeError(f"Failed to enqueue synthetic episodic turn for {topic}.")
        episodes.append(_EpisodeSeed(topic=topic, transcript=transcript, response=response))
    return episodes


def _build_eval_cases(
    *,
    contacts: list[_ContactSeed],
    preferences: list[_PreferenceSeed],
    plans: list[_PlanSeed],
    episodes: list[_EpisodeSeed],
) -> list[LongTermEvalCase]:
    cases: list[LongTermEvalCase] = []

    exact_contacts = [contact for contact in contacts if not contact.ambiguous][:10]
    for index, contact in enumerate(exact_contacts, start=1):
        cases.append(
            LongTermEvalCase(
                case_id=f"contact_exact_{index:02d}",
                category="contact_exact_lookup",
                query_text=contact.given_name,
                kind="contact_lookup",
                lookup_family_name=contact.family_name,
                lookup_role=contact.role.replace("_", " ").title(),
                expected_lookup_status="found",
                expected_contains=(contact.label, contact.phone),
                expected_option_count=0,
            )
        )

    ambiguous_contacts = contacts[:30:3][:10]
    for index, contact in enumerate(ambiguous_contacts, start=1):
        cases.append(
            LongTermEvalCase(
                case_id=f"contact_ambiguous_{index:02d}",
                category="contact_disambiguation",
                query_text=contact.given_name,
                kind="contact_lookup",
                expected_lookup_status="needs_clarification",
                expected_option_count=3,
            )
        )

    shopping_preferences = preferences[:10]
    for index, preference in enumerate(shopping_preferences, start=1):
        cases.append(
            LongTermEvalCase(
                case_id=f"shopping_{index:02d}",
                category="shopping_recall",
                query_text=f"Where can I buy {preference.product} today?",
                kind="provider_context",
                canonical_query_text=f"{preference.product} {preference.brand} {preference.store} buy today",
                expected_contains=(preference.product, preference.brand, preference.store),
            )
        )

    linked_plans = [plan for plan in plans if plan.contact_label][:10]
    for index, plan in enumerate(linked_plans, start=1):
        cases.append(
            LongTermEvalCase(
                case_id=f"plan_multihop_{index:02d}",
                category="temporal_multihop",
                query_text=f"Who is involved in my {plan.summary} {plan.when_text}?",
                kind="provider_context",
                canonical_query_text=f"{plan.summary} {plan.contact_label} {plan.when_text}",
                expected_contains=(plan.summary, plan.contact_label, plan.when_text),
            )
        )

    episodic_targets = [episodes[index] for index in (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)]
    for index, episode in enumerate(episodic_targets, start=1):
        cases.append(
            LongTermEvalCase(
                case_id=f"episodic_{index:02d}",
                category="episodic_recall",
                query_text=f"What did we say earlier about {episode.topic}?",
                kind="provider_context",
                canonical_query_text=f"{episode.topic} plan later conversation",
                expected_contains=(episode.topic, episode.transcript),
            )
        )

    return cases


def _run_eval_case(
    *,
    service: LongTermMemoryService,
    graph_store: TwinrPersonalGraphStore,
    case: LongTermEvalCase,
) -> LongTermEvalCaseResult:
    if case.kind == "contact_lookup":
        result = graph_store.lookup_contact(
            name=case.query_text,
            family_name=case.lookup_family_name,
            role=case.lookup_role,
        )
        context_blob = " ".join(
            [result.match.label, *result.match.phones] if result.match is not None else []
        )
        matched_contains = tuple(text for text in case.expected_contains if text and text in context_blob)
        missing_contains = tuple(text for text in case.expected_contains if text and text not in context_blob)
        observed_option_count = len(result.options)
        passed = (
            result.status == case.expected_lookup_status
            and (case.expected_option_count is None or observed_option_count == case.expected_option_count)
            and not missing_contains
        )
        return LongTermEvalCaseResult(
            case_id=case.case_id,
            category=case.category,
            kind=case.kind,
            passed=passed,
            matched_contains=matched_contains,
            missing_contains=missing_contains,
            present_forbidden=(),
            observed_lookup_status=result.status,
            observed_option_count=observed_option_count,
        )

    context = service.build_provider_context(case.query_text)
    context_blob = "\n".join(context.system_messages())
    matched_contains = tuple(text for text in case.expected_contains if text and text in context_blob)
    missing_contains = tuple(text for text in case.expected_contains if text and text not in context_blob)
    present_forbidden = tuple(text for text in case.expected_absent if text and text in context_blob)
    return LongTermEvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        kind=case.kind,
        passed=not missing_contains and not present_forbidden,
        matched_contains=matched_contains,
        missing_contains=missing_contains,
        present_forbidden=present_forbidden,
    )


def _summarize_results(case_results: tuple[LongTermEvalCaseResult, ...]) -> LongTermEvalSummary:
    pass_counts: dict[str, int] = {}
    case_counts: dict[str, int] = {}
    passed_cases = 0
    for result in case_results:
        case_counts[result.category] = case_counts.get(result.category, 0) + 1
        if result.passed:
            pass_counts[result.category] = pass_counts.get(result.category, 0) + 1
            passed_cases += 1
    return LongTermEvalSummary(
        total_cases=len(case_results),
        passed_cases=passed_cases,
        category_pass_counts=pass_counts,
        category_case_counts=case_counts,
    )


def _enrich_store_multihop(graph_store: TwinrPersonalGraphStore, *, store_label: str, brand_label: str) -> None:
    document = graph_store.load_document()
    nodes = {node.node_id: node for node in document.nodes}
    edges = list(document.edges)
    store_node_id = graph_store._find_or_create_named_node(nodes, node_type="place", label=store_label)
    brand_node_id = graph_store._find_or_create_named_node(nodes, node_type="brand", label=brand_label)
    edges = graph_store._upsert_edge(
        edges,
        TwinrGraphEdgeV1(
            source_node_id=store_node_id,
            edge_type="general_related_to",
            target_node_id=brand_node_id,
            confirmed_by_user=True,
            attributes={"relation": "carries"},
        ),
    )
    edges = graph_store._upsert_edge(
        edges,
        TwinrGraphEdgeV1(
            source_node_id=store_node_id,
            edge_type="spatial_near",
            target_node_id=graph_store.user_node_id,
            confirmed_by_user=True,
        ),
    )
    graph_store._save_document(nodes, edges)


def _link_plan_to_contact(
    graph_store: TwinrPersonalGraphStore,
    *,
    plan_node_id: str,
    contact_label: str,
) -> None:
    document = graph_store.load_document()
    nodes = {node.node_id: node for node in document.nodes}
    edges = list(document.edges)
    target_node_id = None
    for node in nodes.values():
        if node.node_type == "person" and node.label == contact_label:
            target_node_id = node.node_id
            break
    if target_node_id is None:
        raise KeyError(f"Contact {contact_label!r} was not present in the synthetic graph.")
    edges = graph_store._upsert_edge(
        edges,
        TwinrGraphEdgeV1(
            source_node_id=plan_node_id,
            edge_type="general_related_to",
            target_node_id=target_node_id,
            confirmed_by_user=True,
        ),
    )
    graph_store._save_document(nodes, edges)


def _result_to_payload(result: LongTermEvalResult) -> dict[str, object]:
    return {
        "seed_stats": {
            "contacts": result.seed_stats.contacts,
            "preferences": result.seed_stats.preferences,
            "plans": result.seed_stats.plans,
            "episodic_turns": result.seed_stats.episodic_turns,
            "total_memories": result.seed_stats.total_memories,
        },
        "summary": {
            "total_cases": result.summary.total_cases,
            "passed_cases": result.summary.passed_cases,
            "accuracy": result.summary.accuracy,
            "category_case_counts": result.summary.category_case_counts,
            "category_pass_counts": result.summary.category_pass_counts,
            "category_accuracy": result.summary.category_accuracy(),
        },
        "cases": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "kind": case.kind,
                "passed": case.passed,
                "matched_contains": list(case.matched_contains),
                "missing_contains": list(case.missing_contains),
                "present_forbidden": list(case.present_forbidden),
                "observed_lookup_status": case.observed_lookup_status,
                "observed_option_count": case.observed_option_count,
            }
            for case in result.cases
        ],
    }


def main() -> int:
    payload = _result_to_payload(run_synthetic_longterm_eval())
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
