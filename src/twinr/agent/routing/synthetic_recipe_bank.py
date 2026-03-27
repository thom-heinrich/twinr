# CHANGELOG: 2026-03-27
# BUG-1: Fixed malformed "how_to_without_action" utterances by supporting template-specific slot overrides for infinitive vs. clause forms.
# BUG-2: Fixed duplicate utterances across memory families and repaired grammatically broken routine/reminder surfaces that polluted bootstrap labels.
# BUG-3: Fixed invalid home-control Cartesian products by replacing impossible device/state combinations with template-scoped valid combinations.
# SEC-1: Hardened the registry against unsafe placeholder rendering and accidental runtime mutation by validating placeholders and freezing nested mappings.
# IMP-1: Added a typed, validated, immutable registry API with deterministic corpus/example generation for threshold fitting and evaluation workflows.
# IMP-2: Added import-time global registry validation and public lookup/build helpers so bad synthetic data fails fast instead of silently poisoning routing corpora.

"""Synthetic recipe registry for routing bootstrap corpus generation.

This module defines validated, immutable recipe families for generating bootstrap
utterances used by TWINR routing. The registry is optimized for German senior-
assistant traffic and is intentionally lightweight enough for Raspberry Pi 4
deployments.

Frontier-oriented upgrades in this revision:
- safe placeholder parsing and rendering without raw ``str.format`` field access
- template-scoped slot overrides to express grammar-safe and device-safe variants
- deterministic example generation for router ``fit`` / ``evaluate`` workflows
- import-time duplicate detection across families to prevent poisoned labels
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import itertools
from math import prod
from random import Random
from string import Formatter
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

from .contracts import normalize_route_label
from .user_intent import (
    default_user_intent_for_route_label,
    normalize_user_intent_label,
)

REGISTRY_VERSION = "2026-03-27"

_ALLOWED_DIFFICULTIES: frozenset[str] = frozenset({"standard", "boundary", "hard"})
_PLACEHOLDER_FORMATTER = Formatter()

_DEFAULT_NOISE_POOL: tuple[str, ...] = (
    "clean",
    "clean",
    "clean",
    "clean",
    "clean",
    "lowercase",
    "lowercase",
    "no_punct",
    "no_punct",
    "filler",
    "umlaut_flat",
)


def _normalize_string(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _normalize_identifier(value: object, *, field_name: str) -> str:
    text = _normalize_string(value, field_name=field_name).lower()
    if not text.isascii() or not text.replace("_", "").isalnum():
        raise ValueError(
            f"{field_name} must contain only ASCII letters, digits, and underscores."
        )
    return text


def _freeze_weighted_pool(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    pool = tuple(_normalize_string(value, field_name=field_name) for value in values)
    if not pool:
        raise ValueError(f"{field_name} must not be empty.")
    return pool


def _freeze_slot_mapping(
    values: Mapping[str, Sequence[str]] | None,
    *,
    field_name: str,
) -> Mapping[str, tuple[str, ...]]:
    if not values:
        return MappingProxyType({})

    frozen: dict[str, tuple[str, ...]] = {}
    for raw_slot_name, raw_slot_values in values.items():
        slot_name = _normalize_identifier(raw_slot_name, field_name=f"{field_name} slot")
        slot_values = tuple(
            dict.fromkeys(
                _normalize_string(
                    raw_value,
                    field_name=f"{field_name}.{slot_name}",
                )
                for raw_value in raw_slot_values
            )
        )
        if not slot_values:
            raise ValueError(f"{field_name}.{slot_name} must not be empty.")
        frozen[slot_name] = slot_values
    return MappingProxyType(frozen)


def _extract_placeholders(text: str) -> tuple[str, ...]:
    placeholders: list[str] = []
    for _, field_name, format_spec, conversion in _PLACEHOLDER_FORMATTER.parse(text):
        if field_name is None:
            continue
        if format_spec or conversion:
            raise ValueError(
                f"Unsupported placeholder formatting in template {text!r}. "
                "Use bare {slot_name} placeholders only."
            )
        normalized_name = _normalize_identifier(field_name, field_name="placeholder name")
        placeholders.append(normalized_name)
    return tuple(placeholders)


def _render_template(text: str, values: Mapping[str, str]) -> str:
    rendered: list[str] = []
    for literal_text, field_name, format_spec, conversion in _PLACEHOLDER_FORMATTER.parse(text):
        rendered.append(literal_text)
        if field_name is None:
            continue
        if format_spec or conversion:
            raise ValueError(
                f"Unsupported placeholder formatting in template {text!r}. "
                "Use bare {slot_name} placeholders only."
            )
        normalized_name = _normalize_identifier(field_name, field_name="placeholder name")
        if normalized_name not in values:
            raise KeyError(
                f"Missing slot value for placeholder {normalized_name!r} in template {text!r}."
            )
        rendered.append(values[normalized_name])
    return "".join(rendered)


@dataclass(frozen=True, slots=True)
class SyntheticRouteTemplate:
    """Describe one utterance surface form for a semantic-router family."""

    key: str
    text: str
    slot_overrides: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    weight: int = 1
    placeholders: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _normalize_identifier(self.key, field_name="SyntheticRouteTemplate.key"),
        )
        object.__setattr__(
            self,
            "text",
            _normalize_string(self.text, field_name="SyntheticRouteTemplate.text"),
        )
        object.__setattr__(
            self,
            "slot_overrides",
            _freeze_slot_mapping(
                self.slot_overrides,
                field_name=f"SyntheticRouteTemplate[{self.key}].slot_overrides",
            ),
        )
        object.__setattr__(self, "placeholders", _extract_placeholders(self.text))
        if self.weight < 1:
            raise ValueError("SyntheticRouteTemplate.weight must be >= 1.")

    def resolved_slot_values(
        self,
        recipe_slot_values: Mapping[str, tuple[str, ...]],
    ) -> Mapping[str, tuple[str, ...]]:
        if not self.placeholders:
            return MappingProxyType({})
        resolved = dict(recipe_slot_values)
        resolved.update(self.slot_overrides)
        missing = [name for name in self.placeholders if name not in resolved]
        if missing:
            raise ValueError(
                f"Template {self.key!r} is missing slot values for placeholders: {missing!r}."
            )
        return MappingProxyType({name: resolved[name] for name in self.placeholders})

    def iter_slot_assignments(
        self,
        recipe_slot_values: Mapping[str, tuple[str, ...]],
    ) -> Iterator[Mapping[str, str]]:
        resolved = self.resolved_slot_values(recipe_slot_values)
        if not self.placeholders:
            yield MappingProxyType({})
            return

        value_matrix = [resolved[name] for name in self.placeholders]
        for combination in itertools.product(*value_matrix):
            yield MappingProxyType(dict(zip(self.placeholders, combination)))

    def render(self, slot_assignment: Mapping[str, str]) -> str:
        # BREAKING: unknown slot names now raise instead of being silently ignored.
        unknown_slots = sorted(set(slot_assignment) - set(self.placeholders))
        if unknown_slots:
            raise ValueError(
                f"Template {self.key!r} received unknown slots: {unknown_slots!r}."
            )
        return _render_template(self.text, slot_assignment)


@dataclass(frozen=True, slots=True)
class SyntheticRouteExample:
    """One fully rendered bootstrap utterance with provenance metadata."""

    text: str
    label: str
    family_key: str
    difficulty: str
    user_label: str
    template_key: str
    locale: str
    domain: str
    slots: Mapping[str, str]
    noise_pool: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SyntheticRouteRecipe:
    """Describe one transcript family for one semantic-router label."""

    label: str
    family_key: str
    difficulty: str
    templates: tuple[SyntheticRouteTemplate, ...]
    slot_values: Mapping[str, tuple[str, ...]]
    user_label: str | None = None
    noise_pool: tuple[str, ...] = _DEFAULT_NOISE_POOL
    locale: str = "de-de"
    domain: str = "senior_assistant"
    _template_index: Mapping[str, SyntheticRouteTemplate] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_label = normalize_route_label(self.label)
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(
            self,
            "user_label",
            normalize_user_intent_label(
                self.user_label or default_user_intent_for_route_label(normalized_label)
            ),
        )
        object.__setattr__(
            self,
            "family_key",
            _normalize_identifier(
                self.family_key,
                field_name="SyntheticRouteRecipe.family_key",
            ),
        )
        normalized_difficulty = _normalize_identifier(
            self.difficulty or "standard",
            field_name="SyntheticRouteRecipe.difficulty",
        )
        if normalized_difficulty not in _ALLOWED_DIFFICULTIES:
            raise ValueError(
                f"SyntheticRouteRecipe.difficulty must be one of {_ALLOWED_DIFFICULTIES!r}."
            )
        object.__setattr__(self, "difficulty", normalized_difficulty)

        templates = tuple(self.templates)
        if not templates:
            raise ValueError("SyntheticRouteRecipe.templates must not be empty.")
        template_keys = [template.key for template in templates]
        if len(template_keys) != len(set(template_keys)):
            raise ValueError(
                f"SyntheticRouteRecipe {self.family_key!r} contains duplicate template keys."
            )
        object.__setattr__(self, "templates", templates)
        object.__setattr__(
            self,
            "slot_values",
            _freeze_slot_mapping(
                self.slot_values,
                field_name=f"SyntheticRouteRecipe[{self.family_key}].slot_values",
            ),
        )
        object.__setattr__(
            self,
            "noise_pool",
            _freeze_weighted_pool(
                self.noise_pool,
                field_name=f"SyntheticRouteRecipe[{self.family_key}].noise_pool",
            ),
        )
        object.__setattr__(
            self,
            "locale",
            _normalize_identifier(
                str(self.locale).replace("-", "_"),
                field_name=f"SyntheticRouteRecipe[{self.family_key}].locale",
            ).replace("_", "-"),
        )
        object.__setattr__(
            self,
            "domain",
            _normalize_identifier(
                self.domain,
                field_name=f"SyntheticRouteRecipe[{self.family_key}].domain",
            ),
        )

        for template in templates:
            _ = template.resolved_slot_values(self.slot_values)

        object.__setattr__(
            self,
            "_template_index",
            MappingProxyType({template.key: template for template in templates}),
        )

    @property
    def template_count(self) -> int:
        return len(self.templates)

    @property
    def max_example_count(self) -> int:
        total = 0
        for template in self.templates:
            resolved = template.resolved_slot_values(self.slot_values)
            total += (
                1
                if not template.placeholders
                else prod(len(resolved[name]) for name in template.placeholders)
            )
        return total

    def get_template(self, template_key: str) -> SyntheticRouteTemplate:
        normalized_key = _normalize_identifier(template_key, field_name="template_key")
        try:
            return self._template_index[normalized_key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown template_key {template_key!r} for family {self.family_key!r}."
            ) from exc

    def render(self, template_key: str, /, **slots: str) -> SyntheticRouteExample:
        template = self.get_template(template_key)
        normalized_slots = MappingProxyType(
            {
                _normalize_identifier(name, field_name="slot name"): _normalize_string(
                    value,
                    field_name=f"slot[{name}]",
                )
                for name, value in slots.items()
            }
        )
        text = template.render(normalized_slots)
        return SyntheticRouteExample(
            text=text,
            label=self.label,
            family_key=self.family_key,
            difficulty=self.difficulty,
            user_label=self.user_label or "",
            template_key=template.key,
            locale=self.locale,
            domain=self.domain,
            slots=normalized_slots,
            noise_pool=self.noise_pool,
        )

    def iter_examples(
        self,
        *,
        template_keys: Sequence[str] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        limit: int | None = None,
        dedupe_text: bool = True,
    ) -> Iterator[SyntheticRouteExample]:
        selected_templates = (
            tuple(self.get_template(template_key) for template_key in template_keys)
            if template_keys
            else self.templates
        )

        examples: list[SyntheticRouteExample] = []
        seen_texts: set[str] = set()

        for template in selected_templates:
            for slot_assignment in template.iter_slot_assignments(self.slot_values):
                rendered_text = template.render(slot_assignment)
                if dedupe_text and rendered_text in seen_texts:
                    continue
                seen_texts.add(rendered_text)
                examples.append(
                    SyntheticRouteExample(
                        text=rendered_text,
                        label=self.label,
                        family_key=self.family_key,
                        difficulty=self.difficulty,
                        user_label=self.user_label or "",
                        template_key=template.key,
                        locale=self.locale,
                        domain=self.domain,
                        slots=slot_assignment,
                        noise_pool=self.noise_pool,
                    )
                )

        if shuffle:
            Random(seed).shuffle(examples)
        elif seed is not None:
            _ = seed

        if limit is not None:
            examples = examples[:limit]

        yield from examples

    def utterances(
        self,
        *,
        template_keys: Sequence[str] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        limit: int | None = None,
        dedupe_text: bool = True,
    ) -> tuple[str, ...]:
        return tuple(
            example.text
            for example in self.iter_examples(
                template_keys=template_keys,
                shuffle=shuffle,
                seed=seed,
                limit=limit,
                dedupe_text=dedupe_text,
            )
        )


def _validate_recipe_registry(recipes: Sequence[SyntheticRouteRecipe]) -> None:
    # BREAKING: invalid registries now fail fast at import time instead of poisoning
    # downstream corpus generation with silent duplicates / collisions.
    family_keys = [recipe.family_key for recipe in recipes]
    if len(family_keys) != len(set(family_keys)):
        duplicates = sorted(
            family_key
            for family_key, count in Counter(family_keys).items()
            if count > 1
        )
        raise ValueError(f"Duplicate family_key values detected: {duplicates!r}")

    utterance_index: dict[str, tuple[str, str]] = {}
    collisions: list[str] = []
    for recipe in recipes:
        for example in recipe.iter_examples(dedupe_text=False):
            existing = utterance_index.get(example.text)
            current = (recipe.family_key, example.template_key)
            if existing is not None and existing != current:
                collisions.append(
                    f"{example.text!r} -> {existing[0]}/{existing[1]} vs. {current[0]}/{current[1]}"
                )
            else:
                utterance_index[example.text] = current
    if collisions:
        joined = "; ".join(collisions[:10])
        raise ValueError(
            f"Duplicate rendered utterances detected across families: {joined}"
        )


def iter_route_recipes(
    *,
    label: str | None = None,
    user_label: str | None = None,
    difficulty: str | None = None,
) -> Iterator[SyntheticRouteRecipe]:
    normalized_label = normalize_route_label(label) if label is not None else None
    normalized_user_label = (
        normalize_user_intent_label(user_label) if user_label is not None else None
    )
    normalized_difficulty = (
        _normalize_identifier(difficulty, field_name="difficulty")
        if difficulty is not None
        else None
    )

    for recipe in _ROUTE_RECIPES:
        if normalized_label is not None and recipe.label != normalized_label:
            continue
        if normalized_user_label is not None and recipe.user_label != normalized_user_label:
            continue
        if normalized_difficulty is not None and recipe.difficulty != normalized_difficulty:
            continue
        yield recipe


def get_route_recipes(
    *,
    label: str | None = None,
    user_label: str | None = None,
    difficulty: str | None = None,
) -> tuple[SyntheticRouteRecipe, ...]:
    return tuple(
        iter_route_recipes(
            label=label,
            user_label=user_label,
            difficulty=difficulty,
        )
    )


def get_route_recipe(family_key: str) -> SyntheticRouteRecipe:
    normalized_family_key = _normalize_identifier(family_key, field_name="family_key")
    try:
        return _ROUTE_RECIPE_BY_FAMILY_KEY[normalized_family_key]
    except KeyError as exc:
        raise KeyError(f"Unknown SyntheticRouteRecipe family_key: {family_key!r}") from exc


def iter_bootstrap_examples(
    *,
    label: str | None = None,
    user_label: str | None = None,
    difficulty: str | None = None,
    per_recipe_limit: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    dedupe_text: bool = True,
) -> Iterator[SyntheticRouteExample]:
    rng = Random(seed)
    recipes = list(
        iter_route_recipes(
            label=label,
            user_label=user_label,
            difficulty=difficulty,
        )
    )
    if shuffle:
        rng.shuffle(recipes)

    for recipe in recipes:
        recipe_seed = rng.randrange(0, 2**32) if shuffle else seed
        yield from recipe.iter_examples(
            limit=per_recipe_limit,
            shuffle=shuffle,
            seed=recipe_seed,
            dedupe_text=dedupe_text,
        )


def build_bootstrap_corpus(
    *,
    label: str | None = None,
    user_label: str | None = None,
    difficulty: str | None = None,
    per_recipe_limit: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    dedupe_text: bool = True,
) -> tuple[SyntheticRouteExample, ...]:
    return tuple(
        iter_bootstrap_examples(
            label=label,
            user_label=user_label,
            difficulty=difficulty,
            per_recipe_limit=per_recipe_limit,
            shuffle=shuffle,
            seed=seed,
            dedupe_text=dedupe_text,
        )
    )


T = SyntheticRouteTemplate
R = SyntheticRouteRecipe


_ROUTE_RECIPES: tuple[SyntheticRouteRecipe, ...] = (
    R(
        label="parametric",
        family_key="stable_explanation",
        difficulty="standard",
        templates=(
            T("plain", "Erklaer mir {concept} in einfachen Worten."),
            T("question", "Was bedeutet {concept}?"),
            T("function", "Wie funktioniert {concept} genau?"),
            T("spoken", "Kannst du mir {concept} kurz erklaeren?"),
        ),
        slot_values={
            "concept": (
                "Photosynthese",
                "Arthrose",
                "Schwerkraft",
                "Inflation",
                "Demenz",
                "Solarenergie",
                "Blutdruck",
                "Cholesterin",
                "WLAN",
                "Impfung",
                "Osteoporose",
                "Diabetes Typ zwei",
                "Lungenentzuendung",
                "Parkinson",
                "Glukose",
                "Waermepumpe",
                "Staatsverschuldung",
                "Datensicherung",
                "Quantencomputer",
                "Verdauung",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="stable_comparison",
        difficulty="boundary",
        templates=(
            T("difference", "Was ist der Unterschied zwischen {pair}?"),
            T("recognize", "Woran erkennt man den Unterschied zwischen {pair}?"),
            T("simple", "Vergleich mir {pair} in einfachen Worten."),
        ),
        slot_values={
            "pair": (
                "Arthrose und Arthritis",
                "Vitamin D und Calcium",
                "Bakterien und Viren",
                "HDMI und USB C",
                "Herzinfarkt und Schlaganfall",
                "Miete und Nebenkosten",
                "Erkaeltung und Grippe",
                "Demenz und Delir",
                "WLAN und Bluetooth",
                "Tablette und Kapsel",
                "Rente und Pension",
                "Insulin und Blutzucker",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="how_to_without_action",
        difficulty="boundary",
        templates=(
            T(
                "how_to",
                "Wie kann man {task}?",
                slot_overrides={
                    "task": (
                        "einen Timer einstellen",
                        "einen Brief formulieren",
                        "Nudeln kochen",
                        "Blutdruck messen",
                        "einen Videocall vorbereiten",
                        "eine Zimmerpflanze umtopfen",
                        "eine E Mail beantworten",
                        "eine Einkaufsliste schreiben",
                        "das Smartphone lauter stellen",
                        "die Heizung entlueften",
                        "einen Verband wechseln",
                        "ein Passwort sicher aufschreiben",
                    ),
                },
            ),
            T("explain_steps", "Erklaer mir Schritt fuer Schritt, wie man {task}."),
            T("learn", "Ich moechte lernen, wie man {task}."),
        ),
        slot_values={
            "task": (
                "einen Timer einstellt",
                "einen Brief formuliert",
                "Nudeln kocht",
                "Blutdruck misst",
                "einen Videocall vorbereitet",
                "eine Zimmerpflanze umtopft",
                "eine E Mail beantwortet",
                "eine Einkaufsliste schreibt",
                "das Smartphone lauter stellt",
                "die Heizung entlueftet",
                "einen Verband wechselt",
                "ein Passwort sicher aufschreibt",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="history_and_people",
        difficulty="standard",
        templates=(
            T("who", "Wer war {person}?"),
            T("known_for", "Warum ist {person} bekannt?"),
            T("when", "Wann lebte {person}?"),
        ),
        slot_values={
            "person": (
                "Marie Curie",
                "Sophie Scholl",
                "Ada Lovelace",
                "Konrad Adenauer",
                "Albert Einstein",
                "Johann Sebastian Bach",
                "Rosalind Franklin",
                "Hildegard von Bingen",
                "Clara Schumann",
                "Otto von Bismarck",
                "Alan Turing",
                "Nelson Mandela",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="translation_and_phrase_help",
        difficulty="standard",
        templates=(
            T("translate", "Wie sagt man {phrase} auf {language}?"),
            T("means", "Was heisst {phrase} auf {language}?"),
            T("phrase_help", "Uebersetz mir {phrase} in {language}."),
        ),
        slot_values={
            "phrase": (
                "Guten Morgen",
                "Wie geht es dir",
                "Vielen Dank",
                "Ich brauche Hilfe",
                "Bitte langsam sprechen",
                "Wo ist der Bahnhof",
                "Ich moechte einen Kaffee",
                "Bis spaeter",
                "Kannst du mir helfen",
                "Ich komme morgen wieder",
            ),
            "language": (
                "Italienisch",
                "Spanisch",
                "Englisch",
                "Franzoesisch",
                "Portugiesisch",
                "Niederlaendisch",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="stable_short_explanation",
        difficulty="standard",
        templates=(
            T("short", "Erklaer mir kurz {concept}."),
            T("simple", "Kannst du mir einfach erklaeren, was {concept} ist?"),
            T("spoken", "Sag mir bitte kurz, was {concept} bedeutet."),
        ),
        slot_values={
            "concept": (
                "Photosynthese",
                "Osteoporose",
                "Demenz",
                "Parkinson",
                "eine Waermepumpe",
                "Inflation",
                "Cholesterin",
                "Blutdruck",
                "Glukose",
                "Datensicherung",
                "Arthrose",
                "Schwerkraft",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="stable_why_questions",
        difficulty="boundary",
        templates=(
            T("why", "Warum {phenomenon}?"),
            T("wieso", "Wieso {phenomenon}?"),
            T("explain_why", "Kannst du mir erklaeren, warum {phenomenon}?"),
        ),
        slot_values={
            "phenomenon": (
                "ist der Himmel blau",
                "schlafen Menschen nachts",
                "wird Fieber anstrengend",
                "steigt der Blutzucker nach dem Essen",
                "werden Blaetter im Herbst gelb",
                "braucht der Koerper Wasser",
                "rostet Eisen",
                "wird man bei einer Erkaeltung muede",
                "ist Bewegung im Alter wichtig",
                "frieren Menschen bei Wind schneller",
                "ist Stress fuer den Blutdruck schlecht",
                "ist ein Helm beim Fahrradfahren sinnvoll",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="general_tech_and_daily_knowledge",
        difficulty="boundary",
        templates=(
            T("works", "Wie funktioniert {thing} eigentlich?"),
            T("means", "Was genau ist {thing}?"),
            T("learn", "Ich moechte verstehen, wie {thing} funktioniert."),
        ),
        slot_values={
            "thing": (
                "ein Router",
                "Bluetooth",
                "ein Akku",
                "ein Kalender",
                "eine To do Liste",
                "ein Passwortmanager",
                "ein Hoergeraet",
                "eine Fernbedienung",
                "ein Thermostat",
                "ein QR Code",
                "ein Blutdruckmessgeraet",
                "ein Smartphone Update",
            ),
        },
    ),
    R(
        label="parametric",
        family_key="stable_usage_questions",
        difficulty="boundary",
        templates=(
            T("usage", "Wofuer braucht man {thing}?"),
            T("purpose", "Wozu ist {thing} gut?"),
            T("body", "Was macht {system_part}?"),
        ),
        slot_values={
            "thing": (
                "Vitamin B12",
                "eine Brille",
                "eine Impfung",
                "einen Router",
                "eine Waermepumpe",
                "eine Gehhilfe",
                "ein Blutdruckmessgeraet",
                "einen Passwortmanager",
                "eine Patientenverfuegung",
                "eine Einkaufsliste",
                "Bewegung im Alter",
                "ein Hoergeraet",
            ),
            "system_part": (
                "die Schilddruese",
                "die Niere",
                "das Immunsystem",
                "die Lunge",
                "der Blutkreislauf",
                "das Gleichgewichtssystem",
                "die Bauchspeicheldruese",
                "die Verdauung",
            ),
        },
    ),
    R(
        label="web",
        family_key="current_weather",
        difficulty="standard",
        templates=(
            T("forecast", "Wie wird das Wetter {time_ref} in {place}?"),
            T("umbrella", "Brauche ich {time_ref} in {place} einen Schirm?"),
            T("temperature", "Wie warm wird es {time_ref} in {place}?"),
        ),
        slot_values={
            "time_ref": (
                "heute",
                "morgen",
                "am Wochenende",
                "heute Abend",
                "morgen Mittag",
                "in der Nacht",
            ),
            "place": (
                "Berlin",
                "Hamburg",
                "Muenchen",
                "Koeln",
                "Leipzig",
                "Bremen",
                "Dresden",
                "Stuttgart",
                "Frankfurt",
                "Rostock",
                "Freiburg",
                "Hannover",
            ),
        },
    ),
    R(
        label="web",
        family_key="current_news",
        difficulty="standard",
        templates=(
            T("today_news", "Was ist {time_ref} in {place} passiert?"),
            T("latest_topic", "Gibt es {time_ref} Neuigkeiten zu {topic}?"),
            T("summary", "Fass mir die aktuellen Meldungen zu {topic} zusammen."),
        ),
        slot_values={
            "time_ref": (
                "heute",
                "gerade",
                "im Moment",
                "seit heute Morgen",
                "heute Abend",
                "in den letzten Stunden",
            ),
            "place": (
                "Berlin",
                "Deutschland",
                "Europa",
                "Hamburg",
                "Muenchen",
                "Brandenburg",
                "Sachsen",
                "Nordrhein Westfalen",
            ),
            "topic": (
                "der Bahn",
                "dem Bundestag",
                "Unwettern",
                "Pflegepolitik",
                "dem Nahverkehr",
                "dem Gesundheitssystem",
                "Renten",
                "Energiepreisen",
                "dem Wetterchaos",
                "dem Flughafen",
                "Pflegenotstand",
                "dem Stromnetz",
            ),
        },
    ),
    R(
        label="web",
        family_key="live_schedule_and_opening",
        difficulty="boundary",
        templates=(
            T("train", "Wann faehrt der naechste Zug von {origin} nach {destination}?"),
            T("open_now", "Hat {place} {time_ref} geoeffnet?"),
            T("traffic", "Ist {route} {time_ref} gesperrt?"),
        ),
        slot_values={
            "origin": (
                "Berlin",
                "Hamburg",
                "Koeln",
                "Leipzig",
                "Dresden",
                "Kiel",
                "Potsdam",
                "Erfurt",
            ),
            "destination": (
                "Potsdam",
                "Hannover",
                "Dresden",
                "Bremen",
                "Magdeburg",
                "Rostock",
                "Jena",
                "Luebeck",
            ),
            "place": (
                "die Apotheke am Markt",
                "das Buero vom Buergerservice",
                "das Hallenbad",
                "der Supermarkt um die Ecke",
                "die Bibliothek",
                "das Rathaus",
                "die Sparkasse am Platz",
                "die Physiopraxis",
            ),
            "time_ref": (
                "heute",
                "jetzt",
                "morgen frueh",
                "am Sonntag",
                "heute Nachmittag",
                "naechste Woche",
            ),
            "route": (
                "die A9",
                "die Ringbahn",
                "die Strecke nach Potsdam",
                "die Linie U2",
                "die A7",
                "die Strecke nach Dresden",
                "die Regionalbahn nach Jena",
                "die S Bahn zum Flughafen",
            ),
        },
    ),
    R(
        label="web",
        family_key="current_price_and_availability",
        difficulty="standard",
        templates=(
            T("price_now", "Wie teuer ist {asset} gerade?"),
            T("price_current", "Wie hoch ist aktuell der Preis von {asset}?"),
            T("availability", "Ist {product} im Moment lieferbar?"),
        ),
        slot_values={
            "asset": (
                "Heizoel",
                "Gold",
                "Bitcoin",
                "eine Monatskarte",
                "Strom",
                "Gas",
                "Silber",
                "Diesel",
                "Superbenzin",
                "Fernwaerme",
            ),
            "product": (
                "das Diabetes Messgeraet",
                "das Druckerpapier",
                "die Lieblingsschokolade",
                "der Blutdrucksensor",
                "die Tintenpatrone",
                "das Hoergeraet Batteriepack",
                "die Schmerzsalbe",
                "der Wasserfilter",
            ),
        },
    ),
    R(
        label="web",
        family_key="safety_lookup",
        difficulty="boundary",
        templates=(
            T("meds", "Kann ich {med_a} mit {med_b} zusammen nehmen?"),
            T("recall", "Gibt es aktuell Rueckrufe fuer {product}?"),
            T("warning", "Gibt es {time_ref} eine Warnung fuer {hazard} in {place}?"),
        ),
        slot_values={
            "med_a": (
                "Ibuprofen",
                "Aspirin",
                "Paracetamol",
                "Magnesium",
                "Diclofenac",
                "Nasenspray",
                "Johanniskraut",
                "Vitamin K",
            ),
            "med_b": (
                "Blutverduenner",
                "meine Herztabletten",
                "Antibiotika",
                "Schlafmittel",
                "Insulin",
                "meine Schilddruesentabletten",
                "meine Blutdrucksenker",
                "Cortison",
            ),
            "product": (
                "Raeucherlachs",
                "Erdbeeren",
                "Babynahrung",
                "Blutdrucktabletten",
                "Fertigsalat",
                "Eiscreme",
                "Wurstaufschnitt",
                "Tiefkuehlbeeren",
            ),
            "time_ref": ("heute", "gerade", "fuer morgen", "im Moment"),
            "hazard": (
                "Sturm",
                "Glatteis",
                "Hochwasser",
                "starke Hitze",
                "Waldbrand",
                "Feinstaub",
                "starker Regen",
                "Gewitter",
            ),
            "place": (
                "Berlin",
                "Hamburg",
                "Sachsen",
                "Brandenburg",
                "Bayern",
                "Thueringen",
                "Hessen",
                "Rheinland Pfalz",
            ),
        },
    ),
    R(
        label="web",
        family_key="current_role_holder",
        difficulty="boundary",
        templates=(
            T("derzeit", "Wer ist derzeit {role}?"),
            T("aktuell", "Wer ist aktuell {role}?"),
            T("amt", "Wer hat im Moment das Amt {role}?"),
        ),
        slot_values={
            "role": (
                "Bundeskanzler",
                "Bundespraesident",
                "Praesident der USA",
                "Buergermeister von Berlin",
                "Chef der Deutschen Bahn",
                "Praesident von Frankreich",
                "EU Kommissionspraesidentin",
                "Ministerpraesident von Bayern",
                "Trainer der deutschen Fussballnationalmannschaft",
                "Praesident der Ukraine",
            ),
        },
    ),
    R(
        label="web",
        family_key="live_topic_status",
        difficulty="boundary",
        templates=(
            T("what_now", "Was ist aktuell mit {topic} los?"),
            T("situation", "Wie ist momentan die Lage bei {topic}?"),
            T("new", "Gibt es derzeit etwas Neues zu {topic}?"),
        ),
        slot_values={
            "topic": (
                "der Bahn",
                "den Strompreisen",
                "dem Pflegegeld",
                "den Benzinpreisen",
                "dem Krankenhausstreik",
                "dem Wetter in Norddeutschland",
                "dem Nahverkehr in Berlin",
                "den Unwettern in Bayern",
                "den Rentenplaenen der Regierung",
                "dem Flughafen in Hamburg",
                "dem Wohnungsmarkt",
                "der Lage in der Ukraine",
            ),
        },
    ),
    R(
        label="web",
        family_key="mutable_role_plain_question",
        difficulty="boundary",
        templates=(
            T("plain_role", "Wer ist {role}?"),
            T("holds_office", "Wer hat das Amt {role}?"),
            T("leads_org", "Wer leitet {institution}?"),
        ),
        slot_values={
            "role": (
                "der Bundeskanzler",
                "der Bundespraesident",
                "der Praesident der USA",
                "der Buergermeister von Berlin",
                "der Ministerpraesident von Bayern",
                "der Chef der Deutschen Bahn",
                "der Praesident von Frankreich",
                "die EU Kommissionspraesidentin",
            ),
            "institution": (
                "die Deutsche Bahn",
                "die Bundesregierung",
                "die EU Kommission",
                "das Kanzleramt",
                "das Bundesgesundheitsministerium",
                "das Weisse Haus",
            ),
        },
    ),
    R(
        label="memory",
        family_key="personal_fact",
        difficulty="standard",
        templates=(
            T("name", "Wie heisst {relation}?"),
            T("birthday", "Wann hat {relation} Geburtstag?"),
            T("doctor", "Wie heisst mein {role}?"),
        ),
        slot_values={
            "relation": (
                "meine Enkelin",
                "mein Enkel",
                "meine Tochter",
                "mein Sohn",
                "meine Nachbarin",
                "meine Schwester",
                "mein Bruder",
                "meine Pflegerin",
                "mein Schwiegersohn",
                "meine Enkeltochter",
                "mein Patenkind",
                "meine Freundin aus dem Chor",
            ),
            "role": (
                "Hausarzt",
                "Kardiologe",
                "Physiotherapeut",
                "Friseur",
                "Augenarzt",
                "Pflegedienst",
                "Podologe",
                "Zahnarzt",
                "Orthopaede",
                "Fahrdienst",
            ),
        },
    ),
    R(
        label="memory",
        family_key="preferences",
        difficulty="standard",
        templates=(
            T("favorite", "Welchen {item} mag ich am liebsten?"),
            T("music", "Welche Musik hoere ich gern, wenn ich unruhig bin?"),
            T("routine", "Welchen Tee trinke ich abends am liebsten?"),
        ),
        slot_values={
            "item": (
                "Kuchen",
                "Tee",
                "Saft",
                "Pullover",
                "Filmgenre",
                "Suppe",
                "Marmelade",
                "Musikrichtung",
                "Kissen",
                "Schal",
                "Buchgenre",
                "Brot",
            ),
        },
    ),
    R(
        label="memory",
        family_key="household_location",
        difficulty="boundary",
        templates=(
            T("where", "Wo haben wir {object_name} hingelegt?"),
            T("kept", "Wo liegt {object_name} normalerweise?"),
            T("stored", "In welchem Schrank bewahren wir {object_name} auf?"),
        ),
        slot_values={
            "object_name": (
                "den Zweitschluessel",
                "die Impfausweise",
                "das Ladegeraet",
                "die Winterdecken",
                "die Taschenlampe",
                "das Rezeptheft",
                "die Ersatzbrille",
                "den Garagenschluessel",
                "den Reisepass",
                "die Notfallmappe",
                "das Hoergeraet Ladegeraet",
                "die Kerzen",
            ),
        },
    ),
    R(
        label="memory",
        family_key="prior_conversation",
        difficulty="boundary",
        templates=(
            T("remember_plan", "Was hatten wir ueber {topic} festgehalten?"),
            T("reminder_context", "Woran wollte ich mich {time_ref} erinnern?"),
            T("trip_plan", "Welchen Plan hatten wir fuer {event}?"),
        ),
        slot_values={
            "topic": (
                "den Urlaub",
                "den Arztbesuch",
                "den Einkauf",
                "die Geburtstagsfeier",
                "die Steuerunterlagen",
                "die Gartenarbeit",
                "die Reha",
                "das Abendessen mit der Familie",
                "den Zahnarzttermin",
                "die Bahnfahrt",
            ),
            "time_ref": (
                "morgen",
                "naechste Woche",
                "am Wochenende",
                "im April",
                "im Mai",
                "naechsten Monat",
                "an Weihnachten",
                "uebermorgen",
            ),
            "event": (
                "Ostern",
                "den Familienbesuch",
                "den Ausflug",
                "den Geburtstag von Anna",
                "das Sommerfest",
                "den Besuch vom Pflegedienst",
                "die Reise nach Hamburg",
                "mein Jubilaeum",
                "den Chorabend",
                "den Arzttermin am Montag",
            ),
        },
    ),
    R(
        label="memory",
        family_key="personal_health_notes",
        difficulty="boundary",
        templates=(
            T("allergies", "Welche Allergien habe ich?"),
            T("meds", "Welche Tabletten nehme ich morgens?"),
            T("notes", "Was steht in meinen Notizen zu {topic}?"),
        ),
        slot_values={
            "topic": (
                "meinem Ruecken",
                "meinem Schlaf",
                "meinem Blutdruck",
                "meiner Ernaehrung",
                "meinen Knien",
                "meinem Gleichgewicht",
                "meiner Haut",
                "meinem Magen",
                "meiner Bewegung",
                "meinem Blutzucker",
            ),
        },
    ),
    R(
        label="memory",
        family_key="contact_details",
        difficulty="boundary",
        templates=(
            T("phone", "Welche Telefonnummer hat {contact}?"),
            T("address", "Welche Adresse habe ich fuer {contact} gespeichert?"),
            T("details", "Welche Kontaktdaten habe ich zu {contact} notiert?"),
        ),
        slot_values={
            "contact": (
                "meine Tochter",
                "meinen Sohn",
                "meine Schwester",
                "meinen Hausarzt",
                "den Pflegedienst",
                "Frau Schneider",
                "meinen Nachbarn",
                "meine Enkelin",
                "meinen Fahrdienst",
                "die Apotheke",
                "meinen Physiotherapeuten",
                "den Hausmeister",
            ),
        },
    ),
    R(
        label="memory",
        family_key="personal_routines",
        difficulty="boundary",
        templates=(
            T("when_usually", "Wann ist bei mir normalerweise {routine}?"),
            T("day", "An welchem Tag ist bei mir {routine} dran?"),
            T("time", "Zu welcher Uhrzeit ist bei mir meistens {routine}?"),
        ),
        slot_values={
            "routine": (
                "die Gymnastik",
                "der Wocheneinkauf",
                "das Blumen giessen",
                "das Blutdruckmessen",
                "der Anruf bei meiner Tochter",
                "das Sortieren der Medikamente",
                "der Spaziergang",
                "das Tagebuchschreiben",
                "das Laden des Hoergeraets",
                "das Muell rausbringen",
                "das Bett beziehen",
                "der Chorabend",
            ),
        },
    ),
    R(
        label="memory",
        family_key="known_people_preferences",
        difficulty="boundary",
        templates=(
            T("likes", "Was mag {person} besonders gern?"),
            T("prefers", "Worueber freut sich {person} am meisten?"),
            T("notes", "Was hatten wir ueber {person} festgehalten?"),
        ),
        slot_values={
            "person": (
                "Anna",
                "Karl",
                "meine Tochter",
                "mein Enkel",
                "Frau Schneider",
                "der Pfleger Jens",
                "meine Nachbarin",
                "mein Sohn",
                "Oma Lotte",
                "Onkel Peter",
            ),
        },
    ),
    R(
        label="memory",
        family_key="short_known_people_preferences",
        difficulty="boundary",
        templates=(
            T("short_like", "Was mag {person}?"),
            T("likes", "Was mag {person} gern?"),
            T("happy", "Worueber freut sich {person}?"),
        ),
        slot_values={
            "person": (
                "Anna",
                "Karl",
                "meine Tochter",
                "mein Sohn",
                "meine Schwester",
                "mein Enkel",
                "Frau Schneider",
                "der Pfleger Jens",
                "meine Nachbarin",
                "mein Hausarzt",
            ),
        },
    ),
    R(
        label="memory",
        family_key="remembered_people_context",
        difficulty="boundary",
        templates=(
            T("about_person", "Was hatten wir ueber {person} besprochen?"),
            T("notes", "Was weiss ich ueber {person}?"),
            T("stored", "Was hatten wir uns zu {person} notiert?"),
        ),
        slot_values={
            "person": (
                "Anna",
                "Karl",
                "meine Tochter",
                "meinen Sohn",
                "meine Schwester",
                "meinen Nachbarn",
                "Frau Schneider",
                "den Pfleger Jens",
                "meine Enkelin",
                "meinen Hausarzt",
                "den Fahrdienst",
                "den Physiotherapeuten",
            ),
        },
    ),
    R(
        label="memory",
        family_key="person_reason_memory",
        difficulty="boundary",
        templates=(
            T("call_reason", "Warum hat {person} angerufen?"),
            T("visit_reason", "Aus welchem Grund kommt {person} zu mir?"),
            T("wanted_reason", "Weshalb wollte {person} mich sprechen?"),
        ),
        slot_values={
            "person": (
                "Anna",
                "meine Tochter",
                "mein Sohn",
                "meine Schwester",
                "der Pflegedienst",
                "der Fahrdienst",
                "Frau Schneider",
                "mein Hausarzt",
                "meine Enkelin",
                "mein Nachbar",
            ),
        },
    ),
    R(
        label="memory",
        family_key="personal_reason_context",
        difficulty="boundary",
        templates=(
            T("why_visit", "Warum kommt {person} zu mir?"),
            T("why_call", "Warum wollte {person} mich anrufen?"),
            T("why_note", "Weshalb hatten wir {topic} fuer mich festgehalten?"),
        ),
        slot_values={
            "person": (
                "Anna",
                "meine Tochter",
                "der Pflegedienst",
                "mein Sohn",
                "Frau Schneider",
                "mein Hausarzt",
                "der Fahrdienst",
                "meine Enkelin",
            ),
            "topic": (
                "den Arzttermin",
                "den Einkauf",
                "die Reise nach Hamburg",
                "die Medikamente",
                "den Besuch am Wochenende",
                "den Zahnarzttermin",
                "den Chorabend",
                "die Geburtstagsfeier",
            ),
        },
    ),
    R(
        label="tool",
        user_label="persoenlich",
        family_key="personal_schedule_lookup",
        difficulty="boundary",
        templates=(
            T("appointments", "Was habe ich {time_ref} fuer Termine?"),
            T("calendar", "Was steht {time_ref} in meinem Kalender?"),
            T("visit", "Kommt {time_ref} jemand zu mir?"),
        ),
        slot_values={
            "time_ref": (
                "heute",
                "morgen",
                "heute Nachmittag",
                "heute Abend",
                "am Montag",
                "naechste Woche",
                "diese Woche",
                "uebermorgen",
            ),
        },
    ),
    R(
        label="tool",
        user_label="persoenlich",
        family_key="personal_pending_state",
        difficulty="boundary",
        templates=(
            T("meds_open", "Habe ich {time_ref} noch Medikamente offen?"),
            T("reminders", "Welche Erinnerung habe ich {time_ref}?"),
            T("tasks", "Welche Aufgaben habe ich {time_ref} noch offen?"),
        ),
        slot_values={
            "time_ref": (
                "heute",
                "heute Abend",
                "morgen",
                "fuer heute",
                "fuer morgen",
                "am Wochenende",
                "naechste Woche",
                "heute Nacht",
            ),
        },
    ),
    R(
        label="tool",
        user_label="persoenlich",
        family_key="personal_day_overview",
        difficulty="boundary",
        templates=(
            T("today", "Was ist {time_ref} bei mir los?"),
            T("for_me", "Was steht {time_ref} fuer mich an?"),
            T("overview", "Was habe ich {time_ref} noch vor?"),
        ),
        slot_values={
            "time_ref": (
                "heute",
                "heute Nachmittag",
                "heute Abend",
                "morgen",
                "am Montag",
                "diese Woche",
                "naechste Woche",
                "uebermorgen",
            ),
        },
    ),
    R(
        label="tool",
        user_label="persoenlich",
        family_key="personal_short_schedule_status",
        difficulty="boundary",
        templates=(
            T("have_plan", "Hab ich {time_ref} was vor?"),
            T("planned", "Ist {time_ref} bei mir was geplant?"),
            T("coming_up", "Steht {time_ref} bei mir etwas an?"),
        ),
        slot_values={
            "time_ref": (
                "heute",
                "heute Nachmittag",
                "heute Abend",
                "morgen",
                "am Montag",
                "diese Woche",
                "naechste Woche",
                "uebermorgen",
            ),
        },
    ),
    R(
        label="tool",
        user_label="persoenlich",
        family_key="care_support_visits",
        difficulty="boundary",
        templates=(
            T("when_arrive", "Wann kommt {visitor} zu mir?"),
            T("is_scheduled", "Ist {visitor} {time_ref} bei mir eingetragen?"),
            T("coming", "Kommt {visitor} {time_ref} zu mir?"),
        ),
        slot_values={
            "visitor": (
                "der Pflegedienst",
                "meine Tochter",
                "mein Sohn",
                "der Fahrdienst",
                "mein Hausarzt",
                "die Physiotherapie",
                "meine Enkelin",
                "Frau Schneider",
            ),
            "time_ref": (
                "heute",
                "morgen",
                "heute Abend",
                "heute Nachmittag",
                "am Montag",
                "diese Woche",
                "naechste Woche",
                "uebermorgen",
            ),
        },
    ),
    R(
        label="tool",
        family_key="house_state_snapshot",
        difficulty="boundary",
        templates=(
            T("house_now", "Was ist im Haus los?"),
            T("doors", "Ist {door} gerade offen?"),
            T("room_state", "Wie ist der Status im {room}?"),
        ),
        slot_values={
            "door": (
                "die Haustuer",
                "die Balkontuer",
                "das Kuechenfenster",
                "das Schlafzimmerfenster",
                "die Kellertuer",
                "die Terrassentuer",
            ),
            "room": (
                "Wohnzimmer",
                "Kueche",
                "Flur",
                "Schlafzimmer",
                "Bad",
                "Arbeitszimmer",
            ),
        },
    ),
    R(
        label="tool",
        family_key="home_environment_state",
        difficulty="boundary",
        templates=(
            T("temperature", "Wie warm ist es gerade im {room}?"),
            T("light", "Ist im {room} noch Licht an?"),
            T("appliance", "Ist {appliance} gerade an?"),
        ),
        slot_values={
            "room": (
                "Wohnzimmer",
                "Schlafzimmer",
                "Bad",
                "Flur",
                "Kueche",
                "Arbeitszimmer",
                "Gaestezimmer",
                "Keller",
            ),
            "appliance": (
                "der Herd",
                "der Fernseher",
                "die Kaffeemaschine",
                "die Heizdecke",
                "der Ventilator",
                "die Waschmaschine",
                "der Trockner",
                "die Nachttischlampe",
            ),
        },
    ),
    R(
        label="tool",
        family_key="doorbell_and_presence_check",
        difficulty="boundary",
        templates=(
            T("doorbell", "Hat es gerade geklingelt?"),
            T("door", "War jemand an der Haustuer?"),
            T("arrival", "Ist eben jemand gekommen?"),
        ),
        slot_values={},
    ),
    # BREAKING: home_scene_control template keys changed from
    # ("lights", "switch", "set") to domain-safe control groups
    # ("binary_power", "volume", "heating", "fan_speed") so impossible
    # device/state Cartesian products can no longer be generated.
    R(
        label="tool",
        family_key="home_scene_control",
        difficulty="standard",
        templates=(
            T(
                "binary_power",
                "Mach {device} bitte {state}.",
                slot_overrides={
                    "device": (
                        "das Licht im Wohnzimmer",
                        "die Lampe im Flur",
                        "den Fernseher",
                        "die Steckdose an der Kaffeemaschine",
                        "das Radio",
                        "die Nachttischlampe",
                    ),
                    "state": ("an", "aus"),
                },
            ),
            T(
                "volume",
                "Mach {device} bitte {state}.",
                slot_overrides={
                    "device": ("den Fernseher", "das Radio"),
                    "state": ("leiser", "lauter"),
                },
            ),
            T(
                "heating",
                "Stell {device} auf {state}.",
                slot_overrides={
                    "device": ("die Heizung im Bad",),
                    "state": ("waermer", "kuehler"),
                },
            ),
            T(
                "fan_speed",
                "Stell {device} auf Stufe {state}.",
                slot_overrides={
                    "device": ("den Ventilator",),
                    "state": ("eins", "zwei", "drei"),
                },
            ),
        ),
        slot_values={},
    ),
    R(
        label="tool",
        family_key="timers_and_reminders",
        difficulty="standard",
        templates=(
            T("timer", "Stell einen Timer auf {duration}."),
            T("alarm", "Mach mir einen Wecker fuer {clock_time}."),
            T("remind", "Erinnere mich {time_ref} an {task}."),
        ),
        slot_values={
            "duration": (
                "5 Minuten",
                "10 Minuten",
                "15 Minuten",
                "20 Minuten",
                "25 Minuten",
                "30 Minuten",
                "45 Minuten",
                "eine Stunde",
                "zwei Stunden",
                "90 Minuten",
            ),
            "clock_time": (
                "7 Uhr",
                "8 Uhr 30",
                "morgen frueh um 6",
                "heute Abend um 21 Uhr",
                "14 Uhr",
                "16 Uhr 15",
                "morgen um 9",
                "Freitag um 11",
                "naechste Woche Montag um 8",
                "heute Nacht um 23 Uhr",
            ),
            "time_ref": (
                "heute Abend",
                "morgen frueh",
                "naechsten Dienstag",
                "am Wochenende",
                "heute Nachmittag",
                "uebermorgen",
                "am Montagmorgen",
                "jeden Abend um 20 Uhr",
                "naechsten Monat",
                "in einer Stunde",
            ),
            "task": (
                "die Tabletteneinnahme",
                "den Anruf bei meiner Tochter",
                "das Giessen der Blumen",
                "den Arzttermin",
                "das Wasser trinken",
                "die Waesche",
                "die Gymnastikuebungen",
                "das Blutdruckmessen",
                "den Muell",
                "das Abendessen",
                "den Spaziergang",
                "das Laden des Hoergeraets",
            ),
        },
    ),
    R(
        label="tool",
        family_key="print_and_write",
        difficulty="standard",
        templates=(
            T("print", "Druck {item}."),
            T("list", "Schreib {item} auf die Liste."),
            T("note", "Notier dir bitte {item}."),
        ),
        slot_values={
            "item": (
                "meine Einkaufsliste",
                "den Arzttermin fuer Freitag",
                "Milch und Brot",
                "die Telefonnummer vom Pflegedienst",
                "die Rezepte fuer diese Woche",
                "meine Medikamente fuer morgen",
                "den Einkaufszettel fuer Ostern",
                "die Notiz fuer den Hausarzt",
                "die Adresse von Frau Schneider",
                "meine To do Liste fuer heute",
            ),
        },
    ),
    R(
        label="tool",
        family_key="device_control",
        difficulty="standard",
        templates=(
            T("louder", "Mach bitte lauter."),
            T("quieter", "Stell die Lautstaerke leiser."),
            T("play", "Starte {media_item}."),
            T("stop", "Stoppe {media_item}."),
        ),
        slot_values={
            "media_item": (
                "das Radio",
                "meine Abendmusik",
                "die Entspannungsmusik",
                "die Nachrichten",
                "die klassische Musik",
                "das Hoerbuch",
                "die Wettervorhersage",
                "meine Lieblingsplaylist",
                "die Morgenmusik",
                "die Naturgeraesche",
            ),
        },
    ),
    R(
        label="tool",
        family_key="live_state_check",
        difficulty="boundary",
        templates=(
            T("printer", "Pruef, ob der {device} verbunden ist."),
            T("status", "Schau nach, ob {status_item}."),
            T("show", "Zeig mir {system_view}."),
        ),
        slot_values={
            "device": (
                "Drucker",
                "Bildschirm",
                "WLAN Adapter",
                "Lautsprecher",
                "Scanner",
                "Mikrofon",
                "Kopfhoerer",
                "Touchdisplay",
                "Router",
                "Ladestation",
            ),
            "status_item": (
                "noch Erinnerungen offen sind",
                "der Drucker Papier hat",
                "eine neue Nachricht angekommen ist",
                "mein naechster Termin schon eingetragen ist",
                "das WLAN stabil ist",
                "heute noch Medikamente offen sind",
                "die Einkaufsliste leer ist",
                "eine Druckwarteschlange haengt",
                "der Wecker aktiv ist",
                "das Mikrofon verbunden ist",
            ),
            "system_view": (
                "meine naechste Erinnerung",
                "meine offenen Aufgaben",
                "die heutigen Termine",
                "die letzten Notizen",
                "den Status vom Drucker",
                "meine Erinnerungen fuer morgen",
                "die Einkaufsliste",
                "die Wecker fuer diese Woche",
            ),
        },
    ),
    R(
        label="tool",
        family_key="messages_and_actions",
        difficulty="boundary",
        templates=(
            T("message", "Schick {person} die Nachricht {message}."),
            T("call", "Ruf bitte {person} an."),
            T("open", "Oeffne {view}."),
        ),
        slot_values={
            "person": (
                "meine Tochter",
                "meinen Sohn",
                "den Pflegedienst",
                "Frau Schneider",
                "meine Schwester",
                "meinen Enkel",
                "den Nachbarn",
                "den Hausarzt",
                "den Fahrdienst",
                "meine Enkelin",
            ),
            "message": (
                "ich komme spaeter",
                "bitte bring Brot mit",
                "ich denke an dich",
                "der Termin ist bestaetigt",
                "ich bin schon unterwegs",
                "bitte ruf mich spaeter an",
                "ich brauche morgen Hilfe",
                "der Druck ist fertig",
                "wir sehen uns am Sonntag",
                "die Medikamente sind bestellt",
            ),
            "view": (
                "meine Erinnerungen",
                "die Einkaufsliste",
                "die Druckvorschau",
                "die Notizen",
                "den Kalender",
                "die Aufgabenliste",
                "die Nachrichten",
                "die Druckwarteschlange",
            ),
        },
    ),
)

_validate_recipe_registry(_ROUTE_RECIPES)

_ROUTE_RECIPE_BY_FAMILY_KEY: Mapping[str, SyntheticRouteRecipe] = MappingProxyType(
    {recipe.family_key: recipe for recipe in _ROUTE_RECIPES}
)

ROUTE_RECIPES: tuple[SyntheticRouteRecipe, ...] = _ROUTE_RECIPES
ROUTE_RECIPE_BY_FAMILY_KEY: Mapping[str, SyntheticRouteRecipe] = _ROUTE_RECIPE_BY_FAMILY_KEY

__all__ = (
    "REGISTRY_VERSION",
    "ROUTE_RECIPES",
    "ROUTE_RECIPE_BY_FAMILY_KEY",
    "SyntheticRouteTemplate",
    "SyntheticRouteRecipe",
    "SyntheticRouteExample",
    "get_route_recipe",
    "get_route_recipes",
    "iter_route_recipes",
    "iter_bootstrap_examples",
    "build_bootstrap_corpus",
)