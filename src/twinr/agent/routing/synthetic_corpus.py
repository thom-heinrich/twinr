"""Generate and curate synthetic transcript corpora for the local router.

This module owns Twinr's deterministic bootstrap corpora for the local
two-stage router. It produces broad German utterance families, attaches both
backend-route labels and user-centered labels, applies light transcript-noise
variants, removes duplicates and generation leakage, and exports JSONL that the
bundle builders can consume directly.

Example:

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.synthetic_corpus \
  --output-path artifacts/router/synthetic/router_samples.jsonl \
  --samples-per-label 1024
```
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from typing import Iterable, Mapping, Sequence

from .contracts import ROUTE_LABEL_VALUES, normalize_route_label
from .user_intent import (
    USER_INTENT_LABEL_VALUES,
    default_user_intent_for_route_label,
    normalize_user_intent_label,
)


_REFERENCE_DATE = "2026-03-22"
_SPLIT_TRAIN_RATIO = 0.82
_SPLIT_DEV_RATIO = 0.91
_DEFAULT_SOURCE = "synthetic_router_v2_user_centered"
_LABEL_NAMESPACE_BACKEND = "backend"
_LABEL_NAMESPACE_USER = "user"
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
_GENERATION_LEAKAGE_MARKERS: tuple[str, ...] = (
    "label ",
    "klasse ",
    "route ",
    "routing",
    "router",
    "parametric",
    "web research",
    "memory class",
    "tool call",
    "intent ",
)


@dataclass(frozen=True, slots=True)
class SyntheticRouteTemplate:
    """Describe one utterance surface form for a semantic-router family."""

    key: str
    text: str


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", normalize_route_label(self.label))
        object.__setattr__(
            self,
            "user_label",
            normalize_user_intent_label(
                self.user_label or default_user_intent_for_route_label(self.label)
            ),
        )
        object.__setattr__(self, "family_key", str(self.family_key or "").strip().lower())
        object.__setattr__(self, "difficulty", str(self.difficulty or "standard").strip().lower())
        if not self.family_key:
            raise ValueError("SyntheticRouteRecipe.family_key must not be empty.")
        if not self.templates:
            raise ValueError("SyntheticRouteRecipe.templates must not be empty.")


@dataclass(frozen=True, slots=True)
class SyntheticRouteSample:
    """Store one generated router transcript sample plus provenance metadata."""

    text: str
    label: str
    sample_id: str
    split: str
    user_label: str | None = None
    source: str = _DEFAULT_SOURCE
    family_key: str | None = None
    template_key: str | None = None
    difficulty: str = "standard"
    noise_key: str = "clean"
    reference_date: str = _REFERENCE_DATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", " ".join(str(self.text or "").strip().split()))
        object.__setattr__(self, "label", normalize_route_label(self.label))
        object.__setattr__(
            self,
            "user_label",
            normalize_user_intent_label(
                self.user_label or default_user_intent_for_route_label(self.label)
            ),
        )
        object.__setattr__(self, "sample_id", str(self.sample_id or "").strip())
        object.__setattr__(self, "split", str(self.split or "").strip().lower())
        object.__setattr__(self, "source", str(self.source or _DEFAULT_SOURCE).strip())
        object.__setattr__(self, "family_key", str(self.family_key or "").strip().lower() or None)
        object.__setattr__(self, "template_key", str(self.template_key or "").strip().lower() or None)
        object.__setattr__(self, "difficulty", str(self.difficulty or "standard").strip().lower())
        object.__setattr__(self, "noise_key", str(self.noise_key or "clean").strip().lower())
        object.__setattr__(self, "reference_date", str(self.reference_date or _REFERENCE_DATE).strip())
        if not self.text:
            raise ValueError("SyntheticRouteSample.text must not be empty.")
        if not self.sample_id:
            raise ValueError("SyntheticRouteSample.sample_id must not be empty.")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("SyntheticRouteSample.split must be train/dev/test.")

    def label_for_namespace(self, namespace: str) -> str:
        """Return the label used for one export namespace."""

        normalized_namespace = _normalize_label_namespace(namespace)
        if normalized_namespace == _LABEL_NAMESPACE_USER:
            return str(self.user_label)
        return self.label

    def to_json_dict(self, *, label_namespace: str = _LABEL_NAMESPACE_BACKEND) -> dict[str, str]:
        """Return one JSON-serializable payload for JSONL export."""

        payload = {
            "id": self.sample_id,
            "text": self.text,
            "label": self.label_for_namespace(label_namespace),
            "split": self.split,
            "source": self.source,
            "reference_date": self.reference_date,
        }
        payload["backend_label"] = self.label
        payload["user_label"] = str(self.user_label)
        if self.family_key:
            payload["family"] = self.family_key
        if self.template_key:
            payload["template"] = self.template_key
        if self.difficulty:
            payload["difficulty"] = self.difficulty
        if self.noise_key:
            payload["noise"] = self.noise_key
        return payload


@dataclass(frozen=True, slots=True)
class SyntheticRouteCurationReport:
    """Summarize one synthetic-router generation and curation run."""

    generated_count: int
    kept_count: int
    rejected_exact_duplicates: int
    rejected_near_duplicates: int
    rejected_style_collapses: int
    rejected_generation_leakage: int
    per_label: Mapping[str, int]
    per_split: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-friendly report payload."""

        return {
            "generated_count": int(self.generated_count),
            "kept_count": int(self.kept_count),
            "rejected_exact_duplicates": int(self.rejected_exact_duplicates),
            "rejected_near_duplicates": int(self.rejected_near_duplicates),
            "rejected_style_collapses": int(self.rejected_style_collapses),
            "rejected_generation_leakage": int(self.rejected_generation_leakage),
            "per_label": dict(self.per_label),
            "per_split": dict(self.per_split),
        }


_ROUTE_RECIPES: tuple[SyntheticRouteRecipe, ...] = (
    SyntheticRouteRecipe(
        label="parametric",
        family_key="stable_explanation",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("plain", "Erklaer mir {concept} in einfachen Worten."),
            SyntheticRouteTemplate("question", "Was bedeutet {concept}?"),
            SyntheticRouteTemplate("function", "Wie funktioniert {concept} genau?"),
            SyntheticRouteTemplate("spoken", "Kannst du mir {concept} kurz erklaeren?"),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="stable_comparison",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("difference", "Was ist der Unterschied zwischen {pair}?"),
            SyntheticRouteTemplate("recognize", "Woran erkennt man den Unterschied zwischen {pair}?"),
            SyntheticRouteTemplate("simple", "Vergleich mir {pair} in einfachen Worten."),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="how_to_without_action",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("how_to", "Wie kann man {task}?"),
            SyntheticRouteTemplate("explain_steps", "Erklaer mir Schritt fuer Schritt, wie man {task}."),
            SyntheticRouteTemplate("learn", "Ich moechte lernen, wie man {task}."),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="history_and_people",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("who", "Wer war {person}?"),
            SyntheticRouteTemplate("known_for", "Warum ist {person} bekannt?"),
            SyntheticRouteTemplate("when", "Wann lebte {person}?"),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="translation_and_phrase_help",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("translate", "Wie sagt man {phrase} auf {language}?"),
            SyntheticRouteTemplate("means", "Was heisst {phrase} auf {language}?"),
            SyntheticRouteTemplate("phrase_help", "Uebersetz mir {phrase} in {language}."),
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
            "language": ("Italienisch", "Spanisch", "Englisch", "Franzoesisch", "Portugiesisch", "Niederlaendisch"),
        },
    ),
    SyntheticRouteRecipe(
        label="parametric",
        family_key="stable_short_explanation",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("short", "Erklaer mir kurz {concept}."),
            SyntheticRouteTemplate("simple", "Kannst du mir einfach erklaeren, was {concept} ist?"),
            SyntheticRouteTemplate("spoken", "Sag mir bitte kurz, was {concept} bedeutet."),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="stable_why_questions",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("why", "Warum {phenomenon}?"),
            SyntheticRouteTemplate("wieso", "Wieso {phenomenon}?"),
            SyntheticRouteTemplate("explain_why", "Kannst du mir erklaeren, warum {phenomenon}?"),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="general_tech_and_daily_knowledge",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("works", "Wie funktioniert {thing} eigentlich?"),
            SyntheticRouteTemplate("means", "Was genau ist {thing}?"),
            SyntheticRouteTemplate("learn", "Ich moechte verstehen, wie {thing} funktioniert."),
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
    SyntheticRouteRecipe(
        label="parametric",
        family_key="stable_usage_questions",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("usage", "Wofuer braucht man {thing}?"),
            SyntheticRouteTemplate("purpose", "Wozu ist {thing} gut?"),
            SyntheticRouteTemplate("body", "Was macht {system_part}?"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="current_weather",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("forecast", "Wie wird das Wetter {time_ref} in {place}?"),
            SyntheticRouteTemplate("umbrella", "Brauche ich {time_ref} in {place} einen Schirm?"),
            SyntheticRouteTemplate("temperature", "Wie warm wird es {time_ref} in {place}?"),
        ),
        slot_values={
            "time_ref": ("heute", "morgen", "am Wochenende", "heute Abend", "morgen Mittag", "in der Nacht"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="current_news",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("today_news", "Was ist {time_ref} in {place} passiert?"),
            SyntheticRouteTemplate("latest_topic", "Gibt es {time_ref} Neuigkeiten zu {topic}?"),
            SyntheticRouteTemplate("summary", "Fass mir die aktuellen Meldungen zu {topic} zusammen."),
        ),
        slot_values={
            "time_ref": ("heute", "gerade", "im Moment", "seit heute Morgen", "heute Abend", "in den letzten Stunden"),
            "place": ("Berlin", "Deutschland", "Europa", "Hamburg", "Muenchen", "Brandenburg", "Sachsen", "Nordrhein Westfalen"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="live_schedule_and_opening",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("train", "Wann faehrt der naechste Zug von {origin} nach {destination}?"),
            SyntheticRouteTemplate("open_now", "Hat {place} {time_ref} geoeffnet?"),
            SyntheticRouteTemplate("traffic", "Ist {route} {time_ref} gesperrt?"),
        ),
        slot_values={
            "origin": ("Berlin", "Hamburg", "Koeln", "Leipzig", "Dresden", "Kiel", "Potsdam", "Erfurt"),
            "destination": ("Potsdam", "Hannover", "Dresden", "Bremen", "Magdeburg", "Rostock", "Jena", "Luebeck"),
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
            "time_ref": ("heute", "jetzt", "morgen frueh", "am Sonntag", "heute Nachmittag", "naechste Woche"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="current_price_and_availability",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("price_now", "Wie teuer ist {asset} gerade?"),
            SyntheticRouteTemplate("price_current", "Wie hoch ist aktuell der Preis von {asset}?"),
            SyntheticRouteTemplate("availability", "Ist {product} im Moment lieferbar?"),
        ),
        slot_values={
            "asset": ("Heizoel", "Gold", "Bitcoin", "eine Monatskarte", "Strom", "Gas", "Silber", "Diesel", "Superbenzin", "Fernwaerme"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="safety_lookup",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("meds", "Kann ich {med_a} mit {med_b} zusammen nehmen?"),
            SyntheticRouteTemplate("recall", "Gibt es aktuell Rueckrufe fuer {product}?"),
            SyntheticRouteTemplate("warning", "Gibt es {time_ref} eine Warnung fuer {hazard} in {place}?"),
        ),
        slot_values={
            "med_a": ("Ibuprofen", "Aspirin", "Paracetamol", "Magnesium", "Diclofenac", "Nasenspray", "Johanniskraut", "Vitamin K"),
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
            "hazard": ("Sturm", "Glatteis", "Hochwasser", "starke Hitze", "Waldbrand", "Feinstaub", "starker Regen", "Gewitter"),
            "place": ("Berlin", "Hamburg", "Sachsen", "Brandenburg", "Bayern", "Thueringen", "Hessen", "Rheinland Pfalz"),
        },
    ),
    SyntheticRouteRecipe(
        label="web",
        family_key="current_role_holder",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("derzeit", "Wer ist derzeit {role}?"),
            SyntheticRouteTemplate("aktuell", "Wer ist aktuell {role}?"),
            SyntheticRouteTemplate("amt", "Wer hat im Moment das Amt {role}?"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="live_topic_status",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("what_now", "Was ist aktuell mit {topic} los?"),
            SyntheticRouteTemplate("situation", "Wie ist momentan die Lage bei {topic}?"),
            SyntheticRouteTemplate("new", "Gibt es derzeit etwas Neues zu {topic}?"),
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
    SyntheticRouteRecipe(
        label="web",
        family_key="mutable_role_plain_question",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("plain_role", "Wer ist {role}?"),
            SyntheticRouteTemplate("holds_office", "Wer hat das Amt {role}?"),
            SyntheticRouteTemplate("leads_org", "Wer leitet {institution}?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="personal_fact",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("name", "Wie heisst {relation}?"),
            SyntheticRouteTemplate("birthday", "Wann hat {relation} Geburtstag?"),
            SyntheticRouteTemplate("doctor", "Wie heisst mein {role}?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="preferences",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("favorite", "Welchen {item} mag ich am liebsten?"),
            SyntheticRouteTemplate("music", "Welche Musik hoere ich gern, wenn ich unruhig bin?"),
            SyntheticRouteTemplate("routine", "Welchen Tee trinke ich abends am liebsten?"),
        ),
        slot_values={
            "item": ("Kuchen", "Tee", "Saft", "Pullover", "Filmgenre", "Suppe", "Marmelade", "Musikrichtung", "Kissen", "Schal", "Buchgenre", "Brot"),
        },
    ),
    SyntheticRouteRecipe(
        label="memory",
        family_key="household_location",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("where", "Wo haben wir {object_name} hingelegt?"),
            SyntheticRouteTemplate("kept", "Wo liegt {object_name} normalerweise?"),
            SyntheticRouteTemplate("stored", "In welchem Schrank bewahren wir {object_name} auf?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="prior_conversation",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("remember_plan", "Was hatten wir ueber {topic} festgehalten?"),
            SyntheticRouteTemplate("reminder_context", "Woran wollte ich mich {time_ref} erinnern?"),
            SyntheticRouteTemplate("trip_plan", "Welchen Plan hatten wir fuer {event}?"),
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
            "time_ref": ("morgen", "naechste Woche", "am Wochenende", "im April", "im Mai", "naechsten Monat", "an Weihnachten", "uebermorgen"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="personal_health_notes",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("allergies", "Welche Allergien habe ich?"),
            SyntheticRouteTemplate("meds", "Welche Tabletten nehme ich morgens?"),
            SyntheticRouteTemplate("notes", "Was steht in meinen Notizen zu {topic}?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="contact_details",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("phone", "Welche Telefonnummer hat {contact}?"),
            SyntheticRouteTemplate("address", "Welche Adresse habe ich fuer {contact} gespeichert?"),
            SyntheticRouteTemplate("details", "Welche Kontaktdaten habe ich zu {contact} notiert?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="personal_routines",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("when_usually", "Wann mache ich normalerweise {routine}?"),
            SyntheticRouteTemplate("day", "An welchem Tag ist bei mir {routine} dran?"),
            SyntheticRouteTemplate("time", "Zu welcher Uhrzeit mache ich meistens {routine}?"),
        ),
        slot_values={
            "routine": (
                "die Gymnastik",
                "den Wocheneinkauf",
                "das Blumen giessen",
                "den Blutdruck",
                "den Anruf bei meiner Tochter",
                "das Medikamente sortieren",
                "den Spaziergang",
                "das Tagebuch schreiben",
                "das Hoergeraet Laden",
                "den Muell rausbringen",
                "das Bett beziehen",
                "den Chorabend",
            ),
        },
    ),
    SyntheticRouteRecipe(
        label="memory",
        family_key="known_people_preferences",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("likes", "Was mag {person} besonders gern?"),
            SyntheticRouteTemplate("prefers", "Worueber freut sich {person} am meisten?"),
            SyntheticRouteTemplate("notes", "Was hatten wir ueber {person} festgehalten?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="short_known_people_preferences",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("short_like", "Was mag {person}?"),
            SyntheticRouteTemplate("likes", "Was mag {person} gern?"),
            SyntheticRouteTemplate("happy", "Worueber freut sich {person}?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="remembered_people_context",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("about_person", "Was hatten wir ueber {person} besprochen?"),
            SyntheticRouteTemplate("notes", "Was weiss ich ueber {person}?"),
            SyntheticRouteTemplate("stored", "Was hatten wir uns zu {person} notiert?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="person_reason_memory",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("call_reason", "Warum hat {person} angerufen?"),
            SyntheticRouteTemplate("visit_reason", "Warum kommt {person} zu mir?"),
            SyntheticRouteTemplate("wanted_reason", "Weshalb wollte {person} mich sprechen?"),
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
    SyntheticRouteRecipe(
        label="memory",
        family_key="personal_reason_context",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("why_visit", "Warum kommt {person} zu mir?"),
            SyntheticRouteTemplate("why_call", "Warum wollte {person} mich anrufen?"),
            SyntheticRouteTemplate("why_note", "Weshalb hatten wir {topic} fuer mich festgehalten?"),
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
    SyntheticRouteRecipe(
        label="tool",
        user_label="persoenlich",
        family_key="personal_schedule_lookup",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("appointments", "Was habe ich {time_ref} fuer Termine?"),
            SyntheticRouteTemplate("calendar", "Was steht {time_ref} in meinem Kalender?"),
            SyntheticRouteTemplate("visit", "Kommt {time_ref} jemand zu mir?"),
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
    SyntheticRouteRecipe(
        label="tool",
        user_label="persoenlich",
        family_key="personal_pending_state",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("meds_open", "Habe ich {time_ref} noch Medikamente offen?"),
            SyntheticRouteTemplate("reminders", "Welche Erinnerung habe ich {time_ref}?"),
            SyntheticRouteTemplate("tasks", "Welche Aufgaben habe ich {time_ref} noch offen?"),
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
    SyntheticRouteRecipe(
        label="tool",
        user_label="persoenlich",
        family_key="personal_day_overview",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("today", "Was ist {time_ref} bei mir los?"),
            SyntheticRouteTemplate("for_me", "Was steht {time_ref} fuer mich an?"),
            SyntheticRouteTemplate("overview", "Was habe ich {time_ref} noch vor?"),
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
    SyntheticRouteRecipe(
        label="tool",
        user_label="persoenlich",
        family_key="personal_short_schedule_status",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("have_plan", "Hab ich {time_ref} was vor?"),
            SyntheticRouteTemplate("planned", "Ist {time_ref} bei mir was geplant?"),
            SyntheticRouteTemplate("coming_up", "Steht {time_ref} bei mir etwas an?"),
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
    SyntheticRouteRecipe(
        label="tool",
        user_label="persoenlich",
        family_key="care_support_visits",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("when_arrive", "Wann kommt {visitor} zu mir?"),
            SyntheticRouteTemplate("is_scheduled", "Ist {visitor} {time_ref} bei mir eingetragen?"),
            SyntheticRouteTemplate("coming", "Kommt {visitor} {time_ref} zu mir?"),
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
    SyntheticRouteRecipe(
        label="tool",
        family_key="house_state_snapshot",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("house_now", "Was ist im Haus los?"),
            SyntheticRouteTemplate("doors", "Ist {door} gerade offen?"),
            SyntheticRouteTemplate("room_state", "Wie ist der Status im {room}?"),
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
    SyntheticRouteRecipe(
        label="tool",
        family_key="home_environment_state",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("temperature", "Wie warm ist es gerade im {room}?"),
            SyntheticRouteTemplate("light", "Ist im {room} noch Licht an?"),
            SyntheticRouteTemplate("appliance", "Ist {appliance} gerade an?"),
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
    SyntheticRouteRecipe(
        label="tool",
        family_key="doorbell_and_presence_check",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("doorbell", "Hat es gerade geklingelt?"),
            SyntheticRouteTemplate("door", "War jemand an der Haustuer?"),
            SyntheticRouteTemplate("arrival", "Ist eben jemand gekommen?"),
        ),
        slot_values={},
    ),
    SyntheticRouteRecipe(
        label="tool",
        family_key="home_scene_control",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("lights", "Mach {device} bitte {state}."),
            SyntheticRouteTemplate("switch", "Schalte {device} {state}."),
            SyntheticRouteTemplate("set", "Stell {device} auf {state}."),
        ),
        slot_values={
            "device": (
                "das Licht im Wohnzimmer",
                "die Lampe im Flur",
                "den Fernseher",
                "die Heizung im Bad",
                "den Ventilator",
                "die Steckdose an der Kaffeemaschine",
                "das Radio",
                "die Nachttischlampe",
            ),
            "state": (
                "an",
                "aus",
                "leiser",
                "lauter",
                "waermer",
                "kuehler",
                "niedriger",
                "hoeher",
            ),
        },
    ),
    SyntheticRouteRecipe(
        label="tool",
        family_key="timers_and_reminders",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("timer", "Stell einen Timer auf {duration}."),
            SyntheticRouteTemplate("alarm", "Mach mir einen Wecker fuer {clock_time}."),
            SyntheticRouteTemplate("remind", "Erinnere mich {time_ref} an {task}."),
        ),
        slot_values={
            "duration": ("5 Minuten", "10 Minuten", "15 Minuten", "20 Minuten", "25 Minuten", "30 Minuten", "45 Minuten", "eine Stunde", "zwei Stunden", "90 Minuten"),
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
                "die Tabletten",
                "den Anruf bei meiner Tochter",
                "das Blumen giessen",
                "den Arzttermin",
                "das Wasser trinken",
                "die Waesche",
                "die Gymnastikuebungen",
                "den Blutdruck",
                "den Muell",
                "das Abendessen",
                "den Spaziergang",
                "das Hoergeraet Laden",
            ),
        },
    ),
    SyntheticRouteRecipe(
        label="tool",
        family_key="print_and_write",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("print", "Druck {item}."),
            SyntheticRouteTemplate("list", "Schreib {item} auf die Liste."),
            SyntheticRouteTemplate("note", "Notier dir bitte {item}."),
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
    SyntheticRouteRecipe(
        label="tool",
        family_key="device_control",
        difficulty="standard",
        templates=(
            SyntheticRouteTemplate("louder", "Mach bitte lauter."),
            SyntheticRouteTemplate("quieter", "Stell die Lautstaerke leiser."),
            SyntheticRouteTemplate("play", "Starte {media_item}."),
            SyntheticRouteTemplate("stop", "Stoppe {media_item}."),
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
    SyntheticRouteRecipe(
        label="tool",
        family_key="live_state_check",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("printer", "Pruef, ob der {device} verbunden ist."),
            SyntheticRouteTemplate("status", "Schau nach, ob {status_item}."),
            SyntheticRouteTemplate("show", "Zeig mir {system_view}."),
        ),
        slot_values={
            "device": ("Drucker", "Bildschirm", "WLAN Adapter", "Lautsprecher", "Scanner", "Mikrofon", "Kopfhoerer", "Touchdisplay", "Router", "Ladestation"),
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
    SyntheticRouteRecipe(
        label="tool",
        family_key="messages_and_actions",
        difficulty="boundary",
        templates=(
            SyntheticRouteTemplate("message", "Schick {person} die Nachricht {message}."),
            SyntheticRouteTemplate("call", "Ruf bitte {person} an."),
            SyntheticRouteTemplate("open", "Oeffne {view}."),
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


def generate_synthetic_route_samples(
    *,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Generate one curated synthetic dataset for the local semantic router."""

    normalized_namespace = _normalize_label_namespace(label_namespace)
    label_values = _label_values_for_namespace(normalized_namespace)
    target_per_label = max(1, int(samples_per_label))
    raw_target_per_label = max(target_per_label, int(round(target_per_label * float(oversample_factor))))
    rng = random.Random(seed)
    recipes_by_label: dict[str, list[SyntheticRouteRecipe]] = defaultdict(list)
    for recipe in _ROUTE_RECIPES:
        recipes_by_label[_recipe_label_for_namespace(recipe, normalized_namespace)].append(recipe)
    raw_samples: list[SyntheticRouteSample] = []
    raw_counts = Counter()
    recipe_cycles = {
        label: _shuffled_cycle(recipes_by_label[label], rng=rng)
        for label in label_values
    }
    max_attempts = raw_target_per_label * len(label_values) * 50
    attempts = 0
    while any(raw_counts[label] < raw_target_per_label for label in label_values):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Synthetic route generation exhausted its attempt budget before reaching the requested size."
            )
        for label in label_values:
            if raw_counts[label] >= raw_target_per_label:
                continue
            recipe = next(recipe_cycles[label])
            raw_samples.append(_render_recipe_sample(recipe, rng=rng))
            raw_counts[label] += 1
    curated, report = curate_synthetic_route_samples(
        raw_samples,
        target_per_label=target_per_label,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
        label_namespace=normalized_namespace,
    )
    return curated, report


def generate_synthetic_user_intent_samples(
    *,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Generate one curated synthetic dataset labeled by user-centered intent."""

    return generate_synthetic_route_samples(
        samples_per_label=samples_per_label,
        seed=seed,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
        label_namespace=_LABEL_NAMESPACE_USER,
    )


def curate_synthetic_route_samples(
    samples: Sequence[SyntheticRouteSample],
    *,
    target_per_label: int | None = None,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Deduplicate and sanitize synthetic router samples."""

    normalized_namespace = _normalize_label_namespace(label_namespace)
    label_values = _label_values_for_namespace(normalized_namespace)
    requested_per_label = None if target_per_label is None else max(1, int(target_per_label))
    kept: list[SyntheticRouteSample] = []
    exact_seen: set[str] = set()
    shape_counts: Counter[tuple[str, str | None, str | None]] = Counter()
    per_label_counts = Counter()
    accepted_signatures: dict[tuple[str, str | None], list[tuple[frozenset[str], frozenset[str]]]] = defaultdict(list)
    rejected_exact_duplicates = 0
    rejected_near_duplicates = 0
    rejected_style_collapses = 0
    rejected_generation_leakage = 0
    for sample in samples:
        sample_label = sample.label_for_namespace(normalized_namespace)
        if requested_per_label is not None and per_label_counts[sample_label] >= requested_per_label:
            continue
        normalized_text = _normalize_for_dedupe(sample.text)
        if normalized_text in exact_seen:
            rejected_exact_duplicates += 1
            continue
        if _looks_like_generation_leakage(normalized_text):
            rejected_generation_leakage += 1
            continue
        bucket_key = (sample_label, sample.family_key, sample.template_key)
        if shape_counts[bucket_key] >= max_samples_per_template_bucket:
            rejected_style_collapses += 1
            continue
        token_signature = frozenset(_tokenize(normalized_text))
        trigram_signature = frozenset(_char_trigrams(normalized_text))
        family_signatures = accepted_signatures[(sample_label, sample.family_key)]
        if any(
            max(
                _jaccard_similarity(token_signature, existing_tokens),
                _jaccard_similarity(trigram_signature, existing_trigrams),
            ) >= max_near_duplicate_similarity
            for existing_tokens, existing_trigrams in family_signatures
        ):
            rejected_near_duplicates += 1
            continue
        exact_seen.add(normalized_text)
        shape_counts[bucket_key] += 1
        per_label_counts[sample_label] += 1
        family_signatures.append((token_signature, trigram_signature))
        kept.append(sample)
        if requested_per_label is not None and all(per_label_counts[label] >= requested_per_label for label in label_values):
            break
    if requested_per_label is not None and any(per_label_counts[label] < requested_per_label for label in label_values):
        raise RuntimeError(
            "Synthetic route curation could not keep enough samples for every label. "
            f"Counts={dict(per_label_counts)} requested_per_label={requested_per_label}"
        )
    split_counts = Counter(sample.split for sample in kept)
    report = SyntheticRouteCurationReport(
        generated_count=len(samples),
        kept_count=len(kept),
        rejected_exact_duplicates=rejected_exact_duplicates,
        rejected_near_duplicates=rejected_near_duplicates,
        rejected_style_collapses=rejected_style_collapses,
        rejected_generation_leakage=rejected_generation_leakage,
        per_label={label: int(per_label_counts.get(label, 0)) for label in label_values},
        per_split={split: int(split_counts.get(split, 0)) for split in ("train", "dev", "test")},
    )
    return tuple(kept), report


def write_synthetic_route_samples_jsonl(
    samples: Iterable[SyntheticRouteSample],
    path: str | Path,
    *,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> Path:
    """Write synthetic route samples to JSONL on disk."""

    output_path = Path(path).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(sample.to_json_dict(label_namespace=label_namespace), ensure_ascii=True)
        for sample in samples
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_synthetic_user_intent_samples_jsonl(
    samples: Iterable[SyntheticRouteSample],
    path: str | Path,
) -> Path:
    """Write synthetic samples to JSONL with user-centered labels."""

    return write_synthetic_route_samples_jsonl(
        samples,
        path,
        label_namespace=_LABEL_NAMESPACE_USER,
    )


def _render_recipe_sample(
    recipe: SyntheticRouteRecipe,
    *,
    rng: random.Random,
) -> SyntheticRouteSample:
    template = rng.choice(recipe.templates)
    slot_payload = {
        name: rng.choice(tuple(values))
        for name, values in dict(recipe.slot_values).items()
    }
    rendered = " ".join(template.text.format(**slot_payload).split())
    noise_key = rng.choice(recipe.noise_pool)
    noisy_text = _apply_noise_profile(rendered, noise_key=noise_key, rng=rng)
    normalized_text = " ".join(noisy_text.split())
    sample_key = "|".join(
        (
            recipe.label,
            recipe.family_key,
            template.key,
            recipe.difficulty,
            noise_key,
            normalized_text,
        )
    )
    sample_id = hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:16]
    split = _hash_to_split(sample_id)
    return SyntheticRouteSample(
        text=normalized_text,
        label=recipe.label,
        sample_id=f"{recipe.label}_{sample_id}",
        split=split,
        user_label=recipe.user_label,
        family_key=recipe.family_key,
        template_key=template.key,
        difficulty=recipe.difficulty,
        noise_key=noise_key,
    )


def _normalize_label_namespace(value: object) -> str:
    """Return one validated synthetic-label namespace identifier."""

    normalized = str(value or _LABEL_NAMESPACE_BACKEND).strip().lower()
    if normalized not in {_LABEL_NAMESPACE_BACKEND, _LABEL_NAMESPACE_USER}:
        raise ValueError(f"Unsupported synthetic label namespace: {value!r}")
    return normalized


def _label_values_for_namespace(namespace: str) -> tuple[str, ...]:
    """Return the target label set for one synthetic export namespace."""

    normalized_namespace = _normalize_label_namespace(namespace)
    if normalized_namespace == _LABEL_NAMESPACE_USER:
        return USER_INTENT_LABEL_VALUES
    return ROUTE_LABEL_VALUES


def _recipe_label_for_namespace(recipe: SyntheticRouteRecipe, namespace: str) -> str:
    """Return the recipe label for one synthetic export namespace."""

    normalized_namespace = _normalize_label_namespace(namespace)
    if normalized_namespace == _LABEL_NAMESPACE_USER:
        return str(recipe.user_label)
    return recipe.label


def _shuffled_cycle(
    items: Sequence[SyntheticRouteRecipe],
    *,
    rng: random.Random,
):
    """Yield items in repeatedly shuffled order to avoid family collapse."""

    pool = list(items)
    if not pool:
        raise ValueError("Synthetic recipe pools must not be empty.")
    shuffled: list[SyntheticRouteRecipe] = []
    while True:
        if not shuffled:
            shuffled = list(pool)
            rng.shuffle(shuffled)
        yield shuffled.pop()


def _apply_noise_profile(
    text: str,
    *,
    noise_key: str,
    rng: random.Random,
) -> str:
    normalized_key = str(noise_key or "clean").strip().lower()
    if normalized_key == "clean":
        return text
    if normalized_key == "lowercase":
        return text.lower()
    if normalized_key == "no_punct":
        return _strip_simple_punctuation(text).lower()
    if normalized_key == "umlaut_flat":
        flattened = (
            text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "Ae")
            .replace("Ö", "Oe")
            .replace("Ü", "Ue")
            .replace("ß", "ss")
        )
        return _strip_simple_punctuation(flattened).lower()
    if normalized_key == "filler":
        prefix = rng.choice(("aeh ", "hm ", "also ", "sag mal "))
        return _strip_simple_punctuation(f"{prefix}{text}").lower()
    raise ValueError(f"Unsupported synthetic transcript noise profile: {noise_key!r}")


def _hash_to_split(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).digest()[0]
    ratio = float(digest) / 255.0
    if ratio < _SPLIT_TRAIN_RATIO:
        return "train"
    if ratio < _SPLIT_DEV_RATIO:
        return "dev"
    return "test"


def _looks_like_generation_leakage(normalized_text: str) -> bool:
    return any(marker in normalized_text for marker in _GENERATION_LEAKAGE_MARKERS)


def _normalize_for_dedupe(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    cleaned = "".join(character if character.isalnum() or character.isspace() else " " for character in normalized)
    return " ".join(cleaned.split())


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.split() if token)


def _char_trigrams(text: str) -> tuple[str, ...]:
    padded = f"  {text}  "
    if len(padded) < 3:
        return (padded,)
    return tuple(padded[index:index + 3] for index in range(len(padded) - 2))


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _strip_simple_punctuation(text: str) -> str:
    return text.translate(str.maketrans({character: " " for character in ".,!?;:\"'"}))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic JSONL corpus for the Twinr semantic router.")
    parser.add_argument("--output-path", required=True, help="Destination JSONL path")
    parser.add_argument("--samples-per-label", type=int, default=1024, help="Number of kept samples per route label")
    parser.add_argument("--seed", type=int, default=20260322, help="Deterministic generation seed")
    parser.add_argument(
        "--label-namespace",
        default=_LABEL_NAMESPACE_BACKEND,
        choices=(_LABEL_NAMESPACE_BACKEND, _LABEL_NAMESPACE_USER),
        help="Whether JSONL labels should target backend routes or user-centered intents",
    )
    parser.add_argument("--oversample-factor", type=float, default=1.7, help="Raw generation multiplier before curation")
    parser.add_argument(
        "--max-near-duplicate-similarity",
        type=float,
        default=0.92,
        help="Jaccard similarity threshold for near-duplicate rejection",
    )
    parser.add_argument(
        "--max-samples-per-template-bucket",
        type=int,
        default=40,
        help="Cap for one label/family/template bucket before style-collapse rejection",
    )
    parser.add_argument("--report-path", default=None, help="Optional JSON summary path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for synthetic router-dataset generation."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    samples, report = generate_synthetic_route_samples(
        samples_per_label=args.samples_per_label,
        seed=args.seed,
        oversample_factor=args.oversample_factor,
        max_near_duplicate_similarity=args.max_near_duplicate_similarity,
        max_samples_per_template_bucket=args.max_samples_per_template_bucket,
        label_namespace=args.label_namespace,
    )
    output_path = write_synthetic_route_samples_jsonl(
        samples,
        args.output_path,
        label_namespace=args.label_namespace,
    )
    if args.report_path:
        report_path = Path(args.report_path).expanduser().resolve(strict=False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **report.to_dict()}, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
