"""Define the canonical Twinr wakeword training plan.

This module does not train or evaluate models directly. Instead it captures
the current repo-approved plan for:

- the Stage-1 phonetic family detector
- hard-negative mining inputs and exclusions
- Pi-faithful acceptance suites and promotion blockers

The plan can be rendered to Markdown so operators and future retrains use one
shared workflow instead of ad-hoc experiment notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_POSITIVE_FAMILIES = ("twinr", "twinna", "twina", "twinner")
DEFAULT_CONFUSION_FAMILIES = ("twin", "winner", "winter", "tina", "timer", "twitter")


@dataclass(frozen=True, slots=True)
class WakewordTrainingCommand:
    """Describe one concrete command in the canonical training workflow."""

    name: str
    purpose: str
    command: str


@dataclass(frozen=True, slots=True)
class WakewordAcceptanceMetricPlan:
    """Describe one promotion metric or blocker for wakeword candidates."""

    name: str
    target: str
    rationale: str
    blocking: bool = True


@dataclass(frozen=True, slots=True)
class WakewordTrainingPlan:
    """Describe Twinr's current canonical wakeword training workflow."""

    stage1_model_name: str
    stage1_phrase_profile: str
    positive_families: tuple[str, ...]
    confusion_families: tuple[str, ...]
    hard_negative_priority_sources: tuple[str, ...]
    holdout_manifests: tuple[str, ...]
    commands: tuple[WakewordTrainingCommand, ...]
    acceptance_metrics: tuple[WakewordAcceptanceMetricPlan, ...]
    promotion_rules: tuple[str, ...]
    references: tuple[str, ...]


def build_default_wakeword_training_plan(*, project_root: str | Path | None = None) -> WakewordTrainingPlan:
    """Build the current research-backed Twinr wakeword training plan."""

    root = Path(project_root).expanduser().resolve(strict=False) if project_root is not None else REPO_ROOT
    stage1_model_name = "twinr_family_stage1_vnext"
    dataset_root = root / "models" / "openwakeword_dataset" / "runs" / stage1_model_name
    output_model = root / "src" / "twinr" / "proactive" / "wakeword" / "models" / f"{stage1_model_name}.onnx"
    baseline_model = root / "src" / "twinr" / "proactive" / "wakeword" / "models" / "twinr_v1.onnx"
    commands = (
        WakewordTrainingCommand(
            name="Build Stage-1 Dataset",
            purpose=(
                "Generate one family detector dataset that keeps Twinna/Twina/Twinner "
                "positive, mines real false activations back into negatives, and excludes "
                "all held-out Pi acceptance sets from training."
            ),
            command=(
                f"PYTHONPATH=src python3 {root / 'scripts' / 'generate_multivoice_dataset.py'} \\\n"
                f"  --model-name {stage1_model_name} \\\n"
                "  --phrase-profile family \\\n"
                "  --extra-positive-dir <real_room_positive_dir> \\\n"
                "  --extra-negative-dir <real_room_negative_dir> \\\n"
                "  --exclude-manifest <critical16_manifest.json> \\\n"
                "  --exclude-manifest <family34_manifest.json> \\\n"
                "  --exclude-manifest <full160_manifest.json> \\\n"
                "  --hard-negative-manifest <pi_room_capture_manifest.json> \\\n"
                f"  --hard-negative-model {baseline_model} \\\n"
                "  --hard-negative-threshold 0.08 \\\n"
                "  --hard-negative-max-per-text 12"
            ),
        ),
        WakewordTrainingCommand(
            name="Train Stage-1 Detector",
            purpose=(
                "Train the new family Stage-1 detector from the generated dataset root and "
                "select its threshold against the held-out acceptance manifest instead of "
                "shipping the raw upstream default."
            ),
            command=(
                f"PYTHONPATH=src python3 -m twinr --env-file {root / '.env'} \\\n"
                "  --wakeword-train-model \\\n"
                f"  --wakeword-dataset-root {dataset_root} \\\n"
                f"  --wakeword-model-output {output_model} \\\n"
                "  --wakeword-manifest <full160_manifest.json> \\\n"
                "  --wakeword-training-rounds 3 \\\n"
                "  --wakeword-training-model-type mlp \\\n"
                "  --wakeword-training-layer-dim 256 \\\n"
                "  --wakeword-training-steps 20000 \\\n"
                "  --wakeword-training-feature-device gpu"
            ),
        ),
        WakewordTrainingCommand(
            name="Evaluate On Pi Holdouts",
            purpose=(
                "Run the candidate on the held-out Pi replay sets and long-form ambient negatives "
                "before any promotion. This step must use the real Twinr runtime evaluation path, "
                "not an optimistic standalone clip sweep."
            ),
            command=(
                f"PYTHONPATH=src python3 -m twinr --env-file {root / '.env'} \\\n"
                "  --wakeword-promotion-eval \\\n"
                "  --wakeword-promotion-spec <stage1_promotion_spec.json>\n"
                "\n"
                "# The spec should contain the held-out labeled suites plus\n"
                "# the long-form ambient false-accepts/hour guard."
            ),
        ),
    )
    acceptance_metrics = (
        WakewordAcceptanceMetricPlan(
            name="critical16_false_negatives",
            target="0 new false negatives; held-out recall must remain 1.0",
            rationale="Critical must-hit Pi phrases are the first promotion blocker.",
        ),
        WakewordAcceptanceMetricPlan(
            name="family34_false_negatives",
            target="0 false negatives across Twinna/Twina/Twinner family holdout",
            rationale="The spoken phonetic family is the real product target, not only literal Twinr.",
        ),
        WakewordAcceptanceMetricPlan(
            name="full160_false_positives",
            target="must stay at or below the current production baseline",
            rationale="Wide holdout regression is not allowed even if critical recall stays high.",
        ),
        WakewordAcceptanceMetricPlan(
            name="ambient_false_accepts_per_hour",
            target="<= 0.20 false accepts/hour on long-form Pi room negatives",
            rationale="Production quality must be measured on long ambient audio, not only short clips.",
        ),
        WakewordAcceptanceMetricPlan(
            name="runtime_faithful_eval_only",
            target="candidate must pass Twinr's real runtime eval path on /twinr",
            rationale="Earlier standalone sweeps overstated acceptance versus the true Pi runtime path.",
        ),
    )
    promotion_rules = (
        "Train only the broad family Stage-1 detector for production candidates: Twinr, Twinna, Twina, Twinner.",
        "Always mine hard negatives from real Pi false activations before another retrain.",
        "Exclude every held-out acceptance manifest from dataset generation to keep promotion honest.",
        "Do not promote a candidate that improves full-set false positives by introducing even one new family false negative.",
        "Treat candidate-family confusion words as first-class negatives: Twin, Winner, Winter, Tina, Timer, Twitter.",
    )
    references = (
        "Google 2017 cascade KWS on mobile devices",
        "Amazon word-level wakeword verification",
        "Amazon wakeword-independent verification",
        "microWakeWord false-accepts/hour deployment guidance",
        "openWakeWord custom verifier guidance",
        "Recent phoneme-aware and confusable-negative user-defined KWS papers",
    )
    return WakewordTrainingPlan(
        stage1_model_name=stage1_model_name,
        stage1_phrase_profile="family",
        positive_families=DEFAULT_POSITIVE_FAMILIES,
        confusion_families=DEFAULT_CONFUSION_FAMILIES,
        hard_negative_priority_sources=(
            "real Pi false activations from wakeword eval and live room captures",
            "confusable-family negatives: Twin, Winner, Winter, Tina, Timer, Twitter",
            "long-form ambient Pi negatives for false-accepts/hour estimation",
        ),
        holdout_manifests=("critical16", "family34", "full160", "ambient_longform"),
        commands=commands,
        acceptance_metrics=acceptance_metrics,
        promotion_rules=promotion_rules,
        references=references,
    )


def render_wakeword_training_plan_markdown(plan: WakewordTrainingPlan) -> str:
    """Render one human-readable Markdown plan for operators and retrains."""

    lines: list[str] = [
        "# Twinr Wakeword Training Plan",
        "",
        "## Stage-1 Target",
        f"- Model stem: `{plan.stage1_model_name}`",
        f"- Phrase profile: `{plan.stage1_phrase_profile}`",
        f"- Positive families: {', '.join(f'`{item}`' for item in plan.positive_families)}",
        f"- Confusion families: {', '.join(f'`{item}`' for item in plan.confusion_families)}",
        "",
        "## Hard-Negative Mining",
    ]
    lines.extend(f"- {item}" for item in plan.hard_negative_priority_sources)
    lines.extend(
        [
            "",
            "## Held-Out Pi Acceptance Suites",
            f"- {', '.join(f'`{item}`' for item in plan.holdout_manifests)}",
            "",
            "## Promotion Blockers",
        ]
    )
    lines.extend(
        f"- `{metric.name}`: {metric.target}. {metric.rationale}"
        for metric in plan.acceptance_metrics
    )
    lines.extend(["", "## Workflow Commands"])
    for command in plan.commands:
        lines.extend(
            [
                f"### {command.name}",
                command.purpose,
                "",
                "```bash",
                command.command,
                "```",
                "",
            ]
        )
    lines.extend(["## Rules"])
    lines.extend(f"- {rule}" for rule in plan.promotion_rules)
    lines.extend(["", "## Research Basis"])
    lines.extend(f"- {reference}" for reference in plan.references)
    lines.append("")
    return "\n".join(lines)
