"""Shared immutable view models for Twinr's local web control surface.

These dataclasses carry template-ready data between presenters, route handlers,
and templates. Keep them logic-free so the web layer stays predictable and
easy to render safely.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.web.support.store import FileBackedSetting


@dataclass(frozen=True, slots=True)
class DashboardCard:
    """Summarize one dashboard destination card.

    Attributes:
        title: Short card heading shown to operators.
        value: Primary status or metric displayed on the card.
        detail: Secondary explanatory copy under the main value.
        href: Relative link target for the detailed page.
    """

    title: str
    value: str
    detail: str
    href: str


@dataclass(frozen=True, slots=True)
class SettingsSection:
    """Group related settings fields for one page section.

    Attributes:
        title: Section heading shown above the field list.
        description: Plain-language summary of what the section controls.
        fields: File-backed settings rendered inside the section.
    """

    title: str
    description: str
    fields: tuple[FileBackedSetting, ...]


@dataclass(frozen=True, slots=True)
class IntegrationOverviewRow:
    """Describe one integration status row for the overview page.

    Attributes:
        label: Operator-facing integration name.
        status: Short status text such as configured or missing.
        summary: One-line explanation of the current state.
        detail: Secondary detail shown alongside the summary.
    """

    label: str
    status: str
    summary: str
    detail: str


@dataclass(frozen=True, slots=True)
class DetailMetric:
    """Represent one labeled metric/value pair in a detail view.

    Attributes:
        label: Metric name shown to operators.
        value: Current formatted metric value.
        detail: Optional supporting explanation for the metric.
    """

    label: str
    value: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AdaptiveTimingView:
    """Bundle adaptive-timing page data for template rendering.

    Attributes:
        enabled: Whether adaptive timing is currently enabled.
        path: Backing file path shown to operators.
        last_updated_label: Human-readable freshness label for the data file.
        current_metrics: Metrics derived from the latest runtime snapshot.
        counter_metrics: Counter-based metrics aggregated over recent runs.
        baseline_metrics: Baseline timing metrics used for comparison.
    """

    enabled: bool
    path: str
    last_updated_label: str
    current_metrics: tuple[DetailMetric, ...]
    counter_metrics: tuple[DetailMetric, ...]
    baseline_metrics: tuple[DetailMetric, ...]


@dataclass(frozen=True, slots=True)
class WizardCheckRow:
    """Represent one status row inside an operator setup wizard.

    Attributes:
        label: Short check name shown on the left side of the row.
        summary: Operator-facing status summary such as ready or missing.
        detail: Secondary explanation that clarifies the current state.
        status: Template-ready status token such as ok, warn, fail, or muted.
    """

    label: str
    summary: str
    detail: str = ""
    status: str = "muted"


@dataclass(frozen=True, slots=True)
class WizardStep:
    """Describe one template-ready setup step for a server-rendered wizard.

    Attributes:
        key: Stable URL/query key for the step.
        index: Human-readable step number shown in the UI.
        title: Step heading.
        description: Short plain-language explanation of the step goal.
        status: Template-ready status token such as ok, warn, fail, or muted.
        status_label: Short operator-facing label for the current status.
        detail: Supporting copy shown under the step header.
        fields: Optional env-backed fields rendered for this step.
        checks: Optional checklist rows rendered under the step.
        action: Optional POST action token used by the page form.
        action_label: Button label for the step form.
        action_enabled: Whether the main form action should be clickable.
        secondary_action: Optional second POST action token for the same step.
        secondary_action_label: Button label for the optional second step action.
        secondary_action_enabled: Whether the optional second action should be clickable.
        action_hint: Optional plain-language note shown near the action.
        media_src: Optional image or media URL rendered inside the step.
        media_alt: Alternative text for the optional media block.
        media_title: Optional heading shown above the media block.
        media_detail: Optional supporting copy shown under the media block.
        code_block: Optional command or code snippet rendered in a block.
        code_title: Optional heading shown above the code block.
        current: Whether this step is the currently highlighted step.
    """

    key: str
    index: int
    title: str
    description: str
    status: str
    status_label: str
    detail: str = ""
    fields: tuple[FileBackedSetting, ...] = ()
    checks: tuple[WizardCheckRow, ...] = ()
    action: str = ""
    action_label: str = ""
    action_enabled: bool = True
    secondary_action: str = ""
    secondary_action_label: str = ""
    secondary_action_enabled: bool = True
    action_hint: str = ""
    media_src: str = ""
    media_alt: str = ""
    media_title: str = ""
    media_detail: str = ""
    code_block: str = ""
    code_title: str = ""
    current: bool = False
