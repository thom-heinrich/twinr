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
