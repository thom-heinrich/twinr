from __future__ import annotations

from dataclasses import dataclass

from twinr.web.store import FileBackedSetting


@dataclass(frozen=True, slots=True)
class DashboardCard:
    title: str
    value: str
    detail: str
    href: str


@dataclass(frozen=True, slots=True)
class SettingsSection:
    title: str
    description: str
    fields: tuple[FileBackedSetting, ...]


@dataclass(frozen=True, slots=True)
class IntegrationOverviewRow:
    label: str
    status: str
    summary: str
    detail: str


@dataclass(frozen=True, slots=True)
class DetailMetric:
    label: str
    value: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AdaptiveTimingView:
    enabled: bool
    path: str
    last_updated_label: str
    current_metrics: tuple[DetailMetric, ...]
    counter_metrics: tuple[DetailMetric, ...]
    baseline_metrics: tuple[DetailMetric, ...]
