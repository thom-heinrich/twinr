"""Build Twinr Control shell navigation and top-level hub card content."""

from __future__ import annotations

from typing import Any

from twinr.web.support.contracts import DashboardCard

_PRIMARY_NAV_GROUPS: tuple[tuple[str, str, str, frozenset[str]], ...] = (
    ("home", "Home", "/", frozenset({"dashboard", "home"})),
    ("activity", "Activity", "/memory", frozenset({"memory", "activity"})),
    ("automations", "Automations", "/automations", frozenset({"automations"})),
    (
        "settings",
        "Settings",
        "/settings",
        frozenset({"settings", "connect", "integrations", "voice_profile", "personality", "user"}),
    ),
    (
        "advanced",
        "Advanced",
        "/advanced",
        frozenset(
            {
                "advanced",
                "ops_self_coding",
                "ops_self_test",
                "ops_devices",
                "ops_debug",
                "ops_usage",
                "ops_health",
                "ops_logs",
                "ops_config",
                "ops_support",
            }
        ),
    ),
)


def _nav_items(active_page: str | None = None) -> tuple[tuple[str, str, str, bool], ...]:
    """Return the grouped primary shell navigation for Twinr Control."""

    normalized_active_page = str(active_page or "").strip()
    return tuple(
        (key, label, href, normalized_active_page in group_members)
        for key, label, href, group_members in _PRIMARY_NAV_GROUPS
    )


def build_home_destination_cards(
    *,
    snapshot: Any,
    pending_reminders_count: int,
    delivered_reminders_count: int,
    next_reminder_label: str | None,
    health_snapshot: Any,
    checks_summary: dict[str, int],
) -> tuple[DashboardCard, ...]:
    """Return the simplified home destination cards for the flatter shell."""

    reminders_detail = next_reminder_label or f"{delivered_reminders_count} delivered recently"
    advanced_value = f"{checks_summary.get('fail', 0)} fail · {checks_summary.get('warn', 0)} warn"
    return (
        DashboardCard(
            title="Activity",
            value=f"{getattr(snapshot, 'memory_count', 0)} turns",
            detail="Conversation context, reminders, and recent memory",
            href="/memory",
        ),
        DashboardCard(
            title="Reminders",
            value=f"{pending_reminders_count} pending",
            detail=reminders_detail,
            href="/memory",
        ),
        DashboardCard(
            title="Automations",
            value="Schedules and triggers",
            detail="Daily routines, sensor rules, and bounded actions",
            href="/automations",
        ),
        DashboardCard(
            title="Settings",
            value="Device and profile",
            detail="Voice, connect, integrations, personality, and user",
            href="/settings",
        ),
        DashboardCard(
            title="Advanced",
            value=advanced_value,
            detail=f"{str(getattr(health_snapshot, 'status', 'unknown')).upper()} health and support tools",
            href="/advanced",
        ),
    )


def build_settings_shortcut_cards() -> tuple[DashboardCard, ...]:
    """Return quick links to secondary setup areas grouped under Settings."""

    return (
        DashboardCard(
            title="Connect",
            value="Channels and pairing",
            detail="WhatsApp self-chat setup and channel routing",
            href="/connect",
        ),
        DashboardCard(
            title="Integrations",
            value="Email, calendar, smart home",
            detail="Manage external services and import consent",
            href="/integrations",
        ),
        DashboardCard(
            title="Voice profile",
            value="Local familiar-speaker match",
            detail="Enroll, verify, or reset the local profile",
            href="/voice-profile",
        ),
        DashboardCard(
            title="Personality",
            value="Tone and behavior",
            detail="Keep Twinr calm, short, and consistent",
            href="/personality",
        ),
        DashboardCard(
            title="User profile",
            value="Facts and preferences",
            detail="Short user context Twinr can use quietly",
            href="/user",
        ),
    )


def build_advanced_hub_page_context(
    *,
    checks_summary: dict[str, int],
    usage_summary: Any,
    health_snapshot: Any,
    self_coding_status: Any,
) -> dict[str, Any]:
    """Return the template-ready content for the grouped Advanced hub page."""

    self_coding_value = (
        self_coding_status.card_value()
        if bool(getattr(self_coding_status, "has_activity", False))
        else "No live rollouts"
    )
    self_coding_detail = (
        self_coding_status.card_detail()
        if bool(getattr(self_coding_status, "has_activity", False))
        else "Compile telemetry, skill health, rollback, and cleanup"
    )
    featured_cards = (
        DashboardCard(
            title="System health",
            value=str(getattr(health_snapshot, "status", "unknown")).upper(),
            detail="CPU, memory, disk, watchdog, and runtime heartbeat",
            href="/ops/health",
        ),
        DashboardCard(
            title="Config checks",
            value=f"{checks_summary.get('fail', 0)} fail · {checks_summary.get('warn', 0)} warn",
            detail="Guardrails, missing secrets, and configuration drift",
            href="/ops/config",
        ),
        DashboardCard(
            title="LLM usage",
            value=f"{getattr(usage_summary, 'requests_total', 0)} req",
            detail=f"{getattr(usage_summary, 'total_tokens', 0)} tok in 24h",
            href="/ops/usage",
        ),
        DashboardCard(
            title="Self-coding",
            value=self_coding_value,
            detail=self_coding_detail,
            href="/ops/self-coding",
        ),
    )
    sections = (
        {
            "title": "Health and recovery",
            "description": "Use these pages when Twinr needs checking, repair, or a support handoff.",
            "cards": (
                DashboardCard(
                    title="System health",
                    value="Runtime and watchdogs",
                    detail="Current device health, remote-memory watchdog, and host metrics",
                    href="/ops/health",
                ),
                DashboardCard(
                    title="Config checks",
                    value="Safe-to-run guardrails",
                    detail="Review missing values, warnings, and fail-closed checks",
                    href="/ops/config",
                ),
                DashboardCard(
                    title="Hardware self-test",
                    value="Buttons, audio, printer",
                    detail="Run bounded local checks against the real device",
                    href="/ops/self-test",
                ),
                DashboardCard(
                    title="Support bundle",
                    value="Collect local evidence",
                    detail="Build a bounded support artifact for later review",
                    href="/ops/support",
                ),
            ),
        },
        {
            "title": "Operator tools",
            "description": "Use these pages for diagnostics, live traces, usage review, and learned-skill operations.",
            "cards": (
                DashboardCard(
                    title="Debug view",
                    value="Runtime, memory, tools",
                    detail="Open the tabbed operator view for deep inspection",
                    href="/ops/debug",
                ),
                DashboardCard(
                    title="Ops logs",
                    value="Recent events",
                    detail="Review local warnings, errors, and runtime events",
                    href="/ops/logs",
                ),
                DashboardCard(
                    title="LLM usage",
                    value="Requests and tokens",
                    detail="Review 24-hour usage totals and latest model activity",
                    href="/ops/usage",
                ),
                DashboardCard(
                    title="Self-coding",
                    value="Learned skills",
                    detail="Track compile runs, health, watchdogs, rollback, and cleanup",
                    href="/ops/self-coding",
                ),
            ),
        },
        {
            "title": "Devices",
            "description": "Use this area to inspect the device surface Twinr sees locally.",
            "cards": (
                DashboardCard(
                    title="Devices",
                    value="Hardware overview",
                    detail="Buttons, audio path, printer, display, and detected peripherals",
                    href="/ops/devices",
                ),
            ),
        },
    )
    return {
        "featured_cards": featured_cards,
        "sections": sections,
    }


__all__ = [
    "_nav_items",
    "build_advanced_hub_page_context",
    "build_home_destination_cards",
    "build_settings_shortcut_cards",
]
