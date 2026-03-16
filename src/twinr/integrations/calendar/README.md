# calendar

`calendar` owns Twinr's read-only calendar integration package. It provides
the canonical event model, ICS parsing and loading helpers, and the adapter
surface that turns integration requests into agenda responses.

## Responsibility

`calendar` owns:
- define the canonical calendar event model for overlap checks and serialization
- parse ICS payloads into bounded, timezone-aware event records
- translate supported calendar operations into read-only `IntegrationResult` payloads

`calendar` does **not** own:
- integration manifest registration or managed runtime wiring in [../runtime.py](../runtime.py)
- source validation or remote/local calendar policy outside this package
- web form parsing and operator-page rendering in [../../web/viewmodels_integrations.py](../../web/viewmodels_integrations.py)

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public calendar exports |
| [models.py](./models.py) | Event model and overlap rules |
| [adapter.py](./adapter.py) | Read-only agenda adapter |
| [ics.py](./ics.py) | ICS parser and source |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from datetime import UTC, datetime

from twinr.integrations import IntegrationRequest, manifest_for_id
from twinr.integrations.calendar import ICSCalendarSource, ReadOnlyCalendarAdapter

manifest = manifest_for_id("calendar_agenda")
source = ICSCalendarSource.from_path("state/calendar.ics")
adapter = ReadOnlyCalendarAdapter(
    manifest=manifest,
    calendar_reader=source,
    clock=lambda: datetime(2026, 3, 13, 8, 0, tzinfo=UTC),
)
result = adapter.execute(
    IntegrationRequest(
        integration_id="calendar_agenda",
        operation_id="read_today",
    )
)
```

```python
from twinr.integrations.calendar import parse_ics_events

events = parse_ics_events(ics_text)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [managed integrations runtime](../runtime.py)
- [integrations package](../__init__.py)
