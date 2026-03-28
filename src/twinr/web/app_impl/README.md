# app_impl

Internal implementation package for [`twinr.web.app`](../app.py).

`app.py` stays the stable compatibility shim and monkeypatch surface. The real
FastAPI assembly now lives here so route groups remain separated by concern.

## Files

| File | Purpose |
|---|---|
| [`main.py`](./main.py) | Build shared runtime state, create the FastAPI app, and register all route groups |
| [`runtime.py`](./runtime.py) | Shared dataclasses for security config, locks, and wrapper-surface lookups |
| [`compat.py`](./compat.py) | Legacy helper functions extracted from `app.py` without behavior changes |
| [`auth.py`](./auth.py) | Control-plane middleware, error handlers, and managed sign-in routes |
| [`shell.py`](./shell.py) | Home and Advanced page routes |
| [`ops.py`](./ops.py) | `/ops/*` routes, support bundles, self-tests, and debug surfaces |
| [`integrations.py`](./integrations.py) | Integrations, email wizard, provider connect, and WhatsApp wizard routes |
| [`automation.py`](./automation.py) | `/automations` routes |
| [`preferences.py`](./preferences.py) | Voice profile, settings, memory, personality, and user routes |
