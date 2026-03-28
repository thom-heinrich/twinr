"""Provide storage adapters for Twinr long-term memory state.

This package contains the concrete JSON-backed stores for durable objects and
midterm packets plus the remote snapshot adapter used in remote-primary mode.
`store.py`, `remote_catalog.py`, and `remote_state.py` stay as compatibility
wrappers while the structured, remote-catalog, and remote-state internals live
in focused split packages.
Import the concrete store classes from their module files or via
``twinr.memory.longterm`` where re-exported.
"""
