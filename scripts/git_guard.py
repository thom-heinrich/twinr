#!/usr/bin/env python3
"""Run the repo-local git guard from a versioned support script.

Purpose
-------
Expose one stable CLI entrypoint for installing and running Twinr's git guard.
The heavy lifting lives in ``scripts/git_guard_tool/`` so hook wrappers and
humans can both call the same interface.

Usage
-----
Command-line examples::

    python3 scripts/git_guard.py install
    python3 scripts/git_guard.py scan-staged
    python3 scripts/git_guard.py scan-push --remote origin
"""

from __future__ import annotations


def main() -> int:
    from git_guard_tool.cli import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
