"""Shared test-suite warning policy for known third-party noise."""

from __future__ import annotations

import warnings


warnings.filterwarnings(
    "ignore",
    message=r"Passing 'msg' argument to Task\.cancel\(\) is deprecated since Python 3\.11, and scheduled for removal in Python 3\.14\.",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Passing 'msg' argument to Future\.cancel\(\) is deprecated since Python 3\.11, and scheduled for removal in Python 3\.14\.",
    category=DeprecationWarning,
)
