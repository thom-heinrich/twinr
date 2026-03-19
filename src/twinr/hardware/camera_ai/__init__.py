"""Hold the decomposed local AI-camera stack for Twinr hardware code.

Import concrete submodules directly from this package. The package root stays
lightweight on purpose so low-level modules such as ``hand_landmarks.py`` can
depend on shared contracts without pulling the full adapter/runtime graph into
their import path.
"""

__all__: list[str] = []
