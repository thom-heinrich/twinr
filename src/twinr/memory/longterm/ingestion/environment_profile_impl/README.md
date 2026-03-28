# environment_profile_impl

Implementation package for the legacy public module
[`environment_profile.py`](../environment_profile.py).

## Layout

- `compiler.py` owns the public compiler dataclass and compile orchestration
- `models.py` owns the smart-home environment dataclasses and memory rendering
- `pipeline.py` owns event extraction, daily marker assembly, and baselines
- `signals.py` owns deviations, quality states, and regime detection helpers
- `helpers.py` and `constants.py` keep the pure utility surface stable

The public import path stays
`twinr.memory.longterm.ingestion.environment_profile`.
