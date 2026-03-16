# self_coding

`self_coding` owns the first focused package slice of Twinr's Adaptive Skill
Engine (ASE). The current scope now covers the deterministic front-stage core:
versioned contracts, status enums, a file-backed runtime store, the MVP
capability registry, the feasibility checker, and the requirements-dialogue
state machine.

## Responsibility

`self_coding` owns:
- the stable internal contract objects used across future ASE layers
- the status model for capabilities, compile jobs, artifacts, and learned skills
- the file-backed store rooted at `state/self_coding`
- the deterministic capability registry for the MVP module set
- the deterministic feasibility checker for Phase 1
- the persistent requirements-dialogue flow for Phase 2

`self_coding` does **not** own:
- workflow-loop orchestration or runtime hooks
- OpenAI/Codex driver execution
- compile-worker prompting or artifact generation
- sandboxed skill execution and brokered effects

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [contracts.py](./contracts.py) | Versioned ASE contract dataclasses |
| [status.py](./status.py) | Shared ASE enums and lifecycle states |
| [store.py](./store.py) | File-backed store under `state/self_coding` |
| [capability_registry.py](./capability_registry.py) | Deterministic MVP capability readiness view |
| [feasibility.py](./feasibility.py) | Deterministic Phase-1 feasibility checker |
| [requirements_dialogue.py](./requirements_dialogue.py) | Deterministic Phase-2 dialogue state machine |
| [learning_flow.py](./learning_flow.py) | Thin orchestration of feasibility plus dialogue persistence |
| [component.yaml](./component.yaml) | Structured package metadata |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Phased work packages |
| [REQUIREMENTS.md](./REQUIREMENTS.md) | Product and architecture requirements |

## Usage

```python
from twinr.agent.self_coding import (
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    SelfCodingCapabilityRegistry,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
)

store = SelfCodingStore.from_project_root(".")
registry = SelfCodingCapabilityRegistry(project_root=".")
checker = SelfCodingFeasibilityChecker(registry)
dialogue = SelfCodingRequirementsDialogue()
flow = SelfCodingLearningFlow(store=store, checker=checker, dialogue=dialogue)
job = CompileJobRecord(
    job_id="job_demo123",
    skill_id="read_messages",
    skill_name="Read Messages",
    status=CompileJobStatus.QUEUED,
    requested_target=CompileTarget.AUTOMATION_MANIFEST,
    spec_hash="spec-demo",
)
store.save_job(job)
ready_capabilities = registry.configured_capability_ids()
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- [REQUIREMENTS.md](./REQUIREMENTS.md)
