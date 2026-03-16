# self_coding

`self_coding` owns the first focused package slice of Twinr's Adaptive Skill
Engine (ASE). The current scope now covers the deterministic front-stage core
plus the first local compile-worker and activation slice: versioned contracts,
status enums, a file-backed runtime store, the MVP capability registry, the
feasibility checker, the requirements-dialogue state machine, queued compile
jobs, bounded compile workspaces, the first `automation_manifest` compiler
path, compile status persistence, versioned soft-launch activation and
rollback, a compact operator-status summary, and the local Codex drivers used
to draft artifacts. The primary local driver is now a pinned `codex-sdk`
bridge, with `codex exec --json` kept as the bounded fallback path.

## Responsibility

`self_coding` owns:
- the stable internal contract objects used across future ASE layers
- the status model for capabilities, compile jobs, artifacts, and learned skills
- the file-backed store rooted at `state/self_coding`
- the persisted compile-status record used for operator/debug visibility
- the deterministic capability registry for the MVP module set
- the deterministic feasibility checker for Phase 1
- the persistent requirements-dialogue flow for Phase 2
- the compile-worker that creates queued jobs and persists local Codex artifacts
- the first compiler path that validates and canonicalizes `automation_manifest` outputs
- the deterministic compile-prompt builder that anchors Codex on Twinr artifact contracts
- versioned soft-launch staging, confirmation, and rollback for automation-based learned skills
- the compact operator-status summary used by the web dashboard
- the bounded local Codex workspaces and driver adapters under `codex_driver/`

`self_coding` does **not** own:
- workflow-loop orchestration or runtime hooks
- longer-term health scoring, automated pause policy, and operator review workflow beyond the first soft-launch path
- deep compiler validation and compile-target specialization beyond the current slice
- sandboxed skill execution and brokered effects

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [contracts.py](./contracts.py) | Versioned ASE contract dataclasses |
| [status.py](./status.py) | Shared ASE enums and lifecycle states |
| [store.py](./store.py) | File-backed store under `state/self_coding` |
| [compiler/](./compiler) | Deterministic compile-target validation and canonicalization |
| [compiler/prompting.py](./compiler/prompting.py) | Deterministic compile-prompt builder for Twinr artifact contracts |
| [capability_registry.py](./capability_registry.py) | Deterministic MVP capability readiness view |
| [feasibility.py](./feasibility.py) | Deterministic Phase-1 feasibility checker |
| [requirements_dialogue.py](./requirements_dialogue.py) | Deterministic Phase-2 dialogue state machine |
| [learning_flow.py](./learning_flow.py) | Thin orchestration of feasibility, dialogue, and queued compile creation |
| [worker.py](./worker.py) | Queued compile-job execution, status updates, and artifact persistence |
| [activation.py](./activation.py) | Versioned soft-launch staging, enablement, and rollback |
| [operator_status.py](./operator_status.py) | Compact operator-facing compile/activation summary |
| [codex_driver/](./codex_driver) | Pinned Codex SDK bridge, exec fallback, and workspace builders |
| [component.yaml](./component.yaml) | Structured package metadata |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Phased work packages |
| [REQUIREMENTS.md](./REQUIREMENTS.md) | Product and architecture requirements |

## Usage

```python
from twinr.agent.self_coding import (
    CompileJobRecord,
    CompileJobStatus,
    CompileRunStatusRecord,
    CompileTarget,
    LocalCodexCompileDriver,
    SelfCodingActivationService,
    SelfCodingCompileWorker,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    build_self_coding_operator_status,
    SelfCodingCapabilityRegistry,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
)

store = SelfCodingStore.from_project_root(".")
registry = SelfCodingCapabilityRegistry(project_root=".")
checker = SelfCodingFeasibilityChecker(registry)
dialogue = SelfCodingRequirementsDialogue()
worker = SelfCodingCompileWorker(store=store)
flow = SelfCodingLearningFlow(store=store, checker=checker, dialogue=dialogue, compile_worker=worker)
driver = LocalCodexCompileDriver()
status = build_self_coding_operator_status(store)
job = CompileJobRecord(
    job_id="job_demo123",
    skill_id="read_messages",
    skill_name="Read Messages",
    status=CompileJobStatus.QUEUED,
    requested_target=CompileTarget.AUTOMATION_MANIFEST,
    spec_hash="spec-demo",
)
store.save_job(job)
store.save_compile_status(CompileRunStatusRecord(job_id=job.job_id, phase="starting"))
ready_capabilities = registry.configured_capability_ids()
```

Before running live SDK-backed compile jobs on a machine, install the pinned
bridge dependency once:

```bash
cd src/twinr/agent/self_coding/codex_driver/sdk_bridge
npm ci
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- [REQUIREMENTS.md](./REQUIREMENTS.md)
