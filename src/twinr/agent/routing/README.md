# routing

`routing` owns Twinr's local semantic transcript router for the streaming
voice path. It now supports a two-stage setup: first a user-centered
`wissen | nachschauen | persoenlich | machen_oder_pruefen` classifier, then the
backend `parametric | web | memory | tool` router with normal authority gating.
The package keeps bundle formats, ONNX inference, authority policy, and offline
evaluation/calibration code out of the workflow loops.

Internally, the package keeps shared ONNX inference helpers, shared centroid
math, and the synthetic recipe registry in dedicated modules so runtime,
offline training, and dataset-definition concerns stay separated.

## Responsibility

`routing` owns:
- define the canonical `parametric | web | memory | tool` label set
- define the canonical `wissen | nachschauen | persoenlich | machen_oder_pruefen` label set
- load and validate versioned local router bundles
- load and validate versioned local user-intent bundles
- run low-latency ONNX sentence-encoder inference against those bundles
- run a shared-embedding two-stage classifier when both stages are configured
- apply calibrated authority thresholds and margin gates before a route may bypass the supervisor lane
- evaluate scored route decisions against labeled transcript datasets
- keep shared encoder/numeric helpers and centroid math isolated from route-specific runtime code
- build centroid-based or trained linear-head bundles from labeled transcript corpora without pulling training logic into the live workflow path
- build centroid-based or trained linear-head user-intent bundles from labeled transcript corpora
- generate and curate dual-labeled synthetic bootstrap corpora for both stages
- keep synthetic corpora family-balanced so a few surface forms do not dominate one label
- orchestrate backend-only and two-stage synthetic-bootstrap training and split-wise evaluation reports

`routing` does **not** own:
- workflow-loop orchestration or speech-lane timing; that stays in [`workflows`](../workflows/README.md)
- provider-side supervisor prompts or tool-call schemas
- long-term memory retrieval or web-search execution
- operator setup scripts or Pi deployment automation

## Bundle Layout

Each bundle directory must contain:

- `model.onnx`
- `model.ort` is optional but preferred on the Pi when present
- `tokenizer.json`
- `router_metadata.json`

Classifier files depend on `classifier_type`:

- `embedding_centroid_v1` -> `centroids.npy`
- `embedding_linear_softmax_v1` -> `weights.npy` and `bias.npy`

`router_metadata.json` stores:

- the exact label order
- model identifier
- max sequence length
- pooling mode and optional explicit ONNX output name
- confidence thresholds per label
- authoritative label set
- minimum confidence margin
- the frozen reference date for `parametric` versus `web`

User-intent bundles use the same model/tokenizer/classifier-file layout but
store their metadata in `user_intent_metadata.json`.

The bootstrap builders now emit `model.ort` sidecars when ONNX Runtime's
conversion tooling is available. The runtime loaders automatically prefer
`model.ort` over `model.onnx` so Pi startups avoid re-optimizing large graphs
on-device.

## Usage

```python
from twinr.agent.routing import (
    LocalSemanticRouter,
    TwoStageLocalSemanticRouter,
    load_semantic_router_bundle,
    load_user_intent_bundle,
)

bundle = load_semantic_router_bundle("artifacts/router/multilingual-minilm-router")
router = LocalSemanticRouter(bundle)
decision = router.classify("Was ist heute in Berlin passiert?")

user_bundle = load_user_intent_bundle("artifacts/router/user_intent_bundle")
two_stage_router = TwoStageLocalSemanticRouter(user_bundle, bundle)
two_stage_decision = two_stage_router.classify("Was habe ich heute fuer Termine?")
```

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.synthetic_corpus \
  --output-path artifacts/router/synthetic/router_samples.jsonl \
  --samples-per-label 1024
```

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.training bootstrap-synthetic \
  --source-dir /models/paraphrase-multilingual-MiniLM-L12-v2-onnx \
  --work-dir artifacts/router/bootstrap \
  --classifier linear
```

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.training bootstrap-two-stage-synthetic \
  --source-dir /models/paraphrase-multilingual-MiniLM-L12-v2-onnx \
  --work-dir artifacts/router/bootstrap_two_stage \
  --classifier linear
```

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.bootstrap \
  --source-dir /models/paraphrase-multilingual-MiniLM-L12-v2-onnx \
  --dataset test/fixtures/router/router_samples.jsonl \
  --output-dir artifacts/router/multilingual-minilm-router
```

The synthetic bootstrap path writes:

- `synthetic_router_samples.jsonl` with balanced `train/dev/test` rows
- `bundle/` with the runtime-compatible centroid router bundle
- `training_report.json` with split metrics, fallback/authority rates, and dataset curation counts

The two-stage synthetic bootstrap path writes:

- `backend_route_samples.jsonl`
- `user_intent_samples.jsonl`
- `backend_route_bundle/`
- `user_intent_bundle/`
- `training_report.json` with stage-specific reports plus combined two-stage backend-route evals

The synthetic corpus generator deliberately mixes:

- stable family rotation per label instead of unconstrained random family picks
- user-centered boundary families such as short personal lookups versus live checks
- light transcript noise (`lowercase`, `no_punct`, `umlaut_flat`, limited filler) without runtime heuristics

## Labeling

Use [LABELING_HANDBOOK.md](./LABELING_HANDBOOK.md) as the source of truth for
the user-centered taxonomy, precedence rules, and example bank.

## See also

- [AGENTS.md](./AGENTS.md)
- [LABELING_HANDBOOK.md](./LABELING_HANDBOOK.md)
- [component.yaml](./component.yaml)
- [workflows](../workflows/README.md)
