# AGENTS.md — /src/twinr/providers/groq

## Scope

This directory owns Groq client validation, text-turn execution, tool-calling
request translation, and bounded continuation storage. Structural metadata
lives in [component.yaml](./component.yaml).

Out of scope:
- OpenAI support-provider implementation details in `src/twinr/providers/openai`
- provider selection and bundle assembly in `src/twinr/providers/factory.py`
- speech-to-text, text-to-speech, or hardware behavior

## Key files

- `__init__.py` — public export surface; treat changes as API-impacting
- `adapters.py` — text and tool providers, including continuation storage
- `client.py` — validated Groq SDK client builder and base URL policy
- `types.py` — normalized response value object
- `component.yaml` — package metadata, callers, and test map

## Invariants

- `GroqAgentTextProvider` and `GroqToolCallingAgentProvider` remain the only public exports from this package.
- `client.py` must validate API key, timeout, and base URL policy before constructing the SDK client.
- `allow_web_search=True` stays a delegated support-provider path; do not silently pretend native Groq live search exists here.
- Tool continuations must stay bounded by TTL and item count, and continuation reuse must continue to enforce exact expected call IDs.
- Malformed tool arguments or tool results must surface as controlled provider failures, not raw JSON or attribute exceptions.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/providers/groq
PYTHONPATH=src pytest test/test_groq_providers.py test/test_provider_factory.py -q
```

If `adapters.py` changed in a way that affects streaming/tool-turn behavior, also run:

```bash
PYTHONPATH=src pytest test/test_streaming_runner.py -q
```

If `client.py` changed, also run:

```bash
PYTHONPATH=src pytest test/test_config.py -q
```

## Coupling

`adapters.py` changes -> also check:
- `src/twinr/providers/factory.py`
- `src/twinr/agent/workflows/streaming_runner.py`
- `test/test_groq_providers.py`
- `test/test_streaming_runner.py`

`client.py` changes -> also check:
- `src/twinr/agent/base_agent/config.py`
- `src/twinr/providers/factory.py`
- `test/test_config.py`
- `test/test_provider_factory.py`

`__init__.py` changes -> also check:
- `src/twinr/providers/factory.py`
- `test/test_provider_factory.py`

## Security

- Do not widen Groq base URL acceptance without an audited reason; base URL policy protects API-key exfiltration.
- Never log raw API keys, full authorization headers, or arbitrary tool payloads from user turns.
- Keep continuation storage bounded in memory and exact-call-id matched; do not degrade these checks into best-effort behavior.

## Output expectations

- Update docstrings when request translation, fallback behavior, or continuation semantics change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when exports, callers, or verification commands change.
- Keep support-provider delegation clear; cross-provider orchestration changes belong in `src/twinr/providers/factory.py`.
