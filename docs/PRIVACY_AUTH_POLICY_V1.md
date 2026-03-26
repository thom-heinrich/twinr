# Privacy Auth Policy V1

Specify one central multimodal privacy/auth gate for Twinr tool actions.

This design replaces the current voice-only sensitive-action confirmation path
with one explicit decision engine that separates:

- identity/auth
- privacy/listener risk
- media/noise suppression
- forbidden secret-data handling

The intent is regression-safe rollout: one pure policy module, thin handler
adapters, explicit prompt kinds, and a shadow-mode migration before the old
guard is removed.

## Goals

- Replace scattered `require_sensitive_voice_confirmation(...)` checks with one
  reusable policy evaluation entrypoint.
- Keep authorization, privacy warning, and media suppression as separate
  decisions instead of collapsing them into one "blocked" state.
- Make private readouts never proactive.
- Keep "secret" data out of Twinr entirely.
- Make the result easy to unit-test without hardware or live runtime loops.

## Non-goals

- No biometric approval path for truly critical actions.
- No second hidden policy engine inside `service.py`, `voice_session.py`, or
  prompt text.
- No per-user hardcoded exceptions.
- No storing of passwords, PINs, TANs, recovery codes, or similar secrets.

## Proposed module placement

Primary module:

- `src/twinr/agent/tools/policy/privacy_auth_policy.py`

Thin integration points:

- `src/twinr/agent/tools/handlers/support.py`
- `src/twinr/agent/tools/prompting/instructions.py`
- `src/twinr/orchestrator/voice_runtime_intent.py`

Tests:

- `test/test_privacy_auth_policy.py`
- `test/test_privacy_auth_policy_integration.py`

The policy module should stay pure: no hardware calls, no persistence, no
prompt-string composition, no tool execution.

## Runtime inputs to reuse

The new policy should consume existing runtime surfaces instead of creating
parallel truth sources:

- `user_voice_status` and matched user fields from
  `RuntimeContext.update_user_voice_assessment`
- `identity_fusion`
- `portrait_match`
- `speaker_association`
- `ambiguous_room_guard`
- `audio_policy`
- camera presence/person-count facts

Important separation:

- `auth` answers: "Is this probably the intended user?"
- `privacy` answers: "Could others be listening?"
- `media` answers: "Is the audio environment too noisy or TV-like to trust?"

`ambiguous_room_guard` is useful for conservative targeted inference, but V1
policy should not blindly reuse `guard_active` as the sole auth blocker for all
actions. Private-read privacy warnings and target-auth clarity are different
questions and must remain separate in the decision model.

## Core enums

Use `StrEnum` so config values and logs stay readable.

```python
from dataclasses import dataclass
from enum import StrEnum


class ActionClass(StrEnum):
    BENIGN_RESPONSE = "benign_response"
    PRIVATE_READ = "private_read"
    LOCAL_MUTATION = "local_mutation"
    OUTBOUND_OR_EFFECTFUL = "outbound_or_effectful"
    IDENTITY_ENROLLMENT = "identity_enrollment"
    SECRET_DATA = "secret_data"


class AuthRequirement(StrEnum):
    NONE = "none"
    VOICE_ONLY = "voice_only"
    MULTIMODAL_LOW_RISK = "multimodal_low_risk"
    MULTIMODAL_STRICT = "multimodal_strict"
    FORBIDDEN = "forbidden"


class TargetClarityRequirement(StrEnum):
    NONE = "none"
    REQUIRE_CLEAR_TARGET = "require_clear_target"


class PrivacyOverridePolicy(StrEnum):
    IGNORE = "ignore"
    WARN_OVERRIDE = "warn_override"
    BLOCK = "block"


class MediaPolicy(StrEnum):
    IGNORE = "ignore"
    BLOCK_AUTH = "block_auth"
    BLOCK_ACTION = "block_action"


class ConfirmationPolicy(StrEnum):
    NONE = "none"
    EXPLICIT_SAME_TURN = "explicit_same_turn"


class ProactivePolicy(StrEnum):
    ALLOW = "allow"
    FORBID = "forbid"


class DecisionKind(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_PRIVACY_OVERRIDE = "require_privacy_override"
    BLOCK_RETRYABLE = "block_retryable"
    BLOCK_FORBIDDEN = "block_forbidden"


class PromptKind(StrEnum):
    NONE = "none"
    ASK_CONFIRM_PRIVATE_READ = "ask_confirm_private_read"
    ASK_CONFIRM_MUTATION = "ask_confirm_mutation"
    ASK_CONFIRM_ENROLLMENT = "ask_confirm_enrollment"
    ASK_PRIVACY_OVERRIDE_PRIVATE_READ = "ask_privacy_override_private_read"
    BLOCK_IDENTITY = "block_identity"
    BLOCK_CLEAR_TARGET = "block_clear_target"
    BLOCK_MEDIA = "block_media"
    BLOCK_SECRET_DATA = "block_secret_data"


class DecisionReason(StrEnum):
    OK = "ok"
    EXPLICIT_CONFIRMATION_MISSING = "explicit_confirmation_missing"
    PRIVACY_OVERRIDE_REQUIRED = "privacy_override_required"
    IDENTITY_NOT_STRONG_ENOUGH = "identity_not_strong_enough"
    CLEAR_TARGET_REQUIRED = "clear_target_required"
    BACKGROUND_MEDIA_ACTIVE = "background_media_active"
    NON_SPEECH_AUDIO_ACTIVE = "non_speech_audio_active"
    ROOM_BUSY_OR_OVERLAPPING = "room_busy_or_overlapping"
    SECRET_DATA_FORBIDDEN = "secret_data_forbidden"
    PROACTIVE_PRIVATE_ACTION_FORBIDDEN = "proactive_private_action_forbidden"
```

## Core dataclasses

```python
@dataclass(frozen=True, slots=True)
class PolicyRule:
    action_class: ActionClass
    auth_requirement: AuthRequirement
    target_clarity_requirement: TargetClarityRequirement
    privacy_override_policy: PrivacyOverridePolicy
    media_policy: MediaPolicy
    confirmation_policy: ConfirmationPolicy
    proactive_policy: ProactivePolicy


@dataclass(frozen=True, slots=True)
class PrivacyAuthSignals:
    voice_status: str | None = None
    voice_confidence: float | None = None
    voice_matched_user_id: str | None = None
    portrait_match_state: str | None = None
    portrait_matches_reference_user: bool | None = None
    portrait_matched_user_id: str | None = None
    speaker_association_state: str | None = None
    speaker_associated: bool | None = None
    speaker_association_confidence: float | None = None
    identity_fusion_state: str | None = None
    identity_fusion_policy: str | None = None
    identity_fusion_matched_user_id: str | None = None
    person_visible: bool = False
    person_count: int | None = None
    privacy_listener_risk: bool = False
    privacy_listener_risk_reason: str | None = None
    clear_target_available: bool = False
    clear_target_reason: str | None = None
    background_media_likely: bool = False
    non_speech_audio_likely: bool = False
    room_busy_or_overlapping: bool = False


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action_class: ActionClass
    action_label: str
    explicit_user_request: bool = True
    confirmed: bool = False
    privacy_override_confirmed: bool = False
    proactive_attempt: bool = False


@dataclass(frozen=True, slots=True)
class PrivacyAuthDecision:
    kind: DecisionKind
    prompt_kind: PromptKind
    reason: DecisionReason
    action_class: ActionClass
    allow_tool_call: bool
    requires_confirmation: bool = False
    requires_privacy_override: bool = False
    matched_user_id: str | None = None
    log_fields: dict[str, object] | None = None
```

## Default action policy

```python
DEFAULT_RULES: dict[ActionClass, PolicyRule] = {
    ActionClass.BENIGN_RESPONSE: PolicyRule(
        action_class=ActionClass.BENIGN_RESPONSE,
        auth_requirement=AuthRequirement.NONE,
        target_clarity_requirement=TargetClarityRequirement.NONE,
        privacy_override_policy=PrivacyOverridePolicy.IGNORE,
        media_policy=MediaPolicy.IGNORE,
        confirmation_policy=ConfirmationPolicy.NONE,
        proactive_policy=ProactivePolicy.ALLOW,
    ),
    ActionClass.PRIVATE_READ: PolicyRule(
        action_class=ActionClass.PRIVATE_READ,
        auth_requirement=AuthRequirement.MULTIMODAL_LOW_RISK,
        target_clarity_requirement=TargetClarityRequirement.NONE,
        privacy_override_policy=PrivacyOverridePolicy.WARN_OVERRIDE,
        media_policy=MediaPolicy.BLOCK_AUTH,
        confirmation_policy=ConfirmationPolicy.EXPLICIT_SAME_TURN,
        proactive_policy=ProactivePolicy.FORBID,
    ),
    ActionClass.LOCAL_MUTATION: PolicyRule(
        action_class=ActionClass.LOCAL_MUTATION,
        auth_requirement=AuthRequirement.MULTIMODAL_LOW_RISK,
        target_clarity_requirement=TargetClarityRequirement.REQUIRE_CLEAR_TARGET,
        privacy_override_policy=PrivacyOverridePolicy.IGNORE,
        media_policy=MediaPolicy.BLOCK_AUTH,
        confirmation_policy=ConfirmationPolicy.EXPLICIT_SAME_TURN,
        proactive_policy=ProactivePolicy.FORBID,
    ),
    ActionClass.OUTBOUND_OR_EFFECTFUL: PolicyRule(
        action_class=ActionClass.OUTBOUND_OR_EFFECTFUL,
        auth_requirement=AuthRequirement.MULTIMODAL_STRICT,
        target_clarity_requirement=TargetClarityRequirement.REQUIRE_CLEAR_TARGET,
        privacy_override_policy=PrivacyOverridePolicy.IGNORE,
        media_policy=MediaPolicy.BLOCK_ACTION,
        confirmation_policy=ConfirmationPolicy.EXPLICIT_SAME_TURN,
        proactive_policy=ProactivePolicy.FORBID,
    ),
    ActionClass.IDENTITY_ENROLLMENT: PolicyRule(
        action_class=ActionClass.IDENTITY_ENROLLMENT,
        auth_requirement=AuthRequirement.MULTIMODAL_STRICT,
        target_clarity_requirement=TargetClarityRequirement.REQUIRE_CLEAR_TARGET,
        privacy_override_policy=PrivacyOverridePolicy.IGNORE,
        media_policy=MediaPolicy.BLOCK_ACTION,
        confirmation_policy=ConfirmationPolicy.EXPLICIT_SAME_TURN,
        proactive_policy=ProactivePolicy.FORBID,
    ),
    ActionClass.SECRET_DATA: PolicyRule(
        action_class=ActionClass.SECRET_DATA,
        auth_requirement=AuthRequirement.FORBIDDEN,
        target_clarity_requirement=TargetClarityRequirement.NONE,
        privacy_override_policy=PrivacyOverridePolicy.BLOCK,
        media_policy=MediaPolicy.BLOCK_ACTION,
        confirmation_policy=ConfirmationPolicy.NONE,
        proactive_policy=ProactivePolicy.FORBID,
    ),
}
```

## Config surface

This should be operator-facing config, not `update_simple_setting`.

Recommended `TwinrConfig` additions:

```python
privacy_auth_enabled: bool = True
privacy_auth_shadow_mode: bool = True
privacy_auth_log_decisions: bool = True

privacy_private_read_auth_requirement: str = "multimodal_low_risk"
privacy_private_read_confirmation_policy: str = "explicit_same_turn"
privacy_private_read_room_policy: str = "warn_override"
privacy_private_read_proactive_policy: str = "forbid"

privacy_local_mutation_auth_requirement: str = "multimodal_low_risk"
privacy_outbound_auth_requirement: str = "multimodal_strict"
privacy_identity_enrollment_auth_requirement: str = "multimodal_strict"

privacy_effectful_target_clarity_requirement: str = "require_clear_target"
privacy_media_policy_for_auth: str = "block_auth"
privacy_media_policy_for_effectful: str = "block_action"

privacy_secret_data_policy: str = "forbid"
privacy_wake_background_media_policy: str = "block"
```

Implementation note:

- Keep the operator surface intentionally small.
- Parse config strings into the enums above via one `PrivacyAuthConfig`.
- Do not expose these as normal user voice settings.

## Proposed public API

```python
def build_privacy_auth_config(config: TwinrConfig) -> PrivacyAuthConfig: ...

def build_privacy_auth_signals(
    *,
    runtime: RuntimeContext,
    live_facts: Mapping[str, object] | None,
) -> PrivacyAuthSignals: ...

def evaluate_privacy_auth_action(
    *,
    policy: PrivacyAuthConfig,
    request: ActionRequest,
    signals: PrivacyAuthSignals,
) -> PrivacyAuthDecision: ...
```

Optional thin helper in `support.py`:

```python
def enforce_privacy_auth_action(
    owner: Any,
    arguments: dict[str, object],
    *,
    action_class: ActionClass,
    action_label: str,
) -> PrivacyAuthDecision: ...
```

## Evaluation order

The order matters for consistent prompting and safe rollout.

1. If `auth_requirement == FORBIDDEN`, return `BLOCK_FORBIDDEN`.
2. If `proactive_attempt` and the rule forbids proactive behavior, return
   `BLOCK_RETRYABLE`.
3. If media policy blocks and media/noise flags are active, return
   `BLOCK_RETRYABLE`.
4. If auth requirement is not met, return `BLOCK_RETRYABLE`.
5. If clear target is required and unavailable, return `BLOCK_RETRYABLE`.
6. If explicit confirmation is required and `confirmed` is false, return
   `REQUIRE_CONFIRMATION`.
7. If privacy listener risk is active and override is required but not yet
   confirmed, return `REQUIRE_PRIVACY_OVERRIDE`.
8. Otherwise return `ALLOW`.

This keeps auth and quality failures ahead of privacy override. Twinr should not
ask "read it anyway?" when it still cannot authenticate the speaker reliably.

## Auth interpretation

V1 auth should be explicit and inspectable, not hidden in freeform heuristics.

Recommended helpers:

```python
def voice_identity_ok(signals: PrivacyAuthSignals) -> bool:
    return signals.voice_status in {"likely_user", "known_other_user"}


def visual_identity_ok(signals: PrivacyAuthSignals) -> bool:
    return (
        signals.portrait_matches_reference_user is True
        or bool(signals.identity_fusion_matched_user_id)
    )


def multimodal_low_risk_ok(signals: PrivacyAuthSignals) -> bool:
    return voice_identity_ok(signals) and visual_identity_ok(signals)


def multimodal_strict_ok(signals: PrivacyAuthSignals) -> bool:
    return (
        multimodal_low_risk_ok(signals)
        and signals.clear_target_available
        and signals.speaker_associated is True
    )
```

Important:

- `privacy_listener_risk` must not automatically invalidate `multimodal_low_risk_ok`.
- `clear_target_available` is stricter and is mainly for mutations, enrollments,
  and effectful actions.

## Privacy vs auth split

For the user-facing private-read case:

- `auth` decides whether the requester is probably the intended person.
- `privacy_listener_risk` decides whether others may be listening.

That yields the intended behavior:

- authenticated + privacy risk => ask override
- unauthenticated => block

This is the main reason the new module must not simply reuse
`ambiguous_room_guard.guard_active` as a single universal veto.

## Prompt kinds

The policy module should not store full German prose. It should only return a
stable `PromptKind`. Prompt wording stays in the prompting layer.

Recommended mappings:

- `ASK_CONFIRM_PRIVATE_READ`
  - "Soll ich die Nachricht jetzt vorlesen?"
- `ASK_CONFIRM_MUTATION`
  - "Soll ich das jetzt wirklich ändern?"
- `ASK_CONFIRM_ENROLLMENT`
  - "Soll ich deine Stimme beziehungsweise dein Profil jetzt lokal lernen?"
- `ASK_PRIVACY_OVERRIDE_PRIVATE_READ`
  - "Ich glaube, du bist vielleicht nicht allein im Raum. Soll ich trotzdem vorlesen?"
- `BLOCK_IDENTITY`
  - "Ich kann gerade nicht sicher genug erkennen, dass du es bist."
- `BLOCK_CLEAR_TARGET`
  - "Ich kann gerade nicht klar genug zuordnen, wer die Aktion auslösen möchte."
- `BLOCK_MEDIA`
  - "Ich höre gerade eher Hintergrundmedien oder Störgeräusche."
- `BLOCK_SECRET_DATA`
  - "Solche geheimen Daten speichere oder lese ich nicht vor."

## Handler mapping

Map handlers to action classes instead of keeping per-handler security policy.

`PRIVATE_READ`

- contact lookup / detail read
- long-term memory detail inspection
- future WhatsApp or mail readout tools

`LOCAL_MUTATION`

- remember/update profile, preferences, plans, personality
- bounded Twinr device settings
- world-intelligence subscription changes

`OUTBOUND_OR_EFFECTFUL`

- smart-home control
- automation create/update/delete
- future outbound send tools

`IDENTITY_ENROLLMENT`

- local voice-profile enroll/reset
- household face/voice enroll

`SECRET_DATA`

- any future request to store, repeat, or manage secrets

## Integration plan

### 1. Thin adapter in `support.py`

Replace the current voice-only helper with a wrapper around the policy engine.
The adapter should:

- read `confirmed` and `privacy_override_confirmed`
- build `ActionRequest`
- collect `PrivacyAuthSignals`
- call `evaluate_privacy_auth_action(...)`
- raise typed exceptions based on `DecisionKind`

Recommended new exceptions:

- `PrivacyAuthConfirmationRequired`
- `PrivacyAuthPrivacyOverrideRequired`
- `PrivacyAuthBlocked`
- `PrivacyAuthForbidden`

### 2. Handlers

Each handler should declare only:

- `action_class`
- `action_label`

Example:

```python
enforce_privacy_auth_action(
    owner,
    arguments,
    action_class=ActionClass.PRIVATE_READ,
    action_label="read saved contact details",
)
```

### 3. Prompting

Prompting should react to typed guard results or returned `PromptKind`, not
re-implement the policy in natural language.

### 4. Wake gating

Use the same media policy vocabulary for wake scans:

- if `background_media_likely` and wake media policy is `block`, do not keep
  idle wake scanning open for that window
- follow-up windows may remain separate

This belongs in `voice_runtime_intent.py`, but the decision vocabulary should
match the policy module.

## Regression-safe rollout

Roll out in four phases:

1. Add the pure policy module and tests.
2. Shadow mode: evaluate new decisions in parallel, log diffs, keep old guard
   authoritative.
3. Switch `PRIVATE_READ` and `SECRET_DATA` first.
4. Switch `LOCAL_MUTATION`, `OUTBOUND_OR_EFFECTFUL`, and
   `IDENTITY_ENROLLMENT` after diff review.

Wake/media gating should move only after the tool-action path is stable.

## Minimum test matrix

Unit-test the policy matrix, not only individual handlers.

Minimum rows:

- `PRIVATE_READ` + strong auth + no privacy risk + confirmed => allow
- `PRIVATE_READ` + strong auth + privacy risk + confirmed + no override => ask privacy override
- `PRIVATE_READ` + strong auth + privacy risk + confirmed + override => allow
- `PRIVATE_READ` + weak auth => block identity
- `PRIVATE_READ` + background media => block media
- `LOCAL_MUTATION` + strong auth + no confirm => ask confirmation
- `OUTBOUND_OR_EFFECTFUL` + strong auth + unclear target => block clear target
- `IDENTITY_ENROLLMENT` + non-speech/media => block media
- `SECRET_DATA` => forbid

Integration tests should assert that the handler/action mapping returns the
expected `PromptKind` and exception type.

## V1 implementation boundary

V1 can ship without changing the existing multimodal runtime claim producers.
It only needs:

- one new pure policy module
- one support-layer adapter
- handler remapping
- prompt-kind integration
- tests

If later work is needed to better separate "auth target" from "privacy listener
risk" in multi-person scenes, that should extend the signal builder, not the
handler contracts or the decision vocabulary.
