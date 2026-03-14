from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from twinr import TwinrConfig, TwinrRuntime
from twinr.providers.openai import OpenAIBackend


_EXPLICIT_MEMORY_PATTERNS = (
    re.compile(r"\bich erinnere mich\b", re.IGNORECASE),
    re.compile(r"\bwenn ich mich richtig erinnere\b", re.IGNORECASE),
    re.compile(r"\bdu hast gesagt\b", re.IGNORECASE),
    re.compile(r"\bdu meintest\b", re.IGNORECASE),
    re.compile(r"\bi remember\b", re.IGNORECASE),
    re.compile(r"\bif i remember correctly\b", re.IGNORECASE),
    re.compile(r"\byou told me\b", re.IGNORECASE),
    re.compile(r"\byou said earlier\b", re.IGNORECASE),
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SubtextSeedAction:
    kind: Literal["preference", "contact", "plan", "episode"]
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SubtextEvalCase:
    case_id: str
    category: str
    query_text: str
    hidden_context: str
    desired_behavior: str
    should_use_personal_context: bool
    seed_actions: tuple[SubtextSeedAction, ...]


@dataclass(frozen=True, slots=True)
class SubtextJudgeResult:
    helpful_context_used: bool
    subtle_not_explicit: bool
    unforced: bool
    addresses_request: bool
    naturalness_score: int
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SubtextEvalCaseResult:
    case_id: str
    category: str
    query_text: str
    response_text: str
    explicit_memory_announcement: bool
    judge: SubtextJudgeResult
    model: str | None
    request_id: str | None
    response_id: str | None
    prompt_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class SubtextEvalSummary:
    total_cases: int
    passed_cases: int
    accuracy: float
    category_case_counts: dict[str, int]
    category_pass_counts: dict[str, int]
    category_accuracy: dict[str, float]
    explicit_memory_violations: int
    average_naturalness: float


@dataclass(frozen=True, slots=True)
class SubtextEvalResult:
    summary: SubtextEvalSummary
    cases: tuple[SubtextEvalCaseResult, ...]


def default_subtext_eval_cases() -> tuple[SubtextEvalCase, ...]:
    return (
        SubtextEvalCase(
            case_id="coffee_brand",
            category="preference",
            query_text="Wo kann ich heute Kaffee kaufen?",
            hidden_context=(
                "The user strongly prefers Melitta for coffee and usually buys coffee at Markt Z."
            ),
            desired_behavior=(
                "If helpful, gently bias the suggestion toward Melitta or Markt Z without announcing hidden memory."
            ),
            should_use_personal_context=True,
            seed_actions=(
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "brand", "value": "Melitta", "for_product": "coffee"},
                ),
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "store", "value": "Markt Z", "for_product": "coffee"},
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="weather_walk",
            category="situational",
            query_text="Ich überlege, was heute ein guter Plan für draußen wäre.",
            hidden_context=(
                "The user planned to go for a walk in the park today and recently talked about wanting good weather for that walk."
            ),
            desired_behavior=(
                "A helpful answer can naturally suggest the planned walk or a similar outdoor plan, but should not mention hidden memory or prior conversation explicitly."
            ),
            should_use_personal_context=True,
            seed_actions=(
                SubtextSeedAction(
                    kind="plan",
                    args={
                        "summary": "Go for a walk in the park",
                        "when_text": "today",
                        "details": "The user wanted good weather for the walk.",
                    },
                ),
                SubtextSeedAction(
                    kind="episode",
                    args={
                        "transcript": "Tomorrow I want to go for a walk if the weather is nice.",
                        "response": "I can keep the weather in mind for that walk.",
                    },
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="contact_role",
            category="social",
            query_text="Soll ich Corinna heute noch anrufen?",
            hidden_context=(
                "Corinna Maier is the user's physiotherapist and an already-known practical contact."
            ),
            desired_behavior=(
                "The reply should feel like Twinr knows who Corinna is in the user's life, without saying that this information was remembered."
            ),
            should_use_personal_context=True,
            seed_actions=(
                SubtextSeedAction(
                    kind="contact",
                    args={
                        "given_name": "Corinna",
                        "family_name": "Maier",
                        "role": "physiotherapist",
                        "phone": "+49 151 3000 2200",
                    },
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="episodic_knee",
            category="episodic",
            query_text="Soll ich heute spazieren gehen oder lieber zuhause bleiben?",
            hidden_context=(
                "In a recent conversation, the user said their knee hurt today and Twinr suggested taking it easy."
            ),
            desired_behavior=(
                "The answer may gently factor the knee issue into its recommendation, but should not explicitly cite prior memory."
            ),
            should_use_personal_context=True,
            seed_actions=(
                SubtextSeedAction(
                    kind="episode",
                    args={
                        "transcript": "My knee hurts a bit today.",
                        "response": "Then it may be wise to take it easy today.",
                    },
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="tea_store",
            category="preference",
            query_text="Ich brauche heute Tee. Was wäre praktisch?",
            hidden_context=(
                "The user usually buys tea at Laden Seidel and prefers Teekanne for tea."
            ),
            desired_behavior=(
                "If useful, the answer should quietly favor Teekanne or Laden Seidel instead of giving a generic answer."
            ),
            should_use_personal_context=True,
            seed_actions=(
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "store", "value": "Laden Seidel", "for_product": "tea"},
                ),
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "brand", "value": "Teekanne", "for_product": "tea"},
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="tokyo_time_control",
            category="control",
            query_text="Wenn es bei mir 10 Uhr ist, wie spät ist es dann in Tokio?",
            hidden_context=(
                "The user prefers Melitta coffee and wants to go for a walk today, but neither is relevant to the current question."
            ),
            desired_behavior=(
                "The answer should focus only on the requested time conversion and should not inject unrelated personal context."
            ),
            should_use_personal_context=False,
            seed_actions=(
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "brand", "value": "Melitta", "for_product": "coffee"},
                ),
                SubtextSeedAction(
                    kind="plan",
                    args={"summary": "Go for a walk in the park", "when_text": "today"},
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="math_control",
            category="control",
            query_text="Was ist 27 mal 14?",
            hidden_context=(
                "The user has a known physiotherapist contact and a tea preference, but neither should affect a simple arithmetic answer."
            ),
            desired_behavior=(
                "The reply should just answer the math question cleanly and avoid personal details."
            ),
            should_use_personal_context=False,
            seed_actions=(
                SubtextSeedAction(
                    kind="contact",
                    args={
                        "given_name": "Corinna",
                        "family_name": "Maier",
                        "role": "physiotherapist",
                    },
                ),
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "brand", "value": "Teekanne", "for_product": "tea"},
                ),
            ),
        ),
        SubtextEvalCase(
            case_id="rainbow_control",
            category="control",
            query_text="Erklär mir kurz, was ein Regenbogen ist.",
            hidden_context=(
                "The user has shopping preferences and a plan to go for a walk, but those are unrelated to a simple explanation question."
            ),
            desired_behavior=(
                "The reply should stay on the rainbow explanation and not force personal context into the answer."
            ),
            should_use_personal_context=False,
            seed_actions=(
                SubtextSeedAction(
                    kind="preference",
                    args={"category": "store", "value": "Markt Z", "for_product": "coffee"},
                ),
                SubtextSeedAction(
                    kind="plan",
                    args={"summary": "Go for a walk in the park", "when_text": "today"},
                ),
            ),
        ),
    )


def run_subtext_response_eval(
    *,
    env_path: str | Path = ".env",
    cases: tuple[SubtextEvalCase, ...] | None = None,
) -> SubtextEvalResult:
    base_config = TwinrConfig.from_env(env_path)
    if not base_config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the live subtext response eval.")
    eval_cases = cases or default_subtext_eval_cases()
    judge_backend = OpenAIBackend(
        replace(base_config, openai_reasoning_effort="low", openai_realtime_language="en"),
        base_instructions="",
    )
    case_results = tuple(_run_case(case=case, base_config=base_config, judge_backend=judge_backend) for case in eval_cases)
    summary = _summarize(case_results)
    return SubtextEvalResult(summary=summary, cases=case_results)


def _run_case(
    *,
    case: SubtextEvalCase,
    base_config: TwinrConfig,
    judge_backend: OpenAIBackend,
) -> SubtextEvalCaseResult:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = replace(
            base_config,
            project_root=temp_dir,
            personality_dir="personality",
            memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
            long_term_memory_enabled=True,
            long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            long_term_memory_recall_limit=4,
            long_term_memory_background_store_turns=True,
            openai_enable_web_search=False,
            restore_runtime_state_on_startup=False,
            openai_reasoning_effort="low",
        )
        backend = OpenAIBackend(config, base_instructions="")
        runtime = TwinrRuntime(config)
        try:
            _apply_seed_actions(runtime, case.seed_actions)
            runtime.flush_long_term_memory(timeout_s=2.0)
            runtime.last_transcript = case.query_text
            response = backend.respond_with_metadata(
                case.query_text,
                conversation=runtime.provider_conversation_context(),
                allow_web_search=False,
            )
        finally:
            runtime.shutdown(timeout_s=2.0)
    explicit_memory = contains_explicit_memory_announcement(response.text)
    judge = _judge_case(
        case=case,
        response_text=response.text,
        explicit_memory_announcement=explicit_memory,
        judge_backend=judge_backend,
    )
    token_usage = response.token_usage
    return SubtextEvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        query_text=case.query_text,
        response_text=response.text,
        explicit_memory_announcement=explicit_memory,
        judge=judge,
        model=response.model,
        request_id=response.request_id,
        response_id=response.response_id,
        prompt_tokens=getattr(token_usage, "input_tokens", None),
        output_tokens=getattr(token_usage, "output_tokens", None),
    )


def _apply_seed_actions(runtime: TwinrRuntime, actions: tuple[SubtextSeedAction, ...]) -> None:
    for action in actions:
        if action.kind == "preference":
            runtime.remember_preference(**action.args)
        elif action.kind == "contact":
            runtime.remember_contact(**action.args)
        elif action.kind == "plan":
            runtime.remember_plan(**action.args)
        elif action.kind == "episode":
            result = runtime.long_term_memory.enqueue_conversation_turn(**action.args)
            if result is None or not result.accepted:
                raise RuntimeError(f"Failed to enqueue episodic memory for eval action: {action}")
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported seed action kind: {action.kind}")


def _judge_case(
    *,
    case: SubtextEvalCase,
    response_text: str,
    explicit_memory_announcement: bool,
    judge_backend: OpenAIBackend,
) -> SubtextJudgeResult:
    prompt = "\n".join(
        [
            "You are grading whether an assistant used hidden personal context naturally in a reply.",
            "Return JSON only with keys helpful_context_used, subtle_not_explicit, unforced, addresses_request, naturalness_score, passed, reason.",
            f"Should use personal context: {'yes' if case.should_use_personal_context else 'no'}",
            f"Hidden context: {case.hidden_context}",
            f"Desired behavior: {case.desired_behavior}",
            f"User query: {case.query_text}",
            f"Assistant reply: {response_text}",
            f"Detected explicit memory announcement by rule-based scan: {'yes' if explicit_memory_announcement else 'no'}",
            "Rubric: helpful_context_used means the hidden context actually improved or personalized the answer when it should have, or was correctly omitted when it should not be used.",
            "subtle_not_explicit must be false if the assistant openly announces hidden memory or prior conversation.",
            "unforced must be false if the answer feels creepy, off-topic, or inserts irrelevant personal detail.",
            "addresses_request must be true only if the answer directly addresses what the user asked.",
            "naturalness_score must be an integer from 1 to 5.",
            "passed should be true only if the reply addresses the request, avoids explicit memory announcement, and handles personal context appropriately for this case.",
        ]
    )
    payload: dict[str, Any] | None = None
    for attempt in range(2):
        instructions = "Return only compact JSON. Do not use markdown fences."
        max_output_tokens = 220
        if attempt == 1:
            instructions = (
                "Return one single-line JSON object only. Keep reason under 20 words. "
                "Do not use markdown fences or any prose outside the JSON object."
            )
            max_output_tokens = 320
        request = judge_backend._build_response_request(
            prompt,
            instructions=instructions,
            allow_web_search=False,
            model=judge_backend.config.default_model,
            reasoning_effort="low",
            max_output_tokens=max_output_tokens,
        )
        response = judge_backend._client.responses.create(**request)
        text = judge_backend._extract_output_text(response)
        try:
            payload = _extract_json_object(text)
            break
        except ValueError:
            if attempt == 1:
                raise
    if payload is None:  # pragma: no cover - defensive
        raise RuntimeError("Judge payload was not produced.")
    subtle = bool(payload.get("subtle_not_explicit", False)) and not explicit_memory_announcement
    helpful = bool(payload.get("helpful_context_used", False))
    unforced = bool(payload.get("unforced", False))
    addresses = bool(payload.get("addresses_request", False))
    naturalness = int(payload.get("naturalness_score", 1))
    passed = bool(payload.get("passed", False)) and subtle and addresses
    return SubtextJudgeResult(
        helpful_context_used=helpful,
        subtle_not_explicit=subtle,
        unforced=unforced,
        addresses_request=addresses,
        naturalness_score=max(1, min(5, naturalness)),
        passed=passed,
        reason=str(payload.get("reason", "")).strip(),
    )


def contains_explicit_memory_announcement(text: str) -> bool:
    normalized = str(text or "").strip()
    return any(pattern.search(normalized) is not None for pattern in _EXPLICIT_MEMORY_PATTERNS)


def _extract_json_object(text: str) -> dict[str, Any]:
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        raise ValueError(f"Judge response did not contain JSON: {text!r}")
    return json.loads(match.group(0))


def _summarize(case_results: tuple[SubtextEvalCaseResult, ...]) -> SubtextEvalSummary:
    total = len(case_results)
    passed = sum(1 for item in case_results if item.judge.passed)
    category_case_counts: dict[str, int] = {}
    category_pass_counts: dict[str, int] = {}
    for item in case_results:
        category_case_counts[item.category] = category_case_counts.get(item.category, 0) + 1
        if item.judge.passed:
            category_pass_counts[item.category] = category_pass_counts.get(item.category, 0) + 1
    category_accuracy = {
        category: (
            category_pass_counts.get(category, 0) / count if count else 0.0
        )
        for category, count in category_case_counts.items()
    }
    explicit_violations = sum(1 for item in case_results if item.explicit_memory_announcement)
    average_naturalness = (
        sum(item.judge.naturalness_score for item in case_results) / total if total else 0.0
    )
    return SubtextEvalSummary(
        total_cases=total,
        passed_cases=passed,
        accuracy=(passed / total) if total else 0.0,
        category_case_counts=category_case_counts,
        category_pass_counts=category_pass_counts,
        category_accuracy=category_accuracy,
        explicit_memory_violations=explicit_violations,
        average_naturalness=average_naturalness,
    )


def _result_to_payload(result: SubtextEvalResult) -> dict[str, Any]:
    return {
        "summary": asdict(result.summary),
        "cases": [asdict(case) for case in result.cases],
    }


def main() -> int:
    result = run_subtext_response_eval()
    print(json.dumps(_result_to_payload(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
