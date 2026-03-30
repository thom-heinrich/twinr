"""Apply the repo-local git guard policy to changed paths and added lines."""

from __future__ import annotations

import ast
from pathlib import Path

from git_guard_tool.policy import GuardPolicy
from git_guard_tool.types import AddedLine, PathChange, ScanIssue, ScanResult

_PHONE_SEPARATORS = set(" +()-./")
_PLACEHOLDER_CHARS = set("*xX#.")
_HTTP_HEADER_TOKEN_CHARS = set("!#$%&'*+-.^_`|~")
_TEST_PATH_MARKERS = ("/test/", "/tests/", ".test.", "_test.")
_PREFIX_METADATA_KEY_TOKENS = frozenset({"prefix", "namespace"})
_PREFIX_METADATA_DELIMITERS = frozenset({":", "-", "_", "/", "."})


def _trim_excerpt(text: str, *, max_len: int = 120) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 3]}..."


def _normalize_key(text: str) -> str:
    normalized: list[str] = []
    for character in text.casefold():
        if character.isalnum():
            normalized.append(character)
        else:
            normalized.append("_")
    return "".join(normalized)


def _normalize_value(text: str) -> str:
    return text.strip().strip(",").strip("\"'").casefold()


def _key_matches_sensitive_fragment(key_text: str, fragments: tuple[str, ...]) -> bool:
    normalized_key = _normalize_key(key_text)
    key_tokens = tuple(token for token in normalized_key.split("_") if token)
    key_variants = {normalized_key, *key_tokens}
    key_variants.update(
        "_".join(key_tokens[index : index + 2])
        for index in range(len(key_tokens) - 1)
    )
    return any(fragment in key_variants for fragment in fragments)


def _looks_like_placeholder(value: str) -> bool:
    if not value:
        return True
    if all(character in _PLACEHOLDER_CHARS for character in value):
        return True
    return False


def _looks_like_placeholder_token_text(value: str, placeholder_values: tuple[str, ...]) -> bool:
    normalized = _normalize_key(value)
    tokens = tuple(token for token in normalized.split("_") if token)
    if not tokens:
        return False
    return all(token in placeholder_values or _looks_like_placeholder(token) for token in tokens)


def _is_test_like_path(path: str) -> bool:
    lowered_path = path.casefold()
    file_name = Path(lowered_path).name
    if lowered_path.startswith("test/") or lowered_path.startswith("tests/"):
        return True
    if file_name.startswith("test_") or file_name.endswith("_test.py"):
        return True
    return any(marker in lowered_path for marker in _TEST_PATH_MARKERS)


def _line_defines_blocked_terms(text: str) -> bool:
    normalized = _normalize_key(text)
    return "blocked_terms" in normalized or "blocked_term" in normalized


def _iter_token_like_fragments(text: str) -> tuple[str, ...]:
    fragments: list[str] = []
    current: list[str] = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            current.append(character)
            continue
        if current:
            fragments.append("".join(current))
            current.clear()
    if current:
        fragments.append("".join(current))
    return tuple(fragments)


def _is_allowed_test_secret_placeholder(
    fragment: str,
    *,
    prefixes: tuple[str, ...],
    placeholder_values: tuple[str, ...],
) -> bool:
    normalized_fragment = fragment.strip()
    if not normalized_fragment:
        return False
    for prefix in sorted((item for item in prefixes if normalized_fragment.startswith(item)), key=len, reverse=True):
        suffix = normalized_fragment[len(prefix) :]
        if _looks_like_placeholder_token_text(suffix, placeholder_values):
            return True
    return False


def _iter_digit_groups(candidate: str) -> tuple[str, ...]:
    groups: list[str] = []
    current: list[str] = []
    for character in candidate:
        if character.isdigit():
            current.append(character)
            continue
        if current:
            groups.append("".join(current))
            current.clear()
    if current:
        groups.append("".join(current))
    return tuple(groups)


def _find_assignment(text: str) -> tuple[str, str] | None:
    if "=" in text:
        left, right = text.split("=", maxsplit=1)
        key = left.strip()
        if ":" in key:
            key = key.split(":", maxsplit=1)[0].strip()
        if any(character in key for character in "[]{}(),"):
            return None
        if key and right.strip():
            return key, right
    stripped = text.strip()
    if ":" not in stripped:
        return None
    left, right = stripped.split(":", maxsplit=1)
    key = left.strip().strip("\"'")
    if not key or not right.strip():
        return None
    if any(character.isspace() for character in key):
        return None
    if any(character in key for character in "()[],"):
        return None
    return key, right


def _looks_like_date(candidate: str) -> bool:
    for separator in ("-", ".", "/"):
        parts = [part for part in candidate.split(separator) if part]
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        lengths = [len(part) for part in parts]
        if lengths in ([4, 2, 2], [2, 2, 4]):
            return True
    return False


def _looks_like_ip(candidate: str) -> bool:
    parts = candidate.split(".")
    if len(parts) != 4:
        return False
    if not all(part.isdigit() for part in parts):
        return False
    return all(0 <= int(part) <= 255 for part in parts)


def _looks_like_small_version(candidate: str, digit_count: int) -> bool:
    parts = candidate.split(".")
    if len(parts) not in {2, 3}:
        return False
    if not all(part.isdigit() for part in parts):
        return False
    return digit_count <= 6 and all(len(part) <= 3 for part in parts)


def _looks_like_phone_shape(candidate: str) -> bool:
    if candidate and not candidate[-1].isdigit():
        return False
    if candidate.count("+") > 1 or ("+" in candidate and not candidate.startswith("+")):
        return False
    if candidate.count("/") > 1:
        return False
    digit_groups = _iter_digit_groups(candidate)
    if len(digit_groups) < 2:
        return False
    group_lengths = tuple(len(group) for group in digit_groups)
    if any(length == 1 for length in group_lengths[1:]):
        return False
    separator_chars = {
        character
        for character in candidate
        if not character.isdigit() and character != "+"
    }
    if separator_chars == {"."} and len(group_lengths) < 3:
        return False
    if max(group_lengths) >= 5:
        return True
    if len(group_lengths) >= 4 and all(length == 2 for length in group_lengths):
        return separator_chars.issubset({"-", " "})
    if candidate.startswith("+") and sum(group_lengths) >= 8 and len(group_lengths) >= 3:
        return True
    if "(" in candidate and ")" in candidate and max(group_lengths) >= 4:
        return True
    return False


def _iter_phone_candidates(text: str, *, min_digits: int, max_digits: int) -> tuple[str, ...]:
    candidates: list[str] = []
    start: int | None = None
    for index, character in enumerate(text):
        is_phone_character = character.isdigit() or character in _PHONE_SEPARATORS
        if start is None:
            if character.isdigit() or (character == "+" and index + 1 < len(text) and text[index + 1].isdigit()):
                start = index
            continue
        if is_phone_character:
            continue
        candidate = text[start:index].strip()
        start = None
        if not candidate:
            continue
        digit_count = sum(character.isdigit() for character in candidate)
        if digit_count < min_digits or digit_count > max_digits:
            continue
        if not _looks_like_phone_shape(candidate):
            continue
        if _looks_like_date(candidate) or _looks_like_ip(candidate) or _looks_like_small_version(candidate, digit_count):
            continue
        candidates.append(candidate)
    if start is not None:
        candidate = text[start:].strip()
        digit_count = sum(character.isdigit() for character in candidate)
        if digit_count >= min_digits and digit_count <= max_digits and _looks_like_phone_shape(candidate):
            if not _looks_like_date(candidate) and not _looks_like_ip(candidate) and not _looks_like_small_version(candidate, digit_count):
                candidates.append(candidate)
    return tuple(dict.fromkeys(candidates))


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip().strip(",")
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", "\""}:
        return stripped[1:-1]
    return stripped


def _unwrap_literal_value(text: str) -> str | None:
    parsed_string_literal = _parse_plain_string_literal(text)
    if parsed_string_literal is not None:
        return parsed_string_literal
    if _looks_like_bare_literal_value(text):
        return _strip_wrapping_quotes(text)
    return None


def _parse_plain_string_literal(text: str) -> str | None:
    stripped = text.strip().strip(",")
    if not stripped:
        return None
    try:
        parsed = ast.parse(stripped, mode="eval")
    except SyntaxError:
        return None
    if isinstance(parsed.body, ast.Constant):
        if isinstance(parsed.body.value, str):
            return parsed.body.value
        if isinstance(parsed.body.value, bytes):
            try:
                return parsed.body.value.decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


def _looks_like_path_literal(text: str) -> bool:
    return text.startswith(("/", "./", "../", "~", "\\")) or text.startswith(("http://", "https://"))


def _looks_like_regex_literal(text: str) -> bool:
    return "(?:" in text or "(?P" in text or "\\d" in text or "\\w" in text or "[" in text or "|" in text


def _looks_like_bare_literal_value(text: str) -> bool:
    stripped = text.strip().strip(",")
    if not stripped:
        return False
    if any(character in stripped for character in "\"'`()[]{}"):
        return False
    if any(character.isspace() for character in stripped):
        return False
    if any(character in stripped for character in "./:\\"):
        return False
    return True


def _is_custom_header_name_metadata_assignment(key_text: str, value_text: str) -> bool:
    normalized_key = _normalize_key(key_text)
    key_tokens = tuple(token for token in normalized_key.split("_") if token)
    if "header" not in key_tokens or "name" not in key_tokens:
        return False

    literal_value = _unwrap_literal_value(value_text)
    if literal_value is None:
        return False
    normalized_value = literal_value.strip()
    if not normalized_value or not normalized_value.casefold().startswith("x-"):
        return False
    if any(character.isspace() for character in normalized_value):
        return False
    return all(character.isalnum() or character in _HTTP_HEADER_TOKEN_CHARS for character in normalized_value)


def _is_prefix_metadata_assignment(key_text: str, value_text: str) -> bool:
    normalized_key = _normalize_key(key_text)
    key_tokens = tuple(token for token in normalized_key.split("_") if token)
    if not any(token in _PREFIX_METADATA_KEY_TOKENS for token in key_tokens):
        return False

    literal_value = _unwrap_literal_value(value_text)
    if literal_value is None:
        return False
    normalized_value = literal_value.strip()
    if not normalized_value or any(character.isspace() for character in normalized_value):
        return False
    if normalized_value[-1] not in _PREFIX_METADATA_DELIMITERS:
        return False
    return all(character.isalnum() or character in _PREFIX_METADATA_DELIMITERS for character in normalized_value)


def _looks_like_sensitive_literal_value(value_text: str, *, min_length: int) -> bool:
    unwrapped = _unwrap_literal_value(value_text)
    if unwrapped is None:
        return False
    if _looks_like_path_literal(unwrapped) or _looks_like_regex_literal(unwrapped):
        return False
    normalized = _normalize_value(unwrapped)
    if normalized in {"true", "false", "none", "null"}:
        return False
    if normalized in {"yes", "no"}:
        return False
    if normalized in {"0", "1"}:
        return False
    if normalized in {"admin", "password"}:
        return False
    if normalized.isidentifier():
        return False
    if normalized not in {"password.123"} and len(normalized) < min_length:
        return False
    if any(character.isspace() for character in unwrapped):
        return False
    return True


def _scan_path_change(path_change: PathChange, policy: GuardPolicy, issues: list[ScanIssue]) -> None:
    for candidate_path in filter(None, (path_change.path, path_change.previous_path)):
        lowered_path = candidate_path.casefold()
        file_name = Path(candidate_path).name.casefold()
        if policy.ignores_path(candidate_path):
            continue
        if file_name in policy.files.blocked_exact_names:
            issues.append(
                ScanIssue(
                    rule_id="blocked-path-name",
                    message=f"blocked file name detected: {file_name}",
                    path=path_change.path,
                    commit=path_change.commit,
                )
            )
            return
        if any(lowered_path.endswith(suffix) for suffix in policy.files.blocked_suffixes):
            issues.append(
                ScanIssue(
                    rule_id="blocked-path-suffix",
                    message=f"blocked file suffix detected in {candidate_path}",
                    path=path_change.path,
                    commit=path_change.commit,
                )
            )
            return
        for term in policy.content.blocked_terms:
            if term in lowered_path:
                issues.append(
                    ScanIssue(
                        rule_id="blocked-term-path",
                        message=f"blocked term `{term}` detected in path",
                        path=path_change.path,
                        commit=path_change.commit,
                    )
                )
                return


def _scan_added_line(added_line: AddedLine, policy: GuardPolicy, issues: list[ScanIssue]) -> None:
    if policy.ignores_path(added_line.path):
        return
    is_test_like_path = _is_test_like_path(added_line.path)
    lowered = added_line.text.casefold()

    if not is_test_like_path and not _line_defines_blocked_terms(added_line.text):
        for term in policy.content.blocked_terms:
            if term in lowered:
                issues.append(
                    ScanIssue(
                        rule_id="blocked-term-content",
                        message=f"blocked term `{term}` detected in added content",
                        path=added_line.path,
                        line_number=added_line.line_number,
                        excerpt=_trim_excerpt(added_line.text),
                        commit=added_line.commit,
                    )
                )
                return

    stripped = added_line.text.strip()
    if stripped.startswith("-----BEGIN "):
        issues.append(
            ScanIssue(
                rule_id="pem-material",
                message="PEM/private-key material detected in added content",
                path=added_line.path,
                line_number=added_line.line_number,
                excerpt=_trim_excerpt(added_line.text),
                commit=added_line.commit,
            )
        )
        return

    for fragment in _iter_token_like_fragments(added_line.text):
        if is_test_like_path and _is_allowed_test_secret_placeholder(
            fragment,
            prefixes=policy.content.secret_prefixes,
            placeholder_values=policy.content.placeholder_values,
        ):
            continue
        for prefix in policy.content.secret_prefixes:
            if not fragment.startswith(prefix):
                continue
            if len(fragment) < len(prefix) + policy.content.secret_min_length:
                continue
            issues.append(
                ScanIssue(
                    rule_id="secret-prefix",
                    message=f"secret-like token with prefix `{prefix}` detected",
                    path=added_line.path,
                    line_number=added_line.line_number,
                    excerpt=_trim_excerpt(fragment),
                    commit=added_line.commit,
                )
            )
            return

    assignment = _find_assignment(added_line.text)
    if assignment is not None and not is_test_like_path:
        key_text, value_text = assignment
        normalized_value = _normalize_value(value_text)
        if _key_matches_sensitive_fragment(key_text, policy.content.sensitive_key_fragments):
            if normalized_value not in policy.content.placeholder_values and not _looks_like_placeholder(normalized_value):
                if _is_custom_header_name_metadata_assignment(key_text, value_text):
                    return
                if _is_prefix_metadata_assignment(key_text, value_text):
                    return
                if _looks_like_sensitive_literal_value(value_text, min_length=policy.content.secret_min_length):
                    issues.append(
                        ScanIssue(
                            rule_id="sensitive-assignment",
                            message=f"sensitive assignment detected for key `{key_text}`",
                            path=added_line.path,
                            line_number=added_line.line_number,
                            excerpt=_trim_excerpt(added_line.text),
                            commit=added_line.commit,
                        )
                    )
                    return

    if not is_test_like_path:
        phone_candidates = _iter_phone_candidates(
            added_line.text,
            min_digits=policy.phones.min_digits,
            max_digits=policy.phones.max_digits,
        )
        if phone_candidates:
            issues.append(
                ScanIssue(
                    rule_id="phone-number",
                    message="phone-like number detected in added content",
                    path=added_line.path,
                    line_number=added_line.line_number,
                    excerpt=_trim_excerpt(phone_candidates[0]),
                    commit=added_line.commit,
                )
            )


def scan_changes(
    *,
    path_changes: tuple[PathChange, ...],
    added_lines: tuple[AddedLine, ...],
    policy: GuardPolicy,
) -> ScanResult:
    """Run all configured checks over the supplied git changes."""

    issues: list[ScanIssue] = []
    for path_change in path_changes:
        _scan_path_change(path_change, policy, issues)
        if len(issues) >= policy.max_issues:
            return ScanResult(issues=tuple(issues[: policy.max_issues]))
    for added_line in added_lines:
        _scan_added_line(added_line, policy, issues)
        if len(issues) >= policy.max_issues:
            return ScanResult(issues=tuple(issues[: policy.max_issues]))
    return ScanResult(issues=tuple(issues))
