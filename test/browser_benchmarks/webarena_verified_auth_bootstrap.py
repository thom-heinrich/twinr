"""Materialize official WebArena auth state for Twinr benchmark runs.

This module is intentionally benchmark-scoped. It mirrors the public WebArena
reference login setup from `browser_env/env_config.py` and
`browser_env/auto_login.py` so Twinr can run honest external benchmarks
without baking benchmark-specific accounts into runtime browser code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
import time

from playwright.sync_api import (
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from webarena_verified.api import WebArenaVerified
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaVerifiedTask

from test.browser_benchmarks.webarena_verified_auth import (
    WebArenaVerifiedAuthContext,
    derive_task_auth_context,
    load_raw_task_metadata,
    lookup_environment,
    resolve_storage_state_path,
)

_OFFICIAL_AUTH_CACHE_ROOT = Path(gettempdir()) / "twinr_webarena_verified_auth"
_VALIDATED_STORAGE_STATE_CACHE: dict[tuple[str, str, str, int], bool] = {}


@dataclass(frozen=True, slots=True)
class OfficialSiteCredentials:
    """Store one official benchmark account tuple."""

    username: str
    password: str


@dataclass(frozen=True, slots=True)
class SiteBootstrapSpec:
    """Describe one benchmark site's auth bootstrap path."""

    site_name: str
    login_url_suffix: str
    verification_url_suffix: str
    verification_keyword: str
    verification_exact_url: bool


_OFFICIAL_REFERENCE_CREDENTIALS: dict[str, OfficialSiteCredentials] = {
    "shopping": OfficialSiteCredentials(
        username="emma.lopez@gmail.com",
        password="Password.123",
    ),
    "reddit": OfficialSiteCredentials(
        username="MarvelsGrantMan136",
        password="test1234",
    ),
    "gitlab": OfficialSiteCredentials(
        username="byteblaze",
        password="hello1234",
    ),
    "shopping_admin": OfficialSiteCredentials(
        username="admin",
        password="admin1234",
    ),
}

_SITE_BOOTSTRAP_SPECS: dict[str, SiteBootstrapSpec] = {
    "shopping": SiteBootstrapSpec(
        site_name="shopping",
        login_url_suffix="/customer/account/login/",
        verification_url_suffix="/customer/account/",
        verification_keyword="",
        verification_exact_url=True,
    ),
    "reddit": SiteBootstrapSpec(
        site_name="reddit",
        login_url_suffix="/login",
        verification_url_suffix="/user/{username}/account",
        verification_keyword="Delete",
        verification_exact_url=False,
    ),
    "gitlab": SiteBootstrapSpec(
        site_name="gitlab",
        login_url_suffix="/users/sign_in",
        verification_url_suffix="/-/profile",
        verification_keyword="",
        verification_exact_url=True,
    ),
    "shopping_admin": SiteBootstrapSpec(
        site_name="shopping_admin",
        login_url_suffix="",
        verification_url_suffix="/dashboard",
        verification_keyword="Dashboard",
        verification_exact_url=False,
    ),
}


def ensure_task_auth_context(
    *,
    task: WebArenaVerifiedTask,
    config: WebArenaVerifiedConfig,
    benchmark: WebArenaVerified | None = None,
    auth_state_root: Path | None = None,
) -> WebArenaVerifiedAuthContext:
    """Resolve or materialize official login state for one benchmark task."""

    initial = derive_task_auth_context(
        task=task,
        config=config,
        benchmark=benchmark,
        auth_state_root=auth_state_root,
    )
    raw_task = load_raw_task_metadata(task_id=int(task.task_id))
    raw_storage_state = str(raw_task.get("storage_state") or "").strip()
    if initial.storage_state_path is not None and initial.require_login:
        site_name = str(task.sites[0].value)
        if _storage_state_grants_authenticated_access(
            site_name=site_name,
            config=config,
            storage_state_path=initial.storage_state_path,
        ):
            return initial
    elif initial.storage_state_path is not None or not initial.require_login:
        return initial

    if not raw_storage_state:
        return initial

    target_root = _resolve_auth_state_root(auth_state_root)
    candidate_path = _candidate_storage_state_path(
        raw_path=raw_storage_state,
        auth_state_root=target_root,
    )
    site_name = str(task.sites[0].value)
    force_refresh = bool(initial.storage_state_path is not None and initial.require_login)
    _materialize_site_auth_state(
        site_name=site_name,
        config=config,
        benchmark=benchmark,
        target_state_path=candidate_path,
        force_refresh=force_refresh,
    )
    materialized_path = resolve_storage_state_path(
        raw_path=raw_storage_state,
        auth_state_root=target_root,
    )
    return WebArenaVerifiedAuthContext(
        require_login=initial.require_login,
        storage_state_path=materialized_path,
        extra_http_headers=initial.extra_http_headers,
    )


def default_auth_state_root() -> Path:
    """Return the default cache root for generated benchmark auth state."""

    root = _OFFICIAL_AUTH_CACHE_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def official_reference_credentials_for_site(site_name: str) -> OfficialSiteCredentials | None:
    """Return official benchmark credentials for one supported site."""

    return _OFFICIAL_REFERENCE_CREDENTIALS.get(str(site_name or "").strip())


def _resolve_auth_state_root(auth_state_root: Path | None) -> Path:
    """Return the effective auth-state root, creating it when needed."""

    if auth_state_root is None:
        return default_auth_state_root()
    root = Path(auth_state_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _candidate_storage_state_path(*, raw_path: str, auth_state_root: Path) -> Path:
    """Return the concrete path where one storage state should live."""

    cleaned = str(raw_path or "").strip()
    if not cleaned:
        raise ValueError("raw_path is required to materialize benchmark auth state")
    candidate = (Path(auth_state_root).expanduser().resolve() / cleaned).resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _materialize_site_auth_state(
    *,
    site_name: str,
    config: WebArenaVerifiedConfig,
    benchmark: WebArenaVerified | None,
    target_state_path: Path,
    force_refresh: bool = False,
) -> None:
    """Create one storage-state file for a supported official benchmark site."""

    if target_state_path.is_file() and not force_refresh:
        return
    if force_refresh and target_state_path.exists():
        target_state_path.unlink()
    credentials = official_reference_credentials_for_site(site_name)
    if credentials is None:
        raise RuntimeError(f"No official WebArena credentials are known for site {site_name!r}.")
    environment = lookup_environment(config=config, site=site_name)
    if environment is None or not environment.active_url:
        raise RuntimeError(f"Missing active WebArena environment URL for site {site_name!r}.")
    base_url = str(environment.active_url).rstrip("/")
    spec = _SITE_BOOTSTRAP_SPECS.get(site_name)
    if spec is None:
        raise RuntimeError(f"No benchmark auth bootstrap flow is defined for site {site_name!r}.")

    if site_name == "shopping_admin" and environment.use_header_login and benchmark is not None:
        _materialize_header_login_state(
            site_name=site_name,
            base_url=base_url,
            username=credentials.username,
            benchmark=benchmark,
            target_state_path=target_state_path,
            verification_suffix=spec.verification_url_suffix,
        )
        return

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            page = context.new_page()
            _perform_site_login(
                page=page,
                site_name=site_name,
                base_url=base_url,
                credentials=credentials,
            )
            _wait_for_logged_in_state(
                context=context,
                base_url=base_url,
                spec=spec,
                credentials=credentials,
            )
            context.storage_state(path=str(target_state_path))
        finally:
            context.close()
            browser.close()


def _storage_state_grants_authenticated_access(
    *,
    site_name: str,
    config: WebArenaVerifiedConfig,
    storage_state_path: str,
) -> bool:
    """Return whether one cached storage state still authenticates the live env."""

    candidate = Path(str(storage_state_path or "")).expanduser().resolve()
    if not candidate.is_file():
        return False
    spec = _SITE_BOOTSTRAP_SPECS.get(str(site_name or "").strip())
    environment = lookup_environment(config=config, site=site_name)
    credentials = official_reference_credentials_for_site(site_name)
    if spec is None or environment is None or not environment.active_url or credentials is None:
        return False
    verify_url = (
        f"{str(environment.active_url).rstrip('/')}"
        f"{spec.verification_url_suffix.format(username=credentials.username)}"
    )
    valid = False
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(storage_state=str(candidate))
            try:
                page = context.new_page()
                page.goto(verify_url, wait_until="domcontentloaded", timeout=30_000)
                _wait_for_network_idle(page)
                valid = _page_matches_logged_in_target(
                    page=page,
                    expected_url=verify_url,
                    expected_keyword=spec.verification_keyword,
                    exact_url_match=spec.verification_exact_url,
                )
            finally:
                context.close()
                browser.close()
    except Exception:
        valid = False
    return valid


def _materialize_header_login_state(
    *,
    site_name: str,
    base_url: str,
    username: str,
    benchmark: WebArenaVerified,
    target_state_path: Path,
    verification_suffix: str,
) -> None:
    """Use an official header-login site to mint a reusable storage state."""

    header_name = benchmark.get_custom_auth_header_name(site_name)
    if not header_name:
        raise RuntimeError(f"Official header login is unavailable for site {site_name!r}.")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(extra_http_headers={header_name: username})
        try:
            page = context.new_page()
            page.goto(f"{base_url}{verification_suffix}", wait_until="domcontentloaded", timeout=30_000)
            _wait_for_network_idle(page)
            context.storage_state(path=str(target_state_path))
        finally:
            context.close()
            browser.close()


def _perform_site_login(
    *,
    page: Page,
    site_name: str,
    base_url: str,
    credentials: OfficialSiteCredentials,
) -> None:
    """Execute the official login flow for one supported benchmark site."""

    if site_name == "shopping":
        page.goto(f"{base_url}/customer/account/login/", wait_until="domcontentloaded", timeout=30_000)
        page.get_by_label("Email", exact=True).fill(credentials.username)
        page.get_by_label("Password", exact=True).fill(credentials.password)
        page.get_by_role("button", name="Sign In").click()
        return
    if site_name == "reddit":
        page.goto(f"{base_url}/login", wait_until="domcontentloaded", timeout=30_000)
        page.get_by_label("Username").fill(credentials.username)
        page.get_by_label("Password").fill(credentials.password)
        page.get_by_role("button", name="Log in").click()
        return
    if site_name == "shopping_admin":
        page.goto(base_url, wait_until="domcontentloaded", timeout=30_000)
        page.get_by_placeholder("user name").fill(credentials.username)
        page.get_by_placeholder("password").fill(credentials.password)
        page.get_by_role("button", name="Sign in").click()
        return
    if site_name == "gitlab":
        page.goto(f"{base_url}/users/sign_in", wait_until="domcontentloaded", timeout=30_000)
        page.get_by_test_id("username-field").fill(credentials.username)
        page.get_by_test_id("password-field").fill(credentials.password)
        page.get_by_test_id("sign-in-button").click()
        return
    raise RuntimeError(f"Unsupported benchmark auth bootstrap site: {site_name!r}")


def _wait_for_logged_in_state(
    *,
    context: BrowserContext,
    base_url: str,
    spec: SiteBootstrapSpec,
    credentials: OfficialSiteCredentials,
) -> None:
    """Verify that the created auth state actually represents a logged-in session."""

    verify_url = f"{base_url}{spec.verification_url_suffix.format(username=credentials.username)}"
    for _ in range(3):
        page = context.new_page()
        try:
            page.goto(verify_url, wait_until="domcontentloaded", timeout=30_000)
            _wait_for_network_idle(page)
            if _page_matches_logged_in_target(
                page=page,
                expected_url=verify_url,
                expected_keyword=spec.verification_keyword,
                exact_url_match=spec.verification_exact_url,
            ):
                return
        finally:
            page.close()
        time.sleep(0.5)
    raise RuntimeError(f"Failed to verify logged-in benchmark state for site {spec.site_name!r}.")


def _page_matches_logged_in_target(
    *,
    page: Page,
    expected_url: str,
    expected_keyword: str,
    exact_url_match: bool,
) -> bool:
    """Return whether one verification page looks logged-in."""

    current_url = str(page.url or "")
    if exact_url_match and current_url != expected_url:
        return False
    if not exact_url_match and expected_url not in current_url:
        return False
    if not expected_keyword:
        return True
    content = str(page.content() or "")
    return expected_keyword in content


def _wait_for_network_idle(page: Page) -> None:
    """Wait briefly for post-login navigation or rendering to settle."""

    try:
        page.wait_for_load_state("networkidle", timeout=5_000)
    except PlaywrightTimeoutError:
        pass
