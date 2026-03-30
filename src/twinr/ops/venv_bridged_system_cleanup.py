"""Find direct project dependencies shadowed by bridged Pi system packages.

Twinr's Pi deploy intentionally exposes selected OS-managed ``dist-packages``
inside ``/twinr/.venv`` so the acceptance runtime can reuse stable host
packages. A preserved venv can still contain older pip-managed copies of those
same direct dependencies. When that happens, importlib metadata and pip resolve
against the venv copy first while imports may merge or prefer the bridged
system package, which can leave the package graph internally inconsistent.

This helper identifies direct project dependencies where both the venv and one
of the bridged Pi system paths provide the same distribution and the system
version still satisfies the repo's declared requirement. Deploy code can then
remove the stale venv copy so the bridged system package becomes the single
authoritative source again.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as importlib_metadata
from pathlib import Path
import re
import tomllib

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - exercised on older pip-only hosts
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement


_NORMALIZE_NAME_RE = re.compile(r"[-_.]+")


@dataclass(frozen=True, slots=True)
class BridgedSystemShadowedDistribution:
    """Describe one direct dependency duplicated in venv and system paths."""

    name: str
    venv_version: str
    venv_path: str
    system_version: str
    system_path: str


def find_shadowed_direct_dependency_distributions(
    *,
    project_pyproject: Path,
    venv_site_packages_dir: Path,
    bridged_system_paths: tuple[str | Path, ...],
) -> tuple[BridgedSystemShadowedDistribution, ...]:
    """Return direct dependencies whose venv copy should yield to the system copy."""

    direct_requirements = _load_direct_project_requirements(project_pyproject)
    if not direct_requirements:
        return ()

    venv_distributions = _load_distributions_for_path(venv_site_packages_dir)
    system_distributions = _load_system_distributions(tuple(Path(path) for path in bridged_system_paths))
    removable: list[BridgedSystemShadowedDistribution] = []
    for normalized_name, requirement in direct_requirements:
        venv_distribution = venv_distributions.get(normalized_name)
        if venv_distribution is None:
            continue
        system_distribution = system_distributions.get(normalized_name)
        if system_distribution is None:
            continue
        if requirement.specifier and not requirement.specifier.contains(
            system_distribution.version,
            prereleases=True,
        ):
            continue
        removable.append(
            BridgedSystemShadowedDistribution(
                name=_distribution_name(venv_distribution) or requirement.name,
                venv_version=str(venv_distribution.version),
                venv_path=str(venv_distribution.locate_file("")),
                system_version=str(system_distribution.version),
                system_path=str(system_distribution.locate_file("")),
            )
        )
    return tuple(removable)


def _load_direct_project_requirements(project_pyproject: Path) -> tuple[tuple[str, Requirement], ...]:
    payload = tomllib.loads(project_pyproject.read_text(encoding="utf-8"))
    raw_dependencies = payload.get("project", {}).get("dependencies", ())
    if not isinstance(raw_dependencies, list):
        return ()

    environment = default_environment()
    requirements: list[tuple[str, Requirement]] = []
    seen: set[str] = set()
    for raw_dependency in raw_dependencies:
        requirement = Requirement(str(raw_dependency))
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        normalized_name = _normalize_name(requirement.name)
        if normalized_name in seen:
            continue
        seen.add(normalized_name)
        requirements.append((normalized_name, requirement))
    return tuple(requirements)


def _load_distributions_for_path(path: Path) -> dict[str, importlib_metadata.Distribution]:
    if not path.exists():
        return {}
    distributions: dict[str, importlib_metadata.Distribution] = {}
    for distribution in importlib_metadata.distributions(path=[str(path)]):
        name = _distribution_name(distribution)
        if not name:
            continue
        distributions.setdefault(_normalize_name(name), distribution)
    return distributions


def _load_system_distributions(
    bridged_system_paths: tuple[Path, ...],
) -> dict[str, importlib_metadata.Distribution]:
    distributions: dict[str, importlib_metadata.Distribution] = {}
    for system_path in bridged_system_paths:
        for normalized_name, distribution in _load_distributions_for_path(system_path).items():
            distributions.setdefault(normalized_name, distribution)
    return distributions


def _distribution_name(distribution: importlib_metadata.Distribution) -> str:
    return str(distribution.metadata.get("Name", "") or "").strip()


def _normalize_name(value: str) -> str:
    return _NORMALIZE_NAME_RE.sub("-", str(value).strip()).lower()


__all__ = [
    "BridgedSystemShadowedDistribution",
    "find_shadowed_direct_dependency_distributions",
]
