"""Compile self-coding targets into validated Twinr artifacts."""

from twinr.agent.self_coding.compiler.automation_target import (
    AutomationManifestCompilerError,
    CompiledAutomationManifest,
    compile_automation_manifest_content,
)
from twinr.agent.self_coding.compiler.skill_target import SkillPackageCompilerError, compile_skill_package_content
from twinr.agent.self_coding.compiler.prompting import build_compile_prompt
from twinr.agent.self_coding.compiler.validator import CompileArtifactValidationError, validate_compile_artifact

__all__ = [
    "AutomationManifestCompilerError",
    "CompileArtifactValidationError",
    "CompiledAutomationManifest",
    "SkillPackageCompilerError",
    "build_compile_prompt",
    "compile_automation_manifest_content",
    "compile_skill_package_content",
    "validate_compile_artifact",
]
