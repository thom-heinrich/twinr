"""Compile self-coding targets into validated Twinr artifacts."""

from twinr.agent.self_coding.compiler.automation_target import (
    AutomationManifestCompilerError,
    CompiledAutomationManifest,
    compile_automation_manifest_content,
)
from twinr.agent.self_coding.compiler.prompting import build_compile_prompt

__all__ = [
    "AutomationManifestCompilerError",
    "CompiledAutomationManifest",
    "build_compile_prompt",
    "compile_automation_manifest_content",
]
