"""Fix cycle reporter for documenting diagnostic fix cycles.

This module provides utilities for documenting each diagnostic fix cycle
in the balance-core controller validation workflow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class FixCycleRecord:
    """Record of a single diagnostic fix cycle.

    Attributes:
        cycle_number: Sequential fix cycle number (1, 2, 3, ...)
        classified_failure_mode: Failure mode code (e.g., "F2.1", "F3.2")
        responsible_component: Component identified as responsible for failure
        evidence_fields: Dictionary of telemetry fields that provided evidence
        allowed_fix_scope: Description of what can be modified in this cycle
        files_changed: List of file paths that were modified
        parameters_before: Dictionary of parameter values before the fix
        parameters_after: Dictionary of parameter values after the fix
        validation_command: Command used to validate the fix
        validation_result_before: Validation result before the fix
        validation_result_after: Validation result after the fix
        failure_resolved: Whether the original failure was resolved
        new_failure_appeared: Whether a new failure appeared after the fix
        structural_invariants_after_fix: Dictionary of structural invariant check results
        notes: Optional additional notes about the fix cycle
    """
    cycle_number: int
    classified_failure_mode: str
    responsible_component: str
    evidence_fields: Dict[str, Any]
    allowed_fix_scope: str
    files_changed: List[str]
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    validation_command: str
    validation_result_before: str
    validation_result_after: str
    failure_resolved: bool
    new_failure_appeared: bool
    structural_invariants_after_fix: Dict[str, str]
    notes: Optional[str] = None


class FixCycleReporter:
    """Reporter for generating fix cycle documentation."""

    def generate_markdown(self, record: FixCycleRecord) -> str:
        """Generate markdown documentation for a fix cycle.

        Args:
            record: Fix cycle record to document

        Returns:
            Markdown-formatted documentation string
        """
        sections = []

        # Title
        sections.append(f"# Fix Cycle {record.cycle_number}\n")

        # Classification
        sections.append("## Classification\n")
        sections.append(f"- **Failure Mode**: {record.classified_failure_mode}")
        sections.append(f"- **Responsible Component**: {record.responsible_component}")
        sections.append(f"- **Allowed Fix Scope**: {record.allowed_fix_scope}\n")

        # Evidence
        sections.append("## Evidence\n")
        if record.evidence_fields:
            for field, value in record.evidence_fields.items():
                sections.append(f"- `{field}`: {value}")
        else:
            sections.append("- No evidence fields recorded")
        sections.append("")

        # Changes Made
        sections.append("## Changes Made\n")

        # Files changed
        sections.append("### Files Changed\n")
        if record.files_changed:
            for file_path in record.files_changed:
                sections.append(f"- `{file_path}`")
        else:
            sections.append("- No files changed")
        sections.append("")

        # Parameters
        sections.append("### Parameters\n")
        sections.append("**Before:**\n")
        if record.parameters_before:
            sections.append("```python")
            for param, value in record.parameters_before.items():
                sections.append(f"{param}: {value}")
            sections.append("```\n")
        else:
            sections.append("- No parameters recorded\n")

        sections.append("**After:**\n")
        if record.parameters_after:
            sections.append("```python")
            for param, value in record.parameters_after.items():
                sections.append(f"{param}: {value}")
            sections.append("```\n")
        else:
            sections.append("- No parameters recorded\n")

        # Validation
        sections.append("## Validation\n")
        sections.append("### Command\n")
        sections.append(f"```bash\n{record.validation_command}\n```\n")

        sections.append("### Results\n")
        sections.append("**Before Fix:**\n")
        sections.append(f"```\n{record.validation_result_before}\n```\n")

        sections.append("**After Fix:**\n")
        sections.append(f"```\n{record.validation_result_after}\n```\n")

        sections.append("### Outcome\n")
        sections.append(f"- **Failure Resolved**: {'✓ Yes' if record.failure_resolved else '✗ No'}")
        sections.append(f"- **New Failure Appeared**: {'✓ Yes' if record.new_failure_appeared else '✗ No'}\n")

        # Structural Invariants
        sections.append("## Structural Invariants After Fix\n")
        if record.structural_invariants_after_fix:
            for check, result in record.structural_invariants_after_fix.items():
                status_symbol = "✓" if result.upper() == "PASS" else "✗"
                sections.append(f"- **{check}**: {status_symbol} {result}")
        else:
            sections.append("- No structural invariant checks recorded")
        sections.append("")

        # Notes
        if record.notes:
            sections.append("## Notes\n")
            sections.append(f"{record.notes}\n")

        return "\n".join(sections)

    def generate_summary(self, records: List[FixCycleRecord]) -> str:
        """Generate summary of multiple fix cycles.

        Args:
            records: List of fix cycle records

        Returns:
            Markdown-formatted summary string
        """
        if not records:
            return "# Fix Cycle Summary\n\nNo fix cycles recorded.\n"

        sections = []
        sections.append("# Fix Cycle Summary\n")
        sections.append(f"**Total Cycles**: {len(records)}\n")

        # Count resolved vs unresolved
        resolved_count = sum(1 for r in records if r.failure_resolved)
        sections.append(f"**Resolved**: {resolved_count}/{len(records)}\n")

        # List all cycles
        sections.append("## Cycles\n")
        for record in records:
            status = "✓ RESOLVED" if record.failure_resolved else "✗ UNRESOLVED"
            new_failure = " (new failure appeared)" if record.new_failure_appeared else ""
            sections.append(
                f"{record.cycle_number}. {record.classified_failure_mode} - "
                f"{record.responsible_component} - {status}{new_failure}"
            )
        sections.append("")

        return "\n".join(sections)
