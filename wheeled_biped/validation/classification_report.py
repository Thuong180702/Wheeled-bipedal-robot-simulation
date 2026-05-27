# wheeled_biped/validation/classification_report.py
"""Generate structured JSON and markdown reports from classification results."""

import json
from typing import Dict, Any
from wheeled_biped.validation.failure_classifier import ClassificationResult


class ClassificationReportGenerator:
    """Generates JSON and markdown reports from classification results."""

    def to_json(self, result: ClassificationResult) -> str:
        """Convert classification result to JSON string.

        Args:
            result: ClassificationResult to convert

        Returns:
            JSON string representation
        """
        report_dict = self._build_report_dict(result)
        return json.dumps(report_dict, indent=2)

    def to_markdown(self, result: ClassificationResult) -> str:
        """Convert classification result to markdown report.

        Args:
            result: ClassificationResult to convert

        Returns:
            Markdown string representation
        """
        lines = []
        lines.append("# Balance-Core Failure Classification Report")
        lines.append("")
        lines.append(f"**Primary Failure Mode:** {result.primary_failure_mode.value}")
        lines.append(f"**First Threshold Crossing:** Step {result.first_threshold_crossing_step} (t={result.first_threshold_crossing_time_s:.3f}s)")
        lines.append(f"**Responsible Component:** {result.responsible_component}")
        lines.append(f"**Fix Allowed in Balance-Core:** {'Yes' if result.fix_allowed_in_balance_core else 'No'}")
        lines.append("")

        # Recommended fix scope
        lines.append("## Recommended Fix Scope")
        lines.append("")
        lines.append(result.recommended_fix_scope)
        lines.append("")

        # Secondary threshold crossings
        if result.secondary_threshold_crossings:
            lines.append("## Secondary Threshold Crossings")
            lines.append("")
            for crossing in result.secondary_threshold_crossings:
                lines.append(f"- **{crossing.failure_mode.value}** at step {crossing.step} (t={crossing.time_s:.3f}s): value={crossing.value:.4f}, threshold={crossing.threshold:.4f}")
            lines.append("")

        # Evidence fields
        if result.evidence_fields:
            lines.append("## Evidence Fields")
            lines.append("")
            for key, value in result.evidence_fields.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.4f}")
                else:
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)

    def _build_report_dict(self, result: ClassificationResult) -> Dict[str, Any]:
        """Build dictionary representation of classification result.

        Args:
            result: ClassificationResult to convert

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        report = {
            "primary_failure_mode": result.primary_failure_mode.value,
            "first_threshold_crossing_step": result.first_threshold_crossing_step,
            "first_threshold_crossing_time_s": result.first_threshold_crossing_time_s,
            "responsible_component": result.responsible_component,
            "fix_allowed_in_balance_core": result.fix_allowed_in_balance_core,
            "recommended_fix_scope": result.recommended_fix_scope,
            "evidence_fields": result.evidence_fields,
        }

        # Add secondary crossings if present
        if result.secondary_threshold_crossings:
            report["secondary_threshold_crossings"] = [
                {
                    "failure_mode": crossing.failure_mode.value,
                    "step": crossing.step,
                    "time_s": crossing.time_s,
                    "value": crossing.value,
                    "threshold": crossing.threshold,
                }
                for crossing in result.secondary_threshold_crossings
            ]

        return report
