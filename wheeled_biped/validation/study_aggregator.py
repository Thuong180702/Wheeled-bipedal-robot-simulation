# wheeled_biped/validation/study_aggregator.py
"""Study orchestration and setup-validity-aware aggregation for balance-core validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
import json

import pandas as pd

from wheeled_biped.validation.balance_core_validator import (
    BalanceCoreValidator,
    ValidationResult,
)


VALID_START_CONTACT_STATES = {
    "double_contact",
    "left_only",
    "right_only",
    "DOUBLE_CONTACT",
    "SINGLE_LEFT",
    "SINGLE_RIGHT",
}
INVALID_NO_CONTACT_STATES = {
    "flight_or_no_contact",
    "no_contact",
    "FLIGHT_OR_NO_CONTACT",
    "NO_CONTACT",
}
INVALID_UNKNOWN_CONTACT_STATES = {
    "unknown",
    "init",
    "UNKNOWN",
    "INIT",
}
FLOATING_START_DIST_M = 2e-3
EXCESSIVE_PENETRATION_DIST_M = -5e-3


@dataclass
class StudyCaseResult:
    case_id: str
    height_test_type: str
    duration_steps: int
    passed: bool
    actual_steps: int
    setup_valid: bool
    setup_failure_reason: Optional[str]
    initial_contact_state: Optional[str]
    min_wheel_contact_dist_m: Optional[float]
    equilibrium_com_z_m: Optional[float]
    initial_com_z_m: Optional[float]
    failure_mode: Optional[str]
    responsible_component: Optional[str]
    telemetry_path: Optional[str]
    report_path: Optional[str]
    sim_args: list[str]
    summary_metrics: dict[str, Any] = field(default_factory=dict)


class StudyAggregator:
    """Runs setup-aware balance-core study cases and writes summaries."""

    def __init__(self, validator: Optional[BalanceCoreValidator] = None):
        self.validator = validator or BalanceCoreValidator()

    def evaluate_case_from_telemetry(
        self,
        case_id: str,
        height_test_type: str,
        duration_steps: int,
        telemetry_path: str | Path,
        sim_args: Optional[Sequence[str]] = None,
    ) -> StudyCaseResult:
        telemetry_path = Path(telemetry_path)
        df = pd.read_csv(telemetry_path)
        setup_verdict = self._evaluate_setup_validity(df)
        summary_metrics = self._build_summary_metrics(df, duration_steps)

        if not setup_verdict["setup_valid"]:
            return StudyCaseResult(
                case_id=case_id,
                height_test_type=height_test_type,
                duration_steps=duration_steps,
                passed=False,
                actual_steps=len(df),
                setup_valid=False,
                setup_failure_reason=setup_verdict["setup_failure_reason"],
                initial_contact_state=setup_verdict["initial_contact_state"],
                min_wheel_contact_dist_m=setup_verdict["min_wheel_contact_dist_m"],
                equilibrium_com_z_m=self._optional_float(df, "nominal_equilibrium_com_z_m"),
                initial_com_z_m=self._resolve_initial_com_z(df),
                failure_mode="invalid_initial_setup",
                responsible_component=None,
                telemetry_path=str(telemetry_path),
                report_path=None,
                sim_args=list(sim_args or []),
                summary_metrics=summary_metrics,
            )

        validation_result = self.validator.validate_duration(str(telemetry_path), duration_steps)
        return self._to_study_case_result(
            case_id=case_id,
            height_test_type=height_test_type,
            validation_result=validation_result,
            sim_args=sim_args,
            summary_metrics=summary_metrics,
            setup_verdict=setup_verdict,
        )

    def run_case(
        self,
        case_id: str,
        height_test_type: str,
        duration_steps: int,
        output_dir: str | Path,
        sim_args: Optional[Sequence[str]] = None,
    ) -> StudyCaseResult:
        case_output_dir = Path(output_dir) / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = self.validator.run_simulation(
            steps=duration_steps,
            output_dir=str(case_output_dir),
            sim_args=sim_args,
        )
        return self.evaluate_case_from_telemetry(
            case_id=case_id,
            height_test_type=height_test_type,
            duration_steps=duration_steps,
            telemetry_path=telemetry_path,
            sim_args=sim_args,
        )

    def run_longevity_cases(
        self,
        output_dir: str | Path,
        durations: Sequence[int],
        continue_all: bool = False,
        sim_args: Optional[Sequence[str]] = None,
    ) -> list[StudyCaseResult]:
        results: list[StudyCaseResult] = []
        for duration in durations:
            case_id = f"longevity_{duration}"
            result = self.run_case(
                case_id=case_id,
                height_test_type="longevity",
                duration_steps=duration,
                output_dir=Path(output_dir) / "longevity",
                sim_args=sim_args,
            )
            results.append(result)
            if not continue_all and not result.passed:
                break
        return results

    def run_root_z_perturbation_cases(
        self,
        output_dir: str | Path,
        offsets_m: Sequence[float],
        first_duration_steps: int = 1000,
        promoted_duration_steps: int = 5000,
    ) -> list[StudyCaseResult]:
        results: list[StudyCaseResult] = []
        case_root = Path(output_dir) / "root_z_perturbation"

        first_pass_results: list[StudyCaseResult] = []
        for offset_m in offsets_m:
            offset_tag = self._format_offset_tag(offset_m)
            case_id = f"root_z_{offset_tag}_{first_duration_steps}"
            sim_args = ["--initial-root-z-perturbation", str(offset_m)]
            result = self.run_case(
                case_id=case_id,
                height_test_type="root_z_perturbation",
                duration_steps=first_duration_steps,
                output_dir=case_root,
                sim_args=sim_args,
            )
            result.summary_metrics["height_offset_m"] = float(offset_m)
            results.append(result)
            first_pass_results.append(result)

        for result in first_pass_results:
            if not result.passed:
                continue
            offset_m = float(result.summary_metrics["height_offset_m"])
            offset_tag = self._format_offset_tag(offset_m)
            promoted_case_id = f"root_z_{offset_tag}_{promoted_duration_steps}"
            promoted_result = self.run_case(
                case_id=promoted_case_id,
                height_test_type="root_z_perturbation",
                duration_steps=promoted_duration_steps,
                output_dir=case_root,
                sim_args=["--initial-root-z-perturbation", str(offset_m)],
            )
            promoted_result.summary_metrics["height_offset_m"] = offset_m
            results.append(promoted_result)

        return results

    def write_summary_files(
        self,
        results: Sequence[StudyCaseResult],
        json_path: str | Path,
        markdown_path: str | Path,
        conclusion: Optional[str] = None,
    ) -> None:
        json_path = Path(json_path)
        markdown_path = Path(markdown_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.build_summary_payload(results, conclusion=conclusion)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        markdown_path.write_text(self._build_summary_markdown(payload), encoding="utf-8")

    def build_summary_payload(
        self,
        results: Sequence[StudyCaseResult],
        conclusion: Optional[str] = None,
    ) -> dict[str, Any]:
        results_list = [asdict(result) for result in results]
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        invalid = [r for r in results if r.failure_mode == "invalid_initial_setup"]

        # Compute extended longevity fields
        longevity_results = [r for r in passed if r.height_test_type == "longevity"]
        max_confirmed_steps = (
            max((r.duration_steps for r in longevity_results), default=0)
        )
        passed_100k = any(r.passed and r.duration_steps == 100000 for r in longevity_results)
        first_failure = (
            failed[0] if failed else None
        )

        payload = {
            "case_count": len(results),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "invalid_initial_setup_count": len(invalid),
            "max_confirmed_passing_duration_steps": max_confirmed_steps,
            "long_duration_survival_passed_up_to_100000_steps": passed_100k,
            "first_failing_duration_steps": (
                first_failure.duration_steps if first_failure else None
            ),
            "first_failing_primary_failure_mode": (
                first_failure.failure_mode if first_failure else None
            ),
            "first_failing_responsible_component": (
                first_failure.responsible_component if first_failure else None
            ),
            "results": results_list,
        }
        if conclusion is not None:
            payload["conclusion"] = conclusion
        return payload

    def _to_study_case_result(
        self,
        case_id: str,
        height_test_type: str,
        validation_result: ValidationResult,
        sim_args: Optional[Sequence[str]],
        summary_metrics: dict[str, Any],
        setup_verdict: dict[str, Any],
    ) -> StudyCaseResult:
        classification = validation_result.classification_result
        return StudyCaseResult(
            case_id=case_id,
            height_test_type=height_test_type,
            duration_steps=validation_result.duration_steps,
            passed=validation_result.passed,
            actual_steps=validation_result.actual_steps,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=setup_verdict["initial_contact_state"],
            min_wheel_contact_dist_m=setup_verdict["min_wheel_contact_dist_m"],
            equilibrium_com_z_m=setup_verdict["equilibrium_com_z_m"],
            initial_com_z_m=setup_verdict["initial_com_z_m"],
            failure_mode=(classification.primary_failure_mode.value if classification is not None else None),
            responsible_component=(classification.responsible_component if classification is not None else None),
            telemetry_path=str(validation_result.telemetry_path),
            report_path=(str(validation_result.report_path) if validation_result.report_path is not None else None),
            sim_args=list(sim_args or []),
            summary_metrics=summary_metrics,
        )

    def _evaluate_setup_validity(self, df: pd.DataFrame) -> dict[str, Any]:
        initial_contact_state = self._optional_str(df, "contact_supervisor_state")
        min_wheel_contact_dist_m = self._optional_float(df, "min_wheel_contact_dist_m")
        equilibrium_com_z_m = self._optional_float(df, "nominal_equilibrium_com_z_m")
        initial_com_z_m = self._resolve_initial_com_z(df)

        if initial_contact_state in INVALID_NO_CONTACT_STATES:
            return self._setup_verdict(
                False,
                "floating_start",
                initial_contact_state,
                min_wheel_contact_dist_m,
                equilibrium_com_z_m,
                initial_com_z_m,
            )

        if initial_contact_state in INVALID_UNKNOWN_CONTACT_STATES or (
            initial_contact_state is not None and initial_contact_state not in VALID_START_CONTACT_STATES
            and initial_contact_state not in INVALID_NO_CONTACT_STATES
        ):
            return self._setup_verdict(
                False,
                "invalid_contact_state",
                initial_contact_state,
                min_wheel_contact_dist_m,
                equilibrium_com_z_m,
                initial_com_z_m,
            )

        non_wheel_floor_contact_count = self._optional_int(df, "non_wheel_floor_contact_count")
        if non_wheel_floor_contact_count is not None and non_wheel_floor_contact_count > 0:
            return self._setup_verdict(
                False,
                "non_wheel_floor_contact",
                initial_contact_state,
                min_wheel_contact_dist_m,
                equilibrium_com_z_m,
                initial_com_z_m,
            )

        left_wheel_contact = self._optional_bool(df, "left_wheel_contact")
        right_wheel_contact = self._optional_bool(df, "right_wheel_contact")
        if left_wheel_contact is not None and right_wheel_contact is not None:
            if not left_wheel_contact and not right_wheel_contact:
                return self._setup_verdict(
                    False,
                    "floating_start",
                    initial_contact_state,
                    min_wheel_contact_dist_m,
                    equilibrium_com_z_m,
                    initial_com_z_m,
                )

        if min_wheel_contact_dist_m is not None:
            if min_wheel_contact_dist_m < EXCESSIVE_PENETRATION_DIST_M:
                return self._setup_verdict(
                    False,
                    "excessive_penetration",
                    initial_contact_state,
                    min_wheel_contact_dist_m,
                    equilibrium_com_z_m,
                    initial_com_z_m,
                )
            if min_wheel_contact_dist_m > FLOATING_START_DIST_M:
                return self._setup_verdict(
                    False,
                    "floating_start",
                    initial_contact_state,
                    min_wheel_contact_dist_m,
                    equilibrium_com_z_m,
                    initial_com_z_m,
                )

        return self._setup_verdict(
            True,
            None,
            initial_contact_state,
            min_wheel_contact_dist_m,
            equilibrium_com_z_m,
            initial_com_z_m,
        )

    def _setup_verdict(
        self,
        setup_valid: bool,
        setup_failure_reason: Optional[str],
        initial_contact_state: Optional[str],
        min_wheel_contact_dist_m: Optional[float],
        equilibrium_com_z_m: Optional[float],
        initial_com_z_m: Optional[float],
    ) -> dict[str, Any]:
        return {
            "setup_valid": setup_valid,
            "setup_failure_reason": setup_failure_reason,
            "initial_contact_state": initial_contact_state,
            "min_wheel_contact_dist_m": min_wheel_contact_dist_m,
            "equilibrium_com_z_m": equilibrium_com_z_m,
            "initial_com_z_m": initial_com_z_m,
        }

    def _build_summary_metrics(self, df: pd.DataFrame, duration_steps: int) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "requested_steps": int(duration_steps),
            "survival_steps": int(len(df)),
            "terminated": bool(len(df) < duration_steps),
            "final_sim_time_s": self._last_float(df, "sim_time_s") or self._last_float(df, "time") or 0.0,
            "pitch_range_rad": self._range(df, "pitch_x_rad"),
            "roll_range_rad": self._range(df, "roll_y_rad"),
            "com_z_range_m": self._range(df, "com_z_m"),
            "com_z_drift_m": self._drift(df, "com_z_m"),
            "wheel_velocity_left_range_rad_s": self._range(df, "wheel_vel_left_rad_s"),
            "wheel_velocity_right_range_rad_s": self._range(df, "wheel_vel_right_rad_s"),
            "wheel_velocity_mean_range_rad_s": self._range(df, "wheel_vel_mean_rad_s"),
            "contact_state_summary": self._value_counts(df, "contact_supervisor_state"),
            "left_wheel_contact_all": self._all_bool(df, "left_wheel_contact"),
            "right_wheel_contact_all": self._all_bool(df, "right_wheel_contact"),
            "ownership_violation_count_max": self._max_numeric(df, "ownership_violation_count"),
            "hidden_torque_norm_max": self._max_numeric(df, "hidden_torque_norm"),
            "torque_saturation_percent_per_joint": self._mask_percentages(df, "torque_saturation_mask_per_joint"),
            "torque_rate_saturation_percent_per_joint": self._mask_percentages(df, "torque_rate_saturation_mask_per_joint"),
        }
        return metrics

    def _build_summary_markdown(self, payload: dict[str, Any]) -> str:
        lines = [
            "# Balance-Core Study Summary",
            "",
            f"- Cases: {payload['case_count']}",
            f"- Passed: {payload['passed_count']}",
            f"- Failed: {payload['failed_count']}",
            f"- Invalid initial setup: {payload['invalid_initial_setup_count']}",
            f"- Max confirmed passing duration: {payload['max_confirmed_passing_duration_steps']} steps",
            (
                "- Passed 100000 steps: yes"
                if payload["long_duration_survival_passed_up_to_100000_steps"]
                else "- Passed 100000 steps: no"
            ),
        ]
        if payload.get("first_failing_duration_steps") is not None:
            lines.append(
                f"- First failing duration: {payload['first_failing_duration_steps']} steps"
            )
        if payload.get("first_failing_primary_failure_mode") is not None:
            lines.append(
                f"- First failing primary failure mode: {payload['first_failing_primary_failure_mode']}"
            )
        if payload.get("first_failing_responsible_component") is not None:
            lines.append(
                f"- First failing responsible component: {payload['first_failing_responsible_component']}"
            )
        if "conclusion" in payload:
            lines.extend(["", f"**Conclusion:** {payload['conclusion']}"])

        lines.extend(["", "## Cases", ""])
        for result in payload["results"]:
            status = "PASS" if result["passed"] else "FAIL"
            lines.append(
                f"- **{result['case_id']}** [{status}] type={result['height_test_type']} "
                f"duration={result['duration_steps']} actual={result['actual_steps']} "
                f"setup_valid={result['setup_valid']} failure_mode={result['failure_mode']}"
            )
        lines.append("")
        return "\n".join(lines)

    def _format_offset_tag(self, offset_m: float) -> str:
        sign = "plus" if offset_m >= 0 else "minus"
        millimeters = int(round(abs(offset_m) * 1000.0))
        return f"{sign}_{millimeters:03d}mm"

    def _optional_str(self, df: pd.DataFrame, column: str) -> Optional[str]:
        if column not in df.columns or df.empty:
            return None
        value = df.iloc[0][column]
        if pd.isna(value):
            return None
        return str(value)

    def _optional_float(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns or df.empty:
            return None
        value = df.iloc[0][column]
        if pd.isna(value):
            return None
        return float(value)

    def _optional_int(self, df: pd.DataFrame, column: str) -> Optional[int]:
        if column not in df.columns or df.empty:
            return None
        value = df.iloc[0][column]
        if pd.isna(value):
            return None
        return int(value)

    def _optional_bool(self, df: pd.DataFrame, column: str) -> Optional[bool]:
        if column not in df.columns or df.empty:
            return None
        value = df.iloc[0][column]
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False
        return bool(value)

    def _resolve_initial_com_z(self, df: pd.DataFrame) -> Optional[float]:
        if "initial_com_z_m_after_perturbation" in df.columns:
            return self._optional_float(df, "initial_com_z_m_after_perturbation")
        return self._optional_float(df, "com_z_m")

    def _range(self, df: pd.DataFrame, column: str) -> Optional[list[float]]:
        if column not in df.columns or df.empty:
            return None
        series = df[column].astype(float)
        return [float(series.min()), float(series.max())]

    def _drift(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns or df.empty:
            return None
        series = df[column].astype(float)
        return float(series.iloc[-1] - series.iloc[0])

    def _value_counts(self, df: pd.DataFrame, column: str) -> dict[str, int]:
        if column not in df.columns or df.empty:
            return {}
        counts = df[column].astype(str).value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def _all_bool(self, df: pd.DataFrame, column: str) -> Optional[bool]:
        if column not in df.columns or df.empty:
            return None
        return bool(df[column].astype(bool).all())

    def _max_numeric(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns or df.empty:
            return None
        return float(df[column].astype(float).max())

    def _last_float(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns or df.empty:
            return None
        return float(df[column].astype(float).iloc[-1])

    def _mask_percentages(self, df: pd.DataFrame, column: str) -> Optional[list[float]]:
        if column not in df.columns or df.empty:
            return None
        parsed_masks = [self._parse_bool_csv(value) for value in df[column]]
        if not parsed_masks:
            return None
        joint_count = len(parsed_masks[0])
        percentages = []
        for joint_idx in range(joint_count):
            true_count = sum(1 for mask in parsed_masks if mask[joint_idx])
            percentages.append(100.0 * true_count / len(parsed_masks))
        return percentages

    def _parse_bool_csv(self, value: Any) -> list[bool]:
        if isinstance(value, list):
            return [bool(v) for v in value]
        parts = [part.strip() for part in str(value).split(",")]
        parsed = []
        for part in parts:
            lowered = part.lower()
            if lowered in {"true", "1"}:
                parsed.append(True)
            elif lowered in {"false", "0"}:
                parsed.append(False)
            else:
                raise ValueError(f"Invalid boolean CSV value: {part}")
        return parsed
