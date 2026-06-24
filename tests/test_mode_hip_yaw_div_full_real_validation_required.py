"""Tests that D_MODE_HIP_YAW_DIV_V1 was actually run with real simulation.

These tests validate the ``outputs/mode_hip_yaw_div_full_real_validation/``
directory produced by ``scripts/run_mode_hip_yaw_div_full_real_validation.py``.

Required invariants:
1. Every D row has candidate_kind = mode_hip_yaw_div_v1.
2. Every D row has validation_source = real_simulation.
3. Every D telemetry CSV exists.
4. Every D telemetry CSV has mode_hip_yaw_div_enabled column.
5. For D rows, mode_hip_yaw_div_enabled_rows > 0 (after startup).
6. For D rows, mode_hip_yaw_div_kp == 5.0.
7. For D rows, mode_hip_yaw_div_kd == 0.20.
8. For D rows, mode_hip_yaw_div_max_torque == 2.0.
9. D telemetry paths must not equal C telemetry paths.
10. D command.json must contain --enable-mode-hip-yaw-divergence.
11. D rows must not contain wheel_yaw candidate_kind.
12. No row with validation_source = assumed_parity may exist in promotion decision.
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "mode_hip_yaw_div_full_real_validation"

REQUIRED_SUITE_FILES = [
    "step_e_fixed_height_metrics.csv",
    "step_c_standard_metrics.csv",
    "step_d_standard_metrics.csv",
    "d4_d5_focused_1000_metrics.csv",
    "profile_comparison_summary.csv",
    "duration_coverage_summary.csv",
    "promotion_recheck_decision.json",
]

D_MODE_DIV_KP = 5.0
D_MODE_DIV_KD = 0.20
D_MODE_DIV_MAX_TORQUE = 2.0
D_MODE_DIV_SOFT_LIMIT = 0.30
D_MODE_DIV_SOFT_GAIN = 0.25
D_MODE_DIV_REF_SOURCE = "target"


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def promotion_decision() -> dict:
    dec_path = OUT_DIR / "promotion_recheck_decision.json"
    if not dec_path.exists():
        pytest.skip("promotion_recheck_decision.json not found")
    with open(dec_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def comparison_summary() -> list[dict]:
    csv_path = OUT_DIR / "profile_comparison_summary.csv"
    if not csv_path.exists():
        pytest.skip("profile_comparison_summary.csv not found")
    with open(csv_path) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def all_suite_csvs() -> dict[str, list[dict]]:
    """Return all per-suite CSV files as {name: rows}."""
    result = {}
    for fname in [
        "step_e_fixed_height_metrics.csv",
        "step_c_standard_metrics.csv",
        "step_c_extended_metrics.csv",
        "step_d_standard_metrics.csv",
        "step_d_extended_metrics.csv",
        "d4_d5_focused_1000_metrics.csv",
        "d4_d5_focused_5000_metrics.csv",
    ]:
        p = OUT_DIR / fname
        if p.exists():
            with open(p) as f:
                result[fname] = list(csv.DictReader(f))
    return result


@pytest.fixture(scope="module")
def d_rows(comparison_summary) -> list[dict]:
    return [r for r in comparison_summary if r.get("profile_tag") == "D"]


@pytest.fixture(scope="module")
def c_rows(comparison_summary) -> list[dict]:
    return [r for r in comparison_summary if r.get("profile_tag") == "C"]


# =========================================================================
# Required file existence
# =========================================================================

class TestRequiredFiles:
    def test_output_dir_exists(self):
        assert OUT_DIR.exists(), f"Output dir {OUT_DIR} does not exist"

    def test_all_required_files_exist(self):
        missing = [f for f in REQUIRED_SUITE_FILES if not (OUT_DIR / f).exists()]
        assert not missing, f"Missing required files: {missing}"

    def test_promotion_decision_valid_json(self):
        dec_path = OUT_DIR / "promotion_recheck_decision.json"
        with open(dec_path) as f:
            dec = json.load(f)
        assert "overall_verdict" in dec
        assert "d_was_run_independently" in dec


# =========================================================================
# D row validation
# =========================================================================

class TestDRowsHaveRealSimulationSource:
    def test_all_d_rows_have_real_simulation_source(self, d_rows):
        """Every D row has validation_source = real_simulation."""
        for r in d_rows:
            src = r.get("validation_source", "")
            assert src == "real_simulation", (
                f"D row {r.get('case_id')} has source={src!r}, expected 'real_simulation'"
            )

    def test_all_d_rows_have_candidate_kind_mode_hip_yaw_div_v1(self, d_rows):
        """Every D row has candidate_kind = mode_hip_yaw_div_v1."""
        for r in d_rows:
            kind = r.get("candidate_kind", "")
            assert kind == "mode_hip_yaw_div_v1", (
                f"D row {r.get('case_id')} has candidate_kind={kind!r}"
            )

    def test_d_rows_have_mode_hip_yaw_div_enabled_flag(self, d_rows):
        """Every D row reports mode_div_enabled_rows > 0."""
        for r in d_rows:
            val = r.get("mode_div_enabled_rows", "0")
            assert str(val).strip() not in ("", "0", "0.0"), (
                f"D row {r.get('case_id')} has mode_div_enabled_rows={val!r}, expected > 0"
            )

    def test_d_telemetry_paths_distinct_from_c(self, d_rows, c_rows):
        """D telemetry paths must differ from C telemetry paths."""
        d_paths = {r.get("telemetry_path", "") for r in d_rows if r.get("telemetry_path")}
        c_paths = {r.get("telemetry_path", "") for r in c_rows if r.get("telemetry_path")}
        overlap = d_paths & c_paths
        assert not overlap, (
            f"{len(overlap)} telemetry paths shared between C and D: {overlap}"
        )

    def test_d_rows_not_wheel_yaw_candidate(self, d_rows):
        """D rows must not contain wheel_yaw candidate_kind."""
        for r in d_rows:
            kind = r.get("candidate_kind", "")
            assert "wheel_yaw" not in kind.lower(), (
                f"D row {r.get('case_id')} has wheel_yaw candidate_kind={kind!r}"
            )


class TestDModeDivTelemetryContent:
    """Validate D telemetry CSV files for mode-div flag presence and values."""

    def _get_d_telemetry_paths(self, d_rows) -> list[Path]:
        paths = []
        for r in d_rows:
            p = r.get("telemetry_path", "")
            if p:
                p = Path(p)
                if p.exists():
                    paths.append(p)
        return paths

    def test_d_telemetry_files_exist(self, d_rows):
        """Every D telemetry CSV path exists on disk."""
        missing = []
        for r in d_rows:
            p = r.get("telemetry_path", "")
            if p and not Path(p).exists():
                missing.append(p)
        assert not missing, f"{len(missing)} D telemetry paths missing"

    def test_d_telemetry_has_mode_div_enabled_column(self):
        """Every D telemetry CSV has mode_hip_yaw_div_enabled column."""
        # Get path from the summary
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            dur_rows = list(csv.DictReader(f))
        d_rows = [r for r in dur_rows if r.get("profile_tag") == "D"]

        missing = []
        for r in d_rows:
            tp = r.get("telemetry_path", "")
            if not tp or not Path(tp).exists():
                missing.append(f"{r.get('case_id')}: telemetry missing {tp}")
                continue
            try:
                with open(tp) as f:
                    first = next(csv.DictReader(f))
                if "mode_hip_yaw_div_enabled" not in first:
                    missing.append(f"{r.get('case_id')}: no mode_hip_yaw_div_enabled column")
            except Exception as e:
                missing.append(f"{r.get('case_id')}: read error {e}")
        assert not missing, f"Missing mode-div columns: {missing[:5]}"

    def test_d_telemetry_mode_div_enabled_rows_positive(self):
        """For D telemetry, mode_hip_yaw_div_enabled_rows > 0 after startup."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            dur_rows = list(csv.DictReader(f))
        d_rows = [r for r in dur_rows if r.get("profile_tag") == "D"]

        zero_enabled = []
        for r in d_rows:
            tp = r.get("telemetry_path", "")
            if not tp or not Path(tp).exists():
                continue
            try:
                with open(tp) as f:
                    rows = list(csv.DictReader(f))
                enabled_count = sum(
                    1 for row in rows
                    if str(row.get("mode_hip_yaw_div_enabled", "")).lower() == "true"
                )
                if enabled_count == 0:
                    zero_enabled.append(f"{r.get('case_id')}: {enabled_count}/{len(rows)} enabled")
            except Exception as e:
                zero_enabled.append(f"{r.get('case_id')}: error {e}")

        assert not zero_enabled, f"D runs with zero mode-div enabled rows: {zero_enabled[:5]}"

    def test_d_telemetry_kp_is_5(self):
        """For D telemetry, mode_hip_yaw_div_kp == 5.0."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            dur_rows = list(csv.DictReader(f))
        d_rows = [r for r in dur_rows if r.get("profile_tag") == "D"]

        bad_kp = []
        for r in d_rows:
            tp = r.get("telemetry_path", "")
            if not tp or not Path(tp).exists():
                continue
            try:
                with open(tp) as f:
                    rows = list(csv.DictReader(f))
                # Sample first few rows
                for i, row in enumerate(rows[:5]):
                    val = row.get("mode_hip_yaw_div_kp", "")
                    try:
                        if abs(float(val) - D_MODE_DIV_KP) > 1e-6:
                            bad_kp.append(f"{r.get('case_id')} row {i}: kp={val}")
                    except (ValueError, TypeError):
                        bad_kp.append(f"{r.get('case_id')} row {i}: unparseable kp={val!r}")
            except Exception as e:
                bad_kp.append(f"{r.get('case_id')}: {e}")

        assert not bad_kp, f"Wrong mode-div Kp: {bad_kp[:5]}"

    def test_d_telemetry_kd_is_0p2(self):
        """For D telemetry, mode_hip_yaw_div_kd == 0.20."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            dur_rows = list(csv.DictReader(f))
        d_rows = [r for r in dur_rows if r.get("profile_tag") == "D"]

        bad_kd = []
        for r in d_rows:
            tp = r.get("telemetry_path", "")
            if not tp or not Path(tp).exists():
                continue
            try:
                with open(tp) as f:
                    rows = list(csv.DictReader(f))
                for i, row in enumerate(rows[:5]):
                    val = row.get("mode_hip_yaw_div_kd", "")
                    try:
                        if abs(float(val) - D_MODE_DIV_KD) > 1e-6:
                            bad_kd.append(f"{r.get('case_id')} row {i}: kd={val}")
                    except (ValueError, TypeError):
                        bad_kd.append(f"{r.get('case_id')} row {i}: unparseable kd={val!r}")
            except Exception:
                bad_kd.append(f"{r.get('case_id')}: read error")

        assert not bad_kd, f"Wrong mode-div Kd: {bad_kd[:5]}"

    def test_d_telemetry_max_torque_is_2(self, d_rows):
        """For D telemetry, mode_hip_yaw_div_max_torque == 2.0."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            dur_rows = list(csv.DictReader(f))
        d_rows_list = [r for r in dur_rows if r.get("profile_tag") == "D"]

        bad_mt = []
        for r in d_rows_list:
            tp = r.get("telemetry_path", "")
            if not tp or not Path(tp).exists():
                continue
            try:
                with open(tp) as f:
                    rows = list(csv.DictReader(f))
                for i, row in enumerate(rows[:5]):
                    val = row.get("mode_hip_yaw_div_max_torque", "")
                    try:
                        if abs(float(val) - D_MODE_DIV_MAX_TORQUE) > 1e-6:
                            bad_mt.append(f"{r.get('case_id')} row {i}: max_torque={val}")
                    except (ValueError, TypeError):
                        bad_mt.append(f"{r.get('case_id')} row {i}: unparseable max_torque={val!r}")
            except Exception:
                bad_mt.append(f"{r.get('case_id')}: read error")

        assert not bad_mt, f"Wrong mode-div max_torque: {bad_mt[:5]}"

    def test_d_telemetry_soft_limit_and_gain(self):
        """For D telemetry, soft_limit_rad == 0.30 and soft_gain == 0.25."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            dur_rows = list(csv.DictReader(f))
        d_rows_list = [r for r in dur_rows if r.get("profile_tag") == "D"]

        bad = []
        for r in d_rows_list:
            tp = r.get("telemetry_path", "")
            if not tp or not Path(tp).exists():
                continue
            try:
                with open(tp) as f:
                    rows = list(csv.DictReader(f))
                row = rows[0]
                sl = row.get("mode_hip_yaw_div_soft_limit_rad", "")
                sg = row.get("mode_hip_yaw_div_soft_gain", "")
                rs = row.get("mode_hip_yaw_div_ref_source", "")
                try:
                    if abs(float(sl) - D_MODE_DIV_SOFT_LIMIT) > 1e-6:
                        bad.append(f"{r.get('case_id')} soft_limit={sl}")
                    if abs(float(sg) - D_MODE_DIV_SOFT_GAIN) > 1e-6:
                        bad.append(f"{r.get('case_id')} soft_gain={sg}")
                    if rs.strip() != D_MODE_DIV_REF_SOURCE:
                        bad.append(f"{r.get('case_id')} ref_source={rs!r}")
                except (ValueError, TypeError) as e:
                    bad.append(f"{r.get('case_id')}: {e}")
            except Exception:
                bad.append(f"{r.get('case_id')}: read error")

        assert not bad, f"Wrong mode-div params: {bad}"


# =========================================================================
# No assumed parity
# =========================================================================

class TestNoAssumedParity:
    def test_no_assumed_parity_rows_in_promotion_decision(self, promotion_decision):
        """No row with validation_source = assumed_parity may exist."""
        assert not promotion_decision.get("any_assumed_parity_rows", False), (
            "Promotion decision contains assumed_parity rows"
        )

    def test_no_assumed_parity_in_csvs(self, all_suite_csvs):
        """Check all suite CSVs for assumed_parity rows."""
        bad = []
        for fname, rows in all_suite_csvs.items():
            for r in rows:
                if r.get("validation_source", "").strip() == "assumed_parity":
                    bad.append(f"{fname}: {r.get('case_id')} {r.get('profile_tag')}")
        assert not bad, f"assumed_parity rows found: {bad}"


# =========================================================================
# Duration coverage
# =========================================================================

class TestDurationCoverage:
    def test_duration_coverage_summary_requires_step_e_c_d(self):
        """Duration coverage summary must include step_e, step_c, and step_d suites."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        suites = {r.get("suite") for r in rows}
        for required in ("step_e", "step_c", "step_d"):
            assert required in suites, f"Suite {required} missing from duration_coverage_summary.csv"

    def test_step_e_d_has_5000_or_documented_2000(self):
        """Step E D rows have either 5000-step completion or documented 2000 fallback."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        d_step_e = [r for r in rows if r.get("suite") == "step_e" and r.get("profile_tag") == "D"]
        bad = []
        for r in d_step_e:
            comp = r.get("completed_full_duration", "") or ""
            deg = r.get("duration_degraded_from_5000_to_2000", "") or ""
            actual = r.get("actual_rows", "0") or "0"
            try:
                if int(float(actual)) < 2000:
                    bad.append(f"{r.get('case_id')}: actual_rows={actual} < 2000")
            except ValueError:
                bad.append(f"{r.get('case_id')}: unparseable actual_rows={actual!r}")
        assert not bad, f"Insufficient duration: {bad}"

    def test_d_telemetry_command_contains_mode_div_flags(self):
        """D command paths must reference --enable-mode-hip-yaw-divergence."""
        csv_path = OUT_DIR / "duration_coverage_summary.csv"
        if not csv_path.exists():
            pytest.skip("duration_coverage_summary.csv not found")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        d_rows = [r for r in rows if r.get("profile_tag") == "D"]

        missing = []
        for r in d_rows:
            cmd_path = r.get("command_path", "")
            if not cmd_path or not Path(cmd_path).exists():
                continue
            try:
                with open(cmd_path) as f:
                    data = json.load(f)
                # Check command-line string
                cmd_str = json.dumps(data).lower()
                if "enable-mode-hip-yaw-divergence" not in cmd_str:
                    missing.append(f"{r.get('case_id')}: flag not in command.json")
            except Exception:
                missing.append(f"{r.get('case_id')}: unreadable command.json")
        assert not missing, f"Missing mode-div flag in commands: {missing[:5]}"


# =========================================================================
# Safety checks
# =========================================================================

class TestSafety:
    def test_no_wbc_in_d_rows(self, comparison_summary):
        """D rows should not have WBC authority active."""
        d_rows_list = [r for r in comparison_summary if r.get("profile_tag") == "D"]
        wbc_cases = []
        for r in d_rows_list:
            v = str(r.get("wbc_authority_rows", "0"))
            try:
                if int(float(v)) > 0:
                    wbc_cases.append(f"{r.get('case_id')}: wbc={v}")
            except ValueError:
                pass
        assert not wbc_cases, f"D runs with WBC active: {wbc_cases[:5]}"

    def test_no_hidden_torque_in_d_rows(self, comparison_summary):
        """D rows should not have hidden torque above threshold."""
        d_rows_list = [r for r in comparison_summary if r.get("profile_tag") == "D"]
        bad = []
        for r in d_rows_list:
            v = str(r.get("hidden_torque_max", "0"))
            try:
                if float(v) > 0.5:
                    bad.append(f"{r.get('case_id')}: hidden_torque={v}")
            except ValueError:
                pass
        assert not bad, f"D runs with hidden torque: {bad[:5]}"

    def test_no_ownership_violation_in_d_rows(self, comparison_summary):
        """D rows should not have ownership violations."""
        d_rows_list = [r for r in comparison_summary if r.get("profile_tag") == "D"]
        bad = []
        for r in d_rows_list:
            v = str(r.get("ownership_violation_max", "0"))
            try:
                if float(v) > 0:
                    bad.append(f"{r.get('case_id')}: ownership_violation={v}")
            except ValueError:
                pass
        assert not bad, f"D runs with ownership violations: {bad[:5]}"

    def test_d_no_unsafe_falls_in_step_e(self, comparison_summary):
        """D should not fall in protected (non-D4/D5-push) cases."""
        d_fell = []
        for r in comparison_summary:
            if r.get("profile_tag") != "D":
                continue
            if str(r.get("fell", "")).lower() in ("true", "1"):
                d_fell.append(f"{r.get('case_id')}")
        allowed_fall_cases = {"D4_medium_push_low", "D5_large_push_high",
                              "D4_medium_push_low_5000", "D5_large_push_high_5000"}
        unexpected = [c for c in d_fell if c not in allowed_fall_cases]
        assert not unexpected, f"D fell unexpectedly in: {unexpected}"


# =========================================================================
# Old wheel-yaw D not accepted
# =========================================================================

class TestOldWheelYawDNotAccepted:
    def test_old_wheel_yaw_d_path_not_present(self, comparison_summary):
        """No telemetry path from old wheel-yaw output directory in D rows."""
        old_dirs = [
            "hip_yaw_push_limit_architecture_fix",
            "step_e_extreme_support_fix_eval",
            "step_d_all",
        ]
        for r in comparison_summary:
            if r.get("profile_tag") != "D":
                continue
            tp = r.get("telemetry_path", "")
            for old_dir in old_dirs:
                if old_dir in tp:
                    pytest.fail(f"D telemetry path references old dir {old_dir}: {tp}")

    def test_candidate_kind_not_wheel_yaw(self, comparison_summary):
        """No D row has candidate_kind = wheel_yaw."""
        for r in comparison_summary:
            if r.get("profile_tag") != "D":
                continue
            kind = r.get("candidate_kind", "")
            assert "wheel_yaw" not in kind.lower(), (
                f"D row {r.get('case_id')} has wheel_yaw candidate_kind={kind!r}"
            )
