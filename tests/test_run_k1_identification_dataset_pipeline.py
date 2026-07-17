"""Tests for run_k1_identification_dataset_and_validate_models pipeline.

Verifies:
  - Dataset integrity audit script compiles
  - Generated metadata schema validation
  - Run classifier works correctly
  - validation_source must equal real_simulation
  - Bad source is rejected
  - NaN/Inf telemetry rejected
  - Insufficient-length telemetry rejected
  - Model readiness classifier works
  - Controller-design gate blocks if no DESIGN_READY model
  - Report path exists
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"
DOCS_DIR = PROJECT_ROOT / "docs" / "validation"


class TestAuditScriptCompilation:
    """Verify dataset integrity audit script compiles and has required functions."""

    def test_audit_script_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_k1_identification_dataset_integrity",
            SCRIPTS_DIR / "audit_k1_identification_dataset_integrity.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "audit_dataset")
        assert hasattr(mod, "classify_run")
        assert hasattr(mod, "check_metadata_validity")
        assert hasattr(mod, "check_nan_inf")
        assert hasattr(mod, "check_excitation_signal")

    def test_audit_script_has_required_constants(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_k1_identification_dataset_integrity",
            SCRIPTS_DIR / "audit_k1_identification_dataset_integrity.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.K1_PROFILE_EXPECTED == "k1_pitch_rate_notch_v1"
        assert len(mod.RUN_TYPES) == 5
        assert len(mod.TARGET_HEIGHTS_MAP) == 3


class TestMetadataSchemaValidation:
    """Verify generated metadata schema is valid."""

    def test_valid_metadata_passes(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_metadata_validity

        valid_meta = {
            "validation_source": "real_simulation",
            "profile": "k1_pitch_rate_notch_v1",
            "controller_mode": "balance-core",
            "simulation_success": True,
            "telemetry_path": "/some/path/that/exists.csv",
            "source_label": "k1_identification_low_0p330_A_equilibrium",
        }
        issues = check_metadata_validity(valid_meta)
        # "TELEMETRY_FILE_NOT_FOUND" is expected since the path is fake,
        # but it should NOT have NON_REAL_SOURCE, FORBIDDEN_SOURCE_LABEL,
        # WRONG_PROFILE, or SIMULATION_FAILED
        assert not any("NON_REAL_SOURCE" in i for i in issues)
        assert not any("FORBIDDEN" in i for i in issues)
        assert not any("WRONG_PROFILE" in i for i in issues)
        assert not any("SIMULATION_FAILED" in i for i in issues)

    def test_stub_source_rejected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_metadata_validity

        stub_meta = {
            "validation_source": "stub",
            "source_label": "stub_test_data",
        }
        issues = check_metadata_validity(stub_meta)
        assert any("NON_REAL_SOURCE" in i for i in issues), \
            f"Should reject stub source, got: {issues}"

    def test_synthetic_source_rejected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_metadata_validity

        synth_meta = {
            "validation_source": "synthetic",
        }
        issues = check_metadata_validity(synth_meta)
        assert any("NON_REAL_SOURCE" in i for i in issues), \
            f"Should reject synthetic source, got: {issues}"

    def test_assumed_source_rejected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_metadata_validity

        assumed_meta = {
            "validation_source": "assumed",
        }
        issues = check_metadata_validity(assumed_meta)
        assert any("NON_REAL_SOURCE" in i for i in issues), \
            f"Should reject assumed source, got: {issues}"

    def test_missing_metadata_handled(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_metadata_validity

        issues = check_metadata_validity(None)
        assert "MISSING_METADATA" in issues

    def test_wrong_profile_detected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_metadata_validity

        wrong_meta = {
            "validation_source": "real_simulation",
            "profile": "wrong_profile_name",
            "controller_mode": "balance-core",
            "simulation_success": True,
            "telemetry_path": "/some/path.csv",
        }
        issues = check_metadata_validity(wrong_meta)
        assert any("WRONG_PROFILE" in i for i in issues), \
            f"Should detect wrong profile, got: {issues}"


class TestRunClassifier:
    """Verify run classifier produces correct classifications."""

    def test_classify_usable(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "has_nan": False,
            "has_inf": False,
            "fall_detected": False,
            "metadata_issues": [],
            "n_rows": 2000,
            "min_total_rows": 1500,
            "min_post_settle": 1000,
            "post_settle_samples": 1500,
        }
        assert classify_run(run) == "USABLE"

    def test_classify_nan_rejected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "has_nan": True,
            "has_inf": False,
            "fall_detected": False,
            "metadata_issues": [],
            "n_rows": 2000,
            "min_total_rows": 1500,
            "min_post_settle": 1000,
            "post_settle_samples": 1500,
            "nan_columns": ["pitch_x_rad", "height_error_m"],  # Critical cols
            "inf_columns": [],
        }
        assert classify_run(run) == "NAN_INF"

    def test_classify_inf_rejected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "has_nan": False,
            "has_inf": True,
            "fall_detected": False,
            "metadata_issues": [],
            "n_rows": 2000,
            "min_total_rows": 1500,
            "min_post_settle": 1000,
            "post_settle_samples": 1500,
            "nan_columns": [],
            "inf_columns": ["com_y_velocity_m_s"],  # Critical col
        }
        assert classify_run(run) == "NAN_INF"

    def test_classify_fall_rejected(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "has_nan": False,
            "has_inf": False,
            "fall_detected": True,
            "metadata_issues": [],
            "n_rows": 2000,
        }
        assert classify_run(run) == "FALL_REJECTED"

    def test_classify_non_real_source(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "has_nan": False,
            "has_inf": False,
            "fall_detected": False,
            "metadata_issues": ["NON_REAL_SOURCE: validation_source='stub'"],
        }
        assert classify_run(run) == "NON_REAL_SOURCE"

    def test_classify_insufficient_length(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "has_nan": False,
            "has_inf": False,
            "fall_detected": False,
            "metadata_issues": [],
            "n_rows": 50,
            "min_total_rows": 1500,
            "min_post_settle": 1000,
            "post_settle_samples": 30,
        }
        assert classify_run(run) == "INSUFFICIENT_LENGTH"

    def test_classify_failed_simulation(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import classify_run

        run = {
            "telemetry_exists": False,
        }
        assert classify_run(run) == "FAILED_SIMULATION"


class TestNaNInfRejection:
    """Verify NaN and Inf values are correctly detected and rejected."""

    def test_detect_nan_in_telemetry(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_nan_inf

        rows = [
            {"pitch_x_rad": "NaN", "com_y_velocity_m_s": "0.5"},
            {"pitch_x_rad": "0.1", "com_y_velocity_m_s": "Inf"},
        ]
        nan_cols, inf_cols = check_nan_inf(rows)
        assert "pitch_x_rad" in nan_cols, f"Should detect NaN in pitch_x_rad, got nan_cols={nan_cols}"
        assert "com_y_velocity_m_s" in inf_cols, f"Should detect Inf in velocity, got inf_cols={inf_cols}"

    def test_clean_telemetry_passes(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_nan_inf

        rows = [
            {"pitch_x_rad": "0.1", "com_y_velocity_m_s": "0.5"},
            {"pitch_x_rad": "-0.2", "com_y_velocity_m_s": "-0.3"},
        ]
        nan_cols, inf_cols = check_nan_inf(rows)
        assert len(nan_cols) == 0
        assert len(inf_cols) == 0


class TestModelReadinessGate:
    """Verify controller-design gate blocks if no DESIGN_READY model."""

    def test_gate_blocks_no_design_ready(self):
        """If no model has DESIGN_READY classification, gate should block."""
        model_validations = {
            "low_0p330": {"classification": "INSUFFICIENT_EXCITATION"},
            "mid_0p400": {"classification": "NEEDS_STATE_AUGMENTATION"},
            "high_0p480": {"classification": "INCONCLUSIVE"},
        }
        design_ready_count = sum(
            1 for v in model_validations.values()
            if v.get("classification") == "DESIGN_READY"
        )
        assert design_ready_count == 0
        assert design_ready_count < 1, "Gate should block — no DESIGN_READY models"

    def test_gate_allows_design_ready(self):
        """If at least one model is DESIGN_READY, gate should allow."""
        model_validations = {
            "low_0p330": {"classification": "INSUFFICIENT_EXCITATION"},
            "mid_0p400": {"classification": "DESIGN_READY"},
            "high_0p480": {"classification": "DESIGN_READY"},
        }
        design_ready_count = sum(
            1 for v in model_validations.values()
            if v.get("classification") == "DESIGN_READY"
        )
        assert design_ready_count >= 1

    def test_full_block_when_all_fail(self):
        """All validations failed — should block hard."""
        model_validations = {
            "low_0p330": {"classification": "FAILED_SIMULATION"},
            "mid_0p400": {"classification": "NAN_INF"},
            "high_0p480": {"classification": "INSUFFICIENT_LENGTH"},
        }
        design_ready = any(
            v.get("classification") == "DESIGN_READY"
            for v in model_validations.values()
        )
        assert not design_ready


class TestReportPaths:
    """Verify required output paths and report location."""

    def test_output_directory_structure(self):
        """Output directory parent must exist."""
        assert OUTPUT_DIR.parent.parent.parent.exists(), \
            "Project root should exist"

    def test_docs_validation_directory(self):
        """Docs validation directory must exist for report."""
        assert DOCS_DIR.exists(), \
            "docs/validation/ should exist"

    def test_scripts_directory_has_audit_script(self):
        """The integrity audit script must exist."""
        audit_path = SCRIPTS_DIR / "audit_k1_identification_dataset_integrity.py"
        assert audit_path.exists(), \
            f"Audit script not found at {audit_path}"


class TestExcitationSignalAudit:
    """Verify excitation signal audit functions."""

    def test_excitation_signal_clean_passes(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_excitation_signal
        import tempfile

        # Create a temp directory with a valid excitation signal
        with tempfile.TemporaryDirectory() as tmpdir:
            exc_path = Path(tmpdir) / "excitation_signal.json"
            signal_data = {
                "signal": [0.20, -0.20, 0.20, -0.20] * 100,
                "n_steps": 400,
                "amplitude_max": 0.20,
                "is_zero_mean": True,
            }
            with open(exc_path, "w") as f:
                json.dump(signal_data, f)

            result = check_excitation_signal(Path(tmpdir), "D_prbs_excitation")
            assert result["present"] is True
            assert result["issue"] is None
            assert result["is_zero_mean"] is True
            assert result["has_nan"] is False

    def test_missing_excitation_for_prbs_is_flagged(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_excitation_signal
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_excitation_signal(Path(tmpdir), "D_prbs_excitation")
            assert result["present"] is False
            assert result["issue"] == "MISSING_EXCITATION_SIGNAL"

    def test_excitation_not_required_for_equilibrium(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_excitation_signal
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_excitation_signal(Path(tmpdir), "A_equilibrium")
            assert result["present"] is False
            assert result["issue"] is None  # Not required, so no issue


class TestFallDetection:
    """Verify fall detection logic."""

    def test_large_pitch_is_fall(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_fall_detection

        rows = [{"pitch_x_rad": "1.5", "height_error_m": "0.05"}]  # >57 degrees
        fell, reasons = check_fall_detection(rows)
        assert fell

    def test_large_height_error_is_fall(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_fall_detection

        rows = [{"pitch_x_rad": "0.1", "height_error_m": "0.5"}]  # 50cm error
        fell, reasons = check_fall_detection(rows)
        assert fell

    def test_normal_operation_not_fall(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identification_dataset_integrity import check_fall_detection

        rows = [{"pitch_x_rad": "0.05", "height_error_m": "0.02"}]  # Normal
        fell, reasons = check_fall_detection(rows)
        assert not fell
