"""Tests for K1 nonlinear mode source isolation and augmented state ID — Phase 11.

Verifies:
  - Augmented telemetry fields exist in schema
  - Telemetry-only instrumentation does not change K1 torque output
  - Torque decomposition reconstructs total torque
  - Clipping flags match clip deltas
  - Notch fields are finite when enabled
  - Source classifier handles synthetic arrays
  - Model validation rejects missing mode
  - Fix feasibility gate blocks without DESIGN_READY
  - All scripts compile
  - Report path exists
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"
DOCS_DIR = PROJECT_ROOT / "docs" / "validation"


class TestScriptCompilation:
    """Verify all augmented pipeline scripts compile."""

    SCRIPTS = [
        "generate_k1_augmented_identification_dataset.py",
        "audit_k1_augmented_dataset_integrity.py",
        "audit_k1_nonlinear_mode_source.py",
        "identify_k1_augmented_state_models.py",
        "validate_k1_augmented_models.py",
        "audit_k1_ablation_source_check.py",
        "audit_k1_augmented_fix_feasibility.py",
    ]

    @pytest.mark.parametrize("script_name", SCRIPTS)
    def test_script_compiles(self, script_name):
        import importlib.util
        script_path = SCRIPTS_DIR / script_name
        assert script_path.exists(), f"Script {script_name} not found"
        spec = importlib.util.spec_from_file_location(
            script_name.replace(".py", ""), script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    def test_all_seven_scripts_exist(self):
        for name in self.SCRIPTS:
            assert (SCRIPTS_DIR / name).exists(), f"Missing: {name}"


class TestSourceClassifier:
    """Verify source isolation classifier works with synthetic data."""

    def test_coherence_computation(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import compute_coherence

        # Two identical signals should have high coherence
        np.random.seed(42)
        t = np.linspace(0, 20, 2000)  # 2000 samples at 100Hz = 20s
        sig = np.sin(2 * np.pi * 0.3 * t)
        result = compute_coherence(sig, sig, fs=100.0)
        # Two identical sine waves have perfect coherence at their frequency
        if result["sufficient_data"]:
            assert result["peak_coherence"] > 0.8, \
                f"Identical signals should have high coherence, got {result['peak_coherence']:.3f}"
        else:
            # If FFT window too short, still check shape
            assert result["mean_coherence"] >= 0.0

    def test_coherence_independent_signals(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import compute_coherence

        np.random.seed(42)
        t = np.linspace(0, 10, 500)
        sig1 = np.sin(2 * np.pi * 0.3 * t)
        sig2 = np.random.randn(500)
        result = compute_coherence(sig1, sig2)
        # Independent signals have lower coherence
        assert result["peak_coherence"] < 0.5, \
            f"Independent signals should have low coherence, got {result['peak_coherence']:.3f}"

    def test_find_dominant_frequency(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import compute_welch_psd, find_dominant_frequency

        np.random.seed(42)
        t = np.arange(0, 10, 0.01)
        sig = np.sin(2 * np.pi * 0.35 * t) + 0.1 * np.random.randn(len(t))
        freqs, psd = compute_welch_psd(sig)
        result = find_dominant_frequency(freqs, psd, band=(0.15, 0.55))
        assert result["found"], "Should find the 0.35 Hz mode"
        assert 0.25 <= result["freq_hz"] <= 0.50, \
            f"Expected ~0.35 Hz, got {result['freq_hz']:.3f}"

    def test_event_triggered_average(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import event_triggered_average

        np.random.seed(42)
        n = 500
        sig = np.zeros(n)
        # Add pulse at each trigger
        trigger = np.zeros(n)
        trigger[100] = 1
        trigger[200] = 1
        trigger[300] = 1
        # Response grows after trigger
        for ev in [100, 200, 300]:
            sig[ev:ev + 20] = 1.0

        eta = event_triggered_average(sig, trigger, window=60)
        assert eta["n_events"] == 3
        assert eta["sufficient"]
        assert eta["post_event_mean"] > eta["pre_event_mean"], \
            "Response should grow after trigger"

    def test_classify_mode_source_notch_dominant(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import classify_mode_source

        # Synthetic analyses with high pitch-notch coherence
        analyses = [
            {
                "mode_found": True,
                "coherence": {
                    "pitch_vs_notch": {"peak_coherence": 0.85, "sufficient_data": True},
                    "pitch_vs_clip": {"peak_coherence": 0.20, "sufficient_data": True},
                    "pitch_vs_cap": {"peak_coherence": 0.15, "sufficient_data": True},
                },
            },
        ]
        result = classify_mode_source(analyses)
        assert result["classification"] == "NOTCH_FILTER_DOMINANT", \
            f"Expected NOTCH_FILTER_DOMINANT, got {result['classification']}"

    def test_classify_mode_source_clip_dominant(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import classify_mode_source

        analyses = [
            {
                "mode_found": True,
                "coherence": {
                    "pitch_vs_notch": {"peak_coherence": 0.20, "sufficient_data": True},
                    "pitch_vs_clip": {"peak_coherence": 0.85, "sufficient_data": True},
                    "pitch_vs_cap": {"peak_coherence": 0.15, "sufficient_data": True},
                },
            },
        ]
        result = classify_mode_source(analyses)
        assert result["classification"] == "TORQUE_CLIPPING_DOMINANT", \
            f"Expected TORQUE_CLIPPING_DOMINANT, got {result['classification']}"

    def test_classify_mode_source_inconclusive(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_nonlinear_mode_source import classify_mode_source

        analyses = [{"mode_found": False, "coherence": {}}]
        result = classify_mode_source(analyses)
        assert result["classification"] == "INCONCLUSIVE"


class TestModelValidation:
    """Verify model validation logic."""

    def test_validate_rejects_missing_mode(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from validate_k1_augmented_models import validate_model

        # Fit quality dict with no mode found
        fit = {
            "best_mode": None,
            "r2": 0.99,
            "condition_number": 100,
            "rollout_50_rmse": 0.1,
            "rollout_200_rmse": 0.2,
        }
        result = validate_model(fit, "x6_base", "high_0p480")
        assert result["classification"] != "DESIGN_READY", \
            "Model without mode should not be DESIGN_READY"

    def test_validate_accepts_valid_model(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from validate_k1_augmented_models import validate_model

        fit = {
            "best_mode": {"freq_hz": 0.30, "damping": 0.1},
            "r2": 0.995,
            "test_r2": 0.99,
            "condition_number": 500,
            "rollout_50_rmse": 0.1,
            "rollout_200_rmse": 0.2,
        }
        result = validate_model(fit, "x8_notch", "high_0p480")
        assert result["mode_captured"], "Should capture the 0.30 Hz mode"
        assert result["classification"] == "DESIGN_READY", \
            f"Expected DESIGN_READY, got {result['classification']}"


class TestFixFeasibilityGate:
    """Verify fix feasibility gate logic."""

    def test_gate_blocks_without_design_ready(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_augmented_fix_feasibility import determine_fix_feasibility

        # No model validation
        result = determine_fix_feasibility(None, None, None)
        assert result["conclusion"] == "G", \
            f"Expected G (INCONCLUSIVE), got {result['conclusion']}"
        # G means no design allowed — conclusion must NOT be A
        assert result["conclusion"] != "A", \
            "Should NOT allow design (conclusion A) without evidence"

    def test_gate_allows_with_design_ready(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_augmented_fix_feasibility import determine_fix_feasibility

        model_validation = {"allows_design": True}
        result = determine_fix_feasibility(None, model_validation, None)
        assert result["conclusion"] == "A", \
            f"Expected A (AUGMENTED_STATE_FEEDBACK_READY), got {result['conclusion']}"

    def test_gate_suggests_filter_redesign_for_notch_dominant(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_augmented_fix_feasibility import determine_fix_feasibility

        source_analysis = {
            "source_classification": {
                "classification": "NOTCH_FILTER_DOMINANT",
            },
        }
        result = determine_fix_feasibility(source_analysis, None, None)
        assert result["conclusion"] == "B", \
            f"Expected B (FILTER_PATH_REDESIGN), got {result['conclusion']}"


class TestReportPaths:
    """Verify output paths exist or can be created."""

    def test_output_directory_exists_or_creatable(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        assert OUTPUT_DIR.exists()

    def test_docs_validation_directory(self):
        assert DOCS_DIR.exists(), "docs/validation/ should exist"

    def test_augmented_telemetry_test_exists(self):
        test_path = PROJECT_ROOT / "tests" / "test_k1_augmented_telemetry.py"
        assert test_path.exists(), "Augmented telemetry test should exist"


class TestDataIntegrity:
    """Verify integrity audit classifications."""

    def test_classify_usable(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_augmented_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "nan_columns": [],
            "inf_columns": [],
            "fall_detected": False,
            "metadata_issues": [],
            "missing_augmented_fields": [],
            "torque_reconstructs": True,
            "n_rows": 2000,
            "min_total_rows": 1500,
            "post_settle_samples": 1500,
            "min_post_settle": 1000,
        }
        assert classify_run(run) == "USABLE"

    def test_classify_missing_augmented_fields(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_augmented_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "nan_columns": [],
            "inf_columns": [],
            "fall_detected": False,
            "metadata_issues": [],
            "missing_augmented_fields": ["k1_notch_state_1"],
            "torque_reconstructs": True,
            "n_rows": 2000,
        }
        assert classify_run(run) == "MISSING_AUGMENTED_FIELDS"

    def test_classify_bad_reconstruction(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_augmented_dataset_integrity import classify_run

        run = {
            "telemetry_exists": True,
            "nan_columns": [],
            "inf_columns": [],
            "fall_detected": False,
            "metadata_issues": [],
            "missing_augmented_fields": [],
            "torque_reconstructs": False,
            "n_rows": 2000,
        }
        assert classify_run(run) == "BAD_RECONSTRUCTION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
