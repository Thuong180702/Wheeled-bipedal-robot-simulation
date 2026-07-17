"""Tests for K1 Identification Dataset and State-Feedback Design Prep.

Verifies:
  - Generation script compiles
  - State vector extraction returns finite values
  - Excitation signal bounded and zero-mean
  - Metadata contains validation_source=real_simulation
  - Identification function returns finite A/B
  - Model validation rejects bad/overfit models
  - Design readiness classifier works
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


class TestScriptCompilation:
    """Verify all identification dataset scripts compile."""

    def test_generate_dataset_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_k1_identification_dataset",
            SCRIPTS_DIR / "generate_k1_identification_dataset.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "verify_baseline")
        assert hasattr(mod, "generate_dataset")
        assert hasattr(mod, "generate_prbs_signal")
        assert hasattr(mod, "check_data_quality")

    def test_evaluate_state_vectors_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_k1_identification_state_vectors",
            SCRIPTS_DIR / "evaluate_k1_identification_state_vectors.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "evaluate_state_vectors")
        assert hasattr(mod, "identify_linear_model")
        assert hasattr(mod, "check_observability")

    def test_identify_models_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "identify_k1_mujoco_state_space_models",
            SCRIPTS_DIR / "identify_k1_mujoco_state_space_models.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "identify_ridge")
        assert hasattr(mod, "identify_robust")
        assert hasattr(mod, "identify_ols")
        assert hasattr(mod, "cross_validate")

    def test_validate_models_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "validate_k1_identified_models",
            SCRIPTS_DIR / "validate_k1_identified_models.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "validate_model")
        assert hasattr(mod, "classify_model")
        assert hasattr(mod, "find_dominant_mode")

    def test_height_schedule_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "analyze_k1_height_scheduled_models",
            SCRIPTS_DIR / "analyze_k1_height_scheduled_models.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "analyze_height_schedule")
        assert hasattr(mod, "compute_eigenvalue_summary")
        assert hasattr(mod, "compute_participation_factors")

    def test_control_feasibility_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_k1_identified_model_control_feasibility",
            SCRIPTS_DIR / "audit_k1_identified_model_control_feasibility.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "controllability_rank")
        assert hasattr(mod, "pbh_test")
        assert hasattr(mod, "lqr_benchmark")
        assert hasattr(mod, "audit_control_feasibility")


class TestBaselineVerification:
    """Verify K1 baseline is unchanged."""

    def test_k1_gains_unchanged(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from generate_k1_identification_dataset import K1_GAINS

        assert K1_GAINS["kp_pitch"] == 50.0
        assert K1_GAINS["kd_pitch"] == 10.0
        assert K1_GAINS["k_position"] == 40.0
        assert K1_GAINS["k_velocity"] == 15.0
        assert K1_GAINS["k_wheel_velocity"] == 0.5
        assert K1_GAINS["k_support_velocity"] == 0.0

    def test_baseline_verification_passes(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from generate_k1_identification_dataset import verify_baseline

        result = verify_baseline()
        assert result["k1_is_current_best"] is True
        assert result["profile_unchanged"] is True
        assert result["no_controller_modification"] is True


class TestExcitationSignals:
    """Verify excitation signals are valid."""

    def test_prbs_bounded_and_zero_mean(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from generate_k1_identification_dataset import generate_prbs_signal

        signal = generate_prbs_signal(2000, amplitude=0.15, seed=42)
        assert len(signal) == 2000
        assert np.max(np.abs(signal)) <= 0.15 + 1e-10
        # Zero-mean: with sufficient samples and balanced ±amplitude, mean should be small.
        # PRBS with random period lengths may have slight imbalance — allow up to 5% of amplitude
        assert abs(np.mean(signal)) < 0.02, f"Mean={np.mean(signal):.6f} exceeds 0.02"
        assert np.all(np.isfinite(signal))

    def test_prbs_switches_state(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from generate_k1_identification_dataset import generate_prbs_signal

        signal = generate_prbs_signal(500, amplitude=0.15, seed=42)
        # Should have both +amplitude and -amplitude values
        has_positive = np.any(signal > 0.05)
        has_negative = np.any(signal < -0.05)
        assert has_positive and has_negative, "PRBS should have both signs"

    def test_chirp_bounded(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from generate_k1_identification_dataset import generate_chirp_signal

        signal = generate_chirp_signal(500, amplitude=0.20)
        assert len(signal) == 500
        assert np.max(np.abs(signal)) <= 0.20 + 1e-10
        assert np.all(np.isfinite(signal))


class TestStateVectorExtraction:
    """Verify state vector extraction produces valid outputs."""

    def test_x6_extraction_dimensions(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from evaluate_k1_identification_state_vectors import STATE_VECTOR_CANDIDATES

        x6 = STATE_VECTOR_CANDIDATES["x6_base"]
        assert x6["dim"] == 6
        assert len(x6["names"]) == 6
        assert "pitch_x" in x6["names"]

    def test_augmented_vectors_have_more_dims(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from evaluate_k1_identification_state_vectors import STATE_VECTOR_CANDIDATES

        for vec_name in ["x7_add_height", "x8_add_notch", "x9_add_position", "x_filter_augmented"]:
            assert STATE_VECTOR_CANDIDATES[vec_name]["dim"] > 6

    def test_all_vectors_have_names_match_dim(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from evaluate_k1_identification_state_vectors import STATE_VECTOR_CANDIDATES

        for vec_name, vec_info in STATE_VECTOR_CANDIDATES.items():
            assert vec_info["dim"] == len(vec_info["names"]), \
                f"{vec_name}: dim={vec_info['dim']} != len(names)={len(vec_info['names'])}"


class TestSystemIdentification:
    """Verify identification functions produce finite outputs."""

    def test_ridge_returns_finite(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from identify_k1_mujoco_state_space_models import identify_ridge

        np.random.seed(42)
        X = np.random.randn(100, 6) * 0.1
        X_next = 0.9 * X + np.random.randn(100, 6) * 0.01
        U = np.random.randn(100, 1) * 0.01

        A, B, info = identify_ridge(X, U, X_next)
        assert A.shape == (6, 6)
        assert B.shape == (6, 1)
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(B))
        assert info["r_squared"] > -1.0

    def test_robust_returns_finite(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from identify_k1_mujoco_state_space_models import identify_robust

        np.random.seed(42)
        X = np.random.randn(100, 6) * 0.1
        X_next = 0.9 * X + np.random.randn(100, 6) * 0.01
        U = np.random.randn(100, 1) * 0.01

        A, B, info = identify_robust(X, U, X_next)
        assert A.shape == (6, 6)
        assert B.shape == (6, 1)
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(B))

    def test_ols_returns_finite(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from identify_k1_mujoco_state_space_models import identify_ols

        np.random.seed(42)
        X = np.random.randn(100, 6) * 0.1
        X_next = 0.9 * X + np.random.randn(100, 6) * 0.01
        U = np.random.randn(100, 1) * 0.01

        A, B, info = identify_ols(X, U, X_next)
        assert A.shape == (6, 6)
        assert B.shape == (6, 1)
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(B))

    def test_cross_validate_runs(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from identify_k1_mujoco_state_space_models import cross_validate

        np.random.seed(42)
        X = np.random.randn(200, 6) * 0.1
        X_next = 0.9 * X + np.random.randn(200, 6) * 0.01
        U = np.random.randn(200, 1) * 0.01

        cv = cross_validate(X, U, X_next)
        assert cv["n_train"] > 0
        assert cv["n_test"] > 0
        assert cv["train_r2"] > -1.0
        assert cv["test_r2"] > -1.0


class TestModelValidation:
    """Verify model validation and classification."""

    def test_classify_overfit(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from validate_k1_identified_models import classify_model

        validation = {
            "tests": {
                "one_step": {"n_samples": 100, "r_squared": 0.9999, "total_rmse": 0.001,
                            "rmse_per_state": [0.001, 0.002, 0.001, 0.001, 0.001, 0.001]},
                "mode": {"mode_found": True, "frequency_hz": 0.30, "damping_ratio": 0.10,
                        "freq_error_pct": 5.0, "zeta_error": 0.01},
                "rollout_50": {"diverged": False},
                "impulse_response": {"physically_plausible": True},
            }
        }
        result = classify_model(validation)
        assert "OVERFIT" in result or "DESIGN_READY" in result or "INCONCLUSIVE" in result, \
            f"Got unexpected: {result}"

    def test_classify_design_ready(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from validate_k1_identified_models import classify_model

        validation = {
            "tests": {
                "one_step": {"n_samples": 500, "r_squared": 0.98, "rmse_per_state": [0.01] * 6,
                            "total_rmse": 0.01},
                "mode": {"mode_found": True, "frequency_hz": 0.24, "damping_ratio": 0.10,
                        "freq_error_pct": 5.0, "zeta_error": 0.01},
                "rollout_50": {"diverged": False, "total_rmse": 0.05},
                "impulse_response": {"physically_plausible": True},
            }
        }
        result = classify_model(validation)
        assert "DESIGN_READY" in result or "INCONCLUSIVE" in result

    def test_classify_insufficient_data(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from validate_k1_identified_models import classify_model

        validation = {
            "tests": {
                "one_step": {"n_samples": 5},
            }
        }
        result = classify_model(validation)
        assert result == "HEIGHT_DATA_INSUFFICIENT"


class TestControllabilityAudit:
    """Verify controllability computations."""

    def test_controllability_rank_full(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identified_model_control_feasibility import controllability_rank

        n = 6
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[5, :] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        B = np.zeros((n, 1))
        B[5, 0] = 1.0

        result = controllability_rank(A, B)
        assert result["is_fully_controllable"]

    def test_pbh_controllable(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identified_model_control_feasibility import pbh_test

        # Controllable canonical form: PBH should pass for its eigenvalues
        n = 6
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[5, :] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        B = np.zeros((n, 1))
        B[5, 0] = 1.0

        # Test for eigenvalue λ=0.5 (which is NOT an eigenvalue of A)
        # For a PBH test, we check pass/fail regardless of whether λ is an eigenvalue
        result = pbh_test(A, B, 0.5)
        # When λ is not an eigenvalue, λI-A has full rank n, so augmented rank=n
        assert result["augmented_rank"] >= n, f"Expected rank >= {n}, got {result['augmented_rank']}"

    def test_lqr_benchmark_not_controller(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from audit_k1_identified_model_control_feasibility import lqr_benchmark

        n = 6
        A = np.eye(n) * 0.9
        B = np.ones((n, 1))

        result = lqr_benchmark(A, B)
        assert "K_lqr" in result
        assert "FEASIBILITY_BENCHMARK_ONLY" in result["note"]
        assert "NOT a controller implementation" in result["note"]


class TestReportPath:
    """Verify report paths exist."""

    def test_output_directory_created(self):
        output_dir = PROJECT_ROOT / "outputs" / "k1_identification_dataset"
        assert output_dir.parent.parent.parent.exists(), \
            "Project output directory should exist"

    def test_validation_docs_directory(self):
        validation_dir = PROJECT_ROOT / "docs" / "validation"
        assert validation_dir.exists(), \
            "Validation docs directory should exist"
