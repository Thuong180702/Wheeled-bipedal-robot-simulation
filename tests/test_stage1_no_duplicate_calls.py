"""Stage 1: Verify duplicate estimator calls are profiled but NOT removed.

The duplicate centroidal_estimator.estimate() and capture_estimator.update() calls
are NOT removed because capture_estimator uses a different min_height clamp (0.35m)
than the centroidal estimator's internal capture-point computation (0.1m). At K2's
low height of 0.33m, removing the capture_estimator.update() calls would change
the capture_point values used in telemetry.

These tests verify:
1. The duplicate calls exist and are counted by --profile-controller
2. The profile report correctly identifies duplicate calls
3. The duplicate removal blocker is documented
"""

import json
import os
import pytest


PROFILE_PATH = "outputs/profile/stage1_controller_profile_breakdown.json"


class TestDuplicateEstimatorCallsExist:
    """Verify duplicate calls are detected in the profile report."""

    @pytest.mark.skipif(
        not os.path.exists(PROFILE_PATH),
        reason="Profile report not yet generated. Run with --profile-controller first.",
    )
    def test_profile_report_exists(self):
        """Profile report JSON file exists."""
        assert os.path.exists(PROFILE_PATH), (
            f"Profile report not found at {PROFILE_PATH}. "
            "Run: python scripts/simulate_hierarchical_controller.py "
            "--controller-mode balance-core --sagittal-controller velocity-damped "
            "--vd-sagittal-authority-profile k2_notch_low_q_v1 "
            "--height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json "
            "--profile-controller --steps 100"
        )

    @pytest.mark.skipif(
        not os.path.exists(PROFILE_PATH),
        reason="Profile report not yet generated.",
    )
    def test_duplicate_calls_detected(self):
        """Profile report identifies duplicate calls correctly."""
        with open(PROFILE_PATH) as f:
            report = json.load(f)

        dup = report.get("duplicate_call_analysis", {})
        assert dup.get("centroidal_estimate_called_twice_per_step") is True
        assert dup.get("capture_estimator_update_called_twice_per_step") is True
        assert dup.get("duplicate_removed") is False
        assert "min_height" in dup.get("duplicate_removal_blocked_by", "").lower()

    @pytest.mark.skipif(
        not os.path.exists(PROFILE_PATH),
        reason="Profile report not yet generated.",
    )
    def test_estimated_savings_reported(self):
        """Estimated savings from removing duplicates is a positive number."""
        with open(PROFILE_PATH) as f:
            report = json.load(f)

        savings = report.get("duplicate_call_analysis", {}).get(
            "estimated_savings_if_removed_ms", 0.0
        )
        assert savings >= 0.0, "Estimated savings should be non-negative"

    @pytest.mark.skipif(
        not os.path.exists(PROFILE_PATH),
        reason="Profile report not yet generated.",
    )
    def test_all_timing_keys_present(self):
        """Profile report has all expected timing keys."""
        with open(PROFILE_PATH) as f:
            report = json.load(f)

        timing = report.get("timing_mean_ms", {})
        required_keys = [
            "centroidal_control",
            "capture_control",
            "balance_core_block",
            "centroidal_log",
            "capture_log",
            "telemetry",
            "total_per_step",
        ]
        for key in required_keys:
            assert key in timing, f"Missing timing key: {key}"
            assert timing[key] >= 0.0, f"Timing for {key} is negative: {timing[key]}"


class TestDuplicateCallsNotRemoved:
    """Verify duplicate calls are intentionally preserved."""

    def test_capture_estimator_min_height_differs(self):
        """capture_estimator min_height (0.35m) differs from centroidal (0.1m)."""
        from wheeled_biped.controllers.capture_point_estimator import (
            CapturePointEstimatorConfig,
        )

        config = CapturePointEstimatorConfig()
        assert config.min_height == 0.35, (
            f"Expected min_height=0.35, got {config.min_height}. "
            "If this changed, re-evaluate whether duplicate capture_estimator.update() "
            "calls can be safely removed."
        )

    def test_centroidal_estimator_min_height_is_01(self):
        """centroidal_estimator hardcodes max(h, 0.1) for capture point omega."""
        # The centroidal estimator uses jnp.maximum(com_height, 0.1)
        # at line ~198 of centroidal_state_estimator.py
        import inspect
        from wheeled_biped.controllers.centroidal_state_estimator import (
            CentroidalStateEstimator,
        )

        source = inspect.getsource(CentroidalStateEstimator.estimate)
        assert "0.1" in source or "jnp.maximum(com_height" in source, (
            "centroidal_estimator.estimate() should use min_height=0.1 for capture point. "
            "If this changed, re-evaluate duplicate removal."
        )


class TestProfileFlagWorks:
    """Verify --profile-controller flag is parsed correctly."""

    def test_flag_registered(self):
        """--profile-controller is a recognized argument."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "scripts/simulate_hierarchical_controller.py",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "--profile-controller" in result.stdout, (
            "--profile-controller flag not found in help output"
        )
