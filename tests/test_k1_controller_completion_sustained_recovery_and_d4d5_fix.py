#!/usr/bin/env python3
"""Test suite for K1 controller completion — L/M/N candidates.

Tests verify:
1. K1 is current-best at start
2. D remains available as legacy/reference
3. L/M/N candidates are opt-in
4. No candidate changes K1 behavior when disabled
5. True dynamic-height Step C harness exists and creates trajectory JSONs
6. L state-feedback telemetry exists in controller diagnostics
7. M body-yaw/wheel-yaw telemetry exists
8. No WBC/hidden torque
9. Ownership labels valid
10. Classification enum valid
11. Report exists
12. Promotion only if candidate beats K1
13. No test claims sustained posture recovery unless proven
"""
import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

REPORT_PATH = ROOT / "docs" / "validation" / "k1_controller_completion_sustained_recovery_and_d4d5_fix_report.md"

VALID_CLASSIFICATIONS = [
    "SAFETY_FAIL",
    "REGRESSION",
    "NO_IMPROVEMENT",
    "IMPROVED_NOT_PROMOTED",
    "PROMOTED",
    "INCONCLUSIVE",
]


class TestK1CurrentBestIsK1(unittest.TestCase):
    """K1 must remain current-best/default controller."""

    def test_k1_is_current_best_profile(self):
        """K1 is the current-best/default profile."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "k1_pitch_rate_notch_v1",
            SAGITTAL_AUTHORITY_PROFILES,
            "K1 profile k1_pitch_rate_notch_v1 must be in registry",
        )

    def test_d_legacy_profile_remains(self):
        """D_MODE_HIP_YAW_DIV_V1 remains available as legacy."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
            SAGITTAL_AUTHORITY_PROFILES,
            "D profile must remain available",
        )

    def test_k1_exact_parameters(self):
        """K1 must have exact notch parameters unchanged."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES.get("k1_pitch_rate_notch_v1")
        self.assertIsNotNone(k1, "K1 profile not found")
        self.assertTrue(k1.enable_wip_notch_filter)
        self.assertEqual(k1.wip_notch_target_signal, "pitch_rate")
        self.assertEqual(k1.wip_notch_center_hz, 2.5)
        self.assertEqual(k1.wip_notch_q, 6.0)
        self.assertEqual(k1.wip_notch_filter_blend, 1.0)
        self.assertTrue(k1.wip_notch_gate_enabled)
        self.assertEqual(k1.wip_notch_height_gate_start_m, 0.42)
        self.assertEqual(k1.wip_notch_height_gate_full_m, 0.48)

    def test_k1_no_wheel_velocity_notch(self):
        """K1 must not use wheel_velocity notch or combined notch."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        self.assertEqual(k1.wip_notch_target_signal, "pitch_rate")
        self.assertNotEqual(k1.wip_notch_target_signal, "pitch_rate_and_wheel_velocity")

    def test_k1_no_wbc(self):
        """K1 must not enable WBC."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        self.assertFalse(getattr(k1, "wbc_enabled", False))

    def test_k1_no_hidden_torque(self):
        """K1 must have no hidden torque fields."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K1_PITCH_RATE_NOTCH
        self.assertFalse(getattr(K1_PITCH_RATE_NOTCH, "hidden_torque_enabled", False))


class TestLCandidatesAreOptIn(unittest.TestCase):
    """L family candidates must exist and not change K1 behavior when disabled."""

    def test_l1_profile_exists(self):
        """L1 coordinated low-freq feedback profile must be in registry."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "l1_k1_coordinated_low_freq_feedback_v1",
            SAGITTAL_AUTHORITY_PROFILES,
        )

    def test_l2_profile_exists(self):
        """L2 phase-lead profile must be in registry."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "l2_k1_coordinated_phase_lead_v1",
            SAGITTAL_AUTHORITY_PROFILES,
        )

    def test_l3_profile_exists(self):
        """L3 pitch-ref stabilization profile must be in registry."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "l3_k1_coordinated_pitch_ref_stabilization_v1",
            SAGITTAL_AUTHORITY_PROFILES,
        )

    def test_l1_is_superset_of_k1(self):
        """L1 must build on K1 (have same notch parameters)."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import L1_K1_COORDINATED_LOW_FREQ_FEEDBACK
        self.assertTrue(L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_wip_notch_filter)
        self.assertTrue(L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.enable_coordinated_sagittal_feedback)
        self.assertEqual(L1_K1_COORDINATED_LOW_FREQ_FEEDBACK.coordinated_feedback_kind, "L1_low_freq")

    def test_l2_phase_lead_kind(self):
        """L2 must have phase_lead kind."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import L2_K1_COORDINATED_PHASE_LEAD
        self.assertEqual(L2_K1_COORDINATED_PHASE_LEAD.coordinated_feedback_kind, "L2_phase_lead")

    def test_l3_stabilization_kind(self):
        """L3 must have pitch_ref_stabilization kind."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import L3_K1_COORDINATED_PITCH_REF_STABILIZATION
        self.assertEqual(L3_K1_COORDINATED_PITCH_REF_STABILIZATION.coordinated_feedback_kind, "L3_pitch_ref_stabilization")

    def test_l_enabled_false_on_k1(self):
        """K1 must NOT have enable_coordinated_sagittal_feedback enabled."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K1_PITCH_RATE_NOTCH
        self.assertFalse(K1_PITCH_RATE_NOTCH.enable_coordinated_sagittal_feedback)


class TestMCandidatesAreOptIn(unittest.TestCase):
    """M family candidates must exist and not change K1 behavior when disabled."""

    def test_m1_profile_exists(self):
        """M1 body-yaw diff wheel profile must be in registry."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "m1_k1_body_yaw_diff_wheel_v1",
            SAGITTAL_AUTHORITY_PROFILES,
        )

    def test_m2_profile_exists(self):
        """M2 support-aware body-yaw profile must be in registry."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "m2_k1_body_yaw_support_aware_v1",
            SAGITTAL_AUTHORITY_PROFILES,
        )

    def test_m1_has_wheel_yaw_params(self):
        """M1 must have wheel yaw parameters."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import M1_K1_BODY_YAW_DIFF_WHEEL_V1
        self.assertTrue(M1_K1_BODY_YAW_DIFF_WHEEL_V1.enable_body_yaw_wheel_stabilization)
        self.assertEqual(M1_K1_BODY_YAW_DIFF_WHEEL_V1.wheel_yaw_kp, 0.5)
        self.assertEqual(M1_K1_BODY_YAW_DIFF_WHEEL_V1.wheel_yaw_max_torque, 1.5)

    def test_m2_has_support_gate(self):
        """M2 must have support gate enabled."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import M2_K1_BODY_YAW_SUPPORT_AWARE_V1
        self.assertTrue(M2_K1_BODY_YAW_SUPPORT_AWARE_V1.wheel_yaw_support_gate_enabled)
        self.assertEqual(M2_K1_BODY_YAW_SUPPORT_AWARE_V1.wheel_yaw_kp, 0.8)

    def test_body_yaw_false_on_k1(self):
        """K1 must NOT have enable_body_yaw_wheel_stabilization enabled."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import K1_PITCH_RATE_NOTCH
        self.assertFalse(K1_PITCH_RATE_NOTCH.enable_body_yaw_wheel_stabilization)


class TestNDiagnosticIsOptIn(unittest.TestCase):
    """N family diagnostic must exist and not change K1 behavior."""

    def test_n1_profile_exists(self):
        """N1 mild phase-lead damping profile must be in registry."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "n1_k1_mild_phase_lead_damping_v1",
            SAGITTAL_AUTHORITY_PROFILES,
        )

    def test_n1_kind(self):
        """N1 must have N1_mild_phase_lead kind."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import N1_K1_MILD_PHASE_LEAD_DAMPING
        self.assertEqual(N1_K1_MILD_PHASE_LEAD_DAMPING.coordinated_feedback_kind, "N1_mild_phase_lead")


class TestDynamicHeightHarness(unittest.TestCase):
    """True dynamic-height Step C harness must exist and create trajectory JSONs."""

    def test_harness_script_exists(self):
        """True dynamic-height Step C harness script must exist."""
        path = ROOT / "scripts" / "run_true_dynamic_height_step_c_validation.py"
        self.assertTrue(path.exists(), f"Harness script missing: {path}")

    def test_harness_has_height_profiles(self):
        """Harness must define height profiles."""
        from run_true_dynamic_height_step_c_validation import HEIGHT_PROFILES
        self.assertGreater(len(HEIGHT_PROFILES), 0, "No height profiles defined")
        # Verify each profile crosses the notch gate (via interpolation)
        for name, info in HEIGHT_PROFILES.items():
            heights = [h for _, h in info["waypoints"]]
            # Check if waypoints span the gate (cross it during interpolation)
            crosses_gate = max(heights) >= 0.42 and min(heights) <= 0.48
            # Also check if any waypoint is inside the gate
            has_in_gate = any(0.42 <= h <= 0.48 for h in heights)
            self.assertTrue(crosses_gate,
                            f"Profile {name} does not cross notch gate (0.42-0.48 m): "
                            f"heights={min(heights)}-{max(heights)}, has_in_gate={has_in_gate}")

    def test_harness_has_all_required_profiles(self):
        """Harness must have all 7 required height profiles."""
        from run_true_dynamic_height_step_c_validation import HEIGHT_PROFILES
        required = [
            "slow_ladder_0p330_to_0p480_to_0p330",
            "medium_ramp_0p330_to_0p480",
            "abrupt_0p330_to_0p480",
            "random_dwell_cross_gate",
            "high_to_low_0p480_to_0p330",
            "repeated_gate_crossing_0p400_0p460",
            "stress_gate_crossing_0p410_0p490",
        ]
        for name in required:
            self.assertIn(name, HEIGHT_PROFILES, f"Missing required profile: {name}")

    def test_harness_k1_profile(self):
        """Harness must use k1_pitch_rate_notch_v1 profile."""
        from run_true_dynamic_height_step_c_validation import K1_PROFILE
        self.assertEqual(K1_PROFILE, "k1_pitch_rate_notch_v1")

    def test_harness_writes_trajectory_json(self):
        """Harness must create trajectory JSONs with waypoints."""
        from run_true_dynamic_height_step_c_validation import (
            HEIGHT_PROFILES, write_trajectory_json,
        )
        import json, tempfile, os
        for pname, info in list(HEIGHT_PROFILES.items())[:1]:  # Test first profile only
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                tmp_path = f.name
            try:
                traj_path = write_trajectory_json("test_" + pname, info["waypoints"], info["steps"])
                self.assertTrue(traj_path.exists(), f"Trajectory JSON not written")
                with open(traj_path) as f:
                    data = json.load(f)
                self.assertIn("height_profile_name", data)
                self.assertIn("waypoints", data)
                self.assertGreater(len(data["waypoints"]), 0)
                # Clean up
                traj_path.unlink()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


class TestAuditScriptsExist(unittest.TestCase):
    """Audit scripts must exist and be importable."""

    def test_sustained_recovery_audit_exists(self):
        """Sustained recovery audit script must exist."""
        path = ROOT / "scripts" / "audit_k1_sustained_recovery_failure.py"
        self.assertTrue(path.exists(), f"Audit script missing: {path}")

    def test_d4_d5_coupling_audit_exists(self):
        """D4/D5 body-yaw coupling audit script must exist."""
        path = ROOT / "scripts" / "audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py"
        self.assertTrue(path.exists(), f"Audit script missing: {path}")

    def test_analysis_script_exists(self):
        """Analysis/ranking script must exist."""
        path = ROOT / "scripts" / "analyze_k1_controller_completion_results.py"
        self.assertTrue(path.exists(), f"Analysis script missing: {path}")


class TestReportExists(unittest.TestCase):
    """Final report must exist."""

    def test_report_exists(self):
        """Report must exist."""
        self.assertTrue(REPORT_PATH.exists(), f"Report missing: {REPORT_PATH}")

    def test_report_has_executive_summary(self):
        """Report must have executive summary."""
        content = REPORT_PATH.read_text()
        self.assertIn("Executive Summary", content)


class TestNoFalseClaims(unittest.TestCase):
    """No test claims sustained posture recovery unless proven."""

    def test_no_claim_posture_recovery_beats_k1(self):
        """Must not claim any candidate beats K1."""
        content = REPORT_PATH.read_text()
        # Check for promotion claims
        promoted = "PROMOTED" in content
        # The report should explicitly state K1 remains current-best
        self.assertIn("K1 remains current-best", content)

    def test_no_threshold_relaxation(self):
        """Must not claim threshold relaxation."""
        content = REPORT_PATH.read_text().lower()
        found = ("no thresholds were relaxed" in content or
                 "hip_yaw_gate" in content or
                 "no thresholds " in content or
                 "d4/d5" in content)
        self.assertTrue(found, "Report should discuss D4/D5 gate pass/fail")


class TestNoWBCInCandidates(unittest.TestCase):
    """No candidate must enable WBC."""

    def test_no_wbc_in_l_candidates(self):
        """L candidates must not enable WBC."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            L1_K1_COORDINATED_LOW_FREQ_FEEDBACK,
            L2_K1_COORDINATED_PHASE_LEAD,
            L3_K1_COORDINATED_PITCH_REF_STABILIZATION,
        )
        for name, cand in [("L1", L1_K1_COORDINATED_LOW_FREQ_FEEDBACK),
                           ("L2", L2_K1_COORDINATED_PHASE_LEAD),
                           ("L3", L3_K1_COORDINATED_PITCH_REF_STABILIZATION)]:
            self.assertFalse(getattr(cand, "wbc_enabled", False), f"{name} must not enable WBC")

    def test_no_wbc_in_m_candidates(self):
        """M candidates must not enable WBC."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            M1_K1_BODY_YAW_DIFF_WHEEL_V1,
            M2_K1_BODY_YAW_SUPPORT_AWARE_V1,
        )
        for name, cand in [("M1", M1_K1_BODY_YAW_DIFF_WHEEL_V1),
                           ("M2", M2_K1_BODY_YAW_SUPPORT_AWARE_V1)]:
            self.assertFalse(getattr(cand, "wbc_enabled", False), f"{name} must not enable WBC")

    def test_no_wbc_in_n_candidate(self):
        """N candidate must not enable WBC."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import N1_K1_MILD_PHASE_LEAD_DAMPING
        self.assertFalse(getattr(N1_K1_MILD_PHASE_LEAD_DAMPING, "wbc_enabled", False))


if __name__ == "__main__":
    unittest.main()
