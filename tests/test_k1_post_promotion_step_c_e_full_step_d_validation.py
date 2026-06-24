#!/usr/bin/env python3
"""Test suite for K1 Post-Promotion Step C/E/D validation.

Tests verify:
1. K1 is current-best at start of validation
2. K1 identity exact
3. D legacy/reference exists
4. Step E summary exists
5. Step C summary exists
6. Full Step D summary exists
7. Direct hip-yaw telemetry exists in new outputs
8. Notch telemetry exists in new outputs
9. No stub/assumed/synthetic rows accepted
10. Classification enum valid
11. Current-best remains K1 if confirmed
12. Rollback recommendation exists if K1 regresses
13. Report exists
14. No test claims K1 solved sustained posture recovery
"""
import json
import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

K1_OUT = ROOT / "outputs" / "k1_post_promotion_validation"
D_OUT = ROOT / "outputs" / "mode_hip_yaw_div_full_real_validation"
ANALYSIS_OUT = K1_OUT / "analysis"
REPORT_PATH = ROOT / "docs" / "validation" / "k1_post_promotion_step_c_e_full_step_d_validation_report.md"

VALID_CLASSIFICATIONS = [
    "K1_POST_PROMOTION_VALIDATION_CONFIRMED_CURRENT_BEST",
    "K1_POST_PROMOTION_VALIDATION_CONFIRMED_WITH_EXPANDED_LIMITATIONS",
    "K1_POST_PROMOTION_VALIDATION_ROLLBACK_RECOMMENDED_D_BETTER",
    "K1_POST_PROMOTION_VALIDATION_REJECTED_HARD_SAFETY_REGRESSION",
    "K1_POST_PROMOTION_VALIDATION_INCONCLUSIVE",
]


class TestK1IdentityAfterPromotion(unittest.TestCase):
    """K1 must remain current-best at start and D must remain available."""

    def test_k1_is_current_best_profile(self):
        """K1 is the current-best/profile."""
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


class TestStepEOutputs(unittest.TestCase):
    """Step E fixed-height validation outputs must exist."""

    def test_step_e_summary_csv_exists(self):
        """K1 Step E metrics CSV must exist."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        self.assertTrue(path.exists(), f"Step E metrics CSV missing: {path}")

    def test_step_e_has_10_heights(self):
        """Step E must have 10 heights."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        if not path.exists():
            self.skipTest("Step E metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        height_count = len(set(r.get("height", "") for r in rows))
        self.assertEqual(height_count, 10, f"Expected 10 heights, got {height_count}")

    def test_step_e_direct_hip_yaw_telemetry(self):
        """Step E must have hip_yaw_abs_max computed from direct telemetry."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        if not path.exists():
            self.skipTest("Step E metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        hy_vals = [float(r.get("hip_yaw_abs_max", "nan")) for r in rows if r.get("hip_yaw_abs_max", "")]
        self.assertTrue(len(hy_vals) > 0, "No hip_yaw_abs_max values found")
        self.assertTrue(all(v > 0 for v in hy_vals), "hip_yaw_abs_max should be positive")

    def test_step_e_notch_telemetry(self):
        """Step E must have notch_active_fraction."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        if not path.exists():
            self.skipTest("Step E metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        notch_fracs = [float(r.get("notch_active_fraction", "0")) for r in rows if r.get("notch_active_fraction", "")]
        # At least some heights should have notch active (tall heights)
        high_active = sum(1 for v in notch_fracs if v > 0.5)
        self.assertTrue(high_active >= 3,
                        f"Expected >=3 heights with notch active, got {high_active}")

    def test_step_e_no_falls(self):
        """Step E must show no falls."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        if not path.exists():
            self.skipTest("Step E metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        falls = [r.get("fell", "false") for r in rows]
        self.assertTrue(all(f == "False" or f == "false" or not f for f in falls),
                        "Step E should have no falls")

    def test_step_e_no_wbc_no_hidden_torque(self):
        """Step E must show zero WBC and hidden torque."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        if not path.exists():
            self.skipTest("Step E metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        wbc_sum = sum(float(r.get("wbc_authority_rows", "0") or "0") for r in rows)
        ht_sum = sum(float(r.get("hidden_torque_max", "0") or "0") for r in rows)
        self.assertEqual(wbc_sum, 0, "Step E should have zero WBC rows")
        self.assertEqual(ht_sum, 0, "Step E should have zero hidden torque")

    def test_step_e_validation_source_real_simulation(self):
        """Step E must use real_simulation source."""
        path = K1_OUT / "step_e_fixed_height" / "k1_step_e_fixed_height_metrics.csv"
        if not path.exists():
            self.skipTest("Step E metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        sources = [r.get("validation_source", "?") for r in rows]
        self.assertTrue(all(s == "real_simulation" for s in sources),
                        "All Step E rows must be real_simulation")


class TestStepCOutputs(unittest.TestCase):
    """Step C dynamic-height validation outputs must exist."""

    def test_step_c_summary_csv_exists(self):
        """K1 Step C metrics CSV must exist."""
        path = K1_OUT / "step_c_standard" / "k1_step_c_standard_metrics.csv"
        self.assertTrue(path.exists(), f"Step C metrics CSV missing: {path}")

    def test_step_c_has_7_cases(self):
        """Step C must have 7 standard cases."""
        path = K1_OUT / "step_c_standard" / "k1_step_c_standard_metrics.csv"
        if not path.exists():
            self.skipTest("Step C metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        case_count = len(set(r.get("case_id", "") for r in rows))
        self.assertEqual(case_count, 7, f"Expected 7 Step C cases, got {case_count}")

    def test_step_c_no_falls(self):
        """Step C must show no falls."""
        path = K1_OUT / "step_c_standard" / "k1_step_c_standard_metrics.csv"
        if not path.exists():
            self.skipTest("Step C metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        falls = [r.get("fell", "false") for r in rows]
        self.assertTrue(all(f == "False" or f == "false" or not f for f in falls),
                        "Step C should have no falls")


class TestStepDOutputs(unittest.TestCase):
    """Full Step D validation outputs must exist."""

    def test_step_d_summary_csv_exists(self):
        """K1 full Step D metrics CSV must exist."""
        path = K1_OUT / "full_step_d" / "k1_full_step_d_metrics.csv"
        self.assertTrue(path.exists(), f"Full Step D metrics CSV missing: {path}")

    def test_step_d_has_6_cases(self):
        """Step D must have 6 cases (D1-D6)."""
        path = K1_OUT / "full_step_d" / "k1_full_step_d_metrics.csv"
        if not path.exists():
            self.skipTest("Step D metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        case_ids = set(r.get("case_id", "") for r in rows)
        self.assertIn("D1_small_push_high", case_ids)
        self.assertIn("D2_medium_push_high", case_ids)
        self.assertIn("D3_small_push_low", case_ids)
        self.assertIn("D4_medium_push_low", case_ids)
        self.assertIn("D5_large_push_high", case_ids)
        self.assertIn("D6_random_push_high", case_ids)
        self.assertEqual(len(case_ids), 6, f"Expected 6 cases, got {len(case_ids)}")

    def test_step_d_no_falls(self):
        """Step D must show no falls."""
        path = K1_OUT / "full_step_d" / "k1_full_step_d_metrics.csv"
        if not path.exists():
            self.skipTest("Step D metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        falls = [r.get("fell", "false") for r in rows]
        self.assertTrue(all(f == "False" or f == "false" or not f for f in falls),
                        "Step D should have no falls")

    def test_step_d_no_wbc_no_hidden_torque(self):
        """Step D must show zero WBC and hidden torque."""
        path = K1_OUT / "full_step_d" / "k1_full_step_d_metrics.csv"
        if not path.exists():
            self.skipTest("Step D metrics not available")
        with open(path) as f:
            rows = list(__import__("csv").DictReader(f))
        wbc_sum = sum(float(r.get("wbc_authority_rows", "0") or "0") for r in rows)
        ht_sum = sum(float(r.get("hidden_torque_max", "0") or "0") for r in rows)
        ov_sum = sum(float(r.get("ownership_violation_max", "0") or "0") for r in rows)
        self.assertEqual(wbc_sum, 0, "Step D should have zero WBC rows")
        self.assertEqual(ht_sum, 0, "Step D should have zero hidden torque")
        self.assertEqual(ov_sum, 0, "Step D should have zero ownership violations")


class TestAnalysisOutputs(unittest.TestCase):
    """Analysis outputs must exist."""

    def test_step_e_comparison_exists(self):
        """Step E comparison CSV must exist."""
        path = ANALYSIS_OUT / "step_e_comparison.csv"
        self.assertTrue(path.exists(), f"Step E comparison missing: {path}")

    def test_step_c_comparison_exists(self):
        """Step C comparison CSV must exist."""
        path = ANALYSIS_OUT / "step_c_comparison.csv"
        self.assertTrue(path.exists(), f"Step C comparison missing: {path}")

    def test_step_d_comparison_exists(self):
        """Step D comparison CSV must exist."""
        path = ANALYSIS_OUT / "step_d_comparison.csv"
        self.assertTrue(path.exists(), f"Step D comparison missing: {path}")

    def test_summary_json_exists(self):
        """Summary JSON must exist with classification."""
        path = ANALYSIS_OUT / "k1_vs_legacy_summary.json"
        self.assertTrue(path.exists(), f"Summary JSON missing: {path}")
        with open(path) as f:
            summary = json.load(f)
        self.assertIn("classification", summary)

    def test_classification_valid(self):
        """Classification must be a valid enum."""
        path = ANALYSIS_OUT / "k1_vs_legacy_summary.json"
        if not path.exists():
            self.skipTest("Summary JSON not available")
        with open(path) as f:
            summary = json.load(f)
        classification = summary.get("classification", "")
        self.assertIn(classification, VALID_CLASSIFICATIONS,
                      f"Invalid classification: {classification}")

    def test_rollback_recommendation_exists(self):
        """Rollback recommendation JSON must exist."""
        path = ANALYSIS_OUT / "rollback_recommendation.json"
        self.assertTrue(path.exists(), f"Rollback recommendation missing: {path}")
        with open(path) as f:
            rb = json.load(f)
        self.assertIn("rollback_recommended", rb)
        self.assertIn("recommended_current_best", rb)
        self.assertIn("action", rb)

    def test_no_stub_assumed_synthetic(self):
        """Verification must confirm no stub/assumed/synthetic rows."""
        path = ANALYSIS_OUT / "k1_vs_legacy_summary.json"
        if not path.exists():
            self.skipTest("Summary JSON not available")
        with open(path) as f:
            summary = json.load(f)
        verification = summary.get("verification", {})
        self.assertTrue(verification.get("no_stub_assumed_synthetic", False))
        self.assertTrue(verification.get("real_simulation_source", False))
        self.assertTrue(verification.get("direct_hip_yaw_telemetry", False))

    def test_notch_telemetry_available(self):
        """Notch telemetry must be available."""
        path = ANALYSIS_OUT / "k1_vs_legacy_summary.json"
        if not path.exists():
            self.skipTest("Summary JSON not available")
        with open(path) as f:
            summary = json.load(f)
        verification = summary.get("verification", {})
        self.assertTrue(verification.get("notch_telemetry_available", False))


class TestReport(unittest.TestCase):
    """Report must exist and contain required content."""

    def test_report_exists(self):
        """Validation report must exist."""
        self.assertTrue(REPORT_PATH.exists(), f"Report missing: {REPORT_PATH}")

    def test_report_contains_classification(self):
        """Report must contain classification."""
        if not REPORT_PATH.exists():
            self.skipTest("Report not available")
        text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
        has_classification = any(
            c in text for c in VALID_CLASSIFICATIONS
        )
        self.assertTrue(has_classification, "Report must contain a valid classification")

    def test_no_claims_sustained_recovery(self):
        """Report must not claim K1 solved sustained posture recovery."""
        if not REPORT_PATH.exists():
            self.skipTest("Report not available")
        text = REPORT_PATH.read_text(encoding="utf-8", errors="replace").lower()
        forbidden_phrases = [
            "sustained posture recovery achieved",
            "wip solved",
            "posture recovery pass",
            "fully solved",
            "2s hold achieved",
            "k1 achieves sustained recovery",
            "posture recovery solved",
        ]
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, text,
                             f"Report should not claim: {phrase}")


class TestLegacyDData(unittest.TestCase):
    """D legacy data must remain accessible."""

    def test_d_step_d_data_exists(self):
        """D Step D metrics must exist for comparison."""
        path = D_OUT / "step_d_standard_metrics.csv"
        self.assertTrue(path.exists(), f"D Step D metrics missing: {path}")

    def test_d_step_e_data_exists(self):
        """D Step E 5000 metrics must exist for comparison."""
        path = D_OUT / "step_e_fixed_height_metrics.D5000.csv"
        self.assertTrue(path.exists(), f"D Step E 5000 metrics missing: {path}")

    def test_d_step_c_data_exists(self):
        """D Step C metrics must exist for comparison."""
        path = D_OUT / "step_c_standard_metrics.csv"
        self.assertTrue(path.exists(), f"D Step C metrics missing: {path}")


if __name__ == "__main__":
    unittest.main()
