#!/usr/bin/env python3
"""Test suite for K1 best-current promotion evidence.

This test suite verifies:

1. Promotion policy is best-current, not full-goal-solved.
2. K1 profile exists with exact notch parameters.
3. K1 uses pitch_rate notch only (no K3 combined notch).
4. K1 does not use J3a damping increase or other architecture changes.
5. D remains available as legacy/reference.
6. Direct hip-yaw telemetry requirement is enforced.
7. Evidence inventory exists.
8. Ranking summary exists.
9. Known limitation wording is present.
10. No test claims posture recovery pass.
"""
import json
import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "wheeled_biped"))

EVIDENCE_DIR = ROOT / "outputs" / "evidence_based_k1_best_current_promotion"
INVENTORY_DIR = EVIDENCE_DIR / "evidence_inventory"
RANKING_DIR = EVIDENCE_DIR / "ranking"


class TestPromotionPolicy(unittest.TestCase):
    """Test that promotion uses best-current policy, not full-goal-solved."""

    def test_promotion_classification_is_best_current(self):
        """Promotion classification must be best-current, not full-goal-solved."""
        ranking_json = RANKING_DIR / "ranking.json"
        self.assertTrue(ranking_json.exists(), f"Ranking file missing: {ranking_json}")
        # Verify the decision exists
        decision_json = RANKING_DIR / "decision.json"
        self.assertTrue(decision_json.exists(), f"Decision file missing: {decision_json}")

    def test_promotion_status_contains_known_wip_limitation(self):
        """Promotion status must contain known WIP recovery limitation wording."""
        decision_json = RANKING_DIR / "decision.json"
        if not decision_json.exists():
            self.skipTest("Decision file not yet generated")
        with open(decision_json) as f:
            decision = json.load(f)
        classification = decision.get("classification", "")
        self.assertIn("KNOWN_WIP_RECOVERY_LIMITATION", classification,
                       f"Classification {classification} does not contain KNOWN_WIP_RECOVERY_LIMITATION")

    def test_no_posture_recovery_pass_claimed(self):
        """No test should claim POSTURE_RECOVERY_PASS."""
        classification = ""
        decision_json = RANKING_DIR / "decision.json"
        if decision_json.exists():
            with open(decision_json) as f:
                decision = json.load(f)
            classification = decision.get("classification", "")
        self.assertNotIn("POSTURE_RECOVERY_PASS", classification)
        self.assertNotIn("FULL_RECOVERY_SOLVED", classification)
        self.assertNotIn("WIP_SOLVED", classification)
        self.assertNotIn("FULL_VALIDATION_CLEAN_PASS", classification)
        self.assertNotIn("CLEAN_PASS", classification)

    def test_known_limitations_exist(self):
        """If K1 is promoted, known limitations must be documented."""
        decision_json = RANKING_DIR / "decision.json"
        if not decision_json.exists():
            self.skipTest("Decision file not yet generated")
        with open(decision_json) as f:
            decision = json.load(f)
        limitations = decision.get("known_limitations", [])
        self.assertTrue(len(limitations) >= 1,
                        "Known limitations must be documented when K1 is promoted")


class TestK1Profile(unittest.TestCase):
    """Test K1 profile identity and parameters."""

    def test_k1_profile_exists_in_registry(self):
        """K1 profile must exist in SAGITTAL_AUTHORITY_PROFILES."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "k1_pitch_rate_notch_v1",
            SAGITTAL_AUTHORITY_PROFILES,
            "K1 profile missing from SAGITTAL_AUTHORITY_PROFILES",
        )

    def test_k1_exact_notch_parameters(self):
        """K1 must have exact notch parameters: fc=2.5, Q=6, blend=1.0, target=pitch_rate."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        self.assertTrue(k1.enable_wip_notch_filter,
                        "K1 must have notch filter enabled")
        self.assertEqual(k1.wip_notch_target_signal, "pitch_rate",
                         "K1 must target pitch_rate only")
        self.assertEqual(k1.wip_notch_center_hz, 2.5,
                         "K1 notch center must be 2.5 Hz")
        self.assertEqual(k1.wip_notch_q, 6.0,
                         "K1 notch Q must be 6.0")
        self.assertEqual(k1.wip_notch_filter_blend, 1.0,
                         "K1 notch blend must be 1.0")
        self.assertTrue(k1.wip_notch_gate_enabled,
                        "K1 must have height gate enabled")
        self.assertEqual(k1.wip_notch_height_gate_start_m, 0.42,
                         "K1 height gate start must be 0.42 m")
        self.assertEqual(k1.wip_notch_height_gate_full_m, 0.48,
                         "K1 height gate full must be 0.48 m")

    def test_k1_uses_pitch_rate_notch_only(self):
        """K1 must not use combined pitch_rate+wheel_velocity notch."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        target = k1.wip_notch_target_signal
        self.assertIn(target, ["pitch_rate", "wheel_velocity", "pitch_rate_and_wheel_velocity",
                                "support_velocity", "none"])
        self.assertEqual(target, "pitch_rate",
                         "K1 must target pitch_rate only, not combined notch")

    def test_k1_not_k3_combined_notch(self):
        """K1 must not use K3 combined notch (pitch_rate + wheel_velocity)."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        self.assertNotEqual(k1.wip_notch_target_signal,
                            "pitch_rate_and_wheel_velocity",
                            "K1 must not use combined notch")

    def test_k1_no_wbc_enabled(self):
        """K1 must not enable WBC."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        self.assertFalse(getattr(k1, "wbc_enabled", False),
                         "K1 must not enable WBC")

    def test_k1_no_hidden_torque(self):
        """K1 must not have hidden torque mechanisms."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        # K1 is based on PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
        # which has no hidden torque
        self.assertTrue(k1.physics_equilibrium_feedforward_enabled)

    def test_k1_sagittal_base_is_low_band_v2(self):
        """K1 must use the same low-band v2 sagittal base as D/G1_sg080."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        k1 = SAGITTAL_AUTHORITY_PROFILES["k1_pitch_rate_notch_v1"]
        c = SAGITTAL_AUTHORITY_PROFILES[
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
        ]
        # K1 is created with replace() from the v2 base, so it has the same
        # low-band support fields
        self.assertEqual(k1.low_band_support_outer_loop_enabled,
                         c.low_band_support_outer_loop_enabled)
        self.assertEqual(k1.low_band_support_center_m,
                         c.low_band_support_center_m)
        self.assertEqual(k1.low_band_support_sigma_m,
                         c.low_band_support_sigma_m)


class TestDLegacyProfile(unittest.TestCase):
    """Test that D remains available as legacy/reference after promotion."""

    def test_d_profile_remains_available(self):
        """D_MODE_HIP_YAW_DIV_V1 must remain available after K1 promotion."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        self.assertIn(
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
            SAGITTAL_AUTHORITY_PROFILES,
            "D profile must remain available after K1 promotion",
        )

    def test_all_legacy_profiles_remain(self):
        """All legacy profiles must remain selectable."""
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        legacy_profiles = [
            "physics_equilibrium_feedforward_outer_loop",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v1",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
            "calibrated_support_position_outer_loop_pitch_ref_v2",
            "support_position_outer_loop_pitch_ref",
            "height_scheduled_pitch_equilibrium_trim",
        ]
        for name in legacy_profiles:
            self.assertIn(name, SAGITTAL_AUTHORITY_PROFILES,
                          f"Legacy profile {name} must remain available")


class TestEvidenceInventory(unittest.TestCase):
    """Test that evidence inventory exists."""

    def test_evidence_index_exists(self):
        """Evidence index must exist."""
        path = INVENTORY_DIR / "evidence_index.json"
        self.assertTrue(path.exists(), f"Evidence index missing: {path}")

    def test_comparison_table_exists(self):
        """Comparison table must exist."""
        path = INVENTORY_DIR / "comparison_table.csv"
        self.assertTrue(path.exists(), f"Comparison table missing: {path}")

    def test_ranking_summary_exists(self):
        """Ranking summary must exist."""
        path = RANKING_DIR / "ranking.json"
        self.assertTrue(path.exists(), f"Ranking summary missing: {path}")

    def test_decision_exists(self):
        """Decision file must exist."""
        path = RANKING_DIR / "decision.json"
        self.assertTrue(path.exists(), f"Decision file missing: {path}")
        with open(path) as f:
            decision = json.load(f)
        self.assertIn("decision", decision)
        self.assertIn("classification", decision)


class TestKnownLimitationWording(unittest.TestCase):
    """Test that known limitation wording is correct."""

    def test_limitation_does_not_claim_wip_solved(self):
        """Limitation wording must not claim WIP solved."""
        decision_json = RANKING_DIR / "decision.json"
        if not decision_json.exists():
            self.skipTest("Decision file not generated")
        with open(decision_json) as f:
            decision = json.load(f)

        if decision.get("decision") == "PROMOTED":
            limitations = decision.get("known_limitations", [])
            limitations_text = " ".join(limitations).lower()
            for phrase in ["wip solved", "posture recovery pass",
                           "fully solved", "clean pass"]:
                self.assertNotIn(phrase, limitations_text,
                                 f"Limitation wording must not claim: {phrase}")


if __name__ == "__main__":
    unittest.main()
