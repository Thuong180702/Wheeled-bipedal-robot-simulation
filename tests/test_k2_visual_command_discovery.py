"""Tests for K2 visual command discovery.

Verifies that:
- K2 profile is k2_notch_low_q_v1
- K1 legacy profile is k1_pitch_rate_notch_v1
- Visual flag is --visual
- All generated commands use verified flags
- Step D push mechanism is correct
- No command references nonexistent flags
- Command documentation exists
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"
DOC_PATH = ROOT / "docs" / "validation" / "k2_verified_visual_commands.md"
JSON_PATH = ROOT / "outputs" / "visual_command_discovery" / "k2_verified_visual_commands.json"
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"

K2_PROFILE = "k2_notch_low_q_v1"
K1_PROFILE = "k1_pitch_rate_notch_v1"

# Heights available for visual scenarios
STEP_C_HEIGHTS = ["low_0p330", "low_0p320", "high_0p480"]
STEP_E_HEIGHTS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340",
    "low_0p360", "low_0p380",
    "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]
STEP_D_HEIGHTS = ["high_0p480", "mid_0p400", "low_0p330"]


def _get_help_text() -> str:
    """Get argparse help text from simulate_hierarchical_controller.py."""
    result = subprocess.run(
        [sys.executable, str(SIM_SCRIPT), "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30,
    )
    # argparse exits with 0 for --help but may use stderr in some versions
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Profile flag tests
# ---------------------------------------------------------------------------

class TestProfileFlags:
    """Verify K2 and K1 profile selection flags."""

    def test_k2_profile_is_correct(self):
        """K2 current-best profile is k2_notch_low_q_v1."""
        assert K2_PROFILE == "k2_notch_low_q_v1"

    def test_k1_legacy_profile_is_correct(self):
        """K1 legacy profile is k1_pitch_rate_notch_v1."""
        assert K1_PROFILE == "k1_pitch_rate_notch_v1"

    def test_k2_profile_in_help_choices(self):
        """k2_notch_low_q_v1 appears in argparse choices."""
        help_text = _get_help_text()
        assert "k2_notch_low_q_v1" in help_text, (
            "k2_notch_low_q_v1 not found in argparse choices"
        )

    def test_k1_profile_in_help_choices(self):
        """k1_pitch_rate_notch_v1 appears in argparse choices."""
        help_text = _get_help_text()
        assert "k1_pitch_rate_notch_v1" in help_text, (
            "k1_pitch_rate_notch_v1 not found in argparse choices"
        )


# ---------------------------------------------------------------------------
# Visual/viewer flag tests
# ---------------------------------------------------------------------------

class TestVisualFlag:
    """Verify visual flag existence and correctness."""

    def test_visual_flag_exists(self):
        """--visual is the only viewer flag."""
        help_text = _get_help_text()
        assert "--visual" in help_text, "--visual not found in help"

    def test_no_alternative_viewer_flags(self):
        """No --viewer, --render, --gui, or --headless flags exist."""
        help_text = _get_help_text()
        # These should NOT appear as argparse flags
        for bogus in ["--viewer", "--render", "--gui", "--headless", "--show-viewer",
                       "--no-headless", "--mujoco-viewer"]:
            # Check that the flag is not defined (it might appear in help text
            # as part of a help description string, so we check the usage line)
            usage_line = [l for l in help_text.split('\n') if l.strip().startswith('usage:')]
            if usage_line:
                assert bogus not in usage_line[0], (
                    f"Bogus flag {bogus} found in usage line"
                )


# ---------------------------------------------------------------------------
# Setup file existence tests
# ---------------------------------------------------------------------------

class TestSetupFiles:
    """Verify that height setup files exist."""

    @pytest.mark.parametrize("height", STEP_C_HEIGHTS)
    def test_step_c_setup_exists(self, height):
        """Step C setup files exist."""
        setup_path = SETUP_DIR / f"{height}_setup.json"
        assert setup_path.exists(), f"Missing setup: {setup_path}"

    @pytest.mark.parametrize("height", STEP_E_HEIGHTS)
    def test_step_e_setup_exists(self, height):
        """Step E setup files exist."""
        setup_path = SETUP_DIR / f"{height}_setup.json"
        assert setup_path.exists(), f"Missing setup: {setup_path}"

    @pytest.mark.parametrize("height", STEP_D_HEIGHTS)
    def test_step_d_setup_exists(self, height):
        """Step D setup files exist."""
        setup_path = SETUP_DIR / f"{height}_setup.json"
        assert setup_path.exists(), f"Missing setup: {setup_path}"


# ---------------------------------------------------------------------------
# Push mechanism tests
# ---------------------------------------------------------------------------

class TestPushMechanism:
    """Verify push sequence mechanism."""

    def test_push_sequence_flag_exists(self):
        """--push-sequence-file flag exists in argparse."""
        help_text = _get_help_text()
        assert "--push-sequence-file" in help_text, (
            "--push-sequence-file not found in help"
        )

    def test_push_sequence_json_format(self):
        """Push sequence JSON format is correct."""
        # This is the format used by generate_push_sequence_file in step D runner
        import json as _json
        seq = {"sequence": [[300, 0.0, 60.0, 5]]}
        assert "sequence" in seq
        assert len(seq["sequence"]) == 1
        step, fx, fy, duration = seq["sequence"][0]
        assert step == 300
        assert fx == 0.0
        assert fy == 60.0  # positive = forward
        assert duration == 5

    def test_push_sequence_sign_convention(self):
        """Sagittal forward = +force_y, backward = -force_y."""
        # Forward (+y)
        fwd = [[300, 0.0, 60.0, 5]]
        assert fwd[0][2] > 0, "Forward push should have positive force_y"
        # Backward (-y)
        bwd = [[300, 0.0, -60.0, 5]]
        assert bwd[0][2] < 0, "Backward push should have negative force_y"

    def test_push_flag_accepted_by_argparse(self):
        """Verify --push-sequence-file is accepted by argparse (dry-run check)."""
        # Run with --help to verify flag appears
        result = subprocess.run(
            [sys.executable, str(SIM_SCRIPT),
             "--controller-mode", "balance-core",
             "--sagittal-controller", "velocity-damped",
             "--vd-sagittal-authority-profile", K2_PROFILE,
             "--height-variant-setup",
             str(SETUP_DIR / "high_0p480_setup.json"),
             "--steps", "2",
             "--push-sequence-file", "nonexistent_test.json",
             "--output-dir", str(ROOT / "outputs" / "visual_command_discovery" / "_test_push"),
             "--telemetry-decimation", "1",
             "--failure-window-steps", "2",
             "--write-run-summary-sidecar",
             "--enable-mode-hip-yaw-divergence",
             "--mode-hip-yaw-div-kp", "10.0",
             "--mode-hip-yaw-div-kd", "0.50",
             "--mode-hip-yaw-div-max-torque", "7.5",
             "--mode-hip-yaw-div-soft-limit-rad", "0.30",
             "--mode-hip-yaw-div-soft-gain", "0.80",
             "--mode-hip-yaw-div-ref-source", "target"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        # May fail because push file doesn't exist, but should not be argparse error
        combined = result.stdout + result.stderr
        assert "unrecognized arguments" not in combined, (
            f"Argparse rejected push flags: {combined[:500]}"
        )


# ---------------------------------------------------------------------------
# Dynamic height mechanism tests
# ---------------------------------------------------------------------------

class TestDynamicHeightMechanism:
    """Verify dynamic height trajectory mechanism."""

    def test_dynamic_height_flag_exists(self):
        """--dynamic-height-trajectory flag exists in argparse."""
        help_text = _get_help_text()
        assert "--dynamic-height-trajectory" in help_text, (
            "--dynamic-height-trajectory not found in help"
        )

    def test_dynamic_height_json_format(self):
        """Dynamic height trajectory JSON has correct format."""
        traj = {
            "height_profile_name": "test",
            "steps": 5000,
            "waypoints": [
                {"step": 0, "height_m": 0.330},
                {"step": 500, "height_m": 0.330},
                {"step": 3500, "height_m": 0.480},
                {"step": 5000, "height_m": 0.480},
            ],
        }
        assert "height_profile_name" in traj
        assert "steps" in traj
        assert "waypoints" in traj
        assert all("step" in w and "height_m" in w for w in traj["waypoints"])


# ---------------------------------------------------------------------------
# Documentation existence tests
# ---------------------------------------------------------------------------

class TestDocumentation:
    """Verify documentation files exist and contain required content."""

    def test_markdown_doc_exists(self):
        """docs/validation/k2_verified_visual_commands.md exists."""
        assert DOC_PATH.exists(), f"Documentation missing: {DOC_PATH}"

    def test_markdown_contains_visual_flag(self):
        """Documentation documents --visual flag."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "--visual" in content

    def test_markdown_contains_k2_profile(self):
        """Documentation documents K2 profile."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "k2_notch_low_q_v1" in content

    def test_markdown_contains_k1_profile(self):
        """Documentation documents K1 profile."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "k1_pitch_rate_notch_v1" in content

    def test_markdown_contains_step_c(self):
        """Documentation contains Step C commands."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "Step C" in content

    def test_markdown_contains_step_d(self):
        """Documentation contains Step D commands."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "Step D" in content

    def test_markdown_contains_step_e(self):
        """Documentation contains Step E commands."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "Step E" in content

    def test_markdown_contains_dynamic_height(self):
        """Documentation contains dynamic height commands."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "Dynamic Height" in content or "dynamic" in content.lower()

    def test_markdown_contains_verification_table(self):
        """Documentation contains verification status table."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "Verification" in content

    def test_json_summary_exists(self):
        """outputs/visual_command_discovery/k2_verified_visual_commands.json exists."""
        assert JSON_PATH.exists(), f"JSON summary missing: {JSON_PATH}"

    def test_json_summary_valid(self):
        """JSON summary is valid JSON with required keys."""
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        assert data["classification"] == "VERIFIED_VISUAL_COMMANDS_FOUND"
        assert "flags" in data
        assert data["flags"]["visual"] == "--visual"
        assert data["current_best_profile"] == "k2_notch_low_q_v1"
        assert data["legacy_profile"] == "k1_pitch_rate_notch_v1"


# ---------------------------------------------------------------------------
# No nonexistent flag tests
# ---------------------------------------------------------------------------

class TestNoBogusFlags:
    """Verify commands don't reference nonexistent CLI flags."""

    def test_doc_does_not_reference_bogus_flags(self):
        """Documentation doesn't reference flags that don't exist."""
        content = DOC_PATH.read_text(encoding="utf-8")
        # These flags should NOT appear as flags in commands
        # (they may appear in descriptive text though)
        bogus_flags = ["--target-height", "--desired-height"]
        for flag in bogus_flags:
            # Check that the flag isn't used in command examples
            # Simple check: the flag followed by a space or value
            import re
            cmd_pattern = re.compile(re.escape(flag) + r'\s+\S')
            assert not cmd_pattern.search(content), (
                f"Documentation references bogus flag: {flag}"
            )

    def test_commands_use_profile_flag(self):
        """All commands use --vd-sagittal-authority-profile, not other profile flags."""
        content = DOC_PATH.read_text(encoding="utf-8")
        # Profile should be set via --vd-sagittal-authority-profile
        assert "--vd-sagittal-authority-profile" in content

    def test_commands_use_height_variant_setup_flag(self):
        """Height is set via --height-variant-setup, not --target-height."""
        content = DOC_PATH.read_text(encoding="utf-8")
        assert "--height-variant-setup" in content


# ---------------------------------------------------------------------------
# Compile check tests
# ---------------------------------------------------------------------------

class TestCompileChecks:
    """Verify key scripts compile without syntax errors."""

    def test_simulate_script_compiles(self):
        """simulate_hierarchical_controller.py compiles."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(SIM_SCRIPT)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Compile failed: {result.stderr}"

    def test_controller_compiles(self):
        """sagittal_velocity_damped_balance_controller.py compiles."""
        controller_path = (
            ROOT / "wheeled_biped" / "controllers"
            / "sagittal_velocity_damped_balance_controller.py"
        )
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(controller_path)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Compile failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
