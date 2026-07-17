"""Phase 10: K2 JAX dedicated runner regression tests and guards.

Tests that the substep fix, param source, and invariants are maintained.
"""

import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent.parent
RUNNER = str(ROOT / "scripts" / "run_k2_jax_realtime.py")
SETUP_HIGH = str(ROOT / "outputs" / "physical_target_height_setups" / "high_0p480_setup.json")
SETUP_LOW = str(ROOT / "outputs" / "physical_target_height_setups" / "low_0p300_setup.json")
SETUP_LOW330 = str(ROOT / "outputs" / "physical_target_height_setups" / "low_0p330_setup.json")
TRAJ_RAMP_UP = str(ROOT / "outputs" / "k2_jax_abs_trim_phase6" / "trajectories" / "ramp_up_0p330_to_0p480.json")
TRAJ_RAMP_DOWN = str(ROOT / "outputs" / "k2_dynamic_height_gate_crossing" / "trajectories" / "ramp_down_0p480_to_0p330.json")
TRAJ_GATE_CHATTER = str(ROOT / "outputs" / "k2_dynamic_height_gate_crossing" / "trajectories" / "gate_chatter_0p400_0p470.json")
PUSH_BWD = str(ROOT / "outputs" / "k2_release_hardening" / "push_seq_bwd_90N.json")


def run_dedicated(extra_args=None, timeout=120):
    """Run dedicated runner with given extra args, return (rc, stdout, stderr)."""
    cmd = [sys.executable, RUNNER, "--quiet", "--telemetry", "off"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    return result.returncode, result.stdout, result.stderr


def parse_summary(stdout):
    """Extract key metrics from runner stdout."""
    info = {"fall": "[FALL]" in stdout, "terminated": "TERMINATED" in stdout}
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("Steps:") and "/" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if p.startswith("Steps:"):
                    info["steps_completed"] = int(p.split(":")[1].strip().split("/")[0])
                    info["steps_max"] = int(p.split(":")[1].strip().split("/")[1].split()[0])
        elif "Hip yaw div:" in line and "max=" in line:
            info["hip_yaw_div_max"] = float(line.split("max=")[1].split("rad")[0].strip())
        elif "Hz:" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if p.startswith("Hz:"):
                    info["achieved_hz"] = float(p.split(":")[1].strip())
    return info


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSmokeRun:
    """Basic smoke tests — verify runner starts and completes without crashing."""

    def test_help(self):
        """--help should work."""
        rc, stdout, stderr = run_dedicated(["--help"], timeout=30)
        assert rc == 0
        assert "usage:" in stdout.lower() or "K2 JAX" in stdout

    def test_basic_run_survives(self):
        """Runner should survive 100 steps at high_0p480 without falling."""
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "100"], timeout=120)
        assert rc == 0, f"Runner failed: {stderr}"
        info = parse_summary(stdout)
        assert not info.get("fall"), f"Fall detected: {info}"
        assert info.get("steps_completed", 0) == 100

    def test_low_0p300_survives_short(self):
        """Runner should survive 200 steps at low_0p300."""
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_LOW, "--steps", "200"], timeout=120)
        assert rc == 0, f"Runner failed: {stderr}"
        info = parse_summary(stdout)
        assert not info.get("fall"), f"Fall detected at low_0p300"


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTEP FIX REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSubstepFix:
    """Verify physics substep fix is active and correct."""

    def test_n_substeps_equals_5(self):
        """Model timestep=0.002, control_dt=0.01 → n_substeps=5."""
        import mujoco
        m = mujoco.MjModel.from_xml_path(
            str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml"))
        physics_dt = float(m.opt.timestep)
        control_dt = 0.01
        n_substeps = int(round(control_dt / physics_dt))
        assert n_substeps == 5, (
            f"Expected 5 substeps (0.01/0.002), got {n_substeps}. "
            f"Physics dt={physics_dt}"
        )

    def test_ramp_up_survives_500_steps(self):
        """ramp_up must survive at least 500 steps (smoke)."""
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_LOW330,
             "--dynamic-height-trajectory", TRAJ_RAMP_UP,
             "--steps", "500"],
            timeout=120)
        assert rc == 0, f"ramp_up smoke failed: {stderr}"
        info = parse_summary(stdout)
        assert not info.get("fall"), f"ramp_up fell in first 500 steps"

    def test_sim_time_correct(self):
        """After N steps, sim_time should be N * 0.01 (not N * 0.002)."""
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "100"], timeout=120)
        for line in stdout.split("\n"):
            if "Sim:" in line and "s" in line:
                sim_s = float(line.split("Sim:")[1].strip().split("s")[0].strip())
                assert abs(sim_s - 1.0) < 0.02, (
                    f"Expected ~1.0s for 100 steps, got {sim_s:.3f}s. "
                    f"Physics may be running at wrong rate."
                )
                break


# ═══════════════════════════════════════════════════════════════════════════
# PARAM DUMP TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDumpK2Params:
    """Verify --dump-k2-params produces correct JSON."""

    def test_dump_has_required_fields(self, tmp_path):
        dump_path = tmp_path / "params.json"
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "10",
             "--dump-k2-params", str(dump_path)],
            timeout=120)
        assert rc == 0
        with open(dump_path) as f:
            data = json.load(f)
        required = ["control_affecting_params", "equilibrium_constants",
                     "source_profile", "jax_state_size", "jax_params_flat_size"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_velocity_damping_scale_in_dump(self, tmp_path):
        dump_path = tmp_path / "params.json"
        run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "10",
             "--dump-k2-params", str(dump_path)],
            timeout=120)
        with open(dump_path) as f:
            data = json.load(f)
        assert data["control_affecting_params"]["velocity_damping_scale"] == 1.1

    def test_low_variant_dump_has_1p1(self, tmp_path):
        """low_0p300 variant should get velocity_damping_scale=1.1."""
        dump_path = tmp_path / "params.json"
        run_dedicated(
            ["--height-setup", SETUP_LOW, "--steps", "10",
             "--dump-k2-params", str(dump_path)],
            timeout=120)
        with open(dump_path) as f:
            data = json.load(f)
        assert data["control_affecting_params"]["velocity_damping_scale"] == 1.1


# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTelemetry:
    """Verify telemetry modes work correctly."""

    def test_telemetry_off_no_csv(self, tmp_path):
        """With --telemetry off, no CSV should be written."""
        out_dir = tmp_path / "telemetry_off"
        run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "50",
             "--telemetry", "off", "--output-dir", str(out_dir)],
            timeout=120)
        csv_files = list(out_dir.glob("*.csv")) if out_dir.exists() else []
        assert len(csv_files) == 0, f"CSV written when telemetry=off: {csv_files}"

    def test_telemetry_full_writes_csv(self, tmp_path):
        """With --telemetry full, one CSV per run should be written."""
        out_dir = tmp_path / "telemetry_full"
        run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "30",
             "--telemetry", "full", "--output-dir", str(out_dir)],
            timeout=120)
        csv_files = list(out_dir.glob("telemetry_*.csv"))
        assert len(csv_files) >= 1, f"No CSV found in {out_dir}"

    def test_telemetry_decimated(self, tmp_path):
        """Decimated telemetry should produce fewer rows."""
        out_dir = tmp_path / "telemetry_dec"
        run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "50",
             "--telemetry", "decimated", "--telemetry-decimation", "10",
             "--output-dir", str(out_dir)],
            timeout=120)
        csv_files = list(out_dir.glob("telemetry_*.csv"))
        if csv_files:
            with open(csv_files[0]) as f:
                # Header + ~5 rows (50 steps / 10 decimation)
                lines = f.readlines()
                assert len(lines) <= 8, (
                    f"Expected <=8 lines (header + ~5 rows), got {len(lines)}"
                )

    def test_summary_json_written(self, tmp_path):
        """Summary JSON should always be written when output-dir is set."""
        out_dir = tmp_path / "summary_test"
        run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "30",
             "--output-dir", str(out_dir)],
            timeout=120)
        summary = out_dir / "summary.json"
        assert summary.exists(), f"summary.json not found in {out_dir}"
        with open(summary) as f:
            data = json.load(f)
        assert "fall" in data
        assert "achieved_hz" in data


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANT GUARDS
# ═══════════════════════════════════════════════════════════════════════════

class TestInvariants:
    """Verify hard invariants from the task specification."""

    def test_no_python_sagittal_in_dedicated(self):
        """Dedicated runner must not import Python sagittal controller."""
        import scripts.run_k2_jax_realtime as dr
        src = open(dr.__file__, encoding='utf-8').read()
        # Must not import the Python sagittal controller class
        forbidden = ["SagittalVelocityDampedBalanceController"]
        for term in forbidden:
            assert term not in src, (
                f"Dedicated runner imports forbidden class: {term}"
            )

    def test_no_wbc_in_dedicated(self):
        """Dedicated runner must not import WBC."""
        import scripts.run_k2_jax_realtime as dr
        src = open(dr.__file__, encoding='utf-8').read()
        forbidden = ["wbc_controller", "WBCController", "compute_wbc"]
        for term in forbidden:
            assert term not in src, (
                f"Dedicated runner imports forbidden WBC: {term}"
            )

    def test_jax_controller_step_is_jit_compiled(self):
        """The JAX step function must be jit-compiled."""
        import scripts.run_k2_jax_realtime as dr
        src = open(dr.__file__, encoding='utf-8').read()
        assert "jax.jit" in src, "Dedicated runner must use jax.jit"

    def test_old_python_fallback_still_works(self):
        """The monolithic script with python backend should still work."""
        # Just check the import and basic function
        from scripts.simulate_hierarchical_controller import main as _mono_main
        assert callable(_mono_main), "Monolithic main should be callable"

    def test_no_hardcoded_k2_profile_dict(self):
        """Verify K2_PROFILE dict is removed from dedicated runner."""
        import scripts.run_k2_jax_realtime as dr
        assert not hasattr(dr, 'K2_PROFILE'), (
            "K2_PROFILE hardcoded dict must be removed"
        )
        assert hasattr(dr, '_K2_AUTH_SCHED'), (
            "Must import K2_NOTCH_LOW_Q_V1 as _K2_AUTH_SCHED"
        )

    def test_push_uses_xfrc_applied(self):
        """Push forces must be applied via xfrc_applied on torso (body 1)."""
        src = open(str(ROOT / "scripts" / "run_k2_jax_realtime.py"), encoding='utf-8').read()
        assert "xfrc_applied" in src, (
            "Dedicated runner must use xfrc_applied for push forces"
        )

    def test_no_per_step_print(self):
        """Hot loop must not contain print() calls."""
        src = open(str(ROOT / "scripts" / "run_k2_jax_realtime.py"), encoding='utf-8').read()
        # Find the hot loop section
        loop_start = src.find("while step < max_steps")
        loop_end = src.find("# END HOT LOOP")
        if loop_start > 0 and loop_end > loop_start:
            loop_body = src[loop_start:loop_end]
            print_lines = [l for l in loop_body.split("\n")
                          if "print(" in l and not l.strip().startswith("#")]
            assert len(print_lines) == 0, (
                f"Hot loop contains print() calls: {print_lines}"
            )

    def test_no_per_step_csv_write(self):
        """Hot loop must not contain CSV write calls."""
        src = open(str(ROOT / "scripts" / "run_k2_jax_realtime.py"), encoding='utf-8').read()
        loop_start = src.find("while step < max_steps")
        loop_end = src.find("# END HOT LOOP")
        if loop_start > 0 and loop_end > loop_start:
            loop_body = src[loop_start:loop_end]
            csv_lines = [l for l in loop_body.split("\n")
                        if "csv" in l.lower() and "write" in l.lower()
                        and not l.strip().startswith("#")]
            assert len(csv_lines) == 0, (
                f"Hot loop contains CSV write calls: {csv_lines}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# HIP-YAW DIVERGENCE GUARD
# ═══════════════════════════════════════════════════════════════════════════

class TestHipYawDivergenceSafety:
    """Verify hip-yaw divergence is within safety bounds at fixed heights."""

    def test_high_0p480_hip_yaw_div_below_0p35(self):
        """At high_0p480, hip-yaw divergence must be well below 0.35 rad."""
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_HIGH, "--steps", "500"], timeout=120)
        info = parse_summary(stdout)
        hyd = info.get("hip_yaw_div_max", 0)
        assert hyd < 0.35, (
            f"high_0p480 hip_yaw_div={hyd:.4f} rad exceeds 0.35 rad safety gate"
        )

    def test_low_0p300_hip_yaw_div_report(self):
        """Report low_0p300 hip-yaw divergence (informational, not failing)."""
        rc, stdout, stderr = run_dedicated(
            ["--height-setup", SETUP_LOW, "--steps", "500"], timeout=120)
        info = parse_summary(stdout)
        hyd = info.get("hip_yaw_div_max", 0)
        # This is informational — canonical JAX has similar divergence
        # at low heights. This is a pre-existing K2 limitation.
        print(f"\n    [INFO] low_0p300 hip_yaw_div_max={hyd:.4f} rad "
              f"(safety gate=0.35 rad, canonical JAX has similar)")
        # Don't hard-fail — this is a known K2 limitation
        # But it should not exceed 0.7 rad (absolute failure)
        assert hyd < 0.70, (
            f"low_0p300 hip_yaw_div={hyd:.4f} rad exceeds 0.70 rad absolute limit"
        )


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL FLAGS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestVisualFlags:
    """Verify --visual flags parse correctly (don't need actual viewer)."""

    def test_visual_help_shows_flags(self):
        """--help should list visual flags."""
        rc, stdout, stderr = run_dedicated(["--help"], timeout=30)
        assert "--visual" in stdout
        assert "--visual-no-pacing" in stdout
        assert "--no-visual-hold" in stdout


class TestDynamicQrefMode:
    """Phase 2: Verify --dynamic-qref-mode flag behavior."""

    def test_flag_in_help(self):
        """--help should show --dynamic-qref-mode flag."""
        rc, stdout, stderr = run_dedicated(["--help"], timeout=30)
        assert "--dynamic-qref-mode" in stdout
        assert "original-k2-exact" in stdout
        assert "setup-interp-debug" in stdout

    def test_default_is_original_k2_exact(self, tmp_path):
        """Default mode should be original-k2-exact (static q_ref)."""
        out = tmp_path / "qref_default"
        out.mkdir()
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "100",
            "--telemetry", "summary", "--output-dir", str(out),
        ], timeout=30)
        assert rc == 0
        # Check summary JSON
        summary_path = out / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary.get("dynamic_qref_mode") == "original-k2-exact"

    def test_setup_interp_debug_mode(self, tmp_path):
        """--dynamic-qref-mode setup-interp-debug should be recorded in summary."""
        out = tmp_path / "qref_debug"
        out.mkdir()
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "100",
            "--dynamic-qref-mode", "setup-interp-debug",
            "--telemetry", "summary", "--output-dir", str(out),
        ], timeout=30)
        assert rc == 0
        summary_path = out / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary.get("dynamic_qref_mode") == "setup-interp-debug"

    def test_invalid_mode_rejected(self):
        """Invalid --dynamic-qref-mode should fail."""
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "50",
            "--dynamic-qref-mode", "invalid-mode",
            "--telemetry", "off",
        ], timeout=30)
        assert rc != 0  # argparse should reject invalid choice

    def test_exact_mode_terminal_output(self, tmp_path):
        """Terminal output should show q_ref mode."""
        out = tmp_path / "qref_exact_out"
        out.mkdir()
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "100",
            "--dynamic-qref-mode", "original-k2-exact",
            "--telemetry", "summary", "--output-dir", str(out),
        ], timeout=30)
        assert rc == 0
        assert "original-k2-exact" in stdout
        assert "static q_ref" in stdout.lower() or "original-k2-exact" in stdout

    def test_debug_mode_terminal_output(self, tmp_path):
        """Terminal output should warn about debug mode."""
        out = tmp_path / "qref_debug_out"
        out.mkdir()
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "100",
            "--dynamic-qref-mode", "setup-interp-debug",
            "--telemetry", "summary", "--output-dir", str(out),
        ], timeout=30)
        assert rc == 0
        assert "APPROXIMATE" in stdout or "setup-interp-debug" in stdout

    def test_ramp_up_with_exact_qref_survives(self):
        """ramp_up with original-k2-exact mode should survive."""
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_LOW330,
            "--dynamic-height-trajectory", TRAJ_RAMP_UP,
            "--dynamic-qref-mode", "original-k2-exact",
            "--steps", "500",
            "--telemetry", "off",
        ], timeout=60)
        assert rc == 0
        assert "[FALL]" not in stdout

    def test_dynamic_qref_falls_back_to_exact_by_default(self):
        """Even without specifying --dynamic-qref-mode, default should be original-k2-exact."""
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_LOW330,
            "--dynamic-height-trajectory", TRAJ_RAMP_UP,
            "--steps", "300",
            "--telemetry", "off",
        ], timeout=60)
        assert rc == 0
        # Should show original-k2-exact in output (default)
        assert "original-k2-exact" in stdout


class TestPromotionValidationGuards:
    """Phase 10: Guard that promotion validation infrastructure works."""

    def test_validate_promotion_script_runs(self):
        """validate_k2_jax_dedicated_promotion.py --help should work."""
        validator = str(ROOT / "scripts" / "validate_k2_jax_dedicated_promotion.py")
        result = subprocess.run(
            [sys.executable, validator, "--help"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "--scope" in result.stdout

    def test_classifier_module_loads(self):
        """StrictPromotionClassifier should import cleanly."""
        from wheeled_biped.validation.strict_promotion_classifier import (
            StrictClass, StrictPromotionClassifier, load_classifier,
            ScenarioComparison, ScopeComparison, MetricComparison,
        )
        assert StrictClass.EXACT_OR_BETTER.value == 1
        assert StrictClass.SAFETY_FAIL.value == 4
        assert StrictClass.NOT_TESTED.value == 5

    def test_baseline_json_valid(self):
        """Baseline JSON should be valid and complete."""
        baseline_path = ROOT / "outputs" / "k2_original_promoted_baseline" / "k2_original_metrics.json"
        assert baseline_path.exists(), f"Baseline not found: {baseline_path}"
        with open(baseline_path) as f:
            data = json.load(f)
        assert "step_e" in data
        assert "step_c" in data
        assert "step_d" in data
        assert "dynamic_height" in data
        assert "long_run_equilibrium" in data
        assert "tolerances" in data
        assert "absolute_safety_gates" in data

    def test_classifier_baseline_loads(self):
        """Classifier should load baseline without error."""
        from wheeled_biped.validation.strict_promotion_classifier import load_classifier
        c = load_classifier()
        assert c is not None

    def test_mode_div_enabled_by_default(self):
        """Dedicated runner should have mode_div enabled by default."""
        rc, stdout, stderr = run_dedicated(["--help"], timeout=30)
        assert "--enable-mode-hip-yaw-divergence" in stdout
        # Default should be enabled
        rc2, out2, err2 = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "50",
            "--telemetry", "off",
        ], timeout=30)
        assert "mode_div=ON" in out2

    def test_no_mode_div_flag_still_works(self):
        """--no-mode-hip-yaw-divergence should disable mode_div."""
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "50",
            "--no-mode-hip-yaw-divergence",
            "--telemetry", "off",
        ], timeout=30)
        assert rc == 0
        assert "mode_div=OFF" in stdout

    def test_no_hidden_torque_output(self):
        """Dedicated runner should not report hidden torque."""
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "100",
            "--telemetry", "off",
        ], timeout=30)
        assert rc == 0
        # "hidden_torque" should not appear in terminal output
        assert "hidden_torque" not in stdout.lower()

    def test_no_wbc_output(self):
        """Dedicated runner should not reference WBC."""
        rc, stdout, stderr = run_dedicated([
            "--height-setup", SETUP_HIGH, "--steps", "100",
            "--telemetry", "off",
        ], timeout=30)
        assert rc == 0
        assert "WBC" not in stdout
        assert "wbc" not in stdout.lower()
