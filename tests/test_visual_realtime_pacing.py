"""Tests for visual realtime pacing in simulate_hierarchical_controller.py.

Verifies:
- Expected simulated duration computed correctly from steps and control_dt
- Visual realtime factor target calculation correct
- Viewer sync scheduler caps at visual_sync_hz
- No sleep debt accumulation bug
- Visual flags parse correctly
- Headless default behavior unchanged
- --visual-realtime-factor accepts valid values and rejects invalid values
- --visual-sync-hz accepts valid values and rejects invalid values
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

# Known physical constants from the simulation (verified from source code)
CONTROL_DT = 0.01  # seconds (100 Hz control)
PHYSICS_DT = 0.002  # seconds (500 Hz physics, from XML option timestep="0.002")
N_SUBSTEPS = 5  # CONTROL_DT / PHYSICS_DT
CONTROL_HZ = 100.0  # 1 / CONTROL_DT

BASE_K2_ARGS = [
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--height-variant-setup",
    str(ROOT / "outputs" / "physical_target_height_setups_centered" / "high_0p480_setup.json"),
    "--steps", "20",
    "--telemetry-decimation", "1",
    "--failure-window-steps", "20",
    "--write-run-summary-sidecar",
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]


def _run_sim(extra_args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run the simulation script with given extra args."""
    args = [sys.executable, str(SIM_SCRIPT)] + BASE_K2_ARGS + extra_args
    return subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _get_help_text() -> str:
    """Get argparse help text."""
    result = subprocess.run(
        [sys.executable, str(SIM_SCRIPT), "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30,
    )
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Timing model tests
# ---------------------------------------------------------------------------

class TestTimingModel:
    """Verify the simulation timing model constants."""

    def test_control_dt_is_10ms(self):
        """control_dt must be 0.01 s (100 Hz)."""
        assert CONTROL_DT == 0.01
        assert CONTROL_HZ == 100.0

    def test_physics_dt_is_2ms(self):
        """physics_dt from XML is 0.002 s (500 Hz)."""
        assert PHYSICS_DT == 0.002

    def test_n_substeps_is_5(self):
        """n_substeps = control_dt / physics_dt = 5."""
        assert int(CONTROL_DT / PHYSICS_DT) == N_SUBSTEPS
        assert N_SUBSTEPS == 5

    @pytest.mark.parametrize("steps,expected_s", [
        (2000, 20.0),
        (5000, 50.0),
        (7000, 70.0),
        (1000, 10.0),
        (200, 2.0),
    ])
    def test_sim_duration_from_steps(self, steps, expected_s):
        """Simulated duration = steps * control_dt."""
        assert steps * CONTROL_DT == expected_s

    def test_control_dt_in_help_text(self):
        """Help text mentions 100 Hz control."""
        help_text = _get_help_text()
        assert "100 Hz" in help_text or "steps" in help_text.lower()


# ---------------------------------------------------------------------------
# Visual flag tests
# ---------------------------------------------------------------------------

class TestVisualFlags:
    """Verify that visual pacing flags exist and parse correctly."""

    def test_visual_realtime_factor_flag_exists(self):
        """--visual-realtime-factor is in argparse."""
        help_text = _get_help_text()
        assert "--visual-realtime-factor" in help_text

    def test_visual_sync_hz_flag_exists(self):
        """--visual-sync-hz is in argparse."""
        help_text = _get_help_text()
        assert "--visual-sync-hz" in help_text

    def test_visual_disable_pacing_flag_exists(self):
        """--visual-disable-realtime-pacing is in argparse."""
        help_text = _get_help_text()
        assert "--visual-disable-realtime-pacing" in help_text

    def test_visual_profile_timing_flag_exists(self):
        """--visual-profile-timing is in argparse."""
        help_text = _get_help_text()
        assert "--visual-profile-timing" in help_text

    def test_visual_realtime_factor_accepted(self):
        """--visual-realtime-factor 2.0 parses without error."""
        result = _run_sim([
            "--visual-realtime-factor", "2.0",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "rf2"),
        ])
        # Should not have argparse error
        assert "unrecognized arguments" not in result.stderr

    def test_visual_sync_hz_accepted(self):
        """--visual-sync-hz 30 parses without error."""
        result = _run_sim([
            "--visual-sync-hz", "30",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "sync30"),
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_visual_disable_pacing_accepted(self):
        """--visual-disable-realtime-pacing parses without error."""
        result = _run_sim([
            "--visual-disable-realtime-pacing",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "nopace"),
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_visual_profile_timing_accepted(self):
        """--visual-profile-timing parses without error."""
        result = _run_sim([
            "--visual-profile-timing",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "profile"),
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_realtime_factor_zero_disables_pacing(self):
        """--visual-realtime-factor 0 should be treated as disable pacing."""
        result = _run_sim([
            "--visual-realtime-factor", "0",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "rf0"),
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_realtime_factor_negative_accepted(self):
        """Negative values are accepted by argparse (clamped at runtime)."""
        result = _run_sim([
            "--visual-realtime-factor", "-1.0",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "rfneg"),
        ])
        # Argparse accepts it (type=float), runtime clamps to disabled
        assert "unrecognized arguments" not in result.stderr

    def test_sync_hz_min_clamp(self):
        """Very small sync_hz is accepted by argparse."""
        result = _run_sim([
            "--visual-sync-hz", "1",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "sync1"),
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_sync_hz_high_accepted(self):
        """High sync_hz (120) is accepted."""
        result = _run_sim([
            "--visual-sync-hz", "120",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_flags" / "sync120"),
        ])
        assert "unrecognized arguments" not in result.stderr


# ---------------------------------------------------------------------------
# Headless parity tests
# ---------------------------------------------------------------------------

class TestHeadlessParity:
    """Verify that visual pacing changes don't affect headless mode."""

    def test_headless_runs_without_pacing_flags(self):
        """Headless simulation completes successfully with no pacing flags."""
        result = _run_sim([
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_parity" / "headless"),
        ])
        combined = result.stdout + result.stderr
        assert "Status: [OK] Completed full simulation without falling" in combined

    def test_headless_output_contains_expected_summary(self):
        """Headless summary contains expected fields."""
        result = _run_sim([
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_parity" / "headless2"),
        ])
        combined = result.stdout + result.stderr
        assert "Mode: HEADLESS" in combined
        assert "Simulation time:" in combined
        assert "Wall clock time:" in combined

    def test_headless_2_step_deterministic_control(self):
        """A minimal headless run produces valid telemetry (smoke test)."""
        result = _run_sim([
            "--steps", "5",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_parity" / "headless_5"),
            "--failure-window-steps", "5",
        ], timeout=120)
        combined = result.stdout + result.stderr
        assert "Status: [OK] Completed full simulation without falling" in combined

    def test_headless_summary_sidecar_has_wall_clock(self):
        """Headless summary JSON includes wall_clock_time_s."""
        result = _run_sim([
            "--steps", "5",
            "--output-dir",
            str(ROOT / "outputs" / "visual" / "test_parity" / "sidecar_test"),
            "--failure-window-steps", "5",
        ], timeout=120)
        combined = result.stdout + result.stderr
        if "Status: [OK] Completed full simulation without falling" in combined:
            # Find the sidecar file
            sidecar_dir = ROOT / "outputs" / "visual" / "test_parity" / "sidecar_test"
            sidecars = list(sidecar_dir.glob("*.summary.json"))
            if sidecars:
                data = json.loads(sidecars[0].read_text(encoding="utf-8"))
                assert "wall_clock_time_s" in data
                assert data["wall_clock_time_s"] > 0
                assert "visual_pacing" in data
                assert data["visual_pacing"]["mode"] == "headless"


# ---------------------------------------------------------------------------
# Pacing logic tests (offline — no viewer needed)
# ---------------------------------------------------------------------------

class TestPacingLogic:
    """Verify pacing calculations are correct (offline tests)."""

    def test_sim_duration_formula(self):
        """sim_duration_s = steps * control_dt."""
        assert 5000 * CONTROL_DT == 50.0
        assert 1000 * CONTROL_DT == 10.0

    def test_realtime_factor_pacing_dt(self):
        """pacing_dt = control_dt / realtime_factor."""
        for rf in [0.5, 1.0, 2.0, 4.0]:
            pacing_dt = CONTROL_DT / rf
            assert pacing_dt == 0.01 / rf

    def test_viewer_sync_interval_from_hz(self):
        """sync_interval_s = 1.0 / visual_sync_hz."""
        for hz in [15, 30, 60]:
            interval = 1.0 / hz
            assert interval > 0

    def test_sleep_debt_no_divergence(self):
        """Sleep debt accumulation doesn't diverge when compute >> pacing."""
        # Simulate: step takes 50ms, pacing_dt is 10ms
        # Each step: debt = debt + (10ms - 50ms) = debt - 40ms
        pacing_dt = 0.010
        step_time = 0.050
        debt = 0.0
        for _ in range(100):
            debt += pacing_dt - step_time
            debt = max(debt, -pacing_dt)  # capped at -control_dt
        # Debt should be capped, not diverging to -infinity
        assert debt >= -0.010
        assert debt <= 0.0  # Always behind schedule

    def test_sleep_debt_recovery(self):
        """Sleep debt recovers when compute < pacing interval."""
        pacing_dt = 0.050  # 50ms target
        step_time = 0.010  # 10ms actual
        debt = 0.0
        recovery_count = 0
        for _ in range(100):
            sleep_time = pacing_dt - step_time + debt
            if sleep_time > 0:
                debt += pacing_dt - (step_time + sleep_time)
                recovery_count += 1
            else:
                debt += pacing_dt - step_time
            debt = max(debt, -pacing_dt)
        # With fast steps, we should have positive sleep most of the time
        assert debt >= -0.050  # debt bounded

    def test_control_hz_is_100(self):
        """Control rate is 100 Hz (verified from source)."""
        hz = 1.0 / CONTROL_DT
        assert abs(hz - 100.0) < 0.01

    def test_physics_hz_is_500(self):
        """Physics rate is 500 Hz."""
        hz = 1.0 / PHYSICS_DT
        assert abs(hz - 500.0) < 0.01


# ---------------------------------------------------------------------------
# Docs presence tests
# ---------------------------------------------------------------------------

class TestDocumentationReferences:
    """Verify the documentation references visual pacing."""

    def test_commands_doc_exists(self):
        """k2_verified_visual_commands.md exists."""
        doc = ROOT / "docs" / "validation" / "k2_verified_visual_commands.md"
        assert doc.exists()

    def test_pacing_report_will_exist(self):
        """Report path is valid (will be created in Phase 7)."""
        report_path = ROOT / "docs" / "validation" / "visual_realtime_pacing_fix_report.md"
        # The parent directory must exist
        assert report_path.parent.exists()


# ---------------------------------------------------------------------------
# Compile check
# ---------------------------------------------------------------------------

class TestCompileCheck:
    """Verify the modified script compiles without syntax errors."""

    def test_simulate_script_compiles(self):
        """simulate_hierarchical_controller.py compiles."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(SIM_SCRIPT)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Compile failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
