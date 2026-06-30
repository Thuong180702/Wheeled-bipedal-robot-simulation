"""Stage 5: --controller-backend flag and JAX backend integration tests."""

import subprocess
import sys
import pytest

SIMULATE = "scripts/simulate_hierarchical_controller.py"
BASE_ARGS = [
    sys.executable, SIMULATE,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--steps", "50",
]


def _height_setup(variant):
    return f"outputs/physical_target_height_setups_centered/{variant}_setup.json"


def _run(args_extra):
    cmd = list(BASE_ARGS) + args_extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


class TestBackendFlagParses:
    """CLI flag --controller-backend works."""

    def test_help_shows_backend_flag(self):
        result = subprocess.run(
            [sys.executable, SIMULATE, "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert "--controller-backend" in result.stdout
        # Help should mention the default promotion policy
        assert "k2_notch_low_q_v1" in result.stdout or "K2" in result.stdout

    def test_backend_python_parses(self):
        result = _run([
            "--controller-backend", "python",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0

    def test_backend_jax_parses(self):
        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0

    # --- K2 JAX DEFAULT PROMOTION: backend selection tests ---

    def test_k2_profile_defaults_to_jax(self):
        """No --controller-backend + K2 profile → selects JAX."""
        result = _run([
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        assert "JAX BACKEND" in result.stdout
        assert "default for validated K2" in result.stdout

    def test_explicit_python_overrides_k2_default(self):
        """Explicit --controller-backend python + K2 profile → selects Python."""
        result = _run([
            "--controller-backend", "python",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        assert "JAX BACKEND" not in result.stdout
        assert "[BACKEND] Controller backend: python (explicit user override)" in result.stdout

    def test_explicit_jax_with_k2_profile(self):
        """Explicit --controller-backend jax + K2 profile → selects JAX."""
        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        assert "[BACKEND] Controller backend: jax (explicit user override)" in result.stdout

    def test_explicit_both_synced_with_k2_profile(self):
        """Explicit --controller-backend both-synced + K2 profile → selects both-synced."""
        result = _run([
            "--controller-backend", "both-synced",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        assert "[BACKEND] Controller backend: both-synced (explicit user override)" in result.stdout

    def test_non_k2_profile_defaults_to_python(self):
        """No --controller-backend + non-K2 profile (baseline sagittal) → Python fallback."""
        result = subprocess.run(
            [sys.executable, SIMULATE,
             "--controller-mode", "balance-core",
             "--sagittal-controller", "baseline",
             "--steps", "50",
             "--height-variant-setup", _height_setup("high_0p480")],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        assert "JAX BACKEND" not in result.stdout
        assert "default fallback for non-validated profile" in result.stdout

    def test_non_k2_velocity_damped_other_profile_defaults_to_python(self):
        """No --controller-backend + balance-core + velocity-damped but non-K2 profile → Python fallback."""
        result = subprocess.run(
            [sys.executable, SIMULATE,
             "--controller-mode", "balance-core",
             "--sagittal-controller", "velocity-damped",
             "--vd-sagittal-authority-profile", "baseline",
             "--steps", "50",
             "--height-variant-setup", _height_setup("high_0p480")],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        assert "JAX BACKEND" not in result.stdout
        assert "default fallback for non-validated profile" in result.stdout

    def test_invalid_backend_rejected(self):
        """Invalid --controller-backend value is rejected by argparse."""
        result = _run([
            "--controller-backend", "invalid_backend_name",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode != 0


class TestJaxBackendSmoke:
    """JAX backend smoke tests."""

    @pytest.mark.parametrize("variant", ["high_0p480", "low_0p330"])
    def test_jax_smoke_completes(self, variant):
        """JAX backend smoke rollout completes without crash."""
        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup(variant),
        ])
        assert result.returncode == 0, f"JAX smoke failed for {variant}:\n{result.stderr[-500:]}"
        assert "[OK] Completed" in result.stdout, f"No success in output for {variant}"

    def test_jax_no_nan_in_output(self):
        """No NaN or error in JAX output."""
        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert "Traceback" not in result.stdout
        assert "Error" not in result.stdout
        assert "NaN" not in result.stdout

    def test_jax_compile_message(self):
        """JAX backend prints compile time."""
        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert "JAX BACKEND" in result.stdout
        assert "JIT compile time" in result.stdout


class TestPythonBackendUnchanged:
    """Python backend behavior unchanged after Stage 5 changes."""

    def test_python_backend_completes(self):
        result = _run([
            "--controller-backend", "python",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        assert "[OK] Completed" in result.stdout


class TestProfileControllerFlag:
    """--profile-controller works with both backends."""

    def test_profile_python(self):
        result = _run([
            "--controller-backend", "python",
            "--height-variant-setup", _height_setup("high_0p480"),
            "--profile-controller",
        ])
        assert result.returncode == 0
        assert "PROFILE" in result.stdout

    def test_profile_jax(self):
        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup("high_0p480"),
            "--profile-controller",
        ])
        assert result.returncode == 0
        assert "PROFILE" in result.stdout


class TestStage7BenchmarkSmoke:
    """Stage 7: --stage7-benchmark flag and output validation."""

    def test_stage7_flag_parses_and_writes_json(self):
        """--stage7-benchmark flag parses and produces a JSON output file."""
        import json as _json
        from pathlib import Path as _Path

        result = _run([
            "--controller-backend", "python",
            "--height-variant-setup", _height_setup("high_0p480"),
            "--stage7-benchmark",
            "--stage7-benchmark-tag", "cli_test",
            "--stage7-benchmark-warmup-steps", "10",
            "--stage7-benchmark-measured-steps", "40",
            "--steps", "50",
            "--wbc-quiet",
            "--telemetry-decimation", "10",
            "--enable-mode-hip-yaw-divergence",
            "--mode-hip-yaw-div-kp", "10.0",
            "--mode-hip-yaw-div-kd", "0.50",
            "--mode-hip-yaw-div-max-torque", "7.5",
            "--mode-hip-yaw-div-soft-limit-rad", "0.30",
            "--mode-hip-yaw-div-soft-gain", "0.80",
            "--mode-hip-yaw-div-ref-source", "target",
        ])
        assert result.returncode == 0, f"Stage7 benchmark failed:\n{result.stderr[-500:]}"
        assert "STAGE7 BENCHMARK" in result.stdout
        assert "Report saved to" in result.stdout

        # Check JSON file exists and contains required fields
        json_path = _Path("outputs/benchmark/stage7_cli_test_python.json")
        assert json_path.exists(), f"Benchmark JSON not found at {json_path}"
        with open(json_path, encoding="utf-8") as f:
            data = _json.load(f)

        required_top = ["stage", "scenario_tag", "backend", "config", "environment",
                        "compile", "timing_stats_ms", "summary", "validation", "command_line"]
        for key in required_top:
            assert key in data, f"Missing required top-level field: {key}"

        assert data["backend"] == "python"
        assert data["config"]["warmup_steps"] == 10
        assert data["config"]["measured_steps"] > 0
        assert "total_step_s" in data["timing_stats_ms"]

    def test_stage7_benchmark_json_jax_backend(self):
        """Stage7 benchmark produces valid JSON with JAX backend including hot-step timing."""
        import json as _json
        from pathlib import Path as _Path

        result = _run([
            "--controller-backend", "jax",
            "--height-variant-setup", _height_setup("high_0p480"),
            "--stage7-benchmark",
            "--stage7-benchmark-tag", "cli_test_jax",
            "--stage7-benchmark-warmup-steps", "10",
            "--stage7-benchmark-measured-steps", "40",
            "--steps", "50",
            "--wbc-quiet",
            "--telemetry-decimation", "10",
            "--enable-mode-hip-yaw-divergence",
            "--mode-hip-yaw-div-kp", "10.0",
            "--mode-hip-yaw-div-kd", "0.50",
            "--mode-hip-yaw-div-max-torque", "7.5",
            "--mode-hip-yaw-div-soft-limit-rad", "0.30",
            "--mode-hip-yaw-div-soft-gain", "0.80",
            "--mode-hip-yaw-div-ref-source", "target",
        ])
        assert result.returncode == 0, f"Stage7 JAX benchmark failed:\n{result.stderr[-500:]}"
        assert "STAGE7 BENCHMARK" in result.stdout

        json_path = _Path("outputs/benchmark/stage7_cli_test_jax_jax.json")
        assert json_path.exists(), f"JAX benchmark JSON not found at {json_path}"
        with open(json_path, encoding="utf-8") as f:
            data = _json.load(f)

        assert data["backend"] == "jax"
        assert data["compile"]["jit_compile_time_s"] > 0
        assert "jax_pack_input_s" in data["timing_stats_ms"]
        assert "jax_jit_step_s" in data["timing_stats_ms"]
        hot_mean = data["timing_stats_ms"]["jax_jit_step_s"]["mean_ms"]
        assert hot_mean is not None
        assert hot_mean < 100.0, f"JAX hot-step too slow: {hot_mean} ms"

    def test_normal_run_without_benchmark_unchanged(self):
        """Normal run without --stage7-benchmark still works (Python backend)."""
        result = _run([
            "--controller-backend", "python",
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        assert "[OK] Completed" in result.stdout
        # Should NOT contain STAGE7 output
        assert "STAGE7 BENCHMARK" not in result.stdout
