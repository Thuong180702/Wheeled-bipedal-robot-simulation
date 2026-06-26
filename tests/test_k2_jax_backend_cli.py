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

    def test_backend_default_is_python(self):
        """No --controller-backend flag defaults to python."""
        result = _run([
            "--height-variant-setup", _height_setup("high_0p480"),
        ])
        assert result.returncode == 0
        # Verify python backend used (no JAX message)
        assert "JAX BACKEND" not in result.stdout


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
