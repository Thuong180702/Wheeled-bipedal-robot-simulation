import subprocess, sys

def test_all_scripts_compile():
    scripts = [
        "scripts/simulate_hierarchical_controller.py",
        "scripts/run_outer_loop_step_d_push.py",
        "scripts/run_calibrated_outer_loop_v2_step_d.py",
        "wheeled_biped/controllers/physics_equilibrium_feedforward.py",
        "wheeled_biped/controllers/support_outer_loop_low_band.py",
        "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
        "wheeled_biped/validation/hip_yaw_gate_policy.py",
        "wheeled_biped/controllers/hip_yaw_metrics.py",
    ]
    for s in scripts:
        result = subprocess.run([sys.executable, "-m", "py_compile", s], capture_output=True, text=True)
        assert result.returncode == 0, f"Compilation failed for {s}: {result.stderr}"