# Task 1: Local Health Check

**Goal:** Verify that all relevant Python scripts compile without syntax errors.

**Files to compile:**
- scripts/simulate_hierarchical_controller.py
- scripts/run_outer_loop_step_d_push.py
- scripts/run_calibrated_outer_loop_v2_step_d.py
- wheeled_biped/controllers/physics_equilibrium_feedforward.py
- wheeled_biped/controllers/support_outer_loop_low_band.py
- wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
- wheeled_biped/validation/hip_yaw_gate_policy.py
- wheeled_biped/controllers/hip_yaw_metrics.py

**Success criteria:** `python -m py_compile <file>` returns exit code 0 for each file.

**Output:** Create a pytest test file `tests/test_local_health_check.py` that runs the compile checks and asserts success.

**Report file:** `docs/superpowers/plans/2024-06-21-physics-ff-low-band-support-v2-step-d-push-validation-and-promotion_task1_report.md`
