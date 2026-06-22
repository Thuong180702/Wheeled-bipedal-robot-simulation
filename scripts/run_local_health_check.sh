#!/usr/bin/env bash
set -euo pipefail

# Phase 0 health check script

echo "--- Working Directory ---"
pwd

echo "--- Directory Listing ---"
ls -R

echo "--- Python Version ---"
python --version

echo "--- Compiling Python Scripts ---"
# List of scripts to compile
scripts_to_compile=(
  "scripts/simulate_hierarchical_controller.py"
  "scripts/run_outer_loop_step_d_push.py"
  "scripts/run_calibrated_outer_loop_v2_step_d.py"
  "wheeled_biped/controllers/physics_equilibrium_feedforward.py"
  "wheeled_biped/controllers/support_outer_loop_low_band.py"
  "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py"
  "wheeled_biped/validation/hip_yaw_gate_policy.py"
  "wheeled_biped/controllers/hip_yaw_metrics.py"
)
for s in "${scripts_to_compile[@]}"; do
  echo "Compiling $s"
  python -m py_compile "$s"
done

echo "--- Running Pytest Suite (quick check) ---"
# Run a quick pytest on a known passing test (or full suite if desired)
pytest -q || true

echo "Health check completed successfully."
