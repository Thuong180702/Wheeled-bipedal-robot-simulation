"""Analyze HY-FF smoke test results to verify integration bug is fixed."""

import pandas as pd
from pathlib import Path

# Find latest telemetry
sim_dir = Path("outputs/hierarchical_controller_sim")
telemetry_files = sorted(sim_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)

if not telemetry_files:
    print("ERROR: No telemetry files found")
    exit(1)

latest = telemetry_files[-1]
print(f"Analyzing: {latest.name}\n")

df = pd.read_csv(latest)

# Check HY-FF activation
print("=" * 80)
print("HY-FF ACTIVATION STATUS")
print("=" * 80)
print(f"hip_yaw_comp_active: {df['hip_yaw_comp_active'].any()}")
print(f"hip_yaw_comp_k_support: {df['hip_yaw_comp_k_support'].max():.1f}")
print()

# Check height gate (CRITICAL FIX VERIFICATION)
print("=" * 80)
print("HEIGHT GATE STATUS (BUG FIX VERIFICATION)")
print("=" * 80)
gate_min = df['hip_yaw_comp_height_gate'].min()
gate_max = df['hip_yaw_comp_height_gate'].max()
gate_mean = df['hip_yaw_comp_height_gate'].mean()
print(f"hip_yaw_comp_height_gate range: [{gate_min:.3f}, {gate_max:.3f}]")
print(f"hip_yaw_comp_height_gate mean: {gate_mean:.3f}")

if gate_max > 0.9:
    print("[OK] HEIGHT GATE ACTIVATED (bug fixed!)")
else:
    print("[FAIL] HEIGHT GATE STILL ZERO (bug remains)")
print()

# Check support error
print("=" * 80)
print("SUPPORT ERROR STATUS")
print("=" * 80)
support_err_min = df['hip_yaw_comp_support_error_m'].min()
support_err_max = df['hip_yaw_comp_support_error_m'].max()
support_err_mean = df['hip_yaw_comp_support_error_m'].mean()
print(f"hip_yaw_comp_support_error_m range: [{support_err_min:.4f}, {support_err_max:.4f}]")
print(f"hip_yaw_comp_support_error_m mean: {support_err_mean:.4f}")

if support_err_max > 0.01:
    print("[OK] SUPPORT ERROR AVAILABLE (bug fixed!)")
else:
    print("[FAIL] SUPPORT ERROR STILL ZERO (bug remains)")
print()

# Check compensation torque
print("=" * 80)
print("COMPENSATION TORQUE STATUS")
print("=" * 80)
tau_left_min = df['hip_yaw_comp_tau_left'].min()
tau_left_max = df['hip_yaw_comp_tau_left'].max()
tau_right_min = df['hip_yaw_comp_tau_right'].min()
tau_right_max = df['hip_yaw_comp_tau_right'].max()
print(f"hip_yaw_comp_tau_left range: [{tau_left_min:.4f}, {tau_left_max:.4f}]")
print(f"hip_yaw_comp_tau_right range: [{tau_right_min:.4f}, {tau_right_max:.4f}]")

if abs(tau_left_max) > 0.01 or abs(tau_left_min) > 0.01:
    print("[OK] COMPENSATION TORQUE NONZERO (bug fixed!)")
else:
    print("[FAIL] COMPENSATION TORQUE STILL ZERO (bug remains)")
print()

# Check debug telemetry
print("=" * 80)
print("DEBUG TELEMETRY")
print("=" * 80)
print(f"hy_ff_height_passed_to_shape range: [{df['hy_ff_height_passed_to_shape'].min():.3f}, {df['hy_ff_height_passed_to_shape'].max():.3f}]")
print(f"hy_ff_support_error_passed_to_shape range: [{df['hy_ff_support_error_passed_to_shape'].min():.4f}, {df['hy_ff_support_error_passed_to_shape'].max():.4f}]")
print(f"hy_ff_support_error_from_sagittal range: [{df['hy_ff_support_error_from_sagittal'].min():.4f}, {df['hy_ff_support_error_from_sagittal'].max():.4f}]")
print(f"hy_ff_prev_support_error range: [{df['hy_ff_prev_support_error'].min():.4f}, {df['hy_ff_prev_support_error'].max():.4f}]")
print(f"hy_ff_setup_target_com_z_m: {df['hy_ff_setup_target_com_z_m'].mean():.3f} m")
print(f"hy_ff_setup_achieved_com_z_m: {df['hy_ff_setup_achieved_com_z_m'].mean():.3f} m")
print(f"hy_ff_root_z_m range: [{df['hy_ff_root_z_m'].min():.3f}, {df['hy_ff_root_z_m'].max():.3f}]")
print(f"hy_ff_current_com_z_m range: [{df['hy_ff_current_com_z_m'].min():.3f}, {df['hy_ff_current_com_z_m'].max():.3f}]")
print()

# Final verdict
print("=" * 80)
print("INTEGRATION BUG FIX VERIFICATION")
print("=" * 80)

bug_fixed = (gate_max > 0.9) and (support_err_max > 0.01) and (abs(tau_left_max) > 0.01 or abs(tau_left_min) > 0.01)

if bug_fixed:
    print("[OK][OK][OK] INTEGRATION BUG FIXED [OK][OK][OK]")
    print("  - Height gate activates at low_0p300")
    print("  - Support error reaches shape controller")
    print("  - Compensation torque is applied")
    print()
    print("READY FOR PHASE 5 RE-EVALUATION")
else:
    print("[FAIL][FAIL][FAIL] INTEGRATION BUG REMAINS [FAIL][FAIL][FAIL]")
    print("  - Further debugging required")

print("=" * 80)
