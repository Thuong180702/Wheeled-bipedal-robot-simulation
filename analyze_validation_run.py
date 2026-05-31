#!/usr/bin/env python3
"""Analyze validation run with torque-budget-aware position authority."""

import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python analyze_validation_run.py <telemetry_csv>")
    sys.exit(1)

telemetry_path = sys.argv[1]
df = pd.read_csv(telemetry_path)

print('=' * 80)
print(f'VALIDATION ANALYSIS: {telemetry_path}')
print('=' * 80)
print()

# Basic info
steps = len(df)
duration = df['sim_time_s'].iloc[-1]
terminated = df['terminated'].iloc[-1]

print(f'Steps: {steps}')
print(f'Duration: {duration:.1f} s')
print(f'Terminated: {terminated}')
if terminated:
    print(f'Termination reason: {df["termination_reason"].iloc[-1]}')
print()

# Support position error
sup_pos_err = df['support_position_error_m'].values
max_err = np.abs(sup_pos_err).max()
max_err_step = np.abs(sup_pos_err).argmax()
final_err = np.abs(sup_pos_err[-1])

print('Support Position Error:')
print(f'  Range: [{sup_pos_err.min():.4f}, {sup_pos_err.max():.4f}] m')
print(f'  Max absolute: {max_err:.4f} m at step {max_err_step}')
print(f'  Final: {final_err:.4f} m')
print()

# Posture
pitch = df['pitch_x_rad'].values * 180/np.pi
roll = df['roll_y_rad'].values * 180/np.pi
com_z = df['com_z_m'].values

print('Posture:')
print(f'  Pitch range: [{pitch.min():.2f}, {pitch.max():.2f}] deg')
print(f'  Roll range: [{roll.min():.2f}, {roll.max():.2f}] deg')
print(f'  CoM height range: [{com_z.min():.3f}, {com_z.max():.3f}] m')
print()

# Torque budget analysis
tau_balance = df['tau_balance_before_position'].values
tau_pos_raw = df['tau_position_raw'].values
tau_pos_clip = df['tau_position_clipped'].values
tau_budget_avail = df['tau_position_budget_available'].values
tau_budget_allowed = df['tau_position_budget_allowed'].values
tau_budget_cap = df['tau_position_budget_cap'].values
pitch_reserve = df['pitch_reserve_tau'].values
final_margin = df['final_wheel_torque_margin'].values

# Count saturation
sat_reasons = df['tau_position_saturation_reason'].values
sat_fixed_cap = np.sum(sat_reasons == 'fixed_cap')
sat_physical = np.sum(sat_reasons == 'physical_budget')
sat_pitch_reserve = np.sum(sat_reasons == 'pitch_reserve')
sat_none = np.sum(sat_reasons == 'none')

print('Torque Budget:')
print(f'  tau_balance_before_position range: [{tau_balance.min():.3f}, {tau_balance.max():.3f}] Nm')
print(f'  tau_position_raw range: [{tau_pos_raw.min():.3f}, {tau_pos_raw.max():.3f}] Nm')
print(f'  tau_position_clipped range: [{tau_pos_clip.min():.3f}, {tau_pos_clip.max():.3f}] Nm')
print(f'  tau_position_budget_available range: [{tau_budget_avail.min():.3f}, {tau_budget_avail.max():.3f}] Nm')
print(f'  tau_position_budget_allowed range: [{tau_budget_allowed.min():.3f}, {tau_budget_allowed.max():.3f}] Nm')
print(f'  position_tau_budget_cap: {tau_budget_cap[0]:.1f} Nm')
print(f'  pitch_reserve_tau: {pitch_reserve[0]:.1f} Nm')
print(f'  final_wheel_torque_margin range: [{final_margin.min():.3f}, {final_margin.max():.3f}] Nm')
print()

print('Saturation Reasons:')
print(f'  none: {sat_none}/{steps} ({100*sat_none/steps:.1f}%)')
print(f'  fixed_cap: {sat_fixed_cap}/{steps} ({100*sat_fixed_cap/steps:.1f}%)')
print(f'  physical_budget: {sat_physical}/{steps} ({100*sat_physical/steps:.1f}%)')
print(f'  pitch_reserve: {sat_pitch_reserve}/{steps} ({100*sat_pitch_reserve/steps:.1f}%)')
print()

# Gate compliance
print('=' * 80)
print('GATE COMPLIANCE')
print('=' * 80)

preferred_pass = max_err <= 0.10 and final_err <= 0.05
fallback_pass = max_err <= 0.15 and final_err <= 0.10
hard_min_pass = max_err <= 0.30 and final_err <= 0.10

print(f'Preferred (max <=0.10m, final <=0.05m): {"PASS" if preferred_pass else "FAIL"}')
print(f'Fallback (max <=0.15m, final <=0.10m): {"PASS" if fallback_pass else "FAIL"}')
print(f'Hard minimum (max <=0.30m, final <=0.10m): {"PASS" if hard_min_pass else "FAIL"}')
print()

# Summary
print('=' * 80)
print('SUMMARY')
print('=' * 80)
if terminated:
    print(f'[FAIL] Robot terminated: {df["termination_reason"].iloc[-1]}')
elif preferred_pass:
    print('[PASS] Preferred gate')
elif fallback_pass:
    print('[PASS] Fallback gate')
elif hard_min_pass:
    print('[PASS] Hard minimum gate')
else:
    print(f'[FAIL] All gates failed (max error {max_err:.3f}m > 0.30m)')
print()
