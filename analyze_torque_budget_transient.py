#!/usr/bin/env python3
"""Analyze torque budget during transient window to confirm position authority is the limiter."""

import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 4:
    print("Usage: python analyze_torque_budget_transient.py <telemetry_csv> <start_step> <end_step>")
    sys.exit(1)

telemetry_path = sys.argv[1]
start_step = int(sys.argv[2])
end_step = int(sys.argv[3])

df = pd.read_csv(telemetry_path)

# Extract window
window = df.iloc[start_step:end_step+1]

print('=' * 80)
print(f'TORQUE BUDGET ANALYSIS: steps {start_step}-{end_step}')
print('=' * 80)
print()

# Extract torque components from velocity-damped controller
# These are the per-term contributions logged by the controller
tau_pitch = window['sagittal_term_pitch'].values
tau_pitch_rate = window['sagittal_term_pitch_rate'].values
tau_sagittal_velocity = window['stage2c_term_com_vy'].values  # sagittal velocity damping
tau_wheel_velocity_left = window['sagittal_term_wheel_vel_left'].values
tau_wheel_velocity_right = window['sagittal_term_wheel_vel_right'].values
tau_support_velocity = window['tau_support_velocity'].values
tau_position_raw = window['tau_position_raw'].values
tau_position_clipped = window['tau_position_clipped'].values

# Final wheel torques
tau_left = window['tau_total_clipped_l_wheel'].values
tau_right = window['tau_total_clipped_r_wheel'].values

# Compute balance torque before position
tau_balance = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    0.5 * (tau_wheel_velocity_left + tau_wheel_velocity_right) +
    tau_support_velocity
)

# Position authority analysis
pos_saturated = np.sum(np.abs(tau_position_raw) > np.abs(tau_position_clipped) + 0.01)
pos_sat_pct = 100 * pos_saturated / len(window)
authority_deficit = np.abs(tau_position_raw).max() - np.abs(tau_position_clipped).max()

# Final wheel torque
final_wheel_torque_max = max(np.abs(tau_left).max(), np.abs(tau_right).max())
max_tau_wheel = 10.0  # From config

print('BALANCE TORQUE (before position):')
print(f'  tau_pitch range: [{tau_pitch.min():.3f}, {tau_pitch.max():.3f}] Nm')
print(f'  tau_pitch_rate range: [{tau_pitch_rate.min():.3f}, {tau_pitch_rate.max():.3f}] Nm')
print(f'  tau_sagittal_velocity range: [{tau_sagittal_velocity.min():.3f}, {tau_sagittal_velocity.max():.3f}] Nm')
print(f'  tau_wheel_velocity (mean) range: [{0.5*(tau_wheel_velocity_left+tau_wheel_velocity_right).min():.3f}, {0.5*(tau_wheel_velocity_left+tau_wheel_velocity_right).max():.3f}] Nm')
print(f'  tau_support_velocity range: [{tau_support_velocity.min():.3f}, {tau_support_velocity.max():.3f}] Nm')
print(f'  tau_balance range: [{tau_balance.min():.3f}, {tau_balance.max():.3f}] Nm')
print(f'  tau_balance RMS: {np.sqrt(np.mean(tau_balance**2)):.3f} Nm')
print()

print('POSITION AUTHORITY:')
print(f'  tau_position_raw range: [{tau_position_raw.min():.3f}, {tau_position_raw.max():.3f}] Nm')
print(f'  tau_position_clipped: {tau_position_clipped[0]:.3f} Nm (constant)')
print(f'  Position saturation: {pos_saturated}/{len(window)} steps ({pos_sat_pct:.1f}%)')
print(f'  Authority deficit: {authority_deficit:.3f} Nm')
print()

print('FINAL WHEEL TORQUE:')
print(f'  tau_left range: [{tau_left.min():.3f}, {tau_left.max():.3f}] Nm')
print(f'  tau_right range: [{tau_right.min():.3f}, {tau_right.max():.3f}] Nm')
print(f'  Final wheel torque max: {final_wheel_torque_max:.3f} Nm')
print(f'  Physical limit: {max_tau_wheel:.1f} Nm')
print(f'  Margin: {max_tau_wheel - final_wheel_torque_max:.3f} Nm')
print(f'  Final wheel saturated: {"YES" if final_wheel_torque_max >= max_tau_wheel * 0.99 else "NO"}')
print()

print('=' * 80)
print('DIAGNOSIS')
print('=' * 80)

if final_wheel_torque_max >= max_tau_wheel * 0.99:
    print('[LIMITER] Final wheel torque is saturated')
    print('  -> Increasing max_position_tau alone will NOT help')
    print('  -> Must reduce balance torque or increase max_tau_wheel')
else:
    print('[LIMITER] Position term saturation is the TRUE LIMITER')
    print(f'  -> Final wheel torque has {max_tau_wheel - final_wheel_torque_max:.1f} Nm headroom')
    print(f'  -> Safe to increase max_position_tau to ~{authority_deficit + np.abs(tau_position_clipped).max():.1f} Nm')
    print('  -> Torque-budget-aware allocation recommended')
print()

# Compute available budget for position
available_budget_positive = max_tau_wheel - np.maximum(0, tau_balance)
available_budget_negative = max_tau_wheel - np.maximum(0, -tau_balance)

print('AVAILABLE BUDGET FOR POSITION (sign-aware):')
print(f'  When tau_position > 0: available = {available_budget_positive.min():.3f} to {available_budget_positive.max():.3f} Nm')
print(f'  When tau_position < 0: available = {available_budget_negative.min():.3f} to {available_budget_negative.max():.3f} Nm')
print()

# Recommend pitch reserve
pitch_reserve_tau = 2.0
print(f'RECOMMENDED SETTINGS:')
print(f'  pitch_reserve_tau: {pitch_reserve_tau:.1f} Nm (safety margin for pitch recovery)')
print(f'  position_tau_budget_cap: {authority_deficit + np.abs(tau_position_clipped).max():.1f} Nm (covers demanded torque)')
print()
