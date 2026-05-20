"""Analyze roll divergence pattern from telemetry."""

import pandas as pd
import numpy as np

df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1779271317.csv')

# Convert radians to degrees
df['roll_deg'] = np.degrees(df['roll'])
df['pitch_deg'] = np.degrees(df['pitch'])

# Key moments in roll divergence
print('Roll divergence analysis:')
print('='*80)
for step in [0, 5, 10, 20, 30, 40, 50, 60, 67]:
    if step < len(df):
        row = df.iloc[step]
        print(f'Step {step:2d}: roll={row["roll_deg"]:6.2f} deg, '
              f'roll_rate={row["roll_rate_rad_s"]:6.3f} rad/s, '
              f'desired_Mx={row["desired_wrench_Mx"]:6.2f} Nm, '
              f'tau_max={row["tau_total_max"]:6.2f} Nm')

print('\n' + '='*80)
print('Summary:')
print(f'Initial roll (step 0): {df.iloc[0]["roll_deg"]:.2f} deg')
print(f'Final roll (step {len(df)-1}): {df.iloc[-1]["roll_deg"]:.2f} deg')
print(f'Total steps: {len(df)}')
print(f'Roll range: [{df["roll_deg"].min():.2f}, {df["roll_deg"].max():.2f}] deg')
print(f'Max desired Mx: {df["desired_wrench_Mx"].max():.2f} Nm')
print(f'Min desired Mx: {df["desired_wrench_Mx"].min():.2f} Nm')
print(f'Max total torque: {df["tau_total_max"].max():.2f} Nm')
print(f'Max roll rate: {df["roll_rate_rad_s"].abs().max():.3f} rad/s')

# Analyze roll divergence rate
print('\n' + '='*80)
print('Roll divergence rate:')
for i in range(0, len(df)-10, 10):
    roll_change = df.iloc[i+10]['roll_deg'] - df.iloc[i]['roll_deg']
    print(f'Steps {i:2d}-{i+10:2d}: roll change = {roll_change:6.2f} deg')
