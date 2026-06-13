import pandas as pd

df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1780215100.csv')
budget_cols = [c for c in df.columns if 'budget' in c.lower() or 'reserve' in c.lower()]

print('Budget-related columns:')
print('\n'.join(budget_cols))

print(f'\nSample values (step 50):')
for col in budget_cols:
    print(f'{col}: {df[col].iloc[50]}')

print(f'\nSaturation reason (step 50): {df["tau_position_saturation_reason"].iloc[50]}')
print(f'Enable budget mode: {df["enable_torque_budget_aware_position"].iloc[50]}')
print(f'Final wheel torque margin (step 50): {df["final_wheel_torque_margin"].iloc[50]:.3f} Nm')
