"""Check height telemetry."""
import pandas as pd

df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1780543474.csv')

height_var_target = df["height_variant_target_com_z_m"]
target = df["target_com_z_m"]
com_z = df["com_z"]

print(f'height_variant_target_com_z_m: [{height_var_target.min():.3f}, {height_var_target.max():.3f}]')
print(f'target_com_z_m: [{target.min():.3f}, {target.max():.3f}]')
print(f'com_z: [{com_z.min():.3f}, {com_z.max():.3f}]')
print(f'\nFirst 5 rows:')
print(df[['height_variant_target_com_z_m', 'target_com_z_m', 'com_z', 'hip_yaw_comp_height_gate']].head())
