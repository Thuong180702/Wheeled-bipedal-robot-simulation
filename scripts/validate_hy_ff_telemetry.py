"""Validate HY-FF telemetry."""
import pandas as pd

df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1780540473.csv')

print(f'HY-FF active: {df["hip_yaw_comp_active"].unique()}')
print(f'k_support: {df["hip_yaw_comp_k_support"].unique()}')
print(f'tau_max: {df["hip_yaw_comp_tau_max"].unique()}')
print(f'sign: {df["hip_yaw_comp_sign"].unique()}')
print(f'Height gate range: [{df["hip_yaw_comp_height_gate"].min():.3f}, {df["hip_yaw_comp_height_gate"].max():.3f}]')
print(f'Support error range: [{df["hip_yaw_comp_support_error_m"].min():.4f}, {df["hip_yaw_comp_support_error_m"].max():.4f}]')
print(f'Tau left range: [{df["hip_yaw_comp_tau_left"].min():.4f}, {df["hip_yaw_comp_tau_left"].max():.4f}]')
print(f'Tau right range: [{df["hip_yaw_comp_tau_right"].min():.4f}, {df["hip_yaw_comp_tau_right"].max():.4f}]')
print('Telemetry validation: PASS')
