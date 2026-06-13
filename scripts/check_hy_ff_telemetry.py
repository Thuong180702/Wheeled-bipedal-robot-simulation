"""Check HY-FF telemetry from latest run."""
import pandas as pd
from pathlib import Path

sim_dir = Path("outputs/hierarchical_controller_sim")
telem = sorted(sim_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)[-1]
df = pd.read_csv(telem)

print(f"Telemetry file: {telem.name}")
print(f"hip_yaw_comp_active: {df['hip_yaw_comp_active'].any()}")
print(f"hip_yaw_comp_k_support: {df['hip_yaw_comp_k_support'].max()}")
print(f"hip_yaw_comp_height_gate range: [{df['hip_yaw_comp_height_gate'].min():.3f}, {df['hip_yaw_comp_height_gate'].max():.3f}]")
print(f"hip_yaw_comp_support_error_m range: [{df['hip_yaw_comp_support_error_m'].min():.4f}, {df['hip_yaw_comp_support_error_m'].max():.4f}]")
print(f"hip_yaw_comp_tau_left range: [{df['hip_yaw_comp_tau_left'].min():.4f}, {df['hip_yaw_comp_tau_left'].max():.4f}]")
print(f"hip_yaw_abs_max: {df['hip_yaw_abs_max'].max():.4f}")
