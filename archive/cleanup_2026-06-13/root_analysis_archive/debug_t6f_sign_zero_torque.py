"""Debug: Why is transmitted torque 0.00 Nm at high_0p480?"""

import pandas as pd
import numpy as np

T6F_SIGN_PATH = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F_sign_corrected/telemetry_1781269776.csv"

df = pd.read_csv(T6F_SIGN_PATH)

print("T6F_sign_corrected Torque Path Audit")
print("="*80)

# Check key torque fields
torque_fields = [
    "tau_position",
    "tau_position_after_clip",
    "apcr1n_tau_position_after_cap",
    "tau_velocity_damping",
    "tau_pitch",
    "tau_wheel_vel_left",
    "tau_wheel_vel_right",
    "final_wheel_tau_with_apc_left",
    "final_wheel_tau_with_apc_right",
]

print("\n[TORQUE FIELD STATISTICS]")
for field in torque_fields:
    if field in df.columns:
        values = df[field].values
        print(f"{field:40s}: min={np.min(values):7.3f}, max={np.max(values):7.3f}, mean={np.mean(values):7.3f}, std={np.std(values):7.3f}")
    else:
        print(f"{field:40s}: NOT FOUND")

# Check arch fix
arch_fix_active = df["arch_fix_active"].values if "arch_fix_active" in df.columns else np.zeros(len(df))
print(f"\n[ARCH FIX]")
print(f"  arch_fix_active count: {np.sum(arch_fix_active)} / {len(df)} ({100*np.mean(arch_fix_active):.1f}%)")

if np.any(arch_fix_active):
    # When arch fix is active, what torques do we see?
    print(f"\n[TORQUES WHEN ARCH_FIX ACTIVE]")
    for field in torque_fields:
        if field in df.columns:
            values_active = df.loc[arch_fix_active, field].values
            print(f"{field:40s}: min={np.min(values_active):7.3f}, max={np.max(values_active):7.3f}, mean={np.mean(values_active):7.3f}")

# Check effective_max_position_tau_after_arch_fix
if "effective_max_position_tau_after_arch_fix" in df.columns:
    eff_max = df["effective_max_position_tau_after_arch_fix"].values
    print(f"\n[EFFECTIVE MAX POSITION TAU AFTER ARCH FIX]")
    print(f"  min={np.min(eff_max):.3f}, max={np.max(eff_max):.3f}, mean={np.mean(eff_max):.3f}")
    print(f"  When arch_fix_active:")
    eff_max_active = df.loc[arch_fix_active, "effective_max_position_tau_after_arch_fix"].values
    print(f"    min={np.min(eff_max_active):.3f}, max={np.max(eff_max_active):.3f}, mean={np.mean(eff_max_active):.3f}")

# Check WBC authority
wbc_fields = ["wbc_wheel_authority", "wbc_authority_enabled"]
print(f"\n[WBC AUTHORITY CHECK]")
for field in wbc_fields:
    if field in df.columns:
        values = df[field].values
        if field == "wbc_authority_enabled":
            print(f"  {field}: {np.sum(values)} / {len(values)} enabled ({100*np.mean(values):.1f}%)")
        else:
            print(f"  {field}: min={np.min(values):.3f}, max={np.max(values):.3f}, mean={np.mean(values):.3f}")

# Sample rows where arch_fix is active
print(f"\n[SAMPLE ROWS WHERE ARCH_FIX ACTIVE]")
arch_fix_indices = np.where(arch_fix_active)[0]
if len(arch_fix_indices) > 0:
    sample_indices = arch_fix_indices[::len(arch_fix_indices)//5] if len(arch_fix_indices) >= 5 else arch_fix_indices
    for idx in sample_indices[:5]:
        print(f"\nStep {idx}:")
        print(f"  tau_position: {df.loc[idx, 'tau_position']:.3f} Nm")
        print(f"  tau_position_after_clip: {df.loc[idx, 'tau_position_after_clip']:.3f} Nm")
        print(f"  apcr1n_tau_position_after_cap: {df.loc[idx, 'apcr1n_tau_position_after_cap']:.3f} Nm")
        print(f"  tau_velocity_damping: {df.loc[idx, 'tau_velocity_damping']:.3f} Nm" if 'tau_velocity_damping' in df.columns else "  tau_velocity_damping: N/A")
        print(f"  tau_wheel_vel_left: {df.loc[idx, 'tau_wheel_vel_left']:.3f} Nm" if 'tau_wheel_vel_left' in df.columns else "  tau_wheel_vel_left: N/A")
        print(f"  tau_wheel_vel_right: {df.loc[idx, 'tau_wheel_vel_right']:.3f} Nm" if 'tau_wheel_vel_right' in df.columns else "  tau_wheel_vel_right: N/A")
        print(f"  final_wheel_tau_with_apc_left: {df.loc[idx, 'final_wheel_tau_with_apc_left']:.3f} Nm")
        print(f"  final_wheel_tau_with_apc_right: {df.loc[idx, 'final_wheel_tau_with_apc_right']:.3f} Nm")
        print(f"  active_pitch_crossing_signed_error_m: {df.loc[idx, 'active_pitch_crossing_signed_error_m']:.3f} m")
