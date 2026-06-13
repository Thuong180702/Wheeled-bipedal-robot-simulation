"""
APCR1n Phase 2: 2000-step Torque and Stability Comparison
Compares D2, APCR1h, and APCR1n torque and stability metrics.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load telemetry
d2_df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_D2/telemetry_d2.csv")
apcr1h_df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1h/telemetry_apcr1h.csv")
apcr1n_df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv")

print("=" * 80)
print("APCR1N PHASE 2: TORQUE AND STABILITY COMPARISON")
print("=" * 80)

results = {}

for name, df in [("D2", d2_df), ("APCR1h", apcr1h_df), ("APCR1n", apcr1n_df)]:
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")

    results[name] = {}

    # ======== TORQUE METRICS ========
    print("\n--- Torque Metrics ---")

    # Find tau_position columns
    tau_pos_cols = [c for c in df.columns if 'tau_position' in c.lower() and 'hip' not in c.lower()]
    print(f"tau_position columns: {tau_pos_cols}")

    # Find APCR tau columns
    apcr_cols = [c for c in df.columns if 'apcr' in c.lower() and 'tau' in c.lower()]
    print(f"APCR tau columns: {apcr_cols[:5]}")

    # Wheel velocity
    wheel_cols = [c for c in df.columns if 'wheel' in c.lower() and 'vel' in c.lower()]
    print(f"Wheel velocity columns: {wheel_cols[:5]}")

    # Tau position
    tau_pos = df['tau_position'] if 'tau_position' in df.columns else None
    if tau_pos is not None:
        print(f"\ntau_position:")
        print(f"  max: {tau_pos.max():.4f}")
        print(f"  min: {tau_pos.min():.4f}")
        print(f"  mean_abs: {abs(tau_pos).mean():.4f}")

        results[name]['tau_position'] = {
            'max': float(tau_pos.max()),
            'min': float(tau_pos.min()),
            'mean_abs': float(abs(tau_pos).mean())
        }

    # Tau APCR
    if 'tau_apcr' in df.columns:
        tau_apcr = df['tau_apcr']
        print(f"\ntau_apcr:")
        print(f"  max: {tau_apcr.max():.4f}")
        print(f"  min: {tau_apcr.min():.4f}")
        print(f"  mean_abs: {abs(tau_apcr).mean():.4f}")

        results[name]['tau_apcr'] = {
            'max': float(tau_apcr.max()),
            'min': float(tau_apcr.min()),
            'mean_abs': float(abs(tau_apcr).mean())
        }

    # Wheel velocity
    wheel_vel_cols = [c for c in df.columns if 'wheel' in c.lower() and 'vel' in c.lower() and 'mean' not in c.lower() and 'rad_s' in c.lower()]
    if wheel_vel_cols:
        wheel_vel = df[wheel_vel_cols[0]]
        print(f"\nWheel velocity ({wheel_vel_cols[0]}):")
        print(f"  max: {wheel_vel.max():.4f}")
        print(f"  min: {wheel_vel.min():.4f}")
        print(f"  mean_abs: {abs(wheel_vel).mean():.4f}")
        print(f"  > 5 rad/s: {(abs(wheel_vel) > 5).sum()} ({(abs(wheel_vel) > 5).sum() / len(wheel_vel) * 100:.1f}%)")

        results[name]['wheel_velocity'] = {
            'max': float(wheel_vel.max()),
            'min': float(wheel_vel.min()),
            'mean_abs': float(abs(wheel_vel).mean()),
            'gt5_count': int((abs(wheel_vel) > 5).sum()),
            'gt5_pct': float((abs(wheel_vel) > 5).sum() / len(wheel_vel) * 100)
        }

    # ======== STABILITY METRICS ========
    print("\n--- Stability Metrics ---")

    # Contact
    if 'n_contacts' in df.columns:
        n_contacts = df['n_contacts']
        print(f"\nContacts:")
        print(f"  min: {n_contacts.min():.0f}")
        print(f"  max: {n_contacts.max():.0f}")
        print(f"  mean: {n_contacts.mean():.2f}")

        results[name]['contacts'] = {
            'min': float(n_contacts.min()),
            'max': float(n_contacts.max()),
            'mean': float(n_contacts.mean())
        }

    # CoM height
    if 'com_z' in df.columns:
        com_z = df['com_z']
        print(f"\nCoM Z:")
        print(f"  min: {com_z.min():.4f}")
        print(f"  max: {com_z.max():.4f}")
        print(f"  mean: {com_z.mean():.4f}")
        print(f"  final: {com_z.iloc[-1]:.4f}")

        results[name]['com_z'] = {
            'min': float(com_z.min()),
            'max': float(com_z.max()),
            'mean': float(com_z.mean()),
            'final': float(com_z.iloc[-1])
        }

    # Height error
    height_err_cols = [c for c in df.columns if 'height_error' in c.lower() or 'height_err' in c.lower()]
    if height_err_cols:
        height_err = df[height_err_cols[0]]
        print(f"\nHeight error ({height_err_cols[0]}):")
        print(f"  max: {abs(height_err).max():.4f}")
        print(f"  mean: {abs(height_err).mean():.4f}")
        print(f"  final: {height_err.iloc[-1]:.4f}")

        results[name]['height_error'] = {
            'max': float(abs(height_err).max()),
            'mean': float(abs(height_err).mean()),
            'final': float(height_err.iloc[-1])
        }

    # Pitch
    pitch_cols = [c for c in df.columns if 'pitch' in c.lower() and 'x' in c.lower() and 'deg' not in c.lower()]
    if pitch_cols:
        pitch = df[pitch_cols[0]]
        print(f"\nPitch ({pitch_cols[0]}):")
        print(f"  min: {pitch.min():.4f} ({np.degrees(pitch.min()):.2f} deg)")
        print(f"  max: {pitch.max():.4f} ({np.degrees(pitch.max()):.2f} deg)")
        print(f"  RMS: {np.sqrt((pitch**2).mean()):.4f} ({np.degrees(np.sqrt((pitch**2).mean())):.2f} deg)")

        results[name]['pitch'] = {
            'min': float(pitch.min()),
            'max': float(pitch.max()),
            'rms': float(np.sqrt((pitch**2).mean()))
        }

    # Roll
    roll_cols = [c for c in df.columns if 'roll' in c.lower() and 'y' in c.lower() and 'deg' not in c.lower()]
    if roll_cols:
        roll = df[roll_cols[0]]
        print(f"\nRoll ({roll_cols[0]}):")
        print(f"  min: {roll.min():.4f} ({np.degrees(roll.min()):.2f} deg)")
        print(f"  max: {roll.max():.4f} ({np.degrees(roll.max()):.2f} deg)")
        print(f"  RMS: {np.sqrt((roll**2).mean()):.4f} ({np.degrees(np.sqrt((roll**2).mean())):.2f} deg)")

        results[name]['roll'] = {
            'min': float(roll.min()),
            'max': float(roll.max()),
            'rms': float(np.sqrt((roll**2).mean()))
        }

    # Hip yaw
    hip_yaw_cols = [c for c in df.columns if 'hip_yaw' in c.lower() and 'error' in c.lower()]
    if hip_yaw_cols:
        hip_yaw = df[hip_yaw_cols[0]]
        print(f"\nHip yaw error ({hip_yaw_cols[0]}):")
        print(f"  max: {abs(hip_yaw).max():.4f}")
        print(f"  mean: {abs(hip_yaw).mean():.4f}")
        print(f"  final: {hip_yaw.iloc[-1]:.4f}")

        results[name]['hip_yaw_error'] = {
            'max': float(abs(hip_yaw).max()),
            'mean': float(abs(hip_yaw).mean()),
            'final': float(hip_yaw.iloc[-1])
        }

    # Hidden torque
    hidden_cols = [c for c in df.columns if 'hidden' in c.lower() and 'tau' in c.lower()]
    if hidden_cols:
        hidden_tau = df[hidden_cols[0]]
        print(f"\nHidden torque ({hidden_cols[0]}):")
        print(f"  max: {abs(hidden_tau).max():.4f}")
        print(f"  mean: {abs(hidden_tau).mean():.4f}")

        results[name]['hidden_torque'] = {
            'max': float(abs(hidden_tau).max()),
            'mean': float(abs(hidden_tau).mean())
        }

    # Ownership violation
    ownership_cols = [c for c in df.columns if 'ownership' in c.lower() and 'violation' in c.lower()]
    if ownership_cols:
        ownership = df[ownership_cols[0]]
        print(f"\nOwnership violation ({ownership_cols[0]}):")
        print(f"  max: {abs(ownership).max():.4f}")
        print(f"  mean: {abs(ownership).mean():.4f}")

        results[name]['ownership_violation'] = {
            'max': float(abs(ownership).max()),
            'mean': float(abs(ownership).mean())
        }

    # WBC applied
    wbc_cols = [c for c in df.columns if 'wbc' in c.lower() and 'applied' in c.lower()]
    if wbc_cols:
        wbc = df[wbc_cols[0]]
        print(f"\nWBC applied ({wbc_cols[0]}):")
        print(f"  active: {(wbc > 0).sum()} ({(wbc > 0).sum() / len(wbc) * 100:.1f}%)")

        results[name]['wbc_applied'] = {
            'active_count': int((wbc > 0).sum()),
            'active_pct': float((wbc > 0).sum() / len(wbc) * 100)
        }

# ======== APCR1N-SPECIFIC METRICS ========
print("\n" + "=" * 80)
print("APCR1N-SPECIFIC METRICS")
print("=" * 80)

apcr1n_cols = [c for c in apcr1n_df.columns if 'apcr1n' in c]
print(f"\nAPCR1n columns found: {len(apcr1n_cols)}")

# Position cap
if 'apcr1n_position_cap_current' in apcr1n_df.columns:
    pc = apcr1n_df['apcr1n_position_cap_current']
    print(f"\nPosition cap current:")
    print(f"  min: {pc.min():.4f}")
    print(f"  max: {pc.max():.4f}")
    print(f"  mean: {pc.mean():.4f}")

# Position saturated
if 'apcr1n_position_saturated' in apcr1n_df.columns:
    ps = apcr1n_df['apcr1n_position_saturated']
    print(f"\nPosition saturated: {ps.sum()} ({ps.sum() / len(ps) * 100:.2f}%)")

# Tau comparison
if 'apcr1n_tau_position_raw' in apcr1n_df.columns and 'apcr1n_tau_position_after_cap' in apcr1n_df.columns:
    raw = apcr1n_df['apcr1n_tau_position_raw']
    after = apcr1n_df['apcr1n_tau_position_after_cap']
    print(f"\nTau position raw mean: {raw.mean():.4f}")
    print(f"Tau position after cap mean: {after.mean():.4f}")
    diff = abs(raw - after).mean()
    print(f"Difference (cap effect): {diff:.4f}")

# Wheel damping
if 'apcr1n_wheel_damping_scale' in apcr1n_df.columns:
    wd_scale = apcr1n_df['apcr1n_wheel_damping_scale']
    print(f"\nWheel damping scale:")
    print(f"  mean: {wd_scale.mean():.4f}")
    print(f"  min: {wd_scale.min():.4f}")
    print(f"  max: {wd_scale.max():.4f}")

# Save results
output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "apcr1n_phase2_torque_stability_comparison.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save CSV summary
csv_data = []
for name in ["D2", "APCR1h", "APCR1n"]:
    r = results.get(name, {})
    row = {'Profile': name}

    if 'tau_position' in r:
        row['tau_position_max'] = r['tau_position']['max']
        row['tau_position_mean_abs'] = r['tau_position']['mean_abs']

    if 'wheel_velocity' in r:
        row['wheel_vel_max'] = r['wheel_velocity']['max']
        row['wheel_vel_mean_abs'] = r['wheel_velocity']['mean_abs']
        row['wheel_vel_gt5_pct'] = r['wheel_velocity']['gt5_pct']

    if 'com_z' in r:
        row['com_z_min'] = r['com_z']['min']
        row['com_z_mean'] = r['com_z']['mean']

    if 'pitch' in r:
        row['pitch_max_deg'] = np.degrees(r['pitch']['max']) if r['pitch']['max'] else None
        row['pitch_rms_deg'] = np.degrees(r['pitch']['rms']) if r['pitch']['rms'] else None

    if 'roll' in r:
        row['roll_max_deg'] = np.degrees(r['roll']['max']) if r['roll']['max'] else None
        row['roll_rms_deg'] = np.degrees(r['roll']['rms']) if r['roll']['rms'] else None

    csv_data.append(row)

pd.DataFrame(csv_data).to_csv(output_dir / "apcr1n_phase2_torque_stability_comparison.csv", index=False)

print(f"\nResults saved to:")
print(f"  - {output_dir / 'apcr1n_phase2_torque_stability_comparison.json'}")
print(f"  - {output_dir / 'apcr1n_phase2_torque_stability_comparison.csv'}")
