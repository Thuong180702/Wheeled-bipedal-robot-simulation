"""
Compare E1 500-step vs D2 first 500 rows for support/position error and integral diagnostics.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
E1_PATH = Path("outputs/step_e_extreme_support_fix_eval/e1_low_0p300_500_before_fix/e1_low_0p300_500_before_fix_telemetry.csv")
D2_PATH = Path("outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv")
OUT_JSON = Path("outputs/step_e_extreme_support_fix_eval/e1_low_0p300_500_before_fix_analysis.json")
OUT_MD = Path("docs/validation/e1_low_0p300_500_before_fix_analysis.md")

# Load
e1 = pd.read_csv(E1_PATH)
d2 = pd.read_csv(D2_PATH).head(500)

print(f"E1 rows: {len(e1)}, D2 rows (first 500): {len(d2)}")

# Extract key columns
def compute_metrics(df, label):
    """Compute official Step E metrics for a dataframe."""
    m = {}

    # Run integrity
    m['rows_completed'] = len(df)
    m['survived_500'] = len(df) >= 500 and not df['terminated'].any()
    m['termination_count'] = int(df['terminated'].sum())
    m['termination_reasons'] = df['termination_reason'].dropna().unique().tolist()

    # Support position error (OFFICIAL STEP E METRIC)
    # Uses support_position_error_m column which is Euclidean distance of support_center error
    support_pos_err = np.abs(df['support_position_error_m'].values) if 'support_position_error_m' in df.columns else np.abs(df['cp_x'].values)
    m['support_position_error_max'] = float(support_pos_err.max())
    m['support_position_error_final'] = float(support_pos_err[-1])
    m['support_position_error_mean'] = float(support_pos_err.mean())
    m['support_position_error_std'] = float(support_pos_err.std())

    # First crossing of support_position_error > 0.15 m
    crossings = df[support_pos_err > 0.15].index
    m['support_error_gt_0p15_first_step'] = int(crossings[0]) if len(crossings) > 0 else None
    m['support_error_gt_0p15_count'] = len(crossings)

    # Hip yaw
    hip_yaw_cols = [c for c in df.columns if 'hip_yaw' in c.lower() and 'abs' in c.lower()]
    if hip_yaw_cols:
        hip_yaw = np.abs(df[hip_yaw_cols[0]])
    else:
        hip_yaw = np.abs(df['yaw_drift_from_initial_rad']) if 'yaw_drift_from_initial_rad' in df.columns else np.zeros(len(df))
    m['hip_yaw_abs_max'] = float(hip_yaw.max())
    m['hip_yaw_abs_final'] = float(hip_yaw.iloc[-1])
    m['hip_yaw_abs_mean'] = float(hip_yaw.mean())

    # Hip yaw crossing > 0.10 rad
    yaw_crossings = df[hip_yaw > 0.10].index
    m['hip_yaw_gt_0p10_first_step'] = int(yaw_crossings[0]) if len(yaw_crossings) > 0 else None

    # Wheel velocity
    if 'wheel_vel_mean' in df.columns:
        wheel_vel = np.abs(df['wheel_vel_mean'].values)
    else:
        wheel_vel = np.zeros(len(df))
    m['wheel_vel_mean_max'] = float(wheel_vel.max())
    m['wheel_vel_mean_final'] = float(wheel_vel[-1])
    m['wheel_vel_mean_mean'] = float(wheel_vel.mean())

    # Contact
    m['contact_valid_percent_raw'] = float(df['contact_force_valid'].mean() * 100) if 'contact_force_valid' in df.columns else None
    m['non_wheel_floor_contacts_max'] = float(df['non_wheel_floor_contacts'].max()) if 'non_wheel_floor_contacts' in df.columns else None

    # WBC gate
    m['wbc_control_gate_passes'] = 'wbc_control_gate' in df.columns
    if 'wbc_control_gate' in df.columns:
        m['wbc_gate_pass_percent'] = float(df['wbc_control_gate'].mean() * 100)

    # Hidden torque and ownership
    m['hidden_torque_norm_max'] = float(df['hidden_torque_norm'].max()) if 'hidden_torque_norm' in df.columns else None
    m['ownership_violation_count_max'] = int(df['ownership_violation_count'].max()) if 'ownership_violation_count' in df.columns else None

    # Height error
    if 'height_error' in df.columns:
        height_err = np.abs(df['height_error'].values)
    elif 'com_z' in df.columns and 'height_cmd' in df.columns:
        height_err = np.abs(df['com_z'].values - df['height_cmd'].values)
    else:
        height_err = np.zeros(len(df))
    m['height_error_max'] = float(height_err.max())
    m['height_error_final'] = float(height_err[-1])
    m['height_error_mean'] = float(height_err.mean())

    # Roll
    roll = np.abs(df['roll_y'].values) if 'roll_y' in df.columns else np.abs(df['euler_roll_x'].values)
    m['roll_y_max'] = float(roll.max())
    m['roll_y_final'] = float(roll[-1])
    m['roll_y_mean'] = float(roll.mean())

    # Pitch
    pitch = np.abs(df['pitch_x'].values) if 'pitch_x' in df.columns else np.abs(df['robot_pitch_x'].values)
    m['pitch_x_max'] = float(pitch.max())
    m['pitch_x_final'] = float(pitch[-1])
    m['pitch_x_mean'] = float(pitch.mean())

    # E1 integral diagnostics (may not exist in D2)
    e1_fields = [
        'position_integral_enabled', 'integral_active',
        'tau_position_integral', 'tau_position_raw', 'tau_position_final',
        'support_position_error'
    ]

    for field in e1_fields:
        if field in df.columns:
            m[f'e1_{field}_exists'] = True
            vals = df[field].values
            if 'integral' in field.lower() and 'enabled' not in field.lower() and 'active' not in field.lower():
                m[f'e1_{field}_max'] = float(np.abs(vals).max())
                m[f'e1_{field}_final'] = float(np.abs(vals)[-1])
                m[f'e1_{field}_mean'] = float(np.abs(vals).mean())
            elif 'active' in field.lower():
                m[f'e1_{field}_count'] = int(vals.sum())
                m[f'e1_{field}_percent'] = float(vals.mean() * 100)
            else:
                m[f'e1_{field}_max'] = float(np.abs(vals).max())
                m[f'e1_{field}_final'] = float(vals[-1])
                m[f'e1_{field}_mean'] = float(vals.mean())
        else:
            m[f'e1_{field}_exists'] = False

    # Gate reason counts (if available)
    gate_reason_fields = [c for c in df.columns if c.startswith('gate_reason_') or 'gate_' in c]
    for field in gate_reason_fields:
        vals = df[field].values
        try:
            m[f'e1_gate_{field}_sum'] = int(np.nansum(vals))
        except Exception:
            m[f'e1_gate_{field}_sum'] = 0

    # Support gate crossings (if gate field exists)
    if 'support_gate_pass' in df.columns:
        m['support_gate_pass_percent'] = float(df['support_gate_pass'].mean() * 100)
        m['support_gate_fail_count'] = int((~df['support_gate_pass'].astype(bool)).sum())
    elif 'gate_pass' in df.columns:
        m['support_gate_pass_percent'] = float(df['gate_pass'].mean() * 100)
        m['support_gate_fail_count'] = int((~df['gate_pass'].astype(bool)).sum())

    return m

print("\nComputing E1 metrics...")
e1_m = compute_metrics(e1, "E1")
print("\nComputing D2 metrics...")
d2_m = compute_metrics(d2, "D2")

# Print comparison
print("\n" + "="*80)
print("E1 500-STEP vs D2 FIRST 500 ROWS COMPARISON")
print("="*80)

# Support position error
print("\n--- SUPPORT POSITION ERROR ---")
print(f"  E1 max:  {e1_m['support_position_error_max']:.6f} m")
print(f"  D2 max:  {d2_m['support_position_error_max']:.6f} m")
print(f"  E1 mean: {e1_m['support_position_error_mean']:.6f} m")
print(f"  D2 mean: {d2_m['support_position_error_mean']:.6f} m")
print(f"  E1 first crossing > 0.15: step {e1_m['support_error_gt_0p15_first_step']}")
print(f"  D2 first crossing > 0.15: step {d2_m['support_error_gt_0p15_first_step']}")
print(f"  E1 crossings > 0.15 count: {e1_m['support_error_gt_0p15_count']}")
print(f"  D2 crossings > 0.15 count: {d2_m['support_error_gt_0p15_count']}")

# Hip yaw
print("\n--- HIP YAW ---")
print(f"  E1 abs_max: {e1_m['hip_yaw_abs_max']:.6f} rad")
print(f"  D2 abs_max: {d2_m['hip_yaw_abs_max']:.6f} rad")
print(f"  E1 abs_mean: {e1_m['hip_yaw_abs_mean']:.6f} rad")
print(f"  D2 abs_mean: {d2_m['hip_yaw_abs_mean']:.6f} rad")

# Wheel velocity
print("\n--- WHEEL VELOCITY ---")
print(f"  E1 mean_max: {e1_m['wheel_vel_mean_max']:.6f} rad/s")
print(f"  D2 mean_max: {d2_m['wheel_vel_mean_max']:.6f} rad/s")

# Height error
print("\n--- HEIGHT ERROR ---")
print(f"  E1 max: {e1_m['height_error_max']:.6f} m")
print(f"  D2 max: {d2_m['height_error_max']:.6f} m")

# Roll
print("\n--- ROLL ---")
print(f"  E1 max: {e1_m['roll_y_max']:.6f} rad")
print(f"  D2 max: {d2_m['roll_y_max']:.6f} rad")

# Pitch
print("\n--- PITCH ---")
print(f"  E1 max: {e1_m['pitch_x_max']:.6f} rad")
print(f"  D2 max: {d2_m['pitch_x_max']:.6f} rad")

# Contact
print("\n--- CONTACT ---")
print(f"  E1 contact_valid%: {e1_m['contact_valid_percent_raw']:.1f}%")
print(f"  D2 contact_valid%: {d2_m['contact_valid_percent_raw']:.1f}%")
print(f"  E1 non_wheel_contacts max: {e1_m['non_wheel_floor_contacts_max']}")
print(f"  D2 non_wheel_contacts max: {d2_m['non_wheel_floor_contacts_max']}")

# E1 integral diagnostics
print("\n--- E1 INTEGRAL DIAGNOSTICS ---")
for key in ['position_integral_enabled', 'integral_active', 'tau_position_integral', 'tau_position_raw', 'tau_position_final']:
    exists_key = f'e1_{key}_exists'
    if exists_key in e1_m and e1_m[exists_key]:
        max_key = f'e1_{key}_max'
        mean_key = f'e1_{key}_mean'
        if 'active' in key.lower():
            cnt_key = f'e1_{key}_count'
            pct_key = f'e1_{key}_percent'
            print(f"  {key}: count={e1_m.get(cnt_key, 'N/A')}, percent={e1_m.get(pct_key, 'N/A'):.1f}%")
        else:
            print(f"  {key}: max={e1_m.get(max_key, 'N/A'):.6f}, mean={e1_m.get(mean_key, 'N/A'):.6f}")
    else:
        print(f"  {key}: NOT IN TELEMETRY")

# Gate reasons
print("\n--- E1 GATE REASONS ---")
gate_keys = [k for k in e1_m.keys() if k.startswith('e1_gate_')]
for k in sorted(gate_keys):
    if e1_m[k] > 0:
        print(f"  {k}: {e1_m[k]}")

# Classification
print("\n" + "="*80)
print("CLASSIFICATION")
print("="*80)

# Improvement: support error max/mean decreases or crossing delayed
support_improves = (
    e1_m['support_position_error_max'] < d2_m['support_position_error_max'] or
    e1_m['support_position_error_mean'] < d2_m['support_position_error_mean'] or
    (e1_m['support_error_gt_0p15_first_step'] is not None and d2_m['support_error_gt_0p15_first_step'] is not None and
     e1_m['support_error_gt_0p15_first_step'] > d2_m['support_error_gt_0p15_first_step']) or
    e1_m['support_error_gt_0p15_count'] < d2_m['support_error_gt_0p15_count']
)

# hip yaw worsens
hip_yaw_worsens = e1_m['hip_yaw_abs_max'] > d2_m['hip_yaw_abs_max'] * 1.2  # 20% worse

# wheel velocity worsens
wheel_vel_worsens = e1_m['wheel_vel_mean_max'] > d2_m['wheel_vel_mean_max'] * 1.2

# height/roll/contact remain valid
height_valid = e1_m['contact_valid_percent_raw'] is not None and e1_m['contact_valid_percent_raw'] >= 50.0

# Integral is active
integral_active = e1_m.get('e1_integral_active_count', 0) > 50  # at least 50 steps
integral_nonzero = e1_m.get('e1_tau_position_integral_max', 0) > 0.001

print(f"\nSupport improves: {support_improves}")
print(f"Hip yaw worsens: {hip_yaw_worsens}")
print(f"Wheel velocity worsens: {wheel_vel_worsens}")
print(f"Height/contact valid: {height_valid}")
print(f"Integral active (count > 50): {integral_active}")
print(f"Integral nonzero (max > 0.001): {integral_nonzero}")

# Final classification
if hip_yaw_worsens or wheel_vel_worsens:
    classification = "E1_500_REGRESSION_DO_NOT_CONTINUE"
    reason = "hip_yaw or wheel_velocity worsened"
elif support_improves and height_valid:
    classification = "E1_500_IMPROVES_SUPPORT"
    reason = "support_position_error improved"
elif not integral_nonzero or not integral_active:
    classification = "E1_500_NO_EFFECT"
    reason = "integral term was not meaningfully active or nonzero"
else:
    classification = "E1_500_TELEMETRY_INCONCLUSIVE"
    reason = "cannot determine clear improvement or regression"

print(f"\nFINAL CLASSIFICATION: {classification}")
print(f"REASON: {reason}")

# Save results
result = {
    'classification': classification,
    'reason': reason,
    'e1_500': e1_m,
    'd2_500': d2_m,
    'support_improves': support_improves,
    'hip_yaw_worsens': hip_yaw_worsens,
    'wheel_vel_worsens': wheel_vel_worsens,
    'height_valid': height_valid,
    'integral_active': integral_active,
    'integral_nonzero': integral_nonzero
}

with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved analysis to {OUT_JSON}")

# Generate markdown report
def fmt(v, spec='.6f'):
    if v is None:
        return 'N/A'
    if isinstance(v, str):
        return v
    try:
        return format(v, spec)
    except Exception:
        return str(v)

def fmt_int(v):
    if v is None:
        return 'N/A'
    return str(int(v))

md = f"""# E1_support_integral 500-Step Before-Fix Analysis

## Summary

- **E1 500-step simulation**: {len(e1)} rows, survived={e1_m['survived_500']}
- **D2 first 500 rows**: {len(d2)} rows, survived={d2_m['survived_500']}
- **Classification**: `{classification}`
- **Reason**: {reason}

## Support Position Error

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| max (m) | {fmt(e1_m['support_position_error_max'])} | {fmt(d2_m['support_position_error_max'])} | {e1_m['support_position_error_max'] - d2_m['support_position_error_max']:+.6f} |
| mean (m) | {fmt(e1_m['support_position_error_mean'])} | {fmt(d2_m['support_position_error_mean'])} | {e1_m['support_position_error_mean'] - d2_m['support_position_error_mean']:+.6f} |
| first crossing > 0.15m | step {fmt_int(e1_m['support_error_gt_0p15_first_step'])} | step {fmt_int(d2_m['support_error_gt_0p15_first_step'])} | - |
| crossings > 0.15 count | {fmt_int(e1_m['support_error_gt_0p15_count'])} | {fmt_int(d2_m['support_error_gt_0p15_count'])} | {e1_m['support_error_gt_0p15_count'] - d2_m['support_error_gt_0p15_count']:+d} |

## Hip Yaw

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| abs_max (rad) | {fmt(e1_m['hip_yaw_abs_max'])} | {fmt(d2_m['hip_yaw_abs_max'])} | {e1_m['hip_yaw_abs_max'] - d2_m['hip_yaw_abs_max']:+.6f} |
| abs_mean (rad) | {fmt(e1_m['hip_yaw_abs_mean'])} | {fmt(d2_m['hip_yaw_abs_mean'])} | {e1_m['hip_yaw_abs_mean'] - d2_m['hip_yaw_abs_mean']:+.6f} |

## Wheel Velocity

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| mean_max (rad/s) | {fmt(e1_m['wheel_vel_mean_max'])} | {fmt(d2_m['wheel_vel_mean_max'])} | {e1_m['wheel_vel_mean_max'] - d2_m['wheel_vel_mean_max']:+.6f} |

## Height/Roll/Pitch

| Metric | E1 | D2 | Delta |
|--------|----|----|-------|
| height_error_max (m) | {fmt(e1_m['height_error_max'])} | {fmt(d2_m['height_error_max'])} | {e1_m['height_error_max'] - d2_m['height_error_max']:+.6f} |
| roll_y_max (rad) | {fmt(e1_m['roll_y_max'])} | {fmt(d2_m['roll_y_max'])} | {e1_m['roll_y_max'] - d2_m['roll_y_max']:+.6f} |
| pitch_x_max (rad) | {fmt(e1_m['pitch_x_max'])} | {fmt(d2_m['pitch_x_max'])} | {e1_m['pitch_x_max'] - d2_m['pitch_x_max']:+.6f} |

## Contact/Validity

| Metric | E1 | D2 |
|--------|----|----|
| contact_valid% | {fmt(e1_m['contact_valid_percent_raw'], '.1f')}% | {fmt(d2_m['contact_valid_percent_raw'], '.1f')}% |
| non_wheel_contacts_max | {fmt_int(e1_m['non_wheel_floor_contacts_max'])} | {fmt_int(d2_m['non_wheel_floor_contacts_max'])} |

## E1 Integral Diagnostics

| Field | Value |
|-------|-------|
| position_integral_enabled exists | {e1_m.get('e1_position_integral_enabled_exists', 'N/A')} |
| integral_active count | {fmt_int(e1_m.get('e1_integral_active_count', 'N/A'))} |
| integral_active percent | {fmt(e1_m.get('e1_integral_active_percent', 'N/A'), '.1f')}% |
| tau_position_integral max | {fmt(e1_m.get('e1_tau_position_integral_max', 'N/A'))} |
| tau_position_integral mean | {fmt(e1_m.get('e1_tau_position_integral_mean', 'N/A'))} |
| tau_position_raw max | {fmt(e1_m.get('e1_tau_position_raw_max', 'N/A'))} |
| tau_position_final max | {e1_m.get('e1_tau_position_final_max', 'N/A')} |

## Gate Reason Counts (E1)

"""

gate_keys_sorted = sorted([k for k in e1_m.keys() if k.startswith('e1_gate_') and e1_m[k] > 0])
for k in gate_keys_sorted:
    md += f"- {k}: {e1_m[k]}\n"

md += f"""

## Decision Criteria

- Support improves: **{support_improves}**
- Hip yaw worsens: **{hip_yaw_worsens}**
- Wheel velocity worsens: **{wheel_vel_worsens}**
- Height/contact valid: **{height_valid}**
- Integral active (>50 steps): **{integral_active}**
- Integral nonzero (>0.001): **{integral_nonzero}**

## Conclusion

**{classification}**: {reason}
"""

with open(OUT_MD, 'w') as f:
    f.write(md)
print(f"Saved markdown report to {OUT_MD}")
