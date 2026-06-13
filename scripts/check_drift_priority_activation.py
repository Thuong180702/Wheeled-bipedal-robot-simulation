"""Check why drift priority doesn't activate"""
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv")

error_col = 'active_pitch_crossing_signed_error_m'
error = df[error_col]

# Check for drift priority activation conditions
# 1. abs(error) > 0.08
# 2. moving away from zero (error * error_diff > 0)

error_diff = error.diff()
moving_away = (error * error_diff) > 0

print("=" * 80)
print("DRIFT PRIORITY ACTIVATION ANALYSIS")
print("=" * 80)

print(f"\nError statistics:")
print(f"  min: {error.min():.4f}")
print(f"  max: {error.max():.4f}")
print(f"  abs_max: {abs(error).max():.4f}")
print(f"  mean: {error.mean():.4f}")

print(f"\nDrift priority conditions (steps 100+):")
post = df[df['step'] >= 100]
post_error = post[error_col]
post_diff = post_error.diff()
post_moving_away = (post_error * post_diff) > 0

# Condition 1: abs(error) > 0.08
cond1 = abs(post_error) > 0.08
print(f"  abs(error) > 0.08: {cond1.sum()}/{len(post)} ({100*cond1.sum()/len(post):.1f}%)")

# Condition 2: moving away
cond2 = post_moving_away
print(f"  moving away from zero: {cond2.sum()}/{len(post)} ({100*cond2.sum()/len(post):.1f}%)")

# Both conditions
both = cond1 & cond2
print(f"  BOTH conditions: {both.sum()}/{len(post)} ({100*both.sum()/len(post):.1f}%)")

# Also check if error is positive (drift priority might only activate for positive drift)
positive = post_error > 0
print(f"\n  Positive error: {positive.sum()}/{len(post)} ({100*positive.sum()/len(post):.1f}%)")
print(f"  Positive AND abs > 0.08: {(positive & cond1).sum()}/{len(post)}")
print(f"  Positive AND moving away: {(positive & cond2).sum()}/{len(post)}")

# Check error_rate column if available
rate_cols = [c for c in df.columns if 'error_rate' in c.lower() or 'rate' in c.lower()]
print(f"\nRate columns found: {rate_cols[:5]}")

# Check telemetry for drift priority
dp_cols = [c for c in df.columns if 'drift_priority' in c.lower()]
print(f"\nDrift priority telemetry columns: {dp_cols}")

# Check phase_recenter_active which IS active
print(f"\nphase_recenter_active: {df['phase_recenter_active'].sum()}/{len(df)}")
print(f"hysteresis_recenter_active: {df['hysteresis_recenter_active'].sum()}/{len(df)}")

# Check the relationship between phase_recenter and drift priority
print(f"\nCross-analysis:")
phase_recenter = df['phase_recenter_active'] == 1
drift_priority_eligible = both
print(f"  phase_recenter active: {phase_recenter.sum()}")
print(f"  drift_priority eligible: {drift_priority_eligible.sum()}")
print(f"  Both active: {(phase_recenter & drift_priority_eligible).sum()}")
