"""
APCR1i 1000-step principle verification.

Verifies if APCR1i follows the user principle:
- If drift is far from zero, support recentering remains active.
- Recenter does not exit just because pitch is balanced.
- Recenter does not exit just because pitch sign changes.
- Recenter holds until the support error reaches near zero or slightly crosses opposite side.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Paths
CSV_PATH = "outputs/hierarchical_controller_sim/telemetry_1781058071.csv"
OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1i_low_0p300_1000")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading APCR1i 1000-step telemetry...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")

# Key columns
DRIFT_COL = 'support_position_error_m'
STATE_COL = 'active_pitch_crossing_hysteresis_state'
TAU_COL = 'active_pitch_crossing_tau'
PITCH_COL = 'pitch_x'
HEIGHT_COL = 'torso_z'

# Get data
steps = df['step'].values
e = df[DRIFT_COL].values
state = df[STATE_COL].values
tau = df[TAU_COL].values
pitch = df[PITCH_COL].values
height = df[HEIGHT_COL].values if HEIGHT_COL in df.columns else np.ones(len(df)) * 0.3

# Principle thresholds
OUTER_THRESHOLD = 0.08  # m - when to enter RECENTER
INNER_THRESHOLD = 0.03   # m - when to exit RECENTER

# Analysis
print("\n=== APCR1i Principle Verification ===")

# Principle 1: When e > +0.08, APCR should be RECENTER_FROM_POSITIVE
print("\n--- Principle 1: e > +0.08 should enter RECENTER_FROM_POSITIVE ---")
e_positive_large = e > OUTER_THRESHOLD
if np.any(e_positive_large):
    state_when_positive = state[e_positive_large]
    recenter_from_positive = state_when_positive == 'RECENTER_FROM_POSITIVE'
    neutral = state_when_positive == 'NEUTRAL'
    recenter_from_negative = state_when_positive == 'RECENTER_FROM_NEGATIVE'

    print(f"Steps with e > +{OUTER_THRESHOLD}: {np.sum(e_positive_large)}")
    print(f"  RECENTER_FROM_POSITIVE: {np.sum(recenter_from_positive)} ({100*np.sum(recenter_from_positive)/np.sum(e_positive_large):.1f}%)")
    print(f"  NEUTRAL: {np.sum(neutral)} ({100*np.sum(neutral)/np.sum(e_positive_large):.1f}%)")
    print(f"  RECENTER_FROM_NEGATIVE: {np.sum(recenter_from_negative)} ({100*np.sum(recenter_from_negative)/np.sum(e_positive_large):.1f}%)")

    # Check tau sign when e > 0.08
    tau_when_positive = tau[e_positive_large]
    correct_sign = tau_when_positive < 0  # Negative torque for positive drift
    print(f"\n  Torque sign check (should be negative for positive drift):")
    print(f"    Negative tau: {np.sum(correct_sign)} ({100*np.sum(correct_sign)/len(tau_when_positive):.1f}%)")
    print(f"    Positive tau: {np.sum(~correct_sign)} ({100*np.sum(~correct_sign)/len(tau_when_positive):.1f}%)")
    print(f"    Tau range: {tau_when_positive.min():.4f} to {tau_when_positive.max():.4f}")

    principle1_pass = np.sum(recenter_from_positive) / np.sum(e_positive_large) > 0.9
    print(f"\n  Principle 1: {'PASS' if principle1_pass else 'FAIL'}")
else:
    print("No steps with e > +0.08")
    principle1_pass = True

# Principle 2: When e < -0.08, APCR should be RECENTER_FROM_NEGATIVE
print("\n--- Principle 2: e < -0.08 should enter RECENTER_FROM_NEGATIVE ---")
e_negative_large = e < -OUTER_THRESHOLD
if np.any(e_negative_large):
    state_when_negative = state[e_negative_large]
    recenter_from_negative = state_when_negative == 'RECENTER_FROM_NEGATIVE'
    neutral = state_when_negative == 'NEUTRAL'
    recenter_from_positive = state_when_negative == 'RECENTER_FROM_POSITIVE'

    print(f"Steps with e < -{OUTER_THRESHOLD}: {np.sum(e_negative_large)}")
    print(f"  RECENTER_FROM_NEGATIVE: {np.sum(recenter_from_negative)} ({100*np.sum(recenter_from_negative)/np.sum(e_negative_large):.1f}%)")
    print(f"  NEUTRAL: {np.sum(neutral)} ({100*np.sum(neutral)/np.sum(e_negative_large):.1f}%)")
    print(f"  RECENTER_FROM_POSITIVE: {np.sum(recenter_from_positive)} ({100*np.sum(recenter_from_positive)/np.sum(e_negative_large):.1f}%)")

    # Check tau sign when e < -0.08
    tau_when_negative = tau[e_negative_large]
    correct_sign = tau_when_negative > 0  # Positive torque for negative drift
    print(f"\n  Torque sign check (should be positive for negative drift):")
    print(f"    Positive tau: {np.sum(correct_sign)} ({100*np.sum(correct_sign)/len(tau_when_negative):.1f}%)")
    print(f"    Negative tau: {np.sum(~correct_sign)} ({100*np.sum(~correct_sign)/len(tau_when_negative):.1f}%)")
    print(f"    Tau range: {tau_when_negative.min():.4f} to {tau_when_negative.max():.4f}")

    principle2_pass = np.sum(recenter_from_negative) / np.sum(e_negative_large) > 0.9
    print(f"\n  Principle 2: {'PASS' if principle2_pass else 'FAIL'}")
else:
    print("No steps with e < -0.08")
    principle2_pass = True

# Principle 3: RECENTER should hold until inner band or opposite threshold
print("\n--- Principle 3: RECENTER should not exit early ---")

# Find all RECENTER episodes
recenter_mask = np.array(['RECENTER' in s for s in state])
recenter_episodes = []
current_start = None

for i in range(len(state)):
    if recenter_mask[i] and current_start is None:
        current_start = i
    elif not recenter_mask[i] and current_start is not None:
        recenter_episodes.append((current_start, i - 1))
        current_start = None

if current_start is not None:
    recenter_episodes.append((current_start, len(state) - 1))

print(f"Total RECENTER episodes: {len(recenter_episodes)}")

early_exits = 0
correct_exits = 0
for idx, (start, end) in enumerate(recenter_episodes):
    episode_e = e[start:end+1]
    episode_state = state[start]

    # Check if exited correctly
    exit_e = e[end]

    if episode_state == 'RECENTER_FROM_POSITIVE':
        if exit_e <= INNER_THRESHOLD:
            correct_exits += 1
        elif exit_e > OUTER_THRESHOLD and not any(episode_e <= INNER_THRESHOLD):
            early_exits += 1
            print(f"  Episode {idx+1} (RECENTER_FROM_POSITIVE): early exit at e={exit_e:.4f}")
    elif episode_state == 'RECENTER_FROM_NEGATIVE':
        if exit_e >= -INNER_THRESHOLD:
            correct_exits += 1
        elif exit_e < -OUTER_THRESHOLD and not any(episode_e >= -INNER_THRESHOLD):
            early_exits += 1
            print(f"  Episode {idx+1} (RECENTER_FROM_NEGATIVE): early exit at e={exit_e:.4f}")

print(f"\n  Correct exits (reached inner band): {correct_exits}")
print(f"  Early exits (still > outer threshold): {early_exits}")

principle3_pass = early_exits == 0
print(f"\n  Principle 3: {'PASS' if principle3_pass else 'FAIL'}")

# Principle 4: Check if pitch safety gates interrupt recenter
print("\n--- Principle 4: Pitch safety gates should not interrupt RECENTER ---")

# Count how many RECENTER steps are in pitch danger
pitch_danger = np.abs(pitch) > 0.1  # rad
pitch_caution = np.abs(pitch) > 0.05  # rad

recenter_pitch_danger = np.sum(recenter_mask & pitch_danger)
recenter_pitch_caution = np.sum(recenter_mask & pitch_caution)
total_recenter = np.sum(recenter_mask)

print(f"RECENTER steps in pitch danger (|pitch| > 0.1 rad): {recenter_pitch_danger} ({100*recenter_pitch_danger/total_recenter:.1f}%)")
print(f"RECENTER steps in pitch caution (|pitch| > 0.05 rad): {recenter_pitch_caution} ({100*recenter_pitch_caution/total_recenter:.1f}%)")
print(f"Total RECENTER steps: {total_recenter}")

# Check if RECENTER exits correlate with pitch sign changes
print("\n--- Principle 5: Check for opposite direction cycling ---")
pos_recenter_count = sum(1 for start, end in recenter_episodes if state[start] == 'RECENTER_FROM_POSITIVE')
neg_recenter_count = sum(1 for start, end in recenter_episodes if state[start] == 'RECENTER_FROM_NEGATIVE')

print(f"RECENTER_FROM_POSITIVE episodes: {pos_recenter_count}")
print(f"RECENTER_FROM_NEGATIVE episodes: {neg_recenter_count}")

principle5_pass = neg_recenter_count > 0
print(f"\n  Principle 5 (bidirectional recenter): {'PASS' if principle5_pass else 'FAIL'}")

# Overall classification
print("\n=== Overall Classification ===")
all_pass = principle1_pass and principle2_pass and principle3_pass and principle5_pass

if all_pass:
    classification = "APCR1I_PRINCIPLE_SATISFIED"
elif early_exits > 0:
    classification = "APCR1I_PRINCIPLE_FAILS_EARLY_EXIT"
elif not principle5_pass:
    classification = "APCR1I_PRINCIPLE_FAILS_NO_OPPOSITE_CYCLE"
elif np.abs(tau).max() < 1.5:
    classification = "APCR1I_PRINCIPLE_FAILS_TORQUE_CAP"
else:
    classification = "APCR1I_PRINCIPLE_INCONCLUSIVE"

print(f"Classification: {classification}")

# Save results
results = {
    'profile': 'APCR1i_support_hysteresis_recenter',
    'steps': 1000,
    'principle1_e_positive_large': {
        'steps': int(np.sum(e_positive_large)) if np.any(e_positive_large) else 0,
        'recenter_from_positive_pct': float(100*np.sum(recenter_from_positive)/np.sum(e_positive_large)) if np.any(e_positive_large) else 0.0,
        'correct_torque_sign_pct': float(100*np.sum(correct_sign)/len(tau_when_positive)) if np.any(e_positive_large) else 0.0,
        'pass': bool(principle1_pass)
    },
    'principle2_e_negative_large': {
        'steps': int(np.sum(e_negative_large)) if np.any(e_negative_large) else 0,
        'recenter_from_negative_pct': float(100*np.sum(recenter_from_negative)/np.sum(e_negative_large)) if np.any(e_negative_large) else 0.0,
        'correct_torque_sign_pct': float(100*np.sum(correct_sign)/len(tau_when_negative)) if np.any(e_negative_large) else 0.0,
        'pass': bool(principle2_pass)
    },
    'principle3_no_early_exit': {
        'total_recenter_episodes': len(recenter_episodes),
        'correct_exits': correct_exits,
        'early_exits': early_exits,
        'pass': bool(principle3_pass)
    },
    'principle4_pitch_gate_interrupt': {
        'recenter_steps_in_pitch_danger': int(recenter_pitch_danger),
        'recenter_steps_in_pitch_caution': int(recenter_pitch_caution),
        'total_recenter_steps': int(total_recenter),
        'pitch_danger_pct': float(100*recenter_pitch_danger/total_recenter) if total_recenter > 0 else 0.0
    },
    'principle5_bidirectional': {
        'pos_recenter_count': pos_recenter_count,
        'neg_recenter_count': neg_recenter_count,
        'pass': bool(principle5_pass)
    },
    'classification': classification
}

with open(OUTPUT_DIR / 'apcr1i_1000_principle_verification.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / 'apcr1i_1000_principle_verification.json'}")

print("\n=== Phase 6 Complete ===")
