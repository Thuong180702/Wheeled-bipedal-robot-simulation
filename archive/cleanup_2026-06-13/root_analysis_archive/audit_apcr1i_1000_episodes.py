"""
APCR1i 1000-step episode-level state-machine audit using correct column.
Uses active_pitch_crossing_hysteresis_state column which has the actual APCR1i state.
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

# Use the correct APCR1i hysteresis state column (from APC subsystem)
STATE_COL = 'active_pitch_crossing_hysteresis_state'
STATE_ID_COL = 'active_pitch_crossing_hysteresis_state_id'
ENTRY_COUNT_COL = 'active_pitch_crossing_hysteresis_entry_count'
EXIT_COUNT_COL = 'active_pitch_crossing_hysteresis_exit_count'
ENTRY_E_COL = 'active_pitch_crossing_hysteresis_entry_e'
EXIT_E_COL = 'active_pitch_crossing_hysteresis_exit_e'
INNER_EXIT_COL = 'active_pitch_crossing_hysteresis_inner_exit_m'
OPP_RELEASE_COL = 'active_pitch_crossing_hysteresis_opposite_release_m'
EMERGENCY_COL = 'active_pitch_crossing_hysteresis_emergency_active'

# Primary drift column
DRIFT_COL = 'support_position_error_m'

# Extract data
steps = df['step'].values
e = df[DRIFT_COL].values
state = df[STATE_COL].values
state_id = df[STATE_ID_COL].values if STATE_ID_COL in df.columns else None

# Compute e_dot (support velocity)
e_dot = np.gradient(e, steps)

# Check APCR1i hysteresis telemetry
enabled = df.get('active_pitch_crossing_hysteresis_enabled', np.ones(len(df))).values
entry_count = df.get(ENTRY_COUNT_COL, np.zeros(len(df))).values
exit_count = df.get(EXIT_COUNT_COL, np.zeros(len(df))).values
entry_e_vals = df.get(ENTRY_E_COL, np.full(len(df), np.nan)).values
exit_e_vals = df.get(EXIT_E_COL, np.full(len(df), np.nan)).values
inner_exit = df.get(INNER_EXIT_COL, np.full(len(df), 0.03)).values
opp_release = df.get(OPP_RELEASE_COL, np.full(len(df), 0.03)).values
emergency_active = df.get(EMERGENCY_COL, np.zeros(len(df))).values

# Get pitch and contact info
pitch_col = 'pitch_x'
pitch = df[pitch_col].values if pitch_col in df.columns else np.zeros(len(df))
contact_state = df.get('contact_supervisor_state', np.ones(len(df))).values

# APCR tau columns
RAW_TAU_COL = 'active_pitch_crossing_raw_tau'
TAU_COL = 'active_pitch_crossing_tau'
MAX_TAU_COL = 'active_pitch_crossing_max_tau'
apcr_raw_tau = df.get(RAW_TAU_COL, np.zeros(len(df))).values
apcr_tau = df.get(TAU_COL, np.zeros(len(df))).values
apcr_max_tau = df.get(MAX_TAU_COL, np.full(len(df), 1.5)).values

# Gate info
GATE_COL = 'active_pitch_crossing_gate_reason'
HEIGHT_SAFE_COL = 'active_pitch_crossing_height_safe'
CONTACT_SAFE_COL = 'active_pitch_crossing_contact_safe'
gate_reason = df.get(GATE_COL, [''] * len(df)).values if GATE_COL in df.columns else [''] * len(df)
height_safe = df.get(HEIGHT_SAFE_COL, np.ones(len(df))).values if HEIGHT_SAFE_COL in df.columns else np.ones(len(df))
contact_safe = df.get(CONTACT_SAFE_COL, np.ones(len(df))).values if CONTACT_SAFE_COL in df.columns else np.ones(len(df))

print(f"\nState values: {np.unique(state)}")
print(f"Enabled: {np.unique(enabled)}")
print(f"Entry count range: {entry_count.min():.0f} - {entry_count.max():.0f}")
print(f"Exit count range: {exit_count.min():.0f} - {exit_count.max():.0f}")

# Detect episodes
def detect_episodes(steps, state, e, e_dot, pitch, contact_state):
    """Detect state machine episodes."""
    episodes = []
    current_episode = None

    for i in range(len(state)):
        s = state[i]

        if current_episode is None:
            # Start new episode if entering a recenter state
            if 'RECENTER' in s:
                current_episode = {
                    'start_step': int(steps[i]),
                    'start_idx': i,
                    'state': s,
                    'entry_e': float(e[i]),
                    'entry_e_dot': float(e_dot[i]),
                    'entry_pitch': float(pitch[i]),
                    'entry_contact': str(contact_state[i]) if contact_state[i] is not None else 'unknown',
                    'entry_tau': float(apcr_raw_tau[i]),
                    'min_e': float(e[i]),
                    'max_e': float(e[i]),
                    'min_e_dot': float(e_dot[i]),
                    'max_e_dot': float(e_dot[i]),
                    'e_dot_reversed': False,
                    'e_dot_reversal_step': None,
                    'reached_inner_band': False,
                    'crossed_zero': False,
                    'crossed_opposite': False,
                    'inner_exit_threshold': float(inner_exit[i]),
                    'opposite_release_threshold': float(opp_release[i]),
                    'emergency_active': bool(emergency_active[i]),
                    'max_tau_in_episode': float(apcr_raw_tau[i]),
                }
        else:
            # Continue episode
            current_episode['min_e'] = min(current_episode['min_e'], float(e[i]))
            current_episode['max_e'] = max(current_episode['max_e'], float(e[i]))
            current_episode['min_e_dot'] = min(current_episode['min_e_dot'], float(e_dot[i]))
            current_episode['max_e_dot'] = max(current_episode['max_e_dot'], float(e_dot[i]))
            current_episode['max_tau_in_episode'] = max(current_episode['max_tau_in_episode'], float(apcr_raw_tau[i]))
            current_episode['emergency_active'] = current_episode['emergency_active'] or bool(emergency_active[i])

            # Check if crossed zero
            if current_episode['state'] == 'RECENTER_FROM_POSITIVE' and e[i] <= 0:
                current_episode['crossed_zero'] = True

            # Check if crossed opposite threshold
            if current_episode['state'] == 'RECENTER_FROM_POSITIVE' and e[i] < -current_episode['opposite_release_threshold']:
                current_episode['crossed_opposite'] = True
            if current_episode['state'] == 'RECENTER_FROM_NEGATIVE' and e[i] > current_episode['opposite_release_threshold']:
                current_episode['crossed_opposite'] = True

            # Check if reached inner band
            inner_thresh = current_episode['inner_exit_threshold']
            if current_episode['state'] == 'RECENTER_FROM_POSITIVE' and e[i] <= inner_thresh:
                current_episode['reached_inner_band'] = True
            if current_episode['state'] == 'RECENTER_FROM_NEGATIVE' and e[i] >= -inner_thresh:
                current_episode['reached_inner_band'] = True

            # Check for exit
            if s != current_episode['state']:
                # End episode
                current_episode['end_step'] = int(steps[i-1])
                current_episode['end_idx'] = i - 1
                current_episode['duration'] = current_episode['end_step'] - current_episode['start_step']
                current_episode['exit_e'] = float(e[i-1])
                current_episode['exit_e_dot'] = float(e_dot[i-1])
                current_episode['exit_pitch'] = float(pitch[i-1])
                current_episode['exit_contact'] = str(contact_state[i-1]) if contact_state[i-1] is not None else 'unknown'
                current_episode['exit_tau'] = float(apcr_raw_tau[i-1])
                current_episode['exit_gate_reason'] = str(gate_reason[i-1]) if gate_reason[i-1] is not None else ''

                episodes.append(current_episode)
                current_episode = None

    # Close final episode
    if current_episode is not None:
        current_episode['end_step'] = int(steps[-1])
        current_episode['end_idx'] = len(steps) - 1
        current_episode['duration'] = current_episode['end_step'] - current_episode['start_step']
        current_episode['exit_e'] = float(e[-1])
        current_episode['exit_e_dot'] = float(e_dot[-1])
        current_episode['exit_pitch'] = float(pitch[-1])
        current_episode['exit_contact'] = str(contact_state[-1]) if contact_state[-1] is not None else 'unknown'
        current_episode['exit_tau'] = float(apcr_raw_tau[-1])
        episodes.append(current_episode)

    return episodes

episodes = detect_episodes(steps, state, e, e_dot, pitch, contact_state)

print(f"\n=== APCR1i Hysteresis State Machine Episodes ===")
print(f"Total episodes detected: {len(episodes)}")

# State distribution
state_counts = {}
for ep in episodes:
    st = ep['state']
    state_counts[st] = state_counts.get(st, 0) + 1
print(f"State distribution: {state_counts}")

# Count RECENTER episodes
recenter_count = sum(1 for ep in episodes if 'RECENTER' in ep['state'])
neutral_count = sum(1 for ep in episodes if ep['state'] == 'NEUTRAL')
print(f"RECENTER episodes: {recenter_count}")
print(f"NEUTRAL episodes: {neutral_count}")

# Print each episode
print("\n=== Episode Details ===")
for i, ep in enumerate(episodes):
    print(f"\nEpisode {i+1}:")
    print(f"  State: {ep['state']}")
    print(f"  Steps: {ep['start_step']} - {ep['end_step']} ({ep['duration']} steps)")
    print(f"  Entry: e={ep['entry_e']:.4f}, e_dot={ep['entry_e_dot']:.4f}, pitch={ep['entry_pitch']:.4f}")
    print(f"  Exit: e={ep['exit_e']:.4f}, e_dot={ep['exit_e_dot']:.4f}, pitch={ep['exit_pitch']:.4f}")
    print(f"  Min/Max e: {ep['min_e']:.4f} / {ep['max_e']:.4f}")
    print(f"  Reached inner band: {ep['reached_inner_band']} (thresh={ep['inner_exit_threshold']:.3f})")
    print(f"  Crossed zero: {ep['crossed_zero']}")
    print(f"  Crossed opposite: {ep['crossed_opposite']} (thresh={ep['opposite_release_threshold']:.3f})")
    print(f"  Emergency active: {ep['emergency_active']}")
    print(f"  Max tau: {ep['max_tau_in_episode']:.4f}")
    print(f"  Entry tau: {ep['entry_tau']:.4f}, Exit tau: {ep['exit_tau']:.4f}")
    print(f"  Exit gate reason: {ep.get('exit_gate_reason', 'N/A')}")

# Classification
classifications = []
for i, ep in enumerate(episodes):
    state = ep['state']

    # Check if exited correctly
    exited_correctly = False
    exited_early = False
    gate_interrupt = False
    no_opposite_cycle = False

    inner_thresh = ep['inner_exit_threshold']

    if 'RECENTER' in state:
        if state == 'RECENTER_FROM_POSITIVE':
            # Should hold until e <= +inner_thresh or e < -opposite_thresh
            if ep['exit_e'] <= inner_thresh:
                exited_correctly = True
            elif ep['crossed_opposite']:
                exited_correctly = True
            elif ep['exit_e'] > 0.08 and not ep['reached_inner_band']:
                exited_early = True
        elif state == 'RECENTER_FROM_NEGATIVE':
            if ep['exit_e'] >= -inner_thresh:
                exited_correctly = True
            elif ep['crossed_opposite']:
                exited_correctly = True
            elif ep['exit_e'] < -0.08 and not ep['reached_inner_band']:
                exited_early = True

        # Check gate reason
        gate = ep.get('exit_gate_reason', '')
        if 'safety' in gate.lower() or 'pitch' in gate.lower() or 'height' in gate.lower():
            gate_interrupt = True

        # Check if never entered opposite state
        if 'RECENTER_FROM_POSITIVE' in state_counts and 'RECENTER_FROM_NEGATIVE' in state_counts:
            if state_counts.get('RECENTER_FROM_NEGATIVE', 0) == 0:
                no_opposite_cycle = True

    if exited_early:
        classifications.append('APCR1I_EXITS_TOO_EARLY_BEFORE_INNER_BAND')
    elif gate_interrupt:
        classifications.append('APCR1I_GATE_INTERRUPTS_RECENTER')
    elif no_opposite_cycle:
        classifications.append('APCR1I_PRINCIPLE_FAILS_NO_OPPOSITE_CYCLE')
    elif exited_correctly:
        classifications.append('APCR1I_HYSTERESIS_HOLDS_CORRECTLY_BUT_INSUFFICIENT_AUTHORITY')
    else:
        classifications.append('APCR1I_HYSTERESIS_EPISODE_AUDIT_INCONCLUSIVE')

# Summary
class_counts = {}
for c in classifications:
    class_counts[c] = class_counts.get(c, 0) + 1

print("\n=== Episode Classification Summary ===")
for c, count in class_counts.items():
    print(f"  {c}: {count}")

# Check for episodes that stayed too long
print("\n=== Long RECENTER Episode Analysis ===")
for i, ep in enumerate(episodes):
    if 'RECENTER' in ep['state'] and ep['duration'] > 100:
        print(f"Episode {i+1}: {ep['state']} lasted {ep['duration']} steps")
        print(f"  Entry e={ep['entry_e']:.4f}, Exit e={ep['exit_e']:.4f}")
        print(f"  Min/Max e: {ep['min_e']:.4f} / {ep['max_e']:.4f}")
        print(f"  Reached inner band: {ep['reached_inner_band']}")
        print(f"  Crossed opposite: {ep['crossed_opposite']}")
        print(f"  Max tau: {ep['max_tau_in_episode']:.4f}")

# Save episodes
episodes_json = []
for ep in episodes:
    ep_copy = {}
    for k, v in ep.items():
        if isinstance(v, (np.bool_,)):
            ep_copy[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            ep_copy[k] = int(v)
        elif isinstance(v, (np.floating,)):
            ep_copy[k] = float(v)
        else:
            ep_copy[k] = v
    episodes_json.append(ep_copy)

with open(OUTPUT_DIR / 'apcr1i_1000_episode_audit.json', 'w') as f:
    json.dump({
        'profile': 'APCR1i_support_hysteresis_recenter',
        'total_episodes': len(episodes),
        'state_distribution': state_counts,
        'classification_summary': class_counts,
        'episodes': episodes_json
    }, f, indent=2)
print(f"\nSaved episodes to {OUTPUT_DIR / 'apcr1i_1000_episode_audit.json'}")

# Save CSV table
ep_rows = []
for i, ep in enumerate(episodes):
    ep_rows.append({
        'episode_id': i + 1,
        'state': ep['state'],
        'entry_step': ep['start_step'],
        'exit_step': ep['end_step'],
        'duration': ep['duration'],
        'entry_e': ep['entry_e'],
        'exit_e': ep['exit_e'],
        'min_e': ep['min_e'],
        'max_e': ep['max_e'],
        'entry_e_dot': ep['entry_e_dot'],
        'e_dot_reversed': ep['e_dot_reversed'],
        'e_dot_reversal_step': ep['e_dot_reversal_step'],
        'reached_inner_band': ep['reached_inner_band'],
        'crossed_zero': ep['crossed_zero'],
        'crossed_opposite': ep['crossed_opposite'],
        'entry_pitch': ep['entry_pitch'],
        'exit_pitch': ep['exit_pitch'],
        'entry_tau': ep['entry_tau'],
        'exit_tau': ep['exit_tau'],
        'max_tau': ep['max_tau_in_episode'],
        'emergency_active': ep['emergency_active'],
        'gate_reason': ep.get('exit_gate_reason', ''),
        'classification': classifications[i]
    })

ep_df = pd.DataFrame(ep_rows)
ep_df.to_csv(OUTPUT_DIR / 'apcr1i_1000_episode_table.csv', index=False)
print(f"Saved episode table to {OUTPUT_DIR / 'apcr1i_1000_episode_table.csv'}")

print("\n=== Phase 4 Complete ===")
