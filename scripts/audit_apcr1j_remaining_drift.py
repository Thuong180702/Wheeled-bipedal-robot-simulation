#!/usr/bin/env python3
"""
APCR1j root cause audit - Phase 1

Analyzes why APCR1j still reaches max_e = 0.1826 m despite 2.0 Nm APCR torque.

Use correct physical drift only:
1. active_pitch_crossing_signed_error_m
2. sagittal_position_error_m
3. support_position_error_m
4. hip_yaw_comp_support_error_m

Classification:
- APCR1J_REMAINING_DRIFT_FROM_LATE_ENTRY
- APCR1J_REMAINING_DRIFT_FROM_TORQUE_TRANSMISSION_LOSS
- APCR1J_REMAINING_DRIFT_FROM_INSUFFICIENT_FINAL_WHEEL_TORQUE
- APCR1J_REMAINING_DRIFT_FROM_RATE_DELAY
- APCR1J_REMAINING_DRIFT_FROM_GATE_INTERRUPTION
- APCR1J_REMAINING_DRIFT_FROM_MIXED_CAUSES
- APCR1J_REMAINING_DRIFT_INCONCLUSIVE
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def load_telemetry(csv_path):
    """Load telemetry CSV."""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")
    return df

def find_entry_steps(df, error_col='active_pitch_crossing_signed_error_m'):
    """Find first step where error exceeds each threshold."""
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    entries = {}
    e = df[error_col].values

    for thresh in thresholds:
        mask = np.abs(e) > thresh
        if mask.any():
            first_step = df.loc[mask, 'source_step_index'].min()
            entries[thresh] = int(first_step)
        else:
            entries[thresh] = None

    return entries

def analyze_torque_at_thresholds(df, error_col='active_pitch_crossing_signed_error_m'):
    """Analyze APCR torque at each error threshold."""
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
    results = []

    e = df[error_col].values
    apc_tau = df['active_pitch_crossing_tau'].values
    apc_raw_tau = df['active_pitch_crossing_raw_tau'].values
    final_tau = df['final_wheel_tau_with_apc'].values
    wheel_vel = df['wheel_vel_mean_rad_s'].values
    steps = df['source_step_index'].values

    for thresh in thresholds:
        # Find steps where |e| >= thresh
        mask = np.abs(e) >= thresh
        if mask.any():
            idx = np.where(mask)[0]
            first_idx = idx[0]

            results.append({
                'threshold': thresh,
                'first_step': int(steps[first_idx]),
                'e_at_entry': float(e[first_idx]),
                'apc_tau_at_entry': float(apc_tau[first_idx]),
                'apc_raw_tau_at_entry': float(apc_raw_tau[first_idx]),
                'final_tau_at_entry': float(final_tau[first_idx]),
                'wheel_vel_at_entry': float(wheel_vel[first_idx]),
                'max_apc_tau': float(np.max(np.abs(apc_tau[mask]))),
                'max_final_tau': float(np.max(np.abs(final_tau[mask]))),
                'steps_with_e_above': int(mask.sum()),
            })

    return results

def analyze_torque_transmission(df):
    """Trace torque transmission path."""
    apc_tau = df['active_pitch_crossing_tau'].values
    apc_raw_tau = df['active_pitch_crossing_raw_tau'].values
    final_tau = df['final_wheel_tau_with_apc'].values
    steps = df['source_step_index'].values

    # Find steps where APCR is active
    apc_active = np.abs(apc_tau) > 0.01

    results = {
        'apc_active_steps': int(apc_active.sum()),
        'apc_max_tau': float(np.max(np.abs(apc_tau))),
        'apc_raw_max_tau': float(np.max(np.abs(apc_raw_tau))),
        'final_max_tau': float(np.max(np.abs(final_tau))),
        'final_to_apc_ratio': None,
        'apc_reaches_2nm': bool(np.max(np.abs(apc_tau)) >= 1.99),
        'final_reaches_2nm': bool(np.max(np.abs(final_tau)) >= 1.99),
        'apc_at_2nm_steps': [],
        'final_at_2nm_steps': [],
    }

    if results['apc_max_tau'] > 0.01:
        results['final_to_apc_ratio'] = float(results['final_max_tau'] / results['apc_max_tau'])

    # Find steps where APCR reaches 2.0 Nm
    for i, tau in enumerate(apc_tau):
        if abs(tau) >= 1.99:
            results['apc_at_2nm_steps'].append(int(steps[i]))

    for i, tau in enumerate(final_tau):
        if abs(tau) >= 1.99:
            results['final_at_2nm_steps'].append(int(steps[i]))

    results['apc_at_2nm_count'] = len(results['apc_at_2nm_steps'])
    results['final_at_2nm_count'] = len(results['final_at_2nm_steps'])

    return results

def analyze_hysteresis_episodes(df):
    """Analyze hysteresis recenter episodes."""
    state_col = 'active_pitch_crossing_hysteresis_state'
    e_col = 'active_pitch_crossing_signed_error_m'
    apc_tau_col = 'active_pitch_crossing_tau'
    final_tau_col = 'final_wheel_tau_with_apc'
    pitch_col = 'euler_pitch_y'

    states = df[state_col].values
    e = df[e_col].values
    apc_tau = df[apc_tau_col].values
    final_tau = df[final_tau_col].values
    pitch = df[pitch_col].values
    steps = df['source_step_index'].values

    episodes = []
    current_episode = None

    for i in range(len(df)):
        state = states[i]
        step = steps[i]

        if 'RECENTER' in str(state):
            if current_episode is None:
                # Start new episode
                current_episode = {
                    'id': len(episodes) + 1,
                    'state': state,
                    'entry_step': int(step),
                    'entry_e': float(e[i]),
                    'entry_pitch': float(pitch[i]),
                    'entry_apc_tau': float(apc_tau[i]),
                    'max_e': float(e[i]),
                    'min_e': float(e[i]),
                    'max_apc_tau': float(abs(apc_tau[i])),
                    'max_final_tau': float(abs(final_tau[i])),
                    'steps': [],
                }
            else:
                # Update current episode
                current_episode['max_e'] = max(current_episode['max_e'], float(e[i]))
                current_episode['min_e'] = min(current_episode['min_e'], float(e[i]))
                current_episode['max_apc_tau'] = max(current_episode['max_apc_tau'], float(abs(apc_tau[i])))
                current_episode['max_final_tau'] = max(current_episode['max_final_tau'], float(abs(final_tau[i])))

            current_episode['steps'].append(int(step))
        else:
            if current_episode is not None:
                # End episode
                current_episode['exit_step'] = int(step)
                current_episode['exit_e'] = float(e[i])
                current_episode['exit_pitch'] = float(pitch[i])
                current_episode['exit_apc_tau'] = float(apc_tau[i])
                current_episode['duration'] = len(current_episode['steps'])
                del current_episode['steps']
                episodes.append(current_episode)
                current_episode = None

    # Handle incomplete last episode
    if current_episode is not None:
        current_episode['exit_step'] = int(steps[-1])
        current_episode['exit_e'] = float(e[-1])
        current_episode['duration'] = len(current_episode['steps'])
        del current_episode['steps']
        episodes.append(current_episode)

    return episodes

def analyze_gate_interference(df):
    """Check if safety gates block APCR during critical moments."""
    e_col = 'active_pitch_crossing_signed_error_m'
    apc_active_col = 'active_pitch_crossing_active'
    gate_reason_col = 'active_pitch_crossing_gate_reason'
    pitch_safe_col = 'active_pitch_crossing_pitch_safe'
    contact_safe_col = 'active_pitch_crossing_contact_safe'
    height_safe_col = 'active_pitch_crossing_height_safe'
    roll_safe_col = 'active_pitch_crossing_roll_safe'

    e = df[e_col].values
    apc_active = df[apc_active_col].values
    gate_reason = df[gate_reason_col].values
    pitch_safe = df[pitch_safe_col].values
    contact_safe = df[contact_safe_col].values
    height_safe = df[height_safe_col].values
    roll_safe = df[roll_safe_col].values

    # Find large error steps where APCR should be active
    large_error = np.abs(e) > 0.08
    results = {
        'large_error_steps': int(large_error.sum()),
        'apc_inactive_during_large_error': 0,
        'gate_reasons': {},
        'pitch_blocked': 0,
        'contact_blocked': 0,
        'height_blocked': 0,
        'roll_blocked': 0,
    }

    for i in range(len(df)):
        if large_error[i]:
            if not apc_active[i]:
                results['apc_inactive_during_large_error'] += 1
                reason = str(gate_reason[i])
                results['gate_reasons'][reason] = results['gate_reasons'].get(reason, 0) + 1

            if not pitch_safe[i]:
                results['pitch_blocked'] += 1
            if not contact_safe[i]:
                results['contact_blocked'] += 1
            if not height_safe[i]:
                results['height_blocked'] += 1
            if not roll_safe[i]:
                results['roll_blocked'] += 1

    return results

def compute_drift_metrics(df, error_col='active_pitch_crossing_signed_error_m'):
    """Compute drift metrics."""
    e = df[error_col].values

    return {
        'min_e': float(np.min(e)),
        'max_e': float(np.max(e)),
        'P2P': float(np.max(e) - np.min(e)),
        'max_abs': float(np.max(np.abs(e))),
        'mean_e': float(np.mean(e)),
        'abs_mean_e': float(np.mean(np.abs(e))),
        'final_e': float(e[-1]),
        'outside_0.05': int((np.abs(e) > 0.05).sum()),
        'outside_0.08': int((np.abs(e) > 0.08).sum()),
        'outside_0.10': int((np.abs(e) > 0.10).sum()),
        'outside_0.12': int((np.abs(e) > 0.12).sum()),
        'outside_0.15': int((np.abs(e) > 0.15).sum()),
        'outside_0.05_pct': float((np.abs(e) > 0.05).sum() / len(e) * 100),
        'outside_0.08_pct': float((np.abs(e) > 0.08).sum() / len(e) * 100),
        'outside_0.10_pct': float((np.abs(e) > 0.10).sum() / len(e) * 100),
        'outside_0.12_pct': float((np.abs(e) > 0.12).sum() / len(e) * 100),
        'outside_0.15_pct': float((np.abs(e) > 0.15).sum() / len(e) * 100),
    }

def main():
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'outputs' / 'step_e_extreme_support_fix_eval' / 'active_pitch_crossing' / 'apcr1j_low_0p300_1000' / 'telemetry_1781061505.csv'

    df = load_telemetry(str(csv_path))

    print("\n" + "="*60)
    print("A. ENTRY TIMING ANALYSIS")
    print("="*60)

    error_col = 'active_pitch_crossing_signed_error_m'
    entries = find_entry_steps(df, error_col)
    for thresh, step in entries.items():
        print(f"  e > {thresh:.2f}: first at step {step}")

    print("\n" + "="*60)
    print("B. TORQUE RESPONSE AT THRESHOLDS")
    print("="*60)

    torque_analysis = analyze_torque_at_thresholds(df, error_col)
    for r in torque_analysis:
        print(f"\n  Threshold {r['threshold']:.2f} m:")
        print(f"    First step: {r['first_step']}")
        print(f"    e at entry: {r['e_at_entry']:.4f} m")
        print(f"    APCR tau at entry: {r['apc_tau_at_entry']:.4f} Nm")
        print(f"    Final tau at entry: {r['final_tau_at_entry']:.4f} Nm")
        print(f"    Max APCR tau: {r['max_apc_tau']:.4f} Nm")
        print(f"    Max final tau: {r['max_final_tau']:.4f} Nm")
        print(f"    Steps above threshold: {r['steps_with_e_above']}")

    print("\n" + "="*60)
    print("C. TORQUE TRANSMISSION PATH")
    print("="*60)

    transmission = analyze_torque_transmission(df)
    print(f"  APCR active steps: {transmission['apc_active_steps']}")
    print(f"  APCR max tau: {transmission['apc_max_tau']:.4f} Nm")
    print(f"  APCR raw max tau: {transmission['apc_raw_max_tau']:.4f} Nm")
    print(f"  Final max tau: {transmission['final_max_tau']:.4f} Nm")
    print(f"  Final/APCR ratio: {transmission['final_to_apc_ratio']:.4f}")
    print(f"  APCR reaches 2.0 Nm: {transmission['apc_reaches_2nm']}")
    print(f"  Final reaches 2.0 Nm: {transmission['final_reaches_2nm']}")
    print(f"  APCR at 2.0 Nm count: {transmission['apc_at_2nm_count']}")
    print(f"  Final at 2.0 Nm count: {transmission['final_at_2nm_count']}")

    if transmission['apc_at_2nm_count'] > 0:
        print(f"  APCR at 2.0 Nm steps: {transmission['apc_at_2nm_steps'][:10]}...")
    if transmission['final_at_2nm_count'] > 0:
        print(f"  Final at 2.0 Nm steps: {transmission['final_at_2nm_steps'][:10]}...")

    print("\n" + "="*60)
    print("D. HYSTERESIS EPISODES")
    print("="*60)

    episodes = analyze_hysteresis_episodes(df)
    print(f"  Total episodes: {len(episodes)}")
    for ep in episodes:
        print(f"\n  Episode {ep['id']}: {ep['state']}")
        print(f"    Entry: step {ep['entry_step']}, e={ep['entry_e']:.4f} m")
        print(f"    Exit: step {ep['exit_step']}, e={ep['exit_e']:.4f} m")
        print(f"    Duration: {ep['duration']} steps")
        print(f"    Max e: {ep['max_e']:.4f} m")
        print(f"    Max APCR tau: {ep['max_apc_tau']:.4f} Nm")
        print(f"    Max final tau: {ep['max_final_tau']:.4f} Nm")

    print("\n" + "="*60)
    print("E. GATE INTERFERENCE")
    print("="*60)

    gate = analyze_gate_interference(df)
    print(f"  Large error steps (|e|>0.08): {gate['large_error_steps']}")
    print(f"  APCR inactive during large error: {gate['apc_inactive_during_large_error']}")
    print(f"  Pitch blocked: {gate['pitch_blocked']}")
    print(f"  Contact blocked: {gate['contact_blocked']}")
    print(f"  Height blocked: {gate['height_blocked']}")
    print(f"  Roll blocked: {gate['roll_blocked']}")
    print(f"  Gate reasons: {gate['gate_reasons']}")

    print("\n" + "="*60)
    print("F. DRIFT METRICS")
    print("="*60)

    metrics = compute_drift_metrics(df, error_col)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Determine classification
    print("\n" + "="*60)
    print("CLASSIFICATION")
    print("="*60)

    classification = "APCR1J_REMAINING_DRIFT_FROM_MIXED_CAUSES"
    reasons = []

    # Check late entry
    if entries.get(0.08) is not None and entries.get(0.05) is not None:
        if entries[0.08] < 200:
            reasons.append("entry_at_0.08_allowed_momentum")

    # Check torque transmission loss
    if transmission['final_to_apc_ratio'] is not None:
        if transmission['final_to_apc_ratio'] < 0.80:
            reasons.append(f"torque_transmission_loss_ratio_{transmission['final_to_apc_ratio']:.2f}")

    # Check gate interference
    if gate['apc_inactive_during_large_error'] > 50:
        reasons.append(f"gate_blocked_{gate['apc_inactive_during_large_error']}_steps")

    # Check final tau never reaches 2.0
    if not transmission['final_reaches_2nm']:
        reasons.append("final_tau_blocked_from_2nm")

    if transmission['apc_reaches_2nm'] and not transmission['final_reaches_2nm']:
        classification = "APCR1J_REMAINING_DRIFT_FROM_TORQUE_TRANSMISSION_LOSS"
    elif gate['apc_inactive_during_large_error'] > 100:
        classification = "APCR1J_REMAINING_DRIFT_FROM_GATE_INTERRUPTION"
    elif transmission['final_to_apc_ratio'] is not None and transmission['final_to_apc_ratio'] < 0.70:
        classification = "APCR1J_REMAINING_DRIFT_FROM_INSUFFICIENT_FINAL_WHEEL_TORQUE"
    elif entries.get(0.08) is not None and entries[0.08] < 100:
        classification = "APCR1J_REMAINING_DRIFT_FROM_LATE_ENTRY"

    print(f"  Classification: {classification}")
    print(f"  Reasons: {reasons}")

    # Save results
    output_dir = base_dir / 'outputs' / 'step_e_extreme_support_fix_eval' / 'active_pitch_crossing'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'classification': classification,
        'reasons': reasons,
        'entry_timing': entries,
        'torque_at_thresholds': torque_analysis,
        'torque_transmission': transmission,
        'episodes': episodes,
        'gate_interference': gate,
        'drift_metrics': metrics,
    }

    audit_path = output_dir / 'apcr1j_remaining_drift_root_cause_audit.json'
    with open(audit_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {audit_path}")

    # Create threshold event table
    threshold_events = []
    for thresh, step in entries.items():
        if step is not None:
            idx = df[df['source_step_index'] == step].index[0]
            threshold_events.append({
                'threshold_m': thresh,
                'first_step': step,
                'e': float(df.loc[idx, error_col]),
                'apc_tau': float(df.loc[idx, 'active_pitch_crossing_tau']),
                'final_tau': float(df.loc[idx, 'final_wheel_tau_with_apc']),
                'hysteresis_state': str(df.loc[idx, 'active_pitch_crossing_hysteresis_state']),
            })

    events_df = pd.DataFrame(threshold_events)
    events_path = output_dir / 'apcr1j_threshold_event_table.csv'
    events_df.to_csv(events_path, index=False)
    print(f"Saved: {events_path}")

    return results

if __name__ == '__main__':
    main()
