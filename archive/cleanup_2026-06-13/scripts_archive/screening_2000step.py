#!/usr/bin/env python3
"""HY2-DIV 2000-step screening evaluation.

Runs 2000-step simulations for all candidates at all heights.
Collects comprehensive metrics for candidate selection.

Usage:
    python scripts/screening_2000step.py --candidates A0 A1 A2 B1 --heights nominal low high
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

OUTPUT_DIR = Path("outputs/hip_yaw_divergence_fix_authority_eval/screening_2000")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate definitions
CANDIDATES = {
    "A0": {"k": 5.0, "kd": 1.0, "tau_max": 0.5, "z_low": 0.300, "z_high": 0.393, "group": "A"},
    "A1": {"k": 5.0, "kd": 1.0, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.393, "group": "A"},
    "A2": {"k": 5.0, "kd": 1.0, "tau_max": 2.0, "z_low": 0.300, "z_high": 0.393, "group": "A"},
    "A3": {"k": 7.5, "kd": 1.5, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.393, "group": "A"},
    "B1": {"k": 5.0, "kd": 1.0, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.500, "group": "B"},
    "B2": {"k": 5.0, "kd": 1.0, "tau_max": 2.0, "z_low": 0.300, "z_high": 0.500, "group": "B"},
    "B3": {"k": 7.5, "kd": 1.5, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.500, "group": "B"},
    "no_hy2": {"k": 0.0, "kd": 0.0, "tau_max": 0.5, "z_low": 0.300, "z_high": 0.393, "group": "baseline"},
}

# Height configurations
HEIGHT_CONFIGS = {
    "nominal": {"setup": None, "target_z": 0.404},
    "low_0p300": {"setup": "outputs/physical_target_height_setups/low_0p300_setup.json", "target_z": 0.300},
    "high_0p480": {"setup": "outputs/physical_target_height_setups/high_0p480_setup.json", "target_z": 0.480},
}

# Baselines (post-sign-fix without HY2-DIV)
BASELINES = {
    "nominal": {"div_rms": 0.2446, "div_max": 0.5},
    "low_0p300": {"div_rms": 0.3690, "div_max": 0.8},
    "high_0p480": {"div_rms": 0.3399, "div_max": 0.7},
}


@dataclass
class ScreeningResult:
    candidate: str
    height: str
    steps: int
    # Survival
    survived: bool
    final_step: int
    # Divergence
    div_rms: float
    div_max: float
    div_final: float
    # HY2-DIV metrics
    hy2_enabled_pct: float  # percent of steps with enabled=true
    hy2_gate_active_pct: float  # percent of steps with gate_active=true
    hy2_gate_mean: float
    hy2_gate_min: float
    hy2_gate_max: float
    hy2_eff_k_mean: float
    hy2_eff_k_max: float
    hy2_torque_left_max: float
    hy2_torque_right_max: float
    hy2_torque_rms: float
    hy2_clip_pct: float
    # Support/roll metrics
    support_pos_err_max: float
    support_pos_err_rms: float
    support_pos_err_final: float
    roll_max: float
    roll_rms: float
    roll_final: float
    # Height
    height_error_max: float
    height_error_final: float
    # Contact
    contact_valid_pct: float
    # WBC/ownership
    wbc_applied_pct: float
    hidden_torque_max: float
    ownership_violations: int
    # Error
    error: Optional[str] = None
    telemetry_path: Optional[str] = None


def run_simulation(candidate: str, height: str, steps: int = 2000) -> ScreeningResult:
    """Run a single 2000-step simulation."""
    config = CANDIDATES[candidate]
    height_config = HEIGHT_CONFIGS[height]

    # Build command
    cmd = [
        sys.executable, "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "J3",
        "--steps", str(steps),
        "--enable-hip-yaw-divergence-damping",
        "--hip-yaw-divergence-k", str(config["k"]),
        "--hip-yaw-divergence-kd", str(config["kd"]),
        "--hip-yaw-divergence-tau-max", str(config["tau_max"]),
        "--hip-yaw-divergence-z-low", str(config["z_low"]),
        "--hip-yaw-divergence-z-high", str(config["z_high"]),
        "--telemetry-decimation", "1",
    ]

    if height_config["setup"]:
        cmd.extend(["--height-variant-setup", height_config["setup"]])

    run_name = f"{candidate}_{height}_{steps}steps"
    print(f"  Running {run_name}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return ScreeningResult(
                candidate=candidate, height=height, steps=steps,
                survived=False, final_step=0,
                div_rms=float('nan'), div_max=float('nan'), div_final=float('nan'),
                hy2_enabled_pct=0, hy2_gate_active_pct=0, hy2_gate_mean=0, hy2_gate_min=0, hy2_gate_max=0,
                hy2_eff_k_mean=0, hy2_eff_k_max=0,
                hy2_torque_left_max=0, hy2_torque_right_max=0, hy2_torque_rms=0, hy2_clip_pct=0,
                support_pos_err_max=0, support_pos_err_rms=0, support_pos_err_final=0,
                roll_max=0, roll_rms=0, roll_final=0,
                height_error_max=0, height_error_final=0,
                contact_valid_pct=0,
                wbc_applied_pct=0, hidden_torque_max=0, ownership_violations=0,
                error=f"Sim failed: {result.stderr[-200:]}",
            )

        # Find telemetry path
        telemetry_path = None
        for line in result.stdout.split('\n'):
            if 'Telemetry saved to:' in line:
                telemetry_path = line.split('Telemetry saved to:')[-1].strip()
                break

        if telemetry_path and Path(telemetry_path).exists():
            return analyze_screening_telemetry(candidate, height, steps, telemetry_path)
        else:
            return ScreeningResult(
                candidate=candidate, height=height, steps=steps,
                survived=False, final_step=0,
                div_rms=float('nan'), div_max=float('nan'), div_final=float('nan'),
                hy2_enabled_pct=0, hy2_gate_active_pct=0, hy2_gate_mean=0, hy2_gate_min=0, hy2_gate_max=0,
                hy2_eff_k_mean=0, hy2_eff_k_max=0,
                hy2_torque_left_max=0, hy2_torque_right_max=0, hy2_torque_rms=0, hy2_clip_pct=0,
                support_pos_err_max=0, support_pos_err_rms=0, support_pos_err_final=0,
                roll_max=0, roll_rms=0, roll_final=0,
                height_error_max=0, height_error_final=0,
                contact_valid_pct=0,
                wbc_applied_pct=0, hidden_torque_max=0, ownership_violations=0,
                error="No telemetry",
            )

    except subprocess.TimeoutExpired:
        return ScreeningResult(
            candidate=candidate, height=height, steps=steps,
            survived=False, final_step=0,
            div_rms=float('nan'), div_max=float('nan'), div_final=float('nan'),
            hy2_enabled_pct=0, hy2_gate_active_pct=0, hy2_gate_mean=0, hy2_gate_min=0, hy2_gate_max=0,
            hy2_eff_k_mean=0, hy2_eff_k_max=0,
            hy2_torque_left_max=0, hy2_torque_right_max=0, hy2_torque_rms=0, hy2_clip_pct=0,
            support_pos_err_max=0, support_pos_err_rms=0, support_pos_err_final=0,
            roll_max=0, roll_rms=0, roll_final=0,
            height_error_max=0, height_error_final=0,
            contact_valid_pct=0,
            wbc_applied_pct=0, hidden_torque_max=0, ownership_violations=0,
            error="Timeout",
        )
    except Exception as e:
        return ScreeningResult(
            candidate=candidate, height=height, steps=steps,
            survived=False, final_step=0,
            div_rms=float('nan'), div_max=float('nan'), div_final=float('nan'),
            hy2_enabled_pct=0, hy2_gate_active_pct=0, hy2_gate_mean=0, hy2_gate_min=0, hy2_gate_max=0,
            hy2_eff_k_mean=0, hy2_eff_k_max=0,
            hy2_torque_left_max=0, hy2_torque_right_max=0, hy2_torque_rms=0, hy2_clip_pct=0,
            support_pos_err_max=0, support_pos_err_rms=0, support_pos_err_final=0,
            roll_max=0, roll_rms=0, roll_final=0,
            height_error_max=0, height_error_final=0,
            contact_valid_pct=0,
            wbc_applied_pct=0, hidden_torque_max=0, ownership_violations=0,
            error=str(e),
        )


def analyze_screening_telemetry(candidate: str, height: str, steps: int, telemetry_path: str) -> ScreeningResult:
    """Analyze telemetry CSV for comprehensive screening metrics."""
    try:
        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return ScreeningResult(
                candidate=candidate, height=height, steps=steps,
                survived=False, final_step=0,
                div_rms=float('nan'), div_max=float('nan'), div_final=float('nan'),
                hy2_enabled_pct=0, hy2_gate_active_pct=0, hy2_gate_mean=0, hy2_gate_min=0, hy2_gate_max=0,
                hy2_eff_k_mean=0, hy2_eff_k_max=0,
                hy2_torque_left_max=0, hy2_torque_right_max=0, hy2_torque_rms=0, hy2_clip_pct=0,
                support_pos_err_max=0, support_pos_err_rms=0, support_pos_err_final=0,
                roll_max=0, roll_rms=0, roll_final=0,
                height_error_max=0, height_error_final=0,
                contact_valid_pct=0,
                wbc_applied_pct=0, hidden_torque_max=0, ownership_violations=0,
                error="Empty telemetry",
                telemetry_path=telemetry_path,
            )

        final_step = len(rows)

        # Check survival
        terminated = False
        for row in rows:
            if row.get('terminated', 'False').lower() == 'true':
                terminated = True
                break
        survived = not terminated and final_step >= steps - 10

        # Divergence metrics
        divergences = []
        for row in rows:
            try:
                divergences.append(float(row.get('hip_yaw_divergence', 0)))
            except (ValueError, TypeError):
                pass

        if divergences:
            div_arr = np.array(divergences)
            div_rms = float(np.sqrt(np.mean(div_arr**2)))
            div_max = float(np.max(np.abs(div_arr)))
            div_final = float(divergences[-1])
        else:
            div_rms = div_max = div_final = float('nan')

        # HY2-DIV metrics
        hy2_enabled_count = 0
        hy2_gate_active_count = 0
        hy2_gate_values = []
        hy2_eff_k_values = []
        hy2_torque_left = []
        hy2_torque_right = []
        hy2_left_clipped = 0
        hy2_right_clipped = 0

        for row in rows:
            try:
                # enabled
                enabled = row.get('hip_yaw_div_enabled', 'False')
                if enabled and enabled.lower() != 'false':
                    hy2_enabled_count += 1

                # gate_active
                gate_active = row.get('hip_yaw_div_gate_active', 'False')
                if gate_active and gate_active.lower() != 'false':
                    hy2_gate_active_count += 1

                # gate value
                gate = float(row.get('hip_yaw_div_height_gate', 0))
                hy2_gate_values.append(gate)

                # effective k
                eff_k = float(row.get('hip_yaw_div_effective_k', 0))
                hy2_eff_k_values.append(eff_k)

                # torques
                tau_l = float(row.get('hip_yaw_div_left', 0))
                tau_r = float(row.get('hip_yaw_div_right', 0))
                hy2_torque_left.append(tau_l)
                hy2_torque_right.append(tau_r)

                # clipping
                cl = row.get('hip_yaw_div_left_clipped', 'False')
                cr = row.get('hip_yaw_div_right_clipped', 'False')
                if cl and cl.lower() != 'false':
                    hy2_left_clipped += 1
                if cr and cr.lower() != 'false':
                    hy2_right_clipped += 1
            except (ValueError, TypeError):
                pass

        n_rows = len(rows)
        hy2_enabled_pct = hy2_enabled_count / n_rows * 100 if n_rows > 0 else 0
        hy2_gate_active_pct = hy2_gate_active_count / n_rows * 100 if n_rows > 0 else 0
        hy2_gate_mean = np.mean(hy2_gate_values) if hy2_gate_values else 0
        hy2_gate_min = np.min(hy2_gate_values) if hy2_gate_values else 0
        hy2_gate_max = np.max(hy2_gate_values) if hy2_gate_values else 0
        hy2_eff_k_mean = np.mean(hy2_eff_k_values) if hy2_eff_k_values else 0
        hy2_eff_k_max = np.max(hy2_eff_k_values) if hy2_eff_k_values else 0

        tau_all = hy2_torque_left + hy2_torque_right
        hy2_torque_left_max = np.max(np.abs(hy2_torque_left)) if hy2_torque_left else 0
        hy2_torque_right_max = np.max(np.abs(hy2_torque_right)) if hy2_torque_right else 0
        hy2_torque_rms = float(np.sqrt(np.mean(np.array(tau_all)**2))) if tau_all else 0
        hy2_clip_pct = (hy2_left_clipped + hy2_right_clipped) / (2 * n_rows) * 100 if n_rows > 0 else 0

        # Support/roll metrics
        support_errs = []
        rolls = []
        for row in rows:
            try:
                support_errs.append(float(row.get('support_position_error', 0)))
                rolls.append(float(row.get('roll_y_rad', 0)))
            except (ValueError, TypeError):
                pass

        support_arr = np.array(support_errs) if support_errs else np.array([0])
        roll_arr = np.array(rolls) if rolls else np.array([0])
        support_pos_err_max = float(np.max(np.abs(support_arr)))
        support_pos_err_rms = float(np.sqrt(np.mean(support_arr**2)))
        support_pos_err_final = float(support_errs[-1]) if support_errs else 0
        roll_max = float(np.max(np.abs(roll_arr)))
        roll_rms = float(np.sqrt(np.mean(roll_arr**2)))
        roll_final = float(rolls[-1]) if rolls else 0

        # Height error
        height_errors = []
        for row in rows:
            try:
                height_errors.append(float(row.get('height_error_m', 0)))
            except (ValueError, TypeError):
                pass
        height_error_max = float(np.max(np.abs(height_errors))) if height_errors else 0
        height_error_final = float(height_errors[-1]) if height_errors else 0

        # Contact validity
        valid_count = 0
        for row in rows:
            try:
                if row.get('contact_valid', 'True').lower() != 'false':
                    valid_count += 1
            except:
                pass
        contact_valid_pct = valid_count / n_rows * 100 if n_rows > 0 else 0

        # WBC/ownership
        wbc_count = 0
        hidden_torques = []
        ownership_violations = 0
        for row in rows:
            try:
                if row.get('wbc_applied', 'False').lower() != 'false':
                    wbc_count += 1
                hidden_torques.append(float(row.get('hidden_torque_norm', 0)))
                if row.get('ownership_violation', 'False').lower() != 'false':
                    ownership_violations += 1
            except:
                pass

        wbc_applied_pct = wbc_count / n_rows * 100 if n_rows > 0 else 0
        hidden_torque_max = float(np.max(hidden_torques)) if hidden_torques else 0

        return ScreeningResult(
            candidate=candidate, height=height, steps=steps,
            survived=survived, final_step=final_step,
            div_rms=div_rms, div_max=div_max, div_final=div_final,
            hy2_enabled_pct=hy2_enabled_pct,
            hy2_gate_active_pct=hy2_gate_active_pct,
            hy2_gate_mean=hy2_gate_mean, hy2_gate_min=hy2_gate_min, hy2_gate_max=hy2_gate_max,
            hy2_eff_k_mean=hy2_eff_k_mean, hy2_eff_k_max=hy2_eff_k_max,
            hy2_torque_left_max=hy2_torque_left_max, hy2_torque_right_max=hy2_torque_right_max,
            hy2_torque_rms=hy2_torque_rms, hy2_clip_pct=hy2_clip_pct,
            support_pos_err_max=support_pos_err_max, support_pos_err_rms=support_pos_err_rms,
            support_pos_err_final=support_pos_err_final,
            roll_max=roll_max, roll_rms=roll_rms, roll_final=roll_final,
            height_error_max=height_error_max, height_error_final=height_error_final,
            contact_valid_pct=contact_valid_pct,
            wbc_applied_pct=wbc_applied_pct, hidden_torque_max=hidden_torque_max,
            ownership_violations=ownership_violations,
            telemetry_path=telemetry_path,
        )

    except Exception as e:
        return ScreeningResult(
            candidate=candidate, height=height, steps=steps,
            survived=False, final_step=0,
            div_rms=float('nan'), div_max=float('nan'), div_final=float('nan'),
            hy2_enabled_pct=0, hy2_gate_active_pct=0, hy2_gate_mean=0, hy2_gate_min=0, hy2_gate_max=0,
            hy2_eff_k_mean=0, hy2_eff_k_max=0,
            hy2_torque_left_max=0, hy2_torque_right_max=0, hy2_torque_rms=0, hy2_clip_pct=0,
            support_pos_err_max=0, support_pos_err_rms=0, support_pos_err_final=0,
            roll_max=0, roll_rms=0, roll_final=0,
            height_error_max=0, height_error_final=0,
            contact_valid_pct=0,
            wbc_applied_pct=0, hidden_torque_max=0, ownership_violations=0,
            error=str(e),
        )


def print_screening_table(results: list[ScreeningResult]):
    """Print screening results table."""
    print("\n" + "=" * 160)
    print(f"{'Cand':<6} {'Height':<10} {'Surv':<5} {'Div RMS':<10} {'Div Max':<10} {'Gate%':<8} {'GateMean':<9} {'EffKMax':<8} {'Clipped%':<9} {'WBC%':<6} {'Hidden':<8}")
    print("-" * 160)

    for r in results:
        surv = "YES" if r.survived else "NO"
        gate_str = f"{r.hy2_gate_active_pct:.0f}%" if r.hy2_gate_active_pct > 0 else "0%"
        gate_mean_str = f"{r.hy2_gate_mean:.2f}" if r.hy2_gate_mean > 0 else "0.00"
        clipped_str = f"{r.hy2_clip_pct:.1f}%" if r.hy2_clip_pct > 0 else "0.0%"
        wbc_str = f"{r.wbc_applied_pct:.0f}%" if r.wbc_applied_pct > 0 else "0%"
        hidden_str = f"{r.hidden_torque_max:.4f}" if r.hidden_torque_max > 0 else "0.0"

        div_rms_str = f"{r.div_rms:.4f}" if not np.isnan(r.div_rms) else "N/A"
        div_max_str = f"{r.div_max:.4f}" if not np.isnan(r.div_max) else "N/A"
        eff_k_str = f"{r.hy2_eff_k_max:.2f}" if r.hy2_eff_k_max > 0 else "0.00"

        print(f"{r.candidate:<6} {r.height:<10} {surv:<5} {div_rms_str:<10} {div_max_str:<10} {gate_str:<8} {gate_mean_str:<9} {eff_k_str:<8} {clipped_str:<9} {wbc_str:<6} {hidden_str:<8}")

    print("=" * 160)


def select_candidates_for_5000step(results: list[ScreeningResult]) -> list[str]:
    """Select candidates for 5000-step evaluation based on screening criteria."""
    # Group by candidate
    by_candidate = {}
    for r in results:
        if r.candidate not in by_candidate:
            by_candidate[r.candidate] = {}
        by_candidate[r.candidate][r.height] = r

    selected = []
    baseline_div_rms = BASELINES

    for candidate, heights in by_candidate.items():
        if candidate == "no_hy2":
            continue  # Skip baseline for selection

        score = 0
        reasons = []

        # Check nominal
        if "nominal" in heights:
            r = heights["nominal"]
            baseline = baseline_div_rms["nominal"]["div_rms"]
            if r.div_rms < baseline * 0.9:  # 10% improvement
                score += 2
                reasons.append(f"nominal: {r.div_rms:.4f} < {baseline:.4f}")
            elif r.div_rms < baseline:
                score += 1
                reasons.append(f"nominal: {r.div_rms:.4f} < {baseline:.4f}")
            elif r.div_rms > baseline * 1.2:  # Worse by 20%
                score -= 2
                reasons.append(f"nominal WORSE: {r.div_rms:.4f} > {baseline:.4f}")

            # Check WBC/ownership
            if r.wbc_applied_pct > 0 or r.ownership_violations > 0:
                score -= 1
                reasons.append(f"WBC/ownership issue")

        # Check low
        if "low_0p300" in heights:
            r = heights["low_0p300"]
            baseline = baseline_div_rms["low_0p300"]["div_rms"]
            if r.div_rms < baseline:
                score += 1
                reasons.append(f"low: {r.div_rms:.4f} < {baseline:.4f}")

        # Check high
        if "high_0p480" in heights:
            r = heights["high_0p480"]
            baseline = baseline_div_rms["high_0p480"]["div_rms"]
            if r.div_rms < baseline:
                score += 1
                reasons.append(f"high: {r.div_rms:.4f} < {baseline:.4f}")

        # Must survive all heights
        survived_all = all(heights[h].survived for h in heights if h in heights)
        if not survived_all:
            score = -100
            reasons.append("FAILED survival check")

        if score > 0:
            selected.append(candidate)
            print(f"  {candidate}: score={score}, reasons={reasons}")

    # Limit to top 3
    return selected[:3]


def main():
    parser = argparse.ArgumentParser(description="HY2-DIV 2000-step screening")
    parser.add_argument("--candidates", nargs="+", default=["A0", "A1", "A2", "B1", "B2"],
                        help="Candidates to evaluate")
    parser.add_argument("--heights", nargs="+", default=["nominal", "low_0p300", "high_0p480"],
                        help="Heights to evaluate")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps")
    args = parser.parse_args()

    print("=" * 80)
    print("HY2-DIV 2000-Step Screening Evaluation")
    print("=" * 80)
    print(f"Candidates: {args.candidates}")
    print(f"Heights: {args.heights}")
    print(f"Steps: {args.steps}")

    results = []
    for candidate in args.candidates:
        for height in args.heights:
            result = run_simulation(candidate, height, args.steps)
            results.append(result)

    # Print table
    print_screening_table(results)

    # Select candidates
    print("\n" + "=" * 80)
    print("CANDIDATE SELECTION FOR 5000-STEP")
    print("=" * 80)
    selected = select_candidates_for_5000step(results)
    print(f"\nSelected for 5000-step: {selected}")

    # Save results
    output_file = OUTPUT_DIR / "screening_2000_results.json"
    results_data = []
    for r in results:
        results_data.append({
            "candidate": r.candidate,
            "height": r.height,
            "survived": r.survived,
            "final_step": r.final_step,
            "div_rms": r.div_rms,
            "div_max": r.div_max,
            "div_final": r.div_final,
            "hy2_gate_active_pct": r.hy2_gate_active_pct,
            "hy2_gate_mean": r.hy2_gate_mean,
            "hy2_gate_min": r.hy2_gate_min,
            "hy2_gate_max": r.hy2_gate_max,
            "hy2_eff_k_max": r.hy2_eff_k_max,
            "hy2_clip_pct": r.hy2_clip_pct,
            "support_pos_err_rms": r.support_pos_err_rms,
            "roll_rms": r.roll_rms,
            "contact_valid_pct": r.contact_valid_pct,
            "wbc_applied_pct": r.wbc_applied_pct,
            "hidden_torque_max": r.hidden_torque_max,
            "ownership_violations": r.ownership_violations,
            "error": r.error,
        })

    with open(output_file, 'w') as f:
        json.dump({
            "steps": args.steps,
            "candidates": args.candidates,
            "heights": args.heights,
            "selected_for_5000step": selected,
            "results": results_data,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
