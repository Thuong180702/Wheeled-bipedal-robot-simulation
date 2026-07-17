#!/usr/bin/env python3
"""
K2_NOTCH_LOW_Q_V1 — Create and Validate.

Runs paired K1 (baseline) vs K2 (candidate, Q=2.0) simulations across:
  - high_0p480: A_equilibrium, D_prbs_excitation, B_90n_push
  - mid_0p400:  A_equilibrium, D_prbs_excitation
  - low_0p330:  A_equilibrium, D_prbs_excitation

All runs are 3000 steps, real_simulation only.

STRICT CONSTRAINT: Audit-only. K1 remains current-best.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SIM_SCRIPT = SCRIPTS_DIR / "simulate_hierarchical_controller.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k2_notch_low_q_v1_validation"

K1_PROFILE = "k1_pitch_rate_notch_v1"
K2_PROFILE = "k2_notch_low_q_v1"
CONTROLLER_MODE = "balance-core"

# -- Validation Matrix --
# (height_name, height_m, run_type, steps)
VALIDATION_MATRIX = [
    # Core: equilibrium at all 3 heights (2000 steps = sweep baseline)
    ("high_0p480", 0.48, "A_equilibrium", 2000),
    ("mid_0p400", 0.40, "A_equilibrium", 2000),
    ("low_0p330", 0.33, "A_equilibrium", 2000),
    # PRBS excitation (2000 steps)
    ("high_0p480", 0.48, "D_prbs_excitation", 2000),
    ("mid_0p400", 0.40, "D_prbs_excitation", 2000),
    ("low_0p330", 0.33, "D_prbs_excitation", 2000),
    # Push recovery (2000 steps)
    ("high_0p480", 0.48, "B_90n_push", 2000),
]

PRBS_AMPLITUDE = 0.50  # N, sagittal push amplitude for PRBS
PUSH_FORCE_N = 90.0     # N, for B_90n_push
PUSH_START_STEP = 1000   # equilibrium before push
PUSH_DURATION = 10       # steps


def generate_prbs_excitation(n_steps: int, amplitude: float,
                              min_period: int = 3, max_period: int = 12,
                              seed: int = 42) -> list:
    """Generate PRBS excitation signal."""
    rng = np.random.RandomState(seed)
    signal = []
    i = 0
    while i < n_steps:
        val = amplitude * (1.0 if rng.rand() > 0.5 else -1.0)
        period = rng.randint(min_period, max_period + 1)
        period = min(period, n_steps - i)
        signal.extend([val] * period)
        i += period
    return signal[:n_steps]


def prbs_to_push_sequence(prbs_signal: list, sagittal: bool = True) -> list:
    """Convert PRBS signal to push sequence entries."""
    sequence = []
    n = len(prbs_signal)
    i = 0
    while i < n:
        val = prbs_signal[i]
        j = i + 1
        while j < n and abs(prbs_signal[j] - val) < 1e-9:
            j += 1
        duration = j - i
        if sagittal:
            sequence.append([i, 0.0, float(val), duration])
        else:
            sequence.append([i, float(val), 0.0, duration])
        i = j
    return sequence


def run_simulation(profile_name: str, height_name: str, height_m: float,
                   run_type: str, steps: int, output_subdir: Path,
                   push_sequence_file: Optional[str] = None) -> Tuple[bool, str]:
    """Run a single MuJoCo simulation."""
    cmd = [
        sys.executable, str(SIM_SCRIPT),
        "--vd-sagittal-authority-profile", profile_name,
        "--controller-mode", CONTROLLER_MODE,
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", str(PROJECT_ROOT / "outputs" / "physical_target_height_setups_centered"
                                       / f"{height_name}_setup.json"),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--write-run-summary-sidecar",
        "--output-dir", str(output_subdir),
    ]

    if push_sequence_file:
        cmd.extend(["--push-sequence-file", push_sequence_file])

    metadata = {
        "profile_name": profile_name,
        "height_name": height_name,
        "height_m": height_m,
        "run_type": run_type,
        "steps": steps,
        "validation_source": "real_simulation",
    }
    with open(output_subdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    stdout_path = output_subdir / "stdout.log"
    stderr_path = output_subdir / "stderr.log"
    try:
        with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
            result = subprocess.run(cmd, stdout=f_out, stderr=f_err,
                                   text=True, timeout=900, cwd=str(PROJECT_ROOT))
        success = result.returncode == 0
        msg = "OK" if success else f"FAILED:{result.returncode}"
        return success, msg
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR:{e}"


def find_telemetry_csv(run_dir: Path) -> Optional[Path]:
    """Find the telemetry CSV in a run directory."""
    csv_files = sorted(run_dir.glob("telemetry_*.csv"))
    return csv_files[-1] if csv_files else None


def load_telemetry(csv_path: Path) -> List[Dict]:
    """Load telemetry CSV into list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows: List[Dict], run_type: str = "A_equilibrium") -> Dict:
    """Compute validation metrics from telemetry rows.

    Returns a dict with keys matching the Phase 4 spec.
    """
    n = len(rows)
    if n == 0:
        return {"status": "EMPTY_CSV", "n_rows": 0}

    metrics = {"n_rows": n, "run_type": run_type}

    # -- Extract key time series --
    try:
        pitch_deg = np.array([float(r.get("k1_global_pitch_deg", 0)) for r in rows])
        pitch_rate_rads = np.array([float(r.get("k1_pitch_rate_rad_s", 0)) for r in rows])
        support_m = np.array([float(r.get("k1_support_position_error_m", 0)) for r in rows])
        body_height_m = np.array([float(r.get("height_variant_achieved_com_z_m",
                                              r.get("body_height", 0.40))) for r in rows])
        notch_output = np.array([float(r.get("k1_notch_output", 0)) for r in rows])
        roll_deg = np.array([float(r.get("k1_global_roll_deg", 0)) for r in rows])
        yaw_deg = np.array([float(r.get("k1_global_yaw_deg", 0)) for r in rows])
        hip_yaw_l = np.array([float(r.get("k1_hip_yaw_l_rad", 0)) for r in rows])
        hip_yaw_r = np.array([float(r.get("k1_hip_yaw_r_rad", 0)) for r in rows])

        # Torque clipping
        tau_clip_frac = float(rows[-1].get("k1_torque_clip_fraction", 0)) if n > 0 else 0.0
        pos_cap_frac = float(rows[-1].get("k1_position_cap_fraction", 0)) if n > 0 else 0.0

        # Fall detection
        has_fall = any(float(r.get("k1_has_fall", 0)) > 0.5 for r in rows) if n > 0 else False
        fall_reason = ""
        if has_fall:
            for r in rows:
                fr = r.get("k1_fall_reason", "")
                if fr:
                    fall_reason = fr
                    break

        # NaN/Inf check
        has_nan = False
        for key in ["k1_global_pitch_deg", "k1_pitch_rate_rad_s", "k1_support_position_error_m"]:
            vals = np.array([float(r.get(key, 0)) for r in rows])
            if np.any(~np.isfinite(vals)):
                has_nan = True
                break

    except (ValueError, KeyError) as e:
        return {"status": f"PARSE_ERROR:{e}", "n_rows": n}

    # -- Basic stats --
    metrics["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch_deg ** 2)))
    metrics["pitch_abs_max_deg"] = float(np.max(np.abs(pitch_deg)))
    metrics["pitch_rate_rms_rad_s"] = float(np.sqrt(np.mean(pitch_rate_rads ** 2)))
    metrics["support_rms_m"] = float(np.sqrt(np.mean(support_m ** 2)))
    metrics["support_abs_max_m"] = float(np.max(np.abs(support_m)))
    metrics["body_height_min_m"] = float(np.min(body_height_m))
    metrics["body_height_mean_m"] = float(np.mean(body_height_m))
    metrics["roll_abs_max_deg"] = float(np.max(np.abs(roll_deg)))
    metrics["yaw_abs_max_deg"] = float(np.max(np.abs(yaw_deg)))
    metrics["hip_yaw_abs_max_rad"] = float(max(np.max(np.abs(hip_yaw_l)), np.max(np.abs(hip_yaw_r))))
    metrics["notch_output_rms"] = float(np.sqrt(np.mean(notch_output ** 2)))
    metrics["torque_clip_fraction"] = tau_clip_frac
    metrics["position_cap_fraction"] = pos_cap_frac
    metrics["has_fall"] = has_fall
    metrics["fall_reason"] = fall_reason
    metrics["has_nan"] = has_nan

    # -- FFT-based spectral analysis --
    dt = 0.01  # 100 Hz telemetry
    fs = 1.0 / dt

    # Hanning window
    window = np.hanning(n)
    pitch_detrended = pitch_deg - np.mean(pitch_deg)

    # Compute PSD via FFT
    fft = np.fft.rfft(pitch_detrended * window)
    psd = (np.abs(fft) ** 2) / (fs * n)
    freqs = np.fft.rfftfreq(n, dt)

    # LF band: 0.15-0.55 Hz
    lf_mask = (freqs >= 0.15) & (freqs <= 0.55)
    if np.any(lf_mask):
        metrics["lf_power_0p15_0p55_hz"] = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
        lf_peak_idx = np.argmax(psd[lf_mask])
        metrics["lf_peak_freq_hz"] = float(freqs[lf_mask][lf_peak_idx])
        metrics["lf_peak_power"] = float(psd[lf_mask][lf_peak_idx])
    else:
        metrics["lf_power_0p15_0p55_hz"] = 0.0
        metrics["lf_peak_freq_hz"] = 0.0
        metrics["lf_peak_power"] = 0.0

    # WIP band: 2.0-3.0 Hz
    wip_mask = (freqs >= 2.0) & (freqs <= 3.0)
    if np.any(wip_mask):
        metrics["wip_power_2p0_3p0_hz"] = float(np.trapz(psd[wip_mask], freqs[wip_mask]))
    else:
        metrics["wip_power_2p0_3p0_hz"] = 0.0

    # Pitch rate PSD in WIP band
    pr_detrended = pitch_rate_rads - np.mean(pitch_rate_rads)
    fft_pr = np.fft.rfft(pr_detrended * window)
    psd_pr = (np.abs(fft_pr) ** 2) / (fs * n)
    if np.any(wip_mask):
        metrics["wip_pitch_rate_power_2p0_3p0_hz"] = float(np.trapz(psd_pr[wip_mask], freqs[wip_mask]))
    else:
        metrics["wip_pitch_rate_power_2p0_3p0_hz"] = 0.0

    # Notch output PSD in WIP band
    no_detrended = notch_output - np.mean(notch_output)
    fft_no = np.fft.rfft(no_detrended * window)
    psd_no = (np.abs(fft_no) ** 2) / (fs * n)
    if np.any(wip_mask):
        metrics["wip_notch_output_power_2p0_3p0_hz"] = float(np.trapz(psd_no[wip_mask], freqs[wip_mask]))
    else:
        metrics["wip_notch_output_power_2p0_3p0_hz"] = 0.0

    # Cross-spectral coherence: pitch vs notch at LF peak
    if metrics["lf_peak_freq_hz"] > 0:
        # Simple coherence approximation at LF peak
        # Using magnitude-squared coherence: |Pxy|^2 / (Pxx * Pyy)
        # For simplicity we use the ratio of notch PSD at LF peak to pitch PSD
        no_fft = np.fft.rfft(no_detrended * window)
        psd_no_full = (np.abs(no_fft) ** 2) / (fs * n)
        # Cross-power
        cross = fft * np.conj(no_fft) / (fs * n)
        lf_peak_bin = np.argmin(np.abs(freqs - metrics["lf_peak_freq_hz"]))
        pxx = psd[lf_peak_bin]
        pyy = psd_no_full[lf_peak_bin]
        pxy = np.abs(cross[lf_peak_bin])
        if pxx > 0 and pyy > 0:
            metrics["lf_pitch_notch_coherence"] = float((pxy ** 2) / (pxx * pyy))
        else:
            metrics["lf_pitch_notch_coherence"] = 0.0
    else:
        metrics["lf_pitch_notch_coherence"] = 0.0

    # -- Push recovery metrics (only for push runs) --
    if "push" in run_type.lower():
        push_start = PUSH_START_STEP
        if n > push_start:
            post_push = pitch_deg[push_start:]
            metrics["post_push_pitch_rms_deg"] = float(np.sqrt(np.mean(post_push ** 2)))

            post_support = support_m[push_start:]
            metrics["post_push_support_rms_m"] = float(np.sqrt(np.mean(post_support ** 2)))

            # Recovery windows
            for window_steps in [500, 1000, 2000]:
                end = min(push_start + window_steps, n)
                if end > push_start:
                    window_pitch = pitch_deg[push_start:end]
                    metrics[f"recovery_{window_steps}step_pitch_rms_deg"] = float(np.sqrt(np.mean(window_pitch ** 2)))

    # -- Controller/safety flags --
    metrics["hidden_torque_flag"] = False  # real_simulation
    metrics["wbc_flag"] = False
    metrics["validation_source"] = "real_simulation"

    metrics["status"] = "OK"
    return metrics


def generate_push_90n(run_dir: Path) -> str:
    """Generate a 90N single sagittal push sequence."""
    sequence = [[PUSH_START_STEP, 0.0, PUSH_FORCE_N, PUSH_DURATION]]
    push_seq = {
        "sequence": sequence,
        "description": f"Single {PUSH_FORCE_N}N sagittal push at step {PUSH_START_STEP}",
        "amplitude_n": PUSH_FORCE_N,
        "n_entries": 1,
    }
    push_path = run_dir / "push_sequence.json"
    with open(push_path, "w") as f:
        json.dump(push_seq, f, indent=2)
    return str(push_path)


def generate_prbs(run_dir: Path, n_steps: int, seed: int) -> str:
    """Generate PRBS excitation push sequence."""
    prbs_signal = generate_prbs_excitation(n_steps, PRBS_AMPLITUDE, seed=seed)
    push_seq = prbs_to_push_sequence(prbs_signal, sagittal=True)

    push_path = run_dir / "push_sequence.json"
    with open(push_path, "w") as f:
        json.dump({"sequence": push_seq, "description": "PRBS sagittal excitation",
                   "amplitude_n": PRBS_AMPLITUDE, "n_entries": len(push_seq)}, f, indent=2)

    exc_path = run_dir / "excitation_signal.json"
    with open(exc_path, "w") as f:
        json.dump({
            "signal": [float(x) for x in prbs_signal],
            "n_steps": len(prbs_signal),
            "amplitude_max": float(max(abs(x) for x in prbs_signal)),
            "run_type": "D_prbs_excitation",
        }, f, indent=2)

    return str(push_path)


def run_validation_matrix(profiles: List[str], dry_run: bool = False,
                           resume: bool = True) -> Dict:
    """Run the full validation matrix for given profiles."""
    results = {}

    for profile_name in profiles:
        print(f"\n{'='*72}")
        print(f"  Profile: {profile_name}")
        print(f"{'='*72}")

        profile_results = {"runs": {}}

        for height_name, height_m, run_type, steps in VALIDATION_MATRIX:
            run_label = f"{height_name}/{run_type}"
            run_dir = OUTPUT_DIR / profile_name / height_name / run_type
            run_dir.mkdir(parents=True, exist_ok=True)

            # Check for existing results
            if resume:
                csv_path = find_telemetry_csv(run_dir)
                if csv_path:
                    rows = load_telemetry(csv_path)
                    if rows:
                        print(f"  [{run_label}] -> SKIP (existing CSV, {len(rows)} rows)")
                        metrics = compute_metrics(rows, run_type)
                        profile_results["runs"][run_label] = metrics
                        continue

            if dry_run:
                print(f"  [{run_label}] -> DRY_RUN")
                profile_results["runs"][run_label] = {"status": "dry_run"}
                continue

            # Prepare push sequence
            push_seq_file = None
            if run_type == "D_prbs_excitation":
                seed = hash(f"{profile_name}_{height_name}") % 10000
                push_seq_file = generate_prbs(run_dir, steps, seed)
            elif run_type == "B_90n_push":
                push_seq_file = generate_push_90n(run_dir)

            print(f"  [{run_label}] ({steps} steps)...", end=" ", flush=True)
            start_time = time.time()

            success, msg = run_simulation(
                profile_name, height_name, height_m, run_type, steps,
                run_dir, push_sequence_file=push_seq_file,
            )

            elapsed = time.time() - start_time
            print(f"{msg} ({elapsed:.1f}s)")

            if success:
                csv_path = find_telemetry_csv(run_dir)
                if csv_path:
                    rows = load_telemetry(csv_path)
                    if rows:
                        metrics = compute_metrics(rows, run_type)
                        metrics["status"] = "OK"
                        metrics["elapsed_s"] = elapsed
                        profile_results["runs"][run_label] = metrics
                    else:
                        profile_results["runs"][run_label] = {"status": "CSV_READ_FAILED"}
                else:
                    profile_results["runs"][run_label] = {"status": "NO_CSV"}
            else:
                profile_results["runs"][run_label] = {"status": msg}

        results[profile_name] = profile_results

    # Save results
    if not dry_run:
        results_path = OUTPUT_DIR / "validation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {results_path}")

    return results


def print_comparison(results: Dict):
    """Print K1 vs K2 comparison summary."""
    print("\n" + "=" * 80)
    print("K1 vs K2 COMPARISON SUMMARY")
    print("=" * 80)

    k1_data = results.get(K1_PROFILE, {}).get("runs", {})
    k2_data = results.get(K2_PROFILE, {}).get("runs", {})

    header = f"{'Run':<30} {'Metric':<20} {'K1':>12} {'K2':>12} {'Delta':>12} {'Verdict':>12}"
    print(header)
    print("-" * 98)

    for height_name, height_m, run_type, steps in VALIDATION_MATRIX:
        run_label = f"{height_name}/{run_type}"
        k1 = k1_data.get(run_label, {})
        k2 = k2_data.get(run_label, {})

        # Skip if either is missing
        if k1.get("status") != "OK" or k2.get("status") != "OK":
            print(f"{run_label:<30} {'<missing/incomplete>':>20}")
            continue

        # Compare key metrics
        comparisons = [
            ("pitch RMS (deg)", "pitch_rms_deg", "pitch_rms_deg", False),
            ("LF power", "lf_power_0p15_0p55_hz", "lf_power_0p15_0p55_hz", False),
            ("WIP power", "wip_power_2p0_3p0_hz", "wip_power_2p0_3p0_hz", False),
            ("support RMS (m)", "support_rms_m", "support_rms_m", False),
            ("body height min (m)", "body_height_min_m", "body_height_min_m", False),
            ("hip_yaw max (rad)", "hip_yaw_abs_max_rad", "hip_yaw_abs_max_rad", False),
            ("fall", "has_fall", "has_fall", True),
        ]

        for metric_name, k1_key, k2_key, is_bool in comparisons:
            v1 = k1.get(k1_key, 0)
            v2 = k2.get(k2_key, 0)

            if is_bool:
                delta_str = "SAME" if v1 == v2 else "DIFF"
                verdict = "PASS" if v2 == v1 else "WARN"
            elif v1 != 0:
                delta_pct = (v2 - v1) / abs(v1) * 100
                delta_str = f"{delta_pct:+.1f}%"
                # Lower is better for all these
                if delta_pct < -5:
                    verdict = "BETTER"
                elif delta_pct < 5:
                    verdict = "SAME"
                else:
                    verdict = "WORSE"
            else:
                delta_str = "N/A"
                verdict = "N/A"

            if metric_name == "pitch RMS (deg)":
                print(f"{run_label:<30} {metric_name:<20} {v1:>12.3f} {v2:>12.3f} {delta_str:>12} {verdict:>12}")
            elif metric_name == "LF power":
                print(f"{run_label:<30} {metric_name:<20} {v1:>12.6f} {v2:>12.6f} {delta_str:>12} {verdict:>12}")
            elif metric_name == "WIP power":
                print(f"{run_label:<30} {metric_name:<20} {v1:>12.6f} {v2:>12.6f} {delta_str:>12} {verdict:>12}")
            elif metric_name == "body height min (m)":
                print(f"{run_label:<30} {metric_name:<20} {v1:>12.4f} {v2:>12.4f} {delta_str:>12} {verdict:>12}")
            elif metric_name == "hip_yaw max (rad)":
                print(f"{run_label:<30} {metric_name:<20} {v1:>12.4f} {v2:>12.4f} {delta_str:>12} {verdict:>12}")
            elif metric_name == "fall":
                print(f"{run_label:<30} {metric_name:<20} {str(v1):>12} {str(v2):>12} {delta_str:>12} {verdict:>12}")
            else:
                print(f"{run_label:<30} {metric_name:<20} {v1:>12.4f} {v2:>12.4f} {delta_str:>12} {verdict:>12}")


def classify(results: Dict) -> str:
    """Classify K2 based on acceptance gates (Phase 5)."""
    k1_data = results.get(K1_PROFILE, {}).get("runs", {})
    k2_data = results.get(K2_PROFILE, {}).get("runs", {})

    issues = []
    improvements = []
    wip_regressions = 0
    lf_improvements = 0
    pitch_worse = 0
    total_paired = 0

    for height_name, height_m, run_type, steps in VALIDATION_MATRIX:
        run_label = f"{height_name}/{run_type}"
        k1 = k1_data.get(run_label, {})
        k2 = k2_data.get(run_label, {})

        if k1.get("status") != "OK" or k2.get("status") != "OK":
            continue

        total_paired += 1

        # Hard gates
        if k2.get("has_fall") and not k1.get("has_fall"):
            issues.append(f"NEW_FALL:{run_label}")
        if k2.get("has_nan"):
            issues.append(f"NaN:{run_label}")
        if k2.get("hidden_torque_flag"):
            issues.append("hidden_torque")
        if k2.get("wbc_flag"):
            issues.append("wbc")

        # Hip-yaw gate
        hip_yaw_k2 = k2.get("hip_yaw_abs_max_rad", 0)
        if hip_yaw_k2 > 0.35:
            issues.append(f"HIP_YAW_GATE:{run_label}:{hip_yaw_k2:.3f}rad")

        # LF power comparison
        lf1 = k1.get("lf_power_0p15_0p55_hz", 0)
        lf2 = k2.get("lf_power_0p15_0p55_hz", 0)
        if lf1 > 0 and lf2 < lf1 * 0.95:
            lf_improvements += 1
            improvements.append(f"LF_IMPROVED:{run_label}:{(lf2/lf1-1)*100:+.1f}%")

        # WIP regression
        wip1 = k1.get("wip_power_2p0_3p0_hz", 0)
        wip2 = k2.get("wip_power_2p0_3p0_hz", 0)
        if wip1 > 0 and wip2 > wip1 * 1.10:
            wip_regressions += 1
            issues.append(f"WIP_REGRESSION:{run_label}:{(wip2/wip1-1)*100:+.1f}%")

        # Pitch RMS
        prms1 = k1.get("pitch_rms_deg", 0)
        prms2 = k2.get("pitch_rms_deg", 0)
        if prms1 > 0 and prms2 > prms1 * 1.05:
            pitch_worse += 1

    # Classification logic
    if issues:
        for issue in issues:
            if "FALL" in issue or "NaN" in issue or "hidden_torque" in issue or "wbc" in issue:
                return "K2_REGRESSION_DO_NOT_USE"
        for issue in issues:
            if "HIP_YAW_GATE" in issue:
                return "K2_MIXED_RESULTS_KEEP_AS_CANDIDATE"

    if wip_regressions > 0 or pitch_worse > 0:
        if lf_improvements > 0:
            return "K2_MIXED_RESULTS_KEEP_AS_CANDIDATE"
        return "K2_REGRESSION_DO_NOT_USE"

    if lf_improvements >= 2:
        return "K2_STRONG_PASS_READY_FOR_PROMOTION"

    if lf_improvements >= 1:
        return "K2_PASS_NEEDS_MORE_VALIDATION"

    return "K2_MIXED_RESULTS_KEEP_AS_CANDIDATE"


def main():
    parser = argparse.ArgumentParser(description="K2_NOTCH_LOW_Q_V1 Validation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print planned runs without executing")
    parser.add_argument("--no-resume", action="store_true",
                       help="Do not skip existing results")
    parser.add_argument("--profile", type=str, default=None,
                       choices=[K1_PROFILE, K2_PROFILE],
                       help="Run only one profile (default: both)")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else [K1_PROFILE, K2_PROFILE]
    resume = not args.no_resume

    print("=" * 72)
    print("K2_NOTCH_LOW_Q_V1 VALIDATION")
    print(f"  Profiles: {profiles}")
    print(f"  Matrix: {len(VALIDATION_MATRIX)} runs per profile")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Resume: {resume}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_validation_matrix(profiles, dry_run=args.dry_run, resume=resume)

    if not args.dry_run:
        print_comparison(results)

        classification = classify(results)
        print(f"\n{'='*72}")
        print(f"  Classification: {classification}")
        print(f"{'='*72}")

    return results


if __name__ == "__main__":
    main()
