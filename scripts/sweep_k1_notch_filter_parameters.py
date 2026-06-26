#!/usr/bin/env python3
"""
K1 Notch Filter Parameter and Topology Sweep -- Fast Screening.

Runs real MuJoCo simulations for each filter parameter/topology candidate,
collects augmented k1_ telemetry, and computes screening metrics.

STRICT CONSTRAINT: All candidates are audit-only.  K1 remains immutable baseline.
Do NOT tune non-filter gains.  Do NOT promote anything.
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_notch_filter_sweep"

# -- K1 Baseline Profile --
K1_PROFILE = "k1_pitch_rate_notch_v1"
CONTROLLER_MODE = "balance-core"

# -- Sweep Groups --
# Group A: centre frequency sweep (Q=6, blend=1.0, biquad_notch)
GROUP_A_FC_SWEEP = {
    "k_sweep_fc_1p50": {"center_hz": 1.5, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_1p75": {"center_hz": 1.75, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_2p00": {"center_hz": 2.0, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_2p25": {"center_hz": 2.25, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    # K1 baseline at 2.5 Hz -- included for reference
    "k1_pitch_rate_notch_v1": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_2p75": {"center_hz": 2.75, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_3p00": {"center_hz": 3.0, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_3p25": {"center_hz": 3.25, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
    "k_sweep_fc_3p50": {"center_hz": 3.5, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "A_fc"},
}

# Group B: Q sweep (fc=2.5, blend=1.0, biquad_notch)
GROUP_B_Q_SWEEP = {
    "k_sweep_q_2p0": {"center_hz": 2.5, "Q": 2.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "B_Q"},
    "k_sweep_q_3p0": {"center_hz": 2.5, "Q": 3.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "B_Q"},
    "k1d_pitch_rate_notch_q4": {"center_hz": 2.5, "Q": 4.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "B_Q"},
    "k1_pitch_rate_notch_v1": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "B_Q"},
    "k1e_pitch_rate_notch_q8": {"center_hz": 2.5, "Q": 8.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "B_Q"},
    "k_sweep_q_10p0": {"center_hz": 2.5, "Q": 10.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "B_Q"},
}

# Group C: blend sweep (fc=2.5, Q=6, biquad_notch)
GROUP_C_BLEND_SWEEP = {
    "k_sweep_blend_0p00": {"center_hz": 2.5, "Q": 6.0, "blend": 0.0, "filter_type": "biquad_notch", "group": "C_blend"},
    "k_sweep_blend_0p25": {"center_hz": 2.5, "Q": 6.0, "blend": 0.25, "filter_type": "biquad_notch", "group": "C_blend"},
    "k1g_pitch_rate_notch_blend050": {"center_hz": 2.5, "Q": 6.0, "blend": 0.50, "filter_type": "biquad_notch", "group": "C_blend"},
    "k1f_pitch_rate_notch_blend075": {"center_hz": 2.5, "Q": 6.0, "blend": 0.75, "filter_type": "biquad_notch", "group": "C_blend"},
    "k1_pitch_rate_notch_v1": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "biquad_notch", "group": "C_blend"},
}

# Group D: topology variants
GROUP_D_TOPOLOGY = {
    # Notch-disabled diagnostic
    "k_sweep_notch_disabled": {"center_hz": 2.5, "Q": 6.0, "blend": 0.0, "filter_type": "notch_disabled", "group": "D_topo"},
    # First-order lowpass variants
    "k_sweep_lp_3p0": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "first_order_lowpass", "lp_cutoff": 3.0, "group": "D_topo"},
    "k_sweep_lp_4p0": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "first_order_lowpass", "lp_cutoff": 4.0, "group": "D_topo"},
    "k_sweep_lp_5p0": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "first_order_lowpass", "lp_cutoff": 5.0, "group": "D_topo"},
    "k_sweep_lp_6p0": {"center_hz": 2.5, "Q": 6.0, "blend": 1.0, "filter_type": "first_order_lowpass", "lp_cutoff": 6.0, "group": "D_topo"},
}

# Target heights for screening
SCREENING_HEIGHTS = {
    "high_0p480": 0.480,
}

# Run types for screening
SCREENING_RUN_TYPES = ["A_equilibrium", "D_prbs_excitation"]

# Steps per screening run
SCREENING_STEPS = {
    "A_equilibrium": 2000,
    "D_prbs_excitation": 2500,
}

# PRBS amplitude
PRBS_AMPLITUDE = 0.50


def generate_prbs_excitation(n_steps: int, amplitude: float, min_period: int = 3, max_period: int = 12, seed: int = 42) -> list:
    """Generate PRBS excitation signal."""
    rng = np.random.RandomState(seed)
    signal = []
    i = 0
    while i < n_steps:
        period = rng.randint(min_period, max_period + 1)
        val = amplitude if rng.rand() > 0.5 else -amplitude
        signal.extend([val] * min(period, n_steps - i))
        i += period
    return signal[:n_steps]


def prbs_to_push_sequence(prbs_signal: list, sagittal: bool = True) -> list:
    """Convert PRBS signal to push sequence entries."""
    sequence = []
    n = len(prbs_signal)
    i = 0
    while i < n:
        val = prbs_signal[i]
        if abs(val) < 1e-9:
            i += 1
            continue
        j = i + 1
        while j < n and abs(prbs_signal[j] - val) < 1e-9:
            j += 1
        dur = j - i
        if sagittal:
            fx, fy = 0.0, float(val)
        else:
            fx, fy = float(val), 0.0
        sequence.append([i, fx, fy, dur])
        i = j
    return sequence


def run_screening_simulation(profile_name: str, height_name: str, height_m: float,
                              run_type: str, steps: int, output_subdir: Path,
                              push_sequence_file: Optional[str] = None) -> Tuple[bool, str]:
    """Run a single MuJoCo simulation for screening."""
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

    if run_type == "D_prbs_excitation" and push_sequence_file:
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
                                   text=True, timeout=600, cwd=str(PROJECT_ROOT))
        success = result.returncode == 0
        msg = "OK" if success else f"FAILED:{result.returncode}"
        return success, msg
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR:{e}"


def find_telemetry_csv(run_dir: Path) -> Optional[Path]:
    """Find the telemetry CSV file in a run directory."""
    csv_files = list(run_dir.glob("telemetry_*.csv"))
    return csv_files[0] if csv_files else None


def load_telemetry(csv_path: Path) -> Optional[List[dict]]:
    """Load telemetry CSV into list of dicts."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return None


def compute_screening_metrics(rows: List[dict], params: dict) -> dict:
    """Compute screening metrics from telemetry rows."""
    if not rows or len(rows) < 500:
        return {"error": "INSUFFICIENT_DATA", "n_rows": len(rows)}

    n = len(rows)
    dt = 0.01  # 100 Hz

    # Extract key signals
    pitch_x = np.array([float(r.get("pitch_x", 0)) for r in rows])
    pitch_rate_x = np.array([float(r.get("pitch_rate_x", 0)) for r in rows])
    support_error = np.array([float(r.get("k1_support_error_m", 0)) for r in rows])
    notch_output = np.array([float(r.get("k1_notch_output", 0)) for r in rows])
    body_height = np.array([float(r.get("height_variant_achieved_com_z_m", r.get("body_height", 0.40))) for r in rows])
    clip_active_raw = np.array([r.get("k1_tau_total_clip_active", "0") for r in rows])
    clip_active = np.array([1.0 if str(v) in ("True", "true", "1", "1.0") else 0.0 for v in clip_active_raw])

    # Check for fall (body height below 0.20m or NaN)
    has_fall = bool(np.any(body_height[len(body_height)//2:] < 0.20) or
                    np.any(~np.isfinite(pitch_x)))

    # Check for NaN/Inf
    has_nan = bool(np.any(~np.isfinite(pitch_x)) or np.any(~np.isfinite(notch_output)))

    # Basic safety metrics
    pitch_abs_max = float(np.max(np.abs(pitch_x[-500:]))) if n >= 500 else float(np.max(np.abs(pitch_x)))
    pitch_rms = float(np.sqrt(np.mean(pitch_x[-500:]**2))) if n >= 500 else float(np.sqrt(np.mean(pitch_x**2)))
    support_rms = float(np.sqrt(np.mean(support_error[-500:]**2))) if n >= 500 else float(np.sqrt(np.mean(support_error**2)))
    body_height_min = float(np.min(body_height[-500:])) if n >= 500 else float(np.min(body_height))
    notch_output_rms = float(np.sqrt(np.mean(notch_output[-500:]**2))) if n >= 500 else float(np.sqrt(np.mean(notch_output**2)))
    pitch_rate_rms = float(np.sqrt(np.mean(pitch_rate_x[-500:]**2))) if n >= 500 else float(np.sqrt(np.mean(pitch_rate_x**2)))

    # Torque clipping fraction
    clip_fraction = float(np.mean(clip_active[-500:])) if n >= 500 else float(np.mean(clip_active))

    # ---- Spectral analysis ----
    # Use Welch-like approach: compute PSD via FFT on last 1024 samples
    fft_n = min(1024, n)
    signal_section = pitch_x[-fft_n:]
    signal_section = signal_section - np.mean(signal_section)

    # Apply Hanning window
    window = np.hanning(len(signal_section))
    signal_windowed = signal_section * window

    # FFT
    fft = np.fft.rfft(signal_windowed, n=fft_n)
    psd = np.abs(fft)**2 / (np.sum(window**2) * 100.0)  # Normalize by fs
    freqs = np.fft.rfftfreq(fft_n, d=dt)

    # Low-frequency band: 0.15-0.55 Hz
    lf_mask = (freqs >= 0.15) & (freqs <= 0.55)
    lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
    lf_peak_idx = np.argmax(psd[lf_mask]) if np.any(lf_mask) else 0
    lf_peak_freq = float(freqs[lf_mask][lf_peak_idx]) if np.any(lf_mask) else 0.0

    # WIP band: 2.0-3.0 Hz
    wip_mask = (freqs >= 2.0) & (freqs <= 3.0)
    wip_power = float(np.sum(psd[wip_mask])) if np.any(wip_mask) else 0.0

    # Also compute PSD for notch output
    notch_section = notch_output[-fft_n:]
    notch_section = notch_section - np.mean(notch_section)
    notch_windowed = notch_section * window
    notch_fft = np.fft.rfft(notch_windowed, n=fft_n)
    notch_psd = np.abs(notch_fft)**2 / (np.sum(window**2) * 100.0)

    # Notch output power in WIP band
    notch_wip_power = float(np.sum(notch_psd[wip_mask])) if np.any(wip_mask) else 0.0

    # Cross-spectral coherence (magnitude-squared coherence via FFT)
    # Between pitch and notch output
    fft_pitch = np.fft.rfft(signal_windowed, n=fft_n)
    fft_notch = np.fft.rfft(notch_windowed, n=fft_n)

    # Smoothing: average over 3 adjacent frequency bins
    n_freqs = len(freqs)
    cross_spec = fft_pitch * np.conj(fft_notch)
    auto_pitch = np.abs(fft_pitch)**2
    auto_notch = np.abs(fft_notch)**2

    # Simple coherence estimate with 3-bin smoothing
    coherence = np.zeros(n_freqs)
    for i in range(1, n_freqs - 1):
        cs_sm = np.mean(cross_spec[max(0,i-1):i+2])
        ap_sm = np.mean(auto_pitch[max(0,i-1):i+2])
        an_sm = np.mean(auto_notch[max(0,i-1):i+2])
        denom = ap_sm * an_sm
        if denom > 1e-30:
            coherence[i] = np.abs(cs_sm)**2 / denom

    # Peak coherence in low-freq band
    lf_coherence = float(np.max(coherence[lf_mask])) if np.any(lf_mask) else 0.0

    # WIP band pitch_rate power
    pr_section = pitch_rate_x[-fft_n:]
    pr_section = pr_section - np.mean(pr_section)
    pr_windowed = pr_section * window
    pr_fft = np.fft.rfft(pr_windowed, n=fft_n)
    pr_psd = np.abs(pr_fft)**2 / (np.sum(window**2) * 100.0)
    wip_pitch_rate_power = float(np.sum(pr_psd[wip_mask])) if np.any(wip_mask) else 0.0

    metrics = {
        "n_rows": n,
        "has_fall": has_fall,
        "has_nan": has_nan,
        "pitch_abs_max_deg": float(np.degrees(pitch_abs_max)),
        "pitch_rms_deg": float(np.degrees(pitch_rms)),
        "support_rms_m": support_rms,
        "body_height_min_m": body_height_min,
        "notch_output_rms": notch_output_rms,
        "pitch_rate_rms_rad_s": pitch_rate_rms,
        "clip_fraction": clip_fraction,
        "lf_power_0p15_0p55_hz": lf_power,
        "lf_peak_freq_hz": lf_peak_freq,
        "lf_pitch_notch_coherence": lf_coherence,
        "wip_power_2p0_3p0_hz": wip_power,
        "wip_notch_power_2p0_3p0_hz": notch_wip_power,
        "wip_pitch_rate_power_2p0_3p0_hz": wip_pitch_rate_power,
    }
    return metrics


def run_sweep(dry_run: bool = False, resume: bool = True, group_filter: Optional[str] = None):
    """Run the full filter parameter sweep."""
    print("=" * 72)
    print("K1 NOTCH FILTER PARAMETER AND TOPOLOGY SWEEP")
    print(f"  Dry Run: {dry_run}")
    print(f"  Resume: {resume}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all sweep candidates
    all_candidates = {}
    all_candidates.update(GROUP_A_FC_SWEEP)
    all_candidates.update(GROUP_B_Q_SWEEP)
    all_candidates.update(GROUP_C_BLEND_SWEEP)
    all_candidates.update(GROUP_D_TOPOLOGY)

    if group_filter:
        all_candidates = {k: v for k, v in all_candidates.items()
                         if v.get("group", "") == group_filter}

    # Deduplicate (K1 baseline appears in multiple groups)
    unique_candidates = {}
    for name, params in all_candidates.items():
        if name not in unique_candidates:
            unique_candidates[name] = params

    print(f"\n  Candidates: {len(unique_candidates)} unique profiles")
    print(f"  Heights: {list(SCREENING_HEIGHTS.keys())}")
    print(f"  Run Types: {SCREENING_RUN_TYPES}")

    results = {}

    for profile_name, params in unique_candidates.items():
        results[profile_name] = {"params": params, "runs": {}}

        for height_name, height_m in SCREENING_HEIGHTS.items():
            for run_type in SCREENING_RUN_TYPES:
                run_label = f"{height_name}/{run_type}"
                run_dir = OUTPUT_DIR / "screening_runs" / profile_name / height_name / run_type
                run_dir.mkdir(parents=True, exist_ok=True)
                steps = SCREENING_STEPS[run_type]

                # Resume support
                existing_csv = find_telemetry_csv(run_dir)
                if resume and existing_csv:
                    print(f"  [{profile_name}] {run_label} -> SKIP (existing CSV)")
                    rows = load_telemetry(existing_csv)
                    if rows:
                        metrics = compute_screening_metrics(rows, params)
                        results[profile_name]["runs"][run_label] = metrics
                    else:
                        results[profile_name]["runs"][run_label] = {"error": "CSV_READ_FAILED"}
                    continue

                if dry_run:
                    print(f"  [{profile_name}] {run_label} -> DRY_RUN")
                    results[profile_name]["runs"][run_label] = {"status": "dry_run"}
                    continue

                print(f"  [{profile_name}] {run_label} ({steps} steps)...", end=" ", flush=True)

                # Prepare PRBS push sequence if needed
                push_seq_file = None
                if run_type == "D_prbs_excitation":
                    prbs_signal = generate_prbs_excitation(steps, PRBS_AMPLITUDE, seed=hash(profile_name) % 10000)
                    push_seq = prbs_to_push_sequence(prbs_signal, sagittal=True)
                    push_seq_path = run_dir / "push_sequence.json"
                    with open(push_seq_path, "w") as f:
                        json.dump({"sequence": push_seq, "description": "PRBS sagittal excitation",
                                   "amplitude_n": PRBS_AMPLITUDE, "n_entries": len(push_seq)}, f, indent=2)
                    push_seq_file = str(push_seq_path)

                    # Also save excitation signal metadata
                    exc = {
                        "signal": [float(x) for x in prbs_signal],
                        "n_steps": len(prbs_signal),
                        "amplitude_max": float(max(abs(x) for x in prbs_signal)),
                        "run_type": run_type,
                    }
                    with open(run_dir / "excitation_signal.json", "w") as f:
                        json.dump(exc, f, indent=2)

                success, msg = run_screening_simulation(
                    profile_name, height_name, height_m, run_type, steps,
                    run_dir, push_sequence_file=push_seq_file,
                )
                print(msg)

                if success:
                    csv_path = find_telemetry_csv(run_dir)
                    if csv_path:
                        rows = load_telemetry(csv_path)
                        if rows:
                            metrics = compute_screening_metrics(rows, params)
                            metrics["status"] = "OK"
                            results[profile_name]["runs"][run_label] = metrics
                        else:
                            results[profile_name]["runs"][run_label] = {"status": "CSV_READ_FAILED"}
                    else:
                        results[profile_name]["runs"][run_label] = {"status": "NO_CSV"}
                else:
                    results[profile_name]["runs"][run_label] = {"status": msg}

    # Save screening results (skip in dry-run mode)
    if not dry_run:
        screening_output = OUTPUT_DIR / "screening_results.json"
        with open(screening_output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {screening_output}")

    # Print summary
    print("\n" + "=" * 72)
    print("SCREENING SUMMARY")
    print("=" * 72)
    for profile_name, data in sorted(results.items()):
        eq_key = "high_0p480/A_equilibrium"
        eq_metrics = data["runs"].get(eq_key, {})
        if "lf_peak_freq_hz" in eq_metrics:
            print(f"  {profile_name:35s}  LF peak={eq_metrics['lf_peak_freq_hz']:.3f} Hz  "
                  f"coh={eq_metrics.get('lf_pitch_notch_coherence', 0):.3f}  "
                  f"pitch_rms={eq_metrics['pitch_rms_deg']:.2f} deg  "
                  f"wip={eq_metrics.get('wip_power_2p0_3p0_hz', 0):.4f}  "
                  f"fall={eq_metrics.get('has_fall', False)}")
        else:
            status = eq_metrics.get("status", eq_metrics.get("error", "UNKNOWN"))
            print(f"  {profile_name:35s}  {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="K1 Notch Filter Parameter and Topology Sweep")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print planned runs without executing")
    parser.add_argument("--no-resume", action="store_true",
                       help="Do not skip existing results")
    parser.add_argument("--group", type=str, default=None,
                       choices=["A_fc", "B_Q", "C_blend", "D_topo"],
                       help="Run only one sweep group")
    parser.add_argument("--fast", action="store_true",
                       help="Fast screening: equilibrium only, 1000 steps (skip PRBS)")
    args = parser.parse_args()

    if args.fast:
        global SCREENING_RUN_TYPES, SCREENING_STEPS
        SCREENING_RUN_TYPES = ["A_equilibrium"]
        SCREENING_STEPS = {"A_equilibrium": 1000}

    run_sweep(dry_run=args.dry_run, resume=not args.no_resume, group_filter=args.group)


if __name__ == "__main__":
    main()
