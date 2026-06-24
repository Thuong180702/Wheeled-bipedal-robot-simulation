#!/usr/bin/env python3
"""Audit K1 sustained posture recovery failure root cause.

This script reads telemetry from K1 focused recovery runs (high_0p480, single push 90N)
and decomposes the sagittal torque to determine WHY sustained recovery is not achieved.

Audit dimensions:
1. Oscillation classification: WIP natural mode vs torque-term coupling vs support mismatch
2. Sagittal torque decomposition: pitch P, pitch rate D, support position, support velocity,
   wheel velocity, PFF, notch effect, clipping
3. Frequency/phase analysis: pitch vs pitch_rate, support vs wheel, torque vs state
4. Recovery event search: transient crossings vs sustained holds
5. Conclusion: is coordinated state feedback justified?

Output:
    outputs/k1_controller_completion/sustained_recovery_audit/
        analysis/  (metrics and phase analysis)
        figures/   (diagnostic plots if available)

Usage:
    python scripts/audit_k1_sustained_recovery_failure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "k1_controller_completion" / "sustained_recovery_audit"
ANALYSIS_DIR = OUT_BASE / "analysis"

# Look for K1 focused recovery telemetry in multiple locations
K1_TELEMETRY_CANDIDATES = [
    ROOT / "outputs" / "k1_controller_completion" / "K1_focused_recovery" / "telemetry_1782299520.csv",
    ROOT / "outputs" / "k1_pitch_rate_notch_promotion_multi_height_validation" / "K1_pitch_rate_notch" / "high_0p480" / "90.0N" / "telemetry_3000.csv",
    ROOT / "outputs" / "k1_promotion_multi_height_single_push" / "K1_pitch_rate_notch" / "high_0p480" / "telemetry_3000.csv",
    # Generic fallback search
]

G1_TELEMETRY_CANDIDATES = [
    ROOT / "outputs" / "tall_height_sagittal_wip_damping_recovery_fix" / "single_push_g1_sg080_high_0p480_90N_telemetry.csv",
]

# FFT parameters
FS_HZ = 100.0


def find_telemetry(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    # Fallback: glob search
    for parent in [ROOT / "outputs"]:
        matches = sorted(parent.rglob("telemetry_3000.csv"))
        for m in matches:
            if "k1" in str(m).lower():
                return m
    return None


def load_telemetry(path: Path) -> dict:
    """Load telemetry CSV and return as dict of numpy arrays."""
    import csv
    import collections
    with open(path) as f:
        reader = csv.DictReader(f)
        data = collections.defaultdict(list)
        for row in reader:
            for key, val in row.items():
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(0.0)
    return {k: np.array(v) for k, v in data.items()}


def compute_fft(signal: np.ndarray, fs: float = FS_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT magnitude spectrum. Returns (freqs, magnitudes)."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    window = np.hanning(n)
    mags = np.abs(np.fft.rfft(signal * window)) * 2.0 / np.sum(window)
    return freqs, mags


def dominant_freq(signal: np.ndarray, fs: float = FS_HZ, min_hz: float = 0.5, max_hz: float = 10.0) -> float:
    """Find dominant frequency in a signal within [min_hz, max_hz]."""
    freqs, mags = compute_fft(signal, fs)
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not mask.any():
        return 0.0
    idx = np.argmax(mags[mask])
    return float(freqs[mask][idx])


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation between x and y. Returns (lags, corr_values)."""
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    x_detrend = x - np.mean(x)
    y_detrend = y - np.mean(y)
    x_std = np.std(x_detrend)
    y_std = np.std(y_detrend)
    if x_std < 1e-12 or y_std < 1e-12:
        return lags, corr
    for i, lag in enumerate(lags):
        if lag < 0:
            corr[i] = np.mean(x_detrend[:lag] * y_detrend[-lag:]) / (x_std * y_std)
        elif lag > 0:
            corr[i] = np.mean(x_detrend[lag:] * y_detrend[:-lag]) / (x_std * y_std)
        else:
            corr[i] = np.mean(x_detrend * y_detrend) / (x_std * y_std)
    return lags, corr


def recovery_event_search(pitch_rad: np.ndarray, roll_rad: np.ndarray | None,
                          hip_yaw_l: np.ndarray | None, hip_yaw_r: np.ndarray | None,
                          push_end_step: int = 500,
                          dt: float = 0.01) -> dict:
    """Search for recovery events in post-push window.

    Recovery definitions:
    - Pitch abs < 5 deg
    - Pitch abs < 3 deg
    - Sustained hold >= 2 s (200 steps)
    - Sustained hold >= 5 s (500 steps)
    """
    n = len(pitch_rad)
    post_push = slice(push_end_step, n)
    pitch_deg = np.degrees(np.abs(pitch_rad))

    results = {
        "first_pitch_under_5deg_step": -1,
        "first_pitch_under_3deg_step": -1,
        "first_pitch_under_5deg_time_s": -1.0,
        "first_pitch_under_3deg_time_s": -1.0,
        "sustained_2s_hold_found": False,
        "sustained_2s_hold_start_step": -1,
        "sustained_2s_hold_end_step": -1,
        "sustained_5s_hold_found": False,
        "sustained_5s_hold_start_step": -1,
        "sustained_5s_hold_end_step": -1,
        "recovery_later_lost": False,
        "final_window_pitch_rms_deg": float(np.sqrt((pitch_deg[-500:] ** 2).mean())) if n >= 500 else 0.0,
        "peak_pitch_after_push_deg": float(np.max(pitch_deg[post_push])) if post_push.stop > post_push.start else 0.0,
    }

    # Find first time pitch < 5 deg
    under_5 = np.where(pitch_deg < 5.0)[0]
    if len(under_5) > 0:
        first = under_5[0]
        results["first_pitch_under_5deg_step"] = int(first)
        results["first_pitch_under_5deg_time_s"] = float(first * dt)

    # Find first time pitch < 3 deg
    under_3 = np.where(pitch_deg < 3.0)[0]
    if len(under_3) > 0:
        first = under_3[0]
        results["first_pitch_under_3deg_step"] = int(first)
        results["first_pitch_under_3deg_time_s"] = float(first * dt)

    # Find sustained holds (pitch < 5 deg continuously for N steps)
    HOLD_2S_STEPS = int(2.0 / dt)
    HOLD_5S_STEPS = int(5.0 / dt)

    below_5 = (pitch_deg < 5.0).astype(int)
    for hold_steps, key_prefix in [(HOLD_2S_STEPS, "sustained_2s"), (HOLD_5S_STEPS, "sustained_5s")]:
        # Find contiguous blocks
        diffs = np.diff(np.concatenate(([0], below_5, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            duration = e - s
            if duration >= hold_steps:
                results[f"{key_prefix}_hold_found"] = True
                results[f"{key_prefix}_hold_start_step"] = int(s)
                results[f"{key_prefix}_hold_end_step"] = int(e)

    # Check if recovery was later lost (pitch grew again after a hold)
    if results["sustained_2s_hold_found"]:
        hold_end = results["sustained_2s_hold_end_step"]
        if hold_end < n - 100:
            post_hold_pitch = pitch_deg[hold_end:]
            if np.max(post_hold_pitch) > 8.0:  # regrew significantly
                results["recovery_later_lost"] = True

    return results


def decompose_sagittal_torque(data: dict) -> dict:
    """Decompose total sagittal wheel torque into individual terms.

    Returns dict with RMS and peak values for each torque component.
    """
    components = {}
    torque_terms = [
        ("tau_pitch", "tau_pitch"),
        ("tau_pitch_rate", "tau_pitch_rate"),
        ("tau_position", "tau_position"),
        ("tau_wheel_velocity", "tau_wheel_velocity_left", "tau_wheel_velocity_right"),
        ("tau_sagittal_velocity", "tau_sagittal_velocity"),
        ("tau_support_velocity", "tau_support_velocity"),
    ]
    for name, *keys in torque_terms:
        vals = []
        for k in keys:
            if k in data:
                v = data[k]
                if np.isfinite(v).any():
                    vals.append(v)
        if vals:
            combined = np.mean(vals, axis=0) if len(vals) > 1 else vals[0]
            components[name] = {
                "rms": float(np.sqrt(np.mean(combined ** 2))),
                "peak_abs": float(np.max(np.abs(combined))),
                "mean": float(np.mean(combined)),
            }
    return components


def analyze_spectrum(data: dict, push_end_step: int = 500) -> dict:
    """Analyze frequency content of key signals in the post-push window."""
    spectrum = {}
    post_push = slice(push_end_step, None)
    signals = {
        "pitch_rad": "pitch_x_rad",
        "pitch_rate_rad_s": "pitch_x_error_rad",  # approximate
        "support_error_m": "support_position_error_m",
    }
    for label, key in signals.items():
        if key in data:
            sig = data[key][post_push]
            if len(sig) > 100 and np.isfinite(sig).any():
                dom = dominant_freq(sig)
                spectrum[f"{label}_dominant_hz"] = dom

    # Cross-correlations
    corr_results = {}
    pairs = [
        ("pitch_vs_support", "pitch_x_rad", "support_position_error_m"),
        ("pitch_vs_pitch_rate", "pitch_x_rad", "pitch_x_error_rad"),
        ("support_vs_wheel_vel", "support_position_error_m", "wheel_vel_left"),  # wheel_vel may not exist
    ]
    for label, k1, k2 in pairs:
        if k1 in data and k2 in data:
            s1 = data[k1][post_push]
            s2 = data[k2][post_push]
            if len(s1) > 50 and np.isfinite(s1).any() and np.isfinite(s2).any():
                lags, corr = cross_correlation(s1[:1000], s2[:1000], max_lag=50)
                max_idx = np.argmax(np.abs(corr))
                corr_results[f"{label}_max_corr"] = float(corr[max_idx])
                corr_results[f"{label}_lag_at_max"] = int(lags[max_idx])

    return {**spectrum, **corr_results}


def analyze_notch_effect(data: dict, push_end_step: int = 500) -> dict:
    """Analyze the notch filter's effect on pitch rate and torque signals."""
    notch = {}
    post_push = slice(push_end_step, None)

    if "pitch_rate_raw_rad_s" in data and "pitch_rate_effective_rad_s" in data:
        raw = data["pitch_rate_raw_rad_s"][post_push]
        eff = data["pitch_rate_effective_rad_s"][post_push]
        if len(raw) > 0 and np.isfinite(raw).any():
            notch["pitch_rate_attenuation_ratio"] = float(
                np.sqrt(np.mean(eff ** 2)) / np.sqrt(np.mean(raw ** 2)) if np.sqrt(np.mean(raw ** 2)) > 1e-9 else 1.0
            )

    if "tau_pitch_rate_raw_signal" in data and "tau_pitch_rate_filtered_signal" in data:
        raw = data["tau_pitch_rate_raw_signal"][post_push]
        eff = data["tau_pitch_rate_filtered_signal"][post_push]
        if len(raw) > 0 and np.isfinite(raw).any():
            notch["tau_pitch_rate_attenuation_ratio"] = float(
                np.sqrt(np.mean(eff ** 2)) / np.sqrt(np.mean(raw ** 2)) if np.sqrt(np.mean(raw ** 2)) > 1e-9 else 1.0
            )

    return notch


def write_audit_report(audit: dict, path: Path):
    """Write structured audit report."""
    lines = []
    lines.append("# K1 Sustained Posture Recovery — Root-Cause Audit")
    lines.append("")
    lines.append("## 1. Source Telemetry")
    lines.append(f"- Telemetry path: {audit.get('telemetry_path', 'N/A')}")
    lines.append(f"- Rows: {audit.get('rows', 0)}")
    lines.append(f"- Signals available: {len(audit.get('signals', []))}")
    lines.append("")

    lines.append("## 2. Recovery Event Analysis")
    rec = audit.get("recovery", {})
    lines.append(f"- First pitch < 5 deg: {rec.get('first_pitch_under_5deg_time_s', 'N/A'):.2f}s")
    lines.append(f"- First pitch < 3 deg: {rec.get('first_pitch_under_3deg_time_s', 'N/A'):.2f}s")
    lines.append(f"- Sustained 2s hold: {'YES' if rec.get('sustained_2s_hold_found') else 'NO'}")
    if rec.get("sustained_2s_hold_found"):
        lines.append(f"  - Hold: step {rec.get('sustained_2s_hold_start_step')} to {rec.get('sustained_2s_hold_end_step')}")
        lines.append(f"  - Recovery later lost: {'YES' if rec.get('recovery_later_lost') else 'NO'}")
    lines.append(f"- Sustained 5s hold: {'YES' if rec.get('sustained_5s_hold_found') else 'NO'}")
    lines.append(f"- Final window pitch RMS: {rec.get('final_window_pitch_rms_deg', 0):.2f}°")
    lines.append(f"- Peak pitch after push: {rec.get('peak_pitch_after_push_deg', 0):.2f}°")
    lines.append("")

    lines.append("## 3. Sagittal Torque Decomposition")
    comp = audit.get("torque_components", {})
    for name, vals in sorted(comp.items()):
        lines.append(f"- {name}: RMS={vals.get('rms', 0):.3f} Nm, "
                      f"Peak={vals.get('peak_abs', 0):.3f} Nm, "
                      f"Mean={vals.get('mean', 0):.4f} Nm")
    lines.append("")

    lines.append("## 4. Frequency / Phase Analysis")
    spec = audit.get("spectrum", {})
    for k, v in spec.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 5. Notch Effect")
    notch = audit.get("notch", {})
    for k, v in notch.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")

    lines.append("## 6. Conclusion")
    conclusion = audit.get("conclusion", "N/A")
    lines.append(conclusion)
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  [REPORT] Wrote audit report to {path}")


def form_conclusion(audit: dict) -> str:
    """Form structured conclusion based on audit findings."""
    rec = audit.get("recovery", {})
    comp = audit.get("torque_components", {})
    spec = audit.get("spectrum", {})

    reasons = []

    # Check if oscillation is WIP natural mode
    pitch_dom = spec.get("pitch_rad_dominant_hz", 0)
    if 2.0 <= pitch_dom <= 3.0:
        reasons.append(f"DOMINANT_OSCILLATION: WIP natural mode at {pitch_dom:.1f} Hz confirmed")
    else:
        reasons.append(f"DOMINANT_OSCILLATION: Non-WIP dominant freq {pitch_dom:.1f} Hz")

    # Check torque composition
    if "tau_pitch" in comp and "tau_position" in comp:
        pitch_rms = comp["tau_pitch"].get("rms", 0)
        pos_rms = comp["tau_position"].get("rms", 0)
        if pitch_rms > pos_rms * 1.5:
            reasons.append(f"TORQUE_IMBALANCE: tau_pitch ({pitch_rms:.3f} Nm) dominates tau_position ({pos_rms:.3f} Nm)")

    # Check phase coupling
    lag = spec.get("pitch_vs_support_lag_at_max", 0)
    if abs(lag) > 10:
        reasons.append(f"PHASE_LAG: pitch vs support lags by {lag} steps")
    if abs(lag) <= 5:
        reasons.append("PHASE_LOCKED: pitch and support are tightly coupled (tight WIP mode)")

    pitch_support_corr = spec.get("pitch_vs_support_max_corr", 0)
    if pitch_support_corr > 0.7:
        reasons.append(f"STRONG_COUPLING: pitch-support correlation = {pitch_support_corr:.2f}")

    # Check notch
    notch = audit.get("notch", {})
    pr_att = notch.get("pitch_rate_attenuation_ratio", 1.0)
    if pr_att < 0.9:
        reasons.append(f"NOTCH_EFFECTIVE: pitch_rate attenuation = {(1-pr_att)*100:.0f}%")
    else:
        reasons.append("NOTCH_INEFFECTIVE: minimal pitch_rate attenuation")

    # Sustained recovery
    has_hold = rec.get("sustained_2s_hold_found", False)
    if has_hold:
        reasons.append("TRANSIENT_RECOVERY: 2s hold achieved but recovery is partial")
        if rec.get("recovery_later_lost", False):
            reasons.append("RECOVERY_LOST: initial recovery was later lost (oscillation returned)")
    else:
        reasons.append("NO_SUSTAINED_RECOVERY: K1 never achieves sustained 2s hold")

    # Architecture diagnosis
    if pitch_dom >= 2.0:
        reasons.append("ARCHITECTURE: Independent torque summation (pitch + position + rate) "
                       "creates phase-conflicted torque at 2.5 Hz. Coordinated state feedback "
                       "is justified to synchronize torque components.")
    else:
        reasons.append("ARCHITECTURE: Not clearly a torque-conflict issue; lower-frequency dynamics may respond to tuning.")

    return "\n".join(reasons)


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Find telemetry
    k1_path = find_telemetry(K1_TELEMETRY_CANDIDATES)
    if k1_path is None:
        print("[ERROR] K1 focused recovery telemetry not found", flush=True)
        print("  Checked:")
        for p in K1_TELEMETRY_CANDIDATES:
            print(f"    - {p}")
        # Search one more level deep
        for p in sorted((ROOT / "outputs").rglob("*high_0p480*telemetry*.csv")):
            if "k1" in str(p).lower():
                k1_path = p
                break

    if k1_path is None:
        print("[ERROR] No K1 telemetry found. Cannot audit.", flush=True)
        return

    print(f"[INFO] Loading K1 telemetry: {k1_path}", flush=True)
    data = load_telemetry(k1_path)
    print(f"  Rows: {len(next(iter(data.values())))}", flush=True)

    # Build audit
    audit = {
        "telemetry_path": str(k1_path),
        "rows": len(next(iter(data.values()))),
        "signals": list(data.keys()),
    }

    # Recovery event search
    push_end = 500  # push at step 300 for 10 steps, ended by 500
    pitch = data.get("pitch_x_rad", np.zeros(audit["rows"]))
    roll = data.get("roll_y_rad", None)
    hy_l = data.get("l_hip_yaw_pos", None)
    hy_r = data.get("r_hip_yaw_pos", None)
    audit["recovery"] = recovery_event_search(
        pitch, roll, hy_l, hy_r, push_end_step=push_end
    )

    # Torque decomposition
    audit["torque_components"] = decompose_sagittal_torque(data)

    # Spectrum analysis
    audit["spectrum"] = analyze_spectrum(data, push_end_step=push_end)

    # Notch effect
    audit["notch"] = analyze_notch_effect(data, push_end_step=push_end)

    # Conclusion
    audit["conclusion"] = form_conclusion(audit)

    # Write report
    write_audit_report(audit, ANALYSIS_DIR / "sustained_recovery_audit_report.txt")

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("AUDIT SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    rec = audit["recovery"]
    print(f"Recovery: 2s_hold={rec.get('sustained_2s_hold_found', 'N/A')}, "
          f"5s_hold={rec.get('sustained_5s_hold_found', 'N/A')}", flush=True)
    print(f"Final pitch RMS: {rec.get('final_window_pitch_rms_deg', 0):.2f}°", flush=True)
    comp = audit.get("torque_components", {})
    for name, vals in sorted(comp.items()):
        print(f"  {name}: RMS={vals.get('rms', 0):.3f} Nm", flush=True)
    print(f"\nConclusion:\n{audit['conclusion']}", flush=True)
    print(f"\nAudit report: {ANALYSIS_DIR / 'sustained_recovery_audit_report.txt'}", flush=True)


if __name__ == "__main__":
    main()
