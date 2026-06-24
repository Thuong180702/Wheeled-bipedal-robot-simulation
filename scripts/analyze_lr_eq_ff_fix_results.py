"""Quick analysis of K1 + LR focused recovery telemetry after LR EQ/FF fix."""
import csv
import json
import math
from pathlib import Path

import numpy as np

BASE = Path("outputs/k1_lr_eq_ff_fix/focused_recovery")

def analyze_telemetry(csv_path: Path):
    """Extract key metrics from telemetry CSV."""
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "No data rows"}

    n = len(rows)

    def col(name, dtype=float):
        vals = []
        for r in rows:
            v = r.get(name, "")
            if v == "" or v is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(dtype(v))
                except (ValueError, TypeError):
                    vals.append(np.nan)
        return np.array(vals)

    pitch = col("pitch_x_rad")
    roll = col("roll_y_rad")
    support = col("sagittal_position_error_m")
    hip_yaw_l = col("l_hip_yaw_pos_rad")
    hip_yaw_r = col("r_hip_yaw_pos_rad")
    com_z = col("com_z_m")

    # Pitch in degrees
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)

    # LR specific
    lr_enabled_val = rows[0].get("LR_enabled", "False") if rows else "False"
    lr_enabled = lr_enabled_val in ("True", "true", "1")
    lr_eq_ff = col("LR_eq_ff_pass_through_nm")
    lr_feedback = col("LR_feedback_torque_nm")
    lr_k1_est = col("LR_k1_existing_estimate_nm")
    lr_removed = col("LR_removed_dynamic_terms_estimate_nm")
    lr_total_preclip = col("LR_total_command_preclip_nm")
    lr_total_postclip = col("LR_total_command_postclip_nm")

    # Torque components
    tau_pitch = col("tau_pitch_nm")
    tau_pitch_rate = col("tau_pitch_rate_nm")
    tau_sagittal = col("tau_sagittal_velocity_nm")
    tau_support = col("tau_support_velocity_nm")
    tau_position = col("tau_position_nm")
    tau_cp = col("tau_cp_nm")
    tau_com_vy = col("tau_com_vy_nm")
    tau_common = col("tau_common_final_nm")

    # Termination
    terminated_val = rows[-1].get("terminated", "0") if rows else "0"
    terminated = str(terminated_val).strip() in ("True", "true", "1")

    # Fall detection: large pitch or low height
    max_pitch = np.nanmax(np.abs(pitch_deg))
    min_height = np.nanmin(com_z)

    def rms(x):
        return float(np.sqrt(np.nanmean(x**2)))

    # Push recovery metrics (steps 300-600)
    post_push_start = 310
    post_push_end = min(n, 600)
    pre_push = slice(0, 300)
    post_push = slice(post_push_start, post_push_end)
    final_window = slice(max(0, n - 500), n)

    # Sustained hold detection: pitch within +/-3 deg for at least 2s (200 steps)
    pitch_in_band = np.abs(pitch_deg) < 5.0
    sustained_2s = False
    sustained_5s = False
    hold_start = None
    for i in range(n):
        if pitch_in_band[i]:
            if hold_start is None:
                hold_start = i
            elif i - hold_start >= 200:  # 2 seconds at 100Hz
                sustained_2s = True
            if i - hold_start >= 500:  # 5 seconds
                sustained_5s = True
        else:
            if hold_start is not None and i - hold_start >= 200:
                pass  # already counted
            hold_start = None

    # Dominant frequency (FFT on pitch)
    pitch_no_nan = pitch_deg[~np.isnan(pitch_deg)]
    if len(pitch_no_nan) > 256:
        fft = np.abs(np.fft.rfft(pitch_no_nan - np.mean(pitch_no_nan)))
        freqs = np.fft.rfftfreq(len(pitch_no_nan), 0.01)
        # Find peak between 0.3 and 3 Hz
        mask = (freqs >= 0.3) & (freqs <= 3.0)
        if mask.any():
            idx = np.argmax(fft[mask])
            dom_freq = freqs[mask][idx]
            dom_amp = fft[mask][idx] / len(pitch_no_nan) * 2  # amplitude
        else:
            dom_freq = 0.0
            dom_amp = 0.0

        # 0.52 Hz amplitude
        idx_052 = np.argmin(np.abs(freqs - 0.52))
        amp_052 = fft[idx_052] / len(pitch_no_nan) * 2 if idx_052 < len(fft) else 0.0

        # 2.5 Hz amplitude
        idx_25 = np.argmin(np.abs(freqs - 2.5))
        amp_25 = fft[idx_25] / len(pitch_no_nan) * 2 if idx_25 < len(fft) else 0.0
    else:
        dom_freq = 0.0
        dom_amp = 0.0
        amp_052 = 0.0
        amp_25 = 0.0

    hip_yaw_abs_max = float(max(
        np.nanmax(np.abs(hip_yaw_l)),
        np.nanmax(np.abs(hip_yaw_r))
    )) if not np.all(np.isnan(hip_yaw_l)) else 0.0

    result = {
        "n_rows": n,
        "completed_steps": n,
        "terminated": terminated,
        "fall": terminated or max_pitch > 45.0 or min_height < 0.2,
        "pitch_rms_deg": rms(pitch_deg),
        "final_pitch_rms_deg": rms(pitch_deg[final_window]),
        "pitch_max_deg": float(max_pitch),
        "support_rms_m": rms(support),
        "final_support_rms_m": rms(support[final_window]),
        "support_max_m": float(np.nanmax(np.abs(support))),
        "roll_rms_deg": rms(roll_deg),
        "roll_max_deg": float(np.nanmax(np.abs(roll_deg))),
        "hip_yaw_abs_max_rad": hip_yaw_abs_max,
        "sustained_2s_hold": sustained_2s,
        "sustained_5s_hold": sustained_5s,
        "dominant_freq_hz": float(dom_freq),
        "dominant_amp_deg": float(dom_amp),
        "amp_0p52_hz_deg": float(amp_052),
        "amp_2p5_hz_deg": float(amp_25),
        "lr_enabled": lr_enabled,
        "lr_eq_ff_pass_through_rms": rms(lr_eq_ff),
        "lr_eq_ff_pass_through_max": float(np.nanmax(np.abs(lr_eq_ff))),
        "lr_feedback_rms": rms(lr_feedback),
        "lr_feedback_max": float(np.nanmax(np.abs(lr_feedback))),
        "lr_total_preclip_rms": rms(lr_total_preclip),
        "lr_total_postclip_rms": rms(lr_total_postclip),
        "lr_k1_est_rms": rms(lr_k1_est),
        "lr_removed_dynamic_rms": rms(lr_removed),
        "tau_pitch_rms": rms(tau_pitch),
        "tau_pitch_rate_rms": rms(tau_pitch_rate),
        "tau_sagittal_rms": rms(tau_sagittal),
        "tau_support_rms": rms(tau_support),
        "tau_position_rms": rms(tau_position),
        "tau_cp_rms": rms(tau_cp),
        "tau_com_vy_rms": rms(tau_com_vy),
        "tau_common_rms": rms(tau_common),
        "min_height_m": float(min_height),
        "max_height_m": float(np.nanmax(com_z)),
    }

    return result

def main():
    runs = {
        "K1_baseline": BASE / "k1_baseline",
        "LR1": BASE / "lr1",
        "LR2": BASE / "lr2",
        "LR3": BASE / "lr3",
    }

    results = {}
    for name, dirpath in runs.items():
        csvs = list(dirpath.glob("telemetry_*.csv"))
        if csvs:
            csv_path = csvs[0]
            results[name] = analyze_telemetry(csv_path)
        else:
            results[name] = {"error": "No telemetry CSV found"}

    # Convert numpy types to native Python
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj
    print(json.dumps(clean(results), indent=2))

    # Print summary table
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    print(f"{'Metric':<35} {'K1':>15} {'LR1':>15} {'LR2':>15} {'LR3':>15}")
    print("-"*100)

    metrics = [
        ("completed_steps", "Completed steps", "d"),
        ("fall", "Fall?", "b"),
        ("pitch_rms_deg", "Pitch RMS [deg]", ".2f"),
        ("final_pitch_rms_deg", "Final pitch RMS [deg]", ".2f"),
        ("pitch_max_deg", "Pitch max [deg]", ".1f"),
        ("support_rms_m", "Support RMS [m]", ".3f"),
        ("final_support_rms_m", "Final support RMS [m]", ".3f"),
        ("roll_rms_deg", "Roll RMS [deg]", ".2f"),
        ("roll_max_deg", "Roll max [deg]", ".1f"),
        ("hip_yaw_abs_max_rad", "Hip yaw abs max [rad]", ".3f"),
        ("sustained_2s_hold", "Sustained 2s hold?", "b"),
        ("sustained_5s_hold", "Sustained 5s hold?", "b"),
        ("dominant_freq_hz", "Dominant freq [Hz]", ".2f"),
        ("amp_0p52_hz_deg", "0.52 Hz amp [deg]", ".3f"),
        ("amp_2p5_hz_deg", "2.5 Hz amp [deg]", ".3f"),
        ("lr_eq_ff_pass_through_rms", "LR EQ/FF pass-through RMS [Nm]", ".2f"),
        ("lr_eq_ff_pass_through_max", "LR EQ/FF pass-through max [Nm]", ".1f"),
        ("lr_feedback_rms", "LR feedback RMS [Nm]", ".3f"),
        ("lr_total_preclip_rms", "LR total preclip RMS [Nm]", ".2f"),
        ("lr_total_postclip_rms", "LR total postclip RMS [Nm]", ".2f"),
        ("lr_k1_est_rms", "LR K1 existing estimate RMS [Nm]", ".2f"),
        ("lr_removed_dynamic_rms", "LR removed dynamic RMS [Nm]", ".3f"),
        ("tau_pitch_rms", "tau_pitch RMS [Nm]", ".2f"),
        ("tau_pitch_rate_rms", "tau_pitch_rate RMS [Nm]", ".2f"),
        ("tau_sagittal_rms", "tau_sagittal_velocity RMS [Nm]", ".3f"),
        ("tau_support_rms", "tau_support_velocity RMS [Nm]", ".3f"),
        ("tau_position_rms", "tau_position RMS [Nm]", ".2f"),
        ("tau_common_rms", "tau_common_final RMS [Nm]", ".2f"),
    ]

    for key, label, fmt in metrics:
        vals = []
        for name in ["K1_baseline", "LR1", "LR2", "LR3"]:
            r = results.get(name, {})
            v = r.get(key, None)
            if v is None:
                vals.append("N/A")
            elif fmt == "b":
                vals.append("YES" if v else "no")
            elif fmt == "d":
                vals.append(str(int(v)))
            else:
                vals.append(f"{v:{fmt}}")
        print(f"{label:<35} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15} {vals[3]:>15}")

    # Torque decomposition
    print("\n" + "="*100)
    print("TORQUE DECOMPOSITION (RMS, Nm)")
    print("="*100)
    torque_keys = [
        ("tau_pitch_rms", "tau_pitch"),
        ("tau_pitch_rate_rms", "tau_pitch_rate"),
        ("tau_sagittal_rms", "tau_sagittal_velocity"),
        ("tau_support_rms", "tau_support_velocity"),
        ("tau_position_rms", "tau_position"),
        ("tau_common_rms", "tau_common_final"),
        ("lr_eq_ff_pass_through_rms", "LR EQ/FF pass-through"),
        ("lr_feedback_rms", "LR dynamic feedback"),
        ("lr_total_preclip_rms", "LR total preclip"),
    ]
    for key, label in torque_keys:
        vals = []
        for name in ["K1_baseline", "LR1", "LR2", "LR3"]:
            r = results.get(name, {})
            v = r.get(key, 0.0) or 0.0
            vals.append(f"{v:.2f}")
        print(f"{label:<35} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

if __name__ == "__main__":
    main()
