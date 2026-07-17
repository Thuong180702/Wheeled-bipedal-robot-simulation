"""Run D4/D5 focused validation for hip-yaw architecture fix.

Runs D4_medium_push_low and D5_large_push_high for profiles A-D.
Profile D = low-band v2 + differential wheel yaw stabilizer.
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_outer_loop_step_d_push as runner

ROOT = Path(__file__).resolve().parent.parent

PROFILE_A = "calibrated_support_position_outer_loop_pitch_ref_v2"
PROFILE_B = "physics_equilibrium_feedforward_outer_loop"
PROFILE_C = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
PROFILE_D = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"

# Candidate D tuning (opt-in architecture candidate)
D_WHEEL_YAW_KP = 5.0
D_WHEEL_YAW_KD = 1.5
D_WHEEL_YAW_MAX_TORQUE = 5.0
D_WHEEL_YAW_LOWPASS_ALPHA = 1.0
D_WHEEL_YAW_HEIGHT_GATE_LOW = 0.250
D_WHEEL_YAW_HEIGHT_GATE_HIGH = 0.350
D_YAW_CONTROLLER_KP = 8.0
D_YAW_CONTROLLER_KD = 2.0
D_YAW_CONTROLLER_MAX_TORQUE = 5.0

OUT_BASE = ROOT / "outputs" / "hip_yaw_push_limit_architecture_fix" / "d4_d5_validation"

# Only D4 and D5
D4_D5_CASES = [
    ("D4_medium_push_low",  "low_0p330", 1000, 60,  5, 150),
    ("D5_large_push_high",  "high_0p480", 1000, 90,  5, 200),
]


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case_id, height_label, steps, push_mag, push_dur, push_int in D4_D5_CASES:
        for profile, tag, use_wheel_yaw in [
            (PROFILE_A, "A", False),
            (PROFILE_B, "B", False),
            (PROFILE_C, "C", False),
            (PROFILE_D, "D", True),
        ]:
            out_dir = OUT_BASE / f"step_{case_id}_{tag}"
            t0 = time.time()
            print(f"\n[{case_id}] [{tag}] Starting...", end=" ", flush=True)
            tel_path, _ = runner.run_sim(
                height_label, steps, profile, out_dir,
                push_magnitude=push_mag,
                push_duration=push_dur,
                push_interval=push_int,
                enable_wheel_yaw=use_wheel_yaw,
                wheel_yaw_kp=D_WHEEL_YAW_KP if use_wheel_yaw else None,
                wheel_yaw_kd=D_WHEEL_YAW_KD if use_wheel_yaw else None,
                wheel_yaw_max_torque=D_WHEEL_YAW_MAX_TORQUE if use_wheel_yaw else None,
                wheel_yaw_lowpass_alpha=D_WHEEL_YAW_LOWPASS_ALPHA if use_wheel_yaw else None,
                wheel_yaw_height_gate_low=D_WHEEL_YAW_HEIGHT_GATE_LOW if use_wheel_yaw else None,
                wheel_yaw_height_gate_high=D_WHEEL_YAW_HEIGHT_GATE_HIGH if use_wheel_yaw else None,
                yaw_controller_kp=D_YAW_CONTROLLER_KP if use_wheel_yaw else None,
                yaw_controller_kd=D_YAW_CONTROLLER_KD if use_wheel_yaw else None,
                yaw_controller_max_torque=D_YAW_CONTROLLER_MAX_TORQUE if use_wheel_yaw else None,
            )
            metrics = runner.analyze(tel_path)
            row = {
                "case_id": case_id,
                "height": height_label,
                "steps": steps,
                "push_mag_N": push_mag,
                "push_dur": push_dur,
                "push_int": push_int,
                "profile": tag,
                "wheel_yaw": use_wheel_yaw,
                **(metrics or {}),
            }
            all_rows.append(row)
            elapsed = time.time() - t0
            safe, reason = runner.safety_ok(metrics)
            hy = metrics.get("hip_yaw_abs_max_rad", 0.0) if metrics else 0.0
            hy_pass = hy < 0.35
            print(
                f"done ({elapsed:.0f}s) "
                f"safe={safe}({reason}) "
                f"hip_yaw={hy:.4f} {'PASS' if hy_pass else 'FAIL'}",
                flush=True,
            )
    # Write CSV
    csv_path = OUT_BASE / "d4_d5_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nResults: {csv_path}")

    # Summary
    print("\n" + "=" * 78)
    print("HIP-YAW ARCHITECTURE FIX: D4/D5 VALIDATION SUMMARY")
    print("=" * 78)
    for row in all_rows:
        hy = row.get("hip_yaw_abs_max_rad", 0.0)
        fell = row.get("fell", False)
        max_abs = row.get("max_abs", 0.0)
        print(f"  [{row['profile']}] {row['case_id']}: "
              f"hip_yaw={hy:.4f} {'OK' if hy < 0.35 else 'EXCEEDS'}, "
              f"fell={fell}, max_drift={max_abs:.4f}")


if __name__ == "__main__":
    main()
