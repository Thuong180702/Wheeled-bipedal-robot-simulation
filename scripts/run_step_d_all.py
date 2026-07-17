"""Run Step D push cases for profiles A, B, C and log metrics.

Profiles:
    A – calibrated_support_position_outer_loop_pitch_ref_v2 (B2v2)
    B – physics_equilibrium_feedforward_outer_loop (current PFF)
    C – physics_equilibrium_feedforward_outer_loop_low_band_support_v2
    D – same as C + differential wheel yaw stabilizer (architecture fix candidate)
"""
import csv
import sys
import time
from pathlib import Path

# Add scripts/ to path for runner import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_outer_loop_step_d_push as runner

# Profiles per user's Step D comparison matrix:
#   A = calibrated_support_position_outer_loop_pitch_ref_v2 (B2v2 baseline)
#   B = physics_equilibrium_feedforward_outer_loop (current PFF)
#   C = physics_equilibrium_feedforward_outer_loop_low_band_support_v2 (candidate)
#   D = same as C + differential wheel yaw stabilizer (architecture fix)
PROFILE_A = "calibrated_support_position_outer_loop_pitch_ref_v2"
PROFILE_B = "physics_equilibrium_feedforward_outer_loop"
PROFILE_C = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
PROFILE_D = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"  # same sagittal profile + wheel yaw flag

# Base output directory for all Step D runs
ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "hip_yaw_push_limit_architecture_fix" / "step_d_all"


def main() -> None:
    """Run Step D push cases for profiles A, B, C, D and aggregate metrics.

    Profile D uses the same sagittal controller as C but adds the differential
    wheel yaw stabilizer (--enable-wheel-yaw-stabilizer).
    """
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case_id, height_label, steps, push_mag, push_dur, push_int in runner.PUSH_CASES:
        for profile, tag in [
            (PROFILE_A, "A"),
            (PROFILE_B, "B"),
            (PROFILE_C, "C"),
            (PROFILE_D, "D"),
        ]:
            out_dir = OUT_BASE / f"step_d_{case_id}_{tag}"
            t0 = time.time()
            # Profile D uses wheel yaw stabilizer
            use_wheel_yaw = (tag == "D")
            tel_path, _ = runner.run_sim(
                height_label,
                steps,
                profile,
                out_dir,
                push_magnitude=push_mag,
                push_duration=push_dur,
                push_interval=push_int,
                enable_wheel_yaw=use_wheel_yaw,
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
                **(metrics or {}),
            }
            all_rows.append(row)
            elapsed = time.time() - t0
            safe, reason = runner.safety_ok(metrics)
            print(
                f"[{case_id}] [{tag}] {runner.fmt(metrics) if metrics else 'MISSING'} "
                f"safe={safe}({reason}) {elapsed:.0f}s",
                flush=True,
            )
    # Write combined CSV
    csv_path = OUT_BASE / "step_d_all_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Combined metrics CSV written to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
