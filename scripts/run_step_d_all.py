import csv
import time
from pathlib import Path
import run_outer_loop_step_d_push as runner

# Low-band profile constant
LOW_BAND_PROFILE = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"

# Base output directory for all Step D runs
ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "step_d_all"


def main() -> None:
    """Run Step D push cases for profiles A, B, and C and aggregate metrics.

    Profiles:
        A – height_scheduled_pitch_equilibrium_trim (runner.BASE_PROFILE)
        B – support_position_outer_loop_pitch_ref (runner.OL_PROFILE)
        C – low‑band support v2 (LOW_BAND_PROFILE)
    """
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case_id, height_label, steps, push_mag, push_dur, push_int in runner.PUSH_CASES:
        for profile, tag in [
            (runner.BASE_PROFILE, "A"),
            (runner.OL_PROFILE, "B"),
            (LOW_BAND_PROFILE, "C"),
        ]:
            out_dir = OUT_BASE / f"step_d_{case_id}_{tag}"
            t0 = time.time()
            tel_path, _ = runner.run_sim(
                height_label,
                steps,
                profile,
                out_dir,
                push_magnitude=push_mag,
                push_duration=push_dur,
                push_interval=push_int,
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
