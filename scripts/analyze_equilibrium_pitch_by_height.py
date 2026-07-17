"""Extract per-height steady-state equilibrium pitch + drift metrics from the
existing offset-0 adaptive_support_centering_trim height ladder.

Root cause: at each height the robot settles at a height-dependent equilibrium
pitch while pitch_ref is pinned to 0. The mean steady-state robot_pitch_x is
therefore the pitch_ref_offset that height needs to center support drift.

This seeds the height schedule from data instead of a blind 90-run grid.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"

LABELS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340",
    "low_0p360", "low_0p380",
    "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]

DRIFT_COL = "active_pitch_crossing_signed_error_m"
PITCH_COL = "robot_pitch_x"
ROLL_COL = "robot_roll_y"
COMZ_COL = "com_z"


def fcol(rows, key, default=float("nan")):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            out.append(default)
        else:
            try:
                out.append(float(v))
            except ValueError:
                out.append(default)
    return out


def clean(xs):
    return [x for x in xs if x == x]  # drop nan


def analyze(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None
    # steady state = second half
    half = n // 2

    drift = fcol(rows, DRIFT_COL)
    pitch = fcol(rows, PITCH_COL)
    roll = fcol(rows, ROLL_COL)
    comz = fcol(rows, COMZ_COL)

    drift_c = clean(drift)
    pitch_ss = clean(pitch[half:])
    pitch_all = clean(pitch)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def rms(xs):
        return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")

    abs_drift = [abs(x) for x in drift_c]
    nz = len(drift_c)
    pos = sum(1 for x in drift_c if x > 0)
    neg = sum(1 for x in drift_c if x < 0)

    # zero crossings
    zc = 0
    for i in range(1, len(drift_c)):
        if (drift_c[i - 1] <= 0) != (drift_c[i] <= 0):
            zc += 1

    pos_area = sum(x for x in drift_c if x > 0)
    neg_area = -sum(x for x in drift_c if x < 0)

    comz_c = clean(comz)
    roll_c = clean(roll)

    return {
        "steps": n,
        # equilibrium pitch (the candidate offset)
        "pitch_ss_mean_deg": round(math.degrees(mean(pitch_ss)), 2),
        "pitch_all_mean_deg": round(math.degrees(mean(pitch_all)), 2),
        "pitch_min_deg": round(math.degrees(min(pitch_all)), 2) if pitch_all else None,
        "pitch_max_deg": round(math.degrees(max(pitch_all)), 2) if pitch_all else None,
        "pitch_rms_deg": round(math.degrees(rms(pitch_all)), 2),
        # offset-0 drift baseline
        "drift_min": round(min(drift_c), 4) if drift_c else None,
        "drift_max": round(max(drift_c), 4) if drift_c else None,
        "drift_maxabs": round(max(abs_drift), 4) if abs_drift else None,
        "drift_p2p": round(max(drift_c) - min(drift_c), 4) if drift_c else None,
        "pos_pct": round(100 * pos / nz, 1) if nz else None,
        "neg_pct": round(100 * neg / nz, 1) if nz else None,
        "zero_crossings": zc,
        "pos_area": round(pos_area, 3),
        "neg_area": round(neg_area, 3),
        "out15_pct": round(100 * sum(1 for x in abs_drift if x > 0.15) / nz, 1) if nz else None,
        "comz_min": round(min(comz_c), 4) if comz_c else None,
        "comz_max": round(max(comz_c), 4) if comz_c else None,
        "roll_rms_deg": round(math.degrees(rms(roll_c)), 2),
    }


def main():
    results = {}
    print(f"{'label':<12} {'pitch_ss°':>9} {'pitchRMS°':>9} "
          f"{'dMin':>8} {'dMax':>8} {'dMaxAbs':>8} {'P2P':>7} "
          f"{'pos%':>6} {'neg%':>6} {'ZC':>4} {'rollRMS°':>8}")
    print("-" * 100)
    for label in LABELS:
        path = LADDER / f"adaptive_height_ladder_2000_{label}" / "telemetry_2000.csv"
        if not path.exists():
            print(f"{label:<12}  MISSING")
            continue
        m = analyze(path)
        results[label] = m
        print(f"{label:<12} {m['pitch_ss_mean_deg']:>9} {m['pitch_rms_deg']:>9} "
              f"{m['drift_min']:>8} {m['drift_max']:>8} {m['drift_maxabs']:>8} "
              f"{m['drift_p2p']:>7} {m['pos_pct']:>6} {m['neg_pct']:>6} "
              f"{m['zero_crossings']:>4} {m['roll_rms_deg']:>8}")

    out = LADDER / "equilibrium_pitch_by_height.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {out}")

    # Print the suggested schedule: offset ~ steady-state equilibrium pitch
    print("\n=== Candidate schedule (offset = steady-state equilibrium pitch) ===")
    heights = {
        "low_0p300": 0.300, "low_0p320": 0.320, "low_0p330": 0.330,
        "low_0p340": 0.340, "low_0p360": 0.360, "low_0p380": 0.380,
        "high_0p430": 0.430, "high_0p450": 0.450, "high_0p465": 0.465,
        "high_0p480": 0.480,
    }
    for label in LABELS:
        if label in results:
            print(f"  {heights[label]:.3f} m  ->  {results[label]['pitch_ss_mean_deg']:+.2f} deg "
                  f"(offset-0 pos%={results[label]['pos_pct']})")


if __name__ == "__main__":
    main()
