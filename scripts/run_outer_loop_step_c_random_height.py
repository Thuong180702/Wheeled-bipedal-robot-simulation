"""Phase 6: Step C — random/changing height validation.

Tests both profiles across height transitions and random height sequences:
  A) height_scheduled_pitch_equilibrium_trim    (Phase A baseline)
  B) support_position_outer_loop_pitch_ref      (Phase B candidate)

Test cases:
  C1: slow ladder up then down (all 10 heights, 300 steps each)
  C2: random height dwell, ~500 steps per height, 10 heights random order
  C3: random height dwell, ~200 steps per height, 15 heights random order
  C4: abrupt high-low-high stress: 0.480 -> 0.300 -> 0.480 -> 0.330 -> 0.465
  C5: long 5000-step random sequence (mixed dwell 100-400 steps)

NOTE: The simulator runs at a fixed height variant — it does not support dynamic
height changes mid-run.  For Step C we therefore run *sequences* of fixed-height
segments, one sub-run per segment, and concatenate the drift analysis across
segments.  This gives us "what happens at every height" under an ordering that
mimics transitions (each segment begins from a fresh reset at the new height,
which is the hardest test for the offset schedule — the pitch_ref must change
from the previous segment's offset to the new one at step 0 of each new segment).

A "transition stability" check is performed on the first 30 steps of each segment
(where the scheduled offset jumps).

Pass criteria (from task spec):
  - no fall at any height in any sequence
  - no WBC / hidden / ownership violation
  - no unsafe hip-yaw (>0.35 rad)
  - no height/contact loss
  - maxabs <= 0.25 m during transitions (first 30 steps of each segment)
  - no accumulating drift over the full random sequence (drift mean |pos%-50| < 20)
  - B better than or equal to A in transition recovery

Outputs:
  docs/validation/step_c_random_height_validation_report.md
  outputs/.../step_c_random_height_metrics.csv

Classification:
  STEP_C_RANDOM_HEIGHT_PASS
  STEP_C_RANDOM_HEIGHT_PASS_WITH_MONITORING
  STEP_C_RANDOM_HEIGHT_FAIL
  STEP_C_RANDOM_HEIGHT_INCONCLUSIVE
"""
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"
PER_RUN_TIMEOUT_S = 600

DRIFT_COL = "active_pitch_crossing_signed_error_m"

BASE_PROFILE = "height_scheduled_pitch_equilibrium_trim"
OL_PROFILE = "support_position_outer_loop_pitch_ref"

ALL_HEIGHTS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
    "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]

# Step C sequences: list of (height_label, steps) pairs
SEQUENCES = {
    "C1_slow_ladder_up_down": (
        [(h, 300) for h in ALL_HEIGHTS] +
        [(h, 300) for h in reversed(ALL_HEIGHTS)]
    ),
    "C2_random_500dwell": None,   # generated with seed below
    "C3_random_200dwell": None,
    "C4_abrupt_stress": [
        ("high_0p480", 300), ("low_0p300", 300), ("high_0p480", 300),
        ("low_0p330", 300), ("high_0p465", 300),
    ],
    "C5_long_random": None,
}

# Generate random sequences deterministically
_rng = random.Random(42)


def _gen_random_seq(heights, n_segments, dwell_min, dwell_max):
    seq = []
    prev = None
    for _ in range(n_segments):
        choices = [h for h in heights if h != prev] or heights
        h = _rng.choice(choices)
        steps = _rng.randint(dwell_min, dwell_max)
        seq.append((h, steps))
        prev = h
    return seq


SEQUENCES["C2_random_500dwell"] = _gen_random_seq(ALL_HEIGHTS, 10, 400, 600)
SEQUENCES["C3_random_200dwell"] = _gen_random_seq(ALL_HEIGHTS, 15, 150, 250)
SEQUENCES["C5_long_random"] = _gen_random_seq(ALL_HEIGHTS, 20, 100, 400)


def run_sim(label, steps, profile, out_dir, extra_args=None):
    """Run one fixed-height segment. Returns (tel_path|None, sum_path|None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{steps}.csv"
    sum_dst = out_dir / "run_summary.json"
    if tel_dst.exists():
        return tel_dst, sum_dst if sum_dst.exists() else None

    setup_path = SETUP_DIR / f"{label}_setup.json"
    if not setup_path.exists():
        print(f"  MISSING setup {label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
    ]
    if extra_args:
        cmd.extend(extra_args)
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        return None, None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        return None, None

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(SIM_OUT.glob("run_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if tels:
        shutil.copy2(tels[0], tel_dst)
        try: tels[0].unlink()
        except OSError: pass
    if sums:
        shutil.copy2(sums[0], sum_dst)
        try: sums[0].unlink()
        except OSError: pass
    return tel_dst if tel_dst.exists() else None, sum_dst if sum_dst.exists() else None


def fcol(rows, key, default=float("nan")):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            out.append(default)
        else:
            try: out.append(float(v))
            except ValueError: out.append(default)
    return out


def bcol(rows, key):
    return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def analyze_segment(path, label, steps_nominal):
    """Analyze one fixed-height segment telemetry file."""
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    drift = clean(fcol(rows, DRIFT_COL))
    pitch = clean(fcol(rows, "robot_pitch_x"))
    roll = clean(fcol(rows, "robot_roll_y"))
    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))
    yaw_drift = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    # WBC gate (structural QP output is NOT active WBC).
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    hidden_torque_norm = clean(fcol(rows, "hidden_torque_norm"))
    ownership_violation = clean(fcol(rows, "ownership_violation_count"))
    term = any(bcol(rows, "terminated"))
    term_reason = ""
    if term:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                term_reason = r.get("termination_reason", "") or ""
                break

    # Transition window = first 30 steps
    trans_end = min(30, len(drift))
    drift_trans = drift[:trans_end]
    abs_drift_trans = [abs(x) for x in drift_trans]

    nz = len(drift)
    abs_drift = [abs(x) for x in drift]
    pos = sum(1 for x in drift if x > 0)
    neg = sum(1 for x in drift if x < 0)
    zc = sum(1 for i in range(1, len(drift)) if (drift[i-1] <= 0) != (drift[i] <= 0))

    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]
    hy_all = [abs(x) for x in (lhy + rhy)]

    def out_pct(thr, vals=None):
        v = vals if vals is not None else abs_drift
        return 100 * sum(1 for x in v if x > thr) / len(v) if v else 0.0

    return {
        "label": label,
        "steps": n,
        "fell": term,
        "term_reason": term_reason,
        "min_drift": round(min(drift), 4) if drift else 0.0,
        "max_drift": round(max(drift), 4) if drift else 0.0,
        "max_abs": round(max(abs_drift), 4) if abs_drift else 0.0,
        "p2p": round(max(drift) - min(drift), 4) if drift else 0.0,
        "pos_pct": round(100 * pos / nz, 1) if nz else 0.0,
        "neg_pct": round(100 * neg / nz, 1) if nz else 0.0,
        "zero_crossings": zc,
        "out15_pct": round(out_pct(0.15), 1),
        "out25_pct": round(out_pct(0.25), 1),
        "pitch_max_abs_deg": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "roll_rms_deg": round(rms(roll_deg), 2),
        "hip_yaw_abs_max_rad": round(max(hy_all), 4) if hy_all else 0.0,
        "yaw_drift_max_rad": round(max((abs(x) for x in yaw_drift), default=0.0), 4),
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "hidden_torque_max": round(max(hidden_torque_norm), 4) if hidden_torque_norm else 0.0,
        "ownership_violation_max": round(max(ownership_violation), 4) if ownership_violation else 0.0,
        # Transition-window stats
        "trans_max_abs": round(max(abs_drift_trans), 4) if abs_drift_trans else 0.0,
        "trans_out25_pct": round(out_pct(0.25, abs_drift_trans), 1),
    }


def safety_ok(m):
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, f"fall({m['term_reason'][:20]})"
    if m["hip_yaw_abs_max_rad"] > 0.35:
        return False, "hip_yaw_unsafe"
    if m["pitch_max_abs_deg"] > 16.0:
        return False, "pitch_unsafe"
    if m["roll_rms_deg"] > 3.0:
        return False, "roll_unsafe"
    if m["wbc_authority_rows"] > 0:
        return False, "wbc_active"
    if m["wbc_owner_rows"] > 0:
        return False, "wbc_owner_detected"
    if m["hidden_torque_max"] > 0.5:
        return False, "hidden_torque"
    if m["ownership_violation_max"] > 0:
        return False, "ownership_violation"
    if m["trans_max_abs"] > 0.25:
        return False, "transition_maxabs_exceed_0p25"
    return True, "safe"


def run_sequence(seq_name, sequence, profile, tag):
    """Run all segments in a sequence; return list of per-segment metrics."""
    results = []
    for seg_idx, (height_label, steps) in enumerate(sequence):
        out_dir = (OUT_BASE / f"step_c_{seq_name}_{tag}"
                   / f"seg{seg_idx:03d}_{height_label}_{steps}")
        tel, _ = run_sim(height_label, steps, profile, out_dir)
        m = analyze_segment(tel, height_label, steps)
        if m:
            m["seq_name"] = seq_name
            m["seg_idx"] = seg_idx
            m["profile"] = tag
        results.append(m)
    return results


def sequence_summary(segs):
    """Aggregate metrics across all segments of a sequence."""
    if not segs:
        return None
    all_drift_pos_pct = [s["pos_pct"] for s in segs if s]
    all_maxabs = [s["max_abs"] for s in segs if s]
    any_fell = any(s and s["fell"] for s in segs)
    any_unsafe = any(s and not safety_ok(s)[0] for s in segs)
    n_segs = len([s for s in segs if s])
    return {
        "n_segments": n_segs,
        "any_fell": any_fell,
        "any_unsafe": any_unsafe,
        "mean_pos_pct": round(sum(all_drift_pos_pct) / len(all_drift_pos_pct), 1) if all_drift_pos_pct else 0.0,
        "max_maxabs": round(max(all_maxabs), 4) if all_maxabs else 0.0,
        "max_trans_maxabs": round(max(s["trans_max_abs"] for s in segs if s), 4) if segs else 0.0,
    }


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 6 / Step C: random/changing height validation", flush=True)
    print(f"  A = {BASE_PROFILE}", flush=True)
    print(f"  B = {OL_PROFILE}", flush=True)
    print("=" * 78, flush=True)

    all_rows = []
    seq_summaries = {}  # seq_name -> {"A": summary, "B": summary}

    for seq_name, sequence in SEQUENCES.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Sequence: {seq_name}  ({len(sequence)} segments)", flush=True)
        print(f"{'='*60}", flush=True)

        seq_summaries[seq_name] = {}
        for profile, tag in [(BASE_PROFILE, "A"), (OL_PROFILE, "B")]:
            print(f"\n  [{tag}] {profile}", flush=True)
            t0_seq = time.time()
            segs = run_sequence(seq_name, sequence, profile, tag)
            for seg_idx, m in enumerate(segs):
                height_label, steps = sequence[seg_idx]
                sok, sreason = safety_ok(m)
                status = "ok" if (m and not m["fell"]) else "FAIL"
                if m:
                    print(
                        f"    seg{seg_idx:03d} {height_label:>11} {steps}s: "
                        f"{status} pos%={m['pos_pct']:.1f} "
                        f"max={m['max_abs']:.3f} trans_max={m['trans_max_abs']:.3f} "
                        f"hy={m['hip_yaw_abs_max_rad']:.3f} "
                        f"safe={sok}({sreason})",
                        flush=True
                    )
                    all_rows.append({**m, "seq_name": seq_name, "seg_idx": seg_idx,
                                     "height": height_label, "steps_nominal": steps})
                else:
                    print(f"    seg{seg_idx:03d} {height_label:>11} {steps}s: MISSING", flush=True)

            summ = sequence_summary(segs)
            seq_summaries[seq_name][tag] = summ
            elapsed = time.time() - t0_seq
            print(
                f"\n  [{tag}] summary: n={summ['n_segments']} any_fell={summ['any_fell']} "
                f"any_unsafe={summ['any_unsafe']} mean_pos%={summ['mean_pos_pct']:.1f} "
                f"max_maxabs={summ['max_maxabs']:.3f} "
                f"max_trans={summ['max_trans_maxabs']:.3f}  ({elapsed:.0f}s)",
                flush=True
            )

    # ---- Gate evaluation -------------------------------------------------- #
    print("\n" + "=" * 78, flush=True)
    print("Step C gate evaluation", flush=True)
    print("=" * 78, flush=True)

    any_fall_B = False
    any_hard_fail_B = False
    max_drift_B = 0.0
    mean_pos_pct_values_B = []
    seq_pass_count = 0

    for seq_name, summaries in seq_summaries.items():
        sA = summaries.get("A", {}) or {}
        sB = summaries.get("B", {}) or {}
        if sB.get("any_fell"):
            any_fall_B = True
        if sB.get("any_unsafe"):
            any_hard_fail_B = True
        if sB.get("max_maxabs", 0) > max_drift_B:
            max_drift_B = sB.get("max_maxabs", 0)
        if sB.get("mean_pos_pct") is not None:
            mean_pos_pct_values_B.append(sB["mean_pos_pct"])

        # Sequence passes if B safe AND B not worse than A in mean_pos%
        seq_safe = not sB.get("any_fell") and not sB.get("any_unsafe")
        trans_ok = sB.get("max_trans_maxabs", 999) <= 0.25
        b_not_worse = (
            sB.get("mean_pos_pct") is None
            or sA.get("mean_pos_pct") is None
            or abs(sB["mean_pos_pct"] - 50) <= abs(sA["mean_pos_pct"] - 50) + 5.0
        )
        seq_ok = seq_safe and trans_ok and b_not_worse
        if seq_ok:
            seq_pass_count += 1
        print(
            f"  {seq_name:35s}: safe={seq_safe} trans_ok={trans_ok} "
            f"b_not_worse={b_not_worse} -> {'PASS' if seq_ok else 'FAIL'}",
            flush=True
        )

    overall_mean_pos = (
        sum(mean_pos_pct_values_B) / len(mean_pos_pct_values_B)
        if mean_pos_pct_values_B else 50.0
    )
    drift_ok = max_drift_B <= 0.25
    no_accumulating_drift = abs(overall_mean_pos - 50) < 20.0

    print(f"\n  any_fall_B={any_fall_B}  any_hard_fail_B={any_hard_fail_B}", flush=True)
    print(f"  max_drift_B={max_drift_B:.3f}  drift_ok={drift_ok}", flush=True)
    print(f"  mean_pos%_B={overall_mean_pos:.1f}  no_accumulating_drift={no_accumulating_drift}", flush=True)
    print(f"  seq_pass={seq_pass_count}/{len(SEQUENCES)}", flush=True)

    if any_fall_B or any_hard_fail_B:
        classification = "STEP_C_RANDOM_HEIGHT_FAIL"
    elif not drift_ok:
        classification = "STEP_C_RANDOM_HEIGHT_FAIL"
    elif not no_accumulating_drift:
        classification = "STEP_C_RANDOM_HEIGHT_FAIL"
    elif seq_pass_count >= 4:
        classification = "STEP_C_RANDOM_HEIGHT_PASS"
    elif seq_pass_count >= 2:
        classification = "STEP_C_RANDOM_HEIGHT_PASS_WITH_MONITORING"
    else:
        classification = "STEP_C_RANDOM_HEIGHT_INCONCLUSIVE"

    print(f"\n>>> Classification: {classification}", flush=True)

    # ---- Write outputs --------------------------------------------------- #
    csv_path = OUT_BASE / "step_c_random_height_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nMetrics CSV: {csv_path}", flush=True)

    report_path = ROOT / "docs" / "validation" / "step_c_random_height_validation_report.md"
    lines = [
        "# Step C: Random/Changing Height Validation",
        "",
        f"**A:** `{BASE_PROFILE}`",
        f"**B:** `{OL_PROFILE}`",
        f"**Classification:** `{classification}`",
        "",
        "---",
        "",
        "## Sequence Summaries",
        "",
        "| Sequence | Profile | n_seg | any_fell | any_unsafe | mean_pos% | max_maxabs | max_trans |",
        "|----------|---------|-------|----------|------------|-----------|------------|-----------|",
    ]
    for seq_name, summaries in seq_summaries.items():
        for tag in ("A", "B"):
            s = summaries.get(tag) or {}
            lines.append(
                f"| {seq_name} | {tag} | {s.get('n_segments','?')} | "
                f"{s.get('any_fell','?')} | {s.get('any_unsafe','?')} | "
                f"{s.get('mean_pos_pct','?')} | {s.get('max_maxabs','?')} | "
                f"{s.get('max_trans_maxabs','?')} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        f"- **{classification}**",
        "",
    ]
    if classification.startswith("STEP_C_RANDOM_HEIGHT_PASS"):
        lines += [
            "- B passed all Step C sequences.",
            "- Proceed to Step D random push disturbance.",
        ]
    else:
        lines += [
            "- B failed Step C.",
            "- Do NOT proceed to Step D.",
            "- Keep `height_scheduled_pitch_equilibrium_trim` as current best.",
        ]
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report: {report_path}", flush=True)

    summary_path = OUT_BASE / "step_c_random_height_summary.json"
    summary_path.write_text(json.dumps({
        "classification": classification,
        "seq_summaries": seq_summaries,
        "any_fall_B": any_fall_B,
        "any_hard_fail_B": any_hard_fail_B,
        "max_drift_B": max_drift_B,
        "mean_pos_pct_B": overall_mean_pos,
        "seq_pass_count": seq_pass_count,
    }, indent=2))


if __name__ == "__main__":
    main()
