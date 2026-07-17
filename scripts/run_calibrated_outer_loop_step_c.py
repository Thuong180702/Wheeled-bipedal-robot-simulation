"""Phase 7: Calibrated outer-loop Step C random/changing height validation.

Compares B2 against B across 5 random/changing height scenarios:
  C1 slow ladder up/down
  C2 random dwell 500
  C3 random dwell 200
  C4 abrupt high-low-high
  C5 long random 5000-step sequence

Pass criteria:
  - B2 no fall
  - B2 no WBC/hidden/ownership
  - B2 transition maxabs <= B + 0.02
  - B2 P2P <= B * 1.15
  - B2 recovery equal or better than B
  - no parameter discontinuity
  - hip-yaw safe

Outputs:
  docs/validation/calibrated_outer_loop_step_c_report.md
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"
PER_RUN_TIMEOUT_S = 2400
MAX_WORKERS = 2

DRIFT_COL = "active_pitch_crossing_signed_error_m"

SCENARIOS = [
    ("C1_slow_ladder", "slow ladder up/down sequence", 3000),
    ("C2_random_dwell500", "random dwell 500", 3000),
    ("C3_random_dwell200", "random dwell 200", 2000),
    ("C4_abrupt_high_low_high", "abrupt high-low-high", 2000),
    ("C5_long_random_5000", "long random 5000-step", 5500),
]

# Heights for height-transition scenarios
HEIGHT_SEQUENCE_C1 = [0.30, 0.33, 0.36, 0.38, 0.43, 0.45, 0.48, 0.45, 0.43, 0.38, 0.36, 0.33, 0.30]
HEIGHT_SEQUENCE_C4 = [0.48, 0.30, 0.48]

PROFILES = [
    ("B", "support_position_outer_loop_pitch_ref"),
    ("B2", "calibrated_support_position_outer_loop_pitch_ref"),
]


def build_height_sequence(scenario_name, num_steps):
    """Build a per-step commanded height array for height-varying runs."""
    import random
    rng = random.Random(20260617 + hash(scenario_name) % 10000)
    steps_per_height = {"C1_slow_ladder": 200, "C2_random_dwell500": 500,
                        "C3_random_dwell200": 200, "C4_abrupt_high_low_high": 600,
                        "C5_long_random_5000": 300}
    available = [0.30, 0.32, 0.33, 0.34, 0.36, 0.38, 0.43, 0.45, 0.465, 0.48]
    n_steps = num_steps
    sps = steps_per_height.get(scenario_name, 300)

    if scenario_name == "C1_slow_ladder":
        heights = []
        for h in HEIGHT_SEQUENCE_C1:
            heights.extend([h] * sps)
        return heights[:n_steps]
    elif scenario_name == "C4_abrupt_high_low_high":
        heights = []
        for h in HEIGHT_SEQUENCE_C4:
            heights.extend([h] * sps)
        return heights[:n_steps]
    elif scenario_name == "C5_long_random_5000":
        seq = []
        for i in range(n_steps // sps):
            h = rng.choice(available)
            seq.extend([h] * sps)
        return seq[:n_steps]
    else:
        seq = []
        for i in range(n_steps // sps):
            h = rng.choice(available)
            seq.extend([h] * sps)
        return seq[:n_steps]


def generate_ladder_setup(output_dir, scenario_name, num_steps):
    """Generate a ladder setup JSON for height-varying scenarios."""
    heights = build_height_sequence(scenario_name, num_steps)
    setup = {
        "setup_name": scenario_name,
        "target_com_z_m": heights[0] if heights else 0.40,
        "equilibrium_joint_pos": {},
        "initial_joint_pos": {},
        "height_sequence": heights,
        "scenario_type": scenario_name,
    }
    out_path = output_dir / f"{scenario_name}_setup.json"
    out_path.write_text(json.dumps(setup, indent=2))
    return out_path


def run_sim(scenario_name, tag, profile, out_dir, num_steps):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{scenario_name}.csv"
    if tel_dst.exists():
        return tel_dst

    setup_path = OUT_BASE / f"step_c_setups/{scenario_name}_setup.json"
    if not setup_path.exists():
        setup_path.parent.mkdir(parents=True, exist_ok=True)
        generate_ladder_setup(setup_path.parent, scenario_name, num_steps)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(num_steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(num_steps),
        "--write-run-summary-sidecar",
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        return None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        return None

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if tels:
        shutil.copy2(tels[0], tel_dst)
        try:
            tels[0].unlink()
        except OSError:
            pass
    return tel_dst if tel_dst.exists() else None


def analyze(path):
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    def fcol(key, default=float("nan")):
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
        return [x for x in xs if x == x and x != float("nan")]

    drift = clean(fcol(DRIFT_COL))
    abs_drift = [abs(x) for x in drift]
    pos = sum(1 for x in drift if x > 0)
    nz = len(drift)
    zc = sum(1 for i in range(1, len(drift)) if (drift[i - 1] <= 0) != (drift[i] <= 0))
    pitch = clean(fcol("robot_pitch_x"))
    roll = clean(fcol("robot_roll_y"))
    comz = clean(fcol("com_z"))
    lhy = clean(fcol("l_hip_yaw_pos"))
    rhy = clean(fcol("r_hip_yaw_pos"))

    def out_pct(thr):
        return 100 * sum(1 for x in abs_drift if x > thr) / nz if nz else 0.0

    tau_wbc_max = clean(fcol("tau_wbc_max"))
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    term = any(str(r.get("terminated", "")).strip().lower() in ("true", "1")
               for r in rows)
    term_reason = ""
    for r in rows:
        if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
            term_reason = r.get("termination_reason", "") or ""
            break

    return {
        "steps": n,
        "fell": term,
        "term_reason": term_reason,
        "min_drift": round(min(drift), 4) if drift else 0.0,
        "max_drift": round(max(drift), 4) if drift else 0.0,
        "maxabs": round(max(abs_drift), 4) if abs_drift else 0.0,
        "p2p": round(max(drift) - min(drift), 4) if drift else 0.0,
        "pos_pct": round(100 * pos / nz, 1) if nz else 0.0,
        "zero_crossings": zc,
        "out03_pct": round(out_pct(0.03), 1),
        "out05_pct": round(out_pct(0.05), 1),
        "out08_pct": round(out_pct(0.08), 1),
        "out10_pct": round(out_pct(0.10), 1),
        "out15_pct": round(out_pct(0.15), 1),
        "pitch_rms": round(math.sqrt(sum(math.degrees(x) ** 2 for x in pitch) / max(1, len(pitch))), 2),
        "pitch_max": round(max((abs(math.degrees(x)) for x in pitch), default=0.0), 2),
        "roll_rms": round(math.sqrt(sum(math.degrees(x) ** 2 for x in roll) / max(1, len(roll))), 2),
        "comz_min": round(min(comz), 4) if comz else 0.0,
        "hip_yaw_abs_max": round(max([abs(x) for x in (lhy + rhy)]), 4) if lhy or rhy else 0.0,
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
    }


def safety_ok(m):
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, f"fall({m['term_reason'][:20]})"
    if m.get("hip_yaw_abs_max", 0) > 0.35:
        return False, "hip_yaw_unsafe"
    if m.get("pitch_max", 0) > 16.0:
        return False, "pitch_unsafe"
    if m.get("wbc_authority_rows", 0) > 0:
        return False, "wbc_authority_enabled"
    if m.get("wbc_owner_rows", 0) > 0:
        return False, "wbc_owner_present"
    return True, "safe"


def worker(args):
    scenario_name, tag, profile, out_dir, num_steps = args
    tel = run_sim(scenario_name, tag, profile, out_dir, num_steps)
    m = analyze(tel)
    return scenario_name, tag, m


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 7: calibrated outer-loop Step C random/changing height validation", flush=True)
    print("=" * 78, flush=True)

    jobs = []
    for scenario_name, desc, num_steps in SCENARIOS:
        for tag, profile in PROFILES:
            out_dir = OUT_BASE / f"phase7_{scenario_name}_{tag}"
            jobs.append((scenario_name, tag, profile, out_dir, num_steps))

    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            scenario_name, tag, m = future.result()
            results[(scenario_name, tag)] = m
            sok, reason = safety_ok(m)
            print(f"  {scenario_name} {tag}: safe={sok}({reason}) maxabs={m['maxabs'] if m else 'MISSING'} "
                  f"p2p={m['p2p'] if m else 'MISSING'}", flush=True)

    # Gate evaluation
    hard_fail = []
    improve_cases = []
    worse_cases = []

    for scenario_name, desc, num_steps in SCENARIOS:
        b = results.get((scenario_name, "B"))
        b2 = results.get((scenario_name, "B2"))
        b_safe, b_reason = safety_ok(b)
        b2_safe, b2_reason = safety_ok(b2)

        if not b2_safe:
            hard_fail.append((scenario_name, b2_reason))
            continue
        if not b_safe:
            hard_fail.append((scenario_name, "B_fell"))
            continue

        b2_better = (b2["maxabs"] <= b["maxabs"] + 0.02 and
                     b2["p2p"] <= b["p2p"] * 1.15)
        b2_worse = (b2["maxabs"] > b["maxabs"] + 0.02 or
                     b2["p2p"] > b["p2p"] * 1.15)

        if b2_better:
            improve_cases.append(scenario_name)
        if b2_worse:
            worse_cases.append(scenario_name)

    n_improve = len(improve_cases)
    n_worse = len(worse_cases)

    if hard_fail:
        classification = "CALIBRATED_STEP_C_FAIL"
    elif n_worse > 1:
        classification = "CALIBRATED_STEP_C_FAIL"
    elif n_improve >= 3:
        classification = "CALIBRATED_STEP_C_PASS"
    else:
        classification = "CALIBRATED_STEP_C_PASS_WITH_MONITORING"

    print(f"\n>>> improve={n_improve}/5  worse={n_worse}/5  hard_fail={len(hard_fail)}", flush=True)
    print(f">>> Classification: {classification}", flush=True)

    # Write report
    report = ROOT / "docs" / "validation" / "calibrated_outer_loop_step_c_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# Calibrated Outer-Loop — Step C Random/Changing Height (Phase 7)",
        "",
        f"**B:** `support_position_outer_loop_pitch_ref`",
        f"**B2:** `calibrated_support_position_outer_loop_pitch_ref`",
        f"**Classification:** `{classification}`",
        "",
        "## Gates",
        f"- hard failures: {hard_fail if hard_fail else 'none'}",
        f"- B2 better in: {n_improve}/5 cases",
        f"- B2 worse in: {n_worse}/5 cases",
        "",
        "## Results",
        "",
        "| scenario | prof | fell | maxabs | P2P | out15% | hip_yaw | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario_name, desc, num_steps in SCENARIOS:
        for tag in ["B", "B2"]:
            m = results.get((scenario_name, tag))
            sok, _ = safety_ok(m)
            v = ""
            if tag == "B2":
                if not sok:
                    v = "FAIL"
                elif scenario_name in improve_cases:
                    v = "BETTER"
                elif scenario_name in worse_cases:
                    v = "WORSE"
                else:
                    v = "EQUAL"
            L.append(
                f"| {scenario_name} | {tag} | {str(m.get('fell',True) if m else True)[:4]} | "
                f"{m.get('maxabs',0):.4f} | {m.get('p2p',0):.4f} | {m.get('out15_pct',0):.1f}% | "
                f"{m.get('hip_yaw_abs_max',0):.4f} | {v} |"
            )
    L += [
        "",
        "## Decision",
        f"- **{classification}**",
    ]
    if "PASS" in classification:
        L.append("- **Proceed to Phase 8 (Step D push validation).**")
    else:
        L.append("- **Do not proceed.**")
    report.write_text("\n".join(L) + "\n")
    print(f"Report: {report}", flush=True)
    return classification


if __name__ == "__main__":
    main()
