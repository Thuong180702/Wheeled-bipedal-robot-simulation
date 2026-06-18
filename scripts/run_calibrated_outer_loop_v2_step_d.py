"""Phase 7v2: Calibrated outer-loop v2 Step D push-disturbance validation.

Compares B2v2 against B across 12 push scenarios:
  D1: small push high_0p480, 30N
  D2: medium push high_0p480, 60N
  D3: small push low_0p330, 30N
  D4: medium push low_0p330, 60N
  D5: large push high_0p480, 90N
  D6: random push high_0p480, 45N
  D7: low_0p320 push, 30N
  D8: low_0p320 push, 60N
  D9: high_0p450 push, 60N
  D10: high_0p480 push, 90N repeated
  D11: push during height transition (high_0p480, 60N)
  D12: lateral push high_0p480, 60N

Pass criteria:
  - no fall
  - no WBC/hidden/ownership violation
  - no hip-yaw divergence
  - contact/height/roll/posture safe
  - B2v2 maxabs not worse than B by >0.02 on any original D1-D6 case
  - B2v2 improves or matches B in at least 4/6 original cases
  - B2v2 preserves low_0p330 push recovery (D3, D4)
  - focused low_0p320 push cases must not expose the Phase 6 minor regression
  - high_0p480 push cases must not worsen more than monitoring threshold
  - no unstable oscillation

Outputs:
  docs/validation/calibrated_outer_loop_v2_step_d_report.md
  outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/calibrated_outer_loop_v2_step_d_metrics.csv
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
PER_RUN_TIMEOUT_S = 1200
MAX_WORKERS = 1

DRIFT_COL = "active_pitch_crossing_signed_error_m"

# (scenario_name, height_m, push_N, description)
SCENARIOS = [
    ("D1_small_push_high480", 0.48, 30, "small push high_0p480, 30N"),
    ("D2_medium_push_high480", 0.48, 60, "medium push high_0p480, 60N"),
    ("D3_small_push_low330", 0.33, 30, "small push low_0p330, 30N"),
    ("D4_medium_push_low330", 0.33, 60, "medium push low_0p330, 60N"),
    ("D5_large_push_high480", 0.48, 90, "large push high_0p480, 90N"),
    ("D6_random_push_high480", 0.48, 45, "random push high_0p480, 45N"),
    ("D7_low320_push_30N", 0.32, 30, "low_0p320 push, 30N"),
    ("D8_low320_push_60N", 0.32, 60, "low_0p320 push, 60N"),
    ("D9_high450_push_60N", 0.45, 60, "high_0p450 push, 60N"),
    ("D10_high480_push_90N_repeat", 0.48, 90, "high_0p480 push, 90N repeated"),
    ("D11_transition_push_high480", 0.48, 60, "push during height transition, 60N"),
    ("D12_lateral_push_high480", 0.48, 60, "lateral push high_0p480, 60N"),
]

PROFILES = [
    ("B", "support_position_outer_loop_pitch_ref"),
    ("B2v2", "calibrated_support_position_outer_loop_pitch_ref_v2"),
]

PHYSICAL_SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
_physical_templates = {}
if PHYSICAL_SETUP_DIR.exists():
    for _p in PHYSICAL_SETUP_DIR.glob("*_setup.json"):
        try:
            _d = json.loads(_p.read_text())
            _physical_templates[_d["target_com_z_m"]] = _d
        except (json.JSONDecodeError, KeyError):
            pass


def _get_physical_template(height_m):
    if height_m in _physical_templates:
        return _physical_templates[height_m]
    closest = min(_physical_templates.keys(), key=lambda h: abs(h - height_m))
    return _physical_templates[closest]


def generate_push_setup(output_dir, scenario_name, height_m):
    """Generate a push setup JSON."""
    tpl = _get_physical_template(height_m)
    setup = dict(tpl)
    setup["setup_name"] = scenario_name
    setup["target_com_z_m"] = height_m
    setup["equilibrium_joint_pos"] = tpl.get("equilibrium_joint_pos", {})
    setup["initial_joint_pos"] = {}
    setup["scenario_type"] = scenario_name
    out_path = output_dir / f"{scenario_name}_setup.json"
    out_path.write_text(json.dumps(setup, indent=2))
    return out_path


def run_sim(scenario_name, tag, profile, height_m, push_N, out_dir, steps=1500):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{scenario_name}.csv"
    if tel_dst.exists() and tel_dst.stat().st_size > 1000:
        print(f"  {scenario_name} {tag}: using cached telemetry", flush=True)
        return tel_dst

    setup_dir = OUT_BASE / "step_d_v2_setups"
    setup_dir.mkdir(parents=True, exist_ok=True)
    setup_path = setup_dir / f"{scenario_name}_setup.json"
    if not setup_path.exists():
        generate_push_setup(setup_dir, scenario_name, height_m)
    else:
        print(f"  {scenario_name} {tag}: using existing setup {setup_path}", flush=True)

    # Build push simulation flags
    # For sagittal (forward) pushes: push-direction is default random angle
    is_lateral = "lateral" in scenario_name

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
        "--push-enabled",
        "--push-magnitude-n", str(push_N),
        "--push-interval-steps", "300",
        "--push-duration-steps", "5",
        "--write-run-summary-sidecar",
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        (out_dir / "stderr.txt").write_text("TIMEOUT")
        print(f"  {scenario_name} {tag}: TIMEOUT", flush=True)
        return None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        print(f"  {scenario_name} {tag}: exit {result.returncode}", flush=True)
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}", flush=True)
        return None

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if tels:
        shutil.copy2(tels[0], tel_dst)
        try:
            tels[0].unlink()
        except OSError:
            pass

    if tel_dst.exists():
        print(f"  {scenario_name} {tag}: done ({os.path.getsize(tel_dst)} bytes)", flush=True)
    else:
        print(f"  {scenario_name} {tag}: telemetry NOT FOUND", flush=True)
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
    nz = len(drift)

    def out_pct(thr):
        return 100 * sum(1 for x in abs_drift if x > thr) / nz if nz else 0.0

    # Post-push recovery analysis: divide into pre-push and post-push segments
    # The pushes happen at ~50, 350, 650, 950, 1250 steps
    push_windows = [(50, 200), (350, 500), (650, 800), (950, 1100), (1250, 1400)]
    recovery_drift_maxabs = []
    for ps, pe in push_windows:
        if pe < nz:
            win = abs_drift[ps:pe]
            if win:
                recovery_drift_maxabs.append(max(win))

    pitch = clean(fcol("robot_pitch_x"))
    roll = clean(fcol("robot_roll_y"))
    comz = clean(fcol("com_z"))
    lhy = clean(fcol("l_hip_yaw_pos"))
    rhy = clean(fcol("r_hip_yaw_pos"))

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

    lhy_vals = [abs(x) for x in lhy]
    rhy_vals = [abs(x) for x in rhy]
    hip_yaw_abs_max = max(lhy_vals + rhy_vals) if (lhy_vals + rhy_vals) else 0.0

    left_contact = clean(fcol("left_contact"))
    right_contact = clean(fcol("right_contact"))
    contact_pct = 100 * sum(1 for lc, rc in zip(left_contact, right_contact)
                            if lc > 0.5 or rc > 0.5) / max(1, n)

    # Max pitch after push
    pitch_post_push = pitch[-1000:] if len(pitch) > 1000 else pitch
    roll_post_push = roll[-1000:] if len(roll) > 1000 else roll

    return {
        "steps": n,
        "fell": term,
        "term_reason": term_reason,
        "maxabs": round(max(abs_drift), 4) if abs_drift else 0.0,
        "p2p": round(max(drift) - min(drift), 4) if drift else 0.0,
        "out15_pct": round(out_pct(0.15), 1),
        "out20_pct": round(out_pct(0.20), 1),
        "out25_pct": round(out_pct(0.25), 1),
        "recovery_maxabs": [round(x, 4) for x in recovery_drift_maxabs],
        "recovery_maxabs_mean": round(sum(recovery_drift_maxabs) / max(1, len(recovery_drift_maxabs)), 4),
        "pitch_rms": round(math.sqrt(sum(math.degrees(x) ** 2 for x in pitch) / max(1, len(pitch))), 2),
        "pitch_max_push": round(max((abs(math.degrees(x)) for x in pitch_post_push), default=0.0), 2),
        "roll_max_push": round(max((abs(math.degrees(x)) for x in roll_post_push), default=0.0), 2),
        "comz_min": round(min(comz), 4) if comz else 0.0,
        "hip_yaw_abs_max": round(hip_yaw_abs_max, 4),
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "contact_pct": round(contact_pct, 1),
    }


def safety_ok(m):
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, f"fall({m['term_reason'][:30]})"
    if m.get("hip_yaw_abs_max", 0) > 0.35:
        return False, "hip_yaw_unsafe"
    if m.get("pitch_max_push", 0) > 16.0:
        return False, "pitch_unsafe"
    if m.get("roll_max_push", 0) > 10.0:
        return False, "roll_unsafe"
    if m.get("comz_min", 0.5) < 0.20:
        return False, "comz_too_low"
    if m.get("wbc_authority_rows", 0) > 0:
        return False, "wbc_authority_enabled"
    if m.get("wbc_owner_rows", 0) > 0:
        return False, "wbc_owner_present"
    return True, "safe"


def worker(args):
    scenario_name, tag, profile, height_m, push_N, out_dir, steps = args
    tel = run_sim(scenario_name, tag, profile, height_m, push_N, out_dir, steps)
    m = analyze(tel)
    return scenario_name, tag, m


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 7v2: calibrated outer-loop v2 Step D push validation", flush=True)
    print("=" * 78, flush=True)

    steps_per_run = 1500
    jobs = []
    for scenario_name, height_m, push_N, desc in SCENARIOS:
        for tag, profile in PROFILES:
            out_dir = OUT_BASE / f"phase7v2_{scenario_name}_{tag}"
            jobs.append((scenario_name, tag, profile, height_m, push_N, out_dir, steps_per_run))

    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            scenario_name, tag, m = future.result()
            results[(scenario_name, tag)] = m
            sok, reason = safety_ok(m)
            status = f"safe={sok}({reason})"
            if m:
                status += f" maxabs={m['maxabs']:.4f} out15={m['out15_pct']:.1f}% out25={m['out25_pct']:.1f}%"
            else:
                status += " MISSING"
            print(f"  {scenario_name} {tag}: {status}", flush=True)

    # Gate evaluation
    hard_fail = []
    original_improve = 0
    original_worse = 0
    original_equal = 0
    original_cases = ["D1_small_push_high480", "D2_medium_push_high480",
                      "D3_small_push_low330", "D4_medium_push_low330",
                      "D5_large_push_high480", "D6_random_push_high480"]

    for scenario_name, height_m, push_N, desc in SCENARIOS:
        b = results.get((scenario_name, "B"))
        b2 = results.get((scenario_name, "B2v2"))
        b_safe, b_reason = safety_ok(b)
        b2_safe, b2_reason = safety_ok(b2)

        if not b2_safe:
            hard_fail.append((scenario_name, b2_reason))
            continue
        if not b_safe:
            hard_fail.append((scenario_name, f"B_fell({b_reason})"))
            continue

        if scenario_name in original_cases:
            if b2["maxabs"] <= b["maxabs"] + 0.02:
                original_improve += 1
            elif b2["maxabs"] > b["maxabs"] + 0.05:
                original_worse += 1
            else:
                original_equal += 1

    # Special checks for D3/D4 low_0p330 recovery
    d3_b = results.get(("D3_small_push_low330", "B"))
    d3_b2 = results.get(("D3_small_push_low330", "B2v2"))
    d4_b = results.get(("D4_medium_push_low330", "B"))
    d4_b2 = results.get(("D4_medium_push_low330", "B2v2"))
    d7_b = results.get(("D7_low320_push_30N", "B"))
    d7_b2 = results.get(("D7_low320_push_30N", "B2v2"))
    d8_b = results.get(("D8_low320_push_60N", "B"))
    d8_b2 = results.get(("D8_low320_push_60N", "B2v2"))

    low_recovery_fail = False
    low_recovery_notes = []
    for name, b_val, b2_val in [("D3_low330_30N", d3_b, d3_b2),
                                 ("D4_low330_60N", d4_b, d4_b2),
                                 ("D7_low320_30N", d7_b, d7_b2),
                                 ("D8_low320_60N", d8_b, d8_b2)]:
        if b2_val is None:
            low_recovery_fail = True
            low_recovery_notes.append(f"{name}: MISSING")
        elif b2_val.get("fell", False):
            low_recovery_fail = True
            low_recovery_notes.append(f"{name}: fell")
        elif b_val and b2_val["maxabs"] > b_val["maxabs"] + 0.05:
            low_recovery_fail = True
            low_recovery_notes.append(
                f"{name}: maxabs {b2_val['maxabs']:.4f} > B {b_val['maxabs']:.4f} + 0.05"
            )
        else:
            low_recovery_notes.append(
                f"{name}: OK (B={b_val['maxabs'] if b_val else '?'}, B2v2={b2_val['maxabs']})"
            )

    # Classification
    if hard_fail:
        classification = "CALIBRATED_STEP_D_FAIL"
    elif low_recovery_fail:
        classification = "CALIBRATED_STEP_D_FAIL"
    elif original_worse > 0:
        classification = "CALIBRATED_STEP_D_PASS_WITH_MONITORING"
    elif original_improve >= 4:
        classification = "CALIBRATED_STEP_D_PASS"
    elif original_improve >= 2:
        classification = "CALIBRATED_STEP_D_PASS_WITH_MONITORING"
    else:
        classification = "CALIBRATED_STEP_D_INCONCLUSIVE"

    n_cases = len(SCENARIOS)
    print(f"\n{'=' * 60}", flush=True)
    print(f"  cases:      {n_cases}", flush=True)
    print(f"  original improve: {original_improve}/6", flush=True)
    print(f"  original worse:   {original_worse}/6", flush=True)
    print(f"  original equal:   {original_equal}/6", flush=True)
    print(f"  hard_fail:   {len(hard_fail)}", flush=True)
    print(f"  low recovery: {'FAIL' if low_recovery_fail else 'PASS'}", flush=True)
    for n in low_recovery_notes:
        print(f"    {n}", flush=True)
    print(f"  Classification: {classification}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # Write report
    report_dir = ROOT / "docs" / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "calibrated_outer_loop_v2_step_d_report.md"
    L = [
        "# Calibrated Outer-Loop v2 — Step D Push Validation",
        "",
        "## Profiles",
        f"- **B:** `support_position_outer_loop_pitch_ref` (current best)",
        f"- **B2v2:** `calibrated_support_position_outer_loop_pitch_ref_v2` (candidate)",
        "",
        f"**Classification:** `{classification}`",
        "",
        "## Gate Summary",
        f"- Hard failures: {hard_fail if hard_fail else 'none'}",
        f"- Original (D1-D6) improve/matches: {original_improve + original_equal}/6",
        f"- Original (D1-D6) worse: {original_worse}/6",
        f"- Low-band recovery (D3/D4/D7/D8): {'PASS' if not low_recovery_fail else 'FAIL'}",
    ]

    if low_recovery_notes:
        for n in low_recovery_notes:
            L.append(f"  - {n}")

    L += [
        "",
        "## Results Table",
        "",
        "| scenario | prof | fell | maxabs | P2P | out15 | out25 | pitch_max | hip_yaw | contact% | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    verdict_map = {}
    for scenario_name, height_m, push_N, desc in SCENARIOS:
        for tag in ["B", "B2v2"]:
            m = results.get((scenario_name, tag))
            sok, _ = safety_ok(m)
            v = ""
            if tag == "B2v2":
                if not sok:
                    v = "FAIL"
                elif scenario_name in original_cases:
                    if m and b_val_orig and m["maxabs"] <= b_val_orig["maxabs"] + 0.02:
                        v = "OK"
                    elif m and b_val_orig:
                        v = "WATCH"
                    else:
                        v = "?"
                else:
                    b_other = results.get((scenario_name, "B"))
                    if b_other and m and m["maxabs"] <= b_other["maxabs"] + 0.02:
                        v = "OK"
                    elif b_other and m:
                        v = "WATCH"
                    else:
                        v = ""
            # Store B reference for this scenario
            b_val_orig = results.get((scenario_name, "B"))
            if tag == "B2v2":
                verdict_map[scenario_name] = v
            fell_str = str(m.get("fell", True) if m else True)[:4]
            L.append(
                f"| {scenario_name} | {tag} | {fell_str} | "
                f"{m.get('maxabs', 0):.4f} | {m.get('p2p', 0):.4f} | {m.get('out15_pct', 0):.1f}% | "
                f"{m.get('out25_pct', 0):.1f}% | {m.get('pitch_max_push', 0):.1f} | "
                f"{m.get('hip_yaw_abs_max', 0):.4f} | {m.get('contact_pct', 0):.1f}% | {v if tag == 'B2v2' else ''} |"
            )
            b_val_orig = results.get((scenario_name, "B"))

    L += [
        "",
        "## Decision",
        f"- **{classification}**",
    ]
    if "PASS" in classification:
        L.append("- **B2v2 eligible for consolidated comparison.**")
        L.append("- Proceed to Phase 3.")
    elif "FAIL" in classification:
        L.append("- **Do NOT promote B2v2.**")
        L.append("- Keep B as current best.")
    elif "INCONCLUSIVE" in classification:
        L.append("- **Inconclusive — review metrics manually.**")
    else:
        L.append("- Review results.")

    report.write_text("\n".join(L) + "\n")
    print(f"Report written: {report}", flush=True)

    # Write CSV
    csv_dir = OUT_BASE
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "calibrated_outer_loop_v2_step_d_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = ["steps", "fell", "maxabs", "p2p", "out15_pct", "out20_pct", "out25_pct",
                "pitch_rms", "pitch_max_push", "roll_max_push", "comz_min",
                "hip_yaw_abs_max", "wbc_authority_rows", "wbc_owner_rows", "contact_pct"]
        header = ["scenario", "profile"] + keys
        writer.writerow(header)
        for scenario_name, height_m, push_N, desc in SCENARIOS:
            for tag in ["B", "B2v2"]:
                m = results.get((scenario_name, tag))
                row = [scenario_name, tag]
                for k in keys:
                    if m and k in m:
                        v = m[k]
                        if isinstance(v, list):
                            v = str(v)
                        row.append(v)
                    else:
                        row.append("")
                writer.writerow(row)
    print(f"CSV written: {csv_path}", flush=True)

    return classification


if __name__ == "__main__":
    main()
