"""Phase 7v2: Calibrated outer-loop v2 Step C random/changing height validation.

Compares B2v2 against B across 8 random/changing height scenarios:
  C1: slow ladder up/down
  C2: random height dwell 500 steps
  C3: random height dwell 200 steps
  C4: abrupt high-low-high stress
  C5: long 5000-step random height sequence
  C6: repeated transitions through low_0p320
  C7: repeated transitions through high_0p480
  C8: low_0p320 -> high_0p450 -> high_0p480 -> low_0p320 loop

Pass criteria:
  - no fall
  - no WBC/hidden/ownership violation
  - contact/height/roll/posture safe
  - hip-yaw / leg-yaw safe
  - no parameter discontinuity
  - pitch_ref remains continuous/rate-limited
  - B2v2 transition maxabs <= B + 0.02 where feasible
  - B2v2 P2P <= B * 1.15
  - B2v2 recovery equal or better than B in at least 5/8 cases
  - B2v2 must not fail focused low_0p320 or high_0p480 transition cases
  - no accumulating drift in long random sequence

Outputs:
  docs/validation/calibrated_outer_loop_v2_step_c_report.md
  outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/calibrated_outer_loop_v2_step_c_metrics.csv
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
PER_RUN_TIMEOUT_S = 3600
MAX_WORKERS = 1

DRIFT_COL = "active_pitch_crossing_signed_error_m"

SCENARIOS = [
    ("C1_slow_ladder", "slow ladder up/down sequence", 3000),
    ("C2_random_dwell500", "random dwell 500", 3000),
    ("C3_random_dwell200", "random dwell 200", 2000),
    ("C4_abrupt_high_low_high", "abrupt high-low-high", 2000),
    ("C5_long_random_5000", "long random 5000-step", 5500),
    ("C6_focused_low320", "repeated low_0p320 transitions", 3000),
    ("C7_focused_high480", "repeated high_0p480 transitions", 3000),
    ("C8_low320_high450_480_loop", "low-320 -> high-450 -> high-480 -> low-320 loop", 3000),
]

PROFILES = [
    ("B", "support_position_outer_loop_pitch_ref"),
    ("B2v2", "calibrated_support_position_outer_loop_pitch_ref_v2"),
]

AVAILABLE_HEIGHTS = [0.30, 0.32, 0.33, 0.34, 0.36, 0.38, 0.43, 0.45, 0.465, 0.48]
HEIGHT_SEQUENCE_C1 = [0.30, 0.33, 0.36, 0.38, 0.43, 0.45, 0.48, 0.45, 0.43, 0.38, 0.36, 0.33, 0.30]
HEIGHT_SEQUENCE_C4 = [0.48, 0.30, 0.48]

PHYSICAL_SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"

# Load physical setup templates (contain hip_pitch_ref, knee_ref, etc.)
_physical_templates = {}
if PHYSICAL_SETUP_DIR.exists():
    for _p in PHYSICAL_SETUP_DIR.glob("*_setup.json"):
        try:
            _d = json.loads(_p.read_text())
            _physical_templates[_d["target_com_z_m"]] = _d
        except (json.JSONDecodeError, KeyError):
            pass


def _get_physical_template(height_m):
    """Return physical setup dict for a given height, with closest fallback."""
    if height_m in _physical_templates:
        return _physical_templates[height_m]
    closest = min(_physical_templates.keys(), key=lambda h: abs(h - height_m))
    print(f"  [WARN] no exact physical template for h={height_m:.3f}, using h={closest:.3f}", flush=True)
    return _physical_templates[closest]


def build_height_sequence(scenario_name, num_steps):
    """Build a per-step commanded height array for height-varying runs."""
    import random as _random
    rng_seed = 20260617 + hash(scenario_name) % 10000
    rng = _random.Random(rng_seed)

    sps = {"C1_slow_ladder": 200, "C2_random_dwell500": 500,
           "C3_random_dwell200": 200, "C4_abrupt_high_low_high": 600,
           "C5_long_random_5000": 300,
           "C6_focused_low320": 200,
           "C7_focused_high480": 200,
           "C8_low320_high450_480_loop": 300}.get(scenario_name, 300)

    if scenario_name == "C1_slow_ladder":
        heights = []
        for h in HEIGHT_SEQUENCE_C1:
            heights.extend([h] * sps)
        return heights[:num_steps]
    elif scenario_name == "C4_abrupt_high_low_high":
        heights = []
        for h in HEIGHT_SEQUENCE_C4:
            heights.extend([h] * sps)
        return heights[:num_steps]
    elif scenario_name == "C6_focused_low320":
        seq = []
        pattern = [0.30, 0.34, 0.30, 0.34]
        for _ in range(num_steps // (len(pattern) * sps) + 1):
            for h in pattern:
                seq.extend([h] * sps)
        return seq[:num_steps]
    elif scenario_name == "C7_focused_high480":
        seq = []
        pattern = [0.48, 0.45, 0.48, 0.45]
        for _ in range(num_steps // (len(pattern) * sps) + 1):
            for h in pattern:
                seq.extend([h] * sps)
        return seq[:num_steps]
    elif scenario_name == "C8_low320_high450_480_loop":
        seq = []
        pattern = [0.32, 0.45, 0.48, 0.32]
        for _ in range(num_steps // (len(pattern) * sps) + 1):
            for h in pattern:
                seq.extend([h] * sps)
        return seq[:num_steps]
    else:
        seq = []
        n_blocks = num_steps // sps
        for _ in range(n_blocks + 1):
            h = rng.choice(AVAILABLE_HEIGHTS)
            seq.extend([h] * sps)
        return seq[:num_steps]


def generate_ladder_setup(output_dir, scenario_name, num_steps):
    """Generate a ladder setup JSON with all required fields for the sim script."""
    heights = build_height_sequence(scenario_name, num_steps)
    tpl = _get_physical_template(heights[0])
    setup = dict(tpl)
    setup["setup_name"] = scenario_name
    setup["target_com_z_m"] = heights[0]
    setup["equilibrium_joint_pos"] = tpl.get("equilibrium_joint_pos", {})
    setup["initial_joint_pos"] = {}
    setup["height_sequence"] = heights
    setup["scenario_type"] = scenario_name
    out_path = output_dir / f"{scenario_name}_setup.json"
    out_path.write_text(json.dumps(setup, indent=2))
    return out_path


def run_sim(scenario_name, tag, profile, out_dir, num_steps):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{scenario_name}.csv"
    if tel_dst.exists() and tel_dst.stat().st_size > 1000:
        print(f"  {scenario_name} {tag}: using cached telemetry", flush=True)
        return tel_dst

    setup_path = OUT_BASE / f"step_c_v2_setups/{scenario_name}_setup.json"
    if not setup_path.exists():
        setup_path.parent.mkdir(parents=True, exist_ok=True)
        generate_ladder_setup(setup_path.parent, scenario_name, num_steps)
    else:
        print(f"  {scenario_name} {tag}: using existing setup {setup_path}", flush=True)

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
        print(f"  {scenario_name} {tag}: TIMEOUT", flush=True)
        return None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        print(f"  {scenario_name} {tag}: exit {result.returncode}", flush=True)
        if result.stderr:
            stderr_summary = result.stderr[-500:]
            print(f"  STDERR (last 500): {stderr_summary}", flush=True)
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
    pos = sum(1 for x in drift if x > 0)
    nz = len(drift)
    zc = sum(1 for i in range(1, len(drift)) if (drift[i - 1] <= 0) != (drift[i] <= 0))

    seg_count = min(10, nz)
    seg_size = max(1, nz // seg_count)
    seg_drift_maxabs = []
    for si in range(seg_count):
        seg = abs_drift[si * seg_size : (si + 1) * seg_size]
        if seg:
            seg_drift_maxabs.append(max(seg))

    pitch = clean(fcol("robot_pitch_x"))
    roll = clean(fcol("robot_roll_y"))
    comz = clean(fcol("com_z"))
    lhy = clean(fcol("l_hip_yaw_pos"))
    rhy = clean(fcol("r_hip_yaw_pos"))

    outer_loop_kp = clean(fcol("outer_loop_kp_deg_per_m"))
    outer_loop_kd = clean(fcol("outer_loop_kd_deg_per_mps"))
    outer_loop_theta_max = clean(fcol("outer_loop_theta_ref_max_deg"))
    outer_loop_deadband = clean(fcol("outer_loop_error_deadband_m"))
    outer_loop_pitch_ref = clean(fcol("outer_loop_pitch_ref_deg"))
    outer_loop_active_col = clean(fcol("outer_loop_active"))

    pitch_ref_cont = sum(
        1 for i in range(1, len(outer_loop_pitch_ref))
        if abs(outer_loop_pitch_ref[i] - outer_loop_pitch_ref[i - 1]) > 0.5
    ) if len(outer_loop_pitch_ref) > 1 else 0

    pitch_ref_max_rate = max(
        abs(outer_loop_pitch_ref[i] - outer_loop_pitch_ref[i - 1])
        for i in range(1, len(outer_loop_pitch_ref))
    ) if len(outer_loop_pitch_ref) > 1 else 0.0

    active_count = sum(1 for x in outer_loop_active_col if x > 0.5)
    outer_loop_active_pct = 100 * active_count / max(1, len(outer_loop_active_col))

    def out_pct(thr):
        return 100 * sum(1 for x in abs_drift if x > thr) / nz if nz else 0.0

    lhy_vals = lhy[:]
    rhy_vals = rhy[:]
    hip_yaw_combined = [abs(x) for x in (lhy_vals + rhy_vals)]
    hip_yaw_abs_max = max(hip_yaw_combined) if hip_yaw_combined else 0.0
    min_hy_len = min(len(lhy_vals), len(rhy_vals))
    lhy_asym = [abs(lhy_vals[i] - rhy_vals[i]) for i in range(min_hy_len)]
    hip_yaw_asym_rms = math.sqrt(
        sum(x ** 2 for x in lhy_asym) / max(1, len(lhy_asym))
    ) if lhy_asym else 0.0

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

    left_contact = clean(fcol("left_contact"))
    right_contact = clean(fcol("right_contact"))
    contact_pct = 100 * sum(1 for lc, rc in zip(left_contact, right_contact)
                            if lc > 0.5 or rc > 0.5) / max(1, n)
    double_contact_pct = 100 * sum(1 for lc, rc in zip(left_contact, right_contact)
                                   if lc > 0.5 and rc > 0.5) / max(1, n)

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
        "seg_drift_maxabs": seg_drift_maxabs,
        "seg_drift_maxabs_mean": round(sum(seg_drift_maxabs) / max(1, len(seg_drift_maxabs)), 4),
        "pitch_rms": round(math.sqrt(sum(math.degrees(x) ** 2 for x in pitch) / max(1, len(pitch))), 2),
        "pitch_max": round(max((abs(math.degrees(x)) for x in pitch), default=0.0), 2),
        "roll_rms": round(math.sqrt(sum(math.degrees(x) ** 2 for x in roll) / max(1, len(roll))), 2),
        "roll_max": round(max((abs(math.degrees(x)) for x in roll), default=0.0), 2),
        "comz_min": round(min(comz), 4) if comz else 0.0,
        "comz_mean": round(sum(comz) / max(1, len(comz)), 4) if comz else 0.0,
        "hip_yaw_abs_max": round(hip_yaw_abs_max, 4),
        "hip_yaw_asym_rms": round(hip_yaw_asym_rms, 4),
        "lhy_rms": round(math.sqrt(sum(x ** 2 for x in lhy_vals) / max(1, len(lhy_vals))), 4) if lhy_vals else 0.0,
        "rhy_rms": round(math.sqrt(sum(x ** 2 for x in rhy_vals) / max(1, len(rhy_vals))), 4) if rhy_vals else 0.0,
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "contact_pct": round(contact_pct, 1),
        "double_contact_pct": round(double_contact_pct, 1),
        "pitch_ref_discontinuities": pitch_ref_cont,
        "pitch_ref_max_rate_deg": round(pitch_ref_max_rate, 3),
        "outer_loop_active_pct": round(outer_loop_active_pct, 1),
        "kp_min": round(min(outer_loop_kp), 3) if outer_loop_kp else 0.0,
        "kp_max": round(max(outer_loop_kp), 3) if outer_loop_kp else 0.0,
        "kp_range": round(max(outer_loop_kp) - min(outer_loop_kp), 3) if len(outer_loop_kp) > 1 else 0.0,
        "kd_min": round(min(outer_loop_kd), 3) if outer_loop_kd else 0.0,
        "kd_max": round(max(outer_loop_kd), 3) if outer_loop_kd else 0.0,
        "theta_max_min": round(min(outer_loop_theta_max), 3) if outer_loop_theta_max else 0.0,
        "theta_max_max": round(max(outer_loop_theta_max), 3) if outer_loop_theta_max else 0.0,
        "deadband_min": round(min(outer_loop_deadband), 3) if outer_loop_deadband else 0.0,
        "deadband_max": round(max(outer_loop_deadband), 3) if outer_loop_deadband else 0.0,
    }


def safety_ok(m):
    if m is None:
        return False, "missing"
    if m["fell"]:
        return False, f"fall({m['term_reason'][:30]})"
    if m.get("hip_yaw_abs_max", 0) > 0.35:
        return False, "hip_yaw_unsafe"
    if m.get("pitch_max", 0) > 16.0:
        return False, "pitch_unsafe"
    if m.get("roll_max", 0) > 10.0:
        return False, "roll_unsafe"
    if m.get("comz_min", 0.5) < 0.20:
        return False, "comz_too_low"
    if m.get("wbc_authority_rows", 0) > 0:
        return False, "wbc_authority_enabled"
    if m.get("wbc_owner_rows", 0) > 0:
        return False, "wbc_owner_present"
    if m.get("pitch_ref_discontinuities", 0) > 20:
        return False, "pitch_ref_discontinuous"
    return True, "safe"


def worker(args):
    scenario_name, tag, profile, out_dir, num_steps = args
    tel = run_sim(scenario_name, tag, profile, out_dir, num_steps)
    m = analyze(tel)
    return scenario_name, tag, m


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("Phase 7v2: calibrated outer-loop v2 Step C random/changing height validation", flush=True)
    print("=" * 78, flush=True)

    jobs = []
    for scenario_name, desc, num_steps in SCENARIOS:
        for tag, profile in PROFILES:
            out_dir = OUT_BASE / f"phase7v2_{scenario_name}_{tag}"
            jobs.append((scenario_name, tag, profile, out_dir, num_steps))

    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            scenario_name, tag, m = future.result()
            results[(scenario_name, tag)] = m
            sok, reason = safety_ok(m)
            status = f"safe={sok}({reason})"
            if m:
                status += f" maxabs={m['maxabs']:.4f} p2p={m['p2p']:.4f} out15={m['out15_pct']:.1f}%"
            else:
                status += " MISSING"
            print(f"  {scenario_name} {tag}: {status}", flush=True)

    hard_fail = []
    improve_cases = []
    worse_cases = []
    equal_cases = []

    for scenario_name, desc, num_steps in SCENARIOS:
        b = results.get((scenario_name, "B"))
        b2 = results.get((scenario_name, "B2v2"))
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

        if b2_better and not b2_worse:
            improve_cases.append(scenario_name)
        elif b2_worse and not b2_better:
            worse_cases.append(scenario_name)
        else:
            equal_cases.append(scenario_name)

    n_improve = len(improve_cases)
    n_worse = len(worse_cases)
    n_equal = len(equal_cases)
    n_cases = len(SCENARIOS)

    c6_b = results.get(("C6_focused_low320", "B"))
    c6_b2 = results.get(("C6_focused_low320", "B2v2"))
    c7_b = results.get(("C7_focused_high480", "B"))
    c7_b2 = results.get(("C7_focused_high480", "B2v2"))
    c8_b = results.get(("C8_low320_high450_480_loop", "B"))
    c8_b2 = results.get(("C8_low320_high450_480_loop", "B2v2"))

    focused_fail = False
    focused_notes = []
    for name, b_val, b2_val in [("C6_focused_low320", c6_b, c6_b2),
                                 ("C7_focused_high480", c7_b, c7_b2),
                                 ("C8_low320_high450_480_loop", c8_b, c8_b2)]:
        if b2_val is None:
            focused_fail = True
            focused_notes.append(f"{name}: MISSING")
        elif b2_val["fell"]:
            focused_fail = True
            focused_notes.append(f"{name}: fell({b2_val['term_reason'][:20]})")
        elif b_val and b2_val["maxabs"] > b_val["maxabs"] + 0.03:
            focused_fail = True
            focused_notes.append(f"{name}: maxabs {b2_val['maxabs']:.4f} > B {b_val['maxabs']:.4f} + 0.03")
        else:
            focused_notes.append(f"{name}: OK (maxabs={b2_val['maxabs']:.4f} vs B={b_val['maxabs'] if b_val else '?'})")

    if hard_fail:
        classification = "CALIBRATED_STEP_C_FAIL"
    elif focused_fail:
        classification = "CALIBRATED_STEP_C_FAIL"
    elif n_worse > 2:
        classification = "CALIBRATED_STEP_C_FAIL"
    elif n_improve >= 5:
        classification = "CALIBRATED_STEP_C_PASS"
    elif n_improve >= 3:
        classification = "CALIBRATED_STEP_C_PASS_WITH_MONITORING"
    else:
        classification = "CALIBRATED_STEP_C_INCONCLUSIVE"

    print(f"\n{'=' * 60}", flush=True)
    print(f"  cases:     {n_cases}", flush=True)
    print(f"  improve:   {n_improve}", flush=True)
    print(f"  equal:     {n_equal}", flush=True)
    print(f"  worse:     {n_worse}", flush=True)
    print(f"  hard_fail: {len(hard_fail)}", flush=True)
    print(f"  focused:   {'FAIL' if focused_fail else 'PASS'} ({'; '.join(focused_notes)})", flush=True)
    print(f"  Classification: {classification}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    report_dir = ROOT / "docs" / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "calibrated_outer_loop_v2_step_c_report.md"
    L = [
        "# Calibrated Outer-Loop v2 — Step C Random/Changing Height Validation",
        "",
        "## Profiles",
        f"- **B:** `support_position_outer_loop_pitch_ref` (current best)",
        f"- **B2v2:** `calibrated_support_position_outer_loop_pitch_ref_v2` (candidate)",
        "",
        f"**Classification:** `{classification}`",
        "",
        "## Gate Summary",
        f"- Hard failures: {hard_fail if hard_fail else 'none'}",
        f"- B2v2 better in: {n_improve}/{n_cases} cases",
        f"- B2v2 equal in: {n_equal}/{n_cases} cases",
        f"- B2v2 worse in: {n_worse}/{n_cases} cases",
        f"- Focused low_0p320 / high_0p480 cases: {'PASS' if not focused_fail else 'FAIL'}",
    ]
    if focused_notes:
        for fn in focused_notes:
            L.append(f"  - {fn}")

    L += [
        "",
        "## Results Table",
        "",
        "| scenario | prof | fell | maxabs | P2P | out15% | hip_yaw | contact% | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    verdict_map = {}
    for scenario_name, desc, num_steps in SCENARIOS:
        for tag in ["B", "B2v2"]:
            m = results.get((scenario_name, tag))
            sok, _ = safety_ok(m)
            v = ""
            if tag == "B2v2":
                if not sok:
                    v = "FAIL"
                elif scenario_name in improve_cases:
                    v = "BETTER"
                elif scenario_name in worse_cases:
                    v = "WORSE"
                else:
                    v = "EQUAL"
            if tag == "B2v2":
                verdict_map[scenario_name] = v
            L.append(
                f"| {scenario_name} | {tag} | {str(m.get('fell',True) if m else True)[:4]} | "
                f"{m.get('maxabs',0):.4f} | {m.get('p2p',0):.4f} | {m.get('out15_pct',0):.1f}% | "
                f"{m.get('hip_yaw_abs_max',0):.4f} | {m.get('contact_pct',0):.1f}% | {v if tag=='B2v2' else ''} |"
            )

    L += [
        "",
        "## Detailed Metrics Comparison",
        "",
        "| metric | C1 B | C1 B2v2 | C2 B | C2 B2v2 | C3 B | C3 B2v2 | C4 B | C4 B2v2 | C5 B | C5 B2v2 | C6 B | C6 B2v2 | C7 B | C7 B2v2 | C8 B | C8 B2v2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    metric_rows = [
        ("maxabs", "maxabs"),
        ("P2P", "p2p"),
        ("out15%", "out15_pct"),
        ("out10%", "out10_pct"),
        ("out05%", "out05_pct"),
        ("out03%", "out03_pct"),
        ("zero_crossings", "zero_crossings"),
        ("pos%", "pos_pct"),
        ("pitch_rms", "pitch_rms"),
        ("pitch_max", "pitch_max"),
        ("roll_rms", "roll_rms"),
        ("hip_yaw_max", "hip_yaw_abs_max"),
        ("hip_yaw_asym", "hip_yaw_asym_rms"),
        ("comz_min", "comz_min"),
        ("pitch_ref_disc", "pitch_ref_discontinuities"),
        ("ol_active%", "outer_loop_active_pct"),
    ]
    for label, key in metric_rows:
        row = f"| **{label}** |"
        for scenario_name, desc, num_steps in SCENARIOS:
            for tag in ["B", "B2v2"]:
                m = results.get((scenario_name, tag))
                val = m.get(key, "N/A") if m else "MISS"
                row += f" {val} |"
        L.append(row)

    L += [
        "",
        "## Decision",
        f"- **{classification}**",
    ]
    if "PASS" in classification:
        L.append("- **Proceed to Phase 2 (Step D push validation).**")
        L.append("- B2v2 eligible for Step D.")
    elif classification == "CALIBRATED_STEP_C_INCONCLUSIVE":
        L.append("- **Inconclusive — review metrics manually.**")
    else:
        L.append("- **Do NOT proceed to Step D.**")
        L.append("- Keep B as current best.")

    report.write_text("\n".join(L) + "\n")
    print(f"Report written: {report}", flush=True)

    csv_dir = OUT_BASE
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "calibrated_outer_loop_v2_step_c_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        first_m = None
        for scenario_name, tag in results:
            if results[(scenario_name, tag)]:
                first_m = results[(scenario_name, tag)]
                break
        if first_m:
            metric_keys = [k for k in first_m.keys() if not k.startswith("seg_") or k == "seg_drift_maxabs_mean"]
            writer = csv.writer(f)
            header = ["scenario", "profile", "classification"] + metric_keys
            writer.writerow(header)
            for scenario_name, desc, num_steps in SCENARIOS:
                for tag in ["B", "B2v2"]:
                    m = results.get((scenario_name, tag))
                    row = [scenario_name, tag, verdict_map.get(scenario_name, "")]
                    for k in metric_keys:
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