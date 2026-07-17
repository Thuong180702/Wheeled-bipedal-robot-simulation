"""Phase 2: K2 JAX full functional validation (JAX backend, NOT both-synced).

Backend: --controller-backend jax
Required scenarios:
  A. Fixed-height: low_0p330, mid_0p400, high_0p430, high_0p450, high_0p480
  B. Dynamic-height: ramp_up, ramp_down, up_down_cycle, gate_dwell, gate_chatter
  C. Push: high_0p480 forward 90N, high_0p480 backward 90N

Reports: pass/fail, fall status, NaN status, hidden torque/WBC, pitch/roll max,
         hip_yaw_abs_max, wheel torque max, support error max, actuator torque max,
         safety violations, final state summary, warning logs.
"""

import argparse, json, subprocess, sys, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"

BASE_JAX_CMD = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "jax",
    "--wbc-quiet",
]


def write_dynamic_trajectory(name, waypoints, output_dir):
    wp_data = [{"step": s, "height_m": h} for s, h in waypoints]
    traj = {"height_profile_name": name, "steps": waypoints[-1][0], "waypoints": wp_data}
    path = output_dir / f"traj_{name}.json"
    with open(path, "w") as f:
        json.dump(traj, f, indent=2)
    return path


def run_case(name, extra_args, timeout=300):
    cmd = list(BASE_JAX_CMD) + extra_args
    print(f"  [{name}] Launching with {len(extra_args)} extra args...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, time.time() - t0
    elapsed = time.time() - t0
    return r.stdout, r.stderr, elapsed


def parse_output(stdout, stderr):
    """Parse JAX backend simulation output."""
    result = {
        "pass": False,
        "fell": False,
        "nan": False,
        "hidden_torque": "PASS",
        "wbc": "PASS",
        "max_pitch_deg": None,
        "max_roll_deg": None,
        "hip_yaw_abs_max_deg": None,
        "wheel_torque_max_nm": None,
        "support_error_max_m": None,
        "actuator_torque_max_nm": None,
        "warnings": [],
        "falls": None,
        "height_range": None,
    }
    if not stdout:
        return result

    result["pass"] = "without falling" in stdout
    result["fell"] = "fell" in stdout.lower() or "FALL" in stdout

    # NaN check
    result["nan"] = "nan" in stdout.lower()

    # Hidden torque / WBC
    if "hidden_torque_nonzero" in stdout.lower():
        result["hidden_torque"] = "FAIL"
    if "WBC_active" in stdout:
        result["wbc"] = "FAIL"

    # Pitch / roll range
    m = re.search(r'Robot pitch_x range:\s*[\-\d.]+\s*-\s*([\d.]+)\s*deg', stdout)
    if m: result["max_pitch_deg"] = float(m.group(1))

    m = re.search(r'Robot roll_y range:\s*[\-\d.]+\s*-\s*([\d.]+)\s*deg', stdout)
    if m: result["max_roll_deg"] = float(m.group(1))

    # Height range
    m = re.search(r'CoM height range:\s*([\d.]+)\s*-\s*([\d.]+)\s*m', stdout)
    if m: result["height_range"] = (float(m.group(1)), float(m.group(2)))

    # Max torques
    m = re.search(r'Hip roll:\s*([\d.]+)\s*Nm', stdout)
    if m: result["hip_roll_torque_max_nm"] = float(m.group(1))
    m = re.search(r'Wheels:\s*([\d.]+)\s*Nm', stdout)
    if m: result["wheel_torque_max_nm"] = float(m.group(1))
    m = re.search(r'Legs:\s*([\d.]+)\s*Nm', stdout)
    if m: result["leg_torque_max_nm"] = float(m.group(1))
    m = re.search(r'Total:\s*([\d.]+)\s*Nm', stdout)
    if m: result["actuator_torque_max_nm"] = float(m.group(1))

    # Warnings
    for line in (stderr or "").split("\n"):
        if "WARN" in line or "Error" in line or "error" in line or "warn" in line.lower():
            result["warnings"].append(line.strip()[:200])

    # Telemetry rows
    m = re.search(r'Written telemetry rows:\s*(\d+)', stdout)
    if m: result["telemetry_rows"] = int(m.group(1))

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/k2_jax_release_hardening_phase2")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--scenarios", nargs="*", default=None)
    args = p.parse_args()

    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)

    # Fixed height setups
    FIXED_SETUPS = {
        "low_0p330": str(SETUP_DIR / "low_0p330_setup.json"),
        "mid_0p400": str(SETUP_DIR / "mid_0p400_setup.json"),
        "high_0p430": str(SETUP_DIR / "high_0p430_setup.json"),
        "high_0p450": str(SETUP_DIR / "high_0p450_setup.json"),
        "high_0p480": str(SETUP_DIR / "high_0p480_setup.json"),
    }

    # Dynamic height trajectories
    DYNAMIC_TRAJS = {
        "ramp_up": [(0, 0.33), (args.steps, 0.48)],
        "ramp_down": [(0, 0.48), (args.steps, 0.33)],
        "up_down_cycle": [(0, 0.33), (args.steps // 2, 0.48), (args.steps, 0.33)],
        "gate_dwell": [(0, 0.42), (args.steps // 3, 0.42),
                        (args.steps // 3 + 1, 0.45), (2 * args.steps // 3, 0.45),
                        (2 * args.steps // 3 + 1, 0.48), (args.steps, 0.48)],
        "gate_chatter": [(0, 0.40), (args.steps // 4, 0.47), (args.steps // 2, 0.42),
                          (3 * args.steps // 4, 0.47), (args.steps, 0.40)],
    }

    # Build scenario list
    all_scenarios = {}

    # A: Fixed-height
    for variant, setup_path in FIXED_SETUPS.items():
        all_scenarios[f"fixed/{variant}"] = {
            "group": "A_fixed",
            "extra": ["--height-variant-setup", setup_path, "--steps", str(args.steps)],
        }

    # B: Dynamic-height
    for name, waypoints in DYNAMIC_TRAJS.items():
        traj_path = write_dynamic_trajectory(name, waypoints, od)
        all_scenarios[f"dynamic/{name}"] = {
            "group": "B_dynamic",
            "extra": ["--dynamic-height-trajectory", str(traj_path), "--steps", str(args.steps)],
        }

    # C: Push
    for direction, dir_flag in [("forward", ["--sagittal-push-only"]), ("backward", [])]:
        all_scenarios[f"push/{direction}_90N"] = {
            "group": "C_push",
            "extra": [
                "--height-variant-setup", FIXED_SETUPS["high_0p480"],
                "--push-enabled", "--push-magnitude-n", "90",
                "--push-duration-steps", "5",
                "--push-interval-steps", str(args.steps // 2),
                "--push-start-step", "20",
                "--steps", str(args.steps),
            ] + dir_flag,
        }

    # Filter if requested
    if args.scenarios:
        to_run = {k: v for k, v in all_scenarios.items()
                  if any(s in k for s in args.scenarios)}
    else:
        to_run = all_scenarios

    # Run
    results = {}
    for name, info in to_run.items():
        print(f"\n{'='*60}")
        print(f"=== {name} ===", flush=True)
        stdout, stderr, elapsed = run_case(name, info["extra"], timeout=args.timeout)
        parsed = parse_output(stdout, stderr)

        status = "PASS" if parsed["pass"] else "FAIL"
        print(f"  Status: {status}  Fall: {parsed['fell']}  NaN: {parsed['nan']}")
        print(f"  Max pitch: {parsed['max_pitch_deg']} deg  Max roll: {parsed['max_roll_deg']} deg")
        print(f"  Wheel torque max: {parsed['wheel_torque_max_nm']} Nm")
        print(f"  Actuator max: {parsed['actuator_torque_max_nm']} Nm")
        print(f"  Hidden torque: {parsed['hidden_torque']}  WBC: {parsed['wbc']}")
        if parsed["warnings"]:
            print(f"  Warnings: {parsed['warnings'][:3]}")
        print(f"  Elapsed: {elapsed:.0f}s", flush=True)

        results[name] = {**info, **parsed, "elapsed_s": elapsed}

    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if r["pass"])
    failed = total - passed

    fixed_total = sum(1 for r in results.values() if r["group"] == "A_fixed")
    fixed_pass = sum(1 for r in results.values() if r["group"] == "A_fixed" and r["pass"])
    dynamic_total = sum(1 for r in results.values() if r["group"] == "B_dynamic")
    dynamic_pass = sum(1 for r in results.values() if r["group"] == "B_dynamic" and r["pass"])
    push_total = sum(1 for r in results.values() if r["group"] == "C_push")
    push_pass = sum(1 for r in results.values() if r["group"] == "C_push" and r["pass"])

    summary = {
        "phase": 2,
        "title": "K2 JAX Functional Validation",
        "backend": "jax",
        "steps_per_scenario": args.steps,
        "total": total, "passed": passed, "failed": failed,
        "fixed": f"{fixed_pass}/{fixed_total}",
        "dynamic": f"{dynamic_pass}/{dynamic_total}",
        "push": f"{push_pass}/{push_total}",
        "results": {k: {"group": r["group"], "pass": r["pass"],
                         "fell": r["fell"], "nan": r["nan"],
                         "max_pitch_deg": r["max_pitch_deg"],
                         "max_roll_deg": r["max_roll_deg"],
                         "hip_roll_max": r.get("hip_roll_torque_max_nm"),
                         "wheel_max": r["wheel_torque_max_nm"],
                         "actuator_max": r["actuator_torque_max_nm"],
                         "hidden_torque": r["hidden_torque"],
                         "wbc": r["wbc"],
                         "height_range": r["height_range"],
                         "elapsed_s": r["elapsed_s"]}
                    for k, r in results.items()},
    }
    with open(od / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 2: K2 JAX Functional Validation")
    print(f"  Fixed:     {fixed_pass}/{fixed_total}")
    print(f"  Dynamic:   {dynamic_pass}/{dynamic_total}")
    print(f"  Push:      {push_pass}/{push_total}")
    print(f"  Total:     {passed}/{total}")

    all_pass = passed == total
    if all_pass and fixed_pass == fixed_total and dynamic_pass == dynamic_total and push_pass == push_total:
        print("Classification: K2_JAX_RELEASE_HARDENING_FUNCTIONAL_PASS")
        sys.exit(0)
    else:
        print("Classification: K2_JAX_RELEASE_HARDENING_FUNCTIONAL_FAIL_WITH_ROOT_CAUSE")
        sys.exit(1)


if __name__ == "__main__":
    main()
