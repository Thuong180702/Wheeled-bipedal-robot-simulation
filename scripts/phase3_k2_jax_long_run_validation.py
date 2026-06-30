"""Phase 3: K2 JAX full long-run validation (5 heights x 6000 steps = 30000 JAX steps).

Backend: --controller-backend jax
Required heights: low_0p330, mid_0p400, high_0p430, high_0p450, high_0p480
Required duration: 6000 steps per height
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

HEIGHTS = [
    ("low_0p330", "low_0p330_setup.json"),
    ("mid_0p400", "mid_0p400_setup.json"),
    ("high_0p430", "high_0p430_setup.json"),
    ("high_0p450", "high_0p450_setup.json"),
    ("high_0p480", "high_0p480_setup.json"),
]


def parse_output(stdout, stderr):
    """Parse JAX backend simulation output for long-run metrics."""
    result = {
        "pass": False, "fell": False, "nan": False,
        "hidden_torque": "PASS", "wbc": "PASS",
        "max_pitch_deg": None, "max_roll_deg": None,
        "max_wheel_torque_nm": None, "max_actuator_torque_nm": None,
        "max_hip_roll_nm": None, "max_leg_nm": None,
        "height_range": None, "telemetry_rows": 0,
        "warnings": [], "steps_completed": 0,
    }
    if not stdout:
        return result

    result["pass"] = "without falling" in stdout and "Status: [OK]" in stdout
    result["fell"] = "fell" in stdout.lower() or "FALL" in stdout
    result["nan"] = "nan" in stdout.lower()

    if "hidden_torque_nonzero" in stdout.lower():
        result["hidden_torque"] = "FAIL"
    if "WBC_active" in stdout:
        result["wbc"] = "FAIL"

    m = re.search(r'Robot pitch_x range:.*?([\d.]+)\s*deg', stdout)
    if m: result["max_pitch_deg"] = float(m.group(1))
    m = re.search(r'Robot roll_y range:.*?([\d.]+)\s*deg', stdout)
    if m: result["max_roll_deg"] = float(m.group(1))
    m = re.search(r'Hip roll:\s*([\d.]+)\s*Nm', stdout)
    if m: result["max_hip_roll_nm"] = float(m.group(1))
    m = re.search(r'Wheels:\s*([\d.]+)\s*Nm', stdout)
    if m: result["max_wheel_torque_nm"] = float(m.group(1))
    m = re.search(r'Legs:\s*([\d.]+)\s*Nm', stdout)
    if m: result["max_leg_nm"] = float(m.group(1))
    m = re.search(r'Total:\s*([\d.]+)\s*Nm', stdout)
    if m: result["max_actuator_torque_nm"] = float(m.group(1))
    m = re.search(r'CoM height range:\s*([\d.]+)\s*-\s*([\d.]+)\s*m', stdout)
    if m: result["height_range"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r'Total simulated steps:\s*(\d+)', stdout)
    if m: result["steps_completed"] = int(m.group(1))
    m = re.search(r'Written telemetry rows:\s*(\d+)', stdout)
    if m: result["telemetry_rows"] = int(m.group(1))

    for line in (stderr or "").split("\n"):
        if "WARN" in line or "error" in line.lower():
            result["warnings"].append(line.strip()[:200])

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/k2_jax_release_hardening_phase3")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--heights", nargs="*", default=None)
    args = p.parse_args()

    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)

    heights_to_run = [(n, s) for n, s in HEIGHTS
                      if args.heights is None or n in args.heights]

    results = {}
    total_steps = 0
    ok = True

    for name, setup_file in heights_to_run:
        print(f"\n{'='*60}")
        print(f"=== {name} ({args.steps} steps) ===", flush=True)

        cmd = list(BASE_JAX_CMD) + [
            "--height-variant-setup", str(SETUP_DIR / setup_file),
            "--steps", str(args.steps),
        ]

        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {args.timeout}s", flush=True)
            results[name] = {"pass": False, "fell": False, "nan": False,
                           "error": "TIMEOUT", "steps": args.steps}
            ok = False
            continue

        elapsed = time.time() - t0
        parsed = parse_output(r.stdout, r.stderr)

        status = "PASS" if parsed["pass"] else "FAIL"
        print(f"  Status: {status}  Fell: {parsed['fell']}  NaN: {parsed['nan']}")
        print(f"  Steps completed: {parsed['steps_completed']}")
        print(f"  Max pitch: {parsed['max_pitch_deg']} deg  Max roll: {parsed['max_roll_deg']} deg")
        print(f"  Max actuator torque: {parsed['max_actuator_torque_nm']} Nm")
        print(f"  Height range: {parsed['height_range']}")
        print(f"  Hidden torque: {parsed['hidden_torque']}  WBC: {parsed['wbc']}")
        print(f"  Wall-clock: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

        results[name] = {
            "height": name, "steps_requested": args.steps,
            "steps_completed": parsed["steps_completed"],
            "pass": parsed["pass"], "fell": parsed["fell"], "nan": parsed["nan"],
            "max_pitch_deg": parsed["max_pitch_deg"],
            "max_roll_deg": parsed["max_roll_deg"],
            "max_wheel_torque_nm": parsed["max_wheel_torque_nm"],
            "max_actuator_torque_nm": parsed["max_actuator_torque_nm"],
            "max_hip_roll_nm": parsed["max_hip_roll_nm"],
            "max_leg_nm": parsed["max_leg_nm"],
            "height_range": parsed["height_range"],
            "hidden_torque": parsed["hidden_torque"],
            "wbc": parsed["wbc"],
            "warnings_count": len(parsed["warnings"]),
            "wall_clock_s": elapsed,
        }
        total_steps += parsed["steps_completed"]

        if not parsed["pass"]:
            ok = False

    # Summary
    passed = sum(1 for r in results.values() if r["pass"])
    total = len(results)

    summary = {
        "phase": 3,
        "title": "K2 JAX Long-Run Validation",
        "backend": "jax",
        "steps_per_height": args.steps,
        "total_jax_steps": total_steps,
        "heights_run": total,
        "heights_passed": passed,
        "all_pass": ok,
        "results": results,
    }
    with open(od / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 3: K2 JAX Long-Run Validation")
    for name, r in results.items():
        s = "PASS" if r["pass"] else "FAIL"
        print(f"  {s:4s} {name:15s}  steps={r['steps_completed']}  pitch={r['max_pitch_deg']}deg  "
              f"wall={r['wall_clock_s']:.0f}s")
    print(f"\n  Total JAX steps: {total_steps}")
    print(f"  {passed}/{total} heights pass")

    if ok and passed == total and total_steps >= 30000:
        print("Classification: K2_JAX_RELEASE_HARDENING_LONG_RUN_PASS")
        sys.exit(0)
    else:
        print("Classification: K2_JAX_RELEASE_HARDENING_LONG_RUN_FAIL_WITH_ROOT_CAUSE")
        sys.exit(1)


if __name__ == "__main__":
    main()
