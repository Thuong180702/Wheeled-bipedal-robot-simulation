"""Phase 1: Full 9-scenario both-synced parity using simulate_hierarchical_controller.py.

Runs each scenario with --controller-backend both-synced, which does
teacher-forcing: Python drives the physics, JAX runs in parallel with
identical state, and torque output is compared per-step.

Scenarios:
  1. fixed_high_0p480 — constant 0.48m, 500 steps
  2. fixed_low_0p330  — constant 0.33m, 500 steps
  3. ramp_up          — 0.33→0.48m, 500 steps (dynamic trajectory)
  4. ramp_down        — 0.48→0.33m, 500 steps (dynamic trajectory)
  5. up_down_cycle    — 0.33→0.48→0.33m, 700 steps (dynamic trajectory)
  6. gate_dwell       — dwell 0.42/0.45/0.48m, 600 steps (dynamic trajectory)
  7. gate_chatter     — transitions 0.40-0.47m, 500 steps (dynamic trajectory)
  8. push_fwd_90N     — 0.48m + forward 90N push, 500 steps
  9. push_bwd_90N     — 0.48m + backward 90N push, 500 steps
"""

import argparse, json, subprocess, sys, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"

BASE_CMD = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "both-synced",
    "--wbc-quiet",
]


def run_cmd(name, extra_args, timeout=300):
    """Run simulation and return (stdout, stderr, elapsed, returncode)."""
    cmd = list(BASE_CMD) + extra_args
    print(f"  [{name}] Running: {' '.join(cmd[-5:])}...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, time.time() - t0, -1
    elapsed = time.time() - t0
    return r.stdout, r.stderr, elapsed, r.returncode


def parse_both_synced_output(stdout):
    """Parse the both-synced max_abs_diff and classification from stdout."""
    result = {
        "max_abs_diff": None,
        "max_diff_step": None,
        "max_diff_actuator": None,
        "classification": None,
        "fall_status": "no_fall",
        "hidden_torque": "PASS",
        "wbc_active": "PASS",
    }
    if not stdout:
        return result

    # Parse max_abs_diff
    m = re.search(r'Worst max_abs_diff:\s*([\d.e+\-]+)', stdout)
    if m:
        result["max_abs_diff"] = float(m.group(1))

    m = re.search(r'at step\s+(\d+),\s*actuator index\s+(\d+)', stdout)
    if m:
        result["max_diff_step"] = int(m.group(1))
        result["max_diff_actuator"] = int(m.group(2))

    m = re.search(r'Classification:\s*(\S+)', stdout)
    if m:
        result["classification"] = m.group(1)

    # Fall detection
    if "without falling" not in stdout:
        result["fall_status"] = "fell" if "fell" in stdout.lower() or "FALL" in stdout else "check"

    # Hidden torque
    if "hidden_torque_nonzero" in stdout.lower():
        result["hidden_torque"] = "FAIL"

    # WBC
    if "WBC_active" in stdout:
        result["wbc_active"] = "FAIL"

    return result


def write_dynamic_trajectory(name, waypoints, output_dir):
    """Write a dynamic height trajectory JSON file."""
    wp_data = []
    for step, height_m in waypoints:
        wp_data.append({"step": step, "height_m": height_m})
    traj = {
        "height_profile_name": name,
        "steps": waypoints[-1][0],  # last step
        "waypoints": wp_data,
    }
    path = output_dir / f"traj_{name}.json"
    with open(path, "w") as f:
        json.dump(traj, f, indent=2)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/k2_jax_release_hardening_phase1")
    p.add_argument("--steps-per-scenario", type=int, default=500,
                   help="Steps for most scenarios")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout per scenario (seconds)")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="Specific scenarios to run (default: all 9)")
    args = p.parse_args()

    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)

    std_steps = args.steps_per_scenario

    # =====================================================================
    # Scenario definitions
    # =====================================================================
    FIXED_HIGH = str(SETUP_DIR / "high_0p480_setup.json")
    FIXED_LOW = str(SETUP_DIR / "low_0p330_setup.json")

    # Dynamic height trajectory definitions
    dynamic_trajs = {
        "ramp_up": [
            (0, 0.33), (std_steps, 0.48),
        ],
        "ramp_down": [
            (0, 0.48), (std_steps, 0.33),
        ],
        "up_down_cycle": [
            (0, 0.33),
            (std_steps // 2, 0.48),
            (std_steps, 0.33),
        ],
        "gate_dwell": [
            (0, 0.42),
            (std_steps // 3, 0.42),
            (std_steps // 3 + 1, 0.45),
            (2 * std_steps // 3, 0.45),
            (2 * std_steps // 3 + 1, 0.48),
            (std_steps, 0.48),
        ],
        "gate_chatter": [
            (0, 0.40),
            (std_steps // 4, 0.47),
            (std_steps // 2, 0.42),
            (3 * std_steps // 4, 0.47),
            (std_steps, 0.40),
        ],
    }

    # All 9 scenarios
    ALL_SCENARIOS = {
        "fixed_high_0p480": {
            "desc": "Fixed height 0.480m",
            "type": "fixed",
            "setup": FIXED_HIGH,
            "steps": std_steps,
        },
        "fixed_low_0p330": {
            "desc": "Fixed height 0.330m",
            "type": "fixed",
            "setup": FIXED_LOW,
            "steps": std_steps,
        },
        "ramp_up": {
            "desc": "Ramp up 0.33 to 0.48m",
            "type": "dynamic",
            "traj_key": "ramp_up",
            "steps": std_steps,
        },
        "ramp_down": {
            "desc": "Ramp down 0.48 to 0.33m",
            "type": "dynamic",
            "traj_key": "ramp_down",
            "steps": std_steps,
        },
        "up_down_cycle": {
            "desc": "Up-down cycle 0.33 to 0.48 to 0.33m",
            "type": "dynamic",
            "traj_key": "up_down_cycle",
            "steps": std_steps,
        },
        "gate_dwell": {
            "desc": "Gate dwell 0.42/0.45/0.48m",
            "type": "dynamic",
            "traj_key": "gate_dwell",
            "steps": std_steps,
        },
        "gate_chatter": {
            "desc": "Gate chatter 0.40-0.47m",
            "type": "dynamic",
            "traj_key": "gate_chatter",
            "steps": std_steps,
        },
        "push_fwd_90N": {
            "desc": "Push forward 90N at 0.480m",
            "type": "push",
            "setup": FIXED_HIGH,
            "steps": std_steps,
            "push_dir": "forward",
            "push_N": 90,
        },
        "push_bwd_90N": {
            "desc": "Push backward 90N at 0.480m",
            "type": "push",
            "setup": FIXED_HIGH,
            "steps": std_steps,
            "push_dir": "backward",
            "push_N": 90,
        },
    }

    # Filter scenarios if requested
    scenarios_to_run = {}
    if args.scenarios:
        for s in args.scenarios:
            if s in ALL_SCENARIOS:
                scenarios_to_run[s] = ALL_SCENARIOS[s]
            else:
                print(f"WARNING: Unknown scenario '{s}', skipping")
    else:
        scenarios_to_run = dict(ALL_SCENARIOS)

    # Write dynamic trajectories
    for name, info in scenarios_to_run.items():
        if info["type"] == "dynamic":
            traj_key = info["traj_key"]
            info["traj_path"] = write_dynamic_trajectory(
                traj_key, dynamic_trajs[traj_key], od)

    # =====================================================================
    # Run all scenarios
    # =====================================================================
    results = {}
    ok = True

    for name, info in scenarios_to_run.items():
        print(f"\n{'='*60}")
        print(f"=== {name}: {info['desc']} ===", flush=True)

        extra = ["--steps", str(info["steps"])]

        if info["type"] == "fixed":
            extra += ["--height-variant-setup", info["setup"]]
        elif info["type"] == "dynamic":
            extra += ["--dynamic-height-trajectory", str(info["traj_path"])]
        elif info["type"] == "push":
            extra += [
                "--height-variant-setup", info["setup"],
                "--push-enabled",
                "--push-magnitude-n", str(info["push_N"]),
                "--push-duration-steps", "5",
                "--push-interval-steps", str(info["steps"] // 2),
                "--push-start-step", "20",
            ]
            if info["push_dir"] == "forward":
                extra += ["--sagittal-push-only"]

        t0 = time.time()
        stdout, stderr, elapsed, rc = run_cmd(name, extra, timeout=args.timeout)

        if rc != 0:
            print(f"  FAILED: returncode={rc}, elapsed={elapsed:.0f}s", flush=True)
            if stderr:
                print(f"  STDERR: {stderr[:500]}", flush=True)
            parsed = {"max_abs_diff": float("inf"), "classification": "COMMAND_FAILED",
                     "fall_status": "unknown"}
            ok = False
        else:
            parsed = parse_both_synced_output(stdout)
            elapsed = time.time() - t0

        max_diff = parsed["max_abs_diff"]
        classification = parsed.get("classification", "unknown")

        # Determine pass/fail
        threshold = 1e-5
        passed = False
        if max_diff is not None and max_diff < threshold:
            passed = True

        status = "PASS" if passed else "FAIL"

        print(f"  Status: {status}  Max 10-dim diff: {max_diff if max_diff is not None else 'N/A'}")
        print(f"  Classification: {classification}")
        print(f"  Diff step: {parsed.get('max_diff_step')}  actuator: {parsed.get('max_diff_actuator')}")
        print(f"  Fall: {parsed['fall_status']}  Hidden torque: {parsed['hidden_torque']}  WBC: {parsed['wbc_active']}")
        print(f"  Elapsed: {elapsed:.0f}s", flush=True)

        results[name] = {
            "scenario": name,
            "description": info["desc"],
            "scenario_type": info["type"],
            "steps": info["steps"],
            "max_abs_diff": max_diff,
            "max_diff_step": parsed["max_diff_step"],
            "max_diff_actuator": parsed["max_diff_actuator"],
            "classification": classification,
            "fall_status": parsed["fall_status"],
            "hidden_torque": parsed["hidden_torque"],
            "wbc_active": parsed["wbc_active"],
            "threshold": threshold,
            "passed": passed,
            "elapsed_s": elapsed,
        }

        if not passed:
            ok = False

    # =====================================================================
    # Summary
    # =====================================================================
    passed_count = sum(1 for r in results.values() if r["passed"])
    total = len(results)
    failed_count = total - passed_count

    summary = {
        "phase": 1,
        "title": "Full 9-Scenario Both-Synced Parity",
        "threshold": 1e-5,
        "total_scenarios": total,
        "passed": passed_count,
        "failed": failed_count,
        "all_pass": ok,
        "results": results,
    }

    with open(od / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 1: Full 9-Scenario Both-Synced Parity")
    print(f"  {passed_count}/{total} PASS ({failed_count} FAIL)")
    for name, r in results.items():
        s = "PASS" if r["passed"] else "FAIL"
        d = f"{r['max_abs_diff']:.2e}" if r["max_abs_diff"] is not None else "N/A"
        print(f"  {s:4s} {name:20s}  max_diff={d}")
    print()

    if ok:
        print("Classification: K2_JAX_RELEASE_HARDENING_9_SCENARIO_PARITY_PASS")
    else:
        print("Classification: K2_JAX_RELEASE_HARDENING_9_SCENARIO_PARITY_FAIL_WITH_ROOT_CAUSE")
        # Identify failing scenarios
        failing = [name for name, r in results.items() if not r["passed"]]
        print(f"Failing scenarios: {failing}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
