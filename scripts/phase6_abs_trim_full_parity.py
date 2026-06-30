"""Phase 6: Full both-synced parity rerun after ABS trim fix."""
import subprocess, sys, time, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_DIR = ROOT / "outputs" / "k2_jax_abs_trim_phase6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CMD = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "both-synced",
    "--wbc-quiet",
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]


def write_traj(name, waypoints, steps):
    traj_dir = OUT_DIR / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    wp_data = [{"step": int(s), "height_m": float(h)} for s, h in waypoints]
    traj = {"height_profile_name": name, "steps": steps, "waypoints": wp_data}
    path = traj_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(traj, f, indent=2)
    return path


def run_scenario(name, extra_args, timeout=1800):
    cmd = list(BASE_CMD) + extra_args
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"scenario": name, "status": "TIMEOUT", "elapsed_s": timeout}
    elapsed = time.time() - t0
    stdout = r.stdout
    max_abs_diff = None
    max_diff_step = None
    max_diff_actuator = None
    classification = None
    fell = "without falling" not in stdout
    for line in stdout.splitlines():
        m = re.search(r'Worst max_abs_diff:\s*([\d.e+\-]+)', line)
        if m: max_abs_diff = float(m.group(1))
        m = re.search(r'at step\s+(\d+),\s+actuator index\s+(\d+)', line)
        if m: max_diff_step = int(m.group(1)); max_diff_actuator = int(m.group(2))
        m = re.search(r'Classification:\s*(\S+)', line)
        if m: classification = m.group(1)
    # Parse wheel and hip-yaw diffs
    wheel_4_diffs = []; wheel_9_diffs = []; hy1_diffs = []; hy6_diffs = []
    for line in stdout.splitlines():
        if "DIFF=   [" in line:
            m = re.search(r'DIFF=\s+\[(.+)\]', line)
            if m:
                raw = m.group(1).replace(",", " ").strip()
                vals = [float(x.strip()) for x in raw.split()]
                if len(vals) > 4: wheel_4_diffs.append(abs(vals[4]))
                if len(vals) > 9: wheel_9_diffs.append(abs(vals[9]))
                if len(vals) > 1: hy1_diffs.append(abs(vals[1]))
                if len(vals) > 6: hy6_diffs.append(abs(vals[6]))
    result = {
        "scenario": name, "status": "COMPLETED", "fell": fell, "elapsed_s": elapsed,
        "max_abs_diff": max_abs_diff, "max_diff_step": max_diff_step,
        "max_diff_actuator": max_diff_actuator, "classification": classification,
        "wheel_4_max": max(wheel_4_diffs) if wheel_4_diffs else None,
        "wheel_9_max": max(wheel_9_diffs) if wheel_9_diffs else None,
        "hy_1_max": max(hy1_diffs) if hy1_diffs else None,
        "hy_6_max": max(hy6_diffs) if hy6_diffs else None,
        "returncode": r.returncode,
    }
    log_path = OUT_DIR / f"{name}_stdout.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(r.stderr)
    print(f"  Result: max_abs_diff={max_abs_diff:.6e}" if max_abs_diff else "  Result: parse failed")
    print(f"  Divergence: step={max_diff_step}, actuator={max_diff_actuator}")
    print(f"  Wheel[4] max: {result['wheel_4_max']}, Wheel[9] max: {result['wheel_9_max']}")
    print(f"  Fell: {fell}, Elapsed: {elapsed:.0f}s, Classification: {classification}")
    return result


results = []
all_start = time.time()

# 1. fixed_high_0p480
r = run_scenario("fixed_high_0p480", [
    "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
    "--steps", "50",
], timeout=300)
results.append(r)

# 2. fixed_low_0p330
r = run_scenario("fixed_low_0p330", [
    "--height-variant-setup", str(SETUP_DIR / "low_0p330_setup.json"),
    "--steps", "50",
], timeout=300)
results.append(r)

# 3. ramp_up
traj = write_traj("ramp_up_0p330_to_0p480", [(0, 0.330), (500, 0.330), (3500, 0.480), (5000, 0.480)], 500)
r = run_scenario("ramp_up", [
    "--height-variant-setup", str(SETUP_DIR / "low_0p330_setup.json"),
    "--steps", "500", "--dynamic-height-trajectory", str(traj),
], timeout=1800)
results.append(r)

# 4. ramp_down
traj = write_traj("ramp_down_0p480_to_0p330", [(0, 0.480), (500, 0.480), (3500, 0.330), (5000, 0.330)], 500)
r = run_scenario("ramp_down", [
    "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
    "--steps", "500", "--dynamic-height-trajectory", str(traj),
], timeout=1800)
results.append(r)

# 5. up_down_cycle
traj = write_traj("up_down_cycle", [(0, 0.330), (500, 0.330), (2500, 0.480), (3500, 0.480), (4500, 0.330), (5000, 0.330)], 500)
r = run_scenario("up_down_cycle", [
    "--height-variant-setup", str(SETUP_DIR / "low_0p330_setup.json"),
    "--steps", "500", "--dynamic-height-trajectory", str(traj),
], timeout=1800)
results.append(r)

# 6. gate_dwell
traj = write_traj("gate_dwell", [(0, 0.330), (500, 0.330), (3500, 0.400), (5000, 0.400)], 500)
r = run_scenario("gate_dwell", [
    "--height-variant-setup", str(SETUP_DIR / "low_0p330_setup.json"),
    "--steps", "500", "--dynamic-height-trajectory", str(traj),
], timeout=1800)
results.append(r)

# 7. gate_chatter
traj = write_traj("gate_chatter", [
    (0, 0.400), (400, 0.400), (700, 0.430), (1000, 0.400),
    (1300, 0.450), (1600, 0.400), (1900, 0.470), (2200, 0.400),
    (2500, 0.430), (2800, 0.400), (3100, 0.450), (3400, 0.400),
    (3700, 0.470), (4000, 0.400), (4300, 0.430), (4600, 0.400),
    (5000, 0.400),
], 500)
r = run_scenario("gate_chatter", [
    "--height-variant-setup", str(SETUP_DIR / "low_0p330_setup.json"),
    "--steps", "500", "--dynamic-height-trajectory", str(traj),
], timeout=1800)
results.append(r)

# 8. push_fwd_90N
push_seq = OUT_DIR / "push_fwd_seq.json"
with open(push_seq, "w") as f:
    json.dump([[100, 0.0, 90.0, 5]], f)
r = run_scenario("push_fwd_90N", [
    "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
    "--steps", "300", "--push-sequence-file", str(push_seq),
], timeout=1800)
results.append(r)

# 9. push_bwd_90N
push_seq = OUT_DIR / "push_bwd_seq.json"
with open(push_seq, "w") as f:
    json.dump([[100, 180.0, 90.0, 5]], f)
r = run_scenario("push_bwd_90N", [
    "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
    "--steps", "300", "--push-sequence-file", str(push_seq),
], timeout=1800)
results.append(r)

# Summary
total_elapsed = time.time() - all_start
print("\n\n" + "="*140)
print("PHASE 6 FULL BOTH-SYNCED PARITY RERUN")
print("="*140)
print(f"{'Scenario':<25s} {'MaxDiff':>14s} {'Step':>6s} {'Act':>4s} {'Wheel[4]':>14s} {'Wheel[9]':>14s} {'HY[1]':>14s} {'HY[6]':>14s} {'Fell':>5s} {'Class':>45s}")
print("-" * 140)

pass_count = 0; fail_count = 0
for r in results:
    md = f"{r['max_abs_diff']:.6e}" if r['max_abs_diff'] is not None else "N/A"
    ms = str(r['max_diff_step']) if r['max_diff_step'] is not None else "N/A"
    ma = str(r['max_diff_actuator']) if r['max_diff_actuator'] is not None else "N/A"
    w4 = f"{r['wheel_4_max']:.6e}" if r['wheel_4_max'] is not None else "N/A"
    w9 = f"{r['wheel_9_max']:.6e}" if r['wheel_9_max'] is not None else "N/A"
    h1 = f"{r['hy_1_max']:.6e}" if r['hy_1_max'] is not None else "N/A"
    h6 = f"{r['hy_6_max']:.6e}" if r['hy_6_max'] is not None else "N/A"
    fell = "YES" if r['fell'] else "no"
    cls = r['classification'] or "N/A"
    is_pass = (r['max_abs_diff'] is not None and r['max_abs_diff'] < 1e-5)
    if is_pass: pass_count += 1
    else: fail_count += 1
    print(f"{r['scenario']:<25s} {md:>14s} {ms:>6s} {ma:>4s} {w4:>14s} {w9:>14s} {h1:>14s} {h6:>14s} {fell:>5s} {cls:>45s}")

print("-" * 140)
print(f"\nPassed (<1e-5): {pass_count}/{len(results)}")
print(f"Failed: {fail_count}/{len(results)}")

summary = {
    "phase": 6, "title": "ABS trim full both-synced parity rerun",
    "scenarios": results, "pass_count": pass_count, "fail_count": fail_count,
    "total_elapsed_s": total_elapsed,
}
json_path = OUT_DIR / "phase6_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSummary: {json_path}")
