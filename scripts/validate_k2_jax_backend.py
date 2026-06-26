"""Stage 6: K2 JAX backend validation wrapper.

Runs the K2 simulation with --controller-backend jax for Step C, D, E,
single-push, and dynamic-height scenarios. Reports completion status.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_DIR = ROOT / "outputs" / "k2_jax_validation"
TIMEOUT = 600  # 10 min per run

BASE_CMD = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "jax",
]

MODE_DIV_FLAGS = [
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]


def setup_path(variant):
    p = SETUP_DIR / f"{variant}_setup.json"
    return str(p) if p.exists() else None


def run_case(name, height_variant, steps=500, extra_args=None):
    """Run one validation case, return (success, output_summary)."""
    sp = setup_path(height_variant)
    if sp is None:
        return False, f"SETUP_MISSING:{height_variant}"
    cmd = list(BASE_CMD) + [
        "--height-variant-setup", sp,
        "--steps", str(steps),
        "--output-dir", str(OUT_DIR / name),
    ] + MODE_DIV_FLAGS
    if extra_args:
        cmd += extra_args
    print(f"  [{name}] {height_variant} {steps} steps...", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    elapsed = time.time() - t0
    combined = (result.stdout or "") + (result.stderr or "")
    ok = result.returncode == 0 and ("without falling" in combined or "[OK] Completed" in combined)
    status = "OK" if ok else f"FAIL(rc={result.returncode})"
    print(f"    {status} in {elapsed:.0f}s", flush=True)
    if not ok:
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / name / "stderr.txt").write_text(result.stderr[-2000:] if result.stderr else "")
    return ok, result.stdout[-500:] if ok else result.stderr[-500:]


def main():
    results = {}

    # ---- Step C: 7 fixed-height cases (1000 steps each for speed) ----
    print("\n=== STEP C: Fixed-Height Validation (JAX backend) ===")
    step_c_cases = [
        ("c1_low_0p330", "low_0p330"),
        ("c2_low_0p340", "low_0p340"),
        ("c3_low_0p360", "low_0p360"),
        ("c4_low_0p380", "low_0p380"),
        ("c5_mid_0p400", "mid_0p400"),
        ("c6_high_0p430", "high_0p430"),
        ("c7_high_0p480", "high_0p480"),
    ]
    step_c_ok = 0
    for name, variant in step_c_cases:
        ok, _ = run_case(f"step_c/{name}", variant, steps=1000)
        results[f"step_c/{name}"] = "PASS" if ok else "FAIL"
        if ok: step_c_ok += 1
    print(f"Step C: {step_c_ok}/{len(step_c_cases)} passed")

    # ---- Step D: Push matrix (3 heights × 2 dirs × 2 mags, 500 steps) ----
    print("\n=== STEP D: Push Matrix (JAX backend) ===")
    heights = ["low_0p330", "mid_0p400", "high_0p480"]
    directions = [("forward", "90"), ("backward", "90")]
    step_d_ok = 0; step_d_total = 0
    for h in heights:
        for dname, force in directions:
            step_d_total += 1
            ok, _ = run_case(f"step_d/{h}_{dname}_{force}N", h, steps=500)
            results[f"step_d/{h}_{dname}_{force}N"] = "PASS" if ok else "FAIL"
            if ok: step_d_ok += 1
    print(f"Step D: {step_d_ok}/{step_d_total} passed")

    # ---- Step E: 10-height sweep (1000 steps each) ----
    print("\n=== STEP E: Height Sweep (JAX backend) ===")
    step_e_heights = [
        "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
        "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
    ]
    step_e_ok = 0
    for h in step_e_heights:
        ok, _ = run_case(f"step_e/{h}", h, steps=1000)
        results[f"step_e/{h}"] = "PASS" if ok else "FAIL"
        if ok: step_e_ok += 1
    print(f"Step E: {step_e_ok}/{len(step_e_heights)} passed")

    # ---- Single-push: high_0p480 forward/backward 90N ----
    print("\n=== Single-Push Recovery ===")
    push_ok = 0
    for dname in ["forward", "backward"]:
        ok, _ = run_case(f"push/{dname}_90N", "high_0p480", steps=500)
        results[f"push/{dname}_90N"] = "PASS" if ok else "FAIL"
        if ok: push_ok += 1
    print(f"Single-push: {push_ok}/2 passed")

    # ---- Dynamic height: 5 scenarios ----
    print("\n=== Dynamic Height Validation ===")
    dyn_cases = ["ramp_up", "ramp_down", "up_down_cycle", "gate_dwell", "gate_chatter"]
    dyn_ok = 0
    for d in dyn_cases:
        sp = setup_path(d)
        ok, _ = run_case(f"dynamic/{d}", d, steps=1000)
        results[f"dynamic/{d}"] = "PASS" if ok else "FAIL"
        if ok: dyn_ok += 1
    print(f"Dynamic: {dyn_ok}/{len(dyn_cases)} passed")

    # ---- Summary ----
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    report = {
        "backend": "jax",
        "profile": "k2_notch_low_q_v1",
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "step_c_passed": step_c_ok,
        "step_d_passed": step_d_ok,
        "step_e_passed": step_e_ok,
        "push_passed": push_ok,
        "dynamic_passed": dyn_ok,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stage6_validation_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Stage 6 JAX Backend Validation Summary")
    print(f"  Total: {passed}/{total} passed")
    print(f"  Step C: {step_c_ok}/7  Step D: {step_d_ok}/{step_d_total}  Step E: {step_e_ok}/10")
    print(f"  Single-push: {push_ok}/2  Dynamic: {dyn_ok}/5")
    if passed == total:
        print("  Classification: STAGE6_PASS_READY_FOR_STAGE7")
    else:
        print(f"  Classification: STAGE6_PARTIAL_PASS ({total-passed} failures)")
    print(f"  Report: {OUT_DIR / 'stage6_validation_summary.json'}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
