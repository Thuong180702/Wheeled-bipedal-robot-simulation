"""Stage 6H: Full K2 JAX backend validation gate."""
import subprocess, sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_DIR = ROOT / "outputs" / "k2_jax_stage6h"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "jax",
    "--wbc-quiet",
]


def run_case(name, variant, steps=300, extra_args=None):
    sp = SETUP_DIR / f"{variant}_setup.json"
    if not sp.exists():
        return "SKIP"
    cmd = list(BASE) + [
        "--height-variant-setup", str(sp),
        "--steps", str(steps),
    ]
    if extra_args:
        cmd += extra_args
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    elapsed = time.time() - t0
    ok = "without falling" in r.stdout
    status = "PASS" if ok else "FAIL"
    print(f"  {name:30s} {status} ({elapsed:.0f}s)", flush=True)
    return status


results = {}

# Step C: 7 fixed heights
print("\n=== STEP C (7 heights) ===")
for h in ["low_0p330","low_0p340","low_0p360","low_0p380","mid_0p400","high_0p430","high_0p480"]:
    results[f"step_c/{h}"] = run_case(f"step_c/{h}", h)

# Step E: 10 heights
print("\n=== STEP E (10 heights) ===")
for h in ["low_0p300","low_0p320","low_0p330","low_0p340","low_0p360","low_0p380","high_0p430","high_0p450","high_0p465","high_0p480"]:
    results[f"step_e/{h}"] = run_case(f"step_e/{h}", h)

# Step D: Push matrix — 3 heights × 2 dirs × 90N
print("\n=== STEP D (push matrix) ===")
for h in ["low_0p330","mid_0p400","high_0p480"]:
    for d in ["forward","backward"]:
        results[f"step_d/{h}_{d}_90N"] = run_case(f"step_d/{h}_{d}_90N", h, steps=300)

# Single-push: high_0p480
print("\n=== SINGLE PUSH ===")
for d in ["forward","backward"]:
    results[f"push/high_0p480_{d}_90N"] = run_case(f"push/{d}_90N", "high_0p480", steps=300)

# Dynamic height
print("\n=== DYNAMIC HEIGHT ===")
for v in ["ramp_up","ramp_down","up_down_cycle","gate_dwell","gate_chatter"]:
    results[f"dynamic/{v}"] = run_case(f"dynamic/{v}", v, steps=500)

# Summary
total = len([v for v in results.values() if v != "SKIP"])
passed = sum(1 for v in results.values() if v == "PASS")
failed = sum(1 for v in results.values() if v == "FAIL")
skipped = sum(1 for v in results.values() if v == "SKIP")

summary = {
    "backend": "jax",
    "profile": "k2_notch_low_q_v1",
    "total": total,
    "passed": passed,
    "failed": failed,
    "skipped": skipped,
    "results": results,
}
with open(OUT_DIR / "stage6h_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"Stage 6H Summary: {passed}/{total} passed ({failed} failed, {skipped} skipped)")
if failed == 0 and passed == total:
    print("Classification: STAGE6H_PASS_READY_FOR_STAGE7")
else:
    print(f"Classification: STAGE6H_PARTIAL_PASS ({failed} failures)")
print(f"Report: {OUT_DIR / 'stage6h_summary.json'}")
