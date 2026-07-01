#!/usr/bin/env python3
"""Run Phase 3 instrumented scenarios — worst cases + controls with full telemetry."""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = str(ROOT / "scripts" / "run_k2_jax_realtime.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_DIR = ROOT / "outputs" / "k2_phase3_instrumented"

SCENARIOS = [
    # Worst cases (SAFE_BUT_WORSE due to pitch_rms)
    {"name": "low_0p380", "setup": "low_0p380_setup.json", "steps": 2000},
    {"name": "high_0p450", "setup": "high_0p450_setup.json", "steps": 2000},
    {"name": "focused_low_0p320", "setup": "low_0p320_setup.json", "steps": 2000},
    # Dynamic worst cases
    {"name": "gate_dwell", "setup": "high_0p480_setup.json",
     "trajectory": "outputs/k2_dynamic_height_gate_crossing/trajectories/gate_dwell_0p420_0p450_0p480.json",
     "steps": 6000, "qref_mode": "original-k2-exact"},
    {"name": "ramp_down", "setup": "high_0p480_setup.json",
     "trajectory": "outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_down_0p480_to_0p330.json",
     "steps": 5000, "qref_mode": "original-k2-exact"},
    # Control (PASSING)
    {"name": "high_0p430", "setup": "high_0p430_setup.json", "steps": 2000},
]

RESULTS = {}

for sc in SCENARIOS:
    out_dir = OUT_DIR / sc["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / sc["setup"]
    args = [
        sys.executable, RUNNER,
        "--height-setup", str(setup_path),
        "--steps", str(sc["steps"]),
        "--telemetry", "full",
        "--quiet",
        "--output-dir", str(out_dir),
    ]
    if "trajectory" in sc:
        args += ["--dynamic-height-trajectory", sc["trajectory"]]
        if "qref_mode" in sc:
            args += ["--dynamic-qref-mode", sc["qref_mode"]]

    print(f"  {sc['name']} ({sc['steps']} steps) ... ", end="", flush=True)
    result = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    fell = "[FALL]" in result.stdout
    status = "FALL" if fell else ("OK" if result.returncode == 0 else f"ERR({result.returncode})")
    RESULTS[sc["name"]] = {"status": status, "returncode": result.returncode}
    print(status)
    if result.returncode != 0:
        print(f"    stderr: {result.stderr[:200]}")

print(f"\n{'='*50}")
print("RESULTS:")
for name, r in RESULTS.items():
    print(f"  {name}: {r['status']}")
