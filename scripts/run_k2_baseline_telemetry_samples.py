#!/usr/bin/env python3
"""Run key K2 scenarios with full telemetry for rich quality analysis."""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = str(ROOT / "scripts" / "run_k2_jax_realtime.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_DIR = ROOT / "outputs" / "k2_improvement_baseline_telemetry"

SCENARIOS = [
    # Fixed-height: one per height region
    {"name": "step_e/low_0p320", "setup": "low_0p320_setup.json", "steps": 2000},
    {"name": "step_e/low_0p360", "setup": "low_0p360_setup.json", "steps": 2000},
    {"name": "step_e/high_0p450", "setup": "high_0p450_setup.json", "steps": 2000},
    {"name": "step_e/high_0p480", "setup": "high_0p480_setup.json", "steps": 2000},
    # Push: one representative
    {"name": "step_d/push_fwd_60N_mid", "setup": "mid_0p400_setup.json",
     "push": "outputs/k2_step_d_push_matrix_validation/push_sequences/push_sagittal_forward_60N_step300.json",
     "steps": 2000},
    {"name": "step_d/push_bwd_90N_low", "setup": "low_0p330_setup.json",
     "push": "outputs/k2_step_d_push_matrix_validation/push_sequences/push_sagittal_backward_90N_step300.json",
     "steps": 2000},
    # Dynamic height
    {"name": "dynamic/ramp_up", "setup": "low_0p330_setup.json",
     "trajectory": "outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json",
     "steps": 5000, "qref_mode": "setup-interp-debug"},
    {"name": "dynamic/ramp_down", "setup": "high_0p480_setup.json",
     "trajectory": "outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_down_0p480_to_0p330.json",
     "steps": 5000, "qref_mode": "original-k2-exact"},
    # Long run
    {"name": "long_run/mid_0p400", "setup": "mid_0p400_setup.json", "steps": 6000},
]

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
    if "push" in sc:
        args += ["--push-seq", sc["push"]]
    if "trajectory" in sc:
        args += ["--dynamic-height-trajectory", sc["trajectory"]]
        if "qref_mode" in sc:
            args += ["--dynamic-qref-mode", sc["qref_mode"]]

    print(f"Running {sc['name']} ({sc['steps']} steps, --telemetry full) ... ", end="", flush=True)
    result = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    fell = "[FALL]" in result.stdout
    status = "FALL" if fell else ("OK" if result.returncode == 0 else f"ERR({result.returncode})")
    print(status)
