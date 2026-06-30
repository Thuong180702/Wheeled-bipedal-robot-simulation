#!/usr/bin/env python3
"""K2 Both-Synced Full-Sim Parity Harness.

Runs the Python monolithic K2 controller and the JAX dedicated controller in a
synchronized full-simulation setting, exporting detailed per-step JSONL traces
of every control-affecting scalar for divergence analysis.

Modes:
  source-python       Run simulate_hierarchical_controller.py with python backend
  source-jax-mono     Run simulate_hierarchical_controller.py --controller-backend jax
  dedicated-jax       Run run_k2_jax_realtime.py standalone
  both-synced         Run simulate_hierarchical_controller.py --controller-backend both-synced
                      with structured JSONL trace export

Both-synced sub-experiments:
  A  Same source state → source controller vs dedicated controller (default)
  B  Same dedicated state → source controller vs dedicated controller
  D  Dedicated controller with source state reset every step
  E  Source physics with dedicated torque sequence
  F  Dedicated physics with source torque sequence

Usage:
  # Step E fixed-height trace (both-synced, experiment A)
  python scripts/trace_k2_both_synced_fullsim_parity.py \\
    --mode both-synced --scenario step_e --height low_0p380 --steps 200

  # Source-only trace
  python scripts/trace_k2_both_synced_fullsim_parity.py \\
    --mode source-python --scenario step_e --height low_0p380 --steps 200

  # Dedicated-only trace
  python scripts/trace_k2_both_synced_fullsim_parity.py \\
    --mode dedicated-jax --scenario step_e --height low_0p380 --steps 200

  # Both-synced experiment D (state reset each step)
  python scripts/trace_k2_both_synced_fullsim_parity.py \\
    --mode both-synced --experiment D --scenario step_e --height low_0p380 --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
TRACE_DIR = ROOT / "outputs" / "k2_both_synced_traces"
CONTROL_DT = 0.01


# =============================================================================
# Scenario definitions
# =============================================================================

STEP_E_SCENARIOS = {
    "low_0p300": {"setup": "low_0p300_setup.json", "steps": 2000, "height_ref": 0.300},
    "low_0p320": {"setup": "low_0p320_setup.json", "steps": 2000, "height_ref": 0.320},
    "low_0p330": {"setup": "low_0p330_setup.json", "steps": 2000, "height_ref": 0.330},
    "low_0p340": {"setup": "low_0p340_setup.json", "steps": 2000, "height_ref": 0.340},
    "low_0p360": {"setup": "low_0p360_setup.json", "steps": 2000, "height_ref": 0.360},
    "low_0p380": {"setup": "low_0p380_setup.json", "steps": 2000, "height_ref": 0.380},
    "high_0p430": {"setup": "high_0p430_setup.json", "steps": 2000, "height_ref": 0.430},
    "high_0p450": {"setup": "high_0p450_setup.json", "steps": 2000, "height_ref": 0.450},
    "high_0p465": {"setup": "high_0p465_setup.json", "steps": 2000, "height_ref": 0.465},
    "high_0p480": {"setup": "high_0p480_setup.json", "steps": 2000, "height_ref": 0.480},
}

DYNAMIC_SCENARIOS = {
    "ramp_up": {"setup": "low_0p330_setup.json", "trajectory": "ramp_up_0p330_to_0p480.json", "steps": 5000},
    "ramp_down": {"setup": "high_0p480_setup.json", "trajectory": "ramp_down_0p480_to_0p330.json", "steps": 5000},
    "gate_dwell": {"setup": "high_0p480_setup.json", "trajectory": "gate_dwell_0p420_0p450_0p480.json", "steps": 6000},
    "gate_chatter": {"setup": "high_0p480_setup.json", "trajectory": "gate_chatter_0p400_0p470.json", "steps": 6000},
    "up_down_cycle": {"setup": "low_0p330_setup.json", "trajectory": "up_down_cycle_0p330_0p480_0p330.json", "steps": 6000},
}


# =============================================================================
# Mode 1: source-python — simulate_hierarchical_controller.py with python backend
# =============================================================================

def run_source_python(
    scenario: str,
    height_label: str,
    steps: int,
    output_dir: Optional[Path] = None,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> Optional[Path]:
    """Run source-of-truth Python monolithic K2 and return telemetry CSV path."""
    if output_dir is None:
        output_dir = TRACE_DIR / "source_python" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / f"{height_label}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--controller-backend", "python",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(output_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
    ]

    if push_seq:
        push_path = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences" / push_seq
        if push_path.exists():
            cmd += ["--push-sequence-file", str(push_path)]

    if trajectory:
        traj_path = ROOT / "outputs" / "physical_target_height_trajectories" / trajectory
        if traj_path.exists():
            cmd += ["--height-trajectory-file", str(traj_path)]

    print(f"  [source-python] Running: simulate_hierarchical_controller.py ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print("  [source-python] TIMEOUT")
        return None

    elapsed = time.time() - t0
    print(f"  [source-python] Completed in {elapsed:.0f}s (rc={result.returncode})")

    if result.returncode != 0:
        print(f"  [source-python] STDERR (last 50 lines):")
        for line in result.stderr.splitlines()[-50:]:
            print(f"    {line}")
        return None

    # Save stdout for diagnostics
    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    # Find telemetry
    tel_files = sorted(output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tel_files:
        print(f"  [source-python] ERROR: No telemetry CSV found in {output_dir}")
        return None

    return tel_files[0]


# =============================================================================
# Mode 2: source-jax-monolithic — simulate_hierarchical_controller.py --controller-backend jax
# =============================================================================

def run_source_jax_monolithic(
    scenario: str,
    height_label: str,
    steps: int,
    output_dir: Optional[Path] = None,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> Optional[Path]:
    """Run JAX monolithic K2 via simulate_hierarchical_controller.py --controller-backend jax."""
    if output_dir is None:
        output_dir = TRACE_DIR / "source_jax_mono" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / f"{height_label}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--controller-backend", "jax",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(output_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
        "--quiet",
    ]

    if push_seq:
        push_path = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences" / push_seq
        if push_path.exists():
            cmd += ["--push-sequence-file", str(push_path)]

    if trajectory:
        traj_path = ROOT / "outputs" / "physical_target_height_trajectories" / trajectory
        if traj_path.exists():
            cmd += ["--height-trajectory-file", str(traj_path)]

    print(f"  [source-jax-mono] Running: simulate_hierarchical_controller.py --controller-backend jax ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print("  [source-jax-mono] TIMEOUT")
        return None

    elapsed = time.time() - t0
    print(f"  [source-jax-mono] Completed in {elapsed:.0f}s (rc={result.returncode})")

    if result.returncode != 0:
        print(f"  [source-jax-mono] STDERR (last 50 lines):")
        for line in result.stderr.splitlines()[-50:]:
            print(f"    {line}")
        return None

    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    tel_files = sorted(output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tel_files:
        print(f"  [source-jax-mono] ERROR: No telemetry CSV found in {output_dir}")
        return None

    return tel_files[0]


# =============================================================================
# Mode 3: dedicated-jax — run_k2_jax_realtime.py
# =============================================================================

def run_dedicated_jax(
    scenario: str,
    height_label: str,
    steps: int,
    output_dir: Optional[Path] = None,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> Optional[Path]:
    """Run dedicated JAX K2 runner and return telemetry CSV path."""
    if output_dir is None:
        output_dir = TRACE_DIR / "dedicated_jax" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / f"{height_label}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_k2_jax_realtime.py"),
        "--height-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry", "full",
        "--output-dir", str(output_dir),
        "--quiet",
    ]

    if push_seq:
        push_path = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences" / push_seq
        if push_path.exists():
            cmd += ["--push-seq", str(push_path)]

    if trajectory:
        traj_path = ROOT / "outputs" / "physical_target_height_trajectories" / trajectory
        if traj_path.exists():
            cmd += ["--height-trajectory", str(traj_path)]

    print(f"  [dedicated-jax] Running: run_k2_jax_realtime.py ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print("  [dedicated-jax] TIMEOUT")
        return None

    elapsed = time.time() - t0
    print(f"  [dedicated-jax] Completed in {elapsed:.0f}s (rc={result.returncode})")

    if result.returncode != 0:
        print(f"  [dedicated-jax] STDERR (last 50 lines):")
        for line in result.stderr.splitlines()[-50:]:
            print(f"    {line}")
        return None

    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    tel_files = sorted(output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tel_files:
        print(f"  [dedicated-jax] ERROR: No telemetry CSV found in {output_dir}")
        return None

    return tel_files[0]


# =============================================================================
# Mode 4: both-synced — state-synced comparison via simulate_hierarchical_controller.py
# =============================================================================

def run_both_synced(
    scenario: str,
    height_label: str,
    steps: int,
    output_dir: Optional[Path] = None,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
    experiment: str = "A",
) -> Optional[Dict]:
    """Run both-synced comparison and return structured results.

    Uses simulate_hierarchical_controller.py --controller-backend both-synced
    which captures Python K2 state before each step, syncs it to JAX, and
    compares torque outputs step-by-step.

    Returns dict with:
      - telemetry_path: Path to telemetry CSV
      - stdout_path: Path to captured stdout
      - max_abs_diff: float
      - max_diff_step: int
      - max_diff_actuator: int
      - classification: str
      - parsed_diagnostics: list of dicts (one per synced print step)
    """
    if output_dir is None:
        suffix = f"_exp{experiment}" if experiment != "A" else ""
        output_dir = TRACE_DIR / f"both_synced{suffix}" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / f"{height_label}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--controller-backend", "both-synced",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(output_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
    ]

    if push_seq:
        push_path = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences" / push_seq
        if push_path.exists():
            cmd += ["--push-sequence-file", str(push_path)]

    if trajectory:
        traj_path = ROOT / "outputs" / "physical_target_height_trajectories" / trajectory
        if traj_path.exists():
            cmd += ["--height-trajectory-file", str(traj_path)]

    print(f"  [both-synced] Running: simulate_hierarchical_controller.py --controller-backend both-synced ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print("  [both-synced] TIMEOUT")
        return None

    elapsed = time.time() - t0
    print(f"  [both-synced] Completed in {elapsed:.0f}s (rc={result.returncode})")

    # Save full stdout/stderr
    stdout_path = output_dir / "stdout.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    # Parse the synced diagnostics from stdout
    parsed = _parse_synced_stdout(result.stdout)

    # Find telemetry
    tel_files = sorted(output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    telemetry_path = tel_files[0] if tel_files else None

    # Extract final classification from stdout
    classification = "UNKNOWN"
    max_abs_diff = float("inf")
    max_diff_step = -1
    max_diff_actuator = -1
    for line in result.stdout.splitlines():
        if "Worst max_abs_diff:" in line:
            try:
                max_abs_diff = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "at step" in line and "actuator index" in line:
            parts = line.split(",")
            try:
                max_diff_step = int(parts[0].split("step")[-1].strip())
                max_diff_actuator = int(parts[1].split("index")[-1].strip())
            except (ValueError, IndexError):
                pass
        if "Classification:" in line:
            classification = line.split(":")[-1].strip()

    return {
        "telemetry_path": str(telemetry_path) if telemetry_path else None,
        "stdout_path": str(stdout_path),
        "max_abs_diff": max_abs_diff,
        "max_diff_step": max_diff_step,
        "max_diff_actuator": max_diff_actuator,
        "classification": classification,
        "parsed_diagnostics": parsed,
        "n_diag_steps": len(parsed),
    }


def _parse_synced_stdout(stdout: str) -> List[Dict]:
    """Parse both-synced stdout into structured per-step diagnostic dicts.

    Extracts [SYNCED@N] lines with torque vectors, state snapshots, and
    per-component comparisons.
    """
    diagnostics = []
    current = None

    for line in stdout.splitlines():
        if line.startswith("[SYNCED@"):
            if current is not None:
                diagnostics.append(current)
            # Parse step number
            try:
                step_str = line.split("[SYNCED@")[1].split("]")[0]
                step = int(step_str)
            except (ValueError, IndexError):
                step = -1

            # Parse max_abs_diff, first_divergent_idx, val
            current = {"step": step, "raw": [line]}
            parts = line.split()
            for i, p in enumerate(parts):
                if p.startswith("max_abs_diff="):
                    try:
                        current["max_abs_diff"] = float(p.split("=")[1])
                    except ValueError:
                        pass
                if p.startswith("first_divergent_idx="):
                    try:
                        current["first_divergent_idx"] = int(p.split("=")[1])
                    except ValueError:
                        pass
                if p.startswith("val="):
                    try:
                        current["first_divergent_val"] = float(p.split("=")[1])
                    except ValueError:
                        pass

        elif current is not None:
            current["raw"].append(line)

            # Parse structured lines
            if line.startswith("  PY_tau="):
                _parse_vector(line, current, "py_tau")
            elif line.startswith("  JX_tau="):
                _parse_vector(line, current, "jx_tau")
            elif line.startswith("  DIFF=   "):
                _parse_vector(line, current, "tau_diff")
            elif line.startswith("  PY_STATE: notch="):
                current["py_notch_state"] = line.split("notch=")[1].split(" filt_com_z")[0].strip()
            elif "filt_com_z=" in line and "PY_STATE:" in line:
                for token in line.split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        current[f"py_{k}"] = _try_float(v)
            elif line.startswith("  INPUT:"):
                for token in line.replace("  INPUT:", "").split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        current[f"input_{k}"] = _try_float(v)
            elif line.startswith("  SV:"):
                for token in line.replace("  SV:", "").split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        current[f"sv_{k}"] = _try_float(v)
            elif line.startswith("  SAG_TERMS:"):
                _parse_key_value_pairs(line.replace("  SAG_TERMS:", ""), current, "sag")
            elif line.startswith("  MODE_DIV:"):
                for token in line.replace("  MODE_DIV:", "").split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        current[f"md_{k}"] = _try_float(v)
            elif line.startswith("  APCR1ND:"):
                for token in line.replace("  APCR1ND:", "").split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        current[f"apcr1nd_{k}"] = _try_float(v)
            elif line.startswith("  APCR1ND_JX:"):
                for token in line.replace("  APCR1ND_JX:", "").split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        current[f"apcr1nd_jx_{k}"] = _try_float(v)
            elif line.startswith("  TAU_COMP@"):
                _parse_tau_comp_line(line, current)
            elif line.startswith("    PY:"):
                _parse_key_value_pairs(line.replace("    PY:", ""), current, "py_tau_comp")
            elif line.startswith("    JX:"):
                _parse_key_value_pairs(line.replace("    JX:", ""), current, "jx_tau_comp")
            elif line.startswith("    ABS:"):
                _parse_key_value_pairs(line.replace("    ABS:", ""), current, "abs_state")
            elif line.startswith("    JX_sag:"):
                _parse_key_value_pairs(line.replace("    JX_sag:", ""), current, "jx_sag")
            elif line.startswith("    FINAL:"):
                _parse_key_value_pairs(line.replace("    FINAL:", ""), current, "jx_final")
            elif line.startswith("    JX_IN:"):
                _parse_key_value_pairs(line.replace("    JX_IN:", ""), current, "jx_input")
            elif line.startswith("    RING_BUF"):
                _parse_ring_buf_line(line, current)

    if current is not None:
        diagnostics.append(current)

    return diagnostics


def _parse_vector(line: str, target: Dict, key: str):
    """Parse a vector like PY_tau=[1.0, 2.0, ...] from a line."""
    try:
        vec_str = line.split("[")[1].split("]")[0]
        target[key] = [float(x.strip()) for x in vec_str.split(",")]
    except (ValueError, IndexError):
        pass


def _parse_key_value_pairs(text: str, target: Dict, prefix: str):
    """Parse key=value pairs from text like 'tau_p=1.0 tau_pr=2.0 ...'."""
    for token in text.strip().split():
        token = token.strip().rstrip(",")
        if "=" in token:
            k, v = token.split("=", 1)
            target[f"{prefix}_{k}"] = _try_float(v)


def _parse_tau_comp_line(line: str, target: Dict):
    """Parse TAU_COMP@N: max_diff=X act[Y]."""
    for token in line.replace("  TAU_COMP@", "").split():
        if "=" in token:
            k, v = token.split("=", 1)
            target[f"tau_comp_{k}"] = _try_float(v)
        elif token.startswith("act["):
            try:
                target["tau_comp_actuator"] = int(token.split("[")[1].split("]")[0])
            except (ValueError, IndexError):
                pass


def _parse_ring_buf_line(line: str, target: Dict):
    """Parse RING_BUF[N]: key=value pairs."""
    for token in line.split("]:", 1)[-1].strip().split():
        if "=" in token:
            k, v = token.split("=", 1)
            target[f"rb_{k}"] = _try_float(v)


def _try_float(s: str) -> Any:
    """Try to convert string to float, return string if fails."""
    try:
        return float(s)
    except ValueError:
        return s


# =============================================================================
# Combined harness runner
# =============================================================================

def run_harness(
    modes: List[str],
    scenario: str,
    height_label: str,
    steps: int,
    output_dir: Optional[Path] = None,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
    experiment: str = "A",
) -> Dict[str, Any]:
    """Run specified modes and return results dict keyed by mode name."""
    if output_dir is None:
        output_dir = TRACE_DIR / "combined" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    scenario_info = {
        "scenario": scenario,
        "height_label": height_label,
        "steps": steps,
        "push_seq": push_seq,
        "trajectory": trajectory,
        "experiment": experiment,
    }

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        try:
            if mode == "source-python":
                tel_path = run_source_python(scenario, height_label, steps, mode_dir, push_seq, trajectory)
                results[mode] = {"telemetry_path": str(tel_path) if tel_path else None}
            elif mode == "source-jax-mono":
                tel_path = run_source_jax_monolithic(scenario, height_label, steps, mode_dir, push_seq, trajectory)
                results[mode] = {"telemetry_path": str(tel_path) if tel_path else None}
            elif mode == "dedicated-jax":
                tel_path = run_dedicated_jax(scenario, height_label, steps, mode_dir, push_seq, trajectory)
                results[mode] = {"telemetry_path": str(tel_path) if tel_path else None}
            elif mode == "both-synced":
                synced_result = run_both_synced(scenario, height_label, steps, mode_dir, push_seq, trajectory, experiment)
                results[mode] = synced_result
            else:
                print(f"  Unknown mode: {mode}")
        except Exception as e:
            print(f"  ERROR in {mode}: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = {"error": str(e)}

    # Write harness manifest
    manifest = {
        "scenario_info": scenario_info,
        "modes_run": modes,
        "results_summary": {},
    }
    for mode, r in results.items():
        if isinstance(r, dict):
            summary = {}
            if "telemetry_path" in r:
                summary["telemetry"] = r["telemetry_path"]
            if "max_abs_diff" in r:
                summary["max_abs_diff"] = r["max_abs_diff"]
            if "classification" in r:
                summary["classification"] = r["classification"]
            if "n_diag_steps" in r:
                summary["n_diag_steps"] = r["n_diag_steps"]
            if "error" in r:
                summary["error"] = r["error"]
            manifest["results_summary"][mode] = summary

    manifest_path = output_dir / "harness_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")

    return results


# =============================================================================
# Trace comparison (cross-mode)
# =============================================================================

def compare_mode_telemetry(
    source_path: Path,
    dedicated_path: Path,
    output_path: Optional[Path] = None,
) -> Dict:
    """Compare telemetry CSVs from two modes, report first divergence per field."""
    import csv

    with open(source_path, encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    with open(dedicated_path, encoding="utf-8") as f:
        dedicated_rows = list(csv.DictReader(f))

    # Align by step
    src_by_step = {}
    for row in source_rows:
        try:
            step = int(float(row.get("step", row.get("Step", -1))))
            if step >= 0:
                src_by_step[step] = row
        except (ValueError, TypeError):
            continue

    ded_by_step = {}
    for row in dedicated_rows:
        try:
            step = int(float(row.get("step", -1)))
            if step >= 0:
                ded_by_step[step] = row
        except (ValueError, TypeError):
            continue

    common_steps = sorted(set(src_by_step) & set(ded_by_step))
    if not common_steps:
        return {"error": "No common steps", "source_steps": len(src_by_step), "dedicated_steps": len(ded_by_step)}

    # Common numeric fields to compare
    common_fields = set(src_by_step[common_steps[0]].keys()) & set(ded_by_step[common_steps[0]].keys())
    # Filter to numeric fields
    numeric_fields = []
    for field in common_fields:
        try:
            float(src_by_step[common_steps[0]][field])
            float(ded_by_step[common_steps[0]][field])
            numeric_fields.append(field)
        except (ValueError, TypeError):
            continue

    divergences = []
    for field in sorted(numeric_fields):
        max_delta = 0.0
        max_delta_step = -1
        src_at_max = 0.0
        ded_at_max = 0.0

        for step in common_steps:
            try:
                sv = float(src_by_step[step].get(field, 0))
                dv = float(ded_by_step[step].get(field, 0))
            except (ValueError, TypeError):
                continue
            delta = abs(sv - dv)
            if delta > max_delta:
                max_delta = delta
                max_delta_step = step
                src_at_max = sv
                ded_at_max = dv

        if max_delta > 1e-12:
            divergences.append({
                "field": field,
                "max_delta": max_delta,
                "step": max_delta_step,
                "source_value": src_at_max,
                "dedicated_value": ded_at_max,
            })

    divergences.sort(key=lambda d: d["max_delta"], reverse=True)

    result = {
        "n_common_steps": len(common_steps),
        "n_divergent_fields": len(divergences),
        "divergences": divergences[:100],  # top 100
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="K2 Both-Synced Full-Sim Parity Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", nargs="+",
                        choices=["source-python", "source-jax-mono", "dedicated-jax", "both-synced", "all"],
                        default=["both-synced"],
                        help="Mode(s) to run. 'all' runs all four modes.")
    parser.add_argument("--scenario", default="step_e",
                        choices=["step_e", "step_c", "step_d", "dynamic"],
                        help="Scenario type")
    parser.add_argument("--height", default="low_0p380",
                        help="Height label (e.g., low_0p300, high_0p480)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of simulation steps")
    parser.add_argument("--push", help="Push sequence filename")
    parser.add_argument("--trajectory", help="Height trajectory name (for dynamic scenarios)")
    parser.add_argument("--experiment", default="A",
                        choices=["A", "B", "D", "E", "F"],
                        help="Both-synced sub-experiment (A/B/D/E/F)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for traces")
    parser.add_argument("--compare", nargs=2, metavar=("SOURCE_CSV", "DEDICATED_CSV"),
                        help="Compare two existing telemetry CSVs")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch of failing and passing cases")
    parser.add_argument("--batch-scope", default="step_e_failing",
                        choices=["step_e_failing", "step_e_passing", "step_e_all", "dynamic", "all"],
                        help="Which cases to run in batch mode")

    args = parser.parse_args()

    # Compare-only mode
    if args.compare:
        print("Compare-only mode")
        result = compare_mode_telemetry(Path(args.compare[0]), Path(args.compare[1]),
                                        Path(args.output_dir) / "comparison.json" if args.output_dir else None)
        print(json.dumps(result, indent=2, default=str))
        return

    # Batch mode
    if args.batch:
        _run_batch(args)
        return

    # Single-run mode
    modes = ["source-python", "source-jax-mono", "dedicated-jax", "both-synced"] if "all" in args.mode else args.mode
    output_dir = Path(args.output_dir) if args.output_dir else None

    results = run_harness(
        modes=modes,
        scenario=args.scenario,
        height_label=args.height,
        steps=args.steps,
        output_dir=output_dir,
        push_seq=args.push,
        trajectory=args.trajectory,
        experiment=args.experiment,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("HARNESS SUMMARY")
    print(f"{'='*60}")
    for mode, r in results.items():
        if isinstance(r, dict):
            status = "ERROR" if "error" in r else "OK"
            extras = ""
            if "max_abs_diff" in r and r["max_abs_diff"] != float("inf"):
                extras += f" max_diff={r['max_abs_diff']:.2e}"
            if "classification" in r:
                extras += f" class={r['classification']}"
            print(f"  {mode}: {status}{extras}")


def _run_batch(args):
    """Run harness in batch mode across multiple scenarios."""
    batch_scenarios = []

    if args.batch_scope == "step_e_failing":
        batch_scenarios = [
            ("step_e", "low_0p320", 2000),
            ("step_e", "low_0p360", 2000),
            ("step_e", "low_0p380", 2000),
            ("step_e", "high_0p450", 2000),
        ]
    elif args.batch_scope == "step_e_passing":
        batch_scenarios = [
            ("step_e", "low_0p300", 2000),
            ("step_e", "low_0p330", 2000),
            ("step_e", "low_0p340", 2000),
            ("step_e", "high_0p430", 2000),
            ("step_e", "high_0p465", 2000),
            ("step_e", "high_0p480", 2000),
        ]
    elif args.batch_scope == "step_e_all":
        batch_scenarios = [(k, v["height_ref"], v["steps"]) for k, v in STEP_E_SCENARIOS.items()]
        batch_scenarios = [(args.scenario, h, s) for args.scenario, h, s in batch_scenarios]
    elif args.batch_scope == "dynamic":
        batch_scenarios = [
            ("dynamic", "gate_dwell", 1000),
            ("dynamic", "gate_chatter", 1000),
            ("dynamic", "up_down_cycle", 1000),
        ]

    modes = ["source-python", "source-jax-mono", "dedicated-jax", "both-synced"] if "all" in args.mode else args.mode
    all_results = {}

    for scenario, height, steps in batch_scenarios:
        print(f"\n{'#'*60}")
        print(f"BATCH: {scenario}/{height} ({steps} steps)")
        print(f"{'#'*60}")

        output_dir = TRACE_DIR / "batch" / scenario / height
        results = run_harness(
            modes=modes,
            scenario=scenario,
            height_label=height,
            steps=steps,
            output_dir=output_dir,
            experiment=args.experiment,
        )
        all_results[f"{scenario}/{height}"] = results

    # Batch summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    for key, results in all_results.items():
        both = results.get("both-synced", {})
        if isinstance(both, dict):
            diff = both.get("max_abs_diff", "N/A")
            cls = both.get("classification", "N/A")
            print(f"  {key}: max_diff={diff} class={cls}")


if __name__ == "__main__":
    main()
