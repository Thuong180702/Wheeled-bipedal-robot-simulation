"""K2 Python→JAX Correctness Audit — Temporary Diagnostic Instrumentation.

Runs the simulation with --controller-backend both for each scenario,
capturing the detailed [BOTH@step] output.

Usage:
  python scripts/_k2_correctness_audit_instrument.py --scenario push_fwd_90N --steps 25
  python scripts/_k2_correctness_audit_instrument.py --all-scenarios --steps 25
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

SETUP_DIR = _project_root / "outputs" / "physical_target_height_setups_centered"
TRAJ_DIR = _project_root / "outputs" / "k2_dynamic_height_gate_crossing" / "trajectories"

SCENARIOS = {
    "fixed_high_0p480": {
        "height_setup": str(SETUP_DIR / "high_0p480_setup.json"),
        "description": "Fixed high height 0.48m, notch fully active",
        "extra_args": [],
    },
    "ramp_down": {
        "height_setup": str(SETUP_DIR / "high_0p480_setup.json"),
        "description": "Dynamic descending height 0.48->0.33m",
        "extra_args": ["--dynamic-height-trajectory", str(TRAJ_DIR / "ramp_down_0p480_to_0p330.json")],
    },
    "push_fwd_90N": {
        "height_setup": str(SETUP_DIR / "mid_0p400_setup.json"),
        "description": "Forward push 90N at 0.40m",
        "extra_args": ["--push-enabled", "--push-magnitude-n", "90", "--sagittal-push-only"],
    },
    "push_bwd_90N": {
        "height_setup": str(SETUP_DIR / "mid_0p400_setup.json"),
        "description": "Backward push -90N at 0.40m",
        "extra_args": ["--push-enabled", "--push-magnitude-n", "-90", "--sagittal-push-only"],
    },
    "ramp_up": {
        "height_setup": str(SETUP_DIR / "low_0p330_setup.json"),
        "description": "Dynamic ascending height 0.33->0.48m",
        "extra_args": ["--dynamic-height-trajectory", str(TRAJ_DIR / "ramp_up_0p330_to_0p480.json")],
    },
    "gate_chatter": {
        "height_setup": str(SETUP_DIR / "low_0p330_setup.json"),
        "description": "Notch gate boundary oscillation 0.40-0.47m",
        "extra_args": ["--dynamic-height-trajectory", str(TRAJ_DIR / "gate_chatter_0p400_0p470.json")],
    },
}


def run_scenario(scenario_name: str, config: dict, output_dir: Path, num_steps: int = 25):
    """Run a single scenario, capture [BOTH@step] output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_project_root / "scripts" / "simulate_hierarchical_controller.py"),
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--sagittal-controller", "velocity-damped",
        "--controller-backend", "both",
        "--controller-mode", "balance-core",
        "--height-variant-setup", config["height_setup"],
        "--steps", str(num_steps),
        "--enable-mode-hip-yaw-divergence",
        "--output-dir", str(output_dir),
    ] + config["extra_args"]

    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name} — {config['description']}")
    print(f"{'='*80}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

    stdout = result.stdout
    stderr = result.stderr

    # Extract all [BOTH@...] lines
    both_lines = [l.strip() for l in stdout.split('\n') if '[BOTH@' in l]
    print(f"Captured {len(both_lines)} [BOTH@step] lines")

    if not both_lines:
        print("NO [BOTH@step] OUTPUT!")
        # Check for errors
        for line in stderr.split('\n'):
            if 'Error' in line or 'Traceback' in line or 'error' in line:
                print(f"  STDERR: {line[:300]}")
        for line in stdout.split('\n'):
            if 'JAX BACKEND' in line or 'Error' in line or 'Step' in line:
                print(f"  STDOUT: {line[:300]}")
        return None, []

    if result.returncode != 0:
        print(f"Non-zero exit: {result.returncode}")
        print(f"STDERR (last 1000 chars):\n{stderr[-1000:]}")

    # Print first few lines for inspection
    for line in both_lines[:3]:
        print(f"  {line[:250]}")

    # Parse into records
    records = []
    current_rec = {}

    for line in both_lines:
        if line.startswith('[BOTH@'):
            if current_rec and 'step' in current_rec:
                records.append(current_rec)
            current_rec = {}
            try:
                end = line.index(']')
                current_rec['step'] = int(line[7:end])
                rest = line[end + 1:].strip()
                for part in rest.split():
                    if '=' in part:
                        k, v = part.split('=', 1)
                        k = k.rstrip(',')
                        try:
                            current_rec[k] = float(v)
                        except ValueError:
                            current_rec[k] = v
            except (ValueError, IndexError):
                continue

        elif line.startswith('PY_tau='):
            try:
                vals = json.loads(line.split('=', 1)[1])
                for i, v in enumerate(vals):
                    current_rec[f'py_tau_{i}'] = v
            except Exception:
                pass

        elif line.startswith('JX_tau='):
            try:
                vals = json.loads(line.split('=', 1)[1])
                for i, v in enumerate(vals):
                    current_rec[f'jx_tau_{i}'] = v
            except Exception:
                pass

        elif ':' in line and '=' in line:
            try:
                sep = line.index(':')
                prefix = line[:sep].strip().lower().replace(' ', '_').replace('[', '_').replace(']', '_')
                rest = line[sep + 1:].strip()
                for part in rest.split():
                    if '=' in part:
                        k, v = part.split('=', 1)
                        k = k.rstrip(',')
                        try:
                            current_rec[f'{prefix}_{k}'] = float(v)
                        except ValueError:
                            current_rec[f'{prefix}_{k}'] = v
            except ValueError:
                pass

    if current_rec and 'step' in current_rec:
        records.append(current_rec)

    if not records:
        print("  WARNING: No records parsed!")
        return None, []

    # Write CSV
    csv_path = output_dir / f"teacher_forcing_{scenario_name}_steps0_{num_steps - 1}.csv"
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    fieldnames = ['step'] + sorted(k for k in all_keys if k != 'step')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"  Wrote {len(records)} rows -> {csv_path}")

    # Quick summary
    max_row = max(records, key=lambda r: r.get('max_tau_diff', 0))
    print(f"  Max diff: {max_row.get('max_tau_diff', 0):.6e} at step {max_row.get('step', '?')}")

    return csv_path, records


def main():
    parser = argparse.ArgumentParser(description="K2 Correctness Audit")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()))
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output-dir", type=str,
                        default=str(_project_root / "outputs" / "k2_jax_correctness_audit"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(SCENARIOS.keys()) if args.all_scenarios else ([args.scenario] if args.scenario else [])
    if not names:
        print("ERROR: Specify --scenario or --all-scenarios")
        sys.exit(1)

    results = {}
    for name in names:
        scenario_dir = output_dir / name
        csv_path, rows = run_scenario(name, SCENARIOS[name], scenario_dir, args.steps)
        results[name] = {"csv_path": str(csv_path) if csv_path else None, "num_rows": len(rows)}
        if name != names[-1]:
            time.sleep(1)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, info in results.items():
        s = f"{info['num_rows']} rows" if info['csv_path'] else "FAILED"
        print(f"  {name}: {s}")

    manifest_path = output_dir / "audit_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({"audit": "k2_jax_correctness", "date": time.strftime("%Y-%m-%d"),
                    "scenarios": results}, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
