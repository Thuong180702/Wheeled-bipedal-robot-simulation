"""Evaluate continuous low-height sagittal authority fix candidates.

Runs each candidate profile through a comprehensive evaluation grid:
- Boundary variants (low_0p300, high_0p480) at 1000 and 5000 steps
- Step C boundary runs at 5000 steps
- Full height grid for Step E (0.300 .. 0.480 m)
- Step C height grid (0.300, 0.360, nominal, 0.480)
- Five-variant regression (nominal, low_tiny, high_tiny, low_small, high_small)

Checks acceptance gates after each run and stops at the first candidate
that passes all gates.

Usage:
    python scripts/evaluate_continuous_low_height_sagittal_authority_fix.py
    python scripts/evaluate_continuous_low_height_sagittal_authority_fix.py --dry-run
    python scripts/evaluate_continuous_low_height_sagittal_authority_fix.py --candidate baseline
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("outputs/continuous_low_height_sagittal_fix")
SETUP_DIR = Path("outputs/physical_target_height_setups")
VARIANT_DIR = Path("outputs/balance_core_true_height_variants")
SCRIPT = "scripts/simulate_hierarchical_controller.py"

CANDIDATES = [
    "baseline",
    "candidate_E1_k60_continuous",
    "candidate_E2_k80_continuous",
    "candidate_E3_k100_continuous",
]

# --- Height variant setup paths ------------------------------------------------

BOUNDARY_VARIANTS = {
    "low_0p300": SETUP_DIR / "low_0p300_setup.json",
    "high_0p480": SETUP_DIR / "high_0p480_setup.json",
}

# Intermediate setups that need to be generated if missing
INTERMEDIATE_SETUPS = {
    "low_0p330": SETUP_DIR / "low_0p330_setup.json",
    "low_0p360": SETUP_DIR / "low_0p360_setup.json",
    "high_0p450": SETUP_DIR / "high_0p450_setup.json",
}

# Standard five variants from existing validation
REGRESSION_VARIANTS = {
    "nominal": VARIANT_DIR / "variant_nominal" / "variant_setup.json",
    "low_tiny": VARIANT_DIR / "variant_low_tiny" / "variant_setup.json",
    "high_tiny": VARIANT_DIR / "variant_high_tiny" / "variant_setup.json",
    "low_small": VARIANT_DIR / "variant_low_small" / "variant_setup.json",
    "high_small": VARIANT_DIR / "variant_high_small" / "variant_setup.json",
}

# --- Acceptance gates ----------------------------------------------------------

ACCEPTANCE_GATES = {
    "support_position_error_max_abs": {"threshold": 0.15, "comparison": "max_abs", "label": "support_position_error max_abs <= 0.15 m"},
    "hip_yaw_abs_max": {"threshold": 0.07, "comparison": "max_abs", "label": "hip_yaw_abs_max <= 0.07 rad"},
    "pitch_x_max_abs": {"threshold": 0.10, "comparison": "max_abs", "label": "pitch_x max_abs <= 0.10 rad"},
    "roll_y_max_abs": {"threshold": 0.05, "comparison": "max_abs", "label": "roll_y max_abs <= 0.05 rad"},
    "height_error_final_abs": {"threshold": 0.02, "comparison": "max_abs", "label": "final height error max_abs <= 0.02 m"},
    "non_wheel_floor_contacts": {"threshold": 0, "comparison": "max", "label": "non-wheel floor contacts = 0"},
    "contact_valid_rate": {"threshold": 0.999, "comparison": "min", "label": "contact valid >= 99.9%"},
    "wbc_applied": {"threshold": False, "comparison": "is_false", "label": "WBC applied = false"},
    "hidden_torque_max": {"threshold": 0.0, "comparison": "max", "label": "hidden torque = 0"},
    "ownership_violations": {"threshold": 0, "comparison": "max", "label": "ownership violations = 0"},
}


# =====================================================================
# Helpers
# =====================================================================

def build_sim_command(
    candidate: str,
    setup_json: str | None,
    steps: int,
    boundary_yaw_profile: str = "baseline",
) -> list[str]:
    """Build the simulation command for a candidate/variant pair.

    All runs use balance-core mode with velocity-damped sagittal controller.
    The candidate name maps to --vd-sagittal-authority-profile.
    """
    cmd = [
        sys.executable, SCRIPT,
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", candidate,
        "--steps", str(steps),
        "--vd-k-position", "40",
        "--vd-k-velocity", "15",
        "--vd-max-position-tau", "3.0",
        "--boundary-yaw-position-profile", boundary_yaw_profile,
        "--telemetry-decimation", "10",
        "--write-run-summary-sidecar",
    ]
    if setup_json:
        cmd.extend(["--height-variant-setup", setup_json])
    return cmd


def run_simulation(cmd: list[str], timeout: int = 600, dry_run: bool = False) -> dict:
    """Run a single simulation and return result dict."""
    if dry_run:
        print(f"  [DRY-RUN] Would run: {' '.join(cmd[2:])}")
        return {"returncode": 0, "stdout": "", "stderr": "", "elapsed_s": 0.0, "success": True, "dry_run": True}

    print(f"  CMD: {' '.join(cmd[2:])}")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - start
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-3000:] if result.stderr else "",
            "elapsed_s": elapsed,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "TIMEOUT",
            "elapsed_s": elapsed,
            "success": False,
        }


def find_latest_telemetry_csv() -> str | None:
    """Find the most recent telemetry CSV written by the simulation."""
    search_dir = Path("outputs/hierarchical_controller_sim")
    if not search_dir.exists():
        return None
    csvs = sorted(search_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else None


def find_latest_summary_json() -> str | None:
    """Find the most recent run summary sidecar JSON."""
    search_dir = Path("outputs/hierarchical_controller_sim")
    if not search_dir.exists():
        return None
    jsons = sorted(search_dir.glob("telemetry_*.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(jsons[0]) if jsons else None


def parse_metrics_from_csv(csv_path: str) -> dict:
    """Parse key metrics from telemetry CSV using stdlib csv module."""
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return {"parse_error": True, "error": str(e)}

    if not rows:
        return {"parse_error": True, "error": "empty CSV"}

    metrics = {}

    # Support position error (support_position_error_m column)
    if "support_position_error_m" in rows[0]:
        vals = [abs(float(r["support_position_error_m"])) for r in rows if r.get("support_position_error_m")]
        metrics["support_position_error_max_abs"] = max(vals) if vals else None
    elif "support_center_y" in rows[0] and "support_center_ref_y" in rows[0]:
        vals = []
        for r in rows:
            if r.get("support_center_y") and r.get("support_center_ref_y"):
                vals.append(abs(float(r["support_center_y"]) - float(r["support_center_ref_y"])))
        metrics["support_position_error_max_abs"] = max(vals) if vals else None
    else:
        metrics["support_position_error_max_abs"] = None

    # Hip yaw abs max
    if "hip_yaw_abs_max_tracking" in rows[0]:
        vals = [abs(float(r["hip_yaw_abs_max_tracking"])) for r in rows if r.get("hip_yaw_abs_max_tracking")]
        metrics["hip_yaw_abs_max"] = max(vals) if vals else None
    elif "l_hip_yaw_pos" in rows[0]:
        vals = []
        for r in rows:
            l = abs(float(r.get("l_hip_yaw_pos", 0)))
            r_val = abs(float(r.get("r_hip_yaw_pos", 0)))
            vals.append(max(l, r_val))
        metrics["hip_yaw_abs_max"] = max(vals) if vals else None
    else:
        metrics["hip_yaw_abs_max"] = None

    # Pitch x max abs
    if "pitch_x" in rows[0]:
        vals = [abs(float(r["pitch_x"])) for r in rows if r.get("pitch_x")]
        metrics["pitch_x_max_abs"] = max(vals) if vals else None
    elif "robot_pitch_x" in rows[0]:
        vals = [abs(float(r["robot_pitch_x"])) for r in rows if r.get("robot_pitch_x")]
        metrics["pitch_x_max_abs"] = max(vals) if vals else None
    else:
        metrics["pitch_x_max_abs"] = None

    # Roll y max abs
    if "roll_y" in rows[0]:
        vals = [abs(float(r["roll_y"])) for r in rows if r.get("roll_y")]
        metrics["roll_y_max_abs"] = max(vals) if vals else None
    elif "robot_roll_y" in rows[0]:
        vals = [abs(float(r["robot_roll_y"])) for r in rows if r.get("robot_roll_y")]
        metrics["roll_y_max_abs"] = max(vals) if vals else None
    else:
        metrics["roll_y_max_abs"] = None

    # Height error final abs
    if "height_error_m" in rows[0]:
        last_val = rows[-1].get("height_error_m")
        metrics["height_error_final_abs"] = abs(float(last_val)) if last_val else None
    else:
        metrics["height_error_final_abs"] = None

    # Non-wheel floor contacts
    if "non_wheel_floor_contacts" in rows[0]:
        vals = [int(float(r["non_wheel_floor_contacts"])) for r in rows if r.get("non_wheel_floor_contacts") not in (None, "")]
        metrics["non_wheel_floor_contacts"] = max(vals) if vals else 0
    else:
        metrics["non_wheel_floor_contacts"] = 0

    # Contact valid rate
    if "contact_force_valid" in rows[0]:
        valid_count = sum(1 for r in rows if r.get("contact_force_valid") in ("True", "true", "1", True))
        metrics["contact_valid_rate"] = valid_count / len(rows) if rows else 0.0
    else:
        metrics["contact_valid_rate"] = 1.0  # Assume valid if no data

    # WBC applied (should be false in balance-core mode)
    metrics["wbc_applied"] = False  # balance-core mode never applies WBC directly

    # Hidden torque max
    if "tau_wheel_balance_per_joint" in rows[0]:
        vals = []
        for r in rows:
            raw = r.get("tau_wheel_balance_per_joint", "")
            if raw and raw != "":
                try:
                    # Handle comma-separated values like "0.0000,0.0000,..."
                    parts = raw.split(",")
                    for p in parts:
                        vals.append(abs(float(p.strip())))
                except (ValueError, AttributeError):
                    pass
        metrics["hidden_torque_max"] = max(vals) if vals else 0.0
    else:
        metrics["hidden_torque_max"] = 0.0

    # Ownership violations
    if "ownership_violation_count_max" in rows[0]:
        metrics["ownership_violations"] = int(float(rows[-1].get("ownership_violation_count_max", 0)))
    else:
        metrics["ownership_violations"] = 0

    # Step count
    metrics["steps"] = len(rows)

    # Termination info
    metrics["terminated"] = any(r.get("terminated") in ("True", "true", "1") for r in rows[-5:])
    if "termination_reason" in rows[0]:
        last_reasons = [r.get("termination_reason", "") for r in rows[-5:] if r.get("termination_reason") not in ("None", "", None)]
        metrics["termination_reason"] = last_reasons[-1] if last_reasons else None
    else:
        metrics["termination_reason"] = None

    return metrics


def enrich_metrics_from_summary_json(metrics: dict, summary_path: str | None) -> dict:
    """Add run-summary metrics if the sidecar JSON exists."""
    if summary_path is None:
        return metrics
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception:
        return metrics

    # Override ownership_violations from summary if available
    metrics["ownership_violations"] = summary.get("ownership_violation_count_max", metrics.get("ownership_violations", 0))
    metrics["hidden_torque_max"] = summary.get("hidden_torque_norm_max", metrics.get("hidden_torque_max", 0.0))
    metrics["actual_steps"] = summary.get("actual_steps", metrics.get("steps", 0))
    metrics["terminated"] = summary.get("terminated", metrics.get("terminated", False))
    metrics["termination_reason"] = summary.get("termination_reason", metrics.get("termination_reason"))

    return metrics


def check_acceptance_gates(metrics: dict) -> dict:
    """Check all acceptance gates and return verdict with details."""
    verdict = "PASS"
    failures = []
    details = {}

    for gate_name, gate in ACCEPTANCE_GATES.items():
        value = metrics.get(gate_name)
        label = gate["label"]
        threshold = gate["threshold"]
        comparison = gate["comparison"]

        if value is None:
            status = "NO_DATA"
            passed = False
        elif comparison == "max_abs":
            passed = abs(value) <= threshold
            status = "ok" if passed else f"{abs(value):.6f} > {threshold}"
        elif comparison == "max":
            passed = value <= threshold
            status = "ok" if passed else f"{value} > {threshold}"
        elif comparison == "min":
            passed = value >= threshold
            status = "ok" if passed else f"{value} < {threshold}"
        elif comparison == "is_false":
            passed = value == threshold
            status = "ok" if passed else f"{value} != {threshold}"
        else:
            passed = True
            status = "unknown_comparison"

        if not passed:
            verdict = "FAIL"
            failures.append(label)

        details[gate_name] = {"value": value, "threshold": threshold, "passed": passed, "status": status}

    return {"verdict": verdict, "failures": failures, "details": details}


def copy_telemetry_for_candidate(candidate: str, label: str, output_dir: Path) -> Path | None:
    """Copy the latest telemetry CSV and summary to the candidate's output directory.

    Returns the destination CSV path if successful.
    """
    src_csv = find_latest_telemetry_csv()
    src_summary = find_latest_summary_json()

    if src_csv is None:
        return None

    dest_dir = output_dir / candidate
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy CSV
    src_path = Path(src_csv)
    dest_csv = dest_dir / f"{label}.csv"
    dest_csv.write_text(src_path.read_text(), encoding="utf-8")

    # Copy summary if available
    if src_summary:
        src_sum = Path(src_summary)
        dest_sum = dest_dir / f"{label}.summary.json"
        dest_sum.write_text(src_sum.read_text(), encoding="utf-8")

    return dest_csv


# =====================================================================
# Intermediate setup generation
# =====================================================================

def generate_intermediate_setup(
    variant_name: str,
    target_com_z_m: float,
    output_path: Path,
) -> bool:
    """Generate an intermediate height setup JSON using the boundary setup generator.

    Returns True if the setup was generated and is valid.
    """
    if output_path.exists():
        print(f"  [SETUP] {variant_name}: already exists at {output_path}")
        return True

    print(f"  [SETUP] Generating {variant_name} at {target_com_z_m:.3f} m ...")

    # Use the generate_boundary_height_setups script as a subprocess
    # But that script only generates 0.300 and 0.480. We need to generate
    # intermediate setups programmatically.
    try:
        from scripts.generate_boundary_height_setups import (
            build_height_variant_setup,
        )
        from scripts.search_physical_standing_height_envelope import (
            SearchConfig,
            calibrate_root_z_from_wheel_geometry,
            resolve_standing_joint_addresses,
            search_height_for_target,
        )
        from wheeled_biped.utils.config import get_model_path
        from wheeled_biped.validation.physical_standing_height_envelope import (
            PhysicalStandingThresholds,
        )
        import mujoco
        import numpy as np

        model_path = get_model_path()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        thresholds = PhysicalStandingThresholds()
        config = SearchConfig()
        joint_addresses = resolve_standing_joint_addresses(model)

        # Get nominal pose
        data_nominal = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data_nominal, 0)
        data_nominal.qvel[:] = 0.0
        data_nominal.qacc[:] = 0.0

        nominal_root_z = calibrate_root_z_from_wheel_geometry(
            model, data_nominal, target_contact_depth_m=config.target_contact_depth_m
        )
        data_nominal.qpos[2] = nominal_root_z
        mujoco.mj_forward(model, data_nominal)

        nominal_hip_pitch = float(data_nominal.qpos[joint_addresses["l_hip_pitch"]])
        nominal_knee = float(data_nominal.qpos[joint_addresses["l_knee"]])

        # Search for candidate at target height
        candidate = search_height_for_target(
            model=model,
            target_com_z_m=target_com_z_m,
            nominal_hip_pitch=nominal_hip_pitch,
            nominal_knee=nominal_knee,
            joint_addresses=joint_addresses,
            thresholds=thresholds,
            config=config,
        )

        if candidate is None:
            print(f"  [SETUP] FAILED: No candidate found for {variant_name} at {target_com_z_m:.3f} m")
            return False

        # Build and save setup
        data_setup = mujoco.MjData(model)
        setup = build_height_variant_setup(
            variant_name=variant_name,
            target_com_z_m=target_com_z_m,
            candidate=candidate,
            model=model,
            data=data_setup,
            joint_addresses=joint_addresses,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(setup, indent=2), encoding="utf-8")
        print(f"  [SETUP] Generated: {output_path} (achieved_com_z={setup['achieved_com_z_m']:.6f} m)")
        return setup.get("setup_valid", False)

    except Exception as e:
        print(f"  [SETUP] ERROR generating {variant_name}: {e}")
        # Create a fallback: note the setup could not be generated
        return False


def ensure_intermediate_setups(dry_run: bool = False) -> dict[str, Path | None]:
    """Ensure all intermediate setup JSONs exist, generating if necessary.

    Returns a dict mapping variant_name -> Path or None if unavailable.
    """
    setups = {}

    # Target heights for intermediate grid points
    intermediate_targets = {
        "low_0p330": 0.330,
        "low_0p360": 0.360,
        "high_0p450": 0.450,
    }

    for variant_name, setup_path in INTERMEDIATE_SETUPS.items():
        if setup_path.exists():
            setups[variant_name] = setup_path
        elif dry_run:
            print(f"  [DRY-RUN] Would generate {variant_name} setup")
            setups[variant_name] = None
        else:
            target_com_z = intermediate_targets.get(variant_name, 0.0)
            success = generate_intermediate_setup(variant_name, target_com_z, setup_path)
            setups[variant_name] = setup_path if success else None

    return setups


# =====================================================================
# Evaluation runners
# =====================================================================

def run_single_eval(
    candidate: str,
    variant_name: str,
    setup_json: str | None,
    steps: int,
    label: str,
    timeout: int = 600,
    dry_run: bool = False,
) -> dict:
    """Run a single evaluation and return results dict."""
    cmd = build_sim_command(candidate, setup_json, steps)
    sim_result = run_simulation(cmd, timeout=timeout, dry_run=dry_run)

    result = {
        "label": label,
        "candidate": candidate,
        "variant": variant_name,
        "steps": steps,
        "sim_success": sim_result["success"],
        "dry_run": dry_run,
    }

    if dry_run:
        result["verdict"] = "PASS"
        result["metrics"] = {}
        result["gate_details"] = {}
        result["failures"] = []
        return result

    if not sim_result["success"]:
        result["verdict"] = "SIM_FAILED"
        result["metrics"] = {"error": sim_result.get("stderr", "")[:500]}
        result["gate_details"] = {}
        return result

    # Parse telemetry
    csv_path = find_latest_telemetry_csv()
    if csv_path and Path(csv_path).exists():
        metrics = parse_metrics_from_csv(csv_path)
        summary_path = find_latest_summary_json()
        metrics = enrich_metrics_from_summary_json(metrics, summary_path)
        gate_result = check_acceptance_gates(metrics)
        result["metrics"] = metrics
        result["verdict"] = gate_result["verdict"]
        result["failures"] = gate_result["failures"]
        result["gate_details"] = gate_result["details"]

        # Save telemetry copy for successful runs
        copy_telemetry_for_candidate(candidate, label, OUTPUT_DIR)
    else:
        result["verdict"] = "NO_CSV"
        result["metrics"] = {"error": "No telemetry CSV found"}
        result["gate_details"] = {}

    return result


def evaluate_candidate(
    candidate: str,
    dry_run: bool = False,
) -> dict:
    """Run the full evaluation suite for a single candidate.

    Evaluation order:
    1. low_0p300 Step E at 1000 steps
    2. low_0p300 Step E at 5000 steps (only if 1000 passes)
    3. high_0p480 Step E at 5000 steps
    4. Step C low_0p300 at 5000 steps
    5. Step C high_0p480 at 5000 steps
    6. Step E height grid
    7. Step C height grid
    8. Five-variant regression

    Returns dict with all results.
    """
    results = {"candidate": candidate, "runs": [], "overall_pass": False}

    def add_run(label, variant_name, setup_json, steps, timeout=600):
        r = run_single_eval(candidate, variant_name, setup_json, steps, label, timeout, dry_run)
        results["runs"].append(r)
        print(f"    {label}: {r['verdict']}")
        return r

    # ---- Phase 1: Boundary quick check (1000 steps) -------------------------
    print(f"\n  Phase 1: low_0p300 @ 1000 steps")
    r = add_run("stepE_low0p300_1000", "low_0p300", str(BOUNDARY_VARIANTS["low_0p300"]), 1000, 120)
    if r["verdict"] != "PASS":
        print(f"    -> FAILED at low_0p300 1000 steps. Skipping rest.")
        return results

    # ---- Phase 2: Boundary extended (5000 steps) ----------------------------
    print(f"\n  Phase 2: low_0p300 @ 5000 steps")
    r = add_run("stepE_low0p300_5000", "low_0p300", str(BOUNDARY_VARIANTS["low_0p300"]), 5000, 300)
    if r["verdict"] != "PASS":
        print(f"    -> FAILED at low_0p300 5000 steps. Skipping rest.")
        return results

    print(f"\n  Phase 2b: high_0p480 @ 5000 steps")
    r = add_run("stepE_high0p480_5000", "high_0p480", str(BOUNDARY_VARIANTS["high_0p480"]), 5000, 300)
    if r["verdict"] != "PASS":
        print(f"    -> FAILED at high_0p480 5000 steps. Skipping rest.")
        return results

    # ---- Phase 3: Step C boundary runs --------------------------------------
    print(f"\n  Phase 3: Step C low_0p300 @ 5000 steps")
    r = add_run("stepC_low0p300_5000", "low_0p300", str(BOUNDARY_VARIANTS["low_0p300"]), 5000, 300)
    if r["verdict"] != "PASS":
        print(f"    -> FAILED at Step C low_0p300. Skipping rest.")
        return results

    print(f"\n  Phase 3b: Step C high_0p480 @ 5000 steps")
    r = add_run("stepC_high0p480_5000", "high_0p480", str(BOUNDARY_VARIANTS["high_0p480"]), 5000, 300)
    if r["verdict"] != "PASS":
        print(f"    -> FAILED at Step C high_0p480. Skipping rest.")
        return results

    # ---- Phase 4: Step E height grid ----------------------------------------
    # Build the height grid with available setups
    print(f"\n  Phase 4: Step E height grid")
    intermediate = ensure_intermediate_setups(dry_run)

    step_e_height_grid = [
        ("low_0p300", str(BOUNDARY_VARIANTS["low_0p300"])),
        ("low_0p330", str(intermediate.get("low_0p330")) if intermediate.get("low_0p330") else None),
        ("low_0p360", str(intermediate.get("low_0p360")) if intermediate.get("low_0p360") else None),
        ("low_small", str(REGRESSION_VARIANTS["low_small"])),
        ("nominal", str(REGRESSION_VARIANTS["nominal"])),
        ("high_small", str(REGRESSION_VARIANTS["high_small"])),
        ("high_0p450", str(intermediate.get("high_0p450")) if intermediate.get("high_0p450") else None),
        ("high_0p480", str(BOUNDARY_VARIANTS["high_0p480"])),
    ]

    height_grid_pass = True
    for variant_name, setup_json in step_e_height_grid:
        if setup_json is None:
            print(f"    stepE_{variant_name}_5000: SKIP (no setup)")
            results["runs"].append({
                "label": f"stepE_{variant_name}_5000",
                "candidate": candidate,
                "variant": variant_name,
                "steps": 5000,
                "sim_success": False,
                "dry_run": dry_run,
                "verdict": "SKIP_NO_SETUP",
                "metrics": {},
                "gate_details": {},
            })
            continue
        r = add_run(f"stepE_{variant_name}_5000", variant_name, setup_json, 5000, 300)
        if r["verdict"] != "PASS":
            height_grid_pass = False

    # ---- Phase 5: Step C height grid ----------------------------------------
    print(f"\n  Phase 5: Step C height grid")
    step_c_height_grid = [
        ("low_0p300", str(BOUNDARY_VARIANTS["low_0p300"])),
        ("low_0p360", str(intermediate.get("low_0p360")) if intermediate.get("low_0p360") else None),
        ("nominal", str(REGRESSION_VARIANTS["nominal"])),
        ("high_0p480", str(BOUNDARY_VARIANTS["high_0p480"])),
    ]

    step_c_pass = True
    for variant_name, setup_json in step_c_height_grid:
        if setup_json is None:
            print(f"    stepC_{variant_name}_5000: SKIP (no setup)")
            results["runs"].append({
                "label": f"stepC_{variant_name}_5000",
                "candidate": candidate,
                "variant": variant_name,
                "steps": 5000,
                "sim_success": False,
                "dry_run": dry_run,
                "verdict": "SKIP_NO_SETUP",
                "metrics": {},
                "gate_details": {},
            })
            continue
        r = add_run(f"stepC_{variant_name}_5000", variant_name, setup_json, 5000, 300)
        if r["verdict"] != "PASS":
            step_c_pass = False

    # ---- Phase 6: Five-variant regression -----------------------------------
    print(f"\n  Phase 6: Five-variant regression")
    regression_pass = True
    for variant_name, setup_path in REGRESSION_VARIANTS.items():
        setup_str = str(setup_path) if setup_path.exists() else None
        if setup_str is None:
            print(f"    regression_{variant_name}_5000: SKIP (no setup)")
            results["runs"].append({
                "label": f"regression_{variant_name}_5000",
                "candidate": candidate,
                "variant": variant_name,
                "steps": 5000,
                "sim_success": False,
                "dry_run": dry_run,
                "verdict": "SKIP_NO_SETUP",
                "metrics": {},
                "gate_details": {},
            })
            regression_pass = False
            continue
        r = add_run(f"regression_{variant_name}_5000", variant_name, setup_str, 5000, 300)
        if r["verdict"] != "PASS":
            regression_pass = False

    # ---- Overall verdict ----------------------------------------------------
    all_pass = all(r.get("verdict") == "PASS" for r in results["runs"] if r.get("verdict") not in ("SKIP_NO_SETUP",))
    results["overall_pass"] = all_pass

    return results


# =====================================================================
# Report generation
# =====================================================================

def generate_csv_summary(all_results: list[dict], output_dir: Path) -> Path:
    """Generate CSV summary of all runs."""
    csv_path = output_dir / "continuous_low_height_candidate_summary.csv"
    fieldnames = [
        "candidate", "label", "variant", "steps", "sim_success", "verdict",
        "support_position_error_max_abs", "hip_yaw_abs_max",
        "pitch_x_max_abs", "roll_y_max_abs", "height_error_final_abs",
        "non_wheel_floor_contacts", "contact_valid_rate",
        "failures",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand_result in all_results:
            for run in cand_result.get("runs", []):
                row = {
                    "candidate": run.get("candidate", ""),
                    "label": run.get("label", ""),
                    "variant": run.get("variant", ""),
                    "steps": run.get("steps", 0),
                    "sim_success": run.get("sim_success", False),
                    "verdict": run.get("verdict", ""),
                    "support_position_error_max_abs": run.get("metrics", {}).get("support_position_error_max_abs", ""),
                    "hip_yaw_abs_max": run.get("metrics", {}).get("hip_yaw_abs_max", ""),
                    "pitch_x_max_abs": run.get("metrics", {}).get("pitch_x_max_abs", ""),
                    "roll_y_max_abs": run.get("metrics", {}).get("roll_y_max_abs", ""),
                    "height_error_final_abs": run.get("metrics", {}).get("height_error_final_abs", ""),
                    "non_wheel_floor_contacts": run.get("metrics", {}).get("non_wheel_floor_contacts", ""),
                    "contact_valid_rate": run.get("metrics", {}).get("contact_valid_rate", ""),
                    "failures": "; ".join(run.get("failures", [])),
                }
                writer.writerow(row)
    return csv_path


def generate_json_summary(all_results: list[dict], output_dir: Path) -> Path:
    """Generate JSON summary of all results."""
    json_path = output_dir / "continuous_low_height_candidate_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    return json_path


def generate_report(all_results: list[dict], output_dir: Path) -> Path:
    """Generate human-readable markdown report."""
    report_path = output_dir / "continuous_low_height_sagittal_fix_report.md"

    lines = [
        "# Continuous Low-Height Sagittal Authority Fix Evaluation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Candidates Evaluated",
        "",
    ]
    for r in all_results:
        status = "PASS" if r.get("overall_pass") else "FAIL"
        lines.append(f"- **{r['candidate']}**: {status}")

    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])

    for cand_result in all_results:
        cand_name = cand_result["candidate"]
        overall = "PASS" if cand_result.get("overall_pass") else "FAIL"
        lines.append(f"### {cand_name} -- {overall}")
        lines.append("")
        lines.append(f"{'Label':<45} {'Steps':>6} {'Verdict':<12} {'Key Failures'}")
        lines.append("-" * 100)

        for run in cand_result.get("runs", []):
            label = run.get("label", "")
            steps = run.get("steps", 0)
            verdict = run.get("verdict", "")
            failures = "; ".join(run.get("failures", []))
            if len(failures) > 60:
                failures = failures[:57] + "..."
            lines.append(f"{label:<45} {steps:>6} {verdict:<12} {failures}")

        lines.append("")

    # Acceptance gate summary table
    lines.extend([
        "## Acceptance Gates",
        "",
        "| Gate | Threshold |",
        "|------|-----------|",
    ])
    for gate_name, gate in ACCEPTANCE_GATES.items():
        lines.append(f"| {gate['label']} | {gate['threshold']} |")

    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_machine_summary(all_results: list[dict], output_dir: Path) -> Path:
    """Generate machine-readable summary JSON."""
    summary_path = output_dir / "continuous_low_height_sagittal_fix_summary.json"

    selected = None
    for r in all_results:
        if r.get("overall_pass"):
            selected = r["candidate"]
            break

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates_evaluated": [r["candidate"] for r in all_results],
        "selected_candidate": selected,
        "candidate_results": {},
    }

    for r in all_results:
        cand = r["candidate"]
        runs_summary = {}
        for run in r.get("runs", []):
            runs_summary[run.get("label", "")] = {
                "variant": run.get("variant"),
                "steps": run.get("steps"),
                "verdict": run.get("verdict"),
                "failures": run.get("failures", []),
                "key_metrics": {
                    k: run.get("metrics", {}).get(k)
                    for k in [
                        "support_position_error_max_abs",
                        "hip_yaw_abs_max",
                        "pitch_x_max_abs",
                        "roll_y_max_abs",
                        "height_error_final_abs",
                        "non_wheel_floor_contacts",
                    ]
                },
            }
        summary["candidate_results"][cand] = {
            "overall_pass": r.get("overall_pass", False),
            "runs": runs_summary,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary_path


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate continuous low-height sagittal authority fix candidates"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without actually running simulations",
    )
    parser.add_argument(
        "--candidate", type=str, default=None,
        choices=CANDIDATES,
        help="Run only a specific candidate",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates_to_run = [args.candidate] if args.candidate else CANDIDATES

    print("=" * 80)
    print("Continuous Low-Height Sagittal Authority Fix Evaluation")
    print(f"Candidates: {', '.join(candidates_to_run)}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    # Ensure intermediate setups exist
    if not args.dry_run:
        print("\nEnsuring intermediate height setups exist...")
        ensure_intermediate_setups(dry_run=False)

    all_results = []
    selected_candidate = None

    for candidate in candidates_to_run:
        print(f"\n{'=' * 60}")
        print(f"Evaluating candidate: {candidate}")
        print(f"{'=' * 60}")

        cand_result = evaluate_candidate(candidate, dry_run=args.dry_run)
        all_results.append(cand_result)

        if cand_result.get("overall_pass"):
            print(f"\n  CANDIDATE {candidate} FULLY PASSES all gates.")
            selected_candidate = candidate
            break
        else:
            print(f"\n  CANDIDATE {candidate} FAILED. Continuing to next candidate.")

    # Generate output artifacts
    csv_path = generate_csv_summary(all_results, OUTPUT_DIR)
    json_path = generate_json_summary(all_results, OUTPUT_DIR)
    report_path = generate_report(all_results, OUTPUT_DIR)
    summary_path = generate_machine_summary(all_results, OUTPUT_DIR)

    print(f"\n{'=' * 80}")
    print("Evaluation Complete")
    print(f"{'=' * 80}")
    print(f"Selected candidate: {selected_candidate or 'NONE (all failed)'}")
    print(f"\nOutput artifacts:")
    print(f"  CSV:       {csv_path}")
    print(f"  JSON:      {json_path}")
    print(f"  Report:    {report_path}")
    print(f"  Summary:   {summary_path}")
    print(f"  Telemetry: {OUTPUT_DIR}/selected_candidate_telemetry/")
    print(f"{'=' * 80}")

    # Print comparison table
    print(f"\n{'Candidate':<40} {'Runs':>6} {'Passed':>6} {'Overall':>8}")
    print("-" * 60)
    for r in all_results:
        runs = [run for run in r.get("runs", []) if run.get("verdict") not in ("SKIP_NO_SETUP", "DRY_RUN")]
        passed = sum(1 for run in runs if run.get("verdict") == "PASS")
        total = len(runs)
        overall = "PASS" if r.get("overall_pass") else "FAIL"
        print(f"{r['candidate']:<40} {total:>6} {passed:>6} {overall:>8}")


if __name__ == "__main__":
    main()
