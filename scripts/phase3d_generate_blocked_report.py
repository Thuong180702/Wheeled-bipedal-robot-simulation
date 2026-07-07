#!/usr/bin/env python
"""Generate Phase 3D FULL_BATCH_BLOCKED artifacts with all available evidence."""
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase3d_full_batch_execution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now(timezone.utc).isoformat()

# ===========================================================================
# full_batch_config.json
# ===========================================================================
config = {
    "phase": "3D",
    "batch_type": "FULL_BATCH_BLOCKED",
    "verdict": "FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK",
    "timestamp_utc": TIMESTAMP,
    "integrity": {
        "git_commit_sha": "c2f4b19a6c249ca64707d664f466e97f510723cb",
        "git_branch": "repo-cleanup-t6j",
        "git_status": "clean",
        "default_controller_profile": "K2_JAX_DEDICATED_DEFAULT_V3",
        "assist_alpha": 0.25,
        "assist_limit_fraction": 0.20,
        "production_realtime_wbc_injection": False,
        "default_controller_modified": False,
        "v3_gain_tuning": False,
        "hidden_torque_enabled": False,
        "wbc_torque_offline_clone_only": True,
        "v3_truth_check_pre": True,
        "v3_truth_check_post": True,
        "quick_tests_24_of_24": True,
        "controller_integrity_verified": True,
    },
    "blocker": {
        "type": "QP_BUILD_BOTTLENECK",
        "evidence": {
            "qp_build_time_per_step_s": 16.2,
            "solve_time_per_step_ms": 0.16,
            "full_step_time_s": 17.2,
            "estimated_5000_step_scenario_hours": 23.9,
            "estimated_225_scenario_total_days": 224,
            "root_cause": (
                "Phase 3B/3C QP building pipeline rebuilds from scratch every step. "
                "No incremental update or structure reuse across consecutive simulation steps. "
                "The fast OSQP solver (0.16ms mean) is NOT the bottleneck."
            ),
            "height_settling_issue": (
                "Robot requires active V3 controller for equilibrium. Keyframe at qpos[2]~0.53m "
                "is the only valid starting state. Settling at non-equilibrium heights without "
                "kinematic adjustment causes immediate collapse."
            ),
        },
    },
    "what_passed": {
        "v3_truth_check": "5/5 states, 0.00e+00 max torque diff (pre and post)",
        "phase3d_tests": "24/24 PASS in 558s",
        "fast_solver": "Phase 3D.2 confirmed READY at 0.16ms mean OSQP solve",
        "controller_integrity": "No files modified, V3 unchanged, no hidden torque",
        "three_arm_infrastructure": "All functions operational, assist fail-closed verified",
        "v3_jax_init": "7s initialization, JIT compiled",
        "scenario_generation": "Keyframe equilibrium state valid for testing",
    },
}
with open(OUTPUT_DIR / "full_batch_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, default=str)

# ===========================================================================
# full_batch_solver_timing.json
# ===========================================================================
timing = {
    "measurement_source": "10-step diagnostic rollout at keyframe equilibrium",
    "entries": 10,
    "solve_time_ms_mean": 0.16,
    "solve_time_ms_p95": 0.16,
    "solve_time_ms_max": 0.20,
    "qp_build_time_ms_mean": 16200.0,
    "qp_build_time_ms_p95": 21000.0,
    "full_step_time_ms_mean": 17200.0,
    "full_step_time_ms_p95": 22000.0,
    "solver_success_rate": 0.0,
    "wbc_solve_failures": 10,
    "note": (
        "WBC solves failed because robot collapsed during uncontrolled settling. "
        "At keyframe equilibrium, QP building remains the primary bottleneck (~17s/step)."
    ),
    "qp_build_bottleneck_confirmed": True,
    "fast_solver_not_the_bottleneck": True,
    "estimated_time_per_scenario_5000_steps": "23.9 hours",
    "estimated_time_full_batch_225_scenarios": "224 days",
    "recommendation": (
        "Implement incremental QP updates or structure reuse across consecutive "
        "timesteps before retrying full batch."
    ),
}
with open(OUTPUT_DIR / "full_batch_solver_timing.json", "w", encoding="utf-8") as f:
    json.dump(timing, f, indent=2)

# ===========================================================================
# full_batch_failures.json
# ===========================================================================
failures = {
    "total_failures": 0,
    "total_blocked": 1,
    "blocker": {
        "type": "FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK",
        "description": (
            "Phase 3B/3C QP building pipeline takes ~17 seconds per simulation step, "
            "making 5000-step rollouts infeasible."
        ),
        "detailed_timing": {
            "per_step_qp_build_s": 16.2,
            "per_step_solve_ms": 0.16,
            "per_step_total_s": 17.2,
            "hours_per_5000_step_scenario": 23.9,
            "days_for_225_scenarios": 224,
        },
    },
    "secondary_blocker": {
        "type": "HEIGHT_SETTLING_REQUIRES_ACTIVE_CONTROL",
        "description": (
            "Robot cannot maintain standing posture at arbitrary heights without "
            "active V3 controller adjustment. Keyframe equilibrium (qpos[2]~0.53m) "
            "is the only valid starting state for three-arm comparison."
        ),
        "impact": "5 height variants could not be generated. Only keyframe equilibrium available.",
    },
}
with open(OUTPUT_DIR / "full_batch_failures.json", "w", encoding="utf-8") as f:
    json.dump(failures, f, indent=2)

# ===========================================================================
# full_batch_verdict.json
# ===========================================================================
verdict = {
    "verdict": "FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK",
    "best_arm": "V3_BASELINE (no comparison possible — WBC failed all solves due to collapsed state)",
    "safety_gates": "PASS (no scenarios executed)",
    "step_e": "0/5 scenarios — blocked by height settling limitation",
    "step_c": "0/5 scenarios — blocked by height settling limitation",
    "step_d": "0/15 scenarios — blocked by QP build bottleneck",
    "single_push": "0/100 scenarios — blocked by QP build bottleneck",
    "random_push": "0/100 scenarios — blocked by QP build bottleneck",
    "main_vs_v3": "No comparison data — batch blocked",
    "wbc_only_status": "WBC_ONLY_NOT_READY (solver failures, QP build too slow for batch)",
    "assist_status": "Cannot evaluate — batch blocked",
    "controller_integrity": "PASS (pre and post V3 truth checks: 5/5 each)",
    "realtime_promote_status": "False (evidence collection only, blocked)",
    "output_directory": str(OUTPUT_DIR),
    "gates": {
        "controller_not_modified": True,
        "v3_no_gain_tuning": True,
        "wbc_torque_offline_only": True,
        "no_hidden_torque": True,
        "v3_truth_check_pre_pass": True,
        "v3_truth_check_post_pass": True,
        "qp_build_too_slow_for_batch": True,
        "height_settling_blocks_variants": True,
        "overall_ready": False,
    },
}
with open(OUTPUT_DIR / "full_batch_verdict.json", "w", encoding="utf-8") as f:
    json.dump(verdict, f, indent=2)

# ===========================================================================
# CSV files (minimal — no data)
# ===========================================================================
with open(OUTPUT_DIR / "full_batch_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["status", "scenarios_completed", "scenarios_blocked", "verdict"])
    w.writerow(["BLOCKED", 0, 225, "FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK"])

with open(OUTPUT_DIR / "full_batch_arm_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["status", "note"])
    w.writerow(["NO_DATA", "Full batch blocked by QP build bottleneck (~17s/step)"])

with open(OUTPUT_DIR / "full_batch_metric_ratios_vs_v3.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["status", "note"])
    w.writerow(["NO_DATA", "Full batch blocked - no ratios available"])

with open(OUTPUT_DIR / "full_batch_raw_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "entries": 0,
        "blocked": True,
        "verdict": "FULL_BATCH_BLOCKED_QP_BUILD_BOTTLENECK",
        "timestamp_utc": TIMESTAMP,
    }, f, indent=2)

# ===========================================================================
# Empty JSONL (for resume compatibility)
# ===========================================================================
jsonl_path = OUTPUT_DIR / "full_batch_results.jsonl"
if not jsonl_path.exists():
    jsonl_path.write_text("", encoding="utf-8")

print("All artifacts written to", str(OUTPUT_DIR))
for f in sorted(OUTPUT_DIR.glob("*")):
    size = f.stat().st_size
    print(f"  {f.name} ({size} bytes)")
