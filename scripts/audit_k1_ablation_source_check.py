#!/usr/bin/env python3
"""
K1 Ablation-Only Mode Source Check — Phase 8.

Runs audit-only ablation experiments to isolate the 0.24-0.4 Hz mode source.
These are NOT candidate controllers — they are source-isolation experiments only.

Ablation runs:
  1. baseline K1
  2. notch telemetry only (no behavior change)
  3. notch bypass audit mode
  4. hard clipping baseline
  5. smooth clipping audit mode
  6. position cap disabled audit mode (with strict safety limit)
  7. notch bypass + hard clipping
  8. notch bypass + smooth clipping

STRICT CONSTRAINT: Every run labeled audit_ablation_mode. Never promoted.

Output:
  outputs/k1_augmented_identification_dataset/ablation_source_check.json
  outputs/k1_augmented_identification_dataset/ablation_source_check.md
"""

import json
import sys
from pathlib import Path

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"

# -- Ablation Run Definitions --
ABLATION_RUNS = {
    "A1_baseline_k1": {
        "description": "Baseline K1 — no changes",
        "audit_mode": True,
        "notch_bypass": False,
        "smooth_clip": False,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
    },
    "A2_notch_telemetry_only": {
        "description": "Notch telemetry only — no behavior change",
        "audit_mode": True,
        "notch_bypass": False,
        "smooth_clip": False,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
        "note": "This is the default augmented telemetry state",
    },
    "A3_notch_bypass": {
        "description": "Notch filter bypassed — signal passes through unfiltered",
        "audit_mode": True,
        "notch_bypass": True,
        "smooth_clip": False,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
        "risk": "2.5 Hz WIP mode may reappear",
    },
    "A4_hard_clipping_baseline": {
        "description": "Hard clipping at +-5 Nm (current K1 behavior)",
        "audit_mode": True,
        "notch_bypass": False,
        "smooth_clip": False,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
    },
    "A5_smooth_clipping": {
        "description": "Smooth (tanh) clipping instead of hard clip",
        "audit_mode": True,
        "notch_bypass": False,
        "smooth_clip": True,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
        "note": "Uses tanh saturation to test if hard clip causes mode",
    },
    "A6_position_cap_disabled": {
        "description": "Position cap disabled — SAFETY LIMITED",
        "audit_mode": True,
        "notch_bypass": False,
        "smooth_clip": False,
        "position_cap_disabled": True,
        "profile": "k1_pitch_rate_notch_v1",
        "risk": "HIGH — unrestricted position torque may cause instability",
        "safety_limit": "+-5 Nm total cap still active",
    },
    "A7_notch_bypass_hard_clip": {
        "description": "Notch bypass + hard clipping",
        "audit_mode": True,
        "notch_bypass": True,
        "smooth_clip": False,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
        "risk": "WIP mode + hard clip interaction",
    },
    "A8_notch_bypass_smooth_clip": {
        "description": "Notch bypass + smooth clipping",
        "audit_mode": True,
        "notch_bypass": True,
        "smooth_clip": True,
        "position_cap_disabled": False,
        "profile": "k1_pitch_rate_notch_v1",
        "risk": "Smooth clip may damp WIP mode",
    },
}

# -- Mode Change Classifications --
MODE_OUTCOMES = [
    "MODE_DISAPPEARS_WITH_NOTCH_BYPASS",
    "MODE_DISAPPEARS_WITH_SMOOTH_CLIP",
    "MODE_DISAPPEARS_WITH_POSITION_CAP_CHANGE",
    "MODE_REQUIRES_NOTCH_AND_CLIPPING",
    "MODE_UNAFFECTED_BY_ABLATIONS",
    "ABLATION_INCONCLUSIVE",
]


def assess_ablation_results(ablation_results: dict) -> dict:
    """Synthesize ablation results into a mode source conclusion."""
    # This is a template function — real results require running simulations
    return {
        "classification": "ABLATION_INCONCLUSIVE",
        "reason": "Ablation simulations not yet executed. Run individual ablation profiles "
                  "via simulate_hierarchical_controller.py with audit_ablation_mode flags.",
        "required_actions": [
            "Implement --audit-ablation-mode flag in simulate_hierarchical_controller.py",
            "Run each ablation (A1-A8) for 500-1000 steps at 0.48m height",
            "Compare FFT of pitch_x across ablations",
            "Determine which ablation changes the 0.24-0.4 Hz mode",
        ],
    }


def generate_ablation_report():
    """Generate ablation source check report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Placeholder — real results require simulation runs
    assessment = assess_ablation_results({})

    result = {
        "ablation_runs_defined": len(ABLATION_RUNS),
        "ablation_runs": ABLATION_RUNS,
        "assessment": assessment,
        "status": "TEMPLATE — requires simulation execution",
    }

    # Save JSON
    json_path = OUTPUT_DIR / "ablation_source_check.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Markdown report
    md_lines = [
        "# K1 Ablation-Only Mode Source Check",
        "",
        "**Status:** TEMPLATE — requires simulation execution",
        "",
        "## Ablation Runs Defined",
        "",
        "| ID | Description | Notch Bypass | Smooth Clip | Pos Cap Disabled | Risk |",
        "|----|-------------|-------------|-------------|-----------------|------|",
    ]
    for run_id, run_def in ABLATION_RUNS.items():
        risk = run_def.get("risk", "LOW")
        md_lines.append(
            f"| {run_id} | {run_def['description']} | "
            f"{run_def['notch_bypass']} | {run_def['smooth_clip']} | "
            f"{run_def['position_cap_disabled']} | {risk} |"
        )

    md_lines.extend([
        "",
        "## Assessment",
        "",
        f"**Classification:** {assessment['classification']}",
        f"",
        f"**Reason:** {assessment['reason']}",
        f"",
        "### Required Actions",
    ])
    for action in assessment["required_actions"]:
        md_lines.append(f"- {action}")

    md_lines.extend([
        "",
        "## Mode Outcome Classifications",
        "",
    ])
    for outcome in MODE_OUTCOMES:
        md_lines.append(f"- `{outcome}`")

    md_path = OUTPUT_DIR / "ablation_source_check.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Ablation source check template saved.")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return result


def main():
    generate_ablation_report()


if __name__ == "__main__":
    main()
