#!/usr/bin/env python3
"""
Validate K1 Augmented Models — Phase 7.

Validates identified models from Phase 6 against design-ready criteria.
A model is DESIGN_READY only if it captures the 0.24-0.4 Hz mode.

Output:
  outputs/k1_augmented_identification_dataset/augmented_model_validation.json
  outputs/k1_augmented_identification_dataset/augmented_model_validation.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUGMENTED_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"
LEGACY_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"

# -- Validation Criteria --
TARGET_MODE_FREQ_HZ = 0.30  # Center of 0.24-0.4 Hz
FREQ_TOLERANCE = 0.15  # +-15%
DAMPING_TOLERANCE = 0.05  # absolute
MAX_CONDITION = 1e6
MAX_ROLLOUT_50_RMSE = 0.5
MAX_ROLLOUT_200_RMSE = 1.0

# -- Classification Options --
CLASSIFICATIONS = [
    "DESIGN_READY",
    "NEEDS_MORE_EXCITATION",
    "NEEDS_FILTER_STATE",
    "NEEDS_CLIPPING_STATE",
    "NEEDS_NONLINEAR_LIFTING",
    "OVERFIT",
    "HEIGHT_DATA_INSUFFICIENT",
    "INCONCLUSIVE",
]


def validate_model(fit_quality: dict, state_vector: str, height: str) -> dict:
    """Validate a single model against design-ready criteria."""
    checks = {}

    # Check 1: Mode capture
    best_mode = fit_quality.get("best_mode")
    if best_mode and best_mode.get("freq_hz"):
        freq = best_mode["freq_hz"]
        freq_err = abs(freq - TARGET_MODE_FREQ_HZ) / TARGET_MODE_FREQ_HZ
        checks["mode_captured"] = freq_err <= FREQ_TOLERANCE
        checks["mode_freq_hz"] = freq
        checks["mode_freq_error"] = freq_err
        checks["mode_damping"] = best_mode.get("damping", 0)
    else:
        checks["mode_captured"] = False
        checks["mode_freq_hz"] = None
        checks["mode_freq_error"] = None
        checks["mode_damping"] = None

    # Check 2: Condition number
    cond = fit_quality.get("condition_number", float("inf"))
    checks["condition_number"] = cond
    checks["well_conditioned"] = cond < MAX_CONDITION and cond > 0

    # Check 3: R2
    r2 = fit_quality.get("test_r2", fit_quality.get("r2", 0))
    checks["r2"] = r2
    checks["overfit"] = r2 > 0.9999

    # Check 4: Rollout RMSE
    r50 = fit_quality.get("rollout_50_rmse")
    r200 = fit_quality.get("rollout_200_rmse")
    checks["rollout_50_rmse"] = r50
    checks["rollout_200_rmse"] = r200
    checks["rollout_50_ok"] = r50 is not None and r50 < MAX_ROLLOUT_50_RMSE
    checks["rollout_200_ok"] = r200 is not None and r200 < MAX_ROLLOUT_200_RMSE

    # Classification
    if not checks["mode_captured"]:
        if "notch" in state_vector:
            classification = "NEEDS_NONLINEAR_LIFTING"
        elif "clip" in state_vector:
            classification = "NEEDS_CLIPPING_STATE"
        elif state_vector == "x6_base":
            classification = "NEEDS_FILTER_STATE"
        else:
            classification = "NEEDS_STATE_AUGMENTATION" if hasattr(fit_quality, 'get') else "INCONCLUSIVE"
    elif checks["overfit"]:
        classification = "OVERFIT"
    elif not checks["well_conditioned"]:
        classification = "NEEDS_NONLINEAR_LIFTING"
    elif checks["mode_captured"] and checks["rollout_50_ok"] and checks["rollout_200_ok"]:
        classification = "DESIGN_READY"
    elif not checks["rollout_50_ok"]:
        classification = "NEEDS_MORE_EXCITATION"
    else:
        classification = "INCONCLUSIVE"

    checks["classification"] = classification
    return checks


def validate_all(dataset_dir: Path = None):
    """Validate all identified models."""
    if dataset_dir is None:
        dataset_dir = AUGMENTED_DIR
    if not dataset_dir.exists():
        dataset_dir = LEGACY_DIR
    if not dataset_dir.exists():
        return {"status": "NO_DATASET_FOUND"}

    summary_path = dataset_dir / "augmented_identification_summary.json"
    if not summary_path.exists():
        # Try legacy
        summary_path = dataset_dir / "identification_summary.json"
    if not summary_path.exists():
        return {"status": "NO_IDENTIFICATION_SUMMARY", "path": str(summary_path)}

    with open(summary_path) as f:
        id_results = json.load(f)

    validations = {}
    design_ready_count = 0

    for height_name, candidates in id_results.items():
        if "error" in candidates:
            validations[height_name] = candidates
            continue

        validations[height_name] = {}
        for cand_name, cand_result in candidates.items():
            if "error" in cand_result:
                validations[height_name][cand_name] = cand_result
                continue

            validation = validate_model(cand_result, cand_name, height_name)
            validations[height_name][cand_name] = validation

            if validation["classification"] == "DESIGN_READY":
                design_ready_count += 1

    result = {
        "validations": validations,
        "design_ready_count": design_ready_count,
        "allows_design": design_ready_count >= 1,
    }

    # Save JSON
    json_path = dataset_dir / "augmented_model_validation.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Markdown report
    md_lines = [
        "# K1 Augmented Model Validation",
        f"",
        f"**Design-Ready Models:** {design_ready_count}",
        f"**State-Feedback Design Allowed:** {'YES' if result['allows_design'] else 'NO'}",
        f"",
        "## Per-Height Per-Candidate Results",
        f"",
        "| Height | Candidate | Mode Freq | Mode Err | R2 | Cond | 50-step RMSE | Classification |",
        "|--------|-----------|-----------|----------|-----|------|-------------|----------------|",
    ]
    for height_name, candidates in validations.items():
        if "error" in candidates:
            md_lines.append(f"| {height_name} | — | — | — | — | — | — | {candidates['error']} |")
            continue
        for cand_name, v in candidates.items():
            if "error" in v:
                md_lines.append(f"| {height_name} | {cand_name} | — | — | — | — | — | {v['error']} |")
            else:
                md_lines.append(
                    f"| {height_name} | {cand_name} | "
                    f"{v.get('mode_freq_hz', 'N/A')} | "
                    f"{v.get('mode_freq_error', 'N/A')} | "
                    f"{v.get('r2', 'N/A')} | "
                    f"{v.get('condition_number', 'N/A')} | "
                    f"{v.get('rollout_50_rmse', 'N/A')} | "
                    f"**{v.get('classification', 'N/A')}** |"
                )

    md_path = dataset_dir / "augmented_model_validation.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Model validation complete. Design-ready: {design_ready_count}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate K1 augmented models")
    parser.add_argument("--dataset-dir", type=str, default=None)
    args = parser.parse_args()
    validate_all(Path(args.dataset_dir) if args.dataset_dir else None)


if __name__ == "__main__":
    main()
