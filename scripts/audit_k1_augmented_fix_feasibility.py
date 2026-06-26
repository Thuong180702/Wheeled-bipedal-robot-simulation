#!/usr/bin/env python3
"""
K1 Augmented Fix Feasibility Gate — Phase 9.

Using source isolation + augmented models, determines what kind of fix is justified.

Allowed conclusions:
  A. AUGMENTED_STATE_FEEDBACK_READY
  B. FILTER_PATH_REDESIGN_NEEDED
  C. SATURATION_SMOOTHING_OR_ANTI_WINDUP_NEEDED
  D. POSITION_CAP_REDESIGN_NEEDED
  E. NONLINEAR_KOOPMAN_OR_GAIN_SCHEDULE_NEEDED
  F. DIRECT_DAMPING_AUGMENTATION_ONLY
  G. INCONCLUSIVE_NEED_MORE_DATA

STRICT CONSTRAINT: Do not design the fix. Analysis only.

Output:
  outputs/k1_augmented_identification_dataset/fix_feasibility_gate.json
  outputs/k1_augmented_identification_dataset/fix_feasibility_gate.md
"""

import json
import sys
from pathlib import Path

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUGMENTED_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"
LEGACY_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"

# -- Fix Feasibility Classifications --
FIX_CLASSIFICATIONS = {
    "A": "AUGMENTED_STATE_FEEDBACK_READY",
    "B": "FILTER_PATH_REDESIGN_NEEDED",
    "C": "SATURATION_SMOOTHING_OR_ANTI_WINDUP_NEEDED",
    "D": "POSITION_CAP_REDESIGN_NEEDED",
    "E": "NONLINEAR_KOOPMAN_OR_GAIN_SCHEDULE_NEEDED",
    "F": "DIRECT_DAMPING_AUGMENTATION_ONLY",
    "G": "INCONCLUSIVE_NEED_MORE_DATA",
}


def load_source_analysis(dataset_dir: Path) -> dict | None:
    """Load source isolation analysis results."""
    src_path = dataset_dir / "nonlinear_mode_source_analysis.json"
    if not src_path.exists():
        return None
    with open(src_path) as f:
        return json.load(f)


def load_model_validation(dataset_dir: Path) -> dict | None:
    """Load augmented model validation results."""
    val_path = dataset_dir / "augmented_model_validation.json"
    if not val_path.exists():
        return None
    with open(val_path) as f:
        return json.load(f)


def load_identification_summary(dataset_dir: Path) -> dict | None:
    """Load identification summary."""
    id_path = dataset_dir / "augmented_identification_summary.json"
    if not id_path.exists():
        id_path = dataset_dir / "identification_summary.json"
    if not id_path.exists():
        return None
    with open(id_path) as f:
        return json.load(f)


def determine_fix_feasibility(
    source_analysis: dict | None,
    model_validation: dict | None,
    id_summary: dict | None,
) -> dict:
    """Determine which fix approach is justified by the evidence."""

    # Default: inconclusive
    conclusion = "G"
    evidence = []

    # Check if we have any data at all
    if source_analysis is None and model_validation is None and id_summary is None:
        return {
            "conclusion": "G",
            "classification": FIX_CLASSIFICATIONS["G"],
            "reason": "No analysis data available. Run Phases 3-7 first.",
            "evidence": [],
            "required_controller_states": [],
            "expected_risk": "UNKNOWN",
            "expected_validation_path": "Execute full augmented pipeline",
            "why_other_fixes_rejected": {},
        }

    # Check source analysis
    source_class = None
    if source_analysis:
        sc = source_analysis.get("source_classification", {})
        source_class = sc.get("classification", "INCONCLUSIVE")

    # Check model validation
    design_ready = False
    if model_validation:
        design_ready = model_validation.get("allows_design", False)

    # Check identification summary
    any_model_with_mode = False
    if id_summary:
        for height, candidates in id_summary.items():
            if not isinstance(candidates, dict):
                continue
            if "error" in candidates:
                continue
            for cand_name, cand in candidates.items():
                if not isinstance(cand, dict):
                    continue
                if "error" in cand:
                    continue
                if cand.get("best_mode"):
                    any_model_with_mode = True

    # Decision logic
    if design_ready:
        conclusion = "A"
        evidence.append("At least one model is DESIGN_READY")
        evidence.append(f"Source analysis: {source_class}")

    elif source_class == "NOTCH_FILTER_DOMINANT":
        if any_model_with_mode:
            conclusion = "A"
            evidence.append("Notch-dominant mode, but model DID capture the mode")
            evidence.append("Augmented state feedback feasible")
        else:
            conclusion = "B"
            evidence.append("Notch filter is dominant mode source")
            evidence.append("Filter path redesign needed before state feedback")
            evidence.append("Or: expose filter state and retry identification with x8_notch")

    elif source_class == "TORQUE_CLIPPING_DOMINANT":
        conclusion = "C"
        evidence.append("Torque clipping is dominant mode source")
        evidence.append("Saturation smoothing or anti-windup needed")

    elif source_class == "POSITION_CAP_DOMINANT":
        conclusion = "D"
        evidence.append("Position cap is dominant mode source")
        evidence.append("Position cap redesign needed")

    elif source_class == "NOTCH_CLIPPING_INTERACTION":
        conclusion = "E"
        evidence.append("Notch-clipping interaction is mode source")
        evidence.append("Nonlinear identification (Koopman) or gain scheduling may be needed")

    elif source_class == "INCONCLUSIVE":
        if any_model_with_mode:
            conclusion = "F"
            evidence.append("Mode detected but source unclear")
            evidence.append("Direct damping augmentation may help without full state feedback")
        else:
            conclusion = "G"
            evidence.append("Insufficient data for classification")

    else:
        conclusion = "G"
        evidence.append("Insufficient data for classification")

    # Build rejection rationale for other fixes
    rejected = {}
    for key in FIX_CLASSIFICATIONS:
        if key != conclusion:
            rejected[key] = f"Not supported by current evidence: {source_class or 'no source analysis'}"

    return {
        "conclusion": conclusion,
        "classification": FIX_CLASSIFICATIONS[conclusion],
        "reason": evidence[0] if evidence else "Analysis inconclusive",
        "evidence": evidence,
        "required_controller_states": [],
        "expected_risk": "MEDIUM" if conclusion in ("A", "B") else "HIGH" if conclusion in ("D", "E") else "UNKNOWN",
        "expected_validation_path": "Execute fix, run validation at 3 heights, compare mode frequency/damping",
        "why_other_fixes_rejected": rejected,
    }


def audit_fix_feasibility(dataset_dir: Path = None):
    """Run the fix feasibility gate audit."""
    if dataset_dir is None:
        dataset_dir = AUGMENTED_DIR
    if not dataset_dir.exists():
        dataset_dir = LEGACY_DIR

    source_analysis = load_source_analysis(dataset_dir)
    model_validation = load_model_validation(dataset_dir)
    id_summary = load_identification_summary(dataset_dir)

    feasibility = determine_fix_feasibility(source_analysis, model_validation, id_summary)
    feasibility["data_sources"] = {
        "source_analysis": source_analysis is not None,
        "model_validation": model_validation is not None,
        "identification_summary": id_summary is not None,
    }

    # Save JSON
    json_path = dataset_dir / "fix_feasibility_gate.json"
    with open(json_path, "w") as f:
        json.dump(feasibility, f, indent=2)

    # Markdown report
    md_lines = [
        "# K1 Augmented Fix Feasibility Gate",
        "",
        f"## Conclusion: {feasibility['classification']}",
        "",
        f"**Reason:** {feasibility['reason']}",
        "",
        "### Evidence",
    ]
    for e in feasibility["evidence"]:
        md_lines.append(f"- {e}")

    md_lines.extend([
        "",
        "### Data Sources Available",
    ])
    for k, v in feasibility["data_sources"].items():
        md_lines.append(f"- **{k}:** {'YES' if v else 'NO'}")

    md_lines.extend([
        "",
        f"### Expected Risk: {feasibility['expected_risk']}",
        "",
        f"### Expected Validation: {feasibility['expected_validation_path']}",
        "",
        "### Why Other Fixes Rejected",
        "",
    ])
    for key, reason in feasibility["why_other_fixes_rejected"].items():
        md_lines.append(f"- **{FIX_CLASSIFICATIONS[key]}:** {reason}")

    md_lines.extend([
        "",
        "### Fix Classification Options",
        "",
        "| ID | Classification |",
        "|----|---------------|",
    ])
    for key, name in FIX_CLASSIFICATIONS.items():
        marker = " [SELECTED]" if key == feasibility["conclusion"] else ""
        md_lines.append(f"| **{key}** | {name}{marker} |")

    md_path = dataset_dir / "fix_feasibility_gate.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Fix feasibility gate complete.")
    print(f"  Classification: {feasibility['classification']}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return feasibility


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit K1 augmented fix feasibility")
    parser.add_argument("--dataset-dir", type=str, default=None)
    args = parser.parse_args()
    audit_fix_feasibility(Path(args.dataset_dir) if args.dataset_dir else None)


if __name__ == "__main__":
    main()
