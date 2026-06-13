#!/usr/bin/env python3
"""
Phase 1: WBC / Torque Ownership Invariant Audit for low_0p300

Verifies whether WBC torques are actually applied or only computed diagnostically.
This is critical because balance-core mode should have WBC OFF.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs/low_0p300_initialization_contact_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_low_0p300_30_steps():
    """Run low_0p300 baseline for 30 steps with full telemetry."""

    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--height-variant-setup", "outputs/physical_target_height_setups/low_0p300_setup.json",
        "--steps", "30",
        "--vd-sagittal-authority-profile", "baseline",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"[ERROR] Simulation failed:")
        print(result.stderr)
        return None

    # Find most recent telemetry
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    telemetry_files = sorted(sim_output_dir.glob("telemetry_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)

    if not telemetry_files:
        print(f"[ERROR] No telemetry found")
        return None

    telemetry_path = telemetry_files[0]
    print(f"[OK] Found telemetry: {telemetry_path}")

    # Copy to audit directory
    import shutil
    audit_telemetry = OUTPUT_DIR / "low_0p300_first_30_steps_telemetry.csv"
    shutil.copy(telemetry_path, audit_telemetry)

    return audit_telemetry


def audit_wbc_ownership(telemetry_path):
    """Audit WBC application and ownership from telemetry."""

    df = pd.read_csv(telemetry_path)

    audit = {
        "telemetry_path": str(telemetry_path),
        "total_steps": len(df),
        "terminated": bool(df["terminated"].iloc[-1]) if "terminated" in df.columns else False,
        "termination_reason": df["termination_reason"].iloc[-1] if "termination_reason" in df.columns else "N/A",
        "wbc_fields_present": {},
        "ownership_fields_present": {},
        "wbc_analysis": {},
        "ownership_analysis": {},
        "torque_analysis": {},
        "contact_analysis": {},
        "conclusion": "UNKNOWN",
    }

    # Check WBC-related fields
    wbc_fields = [
        "tau_wbc_norm",
        "tau_wbc_max",
        "tau_posture_max",
        "tau_total_max",
        "qp_converged",
        "wrench_error_norm",
    ]

    for field in wbc_fields:
        audit["wbc_fields_present"][field] = field in df.columns
        if field in df.columns:
            audit["wbc_analysis"][field + "_max"] = float(df[field].max())
            audit["wbc_analysis"][field + "_mean"] = float(df[field].mean())

    # Check ownership fields
    ownership_fields = [
        "ownership_violation_count",
        "hidden_torque_norm",
    ]

    for field in ownership_fields:
        audit["ownership_fields_present"][field] = field in df.columns
        if field in df.columns:
            if df[field].dtype == 'bool':
                audit["ownership_analysis"][field + "_any"] = bool(df[field].any())
            else:
                audit["ownership_analysis"][field + "_max"] = float(df[field].max())

    # Analyze torques
    if "tau_wbc_norm" in df.columns:
        wbc_norm_max = df["tau_wbc_norm"].max()
        wbc_nonzero_count = (df["tau_wbc_norm"] > 0.01).sum()

        audit["torque_analysis"]["wbc_norm_max"] = float(wbc_norm_max)
        audit["torque_analysis"]["wbc_nonzero_steps"] = int(wbc_nonzero_count)
        audit["torque_analysis"]["wbc_nonzero_percent"] = float(100.0 * wbc_nonzero_count / len(df))

    if "tau_total_max" in df.columns:
        audit["torque_analysis"]["tau_total_max"] = float(df["tau_total_max"].max())

    # Analyze contact
    if "active_wheels" in df.columns:
        audit["contact_analysis"]["active_wheels_initial"] = int(df["active_wheels"].iloc[0])
        audit["contact_analysis"]["active_wheels_final"] = int(df["active_wheels"].iloc[-1])
        audit["contact_analysis"]["contact_lost_step"] = None

        for i, val in enumerate(df["active_wheels"]):
            if val == 0:
                audit["contact_analysis"]["contact_lost_step"] = int(i)
                break

    if "contact_force_valid" in df.columns:
        audit["contact_analysis"]["contact_valid_initial"] = bool(df["contact_force_valid"].iloc[0])
        audit["contact_analysis"]["contact_valid_percent"] = float(100.0 * df["contact_force_valid"].mean())

    # Determine conclusion
    if "tau_wbc_norm" in df.columns:
        wbc_norm_max = df["tau_wbc_norm"].max()
        if wbc_norm_max > 1.0:
            # WBC torques are being computed
            # But are they applied?
            # In balance-core mode, WBC should be diagnostic only
            # Check if there's evidence of WBC being added to control
            audit["conclusion"] = "WBC_COMPUTED_HIGH"
            audit["wbc_status"] = "WBC torques computed (max={:.2f} Nm). Need to verify if applied or diagnostic only.".format(wbc_norm_max)
        else:
            audit["conclusion"] = "WBC_NEAR_ZERO"
            audit["wbc_status"] = "WBC torques near zero (max={:.2f} Nm). Likely not applied.".format(wbc_norm_max)
    else:
        audit["conclusion"] = "WBC_FIELDS_MISSING"
        audit["wbc_status"] = "Cannot determine WBC status - telemetry fields missing"

    return audit


def create_report(audit):
    """Create markdown report."""

    report_lines = [
        "# Phase 1: WBC / Torque Ownership Audit for low_0p300",
        "",
        "## Executive Summary",
        "",
        f"**Conclusion:** {audit['conclusion']}",
        f"**WBC Status:** {audit.get('wbc_status', 'Unknown')}",
        "",
        "## Simulation Details",
        "",
        f"- Telemetry: `{audit['telemetry_path']}`",
        f"- Total steps: {audit['total_steps']}",
        f"- Terminated: {audit['terminated']}",
        f"- Termination reason: {audit['termination_reason']}",
        "",
        "## WBC Fields Present",
        "",
    ]

    for field, present in audit["wbc_fields_present"].items():
        status = "[OK]" if present else "[MISSING]"
        report_lines.append(f"- {status} `{field}`")

    report_lines.extend([
        "",
        "## WBC Analysis",
        "",
    ])

    for key, val in audit["wbc_analysis"].items():
        report_lines.append(f"- `{key}`: {val:.4f}")

    report_lines.extend([
        "",
        "## Ownership Analysis",
        "",
    ])

    for key, val in audit["ownership_analysis"].items():
        report_lines.append(f"- `{key}`: {val}")

    report_lines.extend([
        "",
        "## Torque Analysis",
        "",
    ])

    for key, val in audit["torque_analysis"].items():
        report_lines.append(f"- `{key}`: {val}")

    report_lines.extend([
        "",
        "## Contact Analysis",
        "",
    ])

    for key, val in audit["contact_analysis"].items():
        report_lines.append(f"- `{key}`: {val}")

    report_lines.extend([
        "",
        "## Next Steps",
        "",
        "If WBC torques are being applied:",
        "- **STOP IMMEDIATELY**",
        "- Fix WBC routing/invariant before any dynamics conclusions",
        "- Balance-core mode should have WBC OFF",
        "",
        "If WBC is diagnostic only:",
        "- Proceed to Phase 2: Static setup validation",
        "",
    ])

    return "\n".join(report_lines)


def main():
    print("\n" + "="*80)
    print("Phase 1: WBC / Torque Ownership Audit for low_0p300")
    print("="*80 + "\n")

    # Run simulation
    telemetry_path = run_low_0p300_30_steps()

    if telemetry_path is None:
        print("\n[ERROR] Failed to run simulation")
        return 1

    # Audit WBC and ownership
    audit = audit_wbc_ownership(telemetry_path)

    # Save audit JSON
    audit_json_path = OUTPUT_DIR / "low_0p300_wbc_ownership_audit.json"
    with open(audit_json_path, 'w') as f:
        json.dump(audit, f, indent=2)
    print(f"\n[OK] Audit saved: {audit_json_path}")

    # Create report
    report = create_report(audit)
    report_path = OUTPUT_DIR / "low_0p300_wbc_ownership_audit_report.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"[OK] Report saved: {report_path}")

    # Print summary
    print("\n" + "="*80)
    print("Audit Summary")
    print("="*80)
    print(f"Conclusion: {audit['conclusion']}")
    print(f"WBC Status: {audit.get('wbc_status', 'Unknown')}")
    print(f"Steps completed: {audit['total_steps']}")
    print(f"Contact lost at step: {audit['contact_analysis'].get('contact_lost_step', 'N/A')}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
