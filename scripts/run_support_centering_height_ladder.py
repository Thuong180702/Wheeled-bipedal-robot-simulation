"""Support-Centered Height Ladder Runner — Phase 9B/9C.

Runs support_centering_bias_trim (formerly T6J_centering_bias_trim) across all
available height variant setups, saving telemetry and summaries to per-variant
output directories.

Uses subprocess.run() with arg lists to avoid shell quoting issues.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
MANIFEST_PATH = OUT_BASE / "t6j_height_ladder_setup_manifest.json"
STEPS = 2000
PROFILE = "support_centering_bias_trim"

DESIRED_LABELS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340",
    "low_0p360", "low_0p380",
    "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]


def discover_setups():
    """Find all *_setup.json files and match to labels."""
    setups = {}
    for p in sorted(SETUP_DIR.glob("*_setup.json")):
        label = p.stem.replace("_setup", "")
        setups[label] = p
    return setups


def build_manifest_entries(setups):
    entries = []
    for label in DESIRED_LABELS:
        setup_path = setups.get(label)
        entry = {
            "label": label,
            "setup_path": str(setup_path) if setup_path else "",
            "exists": setup_path is not None and setup_path.exists(),
            "launch_status": "pending",
            "return_code": None,
            "telemetry_path": "",
            "summary_path": "",
            "notes": "",
        }
        if not entry["exists"]:
            entry["launch_status"] = "skipped_missing"
            entry["notes"] = "Setup file not found"
        entries.append(entry)
    return entries


def run_variant(entry, manifest_path):
    """Run a single height variant with T6J profile."""
    label = entry["label"]
    setup_path = entry["setup_path"]
    out_dir = OUT_BASE / f"t6j_height_ladder_2000_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RUN {label}: {setup_path}")
    print(f"Output dir: {out_dir}")
    print(f"{'='*60}")

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
    ]

    # Run with output capture
    result = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max per variant
    )

    entry["return_code"] = result.returncode
    print(f"Return code: {result.returncode}")

    if result.returncode != 0:
        entry["launch_status"] = "failed"
        # Save stderr for diagnosis
        err_path = out_dir / "stderr.txt"
        err_path.write_text(result.stderr or "")
        out_path = out_dir / "stdout.txt"
        out_path.write_text(result.stdout or "")
        entry["notes"] = f"rc={result.returncode}, stderr saved"
        print(f"FAILED {label} rc={result.returncode}")
        print(f"  stderr (first 500): {(result.stderr or '')[:500]}")
    else:
        entry["launch_status"] = "completed"
        # Find and copy telemetry + summary
        # The sim script writes to outputs/hierarchical_controller_sim/telemetry_*.csv
        sim_out = ROOT / "outputs" / "hierarchical_controller_sim"
        telemetry_files = sorted(sim_out.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        summary_files = sorted(sim_out.glob("run_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if telemetry_files:
            src = telemetry_files[0]
            dst = out_dir / "telemetry_2000.csv"
            shutil.copy2(src, dst)
            entry["telemetry_path"] = str(dst)
            print(f"  Telemetry copied: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

            # Clean up original to avoid confusion
            try:
                src.unlink()
            except OSError:
                pass
        else:
            entry["notes"] += " No telemetry file found."

        if summary_files:
            src = summary_files[0]
            dst = out_dir / "run_summary.json"
            shutil.copy2(src, dst)
            entry["summary_path"] = str(dst)
            print(f"  Summary copied: {dst}")

            try:
                src.unlink()
            except OSError:
                pass
        else:
            entry["notes"] += " No summary file found."

        # Also check for sidecar summary in sim_out
        sidecar_files = sorted(sim_out.glob("run_summary_sidecar_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if sidecar_files:
            src = sidecar_files[0]
            dst = out_dir / "run_summary_sidecar.json"
            shutil.copy2(src, dst)
            print(f"  Sidecar copied: {dst}")
            try:
                src.unlink()
            except OSError:
                pass

        print(f"COMPLETED {label}")

    # Update manifest incrementally
    save_manifest(manifest_path)

    return entry


def save_manifest(path):
    """Save current manifest state."""
    with open(path, "w") as f:
        json.dump(MANIFEST, f, indent=2)
    print(f"  Manifest updated: {path}")


# Global manifest
MANIFEST = []


def main():
    global MANIFEST

    setups = discover_setups()
    print(f"Discovered {len(setups)} setup files:")
    for label, path in sorted(setups.items()):
        print(f"  {label}: {path}")

    MANIFEST = build_manifest_entries(setups)
    print(f"\nManifest has {len(MANIFEST)} entries, {sum(1 for e in MANIFEST if e['exists'])} with existing setups")

    # Run each variant
    for entry in MANIFEST:
        if not entry["exists"]:
            print(f"\nSKIP {entry['label']}: setup missing")
            continue
        try:
            run_variant(entry, MANIFEST_PATH)
        except subprocess.TimeoutExpired:
            entry["launch_status"] = "timeout"
            entry["notes"] = "Subprocess timed out after 600s"
            print(f"TIMEOUT {entry['label']}")
            save_manifest(MANIFEST_PATH)
        except Exception as ex:
            entry["launch_status"] = "error"
            entry["notes"] = str(ex)
            print(f"ERROR {entry['label']}: {ex}")
            save_manifest(MANIFEST_PATH)

    # Final summary
    completed = sum(1 for e in MANIFEST if e["launch_status"] == "completed")
    failed = sum(1 for e in MANIFEST if e["launch_status"] in ("failed", "error", "timeout"))
    skipped = sum(1 for e in MANIFEST if e["launch_status"] == "skipped_missing")

    print(f"\n{'='*60}")
    print(f"HEIGHT LADDER COMPLETE")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"{'='*60}")

    save_manifest(MANIFEST_PATH)


if __name__ == "__main__":
    main()
