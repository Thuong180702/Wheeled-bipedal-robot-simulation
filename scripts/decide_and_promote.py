"""Decide whether low-band v2 passes all promotion gates and, if so, promote it.

Gates checked (in order):
  1. Step D classification == PASS (no monitoring, no fail)
  2. Step C recheck verdict == STEP_C_RECHECK_PASS
  3. Fixed-height recheck passes (no protected regression)

If all gates pass, the default/current-best references are updated.
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------- paths ----------
STEP_D_SUMMARY = ROOT / "outputs" / "step_d_all" / "step_d_summary.json"
STEP_C_RECHECK_REPORT = ROOT / "docs" / "validation" / "step_c_regression_recheck.md"
FIXED_HEIGHT_CSV = ROOT / "outputs" / "physics_ff_low_band_support_v2_tuning" / "full_fixed_height_metrics.csv"
DECISION_JSON = ROOT / "outputs" / "step_d_all" / "decision_summary.json"

# ---------- docs mentioning "current PFF" that refer to physics_equilibrium_feedforward_outer_loop ----------
DOCS_CURRENT_PFF = [
    ROOT / "docs" / "validation" / "step_d_validation_matrix.md",
    ROOT / "docs" / "validation" / "physics_ff_low_band_support_v2_tuning_report.md",
    ROOT / "docs" / "validation" / "physics_ff_step_c_low_band_support_v1_full_step_c_report.md",
    ROOT / "docs" / "validation" / "physics_ff_step_c_low_band_support_fix_report.md",
    ROOT / "docs" / "validation" / "step_c_regression_recheck.md",
    ROOT / "docs" / "validation" / "physics_equilibrium_feedforward_500_sanity_report.md",
]

OLD_PROFILE_SHORT = "physics_equilibrium_feedforward_outer_loop"
NEW_PROFILE_SHORT = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
OLD_DISPLAY = "Current PFF"
NEW_DISPLAY = "Current PFF (low-band v2 promoted)"


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def load_step_d_summary(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check_step_c_recheck(path: pathlib.Path) -> bool:
    """Return True if the Step C recheck report contains STEP_C_RECHECK_PASS."""
    if not path.is_file():
        print("  MISSING: Step C recheck report not found", file=sys.stderr)
        return False
    text = path.read_text()
    return "STEP_C_RECHECK_PASS" in text


def check_fixed_height_recheck(path: pathlib.Path) -> dict:
    """Verify protected low heights have no regression.

    Returns dict with pass/fail per height and overall verdict.
    """
    if not path.is_file():
        return {"pass": False, "reason": "fixed-height CSV not found"}
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Find Low-band v2 rows for protected heights
    protected = {"low_0p320": 0.15, "low_0p330": 0.22, "low_0p360": 0.19, "high_0p480": 0.30}
    results = {}
    for row in rows:
        height = row.get("height", "").strip()
        profile = row.get("profile", "").strip()
        if profile == "Low-band v2" and height in protected:
            max_abs = float(row.get("maxabs", 0))
            fell = row.get("fell", "False").strip().lower() in ("true", "1")
            hip_yaw = float(row.get("hip_yaw", 0))
            wbc = int(row.get("WBC rows", 0))
            hidden = float(row.get("hidden", 0))
            owner = float(row.get("owner", 0))
            threshold = protected[height]
            h_pass = (
                not fell
                and max_abs <= threshold * 1.15  # allow 15% margin
                and hip_yaw < 0.35
                and wbc == 0
                and hidden == 0.0
                and owner == 0.0
            )
            results[height] = {
                "pass": h_pass,
                "max_abs": max_abs,
                "threshold_115pct": round(threshold * 1.15, 4),
                "fell": fell,
                "hip_yaw": hip_yaw,
            }
    all_pass = all(r["pass"] for r in results.values())
    return {"pass": all_pass, "per_height": results}


# ---------------------------------------------------------------------------
# Promotion (doc updates)
# ---------------------------------------------------------------------------

def promote_docs(dry_run: bool = True) -> list[str]:
    """Update docs that say 'Current PFF' / physics_equilibrium_feedforward_outer_loop.

    Returns a list of changed file paths.
    """
    changed = []
    for doc_path in DOCS_CURRENT_PFF:
        if not doc_path.is_file():
            continue
        text = doc_path.read_text(encoding="utf-8")

        # Strategy: replace profile-specific "current PFF" references.
        # We update the definition sentence near the top that names the profile.
        new_text = text

        # 1) Update "Current PFF: physics_equilibrium_feedforward_outer_loop"
        #    to "Current PFF: physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
        #    This handles bullet-point definitions in matrix docs.
        new_text = re.sub(
            r"(Current PFF)[:\s]*`?" + re.escape(OLD_PROFILE_SHORT) + r"`?",
            r"\1 (low-band v2 promoted): `" + NEW_PROFILE_SHORT + r"`",
            new_text,
        )

        # 2) In tables, "Current PFF" in the profile column stays as is (it's a label)
        #    but the table rows that compare "Current PFF" as baseline should remain
        #    since they're historical records.

        # 3) Update "Current PFF rerun" or similar references
        new_text = new_text.replace(
            "Current PFF rerun",
            "Current PFF (v1, pre-promotion) rerun",
        )

        if new_text != text:
            if not dry_run:
                doc_path.write_text(new_text, encoding="utf-8")
            changed.append(str(doc_path))

    return changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decide promotion of low-band v2 and optionally promote."
    )
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Only check gates; do not write changes (default: true)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Override dry-run and actually promote")
    args = parser.parse_args()

    do_promote = args.force or not args.dry_run

    print("=" * 72)
    print("Promotion Gate Check: physics_equilibrium_feedforward_outer_loop_low_band_support_v2")
    print("=" * 72)

    # --- Gate 1: Step D ---
    if STEP_D_SUMMARY.is_file():
        sd = load_step_d_summary(STEP_D_SUMMARY)
        step_d_class = sd.get("classification", "MISSING")
        step_d_pass = step_d_class == "STEP_D_RANDOM_PUSH_PASS"
        print(f"\nGate 1 — Step D: {step_d_class}")
        print(f"  must_not_fall_pass: {sd.get('must_not_fall_pass')}")
        print(f"  any_hard_fail: {sd.get('any_hard_fail')}")
        print(f"  max_drift_C: {sd.get('max_drift_C'):.3f} m")
        print(f"  c_not_worse_count: {sd.get('c_not_worse_count')}/{sd.get('total_cases')}")
        print(f"  → {'PASS' if step_d_pass else 'FAIL'}")
    else:
        step_d_pass = False
        step_d_class = "NO_DATA"
        print(f"\nGate 1 — Step D: NO_DATA (run step_d_all first)")
        print(f"  → FAIL (cannot check)")

    # --- Gate 2: Step C recheck ---
    step_c_pass = check_step_c_recheck(STEP_C_RECHECK_REPORT)
    print(f"\nGate 2 — Step C recheck: {'PASS' if step_c_pass else 'FAIL'}")

    # --- Gate 3: Fixed-height recheck ---
    fh = check_fixed_height_recheck(FIXED_HEIGHT_CSV)
    fixed_height_pass = fh.get("pass", False)
    print(f"\nGate 3 — Fixed-height recheck: {'PASS' if fixed_height_pass else 'FAIL'}")
    for h, r in fh.get("per_height", {}).items():
        print(f"  {h}: {'PASS' if r['pass'] else 'FAIL'} "
              f"(max_abs={r['max_abs']:.4f}, threshold={r['threshold_115pct']:.4f}, "
              f"fell={r['fell']})")

    # --- Overall ---
    all_pass = step_d_pass and step_c_pass and fixed_height_pass
    print(f"\n{'=' * 72}")
    print(f"Overall: {'ALL GATES PASS' if all_pass else 'GATES NOT MET'}")
    print(f"  Step D:         {'✅' if step_d_pass else '❌'} {step_d_class}")
    print(f"  Step C recheck: {'✅' if step_c_pass else '❌'}")
    print(f"  Fixed-height:   {'✅' if fixed_height_pass else '❌'}")

    # --- Decision ---
    if all_pass:
        if do_promote:
            changed = promote_docs(dry_run=False)
            classification = "PHYSICS_FF_LOW_BAND_V2_STEP_D_PASS_AND_PROMOTED_DEFAULT"
            print(f"\n✅ PROMOTION EXECUTED")
            print(f"   Files updated: {len(changed)}")
            for f in changed:
                print(f"   - {f}")
        else:
            classification = "PHYSICS_FF_LOW_BAND_V2_STEP_D_PASS_NOT_PROMOTED"
            print(f"\n⚠️  All gates pass but dry-run mode — not promoting.")
            print(f"   Re-run with --force to promote.")
            changed_candidates = promote_docs(dry_run=True)
            print(f"   Docs that would change: {len(changed_candidates)}")
            for f in changed_candidates:
                print(f"   - {f}")
    elif step_d_pass and not (step_c_pass and fixed_height_pass):
        classification = "STEP_D_PASS_PROMOTION_BLOCKED_BY_REGRESSION_RECHECK"
        print(f"\n❌ Step D passed but regression recheck(s) failed.")
    elif not step_d_pass:
        classification = "PHYSICS_FF_LOW_BAND_V2_STEP_D_FAIL"
        print(f"\n❌ Step D failed.")
    else:
        classification = "PHYSICS_FF_LOW_BAND_V2_STEP_D_INCONCLUSIVE"
        print(f"\n❓ Inconclusive.")

    print(f"\nClassification: {classification}")

    # --- Write decision JSON ---
    decision = {
        "classification": classification,
        "gates": {
            "step_d_pass": step_d_pass,
            "step_d_classification": step_d_class,
            "step_c_recheck_pass": step_c_pass,
            "fixed_height_recheck_pass": fixed_height_pass,
        },
        "fixed_height_details": {
            h: r for h, r in fh.get("per_height", {}).items()
        },
        "promotion_executed": do_promote if all_pass else False,
    }
    DECISION_JSON.parent.mkdir(parents=True, exist_ok=True)
    DECISION_JSON.write_text(json.dumps(decision, indent=2))
    print(f"\nDecision JSON: {DECISION_JSON}")


if __name__ == "__main__":
    main()
