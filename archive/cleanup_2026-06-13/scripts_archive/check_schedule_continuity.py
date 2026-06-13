"""Schedule continuity check for continuous low-height sagittal authority fix.

Generates schedule_continuity_check.csv proving the smoothstep k_position
schedule is continuous from 0.300 to 0.480m using a 181-point dense sweep,
plus explicit clamp check rows at 0.280 and 0.500.

Usage:
    python scripts/check_schedule_continuity.py
"""

import csv
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    smoothstep01,
    scheduled_k_position,
)

OUTPUT_DIR = Path("outputs/continuous_low_height_sagittal_authority_fix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "candidate_E1_k60": {"k_nominal": 40.0, "k_low_max": 60.0},
    "candidate_E2_k80": {"k_nominal": 40.0, "k_low_max": 80.0},
    "candidate_E3_k100": {"k_nominal": 40.0, "k_low_max": 100.0},
}

Z_LOW = 0.300
Z_HIGH = 0.393
# 181-point dense sweep from 0.300 to 0.480
HEIGHT_DENSE = list(np.linspace(0.300, 0.480, 181).tolist())
# Clamp check points - evaluate outside the dense range
CLAMP_CHECK_HEIGHTS = [0.280, 0.500]


def main():
    output_path = OUTPUT_DIR / "schedule_continuity_check.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate", "sample_type", "z_ref_m", "effective_k_position",
            "delta_k_position_per_step", "k_position_schedule_u",
            "k_position_schedule_smoothstep", "schedule_active"
        ])

        for cand_name, params in CANDIDATES.items():
            prev_k = None

            # --- Dense sweep rows (181 points from 0.300 to 0.480) ---
            for z in HEIGHT_DENSE:
                k_pos = scheduled_k_position(
                    z, params["k_nominal"], params["k_low_max"], Z_LOW, Z_HIGH
                )
                u_raw = (Z_HIGH - z) / (Z_HIGH - Z_LOW)
                u = max(0.0, min(1.0, u_raw))
                s = smoothstep01(u)
                delta_k = k_pos - prev_k if prev_k is not None else 0.0
                schedule_active = s > 1e-6

                writer.writerow([
                    cand_name, "dense", f"{z:.6f}", f"{k_pos:.8f}",
                    f"{delta_k:.10f}", f"{u:.8f}",
                    f"{s:.8f}", schedule_active
                ])
                prev_k = k_pos

            # --- Clamp check rows (evaluate at z < z_low and z > z_high) ---
            for z_clamp in CLAMP_CHECK_HEIGHTS:
                k_pos = scheduled_k_position(
                    z_clamp, params["k_nominal"], params["k_low_max"], Z_LOW, Z_HIGH
                )
                u_raw = (Z_HIGH - z_clamp) / (Z_HIGH - Z_LOW)
                u = max(0.0, min(1.0, u_raw))
                s = smoothstep01(u)
                schedule_active = s > 1e-6
                writer.writerow([
                    cand_name, "clamp_check", f"{z_clamp:.6f}", f"{k_pos:.8f}",
                    "0.0000000000", f"{u:.8f}",
                    f"{s:.8f}", schedule_active
                ])

    print(f"Continuity check written to {output_path}")

    # Analyze continuity from CSV
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = {}
    for cand_name, params in CANDIDATES.items():
        cand_rows = [r for r in rows if r["candidate"] == cand_name]
        dense_rows = [r for r in cand_rows if r["sample_type"] == "dense"]
        clamp_rows = [r for r in cand_rows if r["sample_type"] == "clamp_check"]

        # Delta across consecutive DENSE rows only
        deltas = [abs(float(r["delta_k_position_per_step"])) for r in dense_rows[1:]]
        max_abs_delta = max(deltas) if deltas else 0.0

        # Check monotonic decrease in transition band [z_low, z_high] in dense rows
        in_band = [r for r in dense_rows if Z_LOW <= float(r["z_ref_m"]) <= Z_HIGH]
        k_in_band = [float(r["effective_k_position"]) for r in in_band]
        monotonic = all(k_in_band[i] >= k_in_band[i+1] for i in range(len(k_in_band)-1))

        # Check constant above z_high in dense rows
        above_high = [r for r in dense_rows if float(r["z_ref_m"]) > Z_HIGH]
        k_above_high = [float(r["effective_k_position"]) for r in above_high]
        constant_above = all(abs(k - params["k_nominal"]) < 1e-6 for k in k_above_high) if k_above_high else True

        # Check clamp at z=0.280 (z < z_low) from clamp_check row
        z_0280_row = next((r for r in clamp_rows if float(r["z_ref_m"]) == 0.280), None)
        k_at_0280 = float(z_0280_row["effective_k_position"]) if z_0280_row else None
        clamp_below_ok = (
            k_at_0280 is not None and abs(k_at_0280 - params["k_low_max"]) < 1e-6
        ) if k_at_0280 is not None else False

        # Check clamp at z=0.500 (z > z_high) from clamp_check row
        z_0500_row = next((r for r in clamp_rows if float(r["z_ref_m"]) == 0.500), None)
        k_at_0500 = float(z_0500_row["effective_k_position"]) if z_0500_row else None
        clamp_above_ok = (
            k_at_0500 is not None and abs(k_at_0500 - params["k_nominal"]) < 1e-6
        ) if k_at_0500 is not None else False

        clamp_check_verified = clamp_below_ok and clamp_above_ok

        results[cand_name] = {
            "max_abs_delta_k_position": max_abs_delta,
            "no_discontinuity": max_abs_delta < (params["k_low_max"] - params["k_nominal"]) * 0.1,
            "monotonic_decrease_low_to_high": monotonic,
            "constant_k_nominal_above_z_high": constant_above,
            "constant_k_low_max_below_z_low": clamp_below_ok,
            "clamp_check_verified": clamp_check_verified,
            "k_at_0.280": k_at_0280,
            "k_at_0.500": k_at_0500,
        }

        print(f"\n{cand_name}:")
        print(f"  max_abs_delta_k_position = {max_abs_delta:.8f}")
        print(f"  no_discontinuity = {results[cand_name]['no_discontinuity']}")
        print(f"  monotonic_decrease_low_to_high = {monotonic}")
        print(f"  constant_k_nominal_above_z_high = {constant_above}")
        print(f"  constant_k_low_max_below_z_low = {clamp_below_ok}")
        print(f"  k_at_0.280 = {k_at_0280} (expected {params['k_low_max']})")
        print(f"  k_at_0.500 = {k_at_0500} (expected {params['k_nominal']})")
        print(f"  clamp_check_verified = {clamp_check_verified}")

    all_pass = all(
        r["no_discontinuity"] and r["monotonic_decrease_low_to_high"]
        and r["constant_k_nominal_above_z_high"] and r["clamp_check_verified"]
        for r in results.values()
    )

    if all_pass:
        print("\nPASS: All candidates have continuous, monotonic k_position schedules with verified clamps.")
    else:
        print("\nFAIL: Some candidates have discontinuous or non-monotonic schedules, or clamp checks failed.")


if __name__ == "__main__":
    main()
