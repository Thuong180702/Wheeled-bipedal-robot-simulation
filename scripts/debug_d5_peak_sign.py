"""Investigate mode_div sign at D5 peak step."""

from __future__ import annotations

import csv
from pathlib import Path


def main():
    path = (
        "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high"
        "/D_baseline/telemetry_1782210164.csv"
    )
    with open(path) as f:
        rows = list(csv.DictReader(f))

    hy_vals = [float(r["hip_yaw_abs_max"]) for r in rows]
    peak_idx = max(range(len(hy_vals)), key=lambda i: hy_vals[i])
    r = rows[peak_idx]

    print(f"Peak at step {peak_idx}:")
    print(f"  hip_yaw_abs_max:            {r.get('hip_yaw_abs_max')}")
    print(f"  mode_hip_yaw_div_tau_left:  {r.get('mode_hip_yaw_div_tau_left')}")
    print(f"  mode_hip_yaw_div_tau_right: {r.get('mode_hip_yaw_div_tau_right')}")
    print(f"  mode_hip_yaw_div_error:     {r.get('mode_hip_yaw_div_error')}")
    print(f"  mode_hip_yaw_div_rate:      {r.get('mode_hip_yaw_div_rate')}")
    print(f"  mode_hip_yaw_div_height_gate: {r.get('mode_hip_yaw_div_height_gate')}")
    print(f"  mode_hip_yaw_div_tau_left_raw: {r.get('mode_hip_yaw_div_tau_left_raw')}")
    print(f"  mode_hip_yaw_div_tau_left_sat: {r.get('mode_hip_yaw_div_tau_left_sat')}")
    print(f"  current_com_z_m:            {r.get('current_com_z_m')}")
    print(f"  l_hip_yaw_error:            {r.get('l_hip_yaw_error')}")
    print(f"  r_hip_yaw_error:            {r.get('r_hip_yaw_error')}")
    print(f"  hip_yaw_common_error_rad:   {r.get('hip_yaw_common_error_rad')}")
    print(f"  hip_yaw_divergence_error_rad: {r.get('hip_yaw_divergence_error_rad')}")
    print(f"  euler_yaw_z:                {r.get('euler_yaw_z')}")
    print(f"  yaw_controller_tau_hip_yaw_left: {r.get('yaw_controller_tau_hip_yaw_left')}")
    print(f"  l_hip_yaw_tau_shape_final:  {r.get('l_hip_yaw_tau_shape_final')}")
    print(f"  support_position_error_scaled_m: {r.get('support_position_error_scaled_m')}")
    print(f"  pitch_error:                {r.get('pitch_error')}")

    # Context around peak
    print("\nContext +/-10 steps around peak:")
    for i in range(max(0, peak_idx - 10), min(len(rows), peak_idx + 11)):
        row = rows[i]
        h = float(row["hip_yaw_abs_max"])
        e = float(row.get("mode_hip_yaw_div_error", 0))
        t = float(row.get("mode_hip_yaw_div_tau_left", 0))
        g = float(row.get("mode_hip_yaw_div_height_gate", 1))
        tr = float(row.get("mode_hip_yaw_div_tau_left_raw", 0))
        sign_ok = (e * t <= 0) if abs(e) > 1e-9 else True
        sat = "SAT" if row.get("mode_hip_yaw_div_tau_left_sat", "False") == "True" else ""
        print(f"  step {i:3d}: hy={h:.4f} err={e:+.4f} tau={t:+.5f} raw={tr:+.5f} gate={g:.3f} sign={'OK' if sign_ok else 'WRONG'} {sat}")


if __name__ == "__main__":
    main()
