"""Check yaw-controller and mode-div torque sign alignment at peak D5 step."""

from __future__ import annotations

import csv
from pathlib import Path

FILES: list[tuple[str, str]] = [
    (
        "D5_D_baseline",
        "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high/D_baseline/telemetry_1782210164.csv",
    ),
    (
        "D5_F6_sg050",
        "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg50_D5/telemetry_1782217344.csv",
    ),
    (
        "D5_F8_kp30",
        "outputs/mode_divergence_authority_limit_sweep/d4_quick/F8_kp30_D5/telemetry_1782217922.csv",
    ),
]


def main():
    for label, path in FILES:
        p = Path(path)
        if not p.exists():
            print(f"\n=== {label}: FILE NOT FOUND ===")
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))

        hy_vals = [float(r["hip_yaw_abs_max"]) for r in rows]
        peak_idx = max(range(len(hy_vals)), key=lambda i: hy_vals[i])
        r = rows[peak_idx]

        yaw_left = float(r.get("yaw_controller_tau_hip_yaw_left", 0))
        yaw_right = float(r.get("yaw_controller_tau_hip_yaw_right", 0))
        md_left = float(r.get("mode_hip_yaw_div_tau_left", 0))
        md_right = float(r.get("mode_hip_yaw_div_tau_right", 0))
        l_error = float(r.get("l_hip_yaw_error", 0))
        r_error = float(r.get("r_hip_yaw_error", 0))
        common = float(r.get("hip_yaw_common_error_rad", 0))
        divergence = float(r.get("hip_yaw_divergence_error_rad", 0))
        body_yaw = float(r.get("euler_yaw_z", 0))
        pitch = float(r.get("pitch_error", 0)) * 180 / 3.14159
        sup = float(r.get("support_position_error_scaled_m", 0))
        step = int(r.get("step", peak_idx))

        print(f"\n{'='*60}")
        print(f"  {label} at step {step} (peak hy={hy_vals[peak_idx]:.4f})")
        print(f"{'='*60}")
        print(f"  l_error={l_error:.4f}  r_error={r_error:.4f}")
        print(f"  common={common:.4f}  divergence={divergence:.4f}")
        print(f"  body_yaw={body_yaw:.4f}")
        print(f"  yaw_left={yaw_left:+.3f}  yaw_right={yaw_right:+.3f}")
        print(f"  md_left={md_left:+.3f}  md_right={md_right:+.3f}")
        print(f"  yaw+md_left={yaw_left+md_left:+.3f}  yaw+md_right={yaw_right+md_right:+.3f}")
        print(f"  pitch={pitch:.2f} deg  sup={sup:.3f}")

        # Check sign alignment - both use antisymmetric convention
        yaw_left_sign = "negative" if yaw_left < -0.01 else "positive" if yaw_left > 0.01 else "zero"
        md_left_sign = "negative" if md_left < -0.01 else "positive" if md_left > 0.01 else "zero"
        yaw_sign_same = (yaw_left < 0) == (md_left < 0)
        print(f"  yaw_left_sign={yaw_left_sign}  md_left_sign={md_left_sign}")
        print(f"  yaw+mode-div SAME direction? {yaw_sign_same}")

        # Sign verification: mode-div should OPPOSE divergence error
        # Positive divergence (left ahead) -> left torque negative, right torque positive
        div_sign_correct = (divergence > 0.001 and md_left <= 0) or (divergence < -0.001 and md_left >= 0) or abs(divergence) < 0.001
        print(f"  mode-div sign correct (opposes divergence)? {div_sign_correct}")

        # Check if yaw-controller torque direction opposes or reinforces mode-div
        # Yaw controller: body_yaw negative -> left should be positive (oppose)
        # If yaw_left and md_left have same sign, they cooperate
        print(f"  => Yaw and Mode-Div are {'COOPERATING' if yaw_sign_same else 'FIGHTING'} at peak")


if __name__ == "__main__":
    main()
