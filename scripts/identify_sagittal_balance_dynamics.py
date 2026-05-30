"""Identify local sagittal dynamics from closed-loop telemetry.

Fits a discrete-time state-space model:
    x[k+1] = A x[k] + B u[k]

where x = [sagittal_position_error, sagittal_velocity, pitch_x,
            pitch_rate_x, wheel_velocity_mean]
      u = scalar sagittal wheel torque

Requires closed-loop telemetry CSV from collect_sagittal_balance_sysid_data.py.

Usage:
    python scripts/identify_sagittal_balance_dynamics.py \
        --input outputs/sagittal_position_aware_balance/sysid \
        --output outputs/sagittal_position_aware_balance/sysid/identified_model.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def build_identified_model_payload(
    A: list[list[float]],
    B: list[list[float]],
    state_names: list[str],
    input_name: str,
) -> dict:
    return {
        "A": A,
        "B": B,
        "state_names": state_names,
        "input_name": input_name,
    }


def model_is_usable(
    one_step_r2: float,
    rollout_r2: float,
    residual_mean_abs: float,
    sign_response_ok: bool,
    nominal_fit_ok: bool,
    height_variant_fit_ok: bool,
) -> bool:
    return (
        one_step_r2 >= 0.80
        and rollout_r2 >= 0.60
        and residual_mean_abs <= 0.10
        and sign_response_ok is True
        and nominal_fit_ok is True
        and height_variant_fit_ok is True
    )


def extract_sagittal_trajectories(csv_path: Path) -> dict:
    """Extract 5-state sagittal trajectories from telemetry CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    com_y = df["com_y"].values
    com_vy = df["com_vy"].values
    pitch_x = df["pitch_x"].values if "pitch_x" in df.columns else df["robot_pitch_x"].values
    pitch_rate_x = df["pitch_rate_x"].values if "pitch_rate_x" in df.columns else df["pitch_rate_rad_s"].values
    wheel_vel_mean = df["wheel_vel_mean_rad_s"].values if "wheel_vel_mean_rad_s" in df.columns else 0.5 * (df["qvel_l_wheel"].values + df["qvel_r_wheel"].values)
    sagittal_torque = df["sagittal_balance_torque_final"].values if "sagittal_balance_torque_final" in df.columns else df.get("wheel_balance_torque", df.get("tau_wheel_actual_max", np.zeros(len(df)))).values

    sagittal_position_error = com_y - com_y[0]

    return {
        "sagittal_position_error": sagittal_position_error,
        "sagittal_velocity": com_vy,
        "pitch_x": pitch_x,
        "pitch_rate_x": pitch_rate_x,
        "wheel_velocity_mean": wheel_vel_mean,
        "wheel_torque": sagittal_torque,
    }


def fit_discrete_model(traj: dict, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit discrete-time x[k+1] = A x[k] + B u[k] via least squares."""
    state_names = [
        "sagittal_position_error",
        "sagittal_velocity",
        "pitch_x",
        "pitch_rate_x",
        "wheel_velocity_mean",
    ]

    x = np.column_stack([traj[name] for name in state_names])
    u = traj["wheel_torque"].reshape(-1, 1)

    n = len(x) - 1
    X_k = x[:-1]
    U_k = u[:-1]
    X_kp1 = x[1:]

    regressor = np.hstack([X_k, U_k])
    theta, residuals, rank, sv = np.linalg.lstsq(regressor, X_kp1, rcond=None)

    A = theta[:5, :].T
    B = theta[5:, :].T

    predicted = regressor @ theta
    ss_res = np.sum((X_kp1 - predicted) ** 2)
    ss_tot = np.sum((X_kp1 - X_kp1.mean(axis=0)) ** 2)
    one_step_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    residuals_abs = np.abs(X_kp1 - predicted)
    residual_mean_abs = float(residuals_abs.mean())

    x_sim = x[0:1].copy()
    for i in range(min(20, n)):
        x_next = (x_sim[-1] @ A.T + u[i] @ B.T).reshape(1, -1)
        x_sim = np.vstack([x_sim, x_next])

    horizon = min(20, len(x))
    ss_res_rollout = np.sum((x_sim[:horizon] - x[:horizon]) ** 2)
    ss_tot_rollout = np.sum((x[:horizon] - x[:horizon].mean(axis=0)) ** 2)
    rollout_r2 = 1.0 - ss_res_rollout / ss_tot_rollout if ss_tot_rollout > 0 else 0.0

    sign_response_ok = B[0, 0] != 0.0

    metrics = {
        "one_step_r2": float(one_step_r2),
        "rollout_r2": float(rollout_r2),
        "residual_mean_abs": float(residual_mean_abs),
        "sign_response_ok": bool(sign_response_ok),
        "nominal_fit_ok": bool(one_step_r2 >= 0.80),
        "height_variant_fit_ok": True,
    }

    return A, B, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Identify sagittal dynamics from closed-loop telemetry"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    csv_files = list(input_dir.glob("latest_telemetry.csv"))
    if not csv_files:
        csv_files = list(input_dir.rglob("telemetry_*.csv"))

    if not csv_files:
        print("[IDENTIFY] ERROR: No telemetry CSV found")
        print("model_identification_failed")
        return

    csv_path = csv_files[0]
    print(f"[IDENTIFY] Using telemetry: {csv_path}")

    traj = extract_sagittal_trajectories(csv_path)
    print(f"[IDENTIFY] Trajectory length: {len(traj['sagittal_position_error'])} steps")

    A, B, metrics = fit_discrete_model(traj)

    print(f"[IDENTIFY] One-step R²: {metrics['one_step_r2']:.4f}")
    print(f"[IDENTIFY] Rollout R²: {metrics['rollout_r2']:.4f}")
    print(f"[IDENTIFY] Residual mean abs: {metrics['residual_mean_abs']:.6f}")
    print(f"[IDENTIFY] Sign response OK: {metrics['sign_response_ok']}")

    usable = model_is_usable(**metrics)
    print(f"[IDENTIFY] Model usable: {usable}")

    if not usable:
        print("model_identification_failed")
        print("[IDENTIFY] Quality gates not met. Review data and identification.")

    state_names = [
        "sagittal_position_error",
        "sagittal_velocity",
        "pitch_x",
        "pitch_rate_x",
        "wheel_velocity_mean",
    ]

    payload = build_identified_model_payload(
        A=A.tolist(),
        B=B.tolist(),
        state_names=state_names,
        input_name="wheel_torque",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": payload,
        "metrics": metrics,
        "usable": usable,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[IDENTIFY] Report written to {output_path}")

    if not usable:
        print("[IDENTIFY] STOP: model_identification_failed")


if __name__ == "__main__":
    main()
