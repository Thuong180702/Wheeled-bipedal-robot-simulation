"""Tune height-scheduled CoM feedback LQR gains (Phase B.6).

This script optimizes LQR gains K(h) for the height-scheduled CoM feedback variant
using a TWIP model with CoM error state augmentation.

State vector (6D):
    x = [pitch, pitch_rate, com_y_error, com_y_error_rate, wheel_pos, wheel_vel]

Control input (1D):
    u = wheel_torque (distributed to both wheels)

LQR problem:
    min ∫ (x^T Q x + u^T R u) dt
    s.t. dx/dt = A(h) x + B(h) u

where A(h), B(h) are height-dependent linearized dynamics.

Usage:
    python scripts/tune_height_scheduled_com_lqr.py \
        --heights 0.70 0.65 0.60 0.55 0.50 \
        --output configs/controllers/height_scheduled_com_lqr_tuned.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy import linalg


def compute_twip_dynamics_with_com(
    height: float,
    com_height_nom: float = 0.55,
    wheel_radius: float = 0.0762,
    body_mass: float = 10.0,
    body_inertia: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute linearized TWIP dynamics with CoM error augmentation.

    Args:
        height: Torso height [m].
        com_height_nom: Nominal CoM height above wheel axis [m].
        wheel_radius: Wheel radius [m].
        body_mass: Body mass [kg].
        body_inertia: Body pitch inertia [kg·m²].

    Returns:
        (A, B): State-space matrices for 6D system.
            A: 6x6 state matrix
            B: 6x1 input matrix
    """
    g = 9.81  # Gravity [m/s²]

    # Height-dependent CoM height (approximate)
    # Lower heights → CoM shifts forward relative to wheels
    h = com_height_nom + 0.1 * (height - 0.55)  # Rough approximation

    # TWIP linearized dynamics (pitch subsystem)
    # dx/dt = A_twip * x_twip + B_twip * u
    # x_twip = [pitch, pitch_rate, fwd_vel, fwd_pos]
    A_twip = np.array([
        [0, 1, 0, 0],           # pitch_dot = pitch_rate
        [g/h, 0, 0, 0],         # pitch_rate_dot ≈ (g/h) * pitch
        [0, 0, 0, 0],           # fwd_vel_dot = 0 (no direct coupling)
        [0, 0, 1, 0],           # fwd_pos_dot = fwd_vel
    ])

    B_twip = np.array([
        [0],
        [-1/h],  # wheel accel affects pitch rate
        [wheel_radius],     # wheel accel affects fwd vel
        [0],
    ])

    # Augment with CoM error dynamics
    # Assume CoM error evolves as:
    #   com_y_error_dot = com_y_error_rate
    #   com_y_error_rate_dot ≈ -k_spring * com_y_error - k_damping * com_y_error_rate + coupling
    #
    # For simplicity, model CoM error as weakly coupled to pitch:
    #   com_y_error_rate_dot ≈ pitch_rate (kinematic coupling)
    #
    # Full 6D state: [pitch, pitch_rate, com_y_error, com_y_error_rate, wheel_pos, wheel_vel]
    A = np.zeros((6, 6))
    A[0:4, 0:4] = A_twip  # TWIP subsystem

    # CoM error dynamics
    A[2, 3] = 1.0  # com_y_error_dot = com_y_error_rate
    A[3, 1] = 0.5  # com_y_error_rate_dot ≈ 0.5 * pitch_rate (kinematic coupling)

    # Wheel position/velocity
    A[4, 5] = 1.0  # wheel_pos_dot = wheel_vel

    B = np.zeros((6, 1))
    B[0:4, 0] = B_twip[:, 0]  # TWIP input coupling
    B[5, 0] = 1.0 / wheel_radius  # wheel_vel_dot = u / r (torque to angular accel)

    return A, B


def solve_lqr_6d(
    A: np.ndarray,
    B: np.ndarray,
    Q_diag: list[float],
    R_val: float,
) -> np.ndarray:
    """Solve LQR for 6D system.

    Args:
        A: 6x6 state matrix.
        B: 6x1 input matrix.
        Q_diag: Diagonal of Q matrix (6 elements).
        R_val: Scalar R value.

    Returns:
        K: LQR feedback gains, shape (1, 6).
    """
    Q = np.diag(Q_diag)
    R = np.array([[R_val]])

    # Solve continuous-time algebraic Riccati equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # Compute LQR gains: K = R^-1 * B^T * P
    K = np.linalg.solve(R, B.T @ P)

    return K


def tune_height_scheduled_gains(
    heights: list[float],
    Q_diag: list[float],
    R_val: float,
) -> dict[float, dict[str, float]]:
    """Tune LQR gains for each height in the grid.

    Args:
        heights: Height grid [m].
        Q_diag: Q matrix diagonal (6 elements).
        R_val: R scalar value.

    Returns:
        Dict mapping height to gain dict {k_pitch, k_pitch_rate, ...}.
    """
    gains = {}

    for h in heights:
        # Compute height-dependent dynamics
        A, B = compute_twip_dynamics_with_com(height=h)

        # Solve LQR
        K = solve_lqr_6d(A, B, Q_diag, R_val)

        # Extract gains
        gains[h] = {
            "k_pitch": float(K[0, 0]),
            "k_pitch_rate": float(K[0, 1]),
            "k_com": float(K[0, 2]),
            "k_com_rate": float(K[0, 3]),
            "k_wheel_pos": float(K[0, 4]),
            "k_wheel_vel": float(K[0, 5]),
        }

    return gains


def main():
    parser = argparse.ArgumentParser(description="Tune height-scheduled CoM LQR gains")
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50],
        help="Height grid [m]",
    )
    parser.add_argument(
        "--q-diag",
        type=float,
        nargs=6,
        default=[100.0, 10.0, 50.0, 5.0, 1.0, 1.0],
        help="Q matrix diagonal [pitch, pitch_rate, com, com_rate, wheel_pos, wheel_vel]",
    )
    parser.add_argument(
        "--r-val",
        type=float,
        default=1.0,
        help="R scalar value",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/controllers/height_scheduled_com_lqr_tuned.yaml",
        help="Output YAML file",
    )
    args = parser.parse_args()

    print("Tuning height-scheduled CoM LQR gains...")
    print(f"  Heights: {args.heights}")
    print(f"  Q_diag: {args.q_diag}")
    print(f"  R_val: {args.r_val}")

    # Tune gains
    gains = tune_height_scheduled_gains(
        heights=args.heights,
        Q_diag=args.q_diag,
        R_val=args.r_val,
    )

    # Print results
    print("\nTuned gains:")
    for h, g in gains.items():
        print(f"  h={h:.2f}m: k_pitch={g['k_pitch']:.1f}, k_pitch_rate={g['k_pitch_rate']:.1f}, "
              f"k_com={g['k_com']:.1f}, k_com_rate={g['k_com_rate']:.1f}, "
              f"k_wheel_pos={g['k_wheel_pos']:.1f}, k_wheel_vel={g['k_wheel_vel']:.1f}")

    # Load base config
    base_config_path = Path("configs/controllers/height_scheduled_com_lqr.yaml")
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update gains
    config["lqr_gains"] = {}
    for h, g in gains.items():
        key = f"h_{h:.2f}"
        config["lqr_gains"][key] = g

    # Update tuning metadata
    config["tuning"]["status"] = "tuned"
    config["tuning"]["q_diag"] = args.q_diag
    config["tuning"]["r_val"] = args.r_val

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved tuned config to: {output_path}")


if __name__ == "__main__":
    main()
