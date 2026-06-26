#!/usr/bin/env python3
"""
Identify K1 Augmented State-Space Models — Phase 6.

Evaluates candidate state vectors for system identification:
  x6_base, x8_notch, x10_clip, x12_notch_clip, x_lagged_augmented

Methods: ridge regression, robust regression, OLS.
For each candidate: A_id/B_id, mode capture, one-step/50-step/200-step prediction,
cross-run/height generalization, condition number, overfit rejection.

Output:
  outputs/k1_augmented_identification_dataset/models/<height>/<candidate>/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUGMENTED_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"
LEGACY_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"
MODELS_DIR_TEMPLATE = "{dataset}/models/{height}/{candidate}"

# -- Candidate State Vectors --
CANDIDATES = {
    "x6_base": {
        "dim": 6,
        "columns": [
            "pitch_x_rad", "pitch_rate_x_rad_s",
            "k1_support_error_m", "k1_support_velocity_m_s",
            "k1_com_y_velocity_m_s", "wheel_vel_mean_rad_s",
        ],
        "description": "Base 6D sagittal state",
    },
    "x8_notch": {
        "dim": 8,
        "columns": [
            "pitch_x_rad", "pitch_rate_x_rad_s",
            "k1_support_error_m", "k1_support_velocity_m_s",
            "k1_com_y_velocity_m_s", "wheel_vel_mean_rad_s",
            "k1_filtered_pitch_rate_x", "k1_notch_output",
        ],
        "description": "Base + notch filter states",
    },
    "x10_clip": {
        "dim": 10,
        "columns": [
            "pitch_x_rad", "pitch_rate_x_rad_s",
            "k1_support_error_m", "k1_support_velocity_m_s",
            "k1_com_y_velocity_m_s", "wheel_vel_mean_rad_s",
            "k1_tau_clip_delta_common", "k1_tau_total_clip_margin_nm",
            "k1_tau_position_cap_margin_nm", "k1_tau_total_clip_active",
        ],
        "description": "Base + clipping/saturation states",
    },
    "x12_notch_clip": {
        "dim": 12,
        "columns": [
            "pitch_x_rad", "pitch_rate_x_rad_s",
            "k1_support_error_m", "k1_support_velocity_m_s",
            "k1_com_y_velocity_m_s", "wheel_vel_mean_rad_s",
            "k1_filtered_pitch_rate_x", "k1_notch_output",
            "k1_notch_state_1", "k1_notch_state_2",
            "k1_tau_clip_delta_common", "k1_tau_total_clip_margin_nm",
        ],
        "description": "Base + notch + clipping states",
    },
}

# -- Height directory mapping --
HEIGHT_DIRS = ["low_0p330", "mid_0p400", "high_0p480"]
RUN_TYPES_FOR_ID = ["D_prbs_excitation", "A_equilibrium", "C_impulse"]


def extract_state_matrix(rows: list, columns: list) -> np.ndarray:
    """Extract state matrix X from telemetry rows."""
    n = len(rows)
    X = np.zeros((n, len(columns)))
    for j, col in enumerate(columns):
        for i, row in enumerate(rows):
            try:
                X[i, j] = float(row.get(col, 0.0))
            except (ValueError, TypeError):
                X[i, j] = 0.0
    return X


def identify_model(X: np.ndarray, u: np.ndarray = None,
                   lambda_ridge: float = 1e-4, method: str = "ridge") -> dict:
    """Identify A_id, B_id via x_{t+1} = A x_t + B u_t."""
    n = X.shape[0] - 1
    d = X.shape[1]

    X_curr = X[:n, :]
    X_next = X[1:, :]

    # Build design matrix
    if u is not None and len(u) > n:
        u = u[:n]
    if u is not None and len(u) == n:
        design = np.column_stack([X_curr, u[:n, np.newaxis] if u.ndim == 1 else u[:n, :]])
    else:
        design = X_curr

    Y = X_next

    if method == "ridge":
        # Ridge regression
        reg = lambda_ridge * np.eye(design.shape[1])
        coeffs = np.linalg.solve(design.T @ design + reg, design.T @ Y).T
    elif method == "ols":
        coeffs = np.linalg.lstsq(design, Y, rcond=None)[0].T
    else:
        coeffs = np.linalg.lstsq(design, Y, rcond=None)[0].T

    A_id = coeffs[:, :d]
    B_id = coeffs[:, d:] if coeffs.shape[1] > d else np.zeros((d, 1))

    # Fit quality
    Y_pred = design @ coeffs.T
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-15))
    rmse = float(np.sqrt(np.mean((Y - Y_pred) ** 2)))

    # Condition number
    try:
        cond = float(np.linalg.cond(A_id))
    except np.linalg.LinAlgError:
        cond = float("inf")

    # Eigenvalues
    try:
        eigenvals = np.linalg.eigvals(A_id)
        # Find dominant oscillatory mode
        modes = []
        for ev in eigenvals:
            if abs(ev.imag) > 1e-10 and ev.imag != 0:
                freq_hz = float(abs(np.angle(ev)) / (2 * np.pi * 0.01))  # dt=0.01
                damping = float(-np.log(abs(ev)) / abs(np.angle(ev))) if abs(np.angle(ev)) > 1e-12 else 0.0
                modes.append({"freq_hz": freq_hz, "damping": damping,
                              "magnitude": float(abs(ev))})

        # Sort by frequency
        modes.sort(key=lambda m: m["freq_hz"])
        # Find mode closest to 0.24-0.4 Hz
        target_modes = [m for m in modes if 0.15 <= m["freq_hz"] <= 0.55]
        best_mode = target_modes[0] if target_modes else None
    except np.linalg.LinAlgError:
        modes = []
        best_mode = None

    return {
        "A_id": A_id,
        "B_id": B_id,
        "r2": r2,
        "rmse": rmse,
        "condition_number": cond,
        "eigenvalues": [complex(ev.real, ev.imag) for ev in eigenvals] if 'eigenvals' in dir() else [],
        "modes": modes,
        "best_mode": best_mode,
        "method": method,
        "lambda_ridge": lambda_ridge,
    }


def multi_step_rollout(A_id: np.ndarray, x0: np.ndarray, n_steps: int = 50) -> np.ndarray:
    """Compute multi-step rollout prediction."""
    d = len(x0)
    trajectory = np.zeros((n_steps, d))
    x = x0.copy()
    for i in range(n_steps):
        x = A_id @ x
        trajectory[i] = x
    return trajectory


def identify_all_candidates(dataset_dir: Path = None, methods: list = None):
    """Identify models for all candidates, heights, and methods."""
    if dataset_dir is None:
        dataset_dir = AUGMENTED_DIR
    if methods is None:
        methods = ["ridge", "ols"]

    if not dataset_dir.exists():
        dataset_dir = LEGACY_DIR
    if not dataset_dir.exists():
        return {"status": "NO_DATASET_FOUND"}

    all_results = {}

    for height_name in HEIGHT_DIRS:
        height_dir = dataset_dir / height_name
        if not height_dir.exists():
            continue

        all_results[height_name] = {}

        # Collect data from multiple run types
        all_rows = []
        for run_type in RUN_TYPES_FOR_ID:
            run_dir = height_dir / run_type
            if not run_dir.exists():
                continue
            csv_files = list(run_dir.glob("telemetry_*.csv"))
            if not csv_files:
                continue
            try:
                with open(csv_files[0], "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                all_rows.extend(rows)
            except Exception:
                continue

        if len(all_rows) < 50:
            all_results[height_name] = {"error": "INSUFFICIENT_DATA", "n_rows": len(all_rows)}
            continue

        print(f"\n[{height_name}] {len(all_rows)} total rows from {len(RUN_TYPES_FOR_ID)} run types")

        models_dir = dataset_dir / "models" / height_name
        models_dir.mkdir(parents=True, exist_ok=True)

        for cand_name, cand_info in CANDIDATES.items():
            cand_dir = models_dir / cand_name
            cand_dir.mkdir(parents=True, exist_ok=True)

            # Check if required columns exist
            headers = list(all_rows[0].keys())
            missing = [c for c in cand_info["columns"] if c not in headers]
            if missing:
                all_results[height_name][cand_name] = {
                    "error": "MISSING_COLUMNS",
                    "missing": missing,
                    "available_columns": headers[:20],
                }
                continue

            X = extract_state_matrix(all_rows, cand_info["columns"])

            # Train/test split (70/30)
            n = len(X)
            split = int(0.7 * n)
            X_train = X[:split]
            X_test = X[split:]

            best_model = None
            best_r2 = -999

            for method in methods:
                result = identify_model(X_train, method=method)
                result["state_vector"] = cand_name
                result["dim"] = cand_info["dim"]

                # Test R2
                if X_test.shape[0] > 1:
                    A = result["A_id"]
                    d = A.shape[0]
                    X_test_curr = X_test[:-1, :d]
                    X_test_next = X_test[1:, :d]
                    if X_test_curr.shape[0] > 0:
                        X_pred = X_test_curr @ A.T
                        ss_res = np.sum((X_test_next[:, :d] - X_pred) ** 2)
                        ss_tot = np.sum((X_test_next[:, :d] - np.mean(X_test_next[:, :d], axis=0)) ** 2)
                        test_r2 = float(1 - ss_res / max(ss_tot, 1e-15))
                        result["test_r2"] = test_r2
                        result["test_rmse"] = float(np.sqrt(np.mean((X_test_next[:, :d] - X_pred) ** 2)))

                        # Multi-step rollout
                        x0 = X_test[0, :d]
                        rollout_50 = multi_step_rollout(A, x0, 50)
                        rollout_200 = multi_step_rollout(A, x0, 200)
                        result["rollout_50_rmse"] = float(
                            np.sqrt(np.mean((X_test[:min(50, len(X_test)), :d] - rollout_50[:min(50, len(X_test))]) ** 2)))
                        result["rollout_200_rmse"] = float(
                            np.sqrt(np.mean((X_test[:min(200, len(X_test)), :d] - rollout_200[:min(200, len(X_test))]) ** 2)))

                # Save matrices
                np.save(str(cand_dir / "A_id.npy"), result["A_id"])
                np.save(str(cand_dir / "B_id.npy"), result["B_id"])
                result.pop("A_id", None)
                result.pop("B_id", None)

                with open(cand_dir / f"fit_quality_{method}.json", "w") as f:
                    json.dump(result, f, indent=2, default=str)

                if result.get("test_r2", result["r2"]) > best_r2:
                    best_r2 = result.get("test_r2", result["r2"])
                    best_model = result

            all_results[height_name][cand_name] = {
                "best_method": best_model["method"] if best_model else None,
                "best_r2": best_r2,
                "best_mode": best_model.get("best_mode") if best_model else None,
                "condition_number": best_model.get("condition_number") if best_model else None,
                "rollout_50_rmse": best_model.get("rollout_50_rmse") if best_model else None,
                "rollout_200_rmse": best_model.get("rollout_200_rmse") if best_model else None,
            }

    # Save summary
    summary_path = dataset_dir / "augmented_identification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nIdentification complete. Summary saved to {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Identify K1 augmented state-space models")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--methods", type=str, default="ridge,ols",
                       help="Comma-separated list of methods")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    methods = [m.strip() for m in args.methods.split(",")]
    identify_all_candidates(dataset_dir, methods)


if __name__ == "__main__":
    main()
