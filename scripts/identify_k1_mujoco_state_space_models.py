#!/usr/bin/env python3
"""
Identify K1 MuJoCo State-Space Models — Phase 3.

For each height and selected state vector, identify A_id(h) and B_id(h)
using multiple methods:
  - Regularized least squares (ridge regression)
  - Robust regression (Huber-like weighting) if outliers present

Splits data into train/validation/push-test.
Saves identified models to outputs/k1_identification_dataset/models/.

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains or modify K1.
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# -- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"
MODELS_DIR = OUTPUT_DIR / "models"

# Existing telemetry for fallback
TELEMETRY_DIR = (
    PROJECT_ROOT / "outputs" / "d_baseline_single_90n_10step_push_step300_3000"
)
TELEMETRY_PATH = TELEMETRY_DIR / "telemetry_1782262602.csv"

CONTROL_DT = 0.01
TARGET_HEIGHTS = [0.33, 0.40, 0.48]

# Default selected state vector from Phase 2
DEFAULT_STATE_VECTOR = "x6_base"
STATE_NAMES = [
    "pitch_x", "pitch_rate_x", "support_error",
    "support_velocity", "com_y_velocity", "wheel_vel_mean",
]


def _safe_float(val, default=0.0):
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATE EXTRACTION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_sagittal_state(row, prev_row=None):
    """Extract 6D sagittal state from telemetry row."""
    pitch_x = _safe_float(row.get("pitch_x", row.get("robot_pitch_x", 0)))
    pitch_rate = _safe_float(row.get("pitch_rate_x_rad_s", row.get("pitch_rate_rad_s", 0)))
    support_error = _safe_float(row.get("support_position_error_m", 0))
    com_vy = _safe_float(row.get("com_vy", 0))

    # Compute support_velocity
    if prev_row is not None:
        prev_err = _safe_float(prev_row.get("support_position_error_m", 0))
        support_vel = (support_error - prev_err) / CONTROL_DT
    else:
        support_vel = 0.0

    # Compute wheel_vel_mean
    jv = row.get("joint_vel", "")
    l_wheel_vel = 0.0
    r_wheel_vel = 0.0
    if jv and isinstance(jv, str) and jv.strip():
        parts = [p.strip() for p in jv.replace(",", " ").split()]
        if len(parts) >= 10:
            l_wheel_vel = _safe_float(parts[4])
            r_wheel_vel = _safe_float(parts[9])
    wheel_vel_mean = (l_wheel_vel + r_wheel_vel) / 2.0

    return np.array([pitch_x, pitch_rate, support_error, support_vel, com_vy, wheel_vel_mean])


def extract_input_signal(row):
    """Extract external input signal (sagittal push force) from telemetry."""
    push_fx = _safe_float(row.get("push_force_x", 0))
    push_fy = _safe_float(row.get("push_force_y", 0))
    # Use sagittal component as input
    return np.array([push_fy])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING AND SPLITTING                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_telemetry_data(telemetry_path, target_height, height_tolerance=0.03):
    """Load state-input pairs from telemetry near a target height."""
    if not Path(telemetry_path).exists():
        return None, None, None

    with open(telemetry_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X_list = []
    U_list = []
    X_next_list = []

    for i in range(len(rows) - 1):
        com_z = _safe_float(rows[i].get("com_z", 0))
        if abs(com_z - target_height) > height_tolerance:
            continue

        terminated = rows[i].get("terminated", "False")
        if isinstance(terminated, str):
            is_terminated = terminated.lower() in ("true", "1", "yes")
        else:
            is_terminated = bool(int(float(terminated)))
        if is_terminated:
            continue

        prev_row = rows[i - 1] if i > 0 else None
        x_t = extract_sagittal_state(rows[i], prev_row)
        u_t = extract_input_signal(rows[i])
        x_t1 = extract_sagittal_state(rows[i + 1], rows[i])

        if np.all(np.isfinite(x_t)) and np.all(np.isfinite(x_t1)) and np.all(np.isfinite(u_t)):
            X_list.append(x_t)
            U_list.append(u_t)
            X_next_list.append(x_t1)

    if len(X_list) < 20:
        return None, None, None

    return np.array(X_list), np.array(U_list), np.array(X_next_list)


def load_dataset_telemetry(height_name, run_types=None):
    """Load telemetry from the identification dataset for a given height.

    Returns dict: {run_type: (X, U, X_next)} for each available run type.
    """
    if run_types is None:
        run_types = ["A_equilibrium", "B_90n_push", "C_impulse", "D_prbs_excitation"]

    height_dir = OUTPUT_DIR / height_name
    if not height_dir.exists():
        return {}

    target_h = {"low_0p330": 0.33, "mid_0p400": 0.40, "high_0p480": 0.48}.get(height_name, 0.40)

    results = {}
    for rt in run_types:
        run_dir = height_dir / rt
        if not run_dir.exists():
            continue
        telemetry_files = sorted(run_dir.glob("telemetry_*.csv"))
        if not telemetry_files:
            continue

        X, U, Xn = load_telemetry_data(telemetry_files[-1], target_h)
        if X is not None:
            results[rt] = (X, U, Xn)

    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SYSTEM IDENTIFICATION METHODS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def identify_ridge(X, U, X_next, lambda_reg=1e-4):
    """Ridge regression: [A | B] = X_next @ Z^T @ inv(Z @ Z^T + lambda*I)."""
    n_samples, n_states = X.shape
    n_inputs = U.shape[1] if U.ndim > 1 else 1
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    Z = np.hstack([X, U])
    ZTZ = Z.T @ Z
    reg = lambda_reg * np.eye(n_states + n_inputs)
    AB = X_next.T @ Z @ np.linalg.inv(ZTZ + reg)

    A_id = AB[:, :n_states]
    B_id = AB[:, n_states:]

    # Fit quality
    X_pred = (A_id @ X.T).T + (B_id @ U.T).T
    residuals = X_next - X_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((X_next - np.mean(X_next, axis=0)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return A_id, B_id, {
        "method": "ridge_regression",
        "lambda": lambda_reg,
        "r_squared": r2,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "n_samples": n_samples,
    }


def identify_robust(X, U, X_next, lambda_reg=1e-4, n_iter=3, huber_delta=1.345):
    """Robust regression with Huber-like iterative reweighting."""
    A, B, info = identify_ridge(X, U, X_next, lambda_reg)

    n_inputs = U.shape[1] if U.ndim > 1 else 1
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    for iteration in range(n_iter):
        X_pred = (A @ X.T).T + (B @ U.T).T
        residuals = X_next - X_pred
        residual_norms = np.sqrt(np.sum(residuals ** 2, axis=1))

        # Huber weights
        weights = np.ones(len(X))
        mask = residual_norms > huber_delta * np.median(residual_norms)
        weights[mask] = huber_delta * np.median(residual_norms) / (residual_norms[mask] + 1e-8)

        # Weighted ridge regression
        W = np.diag(weights)
        Z = np.hstack([X, U])
        ZTWZ = Z.T @ W @ Z
        reg = lambda_reg * np.eye(Z.shape[1])
        AB = X_next.T @ W @ Z @ np.linalg.inv(ZTWZ + reg)

        A = AB[:, :X.shape[1]]
        B = AB[:, X.shape[1]:]

    X_pred = (A @ X.T).T + (B @ U.T).T
    residuals = X_next - X_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((X_next - np.mean(X_next, axis=0)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return A, B, {
        "method": "robust_regression",
        "lambda": lambda_reg,
        "n_iterations": n_iter,
        "huber_delta": huber_delta,
        "r_squared": r2,
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "n_samples": X.shape[0],
    }


def identify_ols(X, U, X_next):
    """Ordinary least squares (no regularization)."""
    n_inputs = U.shape[1] if U.ndim > 1 else 1
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    Z = np.hstack([X, U])
    try:
        AB = np.linalg.lstsq(Z, X_next, rcond=None)[0]
        A_id = AB[:X.shape[1], :].T
        B_id = AB[X.shape[1]:, :].T

        X_pred = (A_id @ X.T).T + (B_id @ U.T).T
        residuals = X_next - X_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((X_next - np.mean(X_next, axis=0)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

        return A_id, B_id, {
            "method": "ols",
            "r_squared": r2,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "n_samples": X.shape[0],
        }
    except np.linalg.LinAlgError:
        return identify_ridge(X, U, X_next, lambda_reg=1e-4)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CROSS-VALIDATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def cross_validate(X, U, X_next, train_ratio=0.7, method="ridge", lambda_reg=1e-4):
    """Train/test split validation."""
    n = len(X)
    n_train = int(n * train_ratio)
    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, U_train, Xn_train = X[train_idx], U[train_idx], X_next[train_idx]
    X_test, U_test, Xn_test = X[test_idx], U[test_idx], X_next[test_idx]

    if method == "ridge":
        A, B, info = identify_ridge(X_train, U_train, Xn_train, lambda_reg)
    elif method == "robust":
        A, B, info = identify_robust(X_train, U_train, Xn_train, lambda_reg)
    else:
        A, B, info = identify_ols(X_train, U_train, Xn_train)

    # Test performance
    n_inputs = U_test.shape[1] if U_test.ndim > 1 else 1
    if U_test.ndim == 1:
        U_test = U_test.reshape(-1, 1)

    X_pred = (A @ X_test.T).T + (B @ U_test.T).T
    residuals = Xn_test - X_pred
    test_rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Xn_test - np.mean(Xn_test, axis=0)) ** 2)
    test_r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {
        "train_r2": info["r_squared"],
        "train_rmse": info["rmse"],
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "n_train": n_train,
        "n_test": n - n_train,
        "generalization_gap": info["r_squared"] - test_r2,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN IDENTIFICATION PIPELINE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def identify_models_for_height(height_name, target_h, data_dict):
    """Identify A_id(h), B_id(h) using available data.

    Returns dict with identified models and quality metrics.
    """
    print(f"\n{'-' * 60}")
    print(f"Identifying models for {height_name} (h={target_h}m)")
    print(f"{'-' * 60}")

    if not data_dict:
        print(f"  [SKIP] No dataset telemetry available for {height_name}")
        print(f"  Falling back to legacy telemetry...")
        X, U, Xn = load_telemetry_data(TELEMETRY_PATH, target_h)
        if X is None:
            return {"status": "NO_DATA_AVAILABLE", "height": target_h}
        data_dict = {"legacy": (X, U, Xn)}

    results = {"height_name": height_name, "target_height": target_h, "methods": {}}

    # Combine all run types for training
    all_X = []
    all_U = []
    all_Xn = []
    run_sources = []

    for run_type, (X, U, Xn) in data_dict.items():
        all_X.append(X)
        all_U.append(U)
        all_Xn.append(Xn)
        run_sources.extend([run_type] * len(X))

    if not all_X:
        return {"status": "NO_DATA_AVAILABLE", "height": target_h}

    X_all = np.vstack(all_X)
    U_all = np.vstack(all_U)
    Xn_all = np.vstack(all_Xn)

    print(f"  Total samples: {len(X_all)}")
    print(f"  Sources: {list(data_dict.keys())}")

    # Ridge regression
    print(f"\n  [Method 1] Ridge regression (lambda=1e-4)")
    A_ridge, B_ridge, info_ridge = identify_ridge(X_all, U_all, Xn_all)
    print(f"    R²={info_ridge['r_squared']:.4f}, RMSE={info_ridge['rmse']:.4f}")
    cv_ridge = cross_validate(X_all, U_all, Xn_all, method="ridge")
    print(f"    CV: train_R²={cv_ridge['train_r2']:.4f}, test_R²={cv_ridge['test_r2']:.4f}, "
          f"gap={cv_ridge['generalization_gap']:.4f}")

    results["methods"]["ridge"] = {
        "A": A_ridge.tolist(),
        "B": B_ridge.tolist(),
        "fit": info_ridge,
        "cross_validation": cv_ridge,
    }

    # Robust regression
    print(f"\n  [Method 2] Robust regression")
    A_robust, B_robust, info_robust = identify_robust(X_all, U_all, Xn_all)
    print(f"    R²={info_robust['r_squared']:.4f}, RMSE={info_robust['rmse']:.4f}")
    cv_robust = cross_validate(X_all, U_all, Xn_all, method="robust")
    print(f"    CV: train_R²={cv_robust['train_r2']:.4f}, test_R²={cv_robust['test_r2']:.4f}, "
          f"gap={cv_robust['generalization_gap']:.4f}")

    results["methods"]["robust"] = {
        "A": A_robust.tolist(),
        "B": B_robust.tolist(),
        "fit": info_robust,
        "cross_validation": cv_robust,
    }

    # OLS
    print(f"\n  [Method 3] Ordinary least squares")
    A_ols, B_ols, info_ols = identify_ols(X_all, U_all, Xn_all)
    print(f"    R²={info_ols['r_squared']:.4f}, RMSE={info_ols['rmse']:.4f}")
    cv_ols = cross_validate(X_all, U_all, Xn_all, method="ols")
    print(f"    CV: train_R²={cv_ols['train_r2']:.4f}, test_R²={cv_ols['test_r2']:.4f}, "
          f"gap={cv_ols['generalization_gap']:.4f}")

    results["methods"]["ols"] = {
        "A": A_ols.tolist(),
        "B": B_ols.tolist(),
        "fit": info_ols,
        "cross_validation": cv_ols,
    }

    # Select best method (highest test R² with smallest generalization gap)
    best_method = "ridge"
    best_score = -float("inf")
    for method_name, method_data in results["methods"].items():
        cv = method_data["cross_validation"]
        score = cv["test_r2"] - 0.5 * abs(cv["generalization_gap"])
        if score > best_score:
            best_score = score
            best_method = method_name

    results["selected_method"] = best_method
    results["selected_A"] = results["methods"][best_method]["A"]
    results["selected_B"] = results["methods"][best_method]["B"]
    results["status"] = "IDENTIFIED"

    print(f"\n  Selected: {best_method} (score={best_score:.4f})")

    # Compute eigenvalues of A_id
    A_best = np.array(results["selected_A"])
    eigvals = np.linalg.eigvals(A_best)
    results["eigenvalues"] = [
        {"real": float(v.real), "imag": float(v.imag), "magnitude": float(abs(v))}
        for v in eigvals
    ]

    # Check for oscillatory mode
    for lam in eigvals:
        if abs(lam.imag) > 1e-10 and abs(lam) > 1e-14:
            s = np.log(lam) / CONTROL_DT
            freq = abs(s) / (2 * np.pi)
            zeta = -s.real / abs(s) if abs(s) > 1e-10 else 1.0
            if 0.15 <= freq <= 0.55:
                print(f"  Dominant mode: f={freq:.3f} Hz, zeta={zeta:.3f}")

    return results


def identify_all_models(state_vector=DEFAULT_STATE_VECTOR):
    """Run identification for all three heights."""
    print("=" * 72)
    print("PHASE 3: SYSTEM IDENTIFICATION")
    print("=" * 72)
    print(f"State vector: {state_vector}")

    all_results = {"state_vector": state_vector, "dt_s": CONTROL_DT, "heights": {}}

    height_map = {
        "low_0p330": 0.33,
        "mid_0p400": 0.40,
        "high_0p480": 0.48,
    }

    for height_name, target_h in height_map.items():
        # Try dataset telemetry first
        data_dict = load_dataset_telemetry(height_name)
        result = identify_models_for_height(height_name, target_h, data_dict)

        # Save model files
        model_dir = MODELS_DIR / height_name / state_vector
        model_dir.mkdir(parents=True, exist_ok=True)

        if result.get("status") == "IDENTIFIED":
            np.save(model_dir / "A_id.npy", np.array(result["selected_A"]))
            np.save(model_dir / "B_id.npy", np.array(result["selected_B"]))

            with open(model_dir / "model_metadata.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Simple fit quality
            fit_quality = {
                "height": target_h,
                "method": result["selected_method"],
                "r_squared": result["methods"][result["selected_method"]]["fit"]["r_squared"],
                "train_r2": result["methods"][result["selected_method"]]["cross_validation"]["train_r2"],
                "test_r2": result["methods"][result["selected_method"]]["cross_validation"]["test_r2"],
                "rmse": result["methods"][result["selected_method"]]["fit"]["rmse"],
                "n_samples": result["methods"][result["selected_method"]]["fit"]["n_samples"],
            }
            with open(model_dir / "fit_quality.json", "w") as f:
                json.dump(fit_quality, f, indent=2)

        all_results["heights"][height_name] = result

    # Save summary
    summary_path = OUTPUT_DIR / "identification_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[OK] Identification summary saved: {summary_path}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Identify K1 MuJoCo state-space models"
    )
    parser.add_argument("--state-vector", type=str, default=DEFAULT_STATE_VECTOR,
                        choices=["x6_base", "x7_add_height", "x8_add_notch", "x9_add_position",
                                 "x_filter_augmented"],
                        help="State vector to use for identification")
    parser.add_argument("--telemetry", type=str, default=None,
                        help="Path to telemetry CSV (overrides dataset search)")
    args = parser.parse_args()

    identify_all_models(args.state_vector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
