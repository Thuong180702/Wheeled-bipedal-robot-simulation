#!/usr/bin/env python3
"""
Evaluate K1 Identification State Vectors — Phase 2.

Evaluates multiple candidate state vectors for system identification quality:
  - x6 (base): [pitch_x, pitch_rate_x, support_error, support_velocity, com_y_velocity, wheel_vel_mean]
  - x7 (add_height): x6 + [body_height_error]
  - x8 (add_notch): x6 + [filtered_pitch_rate, notch_output]
  - x9 (add_position): x6 + [com_y_position, wheel_angle_mean, body_height_error]
  - x_filter_augmented: includes K1 filter/notch internal states if accessible

For each candidate:
  - Observability from telemetry
  - Numerical conditioning
  - One-step prediction error
  - Multi-step rollout error
  - Eigenmode consistency
  - 0.24-0.4 Hz mode capture

Selects the minimum state vector that captures the dominant mode.

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains or modify K1.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"
TELEMETRY_DIR = (
    PROJECT_ROOT / "outputs" / "d_baseline_single_90n_10step_push_step300_3000"
)
TELEMETRY_PATH = TELEMETRY_DIR / "telemetry_1782262602.csv"

CONTROL_DT = 0.01


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


# ── State Vector Definitions ───────────────────────────────────────────────

STATE_VECTOR_CANDIDATES = {
    "x6_base": {
        "dim": 6,
        "names": [
            "pitch_x", "pitch_rate_x", "support_error",
            "support_velocity", "com_y_velocity", "wheel_vel_mean",
        ],
        "description": "Base 6D sagittal state — original K1 state vector",
        "extract_fn": "extract_x6",
    },
    "x7_add_height": {
        "dim": 7,
        "names": [
            "pitch_x", "pitch_rate_x", "support_error",
            "support_velocity", "com_y_velocity", "wheel_vel_mean",
            "body_height_error",
        ],
        "description": "x6 + body_height_error — captures height-dependent dynamics",
        "extract_fn": "extract_x7",
    },
    "x8_add_notch": {
        "dim": 8,
        "names": [
            "pitch_x", "pitch_rate_x", "support_error",
            "support_velocity", "com_y_velocity", "wheel_vel_mean",
            "filtered_pitch_rate", "notch_output",
        ],
        "description": "x6 + filter/notch states — captures controller internal dynamics",
        "extract_fn": "extract_x8",
    },
    "x9_add_position": {
        "dim": 9,
        "names": [
            "pitch_x", "pitch_rate_x", "support_error",
            "support_velocity", "com_y_velocity", "wheel_vel_mean",
            "com_y_position", "wheel_angle_mean", "body_height_error",
        ],
        "description": "x6 + position states — captures integral/long-term dynamics",
        "extract_fn": "extract_x9",
    },
    "x_filter_augmented": {
        "dim": 10,
        "names": [
            "pitch_x", "pitch_rate_x", "support_error",
            "support_velocity", "com_y_velocity", "wheel_vel_mean",
            "filtered_pitch_rate", "notch_output",
            "body_height_error", "cp_error",
        ],
        "description": "x6 + filter/notch + height + cp — full augmented state",
        "extract_fn": "extract_x_filter_augmented",
    },
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATE EXTRACTION FUNCTIONS                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _extract_base_x6(row):
    """Extract base 6D state from telemetry row."""
    return np.array([
        _safe_float(row.get("pitch_x", row.get("robot_pitch_x", 0))),
        _safe_float(row.get("pitch_rate_x_rad_s", row.get("pitch_rate_rad_s", 0))),
        _safe_float(row.get("support_position_error_m", 0)),
        0.0,  # support_velocity — not directly in standard telemetry (computed)
        _safe_float(row.get("com_vy", 0)),
        _safe_float(row.get("wheel_vel_mean", 0)),
    ])


def _compute_support_velocity(rows, idx):
    """Compute support_velocity from consecutive rows."""
    if idx == 0:
        return 0.0
    err_curr = _safe_float(rows[idx].get("support_position_error_m", 0))
    err_prev = _safe_float(rows[idx - 1].get("support_position_error_m", 0))
    return (err_curr - err_prev) / CONTROL_DT


def _compute_wheel_vel_mean(row):
    """Compute mean wheel velocity from left/right wheel velocities."""
    l_wheel_vel = _safe_float(row.get("l_wheel_vel", row.get("joint_vel_l_wheel", 0)))
    r_wheel_vel = _safe_float(row.get("r_wheel_vel", row.get("joint_vel_r_wheel", 0)))
    # Try joint_vel array
    jv = row.get("joint_vel", "")
    if jv and isinstance(jv, str) and jv.strip():
        parts = jv.strip().split(",") if "," in jv else jv.strip().split()
        if len(parts) >= 10:
            l_wheel_vel = _safe_float(parts[4]) if len(parts) > 4 else l_wheel_vel
            r_wheel_vel = _safe_float(parts[9]) if len(parts) > 9 else r_wheel_vel
    return (l_wheel_vel + r_wheel_vel) / 2.0


def extract_x6(rows, idx):
    """Extract base 6D state with computed support_velocity."""
    x = _extract_base_x6(rows[idx])
    x[3] = _compute_support_velocity(rows, idx)
    return x


def extract_x7(rows, idx):
    """Extract 7D state: x6 + body_height_error."""
    x6 = extract_x6(rows, idx)
    height_error = _safe_float(rows[idx].get("height_error_m", 0))
    body_height = _safe_float(rows[idx].get("com_z", 0))
    height_cmd = _safe_float(rows[idx].get("height_cmd", body_height))
    if abs(height_error) < 1e-10:
        height_error = body_height - height_cmd
    return np.append(x6, height_error)


def extract_x8(rows, idx):
    """Extract 8D state: x6 + filtered_pitch_rate + notch_output."""
    x6 = extract_x6(rows, idx)
    # K1 notch/filter states — may not be directly in telemetry
    filtered_pitch_rate = _safe_float(rows[idx].get("filtered_pitch_rate", 0))
    notch_output = _safe_float(rows[idx].get("notch_output", 0))
    # Also try pitch_rate_corrected as proxy for filtered
    if abs(filtered_pitch_rate) < 1e-12:
        filtered_pitch_rate = _safe_float(rows[idx].get("pitch_rate_corrected_x_rad_s", 0))
    return np.append(x6, [filtered_pitch_rate, notch_output])


def extract_x9(rows, idx):
    """Extract 9D state: x6 + com_y_position + wheel_angle_mean + body_height_error."""
    x6 = extract_x6(rows, idx)
    com_y = _safe_float(rows[idx].get("com_y", 0))
    wheel_angle_mean = _safe_float(rows[idx].get("wheel_angle_mean", 0))
    height_error = _safe_float(rows[idx].get("height_error_m", 0))
    body_height = _safe_float(rows[idx].get("com_z", 0))
    height_cmd = _safe_float(rows[idx].get("height_cmd", body_height))
    if abs(height_error) < 1e-10:
        height_error = body_height - height_cmd
    return np.append(x6, [com_y, wheel_angle_mean, height_error])


def extract_x_filter_augmented(rows, idx):
    """Extract 10D augmented state with filter/notch, height, and cp."""
    x6 = extract_x6(rows, idx)
    filtered_pitch_rate = _safe_float(rows[idx].get("filtered_pitch_rate", 0))
    if abs(filtered_pitch_rate) < 1e-12:
        filtered_pitch_rate = _safe_float(rows[idx].get("pitch_rate_corrected_x_rad_s", 0))
    notch_output = _safe_float(rows[idx].get("notch_output", 0))
    height_error = _safe_float(rows[idx].get("height_error_m", 0))
    body_height = _safe_float(rows[idx].get("com_z", 0))
    height_cmd = _safe_float(rows[idx].get("height_cmd", body_height))
    if abs(height_error) < 1e-10:
        height_error = body_height - height_cmd
    cp_error = _safe_float(rows[idx].get("cp_error_m", 0))
    return np.append(x6, [filtered_pitch_rate, notch_output, height_error, cp_error])


EXTRACT_FNS = {
    "extract_x6": extract_x6,
    "extract_x7": extract_x7,
    "extract_x8": extract_x8,
    "extract_x9": extract_x9,
    "extract_x_filter_augmented": extract_x_filter_augmented,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SYSTEM IDENTIFICATION (SINGLE-HEIGHT)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def identify_linear_model(X, X_next, lambda_reg=1e-4):
    """Linear system identification via regularized least squares.

    X_next ≈ A @ X  (no input term for autonomous/closed-loop ID)
    Returns A_id.
    """
    n_samples, n_states = X.shape
    # Ridge regression: A = X_next @ X^T @ inv(X @ X^T + lambda * I)
    XTX = X.T @ X
    reg = lambda_reg * np.eye(n_states)
    A_id = X_next.T @ X @ np.linalg.inv(XTX + reg)
    return A_id.T  # (n_states, n_states)


def identify_with_input(X, U, X_next, lambda_reg=1e-4):
    """System identification with input.

    X_next ≈ A @ X + B @ U
    Returns A_id, B_id.
    """
    n_samples, n_states = X.shape
    n_inputs = U.shape[1] if U.ndim > 1 else 1
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    # Augmented regression: [A | B] @ [X; U] ≈ X_next
    Z = np.hstack([X, U])  # (n_samples, n_states + n_inputs)
    ZTZ = Z.T @ Z
    reg = lambda_reg * np.eye(n_states + n_inputs)
    AB = X_next.T @ Z @ np.linalg.inv(ZTZ + reg)

    A_id = AB[:, :n_states]
    B_id = AB[:, n_states:]
    return A_id, B_id


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ONE-STEP AND MULTI-STEP PREDICTION                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def one_step_prediction_error(A_id, X, X_next):
    """Compute one-step prediction RMSE."""
    X_pred = (A_id @ X.T).T
    errors = X_next - X_pred
    rmse_per_state = np.sqrt(np.mean(errors ** 2, axis=0))
    nrmse_per_state = rmse_per_state / (np.std(X, axis=0) + 1e-8)
    return {
        "rmse_per_state": rmse_per_state.tolist(),
        "nrmse_per_state": nrmse_per_state.tolist(),
        "total_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "total_nrmse": float(np.sqrt(np.mean(errors ** 2)) / (np.std(X) + 1e-8)),
        "r_squared": float(1.0 - np.sum(errors ** 2) / (np.sum((X_next - np.mean(X_next, axis=0)) ** 2) + 1e-12)),
    }


def multi_step_rollout_error(A_id, X0, X_true, n_steps=50):
    """Compute multi-step rollout error."""
    n_states = len(X0)
    X_pred = np.zeros((n_steps, n_states))
    x = X0.copy()
    errors = []

    for k in range(n_steps):
        x = A_id @ x
        X_pred[k] = x
        if k < len(X_true):
            errors.append(X_true[k] - x)

    errors = np.array(errors)
    return {
        "n_steps": n_steps,
        "rmse_per_state": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        "total_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "diverged": bool(np.any(np.abs(X_pred[-1]) > 100)),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVABILITY CHECK                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def check_observability(A, C=None, n_outputs=6):
    """Check observability of the identified model.

    If C is not provided, uses identity (full state measurement assumed).
    Returns rank and unobservable subspace dimension.
    """
    n = A.shape[0]
    if C is None:
        C = np.eye(n_outputs, n)
        if C.shape[1] < n:
            C = np.eye(n)[:n_outputs, :]

    # Build observability matrix O = [C; CA; CA^2; ...]
    O_blocks = [C]
    for k in range(1, n):
        O_blocks.append(C @ np.linalg.matrix_power(A, k))
    O = np.vstack(O_blocks)

    U, S, Vt = np.linalg.svd(O, full_matrices=False)
    tol = max(S.max() * 1e-12, 1e-10) if S.size > 0 else 1e-10
    rank = int(np.sum(S > tol))

    return {
        "observability_rank": rank,
        "n_states": n,
        "is_fully_observable": rank >= n,
        "unobservable_dim": n - rank,
        "singular_values": S.tolist()[:10],
    }


def check_numerical_conditioning(A):
    """Check numerical conditioning of the identified model."""
    cond = float(np.linalg.cond(A))
    eigvals = np.linalg.eigvals(A)
    max_eig = float(np.max(np.abs(eigvals)))
    min_eig = float(np.min(np.abs(eigvals)))
    return {
        "condition_number": cond,
        "well_conditioned": cond < 1e6,
        "max_eigenvalue_magnitude": max_eig,
        "min_eigenvalue_magnitude": min_eig,
        "spectral_radius": max_eig,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODE CAPTURE CHECK                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_dominant_oscillatory_mode(A, dt, target_range=(0.15, 0.50)):
    """Find the dominant oscillatory mode in the target frequency range."""
    eigvals = np.linalg.eigvals(A)
    best_mode = None

    for lam in eigvals:
        if abs(lam.imag) < 1e-12:
            continue
        if abs(lam) < 1e-14:
            continue

        s = np.log(lam) / dt
        omega_n = abs(s)
        freq_hz = omega_n / (2 * np.pi)
        zeta = -s.real / omega_n if omega_n > 1e-10 else 1.0

        if target_range[0] <= freq_hz <= target_range[1]:
            if best_mode is None or abs(lam.imag) > abs(best_mode["eigenvalue"].imag):
                best_mode = {
                    "eigenvalue": lam,
                    "frequency_hz": float(freq_hz),
                    "damping_ratio": float(zeta),
                    "magnitude": float(abs(lam)),
                }

    return best_mode


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN EVALUATION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_state_vectors(telemetry_path=None):
    """Evaluate all state vector candidates using available telemetry."""
    if telemetry_path is None:
        telemetry_path = TELEMETRY_PATH

    print("=" * 72)
    print("PHASE 2: AUGMENTED STATE VECTOR STUDY")
    print("=" * 72)

    if not Path(telemetry_path).exists():
        print(f"[SKIP] Telemetry not found: {telemetry_path}")
        print("  Run Phase 1 first to generate telemetry.")
        return {"status": "NO_TELEMETRY_AVAILABLE"}

    # Load telemetry
    print(f"\nLoading telemetry: {telemetry_path}")
    with open(telemetry_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    N = len(rows)
    print(f"  {N} rows loaded")

    # Filter for near-equilibrium samples (low pitch, non-terminated)
    equilibrium_rows = []
    equilibrium_indices = []
    for i, row in enumerate(rows):
        pitch = abs(_safe_float(row.get("pitch_x", row.get("robot_pitch_x", 0))))
        terminated = row.get("terminated", "False")
        if isinstance(terminated, str):
            is_terminated = terminated.lower() in ("true", "1", "yes")
        else:
            is_terminated = bool(int(float(terminated)))
        com_z = _safe_float(row.get("com_z", 0))
        if pitch < 0.1 and not is_terminated and 0.25 < com_z < 0.55:
            equilibrium_rows.append(row)
            equilibrium_indices.append(i)

    print(f"  {len(equilibrium_rows)} near-equilibrium samples")
    if len(equilibrium_rows) < 50:
        print("  [WARN] Insufficient equilibrium samples. Results may be unreliable.")

    # Group by height
    height_groups = {}
    for row in equilibrium_rows:
        h = _safe_float(row.get("com_z", 0))
        for target_h in [0.33, 0.40, 0.48]:
            if abs(h - target_h) < 0.03:
                key = f"h_{str(target_h).replace('.', 'p')}"
                height_groups.setdefault(key, []).append(row)
                break

    results = {}
    selected_vector = None
    best_score = float("inf")

    for vec_name, vec_info in STATE_VECTOR_CANDIDATES.items():
        print(f"\n{'-' * 60}")
        print(f"Evaluating: {vec_name} (dim={vec_info['dim']})")
        print(f"  {vec_info['description']}")
        print(f"{'-' * 60}")

        extract_fn = EXTRACT_FNS[vec_info["extract_fn"]]
        n_states = vec_info["dim"]

        # Build state matrices from equilibrium data
        X_pairs = []
        for i in range(len(equilibrium_rows) - 1):
            try:
                x_t = extract_fn(equilibrium_rows, i)
                x_t1 = extract_fn(equilibrium_rows, i + 1)
                if np.all(np.isfinite(x_t)) and np.all(np.isfinite(x_t1)):
                    X_pairs.append((x_t, x_t1))
            except Exception:
                continue

        if len(X_pairs) < 20:
            print(f"  [WARN] Only {len(X_pairs)} valid state pairs — insufficient")
            results[vec_name] = {"status": "INSUFFICIENT_DATA", "n_pairs": len(X_pairs)}
            continue

        X = np.array([p[0] for p in X_pairs])
        X_next = np.array([p[1] for p in X_pairs])

        print(f"  Valid state pairs: {len(X_pairs)}")

        # System identification
        A_id = identify_linear_model(X, X_next)
        print(f"  A_id: cond={np.linalg.cond(A_id):.2e}")

        # One-step prediction
        pred = one_step_prediction_error(A_id, X, X_next)
        print(f"  One-step NRMSE: {pred['total_nrmse']:.4f}, R²={pred['r_squared']:.4f}")

        # Multi-step rollout
        rollout = multi_step_rollout_error(A_id, X[0], X_next[:50], n_steps=50)
        print(f"  50-step rollout RMSE: {rollout['total_rmse']:.4f}, diverged={rollout['diverged']}")

        # Observability
        obs = check_observability(A_id)
        print(f"  Observability rank: {obs['observability_rank']}/{obs['n_states']}, "
              f"fully observable={obs['is_fully_observable']}")

        # Numerical conditioning
        cond = check_numerical_conditioning(A_id)
        print(f"  Condition number: {cond['condition_number']:.2e}, "
              f"well-conditioned={cond['well_conditioned']}")

        # Mode capture
        mode = find_dominant_oscillatory_mode(A_id, CONTROL_DT)
        if mode:
            print(f"  Dominant mode: f={mode['frequency_hz']:.3f} Hz, zeta={mode['damping_ratio']:.3f}")
            mode_captured = 0.15 <= mode["frequency_hz"] <= 0.50
        else:
            print(f"  [WARN] No oscillatory mode found in 0.15-0.50 Hz range")
            mode_captured = False

        results[vec_name] = {
            "dim": n_states,
            "n_pairs": len(X_pairs),
            "one_step_nrmse": pred["total_nrmse"],
            "one_step_r2": pred["r_squared"],
            "rollout_50_rmse": rollout["total_rmse"],
            "rollout_diverged": rollout["diverged"],
            "observability_rank": obs["observability_rank"],
            "is_fully_observable": obs["is_fully_observable"],
            "condition_number": cond["condition_number"],
            "well_conditioned": cond["well_conditioned"],
            "mode_captured": mode_captured,
            "mode_frequency_hz": mode["frequency_hz"] if mode else None,
            "mode_damping": mode["damping_ratio"] if mode else None,
            "status": "EVALUATED",
        }

        # Scoring for selection (lower is better)
        score = 0.0
        if not mode_captured:
            score += 100.0
        score += pred["total_nrmse"] * 10.0
        score += rollout["total_rmse"] * 5.0
        if rollout["diverged"]:
            score += 50.0
        if not obs["is_fully_observable"]:
            score += 10.0
        if not cond["well_conditioned"]:
            score += 20.0
        score += n_states * 0.1  # Penalize larger state vectors

        results[vec_name]["selection_score"] = score
        print(f"  Selection score: {score:.2f} (lower is better)")

        if mode_captured and score < best_score:
            best_score = score
            selected_vector = vec_name

    # Determine selection
    print(f"\n{'=' * 72}")
    print(f"SELECTION RESULT")
    print(f"{'=' * 72}")

    if selected_vector:
        print(f"Selected: {selected_vector} (dim={STATE_VECTOR_CANDIDATES[selected_vector]['dim']})")
        print(f"  Score: {best_score:.2f}")
        print(f"  Mode captured: {results[selected_vector]['mode_captured']}")
        print(f"  Mode frequency: {results[selected_vector]['mode_frequency_hz']} Hz")
        print(f"  One-step NRMSE: {results[selected_vector]['one_step_nrmse']:.4f}")
    else:
        # Fallback: use x6_base
        selected_vector = "x6_base"
        print(f"[FALLBACK] No vector captured the mode adequately.")
        print(f"  Using: {selected_vector} (dim=6)")
        print(f"  WARNING: Mode may not be captured by linear model without filter states")

    results["_selected"] = selected_vector
    results["_selection_score"] = best_score if selected_vector else None

    # Save results
    out_path = OUTPUT_DIR / "state_vector_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make serializable
    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_make_serializable)

    print(f"\n[OK] Results saved: {out_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate K1 identification state vector candidates"
    )
    parser.add_argument("--telemetry", type=str, default=None,
                        help="Path to telemetry CSV (uses default K1 telemetry if not specified)")
    args = parser.parse_args()

    evaluate_state_vectors(args.telemetry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
