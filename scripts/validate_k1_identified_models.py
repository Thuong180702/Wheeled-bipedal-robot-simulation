#!/usr/bin/env python3
"""
Validate K1 Identified Models — Phase 4.

For each identified model A_id(h), B_id(h):
  1. One-step prediction error
  2. 50-step rollout error
  3. 200-step rollout error
  4. Mode frequency error
  5. Damping ratio error
  6. Response-to-impulse prediction
  7. Response-to-push prediction
  8. Cross-run generalization
  9. Cross-height interpolation feasibility

Classifies each model as: DESIGN_READY, NEEDS_STATE_AUGMENTATION,
  INSUFFICIENT_EXCITATION, HEIGHT_DATA_INSUFFICIENT, UNSTABLE_ID_ARTIFACT,
  OVERFIT, or INCONCLUSIVE.

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
TELEMETRY_DIR = (
    PROJECT_ROOT / "outputs" / "d_baseline_single_90n_10step_push_step300_3000"
)
TELEMETRY_PATH = TELEMETRY_DIR / "telemetry_1782262602.csv"

CONTROL_DT = 0.01
STATE_NAMES = [
    "pitch_x", "pitch_rate_x", "support_error",
    "support_velocity", "com_y_velocity", "wheel_vel_mean",
]
TARGET_MODE_FREQ = (0.15, 0.50)  # Hz
TARGET_MODE_FREQ_CENTER = 0.30  # Hz


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
# ║  MODEL LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_model(height_name, state_vector="x6_base"):
    """Load an identified model from disk."""
    model_dir = MODELS_DIR / height_name / state_vector
    a_path = model_dir / "A_id.npy"
    b_path = model_dir / "B_id.npy"
    meta_path = model_dir / "model_metadata.json"

    if not a_path.exists():
        return None

    A = np.load(str(a_path))
    B = np.load(str(b_path)) if b_path.exists() else np.zeros((A.shape[0], 1))

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return {"A": A, "B": B, "metadata": metadata}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VALIDATION METRICS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def one_step_prediction(A, B, X, U):
    """One-step prediction error."""
    X_pred = (A @ X.T).T + (B @ U.T).T
    residuals = X_pred - X  # actually x_{t+1} but using entire sequence
    return {
        "rmse_per_state": np.sqrt(np.mean(residuals ** 2, axis=0)).tolist(),
        "total_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "nrmse_per_state": (np.sqrt(np.mean(residuals ** 2, axis=0)) /
                            (np.std(X, axis=0) + 1e-8)).tolist(),
    }


def n_step_rollout(A, B, x0, u_seq, n_steps):
    """N-step open-loop rollout."""
    n_states = len(x0)
    traj = np.zeros((n_steps, n_states))
    x = x0.copy()
    for k in range(n_steps):
        u = u_seq[min(k, len(u_seq) - 1)] if len(u_seq) > 0 else np.zeros(B.shape[1] if B.ndim > 1 else 1)
        if B.shape[1] == 0 or (B.ndim > 1 and B.shape[1] == 0):
            x = A @ x
        else:
            x = A @ x + B @ u
        traj[k] = x
    return traj


def rollout_error(traj_pred, traj_true):
    """Compute error between predicted and true trajectories."""
    n = min(len(traj_pred), len(traj_true))
    residuals = traj_true[:n] - traj_pred[:n]
    return {
        "rmse_per_state": np.sqrt(np.mean(residuals ** 2, axis=0)).tolist(),
        "total_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "max_error_per_state": np.max(np.abs(residuals), axis=0).tolist(),
        "diverged": bool(np.any(np.abs(traj_pred[-1]) > 100)),
    }


def find_dominant_mode(A):
    """Find dominant oscillatory mode in the target frequency range."""
    eigvals = np.linalg.eigvals(A)
    best = None
    for lam in eigvals:
        if abs(lam.imag) < 1e-12:
            continue
        if abs(lam) < 1e-14:
            continue
        s = np.log(lam) / CONTROL_DT
        freq = abs(s) / (2 * np.pi)
        if TARGET_MODE_FREQ[0] <= freq <= TARGET_MODE_FREQ[1]:
            zeta = -s.real / abs(s) if abs(s) > 1e-10 else 1.0
            if best is None or abs(freq - TARGET_MODE_FREQ_CENTER) < abs(best["frequency_hz"] - TARGET_MODE_FREQ_CENTER):
                best = {
                    "eigenvalue": lam,
                    "frequency_hz": float(freq),
                    "damping_ratio": float(zeta),
                    "magnitude": float(abs(lam)),
                }
    return best


def mode_frequency_error(A_model, freq_ref_hz=0.239, zeta_ref=0.096):
    """Error in dominant mode frequency and damping vs reference."""
    mode = find_dominant_mode(A_model)
    if mode is None:
        return {"mode_found": False, "freq_error_hz": None, "zeta_error": None}

    return {
        "mode_found": True,
        "frequency_hz": mode["frequency_hz"],
        "damping_ratio": mode["damping_ratio"],
        "freq_error_hz": abs(mode["frequency_hz"] - freq_ref_hz),
        "freq_error_pct": abs(mode["frequency_hz"] - freq_ref_hz) / (freq_ref_hz + 1e-8) * 100,
        "zeta_error": abs(mode["damping_ratio"] - zeta_ref),
    }


def impulse_response(A, B, x0, impulse_magnitude=1.0, n_steps=100):
    """Simulate response to an impulse input."""
    n_inputs = B.shape[1] if B.ndim > 1 else 1
    u_seq = np.zeros((n_steps, n_inputs))
    u_seq[0, 0] = impulse_magnitude  # Unit impulse
    return n_step_rollout(A, B, x0, u_seq, n_steps)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VALIDATION PIPELINE                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def validate_model(height_name, model, telemetry_path=None):
    """Run full validation suite on one model."""
    print(f"\n{'-' * 60}")
    print(f"Validating: {height_name}")
    print(f"{'-' * 60}")

    if model is None:
        print("  [SKIP] No model available")
        return {"status": "NO_MODEL", "classification": "HEIGHT_DATA_INSUFFICIENT"}

    A = model["A"]
    B = model["B"]
    n_states = A.shape[0]

    # Load validation data
    target_h = {"low_0p330": 0.33, "mid_0p400": 0.40, "high_0p480": 0.48}.get(height_name, 0.40)

    validation = {
        "height_name": height_name,
        "target_height": target_h,
        "n_states": n_states,
        "tests": {},
    }

    # -- Test 1: One-step prediction --
    if telemetry_path and Path(telemetry_path).exists():
        with open(telemetry_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        X_vals = []
        U_vals = []
        Xn_vals = []
        for i in range(len(rows) - 1):
            com_z = _safe_float(rows[i].get("com_z", 0))
            if abs(com_z - target_h) > 0.03:
                continue
            terminated = rows[i].get("terminated", "False")
            if isinstance(terminated, str):
                is_terminated = terminated.lower() in ("true", "1", "yes")
            else:
                is_terminated = bool(int(float(terminated)))
            if is_terminated:
                continue

            from scripts.identify_k1_mujoco_state_space_models import (
                extract_sagittal_state, extract_input_signal,
            )
            prev_row = rows[i - 1] if i > 0 else None
            x_t = extract_sagittal_state(rows[i], prev_row)
            u_t = extract_input_signal(rows[i])
            x_t1 = extract_sagittal_state(rows[i + 1], rows[i])

            if np.all(np.isfinite(x_t)) and np.all(np.isfinite(x_t1)):
                X_vals.append(x_t)
                U_vals.append(u_t)
                Xn_vals.append(x_t1)

        if len(X_vals) >= 20:
            X_val = np.array(X_vals)
            U_val = np.array(U_vals)
            Xn_val = np.array(Xn_vals)

            # Prediction
            X_pred = (A @ X_val.T).T + (B @ U_val.T).T
            residuals = Xn_val - X_pred
            test1 = {
                "n_samples": len(X_val),
                "total_rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "rmse_per_state": np.sqrt(np.mean(residuals ** 2, axis=0)).tolist(),
                "nrmse_per_state": (np.sqrt(np.mean(residuals ** 2, axis=0)) /
                                    (np.std(X_val, axis=0) + 1e-8)).tolist(),
                "r_squared": float(1.0 - np.sum(residuals ** 2) /
                                   (np.sum((Xn_val - np.mean(Xn_val, axis=0)) ** 2) + 1e-12)),
            }
            print(f"  [1] One-step: R²={test1['r_squared']:.4f}, RMSE={test1['total_rmse']:.4f}, "
                  f"n={test1['n_samples']}")
        else:
            test1 = {"status": "INSUFFICIENT_VALIDATION_DATA", "n_samples": len(X_vals)}
            print(f"  [1] One-step: INSUFFICIENT_DATA (n={len(X_vals)})")
    else:
        test1 = {"status": "NO_TELEMETRY"}
        print(f"  [1] One-step: NO_TELEMETRY")

    validation["tests"]["one_step"] = test1

    # -- Test 2: 50-step rollout --
    if len(X_vals) >= 1:
        x0 = X_vals[0]
        u_seq = U_vals[:50] if len(U_vals) >= 50 else np.tile(U_vals, (50 // max(1, len(U_vals)) + 1, 1))[:50]
        traj_pred = n_step_rollout(A, B, x0, u_seq, 50)
        traj_true = Xn_vals[:50] if len(Xn_vals) >= 50 else Xn_vals

        err_50 = rollout_error(traj_pred, traj_true)
        test2 = err_50
        print(f"  [2] 50-step rollout: RMSE={test2['total_rmse']:.4f}, "
              f"diverged={test2.get('diverged', False)}")
        validation["tests"]["rollout_50"] = test2
    else:
        validation["tests"]["rollout_50"] = {"status": "NO_DATA"}

    # -- Test 3: 200-step rollout --
    if len(X_vals) >= 1:
        u_seq_200 = U_vals[:200] if len(U_vals) >= 200 else np.tile(U_vals, (200 // max(1, len(U_vals)) + 1, 1))[:200]
        traj_pred_200 = n_step_rollout(A, B, x0, u_seq_200, 200)
        traj_true_200 = Xn_vals[:200] if len(Xn_vals) >= 200 else Xn_vals
        err_200 = rollout_error(traj_pred_200, traj_true_200)
        test3 = err_200
        diverged_200 = test3.get("diverged", False)
        print(f"  [3] 200-step rollout: RMSE={test3['total_rmse']:.4f}, "
              f"diverged={diverged_200}")
        validation["tests"]["rollout_200"] = test3
    else:
        validation["tests"]["rollout_200"] = {"status": "NO_DATA"}

    # -- Test 4 & 5: Mode frequency and damping --
    mode = find_dominant_mode(A)
    if mode:
        freq_err = abs(mode["frequency_hz"] - 0.239)
        zeta_err = abs(mode["damping_ratio"] - 0.096)
        test4_5 = {
            "mode_found": True,
            "frequency_hz": mode["frequency_hz"],
            "damping_ratio": mode["damping_ratio"],
            "freq_error_hz": freq_err,
            "freq_error_pct": freq_err / 0.239 * 100,
            "zeta_error": zeta_err,
        }
        print(f"  [4/5] Mode: f={mode['frequency_hz']:.3f} Hz (err={freq_err:.3f} Hz), "
              f"zeta={mode['damping_ratio']:.3f} (err={zeta_err:.3f})")
    else:
        test4_5 = {"mode_found": False}
        print(f"  [4/5] Mode: NOT FOUND in target range")
    validation["tests"]["mode"] = test4_5

    # -- Test 6: Impulse response --
    if len(X_vals) >= 1:
        imp_traj = impulse_response(A, B, x0, impulse_magnitude=5.0, n_steps=100)
        # Check physical plausibility: no NaN/Inf, bounded
        physically_plausible = (
            np.all(np.isfinite(imp_traj)) and
            not np.any(np.abs(imp_traj) > 100)
        )
        test6 = {
            "physically_plausible": physically_plausible,
            "max_state_excursion": float(np.max(np.abs(imp_traj))),
            "settling_time_steps": _estimate_settling_time(imp_traj),
        }
        print(f"  [6] Impulse response: plausible={physically_plausible}, "
              f"max_excursion={test6['max_state_excursion']:.3f}")
    else:
        test6 = {"status": "NO_DATA"}
    validation["tests"]["impulse_response"] = test6

    # -- Test 7: Response-to-push prediction --
    # Use B_90n_push data if available
    validation["tests"]["push_response"] = {"status": "REQUIRES_DEDICATED_TELEMETRY"}

    # -- Test 8: Cross-run generalization --
    # Check if model generalizes across different run types
    validation["tests"]["cross_run"] = {"status": "REQUIRES_MULTI_RUN_TELEMETRY"}

    # -- Test 9: Cross-height interpolation --
    validation["tests"]["cross_height"] = {"status": "ANALYZED_IN_PHASE_5"}

    # -- Classification --
    classification = classify_model(validation)
    validation["classification"] = classification
    print(f"\n  Classification: {classification}")

    return validation


def _estimate_settling_time(traj, threshold_pct=0.05):
    """Estimate settling time (steps to stay within threshold% of final value)."""
    final = traj[-1]
    threshold = np.abs(final) * threshold_pct + 0.01  # absolute floor
    for k in range(len(traj) - 1, -1, -1):
        if np.any(np.abs(traj[k] - final) > threshold):
            return k + 1
    return 0


def classify_model(validation):
    """Classify model readiness for state-feedback design.

    Returns one of: DESIGN_READY, NEEDS_STATE_AUGMENTATION,
    INSUFFICIENT_EXCITATION, HEIGHT_DATA_INSUFFICIENT, UNSTABLE_ID_ARTIFACT,
    OVERFIT, INCONCLUSIVE.
    """
    mode_test = validation["tests"].get("mode", {})
    one_step = validation["tests"].get("one_step", {})
    rollout_50 = validation["tests"].get("rollout_50", {})
    rollout_200 = validation["tests"].get("rollout_200", {})
    impulse = validation["tests"].get("impulse_response", {})

    # Check for insufficient data
    if one_step.get("n_samples", 0) < 20:
        return "HEIGHT_DATA_INSUFFICIENT"

    # Check for unstable identification
    if not np.all(np.isfinite(one_step.get("total_rmse", float("nan")))):
        return "UNSTABLE_ID_ARTIFACT"

    # Check for overfit
    r2 = one_step.get("r_squared", 0)
    if r2 > 0.9995:
        return "OVERFIT"

    # Check mode capture
    if not mode_test.get("mode_found", False):
        return "NEEDS_STATE_AUGMENTATION"

    freq_err_pct = mode_test.get("freq_error_pct", 100)
    zeta_err = mode_test.get("zeta_error", 100)

    if freq_err_pct > 30 or zeta_err > 0.10:
        return "NEEDS_STATE_AUGMENTATION"

    # Check rollout divergence
    if rollout_50.get("diverged", True):
        return "NEEDS_STATE_AUGMENTATION"

    # Check physical plausibility
    if not impulse.get("physically_plausible", False):
        return "UNSTABLE_ID_ARTIFACT"

    # Check NRMSE using nrmse_per_state if available, otherwise total_rmse vs rmse magnitude
    nrmse_per_state = one_step.get("nrmse_per_state", None)
    if nrmse_per_state and len(nrmse_per_state) > 0:
        mean_nrmse = float(np.mean(nrmse_per_state))
    else:
        rmse_per_state = one_step.get("rmse_per_state", [])
        if rmse_per_state and len(rmse_per_state) > 0:
            mean_rmse = float(np.mean(rmse_per_state))
            mean_nrmse = mean_rmse / (mean_rmse + 1e-8)  # signal scale unknown, use RMSE itself
        else:
            mean_nrmse = 1.0  # Default — cannot determine

    if mean_nrmse > 5.0:
        return "INSUFFICIENT_EXCITATION"

    # DESIGN_READY criteria:
    # - Mode captured within ±15% frequency → freq_err_pct < 15
    # - Damping error within ±0.05 → zeta_err < 0.05
    # - 50-step rollout doesn't diverge
    # - Physically plausible
    # - B_id input response plausible
    if freq_err_pct <= 15 and zeta_err <= 0.05 and not rollout_50.get("diverged", True):
        return "DESIGN_READY"

    return "INCONCLUSIVE"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def validate_all_models(state_vector="x6_base"):
    """Validate all identified models."""
    print("=" * 72)
    print("PHASE 4: MODEL VALIDATION")
    print("=" * 72)

    heights = ["low_0p330", "mid_0p400", "high_0p480"]
    results = {}

    for height_name in heights:
        model = load_model(height_name, state_vector)
        validation = validate_model(height_name, model, str(TELEMETRY_PATH))
        results[height_name] = validation

    # Check cross-height interpolation feasibility
    results["cross_height_analysis"] = analyze_cross_height(results, state_vector)

    # Save
    out_path = OUTPUT_DIR / "model_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[OK] Validation results saved: {out_path}")

    # Print summary table
    print(f"\n{'=' * 72}")
    print(f"CLASSIFICATION SUMMARY")
    print(f"{'=' * 72}")
    for height_name in heights:
        cls = results[height_name].get("classification", "UNKNOWN")
        mode = results[height_name].get("tests", {}).get("mode", {})
        freq_str = f"f={mode['frequency_hz']:.3f}Hz" if mode.get("mode_found") else "NO MODE"
        print(f"  {height_name}: {cls:35s} | {freq_str}")

    return results


def analyze_cross_height(results, state_vector):
    """Analyze cross-height interpolation feasibility."""
    models = {}
    for height_name in results:
        if height_name.startswith("cross"):
            continue
        model = load_model(height_name, state_vector)
        if model is not None:
            models[height_name] = model["A"]

    if len(models) < 2:
        return {"feasible": False, "reason": "Need at least 2 identified models"}

    heights = list(models.keys())
    A_matrices = list(models.values())

    # Check eigenvalue continuity
    eigvals_by_height = {}
    for h_name, A in models.items():
        eigvals_by_height[h_name] = np.linalg.eigvals(A)

    # Check if modes vary smoothly with height
    target_h_map = {"low_0p330": 0.33, "mid_0p400": 0.40, "high_0p480": 0.48}

    mode_freqs = []
    mode_zetas = []
    for h_name, A in models.items():
        mode = find_dominant_mode(A)
        if mode:
            mode_freqs.append((target_h_map.get(h_name, 0.4), mode["frequency_hz"]))
            mode_zetas.append((target_h_map.get(h_name, 0.4), mode["damping_ratio"]))

    # Check linearity of frequency vs height
    linear_freq = len(mode_freqs) >= 2
    if linear_freq:
        hs, fs = zip(*sorted(mode_freqs))
        hs = np.array(hs)
        fs = np.array(fs)
        slope = (fs[-1] - fs[0]) / (hs[-1] - hs[0]) if hs[-1] != hs[0] else 0
        linear_freq = abs(slope) < 5.0  # Frequency shouldn't change too rapidly

    return {
        "feasible": linear_freq and len(models) >= 2,
        "n_models_available": len(models),
        "mode_frequencies_by_height": {h: f for h, f in mode_freqs},
        "mode_damping_by_height": {h: z for h, z in mode_zetas},
        "recommendation": "linear_interpolation" if linear_freq else "needs_more_heights",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate K1 identified state-space models"
    )
    parser.add_argument("--state-vector", type=str, default="x6_base")
    args = parser.parse_args()

    validate_all_models(args.state_vector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
