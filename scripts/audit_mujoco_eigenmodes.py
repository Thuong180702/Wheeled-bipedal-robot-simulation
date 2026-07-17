#!/usr/bin/env python3
"""
MuJoCo Eigenmode Analysis — Phase 6.

For each height and each model (A_open_real, A_closed_K1_real, A_id), compute:
  - Eigenvalues (discrete-time)
  - Continuous-time equivalents
  - Natural frequency
  - Damping ratio
  - Mode stability
  - Left/right eigenvectors
  - Participation factors
  - Mode classification

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains or modify K1.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "mujoco_linearization"
OUTPUT_DIR = INPUT_DIR  # Save alongside input
STATE_SPACE_PATH = INPUT_DIR / "state_space_model.json"

CONTROL_DT = 0.01
STATE_NAMES = [
    "pitch_x", "pitch_rate_x", "support_error",
    "support_velocity", "com_y_velocity", "wheel_vel_mean",
]
TARGET_FREQ_RANGE = (0.25, 0.50)  # Hz — the observed 0.33-0.4 Hz mode


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
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
        return super().default(obj)


def compute_eigenvalues(A: np.ndarray, dt: float, label: str) -> list:
    """Compute eigenvalues and their properties.

    Returns list of dicts with eigenvalue properties.
    """
    if A is None or not np.all(np.isfinite(A)):
        return []

    eigvals, eigvecs = np.linalg.eig(A)
    results = []

    for i, lam in enumerate(eigvals):
        # Continuous-time equivalent: s = ln(λ) / dt
        if abs(lam) < 1e-14:
            s = complex(-np.inf, 0)
            freq_hz = 0.0
            zeta = 1.0
        else:
            s = np.log(lam) / dt
            # Natural frequency ω_n = |s|
            omega_n = abs(s)
            freq_hz = omega_n / (2 * np.pi)
            # Damping ratio ζ = -Re(s) / ω_n
            if omega_n > 1e-10:
                zeta = -s.real / omega_n
            else:
                zeta = 1.0 if s.real < 0 else -1.0

        # Stability
        abs_lam = abs(lam)
        if abs_lam < 0.999:
            stability = "STABLE"
        elif abs_lam < 1.001:
            stability = "MARGINAL"
        else:
            stability = "UNSTABLE"

        results.append({
            "index": i,
            "eigenvalue_dt": complex(lam.real, lam.imag),
            "eigenvalue_dt_real": float(lam.real),
            "eigenvalue_dt_imag": float(lam.imag),
            "magnitude": float(abs_lam),
            "eigenvalue_ct": complex(s.real, s.imag) if np.isfinite(s.real) else None,
            "frequency_hz": float(freq_hz),
            "damping_ratio": float(zeta),
            "stability": stability,
            "is_oscillatory": abs(lam.imag) > 1e-8,
            "label": label,
        })

    # Sort by magnitude descending
    results.sort(key=lambda x: x["magnitude"], reverse=True)
    return results


def compute_participation_factors(A: np.ndarray, dt: float) -> list:
    """Compute participation factors for each mode.

    Participation factor p_ki = |v_ki| * |w_ki| where v are right eigenvectors
    and w are left eigenvectors, normalized such that w^T * v = I.
    """
    if A is None or not np.all(np.isfinite(A)):
        return []

    n = A.shape[0]
    eigvals, right_vecs = np.linalg.eig(A)

    # Left eigenvectors: rows of inv(right_vecs)
    try:
        left_vecs = np.linalg.inv(right_vecs)
    except np.linalg.LinAlgError:
        return []

    # Normalize: w_i^T * v_j = δ_ij already satisfied by inv relationship
    # Participation: p_ki = |v_ki * w_ki|
    results = []
    for i in range(n):
        lam = eigvals[i]
        rv = right_vecs[:, i]   # right eigenvector (column)
        lv = left_vecs[i, :]    # left eigenvector (row)

        # Normalize: max participation = 1
        participation = np.abs(rv * lv)
        participation = participation / max(np.sum(participation), 1e-10)

        # Dominant state
        dominant_idx = int(np.argmax(participation))
        dominant_state = STATE_NAMES[dominant_idx] if dominant_idx < len(STATE_NAMES) else "unknown"

        results.append({
            "mode_index": i,
            "eigenvalue": complex(lam.real, lam.imag),
            "participation": {
                STATE_NAMES[j]: float(participation[j])
                for j in range(min(n, len(STATE_NAMES)))
            },
            "dominant_state": dominant_state,
            "dominant_participation": float(participation[dominant_idx]),
        })

    return results


def classify_mode(
    eig_result: dict,
    pf_result: dict | None,
    is_open_loop: bool,
) -> str:
    """Classify a mode based on its properties.

    Returns classification string.
    """
    freq = eig_result.get("frequency_hz", 0)
    zeta = eig_result.get("damping_ratio", 0)
    mag = eig_result.get("magnitude", 0)
    is_osc = eig_result.get("is_oscillatory", False)
    stability = eig_result.get("stability", "UNKNOWN")

    # Check if mode is in the observed 0.33-0.4 Hz range
    in_target_range = TARGET_FREQ_RANGE[0] <= freq <= TARGET_FREQ_RANGE[1]

    if not is_osc:
        # Real eigenvalue
        if mag > 1.001:
            return "UNSTABLE_REAL_POLE"
        elif mag > 0.999:
            return "MARGINAL_REAL_INTEGRATOR"
        else:
            return "STABLE_REAL_POLE"

    # Complex eigenvalue (oscillatory)
    if in_target_range:
        if abs(zeta) < 0.1:
            if is_open_loop:
                return "PLANT_STRUCTURAL_MODE_CRITICALLY_UNDAMPED"
            else:
                return "COUPLED_PITCH_SUPPORT_VELOCITY_MODE_CRITICALLY_UNDAMPED"
        elif abs(zeta) < 0.5:
            return "COUPLED_PITCH_SUPPORT_VELOCITY_MODE_UNDERDAMPED"
        else:
            return "COUPLED_PITCH_SUPPORT_VELOCITY_MODE_DAMPED"

    if freq < 0.05:
        return "SLOW_INTEGRATOR_MODE"

    if freq > 2.0:
        return "HIGH_FREQUENCY_MODE"  # possible WIP

    if abs(zeta) > 0.7:
        return "WELL_DAMPED_MODE"

    return "OSCILLATORY_MODE_UNCLASSIFIED"


def analyze_model(A: np.ndarray, dt: float, label: str, is_open_loop: bool) -> dict:
    """Full eigenmode analysis for one state matrix."""
    if A is None or not np.all(np.isfinite(A)):
        return {"error": "invalid_matrix", "label": label}

    eig_results = compute_eigenvalues(A, dt, label)
    pf_results = compute_participation_factors(A, dt)

    # Classify each mode
    for eig in eig_results:
        idx = eig["index"]
        pf = pf_results[idx] if idx < len(pf_results) else None
        eig["classification"] = classify_mode(eig, pf, is_open_loop)
        if pf:
            eig["participation"] = pf["participation"]
            eig["dominant_state"] = pf["dominant_state"]

    # Find mode closest to observed 0.33-0.4 Hz
    target_mode = None
    min_dist = float("inf")
    for eig in eig_results:
        if eig["is_oscillatory"]:
            freq = eig["frequency_hz"]
            dist = abs(freq - 0.365)  # center of 0.33-0.4 Hz
            if dist < min_dist:
                min_dist = dist
                target_mode = eig

    return {
        "label": label,
        "is_open_loop": is_open_loop,
        "matrix_shape": list(A.shape),
        "n_modes": len(eig_results),
        "eigenvalues": eig_results,
        "target_mode_0p33_0p4_hz": target_mode,
        "target_mode_confirmed": (
            target_mode is not None
            and TARGET_FREQ_RANGE[0] <= target_mode["frequency_hz"] <= TARGET_FREQ_RANGE[1]
        ),
    }


def main():
    print("=" * 72)
    print("MUJOCO EIGENMODE ANALYSIS — PHASE 6")
    print("=" * 72)

    if not STATE_SPACE_PATH.exists():
        print(f"ERROR: State-space model not found: {STATE_SPACE_PATH}")
        print("Run audit_mujoco_true_linearization.py first.")
        return 1

    with open(STATE_SPACE_PATH, "r") as f:
        model_data = json.load(f)

    dt = model_data.get("control_dt_s", CONTROL_DT)
    print(f"Control dt: {dt} s")
    print(f"Target frequency range: {TARGET_FREQ_RANGE[0]}-{TARGET_FREQ_RANGE[1]} Hz")

    all_results = {}

    # ── Open-loop analysis ──
    print("\n── Open-Loop Eigenmodes ──")
    open_loop_data = model_data.get("open_loop", {})
    for h_str, ol in open_loop_data.items():
        A = np.array(ol.get("A_open_real", []))
        if A.size == 0:
            continue
        print(f"\nHeight {h_str}:")
        result = analyze_model(A, dt, f"open_loop_{h_str}", is_open_loop=True)
        all_results[f"open_loop_{h_str}"] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        for eig in result["eigenvalues"]:
            osc = "↻" if eig["is_oscillatory"] else "→"
            print(f"  {osc} λ={eig['magnitude']:.4f} f={eig['frequency_hz']:.3f}Hz "
                  f"ζ={eig['damping_ratio']:.4f} [{eig['stability']}] "
                  f"→ {eig.get('classification', '?')}")

        target = result.get("target_mode_0p33_0p4_hz")
        if target:
            print(f"  >>> Target mode (0.33-0.4 Hz): f={target['frequency_hz']:.3f}Hz, "
                  f"ζ={target['damping_ratio']:.4f}, confirmed={result['target_mode_confirmed']}")

    # ── Closed-loop analysis ──
    print("\n── Closed-Loop K1 Eigenmodes ──")
    closed_loop_data = model_data.get("closed_loop_k1", {})
    for h_str, cl in closed_loop_data.items():
        A = np.array(cl.get("A_closed_K1_real", []))
        if A.size == 0:
            continue
        print(f"\nHeight {h_str}:")
        result = analyze_model(A, dt, f"closed_loop_k1_{h_str}", is_open_loop=False)
        all_results[f"closed_loop_k1_{h_str}"] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        for eig in result["eigenvalues"]:
            osc = "↻" if eig["is_oscillatory"] else "→"
            print(f"  {osc} λ={eig['magnitude']:.4f} f={eig['frequency_hz']:.3f}Hz "
                  f"ζ={eig['damping_ratio']:.4f} [{eig['stability']}] "
                  f"→ {eig.get('classification', '?')}"
                  f"  dom={eig.get('dominant_state', '?')}")

        target = result.get("target_mode_0p33_0p4_hz")
        if target:
            print(f"  >>> Target mode (0.33-0.4 Hz): f={target['frequency_hz']:.3f}Hz, "
                  f"ζ={target['damping_ratio']:.4f}, |λ|={target['magnitude']:.4f}, "
                  f"confirmed={result['target_mode_confirmed']}")
            if "participation" in target:
                print(f"  >>> Participation: {target['participation']}")

    # ── System ID analysis ──
    print("\n── Empirical System ID Eigenmodes ──")
    sysid_data = model_data.get("system_id", {})
    for h_str, sid in sysid_data.items():
        A_id_raw = sid.get("A_id")
        if A_id_raw is None:
            print(f"\nHeight {h_str}: No A_id available")
            continue
        A = np.array(A_id_raw)
        if A.size == 0:
            continue
        print(f"\nHeight {h_str}:")
        result = analyze_model(A, dt, f"system_id_{h_str}", is_open_loop=False)
        all_results[f"system_id_{h_str}"] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        for eig in result["eigenvalues"]:
            osc = "↻" if eig["is_oscillatory"] else "→"
            print(f"  {osc} λ={eig['magnitude']:.4f} f={eig['frequency_hz']:.3f}Hz "
                  f"ζ={eig['damping_ratio']:.4f} [{eig['stability']}]")

    # ── Save results ──
    output_path = OUTPUT_DIR / "eigenmode_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nEigenmode analysis saved to: {output_path}")

    # ── Key findings ──
    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)

    for key, result in all_results.items():
        if "error" in result:
            continue
        target = result.get("target_mode_0p33_0p4_hz")
        confirmed = result.get("target_mode_confirmed", False)
        print(f"\n{key}:")
        print(f"  0.33-0.4 Hz mode confirmed: {confirmed}")
        if target:
            print(f"  Frequency: {target['frequency_hz']:.3f} Hz")
            print(f"  Damping: ζ={target['damping_ratio']:.4f}")
            print(f"  Stability: {target['stability']}")
            print(f"  Classification: {target.get('classification', '?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
