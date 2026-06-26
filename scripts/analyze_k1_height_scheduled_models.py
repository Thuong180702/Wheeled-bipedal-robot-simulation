#!/usr/bin/env python3
"""
Analyze K1 Height-Scheduled Models — Phase 5.

Analyzes how identified dynamics vary with height (0.33m, 0.40m, 0.48m):
  - Dominant mode frequency by height
  - Damping by height
  - B_id gain by height
  - Participation/coupling structure by height
  - Whether linear interpolation K(h) would be plausible
  - Whether one common K could work across heights

Does NOT design K.

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains or modify K1.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# -- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"
MODELS_DIR = OUTPUT_DIR / "models"

CONTROL_DT = 0.01
STATE_NAMES = [
    "pitch_x", "pitch_rate_x", "support_error",
    "support_velocity", "com_y_velocity", "wheel_vel_mean",
]
TARGET_HEIGHTS_MAP = {"low_0p330": 0.33, "mid_0p400": 0.40, "high_0p480": 0.48}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_model(height_name, state_vector="x6_base"):
    """Load identified model from disk."""
    model_dir = MODELS_DIR / height_name / state_vector
    a_path = model_dir / "A_id.npy"
    b_path = model_dir / "B_id.npy"

    if not a_path.exists():
        return None

    A = np.load(str(a_path))
    B = np.load(str(b_path)) if b_path.exists() else np.zeros((A.shape[0], 1))
    return {"A": A, "B": B}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EIGENVALUE ANALYSIS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_eigenvalue_summary(A):
    """Compute all eigenvalues and their properties."""
    eigvals = np.linalg.eigvals(A)
    summary = []
    for i, lam in enumerate(eigvals):
        if abs(lam) < 1e-14:
            s = complex(-float("inf"), 0)
            freq_hz = 0.0
            zeta = 1.0
        else:
            s = np.log(lam) / CONTROL_DT
            omega_n = abs(s)
            freq_hz = omega_n / (2 * np.pi)
            zeta = -s.real / omega_n if omega_n > 1e-10 else 1.0

        is_oscillatory = abs(lam.imag) > 1e-10

        summary.append({
            "index": i,
            "eigenvalue": complex(lam.real, lam.imag),
            "frequency_hz": float(freq_hz),
            "damping_ratio": float(zeta),
            "magnitude": float(abs(lam)),
            "is_oscillatory": bool(is_oscillatory),
            "stability": "STABLE" if abs(lam) < 0.999 else ("MARGINAL" if abs(lam) < 1.001 else "UNSTABLE"),
        })

    # Sort by magnitude descending
    summary.sort(key=lambda x: x["magnitude"], reverse=True)
    return summary


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARTICIPATION FACTORS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_participation_factors(A):
    """Compute participation factors for each mode."""
    eigvals, eigvecs_right = np.linalg.eig(A)
    try:
        eigvecs_left = np.linalg.inv(eigvecs_right).T
    except np.linalg.LinAlgError:
        return None

    results = []
    for k in range(len(eigvals)):
        p_k = {}
        for i in range(A.shape[0]):
            p_ki = abs(eigvecs_right[i, k] * eigvecs_left[i, k])
            p_k[STATE_NAMES[i]] = float(p_ki)

        # Normalize
        total = sum(p_k.values())
        if total > 1e-12:
            p_k = {k: v / total for k, v in p_k.items()}

        results.append({"mode_index": k, "eigenvalue": complex(eigvals[k].real, eigvals[k].imag), "participation": p_k})

    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B MATRIX ANALYSIS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def analyze_b_matrix(B):
    """Analyze B_id input coupling."""
    b_flat = B.flatten()
    b_norm = float(np.linalg.norm(b_flat))
    b_max_abs = float(np.max(np.abs(b_flat)))
    b_argmax = int(np.argmax(np.abs(b_flat)))

    coupling = {}
    for i in range(len(b_flat)):
        if abs(b_flat[i]) > 0.01 * b_max_abs:
            coupling[STATE_NAMES[i]] = float(b_flat[i])

    return {
        "b_norm": b_norm,
        "b_max_abs": b_max_abs,
        "b_max_state": STATE_NAMES[b_argmax] if b_argmax < len(STATE_NAMES) else f"state_{b_argmax}",
        "direct_coupling": coupling,
        "b_vector": b_flat.tolist(),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HEIGHT SCHEDULE ANALYSIS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def analyze_height_schedule(state_vector="x6_base"):
    """Analyze identified dynamics variation with height."""
    print("=" * 72)
    print("PHASE 5: HEIGHT SCHEDULE ANALYSIS")
    print("=" * 72)

    results = {
        "state_vector": state_vector,
        "dt_s": CONTROL_DT,
    }

    # Load all models
    models = {}
    for height_name, target_h in TARGET_HEIGHTS_MAP.items():
        model = load_model(height_name, state_vector)
        if model is not None:
            models[height_name] = model
            print(f"\n[{height_name}] h={target_h}m: Model loaded "
                  f"(cond={np.linalg.cond(model['A']):.2e})")

    if not models:
        print("[WARN] No models available. Run Phase 3 first.")
        return {"status": "NO_MODELS_AVAILABLE"}

    # -- Dominant mode by height --
    print(f"\n{'-' * 60}")
    print("Dominant Oscillatory Mode by Height")
    print(f"{'-' * 60}")

    mode_by_height = {}
    for h_name, model in models.items():
        A = model["A"]
        eig_summary = compute_eigenvalue_summary(A)
        results.setdefault("eigenvalues", {})[h_name] = eig_summary

        # Find dominant oscillatory mode in target range
        best = None
        for e in eig_summary:
            if e["is_oscillatory"] and 0.15 <= e["frequency_hz"] <= 0.55:
                if best is None or e["magnitude"] > best["magnitude"]:
                    best = e

        target_h = TARGET_HEIGHTS_MAP[h_name]
        if best:
            mode_by_height[h_name] = {
                "height_m": target_h,
                "frequency_hz": best["frequency_hz"],
                "damping_ratio": best["damping_ratio"],
                "magnitude": best["magnitude"],
                "stability": best["stability"],
            }
            print(f"  h={target_h:.2f}m: f={best['frequency_hz']:.3f} Hz, "
                  f"zeta={best['damping_ratio']:.3f}, |lambda|={best['magnitude']:.4f}, "
                  f"{best['stability']}")
        else:
            mode_by_height[h_name] = {
                "height_m": target_h,
                "frequency_hz": None,
                "damping_ratio": None,
                "mode_found": False,
            }
            print(f"  h={target_h:.2f}m: NO oscillatory mode in 0.15-0.55 Hz")

    results["mode_by_height"] = mode_by_height

    # -- B_id gain by height --
    print(f"\n{'-' * 60}")
    print("B Matrix Input Coupling by Height")
    print(f"{'-' * 60}")

    b_by_height = {}
    for h_name, model in models.items():
        b_analysis = analyze_b_matrix(model["B"])
        b_by_height[h_name] = b_analysis
        target_h = TARGET_HEIGHTS_MAP[h_name]
        print(f"  h={target_h:.2f}m: |B|={b_analysis['b_norm']:.4f}, "
              f"max_state={b_analysis['b_max_state']}, "
              f"direct_coupling={list(b_analysis['direct_coupling'].keys())}")

    results["b_by_height"] = b_by_height

    # -- Participation factors by height --
    print(f"\n{'-' * 60}")
    print("Participation / Coupling Structure by Height")
    print(f"{'-' * 60}")

    participation_by_height = {}
    for h_name, model in models.items():
        pf = compute_participation_factors(model["A"])
        if pf is not None:
            participation_by_height[h_name] = pf
            # Find dominant mode participation
            for mode_pf in pf:
                e = mode_pf["eigenvalue"]
                if abs(e.imag) > 1e-10:
                    s = np.log(e) / CONTROL_DT
                    freq = abs(s) / (2 * np.pi)
                    if 0.15 <= freq <= 0.55:
                        top_states = sorted(mode_pf["participation"].items(),
                                          key=lambda x: x[1], reverse=True)[:3]
                        print(f"  {h_name} mode@{freq:.2f}Hz: "
                              f"{', '.join(f'{s}:{p:.2f}' for s, p in top_states)}")

    results["participation_by_height"] = participation_by_height

    # -- Height interpolation feasibility --
    print(f"\n{'-' * 60}")
    print("Height Interpolation Feasibility")
    print(f"{'-' * 60}")

    hs = []
    fs = []
    zs = []
    for h_name, mode_data in mode_by_height.items():
        if mode_data.get("frequency_hz") is not None:
            hs.append(mode_data["height_m"])
            fs.append(mode_data["frequency_hz"])
            zs.append(mode_data["damping_ratio"])

    feasibility = {}

    if len(fs) >= 2:
        # Sort by height
        sorted_pairs = sorted(zip(hs, fs, zs))
        hs_sorted, fs_sorted, zs_sorted = zip(*sorted_pairs)

        # Check linearity of frequency vs height
        hs_arr = np.array(hs_sorted)
        fs_arr = np.array(fs_sorted)
        slope_freq = (fs_arr[-1] - fs_arr[0]) / (hs_arr[-1] - hs_arr[0]) if hs_arr[-1] != hs_arr[0] else 0
        freq_range = max(fs_arr) - min(fs_arr)

        feasibility["linear_interpolation_K_h"] = abs(slope_freq) < 5.0
        feasibility["frequency_variation_hz"] = float(freq_range)
        feasibility["frequency_slope_hz_per_m"] = float(slope_freq)

        # Check if one common K could work
        feasibility["one_common_K"] = freq_range < 0.10  # Less than 0.1 Hz variation
        feasibility["damping_variation"] = float(max(zs_sorted) - min(zs_sorted))
        feasibility["damping_stable"] = all(z >= -0.05 for z in zs_sorted)  # All stable or marginal

        print(f"  Frequency slope: {slope_freq:.3f} Hz/m (range: {freq_range:.3f} Hz)")
        print(f"  Linear interpolation K(h) plausible: {feasibility['linear_interpolation_K_h']}")
        print(f"  One common K feasible: {feasibility['one_common_K']}")
        print(f"  Frequency variation: {freq_range:.3f} Hz")
        print(f"  Damping variation: {feasibility['damping_variation']:.3f}")

    else:
        feasibility = {
            "linear_interpolation_K_h": False,
            "one_common_K": False,
            "reason": f"Only {len(fs)} models with identifiable modes — need >=2",
        }
        print(f"  INSUFFICIENT: {len(fs)} models with modes (need >=2)")

    results["height_scheduling_feasibility"] = feasibility

    # -- Recommendations --
    recommendations = []
    if feasibility.get("one_common_K", False):
        recommendations.append("One common K could work across heights — simplest implementation")
    elif feasibility.get("linear_interpolation_K_h", False):
        recommendations.append("Gain-scheduled K(h) with linear interpolation recommended")
    else:
        recommendations.append("Need non-monotonic or per-height K maps — may need more heights")

    if feasibility.get("damping_stable", True):
        recommendations.append("All modes are stable or marginally stable — no aggressive damping needed")
    else:
        recommendations.append("Some modes are UNSTABLE — need stronger damping at those heights")

    results["recommendations"] = recommendations
    print(f"\nRecommendations:")
    for r in recommendations:
        print(f"  - {r}")

    # Save
    out_path = OUTPUT_DIR / "height_schedule_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"\n[OK] Height schedule analysis saved: {out_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze K1 height-scheduled identified models"
    )
    parser.add_argument("--state-vector", type=str, default="x6_base")
    args = parser.parse_args()

    analyze_height_schedule(args.state_vector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
