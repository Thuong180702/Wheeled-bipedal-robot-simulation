#!/usr/bin/env python3
"""
Audit K1 Identified Model Control Feasibility — Phase 6.

Using identified design-ready models, computes:
  - Controllability rank and PBH test for dominant mode
  - Input authority by mode
  - Observability/estimability
  - Condition number
  - Candidate pole target feasibility
  - Estimated torque demand for moving dominant pole

Includes analysis-only LQR benchmark and pole-placement benchmark.
These are feasibility calculations, NOT controller implementation.

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT create controller candidates.

Output: outputs/k1_identification_dataset/control_feasibility.json
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
MAX_TORQUE = 5.0  # K1 max wheel torque (Nm)

TARGET_HEIGHTS_MAP = {"low_0p330": 0.33, "mid_0p400": 0.40, "high_0p480": 0.48}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_model(height_name, state_vector="x6_base"):
    """Load identified model."""
    model_dir = MODELS_DIR / height_name / state_vector
    a_path = model_dir / "A_id.npy"
    b_path = model_dir / "B_id.npy"

    if not a_path.exists():
        return None

    A = np.load(str(a_path))
    B = np.load(str(b_path)) if b_path.exists() else np.zeros((A.shape[0], 1))
    return {"A": A, "B": B}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONTROLLABILITY                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def controllability_rank(A, B):
    """Compute controllability matrix rank via SVD."""
    n = A.shape[0]
    C = np.zeros((n, n * B.shape[1] if B.ndim > 1 else n))

    col = 0
    B_mat = B.reshape(n, -1)
    for k in range(n):
        C[:, col:col + B_mat.shape[1]] = np.linalg.matrix_power(A, k) @ B_mat
        col += B_mat.shape[1]

    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    tol = max(S.max() * 1e-12, 1e-10) if S.size > 0 else 1e-10
    rank = int(np.sum(S > tol))

    return {
        "controllability_matrix_rank": rank,
        "n_states": n,
        "is_fully_controllable": rank >= n,
        "uncontrollable_dimension": n - rank,
        "singular_values": S.tolist()[:n],
    }


def pbh_test(A, B, target_mode):
    """PBH test: check if target mode is controllable.

    For each eigenvalue lambda, rank([lambda*I - A, B]) must be n.
    """
    n = A.shape[0]
    lam = target_mode
    M = np.hstack([lam * np.eye(n) - A, B.reshape(n, -1)])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    tol = max(S.max() * 1e-12, 1e-10) if S.size > 0 else 1e-10
    rank = int(np.sum(S > tol))

    return {
        "eigenvalue": complex(lam),
        "augmented_rank": rank,
        "n_states": n,
        "is_controllable": rank >= n,
        "rank_deficiency": n - rank,
    }


def input_authority_by_mode(A, B):
    """Compute input authority for each mode: |w_k^T * B|.

    Where w_k is the left eigenvector for mode k.
    """
    eigvals, eigvecs_right = np.linalg.eig(A)
    try:
        eigvecs_left = np.linalg.inv(eigvecs_right).T
    except np.linalg.LinAlgError:
        return []

    B_vec = B.flatten()
    authorities = []
    for k in range(len(eigvals)):
        w_k = eigvecs_left[:, k]
        authority = abs(np.dot(w_k, B_vec))

        if abs(eigvals[k].imag) > 1e-10:
            s = np.log(eigvals[k]) / CONTROL_DT
            freq = abs(s) / (2 * np.pi)
        else:
            freq = 0.0

        authorities.append({
            "mode_index": k,
            "eigenvalue": complex(eigvals[k].real, eigvals[k].imag),
            "frequency_hz": float(freq),
            "input_authority": float(authority),
            "normalized_authority": float(authority / (np.linalg.norm(w_k) + 1e-12)),
        })

    # Sort by authority descending
    authorities.sort(key=lambda x: x["input_authority"], reverse=True)
    return authorities


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVABILITY                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_observability(A, C=None):
    """Compute observability of the system."""
    n = A.shape[0]
    if C is None:
        C = np.eye(n)

    O_blocks = [C]
    for k in range(1, n):
        O_blocks.append(np.linalg.matrix_power(A, k) @ C)
    O = np.vstack(O_blocks)

    U, S, Vt = np.linalg.svd(O, full_matrices=False)
    tol = max(S.max() * 1e-12, 1e-10) if S.size > 0 else 1e-10
    rank = int(np.sum(S > tol))

    return {
        "observability_rank": rank,
        "n_states": n,
        "is_fully_observable": rank >= n,
        "unobservable_dimension": n - rank,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ANALYSIS-ONLY BENCHMARKS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def lqr_benchmark(A, B, Q=None, R=None):
    """Analysis-only LQR benchmark on identified model.

    NOT a controller implementation — feasibility analysis only.
    """
    n = A.shape[0]
    m = B.shape[1] if B.ndim > 1 else 1

    if Q is None:
        # Default: penalize pitch states most, then support, then velocity
        Q = np.diag([50.0, 10.0, 20.0, 5.0, 10.0, 1.0])
    if R is None:
        R = 0.1 * np.eye(m)

    # Discrete-time DARE via iterative method
    P = Q.copy()
    for _ in range(100):
        BtPB = B.T @ P @ B
        if B.ndim > 1:
            reg = R + BtPB
        else:
            reg = R + BtPB if BtPB.ndim == 0 else R + BtPB[0, 0]

        try:
            K = np.linalg.solve(reg, B.T @ P @ A)
        except np.linalg.LinAlgError:
            break

        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < 1e-8:
            P = P_next
            break
        P = P_next

    # Final gain
    try:
        if B.ndim > 1:
            K_final = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        else:
            denom = R + float(B.T @ P @ B)
            K_final = (B.T @ P @ A) / denom
    except (np.linalg.LinAlgError, ValueError):
        K_final = np.zeros((m, n))

    A_cl = A - B @ K_final
    eig_cl = np.linalg.eigvals(A_cl)

    # Estimate torque demand
    max_torque_est = float(np.max(np.abs(K_final)) * 0.1)  # Rough: max_gain * typical_state

    return {
        "K_lqr": K_final.tolist(),
        "closed_loop_eigenvalues": [
            {"real": float(v.real), "imag": float(v.imag)}
            for v in eig_cl
        ],
        "estimated_max_torque_demand": max_torque_est,
        "torque_within_budget": max_torque_est <= MAX_TORQUE,
        "note": "FEASIBILITY_BENCHMARK_ONLY — NOT a controller implementation. Analysis artifact only.",
    }


def pole_placement_benchmark(A, B, target_damping=0.7):
    """Analysis-only pole placement benchmark.

    Computes: if we could place poles, what torque would it take?
    Uses Ackermann-like formula for single-input systems.

    NOT a controller implementation — feasibility analysis only.
    """
    n = A.shape[0]
    b = B.flatten()

    # Check controllability first
    ctrl = controllability_rank(A, B)
    if not ctrl["is_fully_controllable"]:
        return {
            "feasible": False,
            "reason": f"System not fully controllable (rank={ctrl['controllability_matrix_rank']}/{n})",
            "note": "FEASIBILITY_BENCHMARK_ONLY — NOT a controller implementation.",
        }

    # Target: move eigenvalues to desired damping while preserving frequency
    eigvals = np.linalg.eigvals(A)
    target_poles = []
    for lam in eigvals:
        if abs(lam.imag) > 1e-10:
            s = np.log(lam) / CONTROL_DT
            omega_n = abs(s)
            # Set target damping
            s_target = complex(-target_damping * omega_n, omega_n * np.sqrt(max(0, 1 - target_damping ** 2)))
            lam_target = np.exp(s_target * CONTROL_DT)
            target_poles.append(lam_target)
        else:
            # Real pole: try to pull inside unit circle
            if abs(lam) >= 1.0:
                target_poles.append(0.95 * lam / abs(lam))  # Pull to 0.95
            else:
                target_poles.append(lam)  # Leave stable poles

    # Ackermann for single-input
    try:
        # Place at target_poles
        poly_target = np.poly(target_poles)
        # Cayley-Hamilton: p(A) = A^n - poly_target[n-1]*A^{n-1} - ... - poly_target[0]*I
        p_A = np.linalg.matrix_power(A, n)
        for i in range(n):
            p_A -= poly_target[i] * np.linalg.matrix_power(A, n - 1 - i)

        # Controllability matrix
        C = np.zeros((n, n))
        for i in range(n):
            C[:, i] = (np.linalg.matrix_power(A, i) @ b).flatten()

        e_n = np.zeros(n)
        e_n[-1] = 1.0

        K_acker = e_n @ np.linalg.inv(C) @ p_A
        max_torque = float(np.max(np.abs(K_acker)) * 0.1)

        return {
            "feasible": True,
            "target_damping": target_damping,
            "K_acker": K_acker.tolist(),
            "estimated_max_torque_Nm": max_torque,
            "torque_within_budget": max_torque <= MAX_TORQUE,
            "note": "FEASIBILITY_BENCHMARK_ONLY — NOT a controller implementation. Analysis artifact only.",
        }
    except (np.linalg.LinAlgError, ValueError) as e:
        return {
            "feasible": False,
            "reason": str(e),
            "note": "FEASIBILITY_BENCHMARK_ONLY — NOT a controller implementation.",
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN AUDIT                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def audit_control_feasibility(state_vector="x6_base"):
    """Run full control feasibility audit on identified models."""
    print("=" * 72)
    print("PHASE 6: PRE-DESIGN CONTROLLABILITY AND FEASIBILITY")
    print("=" * 72)

    all_results = {"state_vector": state_vector, "heights": {}}

    for height_name, target_h in TARGET_HEIGHTS_MAP.items():
        print(f"\n{'-' * 60}")
        print(f"{height_name} (h={target_h}m)")
        print(f"{'-' * 60}")

        model = load_model(height_name, state_vector)
        if model is None:
            print(f"  [SKIP] No model available")
            all_results["heights"][height_name] = {"status": "NO_MODEL"}
            continue

        A, B = model["A"], model["B"]
        n = A.shape[0]

        result = {"height_name": height_name, "height_m": target_h, "n_states": n}

        # -- Condition number --
        cond = float(np.linalg.cond(A))
        result["condition_number"] = cond
        result["well_conditioned"] = cond < 1e6
        print(f"  Condition: {cond:.2e} (well-conditioned={result['well_conditioned']})")

        # -- Controllability --
        ctrl = controllability_rank(A, B)
        result["controllability"] = ctrl
        print(f"  Controllability rank: {ctrl['controllability_matrix_rank']}/{n} "
              f"(fully={ctrl['is_fully_controllable']})")

        # -- PBH test for dominant mode --
        eigvals = np.linalg.eigvals(A)
        oscillator_modes = [v for v in eigvals if abs(v.imag) > 1e-10]
        if oscillator_modes:
            dominant = max(oscillator_modes, key=lambda v: abs(v))
            pbh = pbh_test(A, B, dominant)
            result["pbh_dominant_mode"] = pbh
            print(f"  PBH (dominant oscillatory mode): controllable={pbh['is_controllable']}")

        # -- Input authority by mode --
        authorities = input_authority_by_mode(A, B)
        result["input_authority"] = authorities[:4]  # Top 4
        if authorities:
            top = authorities[0]
            print(f"  Max input authority: mode {top['mode_index']} "
                  f"(f={top['frequency_hz']:.2f}Hz, authority={top['input_authority']:.4f})")

            # Check if oscillatory mode has authority
            osc_modes = [a for a in authorities if 0.15 <= a["frequency_hz"] <= 0.55]
            if osc_modes:
                print(f"  0.15-0.55 Hz mode authority: {osc_modes[0]['input_authority']:.4f}")
            else:
                print(f"  [WARN] No oscillatory mode in 0.15-0.55 Hz — input doesn't reach target mode")

        # -- Observability --
        obs = compute_observability(A)
        result["observability"] = obs
        print(f"  Observability rank: {obs['observability_rank']}/{n} "
              f"(fully={obs['is_fully_observable']})")

        # -- Analysis-only LQR benchmark --
        print(f"\n  [Feasibility benchmark] LQR analysis:")
        lqr = lqr_benchmark(A, B)
        result["lqr_benchmark"] = lqr
        print(f"    Estimated max torque: {lqr['estimated_max_torque_demand']:.2f} Nm "
              f"(budget={lqr['torque_within_budget']})")

        # -- Analysis-only pole placement benchmark --
        print(f"  [Feasibility benchmark] Pole placement (target zeta=0.7):")
        pole = pole_placement_benchmark(A, B, target_damping=0.7)
        result["pole_placement_benchmark"] = pole
        if pole["feasible"]:
            print(f"    Estimated max torque: {pole['estimated_max_torque_Nm']:.2f} Nm "
                  f"(budget={pole['torque_within_budget']})")
        else:
            print(f"    NOT feasible: {pole.get('reason', 'unknown')}")

        # -- Summary --
        result["design_ready"] = (
            ctrl["is_fully_controllable"] and
            obs["is_fully_observable"] and
            result["well_conditioned"]
        )
        print(f"\n  Design readiness: {result['design_ready']}")

        all_results["heights"][height_name] = result

    # -- Overall feasibility --
    all_heights_ready = all(
        r.get("design_ready", False)
        for r in all_results["heights"].values()
        if r.get("status") != "NO_MODEL"
    )
    all_results["overall_design_ready"] = all_heights_ready

    print(f"\n{'=' * 72}")
    print(f"OVERALL DESIGN READINESS: {all_heights_ready}")
    print(f"{'=' * 72}")

    # -- Recommendations --
    recommendations = []
    if all_heights_ready:
        recommendations.append("State-feedback design IS feasible on identified models")
        recommendations.append("Next: design gain-scheduled K(h) via LQR or pole placement on A_id(h), B_id(h)")
    else:
        for h_name, r in all_results["heights"].items():
            if r.get("status") == "NO_MODEL":
                recommendations.append(f"{h_name}: Need identified model first")
            else:
                if not r.get("controllability", {}).get("is_fully_controllable", False):
                    recommendations.append(f"{h_name}: Augment B matrix or state vector for better controllability")
                if not r.get("observability", {}).get("is_fully_observable", False):
                    recommendations.append(f"{h_name}: Add sensors/estimators for unobservable states")
                if not r.get("well_conditioned", False):
                    recommendations.append(f"{h_name}: Use regularization or reduce state dimension")

    all_results["recommendations"] = recommendations

    # Save
    out_path = OUTPUT_DIR / "control_feasibility.json"
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
        json.dump(all_results, f, indent=2, default=_make_serializable)

    print(f"\n[OK] Control feasibility audit saved: {out_path}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Audit K1 identified model control feasibility"
    )
    parser.add_argument("--state-vector", type=str, default="x6_base")
    args = parser.parse_args()

    audit_control_feasibility(args.state_vector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
