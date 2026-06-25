#!/usr/bin/env python3
"""
MuJoCo Mode Controllability and Observability Audit — Phase 7.

For each height and each model, compute:
  - Controllability matrix rank
  - PBH test for each mode
  - Mode controllability metrics
  - Observability from available telemetry states

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains or modify K1.
"""

import json
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
STATE_SPACE_PATH = INPUT_DIR / "state_space_model.json"

STATE_NAMES = [
    "pitch_x", "pitch_rate_x", "support_error",
    "support_velocity", "com_y_velocity", "wheel_vel_mean",
]
N_STATES = 6


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


def controllability_rank(A: np.ndarray, B: np.ndarray) -> dict:
    """Compute controllability matrix rank and singular values.

    C = [B, AB, A^2 B, ..., A^{n-1} B]
    """
    if A is None or B is None:
        return {"error": "null_matrix"}

    n = A.shape[0]
    if n != N_STATES:
        return {"error": f"unexpected_state_dim_{n}"}

    # Build controllability matrix
    C = np.zeros((n, n))  # B has 1 column, so C is n x n
    col = B.copy().flatten()
    for i in range(n):
        if i == 0:
            C[:, i] = col
        else:
            col = A @ col
            C[:, i] = col

    # Rank and singular values
    try:
        U, S, Vh = np.linalg.svd(C)
        singular_values = S.tolist()
        # Effective rank: count singular values above tolerance
        tol = max(S.max() * 1e-12, 1e-10)
        effective_rank = int(np.sum(S > tol))
    except np.linalg.LinAlgError:
        singular_values = []
        effective_rank = 0

    rank_deficiency = n - effective_rank

    # Identify which modes are uncontrollable via PBH test
    eigvals, right_vecs = np.linalg.eig(A.T)  # left eigenvectors of A
    uncontrollable_modes = []

    for i, lam in enumerate(eigvals):
        # PBH: mode i is uncontrollable if v_i^T @ B = 0
        # where v_i is left eigenvector (eigenvector of A^T)
        v = right_vecs[:, i]
        pbh_value = np.abs(np.dot(v, B.flatten()))
        is_controllable = pbh_value > 1e-8

        # Compute mode controllability metric
        if abs(lam) > 1e-10:
            # Controllability gramian approximation for mode i
            mode_ctrb = float(pbh_value)
        else:
            mode_ctrb = 0.0

        uncontrollable_modes.append({
            "mode_index": i,
            "eigenvalue": complex(lam.real, lam.imag),
            "eigenvalue_magnitude": float(abs(lam)),
            "pbh_value": float(pbh_value),
            "is_controllable": bool(is_controllable),
            "mode_controllability_metric": mode_ctrb,
        })

    # Full system controllability
    is_fully_controllable = effective_rank == n

    return {
        "controllability_matrix_rank": effective_rank,
        "state_dimension": n,
        "rank_deficiency": rank_deficiency,
        "is_fully_controllable": is_fully_controllable,
        "singular_values": singular_values,
        "condition_number": float(S[0] / S[-1]) if len(S) > 0 and S[-1] > 1e-15 else float("inf"),
        "uncontrollable_modes": uncontrollable_modes,
    }


def compute_observability_gramian(A: np.ndarray, C: np.ndarray, n_steps: int = 100) -> dict:
    """Compute observability Gramian for discrete-time system.

    Wo = Σ (A^T)^k C^T C A^k
    """
    if A is None or C is None:
        return {"error": "null_matrix"}

    n = A.shape[0]
    Wo = np.zeros((n, n))

    Ak = np.eye(n)
    for k in range(n_steps):
        term = Ak.T @ C.T @ C @ Ak
        Wo += term
        Ak = Ak @ A

    # Check observability
    try:
        U, S, Vh = np.linalg.svd(Wo)
        singular_values = S.tolist()
        tol = max(S.max() * 1e-12, 1e-10)
        observable_rank = int(np.sum(S > tol))
    except np.linalg.LinAlgError:
        singular_values = []
        observable_rank = 0

    is_observable = observable_rank == n

    return {
        "observability_rank": observable_rank,
        "state_dimension": n,
        "is_observable": is_observable,
        "gramian_singular_values": singular_values[:10],  # top 10
        "gramian_condition": float(S[0] / S[-1]) if len(S) > 0 and S[-1] > 1e-15 else float("inf"),
    }


def main():
    print("=" * 72)
    print("MUJOCO CONTROLLABILITY & OBSERVABILITY AUDIT — PHASE 7")
    print("=" * 72)

    if not STATE_SPACE_PATH.exists():
        print(f"ERROR: State-space model not found: {STATE_SPACE_PATH}")
        print("Run audit_mujoco_true_linearization.py first.")
        return 1

    with open(STATE_SPACE_PATH, "r") as f:
        model_data = json.load(f)

    all_results = {}

    # ── Analyze each open-loop model ──
    print("\n── Open-Loop Controllability ──")
    open_loop_data = model_data.get("open_loop", {})
    for h_str, ol in open_loop_data.items():
        A = np.array(ol.get("A_open_real", []))
        B = np.array(ol.get("B_open_real", []))
        if A.size == 0 or B.size == 0:
            continue

        print(f"\nHeight {h_str}:")
        ctrb = controllability_rank(A, B)

        if "error" in ctrb:
            print(f"  ERROR: {ctrb['error']}")
            continue

        print(f"  Controllability rank: {ctrb['controllability_matrix_rank']}/{ctrb['state_dimension']}")
        print(f"  Fully controllable: {ctrb['is_fully_controllable']}")
        print(f"  Condition number: {ctrb['condition_number']:.2e}")

        # Check dominant mode controllability
        for mode in ctrb.get("uncontrollable_modes", []):
            if mode["eigenvalue_magnitude"] > 0.99:
                print(f"    Mode {mode['mode_index']}: λ={mode['eigenvalue_magnitude']:.4f}, "
                      f"PBH={mode['pbh_value']:.2e}, ctrl={mode['is_controllable']}")

        all_results[f"open_loop_{h_str}"] = ctrb

    # ── Analyze each closed-loop model ──
    print("\n── Closed-Loop K1 Controllability ──")
    closed_loop_data = model_data.get("closed_loop_k1", {})
    for h_str, cl in closed_loop_data.items():
        A_cl = np.array(cl.get("A_closed_K1_real", []))
        if A_cl.size == 0:
            continue

        # For closed-loop, use B from open-loop (same input — wheel torque)
        # The controllability of A_closed with wheel torque as input
        ol_key = h_str
        ol = open_loop_data.get(ol_key, {})
        B = np.array(ol.get("B_open_real", []))
        if B.size == 0:
            print(f"\nHeight {h_str}: No B matrix available for closed-loop")
            continue

        print(f"\nHeight {h_str}:")
        ctrb = controllability_rank(A_cl, B)

        if "error" in ctrb:
            print(f"  ERROR: {ctrb['error']}")
            continue

        print(f"  Controllability rank: {ctrb['controllability_matrix_rank']}/{ctrb['state_dimension']}")
        print(f"  Fully controllable: {ctrb['is_fully_controllable']}")
        print(f"  Condition number: {ctrb['condition_number']:.2e}")

        # Dominant mode PBH
        for mode in ctrb.get("uncontrollable_modes", []):
            if mode["eigenvalue_magnitude"] > 0.99:
                print(f"    Mode {mode['mode_index']}: λ={mode['eigenvalue_magnitude']:.4f}, "
                      f"PBH={mode['pbh_value']:.2e}, ctrl={mode['is_controllable']}")

        all_results[f"closed_loop_{h_str}"] = ctrb

    # ── Observability analysis ──
    print("\n── Observability Analysis ──")

    # Define observation matrix C from available telemetry states
    # We observe: pitch_x, pitch_rate_x, support_error, com_y_velocity, wheel_vel_mean
    # support_velocity must be estimated internally
    C_obs = np.zeros((5, N_STATES))
    C_obs[0, 0] = 1.0  # observe pitch_x
    C_obs[1, 1] = 1.0  # observe pitch_rate_x
    C_obs[2, 2] = 1.0  # observe support_error
    C_obs[3, 4] = 1.0  # observe com_y_velocity
    C_obs[4, 5] = 1.0  # observe wheel_vel_mean
    # support_velocity (index 3) is NOT directly observed — must be estimated

    for h_str, cl in closed_loop_data.items():
        A_cl = np.array(cl.get("A_closed_K1_real", []))
        if A_cl.size == 0:
            continue
        print(f"\nHeight {h_str}:")

        obs = compute_observability_gramian(A_cl, C_obs)
        print(f"  Observability rank: {obs.get('observability_rank', '?')}/{obs.get('state_dimension', '?')}")
        print(f"  Is observable: {obs.get('is_observable', '?')}")

        all_results[f"observability_{h_str}"] = obs

    # ── Specific questions ──
    print("\n── Key Questions ──")

    # 1. Is 0.33-0.4 Hz mode controllable?
    print("\n[Q1] Is the 0.33-0.4 Hz mode controllable by wheel torque?")
    for key, ctrb in all_results.items():
        if not key.startswith("closed_loop_"):
            continue
        for mode in ctrb.get("uncontrollable_modes", []):
            if 0.2 < abs(mode["eigenvalue"].imag) / (2 * np.pi * 0.01) < 0.6:
                print(f"  {key}: PBH={mode['pbh_value']:.2e} → "
                      f"{'CONTROLLABLE' if mode['is_controllable'] else 'UNCONTROLLABLE'}")

    # 2. Is the mode observable?
    print("\n[Q2] Is the 0.33-0.4 Hz mode observable from telemetry?")
    for key, obs in all_results.items():
        if not key.startswith("observability_"):
            continue
        print(f"  {key}: rank={obs.get('observability_rank', '?')}/{obs.get('state_dimension', '?')} "
              f"→ {'OBSERVABLE' if obs.get('is_observable') else 'NOT FULLY OBSERVABLE'}")

    # 3. Does wheel torque have authority to move the dominant pole?
    print("\n[Q3] Does wheel torque have enough authority?")
    for h_str, ol in open_loop_data.items():
        B = np.array(ol.get("B_open_real", []))
        if B.size == 0:
            continue
        # Input sensitivity: norm of B column
        b_norm = float(np.linalg.norm(B))
        # Dominant entries
        dominant_input_states = []
        for i, name in enumerate(STATE_NAMES):
            if abs(B[i, 0]) > 0.01:
                dominant_input_states.append(f"{name}={B[i,0]:.4f}")
        print(f"  {h_str}: |B|={b_norm:.4f}, dominant entries: {dominant_input_states}")

    # ── Save results ──
    output_path = INPUT_DIR / "controllability_audit.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nControllability audit saved to: {output_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
