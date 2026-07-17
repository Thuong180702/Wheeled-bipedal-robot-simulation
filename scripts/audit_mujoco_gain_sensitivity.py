#!/usr/bin/env python3
"""
MuJoCo True Gain/Input Sensitivity — Phase 8.

For each height, perturb K1 gains ±10% and evaluate the effect on:
  - Closed-loop eigenvalues
  - Damping ratio of dominant mode
  - Natural frequency shift
  - Mode stability

Also evaluates paired perturbations to test whether cross-coupled gain
design is necessary.

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains or modify K1 permanently.
"""

import copy
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

CONTROL_DT = 0.01
STATE_NAMES = [
    "pitch_x", "pitch_rate_x", "support_error",
    "support_velocity", "com_y_velocity", "wheel_vel_mean",
]
N_STATES = 6

# K1 gains (read-only reference)
K1_GAINS = {
    "kp_pitch": 50.0,
    "kd_pitch": 10.0,
    "k_position": 40.0,
    "k_velocity": 15.0,
    "k_wheel_velocity": 0.5,
}

# K1 feedback mapping: which state each gain multiplies
# u = kp_pitch * pitch_x + kd_pitch * pitch_rate_x + ...
#   + k_position * (-support_error) + k_velocity * (-com_y_velocity)
#   + k_wheel_velocity * (-wheel_vel_mean)
GAIN_TO_STATE_SIGN = {
    "kp_pitch":       (0, +1),   # +kp * pitch_x
    "kd_pitch":       (1, +1),   # +kd * pitch_rate_x
    "k_position":     (2, -1),   # -k_pos * support_error
    "k_velocity":     (4, -1),   # -k_vel * com_y_velocity
    "k_wheel_velocity": (5, -1), # -k_wv * wheel_vel_mean
}


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


def compute_closed_loop_from_gains(
    A_open: np.ndarray,
    B_open: np.ndarray,
    gains: dict,
) -> np.ndarray:
    """Compute A_closed = A_open + B * K for a given set of gains.

    K is a 1×6 row vector mapping state to wheel torque.
    """
    n = A_open.shape[0]
    K = np.zeros((1, n))

    for gain_name, (state_idx, sign) in GAIN_TO_STATE_SIGN.items():
        K[0, state_idx] = sign * gains.get(gain_name, 0.0)

    A_closed = A_open + B_open @ K
    return A_closed


def get_dominant_oscillatory_mode(
    eigvals: np.ndarray,
    dt: float,
    target_range: tuple = (0.2, 0.6),
) -> dict | None:
    """Find the dominant oscillatory mode in the target frequency range."""
    best = None
    min_dist = float("inf")

    for lam in eigvals:
        if abs(lam.imag) < 1e-8:
            continue  # real eigenvalue

        s = np.log(lam) / dt
        omega_n = abs(s)
        freq_hz = omega_n / (2 * np.pi)

        if target_range[0] <= freq_hz <= target_range[1]:
            zeta = float(-s.real / omega_n) if omega_n > 1e-10 else -1.0
            dist = abs(freq_hz - 0.365)
            if dist < min_dist:
                min_dist = dist
                best = {
                    "eigenvalue": complex(lam.real, lam.imag),
                    "eigenvalue_magnitude": float(abs(lam)),
                    "frequency_hz": float(freq_hz),
                    "damping_ratio": zeta,
                    "stability": "STABLE" if abs(lam) < 0.999 else (
                        "MARGINAL" if abs(lam) < 1.001 else "UNSTABLE"
                    ),
                }

    return best


def main():
    print("=" * 72)
    print("MUJOCO TRUE GAIN SENSITIVITY — PHASE 8")
    print("=" * 72)

    if not STATE_SPACE_PATH.exists():
        print(f"ERROR: State-space model not found: {STATE_SPACE_PATH}")
        print("Run audit_mujoco_true_linearization.py first.")
        return 1

    with open(STATE_SPACE_PATH, "r") as f:
        model_data = json.load(f)

    all_results = {}
    open_loop_data = model_data.get("open_loop", {})

    for h_str, ol in open_loop_data.items():
        A_open = np.array(ol.get("A_open_real", []))
        B_open = np.array(ol.get("B_open_real", []))
        if A_open.size == 0 or B_open.size == 0:
            continue

        print(f"\n{'='*60}")
        print(f"Height: {h_str}")
        print(f"{'='*60}")

        # ── Nominal closed-loop ──
        A_cl_nominal = compute_closed_loop_from_gains(A_open, B_open, K1_GAINS)
        eig_nominal = np.linalg.eigvals(A_cl_nominal)
        mode_nominal = get_dominant_oscillatory_mode(eig_nominal, CONTROL_DT)

        if mode_nominal is None:
            print(f"  WARNING: No oscillatory mode in 0.2-0.6 Hz range for nominal K1")
            all_results[h_str] = {"error": "no_target_mode_found"}
            continue

        print(f"\n  Nominal K1 dominant mode:")
        print(f"    f = {mode_nominal['frequency_hz']:.4f} Hz")
        print(f"    ζ = {mode_nominal['damping_ratio']:.4f}")
        print(f"    |λ| = {mode_nominal['eigenvalue_magnitude']:.4f}")
        print(f"    Stability: {mode_nominal['stability']}")

        # ── Individual gain perturbations ±10% ──
        print(f"\n  ── Individual Gain Sensitivity (±10%) ──")
        individual_sensitivities = {}

        for gain_name, nominal_val in K1_GAINS.items():
            for pct in [-10, +10]:
                pert_val = nominal_val * (1.0 + pct / 100.0)
                gains_pert = copy.deepcopy(K1_GAINS)
                gains_pert[gain_name] = pert_val

                A_cl_pert = compute_closed_loop_from_gains(A_open, B_open, gains_pert)
                eig_pert = np.linalg.eigvals(A_cl_pert)
                mode_pert = get_dominant_oscillatory_mode(eig_pert, CONTROL_DT)

                if mode_pert is None:
                    continue

                delta_zeta = mode_pert["damping_ratio"] - mode_nominal["damping_ratio"]
                delta_freq = mode_pert["frequency_hz"] - mode_nominal["frequency_hz"]
                delta_mag = mode_pert["eigenvalue_magnitude"] - mode_nominal["eigenvalue_magnitude"]

                key = f"{gain_name}_{pct:+d}pct"
                individual_sensitivities[key] = {
                    "gain": gain_name,
                    "perturbation_pct": pct,
                    "nominal_value": nominal_val,
                    "perturbed_value": pert_val,
                    "frequency_hz": mode_pert["frequency_hz"],
                    "delta_frequency_hz": float(delta_freq),
                    "damping_ratio": mode_pert["damping_ratio"],
                    "delta_damping_ratio": float(delta_zeta),
                    "eigenvalue_magnitude": mode_pert["eigenvalue_magnitude"],
                    "delta_magnitude": float(delta_mag),
                    "stability": mode_pert["stability"],
                }

                # Sensitivity metric: |Δζ| / |Δgain%|
                abs_pct = abs(pct)
                sens = abs(delta_zeta) / (abs_pct / 100.0) if abs_pct > 0 else 0.0
                individual_sensitivities[key]["sensitivity_zeta_per_10pct"] = float(sens * 10)

                print(f"    {gain_name:20s} {pct:+4d}%: "
                      f"f={mode_pert['frequency_hz']:.4f}Hz "
                      f"(Δ={delta_freq:+.4f}), "
                      f"ζ={mode_pert['damping_ratio']:.4f} "
                      f"(Δ={delta_zeta:+.4f}), "
                      f"|λ|={mode_pert['eigenvalue_magnitude']:.4f} "
                      f"(Δ={delta_mag:+.4f})")

        # Compute sensitivity ranking
        max_sensitivities = {}
        for gain_name in K1_GAINS:
            sens_vals = []
            for key, val in individual_sensitivities.items():
                if val["gain"] == gain_name:
                    sens_vals.append(val["sensitivity_zeta_per_10pct"])
            max_sensitivities[gain_name] = max(sens_vals) if sens_vals else 0.0

        ranked = sorted(max_sensitivities.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Sensitivity ranking (most → least influential):")
        for i, (gain, sens) in enumerate(ranked):
            label = "NEGLIGIBLE" if sens < 0.1 else ("MODERATE" if sens < 0.5 else "SIGNIFICANT")
            print(f"    {i+1}. {gain}: sensitivity={sens:.4f} [{label}]")

        # ── Paired perturbations ──
        print(f"\n  ── Paired Gain Sensitivity ──")
        paired_results = {}

        pairs_to_test = [
            ("kd_pitch", "k_velocity"),     # pitch damping + velocity damping
            ("k_position", "k_velocity"),   # position centering + velocity
            ("kp_pitch", "k_position"),     # pitch stiffness + position
            ("kd_pitch", "k_position"),     # pitch damping + position
            ("kp_pitch", "k_velocity"),     # pitch + velocity
        ]

        for g1, g2 in pairs_to_test:
            for pct in [-10, +10]:
                gains_pert = copy.deepcopy(K1_GAINS)
                gains_pert[g1] = K1_GAINS[g1] * (1.0 + pct / 100.0)
                gains_pert[g2] = K1_GAINS[g2] * (1.0 + pct / 100.0)

                A_cl_pert = compute_closed_loop_from_gains(A_open, B_open, gains_pert)
                eig_pert = np.linalg.eigvals(A_cl_pert)
                mode_pert = get_dominant_oscillatory_mode(eig_pert, CONTROL_DT)

                if mode_pert is None:
                    continue

                delta_zeta = mode_pert["damping_ratio"] - mode_nominal["damping_ratio"]
                abs_pct = abs(pct)
                sens = abs(delta_zeta) / (abs_pct / 100.0) if abs_pct > 0 else 0.0

                key = f"{g1}+{g2}_{pct:+d}pct"
                paired_results[key] = {
                    "gain_pair": [g1, g2],
                    "perturbation_pct": pct,
                    "delta_damping_ratio": float(delta_zeta),
                    "sensitivity_zeta_per_10pct": float(sens * 10),
                }

                # Compare to sum of individual sensitivities
                ind_sens = (
                    max_sensitivities.get(g1, 0) + max_sensitivities.get(g2, 0)
                )
                synergy = sens * 10 - ind_sens
                paired_results[key]["individual_sensitivity_sum"] = float(ind_sens)
                paired_results[key]["synergy_surplus"] = float(synergy)

                print(f"    {g1}+{g2} {pct:+4d}%: "
                      f"Δζ={delta_zeta:+.4f}, "
                      f"sens={sens*10:.4f}, "
                      f"indiv_sum={ind_sens:.4f}, "
                      f"synergy={synergy:+.4f}")

        # Compile height results
        all_results[h_str] = {
            "nominal_mode": mode_nominal,
            "nominal_gains": K1_GAINS,
            "individual_sensitivities": individual_sensitivities,
            "sensitivity_ranking": [
                {"gain": g, "max_sensitivity": s}
                for g, s in ranked
            ],
            "most_influential_gain": ranked[0][0] if ranked else None,
            "least_influential_gain": ranked[-1][0] if ranked else None,
            "all_gains_negligible": all(s < 0.1 for _, s in ranked),
            "paired_sensitivities": paired_results,
        }

    # ── Cross-height summary ──
    print(f"\n{'='*60}")
    print("CROSS-HEIGHT SUMMARY")
    print(f"{'='*60}")

    for h_str, results in all_results.items():
        if "error" in results:
            print(f"\n  {h_str}: ERROR - {results['error']}")
            continue

        ranking = results.get("sensitivity_ranking", [])
        print(f"\n  Height {h_str}:")
        print(f"    Nominal: f={results['nominal_mode']['frequency_hz']:.3f}Hz, "
              f"ζ={results['nominal_mode']['damping_ratio']:.4f}, "
              f"|λ|={results['nominal_mode']['eigenvalue_magnitude']:.4f}")
        print(f"    All gains negligible: {results.get('all_gains_negligible', '?')}")
        print(f"    Most influential: {results.get('most_influential_gain')}")
        print(f"    Least influential: {results.get('least_influential_gain')}")

        # Answer key questions
        print(f"\n    Key conclusions:")
        if results.get("all_gains_negligible"):
            print(f"      → Independent scalar K1 gains are INSUFFICIENT to damp the mode")
            print(f"      → Cross-coupled gain design (state feedback) is JUSTIFIED")
        else:
            max_sens = ranking[0]["max_sensitivity"] if ranking else 0
            print(f"      → Max sensitivity = {max_sens:.4f}")

        # Check for synergy in paired perturbations
        paired = results.get("paired_sensitivities", {})
        max_synergy = 0.0
        for key, val in paired.items():
            if abs(val.get("synergy_surplus", 0)) > abs(max_synergy):
                max_synergy = val["synergy_surplus"]
        print(f"      → Max paired synergy: {max_synergy:+.4f}")
        if abs(max_synergy) > 0.05:
            print(f"      → Joint gain design has SYNERGY beyond independent gains")

    # ── Save results ──
    output_path = INPUT_DIR / "gain_sensitivity.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nGain sensitivity analysis saved to: {output_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
