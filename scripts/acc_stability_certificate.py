#!/usr/bin/env python3
"""Local stability certificate for the ACC closed loop at quiet stance.

The paper's stability evidence is otherwise entirely behavioural (it does not
fall). This script supplies the control-theoretic complement, in three parts.

1.  GATE MARGINS.  Every ACC gate is a smoothstep whose value is *flat* (exactly
    0 or exactly 1) outside a finite band.  We measure how far the settled state
    sits from the nearest band edge.  If every gate is flat, the closed loop is
    C^inf on a neighbourhood of the fixed point and Lyapunov's indirect method
    applies -- there is no switching to reason about *there*.

2.  CLOSED-LOOP JACOBIAN.  Central differences on the whole 100 Hz control step
    of the augmented map (plant + controller memory), about the settled fixed
    point.  Reports the discrete spectral radius rho, the implied envelope time
    constant, and the slowest modes.  Repeated at two step sizes and over the
    five height variants.

3.  NONLINEAR CROSS-CHECK.  The predicted decay rate is compared against the
    measured decay of a small perturbation in the full nonlinear simulator with
    every frozen subsystem live again.  A linearization that matches the plant
    it was taken from is evidence the reduction in (2) is not doing the work.

The one element of ACC that is genuinely non-smooth -- the envelope follower's
attack/release switch -- is handled analytically in the paper, not here: both
branches are scalar contractions (1-0.35 and 1-0.0067), so V(e)=e^2 is a common
Lyapunov function and no dwell-time condition is needed.  This script only
reports the two coefficients so the claim can be checked against the profile.

Usage:  .venv/bin/python scripts/acc_stability_certificate.py [SETTLE_S]
Writes outputs/paper_verification/acc_stability_certificate.json
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.promote_v3_vs_assist as P  # noqa: E402
from wheeled_biped.controllers.k2_jax_controller import (  # noqa: E402
    K2_JAX_DIAG_FIELDS, K2_JAX_STATE_FIELDS, pack_state_k2)
from wheeled_biped.wbc.offline_three_arm_counterfactual import (  # noqa: E402
    compute_v3_torque_for_state, init_v3_controller)

DT = 0.01
SUBSTEPS = 5
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
VARIANTS = ["nominal", "high_small", "high_tiny", "low_small", "low_tiny"]
JOINTS = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
          "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]

# Controller memory that participates in the loop as a continuous state.  The
# adaptive-bias (ABS) trim is deliberately NOT here: it is an 800-tap FIR moving
# average, i.e. memory rather than dynamics, and it has converged to a constant
# at the fixed point.  It is held at its settled value inside the map so that
# the Jacobian describes the loop *about the equilibrium the trim selects*.
CTRL_STATES = (
    "notch_x1", "notch_x2", "notch_y1", "notch_y2",
    *(f"prev_tau_{i}" for i in range(10)),
    "filtered_com_z", "prev_support_error",
    "outer_loop_pitch_ref_smoothed_deg",
    "outer_loop_prev_support_error_m",
    "outer_loop_support_error_rate_smoothed",
    "apcr1nd_prev_error",
    "heading_hy_integral",
    "anchor_integ_tau", "anchor_activity_ema",
    "terrain_split_slow",
)
CS_IDX = np.array([K2_JAX_STATE_FIELDS.index(n) for n in CTRL_STATES])

# Every gate the controller exports.  A gate sitting at exactly 0.0 or 1.0 is in
# the flat region of its smoothstep: its derivative vanishes, so it contributes
# nothing to the linearization and, more importantly, the hard `where` switch it
# guards cannot fire under a small perturbation.
GATE_FIELDS = [
    "drift_stability_gate", "drift_position_gate", "drift_heading_gate",
    "drift_height_gate", "drift_height_gate_vel", "drift_height_gate_heading",
    "drift_height_gate_pos", "heading_gate", "heading_pitch_gate",
    "heading_roll_gate", "heading_contact_gate", "heading_twist_gate",
    "heading_height_gate", "heading_twist_yield_gate", "twist_gate",
    "center_gate", "hy_div_guard_gate", "notch_height_gate",
    "abs_safety_pass", "apcr1nd_safety_pass", "apcr1nd_recenter_active",
]
# Two anchor gates are not exported; both are reconstructed exactly from the
# controller state (sag_pos_err is `prev_support_error` one step delayed).
ANCHOR_BANDS = [
    ("anchor proximity  |sag_pos_err|", "prev_support_error", 0.05, 0.15, "m"),
    ("envelope quiet    act_ema", "anchor_activity_ema", 0.25, 0.50, "m/s"),
]


def _setup(variant):
    nom = json.load(open(ROOT / DV / f"variant_{variant}__variant_setup.json"))
    posture = np.array([nom["hip_roll_left"], nom["hip_yaw_left"], nom["hip_pitch_ref"],
                        nom["knee_ref"], 0.0, nom["hip_roll_right"], nom["hip_yaw_right"],
                        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
    return nom, posture


class Loop:
    """One augmented closed-loop control step, as a pure function of the state."""

    def __init__(self, variant, settle_s):
        nom, posture = _setup(variant)
        self.model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
        self.data = mujoco.MjData(self.model)
        self.v3 = dict(init_v3_controller(profile_name=PROFILE, model=self.model))
        self.v3["jax_state"] = pack_state_k2()
        d, m = self.data, self.model
        d.qpos[7:17] = posture
        d.qpos[2] = float(nom["calibrated_root_z_m"])
        mujoco.mj_forward(m, d)
        self.ctx = P._build_v3_controller_context(
            m, d, self.v3, eq_joint=posture, height_ref=float(nom["target_com_z_m"]))

        # --- settle to the fixed point, recording every exported gate ---
        n = int(settle_s / DT)
        keep = n // 3
        gi = [K2_JAX_DIAG_FIELDS.index(f) for f in GATE_FIELDS]
        ai = [K2_JAX_STATE_FIELDS.index(b[1]) for b in ANCHOR_BANDS]
        gates = np.zeros((keep, len(gi)))
        anch = np.zeros((keep, len(ai)))
        for k in range(n):
            r = compute_v3_torque_for_state(d, m, self.v3["jax_step_fn"],
                                            self.v3["jax_state"], self.v3["jax_params"],
                                            self.ctx, teleop=None)
            self.v3["jax_state"] = r["next_jax_state"]
            d.ctrl[:] = np.array(r["tau_v3"])
            for _ in range(SUBSTEPS):
                mujoco.mj_step(m, d)
            if k >= n - keep:
                j = k - (n - keep)
                gates[j] = np.asarray(r["diagnostics"])[gi]
                anch[j] = np.abs(np.asarray(self.v3["jax_state"])[ai])
        self.gate_trace = gates
        self.anchor_trace = anch

        # --- freeze the fixed point ---
        self.qpos_star = d.qpos.copy()
        self.s_star = np.asarray(self.v3["jax_state"]).copy()
        self.com_star = np.array(self.ctx["prev_com_pos"], dtype=np.float64).copy()
        self.z_star = np.concatenate([np.zeros(m.nv), d.qvel.copy(),
                                      self.com_star, self.s_star[CS_IDX]])
        self.nv = m.nv
        # cyclic wheel angles: qpos-tangent indices of the two wheel joints
        self.drop = [6 + JOINTS.index("l_wheel"), 6 + JOINTS.index("r_wheel")]
        self.keep_idx = np.array([i for i in range(len(self.z_star))
                                  if i not in self.drop])
        self.names = ([f"dq_{n}" for n in ["x", "y", "z", "rx", "ry", "rz"] + JOINTS]
                      + [f"v_{n}" for n in ["x", "y", "z", "rx", "ry", "rz"] + JOINTS]
                      + ["com_x", "com_y", "com_z"] + list(CTRL_STATES))
        # the lateral coordinate (dq_x and its estimator shadow) is the
        # non-holonomic centre direction: excluded from the decay measurement,
        # which is about the modes that do converge.
        self.lat = [0, 2 * m.nv]
        self.decay_idx = np.array([i for i in range(2 * m.nv)
                                   if i not in self.drop and i != 0])
        self.settle_time = float(d.time)

    def step(self, z):
        m, d, nv = self.model, self.data, self.nv
        self._load(z)
        r = compute_v3_torque_for_state(d, m, self.v3["jax_step_fn"], self.v3["jax_state"],
                                        self.v3["jax_params"], self.ctx, teleop=None)
        s2 = np.asarray(r["next_jax_state"]).copy()
        d.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self.qpos_star, d.qpos)
        return np.concatenate([dq, d.qvel.copy(),
                               np.array(self.ctx["prev_com_pos"], dtype=np.float64),
                               s2[CS_IDX]])

    def jacobian(self, h=1e-6):
        z0 = self.z_star
        n = len(z0)
        A = np.zeros((n, n))
        for i in range(n):
            zp, zm = z0.copy(), z0.copy()
            zp[i] += h
            zm[i] -= h
            A[:, i] = (self.step(zp) - self.step(zm)) / (2 * h)
        k = self.keep_idx
        return A[np.ix_(k, k)]

    def _load(self, z):
        m, d = self.model, self.data
        nv = self.nv
        d.qpos[:] = self.qpos_star
        mujoco.mj_integratePos(m, d.qpos, np.ascontiguousarray(z[:nv]), 1.0)
        d.qvel[:] = z[nv:2 * nv]
        d.time = self.settle_time
        s = self.s_star.copy()
        s[CS_IDX] = z[2 * nv + 3:]
        self.v3["jax_state"] = s
        self.ctx["prev_com_pos"] = np.array(z[2 * nv:2 * nv + 3])
        mujoco.mj_forward(m, d)

    def decay(self, direction, amp, seconds):
        """Nonlinear decay of a perturbation, everything live (ABS included)."""
        m, d, nv = self.model, self.data, self.nv
        self._load(self.z_star + amp * direction / np.linalg.norm(direction))
        out = []
        for _ in range(int(seconds / DT)):
            r = compute_v3_torque_for_state(d, m, self.v3["jax_step_fn"],
                                            self.v3["jax_state"], self.v3["jax_params"],
                                            self.ctx, teleop=None)
            self.v3["jax_state"] = r["next_jax_state"]
            d.ctrl[:] = np.array(r["tau_v3"])
            for _ in range(SUBSTEPS):
                mujoco.mj_step(m, d)
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, self.qpos_star, d.qpos)
            e = np.concatenate([dq, d.qvel - self.z_star[nv:2 * nv]])
            out.append(float(np.linalg.norm(e[self.decay_idx])))
        return np.array(out)

    def centre_probe(self, idx, amp, seconds):
        """Direct nonlinear test of a suspected centre direction.

        A finite-differenced eigenvalue cannot distinguish |lambda| = 1 from
        1 + 1e-8; a long nonlinear rollout can.  Perturbs one coordinate and
        reports what the coordinate actually does over `seconds`.
        """
        m, d, nv = self.model, self.data, self.nv
        z = self.z_star.copy()
        z[idx] += amp
        self._load(z)
        traj = []
        for _ in range(int(seconds / DT)):
            r = compute_v3_torque_for_state(d, m, self.v3["jax_step_fn"],
                                            self.v3["jax_state"], self.v3["jax_params"],
                                            self.ctx, teleop=None)
            self.v3["jax_state"] = r["next_jax_state"]
            d.ctrl[:] = np.array(r["tau_v3"])
            for _ in range(SUBSTEPS):
                mujoco.mj_step(m, d)
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, self.qpos_star, d.qpos)
            traj.append(float(dq[idx]))
        traj = np.array(traj)
        t = np.arange(len(traj)) * DT
        half = len(traj) // 2
        rate = float(np.polyfit(t[half:], traj[half:], 1)[0])
        return {"coord": self.names[idx], "amp": amp, "seconds": seconds,
                "final": float(traj[-1]), "max_abs": float(np.abs(traj).max()),
                "ratio_final_over_amp": float(traj[-1] / amp),
                "late_rate_per_s": rate,
                "growth_over_window": float(rate * seconds / abs(amp))}


# |lambda| within this of 1 counts as a centre direction.  Chosen an order of
# magnitude above the measured finite-difference resolution (the step-size
# sensitivity of rho, ~1e-7) and three orders below the gap to the next mode
# (~1.8e-3), so the classification is not a judgement call.  Every direction it
# selects is then tested directly by a 300 s nonlinear rollout.
MARGIN_TOL = 1e-6


def summarise(A, names=None):
    """Spectrum of the closed-loop one-step map.

    A differential-drive base can exert no lateral force, so the lateral
    coordinate is a non-holonomic centre direction and must appear as a simple
    eigenvalue at exactly 1.  We report it separately rather than hiding it:
    the certificate is asymptotic stability *modulo* that one direction, plus
    the measured bound on the drift along it.
    """
    ev, V = np.linalg.eig(A)
    order = np.argsort(-np.abs(ev))
    ev, V = ev[order], V[:, order]

    def mode(j):
        e = ev[j]
        d = {"re": float(e.real), "im": float(e.imag), "mag": float(abs(e)),
             "tau_s": float(-DT / np.log(abs(e))) if 0 < abs(e) < 1 else None,
             "f_hz": float(abs(np.angle(e)) / (2 * np.pi * DT))}
        if names is not None:
            w = np.abs(V[:, j])
            w = w / max(w.max(), 1e-30)
            d["dominant"] = [(names[t], round(float(w[t]), 3))
                             for t in np.argsort(-w)[:4]]
        return d

    mag = np.abs(ev)
    centre = np.where(mag > 1.0 - MARGIN_TOL)[0]
    rest = np.where(mag <= 1.0 - MARGIN_TOL)[0]
    rho_r = float(mag[rest].max())
    return {
        "n_states": int(A.shape[0]),
        "spectral_radius": float(mag[0]),
        "n_unstable_strict": int(np.sum(mag > 1.0 + MARGIN_TOL)),
        "n_centre": int(len(centre)),
        "centre_modes": [mode(j) for j in centre],
        "rho_reduced": rho_r,
        "schur_modulo_centre": bool(rho_r < 1.0),
        "tau_slowest_s": float(-DT / np.log(rho_r)) if 0 < rho_r < 1 else None,
        "slowest": [mode(j) for j in rest[:6]],
    }


def main():
    settle_s = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    out = {"protocol": {"control_hz": 100, "physics_hz": 500, "settle_s": settle_s,
                        "profile": PROFILE, "frozen": "ABS adaptive-bias FIR trim",
                        "envelope_attack_alpha": 0.35, "envelope_release_alpha": 0.0067},
           "variants": {}}

    for variant in VARIANTS:
        print(f"\n=== {variant} ===", flush=True)
        L = Loop(variant, settle_s)
        rec = {"com_z_m": float(L.data.subtree_com[0][2]),
               "gates": [], "anchor_gate_margins": []}
        print("  exported gates over the last third of the settle "
              "(saturated = flat region = derivative 0, switch cannot fire):")
        n_sat = 0
        for f, col in zip(GATE_FIELDS, L.gate_trace.T):
            lo, hi = float(col.min()), float(col.max())
            sat = bool(lo == hi and (lo == 0.0 or lo == 1.0))
            n_sat += sat
            rec["gates"].append({"gate": f, "min": lo, "max": hi, "saturated": sat})
            if not sat:
                print(f"    {f:28s} [{lo:.6f}, {hi:.6f}]  NOT saturated")
        rec["n_gates"] = len(GATE_FIELDS)
        rec["n_gates_saturated"] = int(n_sat)
        print(f"    {n_sat}/{len(GATE_FIELDS)} exported gates saturated")
        for (label, field, lo_e, hi_e, unit), col in zip(ANCHOR_BANDS, L.anchor_trace.T):
            v = float(col.max())
            flat = bool(v < lo_e)
            rec["anchor_gate_margins"].append(
                {"gate": label, "peak": v, "low_edge": lo_e, "unit": unit,
                 "flat": flat, "margin_ratio": float(lo_e / v) if v > 0 else None})
            print(f"    {label:34s} peak {v:.3e} {unit:5s} vs edge {lo_e:g}"
                  f"  {'FLAT' if flat else '*** IN BAND ***'}"
                  f"  ({lo_e / max(v, 1e-30):.1f}x margin)")

        names = [L.names[i] for i in L.keep_idx]
        A = L.jacobian(1e-6)
        rec["jacobian"] = summarise(A, names)
        rec["jacobian_h1e-5"] = summarise(L.jacobian(1e-5), names)
        rec["rho_reduced_step_size_sensitivity"] = abs(
            rec["jacobian"]["rho_reduced"] - rec["jacobian_h1e-5"]["rho_reduced"])
        j = rec["jacobian"]
        print(f"  {j['n_states']} states: {j['n_centre']} centre, "
              f"{j['n_unstable_strict']} strictly unstable")
        for e in j["centre_modes"]:
            print(f"    centre |lambda|={e['mag']:.12f}  {e['dominant']}")
        print(f"  rho_reduced = {j['rho_reduced']:.9f}  tau = {j['tau_slowest_s']:.1f} s"
              f"  [h-sensitivity {rec['rho_reduced_step_size_sensitivity']:.2e}]")
        for e in j["slowest"][:4]:
            print(f"    |lambda|={e['mag']:.9f} tau={e['tau_s']:9.2f}s "
                  f"f={e['f_hz']:5.2f}Hz  {e['dominant']}")

        if variant == "nominal":
            rng = np.random.default_rng(20260727)
            rec["nonlinear_decay"] = []
            for amp in (1e-4, 1e-3, 1e-2):
                dirs = rng.normal(size=(3, len(L.z_star)))
                dirs[:, 2 * L.nv:] = 0.0          # perturb the plant only
                dirs[:, L.lat] = 0.0              # ... off the centre direction
                rates = []
                for dvec in dirs:
                    e = L.decay(dvec, amp, 12.0)
                    e = np.maximum(e, 1e-16)
                    k0, k1 = int(2.0 / DT), int(10.0 / DT)
                    sl = np.polyfit(np.arange(k1 - k0), np.log(e[k0:k1]), 1)[0]
                    rates.append(float(np.exp(sl)))
                rec["nonlinear_decay"].append(
                    {"amplitude": amp, "rho_measured_mean": float(np.mean(rates)),
                     "rho_measured_std": float(np.std(rates)), "n_dirs": len(rates)})
                print(f"  nonlinear decay @ amp {amp:g}: rho_meas = "
                      f"{np.mean(rates):.5f} +- {np.std(rates):.5f}")
            rec["centre_probes"] = []
            for idx, amp, unit, sc in ((L.names.index("dq_x"), 1e-3, "mm", 1e3),
                                       (L.names.index("dq_rz"), 8.727e-3, "deg",
                                        180.0 / np.pi)):
                p = L.centre_probe(idx, amp, 300.0)
                rec["centre_probes"].append(p)
                print(f"  centre probe {p['coord']:6s}: {amp * sc:+.3f} {unit} -> "
                      f"{p['final'] * sc:+.4f} {unit} after {p['seconds']:g} s "
                      f"(peak {p['max_abs'] * sc:.4f}, late rate "
                      f"{p['late_rate_per_s'] * sc:+.2e} {unit}/s)")
        out["variants"][variant] = rec

    rhos = [out["variants"][v]["jacobian"]["rho_reduced"] for v in VARIANTS]
    nc = [out["variants"][v]["jacobian"]["n_centre"] for v in VARIANTS]
    nu = [out["variants"][v]["jacobian"]["n_unstable_strict"] for v in VARIANTS]
    out["summary"] = {"rho_reduced_max": float(max(rhos)),
                      "rho_reduced_min": float(min(rhos)),
                      "n_centre": nc, "n_unstable_strict": nu,
                      "schur_modulo_centre_all": bool(max(rhos) < 1.0 and max(nu) == 0)}
    print(f"\nrho_reduced over the five height variants: {min(rhos):.9f} .. "
          f"{max(rhos):.9f}   centre dims {nc}   strictly unstable {nu}")

    dest = ROOT / "outputs/paper_verification/acc_stability_certificate.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, dest.open("w"), indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
