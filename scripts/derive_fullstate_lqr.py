#!/usr/bin/env python3
"""Full-state LQR baseline on the true 32-state MuJoCo linearization.

The paper's classical baselines all derive from reduced 4- and 6-state models.
This script removes that restriction: it linearizes the complete 10-actuator,
16-DOF plant about a solved standing equilibrium, finite-differences the
complete 100 Hz control step to get the transition Jacobians, solves the
discrete-time Riccati equation, and evaluates the resulting full-state gain in
closed loop under the same protocol as every other baseline.

Weight selection is a sweep, not a single guess: the R multiplier is swept and
the best-performing gain is reported, so the baseline is tuned rather than
assumed.

Usage:
  .venv/bin/python scripts/derive_fullstate_lqr.py --episodes 10
  .venv/bin/python scripts/derive_fullstate_lqr.py --episodes 20 \
      --output outputs/fullstate_lqr/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "assets" / "robot" / "wheeled_biped_real.xml"

CONTROL_DT = 0.01          # 100 Hz control, matching ACC
SUBSTEPS = 5               # 500 Hz physics
EPISODE_S = 20.0
MIN_HEIGHT = 0.3           # eval_balance.py::_is_fallen
MAX_TILT_RAD = 0.8
JOINT_POS_STD = 0.005      # paper's initial-posture draw
ROOT_POS_STD = 0.001

WHEEL_DOFS = (10, 15)      # l_wheel, r_wheel velocity indices


# ---------------------------------------------------------------------------
# Equilibrium and linearization
# ---------------------------------------------------------------------------
def _qacc_at(
    m: mujoco.MjModel, d: mujoco.MjData, qpos: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Generalized acceleration at rest for pose ``qpos`` and input ``u``."""
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = u
    mujoco.mj_forward(m, d)
    return d.qacc.copy()


def standing_equilibrium(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve for a true static equilibrium (x*, u*) near the standing keyframe.

    The keyframe pose itself is not an equilibrium: it is authored with the
    wheels several millimetres inside the floor, so the contact springs return
    an order of magnitude more force than body weight and the plant is in a
    violent transient at t=0. Inverse dynamics at ``qacc = 0`` therefore reports
    the torque needed to fight that penetration, not the torque needed to stand,
    and forward integration is no help either because the pose is an unstable
    equilibrium that topples.

    The equilibrium is instead solved for directly: damped Gauss-Newton drives
    ``qacc`` to zero with a finite-difference Jacobian. Ten torques against
    sixteen accelerations is underdetermined, so the pose is a free variable too
    -- base height, base roll and pitch, and the eight leg angles. Base height in
    particular must be free because the resting penetration, the depth at which
    the contact springs carry exactly body weight, is part of the equilibrium
    rather than an input to it. Only the cyclic coordinates are pinned: base x,
    y, yaw and the two wheel angles, none of which has a restoring force and so
    none of which has a preferred value. A weak Tikhonov pull keeps the solution
    near the commanded standing pose rather than letting it wander to some other
    equilibrium of the plant.

    Returns (qpos_star, ctrl_star, residual), residual being the norm of the
    unbalanced generalized force at (x*, u*).
    """
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    qpos_ref = d.qpos.copy()
    qpos = qpos_ref.copy()

    # Free pose directions: everything but the cyclic coordinates.
    free = [i for i in range(m.nv)
            if i not in (0, 1, 5, *WHEEL_DOFS)]
    n_pose, nu = len(free), m.nu
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    u = np.zeros(nu)
    eps_u, eps_q = 1e-4, 1e-6
    n_iter = 600

    def perturbed(base: np.ndarray, idx: int, eps: float) -> np.ndarray:
        dq = np.zeros(m.nv)
        dq[idx] = eps
        out = base.copy()
        mujoco.mj_integratePos(m, out, dq, 1.0)
        return out

    w_pose, w_u = 30.0, 0.05      # Tikhonov weights on pose drift and torque
    for _ in range(n_iter):
        r0 = _qacc_at(m, d, qpos, u)
        jac = np.zeros((m.nv, nu + n_pose))
        for i in range(nu):
            up = u.copy()
            up[i] += eps_u
            jac[:, i] = (_qacc_at(m, d, qpos, up) - r0) / eps_u
        for k, i in enumerate(free):
            jac[:, nu + k] = (
                _qacc_at(m, d, perturbed(qpos, i, eps_q), u) - r0) / eps_q

        # Regularized least squares: ||J s + r0||^2 + w_u||u+s_u||^2
        #                                          + w_pose||dpose + s_pose||^2
        dpose = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dpose, 1.0, qpos_ref, qpos)
        reg = np.zeros((nu + n_pose, nu + n_pose))
        reg[:nu, :nu] = w_u * np.eye(nu)
        reg[nu:, nu:] = w_pose * np.eye(n_pose)
        rhs = np.concatenate([-r0, -w_u * u, -w_pose * dpose[free]])
        step, *_ = np.linalg.lstsq(np.vstack([jac, reg]), rhs, rcond=None)

        u_next = np.clip(u + 0.3 * step[:nu], lo, hi)
        dq = np.zeros(m.nv)
        dq[free] = 0.3 * step[nu:]
        qpos_next = qpos.copy()
        mujoco.mj_integratePos(m, qpos_next, dq, 1.0)
        if (np.linalg.norm(u_next - u) < 1e-12
                and np.linalg.norm(qpos_next - qpos) < 1e-12):
            break
        u, qpos = u_next, qpos_next

    # Report the residual as an unbalanced generalized force, not as ||qacc||:
    # the light hip DOFs (armature 0.02) turn a negligible torque into a large
    # acceleration, so ||qacc|| overstates how far from equilibrium the pose is.
    qacc = _qacc_at(m, d, qpos, u)
    force = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, force, qacc)
    return qpos, u, float(np.linalg.norm(force))


def _control_step(
    m: mujoco.MjModel, d: mujoco.MjData, qpos_star: np.ndarray,
    dx: np.ndarray, u: np.ndarray,
) -> np.ndarray:
    """Advance one held-input 100 Hz control step from x* + dx; return the new dx."""
    d.qpos[:] = qpos_star
    mujoco.mj_integratePos(m, d.qpos, dx[:m.nv], 1.0)
    d.qvel[:] = dx[m.nv:]
    d.ctrl[:] = u
    mujoco.mj_forward(m, d)
    for _ in range(SUBSTEPS):
        mujoco.mj_step(m, d)
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, qpos_star, d.qpos)
    return np.concatenate([dq, d.qvel])


def control_rate_jacobians(
    m: mujoco.MjModel, qpos_star: np.ndarray, ctrl_star: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A, B for one 100 Hz control step, by central differences on the step itself.

    Differencing the whole five-substep step is not merely a convenience over
    composing the per-substep Jacobian as ``A^5``. The wheel--ground contact is a
    ~10^6 N/m spring, which the 2 ms linearization sees as a mode with
    |lambda| ~ 50; raising that to the fifth power amplifies it by ~10^8 and
    swamps every rigid-body mode. Differencing the composite map lets the
    contact solver resolve its own stiff mode internally, so the Jacobian
    describes the dynamics the controller actually sees at 100 Hz.
    """
    d = mujoco.MjData(m)
    nx = 2 * m.nv
    eps_x, eps_u = 1e-5, 1e-4

    A = np.zeros((nx, nx))
    for i in range(nx):
        dp, dm = np.zeros(nx), np.zeros(nx)
        dp[i], dm[i] = eps_x, -eps_x
        A[:, i] = (_control_step(m, d, qpos_star, dp, ctrl_star)
                   - _control_step(m, d, qpos_star, dm, ctrl_star)) / (2 * eps_x)

    B = np.zeros((nx, m.nu))
    zero = np.zeros(nx)
    for j in range(m.nu):
        up, um = ctrl_star.copy(), ctrl_star.copy()
        up[j] += eps_u
        um[j] -= eps_u
        B[:, j] = (_control_step(m, d, qpos_star, zero, up)
                   - _control_step(m, d, qpos_star, zero, um)) / (2 * eps_u)
    return A, B


def bryson_weights(m: mujoco.MjModel, r_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Q, R by Bryson's rule from physically meaningful maximum deviations.

    Wheel *angles* carry zero weight: they are cyclic coordinates tied to base
    translation by the rolling contact, so penalizing them fights the very
    motion the wheels use to balance.
    """
    nv = m.nv
    pos_max = np.empty(nv)
    pos_max[0:2] = 0.05      # base x, y  [m]
    pos_max[2] = 0.02        # base z     [m]
    pos_max[3:5] = 0.10      # roll, pitch [rad]
    pos_max[5] = 0.20        # yaw        [rad]
    pos_max[6:] = 0.20       # joints     [rad]
    vel_max = np.empty(nv)
    vel_max[0:3] = 0.50
    vel_max[3:6] = 1.00
    vel_max[6:] = 2.00

    q_diag = np.concatenate([1.0 / pos_max**2, 1.0 / vel_max**2])
    for w in WHEEL_DOFS:
        q_diag[w] = 0.0      # cyclic wheel angle: unweighted

    u_max = np.minimum(np.abs(m.actuator_ctrlrange[:, 1]), 30.0)
    return np.diag(q_diag), np.diag(r_scale / u_max**2)


def solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


# ---------------------------------------------------------------------------
# Closed-loop evaluation
# ---------------------------------------------------------------------------
def state_error(m: mujoco.MjModel, d: mujoco.MjData, qpos_star: np.ndarray) -> np.ndarray:
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, qpos_star, d.qpos)
    dq[list(WHEEL_DOFS)] = 0.0      # cyclic: never drive the wheel back
    return np.concatenate([dq, d.qvel])


def rollout(
    m: mujoco.MjModel, K: np.ndarray, qpos_star: np.ndarray,
    ctrl_star: np.ndarray, seed: int,
) -> dict:
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    d.qpos[:] = qpos_star
    d.qpos[7:] += rng.normal(0.0, JOINT_POS_STD, size=m.nu)
    d.qpos[0:3] += rng.normal(0.0, ROOT_POS_STD, size=3)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    n_ctrl = int(EPISODE_S / CONTROL_DT)
    pitch_sq, n_acc = 0.0, 0

    for step in range(n_ctrl):
        u = np.clip(ctrl_star - K @ state_error(m, d, qpos_star), lo, hi)
        d.ctrl[:] = u
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)

        quat = d.qpos[3:7]
        R_wb = np.zeros(9)
        mujoco.mju_quat2Mat(R_wb, quat)
        tilt = np.arccos(np.clip(R_wb[8], -1.0, 1.0))
        pitch_sq += tilt**2
        n_acc += 1

        if d.qpos[2] < MIN_HEIGHT or tilt > MAX_TILT_RAD or not np.isfinite(d.qpos).all():
            return {
                "survived": False,
                "survival_time_s": (step + 1) * CONTROL_DT,
                "tilt_rms_deg": float(np.degrees(np.sqrt(pitch_sq / n_acc))),
            }

    return {
        "survived": True,
        "survival_time_s": EPISODE_S,
        "tilt_rms_deg": float(np.degrees(np.sqrt(pitch_sq / n_acc))),
    }


def evaluate(m, K, qpos_star, ctrl_star, episodes: int) -> dict:
    eps = [rollout(m, K, qpos_star, ctrl_star, seed=1000 + i) for i in range(episodes)]
    surv = np.array([e["survival_time_s"] for e in eps])
    return {
        "episodes": episodes,
        "fall_rate": float(sum(not e["survived"] for e in eps) / episodes),
        "survival_time_mean_s": float(surv.mean()),
        "survival_time_std_s": float(surv.std(ddof=1)) if episodes > 1 else 0.0,
        "tilt_rms_deg_mean": float(np.mean([e["tilt_rms_deg"] for e in eps])),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--r-scales", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs" / "fullstate_lqr" / "results.json")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(XML_PATH))
    qpos_star, ctrl_star, drift_norm = standing_equilibrium(m)
    print(f"equilibrium: base z = {qpos_star[2]:.6f} m, "
          f"|u*| = {np.linalg.norm(ctrl_star):.3f} Nm, "
          f"residual force = {drift_norm:.4e} N")

    A, B = control_rate_jacobians(m, qpos_star, ctrl_star)
    kappa = float(np.linalg.cond(A))
    rank = int(np.linalg.matrix_rank(A))
    print(f"control-rate linearization: A {A.shape}, kappa(A) = {kappa:.3e}, "
          f"rank = {rank}/{A.shape[0]}")

    # Stabilizability: PBH on every unstable/marginal mode.
    evals = np.linalg.eigvals(A)
    unstable = evals[np.abs(evals) >= 1.0 - 1e-8]
    pbh_ok = all(
        np.linalg.matrix_rank(np.hstack([A - lam * np.eye(A.shape[0]), B]), tol=1e-7)
        == A.shape[0]
        for lam in unstable
    )
    print(f"{len(unstable)} modes with |lambda| >= 1; "
          f"stabilizable = {pbh_ok}; max|lambda| = {np.abs(evals).max():.6f}")

    results = {
        "equilibrium": {
            "base_z_m": float(qpos_star[2]),
            "holding_torque_norm_Nm": float(np.linalg.norm(ctrl_star)),
            "holding_torque_Nm": ctrl_star.tolist(),
            "residual_force_norm_N": drift_norm,
        },
        "linearization": {
            "state_dim": int(A.shape[0]),
            "input_dim": int(B.shape[1]),
            "control_dt_s": CONTROL_DT,
            "condition_number": kappa,
            "rank": rank,
            "n_marginal_or_unstable_modes": int(len(unstable)),
            "max_abs_eigenvalue": float(np.abs(evals).max()),
            "stabilizable_pbh": bool(pbh_ok),
        },
        "sweep": [],
    }

    best = None
    for rs in args.r_scales:
        Q, R = bryson_weights(m, rs)
        try:
            K = solve_lqr(A, B, Q, R)
        except Exception as exc:  # noqa: BLE001
            print(f"  R x{rs:<8g}  DARE FAILED: {type(exc).__name__}")
            results["sweep"].append({"r_scale": rs, "error": str(exc)})
            continue

        cl = float(np.abs(np.linalg.eigvals(A - B @ K)).max())
        metrics = evaluate(m, K, qpos_star, ctrl_star, args.episodes)
        row = {"r_scale": rs, "max_abs_closed_loop_eig": cl,
               "max_abs_gain": float(np.abs(K).max()), **metrics}
        results["sweep"].append(row)
        print(f"  R x{rs:<8g} |eig_cl|={cl:.5f}  max|K|={row['max_abs_gain']:8.1f}  "
              f"falls={metrics['fall_rate']*100:5.1f}%  "
              f"surv={metrics['survival_time_mean_s']:6.3f}s")

        key = (-metrics["fall_rate"], metrics["survival_time_mean_s"])
        if best is None or key > best[0]:
            best = (key, row)

    if best is not None:
        results["best"] = best[1]
        print(f"\nbest: R x{best[1]['r_scale']}  "
              f"falls {best[1]['fall_rate']*100:.1f}%  "
              f"survival {best[1]['survival_time_mean_s']:.3f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
