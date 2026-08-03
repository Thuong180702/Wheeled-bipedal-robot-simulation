"""
Full-state LQR balance baseline — linearization of the complete MuJoCo plant.

PURPOSE
-------
Every other classical baseline in this repository is designed on a reduced
model: a 4-state TWIP, or a 6-state coupled sagittal/roll model.  This one is
designed on the plant itself.  It linearizes the full 16-DOF, 10-actuator
MuJoCo model about a solved standing equilibrium and solves the discrete-time
Riccati equation on the result, so the comparison against the proposed
controller is not bounded by the fidelity of a hand-written model.

DERIVATION (offline, per commanded height, cached)
--------------------------------------------------
1. ``standing_equilibrium`` solves for a true static equilibrium (x*, u*) at
   the commanded height.  The authored keyframe is *not* one — see that
   function's docstring.
2. ``control_rate_jacobians`` finite-differences the whole held-input control
   step to get the transition Jacobians A (32x32) and B (32x10) at the rate
   the controller actually runs.
3. The two wheel-*angle* states are removed.  They are cyclic coordinates:
   nothing in the dynamics depends on absolute wheel angle, and it is not
   observable from the robot's observation vector either.  ``reduce_system``
   asserts the first of those numerically.
4. Bryson's rule sets Q and R from physically meaningful maximum deviations;
   ``solve_discrete_are`` gives the gain K (10x30).

ONLINE (50 Hz)
--------------
The 30-dim state error is reconstructed from the 42-dim BalanceEnv
observation, and the control is the LQR law about the equilibrium::

    tau = clip(u* - K @ dx, ctrl_min, ctrl_max)

emitted as a **direct torque** action, rate-limited to the same 400 Nm/s as
ACC and the direct-torque coupled baseline.  This is the natural action space
for a torque-designed LQR; routing it through a position/velocity PID servo,
as an earlier revision of this file did, discards the design.

MEASURED OUTCOME — read before citing this as a baseline
--------------------------------------------------------
The design is numerically sound at every commanded height in [0.40, 0.70] m:
the equilibrium residual is 0.43 N of the 79.46 N body weight, the reduced
30-state A is full rank and PBH-stabilizable, the Riccati relative residual is
~1e-14, and the obs->state reconstruction is second-order accurate.  cond(A)
is 2.6e7 at 100 Hz and 6.5e9 at 50 Hz; the large 50 Hz figure comes from a
~1e-9 *smallest* singular value, i.e. from strongly contracting contact modes,
and obstructs nothing.  Earlier drafts blamed the gains on that number; that
was wrong.

What it actually does: at 100 Hz with a high control weight it stands and
tracks height well for tens of seconds -- r_scale=1000 at 0.65 m gives 0%
falls over a 20 s horizon with 11.0 mm height RMSE and 0.12 deg pitch RMS --
but it is not stabilizing.  Extending the horizon to 60 s falls 100% in every
configuration tried (mean time-to-fall 22.2-57.0 s), with planar drift growing
monotonically to 0.43-0.63 m.  The 20 s window truncates a slow divergence.
No single weighting holds the band either: r_scale=100 stands 20 s at 0.60 and
0.69 m but falls at 0.65 m, and r_scale=1000 does the reverse; random height
in [0.40, 0.70] falls 45-80%.  Push recovery is where it separates from ACC
outright -- 100% falls under the standard 50 N impulse, max recoverable push
10-20 N against ACC's F_min = 83 N.

So: cite it as a classical baseline that survives ~100x longer than the
reduced-order ones (0.16-0.60 s -> 22-57 s) and still falls, not as one that
could not be derived.  Evidence: outputs/fullstate_lqr_corrected/.

Two defects had to be fixed to get here, both of which flatter the controller
if left in.  (a) The shared ``_build_height_ik`` polynomial saturates -- 0.60,
0.65 and 0.70 m all return near-straight legs standing at ~0.73 m -- so the
design stood at a height it was never commanded and its survival came from not
tracking height at all; ``standing_pose`` replaces it.  (b) eval_balance
rebound ``_N_PHYSICS_SUBSTEPS``, a name this module does not define, so
``--control-hz`` left the linearization at 50 Hz whatever it said.

Things that were tried and changed nothing: tightening the contact solver
(iterations 4->200, tol 1e-8->1e-14, bit-identical Jacobians), sweeping the
finite-difference step over three decades, fitting A and B by least squares
over the operating region instead of pointwise, and removing the remaining
cyclic base-x/y states.  Contact-set flicker is not the mechanism: ncon is
constant across all FD probes at each equilibrium.

Self-check::

    .venv/bin/python -m wheeled_biped.controllers.full_lqr
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IDX = {
    "l_hip_roll": 0, "l_hip_yaw": 1, "l_hip_pitch": 2, "l_knee": 3, "l_wheel": 4,
    "r_hip_roll": 5, "r_hip_yaw": 6, "r_hip_pitch": 7, "r_knee": 8, "r_wheel": 9,
}

_CONTROL_DT = 0.02          # 50 Hz control, matching BalanceEnv
_SUBSTEPS = 10              # 0.02 / 0.002 = 10 physics steps per control step
_MAX_TORQUE_RATE = 400.0    # Nm/s, same limit as ACC and the torque baseline
_MIN_H, _MAX_H = 0.40, 0.70

# Velocity-space indices of the two wheel DOFs.  Their *position* counterparts
# are the cyclic states removed before the Riccati solve.
WHEEL_DOFS = (10, 15)

# 42-dim BalanceEnv observation layout (lin_vel_mode = clean/noisy).
_OBS_GRAVITY = slice(0, 3)
_OBS_LIN_VEL = slice(3, 6)
_OBS_ANG_VEL = slice(6, 9)
_OBS_JOINT_POS = slice(9, 19)
_OBS_JOINT_VEL = slice(19, 29)
_OBS_CUR_HEIGHT = 40        # normalized to [0, 1] over [_MIN_H, _MAX_H]
_OBS_YAW_ERROR = 41


# ---------------------------------------------------------------------------
# Reference pose
# ---------------------------------------------------------------------------

def standing_pose(
    m: mujoco.MjModel, height_m: float, tol: float = 1e-6,
) -> np.ndarray:
    """Symmetric standing pose whose wheels rest on the floor at ``height_m``.

    The legs form a parallel link (knee = 2 x hip returns the shank to
    vertical), so torso height is a single monotone function of the hip angle.
    That function is inverted here by bisection against the real wheel-contact
    geometry.

    The repo's shared ``_build_height_ik`` polynomial is deliberately not used.
    It is fit over a narrow scan range (0.520-0.703 m) and saturates inside it:
    asking for 0.60, 0.65 or 0.70 m all return nearly straight legs standing at
    0.72-0.73 m, and below the scan range it extrapolates to hip angles of
    3.8 rad, twice the joint limit.  Feeding that to the equilibrium solver
    hands it the wrong pose, which it then correctly balances at the wrong
    height.  Fixing the shared polynomial would move the six published
    reduced-order baseline results, so the correction is kept local.
    """
    gids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("l_wheel_collision", "r_wheel_collision")
    ]
    radius = float(m.geom_size[gids[0], 0])
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing")

    def posed(hip: float) -> np.ndarray:
        mujoco.mj_resetDataKeyframe(m, d, key)
        for side in ("l", "r"):
            d.qpos[7 + _IDX[f"{side}_hip_pitch"]] = hip
            d.qpos[7 + _IDX[f"{side}_knee"]] = 2.0 * hip
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        # Lower the base until the lowest wheel just touches z = 0.
        d.qpos[2] = 1.0 - (min(d.geom_xpos[g][2] for g in gids) - radius)
        return d.qpos.copy()

    # Torso height decreases monotonically in hip over this span (0.73 -> 0.17 m).
    lo, hi = 0.0, 1.6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if posed(mid)[2] > height_m:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return posed(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Equilibrium
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


def standing_equilibrium(
    m: mujoco.MjModel, qpos_ref: np.ndarray, n_iter: int = 600,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve for a true static equilibrium (x*, u*) near the reference pose.

    The authored ``standing`` keyframe is not an equilibrium: it places the
    wheels several millimetres inside the floor, so the contact springs return
    an order of magnitude more force than body weight and the plant is in a
    violent transient at t=0.  Inverse dynamics at ``qacc = 0`` therefore
    reports the torque needed to fight that penetration rather than the torque
    needed to stand, and forward integration is no help either, because the
    pose is an unstable equilibrium that topples during any settle.  Both of
    those shortcuts were tried and both produce a linearization of a falling
    robot.

    The equilibrium is solved for directly instead: damped Gauss-Newton drives
    ``qacc`` to zero with a finite-difference Jacobian.  Ten torques against
    sixteen accelerations is underdetermined, so the pose is a free variable
    too -- base height, base roll and pitch, and the eight leg angles.  Base
    height in particular must be free, because the resting penetration (the
    depth at which the contact springs carry exactly body weight) is part of
    the equilibrium rather than an input to it.  Only the cyclic coordinates
    are pinned: base x, y, yaw and the two wheel angles, none of which has a
    restoring force and so none of which has a preferred value.  A weak
    Tikhonov pull keeps the solution near ``qpos_ref`` rather than letting it
    wander to some other equilibrium of the plant.

    Returns (qpos_star, ctrl_star, residual), the residual being the norm of
    the unbalanced generalized force at (x*, u*) in newtons.
    """
    d = mujoco.MjData(m)
    qpos = np.asarray(qpos_ref, dtype=np.float64).copy()

    free = [i for i in range(m.nv) if i not in (0, 1, 5, *WHEEL_DOFS)]
    n_pose, nu = len(free), m.nu
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    u = np.zeros(nu)
    eps_u, eps_q = 1e-4, 1e-6
    w_pose, w_u = 30.0, 0.05      # Tikhonov weights on pose drift and torque

    def perturbed(base: np.ndarray, idx: int, eps: float) -> np.ndarray:
        dq = np.zeros(m.nv)
        dq[idx] = eps
        out = base.copy()
        mujoco.mj_integratePos(m, out, dq, 1.0)
        return out

    def unbalanced_force(qpos_: np.ndarray, u_: np.ndarray) -> float:
        # Reported as a generalized force, not as ||qacc||: the light hip DOFs
        # (armature 0.02) turn a negligible torque into a large acceleration,
        # so ||qacc|| overstates how far from equilibrium the pose is.
        qacc = _qacc_at(m, d, qpos_, u_)
        force = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, force, qacc)
        return float(np.linalg.norm(force))

    # Warm start: sink the base until the contact springs carry body weight.
    # ``standing_pose`` puts the wheels at exact touch, where contact force is
    # zero and the ~1e6 N/m contact spring makes a Gauss-Newton step wildly
    # non-local -- a sub-millimetre move in base z swings the contact force by
    # hundreds of newtons, and the iteration limit-cycles between 80 N and 54 N
    # instead of converging.  Starting inside contact puts it in the smooth
    # region.  The sink depth is well under a millimetre, so the commanded
    # height is unaffected.
    z_hi, z_lo = qpos[2], qpos[2] - 0.02
    for _ in range(60):
        qpos[2] = 0.5 * (z_hi + z_lo)
        if _qacc_at(m, d, qpos, u)[2] > 0.0:
            z_lo = qpos[2]              # too deep, springs over-support
        else:
            z_hi = qpos[2]
    qpos[2] = 0.5 * (z_hi + z_lo)

    best = (unbalanced_force(qpos, u), qpos.copy(), u.copy())

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

        # Keep the best iterate rather than whatever the last one happens to
        # be: the tail of the iteration is not monotone.
        f_now = unbalanced_force(qpos, u)
        if f_now < best[0]:
            best = (f_now, qpos.copy(), u.copy())

    return best[1], best[2], best[0]


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

def _control_step(
    m: mujoco.MjModel, d: mujoco.MjData, qpos_star: np.ndarray,
    dx: np.ndarray, u: np.ndarray, substeps: int,
) -> np.ndarray:
    """Advance one held-input control step from x* + dx; return the new dx."""
    d.qpos[:] = qpos_star
    mujoco.mj_integratePos(m, d.qpos, dx[:m.nv], 1.0)
    d.qvel[:] = dx[m.nv:]
    d.ctrl[:] = u
    mujoco.mj_forward(m, d)
    for _ in range(substeps):
        mujoco.mj_step(m, d)
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, qpos_star, d.qpos)
    return np.concatenate([dq, d.qvel])


def control_rate_jacobians(
    m: mujoco.MjModel, qpos_star: np.ndarray, ctrl_star: np.ndarray,
    substeps: int = _SUBSTEPS, eps_x: float = 1e-5, eps_u: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """A, B for one control step, by central differences on the step itself.

    Differencing the whole multi-substep step is not merely a convenience over
    composing the per-substep Jacobian as ``A_sub ** substeps``.  The
    wheel--ground contact is a ~1e6 N/m spring, which the 2 ms linearization
    sees as a mode with |lambda| ~ 50; raising that to the tenth power
    amplifies it by ~1e17 and swamps every rigid-body mode, leaving a matrix
    that is numerically rank-deficient.  Differencing the composite map lets
    the contact solver resolve its own stiff mode internally, so the Jacobian
    describes the dynamics the controller actually sees at the control rate.
    """
    d = mujoco.MjData(m)
    nx = 2 * m.nv

    A = np.zeros((nx, nx))
    for i in range(nx):
        dp, dm = np.zeros(nx), np.zeros(nx)
        dp[i], dm[i] = eps_x, -eps_x
        A[:, i] = (_control_step(m, d, qpos_star, dp, ctrl_star, substeps)
                   - _control_step(m, d, qpos_star, dm, ctrl_star, substeps)
                   ) / (2 * eps_x)

    B = np.zeros((nx, m.nu))
    zero = np.zeros(nx)
    for j in range(m.nu):
        up, um = ctrl_star.copy(), ctrl_star.copy()
        up[j] += eps_u
        um[j] -= eps_u
        B[:, j] = (_control_step(m, d, qpos_star, zero, up, substeps)
                   - _control_step(m, d, qpos_star, zero, um, substeps)
                   ) / (2 * eps_u)
    return A, B


def kept_state_indices(nv: int) -> np.ndarray:
    """Indices of the 32-dim state that survive into the reduced design.

    The two wheel-angle *positions* are dropped; every velocity is kept.
    """
    pos = [i for i in range(nv) if i not in WHEEL_DOFS]
    return np.array(pos + list(range(nv, 2 * nv)))


def reduce_system(
    A: np.ndarray, B: np.ndarray, nv: int, tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the cyclic wheel-angle states from (A, B).

    Deleting a state is only exact if nothing else depends on it, i.e. if its
    columns in A are zero outside its own rows.  For an absolute wheel angle on
    a translation-invariant plant that is true by symmetry, but it is asserted
    here rather than assumed, because a modelling change (a wheel-angle-
    dependent spring, a non-cylindrical wheel) would silently break it.
    """
    keep = kept_state_indices(nv)
    drop = [i for i in range(2 * nv) if i not in set(keep.tolist())]
    coupling = np.abs(A[np.ix_(keep, drop)]).max()
    if coupling > tol:
        raise AssertionError(
            f"wheel angle is not cyclic: max |dA/d(wheel angle)| = {coupling:.3e} "
            f"> {tol:.0e}; the reduced design would be invalid"
        )
    return A[np.ix_(keep, keep)], B[keep, :]


def bryson_weights(
    m: mujoco.MjModel, r_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Q, R by Bryson's rule from physically meaningful maximum deviations.

    ``r_scale`` uniformly scales R, trading control effort against regulation;
    it is the single knob swept when tuning the baseline.
    """
    nv = m.nv
    pos_max = np.empty(nv)
    pos_max[0:2] = 0.05      # base x, y   [m]
    pos_max[2] = 0.02        # base z      [m]
    pos_max[3:5] = 0.10      # roll, pitch [rad]
    pos_max[5] = 0.20        # yaw         [rad]
    pos_max[6:] = 0.20       # joints      [rad]
    vel_max = np.empty(nv)
    vel_max[0:3] = 0.50
    vel_max[3:6] = 1.00
    vel_max[6:] = 2.00

    q_full = np.concatenate([1.0 / pos_max**2, 1.0 / vel_max**2])
    q_diag = q_full[kept_state_indices(nv)]

    u_max = np.minimum(np.abs(m.actuator_ctrlrange[:, 1]), 30.0)
    return np.diag(q_diag), np.diag(r_scale / u_max**2)


def solve_lqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Discrete-time LQR gain, with the Riccati residual verified.

    A carries strongly contracting contact modes whose singular values reach
    ~1e-9, so the intermediate products underflow and BLAS raises spurious
    overflow/divide warnings.  Those are noise; what matters is whether the
    returned P actually solves the equation, so the relative residual is
    checked directly instead of trusting the absence of warnings.
    """
    P = solve_discrete_are(A, B, Q, R)
    with np.errstate(all="ignore"):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        residual = A.T @ P @ A - P - A.T @ P @ B @ K + Q
        rel = float(np.abs(residual).max() / np.abs(P).max())
    if not np.isfinite(K).all() or rel > tol:
        raise AssertionError(
            f"Riccati solution is unusable: relative residual {rel:.3e} "
            f"(tolerance {tol:.0e}), K finite = {np.isfinite(K).all()}"
        )
    return K


def design(
    m: mujoco.MjModel, qpos_ref: np.ndarray, r_scale: float,
    substeps: int = _SUBSTEPS,
) -> dict[str, Any]:
    """Full offline pipeline: equilibrium -> Jacobians -> reduction -> gain."""
    qpos_star, ctrl_star, residual = standing_equilibrium(m, qpos_ref)
    A_full, B_full = control_rate_jacobians(m, qpos_star, ctrl_star, substeps)
    A, B = reduce_system(A_full, B_full, m.nv)
    Q, R = bryson_weights(m, r_scale)
    K = solve_lqr(A, B, Q, R)

    # A's smallest singular value is ~1e-9 (strongly contracting contact modes),
    # so these products underflow harmlessly.  solve_lqr already asserts the
    # Riccati residual, which is the check that matters.
    with np.errstate(all="ignore"):
        evals = np.linalg.eigvals(A)
        unstable = evals[np.abs(evals) >= 1.0 - 1e-8]
        pbh_ok = all(
            np.linalg.matrix_rank(
                np.hstack([A - lam * np.eye(A.shape[0]), B]), tol=1e-7,
            ) == A.shape[0]
            for lam in unstable
        )
        eig_closed = float(np.abs(np.linalg.eigvals(A - B @ K)).max())
    return {
        "qpos_star": qpos_star,
        "ctrl_star": ctrl_star,
        "K": K,
        "residual_force_N": residual,
        "base_z_m": float(qpos_star[2]),
        "state_dim": int(A.shape[0]),
        "cond_A": float(np.linalg.cond(A)),
        "rank_A": int(np.linalg.matrix_rank(A)),
        "max_abs_eig_open": float(np.abs(evals).max()),
        "n_marginal_or_unstable_modes": int(len(unstable)),
        "stabilizable_pbh": bool(pbh_ok),
        "max_abs_eig_closed": eig_closed,
        "max_abs_gain": float(np.abs(K).max()),
    }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class FullStateLQRController:
    """LQR balance baseline designed on the full MuJoCo linearization.

    Usage::

        ctrl = FullStateLQRController(model_path="assets/robot/....xml")
        ctrl.reset(height_cmd_m=0.65)
        action = ctrl.compute_action(obs)   # obs is 42-dim, action is 10-dim
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        q_config: dict | None = None,
        **_ignored: Any,
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}
        qc = q_config or {}

        self._r_scale = float(qc.get("r_scale", 1.0))
        self._substeps = int(qc.get("substeps", _SUBSTEPS))

        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._ctrl_min = self._model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_max = self._model.actuator_ctrlrange[:, 1].copy()

        pid_cfg = self._config.get("low_level_pid", {})
        self._max_torque_rate = float(
            pid_cfg.get("max_torque_rate", _MAX_TORQUE_RATE)
        )

        # Designs are cached per commanded height: the nominal scenario only
        # ever needs one, the fixed-height sweep needs seven.
        self._designs: dict[float, dict[str, Any]] = {}
        self._height_cmd_m = float(qc.get("nominal_height_m", 0.65))
        self._design = self._design_for(self._height_cmd_m)

        self._xy_drift = np.zeros(2)
        self._tau_prev = np.zeros(self._model.nu)

    # -- offline ------------------------------------------------------------

    def _reference_pose(self, height_m: float) -> np.ndarray:
        """Standing pose that actually stands at ``height_m``."""
        return standing_pose(self._model, float(np.clip(height_m, _MIN_H, _MAX_H)))

    def _design_for(self, height_m: float) -> dict[str, Any]:
        key = round(float(height_m), 4)
        if key not in self._designs:
            self._designs[key] = design(
                self._model, self._reference_pose(key), self._r_scale,
                self._substeps,
            )
        return self._designs[key]

    # -- online -------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._design = self._design_for(self._height_cmd_m)
        self._xy_drift[:] = 0.0
        self._tau_prev[:] = 0.0

    def _state_error(self, obs: np.ndarray) -> np.ndarray:
        """Reconstruct the 30-dim state error x - x* from the observation.

        The mapping mirrors ``mj_differentiatePos`` on the free joint: the
        orientation error reads out of body-frame gravity as (-g_y, g_x) with
        yaw supplied directly by the observation, and base xy drift, which no
        sensor reports, is integrated from body-frame linear velocity.  It is
        verified against ``mj_differentiatePos`` in this module's self-check.
        """
        qpos_star = self._design["qpos_star"]
        grav = obs[_OBS_GRAVITY]
        lin_vel = obs[_OBS_LIN_VEL]
        height = _MIN_H + float(obs[_OBS_CUR_HEIGHT]) * (_MAX_H - _MIN_H)

        self._xy_drift += lin_vel[:2] * _CONTROL_DT

        dq = np.empty(self._model.nv)
        dq[0:2] = self._xy_drift
        dq[2] = height - qpos_star[2]
        dq[3] = -grav[1]
        dq[4] = grav[0]
        dq[5] = float(obs[_OBS_YAW_ERROR])
        dq[6:] = obs[_OBS_JOINT_POS] - qpos_star[7:]

        dv = np.concatenate([lin_vel, obs[_OBS_ANG_VEL], obs[_OBS_JOINT_VEL]])
        return np.concatenate([dq, dv])[kept_state_indices(self._model.nv)]

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map 42-dim obs -> 10-dim normalized direct-torque action."""
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (42,):
            raise ValueError(
                f"FullStateLQRController requires 42-dim obs, got {obs.shape}"
            )

        tau = self._design["ctrl_star"] - self._design["K"] @ self._state_error(obs)
        tau = np.clip(tau, self._ctrl_min, self._ctrl_max)

        # Rate limit, same 400 Nm/s ceiling as ACC and the torque baseline.
        max_step = self._max_torque_rate * _CONTROL_DT
        tau = self._tau_prev + np.clip(tau - self._tau_prev, -max_step, max_step)
        self._tau_prev = tau.copy()

        span = np.where(
            (self._ctrl_max - self._ctrl_min) < 1e-9,
            1.0, self._ctrl_max - self._ctrl_min,
        )
        action = 2.0 * (tau - self._ctrl_min) / span - 1.0
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # -- introspection ------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        d = self._design
        return {
            "controller_type": "FullStateLQRController",
            "action_path": "direct_torque",
            "linearization_method": "central_difference_on_full_control_step",
            "control_rate_hz": round(1.0 / _CONTROL_DT, 1),
            "design_height_m": self._height_cmd_m,
            "state_dim": d["state_dim"],
            "control_dim": int(self._model.nu),
            "K_shape": list(d["K"].shape),
            "r_scale": self._r_scale,
            "equilibrium_base_z_m": round(d["base_z_m"], 6),
            "equilibrium_residual_force_N": round(d["residual_force_N"], 6),
            "equilibrium_torque_norm_Nm": round(
                float(np.linalg.norm(d["ctrl_star"])), 4),
            "cond_A": d["cond_A"],
            "rank_A": d["rank_A"],
            "max_abs_eig_open_loop": d["max_abs_eig_open"],
            "n_marginal_or_unstable_modes": d["n_marginal_or_unstable_modes"],
            "stabilizable_pbh": d["stabilizable_pbh"],
            "max_abs_eig_closed_loop": d["max_abs_eig_closed"],
            "max_abs_gain": d["max_abs_gain"],
            "max_torque_rate_nm_s": self._max_torque_rate,
            "model_path": self._model_path,
        }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Verify the two things the design silently depends on.

    1. The observation-to-state reconstruction agrees with the quantity the
       Jacobian was differentiated in, ``mj_differentiatePos``.
    2. The equilibrium is one: the unbalanced force is a small fraction of
       body weight, and the reduction assertion holds.
    """
    root = Path(__file__).resolve().parents[2]
    xml = root / "assets" / "robot" / "wheeled_biped_real.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing")
    mujoco.mj_resetDataKeyframe(m, d, key)
    qpos_star = d.qpos.copy()
    keep = kept_state_indices(m.nv)

    ctrl = FullStateLQRController.__new__(FullStateLQRController)
    ctrl._model = m
    ctrl._design = {"qpos_star": qpos_star}
    ctrl._xy_drift = np.zeros(2)

    def worst_error(scale: float) -> float:
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(5):
            ctrl._xy_drift[:] = 0.0
            dq_true = rng.normal(0, scale, m.nv)
            qpos = qpos_star.copy()
            mujoco.mj_integratePos(m, qpos, dq_true, 1.0)
            d.qpos[:] = qpos
            d.qvel[:] = rng.normal(0, 2.5 * scale, m.nv)
            mujoco.mj_forward(m, d)

            truth = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, truth, 1.0, qpos_star, d.qpos)

            # Synthesize the observation BalanceEnv would emit for this state.
            quat = d.qpos[3:7]
            qinv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
            grav = np.zeros(3)
            mujoco.mju_rotVecQuat(grav, np.array([0.0, 0.0, -1.0]), qinv)
            lin_b = np.zeros(3)
            mujoco.mju_rotVecQuat(lin_b, d.qvel[:3].copy(), qinv)
            ang_b = np.zeros(3)
            mujoco.mju_rotVecQuat(ang_b, d.qvel[3:6].copy(), qinv)
            w, x, y, z = quat
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            obs = np.zeros(42)
            obs[_OBS_GRAVITY] = grav
            obs[_OBS_LIN_VEL] = lin_b
            obs[_OBS_ANG_VEL] = ang_b
            obs[_OBS_JOINT_POS] = d.qpos[7:]
            obs[_OBS_JOINT_VEL] = d.qvel[6:]
            obs[_OBS_CUR_HEIGHT] = (d.qpos[2] - _MIN_H) / (_MAX_H - _MIN_H)
            obs[_OBS_YAW_ERROR] = yaw

            got = ctrl._state_error(obs)
            want = np.concatenate([truth, d.qvel])[keep]
            # base xy is integrated, not measured; it is zero on the first step.
            worst = max(worst, float(np.abs(got[2:] - want[2:]).max()))
        return worst

    # The reconstruction inverts two small-angle relations (body gravity to
    # tilt, body-frame to world-frame velocity), so it is exact only to first
    # order.  What must hold is that the residual is *second* order: halving
    # the perturbation must quarter the error.  A wrong axis, sign or frame
    # would leave a first-order residual and halve instead.
    coarse, fine = worst_error(0.02), worst_error(0.01)
    ratio = coarse / fine
    assert 3.5 < ratio < 4.5, (
        f"obs->state residual is not second order (error ratio {ratio:.2f}, "
        f"expected ~4): the reconstruction has a first-order error"
    )

    # Equilibrium and reduction.
    des = design(m, qpos_star, r_scale=1.0)
    weight_N = float(m.body_subtreemass[1] * 9.81)
    assert des["residual_force_N"] < 0.01 * weight_N, (
        f"not an equilibrium: {des['residual_force_N']:.3f} N unbalanced "
        f"vs {weight_N:.2f} N body weight"
    )
    assert des["rank_A"] == des["state_dim"], "reduced A is rank-deficient"
    assert des["stabilizable_pbh"], "reduced system is not stabilizable"

    # The design must actually stand at the commanded height across the whole
    # band.  The shared ``_build_height_ik`` polynomial silently failed this
    # (0.60, 0.65 and 0.70 m all designed a robot standing at ~0.73 m), which
    # showed up only as a 7.6 cm height RMSE in evaluation.
    for h in (_MIN_H, 0.50, 0.55, 0.60, _MAX_H):
        pose = standing_pose(m, h)
        assert abs(pose[2] - h) < 1e-4, (
            f"standing_pose({h}) stands at {pose[2]:.5f} m"
        )
        q_eq, _, res = standing_equilibrium(m, pose)
        assert res < 0.05 * weight_N, (
            f"equilibrium at {h} m left {res:.3f} N unbalanced"
        )
        assert abs(q_eq[2] - h) < 5e-3, (
            f"equilibrium at {h} m drifted to {q_eq[2]:.5f} m"
        )
    print(f"height reference exact and equilibrium solved over "
          f"[{_MIN_H}, {_MAX_H}] m")

    print(f"obs->state reconstruction is second order "
          f"(err {coarse:.2e} -> {fine:.2e} on halving, ratio {ratio:.2f})")
    print(f"equilibrium: base z = {des['base_z_m']:.6f} m, "
          f"residual {des['residual_force_N']:.4f} N of {weight_N:.2f} N weight")
    print(f"reduced system: {des['state_dim']} states, "
          f"cond(A) = {des['cond_A']:.3e}, rank {des['rank_A']}, "
          f"max|lambda| = {des['max_abs_eig_open']:.6f}, stabilizable")
    print("all checks passed")


if __name__ == "__main__":
    _self_check()
