"""Torque sign convention probes.

Non-destructive, short rollouts that measure the direction of joint
acceleration produced by positive and negative actuator commands.

Phase 1.5 adds bias-subtracted / delta-based measurement with escalatable
probe torque so that joints dominated by gravitational bias (hip_pitch,
knee) can still be resolved.

No controller logic is used — only direct mj_data.ctrl assignment.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from mujoco import mjtObj


# ── Default probe parameters ─────────────────────────────────────
PROBE_TORQUE_DEFAULT = 10.0       # Nm — request; may be capped by limit
MIN_DELTA_QACC = 1e-3             # rad/s² — below this, delta is noise
ESCALATION_TORQUES = [5.0, 10.0, 20.0, 40.0]  # Nm — ascending
MAX_PROBE_FRACTION = 0.25         # fraction of max |ctrl| limit


def _safe_probe_torque(
    model: mujoco.MjModel,
    actuator_id: int,
    requested: float,
) -> float:
    """Return the probe torque, capped at a fraction of the actuator control limit.

    Args:
        model: MuJoCo MjModel.
        actuator_id: Actuator index.
        requested: Desired probe torque magnitude (Nm).

    Returns:
        Safe probe torque magnitude (positive float).
    """
    ctrlrange = model.actuator_ctrlrange[actuator_id]
    max_ctrl = max(abs(ctrlrange[0]), abs(ctrlrange[1]))
    cap = max_ctrl * MAX_PROBE_FRACTION
    if cap <= 0.0:
        return requested  # no sensible cap available
    return min(requested, cap)


def torque_sign_probe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    actuator_name: str,
    probe_torque: float = PROBE_TORQUE_DEFAULT,
    min_delta_qacc: float = MIN_DELTA_QACC,
    escalate: bool = True,
) -> dict[str, Any]:
    """Bias-subtracted torque sign probe using zero/+probe/−probe measurements.

    For each actuator:
      1. ctrl = 0       → measure qacc_zero
      2. ctrl = +probe  → measure qacc_plus
      3. ctrl = −probe  → measure qacc_minus

    Then:
      delta_plus  = qacc_plus  − qacc_zero
      delta_minus = qacc_minus − qacc_zero
      delta_pair  = qacc_plus  − qacc_minus

    A sign is MEASURED if:
      abs(delta_pair) > min_delta_qacc
      AND sign(delta_plus) is opposite sign(delta_minus)
      OR delta_pair has a stable nonzero direction.

    If the result remains AMBIGUOUS and escalate=True, the probe
    torque is increased (up to MAX_PROBE_FRACTION of the actuator
    ctrl limit) and the measurement is retried.

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData (will be reset to keyframe and restored).
        joint_name: Joint name (e.g., "l_hip_roll").
        actuator_name: Actuator name (e.g., "l_hip_roll_motor").
        probe_torque: Requested torque magnitude in Nm (default 10.0).
            Will be capped at 0.25 × max|ctrlrange| for safety.
        min_delta_qacc: Minimum abs(delta_pair) to consider measurable
            (default 1e-3 rad/s²).
        escalate: If True, retry with higher probe torques when
            AMBIGUOUS (up to 0.25 × actuator limit).

    Returns:
        dict with keys:
            joint_name, actuator_name, joint_id, actuator_id,
            vel_index, probe_torque_requested, probe_torque_used,
            qacc_zero, qacc_plus, qacc_minus,
            delta_plus, delta_minus, delta_pair,
            sign_consistent, sign_consistent_delta,
            measured_sign_convention, outcome, note.
    """
    # ── resolve IDs ─────────────────────────────────────────────
    joint_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
    actuator_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_ACTUATOR, actuator_name)

    if joint_id < 0:
        return _missing_result(
            joint_name, actuator_name, probe_torque,
            f"joint '{joint_name}' not found",
        )
    if actuator_id < 0:
        return _missing_result(
            joint_name, actuator_name, probe_torque,
            f"actuator '{actuator_name}' not found",
        )

    vel_adr = model.jnt_dofadr[joint_id]

    # ── safe probe torque (capped) ──────────────────────────────
    safe_torque = _safe_probe_torque(model, actuator_id, probe_torque)

    # ── primary measurement ─────────────────────────────────────
    primary = _measure_deltas(
        model, data, actuator_id, vel_adr, safe_torque, min_delta_qacc,
    )
    primary["probe_torque_requested"] = probe_torque

    # ── escalation if ambiguous ─────────────────────────────────
    if escalate and primary["outcome"] == "AMBIGUOUS":
        ctrlrange = model.actuator_ctrlrange[actuator_id]
        max_ctrl = max(abs(ctrlrange[0]), abs(ctrlrange[1]))
        cap = max_ctrl * MAX_PROBE_FRACTION

        # Try higher torques from the escalation list
        for candidate in ESCALATION_TORQUES:
            if candidate <= safe_torque + 1e-6:
                continue  # already tried or lower
            if candidate > cap + 1e-6:
                continue  # exceeds safety cap

            escalated = _measure_deltas(
                model, data, actuator_id, vel_adr, candidate, min_delta_qacc,
            )
            if escalated["outcome"] == "MEASURED":
                escalated["probe_torque_requested"] = probe_torque
                escalated["note"] = (
                    f"Escalated from {safe_torque:.1f} → {candidate:.1f} Nm "
                    f"to overcome gravitational bias."
                )
                return _finalise_result(
                    joint_name, actuator_name, joint_id, actuator_id,
                    vel_adr, escalated,
                )

        # No escalation succeeded — attach note
        primary["note"] = (
            f"Escalation up to {cap:.1f} Nm ({MAX_PROBE_FRACTION:.0%} of "
            f"±{max_ctrl:.0f} Nm limit) did not resolve. Gravitational or "
            f"bias loading remains dominant."
        )

    return _finalise_result(
        joint_name, actuator_name, joint_id, actuator_id, vel_adr, primary,
    )


def _measure_deltas(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: int,
    vel_adr: int,
    probe_torque: float,
    min_delta_qacc: float,
) -> dict[str, Any]:
    """Run zero/+probe/−probe measurements and compute deltas.

    State is saved before and restored after.  Each measurement
    resets to keyframe, sets ctrl, steps once, and records qacc.

    Returns a partial result dict (missing joint/actuator name/id
    metadata — those are filled in by the caller).
    """
    # ── save state ──────────────────────────────────────────────
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    ctrl_orig = data.ctrl.copy()

    # ── zero probe ──────────────────────────────────────────────
    _reset_to_keyframe(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    qacc_zero = float(data.qacc[vel_adr])

    # ── positive probe ──────────────────────────────────────────
    _reset_to_keyframe(model, data)
    data.ctrl[:] = 0.0
    data.ctrl[actuator_id] = probe_torque
    mujoco.mj_step(model, data)
    qacc_plus = float(data.qacc[vel_adr])

    # ── negative probe ──────────────────────────────────────────
    _reset_to_keyframe(model, data)
    data.ctrl[:] = 0.0
    data.ctrl[actuator_id] = -probe_torque
    mujoco.mj_step(model, data)
    qacc_minus = float(data.qacc[vel_adr])

    # ── restore state ───────────────────────────────────────────
    data.qpos[:] = qpos_orig
    data.qvel[:] = qvel_orig
    data.ctrl[:] = ctrl_orig
    mujoco.mj_forward(model, data)

    # ── compute deltas ──────────────────────────────────────────
    delta_plus = qacc_plus - qacc_zero
    delta_minus = qacc_minus - qacc_zero
    delta_pair = qacc_plus - qacc_minus

    # ── delta-based sign analysis ───────────────────────────────
    # A sign is measurable via deltas if:
    #   abs(delta_pair) > threshold  AND
    #   delta_plus and delta_minus have opposite signs
    #   (i.e. the probe dominates the bias).
    abs_pair = abs(delta_pair)
    delta_opposite = bool(
        (delta_plus > 0 and delta_minus < 0)
        or (delta_plus < 0 and delta_minus > 0)
    )

    # Also keep the legacy absolute-sign check for backwards compat
    abs_sign_consistent = bool(np.sign(qacc_plus) == -np.sign(qacc_minus))

    # Determine outcome
    if not (np.isfinite(qacc_zero) and np.isfinite(qacc_plus) and np.isfinite(qacc_minus)):
        outcome = "INVALID"
    elif abs_pair > min_delta_qacc and delta_opposite:
        outcome = "MEASURED"
    elif abs_pair > min_delta_qacc:
        # delta has a stable nonzero direction even if both deltas
        # point the same way (should be extremely rare)
        outcome = "MEASURED"
    else:
        outcome = "AMBIGUOUS"

    # Measured sign convention from delta_plus
    if delta_plus > min_delta_qacc:
        measured_convention = "positive_ctrl_increases_joint_acceleration"
    elif delta_plus < -min_delta_qacc:
        measured_convention = "positive_ctrl_decreases_joint_acceleration"
    elif abs(qacc_plus) > min_delta_qacc:
        # fallback to absolute (legacy format)
        if qacc_plus > 0:
            measured_convention = "positive_ctrl_→_positive_qacc"
        elif qacc_plus < 0:
            measured_convention = "positive_ctrl_→_negative_qacc"
        else:
            measured_convention = "positive_ctrl_→_zero_qacc"
    else:
        measured_convention = "ambiguous_bias_dominated"

    return {
        "probe_torque_used": probe_torque,
        "qacc_zero": qacc_zero,
        "qacc_plus": qacc_plus,
        "qacc_minus": qacc_minus,
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "delta_pair": delta_pair,
        "delta_opposite": delta_opposite,
        "sign_consistent": abs_sign_consistent,         # legacy
        "sign_consistent_delta": delta_opposite,         # delta-based
        "measured_sign_convention": measured_convention,
        "outcome": outcome,
        "note": None,
    }


def _finalise_result(
    joint_name: str,
    actuator_name: str,
    joint_id: int,
    actuator_id: int,
    vel_adr: int,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """Attach joint/actuator metadata to a measurement result dict."""
    return {
        "joint_name": joint_name,
        "actuator_name": actuator_name,
        "joint_id": int(joint_id),
        "actuator_id": int(actuator_id),
        "vel_index": int(vel_adr),
        "probe_torque_requested": measurement.get("probe_torque_requested", measurement["probe_torque_used"]),
        "probe_torque_used": measurement["probe_torque_used"],
        "qacc_zero": measurement["qacc_zero"],
        "qacc_plus": measurement["qacc_plus"],
        "qacc_minus": measurement["qacc_minus"],
        "delta_plus": measurement["delta_plus"],
        "delta_minus": measurement["delta_minus"],
        "delta_pair": measurement["delta_pair"],
        "sign_consistent": measurement["sign_consistent"],
        "sign_consistent_delta": measurement.get("sign_consistent_delta"),
        "measured_sign_convention": measurement["measured_sign_convention"],
        "outcome": measurement["outcome"],
        "note": measurement.get("note"),
    }


def _reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset data to keyframe 0 if available, otherwise do nothing."""
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def _missing_result(
    joint_name: str,
    actuator_name: str,
    probe_torque: float,
    note: str,
) -> dict[str, Any]:
    """Return a structured MISSING result when lookup fails."""
    return {
        "joint_name": joint_name,
        "actuator_name": actuator_name,
        "joint_id": -1,
        "actuator_id": -1,
        "vel_index": -1,
        "probe_torque_requested": probe_torque,
        "probe_torque_used": 0.0,
        "qacc_zero": None,
        "qacc_plus": None,
        "qacc_minus": None,
        "delta_plus": None,
        "delta_minus": None,
        "delta_pair": None,
        "sign_consistent": None,
        "sign_consistent_delta": None,
        "measured_sign_convention": None,
        "outcome": "MISSING",
        "note": note,
    }
