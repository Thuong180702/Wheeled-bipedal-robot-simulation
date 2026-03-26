"""
Standing quality analysis -- pure numpy, no JAX/MuJoCo dependencies.

Computes per-signal metrics from a TelemetryRecorder.to_numpy() dict to
surface common reward-exploitation patterns in standing policies.

Exploit patterns detected
-------------------------
wheel_spin_mean_rads    Robot uses wheel momentum to "balance" rather than
                        holding posture -- wheels spin while body looks upright.
height_std_m            Vertical oscillation: policy briefly hits the
                        exp-kernel reward peak then bounces.
xy_drift_max_m          Slow rolling / drifting while appearing balanced.
                        position_drift reward only penalises far-from-anchor;
                        a drifting robot can still score well per step.
roll_mean_abs_deg       Chronic sideways lean below the 46 deg termination
                        threshold that reward tolerates silently.
pitch_mean_abs_deg      Chronic forward/back lean (same blind spot).
ctrl_jitter_mean_nm     Chattering actuation: action_rate penalty may be
                        too small or the policy oscillates near a saddle.
leg_asymmetry_mean_rad  One leg crouched more than the other; symmetry
                        reward weight may be insufficient.
ang_vel_rms_rads        Chronic pitch/roll wobble below termination -- robot
                        is oscillating but survives.

Usage
-----
from wheeled_biped.utils.telemetry import TelemetryRecorder
from wheeled_biped.eval.standing_quality import compute_standing_signals

recorder = TelemetryRecorder()
# ... fill recorder via recorder.record(mj_data) in a rollout loop ...
signals = compute_standing_signals(recorder.to_numpy(), height_cmd=0.69)
print(signals["flags"])        # list of human-readable warning strings
print(signals["num_suspicious"])  # count of triggered flags
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Warning thresholds
# ---------------------------------------------------------------------------
# Values *above* these thresholds are flagged as suspicious.  Chosen to be
# conservative: normal well-trained standing should stay comfortably below
# all of them.

THRESHOLDS: dict[str, float] = {
    "wheel_spin_mean_rads":    3.0,   # rad/s  -- still-standing should be ~0
    "height_std_m":            0.025, # m      -- > 2.5 cm bounce is suspicious
    "xy_drift_max_m":          0.15,  # m      -- > 15 cm drift is suspicious
    "roll_mean_abs_deg":       4.0,   # deg    -- chronic sideways lean
    "pitch_mean_abs_deg":      4.0,   # deg    -- chronic forward/back lean
    "ctrl_jitter_mean_nm":     0.5,   # Nm     -- mean step-to-step ctrl change
    "leg_asymmetry_mean_rad":  0.10,  # rad    -- hip_pitch / knee diff L vs R
    "ang_vel_rms_rads":        0.5,   # rad/s  -- torso pitch/roll oscillation
}

_FLAG_MESSAGES: dict[str, str] = {
    "wheel_spin_mean_rads": (
        "HIGH wheel spin ({val:.2f} rad/s mean) -- "
        "possible wheel-momentum exploit: robot spins wheels to stay upright "
        "rather than holding posture"
    ),
    "height_std_m": (
        "HEIGHT oscillation (std={val:.3f} m) -- "
        "vertical bounce or unstable crouching: reward may peak briefly then drift"
    ),
    "xy_drift_max_m": (
        "XY drift ({val:.2f} m max from start) -- "
        "robot rolling or drifting while appearing balanced; "
        "position_drift reward may be insufficient"
    ),
    "roll_mean_abs_deg": (
        "CHRONIC ROLL ({val:.1f} deg mean |roll|) -- "
        "robot leaning sideways: visually bad even if below 46 deg termination"
    ),
    "pitch_mean_abs_deg": (
        "CHRONIC PITCH ({val:.1f} deg mean |pitch|) -- "
        "robot leaning forward/back; may exploit reward curvature near target"
    ),
    "ctrl_jitter_mean_nm": (
        "CONTROL JITTER ({val:.3f} Nm mean step-to-step change) -- "
        "chattering actuation: action_rate penalty may be too weak"
    ),
    "leg_asymmetry_mean_rad": (
        "LEG ASYMMETRY ({val:.3f} rad mean hip_pitch+knee diff) -- "
        "one leg crouched more than the other; symmetry reward may be too weak"
    ),
    "ang_vel_rms_rads": (
        "TORSO WOBBLE (ang_vel RMS={val:.2f} rad/s) -- "
        "chronic pitch/roll oscillation below termination threshold"
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_standing_signals(
    tele: dict[str, np.ndarray],
    height_cmd: float | None = None,
) -> dict:
    """Compute standing quality signals from a telemetry numpy dict.

    Args:
        tele: Output of ``TelemetryRecorder.to_numpy()``.  Must contain the
              standard keys produced by ``TelemetryRecorder.record()``.
              Missing keys are treated as zeros (signal is skipped gracefully).
        height_cmd: Height command in metres used during the rollout (e.g.
                    0.69).  When provided, ``height_tracking_rmse_m`` is
                    computed.  Pass ``None`` to skip (value will be NaN).

    Returns:
        Dict with:
        - One float per quality signal (see module docstring for names).
        - ``"flags"`` -- list of human-readable warning strings for any
          signal that exceeded its threshold.
        - ``"num_suspicious"`` -- int count of triggered flags.

        Returns ``{"flags": [...], "num_suspicious": 1}`` for empty input.
    """
    if not tele:
        return {"flags": ["no telemetry data collected"], "num_suspicious": 1}

    signals: dict = {}
    flags: list[str] = []

    def _record(key: str, val: float) -> None:
        """Store val and append a flag if it exceeds the threshold."""
        signals[key] = val
        thresh = THRESHOLDS.get(key)
        if thresh is not None and val > thresh:
            flags.append(_FLAG_MESSAGES[key].format(val=val))

    # ── 1. Wheel spin ────────────────────────────────────────────────────────
    l_wv = tele.get("l_wheel_vel", np.zeros(1))
    r_wv = tele.get("r_wheel_vel", np.zeros(1))
    wheel_spin = float(np.mean((np.abs(l_wv) + np.abs(r_wv)) / 2.0))
    _record("wheel_spin_mean_rads", wheel_spin)

    # ── 2. Height stability ──────────────────────────────────────────────────
    tz = tele.get("torso_z", np.zeros(1))
    signals["height_mean_m"] = float(np.mean(tz))
    _record("height_std_m", float(np.std(tz)))

    if height_cmd is not None:
        signals["height_tracking_rmse_m"] = float(
            np.sqrt(np.mean((tz - height_cmd) ** 2))
        )
    else:
        signals["height_tracking_rmse_m"] = float("nan")

    # ── 3. XY drift from start position ─────────────────────────────────────
    tx = tele.get("torso_x", np.zeros(1))
    ty = tele.get("torso_y", np.zeros(1))
    xy_disp = np.sqrt((tx - tx[0]) ** 2 + (ty - ty[0]) ** 2)
    signals["xy_drift_final_m"] = float(xy_disp[-1])
    _record("xy_drift_max_m", float(np.max(xy_disp)))

    # ── 4. Chronic roll / pitch lean ────────────────────────────────────────
    roll_abs = np.abs(np.degrees(tele.get("roll_rad", np.zeros(1))))
    pitch_abs = np.abs(np.degrees(tele.get("pitch_rad", np.zeros(1))))
    _record("roll_mean_abs_deg",  float(np.mean(roll_abs)))
    _record("pitch_mean_abs_deg", float(np.mean(pitch_abs)))

    # ── 5. Control jitter ────────────────────────────────────────────────────
    ctrl_keys = [
        "l_hip_roll_ctrl", "l_hip_yaw_ctrl", "l_hip_pitch_ctrl",
        "l_knee_ctrl",     "l_wheel_ctrl",
        "r_hip_roll_ctrl", "r_hip_yaw_ctrl", "r_hip_pitch_ctrl",
        "r_knee_ctrl",     "r_wheel_ctrl",
    ]
    ctrl_present = [tele[k] for k in ctrl_keys if k in tele]
    if ctrl_present:
        ctrl_mat = np.stack(ctrl_present, axis=1)        # (T, n_channels)
        jitter = float(np.mean(np.abs(np.diff(ctrl_mat, axis=0))))
        _record("ctrl_jitter_mean_nm", jitter)
    else:
        signals["ctrl_jitter_mean_nm"] = float("nan")

    # ── 6. Leg asymmetry (hip_pitch and knee, L vs R) ────────────────────────
    l_hp = tele.get("l_hip_pitch_pos", np.zeros(1))
    r_hp = tele.get("r_hip_pitch_pos", np.zeros(1))
    l_kn = tele.get("l_knee_pos",      np.zeros(1))
    r_kn = tele.get("r_knee_pos",      np.zeros(1))
    hp_asym = float(np.mean(np.abs(l_hp - r_hp)))
    kn_asym = float(np.mean(np.abs(l_kn - r_kn)))
    signals["leg_asymmetry_hip_pitch_rad"] = hp_asym
    signals["leg_asymmetry_knee_rad"]      = kn_asym
    _record("leg_asymmetry_mean_rad", (hp_asym + kn_asym) / 2.0)

    # ── 7. Torso angular velocity RMS (roll + pitch axes) ────────────────────
    wx = tele.get("body_wx", np.zeros(1))
    wy = tele.get("body_wy", np.zeros(1))
    _record("ang_vel_rms_rads", float(np.sqrt(np.mean(wx ** 2 + wy ** 2))))

    signals["flags"]         = flags
    signals["num_suspicious"] = len(flags)
    return signals
