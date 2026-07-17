"""LR support-drift sign/phase audit for the lr_support_drift_sign_phase sweep task.

Computes per-candidate:

A. State/torque sign checks:
   - correlation(state, LR_feedback)
   - correlation(state, total torque)
   - sign agreement with stabilizing direction
   - windows where error grows while LR feedback assists the growth

B. Phase analysis:
   - dominant frequency in post-push window
   - phase/correlation between pitch/pitch_rate/support_error/support_velocity and LR_feedback
   - 0.35-0.65 Hz band evaluation

C. Support-drift event analysis:
   - first time support error exceeds 0.25/0.50/1.00 m
   - LR torque sign at each threshold

D. Torque magnitude analysis

E. Recovery metrics
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ============================================================
# Column-name resolver — hip-yaw + all known variants
# ============================================================
_HIP_YAW_LEFT_CANDIDATES = [
    "l_hip_yaw_pos", "l_hip_yaw_pos_rad",
    "l_hip_yaw_joint_pos", "l_hip_yaw_angle_rad",
]
_HIP_YAW_RIGHT_CANDIDATES = [
    "r_hip_yaw_pos", "r_hip_yaw_pos_rad",
    "r_hip_yaw_joint_pos", "r_hip_yaw_angle_rad",
]
_HIP_YAW_ABS_MAX_CANDIDATES = [
    "hip_yaw_abs_max", "hip_yaw_abs_max_rad",
]
_HIP_YAW_COMMON_ERROR_CANDIDATES = [
    "hip_yaw_common_error_rad", "hip_yaw_common_error",
]
_HIP_YAW_DIV_ERROR_CANDIDATES = [
    "hip_yaw_divergence_error_rad", "hip_yaw_divergence_error",
]


def _resolve_column(header: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in header:
            return c
    return None


def load_telemetry(csv_path: Path) -> dict:
    """Load telemetry CSV and return structured data."""
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("No data rows")

    header = rows[0].keys()
    n = len(rows)

    def col(name, dtype=float):
        """Extract column, falling back to resolution list."""
        vals = []
        for r in rows:
            v = r.get(name, "")
            if v == "" or v is None:
                vals.append(np.nan)
            else:
                try:
                    vals.append(dtype(v))
                except (ValueError, TypeError):
                    vals.append(np.nan)
        return np.array(vals)

    # Resolve hip-yaw columns
    hy_left_name = _resolve_column(header, _HIP_YAW_LEFT_CANDIDATES)
    hy_right_name = _resolve_column(header, _HIP_YAW_RIGHT_CANDIDATES)
    hy_abs_max_name = _resolve_column(header, _HIP_YAW_ABS_MAX_CANDIDATES)
    hy_common_name = _resolve_column(header, _HIP_YAW_COMMON_ERROR_CANDIDATES)
    hy_div_name = _resolve_column(header, _HIP_YAW_DIV_ERROR_CANDIDATES)

    # Core state
    pitch_x = col("pitch_x_rad")
    pitch_rate = col("pitch_rate_effective_rad_s")
    support_error = col("sagittal_position_error_m")
    support_vel = col("support_position_velocity_m_s")
    com_z = col("com_z_m")
    roll = col("roll_y_rad")

    # LR-specific
    lr_enabled_raw = rows[0].get("LR_enabled", "False") if rows else "False"
    lr_enabled = str(lr_enabled_raw).strip() in ("True", "true", "1")

    lr_feedback = col("LR_feedback_torque_nm")
    lr_eq_ff = col("LR_eq_ff_pass_through_nm")
    lr_total_preclip = col("LR_total_command_preclip_nm")
    lr_total_postclip = col("LR_total_command_postclip_nm")
    lr_replacement_mode = rows[0].get("LR_replacement_mode", "none") if rows else "none"

    # Torque components
    tau_pitch = col("tau_pitch_nm")
    tau_pitch_rate = col("tau_pitch_rate_nm")
    tau_sagittal_vel = col("tau_sagittal_velocity_nm")
    tau_support_vel = col("tau_support_velocity_nm")
    tau_position = col("tau_position_nm")
    tau_cp = col("tau_cp_nm")
    tau_com_vy = col("tau_com_vy_nm")
    tau_common = col("tau_common_final_nm")

    # Hip-yaw
    hip_yaw_l = col(hy_left_name) if hy_left_name else np.full(n, np.nan)
    hip_yaw_r = col(hy_right_name) if hy_right_name else np.full(n, np.nan)
    hip_yaw_abs_max_direct = col(hy_abs_max_name) if hy_abs_max_name else np.full(n, np.nan)
    hip_yaw_common = col(hy_common_name) if hy_common_name else np.full(n, np.nan)
    hip_yaw_div = col(hy_div_name) if hy_div_name else np.full(n, np.nan)

    # Compute hip_yaw_abs_max from raw joint columns if direct not available
    if hy_left_name and hy_right_name:
        hip_yaw_abs_max_computed = np.maximum(np.abs(hip_yaw_l), np.abs(hip_yaw_r))
    else:
        hip_yaw_abs_max_computed = np.full(n, np.nan)

    # Termination
    terminated_raw = rows[-1].get("terminated", "0") if rows else "0"
    terminated = str(terminated_raw).strip() in ("True", "true", "1")
    term_reason = rows[-1].get("termination_reason", "none") if rows else "none"

    # Candidate name
    variant_name_raw = rows[0].get("variant_name", "unknown") if rows else "unknown"
    profile = rows[0].get("vd_sagittal_authority_profile", "unknown") if rows else "unknown"

    return {
        "n": n,
        "pitch_x_rad": pitch_x,
        "pitch_rate_rad_s": pitch_rate,
        "support_error_m": support_error,
        "support_velocity_m_s": support_vel,
        "com_z_m": com_z,
        "roll_rad": roll,
        "lr_enabled": lr_enabled,
        "lr_feedback_nm": lr_feedback,
        "lr_eq_ff_nm": lr_eq_ff,
        "lr_total_preclip_nm": lr_total_preclip,
        "lr_total_postclip_nm": lr_total_postclip,
        "lr_replacement_mode": lr_replacement_mode,
        "tau_pitch_nm": tau_pitch,
        "tau_pitch_rate_nm": tau_pitch_rate,
        "tau_sagittal_vel_nm": tau_sagittal_vel,
        "tau_support_vel_nm": tau_support_vel,
        "tau_position_nm": tau_position,
        "tau_cp_nm": tau_cp,
        "tau_com_vy_nm": tau_com_vy,
        "tau_common_nm": tau_common,
        "hip_yaw_l": hip_yaw_l,
        "hip_yaw_r": hip_yaw_r,
        "hip_yaw_abs_max_direct": hip_yaw_abs_max_direct,
        "hip_yaw_abs_max_computed": hip_yaw_abs_max_computed,
        "hip_yaw_common_error_rad": hip_yaw_common,
        "hip_yaw_div_error_rad": hip_yaw_div,
        "hip_yaw_left_col": hy_left_name,
        "hip_yaw_right_col": hy_right_name,
        "terminated": terminated,
        "termination_reason": str(term_reason),
        "variant_name": str(variant_name_raw),
        "profile": str(profile),
    }


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x ** 2)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, safe for NaN."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 10:
        return np.nan
    xc = x[mask] - np.mean(x[mask])
    yc = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    if denom < 1e-16:
        return np.nan
    return float(np.sum(xc * yc) / denom)


def compute_sign_agreement(
    state: np.ndarray,
    torque: np.ndarray,
    stabilizing_sign: int,
    label: str,
) -> dict:
    """Check whether torque sign agrees with stabilizing direction.

    stabilizing_sign: +1 if torque should be same sign as state to stabilize,
                     -1 if torque should oppose state to stabilize.
    """
    mask = ~np.isnan(state) & ~np.isnan(torque)
    s = state[mask]
    t = torque[mask]
    n = len(s)
    if n < 10:
        return {"label": label, "error": "insufficient_data"}

    # Correct sign: torque sign = stabilizing_sign * sign(state)
    expected_torque_sign = stabilizing_sign * np.sign(s)
    expected_torque_sign[expected_torque_sign == 0] = 1  # neutral → any sign OK
    actual_torque_sign = np.sign(t)
    actual_torque_sign[actual_torque_sign == 0] = 1

    agreement = np.mean(expected_torque_sign == actual_torque_sign)

    # When state magnitude is large (>1 std), agreement should be higher
    large_state_mask = np.abs(s) > np.nanstd(s)
    large_agreement = (
        np.mean(expected_torque_sign[large_state_mask] == actual_torque_sign[large_state_mask])
        if large_state_mask.sum() > 10 else np.nan
    )

    # Correlation
    corr_val = safe_corr(s, t)

    # Destabilizing windows: error grows while torque has same sign as error (assisting growth)
    ds_mask = np.abs(s) > np.nanstd(s) * 0.5
    if ds_mask.sum() > 10:
        error_growing = np.zeros(n, dtype=bool)
        error_growing[1:] = np.abs(s[1:]) > np.abs(s[:-1])
        assisting = expected_torque_sign != actual_torque_sign  # wrong sign = assisting error
        destabilizing_rate = float(np.mean(error_growing[ds_mask] & assisting[ds_mask]))
    else:
        destabilizing_rate = np.nan

    return {
        "label": label,
        "n_samples": int(n),
        "correlation": float(corr_val) if not np.isnan(corr_val) else None,
        "sign_agreement_pct": float(agreement * 100),
        "large_state_sign_agreement_pct": float(large_agreement * 100) if not np.isnan(large_agreement) else None,
        "destabilizing_window_rate_pct": float(destabilizing_rate * 100) if not np.isnan(destabilizing_rate) else None,
        "stabilizing_sign_convention": stabilizing_sign,
        # stabilizing_sign encodes expected correlation sign:
        #   -1 → torque should OPPOSE state → expect negative correlation
        #   +1 → torque should ALIGN with state → expect positive correlation
        # pass_ = True when actual correlation sign matches expected
        "pass_": (
            (corr_val < 0) if stabilizing_sign < 0 else (corr_val > 0)
            if not np.isnan(corr_val) else None
        ),
    }


def compute_phase_analysis(d: dict) -> dict:
    """Phase/correlation analysis in post-push window."""
    n = d["n"]
    post_push_start = 310
    post_push_end = min(n, 900)
    pp = slice(post_push_start, post_push_end)

    pitch = d["pitch_x_rad"][pp]
    pitch_rate = d["pitch_rate_rad_s"][pp]
    support_err = d["support_error_m"][pp]
    support_vel = d["support_velocity_m_s"][pp]
    lr_fb = d["lr_feedback_nm"][pp]
    tau_common = d["tau_common_nm"][pp]

    signals = {
        "pitch_vs_LR_feedback": (pitch, lr_fb),
        "pitch_rate_vs_LR_feedback": (pitch_rate, lr_fb),
        "support_error_vs_LR_feedback": (support_err, lr_fb),
        "support_velocity_vs_LR_feedback": (support_vel, lr_fb),
        "support_error_vs_total_torque": (support_err, tau_common),
        "pitch_vs_total_torque": (pitch, tau_common),
    }

    correlations = {}
    for label, (x, y) in signals.items():
        c = safe_corr(x, y)
        correlations[label] = float(c) if not np.isnan(c) else None

    # FFT in post-push window for 0.35-0.65 Hz band
    pitch_clean = pitch[~np.isnan(pitch)]
    fft_result = {}
    if len(pitch_clean) > 128:
        fft = np.abs(np.fft.rfft(pitch_clean - np.mean(pitch_clean)))
        freqs = np.fft.rfftfreq(len(pitch_clean), 0.01)

        # Dominant in 0.3-3.0 Hz
        mask = (freqs >= 0.3) & (freqs <= 3.0)
        if mask.any():
            idx = np.argmax(fft[mask])
            dom_freq = freqs[mask][idx]
            dom_amp = fft[mask][idx] / len(pitch_clean) * 2
        else:
            dom_freq = 0.0
            dom_amp = 0.0

        # 0.52 Hz
        idx_052 = np.argmin(np.abs(freqs - 0.52))
        amp_052 = fft[idx_052] / len(pitch_clean) * 2 if idx_052 < len(fft) else 0.0

        # 2.5 Hz
        idx_25 = np.argmin(np.abs(freqs - 2.5))
        amp_25 = fft[idx_25] / len(pitch_clean) * 2 if idx_25 < len(fft) else 0.0

        # Band energy 0.35-0.65 Hz vs 0.65-3.0 Hz
        band_low_mask = (freqs >= 0.35) & (freqs <= 0.65)
        band_high_mask = (freqs >= 0.65) & (freqs <= 3.0)
        low_band_energy = float(np.sum(fft[band_low_mask] ** 2))
        high_band_energy = float(np.sum(fft[band_high_mask] ** 2))
        low_to_high_ratio = low_band_energy / high_band_energy if high_band_energy > 0 else np.inf

        fft_result = {
            "dominant_freq_hz": float(dom_freq),
            "dominant_amp_deg": float(np.degrees(dom_amp)),
            "amp_0p52_hz_deg": float(np.degrees(amp_052)),
            "amp_2p5_hz_deg": float(np.degrees(amp_25)),
            "low_band_0p35_0p65_energy_ratio": float(low_to_high_ratio),
        }

    return {
        "post_push_correlations": correlations,
        "post_push_fft": fft_result,
    }


def compute_drift_events(d: dict) -> dict:
    """Support-drift event analysis."""
    support_err = d["support_error_m"]
    lr_fb = d["lr_feedback_nm"]
    n = d["n"]

    thresholds = [0.25, 0.50, 1.00]
    events = {}

    for thresh in thresholds:
        idx = np.where(np.abs(support_err) > thresh)[0]
        if len(idx) == 0:
            events[f"threshold_{thresh:.2f}m"] = {"crossed": False, "first_step": None}
            continue

        first = int(idx[0])
        supp_at_thresh = float(support_err[first])
        lr_at_thresh = float(lr_fb[first]) if first < len(lr_fb) else np.nan

        # Is LR torque opposing or assisting the support error?
        # If support_error > 0 (robot too far forward) and LR torque > 0 (pushing forward),
        # LR is assisting the drift.
        lr_assisting = bool(
            (supp_at_thresh > 0 and lr_at_thresh > 0) or
            (supp_at_thresh < 0 and lr_at_thresh < 0)
        ) if not np.isnan(lr_at_thresh) else None

        # Check surrounding window (50 steps before)
        win_start = max(0, first - 50)
        win = slice(win_start, min(first + 10, n))
        supp_win = support_err[win]
        lr_win = lr_fb[win]
        supp_growing = bool(np.abs(supp_at_thresh) > np.nanmax(np.abs(support_err[:first])) * 0.99)

        events[f"threshold_{thresh:.2f}m"] = {
            "crossed": True,
            "first_step": first,
            "support_error_m": supp_at_thresh,
            "lr_feedback_nm": lr_at_thresh,
            "lr_assisting_drift": lr_assisting,
            "support_was_growing_into_threshold": supp_growing,
            "window_corr_support_vs_lr": float(safe_corr(supp_win, lr_win)),
        }

    return events


def compute_torque_magnitudes(d: dict) -> dict:
    """Torque decomposition and saturation."""
    def safe_rms(x):
        return rms(x) if not np.all(np.isnan(x)) else np.nan

    def safe_max(x):
        return float(np.nanmax(np.abs(x))) if not np.all(np.isnan(x)) else np.nan

    tau_pitch = rms(d["tau_pitch_nm"])
    tau_pitch_rate_nm = safe_rms(d["tau_pitch_rate_nm"])
    tau_position = safe_rms(d["tau_position_nm"])
    tau_cp = safe_rms(d["tau_cp_nm"])
    tau_com_vy = safe_rms(d["tau_com_vy_nm"])
    tau_sagittal_vel = safe_rms(d["tau_sagittal_vel_nm"])
    tau_support_vel = safe_rms(d["tau_support_vel_nm"])

    # EQ/FF estimate (for LR) = tau_pitch + tau_position + tau_cp + tau_com_vy
    eq_ff_total = (
        d["tau_pitch_nm"] + d["tau_position_nm"] + d["tau_cp_nm"] + d["tau_com_vy_nm"]
    )
    eq_ff_rms = safe_rms(eq_ff_total)

    lr_fb_rms = safe_rms(d["lr_feedback_nm"])
    lr_eq_ff_rms = safe_rms(d["lr_eq_ff_nm"])
    lr_total_rms = safe_rms(d["lr_total_preclip_nm"])
    tau_common_rms = safe_rms(d["tau_common_nm"])

    # Saturation rows: when |LR_total_preclip| > |LR_total_postclip| + 0.01
    pre = d["lr_total_preclip_nm"]
    post = d["lr_total_postclip_nm"]
    sat_mask = np.abs(pre) > (np.abs(post) + 0.01)
    saturation_rate = float(np.mean(sat_mask)) * 100 if sat_mask.any() else 0.0

    return {
        "eq_ff_estimate_rms_nm": float(eq_ff_rms) if not np.isnan(eq_ff_rms) else None,
        "eq_ff_estimate_max_nm": safe_max(eq_ff_total),
        "lr_feedback_rms_nm": float(lr_fb_rms) if not np.isnan(lr_fb_rms) else None,
        "lr_feedback_max_nm": safe_max(d["lr_feedback_nm"]),
        "lr_eq_ff_pass_through_rms_nm": float(lr_eq_ff_rms) if not np.isnan(lr_eq_ff_rms) else None,
        "lr_total_preclip_rms_nm": float(lr_total_rms) if not np.isnan(lr_total_rms) else None,
        "lr_total_preclip_max_nm": safe_max(d["lr_total_preclip_nm"]),
        "tau_common_final_rms_nm": float(tau_common_rms) if not np.isnan(tau_common_rms) else None,
        "tau_common_final_max_nm": safe_max(d["tau_common_nm"]),
        "lr_saturation_rate_pct": float(saturation_rate),
        # Component breakdown
        "tau_pitch_rms_nm": float(tau_pitch) if not np.isnan(tau_pitch) else None,
        "tau_pitch_rate_rms_nm": float(tau_pitch_rate_nm) if not np.isnan(tau_pitch_rate_nm) else None,
        "tau_position_rms_nm": float(tau_position) if not np.isnan(tau_position) else None,
        "tau_cp_rms_nm": float(tau_cp) if not np.isnan(tau_cp) else None,
        "tau_com_vy_rms_nm": float(tau_com_vy) if not np.isnan(tau_com_vy) else None,
        "tau_sagittal_vel_rms_nm": float(tau_sagittal_vel) if not np.isnan(tau_sagittal_vel) else None,
        "tau_support_vel_rms_nm": float(tau_support_vel) if not np.isnan(tau_support_vel) else None,
    }


def compute_recovery_metrics(d: dict) -> dict:
    """Recovery and survival metrics."""
    n = d["n"]
    pitch_deg = np.degrees(d["pitch_x_rad"])
    support_err = d["support_error_m"]
    roll_deg = np.degrees(d["roll_rad"])
    com_z = d["com_z_m"]

    # Sustained hold: pitch within +/-5 deg for >=200 consecutive steps
    pitch_in_band = np.abs(pitch_deg) < 5.0
    sustained_2s = False
    sustained_5s = False
    first_recovery_interval = None
    recovery_later_lost = False

    hold_start = None
    for i in range(n):
        if pitch_in_band[i]:
            if hold_start is None:
                hold_start = i
            else:
                dur = i - hold_start
                if dur >= 200 and not sustained_2s:
                    sustained_2s = True
                    first_recovery_interval = (int(hold_start), int(i))
                if dur >= 500 and not sustained_5s:
                    sustained_5s = True
        else:
            if hold_start is not None and i - hold_start >= 200:
                # Had recovery, now lost it
                recovery_later_lost = True
            hold_start = None

    # Check if terminated by checking last few steps for large pitch
    final_window = slice(max(0, n - 500), n)
    fall = d["terminated"] or np.nanmax(np.abs(pitch_deg)) > 45.0 or np.nanmin(com_z) < 0.2

    # Hip-yaw
    hy_abs = d["hip_yaw_abs_max_computed"]
    hy_direct = d["hip_yaw_abs_max_direct"]
    hy_abs_max = float(np.nanmax(hy_abs)) if not np.all(np.isnan(hy_abs)) else (
        float(np.nanmax(hy_direct)) if not np.all(np.isnan(hy_direct)) else None
    )
    hy_common_max = float(np.nanmax(np.abs(d["hip_yaw_common_error_rad"]))) if not np.all(np.isnan(d["hip_yaw_common_error_rad"])) else None

    return {
        "completed_steps": n,
        "terminated": bool(d["terminated"]),
        "termination_reason": d["termination_reason"],
        "fall": bool(fall),
        "pitch_rms_deg": rms(pitch_deg),
        "final_pitch_rms_deg": rms(pitch_deg[final_window]),
        "pitch_max_deg": float(np.nanmax(np.abs(pitch_deg))),
        "support_rms_m": rms(support_err),
        "final_support_rms_m": rms(support_err[final_window]),
        "support_max_m": float(np.nanmax(np.abs(support_err))),
        "roll_rms_deg": rms(roll_deg),
        "roll_max_deg": float(np.nanmax(np.abs(roll_deg))),
        "min_height_m": float(np.nanmin(com_z)),
        "max_height_m": float(np.nanmax(com_z)),
        "sustained_2s_hold": sustained_2s,
        "sustained_5s_hold": sustained_5s,
        "first_recovery_interval": first_recovery_interval,
        "recovery_later_lost": recovery_later_lost,
        "hip_yaw_abs_max_rad": hy_abs_max,
        "hip_yaw_common_error_max_rad": hy_common_max,
        "hip_yaw_left_col_used": d["hip_yaw_left_col"],
        "hip_yaw_right_col_used": d["hip_yaw_right_col"],
    }


def analyze_candidate(csv_path: Path, name: str) -> dict:
    """Full analysis of one candidate."""
    d = load_telemetry(csv_path)

    # Sign checks for LR-enabled candidates
    sign_checks = {}
    if d["lr_enabled"]:
        lr_fb = d["lr_feedback_nm"]
        # Stabilizing sign conventions:
        # - support_error: torque should OPPOSE error (stabilizing_sign=-1)
        # - support_velocity: torque should OPPOSE velocity (stabilizing_sign=-1)
        # - pitch: torque should OPPOSE pitch (stabilizing_sign=-1)
        # - pitch_rate: torque should OPPOSE rate (stabilizing_sign=-1)
        sign_checks = {
            "support_error": compute_sign_agreement(
                d["support_error_m"], lr_fb, -1, "Support Error vs LR Feedback"
            ),
            "support_velocity": compute_sign_agreement(
                d["support_velocity_m_s"], lr_fb, -1, "Support Velocity vs LR Feedback"
            ),
            "pitch": compute_sign_agreement(
                d["pitch_x_rad"], lr_fb, -1, "Pitch vs LR Feedback"
            ),
            "pitch_rate": compute_sign_agreement(
                d["pitch_rate_rad_s"], lr_fb, -1, "Pitch Rate vs LR Feedback"
            ),
        }

        # Total torque sign checks
        tau_c = d["tau_common_nm"]
        sign_checks["support_error_vs_total"] = compute_sign_agreement(
            d["support_error_m"], tau_c, -1, "Support Error vs Total Torque"
        )
        sign_checks["pitch_vs_total"] = compute_sign_agreement(
            d["pitch_x_rad"], tau_c, -1, "Pitch vs Total Torque"
        )

    phase = compute_phase_analysis(d)
    drift_events = compute_drift_events(d)
    torque = compute_torque_magnitudes(d)
    recovery = compute_recovery_metrics(d)

    # Audited LR profile info
    audited = {
        "lr_profile": d["profile"],
        "lr_candidate_kind": d.get("lr_candidate_kind", "none"),
    }

    result = {
        "candidate_name": name,
        "lr_enabled": d["lr_enabled"],
        "lr_replacement_mode": d["lr_replacement_mode"],
        "audited_profile": audited,
        "sign_checks": sign_checks,
        "phase_analysis": phase,
        "drift_events": drift_events,
        "torque_magnitudes": torque,
        "recovery_metrics": recovery,
        "csv_path": str(csv_path),
    }

    return result


def _clean(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and math.isinf(obj):
        return "inf" if obj > 0 else "-inf"
    return obj


def print_summary(results: dict):
    """Print formatted summary table."""
    names = list(results.keys())
    print("\n" + "=" * 120)
    print("LR SUPPORT-DRIFT SIGN/PHASE AUDIT — SUMMARY")
    print("=" * 120)

    # Recovery metrics
    print(f"\n{'Metric':<40}", end="")
    for name in names:
        print(f" {name:>25}", end="")
    print()
    print("-" * 120)

    metrics = [
        ("completed_steps", "Completed steps", "d"),
        ("fall", "Fall?", "b"),
        ("termination_reason", "Termination reason", "s"),
        ("pitch_rms_deg", "Pitch RMS [deg]", ".2f"),
        ("final_pitch_rms_deg", "Final pitch RMS [deg]", ".2f"),
        ("pitch_max_deg", "Pitch max [deg]", ".1f"),
        ("support_rms_m", "Support RMS [m]", ".3f"),
        ("final_support_rms_m", "Final support RMS [m]", ".3f"),
        ("support_max_m", "Support max [m]", ".2f"),
        ("roll_rms_deg", "Roll RMS [deg]", ".2f"),
        ("roll_max_deg", "Roll max [deg]", ".1f"),
        ("min_height_m", "Min height [m]", ".3f"),
        ("sustained_2s_hold", "Sustained 2s?", "b"),
        ("sustained_5s_hold", "Sustained 5s?", "b"),
        ("hip_yaw_abs_max_rad", "Hip yaw abs max [rad]", ".3f"),
        ("hip_yaw_common_error_max_rad", "Hip yaw common max [rad]", ".3f"),
    ]

    for key, label, fmt in metrics:
        print(f"{label:<40}", end="")
        for name in names:
            r = results[name].get("recovery_metrics", {})
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                print(f" {'N/A':>25}", end="")
            elif fmt == "b":
                print(f" {'YES' if v else 'no':>25}", end="")
            elif fmt == "d":
                print(f" {int(v):>25}", end="")
            elif fmt == "s":
                s = str(v)[:25]
                print(f" {s:>25}", end="")
            else:
                print(f" {v:{fmt}}", end="")
        print()

    # Sign checks (LR only)
    print("\n--- SIGN AGREEMENT (LR candidates only) ---")
    lr_names = [n for n in names if results[n].get("lr_enabled")]
    if lr_names:
        print(f"{'Check':<40}", end="")
        for name in lr_names:
            print(f" {name:>25}", end="")
        print()

        sign_keys = [
            "support_error",
            "support_velocity",
            "pitch",
            "pitch_rate",
            "support_error_vs_total",
            "pitch_vs_total",
        ]
        for sk in sign_keys:
            print(f"{sk:<40}", end="")
            for name in lr_names:
                sc = results[name].get("sign_checks", {}).get(sk, {})
                corr = sc.get("correlation")
                sign_ok = sc.get("pass_")
                if corr is not None:
                    print(f" r={corr:+.3f} {'OK' if sign_ok else 'FLIP'}", end="")
                else:
                    print(f" {'N/A':>25}", end="")
            print()

    # Phase correlations
    print("\n--- POST-PUSH CORRELATIONS ---")
    corr_keys = [
        "pitch_vs_LR_feedback",
        "pitch_rate_vs_LR_feedback",
        "support_error_vs_LR_feedback",
        "support_velocity_vs_LR_feedback",
        "support_error_vs_total_torque",
        "pitch_vs_total_torque",
    ]
    print(f"{'Correlation':<40}", end="")
    for name in names:
        print(f" {name:>25}", end="")
    print()
    for ck in corr_keys:
        key_s = ck[:40]
        print(f"{key_s:<40}", end="")
        for name in names:
            c = results[name].get("phase_analysis", {}).get("post_push_correlations", {}).get(ck)
            if c is not None:
                print(f" {c:+.3f}", end="")
            else:
                print(f" {'N/A':>25}", end="")
        print()

    # FFT
    print("\n--- POST-PUSH FFT ---")
    fft_keys = [
        ("dominant_freq_hz", "Dominant freq [Hz]", ".2f"),
        ("dominant_amp_deg", "Dominant amp [deg]", ".2f"),
        ("amp_0p52_hz_deg", "0.52 Hz amp [deg]", ".3f"),
        ("amp_2p5_hz_deg", "2.5 Hz amp [deg]", ".3f"),
        ("low_band_0p35_0p65_energy_ratio", "0.35-0.65 Hz energy ratio", ".3f"),
    ]
    print(f"{'Metric':<40}", end="")
    for name in names:
        print(f" {name:>25}", end="")
    print()
    for key, label, fmt in fft_keys:
        print(f"{label:<40}", end="")
        for name in names:
            v = results[name].get("phase_analysis", {}).get("post_push_fft", {}).get(key)
            if v is not None:
                print(f" {v:{fmt}}", end="")
            else:
                print(f" {'N/A':>25}", end="")
        print()

    # Torque magnitudes
    print("\n--- TORQUE MAGNITUDES (RMS, Nm) ---")
    tq_keys = [
        "eq_ff_estimate_rms_nm",
        "lr_feedback_rms_nm",
        "lr_eq_ff_pass_through_rms_nm",
        "lr_total_preclip_rms_nm",
        "tau_common_final_rms_nm",
        "lr_saturation_rate_pct",
    ]
    print(f"{'Metric':<40}", end="")
    for name in names:
        print(f" {name:>25}", end="")
    print()
    for key in tq_keys:
        label = key.replace("_", " ")[:40]
        print(f"{label:<40}", end="")
        for name in names:
            v = results[name].get("torque_magnitudes", {}).get(key)
            if v is not None:
                print(f" {v:25.3f}" if isinstance(v, float) else f" {str(v):>25}", end="")
            else:
                print(f" {'N/A':>25}", end="")
        print()

    # Drift events
    print("\n--- SUPPORT-DRIFT THRESHOLD EVENTS ---")
    for name in lr_names:
        drift = results[name].get("drift_events", {})
        print(f"\n{name}:")
        for thresh_key, event in sorted(drift.items()):
            if event.get("crossed"):
                print(f"  {thresh_key}: step={event['first_step']}, "
                      f"support={event['support_error_m']:.3f}m, "
                      f"lr_fb={event['lr_feedback_nm']:.2f}Nm, "
                      f"lr_assisting={event['lr_assisting_drift']}")
            else:
                print(f"  {thresh_key}: NOT crossed")

    # Audit conclusions
    print("\n--- AUDIT CONCLUSIONS ---")
    for name in lr_names:
        sc = results[name].get("sign_checks", {})
        support_sign = sc.get("support_error", {})
        support_vel_sign = sc.get("support_velocity", {})
        pitch_rate_sign = sc.get("pitch_rate", {})
        pitch_sign = sc.get("pitch", {})

        print(f"\n{name}:")
        print(f"  Support feedback sign correct? "
              f"{'YES' if support_sign.get('pass_') else 'NO' if support_sign.get('pass_') is False else 'INCONCLUSIVE'}")
        print(f"  Support velocity damping sign correct? "
              f"{'YES' if support_vel_sign.get('pass_') else 'NO' if support_vel_sign.get('pass_') is False else 'INCONCLUSIVE'}")
        print(f"  Pitch-rate damping sign correct? "
              f"{'YES' if pitch_rate_sign.get('pass_') else 'NO' if pitch_rate_sign.get('pass_') is False else 'INCONCLUSIVE'}")
        print(f"  Pitch stiffness sign correct? "
              f"{'YES' if pitch_sign.get('pass_') else 'NO' if pitch_sign.get('pass_') is False else 'INCONCLUSIVE'}")

        # Determine main failure mode
        rec = results[name].get("recovery_metrics", {})
        if rec.get("support_rms_m", 999) > 0.5:
            failure = "SUPPORT_DRIFT_DOMINANT"
        elif rec.get("pitch_rms_deg", 999) > 10.0:
            failure = "PITCH_OSCILLATION_DOMINANT"
        elif rec.get("pitch_max_deg", 0) > 45.0:
            failure = "PITCH_DIVERGENCE"
        else:
            failure = "MIXED"
        print(f"  Main failure mode: {failure}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LR sign/phase/support-drift audit")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing candidate subdirectories with telemetry_*.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/lr_support_drift_audit",
                        help="Output directory for audit JSON")
    parser.add_argument("--candidates", type=str, nargs="*",
                        help="Specific candidate names (subdirectories); auto-discovered if omitted")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover candidates
    if args.candidates:
        candidate_dirs = {n: input_dir / n for n in args.candidates}
    else:
        candidate_dirs = {}
        for d in sorted(input_dir.iterdir()):
            if d.is_dir():
                candidate_dirs[d.name] = d

    results = {}
    for name, dirpath in candidate_dirs.items():
        csvs = list(dirpath.glob("telemetry_*.csv"))
        if not csvs:
            print(f"WARNING: No telemetry CSV found for {name}, skipping")
            continue
        print(f"Analyzing {name}...")
        try:
            results[name] = analyze_candidate(csvs[0], name)
        except Exception as e:
            print(f"ERROR analyzing {name}: {e}")
            results[name] = {"error": str(e)}

    # Save JSON
    json_path = output_dir / "audit_results.json"
    with open(json_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nAudit JSON saved to: {json_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
