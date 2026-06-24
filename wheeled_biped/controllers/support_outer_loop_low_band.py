"""Low-band support outer-loop shaping for the PFF candidate profile."""
from __future__ import annotations

import math
from typing import Dict


LOW_BAND_SUPPORT_PROFILE_NAME = "pff_low_band_support_v1"
LOW_BAND_SUPPORT_CENTER_M = 0.320
LOW_BAND_SUPPORT_SIGMA_M = 0.006
LOW_BAND_SUPPORT_KP_PEAK_DEG_PER_M = 1.5
LOW_BAND_SUPPORT_THETA_REF_MAX_PEAK_DEG = 3.00
LOW_BAND_SUPPORT_PITCH_REF_OFFSET_PEAK_DEG = 1.00

KP_BOUNDS = (0.00, 8.00)
KD_BOUNDS = (0.00, 0.50)
THETA_REF_MAX_BOUNDS_DEG = (0.50, 3.00)
PITCH_REF_OFFSET_BOUNDS_DEG = (-2.00, 2.00)


def _finite(value: float, fallback: float) -> float:
    return value if value == value and value not in (float("inf"), float("-inf")) else fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def low_band_support_height_scale(
    height_m: float,
    *,
    center_m: float = LOW_BAND_SUPPORT_CENTER_M,
    sigma_m: float = LOW_BAND_SUPPORT_SIGMA_M,
) -> float:
    """Smooth Gaussian weight centered on the low 0.320 m band."""
    h = _finite(float(height_m), center_m)
    sigma = max(float(sigma_m), 1e-6)
    u = (h - float(center_m)) / sigma
    return _clamp(math.exp(-0.5 * u * u), 0.0, 1.0)


def low_band_support_outer_loop_params(
    height_m: float,
    *,
    base_kp_deg_per_m: float,
    base_kd_deg_per_mps: float,
    base_theta_ref_max_deg: float,
    center_m: float = LOW_BAND_SUPPORT_CENTER_M,
    sigma_m: float = LOW_BAND_SUPPORT_SIGMA_M,
    peak_kp_deg_per_m: float = LOW_BAND_SUPPORT_KP_PEAK_DEG_PER_M,
    peak_theta_ref_max_deg: float = LOW_BAND_SUPPORT_THETA_REF_MAX_PEAK_DEG,
    peak_pitch_ref_offset_deg: float = LOW_BAND_SUPPORT_PITCH_REF_OFFSET_PEAK_DEG,
    blend_with_base: bool = False,
) -> Dict[str, float | str]:
    """Apply a bounded support correction only inside the low-height band.

    The blend is continuous in height and keyed only by commanded height. It
    intentionally leaves Kd unchanged; the local telemetry points to a low-band
    operating-point issue, not a support-rate overshoot.

    When ``blend_with_base=True``, the Kp is a smooth blend between the base Kp
    (calibrated outer loop value at the current height) and the peak Kp:
        kp = (1 - scale) * base_kp + scale * peak_kp
    This ensures the support correction is active at ALL heights — the base Kp
    provides centering feedback at tall heights while the peak Kp augments it in
    the low band. This is the correct behavior for tall-height push recovery.

    When ``blend_with_base=False`` (default, backward-compatible), the Kp is the
    scaling-only version:
        kp = scale * peak_kp
    which drops to zero at heights far from the low-band center. This preserves
    the original v1/v2 behavior for existing validations.
    """
    scale = low_band_support_height_scale(height_m, center_m=center_m, sigma_m=sigma_m)
    base_kp = _clamp(_finite(float(base_kp_deg_per_m), 1.0), *KP_BOUNDS)
    base_kd = _clamp(_finite(float(base_kd_deg_per_mps), 0.0), *KD_BOUNDS)
    base_theta = _clamp(_finite(float(base_theta_ref_max_deg), 3.0), *THETA_REF_MAX_BOUNDS_DEG)
    peak_kp = _clamp(_finite(float(peak_kp_deg_per_m), LOW_BAND_SUPPORT_KP_PEAK_DEG_PER_M), *KP_BOUNDS)
    peak_theta = _clamp(
        _finite(float(peak_theta_ref_max_deg), LOW_BAND_SUPPORT_THETA_REF_MAX_PEAK_DEG),
        *THETA_REF_MAX_BOUNDS_DEG,
    )
    peak_offset = _clamp(
        _finite(float(peak_pitch_ref_offset_deg), LOW_BAND_SUPPORT_PITCH_REF_OFFSET_PEAK_DEG),
        *PITCH_REF_OFFSET_BOUNDS_DEG,
    )
    if blend_with_base:
        # Blend: at scale=1 (center), kp ≈ peak_kp; at scale=0 (far), kp ≈ base_kp.
        kp = (1.0 - scale) * base_kp + scale * peak_kp
    else:
        # Legacy: kp is nonzero only near the low-band center.
        kp = scale * peak_kp
    theta = base_theta + scale * (peak_theta - base_theta)
    return {
        "support_outer_loop_profile_name": LOW_BAND_SUPPORT_PROFILE_NAME,
        "support_outer_loop_height_scale": float(scale),
        "support_outer_loop_kp_effective": float(_clamp(kp, *KP_BOUNDS)),
        "support_outer_loop_kd_effective": float(base_kd),
        "support_outer_loop_theta_ref_max_effective_deg": float(_clamp(theta, *THETA_REF_MAX_BOUNDS_DEG)),
        "support_outer_loop_pitch_ref_offset_deg": float(scale * peak_offset),
    }
