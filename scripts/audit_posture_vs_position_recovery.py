#!/usr/bin/env python3
"""
Phase 4: Posture vs Position Recovery Audit

Separates and analyzes post-push recovery into:
- Pitch recovery
- Roll recovery
- Yaw recovery
- Support recovery
- Position recovery

Key questions:
- Did K1 recover posture?
- Did K1 recover position?
- Did K1 recover support?
- Which one actually fails?
"""

import csv
import json
import statistics
import sys
import os
import math
import numpy as np

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
TELEMETRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "d_baseline_single_90n_10step_push_step300_3000",
    "telemetry_1782262602.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "system_audit", "posture_vs_position",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "posture_vs_position_audit.json")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "posture_vs_position_report.md")

# ── Load telemetry ─────────────────────────────────────────────────────────
print("Loading K1 telemetry...")
with open(TELEMETRY_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

N = len(rows)
dt = float(rows[1]["sim_time_s"]) - float(rows[0]["sim_time_s"]) if N > 1 else 0.005

print(f"  {N} steps, dt={dt:.4f}s")

# ── Extract signals ────────────────────────────────────────────────────────
pitch_x = np.array([float(r["pitch_x"]) for r in rows])
pitch_rate = np.array([float(r["pitch_rate_x"]) for r in rows])
roll_y = np.array([float(r["roll_y"]) for r in rows])
roll_rate = np.array([float(r["roll_rate_y"]) for r in rows])
yaw_z = np.array([float(r.get("yaw_z", 0)) for r in rows])
yaw_rate = np.array([float(r.get("yaw_rate_z_rad_s", 0)) for r in rows])
com_z = np.array([float(r["com_z"]) for r in rows])
com_y = np.array([float(r["com_y"]) for r in rows])
com_x = np.array([float(r["com_x"]) for r in rows])
com_vy = np.array([float(r.get("com_vy", 0)) for r in rows])
support_error = np.array([float(r.get("support_position_error_m", 0)) for r in rows])
com_error_x = np.array([float(r.get("com_error_x", 0)) for r in rows])
cp_error_y = np.array([float(r.get("cp_error_y", 0)) for r in rows])
def _safe_float(val, default=0.0):
    """Handle boolean string values like 'True'/'False' in telemetry."""
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

push_active = np.array([_safe_float(r.get("push_active", 0)) for r in rows])
terminated = np.array([_safe_float(r.get("terminated", 0)) for r in rows])
termination_reason = [r.get("termination_reason", "") for r in rows]
height_error = np.array([float(r.get("height_error", 0)) for r in rows])

# Wheel torque
tau_left = np.array([float(r["tau_left"]) for r in rows])
tau_right = np.array([float(r["tau_right"]) for r in rows])
tau_common = tau_left + tau_right

# ── Find push and termination ─────────────────────────────────────────────
push_steps = np.where(push_active > 0.5)[0]
push_start = int(push_steps[0]) if len(push_steps) > 0 else 300
push_end = int(push_steps[-1]) if len(push_steps) > 0 else 310

# Termination info
term_steps = np.where(terminated > 0.5)[0]
term_step = int(term_steps[0]) if len(term_steps) > 0 else N - 1
term_reason = termination_reason[term_step] if term_step < len(termination_reason) else "none"
if not term_reason or term_reason == "False" or term_reason == "0":
    term_reason = "completed_3000_steps"

# Actually, the run completed 3000 steps successfully
print(f"  Push: step {push_start} to {push_end}")
print(f"  Termination: step {term_step}, reason: {term_reason}")

# ── Define phases ──────────────────────────────────────────────────────────
PRE_PUSH = (0, push_start)
DURING_PUSH = (push_start, push_end + 1)
POST_PUSH_EARLY = (push_end + 1, min(push_end + 101, N))   # 100 steps after push
POST_PUSH_RECOVERY = (push_end + 1, min(push_end + 501, N)) # 500 steps recovery window
POST_PUSH_LATE = (push_end + 501, N)                         # remainder

# ── Recovery metrics ───────────────────────────────────────────────────────
def compute_recovery_metrics(signal, time_axis, push_end_idx, label, threshold=None, settling_frac=0.1):
    """
    Compute recovery time: how long after push ends until signal returns to
    within settling_frac of pre-push range (or below threshold).
    """
    # Pre-push baseline
    pre_mean = float(np.mean(signal[:push_end_idx]))
    pre_std = float(np.std(signal[:push_end_idx]))
    pre_max = float(np.max(np.abs(signal[:push_end_idx])))

    # Post-push signal
    post_signal = signal[push_end_idx + 1:]
    post_time = time_axis[push_end_idx + 1:] - time_axis[push_end_idx + 1]

    if len(post_signal) == 0:
        return {
            "label": label,
            "pre_mean": pre_mean,
            "pre_std": pre_std,
            "pre_max_abs": pre_max,
            "max_excursion": 0.0,
            "max_excursion_step": 0,
            "recovered": True,
            "recovery_steps": 0,
            "recovery_time_s": 0.0,
            "final_error": 0.0,
            "settled": True,
        }

    # Max excursion
    abs_post = np.abs(post_signal - pre_mean)
    max_exc_idx = np.argmax(abs_post)
    max_exc = float(abs_post[max_exc_idx])

    # Recovery: when |signal - pre_mean| < max(settling_frac * pre_max_abs, threshold)
    settle_band = max(settling_frac * pre_max, threshold if threshold else 0.01)

    recovered_idx = None
    for i in range(len(post_signal)):
        if abs(float(post_signal[i]) - pre_mean) < settle_band:
            recovered_idx = i
            break

    if recovered_idx is not None:
        # Check if it STAYS settled (no re-excursion)
        settled = True
        for j in range(recovered_idx + 1, min(recovered_idx + 50, len(post_signal))):
            if abs(float(post_signal[j]) - pre_mean) > settle_band * 3:
                settled = False
                break

        return {
            "label": label,
            "pre_mean": pre_mean,
            "pre_std": pre_std,
            "pre_max_abs": pre_max,
            "settle_band": settle_band,
            "max_excursion": max_exc,
            "max_excursion_at_step": int(max_exc_idx) + push_end_idx + 1,
            "recovered": recovered_idx is not None,
            "recovery_steps": int(recovered_idx) if recovered_idx is not None else 0,
            "recovery_time_s": float(recovered_idx * dt) if recovered_idx is not None else 0.0,
            "final_error": float(post_signal[-1] - pre_mean),
            "settled": settled,
        }
    else:
        return {
            "label": label,
            "pre_mean": pre_mean,
            "pre_std": pre_std,
            "pre_max_abs": pre_max,
            "settle_band": settle_band,
            "max_excursion": max_exc,
            "max_excursion_at_step": int(max_exc_idx) + push_end_idx + 1,
            "recovered": False,
            "recovery_steps": len(post_signal),
            "recovery_time_s": float(len(post_signal) * dt),
            "final_error": float(post_signal[-1] - pre_mean),
            "settled": False,
        }

# ── Compute recovery for each dimension ────────────────────────────────────
print("\n── Recovery Analysis ──")

time_axis = np.arange(N) * dt

signals = {
    "pitch": (pitch_x, 0.01, math.radians(2.0)),  # signal, threshold, settling frac * pre_max
    "roll": (roll_y, 0.005, math.radians(1.0)),
    "yaw": (yaw_z, 0.01, math.radians(3.0)),
    "support_error": (support_error, 0.02, 0.05),
    "height": (com_z, 0.02, 0.03),
    "com_y": (com_y, 0.02, 0.05),
}

recovery_metrics = {}
for name, (sig, threshold, sett_frac) in signals.items():
    rm = compute_recovery_metrics(sig, time_axis, push_end, name, threshold=threshold)
    recovery_metrics[name] = rm
    deg_str = "deg" if name in ("pitch", "roll", "yaw") else "m"
    multiplier = 180.0 / math.pi if name in ("pitch", "roll", "yaw") else 1.0
    print(f"  {name}: pre_mean={rm['pre_mean']*multiplier:.4f}{deg_str}, "
          f"max_exc={rm['max_excursion']*multiplier:.3f}{deg_str}, "
          f"recovered={rm['recovered']}, "
          f"recovery_time={rm['recovery_time_s']:.2f}s, "
          f"final_error={rm['final_error']*multiplier:.3f}{deg_str}, "
          f"settled={rm['settled']}")

# ── Which dimensions recover and which don't? ──────────────────────────────
recovered_dims = [name for name, rm in recovery_metrics.items() if rm["recovered"] and rm["settled"]]
unrecovered_dims = [name for name, rm in recovery_metrics.items() if not rm["recovered"] or not rm["settled"]]
partially_recovered = [name for name, rm in recovery_metrics.items() if rm["recovered"] and not rm["settled"]]

print(f"\n  Fully recovered: {recovered_dims}")
print(f"  Partially recovered: {partially_recovered}")
print(f"  Not recovered: {unrecovered_dims}")

# ── Detailed pitch recovery ────────────────────────────────────────────────
print("\n── Pitch Recovery Detail ──")
# Look at specific pitch behavior in recovery windows
pp_early_pitch = np.abs(pitch_x[push_end + 1:push_end + 101])
pp_recovery_pitch = np.abs(pitch_x[push_end + 1:push_end + 501])
pp_late_pitch = np.abs(pitch_x[push_end + 501:])

print(f"  Pre-push pitch RMS: {np.sqrt(np.mean(pitch_x[:push_start]**2))*180/math.pi:.3f} deg")
print(f"  During-push pitch max: {np.max(np.abs(pitch_x[push_start:push_end+1]))*180/math.pi:.3f} deg")
print(f"  Post-push early (100 steps) pitch RMS: {np.sqrt(np.mean(pp_early_pitch**2))*180/math.pi:.3f} deg")
print(f"  Post-push recovery (500 steps) pitch RMS: {np.sqrt(np.mean(pp_recovery_pitch**2))*180/math.pi:.3f} deg")
if len(pp_late_pitch) > 0:
    print(f"  Post-push late pitch RMS: {np.sqrt(np.mean(pp_late_pitch**2))*180/math.pi:.3f} deg")

# ── Detailed support recovery ──────────────────────────────────────────────
print("\n── Support Recovery Detail ──")
pp_early_sup = np.abs(support_error[push_end + 1:push_end + 101])
pp_recovery_sup = np.abs(support_error[push_end + 1:push_end + 501])
pp_late_sup = np.abs(support_error[push_end + 501:])

print(f"  Pre-push support RMS: {np.sqrt(np.mean(support_error[:push_start]**2)):.4f} m")
print(f"  During-push support max: {np.max(np.abs(support_error[push_start:push_end+1])):.4f} m")
print(f"  Post-push early (100 steps) support RMS: {np.sqrt(np.mean(pp_early_sup**2)):.4f} m")
print(f"  Post-push recovery (500 steps) support RMS: {np.sqrt(np.mean(pp_recovery_sup**2)):.4f} m")
if len(pp_late_sup) > 0:
    print(f"  Post-push late support RMS: {np.sqrt(np.mean(pp_late_sup**2)):.4f} m")

# ── Recovery scorecard ─────────────────────────────────────────────────────
# For each dimension: did it recover within 100 steps? 500 steps? 2000 steps?
windows = {
    "100_steps": (push_end + 1, push_end + 101),
    "500_steps": (push_end + 1, push_end + 501),
    "2000_steps": (push_end + 1, push_end + 2001),
}

scorecard = {}
for name, (sig, threshold, _) in signals.items():
    pre_mean = float(np.mean(sig[:push_start]))
    pre_std = float(np.std(sig[:push_start]))
    settle_band = max(threshold, 3 * pre_std)
    results_w = {}
    for wname, (wstart, wend) in windows.items():
        wend = min(wend, N)
        if wstart >= N:
            results_w[wname] = False
            continue
        wsignal = sig[wstart:wend]
        if len(wsignal) == 0:
            results_w[wname] = False
            continue
        # Check if last 20 steps are within settle_band
        check_len = min(20, len(wsignal))
        last_segment = wsignal[-check_len:]
        recovered_w = all(abs(float(s) - pre_mean) < settle_band for s in last_segment)
        results_w[wname] = recovered_w

    scorecard[name] = results_w

print("\n── Recovery Scorecard ──")
print(f"{'Dimension':<20} {'100 steps':<12} {'500 steps':<12} {'2000 steps':<12}")
print("-" * 56)
for name, results_w in scorecard.items():
    print(f"{name:<20} {'YES' if results_w['100_steps'] else 'NO':<12} {'YES' if results_w['500_steps'] else 'NO':<12} {'YES' if results_w['2000_steps'] else 'NO':<12}")

# ── Which dimension actually fails? ────────────────────────────────────────
# Only count as "failure" if it doesn't recover even after 2000 steps
failing_dims = [name for name, results_w in scorecard.items() if not results_w["2000_steps"]]
all_recovered = len(failing_dims) == 0

print(f"\n  Dimensions NOT recovered after 2000 steps: {failing_dims if failing_dims else 'NONE'}")

# ── Position vs posture analysis ───────────────────────────────────────────
# "Posture" = orientation (pitch, roll, yaw)
# "Position" = COM position, support centering, height
posture_recovered = all(scorecard.get(d, {}).get("500_steps", False) for d in ["pitch", "roll", "yaw"])
position_recovered = all(scorecard.get(d, {}).get("500_steps", False) for d in ["support_error", "com_y", "height"])

print(f"\n── Posture vs Position ──")
print(f"  Posture recovered (500 steps): {posture_recovered}")
print(f"  Position recovered (500 steps): {position_recovered}")

# ── Which fails? ───────────────────────────────────────────────────────────
if all_recovered:
    primary_failure = "NONE_ALL_RECOVERED"
elif not posture_recovered and not position_recovered:
    primary_failure = "BOTH_POSTURE_AND_POSITION"
elif not posture_recovered:
    primary_failure = "POSTURE"
else:
    primary_failure = "POSITION"

# ── Specific failure mechanism ─────────────────────────────────────────────
# Is it:
# - pitch oscillation that never damps out?
# - support drift that never recenters?
# - height loss that accumulates?

# Compute recovery completeness
def recovery_completeness(sig, pre_mean, post_start, window_size=500):
    """How close does the signal get to pre-push mean by the end of window?"""
    end = min(post_start + window_size, len(sig))
    segment = sig[post_start:end]
    if len(segment) == 0:
        return float("inf")
    initial_error = abs(float(sig[post_start]) - pre_mean)
    final_error = abs(float(np.mean(segment[-50:])) - pre_mean) if len(segment) >= 50 else abs(float(segment[-1]) - pre_mean)
    if initial_error < 1e-9:
        return 0.0
    return final_error / initial_error

pitch_pre_mean = float(np.mean(pitch_x[:push_start]))
sup_pre_mean = float(np.mean(support_error[:push_start]))

pitch_completeness = recovery_completeness(pitch_x, pitch_pre_mean, push_end + 1, 500)
sup_completeness = recovery_completeness(support_error, sup_pre_mean, push_end + 1, 500)

print(f"\n── Recovery Completeness (500 steps post-push) ──")
print(f"  Pitch: {pitch_completeness:.3f} (0=fully recovered, 1=no recovery, >1=worse)")
print(f"  Support: {sup_completeness:.3f} (0=fully recovered, 1=no recovery, >1=worse)")

# ── Accumulation analysis ──────────────────────────────────────────────────
# Does the error GROW over time or oscillate around a constant offset?
# If it grows => accumulating failure (e.g., integral drift)
# If it oscillates => underdamped mode

# Fit linear trend
pp_range = np.arange(push_end + 500, N) if N > push_end + 500 else np.arange(push_end + 1, N)
if len(pp_range) > 10:
    A = np.vstack([pp_range - pp_range[0], np.ones_like(pp_range)]).T
    slope_pitch_late, _ = np.linalg.lstsq(A, np.abs(pitch_x[pp_range]), rcond=None)[0]
    slope_sup_late, _ = np.linalg.lstsq(A, np.abs(support_error[pp_range]), rcond=None)[0]

    pitch_trend = "GROWING" if slope_pitch_late > 1e-6 else "STABLE_OR_DECAYING"
    sup_trend = "GROWING" if slope_sup_late > 1e-6 else "STABLE_OR_DECAYING"

    print(f"\n── Late-Phase Error Trends ──")
    print(f"  Pitch trend: {pitch_trend} (slope = {slope_pitch_late*180/math.pi:.6f} deg/step)")
    print(f"  Support trend: {sup_trend} (slope = {slope_sup_late:.6f} m/step)")
else:
    pitch_trend = "INSUFFICIENT_DATA"
    sup_trend = "INSUFFICIENT_DATA"

# ── Final answer ───────────────────────────────────────────────────────────
if all_recovered:
    which_fails = "NO_DIMENSION_FAILS_AT_3000_STEPS"
elif pitch_trend == "GROWING" and sup_trend == "GROWING":
    which_fails = "BOTH_PITCH_AND_SUPPORT_ACCUMULATE"
elif pitch_trend == "GROWING":
    which_fails = "PITCH_OSCILLATION_ACCUMULATES"
elif sup_trend == "GROWING":
    which_fails = "SUPPORT_DRIFT_ACCUMULATES"
elif not posture_recovered and not position_recovered:
    which_fails = "BOTH_POSTURE_AND_POSITION_FAIL_TO_SETTLE"
elif not position_recovered:
    which_fails = "POSITION_FAILS_TO_SETTLE_POSTURE_RECOVERS"
elif not posture_recovered:
    which_fails = "POSTURE_FAILS_TO_SETTLE_POSITION_RECOVERS"
else:
    which_fails = "ALL_DIMENSIONS_EVENTUALLY_RECOVER"

print(f"\n── WHICH ACTUALLY FAILS? ──")
print(f"  {which_fails}")

# ── Compile results ────────────────────────────────────────────────────────
def convert_to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj

audit = convert_to_native({
    "audit": "posture_vs_position_recovery",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "push_window": {"start": push_start, "end": push_end},
    "termination": {"step": term_step, "reason": term_reason},
    "dt_s": dt,
    "recovery_metrics": recovery_metrics,
    "recovery_scorecard": scorecard,
    "summary": {
        "fully_recovered": recovered_dims,
        "partially_recovered": partially_recovered,
        "not_recovered": unrecovered_dims,
        "posture_recovered_500steps": posture_recovered,
        "position_recovered_500steps": position_recovered,
        "failing_dimensions_after_2000": failing_dims,
    },
    "completeness_500steps": {
        "pitch": float(pitch_completeness),
        "support": float(sup_completeness),
    },
    "late_phase_trends": {
        "pitch": pitch_trend,
        "support": sup_trend,
    },
    "verdict": which_fails,
})

# ── Save JSON ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"\nJSON audit saved to: {OUTPUT_JSON}")

# ── Print summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("POSTURE VS POSITION RECOVERY AUDIT — SUMMARY")
print("=" * 70)

print(f"\n── Recovery ──")
for name, rm in recovery_metrics.items():
    mul = 180.0 / math.pi if name in ("pitch", "roll", "yaw") else 1.0
    unit = "deg" if name in ("pitch", "roll", "yaw") else "m"
    print(f"  {name}: max_exc={rm['max_excursion']*mul:.3f}{unit}, "
          f"recovery={rm['recovery_time_s']:.2f}s, "
          f"settled={'YES' if rm['settled'] else 'NO'}")

print(f"\n── Scorecard ──")
for name, results_w in scorecard.items():
    print(f"  {name}: 100s={'Y' if results_w['100_steps'] else 'N'} "
          f"500s={'Y' if results_w['500_steps'] else 'N'} "
          f"2000s={'Y' if results_w['2000_steps'] else 'N'}")

print(f"\n── Posture vs Position ──")
print(f"  Posture recovered: {posture_recovered}")
print(f"  Position recovered: {position_recovered}")

print(f"\n── Recovery Completeness (lower=better) ──")
print(f"  Pitch: {pitch_completeness:.3f}")
print(f"  Support: {sup_completeness:.3f}")

print(f"\n── Late-Phase Trends ──")
print(f"  Pitch: {pitch_trend}")
print(f"  Support: {sup_trend}")

print(f"\n── WHICH ACTUALLY FAILS? ──")
print(f"  {which_fails}")

# ── Write report ──────────────────────────────────────────────────────────
report_lines = [
    "# Posture vs Position Recovery Audit",
    "",
    f"**Verdict:** `{which_fails}`",
    "",
    "## Setup",
    "",
    f"- **Controller:** K1_PITCH_RATE_NOTCH_V1",
    f"- **Scenario:** high_0p480, 90N sagittal push at step {push_start}, 10-step duration",
    f"- **Termination:** {term_reason} at step {term_step}",
    f"- **DT:** {dt:.4f} s",
    "",
    "## Recovery by Dimension",
    "",
    "| Dimension | Pre-Mean | Max Excursion | Recovery Time (s) | Settled? | Final Error |",
    "|-----------|----------|---------------|-------------------|----------|-------------|",
]

for name, rm in recovery_metrics.items():
    mul = 180.0 / math.pi if name in ("pitch", "roll", "yaw") else 1.0
    unit = "deg" if name in ("pitch", "roll", "yaw") else "m"
    row = f"| {name} | {rm['pre_mean']*mul:.3f}{unit} |"
    row += f" {rm['max_excursion']*mul:.3f}{unit} |"
    row += f" {rm['recovery_time_s']:.2f} |"
    row += f" {'YES' if rm['settled'] else 'NO'} |"
    row += f" {rm['final_error']*mul:.3f}{unit} |"
    report_lines.append(row)

report_lines += [
    "",
    "## Recovery Scorecard",
    "",
    "| Dimension | 100 steps | 500 steps | 2000 steps |",
    "|-----------|-----------|-----------|------------|",
]

for name, results_w in scorecard.items():
    report_lines.append(
        f"| {name} | {'YES' if results_w['100_steps'] else 'NO'} |"
        f" {'YES' if results_w['500_steps'] else 'NO'} |"
        f" {'YES' if results_w['2000_steps'] else 'NO'} |"
    )

report_lines += [
    "",
    "## Posture vs Position",
    "",
    f"| Category | Recovered (500 steps)? |",
    f"|----------|----------------------|",
    f"| Posture (pitch, roll, yaw) | {'YES' if posture_recovered else 'NO'} |",
    f"| Position (support, COM, height) | {'YES' if position_recovered else 'NO'} |",
    "",
    "## Recovery Completeness (500 steps post-push)",
    "",
    "Completeness = final_error / initial_error. 0 = fully recovered, >1 = worse than initial.",
    "",
    f"| Dimension | Completeness |",
    f"|-----------|-------------|",
    f"| Pitch | {pitch_completeness:.3f} |",
    f"| Support | {sup_completeness:.3f} |",
    "",
    "## Late-Phase Error Trends",
    "",
    f"| Dimension | Trend |",
    f"|-----------|-------|",
    f"| Pitch | {pitch_trend} |",
    f"| Support | {sup_trend} |",
    "",
    "## Diagnosis",
    "",
    "### Did K1 recover posture?",
    "",
    f"**{'YES' if posture_recovered else 'PARTIALLY' if len(recovered_dims) >= 1 else 'NO'}**",
    "",
    "### Did K1 recover position?",
    "",
    f"**{'YES' if position_recovered else 'PARTIALLY' if sup_completeness < 1.0 else 'NO'}**",
    "",
    "### Did K1 recover support?",
    "",
    f"**{'YES' if recovery_metrics.get('support_error', {}).get('recovered') else 'NO'}**",
    "",
    "### Which one actually fails?",
    "",
    f"**{which_fails}**",
    "",
]

if "SUPPORT_DRIFT" in which_fails:
    report_lines += [
        "Support drift is the primary failure mechanism. The pitch/posture recovery is adequate",
        "(or at least non-accumulating), but the COM drifts away from the support center over time.",
        "This is consistent with the height_too_low termination: as support drifts, the robot",
        "squats to compensate, eventually violating the height limit.",
        "",
        "**Implication:** The architecture needs more effective support centering, either through",
        "higher position gain, integral action, or a dedicated drift-correction term.",
    ]
elif "PITCH" in which_fails:
    report_lines += [
        "Pitch oscillation is the primary failure mechanism. The posture never fully settles",
        "after the push, and the oscillation grows or persists at constant amplitude.",
        "Support centering may be adequate but pitch instability drives the failure cascade.",
        "",
        "**Implication:** The architecture needs more effective pitch damping, potentially",
        "through higher derivative gain, phase-lead compensation, or adaptive damping.",
    ]
elif "BOTH" in which_fails:
    report_lines += [
        "Both posture and position fail to recover. The push excites coupled pitch-support",
        "dynamics that neither the pitch damping nor the position centering terms can fully suppress.",
        "",
        "**Implication:** The architecture needs improvement in BOTH pitch damping and",
        "support centering. These may need to be addressed simultaneously due to coupling.",
    ]
else:
    report_lines += [
        "All dimensions eventually recover. The system is stable and the controller",
        "successfully damps the perturbation. The limiting factor for even better",
        "performance may be the fundamental physics (inertia, damping, geometry)",
        "rather than the controller architecture.",
    ]

report_lines += [
    "",
    f"**Classification:** `{which_fails}`",
]

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report saved to: {OUTPUT_REPORT}")
print("\nDone.")
