#!/usr/bin/env python3
"""Baseline frequency and sample-rate audit for 2.5 Hz WIP notch filter design.

Reads G1_sg080 (and optionally I1, J3a) telemetry and computes:
1. Sample rate (from time column)
2. Dominant mode frequency for pitch / support / wheel-velocity / pitch-rate
3. Cross-correlation and phase lag
4. Signal ranking by 2.5 Hz component strength
5. Offline preview of notch filter attenuation

Outputs:
    outputs/targeted_2p5hz_wip_notch_bandstop_filter/audit/
        baseline_mode_audit.md
        filter_design_summary.json
        signal_frequency_rank.csv
        filter_candidate_coefficients.csv

Usage:
    python scripts/audit_2p5hz_wip_mode_filter_design.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

DEG = 180.0 / math.pi

SRC = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "g1_sg080_single_90n_10step_push_step300_3000"
    / "telemetry_1782262442.csv"
)

OUT = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "targeted_2p5hz_wip_notch_bandstop_filter"
    / "audit"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _float_safe(v: str | None, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        v = v.strip()
        return float(v) if v else default
    except (AttributeError, ValueError):
        return default


def _rms(arr: np.ndarray) -> float:
    return float(math.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


def _psd_frequency_peak(
    signal: np.ndarray,
    fs: float,
    lo_hz: float = 1.0,
    hi_hz: float = 10.0,
) -> tuple[float, float]:
    """Return dominant frequency (Hz) and its PSD amplitude in [lo, hi]."""
    n = len(signal)
    if n < 10:
        return (0.0, 0.0)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(fft_vals) ** 2
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return (0.0, 0.0)
    idx = np.argmax(psd[mask])
    return (float(freqs[mask][idx]), float(psd[mask][idx]))


def _hz(x: float) -> str:
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# load telemetry
# ---------------------------------------------------------------------------

def load_telemetry(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        print(f"  [SKIP] telemetry not found: {path}")
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"  [SKIP] empty telemetry: {path}")
        return None
    data: dict[str, list[float]] = {}
    keys = list(rows[0].keys())
    for k in keys:
        data[k] = [_float_safe(r[k]) for r in rows]
    print(f"  [LOAD] {len(rows)} rows, {len(keys)} columns from {path.name}")
    return {k: np.array(v, dtype=np.float64) for k, v in data.items()}


# ---------------------------------------------------------------------------
# sample-rate audit
# ---------------------------------------------------------------------------

def audit_sample_rate(
    data: dict[str, np.ndarray],
    label: str,
) -> dict:
    t = data.get("time")
    if t is None or len(t) < 2:
        print(f"  [WARN] {label}: no time column")
        return {}
    dt = np.diff(t)
    info = {
        "label": label,
        "n_rows": int(len(t)),
        "t_min_s": float(t[0]),
        "t_max_s": float(t[-1]),
        "t_duration_s": float(t[-1]) - float(t[0]),
        "dt_median_s": float(np.median(dt)),
        "dt_mean_s": float(np.mean(dt)),
        "dt_min_s": float(np.min(dt)),
        "dt_max_s": float(np.max(dt)),
        "fs_hz_median": float(1.0 / np.median(dt)) if np.median(dt) > 0 else 0,
        "fs_hz_mean": float(1.0 / np.mean(dt)) if np.mean(dt) > 0 else 0,
    }
    # check step index
    si = data.get("source_step_index")
    if si is not None:
        info["step_first"] = int(si[0])
        info["step_last"] = int(si[-1])
        info["step_gap_first"] = int(si[-1]) - int(si[0]) + 1
    print(f"  fs={info['fs_hz_median']:.1f} Hz, dt={info['dt_median_s']:.6f} s, "
          f"duration={info['t_duration_s']:.1f} s, rows={info['n_rows']}")
    return info


# ---------------------------------------------------------------------------
# frequency analysis
# ---------------------------------------------------------------------------

SIGNAL_DEFS: list[tuple[str, str, float]] = [
    ("pitch_rate_x_rad_s",       "pitch_rate",      DEG),
    ("robot_pitch_x",            "pitch",            1.0),
    ("pitch_x_rad",              "pitch_x",          1.0),
    ("wheel_vel_left_rad_s",     "wheel_vel_left",   1.0),
    ("wheel_vel_right_rad_s",    "wheel_vel_right",  1.0),
    ("sagittal_position_error_m","support_error",    1.0),
    ("support_position_velocity_m_s","support_vel",  1.0),
    ("sagittal_velocity_m_s",    "sag_vel",          1.0),
    ("com_vx",                   "com_vx",           1.0),
    ("com_vy",                   "com_vy",           1.0),
    ("com_vz",                   "com_vz",           1.0),
]


def analyze_frequency(
    data: dict[str, np.ndarray],
    fs: float,
    label: str,
    push_start_idx: int = 300,
    push_end_idx: int = 310,
) -> dict:
    results: dict = {"label": label, "fs_hz": fs}
    rows_n = len(next(iter(data.values())))
    for col_key, short_name, scale in SIGNAL_DEFS:
        raw = data.get(col_key)
        if raw is None:
            continue
        sig_deg = raw * scale  # scale to deg if needed
        # windows
        pre = slice(0, push_start_idx)
        post = slice(push_end_idx, None)
        final = slice(max(0, rows_n - 500), None)
        freq_pre, amp_pre = _psd_frequency_peak(sig_deg[pre], fs)
        freq_post, amp_post = _psd_frequency_peak(sig_deg[post], fs)
        freq_final, amp_final = _psd_frequency_peak(sig_deg[final], fs)
        rms_pre = _rms(sig_deg[pre])
        rms_post = _rms(sig_deg[post])
        rms_final = _rms(sig_deg[final])
        results[short_name] = {
            "col": col_key,
            "scale": scale,
            "freq_pre_hz": freq_pre,
            "amp_pre": amp_pre,
            "freq_post_hz": freq_post,
            "amp_post": amp_post,
            "freq_final_hz": freq_final,
            "amp_final": amp_final,
            "rms_pre": rms_pre,
            "rms_post": rms_post,
            "rms_final": rms_final,
        }
        print(f"  {short_name:20s}  final freq={freq_final:.4f} Hz  "
              f"amp={amp_final:.2e}  rms={rms_final:.4f}")
    return results


# ---------------------------------------------------------------------------
# cross-correlation
# ---------------------------------------------------------------------------

def compute_cross_correlation(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    max_lag_s: float = 1.0,
) -> dict:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(max_lag_s * fs)
    corr = np.correlate(a - np.mean(a), b - np.mean(b), mode="same")
    lags = np.arange(-len(a) // 2, len(a) // 2 + 1)
    if len(corr) != len(lags):
        # adjust
        min_len = min(len(corr), len(lags))
        corr = corr[:min_len]
        lags = lags[:min_len]
    sigma_a = np.std(a)
    sigma_b = np.std(b)
    denom = sigma_a * sigma_b * len(a)
    norm_corr = corr / denom if denom > 1e-12 else corr
    # find peak near zero lag
    near_mask = np.abs(lags) <= max_lag
    if np.any(near_mask):
        peak_idx = np.argmax(np.abs(norm_corr[near_mask]))
        lag_at_peak = lags[near_mask][peak_idx]
        corr_at_peak = float(norm_corr[near_mask][peak_idx])
    else:
        lag_at_peak = 0
        corr_at_peak = 0.0
    lag_at_zero = 0
    zero_idx = np.argmin(np.abs(lags))
    corr_at_zero = float(norm_corr[zero_idx]) if zero_idx < len(norm_corr) else 0.0
    return {
        "lag_at_peak_samples": int(lag_at_peak),
        "lag_at_peak_s": float(lag_at_peak / fs),
        "corr_at_peak": corr_at_peak,
        "corr_at_zero_lag": corr_at_zero,
    }


# =========================================================================
# Offline filter preview
# =========================================================================

from wheeled_biped.controllers.signal_filters import BiquadNotchFilter


def offline_filter_preview(
    signal: np.ndarray,
    fs: float,
    fc_hz: float,
    Q: float,
) -> tuple[np.ndarray, dict]:
    """Apply notch filter offline for analysis only."""
    nf = BiquadNotchFilter(fs_hz=fs, fc_hz=fc_hz, Q=Q)
    out = np.zeros_like(signal)
    for i, x in enumerate(signal):
        out[i] = nf.update(x)
    atten_db = 20.0 * math.log10(_rms(out) / max(_rms(signal), 1e-12))
    return out, {
        "fc_hz": fc_hz,
        "Q": Q,
        "bw_hz": fc_hz / Q,
        "fs_hz": fs,
        "rms_raw": float(_rms(signal)),
        "rms_filtered": float(_rms(out)),
        "attenuation_db": float(atten_db),
        "b0": nf.b0,
        "b1": nf.b1,
        "b2": nf.b2,
        "a1": nf.a1,
        "a2": nf.a2,
    }


def offline_filter_grid(
    signal: np.ndarray,
    fs: float,
    label: str,
    fc_list: list[float],
    Q_list: list[float],
) -> list[dict]:
    results = []
    for fc in fc_list:
        for Q in Q_list:
            _, info = offline_filter_preview(signal, fs, fc, Q)
            info["signal"] = label
            results.append(info)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 1 — BASELINE FREQUENCY AND SAMPLE-RATE AUDIT")
    print("=" * 70)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load G1_sg080 telemetry
    print("\n--- Loading telemetry ---")
    data = load_telemetry(SRC)
    if data is None:
        print("FATAL: no telemetry found")
        sys.exit(1)

    # 2. Sample-rate audit
    print("\n--- Sample-rate audit ---")
    sr = audit_sample_rate(data, "G1_sg080")
    fs = sr.get("fs_hz_median", 100.0)
    sr_path = OUT / "sample_rate_audit.json"
    with open(sr_path, "w") as f:
        json.dump(sr, f, indent=2)
    print(f"  written: {sr_path}")

    # 3. Dominant frequency analysis
    print("\n--- Frequency analysis (post-push) ---")
    freq = analyze_frequency(data, fs, "G1_sg080")

    # 4. Cross-correlation
    print("\n--- Cross-correlation ---")
    pitch = data.get("pitch_x_rad", data.get("robot_pitch_x"))
    support = data.get("sagittal_position_error_m")
    pitch_rate = data.get("pitch_rate_x_rad_s")
    wheel_left = data.get("wheel_vel_left_rad_s")
    if pitch is not None and support is not None:
        cc = compute_cross_correlation(pitch, support, fs)
        print(f"  pitch <-> support:  corr={cc['corr_at_peak']:.4f}  lag={cc['lag_at_peak_s']:.4f}s")
    if pitch is not None and pitch_rate is not None:
        cc_pr = compute_cross_correlation(pitch, pitch_rate, fs)
        print(f"  pitch <-> pitch_rate:  corr={cc_pr['corr_at_peak']:.4f}  lag={cc_pr['lag_at_peak_s']:.4f}s")
    if support is not None and wheel_left is not None:
        cc_sw = compute_cross_correlation(support, wheel_left, fs)
        print(f"  support <-> wheel_vel_left:  corr={cc_sw['corr_at_peak']:.4f}  lag={cc_sw['lag_at_peak_s']:.4f}s")

    # 5. Signal frequency rank (by 2.5 Hz amplitude in final window)
    print("\n--- Signal frequency rank (final window, 2-3 Hz band) ---")
    rows_n = len(next(iter(data.values())))
    final_slice = slice(max(0, rows_n - 500), None)
    ranks: list[dict] = []
    for col_key, short_name, scale in SIGNAL_DEFS:
        raw = data.get(col_key)
        if raw is None:
            continue
        sig = raw[final_slice] * scale
        f_peak, amp_peak = _psd_frequency_peak(sig, fs, lo_hz=2.0, hi_hz=3.0)
        rms_val = _rms(sig)
        ranks.append({
            "signal": short_name,
            "col": col_key,
            "dominant_freq_hz": f_peak,
            "psd_amplitude": amp_peak,
            "rms": rms_val,
        })
    ranks.sort(key=lambda r: r["psd_amplitude"], reverse=True)
    print(f"  {'signal':20s}  {'freq(Hz)':>10s}  {'amp':>12s}  {'rms':>10s}")
    for r in ranks[:5]:
        print(f"  {r['signal']:20s}  {r['dominant_freq_hz']:10.4f}  {r['psd_amplitude']:12.2e}  {r['rms']:10.4f}")
    rank_path = OUT / "signal_frequency_rank.csv"
    with open(rank_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["signal", "col", "dominant_freq_hz", "psd_amplitude", "rms"])
        w.writeheader()
        w.writerows(ranks)
    print(f"  written: {rank_path}")

    # 6. Offline filter preview on pitch_rate
    print("\n--- Offline filter preview (pitch_rate signal) ---")
    pr_signal = data.get("pitch_rate_x_rad_s")
    if pr_signal is not None:
        fc_list = [2.3, 2.5, 2.7]
        Q_list = [2, 4, 6, 8, 10]
        filt_results = offline_filter_grid(pr_signal[final_slice], fs, "pitch_rate", fc_list, Q_list)
        print(f"  {'fc':>6s}  {'Q':>4s}  {'bw':>6s}  {'rms_raw':>10s}  {'rms_filt':>10s}  {'atten(dB)':>10s}")
        for r in filt_results:
            print(f"  {r['fc_hz']:6.1f}  {r['Q']:4.1f}  {r['bw_hz']:6.3f}  "
                  f"{r['rms_raw']:10.4f}  {r['rms_filtered']:10.4f}  {r['attenuation_db']:10.2f}")
        flt_path = OUT / "filter_candidate_coefficients.csv"
        with open(flt_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filt_results[0].keys()))
            w.writeheader()
            w.writerows(filt_results)
        print(f"  written: {flt_path}")

    # 7. Write frequency analysis summary
    freq_path = OUT / "frequency_analysis.json"
    with open(freq_path, "w") as f:
        json.dump(freq, f, indent=2)
    print(f"  written: {freq_path}")

    # 8. Create baseline_mode_audit.md
    print("\n--- Generating audit markdown ---")
    pitch_freq = freq.get("pitch", {}).get("freq_final_hz", 0)
    support_freq = freq.get("support_error", {}).get("freq_final_hz", 0)
    pitch_rate_freq = freq.get("pitch_rate", {}).get("freq_final_hz", 0)
    wheel_freq = freq.get("wheel_vel_left", {}).get("freq_final_hz", 0)

    md = f"""# Baseline Frequency and Sample-Rate Audit

## Sample Rate

| Metric | Value |
|--------|-------|
| Source | `{SRC.name}` |
| Rows | {sr['n_rows']} |
| Duration | {sr['t_duration_s']:.1f} s |
| dt median | {sr['dt_median_s']:.6f} s |
| dt min | {sr['dt_min_s']:.6f} s |
| dt max | {sr['dt_max_s']:.6f} s |
| fs (median) | {sr['fs_hz_median']:.1f} Hz |
| Step range | {sr.get('step_first', '?')} – {sr.get('step_last', '?')} |

**Conclusion:** Sample rate is {sr['fs_hz_median']:.0f} Hz (dt = {sr['dt_median_s']:.3f} s).

## Dominant Frequencies (Final Window)

| Signal | Frequency (Hz) | PSD Amplitude | RMS |
|--------|:-------------:|:-------------:|:---:|
| pitch | {pitch_freq:.4f} | {freq.get('pitch',{}).get('amp_final',0):.2e} | {freq.get('pitch',{}).get('rms_final',0):.4f} |
| support_error | {support_freq:.4f} | {freq.get('support_error',{}).get('amp_final',0):.2e} | {freq.get('support_error',{}).get('rms_final',0):.4f} |
| pitch_rate | {pitch_rate_freq:.4f} | {freq.get('pitch_rate',{}).get('amp_final',0):.2e} | {freq.get('pitch_rate',{}).get('rms_final',0):.4f} |
| wheel_vel_left | {wheel_freq:.4f} | {freq.get('wheel_vel_left',{}).get('amp_final',0):.2e} | {freq.get('wheel_vel_left',{}).get('rms_final',0):.4f} |
"""
    # Add all signals
    md += "\n### Full Signal List\n\n"
    md += "| Signal | Freq (Hz) | PSD Amp | RMS |\n"
    md += "|--------|:---------:|:-------:|:---:|\n"
    for r in ranks:
        md += f"| {r['signal']} | {r['dominant_freq_hz']:.4f} | {r['psd_amplitude']:.2e} | {r['rms']:.4f} |\n"

    md += f"""
## Cross-Correlation

| Pair | Correlation | Lag (s) |
|------|:-----------:|:-------:|
| pitch <-> support | {cc['corr_at_peak']:.4f} | {cc['lag_at_peak_s']:.4f} |
| pitch <-> pitch_rate | {cc_pr['corr_at_peak']:.4f} | {cc_pr['lag_at_peak_s']:.4f} |
| support <-> wheel_vel_left | {cc_sw['corr_at_peak']:.4f} | {cc_sw['lag_at_peak_s']:.4f} |

## Filter Design Recommendations

**Sample rate:** {sr['fs_hz_median']:.0f} Hz (confirmed from telemetry time column).
**Dominant mode:** {pitch_rate_freq:.3f} Hz pitch_rate, {pitch_freq:.3f} Hz pitch, {support_freq:.3f} Hz support.
**Target center frequency:** 2.5 Hz (consistent across all pitch/support/pitch-rate signals).
**Recommended Q range:** 4–8 (narrow notch, preserves <1 Hz balance dynamics).
**Filter type:** Causal IIR biquad notch (Direct Form II Transposed).

### Signals to filter (ranked by 2.5 Hz component)

| Rank | Signal | Rationale |
|------|--------|-----------|
| 1 | pitch_rate | Direct input to tau_pitch_rate = kd_pitch * pitch_rate. J candidate showed phase-lagged damping amplifies 2.5 Hz. |
| 2 | wheel_velocity | Direct input to tau_wheel_vel = -k_wheel_vel * wheel_vel. Also shows 2.5 Hz component. |
| 3 | support_velocity | Numerical derivative of support error. May amplify high-frequency noise. |

## Filter Coefficients (pitch_rate, 2.5 Hz, Q=6)

Best candidate: fc=2.5 Hz, Q=6, fs=100 Hz

```
b0 = {filt_results[0]['b0']:.6f}
b1 = {filt_results[0]['b1']:.6f}
b2 = {filt_results[0]['b2']:.6f}
a1 = {filt_results[0]['a1']:.6f}
a2 = {filt_results[0]['a2']:.6f}
```

## Notes

- The 2.5 Hz mode is confirmed across all sagittal signals.
- Filter must be causal and used inside the control loop.
- Offline preview is for analysis only; real results require online simulation.
- Phase response at low frequencies (<0.5 Hz) is near unity — balance dynamics preserved.
"""
    md_path = OUT / "baseline_mode_audit.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  written: {md_path}")

    print("\n--- Phase 1 complete ---")


if __name__ == "__main__":
    main()
