#!/usr/bin/env python3
"""
K1 Nonlinear Mode Source Isolation — Phase 5.

Uses augmented telemetry to determine which mechanism drives the 0.24-0.4 Hz mode.

STRICT CONSTRAINT: Analysis only. No controller modifications.

Analyses per height and run type:
  A. Spectral analysis (FFT/Welch PSD)
  B. Cross-spectral/coherence analysis
  C. Event-triggered averaging around clipping/cap/notch events
  D. Causality/predictive tests (lagged regression, ablation prediction)

Output:
  outputs/k1_augmented_identification_dataset/nonlinear_mode_source_analysis.json
  outputs/k1_augmented_identification_dataset/nonlinear_mode_source_analysis.md
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUGMENTED_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"
LEGACY_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"

# -- Analysis Parameters --
TARGET_BAND_HZ = (0.15, 0.55)  # Frequency band of interest
FS_HZ = 100.0  # Control frequency

# -- Source Classifications --
SOURCE_CLASSIFICATIONS = [
    "NOTCH_FILTER_DOMINANT",
    "TORQUE_CLIPPING_DOMINANT",
    "POSITION_CAP_DOMINANT",
    "NOTCH_CLIPPING_INTERACTION",
    "CONTACT_NONLINEARITY_DOMINANT",
    "MISSING_STATE_OBSERVABILITY",
    "INCONCLUSIVE",
]


def compute_welch_psd(signal: np.ndarray, fs: float = FS_HZ, nperseg: int = None) -> tuple:
    """Compute Welch PSD estimate. Returns (freqs, psd)."""
    if nperseg is None:
        nperseg = min(256, len(signal) // 4)
    if nperseg < 16:
        return np.array([]), np.array([])

    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)  # Remove DC

    n = len(signal)
    n_overlap = nperseg // 2
    n_segments = (n - n_overlap) // (nperseg - n_overlap)
    if n_segments < 2:
        return np.array([]), np.array([])

    window = np.hanning(nperseg)
    psd = np.zeros(nperseg // 2 + 1)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)

    for i in range(n_segments):
        start = i * (nperseg - n_overlap)
        segment = signal[start:start + nperseg] * window
        fft_seg = np.fft.rfft(segment)
        psd += np.abs(fft_seg) ** 2

    psd /= (n_segments * fs * np.sum(window ** 2))
    return freqs, psd


def find_dominant_frequency(freqs: np.ndarray, psd: np.ndarray,
                            band: tuple = TARGET_BAND_HZ) -> dict:
    """Find dominant frequency within target band."""
    if len(freqs) == 0:
        return {"freq_hz": None, "power": 0.0, "found": False}

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return {"freq_hz": None, "power": 0.0, "found": False}

    band_psd = psd[mask]
    band_freqs = freqs[mask]
    idx = np.argmax(band_psd)
    return {
        "freq_hz": float(band_freqs[idx]),
        "power": float(band_psd[idx]),
        "found": True,
        "total_band_power": float(np.sum(band_psd)),
    }


def compute_coherence(x: np.ndarray, y: np.ndarray, fs: float = FS_HZ,
                      nperseg: int = None) -> dict:
    """Estimate coherence between two signals in target band."""
    if nperseg is None:
        nperseg = min(256, len(x) // 4)
    if nperseg < 16 or len(x) < nperseg:
        return {"mean_coherence": 0.0, "peak_coherence": 0.0,
                "peak_freq_hz": None, "sufficient_data": False}

    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)

    n_overlap = nperseg // 2
    n_segments = (len(x) - n_overlap) // (nperseg - n_overlap)
    if n_segments < 2:
        return {"mean_coherence": 0.0, "peak_coherence": 0.0,
                "peak_freq_hz": None, "sufficient_data": False}

    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)

    # Cross-spectral density estimate
    pxx = np.zeros(len(freqs))
    pyy = np.zeros(len(freqs))
    pxy = np.zeros(len(freqs), dtype=complex)

    for i in range(n_segments):
        start = i * (nperseg - n_overlap)
        x_seg = x[start:start + nperseg] * window
        y_seg = y[start:start + nperseg] * window
        fx = np.fft.rfft(x_seg)
        fy = np.fft.rfft(y_seg)
        pxx += np.abs(fx) ** 2
        pyy += np.abs(fy) ** 2
        pxy += fx * np.conj(fy)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        coh = np.abs(pxy) ** 2 / (pxx * pyy)
        coh = np.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0)

    # Band-limited coherence
    band_mask = (freqs >= TARGET_BAND_HZ[0]) & (freqs <= TARGET_BAND_HZ[1])
    if np.any(band_mask):
        coh_band = coh[band_mask]
        peak_idx = np.argmax(coh_band)
        return {
            "mean_coherence": float(np.mean(coh_band)),
            "peak_coherence": float(coh_band[peak_idx]),
            "peak_freq_hz": float(freqs[band_mask][peak_idx]),
            "sufficient_data": True,
        }
    return {"mean_coherence": 0.0, "peak_coherence": 0.0,
            "peak_freq_hz": None, "sufficient_data": False}


def event_triggered_average(signal: np.ndarray, trigger: np.ndarray,
                            window: int = 50) -> dict:
    """Compute event-triggered average around trigger events."""
    events = np.where(trigger > 0)[0]
    if len(events) < 3:
        return {"n_events": len(events), "eta": None, "sufficient": False}

    segments = []
    half = window // 2
    for ev in events:
        start = max(0, ev - half)
        end = min(len(signal), ev + half)
        seg = signal[start:end]
        if len(seg) >= window // 2:
            # Pad to full window if needed
            padded = np.zeros(window)
            offset = half - min(half, ev)
            actual = min(window - offset, len(seg))
            padded[offset:offset + actual] = seg[:actual]
            segments.append(padded)

    if not segments:
        return {"n_events": len(events), "eta": None, "sufficient": False}

    eta = np.mean(segments, axis=0)
    eta_std = np.std(segments, axis=0)
    return {
        "n_events": len(events),
        "eta": eta.tolist(),
        "eta_std": eta_std.tolist(),
        "sufficient": True,
        "pre_event_mean": float(np.mean(eta[:window // 2])),
        "post_event_mean": float(np.mean(eta[window // 2:])),
        "peak_response": float(np.max(np.abs(eta))),
    }


def lagged_regression_importance(X: np.ndarray, y: np.ndarray,
                                  lags: list = None) -> dict:
    """Compute lagged feature importance via linear regression coefficients."""
    if lags is None:
        lags = list(range(1, 21))  # 1-20 step lags

    n = len(y)
    features = []
    feature_names = []

    # Add current values
    for j in range(X.shape[1]):
        features.append(X[:n, j])
        feature_names.append(f"x{j}_lag0")

    # Add lagged values
    for lag in lags:
        for j in range(X.shape[1]):
            col = np.zeros(n)
            col[lag:] = X[:n - lag, j]
            features.append(col)
            feature_names.append(f"x{j}_lag{lag}")

    X_design = np.column_stack(features)

    # Ridge regression for importance
    try:
        coeffs = np.linalg.lstsq(X_design, y, rcond=None)[0]
        importance = {name: float(abs(c))
                      for name, c in zip(feature_names, coeffs)
                      if abs(c) > 1e-6}
        return {
            "n_features": len(feature_names),
            "n_significant": len(importance),
            "top_features": sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10],
            "sufficient_data": True,
        }
    except np.linalg.LinAlgError:
        return {"n_features": len(feature_names), "error": "LinAlgError", "sufficient_data": False}


def analyze_run(rows: list, height_name: str, run_type: str) -> dict:
    """Analyze a single run's telemetry for mode source evidence."""
    if len(rows) < 100:
        return {"height": height_name, "run_type": run_type,
                "error": "INSUFFICIENT_DATA", "n_rows": len(rows)}

    n_rows = len(rows)
    analysis = {"height": height_name, "run_type": run_type, "n_rows": n_rows}

    # Extract signals
    try:
        pitch_x = np.array([float(r.get("pitch_x", 0)) for r in rows])
        support_err = np.array([float(r.get("k1_support_error_m", 0)) for r in rows])
        com_vy = np.array([float(r.get("k1_com_y_velocity_m_s", 0)) for r in rows])
        notch_out = np.array([float(r.get("k1_notch_output", 0)) for r in rows])
        filt_pitch_rate = np.array([float(r.get("k1_filtered_pitch_rate_x", 0)) for r in rows])
        clip_delta = np.array([float(r.get("k1_tau_clip_delta_common", 0)) for r in rows])
        cap_margin = np.array([float(r.get("k1_tau_position_cap_margin_nm", 0)) for r in rows])
        clip_active = np.array([1.0 if r.get("k1_tau_total_clip_active", "0") in ("True", "true", "1", "1.0") else 0.0 for r in rows])
        cap_active = np.array([1.0 if r.get("k1_tau_position_cap_active", "0") in ("True", "true", "1", "1.0") else 0.0 for r in rows])
        notch_enabled = np.array([1.0 if r.get("k1_notch_enabled", "0") in ("True", "true", "1", "1.0") else 0.0 for r in rows])
    except (KeyError, ValueError) as e:
        analysis["error"] = f"MISSING_FIELD:{e}"
        return analysis

    # A. Spectral analysis
    spectral = {}
    for name, sig in [("pitch_x", pitch_x), ("support_error", support_err),
                       ("com_y_velocity", com_vy), ("notch_output", notch_out),
                       ("filtered_pitch_rate", filt_pitch_rate),
                       ("clip_delta_common", clip_delta)]:
        freqs, psd = compute_welch_psd(sig)
        dom = find_dominant_frequency(freqs, psd)
        spectral[name] = dom

    analysis["spectral"] = spectral

    # B. Cross-spectral coherence
    coherence = {}
    for name, a, b in [
        ("pitch_vs_notch", pitch_x, notch_out),
        ("pitch_vs_clip", pitch_x, clip_delta),
        ("pitch_vs_cap", pitch_x, cap_active),
        ("pitch_vs_filt_rate", pitch_x, filt_pitch_rate),
        ("support_vs_notch", support_err, notch_out),
    ]:
        coherence[name] = compute_coherence(a, b)

    analysis["coherence"] = coherence

    # C. Event-triggered averaging
    eta = {}
    # Around clipping events
    clip_events = (np.abs(clip_delta) > 0.01).astype(float)
    eta["pitch_around_clip"] = event_triggered_average(pitch_x, clip_events)
    # Around position cap active transitions
    cap_events = (np.abs(np.diff(np.concatenate([[0], cap_active]))) > 0.5).astype(float)
    eta["pitch_around_cap_transition"] = event_triggered_average(pitch_x, cap_events)

    analysis["event_triggered"] = eta

    # D. Predictive tests
    # Base x6 model vs augmented
    x6 = np.column_stack([pitch_x, np.gradient(pitch_x), support_err,
                           np.gradient(support_err), com_vy,
                           0.5 * (np.zeros_like(pitch_x))])  # wheel_vel not in these rows without header
    y_next = np.roll(pitch_x, -1)
    y_next = y_next[:len(y_next) - 20]  # Reserve for test

    # Check if augmented fields are constant (no notch data)
    notch_variance = np.var(notch_out)
    clip_variance = np.var(clip_delta)

    analysis["predictive"] = {
        "notch_output_variance": float(notch_variance),
        "clip_delta_variance": float(clip_variance),
        "notch_is_constant": bool(notch_variance < 1e-12),
        "clip_is_constant": bool(clip_variance < 1e-12),
    }

    # Source classification
    dom_freq = spectral.get("pitch_x", {}).get("freq_hz")
    if dom_freq and TARGET_BAND_HZ[0] <= dom_freq <= TARGET_BAND_HZ[1]:
        analysis["mode_found"] = True
        analysis["mode_freq_hz"] = dom_freq
    else:
        analysis["mode_found"] = False

    return analysis


def classify_mode_source(all_analyses: list) -> dict:
    """Synthesize source classification from all run analyses."""
    mode_found_any = any(a.get("mode_found") for a in all_analyses)

    if not mode_found_any:
        return {
            "classification": "INCONCLUSIVE",
            "reason": "Mode not found in any run",
            "evidence": "No dominant frequency in 0.15-0.55 Hz band",
        }

    # Aggregate coherence evidence
    pitch_notch_coh = []
    pitch_clip_coh = []
    pitch_cap_coh = []
    for a in all_analyses:
        coh = a.get("coherence", {})
        if coh.get("pitch_vs_notch", {}).get("sufficient_data"):
            pitch_notch_coh.append(coh["pitch_vs_notch"]["peak_coherence"])
        if coh.get("pitch_vs_clip", {}).get("sufficient_data"):
            pitch_clip_coh.append(coh["pitch_vs_clip"]["peak_coherence"])
        if coh.get("pitch_vs_cap", {}).get("sufficient_data"):
            pitch_cap_coh.append(coh["pitch_vs_cap"]["peak_coherence"])

    mean_notch_coh = np.mean(pitch_notch_coh) if pitch_notch_coh else 0.0
    mean_clip_coh = np.mean(pitch_clip_coh) if pitch_clip_coh else 0.0
    mean_cap_coh = np.mean(pitch_cap_coh) if pitch_cap_coh else 0.0

    # Classification logic
    if mean_notch_coh > 0.7 and mean_notch_coh > mean_clip_coh:
        classification = "NOTCH_FILTER_DOMINANT"
        reason = f"High pitch-notch coherence ({mean_notch_coh:.3f}) exceeds clip coherence ({mean_clip_coh:.3f})"
    elif mean_clip_coh > 0.7 and mean_clip_coh > mean_notch_coh:
        classification = "TORQUE_CLIPPING_DOMINANT"
        reason = f"High pitch-clip coherence ({mean_clip_coh:.3f}) exceeds notch coherence ({mean_notch_coh:.3f})"
    elif mean_cap_coh > 0.7:
        classification = "POSITION_CAP_DOMINANT"
        reason = f"High pitch-position-cap coherence ({mean_cap_coh:.3f})"
    elif mean_notch_coh > 0.35 and mean_clip_coh > 0.35:
        classification = "NOTCH_CLIPPING_INTERACTION"
        reason = f"Both notch ({mean_notch_coh:.3f}) and clip ({mean_clip_coh:.3f}) show significant coherence"
    else:
        classification = "INCONCLUSIVE"
        reason = f"Insufficient coherence: notch={mean_notch_coh:.3f}, clip={mean_clip_coh:.3f}"

    return {
        "classification": classification,
        "reason": reason,
        "evidence": {
            "mean_pitch_notch_coherence": float(mean_notch_coh),
            "mean_pitch_clip_coherence": float(mean_clip_coh),
            "mean_pitch_cap_coherence": float(mean_cap_coh),
            "n_runs_with_mode": sum(1 for a in all_analyses if a.get("mode_found")),
            "total_runs_analyzed": len(all_analyses),
        },
    }


def analyze_source(dataset_dir: Path = None):
    """Run the full source isolation analysis."""
    if dataset_dir is None:
        dataset_dir = AUGMENTED_DIR

    if not dataset_dir.exists():
        dataset_dir = LEGACY_DIR
    if not dataset_dir.exists():
        return {"status": "NO_DATASET_FOUND"}

    all_analyses = []

    for height_name in sorted(["low_0p330", "mid_0p400", "high_0p480"]):
        height_dir = dataset_dir / height_name
        if not height_dir.exists():
            continue

        for run_type in ["A_equilibrium", "D_prbs_excitation", "B_90n_push"]:
            run_dir = height_dir / run_type
            if not run_dir.exists():
                continue

            csv_files = list(run_dir.glob("telemetry_*.csv"))
            if not csv_files:
                continue

            csv_path = csv_files[0]
            try:
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except Exception:
                continue

            print(f"  Analyzing {height_name}/{run_type} ({len(rows)} rows)...")
            analysis = analyze_run(rows, height_name, run_type)
            all_analyses.append(analysis)

    source_classification = classify_mode_source(all_analyses)

    result = {
        "source_classification": source_classification,
        "per_run_analyses": all_analyses,
        "n_runs_analyzed": len(all_analyses),
    }

    # Save
    json_path = dataset_dir / "nonlinear_mode_source_analysis.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Markdown report
    md_lines = [
        "# K1 Nonlinear Mode Source Isolation",
        f"",
        f"## Classification: {source_classification['classification']}",
        f"",
        f"**Reason:** {source_classification['reason']}",
        f"",
        f"### Evidence",
    ]
    ev = source_classification.get("evidence", [])
    if isinstance(ev, dict):
        for k, v in ev.items():
            md_lines.append(f"- **{k}:** {v}")
    elif isinstance(ev, list):
        for item in ev:
            md_lines.append(f"- {item}")
    elif ev:
        md_lines.append(f"- {ev}")
    md_lines.append("")
    md_lines.append("### Spectral Results")
    md_lines.append("")
    md_lines.append("| Height | Run | Pitch Mode Freq | Support Mode | COM Vy Mode |")
    md_lines.append("|--------|-----|----------------|-------------|------------|")
    for a in all_analyses:
        spec = a.get("spectral", {})
        p = spec.get("pitch_x", {})
        s = spec.get("support_error", {})
        c = spec.get("com_y_velocity", {})
        md_lines.append(
            f"| {a['height']} | {a['run_type']} | "
            f"{p.get('freq_hz', 'N/A')} | {s.get('freq_hz', 'N/A')} | "
            f"{c.get('freq_hz', 'N/A')} |"
        )
    md_lines.append("")
    md_lines.append("### Coherence Results")
    md_lines.append("")
    for a in all_analyses:
        coh = a.get("coherence", {})
        pn = coh.get("pitch_vs_notch", {})
        pc = coh.get("pitch_vs_clip", {})
        md_lines.append(
            f"- **{a['height']}/{a['run_type']}:** pitch-notch={pn.get('peak_coherence', 'N/A')}, "
            f"pitch-clip={pc.get('peak_coherence', 'N/A')}"
        )

    md_path = dataset_dir / "nonlinear_mode_source_analysis.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nSource analysis complete. Classification: {source_classification['classification']}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze K1 nonlinear mode source")
    parser.add_argument("--dataset-dir", type=str, default=None)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    analyze_source(dataset_dir)


if __name__ == "__main__":
    main()
