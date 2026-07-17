#!/usr/bin/env python3
"""Phase 12 stress tests for Candidate E v2."""
import subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = str(ROOT / "scripts" / "run_k2_jax_realtime.py")
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_DIR = ROOT / "outputs" / "k2_final_stress"

# Dense height sweep: 0.300 to 0.480 every 0.010m
heights = [f"low_0p{h:03d}" for h in range(300, 481, 10)]
# Map to actual setup files
height_map = {}
for h_mm in range(300, 481, 10):
    h_str = f"{h_mm/1000:.3f}"
    for sfx in [f"low_0p{h_mm}", f"mid_0p{h_mm}", f"high_0p{h_mm}"]:
        fpath = SETUP_DIR / f"{sfx}_setup.json"
        if fpath.exists():
            height_map[h_str] = sfx
            break

results = {}
for h_str, sfx in sorted(height_map.items()):
    out_dir = OUT_DIR / f"h_{h_str.replace('.','p')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, RUNNER, "--height-setup", str(SETUP_DIR / f"{sfx}_setup.json"),
            "--steps", "2000", "--telemetry", "summary", "--quiet",
            "--output-dir", str(out_dir)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    fell = "[FALL]" in result.stdout
    status = "FALL" if fell else "OK"
    # Extract pitch RMS
    pitch_rms = 0.0
    for line in result.stdout.split("\n"):
        if "Pitch X:" in line and "RMS:" in line:
            try:
                pitch_rms = float(line.split("RMS:")[1].split("deg")[0].strip())
            except: pass
    results[h_str] = {"status": status, "pitch_rms": pitch_rms}
    print(f"h={h_str}: {status} pitch_rms={pitch_rms:.1f}")

# Summarize
falls = sum(1 for r in results.values() if r["status"] == "FALL")
oks = sum(1 for r in results.values() if r["status"] == "OK")
pitch_vals = [r["pitch_rms"] for r in results.values() if r["pitch_rms"] > 0]
mean_pitch = sum(pitch_vals)/len(pitch_vals) if pitch_vals else 0
print(f"\nStress: {oks} OK, {falls} FALLS, mean pitch RMS={mean_pitch:.1f} deg")
with open(OUT_DIR / "stress_results.json", "w") as f:
    json.dump({"results": results, "oks": oks, "falls": falls, "mean_pitch_rms": mean_pitch}, f)
