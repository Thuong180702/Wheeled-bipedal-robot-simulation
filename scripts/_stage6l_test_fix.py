"""Test JAX ring buffer fix at fixed 0.33 for 1000 steps."""
import subprocess, sys, time, csv
from pathlib import Path

ROOT = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation")
outdir = ROOT / "outputs" / "k2_jax_runtime" / "fixed_0p330_ringbuf_fix"

# Delete old telemetry
for f in outdir.glob("telemetry_*.csv"):
    f.unlink()

cmd = [
    sys.executable, str(ROOT/"scripts"/"simulate_hierarchical_controller.py"),
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "jax",
    "--height-variant-setup", str(ROOT/"outputs"/"physical_target_height_setups_centered"/"low_0p330_setup.json"),
    "--steps", "1000",
    "--telemetry-decimation", "1",
    "--failure-window-steps", "1000",
    "--output-dir", str(outdir),
]

t0 = time.time()
r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
elapsed = time.time() - t0

print(f"Completed in {elapsed:.0f}s, rc={r.returncode}")
for line in r.stdout.splitlines()[-10:]:
    print(line)

for f in sorted(outdir.glob("telemetry_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
    n = sum(1 for _ in open(f))
    print(f"Rows: {n}")
    rows = list(csv.DictReader(open(f)))
    last = rows[-1]
    cz = last.get("current_com_z_m", "?")
    px = float(last.get("pitch_x_rad", "0"))
    print(f"Last step: com_z={cz}, pitch={px*57.3:.1f}deg")

    # Check ABS status evolution
    for i in [0, 100, 200, 300, 500, 999]:
        if i < len(rows):
            r = rows[i]
            print(f"  [{i}] abs_active={r.get('adaptive_bias_trim_active')} "
                  f"mean_err={float(r.get('adaptive_bias_mean_error_m',0) or 0):.4f} "
                  f"trim={float(r.get('adaptive_bias_tau_nm',0) or 0):.4f} "
                  f"ol_active={r.get('outer_loop_active')} "
                  f"ol_block={r.get('outer_loop_block_reason','?')}")
    break
