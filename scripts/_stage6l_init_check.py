"""Check initial state difference between Python and JAX backends."""
import subprocess, sys, csv, time
from pathlib import Path

ROOT = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation")

for backend in ["python", "jax"]:
    outdir = ROOT / "outputs" / "k2_jax_runtime" / f"init_check_{backend}"
    for f in outdir.glob("telemetry_*.csv"):
        f.unlink()

    cmd = [
        sys.executable, str(ROOT/"scripts"/"simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--controller-backend", backend,
        "--height-variant-setup", str(ROOT/"outputs"/"physical_target_height_setups_centered"/"low_0p330_setup.json"),
        "--steps", "10",
        "--telemetry-decimation", "1",
        "--failure-window-steps", "10",
        "--output-dir", str(outdir),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    print(f"[{backend}] rc={r.returncode} in {elapsed:.0f}s")

    for f in sorted(outdir.glob("telemetry_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        rows = list(csv.DictReader(open(f)))
        r0 = rows[0]
        print(f"  Step 0:")
        print(f"    com_vy={r0.get('com_vy_m_s','?')}  com_vz={r0.get('com_vz_m_s','?')}")
        print(f"    com_x={r0.get('com_x_m','?')}  com_y={r0.get('com_y_m','?')}  com_z={r0.get('com_z_m','?')}")
        print(f"    pitch={r0.get('pitch_x_rad','?')}  pitch_rate={r0.get('pitch_rate_rad_s','?')}")
        print(f"    tau_pitch={r0.get('tau_pitch','?')}  tau_wheel_left={r0.get('tau_wheel_velocity_left','?')}")
        # Check sagittal input
        print(f"    sagittal_position_error_m={r0.get('sagittal_position_error_m','?')}")
        print(f"    sagittal_velocity_m_s={r0.get('sagittal_velocity_m_s','?')}")
        print(f"    support_position_error_m={r0.get('support_position_error_m','?')}")
        print(f"    commanded_height_ref_m (scheduled)= {r0.get('schedule_height_reference_m','?')}")
        break
