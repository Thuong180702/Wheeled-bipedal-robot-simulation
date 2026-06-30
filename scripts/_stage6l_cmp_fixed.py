"""Compare fixed-height 0.33 Python vs JAX per-step traces."""
import csv

root = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/k2_jax_runtime"
py_path = root + "/fixed_0p330_py_1000/telemetry_1782491655.csv"
jx_path = root + "/fixed_0p330_1000steps/telemetry_1782491529.csv"

with open(py_path) as f:
    py = list(csv.DictReader(f))
with open(jx_path) as f:
    jx = list(csv.DictReader(f))

print(f"PY: {len(py)} rows, JX: {len(jx)} rows")

# Step 0 state
print(f"Step 0 com_z: PY={py[0]['current_com_z_m']}  JX={jx[0]['current_com_z_m']}")
print(f"Step 0 tau_pitch: PY={py[0].get('tau_pitch','?')}  JX={jx[0].get('tau_pitch','?')}")

# Find first torque divergence
n = min(len(py), len(jx))
for i in range(n):
    rp, rj = py[i], jx[i]
    for k in ["tau_pitch", "tau_pitch_rate", "tau_position", "tau_total_after_final_clip"]:
        vp = float(rp.get(k, 0) or 0)
        vj = float(rj.get(k, 0) or 0)
        if abs(vp - vj) > 0.01:
            print(f"\nStep {i}: FIRST TORQUE DIVERGENCE in {k}")
            print(f"  PY={vp:.6f} JX={vj:.6f} diff={abs(vp-vj):.6f}")
            pp = float(rp.get("pitch_x_rad", 0) or 0)
            pj = float(rj.get("pitch_x_rad", 0) or 0)
            prp = float(rp.get("pitch_rate_effective_rad_s", 0) or 0)
            prj = float(rj.get("pitch_rate_effective_rad_s", 0) or 0)
            print(f"  pitch: PY={pp:.6f} JX={pj:.6f} diff={abs(pp-pj):.6e}")
            print(f"  pitch_rate_eff: PY={prp:.6f} JX={prj:.6f} diff={abs(prp-prj):.6e}")
            print(f"  adaptive_bias_trim_active: PY={rp.get('adaptive_bias_trim_active')} JX={rj.get('adaptive_bias_trim_active')}")
            print(f"  adaptive_bias_mean_error_m: PY={rp.get('adaptive_bias_mean_error_m')} JX={rj.get('adaptive_bias_mean_error_m')}")
            print(f"  outer_loop_active: PY={rp.get('outer_loop_active')} JX={rj.get('outer_loop_active')}")
            # Show surrounding steps
            for step in range(max(0, i-1), min(n, i+4)):
                rp2, rj2 = py[step], jx[step]
                vp2 = float(rp2.get("tau_pitch_rate", 0) or 0)
                vj2 = float(rj2.get("tau_pitch_rate", 0) or 0)
                print(f"  [{step}] tau_pitch_rate: PY={vp2:.6f} JX={vj2:.6f}")
            # Show notch state
            print(f"\n  Notch state:")
            print(f"    pitch_rate_raw: PY={rp.get('pitch_rate_raw_rad_s')} JX={rj.get('pitch_rate_raw_rad_s')}")
            print(f"    pitch_rate_notched: PY={rp.get('pitch_rate_notched_rad_s')} JX={rj.get('pitch_rate_notched_rad_s')}")
            print(f"    pitch_rate_effective: PY={rp.get('pitch_rate_effective_rad_s')} JX={rj.get('pitch_rate_effective_rad_s')}")
            print(f"    wip_notch_height_gate: PY={rp.get('wip_notch_height_gate')} JX={rj.get('wip_notch_height_gate')}")
            break
    else:
        continue
    break

# Track adaptive bias evolution
print("\n=== Adaptive Bias Evolution (JAX) ===")
for i in range(0, min(550, len(jx)), 50):
    r = jx[i]
    print(f"  [{i:3d}] active={r.get('adaptive_bias_trim_active'):5s} "
          f"mean_err={float(r.get('adaptive_bias_mean_error_m',0) or 0):.6f} "
          f"trim={float(r.get('adaptive_bias_tau_nm',0) or 0):.4f} "
          f"block={r.get('adaptive_bias_block_reason','?'):20s} "
          f"ol_active={r.get('outer_loop_active'):5s} "
          f"ol_block={r.get('outer_loop_block_reason','?'):20s}")

print("\n=== Adaptive Bias Evolution (Python) ===")
for i in range(0, min(550, len(py)), 50):
    r = py[i]
    print(f"  [{i:3d}] active={r.get('adaptive_bias_trim_active'):5s} "
          f"mean_err={float(r.get('adaptive_bias_mean_error_m',0) or 0):.6f} "
          f"trim={float(r.get('adaptive_bias_tau_nm',0) or 0):.4f} "
          f"block={r.get('adaptive_bias_block_reason','?'):20s} "
          f"ol_active={r.get('outer_loop_active'):5s} "
          f"ol_block={r.get('outer_loop_block_reason','?'):20s}")

# When does JAX outer_loop first block?
for i in range(n):
    rj = jx[i]
    if rj.get("outer_loop_active") == "False":
        print(f"\nJAX outer_loop FIRST BLOCKED at step {i}: reason={rj.get('outer_loop_block_reason')}")
        print(f"  support_error_m={rj.get('outer_loop_support_error_m')}")
        break

# When does JAX adaptive_bias first block?
for i in range(n):
    rj = jx[i]
    if rj.get("adaptive_bias_trim_active") == "False" and i > 5:
        print(f"JAX adaptive_bias FIRST BLOCKED at step {i}: reason={rj.get('adaptive_bias_block_reason')}")
        print(f"  mean_error_m={rj.get('adaptive_bias_mean_error_m')}")
        break

# Compare JAX ABS state: what's the sagittal_position_error_m evolution?
print("\n=== Support Error Evolution ===")
for i in range(0, min(550, n), 50):
    rp, rj = py[i], jx[i]
    vp = float(rp.get("outer_loop_support_error_m", 0) or 0)
    vj = float(rj.get("outer_loop_support_error_m", 0) or 0)
    print(f"  [{i:3d}] PY={vp:.6f}  JX={vj:.6f}  diff={abs(vp-vj):.6f}")
