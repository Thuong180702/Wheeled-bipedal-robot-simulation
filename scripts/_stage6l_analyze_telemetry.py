"""Quick diagnostic: analyze ramp_up telemetry to understand JAX failure."""
import csv, os

root = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation"

# Python backend (K1, default backend)
path_py = root + "/outputs/k2_dynamic_height_gate_crossing/raw/ramp_up_0p330_to_0p480_K1/telemetry_5000.csv"
with open(path_py) as f:
    rows_py = list(csv.DictReader(f))
print(f"=== PYTHON BACKEND (K1): {len(rows_py)} rows ===")
for i in [0, 250, 500, 1000, 2000, 4999]:
    if i < len(rows_py):
        r = rows_py[i]
        cz = float(r.get("current_com_z_m", 0))
        target = float(r.get("dynamic_height_target_m", 0))
        pitch = float(r.get("pitch_x_rad", 0))
        print(f"  [{i}] com_z={cz:.4f} target={target:.4f} pitch={pitch:.4f}")
last = rows_py[-1]
print(f"  Last: com_z={float(last['current_com_z_m']):.4f} target={float(last['dynamic_height_target_m']):.4f}")

# JAX backend (K2)
path_jx = root + "/outputs/k2_dynamic_height_gate_crossing/raw/ramp_up_0p330_to_0p480_K2/telemetry_5000.csv"
with open(path_jx) as f:
    rows_jx = list(csv.DictReader(f))
print(f"\n=== JAX BACKEND (K2): {len(rows_jx)} rows ===")
for i in [0, 250, 500, 550]:
    if i < len(rows_jx):
        r = rows_jx[i]
        cz = float(r.get("current_com_z_m", 0))
        target = float(r.get("dynamic_height_target_m", 0))
        pitch = float(r.get("pitch_x_rad", 0))
        print(f"  [{i}] com_z={cz:.4f} target={target:.4f} pitch={pitch:.4f}")

# Compare at step 0 and step 250
print("\n=== STEP 0 COMPARISON ===")
r0_py = rows_py[0]
r0_jx = rows_jx[0]
for k in ["current_com_z_m", "pitch_x_rad", "dynamic_height_target_m", "schedule_height_reference_m",
          "effective_k_position", "effective_k_wheel_velocity", "effective_kd_pitch",
          "wip_notch_height_gate", "pitch_ref_total_after_outer_loop_deg",
          "tau_pitch", "tau_pitch_rate", "tau_position"]:
    v_py = r0_py.get(k, "N/A")
    v_jx = r0_jx.get(k, "N/A")
    print(f"  {k}: PY={v_py}  JX={v_jx}")

# Find first step where JAX torque diverges from Python
print("\n=== FIRST TORQUE DIVERGENCE ===")
for i in range(min(len(rows_py), len(rows_jx))):
    rp = rows_py[i]
    rj = rows_jx[i]
    for k in ["tau_pitch", "tau_pitch_rate", "tau_position"]:
        vp = float(rp.get(k, 0))
        vj = float(rj.get(k, 0))
        if abs(vp - vj) > 0.01:
            print(f"Step {i}: {k} diverge: PY={vp:.4f} JX={vj:.4f} diff={vp-vj:.4f}")
            # Show relevant state
            for sk in ["pitch_x_rad", "pitch_rate_effective_rad_s", "current_com_z_m",
                       "adaptive_bias_trim_tau_nm", "adaptive_bias_trim_active",
                       "adaptive_bias_mean_error_m",
                       "outer_loop_pitch_ref_total_deg", "outer_loop_support_error_m",
                       "pitch_ref_total_after_outer_loop_deg",
                       "wip_notch_height_gate", "pitch_rate_raw_rad_s", "pitch_rate_notched_rad_s"]:
                vp2 = rp.get(sk, "N/A")
                vj2 = rj.get(sk, "N/A")
                print(f"    {sk}: PY={vp2}  JX={vj2}")
            # Show next 5 steps
            for j in range(i+1, min(i+6, len(rows_py), len(rows_jx))):
                rpj = rows_py[j]
                rjj = rows_jx[j]
                print(f"  Step {j}: tau_pos PY={float(rpj.get('tau_position',0)):.4f} JX={float(rjj.get('tau_position',0)):.4f}  tau_pitch PY={float(rpj.get('tau_pitch',0)):.4f} JX={float(rjj.get('tau_pitch',0)):.4f}")
            break
    else:
        continue
    break

# Also compare adaptive_bias_trim at step 250
print("\n=== ADAPTIVE BIAS TRIM at step 250 ===")
r250_py = rows_py[250]
r250_jx = rows_jx[250]
for k in sorted(r250_py.keys()):
    if "adaptive_bias" in k or "bias_trim" in k or "bias_cancel" in k:
        vp = r250_py.get(k, "N/A")
        vj = r250_jx.get(k, "N/A")
        print(f"  {k}: PY={vp}  JX={vj}")

# Check outer_loop fields
print("\n=== OUTER LOOP at step 250 ===")
for k in sorted(r250_py.keys()):
    if "outer_loop" in k or "calibrated" in k:
        vp = r250_py.get(k, "N/A")
        vj = r250_jx.get(k, "N/A")
        print(f"  {k}: PY={vp}  JX={vj}")
