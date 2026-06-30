"""Check initial conditions in both Python and JAX telemetry."""
import csv, json

root = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/k2_jax_runtime"
py_path = root + "/fixed_0p330_py_1000/telemetry_1782491655.csv"
jx_path = root + "/fixed_0p330_1000steps/telemetry_1782491529.csv"

with open(py_path) as f:
    py = list(csv.DictReader(f))
with open(jx_path) as f:
    jx = list(csv.DictReader(f))

# Compare STEP 0 completely
r0p, r0j = py[0], jx[0]
print("=== STEP 0 COMPARISON ===")
all_same = True
diffs = []
for k in sorted(r0p.keys()):
    vp = r0p.get(k, "")
    vj = r0j.get(k, "")
    if vp != vj:
        try:
            dp = float(vp)
            dj = float(vj)
            if abs(dp - dj) > 1e-15:
                diffs.append((k, dp, dj, abs(dp-dj)))
                all_same = False
        except:
            if vp != vj:
                diffs.append((k, vp, vj, "string diff"))
                all_same = False

print(f"All fields same: {all_same}")
print(f"Fields differing: {len(diffs)}")
for k, vp, vj, d in diffs[:30]:
    print(f"  {k}: PY={vp} JX={vj} diff={d}")

# Compare key physical state at step 1
print("\n=== STEP 1 PHYSICAL STATE ===")
for k in ["pitch_x_rad", "pitch_rate_rad_s", "roll_y_rad", "roll_rate_rad_s",
          "com_x_m", "com_y_m", "com_z_m", "com_vx_m_s", "com_vy_m_s", "com_vz_m_s",
          "joint_pos", "joint_vel"]:
    vp = py[1].get(k, "N/A")
    vj = jx[1].get(k, "N/A")
    print(f"  {k}: PY={vp}  JX={vj}")

# Compare torque at step 0
print("\n=== STEP 0 TORQUE ===")
for k in ["tau_pitch", "tau_pitch_rate", "tau_position",
          "tau_total_unclipped", "tau_total_clipped",
          "tau_total_after_final_clip",
          "tau_wheel_velocity_left", "tau_wheel_velocity_right"]:
    vp = py[0].get(k, "N/A")
    vj = jx[0].get(k, "N/A")
    print(f"  {k}: PY={vp}  JX={vj}")

# Compare key fields that might explain ABS divergence
# Check the SAFETY gate values at step 200
print("\n=== STEP 200 SAFETY GATES ===")
r200p, r200j = py[200], jx[200]
for k in sorted(r200p.keys()):
    if "adaptive_bias" in k or "safety" in k or "abs_error" in k or "pitch_ok" in k:
        vp = r200p.get(k, "N/A")
        vj = r200j.get(k, "N/A")
        print(f"  {k}: PY={vp}  JX={vj}")

# Check what ZC (zero crossing) values differ
print("\n=== STEP 200 ZERO CROSSING ===")
for k in sorted(r200p.keys()):
    if "zc_" in k.lower() or "zero_cross" in k.lower():
        vp = r200p.get(k, "N/A")
        vj = r200j.get(k, "N/A")
        print(f"  {k}: PY={vp}  JX={vj}")
