"""Check all G candidate results with full metrics."""
import csv, math, glob

def check(label, path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    hy = max(float(r["hip_yaw_abs_max"]) for r in rows)
    pitch = max(abs(float(r["pitch_error"])) for r in rows) * 180 / math.pi
    sup = max(abs(float(r.get("support_position_error_scaled_m", 0))) for r in rows)
    roll_rms = (sum(float(r["roll_y"])**2 for r in rows)/n)**0.5 * 180/math.pi
    yaw = max(abs(float(r.get("euler_yaw_z", 0))) for r in rows)
    md_tau = max(abs(float(r["mode_hip_yaw_div_tau_left"])) for r in rows)
    final_tau = max(abs(float(r.get("l_hip_yaw_tau_shape_final", 0))) for r in rows)
    sat = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True")
    gate = sum(float(r.get("mode_hip_yaw_div_height_gate", 1)) for r in rows) / n
    falls = sum(1 for r in rows if r.get("terminated", "False") == "True")
    sign_ok = sum(1 for r in rows if abs(float(r["mode_hip_yaw_div_error"])) < 1e-9 or float(r["mode_hip_yaw_div_error"]) * float(r["mode_hip_yaw_div_tau_left"]) <= 0)
    common = max(abs(float(r.get("hip_yaw_common_error_rad", 0))) for r in rows)
    div_ = max(abs(float(r.get("hip_yaw_divergence_error_rad", 0))) for r in rows)
    yaw_left_max = max(abs(float(r.get("yaw_controller_tau_hip_yaw_left", 0))) for r in rows)

    p = "PASS" if hy <= 0.35 else "FAIL"
    r_str = f"{label}: hy={hy:.4f} {p} sup={sup:.4f} pitch={pitch:.2f} roll={roll_rms:.2f} yaw={yaw:.3f} gate={gate:.3f} sat={sat}/{n} sign%={100*sign_ok/n:.1f} com={common:.4f} div={div_:.4f} yL={yaw_left_max:.3f} falls={falls}"
    print(r_str)

base = "outputs/d5_high_height_mode_div_gate_and_common_mode_coupling_fix/sweep"
d5 = f"{base}/D5_large_push_high"
d4 = f"{base}/D4_medium_push_low"

print("=== D4 references ===")
check("D4_D_baseline", "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D4_medium_push_low/D_baseline/telemetry_1782210053.csv")
check("D4_F6", "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_kp10_mt75/telemetry_1782216842.csv")

print("\n=== D4 G candidates ===")
tg = glob.glob(f"{d4}/G1_sg060/telemetry_*.csv")
if tg: check("D4_G1_sg060", tg[0])
tg = glob.glob(f"{d4}/G1_sg070/telemetry_*.csv")
if tg: check("D4_G1_sg070", tg[0])
tg = glob.glob(f"{d4}/G1_sg080/telemetry_*.csv")
if tg: check("D4_G1_sg080", tg[0])

print("\n=== D5 references ===")
check("D5_D_baseline", "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high/D_baseline/telemetry_1782210164.csv")
check("D5_F6+sg050", "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg50_D5/telemetry_1782217344.csv")

print("\n=== D5 G candidates ===")
tg = glob.glob(f"{d5}/G1_sg060/telemetry_*.csv")
if tg: check("D5_G1_sg060", tg[0])
tg = glob.glob(f"{d5}/G1_sg070/telemetry_*.csv")
if tg: check("D5_G1_sg070", tg[0])
tg = glob.glob(f"{d5}/G1_sg080/telemetry_*.csv")
if tg: check("D5_G1_sg080", tg[0])
tg = glob.glob(f"{d5}/G1_sg085/telemetry_*.csv")
if tg: check("D5_G1_sg085", tg[0])
tg = glob.glob(f"{d5}/G1_sg090/telemetry_*.csv")
if tg: check("D5_G1_sg090", tg[0])
tg = glob.glob(f"{d5}/G2_mt85_sg080/telemetry_*.csv")
if tg: check("D5_G2_mt85_sg080", tg[0])
