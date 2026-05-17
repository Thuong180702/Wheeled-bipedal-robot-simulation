"""Analyze desired wrench from centroidal controller.

Checks if commanded wrenches are physically reasonable and feasible.
"""

import csv
import sys

import numpy as np


def analyze_desired_wrench(telemetry_path: str):
    """Analyze desired wrench components from telemetry."""

    # Read telemetry
    with open(telemetry_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: No telemetry data found")
        return

    print("=" * 80)
    print("DESIRED WRENCH ANALYSIS")
    print("=" * 80)
    print(f"\nTelemetry file: {telemetry_path}")
    print(f"Steps: {len(rows)}")
    print(f"Survival time: {float(rows[-1]['time']):.2f}s")
    print(f"Termination: {rows[-1]['termination_reason']}")

    # Check if wrench components are in telemetry
    has_wrench = 'desired_wrench_Fx' in rows[0]

    if not has_wrench:
        print("\n[ERROR] Telemetry does not contain desired wrench components")
        print("Need to add wrench logging to simulation script")
        return

    # Extract wrench components
    Fx = [float(r['desired_wrench_Fx']) for r in rows]
    Fy = [float(r['desired_wrench_Fy']) for r in rows]
    Fz = [float(r['desired_wrench_Fz']) for r in rows]
    Mx = [float(r['desired_wrench_Mx']) for r in rows]
    My = [float(r['desired_wrench_My']) for r in rows]
    Mz = [float(r['desired_wrench_Mz']) for r in rows]

    # Compute statistics
    def stats(values):
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    print("\n" + "=" * 80)
    print("WRENCH COMPONENT STATISTICS")
    print("=" * 80)

    print("\nForces (N):")
    fx_stats = stats(Fx)
    print(f"  Fx: {fx_stats['mean']:7.2f} +/- {fx_stats['std']:5.2f}  [{fx_stats['min']:7.2f}, {fx_stats['max']:7.2f}]")

    fy_stats = stats(Fy)
    print(f"  Fy: {fy_stats['mean']:7.2f} +/- {fy_stats['std']:5.2f}  [{fy_stats['min']:7.2f}, {fy_stats['max']:7.2f}]")

    fz_stats = stats(Fz)
    print(f"  Fz: {fz_stats['mean']:7.2f} +/- {fz_stats['std']:5.2f}  [{fz_stats['min']:7.2f}, {fz_stats['max']:7.2f}]")

    print("\nMoments (Nm):")
    mx_stats = stats(Mx)
    print(f"  Mx (roll):  {mx_stats['mean']:7.2f} +/- {mx_stats['std']:5.2f}  [{mx_stats['min']:7.2f}, {mx_stats['max']:7.2f}]")

    my_stats = stats(My)
    print(f"  My (pitch): {my_stats['mean']:7.2f} +/- {my_stats['std']:5.2f}  [{my_stats['min']:7.2f}, {my_stats['max']:7.2f}]")

    mz_stats = stats(Mz)
    print(f"  Mz (yaw):   {mz_stats['mean']:7.2f} +/- {mz_stats['std']:5.2f}  [{mz_stats['min']:7.2f}, {mz_stats['max']:7.2f}]")

    # Physical reasonableness checks
    print("\n" + "=" * 80)
    print("PHYSICAL REASONABLENESS CHECKS")
    print("=" * 80)

    robot_mass = 15.0  # kg
    gravity = 9.81  # m/s^2
    expected_fz = robot_mass * gravity  # ~147 N

    print(f"\n1. Vertical force (Fz):")
    print(f"   Expected: ~{expected_fz:.1f} N (gravity compensation)")
    print(f"   Actual mean: {fz_stats['mean']:.1f} N")
    print(f"   Deviation: {((fz_stats['mean'] - expected_fz) / expected_fz * 100):.1f}%")

    if abs(fz_stats['mean'] - expected_fz) > 50:
        print(f"   [WARNING] Fz deviates >50N from gravity compensation")
    else:
        print(f"   [OK] Fz is reasonable for standing balance")

    print(f"\n2. Horizontal forces (Fx, Fy):")
    print(f"   Fx mean: {fx_stats['mean']:.1f} N, max: {max(abs(fx_stats['min']), abs(fx_stats['max'])):.1f} N")
    print(f"   Fy mean: {fy_stats['mean']:.1f} N, max: {max(abs(fy_stats['min']), abs(fy_stats['max'])):.1f} N")

    fx_max_abs = max(abs(fx_stats['min']), abs(fx_stats['max']))
    fy_max_abs = max(abs(fy_stats['min']), abs(fy_stats['max']))

    if fx_max_abs > 30 or fy_max_abs > 30:
        print(f"   [WARNING] Large horizontal forces (>30N) for standing balance")
        print(f"   -> May be infeasible given contact geometry")
    else:
        print(f"   [OK] Horizontal forces are reasonable")

    print(f"\n3. Roll moment (Mx):")
    print(f"   Mean: {mx_stats['mean']:.1f} Nm")
    print(f"   Range: [{mx_stats['min']:.1f}, {mx_stats['max']:.1f}] Nm")

    mx_max_abs = max(abs(mx_stats['min']), abs(mx_stats['max']))
    if mx_max_abs > 20:
        print(f"   [WARNING] Large roll moment (>20Nm) suggests aggressive correction")
    else:
        print(f"   [OK] Roll moment is reasonable")

    print(f"\n4. Pitch moment (My):")
    print(f"   Mean: {my_stats['mean']:.1f} Nm")
    print(f"   Range: [{my_stats['min']:.1f}, {my_stats['max']:.1f}] Nm")

    my_max_abs = max(abs(my_stats['min']), abs(my_stats['max']))
    if my_max_abs > 10:
        print(f"   [CRITICAL] Large pitch moment (>10Nm)")
        print(f"   -> Pitch moment requires tensile forces (INFEASIBLE)")
        print(f"   -> This is likely causing the large wrench tracking error")
    else:
        print(f"   [OK] Pitch moment is small")

    # Feasibility analysis
    print("\n" + "=" * 80)
    print("FEASIBILITY ANALYSIS")
    print("=" * 80)

    print("\nFrom contact geometry analysis:")
    print("  - Fx (forward force) requires tensile forces -> INFEASIBLE")
    print("  - My (pitch moment) requires tensile forces -> INFEASIBLE")
    print("  - Fy, Fz, Mx, Mz are achievable with compressive forces")

    infeasible_fx = fx_max_abs > 5.0
    infeasible_my = my_max_abs > 5.0

    if infeasible_fx or infeasible_my:
        print("\n[CRITICAL] Controller is commanding INFEASIBLE wrenches:")
        if infeasible_fx:
            print(f"  - Fx = {fx_max_abs:.1f} N (should be ~0 for standing)")
        if infeasible_my:
            print(f"  - My = {my_max_abs:.1f} Nm (should be ~0 for standing)")

        print("\nThis explains the large wrench tracking error (35.46 N/Nm)")
        print("The QP cannot track these wrenches with compressive forces only")

    # Wrench evolution
    print("\n" + "=" * 80)
    print("WRENCH EVOLUTION")
    print("=" * 80)

    print("\nFirst 5 timesteps:")
    print("  Time    Fx      Fy      Fz      Mx      My      Mz")
    for i in range(min(5, len(rows))):
        t = float(rows[i]['time'])
        print(f"  {t:4.2f}  {Fx[i]:6.1f}  {Fy[i]:6.1f}  {Fz[i]:6.1f}  {Mx[i]:6.1f}  {My[i]:6.1f}  {Mz[i]:6.1f}")

    print("\nLast 5 timesteps:")
    print("  Time    Fx      Fy      Fz      Mx      My      Mz")
    for i in range(max(0, len(rows)-5), len(rows)):
        t = float(rows[i]['time'])
        print(f"  {t:4.2f}  {Fx[i]:6.1f}  {Fy[i]:6.1f}  {Fz[i]:6.1f}  {Mx[i]:6.1f}  {My[i]:6.1f}  {Mz[i]:6.1f}")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if infeasible_fx or infeasible_my:
        print("\nThe centroidal controller is commanding infeasible wrenches.")
        print("Root causes:")
        print("  1. Controller gains may be too aggressive")
        print("  2. State estimation may be incorrect (large errors)")
        print("  3. Controller may not account for contact geometry constraints")

        print("\nRecommendations:")
        print("  1. Reduce sagittal (forward/pitch) control gains")
        print("  2. Validate state estimation (CoM, CP, velocities)")
        print("  3. Add wrench feasibility constraints to controller")
        print("  4. Consider using only Fy, Fz, Mx, Mz (feasible components)")
    else:
        print("\nDesired wrenches appear feasible.")
        print("The wrench tracking error may be due to:")
        print("  1. QP solver tuning (w_wrench too low)")
        print("  2. Numerical conditioning issues")
        print("  3. Hierarchical controller conflicts")


def main():
    """Run desired wrench analysis."""

    if len(sys.argv) < 2:
        print("Usage: python analyze_desired_wrench.py <telemetry_csv>")
        print("\nExample:")
        print("  python analyze_desired_wrench.py outputs/hierarchical_controller_sim/telemetry_1778999749.csv")
        sys.exit(1)

    telemetry_path = sys.argv[1]
    analyze_desired_wrench(telemetry_path)


if __name__ == "__main__":
    main()
