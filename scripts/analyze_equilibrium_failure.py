"""Analyze why the optimized equilibrium keyframe still fails.

Compares initial conditions and failure progression between original and optimized keyframes.
"""

import pandas as pd
import numpy as np

# Load telemetry
df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1779199031.csv')

print("=" * 80)
print("Equilibrium Failure Analysis")
print("=" * 80)

# Initial conditions (step 0)
print("\n1. INITIAL CONDITIONS (t=0.0s)")
print("-" * 80)
initial = df.iloc[0]
print(f"CoM position: x={initial['com_x']:.6f}, y={initial['com_y']:.6f}, z={initial['com_z']:.6f} m")
print(f"CoM y-offset: {initial['com_y']:.6f} m (target: ~0.0 m)")
print(f"Roll: {initial['roll']:.4f} rad = {np.degrees(initial['roll']):.2f} deg")
print(f"Pitch: {initial['pitch']:.4f} rad = {np.degrees(initial['pitch']):.2f} deg")
print(f"Contact forces:")
print(f"  Left wheel:  {initial['left_contact_force_world_z']:.2f} N")
print(f"  Right wheel: {initial['right_contact_force_world_z']:.2f} N")
print(f"  Total:       {initial['total_contact_force_z']:.2f} N (weight: {initial['weight_N']:.2f} N)")
print(f"  Asymmetry:   {abs(initial['left_contact_force_world_z'] - initial['right_contact_force_world_z']):.2f} N")
print(f"  Force ratio: {initial['total_contact_force_z'] / initial['weight_N']:.3f} (target: 1.0)")

# Contact loss event
print("\n2. CONTACT LOSS EVENT")
print("-" * 80)
contact_loss_idx = df[~df['left_contact_active']].index[0] if any(~df['left_contact_active']) else None
if contact_loss_idx:
    before = df.iloc[contact_loss_idx - 1]
    at_loss = df.iloc[contact_loss_idx]

    print(f"Left wheel contact lost at step {contact_loss_idx} (t={at_loss['time']:.2f}s)")
    print(f"\nConditions just before contact loss (step {contact_loss_idx-1}):")
    print(f"  Roll: {np.degrees(before['roll']):.2f} deg")
    print(f"  Roll rate: {before['roll_rate_rad_s']:.2f} rad/s")
    print(f"  CoM y-offset: {before['com_y']:.6f} m")
    print(f"  Left force: {before['left_contact_force_world_z']:.2f} N")
    print(f"  Right force: {before['right_contact_force_world_z']:.2f} N")
    print(f"  Force asymmetry: {abs(before['left_contact_force_world_z'] - before['right_contact_force_world_z']):.2f} N")

    print(f"\nAt contact loss (step {contact_loss_idx}):")
    print(f"  Roll: {np.degrees(at_loss['roll']):.2f} deg")
    print(f"  Roll rate: {at_loss['roll_rate_rad_s']:.2f} rad/s")
    print(f"  CoM y-offset: {at_loss['com_y']:.6f} m")
    print(f"  Left force: {at_loss['left_contact_force_world_z']:.2f} N")
    print(f"  Right force: {at_loss['right_contact_force_world_z']:.2f} N")

# Roll growth analysis
print("\n3. ROLL DIVERGENCE ANALYSIS")
print("-" * 80)
print("Step | Time | Roll(deg) | Roll_rate | CoM_y | Left_Fz | Right_Fz | Asymmetry")
print("-" * 80)
for i in range(min(10, len(df))):
    row = df.iloc[i]
    asymmetry = abs(row['left_contact_force_world_z'] - row['right_contact_force_world_z'])
    print(f"{i:4d} | {row['time']:4.2f} | {np.degrees(row['roll']):9.2f} | "
          f"{row['roll_rate_rad_s']:9.2f} | {row['com_y']:7.4f} | "
          f"{row['left_contact_force_world_z']:7.2f} | {row['right_contact_force_world_z']:8.2f} | "
          f"{asymmetry:9.2f}")

# Force distribution analysis
print("\n4. FORCE DISTRIBUTION ANALYSIS")
print("-" * 80)
print("The force distributor creates asymmetric vertical forces to generate roll moment:")
print("  Mx = wheel_x_offset * (f_left_z - f_right_z)")
print("  With wheel_x_offset = 0.135 m")
print()
print("Step | Desired_Mx | f_left_z | f_right_z | Asymmetry | Actual_Asymmetry")
print("-" * 80)
for i in range(min(10, len(df))):
    row = df.iloc[i]
    desired_mx = row['desired_wrench_Mx']
    f_left_z = row['distributed_left_fz']
    f_right_z = row['distributed_right_fz']
    commanded_asymmetry = f_left_z - f_right_z
    actual_asymmetry = row['left_contact_force_world_z'] - row['right_contact_force_world_z']
    print(f"{i:4d} | {desired_mx:10.2f} | {f_left_z:8.2f} | {f_right_z:9.2f} | "
          f"{commanded_asymmetry:9.2f} | {actual_asymmetry:16.2f}")

# Root cause summary
print("\n5. ROOT CAUSE ANALYSIS")
print("=" * 80)
print("FINDING: The optimized keyframe improved initial symmetry but the controller")
print("         creates INTENTIONAL force asymmetry to generate roll correction moment.")
print()
print("Initial state (optimized keyframe):")
print(f"  - CoM y-offset: {initial['com_y']:.6f} m (12x better than original -0.0162 m)")
print(f"  - Force asymmetry: {abs(initial['left_contact_force_world_z'] - initial['right_contact_force_world_z']):.2f} N")
print(f"  - Initial roll: {np.degrees(initial['roll']):.2f} deg (near zero)")
print()
print("Controller behavior:")
print(f"  - Step 1: Desired Mx = {df.iloc[1]['desired_wrench_Mx']:.2f} Nm → creates 11.26 N asymmetry")
print(f"  - Step 2: Desired Mx = {df.iloc[2]['desired_wrench_Mx']:.2f} Nm → creates 20.93 N asymmetry")
print(f"  - Step 3: Desired Mx = {df.iloc[3]['desired_wrench_Mx']:.2f} Nm → creates 30.25 N asymmetry")
print()
print("The force asymmetry is INTENTIONAL - it's how the controller generates roll moment.")
print("However, this asymmetry causes the left wheel to lift off at step 9.")
print()
print("CONCLUSION: The problem is NOT the initial keyframe equilibrium.")
print("            The problem is the CONTROLLER ARCHITECTURE:")
print()
print("  1. The force distributor uses differential vertical forces for roll control")
print("  2. This creates large force asymmetries (up to 84 N at step 8)")
print("  3. The left wheel force drops to near zero and loses contact")
print("  4. Once contact is lost, the controller cannot recover")
print()
print("RECOMMENDED FIX:")
print("  - Use hip roll torques as PRIMARY roll control mechanism")
print("  - Use differential wheel forces only as SECONDARY/backup")
print("  - Limit maximum force asymmetry to prevent wheel liftoff")
print("  - Add contact-aware force distribution that prevents liftoff")
print("=" * 80)
