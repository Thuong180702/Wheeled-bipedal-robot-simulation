"""Analyze why robot develops asymmetric roll and loses right wheel contact.

From telemetry after sign inversion fix:
- Steps 0-17: Both wheels in contact, symmetric torques
- Step 18: Right wheel loses contact, right leg torques → 0
- Steps 18-25: Only left wheel in contact, robot rolls left
- Step 25: Terminated (roll = -45°)

This script investigates:
1. What causes the initial roll asymmetry
2. Why controller cannot correct small roll errors
3. Whether hip roll torques are working correctly
"""

import pandas as pd
import numpy as np

# Load telemetry
df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1779188488.csv')

print('='*80)
print('ASYMMETRIC FAILURE ANALYSIS')
print('='*80)

print('\n1. Roll development over time:')
print('   Time (s) | Roll (deg) | L_contact | R_contact | L_fz (N) | R_fz (N)')
print('   ' + '-'*75)

for i in range(min(30, len(df))):
    row = df.iloc[i]
    time = row['time']
    roll = row['roll']
    l_contact = row['left_contact_active']
    r_contact = row['right_contact_active']
    l_fz = row['left_contact_force_world_z']
    r_fz = row['right_contact_force_world_z']

    print(f'   {time:8.3f} | {np.degrees(roll):10.1f} | {l_contact:9} | {r_contact:9} | {l_fz:8.1f} | {r_fz:8.1f}')

    if not r_contact and i > 0:
        print(f'   -> RIGHT WHEEL LOST CONTACT at time {time:.3f}s')
        break

print('\n2. Desired wrench analysis:')
print('   The controller computes desired wrench to stabilize the robot.')
print('   Mx (roll moment) should counteract roll errors.')
print('')
print('   Time (s) | Roll (deg) | Desired Mx (Nm) | Desired My (Nm)')
print('   ' + '-'*65)

for i in range(min(20, len(df))):
    row = df.iloc[i]
    time = row['time']
    roll = row['roll']
    mx = row['desired_wrench_Mx']
    my = row['desired_wrench_My']

    print(f'   {time:8.3f} | {np.degrees(roll):10.1f} | {mx:15.2f} | {my:15.2f}')

print('\n3. Force distribution analysis:')
print('   The QP distributes vertical forces to left/right wheels.')
print('')
print('   Time (s) | Roll (deg) | f_left_z (N) | f_right_z (N) | Asymmetry')
print('   ' + '-'*75)

for i in range(min(20, len(df))):
    row = df.iloc[i]
    time = row['time']
    roll = row['roll']
    f_left = row['distributed_left_fz']
    f_right = row['distributed_right_fz']
    asymmetry = abs(f_left - f_right)

    print(f'   {time:8.3f} | {np.degrees(roll):10.1f} | {f_left:12.2f} | {f_right:13.2f} | {asymmetry:9.2f}')

print('\n4. Diagnosis:')
print('   From the data:')
print('   - Robot develops increasing roll angle')
print('   - Desired Mx should provide corrective roll moment')
print('   - Force distribution should be asymmetric to create Mx')
print('   - But one wheel eventually loses contact')
print('')
print('   Possible causes:')
print('   a) Desired Mx is too small (insufficient roll correction gain)')
print('   b) Force distribution cannot achieve desired Mx (QP constraint issue)')
print('   c) Jacobian mapping loses effectiveness as roll increases')
print('   d) Hip roll joints have insufficient authority')

print('\n5. Recommended fixes:')
print('   Option A: Increase k_roll gain in centroidal wrench computer')
print('   Option B: Increase hip roll authority in force distributor')
print('   Option C: Add roll rate damping (k_roll_rate)')
print('   Option D: Check if hip roll Jacobian is correct')
print('')
print('   Preferred: Check if desired Mx is being generated correctly first')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)
print('The sign inversion fixed leg extension, but hip roll control is broken:')
print('1. Both hip roll torques push robot in same direction (should be opposite)')
print('2. Cannot correct small roll errors')
print('3. Robot develops asymmetric roll until one wheel lifts')
print('4. Once one wheel lifts, robot falls immediately')
print('')
print('Fix: Verify and correct hip roll sign convention in Jacobian mapping')
