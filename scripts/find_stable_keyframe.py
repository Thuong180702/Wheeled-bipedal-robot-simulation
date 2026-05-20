"""Find a stable keyframe configuration by testing different joint angles."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('SEARCHING FOR STABLE KEYFRAME CONFIGURATION')
print('='*80)

# Test different configurations
configs = [
    # (hip_pitch, knee, description)
    (0.95, 1.70, "Current (UNSTABLE)"),
    (0.50, 1.00, "Moderate bend"),
    (0.40, 0.80, "Less bend"),
    (0.30, 0.60, "Straighter"),
    (0.20, 0.40, "Very straight"),
    (0.10, 0.20, "Nearly straight"),
]

print('\nTesting configurations:')
print('Hip_pitch | Knee | Total | CoM_z | Stable? | Description')
print('-'*80)

best_config = None
best_stability = float('inf')

for hip_pitch, knee, desc in configs:
    # Set configuration
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, 0.545]  # Base position
    d.qpos[3:7] = [1, 0, 0, 0]   # Base orientation (quaternion)
    d.qpos[7:12] = [0, 0, hip_pitch, knee, 0]  # Left leg
    d.qpos[12:17] = [0, 0, hip_pitch, knee, 0]  # Right leg

    # Forward kinematics
    mujoco.mj_forward(m, d)

    # Check if configuration is valid (no penetration, has contacts)
    if d.ncon == 0:
        print(f'{np.degrees(hip_pitch):9.1f}° | {np.degrees(knee):4.1f}° | {np.degrees(hip_pitch+knee):5.1f}° | N/A   | NO      | {desc} (no contact)')
        continue

    # Get CoM height
    com_z = d.subtree_com[1][2]

    # Simulate 20 steps with zero control
    max_vel = 0.0
    for step in range(20):
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        max_vel = max(max_vel, np.max(np.abs(d.qvel)))

    # Check stability
    final_contacts = d.ncon
    final_com_z = d.subtree_com[1][2]
    height_drop = com_z - final_com_z

    is_stable = (final_contacts > 0 and max_vel < 0.3 and height_drop < 0.01)
    stability_score = max_vel + height_drop * 10  # Lower is better

    status = "YES" if is_stable else "NO"
    print(f'{np.degrees(hip_pitch):9.1f}° | {np.degrees(knee):4.1f}° | {np.degrees(hip_pitch+knee):5.1f}° | {com_z:.3f} | {status:7s} | {desc}')

    if is_stable and stability_score < best_stability:
        best_config = (hip_pitch, knee, desc)
        best_stability = stability_score

print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

if best_config:
    hip_pitch, knee, desc = best_config
    print(f'\nBest stable configuration found: {desc}')
    print(f'  hip_pitch = {hip_pitch:.4f} rad = {np.degrees(hip_pitch):.1f}°')
    print(f'  knee      = {knee:.4f} rad = {np.degrees(knee):.1f}°')
    print(f'  Total bend = {np.degrees(hip_pitch + knee):.1f}°')
    print('')
    print('Update keyframe in wheeled_biped_real.xml:')
    print(f'  qpos="0 0 0.545')
    print(f'        1 0 0 0')
    print(f'        0 0 {hip_pitch:.2f} {knee:.2f} 0')
    print(f'        0 0 {hip_pitch:.2f} {knee:.2f} 0"')
else:
    print('\nNo stable configuration found in tested range.')
    print('May need to:')
    print('  1. Test even straighter legs')
    print('  2. Adjust base height')
    print('  3. Add active control for stabilization')
