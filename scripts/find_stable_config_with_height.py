"""Find stable configuration by adjusting both leg angles AND base height."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('FINDING STABLE CONFIGURATION WITH PROPER BASE HEIGHT')
print('='*80)

def test_configuration(hip_pitch, knee, base_height):
    """Test a configuration and return stability metrics."""
    # Set configuration
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, base_height]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:12] = [0, 0, hip_pitch, knee, 0]
    d.qpos[12:17] = [0, 0, hip_pitch, knee, 0]

    # Forward kinematics
    mujoco.mj_forward(m, d)

    # Check initial state
    if d.ncon == 0:
        return None, "No contact"

    # Check for severe penetration (contact distance < -1mm)
    max_penetration = 0.0
    for i in range(d.ncon):
        if d.contact[i].dist < max_penetration:
            max_penetration = d.contact[i].dist

    if max_penetration < -0.002:  # More than 2mm penetration
        return None, f"Severe penetration ({max_penetration*1000:.1f}mm)"

    # Check initial contact forces
    total_fx = 0.0
    total_fz = 0.0
    for i in range(d.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        total_fx += force[0]
        total_fz += force[2]

    if abs(total_fx) > 1000:  # Absurd horizontal force
        return None, f"Excessive force ({total_fx:.0f}N)"

    # Simulate 20 steps
    max_vel = 0.0
    for step in range(20):
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        max_vel = max(max_vel, np.max(np.abs(d.qvel)))

        if d.ncon == 0:  # Lost contact
            return None, "Lost contact"

    # Check final state
    final_com_z = d.subtree_com[1][2]

    # Stability score (lower is better)
    stability_score = max_vel

    return stability_score, "OK"

# Test configurations with adjusted base heights
print('\nTesting configurations:')
print('Hip_pitch | Knee | Base_z | Total | Stability | Status')
print('-'*80)

best_config = None
best_score = float('inf')

# Try different leg configurations
for hip_pitch in [0.95, 0.70, 0.50, 0.40, 0.30, 0.25, 0.20]:
    for knee in [1.70, 1.40, 1.20, 1.00, 0.80, 0.60, 0.50, 0.40]:
        # Try different base heights around 0.545
        for base_height in [0.545, 0.500, 0.450, 0.400, 0.380, 0.360]:
            score, status = test_configuration(hip_pitch, knee, base_height)

            if score is not None and score < best_score:
                best_score = score
                best_config = (hip_pitch, knee, base_height)

                total_bend = np.degrees(hip_pitch + knee)
                print(f'{np.degrees(hip_pitch):9.1f}° | {np.degrees(knee):4.1f}° | {base_height:.3f} | {total_bend:5.1f}° | {score:9.4f} | {status}')

print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

if best_config:
    hip_pitch, knee, base_height = best_config
    print(f'\nBest stable configuration found:')
    print(f'  hip_pitch   = {hip_pitch:.4f} rad = {np.degrees(hip_pitch):.1f}°')
    print(f'  knee        = {knee:.4f} rad = {np.degrees(knee):.1f}°')
    print(f'  base_height = {base_height:.3f} m')
    print(f'  Total bend  = {np.degrees(hip_pitch + knee):.1f}°')
    print(f'  Stability score = {best_score:.4f} rad/s')
    print('')
    print('Update keyframe in wheeled_biped_real.xml:')
    print(f'  qpos="0 0 {base_height:.3f}')
    print(f'        1 0 0 0')
    print(f'        0 0 {hip_pitch:.2f} {knee:.2f} 0')
    print(f'        0 0 {hip_pitch:.2f} {knee:.2f} 0"')
else:
    print('\nNo stable configuration found.')
    print('The robot may require active control for stabilization.')
