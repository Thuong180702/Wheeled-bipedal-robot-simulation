"""Show all tested configurations to understand why current one is best."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('COMPREHENSIVE CONFIGURATION TEST')
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

    # Check for severe penetration
    max_penetration = 0.0
    for i in range(d.ncon):
        if d.contact[i].dist < max_penetration:
            max_penetration = d.contact[i].dist

    if max_penetration < -0.002:
        return None, f"Penetration {max_penetration*1000:.1f}mm"

    # Check initial contact forces
    total_fx = 0.0
    for i in range(d.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        total_fx += force[0]

    if abs(total_fx) > 1000:
        return None, f"Force {total_fx:.0f}N"

    # Simulate 20 steps
    max_vel = 0.0
    for step in range(20):
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        max_vel = max(max_vel, np.max(np.abs(d.qvel)))

        if d.ncon == 0:
            return None, f"Lost@step{step}"

    return max_vel, "OK"

# Test a grid of configurations
print('\nTesting grid of configurations:')
print('(showing only valid configurations that maintain contact)')
print('')
print('Hip_pitch | Knee | Base_z | Total | Max_vel | Status')
print('-'*80)

results = []

for hip_pitch in [0.95, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]:
    for knee in [1.70, 1.50, 1.30, 1.10, 0.90, 0.70, 0.50]:
        for base_height in [0.545, 0.520, 0.500, 0.480, 0.460, 0.440, 0.420, 0.400]:
            score, status = test_configuration(hip_pitch, knee, base_height)

            if score is not None:
                total_bend = np.degrees(hip_pitch + knee)
                results.append((score, hip_pitch, knee, base_height, total_bend, status))

# Sort by stability score (lower is better)
results.sort(key=lambda x: x[0])

# Show top 20 most stable configurations
print('\nTop 20 most stable configurations:')
for i, (score, hp, kn, bh, tb, st) in enumerate(results[:20]):
    print(f'{i+1:2d}. {np.degrees(hp):5.1f}° | {np.degrees(kn):5.1f}° | {bh:.3f} | {tb:5.1f}° | {score:7.4f} | {st}')

print(f'\nTotal valid configurations found: {len(results)}')
print(f'Total configurations tested: {7 * 7 * 8} = {7*7*8}')

print('\n' + '='*80)
print('ANALYSIS')
print('='*80)

if len(results) > 0:
    best_score, best_hp, best_kn, best_bh, best_tb, _ = results[0]
    print(f'\nBest configuration:')
    print(f'  hip_pitch   = {best_hp:.4f} rad = {np.degrees(best_hp):.1f}°')
    print(f'  knee        = {best_kn:.4f} rad = {np.degrees(best_kn):.1f}°')
    print(f'  base_height = {best_bh:.3f} m')
    print(f'  Total bend  = {best_tb:.1f}°')
    print(f'  Max velocity after 20 steps = {best_score:.4f} rad/s')
    print('')

    if best_score > 0.5:
        print('Even the best configuration is UNSTABLE (develops >0.5 rad/s velocity).')
        print('This robot design requires ACTIVE CONTROL for stabilization.')
        print('A purely passive stable standing configuration does not exist.')
    else:
        print('Found a stable configuration!')
        print('')
        print('Update keyframe in wheeled_biped_real.xml:')
        print(f'  qpos="0 0 {best_bh:.3f}')
        print(f'        1 0 0 0')
        print(f'        0 0 {best_hp:.2f} {best_kn:.2f} 0')
        print(f'        0 0 {best_hp:.2f} {best_kn:.2f} 0"')
else:
    print('\nNo valid configurations found.')
    print('All tested configurations either:')
    print('  - Lose contact immediately')
    print('  - Have severe penetration')
    print('  - Generate excessive forces')
