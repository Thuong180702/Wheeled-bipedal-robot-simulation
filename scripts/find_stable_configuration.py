"""Find stable standing configuration through simulation.

Tests different joint configurations to find one that maintains balance
with zero control input (passive stability).
"""

import mujoco
import numpy as np

def test_configuration(mj_model, hip_pitch, knee, duration=2.0):
    """Test if a joint configuration is stable.

    Returns:
        (stable, final_pitch, final_height)
    """
    mj_data = mujoco.MjData(mj_model)

    # Set joint configuration
    mj_data.qpos[7:17] = np.array([
        0.0, 0.0, hip_pitch, knee, 0.0,  # left leg
        0.0, 0.0, hip_pitch, knee, 0.0,  # right leg
    ])

    # Set base height to let robot settle
    mj_data.qpos[2] = 0.75  # Start high

    # Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)

    # Simulate with zero control
    steps = int(duration / mj_model.opt.timestep)
    for _ in range(steps):
        mj_data.ctrl[:] = 0
        mujoco.mj_step(mj_model, mj_data)

    # Check final state
    quat = mj_data.qpos[3:7]
    pitch = float(np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                              1 - 2 * (quat[1]**2 + quat[2]**2)))
    height = mj_data.qpos[2]

    # Stable if pitch < 10 degrees and height > 0.4m
    stable = abs(pitch) < 0.174 and height > 0.4

    return stable, np.rad2deg(pitch), height

def main():
    print("=" * 80)
    print("Searching for Stable Standing Configuration")
    print("=" * 80)

    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)

    print("\nTesting configurations...")
    print(f"{'hip_pitch':>10} {'knee':>10} {'stable':>10} {'pitch':>10} {'height':>10}")
    print("-" * 60)

    best_config = None
    best_pitch_error = float('inf')

    # Search grid
    for hip_pitch in np.linspace(0.2, 0.8, 13):  # 0.2 to 0.8 rad
        for knee in np.linspace(0.4, 1.6, 13):  # 0.4 to 1.6 rad
            stable, pitch_deg, height = test_configuration(mj_model, hip_pitch, knee)

            pitch_error = abs(pitch_deg)
            if pitch_error < best_pitch_error:
                best_pitch_error = pitch_error
                best_config = (hip_pitch, knee, pitch_deg, height)

            if stable or pitch_error < 15:  # Show promising configs
                status = "STABLE" if stable else f"pitch={pitch_deg:.1f}°"
                print(f"{hip_pitch:10.3f} {knee:10.3f} {status:>10} {pitch_deg:10.1f} {height:10.3f}")

    print("\n" + "=" * 80)
    if best_config:
        hp, k, p, h = best_config
        print(f"Best configuration found:")
        print(f"  hip_pitch = {hp:.3f} rad ({np.rad2deg(hp):.1f}°)")
        print(f"  knee = {k:.3f} rad ({np.rad2deg(k):.1f}°)")
        print(f"  Final pitch = {p:.1f}°")
        print(f"  Final height = {h:.3f}m")
        print(f"\nUpdate keyframe in wheeled_biped_real.xml:")
        print(f'  qpos="0 0 {h:.2f}')
        print(f'        1 0 0 0')
        print(f'        0 0 {hp:.1f} {k:.1f} 0')
        print(f'        0 0 {hp:.1f} {k:.1f} 0"')
    else:
        print("No stable configuration found in search range")
    print("=" * 80)

if __name__ == "__main__":
    main()
