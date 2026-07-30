#!/usr/bin/env python3
"""Script to re-optimize gains for the Direct Torque LQR baseline.

Sweeps LQR Q/R matrices (with direct torque plant dynamics, removing tau_s lag),
hip-roll PD gains, and yaw PD gains to find the maximum possible standing survival time
and stability for the Direct Torque LQR baseline.
"""

from __future__ import annotations
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.controllers.fair_lqr_torque import FairLQRTorqueController
from wheeled_biped.utils.config import get_model_path


def eval_dt_lqr(lqr_q, lqr_r, tau_s, kp_roll, kd_roll, kp_yaw, kd_yaw, num_episodes=5, num_steps=1000):
    model_path = str(get_model_path())
    
    from scipy.linalg import solve_continuous_are
    g = 9.81
    l_com = 0.54
    r_wheel = 0.06
    m_tot = 8.1
    
    if tau_s > 0:
        a_mat = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [g / l_com, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        b_mat = np.array([[0.0], [-r_wheel / (l_com * tau_s)], [r_wheel], [0.0]])
    else:
        # Direct torque plant without servo lag
        a_mat = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [g / l_com, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        b_mat = np.array([[0.0], [-1.0 / (m_tot * l_com**2)], [r_wheel / m_tot], [0.0]])

    q_mat = np.diag(lqr_q)
    r_mat = np.array([[lqr_r]])
    p_mat = solve_continuous_are(a_mat, b_mat, q_mat, r_mat)
    k_lqr = (np.linalg.inv(r_mat) @ b_mat.T @ p_mat).flatten()
    
    ctrl = FairLQRTorqueController(
        model_path=model_path,
        lqr_q=tuple(lqr_q),
        lqr_r=lqr_r,
        kp_roll=kp_roll,
        kd_roll=kd_roll,
        kp_yaw=kp_yaw,
        kd_yaw=kd_yaw,
    )
    ctrl._K_lqr = k_lqr
    
    env_config = {
        'episode_length': num_steps,
        'height_command_mode': 'fixed',
        'target_height': 0.65,
        'enable_push_disturbance': False,
    }
    env = BalanceEnv(env_config)
    
    survivals = []
    pitch_rmss = []
    roll_rmss = []
    
    import jax
    rng = jax.random.PRNGKey(0)
    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)
        ctrl.reset(0.65)
        
        steps = 0
        p_sq = 0.0
        r_sq = 0.0
        
        for s in range(num_steps):
            obs = np.array(state.obs)
            action = ctrl.compute_action(obs)
            state = env.step(state, action)
            steps += 1
            
            p_sq += float(state.obs[3])**2  # pitch
            r_sq += float(state.obs[0])**2  # roll
            
            if bool(state.done):
                break
                
        survivals.append(steps * 0.02)
        pitch_rmss.append(np.rad2deg(np.sqrt(p_sq / max(steps, 1))))
        roll_rmss.append(np.rad2deg(np.sqrt(r_sq / max(steps, 1))))
        
    return {
        'survival_time': float(np.mean(survivals)),
        'pitch_rms_deg': float(np.mean(pitch_rmss)),
        'roll_rms_deg': float(np.mean(roll_rmss)),
        'k_lqr': k_lqr.tolist(),
    }


def main():
    print("=" * 72)
    print("SWEEPING & RE-OPTIMIZING GAINS FOR DIRECT TORQUE LQR BASELINE")
    print("=" * 72)
    
    # Param grids
    tau_s_list = [0.0, 0.05, 0.1, 0.25]
    lqr_q_list = [
        [10.0, 2.0, 3.0, 0.3],
        [50.0, 5.0, 10.0, 1.0],
        [100.0, 10.0, 15.0, 2.0],
        [20.0, 1.0, 5.0, 0.1],
    ]
    lqr_r_list = [0.01, 0.1, 0.8, 2.0]
    kp_roll_list = [0.4, 10.0, 35.0, 55.0, 75.0]
    
    best_res = None
    best_surv = -1.0
    best_params = None
    
    results = []
    
    for tau_s in tau_s_list:
        for q in lqr_q_list:
            for r in lqr_r_list:
                for kp_r in kp_roll_list:
                    kd_r = kp_r * 0.1  # maintain D gain ratio
                    for kp_y in [2.5, 5.0]:
                        kd_y = kp_y * 0.1
                        
                        res = eval_dt_lqr(q, r, tau_s, kp_r, kd_r, kp_y, kd_y, num_episodes=3, num_steps=1000)
                        
                        surv = res['survival_time']
                        params = {
                            'tau_s': tau_s,
                            'lqr_q': q,
                            'lqr_r': r,
                            'kp_roll': kp_r,
                            'kd_roll': kd_r,
                            'kp_yaw': kp_y,
                            'kd_yaw': kd_y,
                        }
                        results.append((surv, res, params))
                        
                        if surv > best_surv:
                            best_surv = surv
                            best_res = res
                            best_params = params
                            print(f"NEW BEST! Surv: {surv:.2f}s | Pitch RMS: {res['pitch_rms_deg']:.2f}° | Roll RMS: {res['roll_rms_deg']:.2f}° | params: {params}")

    print("\n" + "=" * 72)
    print("TOP 10 RESULTS:")
    print("=" * 72)
    results.sort(key=lambda x: x[0], reverse=True)
    for idx, (surv, res, params) in enumerate(results[:10]):
        print(f"#{idx+1}: Surv={surv:.2f}s | Pitch RMS={res['pitch_rms_deg']:.2f}° | Roll RMS={res['roll_rms_deg']:.2f}°")
        print(f"     Params: {params}")
        print(f"     K_LQR: {res['k_lqr']}")


if __name__ == '__main__':
    main()
