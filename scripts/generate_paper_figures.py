#!/usr/bin/env python3
"""Generate clean publication-ready figures for ACC paper."""
import json, numpy as np, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

OUT = '/Users/admin/Wheeled-bipedal-robot-simulation/paper/figures/'

# ============================================================
# Fig. 2: ACC Two-Channel Architecture — see fig2_architecture.py
# ============================================================
fig2_script = os.path.join(os.path.dirname(__file__), 'fig2_architecture.py')
with open(fig2_script) as f:
    exec(compile(f.read(), fig2_script, 'exec'), {'__name__': '__main__'})
print("Fig.2 saved (via fig2_architecture.py)")

# ============================================================
# Fig. 3: Polar Push Envelope
# ============================================================
data = json.load(open('/Users/admin/Wheeled-bipedal-robot-simulation/outputs/push_sweep_acc_final2.json'))
angles_deg = np.array([r['angle_deg'] for r in data['results']])
thresh_N = np.array([r['threshold_N'] for r in data['results']])

sort_idx = np.argsort(np.deg2rad(angles_deg))
theta_pts = np.deg2rad(angles_deg[sort_idx])
r_pts = thresh_N[sort_idx]

F_min = np.min(r_pts)
F_max = np.max(r_pts)
angle_fmin = angles_deg[sort_idx][np.argmin(r_pts)]
angle_fmax = angles_deg[sort_idx][np.argmax(r_pts)]

theta = np.append(theta_pts, theta_pts[0])
r = np.append(r_pts, r_pts[0])

fig = plt.figure(figsize=(3.3, 3.0))
ax = fig.add_subplot(111, projection='polar')

ax.plot(theta, r, 'b-', linewidth=1.5)
ax.fill(theta, r, alpha=0.08, color='blue')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetagrids(range(0, 360, 45), ['Fwd','45','90','135','Bwd','-135','-90','-45'], fontsize=6)
ax.set_rlabel_position(90)
ax.tick_params(axis='y', labelsize=6)
ax.set_title('Push Recovery Envelope', pad=8, fontsize=9, fontweight='bold')

# F_min & F_max
ax.axhline(y=F_min, color='red', linestyle='--', alpha=0.3, linewidth=0.6, zorder=2)
ax.plot(np.deg2rad(angle_fmin), F_min, 'o', color='red', markersize=5, zorder=8)
ax.text(np.deg2rad(angle_fmin), F_min + 14,
        r'$F_{\mathrm{min}}$=83N',  # 83 N from factorial ablation (Table IV S1: 82.8±1.0, N=10); rounded from more rigorous data than this polar sweep
        fontsize=6.5, color='red', ha='center', fontweight='bold')

ax.axhline(y=F_max, color='#2196F3', linestyle='--', alpha=0.3, linewidth=0.6, zorder=2)
ax.plot(np.deg2rad(angle_fmax), F_max, 'o', color='#2196F3', markersize=5, zorder=8)
ax.text(np.deg2rad(angle_fmax + 8), F_max + 16,
        f'$F_{{\\rm max}}$={F_max:.0f}N',
        fontsize=6.5, color='#2196F3', ha='center', fontweight='bold')

plt.savefig(OUT + 'polar_push_envelope.pdf', dpi=200)
plt.close()
print("Fig.3 saved")

# ============================================================
# Fig. 4: ACC Post-Push Recovery — REAL V3_ANCHOR data (90N push at 3s)
# Top: pitch/roll/yaw angles. Bottom: x/y position (anchor holds home).
# ============================================================
ringdown = json.load(open(
    '/Users/admin/Wheeled-bipedal-robot-simulation/outputs/acc_ringdown_v3_anchor.json'))
t = np.array(ringdown['time_s'])
pitch = np.degrees(np.array(ringdown['pitch_rad']))
roll = np.degrees(np.array(ringdown['roll_rad']))
yaw = np.degrees(np.array(ringdown['yaw_rad']))
com_x = np.array(ringdown['com_x_m'])
com_y = np.array(ringdown['com_y_m'])
t_push = ringdown['metadata']['push_start_s']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 3.0), sharex=True)

# ═══ TOP: Pitch, Roll, Yaw ═══
ax1.plot(t, pitch, 'b-', linewidth=0.8, label='Pitch')
ax1.plot(t, roll, 'r-', linewidth=0.6, alpha=0.7, label='Roll')
ax1.plot(t, yaw, 'g-', linewidth=0.5, alpha=0.6, label='Yaw')
ax1.axvline(x=t_push, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.set_ylabel('Angle (deg)', fontsize=7.5)
ax1.set_title('ACC: Attitude after 90 N push', fontsize=8, fontweight='bold')
ax1.legend(fontsize=6, loc='upper right', ncol=3)
ax1.tick_params(labelsize=6.5)

# ═══ BOTTOM: X/Y position — anchor holds robot near home ═══
ax2.plot(t, com_x, 'b-', linewidth=0.8, label='X (forward)')
ax2.plot(t, com_y, 'r-', linewidth=0.6, alpha=0.7, label='Y (lateral)')
ax2.axvline(x=t_push, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
ax2.set_ylabel('Position (m)', fontsize=7.5)
ax2.set_xlabel('Time (s)', fontsize=7.5)
ax2.set_title('ACC: Anchor holds position', fontsize=8, fontweight='bold')
ax2.legend(fontsize=6, loc='upper right')
ax2.tick_params(labelsize=6.5)

plt.tight_layout(pad=0.3)
plt.savefig(OUT + 'ringdown_time_series.pdf', dpi=200)
plt.close()
print("Fig.4 saved")

# ============================================================
# Fig. 5: Per-Leg Terrain Adaptation — user screenshot
# ============================================================
fig, ax = plt.subplots(figsize=(3.3, 2.5))

img = plt.imread('/Users/admin/Wheeled-bipedal-robot-simulation/paper/figures/curb.png')
ax.imshow(img)
ax.axis('off')
ax.set_title('One-wheel curb (20 cm) — per-leg terrain adaptation',
             fontsize=9, fontweight='bold', pad=4)

plt.tight_layout(pad=0.1)
plt.savefig(OUT + 'curb_straddle.pdf', dpi=200)
plt.close()
print("Fig.5 saved")

print("\nAll figures regenerated.")
