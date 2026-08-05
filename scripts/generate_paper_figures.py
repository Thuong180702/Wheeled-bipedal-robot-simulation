#!/usr/bin/env python3
"""Generate the black-and-white figure variants for the JCD submission.

Writes `*_bw.pdf` only. The colour figures used by paper/main.tex are the
committed `*.pdf` files; their generator is this script at revision fd6f57e^.
"""
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
# Fig. 3: Polar Push Envelope -- see fig3_polar_envelope.py
# ============================================================
fig3_script = os.path.join(os.path.dirname(__file__), 'fig3_polar_envelope.py')
with open(fig3_script) as f:
    exec(compile(f.read(), fig3_script, 'exec'), {'__name__': '__main__'})

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True)

# ═══ TOP: Pitch, Roll, Yaw ═══
ax1.plot(t, pitch, '-', color='black', linewidth=0.8, label='Pitch')
ax1.plot(t, roll, '--', color='0.35', linewidth=0.7, label='Roll')
ax1.plot(t, yaw, ':', color='0.55', linewidth=0.7, label='Yaw')
ax1.axvline(x=t_push, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.set_ylabel('Angle (deg)', fontsize=7.5)
ax1.set_title('ACC: Attitude after 90 N push', fontsize=8, fontweight='bold')
ax1.legend(fontsize=7, loc='upper right', ncol=3)
ax1.tick_params(labelsize=7)

# ═══ BOTTOM: X/Y position — anchor holds robot near home ═══
ax2.plot(t, com_x, '-', color='black', linewidth=0.8, label='X (lateral, pushed)')
ax2.plot(t, com_y, '--', color='0.4', linewidth=0.7, label='Y (sagittal)')
ax2.axvline(x=t_push, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
ax2.set_ylabel('Position (m)', fontsize=7.5)
ax2.set_xlabel('Time (s)', fontsize=7.5)
ax2.set_title('ACC: Anchor holds position', fontsize=8, fontweight='bold')
ax2.legend(fontsize=6.5, loc='upper right')
ax2.tick_params(labelsize=7)

plt.tight_layout(pad=0.3)
plt.savefig(OUT + 'ringdown_time_series_bw.pdf', dpi=200)
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
plt.savefig(OUT + 'curb_straddle_bw.pdf', dpi=200)
plt.close()
print("Fig.5 saved")

print("\nAll figures regenerated.")
