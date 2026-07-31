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
# Literally the same run as the factorial ablation (Table IV): this reads the S1
# row -- fixed k_p=50, i.e. canonical ACC = L3 -- straight out of the ablation
# result file, so the figure and the table cannot drift apart. Do NOT plot
# push_ci_all_ablations.json's "L3_S0_Full-ACC": that key was produced with empty
# param_overrides, which is the S0 *scheduled*-k_p variant the paper rejects.
_ST = '/Users/admin/Wheeled-bipedal-robot-simulation/outputs/paper_statistics/'
_abl = [json.load(open(_ST + f)) for f in
        ('ablation_n10_results.json', 'ablation_n10_partial.json') if os.path.exists(_ST + f)]
data = next(d['S1'] for d in _abl if 'S1' in d)   # whichever file the completed run landed in
reps = {int(a): np.asarray(v) for a, v in data['all_reps'].items()}
angles_deg = np.array(sorted(reps))
r_mean = np.array([reps[a].mean() for a in angles_deg])
r_sd = np.array([reps[a].std(ddof=1) for a in angles_deg])

theta_pts = np.deg2rad(angles_deg)
F_min = r_mean.min()
F_max = r_mean.max()
angle_fmin = angles_deg[np.argmin(r_mean)]
angle_fmax = angles_deg[np.argmax(r_mean)]

# close the loop
theta = np.append(theta_pts, theta_pts[0])
r = np.append(r_mean, r_mean[0])
r_lo = np.append(r_mean - r_sd, r_mean[0] - r_sd[0])
r_hi = np.append(r_mean + r_sd, r_mean[0] + r_sd[0])

fig = plt.figure(figsize=(3.45, 3.15))
ax = fig.add_subplot(111, projection='polar')

ax.fill_between(theta, r_lo, r_hi, color='blue', alpha=0.18, linewidth=0, zorder=3)
ax.plot(theta, r, 'b-', linewidth=1.5, zorder=4)
ax.fill(theta, r, alpha=0.07, color='blue', zorder=1)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
# Bearings are world-frame f = F*(cos T, sin T, 0). This robot's wheel axles lie
# along world x, so world y is the sagittal (rolling/driving) axis and world x is
# the lateral (track) axis -- see scripts/verify_body_axes.py. Hence +-90 deg are
# the fore/aft bearings and 0/180 deg the lateral ones, NOT the other way round.
ax.set_thetagrids(range(0, 360, 45),
                  ['$+x$ lat', '45', '$+y$ sag', '135',
                   '$-x$ lat', '-135', '$-y$ sag', '-45'], fontsize=7)
# Radial ticks: coarse set on a spoke-free bearing (202.5 deg) so the numbers
# neither run together nor collide with the angular tick labels.
ax.set_rlim(0.0, 170.0)
ax.autoscale(enable=False)   # the single-point F_min/F_max markers below would
                             # otherwise re-autoscale r and shift the origin off 0
ax.set_rticks([50, 100, 150])
ax.set_yticklabels(['50', '100', '150 N'], fontsize=7)
ax.set_rlabel_position(202.5)
ax.tick_params(axis='y', pad=0)
for lbl in ax.get_yticklabels():
    lbl.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.8))
ax.set_title('Push Recovery Envelope ($N{=}10$)', pad=8, fontsize=9, fontweight='bold')

# weakest and strongest bearing (means over the 10 reps)
ax.axhline(y=F_min, color='red', linestyle='--', alpha=0.35, linewidth=0.6, zorder=2)
ax.plot(np.deg2rad(angle_fmin), F_min, 'o', color='red', markersize=5, zorder=8)
ax.text(np.deg2rad(angle_fmin - 18), 0.52 * F_min,
        f'$F_{{\\mathrm{{min}}}}$={F_min:.0f} N',
        fontsize=9, color='red', ha='center', va='center', fontweight='bold', zorder=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))

ax.axhline(y=F_max, color='#2196F3', linestyle='--', alpha=0.35, linewidth=0.6, zorder=2)
ax.plot(np.deg2rad(angle_fmax), F_max, 'o', color='#2196F3', markersize=5, zorder=8)
# Keep the label inside the rlim (170) and off the rim so it cannot collide with
# the angular tick ring; the white bbox guards against the radial grid lines.
ax.text(np.deg2rad(angle_fmax + 26), 0.80 * F_max,
        f'$F_{{\\mathrm{{max}}}}$={F_max:.0f} N',
        fontsize=9, color='#2196F3', ha='center', va='center', fontweight='bold', zorder=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.2), sharex=True)

# ═══ TOP: Pitch, Roll, Yaw ═══
ax1.plot(t, pitch, 'b-', linewidth=0.8, label='Pitch')
ax1.plot(t, roll, 'r-', linewidth=0.6, alpha=0.7, label='Roll')
ax1.plot(t, yaw, 'g-', linewidth=0.5, alpha=0.6, label='Yaw')
ax1.axvline(x=t_push, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.set_ylabel('Angle (deg)', fontsize=7.5)
ax1.set_title('ACC: Attitude after 90 N push', fontsize=8, fontweight='bold')
ax1.legend(fontsize=7, loc='upper right', ncol=3)
ax1.tick_params(labelsize=7)

# ═══ BOTTOM: X/Y position — anchor holds robot near home ═══
ax2.plot(t, com_x, 'b-', linewidth=0.8, label='X (lateral, pushed)')
ax2.plot(t, com_y, 'r-', linewidth=0.6, alpha=0.7, label='Y (sagittal)')
ax2.axvline(x=t_push, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
ax2.set_ylabel('Position (m)', fontsize=7.5)
ax2.set_xlabel('Time (s)', fontsize=7.5)
ax2.set_title('ACC: Anchor holds position', fontsize=8, fontweight='bold')
ax2.legend(fontsize=6.5, loc='upper right')
ax2.tick_params(labelsize=7)

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
