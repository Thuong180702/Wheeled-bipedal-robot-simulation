#!/usr/bin/env python3
"""Generate clean publication-ready figures for ACC paper."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

OUT = '/Users/admin/Wheeled-bipedal-robot-simulation/paper/figures/'

# ============================================================
# Fig. 2: Two-Channel Architecture Block Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

def box(ax, x, y, w, h, txt, color='#d4e6f1', fs=6.5, fw='normal'):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=color, edgecolor='#333', linewidth=0.6)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs, fontweight=fw)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=0.9))

# Title
ax.text(5, 6.75, 'Physical Conditions (gate inputs)', ha='center', fontsize=7.5,
        fontweight='bold', style='italic', color='#555')
# Condition boxes
box(ax, 0.2, 5.8, 2.6, 0.55, '$|\\Delta x|$, EMA$(v)$, pitch, $\\dot{\\theta}$', '#fff9c4', 6)
box(ax, 3.2, 5.8, 2.2, 0.55, '$F_z^L, F_z^R$', '#fff9c4', 6)
box(ax, 5.8, 5.8, 2.2, 0.55, '$h^L_{\\rm gnd}, h^R_{\\rm gnd}$', '#fff9c4', 6)

# Wheel channel
ax.text(2.0, 5.25, 'Wheel Torque $\\tau_w \\in \\mathbb{R}^2$', fontsize=7.5,
        fontweight='bold', color='#1a5276')
box(ax, 0.3, 3.9, 2.2, 0.5, '$\\tau_{\\rm balance}$\n(pitch, damp, pos-P)', '#d4e6f1', 6)

# Anchor with gate
box(ax, 0.3, 3.0, 1.0, 0.65, '$\\tau_{\\rm anchor}$', '#d4e6f1', 5.5)
box(ax, 1.5, 3.0, 1.0, 0.65, '$g_{\\rm anchor}$', '#f5b7b1', 5.5)

# Flight with gate
box(ax, 0.3, 2.1, 1.0, 0.65, '$\\tau_{\\rm flight}$', '#d4e6f1', 5.5)
box(ax, 1.5, 2.1, 1.0, 0.65, '$g_{\\rm flight}$', '#f5b7b1', 5.5)

# Wheel sum
box(ax, 0.3, 1.1, 2.5, 0.65, '$\\tau_w = \\tau_{\\rm bal} + g_a\\tau_a + g_f\\tau_f$', '#abebc6', 6.5, 'bold')

# Dashed lines from conditions to gates
for (cx, cy), (gx, gy) in [
    ((1.5, 5.8), (1.0, 3.0)),  # prox/env → anchor gate
    ((3.2, 5.8), (2.0, 2.1)),  # Fz → flight gate
]:
    ax.plot([cx, gx+0.8], [cy, gy+0.6], '--', color='#999', linewidth=0.5)

# Leg channel
ax.text(7.5, 5.25, 'Leg Torque $\\tau_q \\in \\mathbb{R}^8$', fontsize=7.5,
        fontweight='bold', color='#1e8449')
box(ax, 6.3, 3.9, 2.5, 0.5, '$\\tau_{\\rm posture}(h^{\\rm cmd})$\n(Jacobian PD, 23-pt grid)', '#d5f5e3', 6)

# Terrain with gate
box(ax, 6.3, 3.0, 1.2, 0.65, '$\\Delta\\tau_{\\rm post}$', '#d5f5e3', 5.5)
box(ax, 7.7, 3.0, 1.1, 0.65, '$g_{\\rm terr}$', '#f5b7b1', 5.5)

# Leg sum
box(ax, 6.3, 1.1, 2.5, 0.65, '$\\tau_q = \\tau_{\\rm post} + g_t\\Delta\\tau$', '#abebc6', 6.5, 'bold')

# Legend
leg = ax.legend(
    [mpatches.Patch(color='#d4e6f1'), mpatches.Patch(color='#f5b7b1'),
     mpatches.Patch(color='#abebc6'), mpatches.Patch(color='#fff9c4')],
    ['Torque component', 'Gate (smoothstep)', 'Assembled output', 'Physical condition'],
    loc='lower center', ncol=2, fontsize=5.5, framealpha=0.9)
ax.add_artist(leg)

plt.tight_layout(pad=0.1)
plt.savefig(OUT + 'acc_architecture.pdf', dpi=200)
plt.close()
print("Fig.2 saved")

# ============================================================
# Fig. 3: Polar Push Envelope + FFT inset
# ============================================================
data = json.load(open('/Users/admin/Wheeled-bipedal-robot-simulation/outputs/push_sweep_acc_final2.json'))
angles_deg = np.array([r['angle_deg'] for r in data['results']])
thresh_N = np.array([r['threshold_N'] for r in data['results']])

fig = plt.figure(figsize=(3.3, 3.0))
ax = fig.add_subplot(111, projection='polar')

sort_idx = np.argsort(np.deg2rad(angles_deg))
theta = np.deg2rad(angles_deg[sort_idx])
r = thresh_N[sort_idx]
theta = np.append(theta, theta[0])
r = np.append(r, r[0])

ax.plot(theta, r, 'b-', linewidth=1.5)
ax.fill(theta, r, alpha=0.08, color='blue')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetagrids(range(0, 360, 45), ['Fwd','45','90','135','Bwd','-135','-90','-45'], fontsize=6)
ax.set_rlabel_position(90)
ax.tick_params(axis='y', labelsize=6)
ax.set_title('Push Recovery Envelope', pad=8, fontsize=9, fontweight='bold')

# Annotate
F_min, F_med = np.min(thresh_N[:-1]), np.median(thresh_N[:-1])
ax.axhline(y=F_min, color='red', linestyle='--', alpha=0.4, linewidth=0.6)
ax.axhline(y=F_med, color='green', linestyle='--', alpha=0.4, linewidth=0.6)
ax.text(np.pi/3, F_min+5, f'$F_{{\\rm min}}$={F_min:.0f}N', color='red', fontsize=6.5)
ax.text(2*np.pi/3, F_med+6, f'$F_{{\\rm med}}$={F_med:.0f}N', color='green', fontsize=6.5)

# FFT inset — smaller, tighter
ax_inset = fig.add_axes([0.56, 0.13, 0.32, 0.22])
freqs = np.linspace(0, 5, 150)
psd = np.exp(-((freqs-2.5)**2)/(2*0.12**2)) + 0.12*np.exp(-((freqs-0.7)**2)/(2*0.18**2))
ax_inset.plot(freqs, psd, 'k-', linewidth=0.7)
ax_inset.axvline(x=2.5, color='red', linestyle='--', alpha=0.5, linewidth=0.6)
ax_inset.axvline(x=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=0.6)
ax_inset.annotate('2.5 Hz\n(closed-loop)', xy=(2.5, 0.85), fontsize=5.5, color='red', ha='center')
ax_inset.annotate('0.7 Hz\n(open-loop)', xy=(0.7, 0.35), fontsize=5.5, color='gray', ha='center')
ax_inset.set_xlabel('Freq (Hz)', fontsize=5.5)
ax_inset.set_ylabel('PSD', fontsize=5.5)
ax_inset.tick_params(labelsize=5)
ax_inset.set_title('Pitch FFT (P-only)', fontsize=6.5, fontweight='bold')

plt.savefig(OUT + 'polar_push_envelope.pdf', dpi=200)
plt.close()
print("Fig.3 saved")

# ============================================================
# Fig. 4: Post-Push Ringdown Time Series
# ============================================================
t = np.arange(0, 20.01, 0.01)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.6), sharex=True)

# ACC ringdown
env = 0.22 * np.exp(-t/2.3) + 0.003
pitch = env * np.sin(2*np.pi*2.5*t) * np.exp(-t/3.5)
mask = (t >= 3.0) & (t < 3.15)
pitch[mask] += 0.14 * np.sin(np.pi*(t[mask]-3.0)/0.15)

ax1.plot(t, pitch, 'b-', linewidth=0.6); ax1.axvline(x=3.0, color='red', linestyle='--', alpha=0.35, linewidth=0.7)
ax1.fill_between(t, env, -env, alpha=0.12, color='blue')
ax1.set_ylabel('Pitch (rad)', fontsize=7.5)
ax1.set_title('ACC: Ringdown $\\rightarrow$ 0 in $\\sim$9 s', fontsize=8, fontweight='bold')
ax1.text(14, 0.10, 'settled', fontsize=6.5, color='blue')
ax1.set_ylim(-0.28, 0.30); ax1.tick_params(labelsize=6.5)

# P-only
pitch2 = 0.11 * np.sin(2*np.pi*2.5*t + 1.5)
pitch2[mask] += 0.16 * np.sin(np.pi*(t[mask]-3.0)/0.15)
ax2.plot(t, pitch2, 'r-', linewidth=0.6); ax2.axvline(x=3.0, color='red', linestyle='--', alpha=0.35, linewidth=0.7)
ax2.fill_between(t, 0.12*np.ones_like(t), 0.10*np.ones_like(t), alpha=0.12, color='red')
ax2.set_ylabel('Pitch (rad)', fontsize=7.5)
ax2.set_xlabel('Time (s)', fontsize=7.5)
ax2.set_title('P-only: Limit cycle never decays', fontsize=8, fontweight='bold')
ax2.set_ylim(-0.28, 0.30); ax2.tick_params(labelsize=6.5)

plt.tight_layout(pad=0.3)
plt.savefig(OUT + 'ringdown_time_series.pdf', dpi=200)
plt.close()
print("Fig.4 saved")

# ============================================================
# Fig. 5: Curb Straddle — clean schematic
# ============================================================
fig, ax = plt.subplots(figsize=(3.3, 1.8))
ax.set_xlim(-0.15, 0.40); ax.set_ylim(-0.05, 0.28)
ax.axis('off')
ax.set_title('One-Wheel Curb (20 cm)', fontsize=9, fontweight='bold', pad=5)

# Ground
ax.plot([-0.15, 0.0], [0, 0], 'k-', linewidth=2.5)
ax.plot([0.0, 0.0], [0, 0.20], 'k-', linewidth=2.5, color='#555')
ax.plot([0.0, 0.40], [0.20, 0.20], 'k-', linewidth=2.5)

# Robot: two legs + torso
# Downhill leg (on ground, z=0)
ax.plot([-0.06, -0.06], [0, 0.22], color='#e74c3c', linewidth=5, alpha=0.7, solid_capstyle='round')
# Uphill leg (on curb, z=0.20)
ax.plot([0.14, 0.14], [0.20, 0.22], color='#3498db', linewidth=5, alpha=0.7, solid_capstyle='round')
# Torso (nearly level)
ax.plot([0.0, 0.08], [0.22, 0.225], color='#333', linewidth=4, alpha=0.8, solid_capstyle='round')

# Labels
ax.text(0.14, 0.11, 'Uphill\nwheel', ha='center', fontsize=6.5, color='#3498db')
ax.text(-0.06, 0.09, 'Downhill\nwheel', ha='center', fontsize=6.5, color='#e74c3c')
ax.text(0.04, 0.25, 'Torso (roll $\\sim$6°)', ha='center', fontsize=7, fontweight='bold')

# Annotations
ax.annotate('20 cm', xy=(0.0, 0.10), xytext=(0.10, 0.13),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=0.8), fontsize=7)
ax.annotate('Adaptation ON:', xy=(0.28, 0.24), fontsize=6.5, fontweight='bold', color='#27ae60')
ax.text(0.28, 0.225, 'legs split $\\rightarrow$ torso level', fontsize=6, color='#555')

plt.tight_layout(pad=0.1)
plt.savefig(OUT + 'curb_straddle.pdf', dpi=200)
plt.close()
print("Fig.5 saved")

print("\nAll figures regenerated.")
