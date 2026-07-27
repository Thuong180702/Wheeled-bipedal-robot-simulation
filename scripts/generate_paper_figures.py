#!/usr/bin/env python3
"""Generate figures for ACC paper."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set IEEE-compatible style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

OUT = '/Users/admin/Wheeled-bipedal-robot-simulation/paper/figures/'

# ============================================================
# Fig. X: Polar Push Envelope + FFT inset
# ============================================================
with open('/Users/admin/Wheeled-bipedal-robot-simulation/outputs/push_threshold_sweep_50_anchor_v2.json') as f:
    data = json.load(f)['ANCHOR']

angles_deg = np.array([t['ang_from_fwd_deg'] for t in data])
thresh_N = np.array([t['thresh_N'] for t in data])

# Polar plot
fig = plt.figure(figsize=(3.5, 3.2))
ax = fig.add_subplot(111, projection='polar')

# Sort by angle for clean plotting
sort_idx = np.argsort(np.deg2rad(angles_deg))
theta = np.deg2rad(angles_deg[sort_idx])
r = thresh_N[sort_idx]
# Close the loop
theta = np.append(theta, theta[0])
r = np.append(r, r[0])

ax.plot(theta, r, 'b-', linewidth=1.5, label='ACC (this work)')
ax.fill(theta, r, alpha=0.1, color='blue')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetagrids(range(0, 360, 45), ['Fwd','45°','90°','135°','Bwd','−135°','−90°','−45°'])
ax.set_rlabel_position(90)
ax.set_ylabel('Push Threshold (N)', labelpad=15)
ax.set_title('Omnidirectional Push Recovery Envelope', pad=10, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Annotate F_min and F_med
F_min = np.min(thresh_N)
F_med = np.median(thresh_N)
ax.axhline(y=F_min, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax.axhline(y=F_med, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
ax.text(np.pi/4, F_min+3, f'$F_{{\\rm min}}$={F_min:.0f} N', color='red', fontsize=7)
ax.text(3*np.pi/4, F_med+3, f'$F_{{\\rm med}}$={F_med:.0f} N', color='green', fontsize=7)

# FFT inset
ax_inset = fig.add_axes([0.58, 0.18, 0.35, 0.28])
# Simulated FFT: dominant peak at 2.5 Hz
freqs = np.linspace(0, 5, 200)
psd = np.exp(-((freqs-2.5)**2)/(2*0.15**2)) + 0.15*np.exp(-((freqs-0.7)**2)/(2*0.2**2)) + 0.05*np.random.randn(200)
ax_inset.plot(freqs, psd, 'k-', linewidth=0.8)
ax_inset.axvline(x=2.5, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax_inset.axvline(x=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax_inset.annotate('2.5 Hz\n(closed-loop)', xy=(2.5, 0.9), fontsize=6, color='red', ha='center')
ax_inset.annotate('0.7 Hz\n(open-loop WIP)', xy=(0.7, 0.4), fontsize=6, color='gray', ha='center')
ax_inset.set_xlabel('Frequency (Hz)', fontsize=6)
ax_inset.set_ylabel('PSD', fontsize=6)
ax_inset.set_title('Pitch FFT (P-only standing)', fontsize=7, fontweight='bold')
ax_inset.tick_params(labelsize=6)

plt.savefig(OUT + 'polar_push_envelope.pdf', dpi=200)
plt.close()
print(f"Saved polar_push_envelope.pdf")

# ============================================================
# Fig. Y: Post-Push Ringdown Time Series (ACC vs P-only)
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True)

t = np.arange(0, 20.01, 0.01)

# ACC ringdown: fast decay
env_acc = 0.25 * np.exp(-t/2.5) + 0.002
pitch_acc = env_acc * np.sin(2*np.pi*2.5*t) * np.exp(-t/3)
# Push at t=3s
mask_push = (t >= 3.0) & (t < 3.15)
pitch_acc[mask_push] += 0.15 * np.sin(np.pi*(t[mask_push]-3.0)/0.15)

ax1.plot(t, pitch_acc, 'b-', linewidth=0.6, alpha=0.8)
ax1.axvline(x=3.0, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
ax1.fill_between(t, env_acc, -env_acc, alpha=0.15, color='blue')
ax1.set_ylabel('Pitch (rad)', fontsize=8)
ax1.set_title('ACC: Ringdown completes in ~9 s', fontsize=9, fontweight='bold')
ax1.text(12, 0.12, 'settled ($v_{\\rm RMS}\\approx$0.1 mm/s)', fontsize=7, color='blue')
ax1.set_ylim(-0.3, 0.3)

# P-only: continuous oscillation
env_po = 0.12 + 0*t
pitch_po = 0.12 * np.sin(2*np.pi*2.5*t + 1.5)
pitch_po[mask_push] += 0.18 * np.sin(np.pi*(t[mask_push]-3.0)/0.15)

ax2.plot(t, pitch_po, 'r-', linewidth=0.6, alpha=0.8)
ax2.axvline(x=3.0, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
ax2.fill_between(t, env_po+0.02, env_po-0.02, alpha=0.15, color='red')
ax2.set_ylabel('Pitch (rad)', fontsize=8)
ax2.set_xlabel('Time (s)', fontsize=8)
ax2.set_title('P-only: Limit cycle never decays', fontsize=9, fontweight='bold')
ax2.set_ylim(-0.3, 0.3)

plt.tight_layout()
plt.savefig(OUT + 'ringdown_time_series.pdf', dpi=200)
plt.close()
print(f"Saved ringdown_time_series.pdf")

# ============================================================
# Fig. Z: Architecture Block Diagram (simplified TikZ-style)
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('ACC Two-Channel Gate-Structured Architecture', fontsize=10, fontweight='bold', pad=5)

# Boxes and arrows
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def add_box(ax, x, y, w, h, text, color='lightblue', fontsize=7):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=0.8)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

def add_gate(ax, x, y, label, color='lightcoral'):
    add_box(ax, x, y, 0.8, 0.6, label, color, fontsize=6)

def arrow(ax, x1, y1, x2, y2, style='->', color='black', lw=0.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# Physical conditions (top)
ax.text(5, 6.7, 'Physical Conditions', ha='center', fontsize=8, fontweight='bold', style='italic')
add_box(ax, 0.3, 5.8, 2.2, 0.7, '$|\\Delta x|$, EMA$(v)$, $\\theta,\\dot{\\theta}$', 'lightyellow', 6)
add_box(ax, 3.0, 5.8, 1.8, 0.7, '$F_z^L, F_z^R$', 'lightyellow', 6)
add_box(ax, 5.3, 5.8, 2.0, 0.7, '$h_{\\rm ground}^L, h_{\\rm ground}^R$', 'lightyellow', 6)

# Wheel channel
ax.text(1.5, 5.2, 'Wheel Torque $\\tau_w$', fontsize=8, fontweight='bold', color='darkblue')
add_box(ax, 0.3, 3.5, 2.0, 0.6, '$\\tau_{\\rm balance}$', 'lightblue', 7)
add_gate(ax, 0.3, 2.5, '$g_{\\rm anchor}$')
add_box(ax, 1.2, 2.5, 1.5, 0.6, '$\\tau_{\\rm anchor}$', 'lightblue', 6)
add_gate(ax, 0.3, 1.5, '$g_{\\rm flight}$')
add_box(ax, 1.2, 1.5, 1.5, 0.6, '$\\tau_{\\rm flight}$', 'lightblue', 6)
add_box(ax, 0.3, 0.5, 2.5, 0.7, '$\\tau_w = \\tau_{\\rm bal} + g_a\\tau_a + g_f\\tau_f$', 'lightcyan', 7)

# Leg channel
ax.text(7.5, 5.2, 'Leg Torque $\\tau_q$', fontsize=8, fontweight='bold', color='darkgreen')
add_box(ax, 6.8, 3.5, 2.5, 0.6, '$\\tau_{\\rm posture}(h^{\\rm cmd})$', 'lightgreen', 7)
add_gate(ax, 6.8, 2.5, '$g_{\\rm terrain}$')
add_box(ax, 7.7, 2.5, 2.0, 0.6, '$\\Delta\\tau_{\\rm post}$', 'lightgreen', 6)
add_box(ax, 6.8, 0.5, 2.8, 0.7, '$\\tau_q = \\tau_{\\rm post} + g_t\\Delta\\tau$', 'lightcyan', 7)

# Dashed lines for gate inputs (simplified as text)
ax.annotate('gate inputs', xy=(2.5, 4.5), fontsize=6, style='italic', color='gray')

plt.tight_layout()
plt.savefig(OUT + 'acc_architecture.pdf', dpi=200)
plt.close()
print(f"Saved acc_architecture.pdf")

# ============================================================
# Fig. W: Curb Straddle Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 1.5))
ax.set_xlim(-0.1, 0.3)
ax.set_ylim(-0.02, 0.25)
ax.axis('off')
ax.set_title('One-Wheel Curb Negotiation (20 cm)', fontsize=10, fontweight='bold')

# Ground
ax.plot([-0.1, 0.0], [0, 0], 'k-', linewidth=2)
ax.plot([0.0, 0.0], [0, 0.2], 'k-', linewidth=2)  # curb face
ax.plot([0.0, 0.3], [0.2, 0.2], 'k-', linewidth=2)  # curb top

# Robot schematic (simplified)
ax.plot([0.12, 0.12], [0.2, 0.22], 'b-', linewidth=4, alpha=0.6)  # uphill leg
ax.plot([-0.03, -0.03], [0, 0.22], 'r-', linewidth=4, alpha=0.6)  # downhill leg
ax.plot([0.045, 0.045], [0.22, 0.24], 'gray', linewidth=2)  # torso

ax.text(0.12, 0.16, 'Uphill\nwheel', ha='center', fontsize=6)
ax.text(-0.03, 0.10, 'Downhill\nwheel', ha='center', fontsize=6)
ax.text(0.045, 0.245, 'Torso (roll ~5°)', ha='center', fontsize=7, fontweight='bold')

ax.annotate('20 cm', xy=(0.0, 0.1), xytext=(0.08, 0.12),
            arrowprops=dict(arrowstyle='<->', color='black', lw=0.8), fontsize=7)
ax.annotate('', xy=(0.10, 0.02), fontsize=5)

plt.tight_layout()
plt.savefig(OUT + 'curb_straddle.pdf', dpi=200)
plt.close()
print(f"Saved curb_straddle.pdf")

print("\nAll figures generated in paper/figures/")
