#!/usr/bin/env python3
"""Fig.2: ACC Two-Channel Torque Assembly — paper terminology, zero-overlap verified."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 6.5,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.0,
})
OUT = '/Users/admin/Wheeled-bipedal-robot-simulation/paper/figures/acc_architecture.pdf'

fig, ax = plt.subplots(figsize=(3.45, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 22); ax.axis('off')

def abox(x, y, txt, fc='#d4e6f1', fs=5.3, fw='normal', tc='#222', max_w=30):
    wrapped = '\n'.join(textwrap.wrap(txt, width=max_w))
    return ax.text(x, y, wrapped, ha='center', va='center',
                   fontsize=fs, fontweight=fw, color=tc, zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=fc,
                             edgecolor='#555', linewidth=0.6))

# ── Component box positions (y) ──
YC1, YC2, YC3 = 16.4, 13.8, 11.2   # wheel: 3 components
YL1, YL2 = 16.4, 13.8                # leg: 2 components
YOUT = 9.5                           # both output boxes

# ── Exact midpoints for + signs ──
YP_W1 = (YC1 + YC2) / 2  # 15.1
YP_W2 = (YC2 + YC3) / 2  # 12.5
YP_L1 = (YL1 + YL2) / 2  # 15.1

# ═══ TITLE ═══
ax.text(5, 21.3, 'ACC Two-Channel Torque Assembly', ha='center', fontsize=8, fontweight='bold')

# ═══ PHYSICAL CONDITIONS ═══
ax.text(5, 20.3, 'Physical Conditions  —  Smoothstep Gate Inputs', ha='center',
        fontsize=6.5, fontweight='bold', color='#666')
abox(5, 19.2, 'Proximity |Δx|  ·  Quietness EMA(|v|)  ·  Stability θ, θ̇  ·  Contact Fz^L, Fz^R  ·  Terrain h^L_gnd, h^R_gnd',
     '#fff9c4', 4.8, max_w=54)

# ═══ CHANNEL HEADERS ═══
ax.text(2.5, 18.0, 'Wheel Torque  τ_w',
        ha='center', fontsize=7, fontweight='bold', color='#1a5276')
ax.text(7.5, 18.0, 'Leg-Joint Torque  τ_q',
        ha='center', fontsize=7, fontweight='bold', color='#1e8449')

# ═══ WHEEL CHANNEL (left, 3 components) ═══
abox(2.5, YC1, 'τ_balance\nSagittal Balance Core\n(pitch PD + vel. damp + pos-P\n + lateral + yaw)\nalways active',
     '#d4e6f1', 5.2, 'bold', max_w=28)
abox(2.5, YC2, 'g_anchor · τ_anchor\nProximity-Gated Anchor\n(integral + damping boost)\ngated by g_prox, g_env',
     '#d4e6f1', 5.2, 'bold', max_w=28)
abox(2.5, YC3, 'g_flight · τ_flight\nContact-Loss Recovery\n(reaction-wheel attitude PD)\ngated by g_flight',
     '#d4e6f1', 5.2, 'bold', max_w=28)

# + signs: exact midpoint, same column (x=0.4)
for yp in [YP_W1, YP_W2]:
    ax.text(0.4, yp, '+', fontsize=9, fontweight='bold', color='#444', va='center', ha='center', zorder=6)

abox(2.5, YOUT, 'τ_w = τ_bal + g_anc·τ_anc + g_flt·τ_flt', '#abebc6', 5.5, 'bold', '#1a5276', max_w=32)

# ═══ LEG CHANNEL (right, 2 components) ═══
abox(7.5, YL1, 'τ_posture(h_cmd)\nPosture & Yaw Stability\n(Jacobian PD + hip-yaw\ndivergence + homing)\nalways active, height-scheduled',
     '#d5f5e3', 5.2, 'bold', max_w=28)
abox(7.5, YL2, 'g_terrain · Δτ_posture\nPer-Leg Ground Adaptation\n(split height commands\n+ leveling integrator)\ngated by g_terrain',
     '#d5f5e3', 5.2, 'bold', max_w=28)

# + sign: exact midpoint, same column (x=9.6 — symmetric with left x=0.4)
ax.text(9.6, YP_L1, '+', fontsize=9, fontweight='bold', color='#444', va='center', ha='center', zorder=6)

# Leg flow arrow
ax.annotate('', xy=(7.5, 10.8), xytext=(7.5, 12.3),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.0))

abox(7.5, YOUT, 'τ_q = τ_post + g_terr·Δτ_post', '#abebc6', 5.5, 'bold', '#1e8449', max_w=32)

# ═══ DASHED CONNECTIONS (through margins, zorder=0 behind boxes) ═══
# Left: prox/env → anchor + contact → flight
ax.plot([2.0, 0.1, 0.1, 0.5], [18.9, 18.9, 8.5, 8.5],
        '--', color='#c0392b', linewidth=0.6, zorder=0, alpha=0.5)
ax.plot([0.5, 0.5], [8.5, YC3 + 0.1], '--', color='#c0392b', linewidth=0.6, zorder=0, alpha=0.5)
ax.plot([0.5, 0.5], [8.5, YC2 + 0.1], '--', color='#c0392b', linewidth=0.6, zorder=0, alpha=0.5)
# Right: terrain → terrain component
ax.plot([8.0, 9.9, 9.9, 9.5], [18.9, 18.9, 8.5, 8.5],
        '--', color='#c0392b', linewidth=0.6, zorder=0, alpha=0.5)
ax.plot([9.5, 9.5], [8.5, YL2 + 0.1], '--', color='#c0392b', linewidth=0.6, zorder=0, alpha=0.5)

# ═══ GATE DEFINITIONS (consistent spacing, 3 rows) ═══
ax.text(5, 8.6, 'Smoothstep Gate Definitions', ha='center', fontsize=5.8, fontweight='bold', color='#888')

abox(5, 7.7, 'g_prox = 1−ss(|Δx|, 0.05, 0.15)    g_env = 1−ss(EMA|v|, 0.18, 0.30)',
     '#fce4ec', 4.6, max_w=56)

abox(5, 6.5, 'Asymmetric EMA: α_attack=0.35 (τ≈30ms), α_release=0.007 (τ≈1.5s)    k_p = 50−15·(1−g_env·g_prox)',
     '#e8daef', 4.6, max_w=56)

abox(5, 5.3, 'g_flight: Fz < 0.5mg for ≥2 steps, release ≥5 steps + 150ms ramp    g_terrain: |Δh| > 2cm, unloaded freeze',
     '#fce4ec', 4.6, max_w=56)

# ═══ BOTTOM NOTE + LEGEND ═══
ax.text(5, 4.2, 'Both channels assembled independently → final 10-DoF torque command',
        ha='center', fontsize=5.2, color='#888', style='italic')

leg = ax.legend(
    [mpatches.Patch(color='#d4e6f1'), mpatches.Patch(color='#d5f5e3'),
     mpatches.Patch(color='#abebc6'), mpatches.Patch(color='#fff9c4'),
     mpatches.Patch(color='#fce4ec'), mpatches.Patch(color='#e8daef')],
    ['Wheel torque', 'Leg-joint torque', 'Channel output',
     'Physical condition', 'Gate definition', 'Scheduled gain'],
    loc='lower center', ncol=3, fontsize=4.8, framealpha=0.9,
    bbox_to_anchor=(0.5, 0.12))
ax.add_artist(leg)

# ═══ SEPARATOR (span from below title to above legend area) ═══
ax.axvline(x=5.0, ymin=0.25, ymax=0.96, color='#bbb', linewidth=1.2, zorder=0)

# ═══════════════════════════════════════════════════════════════
# SELF-CHECK
# ═══════════════════════════════════════════════════════════════
print("Self-check...")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

overlaps = []
text_items = [(c, c.get_window_extent(renderer)) for c in ax.get_children()
              if isinstance(c, plt.Text)]
text_items = [(t, b) for t, b in text_items if b and b.width > 5 and b.height > 5]

for i in range(len(text_items)):
    for j in range(i+1, len(text_items)):
        ti, bi = text_items[i]; tj, bj = text_items[j]
        if not bi.overlaps(bj):
            continue
        ti_t = ti.get_text()[:40].strip() or '·'
        tj_t = tj.get_text()[:40].strip() or '·'
        if ti_t == '+' or tj_t == '+':
            continue
        if ti_t == tj_t:
            continue
        x_ov = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
        y_ov = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
        overlaps.append((ti_t, tj_t, x_ov, y_ov))

if overlaps:
    print(f"\n*** {len(overlaps)} OVERLAPS ***")
    for ti, tj, x, y in overlaps[:15]:
        print(f"  '{ti}' <> '{tj}' (x={x:.0f} y={y:.0f}px)")
    print("FAILED")
else:
    print("Zero overlaps — PASSED")

plt.savefig(OUT, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
print("Fig.2 saved to", OUT)
