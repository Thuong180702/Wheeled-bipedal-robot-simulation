#!/usr/bin/env python3
"""Fig.2: ACC Two-Channel Torque Assembly — clean layout, no overlap, uniform boxes."""
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

# ─── Layout ───
XL, XR, XC = 2.0, 8.0, 5.0             # left col, right col, centre divider
BW   = 3.2                               # uniform box width
BFS  = 4.9                               # box content fontsize
GAP  = 1.1                               # vertical gap between boxes
OUT_H = 0.5                              # output box height

# ─── Pre-compute all box texts (wrapped), find max lines ───
BOX_TEXTS = {
    'balance':  ('τ_balance\nSagittal Balance Core\n(pitch PD + vel. damp + pos-P\n + lateral + yaw)\nalways active', '#d4e6f1', '#222'),
    'anchor':   ('g_anchor · τ_anchor\nProximity-Gated Anchor\n(integral + damping boost)\ngated by g_prox, g_env', '#d4e6f1', '#222'),
    'flight':   ('g_flight · τ_flight\nContact-Loss Recovery\n(reaction-wheel attitude PD)\ngated by g_flight', '#d4e6f1', '#222'),
    'posture':  ('τ_posture(h_cmd)\nPosture & Yaw Stability\n(Jacobian PD + hip-yaw\ndivergence + homing)\nalways active, height-scheduled', '#d5f5e3', '#222'),
    'terrain':  ('g_terrain · Δτ_posture\nPer-Leg Ground Adaptation\n(split height commands\n+ leveling integrator)\ngated by g_terrain', '#d5f5e3', '#222'),
}

def line_count(txt):
    return txt.count('\n') + 1

max_lines = max(line_count(t[0]) for t in BOX_TEXTS.values())  # 5

# Box height: max_lines * line_height + padding
LINE_H = 0.13                         # per-line height in data coords at BFS
BH = max_lines * LINE_H + 0.28        # uniform box height ≈ 0.93 → round to 0.95

# ─── Y-layout: bottom-up ───
Y_OUT    = 0.6
Y_COMP3  = Y_OUT + OUT_H + GAP
Y_PLUS2  = Y_COMP3 + BH + GAP/2
Y_COMP2  = Y_COMP3 + BH + GAP
Y_PLUS1  = Y_COMP2 + BH + GAP/2
Y_COMP1  = Y_COMP2 + BH + GAP
Y_HEAD   = Y_COMP1 + BH + 0.25
Y_COND   = Y_HEAD + 0.6
Y_TITLE  = Y_COND + 1.0

# Bottom section
Y_GATE_HDR = Y_OUT - 0.3
Y_GATE1    = Y_GATE_HDR - 0.7
Y_GATE2    = Y_GATE1 - 0.7
Y_GATE3    = Y_GATE2 - 0.7
Y_NOTE     = Y_GATE3 - 0.65

Y_MIN = Y_NOTE - 0.8
Y_MAX = Y_TITLE + 0.4

# ═══════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.45, 5.0))
ax.set_xlim(0, 10); ax.set_ylim(Y_MIN, Y_MAX); ax.axis('off')

# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════
def draw_box(x, y, txt, fc, tc, fw='bold'):
    """Uniform box with pre-wrapped text centred inside."""
    rect = mpatches.FancyBboxPatch(
        (x - BW/2, y), BW, BH,
        boxstyle='round,pad=0.2', facecolor=fc,
        edgecolor='#555', linewidth=0.6, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y + BH/2, txt, ha='center', va='center',
            fontsize=BFS, fontweight=fw, color=tc, zorder=6)

def draw_out(x, y, txt, fc, tc):
    """Shorter output box."""
    rect = mpatches.FancyBboxPatch(
        (x - BW/2, y), BW, OUT_H,
        boxstyle='round,pad=0.12', facecolor=fc,
        edgecolor='#555', linewidth=0.6, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y + OUT_H/2, txt, ha='center', va='center',
            fontsize=BFS, fontweight='bold', color=tc, zorder=6)

def plus(x, y):
    ax.text(x, y, '+', fontsize=11, fontweight='bold', color='#555',
            va='center', ha='center', zorder=8)

def arrow(x, yt, yb):
    ax.annotate('', xy=(x, yb), xytext=(x, yt),
                arrowprops=dict(arrowstyle='->', color='#777', lw=1.0), zorder=4)

def gate_box(x, y, txt, fc='#fce4ec'):
    lines = textwrap.wrap(txt, width=54)
    n = len(lines)
    gh = 0.2 + 0.13 * n
    rect = mpatches.FancyBboxPatch(
        (x - 4.8, y), 9.6, gh,
        boxstyle='round,pad=0.1', facecolor=fc,
        edgecolor='#ccc', linewidth=0.4, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y + gh/2, '\n'.join(lines), ha='center', va='center',
            fontsize=4.4, color='#444', zorder=6)

# ═══════════════════════════════════════════════════
# TITLE + PHYSICAL CONDITIONS
# ═══════════════════════════════════════════════════
ax.text(XC, Y_TITLE, 'ACC Two-Channel Torque Assembly',
        ha='center', fontsize=9, fontweight='bold')

ax.text(XC, Y_COND + 0.55, 'Physical Conditions  —  Smoothstep Gate Inputs',
        ha='center', fontsize=6.5, fontweight='bold', color='#555')

cond = ('Proximity |Δx|  ·  Quietness EMA(|v|)  ·  Stability θ, θ̇  ·  '
        'Contact Fzᴸ, Fzᴿ  ·  Terrain hᴸ_gnd, hᴿ_gnd')
gate_box(XC, Y_COND - 0.1, cond, '#fff9c4')

# ═══════════════════════════════════════════════════
# CHANNEL HEADERS
# ═══════════════════════════════════════════════════
ax.text(XL, Y_HEAD, 'Wheel Torque    τ_w', ha='center',
        fontsize=7.5, fontweight='bold', color='#1a5276')
ax.text(XR, Y_HEAD, 'Leg-Joint Torque    τ_q', ha='center',
        fontsize=7.5, fontweight='bold', color='#1e8449')

# ═══════════════════════════════════════════════════
# WHEEL CHANNEL — 3 comps + 2 plus + arrow + output
# ═══════════════════════════════════════════════════
draw_box(XL, Y_COMP1, *BOX_TEXTS['balance'])
plus(XL, Y_PLUS1)
draw_box(XL, Y_COMP2, *BOX_TEXTS['anchor'])
plus(XL, Y_PLUS2)
draw_box(XL, Y_COMP3, *BOX_TEXTS['flight'])
arrow(XL, Y_COMP3 - 0.05, Y_OUT + OUT_H + 0.08)
draw_out(XL, Y_OUT, 'τ_w = τ_bal + g_anc·τ_anc + g_flt·τ_flt', '#abebc6', '#1a5276')

# ═══════════════════════════════════════════════════
# LEG CHANNEL — 2 comps + 1 plus + arrow + output
# ═══════════════════════════════════════════════════
draw_box(XR, Y_COMP1, *BOX_TEXTS['posture'])
plus(XR, Y_PLUS1)
draw_box(XR, Y_COMP2, *BOX_TEXTS['terrain'])
arrow(XR, Y_COMP2 - 0.05, Y_OUT + OUT_H + 0.08)
draw_out(XR, Y_OUT, 'τ_q = τ_post + g_terr·Δτ_post', '#abebc6', '#1e8449')

# ═══════════════════════════════════════════════════
# DASHED GATE CONNECTIONS
# ═══════════════════════════════════════════════════
# Left: prox/env → anchor + flight
ax.plot([XL-0.6, 0.15, 0.15, XL-0.6],
        [Y_COND-0.55, Y_COND-0.55, Y_PLUS2, Y_PLUS2],
        '--', color='#c0392b', linewidth=0.7, zorder=0, alpha=0.5)
ax.plot([XL-0.6, XL-0.6], [Y_PLUS2, Y_COMP3+BH*0.55],
        '--', color='#c0392b', linewidth=0.7, zorder=0, alpha=0.5)
ax.plot([XL-0.6, XL-0.6], [Y_PLUS2, Y_COMP2+BH*0.55],
        '--', color='#c0392b', linewidth=0.7, zorder=0, alpha=0.5)
# Right: terrain → terrain
ax.plot([XR+0.6, 9.85, 9.85, XR+0.6],
        [Y_COND-0.55, Y_COND-0.55, Y_PLUS1, Y_PLUS1],
        '--', color='#c0392b', linewidth=0.7, zorder=0, alpha=0.5)
ax.plot([XR+0.6, XR+0.6], [Y_PLUS1, Y_COMP2+BH*0.55],
        '--', color='#c0392b', linewidth=0.7, zorder=0, alpha=0.5)

# ═══════════════════════════════════════════════════
# CENTRE DIVIDER (short, decorative)
# ═══════════════════════════════════════════════════
ax.plot([XC, XC], [Y_OUT - 0.2, Y_HEAD + 0.1],
        color='#ccc', linewidth=1.2, zorder=0)

# ═══════════════════════════════════════════════════
# GATE DEFINITIONS
# ═══════════════════════════════════════════════════
ax.text(XC, Y_GATE_HDR, 'Smoothstep Gate Definitions', ha='center',
        fontsize=5.5, fontweight='bold', color='#999')

gate_box(XC, Y_GATE1,
         'g_prox = 1−ss(|Δx|, 0.05, 0.15)    g_env = 1−ss(EMA|v|, 0.18, 0.30)',
         '#fce4ec')
gate_box(XC, Y_GATE2,
         'Asymmetric EMA: α_attack=0.35 (τ≈30ms), α_release=0.007 (τ≈1.5s)    k_p = 50−15·(1−g_env·g_prox)',
         '#e8daef')
gate_box(XC, Y_GATE3,
         'g_flight: Fz < 0.5mg for ≥2 steps, release ≥5 steps + 150ms ramp    g_terrain: |Δh| > 2cm, unloaded freeze',
         '#fce4ec')

# ═══════════════════════════════════════════════════
# BOTTOM NOTE + LEGEND
# ═══════════════════════════════════════════════════
ax.text(XC, Y_NOTE, 'Both channels assembled independently → final 10-DoF torque command',
        ha='center', fontsize=5.0, color='#888', style='italic')

leg = ax.legend(
    [mpatches.Patch(color='#d4e6f1'), mpatches.Patch(color='#d5f5e3'),
     mpatches.Patch(color='#abebc6'), mpatches.Patch(color='#fff9c4'),
     mpatches.Patch(color='#fce4ec'), mpatches.Patch(color='#e8daef')],
    ['Wheel torque', 'Leg-joint torque', 'Channel output',
     'Physical condition', 'Gate definition', 'Scheduled gain'],
    loc='lower center', ncol=3, fontsize=4.5, framealpha=0.85,
    bbox_to_anchor=(0.5, 0.0))
ax.add_artist(leg)

# ═══════════════════════════════════════════════════
# SELF-CHECK
# ═══════════════════════════════════════════════════
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
        if x_ov <= 15 and y_ov <= 15:      # ignore ≤15px trivial touches
            continue
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
