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
BW   = 3.5                               # uniform box width
BFS  = 4.9                               # box content fontsize
GAP  = 1.1                               # vertical gap between boxes
OUT_H = 0.5                              # output box height
OUT_W = 4.0                              # output box width (text is long)
OUT_FS = 4.6                             # output box fontsize

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
Y_HEAD   = Y_COMP1 + BH + 0.50
Y_COND   = Y_HEAD + 0.6
Y_TITLE  = Y_COND + 1.0

# Bottom section — gate boxes are stacked dynamically below, since their height
# depends on how many lines the definition text wraps to.
Y_GATE_HDR = Y_OUT - 0.45

Y_MAX = Y_TITLE + 0.4

# ═══════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.45, 5.0))
ax.set_xlim(0, 10); ax.axis('off')   # ylim set once the gate stack is laid out

# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════
GATE_ITEMS = []   # (text, rect) pairs; every box is checked for text containment

def fit_text(t, rect, pad_px=4):
    """Shrink a text until it fits inside its own box. Wrapping the multi-line
    component labels would break their line structure, so scale the font instead."""
    for _ in range(12):
        fig.canvas.draw()
        tb = t.get_window_extent(fig.canvas.get_renderer())
        rb = rect.get_window_extent(fig.canvas.get_renderer())
        over = max(tb.width - (rb.width - 2*pad_px), tb.height - (rb.height - 2*pad_px))
        if over <= 0:
            return
        t.set_fontsize(t.get_fontsize() * 0.94)

def draw_box(x, y, txt, fc, tc, fw='bold'):
    """Uniform box with pre-wrapped text centred inside."""
    rect = mpatches.FancyBboxPatch(
        (x - BW/2, y), BW, BH,
        boxstyle='round,pad=0.2', facecolor=fc,
        edgecolor='#555', linewidth=0.6, zorder=5)
    ax.add_patch(rect)
    t = ax.text(x, y + BH/2, txt, ha='center', va='center',
                fontsize=BFS, fontweight=fw, color=tc, zorder=6)
    fit_text(t, rect)
    GATE_ITEMS.append((t, rect))

def draw_out(x, y, txt, fc, tc):
    """Shorter output box."""
    rect = mpatches.FancyBboxPatch(
        (x - OUT_W/2, y), OUT_W, OUT_H,
        boxstyle='round,pad=0.12', facecolor=fc,
        edgecolor='#555', linewidth=0.6, zorder=5)
    ax.add_patch(rect)
    t = ax.text(x, y + OUT_H/2, txt, ha='center', va='center',
                fontsize=OUT_FS, fontweight='bold', color=tc, zorder=6)
    fit_text(t, rect)
    GATE_ITEMS.append((t, rect))

def plus(x, y):
    ax.text(x, y, '+', fontsize=11, fontweight='bold', color='#555',
            va='center', ha='center', zorder=8)

def arrow(x, yt, yb):
    ax.annotate('', xy=(x, yb), xytext=(x, yt),
                arrowprops=dict(arrowstyle='->', color='#777', lw=1.0), zorder=4)

GATE_FS = 4.4           # gate-definition font size
_wrap_cache = {}

def gate_wrap(txt):
    """Chars per line that actually fit the 9.6-wide box, measured not guessed."""
    if txt not in _wrap_cache:
        probe = ax.text(0, 0, txt, fontsize=GATE_FS, alpha=0)
        fig.canvas.draw()
        w_px = probe.get_window_extent(fig.canvas.get_renderer()).width
        probe.remove()
        box_px = 9.2 / (ax.get_xlim()[1] - ax.get_xlim()[0]) * ax.bbox.width
        _wrap_cache[txt] = max(20, int(len(txt) * box_px / w_px))
    return _wrap_cache[txt]

def gate_h(txt):
    return 0.2 + 0.13 * len(textwrap.wrap(txt, width=gate_wrap(txt)))

def gate_box(x, y, txt, fc='#fce4ec'):
    """Draw a full-width definition box with its BOTTOM edge at y."""
    lines = textwrap.wrap(txt, width=gate_wrap(txt))
    gh = 0.2 + 0.13 * len(lines)
    rect = mpatches.FancyBboxPatch(
        (x - 4.8, y), 9.6, gh,
        boxstyle='round,pad=0.1', facecolor=fc,
        edgecolor='#ccc', linewidth=0.4, zorder=5)
    ax.add_patch(rect)
    t = ax.text(x, y + gh/2, '\n'.join(lines), ha='center', va='center',
                fontsize=GATE_FS, color='#444', zorder=6)
    GATE_ITEMS.append((t, rect))
    return gh

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
HDR_L = ax.text(XL, Y_HEAD, 'Wheel Torque  τ_w', ha='center',
                fontsize=7.5, fontweight='bold', color='#1a5276')
HDR_R = ax.text(XR, Y_HEAD, 'Leg-Joint Torque  τ_q', ha='center',
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
# Routed down the inner corridor between each column and the centre divider.
# The outer margins are no longer wide enough: the boxes reach x = 0.05 / 9.95
# and the channel headers are wider still, so an outer rail would cross both.
BOX_L, BOX_R = XL + BW/2 + 0.2, XR - BW/2 - 0.2     # inner padded box edges
RAIL_L, RAIL_R = BOX_L + 0.45, BOX_R - 0.45
DASH = dict(ls='--', color='#c0392b', linewidth=0.7, zorder=0, alpha=0.55)

def gate_feed(rail, box_edge, y_from, y_targets):
    ax.plot([rail, rail], [y_from, min(y_targets)], **DASH)
    for y in y_targets:
        ax.plot([rail, box_edge], [y, y], **DASH)

# prox / env / flight gates → anchor and contact-loss boxes (wheel channel)
gate_feed(RAIL_L, BOX_L, Y_COND - 0.28,
          [Y_COMP2 + BH*0.5, Y_COMP3 + BH*0.5])
# terrain gate → per-leg adaptation box (leg channel)
gate_feed(RAIL_R, BOX_R, Y_COND - 0.28, [Y_COMP2 + BH*0.5])

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

GATE_DEFS = [
    ('g_prox = 1−ss(|Δx|; 0.05 m, 0.15 m)   │   '
     'g_env = 1−ss(EMA(|v_sag|); 0.25 m/s, 0.50 m/s)', '#fce4ec'),
    ('Asymmetric EMA on |v_sag|:  α_attack = 0.35 (τ ≈ 23 ms)   │   '
     'α_release = 0.0067 (τ ≈ 1.5 s)   │   pitch stiffness fixed at k_p = 50', '#e8daef'),
    ('g_flight: Fz < 0.5 mg for ≥2 steps, release ≥5 steps + 150 ms ramp   │   '
     'g_terrain: per-leg ground estimate, frozen while unloaded '
     '(seek on 3 mm drop-below); leveling trim on |Δh| ≥ 2 cm', '#fce4ec'),
]

_y = Y_GATE_HDR - 0.20
for _txt, _fc in GATE_DEFS:
    _y -= gate_h(_txt)
    gate_box(XC, _y, _txt, _fc)
    _y -= 0.30                     # gap (boxstyle pad adds 0.1 per edge)

Y_NOTE = _y - 0.28
Y_MIN = Y_NOTE - 1.15
ax.set_ylim(Y_MIN, Y_MAX)

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
     'Physical condition', 'Gate definition', 'Gate parameters'],
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

# channel headers must clear the dashed gate rails and each other
_inv = ax.transData.inverted()
for _h, _rail in ((HDR_L, RAIL_L), (HDR_R, RAIL_R)):
    _b = _h.get_window_extent(renderer)
    _x0, _x1 = _inv.transform((_b.x0, 0))[0], _inv.transform((_b.x1, 0))[0]
    if _x0 - 0.05 < _rail < _x1 + 0.05:
        overlaps.append((_h.get_text(), 'CROSSES GATE RAIL', 0, 0))

# text must stay inside its own definition box (wrapping can outgrow it)
for t, rect in GATE_ITEMS:
    tb = t.get_window_extent(renderer)
    rb = rect.get_window_extent(renderer)
    if tb.x0 < rb.x0 - 1 or tb.x1 > rb.x1 + 1 or tb.y0 < rb.y0 - 1 or tb.y1 > rb.y1 + 1:
        overlaps.append((t.get_text()[:40], 'OVERFLOWS ITS BOX', 0, 0))

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
