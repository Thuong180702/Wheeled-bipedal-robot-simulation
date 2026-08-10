#!/usr/bin/env python3
"""Fig. 3: polar push envelope, colour and black-and-white variants.

Split out of generate_paper_figures.py so both variants come from one source.
They previously lived in two revisions of that script and differed only in
palette, which is how the colour figure ended up frozen at N=10 while the table
moved to N=30.

Literally the same run as the factorial ablation (Table IV): this reads the S1
row -- fixed k_p=50, i.e. canonical ACC = L3 -- straight out of the ablation
result file, so the figure and the table cannot drift apart. Do NOT plot
push_ci_all_ablations.json's "L3_S0_Full-ACC": that key was produced with empty
param_overrides, which is the S0 *scheduled*-k_p variant the paper rejects.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository root, resolved from this file so the script runs from any checkout.
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

ROOT = str(_ROOT) + '/'
OUT = ROOT + 'paper/figures/'
SRC = ROOT + 'outputs/paper_statistics/ablation_n30_results.json'

reps = {int(a): np.asarray(v)
        for a, v in json.load(open(SRC))['S1']['all_reps'].items()}
N = len(next(iter(reps.values())))
angles_deg = np.array(sorted(reps))
r_mean = np.array([reps[a].mean() for a in angles_deg])
r_sd = np.array([reps[a].std(ddof=1) for a in angles_deg])

theta_pts = np.deg2rad(angles_deg)
F_min, F_max = r_mean.min(), r_mean.max()
angle_fmin = angles_deg[np.argmin(r_mean)]
angle_fmax = angles_deg[np.argmax(r_mean)]

# close the loop
theta = np.append(theta_pts, theta_pts[0])
r = np.append(r_mean, r_mean[0])
r_lo = np.append(r_mean - r_sd, r_mean[0] - r_sd[0])
r_hi = np.append(r_mean + r_sd, r_mean[0] + r_sd[0])

# The journal prints in black and white, so the bw variant may not depend on
# hue: the band is a grey fill, the envelope a black line, and F_min / F_max are
# told apart by marker shape (circle vs. square) and by their own labels.
PALETTES = {
    'polar_push_envelope.pdf': dict(
        band='blue', band_alpha=0.18, line='-', line_c='blue', fill='blue',
        fill_alpha=0.07, c_min='red', c_max='#2196F3',
        ls_min='--', ls_max='--', mk_max='o', hl_alpha=0.35),
    'polar_push_envelope_bw.pdf': dict(
        band='0.55', band_alpha=0.35, line='-', line_c='black', fill='0.4',
        fill_alpha=0.10, c_min='black', c_max='black',
        ls_min='--', ls_max=':', mk_max='s', hl_alpha=0.45),
}

for fname, p in PALETTES.items():
    fig = plt.figure(figsize=(3.45, 3.15))
    ax = fig.add_subplot(111, projection='polar')

    ax.fill_between(theta, r_lo, r_hi, color=p['band'], alpha=p['band_alpha'],
                    linewidth=0, zorder=3)
    ax.plot(theta, r, p['line'], color=p['line_c'], linewidth=1.5, zorder=4)
    ax.fill(theta, r, alpha=p['fill_alpha'], color=p['fill'], zorder=1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # Bearings are world-frame f = F*(cos T, sin T, 0). This robot's wheel axles
    # lie along world x, so world y is the sagittal (rolling/driving) axis and
    # world x is the lateral (track) axis -- see scripts/verify_body_axes.py.
    # Hence +-90 deg are the fore/aft bearings and 0/180 deg the lateral ones,
    # NOT the other way round.
    ax.set_thetagrids(range(0, 360, 45),
                      ['$+x$ lat', '45', '$+y$ sag', '135',
                       '$-x$ lat', '-135', '$-y$ sag', '-45'], fontsize=7)
    # Radial ticks: coarse set on a spoke-free bearing (202.5 deg) so the
    # numbers neither run together nor collide with the angular tick labels.
    ax.set_rlim(0.0, 170.0)
    ax.autoscale(enable=False)  # the single-point F_min/F_max markers below
                                # would otherwise re-autoscale r off 0
    ax.set_rticks([50, 100, 150])
    ax.set_yticklabels(['50', '100', '150 N'], fontsize=7)
    ax.set_rlabel_position(202.5)
    ax.tick_params(axis='y', pad=0)
    for lbl in ax.get_yticklabels():
        lbl.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.75,
                          pad=0.8))
    ax.set_title(f'Push Recovery Envelope ($N{{=}}{N}$)', pad=8, fontsize=9,
                 fontweight='bold')

    # weakest and strongest bearing (means over the N reps)
    ax.axhline(y=F_min, color=p['c_min'], linestyle=p['ls_min'],
               alpha=p['hl_alpha'], linewidth=0.6, zorder=2)
    ax.plot(np.deg2rad(angle_fmin), F_min, 'o', color=p['c_min'], markersize=5,
            zorder=8)
    ax.text(np.deg2rad(angle_fmin - 18), 0.52 * F_min,
            f'$F_{{\\mathrm{{min}}}}$={F_min:.0f} N',
            fontsize=9, color=p['c_min'], ha='center', va='center',
            fontweight='bold', zorder=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))

    ax.axhline(y=F_max, color=p['c_max'], linestyle=p['ls_max'],
               alpha=p['hl_alpha'], linewidth=0.6, zorder=2)
    ax.plot(np.deg2rad(angle_fmax), F_max, p['mk_max'], color=p['c_max'],
            markersize=5, zorder=8)
    # Keep the label inside the rlim (170) and off the rim so it cannot collide
    # with the angular tick ring; the white bbox guards the radial grid lines.
    ax.text(np.deg2rad(angle_fmax + 26), 0.80 * F_max,
            f'$F_{{\\mathrm{{max}}}}$={F_max:.0f} N',
            fontsize=9, color=p['c_max'], ha='center', va='center',
            fontweight='bold', zorder=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))

    plt.savefig(OUT + fname, dpi=200)
    plt.close()

# The caption quotes these; print them so a re-render that moves a number is
# visible instead of silently disagreeing with the text.
print(f"Fig.3 saved (N={N}): " + " > ".join(
    f"{a}deg({v:.1f}N)" for a, v in
    sorted(zip(angles_deg, r_mean), key=lambda x: -x[1])))
