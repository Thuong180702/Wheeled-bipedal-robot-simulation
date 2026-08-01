"""
Static torque balance of the sagittal parking offset.
Generates: paper/figures/parking_offset.pdf

Data: outputs/paper_verification/parking_offset_diag.json
      (scripts/diagnose_parking_offset.py)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
D = json.load(open(ROOT / "outputs" / "paper_verification" / "parking_offset_diag.json"))

KPOS = 40.0
DT = 0.01


def g(tag, key):
    v = D[tag][key]
    return v["mean"] if isinstance(v, dict) else v


B = "acc_baseline"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.2, "figure.dpi": 150,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75))

# ── (a) Torque balance at the -27 mm parking point ──────────────────────────
labels = [r"$\tau_{\mathrm{gravity}}$",
          r"$I_{\mathrm{anchor}}$",
          r"$I_{\max}$ (cap)",
          r"$|\tau_{\mathrm{pitch}}|$",
          r"$\tau_{\mathrm{position}}$"]
vals = [g(B, "grav_tau"), g(B, "anchor_integ"), g(B, "integ_cap_nm"),
        abs(g(B, "tau_pitch")), g(B, "tau_position")]
cols = ["#bbbbbb", "#6699cc", "#dddddd", "#cc5544", "#44aa77"]
y = np.arange(len(labels))[::-1]
ax1.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.62)
for yi, v in zip(y, vals):
    ax1.text(v + 0.05, yi, f"{v:.3f}", va="center", fontsize=8)
ax1.set_yticks(y)
ax1.set_yticklabels(labels)
ax1.set_xlim(0, 2.45)
ax1.set_xlabel("sagittal torque about the wheel axle (N$\\cdot$m)")
ax1.set_title("(a) torque balance at $\\bar{s}=-27$ mm", loc="left")
ax1.grid(axis="x", alpha=0.3, linewidth=0.5)
ax1.set_axisbelow(True)
# The hypothesis under test: gravity balanced by a saturated integral.
ax1.text(0.72, y[0], "gravity $\\approx$ 0 (CoM lever 0.04 mm)",
         fontsize=7.5, va="center", color="#555555")
ax1.text(0.72, y[1], "10% of cap: not saturated",
         fontsize=7.5, va="center", color="#3366aa")

# ── (b) Steady-state offset vs. integrator DC gain ──────────────────────────
tags = [t for t in D if t.startswith("ki_")] + [B]
pts = []
for t in tags:
    ki, leak = g(t, "ki"), g(t, "leak")
    pts.append((ki * DT * (1.0 - leak) / leak, abs(g(t, "sag_pos_err_mm"))))
pts.sort()
kdc = np.array([p[0] for p in pts])
emeas = np.array([p[1] for p in pts])

tau_p, trim = abs(g(B, "tau_pitch")), g(B, "ext_trim")
kk = np.linspace(0, 340, 400)
ax2.plot(kk, 1000.0 * (tau_p - trim) / (KPOS + kk), "-", color="#333333",
         label="closed form (see caption)")
ax2.plot(kdc, emeas, "o", ms=5, color="#cc5544", mec="black", mew=0.5,
         label="measured, $N{=}3$")
ax2.axhline(abs(g("pitcheq_fix_n3", "sag_pos_err_mm")), ls=":", lw=1.0,
            color="#44aa77")
ax2.text(58, 1.2, "feedforward pitch recalibrated,\n"
                  "$k_{I,\\mathrm{dc}}$ unchanged: 0.05 mm",
         fontsize=7.5, color="#2e7d55")
ax2.annotate("ACC as published ($k_i{=}4$)", xy=(kdc[0] + 3, emeas[0]),
             xytext=(52, 24.6), fontsize=7.5, va="center",
             arrowprops=dict(arrowstyle="->", lw=0.7))
ax2.set_xlabel(r"integrator DC gain $k_{I,\mathrm{dc}}$ (N$\cdot$m/m)")
ax2.set_ylabel("steady-state position error (mm)")
ax2.set_title("(b) offset is set by finite DC gain", loc="left")
ax2.set_xlim(0, 340)
ax2.set_ylim(0, 28)
ax2.grid(alpha=0.3, linewidth=0.5)
ax2.set_axisbelow(True)
ax2.legend(loc=(0.40, 0.66), frameon=False, handlelength=1.4,
           borderpad=0.1, labelspacing=0.3)

fig.tight_layout(pad=0.4)
out = ROOT / "paper" / "figures" / "parking_offset.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

# Self-check: the closed form evaluated with each run's OWN measured tau_pitch
# and trim must track every swept point. (The curve drawn above instead holds
# tau_pitch at its baseline value, so it drifts a few percent at high gain,
# where the shifted parking point slightly changes the outer-loop term.)
tags_all = [t for t in D if t.startswith(("ki_", "heightchk_")) or t == B]
err = np.array([abs(g(t, "pred_err_mm") - abs(g(t, "sag_pos_err_mm")))
                / abs(g(t, "sag_pos_err_mm")) for t in tags_all])
assert err.max() < 0.02, f"closed form off by {err.max():.1%} for {tags_all[int(err.argmax())]}"
print(f"closed form, per-run tau_pitch: max deviation {100 * err.max():.2f}% "
      f"over {len(tags_all)} gain/height points")
