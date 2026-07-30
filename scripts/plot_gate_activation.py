"""
Plot gate activation time series for paper Figure.
Generates: paper/figures/gate_activation.pdf
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
data_path = ROOT / "outputs" / "paper_statistics" / "gate_activation_timeseries.json"
d = json.load(open(data_path))
recs = d["records"]

t = np.array([r["t"] for r in recs])
ema = np.array([r["ema"] for r in recs])
abs_vel = np.array([r["abs_sag_vel"] for r in recs])
g_prox = np.array([r["g_prox"] for r in recs])
g_env = np.array([r["g_env"] for r in recs])
g_stab = np.array([r["g_stab"] for r in recs])
g_boost = np.array([r["g_boost"] for r in recs])
pushing = np.array([r["pushing"] for r in recs])

PUSH_T0 = d["metadata"]["push_start_s"]
PUSH_DUR = d["metadata"]["push_dur_steps"] * d["metadata"]["dt_s"]

# ── Figure ──
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.0, "figure.dpi": 150,
})
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.0, 5.0), sharex=True)

# ── Shaded push region ──
for ax in (ax1, ax2, ax3):
    ax.axvspan(PUSH_T0, PUSH_T0 + PUSH_DUR, color="grey", alpha=0.12, lw=0)

# ── Panel 1: Raw |v_sag| + EMA envelope ──
ax1.plot(t, abs_vel, color="#4A90D9", alpha=0.35, lw=0.6, label=r"$|v_{\mathrm{sag}}|$ (raw)")
ax1.plot(t, ema, color="#D94A4A", lw=1.4, label=r"EMA envelope")
ax1.axhline(0.25, color="grey", ls="--", lw=0.6, alpha=0.5)
ax1.axhline(0.50, color="grey", ls="--", lw=0.6, alpha=0.5)
ax1.text(19.2, 0.255, r"$g_{\mathrm{env}}$ low", fontsize=6, color="grey", va="bottom")
ax1.text(19.2, 0.505, r"$g_{\mathrm{env}}$ high", fontsize=6, color="grey", va="bottom")
ax1.set_ylabel(r"Velocity (m/s)")
ax1.set_ylim(-0.02, 0.65)
ax1.legend(loc="upper right", framealpha=0.8, ncol=2)
ax1.set_title("Envelope follower + gate activation during push-recovery cycle")

# ── Panel 2: Gate states ──
ax2.plot(t, g_prox, color="#2E8B57", lw=1.2, label=r"$g_{\mathrm{prox}}$ (proximity)")
ax2.plot(t, g_env, color="#CD853F", lw=1.2, label=r"$g_{\mathrm{env}}$ (quiet-stance)")
ax2.plot(t, g_stab, color="#9370DB", lw=1.2, label=r"$g_{\theta}$ (pitch stability)")
ax2.set_ylabel("Gate state")
ax2.set_ylim(-0.05, 1.12)
ax2.legend(loc="upper right", framealpha=0.8, ncol=3)
ax2.set_yticks([0, 0.5, 1])

# ── Panel 3: Composite output ──
ax3.plot(t, g_boost, color="#1E3A5F", lw=1.4, label=r"$g_{\mathrm{boost}}$ (damping boost)")
ax3.set_ylabel(r"$g_{\mathrm{boost}}$")
ax3.set_ylim(-0.05, 1.12)
ax3.set_xlabel("Time (s)")
ax3.legend(loc="upper right", framealpha=0.8)

# Annotations
ax2.annotate("push\n(90 N)", xy=(PUSH_T0 + 0.03, 0.15), fontsize=7, color="#333333",
            xytext=(PUSH_T0 + 0.5, 0.25), arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))

plt.tight_layout(pad=0.8)
out_path = ROOT / "paper" / "figures" / "gate_activation.pdf"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
