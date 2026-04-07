"""Generate DSQ vs success rate scatter plot for thesis domain invariance section.

Reads all domain_invariance_*.json files and plots DSQ (global and edge-weighted)
vs success rate, with different markers per obstacle family.
"""

import json
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
})

# Load all results
results_dir = Path("thesis")
all_points = []

for jf in sorted(results_dir.glob("domain_invariance_*.json")):
    if "edsq" in jf.name:
        continue  # skip re-run files that duplicate
    data = json.load(open(jf))
    cfg = data["config"]
    family = cfg["force_family"]
    row = cfg["force_row"]
    label = f"{family} R{row}"
    for r in data["results"]:
        dsq = r.get("depth_signal_quality")
        edsq = r.get("edge_dsq")
        sr = r["success_rate"]
        if dsq is None or sr < 0:
            continue
        cat = "baseline" if r["perturbation"] == "baseline" else \
              "featureless" if r["perturbation"] in ("tex_none", "tex_solid_red", "tex_solid_blue") else \
              "textured"
        all_points.append({
            "label": label,
            "perturbation": r["perturbation"],
            "family": family,
            "row": row,
            "dsq": dsq,
            "edsq": edsq,
            "success": sr,
            "cat": cat,
        })

# ── Figure 1: DSQ vs Success, colored by obstacle family ──
family_styles = {
    ("gap", 0):          {"color": "#1f77b4", "marker": "o", "label": "Gap R0"},
    ("gap", 3):          {"color": "#1f77b4", "marker": "s", "label": "Gap R3"},
    ("stairs", 0):       {"color": "#2ca02c", "marker": "o", "label": "Stairs R0"},
    ("stairs", 3):       {"color": "#2ca02c", "marker": "s", "label": "Stairs R3"},
    ("hurdle_block", 0): {"color": "#ff7f0e", "marker": "o", "label": "Hurdle R0"},
    ("hurdle_block", 3): {"color": "#ff7f0e", "marker": "s", "label": "Hurdle R3"},
}

fig, ax = plt.subplots(figsize=(4.5, 3.5))

for (fam, row), style in family_styles.items():
    pts = [p for p in all_points if p["family"] == fam and p["row"] == row]
    if not pts:
        continue
    dsqs = [p["dsq"] for p in pts]
    srs = [p["success"] * 100 for p in pts]
    ax.scatter(dsqs, srs, label=style["label"],
               c=style["color"], marker=style["marker"], s=35,
               edgecolors="white", linewidths=0.3, alpha=0.8)

ax.set_xlabel("Depth Signal Quality (DSQ)")
ax.set_ylabel("Success Rate (%)")
ax.set_xlim(0.68, 0.96)
ax.set_ylim(-2, 102)
ax.legend(loc="upper left", framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
ax.axvline(x=0.85, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.855, 5, "DSQ = 0.85", fontsize=6.5, color="grey", alpha=0.7)

fig.tight_layout()
fig.savefig("thesis/figures/dsq_scatter.pdf")
fig.savefig("thesis/figures/dsq_scatter.png")
print("Saved dsq_scatter.pdf/png")

# ── Figure 2: Edge-DSQ vs DSQ, colored by success ──
pts_with_edsq = [p for p in all_points if p["edsq"] is not None]
if pts_with_edsq:
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
    dsqs = np.array([p["dsq"] for p in pts_with_edsq])
    edsqs = np.array([p["edsq"] for p in pts_with_edsq])
    srs = np.array([p["success"] * 100 for p in pts_with_edsq])

    sc = ax2.scatter(dsqs, edsqs, c=srs, cmap="RdYlGn", s=35,
                     edgecolors="grey", linewidths=0.3, vmin=0, vmax=100)
    cbar = fig2.colorbar(sc, ax=ax2, label="Success Rate (%)")
    ax2.plot([0.65, 0.96], [0.65, 0.96], 'k--', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Global DSQ")
    ax2.set_ylabel("Edge-weighted DSQ (eDSQ)")
    ax2.set_xlim(0.68, 0.96)
    ax2.set_ylim(0.68, 0.96)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("thesis/figures/edsq_vs_dsq.pdf")
    fig2.savefig("thesis/figures/edsq_vs_dsq.png")
    print("Saved edsq_vs_dsq.pdf/png")

    # Print correlation analysis
    from scipy import stats
    r_dsq, p_dsq = stats.pearsonr(dsqs, srs)
    r_edsq, p_edsq = stats.pearsonr(edsqs, srs)
    print(f"\nCorrelation with success rate:")
    print(f"  Global DSQ:  r={r_dsq:.3f}, p={p_dsq:.4f}")
    print(f"  Edge DSQ:    r={r_edsq:.3f}, p={p_edsq:.4f}")

plt.close("all")
