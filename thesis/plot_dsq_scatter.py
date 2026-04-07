"""Generate DSQ vs success rate scatter plot for thesis domain invariance section.

Reads all domain_invariance_*.json files and plots DSQ vs success rate.
Color = obstacle type + row, Shape = perturbation category.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 6,
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

PERT_CATEGORIES = {
    "baseline": "baseline",
    "tex_concrete": "realistic", "tex_stone_tiles": "realistic",
    "tex_gravel": "realistic", "tex_wood": "realistic", "tex_grass": "realistic",
    "tex_noise": "structured", "tex_bricks": "structured",
    "tex_none": "featureless", "tex_solid_red": "featureless",
    "tex_solid_blue": "featureless",
    "light_dim": "lighting", "light_bright": "lighting",
    "light_low_gamma": "lighting",
    "jitter_brightness_mild": "jitter", "jitter_brightness_strong": "jitter",
    "jitter_color_mild": "jitter", "jitter_color_strong": "jitter",
    "combined_mild": "combined", "combined_extreme": "combined",
}

for jf in sorted(results_dir.glob("domain_invariance_*.json")):
    if "edsq" in jf.name:
        continue
    data = json.load(open(jf))
    cfg = data["config"]
    family = cfg["force_family"]
    row = cfg["force_row"]
    for r in data["results"]:
        dsq = r.get("depth_signal_quality")
        edsq = r.get("edge_dsq")
        sr = r["success_rate"]
        if dsq is None or sr < 0:
            continue
        pert = r["perturbation"]
        cat = PERT_CATEGORIES.get(pert, "other")
        all_points.append({
            "family": family, "row": row,
            "perturbation": pert, "cat": cat,
            "dsq": dsq, "edsq": edsq, "success": sr,
        })

# Color = obstacle + row
terrain_colors = {
    ("gap", 0):          "#1f77b4",  # blue
    ("gap", 3):          "#08306b",  # dark blue
    ("stairs", 0):       "#2ca02c",  # green
    ("stairs", 3):       "#006d2c",  # dark green
    ("hurdle_block", 0): "#ff7f0e",  # orange
    ("hurdle_block", 3): "#d62728",  # red
}
terrain_labels = {
    ("gap", 0): "Gap R0", ("gap", 3): "Gap R3",
    ("stairs", 0): "Stairs R0", ("stairs", 3): "Stairs R3",
    ("hurdle_block", 0): "Hurdle R0", ("hurdle_block", 3): "Hurdle R3",
}

# Shape = perturbation category
cat_markers = {
    "baseline":    "*",
    "realistic":   "o",
    "structured":  "s",
    "featureless": "X",
    "lighting":    "D",
    "jitter":      "^",
    "combined":    "P",
}
cat_labels = {
    "baseline":    "Baseline",
    "realistic":   "Realistic textures",
    "structured":  "Structured textures",
    "featureless": "Featureless",
    "lighting":    "Lighting",
    "jitter":      "Color/brightness",
    "combined":    "Combined",
}
cat_sizes = {
    "baseline": 80, "realistic": 35, "structured": 35,
    "featureless": 45, "lighting": 35, "jitter": 35, "combined": 45,
}

fig, ax = plt.subplots(figsize=(4.8, 3.8))

for p in all_points:
    color = terrain_colors[(p["family"], p["row"])]
    marker = cat_markers[p["cat"]]
    size = cat_sizes[p["cat"]]
    ax.scatter(p["dsq"], p["success"] * 100,
               c=color, marker=marker, s=size,
               edgecolors="white", linewidths=0.3, alpha=0.85, zorder=5)

ax.set_xlabel("Depth Signal Quality (DSQ)")
ax.set_ylabel("Success Rate (%)")
ax.set_xlim(0.68, 0.96)
ax.set_ylim(-3, 103)
ax.grid(True, alpha=0.3)
ax.axvline(x=0.85, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.855, 3, "DSQ = 0.85", fontsize=6.5, color="grey", alpha=0.7)

# Two-part legend: colors (terrain) + shapes (perturbation)
color_handles = [Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=terrain_colors[k], markersize=6,
                        label=terrain_labels[k])
                 for k in terrain_colors]
shape_handles = [Line2D([0], [0], marker=m, color="w",
                        markerfacecolor="grey", markersize=6,
                        label=cat_labels[cat])
                 for cat, m in cat_markers.items()]

leg1 = ax.legend(handles=color_handles, loc="upper left", framealpha=0.9,
                 title="Terrain", title_fontsize=6.5, borderpad=0.4)
ax.add_artist(leg1)
ax.legend(handles=shape_handles, loc="center left", framealpha=0.9,
          title="Perturbation", title_fontsize=6.5, borderpad=0.4,
          bbox_to_anchor=(0.0, 0.38))

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
    fig2.colorbar(sc, ax=ax2, label="Success Rate (%)")
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

    from scipy import stats
    r_dsq, p_dsq = stats.pearsonr(dsqs, srs)
    r_edsq, p_edsq = stats.pearsonr(edsqs, srs)
    print(f"\nCorrelation with success rate:")
    print(f"  Global DSQ:  r={r_dsq:.3f}, p={p_dsq:.4f}")
    print(f"  Edge DSQ:    r={r_edsq:.3f}, p={p_edsq:.4f}")

plt.close("all")
