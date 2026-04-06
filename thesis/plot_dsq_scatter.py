"""Generate DSQ vs success rate scatter plot for thesis domain invariance section."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# Data: (DSQ, success_rate, category)
# Category determines color; terrain determines marker shape.
gap0 = {
    # Realistic textures
    "Concrete":     (0.914, 0.793, "realistic"),
    "Wood":         (0.916, 0.348, "realistic"),
    "Grass":        (0.929, 0.660, "realistic"),
    "Stone tiles":  (0.910, 0.930, "realistic"),
    "Gravel":       (0.910, 0.914, "realistic"),
    # Other structured
    "Bricks":       (0.913, 0.590, "structured"),
    "Noise":        (0.928, 0.828, "structured"),
    # Featureless
    "No texture":   (0.756, 0.168, "featureless"),
    "Solid red":    (0.764, 0.227, "featureless"),
    "Solid blue":   (0.719, 0.086, "featureless"),
    # Lighting
    "Dim":          (0.938, 0.828, "lighting"),
    "Bright":       (0.916, 0.945, "lighting"),
    "Low contrast": (0.935, 0.762, "lighting"),
    # Jitter
    "B. ±30%":      (0.923, 0.930, "jitter"),
    "B. ±60%":      (0.923, 0.941, "jitter"),
    "C. ±15%":      (0.926, 0.891, "jitter"),
    "C. ±30%":      (0.923, 0.793, "jitter"),
    # Combined
    "Comb. mild":   (0.917, 0.766, "combined"),
    "Comb. extreme":(0.856, 0.340, "combined"),
    # Baseline
    "Baseline":     (0.922, 0.961, "baseline"),
}

gap3 = {
    "Baseline":     (0.924, 0.941, "baseline"),
    "No texture":   (0.703, 0.000, "featureless"),
    "Concrete":     (0.818, 0.121, "realistic"),
    "Stone tiles":  (0.905, 0.789, "realistic"),
    "Gravel":       (0.863, 0.488, "realistic"),
    "Noise":        (0.862, 0.293, "structured"),
    "Dim":          (0.910, 0.582, "lighting"),
    "C. ±30%":      (0.873, 0.320, "jitter"),
    "Comb. mild":   (0.803, 0.082, "combined"),
    "Comb. extreme":(0.746, 0.004, "combined"),
}

# Color by perturbation category
cat_colors = {
    "baseline":    "black",
    "realistic":   "#2ca02c",
    "structured":  "#1f77b4",
    "featureless": "#d62728",
    "lighting":    "#ff7f0e",
    "jitter":      "#9467bd",
    "combined":    "#8c564b",
}
cat_labels = {
    "baseline":    "Baseline",
    "realistic":   "Realistic textures",
    "structured":  "Other structured",
    "featureless": "Featureless",
    "lighting":    "Lighting",
    "jitter":      "Color/brightness jitter",
    "combined":    "Combined",
}

# Shape by terrain
terrain_markers = {"Gap R0": "o", "Gap R3": "s"}
terrain_sizes   = {"Gap R0": 35,  "Gap R3": 40}

fig, ax = plt.subplots(figsize=(4.8, 3.8))

# Plot each terrain × category combination
for terrain_label, data in [("Gap R0", gap0), ("Gap R3", gap3)]:
    mk = terrain_markers[terrain_label]
    sz = terrain_sizes[terrain_label]
    for name, (dsq, sr, cat) in data.items():
        ax.scatter(dsq, sr * 100, color=cat_colors[cat],
                   marker=mk, s=sz, alpha=0.85,
                   edgecolors="white", linewidths=0.3, zorder=5)

# Annotate select points
annotations = {
    "Baseline (R0)":      (0.922, 96.1, (-55, 6)),
    "Stone tiles (R0)":   (0.910, 93.0, (-78, -3)),
    "No tex (R0)":        (0.756, 16.8, (5, -10)),
    "Wood (R0)":          (0.916, 34.8, (5, -8)),
    "Baseline (R3)":      (0.924, 94.1, (5, -12)),
    "Stone tiles (R3)":   (0.905, 78.9, (5, 4)),
    "No tex (R3)":        (0.703, 0.0, (5, 4)),
    "Comb. ext (R3)":     (0.746, 0.4, (-15, 8)),
}
for label, (x, y, offset) in annotations.items():
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=offset, fontsize=6, color="grey")

ax.set_xlabel("Depth Signal Quality (DSQ)")
ax.set_ylabel("Success Rate (%)")
ax.set_xlim(0.67, 0.96)
ax.set_ylim(-3, 103)
ax.grid(True, alpha=0.3)

# Threshold line
ax.axvline(x=0.85, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.855, 3, "DSQ = 0.85", fontsize=6.5, color="grey", alpha=0.7)

# Build two-part legend: colors (category) + shapes (terrain)
color_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                        markersize=6, label=cat_labels[cat])
                 for cat, c in cat_colors.items()]
shape_handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor="grey",
                        markersize=6, label=terrain)
                 for terrain, mk in terrain_markers.items()]
leg1 = ax.legend(handles=color_handles, loc="upper left", framealpha=0.9,
                 title="Perturbation type", title_fontsize=6.5)
ax.add_artist(leg1)
ax.legend(handles=shape_handles, loc="center left", framealpha=0.9,
          title="Terrain", title_fontsize=6.5)

fig.tight_layout()
fig.savefig("thesis/figures/dsq_scatter.pdf")
fig.savefig("thesis/figures/dsq_scatter.png")
print("Saved thesis/figures/dsq_scatter.pdf and .png")
