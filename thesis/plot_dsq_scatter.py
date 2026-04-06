"""Generate DSQ vs success rate scatter plot for thesis domain invariance section."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
})

# Results from full domain invariance evaluation (256 episodes, gap Row 0)
data = {
    # (label, DSQ, success_rate, category)
    "Baseline (checkerboard)": (0.922, 0.961, "baseline"),
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
    "Dim (0.3x)":        (0.938, 0.828, "lighting"),
    "Bright (2x)":       (0.916, 0.945, "lighting"),
    "Low contrast":      (0.935, 0.762, "lighting"),
    # Jitter
    "Brightness ±30%":   (0.923, 0.930, "jitter"),
    "Brightness ±60%":   (0.923, 0.941, "jitter"),
    "Color ±15%":        (0.926, 0.891, "jitter"),
    "Color ±30%":        (0.923, 0.793, "jitter"),
    # Combined
    "Combined mild":     (0.917, 0.766, "combined"),
    "Combined extreme":  (0.856, 0.340, "combined"),
}

cat_styles = {
    "baseline":    {"color": "black",   "marker": "*", "s": 120, "zorder": 10},
    "realistic":   {"color": "#2ca02c", "marker": "o", "s": 50,  "zorder": 5},
    "structured":  {"color": "#1f77b4", "marker": "s", "s": 50,  "zorder": 5},
    "featureless": {"color": "#d62728", "marker": "X", "s": 60,  "zorder": 5},
    "lighting":    {"color": "#ff7f0e", "marker": "D", "s": 50,  "zorder": 5},
    "jitter":      {"color": "#9467bd", "marker": "^", "s": 50,  "zorder": 5},
    "combined":    {"color": "#8c564b", "marker": "P", "s": 60,  "zorder": 5},
}

cat_labels = {
    "baseline":    "Baseline",
    "realistic":   "Realistic textures",
    "structured":  "Other structured",
    "featureless": "Featureless surfaces",
    "lighting":    "Lighting",
    "jitter":      "Color/brightness jitter",
    "combined":    "Combined",
}

fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Plot each category
for cat, style in cat_styles.items():
    pts = [(name, dsq, sr) for name, (dsq, sr, c) in data.items() if c == cat]
    if not pts:
        continue
    dsqs = [p[1] for p in pts]
    srs = [p[2] for p in pts]
    ax.scatter(dsqs, [s * 100 for s in srs], label=cat_labels[cat],
               **style, edgecolors="white", linewidths=0.3)

# Add a few select labels
labels_to_show = {"Wood", "Stone tiles", "No texture", "Combined extreme", "Baseline (checkerboard)"}
for name, (dsq, sr, cat) in data.items():
    if name in labels_to_show:
        offset = (5, 5)
        if name == "Wood":
            offset = (5, -10)
        elif name == "Baseline (checkerboard)":
            offset = (-60, 5)
            name = "Baseline"
        elif name == "Combined extreme":
            offset = (5, 5)
            name = "Comb. extreme"
        ax.annotate(name, (dsq, sr * 100), textcoords="offset points",
                    xytext=offset, fontsize=6.5, color="grey")

ax.set_xlabel("Depth Signal Quality (DSQ)")
ax.set_ylabel("Success Rate (%)")
ax.set_xlim(0.68, 0.96)
ax.set_ylim(-2, 102)
ax.legend(loc="upper left", framealpha=0.9, ncol=1)
ax.grid(True, alpha=0.3)

# Add vertical threshold line
ax.axvline(x=0.85, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.855, 5, "DSQ = 0.85", fontsize=6.5, color="grey", alpha=0.7)

fig.tight_layout()
fig.savefig("thesis/figures/dsq_scatter.pdf")
fig.savefig("thesis/figures/dsq_scatter.png")
print("Saved thesis/figures/dsq_scatter.pdf and .png")
