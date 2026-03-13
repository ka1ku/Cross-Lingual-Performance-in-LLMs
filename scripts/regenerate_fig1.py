"""Regenerate fig1_accuracy_heatmap.png with colorbar off to the side."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

CATEGORIES = ["math", "factual", "reasoning"]
LANGS = ["English", "Spanish", "Basque"]

def acc_from_cache(path):
    with open(path) as f:
        cache = json.load(f)
    acc = {l: {c: [] for c in CATEGORIES} for l in LANGS}
    for key, correct in cache.items():
        # key is e.g. "math_1_English" or "reasoning_10_Basque"
        parts = key.split("_")
        if len(parts) < 3:
            continue
        cat, lang = parts[0], parts[-1]
        if cat not in CATEGORIES or lang not in LANGS:
            continue
        acc[lang][cat].append(100.0 if correct else 0.0)
    for l in LANGS:
        for c in CATEGORIES:
            acc[l][c] = np.mean(acc[l][c]) if acc[l][c] else 0.0
    return acc

m_acc = acc_from_cache("results/mistral_accuracy_cache.json")
q_acc = acc_from_cache("results/qwen_accuracy_cache.json")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (acc, title) in zip(axes, [(m_acc, "Mistral-7B-Instruct-v0.2"), (q_acc, "Qwen2-7B-Instruct")]):
    data = np.array([[acc[l][c] for c in CATEGORIES] for l in LANGS])
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES], fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(LANGS, fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.0f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if data[i, j] < 40 else "black")
    ax.set_title(title, fontsize=12, pad=8)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.12)
fig.colorbar(im, cax=cax, label="Accuracy (%)")
plt.suptitle("Task Accuracy by Language and Category", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig1_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig1_accuracy_heatmap.png")
