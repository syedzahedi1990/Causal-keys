"""Figure 1 schematic: the writing token's key (looked up by later candidate mentions, hop 1; the answer then reads the option word, hop 2) vs value (copied), and the format arms."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, INK, INK2, FILL = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#f1f0ec"
DARK = "#a8401a"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")


def box(x, y, w, h, text, fc=FILL, ec=INK2, fs=6.6, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color)


def arrow(x0, y0, x1, y1, color, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=color, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# panel (a): token stream
ax.text(0.5, 38.3, "(a) Two ways later tokens can read the token that wrote a value", fontsize=7.4, weight="bold", color=INK)
y = 21
box(1, y, 20, 5, "…moved to the")
box(23, y, 9, 5, "shelf", fc="#dcebfb", ec=BLUE, weight="bold")
box(34, y, 15, 5, "… Question …")
box(51, y, 17, 5, "Choices: box,")
box(70, y, 8, 5, "shelf", fc="#fde3d8", ec=ORANGE, weight="bold")
box(80, y, 18, 5, "… Answer: ▢")
ax.text(27.5, 18.2, "writing token\nkey $K$ · value $V$", ha="center", va="top", fontsize=6.2, color=BLUE)
arrow(74, 26.4, 28.5, 26.4, ORANGE, rad=0.25)
ax.text(47, 33.2, "key lookup, hop 1 (duplicate-token heads): a later mention of the candidate matches the writing token's key",
        ha="center", fontsize=6.0, color=ORANGE)
arrow(88, 26.4, 75.5, 26.4, DARK, rad=0.45)
ax.text(89.5, 30.6, "hop 2: the answer\nreads the option word", ha="center", va="bottom", fontsize=5.6, color=DARK)
arrow(31, 20.6, 88, 20.6, BLUE, rad=0.18, ls="--")
ax.text(60, 13.4, "value copy: later tokens copy the writing token's value", ha="center", fontsize=6.3, color=BLUE)

# panel (b): format arms
ax.text(0.5, 8.6, "(b) Prompt formats (same story; candidates named after, before or not at all)", fontsize=7.4, weight="bold", color=INK)
arms = [("options-after", "story · Q · Choices: … · Answer:"), ("letters-after", "story · Q · A) … F) · Answer:"),
        ("sentence-after", "story · \"…has a box, …\" · Q"), ("no-mention", "story · Q · Answer:"),
        ("list-before", "Choices: … · story · Q")]
for i, (name, desc) in enumerate(arms):
    x = 1 + i * 19.8
    box(x, 0.6, 18.6, 5.6, "", fc="white")
    ax.text(x + 9.3, 4.6, name, ha="center", fontsize=6.4, weight="bold", color=INK)
    ax.text(x + 9.3, 2.1, desc, ha="center", fontsize=5.2, color=INK2)
fig.tight_layout(pad=0.2)
out = Path(__file__).resolve().parent / "figures" / "fig_schematic.pdf"
fig.savefig(out)
print("wrote", out)
