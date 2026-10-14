"""Figure 1, panel (a): the two routes from the token that wrote a value. The later mention of the written word matches the
writing token's key (hop 1), and the heads that make the match write a flag into its row; the answer reads the
flagged mention by key (hop 2). Without a later mention the answer copies the writing token's value. A mention placed
before the writing token cannot attend to it (causal mask). Panels (b) and (c) of Figure 1 are fig_fresh.pdf and
fig_crossover_main.pdf (paper/make_figures.py); the LaTeX figure sets the three side by side.

Run from the repository root: python paper/make_schematic.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

BLUE, ORANGE, INK, INK2, FILL = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#f1f0ec"
DARK, FADE, LBLUE, LORANGE = "#a8401a", "#a3a19c", "#dcebfb", "#fde3d8"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})

W_IN, H_IN = 6.8, 1.9
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100 * H_IN / W_IN)   # equal units in x and y (1 unit = 0.068 in), so the arcs are not distorted
ax.axis("off")

ROW, RH = 11.0, 3.4                 # token row: bottom and height
CELL = 2.3                          # height of the cache cells K (above the writing token) and V (below it)


def box(x0, x1, y, h, text, fc=FILL, ec=INK2, fs=7.6, weight="normal", color=INK, ls="-", lw=0.7):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.7", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text((x0 + x1) / 2, y + h / 2, text, ha="center", va="center_baseline", fontsize=fs, weight=weight, color=color)


def arrow(p0, p1, color, rad=0.0, ls="-", lw=1.1):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                        linestyle=ls, shrinkA=0.5, shrinkB=0.8)
    ax.add_patch(a)
    return a


# panel title
ax.text(0.6, 26.55, "(a) Two routes from the token that wrote a value", fontsize=8.2, weight="bold", color=INK, va="center")

# token stream: a faded list before the story (the alternative placement), the story, the question, a later list, the answer
xs = {"early": (0.3, 16.4), "moved": (17.6, 31.8), "write": (33.4, 41.6), "quest": (43.4, 55.0), "list": (56.8, 68.6),
      "mention": (70.4, 78.6), "answer": (80.4, 92.0)}
box(*xs["early"], ROW, RH, "Choices: … shelf …", fc="white", ec=FADE, color=FADE, ls=(0, (2, 1.5)))
box(*xs["moved"], ROW, RH, "… moved to the")
box(*xs["write"], ROW, RH, "shelf", fc=LBLUE, ec=BLUE, weight="bold")
box(*xs["quest"], ROW, RH, "… Question …")
box(*xs["list"], ROW, RH, "Choices: box,")
box(*xs["mention"], ROW, RH, "shelf", fc=LORANGE, ec=ORANGE, weight="bold")
box(*xs["answer"], ROW, RH, "… Answer: ▢")

# the writing token's cache: key K above it, value V below it
wx0, wx1 = xs["write"]
wc = (wx0 + wx1) / 2
kb, vb = ROW + RH + 0.45, ROW - 0.45 - CELL
box(wx0, wx1, kb, CELL, "key $K$", fc="white", ec=BLUE, color=BLUE, fs=7.5, lw=0.6)
box(wx0, wx1, vb, CELL, "value $V$", fc="white", ec=BLUE, color=BLUE, fs=7.5, lw=0.6)
ax.text(wx0 - 0.9, kb + CELL / 2, "writing token", ha="right", va="center_baseline", fontsize=7.5, color=BLUE)

# the flag written into the later mention's own row (a small pennant on its box)
mx0, mx1 = xs["mention"]
mc = (mx0 + mx1) / 2
px, py = mx1 - 1.6, ROW + RH
ax.plot([px, px], [py, py + 2.7], color=DARK, lw=0.8, solid_capstyle="butt")
ax.add_patch(Polygon([(px, py + 2.7), (px + 2.1, py + 2.1), (px, py + 1.5)], closed=True, fc=DARK, ec=DARK, lw=0.4))

# hop 1: the later mention matches the writing token's key (query at the mention, key at the writing token)
arrow((mc - 1.2, ROW + RH), (wc + 1.0, kb + CELL), ORANGE, rad=0.22)
ax.text(57.0, 23.15, "hop 1: the later mention of the written word matches the writing token's key,\n"
        "and the heads that make the match write a flag into its row",
        ha="center", va="center", fontsize=7.5, color=ORANGE, linespacing=1.2)

# hop 2: the answer reads the flagged mention by key
ax.add_patch(FancyArrowPatch(((xs["answer"][0] + xs["answer"][1]) / 2 - 1.5, ROW + RH), (px + 1.2, py + 2.5), arrowstyle="-|>",
                             mutation_scale=7, color=DARK, lw=1.1, connectionstyle="arc3,rad=0.55", shrinkA=0.5, shrinkB=0.6))
ax.text(99.6, 17.0, "hop 2: the answer\nreads the flagged\nmention by key", ha="right", va="center", fontsize=7.5, color=DARK,
        linespacing=1.2)

# value copy: without a later mention the answer copies the writing token's value
ax.add_patch(FancyArrowPatch((wc, vb), ((xs["answer"][0] + xs["answer"][1]) / 2 + 1.0, ROW), arrowstyle="-|>", mutation_scale=7,
                             color=BLUE, lw=1.1, linestyle=(0, (3, 2)), connectionstyle="arc3,rad=0.16", shrinkA=0.5, shrinkB=0.8))
ax.text(62.0, 3.0, "value copy: later tokens copy the writing token's value", ha="center", va="center", fontsize=7.5, color=BLUE)

# a mention before the writing token cannot attend to it: a dotted arrow, crossed out
ex0, ex1 = xs["early"]
a = FancyArrowPatch(((ex0 + ex1) / 2, ROW), (wx0 - 0.1, ROW + 0.9), arrowstyle="-|>", mutation_scale=6, color=FADE, lw=0.9,
                    linestyle=(0, (1, 1.4)), connectionstyle="arc3,rad=0.3", shrinkA=0.5, shrinkB=0.8)
ax.add_patch(a)
mid_x = ((ex0 + ex1) / 2 + wx0 - 0.1) / 2
mid_y = ROW + 0.45 - 0.3 * ((wx0 - 0.1) - (ex0 + ex1) / 2) / 2   # the arc's lowest point (arc3: sagitta = rad x chord / 2)
ax.text(mid_x, mid_y, "×", ha="center", va="center", fontsize=9, color=INK2, weight="bold")
ax.text(0.6, 4.3, "a mention before the token cannot\nattend to it (causal mask)", ha="left", va="center", fontsize=7.5, color=INK2,
        linespacing=1.2)

out = Path(__file__).resolve().parent / "figures" / "fig_schematic.pdf"
fig.savefig(out, metadata={"CreationDate": None})   # reproducible PDF; no metadata beyond matplotlib's defaults
print("wrote", out)
