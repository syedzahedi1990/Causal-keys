"""Build paper figures (PDF) and a LaTeX macro file of every reported number, directly from saved results.

Run from the repository root: python paper/make_figures.py
Outputs: paper/figures/*.pdf and paper/numbers.tex (\\newcommand macros used by the text).
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from stage1_prereg import ci, per_core  # noqa: E402
import stage2_score as s2  # noqa: E402

FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})

SIZES = {"Qwen2.5-1.5B-Instruct": 1.5, "Qwen2.5-3B-Instruct": 3.1, "Qwen2.5-7B-Instruct": 7.6,
         "Qwen2.5-14B-Instruct": 14.7, "Qwen2.5-32B-Instruct": 32.8, "Qwen2.5-72B-Instruct": 72.7,
         "Qwen3-8B": 8.2, "Mistral-7B-Instruct-v0.3": 7.2, "Mistral-Small-24B-Instruct-2501": 23.6,
         "OLMo-2-1124-7B-Instruct": 7.3}
SHORT = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen2.5-3B-Instruct": "Qwen2.5-3B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "Qwen2.5-14B-Instruct": "Qwen2.5-14B", "Qwen2.5-32B-Instruct": "Qwen2.5-32B", "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
         "Qwen3-8B": "Qwen3-8B", "Mistral-7B-Instruct-v0.3": "Mistral-7B", "Mistral-Small-24B-Instruct-2501": "Mistral-24B",
         "OLMo-2-1124-7B-Instruct": "OLMo-2-7B"}
ARMS = ["LETTER", "P1", "POST", "NONE", "BEFORE"]
ARM_TAG = {"P1": "Opt", "LETTER": "Letter", "POST": "Post", "NONE": "None", "BEFORE": "Before"}
MODEL_TAG = {"Qwen2.5-1.5B-Instruct": "QwenOnefive", "Qwen2.5-3B-Instruct": "QwenThree", "Qwen2.5-7B-Instruct": "QwenSeven",
             "Qwen2.5-14B-Instruct": "QwenFourteen", "Qwen2.5-32B-Instruct": "QwenThirtytwo", "Qwen2.5-72B-Instruct": "QwenSeventytwo",
             "Qwen3-8B": "QwenThreeEight", "Mistral-7B-Instruct-v0.3": "MistralSeven",
             "Mistral-Small-24B-Instruct-2501": "MistralTwentyfour", "OLMo-2-1124-7B-Instruct": "OlmoSeven"}
ARM_LABEL = {"P1": "options\nafter", "LETTER": "lettered\noptions", "POST": "re-mention\nsentence",
             "NONE": "no\nre-mention", "BEFORE": "options\nbefore"}
macros = {}


def mac(name, value, fmt="{:.2f}"):
    assert name.isalpha(), f"LaTeX macro names must be letters only: {name}"
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def ratio_ci(a, b, B=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (B, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


# ---------------------------------------------------------------- natural factorial across models
files = sorted(glob.glob(str(ROOT / "results/gpu_stage1/*_s0.json")) + glob.glob(str(ROOT / "results/gpu_stage2/format_factorial/*_s0.json")))
nat = {}
for f in files:
    name = Path(f).name[: -len("_s0.json")]
    res = json.load(open(f))["results"]
    arms = {a: per_core(res, a) for a in ARMS}
    p1 = arms["P1"]
    L = next(iter(p1.values()))["L"]
    l0s = sorted({int(k.split("@")[1]) for k in next(iter(p1.values()))["d"] if k.startswith("KV_S@")})
    nat[name] = {
        "L": L,
        "onset": {l0: ratio_ci([v["d"][f"K_S@{l0}"] for v in p1.values()],
                               [v["d"][f"K_S@{l0}"] + v["d"][f"V_S@{l0}"] for v in p1.values()]) for l0 in l0s},
        "dec": {a: {q: ci([v["d"][k] for v in arms[a].values()]) for q, k in
                    (("dK", "K_S@0"), ("dV", "V_S@0"), ("dKV", "KV_S@0"))} | {
                    "int": ci([v["d"]["KV_S@0"] - v["d"]["K_S@0"] - v["d"]["V_S@0"] for v in arms[a].values()]),
                    "idV": ci([v["idV"] for v in arms[a].values()])} for a in ARMS},
        "idK": {a: ci([v["idK"] for v in arms[a].values()]) for a in ARMS},
        "share": ratio_ci([v["d"]["K_S@0"] for v in p1.values()], [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in p1.values()]),
        "shares": {a: ratio_ci([v["d"]["K_S@0"] for v in arms[a].values()],
                               [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in arms[a].values()]) for a in ARMS},
    }
order = sorted(nat, key=lambda m: SIZES[m])
for m in order:
    key = MODEL_TAG[m]
    mac(f"share{key}", nat[m]["share"][0])
    mac(f"shareLo{key}", nat[m]["share"][1])
    mac(f"shareHi{key}", nat[m]["share"][2])
    for a in ARMS:
        mac(f"idK{ARM_TAG[a]}{key}", nat[m]["idK"][a][0], "{:.1f}")
mac("nModels", len(nat), "{:d}")
before_max = max(nat[m]["idK"]["BEFORE"][0] for m in nat)
mac("beforeMax", before_max, "{:+.2f}")
none_max = max(nat[m]["idK"]["NONE"][0] for m in nat)
mac("noneMax", none_max, "{:.2f}")

# Figure 1: key share of the multiple-choice answer vs model size
fig, ax = plt.subplots(figsize=(3.3, 2.3))
qw = [m for m in order if m.startswith("Qwen2.5")]
x = [SIZES[m] for m in qw]
y = [nat[m]["share"][0] for m in qw]
lo = [nat[m]["share"][1] for m in qw]
hi = [nat[m]["share"][2] for m in qw]
ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=4, label="Qwen2.5 (1.5B–72B)")
others = [m for m in order if not m.startswith("Qwen2.5")]
ax.scatter([SIZES[m] for m in others], [nat[m]["share"][0] for m in others], s=22, color=ORANGE,
           edgecolor="white", linewidth=1, zorder=3, label="other families")
OFFS = {"Mistral-7B-Instruct-v0.3": (-44, 2), "Qwen3-8B": (6, -9), "OLMo-2-1124-7B-Instruct": (5, -8),
        "Mistral-Small-24B-Instruct-2501": (6, -4)}
for m in others:
    ax.annotate(SHORT[m], (SIZES[m], nat[m]["share"][0]), textcoords="offset points", xytext=OFFS.get(m, (4, -9)),
                fontsize=6.3, color=INK2)
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([1.5, 3, 7, 14, 32, 72])
ax.set_xticklabels(["1.5", "3", "7", "14", "32", "72"])
ax.set_xlabel("parameters (billions, log scale)")
ax.set_ylabel("key share of answer log-odds")
ax.set_ylim(0, 1)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "fig_scale.pdf")
plt.close(fig)

# Figure 2: identity carried by keys, by format, normalised to the Paper 1 format, across all models
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for j, a in enumerate(ARMS):
    vals = [nat[m]["idK"][a][0] / nat[m]["idK"]["P1"][0] for m in order]
    jitter = np.linspace(-0.18, 0.18, len(vals))
    ax.scatter(np.full(len(vals), j) + jitter, vals, s=12, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    ax.hlines(np.median(vals), j - 0.3, j + 0.3, color=INK, lw=1.5, zorder=4)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=6.3)
ax.set_ylabel("identity carried by keys\n(relative to options-after)")
ax.grid(axis="y", color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "fig_formats.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Paper 1 frames (stage 2)
frames = {}
FRAME_ARMS = ["LETTER", "P1", "POST", "NONE", "BEFORE"]
ARM_TAG = {"P1": "Opt", "LETTER": "Letter", "POST": "Post", "NONE": "None", "BEFORE": "Before"}
MODEL_TAG = {"Qwen2.5-1.5B-Instruct": "QwenOnefive", "Qwen2.5-3B-Instruct": "QwenThree", "Qwen2.5-7B-Instruct": "QwenSeven",
             "Qwen2.5-14B-Instruct": "QwenFourteen", "Qwen2.5-32B-Instruct": "QwenThirtytwo", "Qwen2.5-72B-Instruct": "QwenSeventytwo",
             "Qwen3-8B": "QwenThreeEight", "Mistral-7B-Instruct-v0.3": "MistralSeven",
             "Mistral-Small-24B-Instruct-2501": "MistralTwentyfour", "OLMo-2-1124-7B-Instruct": "OlmoSeven"}
for model, label in (("mistral", "Mistral-Small-24B"), ("qwen", "Qwen2.5-72B")):
    f = ROOT / f"results/gpu_stage2/paper1_frames/{model}.json"
    res = json.load(open(f))["results"]
    rows = {a: s2.per_core(res, a) for a in FRAME_ARMS}
    out = {}
    for a in FRAME_ARMS:
        R = rows[a]
        out[a] = {"phi": s2.ratio(R, lambda x: x["M"] - x["P"], lambda x: x["T"] - x["S"])[:3],
                  "psi": s2.ratio(R, lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"])[:3],
                  "rho": s2.ratio(R, lambda x: x["rem"] - x["M"], lambda x: x["P"] - x["M"])[:3],
                  "mT": float(np.mean([x["M_T_rate"] for x in R.values()])),
                  "n": len(R)}
        mtag = model.capitalize()
        for q in ("phi", "psi", "rho"):
            mac(f"{q}{ARM_TAG[a]}{mtag}", out[a][q][0])
            mac(f"{q}Lo{ARM_TAG[a]}{mtag}", out[a][q][1])
            mac(f"{q}Hi{ARM_TAG[a]}{mtag}", out[a][q][2])
        mac(f"mT{ARM_TAG[a]}{mtag}", out[a]["mT"])
    frames[label] = out

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
w = 0.26
for ax, (label, out) in zip(axes, frames.items()):
    xs = np.arange(len(FRAME_ARMS))
    for k, (q, col, name) in enumerate((("phi", BLUE, "full remap (behaviour)"), ("psi", ORANGE, "key-only addition"),
                                         ("rho", AQUA, "key-only removal"))):
        v = [out[a][q][0] for a in FRAME_ARMS]
        e = np.array([[out[a][q][0] - out[a][q][1] for a in FRAME_ARMS], [out[a][q][2] - out[a][q][0] for a in FRAME_ARMS]])
        ax.bar(xs + (k - 1) * w, v, w * 0.9, color=col, label=name, yerr=e, error_kw=dict(lw=0.6, ecolor=INK2), zorder=3)
        if q == "rho":
            for xi, vi in zip(xs + (k - 1) * w, v):
                ax.text(xi, max(vi, 0) + 0.04, f"{vi:.2f}", ha="center", fontsize=5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a] for a in FRAME_ARMS], fontsize=6)
    ax.set_title(label, fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
axes[0].set_ylabel("fraction of effect")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIG / "fig_frames.pdf")
plt.close(fig)

# ---------------------------------------------------------------- localisation at 1.5B (CPU)
loc = json.load(open(ROOT / "results/row_restricted_windows/Qwen2.5-1.5B-Instruct_direct.json"))
groups = [("choice_words", "option words"), ("question", "question"), ("story_tail", "story after state"),
          ("rest_after_p", "instruction / answer tail")]
fig, ax = plt.subplots(figsize=(3.5, 1.9))
for k, (arm, col) in enumerate((("P1", BLUE), ("LETTER", ORANGE))):
    R = [r for r in loc if r["arm"] == arm]
    full = [r["m"]["all"] - r["m_B"] for r in R]
    vals = [ratio_ci([r["m"][g] - r["m_B"] for r in R], full)[0] for g, _ in groups]
    for g, v in zip(groups, vals):
        mac(f"loc{ARM_TAG[arm]}{g[0].replace('_', '').capitalize()}", v)
    ys = np.arange(len(groups)) + (k - 0.5) * 0.36
    ax.barh(ys, vals, 0.34, color=col, label={"P1": "options after", "LETTER": "lettered options"}[arm], zorder=3)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([g[1] for g in groups])
ax.invert_yaxis()
ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlabel("fraction of full key effect recovered")
ax.set_xlim(-0.1, 1.1)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=6.3, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig_localisation.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Paper 1 fixed-value key share (CPU reanalysis anchors)
mac("sKQwenFV", 0.745)
mac("sKMistralFV", 0.730)

# ---------------------------------------------------------------- auto-generated tables
TAB = ROOT / "paper" / "tables"
TAB.mkdir(exist_ok=True)
def c1(t):
    return f"{t[0]:+.1f} \\tiny[{t[1]:+.1f}, {t[2]:+.1f}]"


lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Model & $s_K$ (\fmtOpt) & \multicolumn{5}{c}{$\mathrm{ID}_K$, identity carried by keys (nats)} \\",
         r"\cmidrule(lr){3-7}",
         r" & & \fmtLetter & \fmtOpt & \fmtPost & \fmtNone & \fmtBefore \\", r"\midrule"]
for m in order:
    sh = nat[m]["share"]
    cells = " & ".join(c1(nat[m]["idK"][a]) for a in ["LETTER", "P1", "POST", "NONE", "BEFORE"])
    lines.append(f"{SHORT[m]} & {sh[0]:.2f} \\tiny[{sh[1]:.2f}, {sh[2]:.2f}] & {cells} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_natural.tex").write_text("\n".join(lines) + "\n")

# appendix: full decomposition per model x format
lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Format & $d_K$ & $d_V$ & $d_{KV}$ & interaction & $\mathrm{ID}_V$ \\", r"\midrule"]
for m in order:
    for i, a in enumerate(ARMS):
        d = nat[m]["dec"][a]
        lines.append(f"{SHORT[m] if i == 0 else ''} & {ARM_LABEL[a].replace(chr(10), ' ')} & {d['dK'][0]:+.1f} & {d['dV'][0]:+.1f} & "
                     f"{d['dKV'][0]:+.1f} & {d['int'][0]:+.1f} \\tiny[{d['int'][1]:+.1f}, {d['int'][2]:+.1f}] & {d['idV'][0]:+.1f} \\\\")
    lines.append(r"\midrule" if m != order[-1] else r"\bottomrule")
lines.append(r"\end{tabular}")
(TAB / "tab_decomposition.tex").write_text("\n".join(lines) + "\n")

# appendix: key share vs clamp onset (rules out a pure input-embedding account)
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Model & layers $L$ & $s_K$, onset $\ell_0{=}0$ & $s_K$, $\ell_0{\approx}0.06L$ & $s_K$, $\ell_0{\approx}0.3L$ \\", r"\midrule"]
for m in order:
    on = nat[m]["onset"]
    ks = sorted(on)
    cells = " & ".join(f"{on[k][0]:.2f} ($\\ell_0{{=}}{k}$)" for k in ks)
    lines.append(f"{SHORT[m]} & {nat[m]['L']} & {cells} \\\\")
    mac(f"onsetLate{MODEL_TAG[m]}", on[ks[-1]][0])
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_onset.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Model & Format & $\varphi$ (full remap) & $\psi$ (key addition) & $\rho$ (key removal) \\", r"\midrule"]
for label, out in frames.items():
    for i, a in enumerate(FRAME_ARMS):
        f3 = lambda q: f"{out[a][q][0]:.2f} [{out[a][q][1]:.2f}, {out[a][q][2]:.2f}]"
        name = label if i == 0 else ""
        lines.append(f"{name} & {ARM_LABEL[a].replace(chr(10), ' ')} & {f3('phi')} & {f3('psi')} & {f3('rho')} \\\\")
    lines.append(r"\midrule" if label != list(frames)[-1] else r"\bottomrule")
lines.append(r"\end{tabular}")
(TAB / "tab_frames.tex").write_text("\n".join(lines) + "\n")
print("wrote tables to", TAB)


# appendix: layer-window and KV-group localisation at 1.5B (choice-word rows only)
win = json.load(open(ROOT / "results/row_restricted_windows/Qwen2.5-1.5B-Instruct_direct.json"))
lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Rows/layers that see the swapped key & \fmtOpt & \fmtLetter \\", r"\midrule"]
keys = sorted(k for k in win[0]["m"] if k.startswith("win")) + sorted(k for k in win[0]["m"] if k.startswith("kvgroup"))
for k in keys:
    cells = []
    for arm in ("P1", "LETTER"):
        R = [r for r in win if r["arm"] == arm]
        fr = ratio_ci([r["m"][k] - r["m_B"] for r in R], [r["m"]["choice_words"] - r["m_B"] for r in R])
        cells.append(f"{fr[0]:+.2f} \\tiny[{fr[1]:+.2f}, {fr[2]:+.2f}]")
    lab = f"layers {int(k[3:])}--{int(k[3:]) + 3}" if k.startswith("win") else f"KV group {k[-1]} (all layers)"
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_windows.tex").write_text("\n".join(lines) + "\n")

with open(ROOT / "paper" / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by paper/make_figures.py from saved results. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and figures to {FIG}")
