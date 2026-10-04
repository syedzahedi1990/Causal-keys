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
ARM_LABEL = {"P1": "options-\nafter", "LETTER": "letters-\nafter", "POST": "sentence-\nafter",
             "NONE": "no-\nmention", "BEFORE": "list-\nbefore"}
ARM_TEX = {"P1": "\\fmtOpt{}", "LETTER": "\\fmtLetter{}", "POST": "\\fmtPost{}", "NONE": "\\fmtNone{}",
           "BEFORE": "\\fmtBefore{}", "AFTER": "\\fmtListA{}", "PRE": "\\fmtSentB{}"}
macros = {}


def mac(name, value, fmt="{:.2f}"):
    assert name.isalpha(), f"LaTeX macro names must be letters only: {name}"
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def cell(t, f="{:+.1f}", fci=None):
    """Estimate on one line, its 95% interval below in scriptsize."""
    fci = fci or f
    return (f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{f.format(t[0])}\\\\[-1pt]"
            f"{{\\scriptsize[{fci.format(t[1])}, {fci.format(t[2])}]}}\\end{{tabular}}")


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
        "idshare": {a: ratio_ci([v["idK"] for v in arms[a].values()],
                                [v["idK"] + v["idV"] for v in arms[a].values()]) for a in ARMS},
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
for a in ARMS:
    vals = [nat[m]["idshare"][a][0] for m in order]
    mac(f"idShare{ARM_TAG[a]}Min", min(vals)); mac(f"idShare{ARM_TAG[a]}Max", max(vals))
    big = [nat[m]["idshare"][a][0] for m in order if m != "Qwen2.5-1.5B-Instruct"]
    mac(f"idShare{ARM_TAG[a]}MinNoOnefive", min(big)); mac(f"idShare{ARM_TAG[a]}MaxNoOnefive", max(big))
for m in order:
    mac(f"idShareOpt{MODEL_TAG[m]}", nat[m]["idshare"]["P1"][0])
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
ax.set_ylabel("value identity carried by keys\n(relative to options-after)")
ax.grid(axis="y", color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "fig_formats.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Paper 1 frames: key-only vs value-only exchange (stage 3b)
frames = {}
FRAME_ARMS = ["LETTER", "P1", "POST", "NONE", "BEFORE"]
for model, label in (("mistral", "Mistral-Small-24B"), ("qwen", "Qwen2.5-72B")):
    res = json.load(open(ROOT / f"results/gpu_stage3b/paper1_frames_v/{model}.json"))["results"]
    rows = {a: s2.per_core(res, a) for a in FRAME_ARMS}
    out = {}
    mtag = model.capitalize()
    for a in FRAME_ARMS:
        R = rows[a]
        out[a] = {"phi": s2.ratio(R, lambda x: x["M"] - x["P"], lambda x: x["T"] - x["S"])[:3],
                  "psi": s2.ratio(R, lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"])[:3],
                  "rho": s2.ratio(R, lambda x: x["rem"] - x["M"], lambda x: x["P"] - x["M"])[:3],
                  "psiV": s2.ratio(R, lambda x: x["addv"] - x["P"], lambda x: x["M"] - x["P"])[:3],
                  "rhoV": s2.ratio(R, lambda x: x["remv"] - x["M"], lambda x: x["P"] - x["M"])[:3],
                  "mT": float(np.mean([x["M_T_rate"] for x in R.values()])), "n": len(R)}
        for q in ("phi", "psi", "rho", "psiV", "rhoV"):
            mac(f"{q}{ARM_TAG[a]}{mtag}", out[a][q][0])
            mac(f"{q}Lo{ARM_TAG[a]}{mtag}", out[a][q][1])
            mac(f"{q}Hi{ARM_TAG[a]}{mtag}", out[a][q][2])
        mac(f"mT{ARM_TAG[a]}{mtag}", out[a]["mT"])
    frames[label] = out

# exploratory cross-check: does the intervention's channel follow the natural identity read? (same pinned models)
NATM = {"Mistral-Small-24B": "Mistral-Small-24B-Instruct-2501", "Qwen2.5-72B": "Qwen2.5-72B-Instruct"}
cross = {}
for (model, label) in (("mistral", "Mistral-Small-24B"), ("qwen", "Qwen2.5-72B")):
    res = json.load(open(ROOT / f"results/gpu_stage3b/paper1_frames_v/{model}.json"))["results"]
    for a in FRAME_ARMS:
        R = s2.per_core(res, a)
        ak = [x["add"] - x["P"] for x in R.values()]; av = [x["addv"] - x["P"] for x in R.values()]
        cross[(label, a)] = (nat[NATM[label]]["idshare"][a], ratio_ci(ak, [k + v for k, v in zip(ak, av)]))
        mac(f"chShare{ARM_TAG[a]}{model.capitalize()}", cross[(label, a)][1][0])
        mac(f"natShare{ARM_TAG[a]}{model.capitalize()}", cross[(label, a)][0][0])
cx = np.array([v[0][0] for v in cross.values()]); cy = np.array([v[1][0] for v in cross.values()])
mac("crossR", float(np.corrcoef(cx, cy)[0, 1]))
mac("crossMaxGap", float(np.max(np.abs(cx - cy))))

# ---------------------------------------------------------------- stage 4 (preregistration F, paper v2): remap refit under NO-MENTION
import contextlib, io, re  # noqa: E402
import stage4_score as s4  # noqa: E402
REFAM = {"none/": "RefNone", "p1/": "RefOpt"}
with contextlib.redirect_stdout(io.StringIO()):
    _, rows4 = s4.load_frames(ROOT / "results/gpu_stage4/frames/mistral.json")
    est4 = s4.table(rows4, "")
    _, rows4n = s4.load_frames(ROOT / "results/gpu_stage4/frames_noprefill/mistral.json")
QK = {"phi": "phi", "psiK": "psi", "psiV": "psiV", "rhoK": "rho", "rhoV": "rhoV"}
for fam, tag in REFAM.items():
    for a in FRAME_ARMS:
        for k, q in QK.items():
            t = est4[fam, a, k]
            mac(f"{q}{ARM_TAG[a]}{tag}", t[0]); mac(f"{q}Lo{ARM_TAG[a]}{tag}", t[1]); mac(f"{q}Hi{ARM_TAG[a]}{tag}", t[2])
D4 = s4.did(rows4); D4n = s4.did(rows4n)
mac("refD", D4[0], "{:+.3f}"); mac("refDLo", D4[1], "{:+.3f}"); mac("refDHi", D4[2], "{:+.3f}")
mac("refDNoPre", D4n[0], "{:+.3f}"); mac("refDNoPreLo", D4n[1], "{:+.3f}"); mac("refDNoPreHi", D4n[2], "{:+.3f}")
sid4 = s4.s_id(ROOT / "results/gpu_stage2/format_factorial/Mistral-Small-24B-Instruct-2501_s0.json")
refshare = {}
for fam, tag in REFAM.items():
    sh = {a: est4[fam, a, "psiK"][0] / (est4[fam, a, "psiK"][0] + est4[fam, a, "psiV"][0]) for a in FRAME_ARMS}
    refshare[fam] = sh
    mac(f"refR{tag[3:]}", float(np.corrcoef([sh[a] for a in FRAME_ARMS], [sid4[a] for a in FRAME_ARMS])[0, 1]), "{:.2f}")
three = [("", "none/"), ("", "p1/"), ("none/", "p1/")]
mac("refMaxDiff", max(abs(est4[x, a, k][0] - est4[y, a, k][0]) for x, y in three for a in FRAME_ARMS for k in ("psiK", "psiV")))
mac("refMaxDiffOpt", max(abs(est4["", a, k][0] - est4["p1/", a, k][0]) for a in FRAME_ARMS for k in ("phi", "psiK", "psiV", "rhoK", "rhoV")))
cos = {}
for line in open(ROOT / "results/gpu_stage4/STAGE4_SCORE.txt"):
    m_ = re.match(r"\s+(\w+) m3_(\d+) vs (\w+) m3_(\d+): ([0-9.]+)$", line)
    if m_:
        cos.setdefault((m_[1], m_[3]), []).append(float(m_[5]))
    g_ = re.match(r"\s+fit_\w+ seed \d+: ([0-9.]+)$", line)
    if g_:
        cos.setdefault("G1", []).append(float(g_[1]))
for key, tag in ((("fit_none", "released"), "None"), (("fit_p1", "released"), "Opt"), (("released", "released"), "Seed"),
                 (("fit_none", "fit_p1"), "NoneOpt")):
    mac(f"refCos{tag}Min", min(cos[key])); mac(f"refCos{tag}Max", max(cos[key]))
mac("refGOneMin", min(cos["G1"]), "{:.3f}")
lines = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Remap & Format & $\varphi$ & $\psi_K$ & $\rho_K$ & $\psi_V$ & $\rho_V$ \\", r"\midrule"]
for fam, name in (("none/", "fit\\_none"), ("p1/", "fit\\_p1"), ("", "released")):
    for i, a in enumerate(FRAME_ARMS):
        f3 = lambda k: f"{est4[fam, a, k][0]:.2f} [{est4[fam, a, k][1]:.2f}, {est4[fam, a, k][2]:.2f}]"
        lines.append(f"{name if i == 0 else ''} & {ARM_TEX[a]} & {f3('phi')} & {f3('psiK')} & {f3('rhoK')} & {f3('psiV')} & {f3('rhoV')} \\\\")
    lines.append(r"\midrule" if fam != "" else r"\bottomrule")
lines.append(r"\end{tabular}")
(ROOT / "paper/tables").mkdir(exist_ok=True)
(ROOT / "paper/tables/tab_refit.tex").write_text("\n".join(lines) + "\n")

fig = plt.figure(figsize=(6.8, 2.35))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.2, 0.8], wspace=0.08)
axes = [fig.add_subplot(gs[0, 0])]
axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
w = 0.26
for ax, (label, out) in zip(axes, frames.items()):
    xs = np.arange(len(FRAME_ARMS))
    for k, (q, col, name) in enumerate((("phi", BLUE, "full remap: behavioural effect $\\varphi$"),
                                         ("psi", ORANGE, "keys only: $\\psi_K$"), ("psiV", AQUA, "values only: $\\psi_V$"))):
        v = [out[a][q][0] for a in FRAME_ARMS]
        e = np.array([[out[a][q][0] - out[a][q][1] for a in FRAME_ARMS], [out[a][q][2] - out[a][q][0] for a in FRAME_ARMS]])
        ax.bar(xs + (k - 1) * w, v, w * 0.9, color=col, label=name, yerr=e, error_kw=dict(lw=0.6, ecolor=INK2), zorder=3)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a].replace("-\n", "\n") for a in FRAME_ARMS], fontsize=5.8)
    ax.set_title(f"({'ab'[list(frames).index(label)]}) {label}", fontsize=7.5, color=INK)
    ax.set_ylim(-0.1, 1.15)
    ax.grid(axis="y", color=GRID, lw=0.6)
axes[0].set_ylabel("fraction of the remap's effect")
plt.setp(axes[1].get_yticklabels(), visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=3)

ax = fig.add_subplot(gs[0, 3])
ax.plot([-0.15, 1.05], [-0.15, 1.05], color=INK2, lw=0.6, ls=":", zorder=1)
ABBR = {"LETTER": "letters", "P1": "options", "POST": "sentence", "NONE": "none", "BEFORE": "before"}
for (label, a), (nx, cy_) in cross.items():
    mist = label.startswith("Mistral")
    ax.errorbar(nx[0], cy_[0], xerr=[[nx[0] - nx[1]], [nx[2] - nx[0]]], yerr=[[cy_[0] - cy_[1]], [cy_[2] - cy_[0]]],
                fmt="o" if mist else "s", ms=3.4, color=INK2, mfc="white" if mist else INK, mec=INK, mew=0.7,
                elinewidth=0.6, zorder=3, label=(label.split("-")[-1] + " released") if a == "LETTER" else None)
    if label.startswith("Qwen"):
        ax.annotate(ABBR[a], (nx[0], cy_[0]), textcoords="offset points",
                    xytext={"LETTER": (-16, -10), "P1": (5, -5), "POST": (5, -6), "NONE": (6, 0), "BEFORE": (5, -7)}[a],
                    fontsize=5.6, color=INK2)
for fam, mk, nm in (("none/", "^", "24B refit, no mention"), ("p1/", "v", "24B refit, options")):
    ax.scatter([sid4[a] for a in FRAME_ARMS], [refshare[fam][a] for a in FRAME_ARMS], marker=mk, s=13,
               facecolor="#9a9893" if fam == "none/" else "white", edgecolor=INK, linewidth=0.6, zorder=4, label=nm)
ax.set_xlim(-0.15, 1.05); ax.set_ylim(-0.15, 1.05)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("natural read: identity key share", fontsize=6.5)
ax.set_ylabel("remap: $\\psi_K/(\\psi_K+\\psi_V)$", fontsize=6.5, labelpad=2)
ax.set_title("(c) remap vs. natural read", fontsize=7.5, color=INK)
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=5.2, loc="upper left", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.8)
fig.savefig(FIG / "fig_frames.pdf")
plt.close(fig)

# ---------------------------------------------------------------- localisation at 1.5B (CPU): macros only
loc = json.load(open(ROOT / "results/row_restricted_windows/Qwen2.5-1.5B-Instruct_direct.json"))
groups = [("choice_words", "option words"), ("question", "question"), ("story_tail", "story after writing token"),
          ("rest_after_p", "instruction / answer tail")]
for k, (arm, col) in enumerate((("P1", BLUE), ("LETTER", ORANGE))):
    R = [r for r in loc if r["arm"] == arm]
    full = [r["m"]["all"] - r["m_B"] for r in R]
    vals = [ratio_ci([r["m"][g] - r["m_B"] for r in R], full)[0] for g, _ in groups]
    for g, v in zip(groups, vals):
        mac(f"loc{ARM_TAG[arm]}{g[0].replace('_', '').capitalize()}", v)

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

lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & \fmtLetter & \fmtOpt & \fmtPost & \fmtNone & \fmtBefore \\", r"\midrule"]
for m in order:
    lines.append(f"{SHORT[m]} & " + " & ".join(cell(nat[m]["idshare"][a], "{:+.2f}") for a in ARMS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_idshare.tex").write_text("\n".join(lines) + "\n")

# appendix: full decomposition per model x format
lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Format & $d_K$ & $d_V$ & $d_{KV}$ & interaction & $\mathrm{ID}_V$ \\", r"\midrule"]
for m in order:
    for i, a in enumerate(ARMS):
        d = nat[m]["dec"][a]
        lines.append(f"{SHORT[m] if i == 0 else ''} & {ARM_TEX[a]} & {d['dK'][0]:+.1f} & {d['dV'][0]:+.1f} & "
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

lines = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Model & Format & $\varphi$ & $\psi_K$ & $\rho_K$ & $\psi_V$ & $\rho_V$ \\", r"\midrule"]
for label, out in frames.items():
    for i, a in enumerate(FRAME_ARMS):
        f3 = lambda q: f"{out[a][q][0]:.2f} [{out[a][q][1]:.2f}, {out[a][q][2]:.2f}]"
        name = label if i == 0 else ""
        lines.append(f"{name} & {ARM_TEX[a]} & {f3('phi')} & {f3('psi')} & {f3('rho')} & {f3('psiV')} & {f3('rhoV')} \\\\")
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


# ---------------------------------------------------------------- stage 3: new tasks
TASK_MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen3-8B", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Task & Model & $s_K$ (\fmtOpt) & \multicolumn{5}{c}{$\mathrm{ID}_K$ (nats)} \\", r"\cmidrule(lr){4-8}",
         r" & & & \fmtLetter & \fmtOpt & \fmtPost & \fmtNone & \fmtBefore \\", r"\midrule"]
for task in ("paint", "schedule"):
    vals = []
    for i, m in enumerate(TASK_MODELS):
        res = json.load(open(ROOT / f"results/gpu_stage3/task_factorial/{task}_{m}_s0.json"))["results"]
        arms = {a: per_core(res, a) for a in ARMS}
        idk = {a: ci([v["idK"] for v in arms[a].values()]) for a in ARMS}
        sh = ratio_ci([v["d"]["K_S@0"] for v in arms["P1"].values()], [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in arms["P1"].values()])
        vals.append((idk, sh))
        cells = " & ".join(cell(idk[a]) for a in ["LETTER", "P1", "POST", "NONE", "BEFORE"])
        lines.append(f"{task if i == 0 else ''} & {SHORT[m]} & {cell(sh, '{:.2f}')} & {cells} \\\\")
    lines.append(r"\midrule" if task == "paint" else r"\bottomrule")
    tt = task.capitalize()
    mac(f"task{tt}OptMin", min(v[0]["P1"][0] for v in vals), "{:.1f}")
    mac(f"task{tt}OptMax", max(v[0]["P1"][0] for v in vals), "{:.1f}")
    mac(f"task{tt}BeforeMax", max(v[0]["BEFORE"][0] for v in vals), "{:+.2f}")
    mac(f"task{tt}ShareMin", min(v[1][0] for v in vals))
    mac(f"task{tt}ShareMax", max(v[1][0] for v in vals))
lines.append(r"\end{tabular}")
(TAB / "tab_tasks.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- stage 3b: instruction-matched 2x2
M4 = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
ARMS2 = ["AFTER", "POST", "NONE", "PRE", "BEFORE"]
lab2 = {"AFTER": "\\fmtListA", "POST": "\\fmtPost", "NONE": "\\fmtNone", "PRE": "\\fmtSentB", "BEFORE": "\\fmtBefore"}
ROWS2 = ["AFTER", "POST", "BEFORE", "PRE", "NONE"]
SH2 = {"Qwen2.5-7B-Instruct": "Qwen-7B", "Qwen2.5-14B-Instruct": "Qwen-14B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
       "OLMo-2-1124-7B-Instruct": "OLMo-2-7B"}
agg = {a: [] for a in ARMS2}
idk2 = {}
for m in M4:
    res = json.load(open(ROOT / f"results/gpu_stage3b/format_2x2/{m}_s0.json"))["results"]
    arms = {a: per_core(res, a) for a in ARMS2}
    idk2[m] = {a: ci([v["idK"] for v in arms[a].values()]) for a in ARMS2}
    for a in ARMS2:
        agg[a].append(idk2[m][a][0])
lines = [r"\begin{tabular}{@{}l" + "c" * len(M4) + "@{}}", r"\toprule",
         "Format & " + " & ".join(SH2[m] for m in M4) + r" \\", r"\midrule"]
for a in ROWS2:
    if a == "NONE":
        lines.append(r"\midrule")
    lines.append(f"{lab2[a]} & " + " & ".join(cell(idk2[m][a], "{:+.1f}", "{:.1f}") for m in M4) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_2x2.tex").write_text("\n".join(lines) + "\n")
mac("twoAfterMin", min(agg["AFTER"]), "{:.1f}"); mac("twoAfterMax", max(agg["AFTER"]), "{:.1f}")
mac("twoPostMin", min(agg["POST"]), "{:.1f}"); mac("twoPostMax", max(agg["POST"]), "{:.1f}")
mac("twoBeforeMax", max(agg["BEFORE"] + agg["PRE"]), "{:+.2f}")

# ---------------------------------------------------------------- stage 3b: role control
RM = M4 + ["Qwen2.5-72B-Instruct"]
lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Model & Format & role effect (nats) & $f_K$ & $f_V$ \\", r"\midrule"]
role_eff = []
fmax = 0.0
for m in RM:
    res = json.load(open(ROOT / f"results/gpu_stage3b/role_factorial/{m}_s0.json"))["results"]
    for i, arm in enumerate(("P1", "NONE", "LETTER")):
        R = [r for r in res if r["arm"] == arm and r["view"] == "direct"]
        full = [r["clean"]["R"]["m"] - r["m"]["ID"] for r in R]
        fk = ratio_ci([r["m"]["K_R"] - r["m"]["ID"] for r in R], full)
        fv = ratio_ci([r["m"]["V_R"] - r["m"]["ID"] for r in R], full)
        fmax = max(fmax, abs(fk[0]), abs(fv[0]))
        role_eff.append(np.mean(full))
        lines.append(f"{SHORT[m] if i == 0 else ''} & {ARM_TEX[arm]} & {np.mean(full):+.1f} & {cell(fk, '{:+.3f}')} & {cell(fv, '{:+.3f}')} \\\\")
    lines.append(r"\midrule" if m != RM[-1] else r"\bottomrule")
lines.append(r"\end{tabular}")
(TAB / "tab_role.tex").write_text("\n".join(lines) + "\n")
mac("roleFracMax", fmax)
mac("roleEffMin", min(role_eff), "{:.1f}"); mac("roleEffMax", max(role_eff), "{:.1f}")

# ---------------------------------------------------------------- stage 3b: environment check
for sub, m, tag in (("tf59_2gpu", "Qwen2.5-14B-Instruct", "EnvFourteen"), ("tf518_1gpu", "Qwen2.5-32B-Instruct", "EnvThirtytwo")):
    p1 = per_core(json.load(open(ROOT / f"results/gpu_stage3b/env_check/{sub}/{m}_s0.json"))["results"], "P1")
    sh = ratio_ci([v["d"]["K_S@0"] for v in p1.values()], [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in p1.values()])
    mac(f"share{tag}", sh[0])

# ---------------------------------------------------------------- localisation at scale (stage 3) + 1.5B (CPU)
LOCM = [("Qwen2.5-1.5B-Instruct", ROOT / "results/row_restricted_windows/Qwen2.5-1.5B-Instruct_direct.json"),
        ("Qwen2.5-7B-Instruct", ROOT / "results/gpu_stage3/row_restricted/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct_direct.json"),
        ("Qwen2.5-14B-Instruct", ROOT / "results/gpu_stage3/row_restricted/Qwen2.5-14B-Instruct/Qwen2.5-14B-Instruct_direct.json"),
        ("Mistral-7B-Instruct-v0.3", ROOT / "results/gpu_stage3/row_restricted/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_direct.json")]
GR = [("mention", "later-mention words"), ("question", "question"), ("story_tail", "story after writing token"), ("rest_after_p", "other later tokens")]
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Model & Format & " + " & ".join(g[1] for g in GR) + r" \\", r"\midrule"]
locv = {}
for m, f in LOCM:
    res = json.load(open(f))
    for arm in ("P1", "LETTER", "POST"):
        R = [r for r in res if r["arm"] == arm]
        if not R:
            continue
        full = [r["m"]["all"] - r["m_B"] for r in R]
        row = []
        for g, _ in GR:
            gk = ("remention_words" if arm == "POST" else "choice_words") if g == "mention" else g
            fr = ratio_ci([r["m"][gk] - r["m_B"] for r in R], full)
            row.append(fr)
            locv[(m, arm, g)] = fr
        lines.append(f"{SHORT[m]} & {ARM_TEX[arm]} & " + " & ".join(cell(x, "{:+.2f}") for x in row) + r" \\")
    lines.append(r"\midrule" if m != LOCM[-1][0] else r"\bottomrule")
lines.append(r"\end{tabular}")
(TAB / "tab_localisation.tex").write_text("\n".join(lines) + "\n")
big = [m for m, _ in LOCM[1:]]
mac("locScaleOptMin", min(locv[(m, "P1", "mention")][0] for m in big)); mac("locScaleOptMax", max(locv[(m, "P1", "mention")][0] for m in big))
mac("locScalePostMin", min(locv[(m, "POST", "mention")][0] for m in big)); mac("locScalePostMax", max(locv[(m, "POST", "mention")][0] for m in big))
mac("locScaleQuestionMax", max(abs(locv[(m, "P1", "question")][0]) for m in big))
mac("locScaleOptOtherMax", max(abs(locv[(m, "P1", g)][0]) for m in big for g in ("question", "story_tail", "rest_after_p")))
mac("locScalePostOtherMin", min(locv[(m, "POST", "rest_after_p")][0] for m in big))
mac("locScalePostOtherMax", max(locv[(m, "POST", "rest_after_p")][0] for m in big))
left = [1 - sum(locv[(m, "POST", g)][0] for g, _ in GR) for m in big]
mac("locScalePostLeftMin", min(left)); mac("locScalePostLeftMax", max(left))

fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), sharey=True)
for ax, arm, title in ((axes[0], "P1", "options-after"), (axes[1], "POST", "sentence-after")):
    for k, (m, col) in enumerate(zip(big, (BLUE, ORANGE, AQUA))):
        v = [locv[(m, arm, g)][0] for g, _ in GR]
        ys = np.arange(len(GR)) + (k - 1) * 0.26
        ax.barh(ys, v, 0.24, color=col, label=SHORT[m], zorder=3)
    ax.set_yticks(range(len(GR)))
    ax.set_yticklabels([g[1] for g in GR])
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-0.1, 1.1)
    ax.set_title(title, fontsize=7.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
axes[0].set_xlabel("fraction of the key effect recovered"); axes[1].set_xlabel("fraction of the key effect recovered")
axes[0].invert_yaxis()  # shared y: invert once
axes[1].legend(frameon=False, fontsize=6.3, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig_localisation.pdf")
plt.close(fig)

with open(ROOT / "paper" / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by paper/make_figures.py from saved results. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and figures to {FIG}")
