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
PURPLE, BROWN, LGREY = "#7d3fa8", "#a8661f", "#b9b7b2"
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


def tnum(s):
    """formatted number -> LaTeX: a math minus for negatives; a value that rounds to zero is printed unsigned."""
    if s.lstrip("+-").strip("0.") == "":
        return s.lstrip("+-")
    return f"$-${s[1:]}" if s.startswith("-") else s


def cell(t, f="{:+.1f}", fci=None):
    """Estimate on one line, its 95% interval below in scriptsize."""
    fci = fci or f
    return (f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{tnum(f.format(t[0]))}\\\\[-1pt]"
            f"{{\\scriptsize[{tnum(fci.format(t[1]))}, {tnum(fci.format(t[2]))}]}}\\end{{tabular}}")


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
# subspace overlaps: refit-vs-refit and floors computed from the saved bases; refit-vs-released (released bases are
# not in this repository) parsed from the preregistered score output, whose 3-decimal values are not on a .xx5 boundary
cos = {}
for line in open(ROOT / "results/gpu_stage4/STAGE4_SCORE.txt"):
    m_ = re.match(r"\s+(\w+) m3_(\d+) vs (\w+) m3_(\d+): ([0-9.]+)$", line)
    if m_:
        cos.setdefault((m_[1], m_[3]), []).append(float(m_[5]))
    g_ = re.match(r"\s+fit_\w+ seed \d+: ([0-9.]+)$", line)
    if g_:
        cos.setdefault("G1", []).append(float(g_[1]))
RB = {(f, o, sd): s4.load_basis(ROOT / f"results/gpu_stage4/fit_{f}/run/bases/{o}_ts{sd}.npz")
      for f in ("none", "p1") for o in ("m3", "pca") for sd in s4.SEEDS}
SP4 = ((101, 102), (101, 103), (102, 103))
cos["noneopt"] = [s4.msq_cos(RB["none", "m3", sd], RB["p1", "m3", sd]) for sd in s4.SEEDS]
cos["seed"] = cos[("released", "released")] + [s4.msq_cos(RB[f, "m3", a], RB[f, "m3", b]) for f in ("none", "p1") for a, b in SP4]
cos["floor"] = [s4.msq_cos(RB[f, "m3", sd], RB[f, "pca", sd]) for f in ("none", "p1") for sd in s4.SEEDS]
for key, tag in ((("fit_none", "released"), "None"), (("fit_p1", "released"), "Opt"), ("seed", "Seed"), ("noneopt", "NoneOpt"),
                 ("floor", "Floor")):
    mac(f"refCos{tag}Min", min(cos[key])); mac(f"refCos{tag}Max", max(cos[key]))
mac("refCosChance", 16 / RB["none", "m3", 101].shape[1], "{:.3f}")
mac("refPcaSame", min(s4.msq_cos(RB[f, "pca", 101], RB[f, "pca", sd]) for f in ("none", "p1") for sd in s4.SEEDS), "{:.3f}")
# post hoc: paired fit_none - fit_opt difference in psi_K per format (same cores, same resamples)
for a in FRAME_ARMS:
    ids = sorted(set(rows4["none/", a]) & set(rows4["p1/", a]))
    x = s4.stat(rows4["none/", a], ids, *s4.STATS["psiK"]); y = s4.stat(rows4["p1/", a], ids, *s4.STATS["psiK"])
    dd = x[3] - y[3]
    mac(f"refDiff{ARM_TAG[a]}", -(x[0] - y[0])); mac(f"refDiffLo{ARM_TAG[a]}", -np.percentile(dd, 97.5)); mac(f"refDiffHi{ARM_TAG[a]}", -np.percentile(dd, 2.5))
# identity-objective (f_star) fits, exploratory: (F - P) / (T - S)
for fam, tag in REFAM.items():
    mac(f"refFstarMax{tag[3:]}", max(abs(est4[fam, a, "phiF"][0]) for a in FRAME_ARMS))
    mac(f"refFstarLetter{tag[3:]}", est4[fam, "LETTER", "phiF"][0])
# training loss of the remap (m3) fits, first and last 100 updates
loss = {}
for f in ("none", "p1"):
    for line in open(ROOT / f"results/gpu_stage4/fit_{f}/run/updates.jsonl"):
        r = json.loads(line)
        if r["fit"].startswith("m3"):
            loss.setdefault(r["fit"] + f, []).append(r["loss"])
mac("refLossFirstMin", min(np.mean(v[:100]) for v in loss.values()), "{:.1f}"); mac("refLossFirstMax", max(np.mean(v[:100]) for v in loss.values()), "{:.1f}")
mac("refLossLastMin", min(np.mean(v[-100:]) for v in loss.values())); mac("refLossLastMax", max(np.mean(v[-100:]) for v in loss.values()))
mac("refGOneMin", min(cos["G1"]), "{:.3f}")
lines = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Remap & Format & $\varphi$ & $\psi_K$ & $\chi_K$ & $\psi_V$ & $\chi_V$ \\", r"\midrule"]
for fam, name in (("none/", "fit\\_none"), ("p1/", "fit\\_opt"), ("", "released")):
    for i, a in enumerate(FRAME_ARMS):
        f3 = lambda k: f"{est4[fam, a, k][0]:.2f} [{est4[fam, a, k][1]:.2f}, {est4[fam, a, k][2]:.2f}]"
        lines.append(f"{name if i == 0 else ''} & {ARM_TEX[a]} & {f3('phi')} & {f3('psiK')} & {f3('rhoK')} & {f3('psiV')} & {f3('rhoV')} \\\\")
    lines.append(r"\midrule" if fam != "" else r"\bottomrule")
lines.append(r"\end{tabular}")
(ROOT / "paper/tables").mkdir(exist_ok=True)
(ROOT / "paper/tables/tab_refit.tex").write_text("\n".join(lines) + "\n")

def draw_frames(pk=None):
    """fig_frames; pk adds Prakash et al.'s cells to panel (c) (v3)."""
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
    axes[0].set_ylabel("fraction ($\\varphi$: of natural shift;\n$\\psi$: of M$-$P)", fontsize=6.5, labelpad=1)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=3)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot([-0.15, 1.05], [-0.15, 1.05], color=INK2, lw=0.6, ls=":", zorder=1)
    ABBR = {"LETTER": "letters", "P1": "options", "POST": "sentence", "NONE": "none", "BEFORE": "before"}
    for (label, a), (nx, cy_) in cross.items():
        mist = label.startswith("Mistral")
        ax.errorbar(nx[0], cy_[0], xerr=[[nx[0] - nx[1]], [nx[2] - nx[0]]], yerr=[[cy_[0] - cy_[1]], [cy_[2] - cy_[0]]],
                    fmt="o" if mist else "s", ms=5.2 if mist else 3.4, color=INK2, mfc="white" if mist else INK, mec=INK, mew=0.7,
                    elinewidth=0.6, zorder=3, label=(label.split("-")[-1] + " released") if a == "LETTER" else None)
        if label.startswith("Qwen"):
            ax.annotate(ABBR[a], (nx[0], cy_[0]), textcoords="offset points",
                        xytext={"LETTER": (-16, -10), "P1": (5, -5), "POST": (5, -6), "NONE": (6, 0), "BEFORE": (5, -7)}[a],
                        fontsize=5.6, color=INK2)
    for fam, mk, nm in (("none/", "^", "24B refit, no mention"), ("p1/", "v", "24B refit, options")):
        ax.scatter([sid4[a] for a in FRAME_ARMS], [refshare[fam][a] for a in FRAME_ARMS], marker=mk, s=9,
                   facecolor="#9a9893" if fam == "none/" else "white", edgecolor=INK, linewidth=0.6, zorder=4, label=nm)
    ax.set_xlim(-0.15, 1.05); ax.set_ylim(-0.15, 1.05)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("natural read: identity key share", fontsize=6.5)
    ax.set_ylabel("edit's key share $\\psi_K/(\\psi_K+\\psi_V)$ or $\\kappa$", fontsize=6.5, labelpad=2)
    ax.set_title("(c) edit vs. natural read", fontsize=7.5, color=INK)
    ax.grid(color=GRID, lw=0.6)
    for lab, mk, col, pts in (pk or []):
        for (sx, ky, f) in pts:
            ax.errorbar(sx[0], ky[0], xerr=[[sx[0] - sx[1]], [sx[2] - sx[0]]], yerr=[[ky[0] - ky[1]], [ky[2] - ky[0]]], fmt=mk, ms=3.4,
                        color=col, mfc=col if mk == "D" else "white", mec=col, mew=0.8, elinewidth=0.6, zorder=5,
                        label=lab if f == "NO-MENTION" else None)
        xs_, ys_ = [p[0][0] for p in pts], [p[1][0] for p in pts]
        ax.annotate("H7" if mk == "D" else "H11", (max(xs_), max(ys_)), textcoords="offset points", xytext=(4, -2) if mk == "D" else (-14, 5),
                    fontsize=6, color=col, weight="bold")
    if not pk:
        ax.legend(frameon=False, fontsize=5.2, loc="upper left", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
    else:
        h_, l_ = ax.get_legend_handles_labels()
        fig.legend(h_, l_, frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.66, 1.02), ncol=2, handletextpad=0.1,
                   columnspacing=0.6, labelspacing=0.2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.2, top=0.8)
    fig.savefig(FIG / "fig_frames.pdf")
    plt.close(fig)


draw_frames()

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
         r"Model & Format & $\varphi$ & $\psi_K$ & $\chi_K$ & $\psi_V$ & $\chi_V$ \\", r"\midrule"]
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

# ================================================================ paper v3: stages 5 (preregistration G) and 6 (H)
# Every new macro is registered with a description and its source (score-file line or raw file). Values that the
# scorers print are taken verbatim from STAGE5_SCORE.txt / STAGE6_SCORE.txt at their printed precision (a leading '+'
# is dropped; negative values keep '-', so use them in math mode). Values the scorers do not print are recomputed from
# the raw results and registered as such ("recomputed"); the paper must label them exploratory or post hoc.
# A check at the end re-derives the decisive stage-6 values from the raw JSON through the committed scorer classes and
# asserts that every score-file macro still matches its line. v2 macros are never overwritten (nm() asserts this).
import os  # noqa: E402

V2_MACROS = set(macros)
REG = []          # (group, name, value, description, source)
SRCLINE = {}      # name -> (file tag, line index, printed string) for score-file macros
S5F = ROOT / "results/gpu_stage5/STAGE5_SCORE.txt"
S6F = ROOT / "results/gpu_stage6/STAGE6_SCORE.txt"
S5 = S5F.read_text().splitlines()
S6 = S6F.read_text().splitlines()
SCORE = {"S5": S5, "S6": S6}
SNAME = {"S5": "results/gpu_stage5/STAGE5_SCORE.txt", "S6": "results/gpu_stage6/STAGE6_SCORE.txt"}
NUM = r"([+-]?(?:\d+\.\d+(?:e[+-]\d+)?|nan))"
CIR = NUM + r" \[" + NUM + r"," + NUM + r"\]"
WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
        10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen",
        17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty",
        60: "Sixty", 70: "Seventy", 80: "Eighty", 90: "Ninety"}


def word(n):
    n = int(n)
    if n in WORD:
        return WORD[n]
    if n < 100:
        return WORD[n // 10 * 10] + WORD[n % 10].lower()
    if n == 100:
        return "Hundred"
    return "Hundred" + ("and" if n % 100 else "") + word(n - 100).lower()


def sv(s):
    """score-file string -> macro string: drop a leading '+', keep everything else as printed."""
    s = s.strip()
    return s[1:] if s.startswith("+") else s


def nm(name, value, desc, src, group, fmt="{:.2f}"):
    """register a new v3 macro (never overwrites a v2 macro or another v3 macro)."""
    assert name not in macros, f"macro {name} already defined"
    mac(name, value, fmt)
    REG.append((group, name, macros[name], desc, src))


def smac(name, tag, idx, s, desc, group):
    """macro from a printed score-file string s on line idx of score file tag."""
    assert s in SCORE[tag][idx], (name, s, SCORE[tag][idx][:120])
    nm(name, sv(s), desc, f"{SNAME[tag]}:{idx + 1}", group)
    SRCLINE[name] = (tag, idx, s)


def smac3(name, tag, idx, t, desc, group):
    """estimate and its 95 % interval (three printed strings) -> name, nameLo, nameHi."""
    smac(name, tag, idx, t[0], desc, group)
    smac(name + "Lo", tag, idx, t[1], desc + " (lower 95 % bound)", group)
    smac(name + "Hi", tag, idx, t[2], desc + " (upper 95 % bound)", group)


def rmac(name, value, desc, src, group, fmt="{:.3f}"):
    """recomputed from raw results (not printed by a scorer). Stage 8 (groups S8*): a value that rounds to zero is printed
    unsigned ('0.000', never '-0.000'); the earlier groups keep their printed strings unchanged."""
    if group.startswith("S8") and not isinstance(value, str) and fmt.format(value).lstrip("-").strip("0.") == "":
        value = abs(value)
    nm(name, value, desc + " [recomputed; label exploratory/post hoc]", src, group, fmt)


def amac(name, value, desc, src, group, fmt="{:.2f}"):
    """arithmetic on printed score-file values."""
    nm(name, value, desc + " [arithmetic on printed values]", src, group, fmt)


def find(L, *subs, start=0, end=None, many=False):
    hits = [i for i in range(start, len(L) if end is None else end) if all(s in L[i] for s in subs)]
    if many:
        return hits
    assert len(hits) == 1, (subs, start, end, hits[:6])
    return hits[0]


def grab(L, i, pat):
    m_ = re.search(pat, L[i])
    assert m_, (pat, L[i][:200])
    return m_.groups()


def sec(L, title):
    return find(L, title)


def fl(s):
    return float(s)


MT = {"Qwen2.5-7B-Instruct": "QwenSeven", "Qwen2.5-14B-Instruct": "QwenFourteen", "Qwen2.5-1.5B-Instruct": "QwenOnefive",
      "Qwen2.5-3B-Instruct": "QwenThree", "Mistral-7B-Instruct-v0.3": "MistralSeven", "OLMo-2-1124-7B-Instruct": "OlmoSeven",
      "gpt2": "GptTwo", "gpt2-xl": "GptTwoXl", "Qwen2.5-7B": "QwenSevenBase"}
SH5 = dict(SHORT, **{"gpt2": "GPT-2", "gpt2-xl": "GPT-2 XL", "Qwen2.5-7B": "Qwen2.5-7B base"})

# ---------------------------------------------------------------- stage 6, part (a): reader heads and the second hop
G6A = "c.4 sec:heads / tab:heads (stage 6 part a, preregistration H1-H6)"
A6 = sec(S6, "######## PART (a)")
B6 = sec(S6, "######## PART (b)")
HM = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"]
HARM = {"P1": "Opt", "POST": "Post"}
h6 = {m: {} for m in HM}   # parsed floats for tables and figures
for m in HM:
    t = MT[m]
    i = find(S6, f"   {m}: git_commit", start=A6, end=B6)
    nh, ks = grab(S6, i + 1, r"n_heads=(\d+) .*?k\*=(\d+)")
    smac(f"Hkstar{t}", "S6", i + 1, ks, f"k* = round(0.05 x n_heads) at {SHORT[m]}", G6A)
    smac(f"HnHeads{t}", "S6", i + 1, nh, f"number of attention heads at {SHORT[m]}", G6A)
    smac(f"HnRank{t}", "S6", i + 2, grab(S6, i + 2, r"n_rank=(\d+)")[0], "ranking stories R per format", G6A)
    smac(f"HnEval{t}", "S6", i + 2, grab(S6, i + 2, r"n_eval=(\d+)")[0], "evaluation stories E per format", G6A)
    h6[m]["kstar"], h6[m]["nheads"] = int(ks), int(nh)
    for arm, at in HARM.items():
        i = find(S6, "Gate a2", m, f" {arm} ", start=A6, end=B6)
        nf, af, bd, hp, hpm, raw, verd = grab(S6, i, r"\|none - clean\| " + NUM + r", \|all_T - full\| " + NUM + r" <= " + NUM
                                         + r"; hop \|exact - K_S\| max (\S+) \(mean (\S+)\).*\|all_G - full\| " + NUM + r"\] -> (.+)$")
        for nmx, v, d in (("NoneClean", nf, "|none - clean|"), ("AllFull", af, "|all_T - full|"), ("Bound", bd, "bound max(0.5, 0.02 mean d_full)"),
                          ("Hop", hp, "hop exactness max |exact - K_S|"), ("Raw", raw, "raw floor |all_G - full|")):
            smac(f"Hatwo{nmx}{at}{t}", "S6", i, v, f"Gate a2 {d}, {at}, {SHORT[m]}", G6A)
        h6[m][f"a2{arm}"] = verd
    i = find(S6, "Gate a3", m, start=A6, end=B6)
    d1, d2, g = (grab(S6, i, p) for p in (r"d_full OPTIONS-AFTER " + CIR, r"SENTENCE-AFTER " + CIR, r"d_G/d_full OPTIONS-AFTER " + CIR))
    smac3(f"HathreeDfullOpt{t}", "S6", i, d1, f"Gate a3 mean d_full (nats), OPTIONS-AFTER, {SHORT[m]}", G6A)
    smac3(f"HathreeDfullPost{t}", "S6", i, d2, f"Gate a3 mean d_full (nats), SENTENCE-AFTER, {SHORT[m]}", G6A)
    smac3(f"HathreeRatio{t}", "S6", i, g, f"Gate a3 d_G/d_full, OPTIONS-AFTER, {SHORT[m]}", G6A)
    h6[m]["a3"] = [d1, d2, g]
    # H1
    i = find(S6, "  H1   " + m, start=A6, end=B6)
    k_, r1, lo, hi, k80a, k80f, k80d = grab(S6, i, r"R\((\d+)\) " + CIR + r" .*k80 a3 (\S+), fplus (\S+), dminus (\S+) ->")
    smac3(f"HoneR{t}", "S6", i, (r1, lo, hi), f"H1 R(k*): option-row read recovered by the top-k* heads by a3, {SHORT[m]}", G6A)
    smac(f"HoneKeighty{t}", "S6", i, k80a, f"H1 k80 by a3 (smallest k with R(k) >= 0.8), {SHORT[m]}", G6A)
    smac(f"HoneKeightyFplus{t}", "S6", i, k80f, f"k80 by the single-head f+ ranking ('None' = never), {SHORT[m]}", G6A)
    smac(f"HoneKeightyDminus{t}", "S6", i, k80d, f"k80 by the single-head d- ranking ('None' = never), {SHORT[m]}", G6A)
    h6[m]["H1"] = (r1, lo, hi)
    h6[m]["k80"] = (k80a, k80f, k80d)
    # H2
    i = find(S6, "  H2   " + m, start=A6, end=B6)
    ko = grab(S6, i, r"KO\(\d+\) " + CIR)
    rr = grab(S6, i, r"R_rand " + CIR)
    kr = grab(S6, i, r"KO_rand " + CIR)
    smac3(f"HtwoKO{t}", "S6", i, ko, f"H2 KO(k*): read removed by blinding the top-k* heads, {SHORT[m]}", G6A)
    smac3(f"HtwoRrand{t}", "S6", i, rr, f"H2 R of random sets of k* heads (mean over 3 draws), {SHORT[m]}", G6A)
    smac3(f"HtwoKOrand{t}", "S6", i, kr, f"H2 KO of random sets of k* heads (mean over 3 draws), {SHORT[m]}", G6A)
    h6[m]["H2"] = (ko, rr, kr)
    # H3
    i = find(S6, "  H3   " + m, start=A6, end=B6)
    rho = grab(S6, i, r"rho_K " + CIR)
    mass = grab(S6, i, r"mass >= 0.9 in (\S+) of stories")[0]
    rc = grab(S6, i, r"rand0 " + NUM + ", rand1 " + NUM + ", rand2 " + NUM + ", active " + NUM)
    dv = grab(S6, i, r"dV " + CIR)
    floor = grab(S6, i, r"floor (\S+):")[0]
    base = grab(S6, i, r"base-argmax rate (\S+)")[0]
    smac3(f"HthreeRho{t}", "S6", i, rho, f"H3(a) rho_K: ID_K under mean-ablation of the top-k* heads at the option rows / ID_K, {SHORT[m]}", G6A)
    smac(f"HthreeMass{t}", "S6", i, mass, f"H3(b) fraction of stories with clean-B candidate mass >= 0.9 under ablation, {SHORT[m]}", G6A)
    for j, w in enumerate(("RandZero", "RandOne", "RandTwo", "Active")):
        smac(f"HthreeRho{w}{t}", "S6", i, rc[j], f"H3(c) rho_K under ablation of a control set ({w}), {SHORT[m]}", G6A)
    smac3(f"HthreeDV{t}", "S6", i, dv, f"H3(d) dV: rise of ID_V (nats) under ablation of the top-k* heads, {SHORT[m]}", G6A)
    smac(f"HthreeFloor{t}", "S6", i, floor, f"H3(d) preregistered dV floor (nats), {SHORT[m]}", G6A)
    smac(f"HthreeBase{t}", "S6", i, base, f"H3 base-argmax rate (answer kept) under ablation, {SHORT[m]}", G6A)
    h6[m]["H3"] = dict(rho=rho, mass=mass, rc=rc, dv=dv, floor=floor, base=base)
    # H4
    i = find(S6, "  H4   " + m, start=A6, end=B6)
    kc = grab(S6, i, r"C = top-(\d+) by a3")[0]
    D_, I_, T_ = (grab(S6, i, p + CIR) for p in (r"median D ", r"median I ", r"median T_dup "))
    ov, P_ = grab(S6, i, r"top10\(D\)\| = (\d+) \(>= 3: \w+, P = (\S+)\)")
    rd, rt = grab(S6, i, r"Spearman\(a3, D\) " + NUM + r", Spearman\(a3, T_dup\) " + NUM)
    smac(f"HfourKC{t}", "S6", i, kc, f"H4 size of C = min(k80, k*), {SHORT[m]}", G6A)
    smac3(f"HfourD{t}", "S6", i, D_, f"H4(i) median duplicate-token score D over C, {SHORT[m]}", G6A)
    smac3(f"HfourI{t}", "S6", i, I_, f"H4(i) median induction score I over C, {SHORT[m]}", G6A)
    smac3(f"HfourT{t}", "S6", i, T_, f"H4(ii) median in-task duplicate score T_dup over C (E stories), {SHORT[m]}", G6A)
    smac(f"HfourOverlap{t}", "S6", i, ov, f"H4 secondary |top-10(a3) & top-10(D)|, {SHORT[m]}", G6A)
    mant, ex_ = P_.split("e")
    nm(f"HfourP{t}", f"{mant}\\times 10^{{{int(ex_)}}}", f"H4 secondary hypergeometric P(overlap >= observed) (math mode; printed {P_}), {SHORT[m]}",
       f"{SNAME['S6']}:{i + 1}", G6A)
    smac(f"HfourRhoD{t}", "S6", i, rd, f"H4 Spearman(a3, D) over all heads, {SHORT[m]}", G6A)
    smac(f"HfourRhoT{t}", "S6", i, rt, f"H4 Spearman(a3, T_dup) over all heads, {SHORT[m]}", G6A)
    h6[m]["H4"] = dict(kc=kc, D=D_, I=I_, T=T_, ov=ov, P=P_, rd=rd, rt=rt)
    # H5, H5s
    i = find(S6, "  H5   " + m, start=A6, end=B6)
    ra, df_, ro, rl, kvd = (grab(S6, i, p + CIR) for p in (r"r_ans\(KV\) ", r"r_ans - r_other ", r"\[r_other ", r"r_all\(KV\) ", r"r_ans\(K\) - r_ans\(V\) "))
    rk, rv = grab(S6, i, r"\[r_ans\(K\) " + NUM + r", r_ans\(V\) " + NUM + r"\]")
    smac3(f"HfiveAns{t}", "S6", i, ra, f"H5 r_ans(KV): fraction of the key-clamp effect removed when only the answer row sees the option rows' base K/V, {SHORT[m]}", G6A)
    smac3(f"HfiveDiff{t}", "S6", i, df_, f"H5 paired r_ans - r_other, {SHORT[m]}", G6A)
    smac3(f"HfiveOther{t}", "S6", i, ro, f"H5 r_other (every other row restored), {SHORT[m]}", G6A)
    smac3(f"HfiveAll{t}", "S6", i, rl, f"H5 r_all(KV) (all rows restored), {SHORT[m]}", G6A)
    smac3(f"HfiveKminusV{t}", "S6", i, kvd, f"H5 secondary (reported, not scored) r_ans(K) - r_ans(V), {SHORT[m]}", G6A)
    smac(f"HfiveAnsK{t}", "S6", i, rk, f"H5 secondary (not scored) r_ans(K): answer row sees base keys only, {SHORT[m]}", G6A)
    smac(f"HfiveAnsV{t}", "S6", i, rv, f"H5 secondary (not scored) r_ans(V): answer row sees base values only, {SHORT[m]}", G6A)
    h6[m]["H5"] = dict(ans=ra, diff=df_, other=ro, all=rl, kv=kvd, K=rk, V=rv)
    i = find(S6, "  H5s  " + m, start=A6, end=B6)
    ra, rl, ro = (grab(S6, i, p + CIR) for p in (r"r_ans\(KV\) ", r"r_all\(KV\) ", r"r_other "))
    smac3(f"HfivesAns{t}", "S6", i, ra, f"H5 SENTENCE-AFTER line (reported; not evaluable overall) r_ans(KV), {SHORT[m]}", G6A)
    smac3(f"HfivesAll{t}", "S6", i, rl, f"H5 SENTENCE-AFTER line r_all(KV), {SHORT[m]}", G6A)
    smac3(f"HfivesOther{t}", "S6", i, ro, f"H5 SENTENCE-AFTER line r_other, {SHORT[m]}", G6A)
    h6[m]["H5s"] = dict(ans=ra, all=rl, other=ro, verd=S6[i].split("-> ")[-1])
    i = find(S6, "  H6   " + m, start=A6, end=B6)
    ov, P_ = grab(S6, i, r"\| = (\d+) >= 10 \(null P = (\S+)\)")
    smac(f"HsixOverlap{t}", "S6", i, ov, f"H6 |top-20(a3, OPTIONS-AFTER) & top-20(a3, SENTENCE-AFTER)|, {SHORT[m]}", G6A)
    mant, ex_ = P_.split("e")
    nm(f"HsixP{t}", f"{mant}\\times 10^{{{int(ex_)}}}", f"H6 null P(overlap >= observed) (math mode; printed {P_}), {SHORT[m]}", f"{SNAME['S6']}:{i + 1}", G6A)
    h6[m]["H6"] = (ov, P_, S6[i].split("-> ")[-1])
    for H in ("H1", "H2", "H3", "H4", "H5", "H5s", "H6"):
        i = find(S6, f"  {H:4s} {m}", start=A6, end=B6)
        h6[m][f"v{H}"] = S6[i].split("-> ")[-1]

# exploratory lines per model and format
GX6 = "c.4 sec:heads exploratory (stage 6 part a; E-c, single-head rankings, ablation conditions, hop rows)"
CURV = {}
COND = {"none": "None", "top_kstar_mean": "Topk", "top_10_mean": "Topten", "top_20_mean": "Toptwenty", "top_kstar_zero": "Topkzero",
        "rand0_kstar_mean": "RandZero", "rand1_kstar_mean": "RandOne", "rand2_kstar_mean": "RandTwo", "active_kstar_mean": "Active",
        "next_kstar_mean": "Next"}
RANKW = {"a3": "Athree", "fplus": "Fplus", "dminus": "Dminus", "rand0": "RandZero", "rand1": "RandOne", "rand2": "RandTwo"}
ABL = {}
for m in HM:
    t = MT[m]
    for arm, at in HARM.items():
        i0 = find(S6, f"   -- {m} {arm}", start=A6, end=B6)
        i1 = next(j for j in range(i0 + 1, B6 + 1) if S6[j].startswith("   -- ") or S6[j].startswith("######"))
        for j in range(i0 + 1, i1):
            s = S6[j]
            mm = re.match(r"\s+(a3|fplus|dminus|rand\d)\s+(R|KO)\s+(.*)$", s)
            if mm:
                pts = [x.split(":") for x in mm[3].split()]
                CURV[m, arm, mm[1], mm[2]] = {int(k): fl(v) for k, v in pts}
                if arm == "P1":
                    for k, v in pts:
                        smac(f"Hcurve{mm[2]}{RANKW[mm[1]]}{word(k)}{at}{t}", "S6", j, f"{k}:{v}".split(":")[1],
                             f"{mm[2]}({k}) for ranking {mm[1]}, {at}, {SHORT[m]} (2 decimals)", GX6)
            elif "layer profile (fraction of d_G)" in s:
                prof = [fl(x) for x in s.split(":")[1].split()]
                CURV[m, arm, "layer"] = prof
                k = int(np.argmax(prof))
                smac(f"HlayerMax{at}{t}", "S6", j, s.split(":")[1].split()[k], f"largest single-layer share of d_G (all heads of one layer read K_S), {at}, {SHORT[m]}", GX6)
                nm(f"HlayerArg{at}{t}", str(k), f"layer index of that maximum, {at}, {SHORT[m]}", f"{SNAME['S6']}:{j + 1}", GX6)
            elif "split-half a3" in s:
                smac(f"Hsplit{at}{t}", "S6", j, grab(S6, j, r"Spearman " + NUM)[0], f"split-half Spearman of a3 (R vs E), {at}, {SHORT[m]}", GX6)
            elif "ablation: ID_K(none)" in s:
                a, b = grab(S6, j, r"ID_K\(none\) " + CIR), grab(S6, j, r"ID_V\(none\) " + CIR)
                smac3(f"HablIDKnone{at}{t}", "S6", j, a, f"ID_K without ablation (nats, E stories), {at}, {SHORT[m]}", GX6)
                smac3(f"HablIDVnone{at}{t}", "S6", j, b, f"ID_V without ablation (nats, E stories), {at}, {SHORT[m]}", GX6)
                ABL[m, arm, "none_idk"], ABL[m, arm, "none_idv"] = a, b
            elif re.match(r"\s{8}(\w+)\s+rho_K", s):
                c = s.split()[0]
                r_, d_ = grab(S6, j, r"rho_K " + CIR), grab(S6, j, r"dV " + CIR)
                ms, ba = grab(S6, j, r"mass>=0.9 (\S+)  base-argmax (\S+)")
                ABL[m, arm, c] = (r_, d_, ms, ba)
                w = COND[c]
                smac3(f"Habl{w}Rho{at}{t}", "S6", j, r_, f"rho_K under ablation condition {c}, {at}, {SHORT[m]}", GX6)
                smac3(f"Habl{w}DV{at}{t}", "S6", j, d_, f"dV (nats) under ablation condition {c}, {at}, {SHORT[m]}", GX6)
                smac(f"Habl{w}Mass{at}{t}", "S6", j, ms, f"fraction of stories with clean-B mass >= 0.9, condition {c}, {at}, {SHORT[m]}", GX6)
                smac(f"Habl{w}Base{at}{t}", "S6", j, ba, f"base-argmax rate, condition {c}, {at}, {SHORT[m]}", GX6)
            elif s.strip().startswith("hop: r_ans_K"):
                vals = dict(re.findall(r"r_(\w+) " + NUM, s))
                for k_, w in (("ans_K", "AnsK"), ("ans_V", "AnsV"), ("ans_KV", "AnsKV"), ("all_KV", "AllKV"), ("other_KV", "OtherKV"), ("exact", "Exact")):
                    smac(f"Hhop{w}{at}{t}", "S6", j, vals[k_], f"hop row r_{k_}, {at}, {SHORT[m]}", GX6)
                smac(f"HhopSum{at}{t}", "S6", j, grab(S6, j, r"r_ans\(KV\) \+ r_other " + NUM)[0], f"r_ans(KV) + r_other (additivity check), {at}, {SHORT[m]}", GX6)
                CURV[m, arm, "hop"] = {k: fl(v) for k, v in vals.items()}
            elif re.match(r"\s+rows (G|rowS|rowB|sep)\s*:", s):
                g = re.match(r"\s+rows (G|rowS|rowB|sep)\s*:", s)[1]
                for k_, v in re.findall(r"r_(\w+) " + NUM, s):
                    nmx = "".join(x[0].upper() + x[1:] for x in k_.split("_"))
                    smac(f"Hhoprow{g.capitalize()}{nmx}{at}{t}", "S6", j, v, f"hop row restriction '{g}' r_{k_}, {at}, {SHORT[m]}", GX6)
            elif "over C (top-" in s:
                a, b, c = grab(S6, j, r"median T_dup " + NUM + r", T_ctrl " + NUM + r", previous-token P " + NUM)
                smac(f"HcTdup{at}{t}", "S6", j, a, f"median T_dup over C (P1 ranking), on {at} stories, {SHORT[m]}", GX6)
                smac(f"HcTctrl{at}{t}", "S6", j, b, f"median control-word T_ctrl over C, on {at} stories, {SHORT[m]}", GX6)
                smac(f"HcPrev{at}{t}", "S6", j, c, f"median previous-token score over C, {SHORT[m]}", GX6)
            elif "over the top-" in s:
                a, b, c = grab(S6, j, r"median D " + NUM + r", I " + NUM + r", T_dup " + NUM)
                smac(f"HcDown{at}{t}", "S6", j, a, f"median D over the top-k_C heads by this format's own a3, {at}, {SHORT[m]}", GX6)
                smac(f"HcIown{at}{t}", "S6", j, b, f"median I over the top-k_C heads by this format's own a3, {at}, {SHORT[m]}", GX6)
                smac(f"HcTown{at}{t}", "S6", j, c, f"median T_dup over the top-k_C heads by this format's own a3, {at}, {SHORT[m]}", GX6)
    # arithmetic (E-d): ID_V under ablation = ID_V(none) + dV
    amac(f"HablIDVafter{t}", fl(ABL[m, "P1", "none_idv"][0]) + fl(h6[m]["H3"]["dv"][0]),
         f"ID_V under ablation of the top-k* = ID_V(none) + dV (nats), OPTIONS-AFTER, {SHORT[m]}", f"{SNAME['S6']} (H3 line + ablation line)", GX6)

# recomputed part (a) quantities (not printed by the scorer), through the committed scorer's own Model class
sys.path.insert(0, str(ROOT / "analysis" / "stage6_parts"))
import heads as H6S  # noqa: E402
import prakash as PKS  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HSRC = {m: f"results/gpu_stage6/heads/{m}.json" for m in HM}
HMOD = {m: H6S.Model(m, json.load(open(ROOT / HSRC[m]))) for m in HM}
for m in HM:
    M, t = HMOD[m], MT[m]
    ik = M.ik
    rr = [M.R("P1", f"rand{r}", ik)[0] for r in range(M.n_rand)]
    kr = [M.KO("P1", f"rand{r}", ik)[0] for r in range(M.n_rand)]
    rmac(f"HtwoRrandMax{t}", max(rr), f"largest single random draw R(k*), OPTIONS-AFTER, {SHORT[m]}", HSRC[m], G6A)
    rmac(f"HtwoKOrandMax{t}", max(kr), f"largest single random draw KO(k*), OPTIONS-AFTER, {SHORT[m]}", HSRC[m], G6A)
    kc, C = H6S.causal_set(M)
    Dg = M.dup["D_seq"].mean(0)
    rmac(f"HfourDkstar{t}", H6S.med_over(M, "D", M.rank("P1")[:M.kstar])[0], f"median D over the whole top-k* set, {SHORT[m]}", HSRC[m], G6A)
    rmac(f"HfourDwhole{t}", float(np.median(Dg)), f"median D over all heads of the model, {SHORT[m]}", HSRC[m], G6A)
    nm(f"HfourNDge{t}", str(int(sum(Dg[c] >= 0.2 for c in C))), f"number of heads in C with D >= 0.2 [recomputed]", HSRC[m], G6A)
    nm(f"HfourNDgeWhole{t}", str(int((Dg >= 0.2).sum())), f"number of heads in the whole model with D >= 0.2 [recomputed]", HSRC[m], G6A)
    rmac(f"HfourPrevMax{t}", max(M.dup["P"][c] for c in C), f"largest previous-token score over C, {SHORT[m]}", HSRC[m], G6A)
    R_ = M.J["arms"]["P1"]["eval"]
    S_ = np.array([np.array(r["tdup"])[[c[0] for c in C], [c[1] for c in C]] / max(r["n_dup"], 1) for r in R_ if r["n_dup"]], float)
    rmac(f"HfourTstory{t}", float(np.median(S_.mean(0))), f"H4(ii) T_dup averaged per story (not pooled), median over C, {SHORT[m]}", HSRC[m], G6A)
    for k in ((12, 20) if m.startswith("Mistral") else (16,)):
        rmac(f"HfourDtop{word(k)}{t}", H6S.med_over(M, "D", M.rank("P1")[:k])[0], f"median D over the top-{k} heads by a3 (H4 sensitivity to |C|), {SHORT[m]}", HSRC[m], G6A)
    rmac(f"HoneRsixteen{t}", M.R("P1", "a3", M.KS.index(16))[0], f"R(16) by a3, OPTIONS-AFTER, {SHORT[m]}", HSRC[m], G6A)
    nm(f"HlayerSpan{t}", str(len({c[0] for c in C})), f"number of distinct layers spanned by the {kc} heads of C [recomputed]", HSRC[m], GX6)
    for arm, at in HARM.items():
        E = M.E(arm)
        L = np.array([e["layer"]["m"] for e in E]) - np.array([[e["layer"]["none"]] for e in E])
        dg = np.mean([e["layer"]["allG"] - e["layer"]["none"] for e in E])
        rmac(f"HlayerSum{at}{t}", float(L.mean(0).sum() / dg), f"sum over layers of the single-layer shares of d_G, {at}, {SHORT[m]}", HSRC[m], GX6)
    nm(f"HcrossTopten{t}", str(len(set(M.rank("P1")[:10]) & set(M.rank("POST")[:10]))),
       "overlap of the top-10 a3 heads between OPTIONS-AFTER and SENTENCE-AFTER [recomputed]", HSRC[m], GX6)
    rmac(f"HcrossRho{t}", float(spearmanr(M.grid("P1", "a3").ravel(), M.grid("POST", "a3").ravel())[0]),
         f"Spearman of the full a3 grids across the two formats, {SHORT[m]}", HSRC[m], GX6)
    g = H6S.est(H6S.ratio, M.dG("POST"), M.dfull("POST"))
    rmac(f"HathreeRatioPost{t}", g[0], f"d_G/d_full under SENTENCE-AFTER, {SHORT[m]}", HSRC[m], G6A)
    af = H6S.est(lambda x: x, np.array([abs(e["curves"]["a3"]["allT"] - e["mF"]) for e in M.E("P1")]))
    rmac(f"HatwoAllFullOpt{t}Lo", af[1], f"Gate a2 |all_T - full| OPTIONS-AFTER, lower 95 % bound, {SHORT[m]}", HSRC[m], G6A)
    rmac(f"HatwoAllFullOpt{t}Hi", af[2], f"Gate a2 |all_T - full| OPTIONS-AFTER, upper 95 % bound, {SHORT[m]}", HSRC[m], G6A)
    CURV[m, "D_whole"] = Dg
# the stage-1 NO-MENTION ID_V for the cross-stage comparison E-d (same estimator as tab_decomposition, 2 decimals)
for m in HM:
    nm(f"idVNone{MODEL_TAG[m]}", nat[m]["dec"]["NONE"]["idV"][0], f"stage-1 ID_V under NO-MENTION (nats), {SHORT[m]} (as tab_decomposition, 2 decimals)",
       f"results/gpu_stage1/{m}_s0.json", "c.4 E-d (cross-stage)")

# ---------------------------------------------------------------- stage 6, part (b): the exchange on Prakash et al.'s binding swap
G6B = "c.4 sec:prakash / tab:prakash (stage 6 part b, preregistration H7-H12, Qwen2.5-14B)"
GX6B = "c.4 sec:prakash exploratory (E-a, E-b, sweeps, clamp onsets, replication)"
PKF = {"NO-MENTION": "None", "QNAMES": "Qnames", "OPTIONS-AFTER": "Opt", "LETTERS-AFTER": "Letter", "QNAMES2": "Qnamestwo"}
PKA = {("BIND", 28): "Bind", ("ID", 0): "IdZero", ("ID", 28): "IdTwentyeight"}
i = find(S6, "LM filter (NO-MENTION, both prompts)", start=B6)
a, b, c, d, e = grab(S6, i, r"(\d+)/(\d+) pass \(accuracy (\S+)\); population n = (\d+), .*: (\d+)/80")
smac("PkNpass", "S6", i, a, "pairs passing the LM filter (NO-MENTION, both prompts)", G6B)
smac("PkNpool", "S6", i, b, "pairs in Prakash et al.'s seed-10 pool (template 2)", G6B)
smac("PkFilterAcc", "S6", i, c, "LM-filter accuracy", G6B)
smac("PkN", "S6", i, d, "population size (the first passing pairs in pool order)", G6B)
smac("PkOverlap", "S6", i, e, "overlap of the population with the 81st-160th passing pairs (their validation split, approximate), of 80", G6B)
i = find(S6, "   l* = ", start=B6)
a, b = grab(S6, i, r"l\* = (\d+) .*l\*_ID = (\d+)")
smac("PkLstar", "S6", i, a, "l*: earliest maximum of BIND IIA on the NO-MENTION sweep", G6B)
smac("PkLstarId", "S6", i, b, "l*_ID: earliest maximum of ID IIA on the NO-MENTION sweep (tie rule on a flat curve)", G6B)
i = find(S6, "Gate b1 reproduction", start=B6)
a, b, c = grab(S6, i, r"BIND IIA\(l\* = 28\) = (\S+) .*ID IIA_ID\(l\*_ID = 0\) = (\S+) .*IIA_ID\(l\*\) = (\S+) ")
smac("PkGatebOneBind", "S6", i, a, "Gate b1: BIND IIA at l* (NO-MENTION sweep)", G6B)
smac("PkGatebOneId", "S6", i, b, "Gate b1: ID IIA at l*_ID", G6B)
smac("PkGatebOneIdAtLstar", "S6", i, c, "ID IIA at l* (H11 Part 2 condition)", G6B)
PK = {}
EXR = (r"(BIND|ID)\s+@\s*(\d+) (\S+)\s+n=(\d+) Gate b0 mean\|m\(r4\)-m\(r1\)\| (\S+) mean\|m\(r0\)-m\(B\)\| (\S+) \(<= 0.3\) -> (\w+);\s+"
       r"Gate b2 Phi " + CIR + r" \(>= 3\) -> (MET|NOT MET);\s+rule (holds|fails);\s+kappa (.*)$")
b0a, b0b = [], []
for i in find(S6, "Gate b0 mean", start=B6, many=True):
    arm, dep, f, n, ba, bb, b0v, ph, phl, phh, b2v, rl, rest = grab(S6, i, EXR)
    a_, ft = PKA[arm, int(dep)], PKF[f]
    PK[a_, ft] = dict(b0a=fl(ba), b0b=fl(bb), b0=b0v, phi=(ph, phl, phh), b2=b2v, rule=rl, n=n)
    b0a.append(fl(ba)); b0b.append(fl(bb))
    smac3(f"PkPhi{a_}{ft}", "S6", i, (ph, phl, phh), f"Phi = mean m(r1) - m(r0) (nats; Gate b2 >= 3), {arm}@{dep} {f}", G6B)
    nm(f"PkBzeroA{a_}{ft}", f"{fl(ba):.3f}", f"Gate b0 mean|m(r4)-m(r1)| (printed {ba}), {arm}@{dep} {f}", f"{SNAME['S6']}:{i + 1}", G6B)
    nm(f"PkBzeroB{a_}{ft}", f"{fl(bb):.3f}", f"Gate b0 mean|m(r0)-m(B)| (printed {bb}), {arm}@{dep} {f}", f"{SNAME['S6']}:{i + 1}", G6B)
    k = re.match(CIR + r" \(dropped (\S+)%\)", rest)
    if k:
        smac3(f"PkKappa{a_}{ft}", "S6", i, k.groups()[:3], f"kappa = psi_K/(psi_K+psi_V), {arm}@{dep} {f}", G6B)
        smac(f"PkDrop{a_}{ft}", "S6", i, k[4], f"percent of resamples dropped from kappa's CI, {arm}@{dep} {f}", G6B)
        PK[a_, ft]["kappa"] = k.groups()[:3]
    else:
        PK[a_, ft]["kappa"] = None
        PK[a_, ft]["ne"] = grab(S6, i, r"psi_K " + NUM + r", psi_V " + NUM + r", interaction " + NUM)
nm("PkBzeroAMin", f"{min(b0a):.3f}", "smallest Gate b0 mean|m(r4)-m(r1)| over the 13 cells", SNAME["S6"], G6B)
nm("PkBzeroAMax", f"{max(b0a):.3f}", "largest Gate b0 mean|m(r4)-m(r1)| over the 13 cells", SNAME["S6"], G6B)
nm("PkBzeroBMin", f"{min(b0b):.3f}", "smallest Gate b0 mean|m(r0)-m(B)| over the 13 cells", SNAME["S6"], G6B)
nm("PkBzeroBMax", f"{max(b0b):.3f}", "largest Gate b0 mean|m(r0)-m(B)| over the 13 cells", SNAME["S6"], G6B)
i = find(S6, "Gate b3 (H10 as a dissociation)", start=B6)
a = grab(S6, i, r"ID_K\(OPTIONS-AFTER, l\*\+1\) " + CIR)
smac3("PkGatebThreeIDK", "S6", i, a, "Gate b3(a): ID_K(OPTIONS-AFTER, l*+1) (nats)", G6B)
smac("PkGatebThreeSid", "S6", i, grab(S6, i, r"; s_ID " + NUM)[0], "Gate b3(a): s_ID(OPTIONS-AFTER, l*+1)", G6B)
smac3("PkGatebThreeKappa", "S6", i, grab(S6, i, r"\(OPTIONS-AFTER\) kappa " + CIR), "Gate b3(b): kappa_ID at l* under OPTIONS-AFTER", G6B)
i = find(S6, "(H7 secondary, against s_ID(f, 0)", start=B6)
for f, ft in list(PKF.items())[:3]:
    k_, s_, g_ = grab(S6, i, re.escape(f) + r" kappa " + NUM + r" s_ID " + NUM + r" gap " + NUM)
    smac(f"HsevenGapZero{ft}", "S6", i, g_, f"H7 secondary: |kappa_BIND(f) - s_ID(f, 0)|, {f}", G6B)
    smac(f"HsevenSidZero{ft}", "S6", i, s_, f"H7 secondary: s_ID(f, 0), {f}", G6B)
P6 = find(S6, "   PREDICTIONS (Qwen2.5-14B-Instruct)", start=B6)
i = find(S6, " -> NOT MET", "NO-MENTION kappa", "r = ", start=P6, end=P6 + 3)
gaps = []
for f, ft in list(PKF.items())[:3]:
    k_, s_, g_ = grab(S6, i, re.escape(f) + r" kappa " + NUM + r" s_ID " + NUM + r" gap " + NUM)
    smac(f"HsevenGap{ft}", "S6", i, g_, f"H7: |kappa_BIND(f) - s_ID(f, l*+1)| (threshold 0.25), {f}", G6B)
    smac(f"HsevenSid{ft}", "S6", i, s_, f"H7: s_ID(f, l*+1 = 29), {f}", G6B)
    gaps.append(g_)
smac("HsevenR", "S6", i, grab(S6, i, r"r = " + NUM)[0], "H7: Pearson r(kappa, s_ID(., 29)) over the three formats", G6B)
amac("HsevenGapMin", min(map(fl, gaps)), "smallest H7 gap, 2 decimals", f"{SNAME['S6']}:{i + 1}", G6B)
amac("HsevenGapMax", max(map(fl, gaps)), "largest H7 gap, 2 decimals", f"{SNAME['S6']}:{i + 1}", G6B)
i = find(S6, "psi_V - psi_K ", "-> NOT MET", start=P6, end=P6 + 20)
a, b = grab(S6, i, r"psi_V " + NUM + r" psi_K " + NUM)
smac("HeightPsiV", "S6", i, a, "H8: psi_V of BIND@28 under NO-MENTION (predicted >= 0.5)", G6B)
smac("HeightPsiK", "S6", i, b, "H8: psi_K of BIND@28 under NO-MENTION (predicted <= 0.25)", G6B)
smac3("HeightDiff", "S6", i, grab(S6, i, r"psi_V - psi_K " + CIR), "H8: psi_V - psi_K under NO-MENTION", G6B)
i = find(S6, "kappa(OA) - kappa(NM)", start=P6, end=P6 + 20)
smac3("HnineDiff", "S6", i, grab(S6, i, r"kappa\(OA\) - kappa\(NM\) " + CIR), "H9: kappa(OPTIONS-AFTER) - kappa(NO-MENTION) for BIND@28 (predicted >= 0.4)", G6B)
smac("HnineDrop", "S6", i, grab(S6, i, r"dropped (\S+)%")[0], "H9: percent of resamples dropped", G6B)
i = find(S6, "Part 1 MET", start=P6, end=P6 + 20)
p1, p2 = S6[i].split("Part 2")
for f, ft in list(PKF.items())[:3]:
    k_, s_, g_ = re.search(re.escape(f) + r" kappa " + NUM + r" s_ID " + NUM + r" gap " + NUM, p1).groups()
    smac(f"HelevenGap{ft}", "S6", i, f"{f} kappa {k_} s_ID {s_} gap {g_}".split("gap ")[1], f"H11 Part 1: |kappa_ID@0(f) - s_ID(f, 1)|, {f}", G6B)
    smac(f"HelevenSid{ft}", "S6", i, s_, f"H11 Part 1: s_ID(f, l*_ID+1 = 1), {f}", G6B)
    k_, s_, g_ = re.search(re.escape(f) + r" kappa " + NUM + r" s_ID " + NUM + r" gap " + NUM, p2).groups()
    smac(f"HelevenGapPartTwo{ft}", "S6", i, g_, f"H11 Part 2: |kappa_ID@28(f) - s_ID(f, 29)|, {f}", G6B)
d = re.search(r"kappa_ID\(OA\) - kappa_ID\(NM\) " + CIR, p1).groups()
smac3("HelevenDiff", "S6", i, d, "H11 Part 1: kappa_ID(OPTIONS-AFTER) - kappa_ID(NO-MENTION) at block 0 (predicted >= 0.4)", G6B)
# verdict strings (for the tables)
VH = {}
for i in range(find(S6, "VERDICTS (H1-H12"), find(S6, "SUMMARY: ")):
    mm = re.match(r"  (H\d+)\s+(.*?)\s+(?:Qwen.*?)?-> (MET|NOT MET|NOT EVALUABLE|NOT RUN)", S6[i])
    if mm:
        VH[mm[1]] = mm[3]
i = find(S6, "SUMMARY: ", end=B6)
a, b, c, d = grab(S6, i, r"(\d+) MET, (\d+) NOT MET, (\d+) NOT EVALUABLE, (\d+) NOT RUN of 12")
for nmx, v in (("Met", a), ("NotMet", b), ("NotEval", c), ("NotRun", d)):
    smac(f"Hcount{nmx}", "S6", i, v, f"stage 6: number of predictions {nmx}", "c.9 tab:prereg-h")

# exploratory exchange lines (psi_K, psi_V, interaction, rho, IIA, kappa_w, per-question split)
E6 = find(S6, "-- Qwen2.5-14B-Instruct: EXPLORATORY", start=B6)
for i in range(E6 + 1, len(S6)):
    mm = re.match(r"\s+(BIND|ID)\s+@\s*(\d+) (\S+)\s+psi_K " + NUM + r" psi_V " + NUM + r" interaction " + NUM + r" rho_K " + NUM
                  + r" rho_V " + NUM + r" IIA (\S+) kappa_w " + NUM, S6[i])
    if not mm:
        continue
    arm, dep, f, pk, pv, it, rk, rv, iia, kw = mm.groups()
    a_, ft = PKA[arm, int(dep)], PKF[f]
    PK[a_, ft].update(pk=pk, pv=pv, inter=it, rk=rk, rv=rv, iia=iia, kw=kw)
    for nmx, v, dsc in (("PsiK", pk, "psi_K: share of Phi reproduced by exchanging the keys alone"),
                        ("PsiV", pv, "psi_V: share of Phi reproduced by exchanging the values alone"),
                        ("Inter", it, "interaction 1 - psi_K - psi_V"), ("RhoK", rk, "rho_K (removal of the patched keys)"),
                        ("RhoV", rv, "rho_V (removal of the patched values)"), ("Flip", iia, "IIA of the full patch in this cell (fraction of answers flipped)"),
                        ("KappaW", kw, "kappa_w (words only, punctuation positions excluded)")):
        smac(f"Pk{nmx}{a_}{ft}", "S6", i, v, f"{dsc}, {arm}@{dep} {f}", GX6B if nmx in ("RhoK", "RhoV", "KappaW") else G6B)
    for q, pk_, pv_, n_ in re.findall(r"q=(\d): psi_K " + NUM + r" psi_V " + NUM + r" \(n=(\d+)\)", S6[i]):
        smac(f"PkPsiKq{word(q)}{a_}{ft}", "S6", i, pk_, f"psi_K for question q={q} (n={n_}), {arm}@{dep} {f}", GX6B)
        smac(f"PkPsiVq{word(q)}{a_}{ft}", "S6", i, pv_, f"psi_V for question q={q} (n={n_}), {arm}@{dep} {f}", GX6B)
        smac(f"PkNq{word(q)}{a_}{ft}", "S6", i, n_, f"pairs with question q={q}, {arm}@{dep} {f}", GX6B)
# natural clamp at p (s_ID by onset and format)
CL = {}
for i in find(S6, "   clamp ", start=E6, many=True):
    f, l0, n, ik, ikl, ikh, iv, ikv, s_, sl, sh, it = grab(S6, i, r"clamp (\S+)\s+l0=\s*(\d+) n=(\d+) ID_K " + CIR + r"  ID_V " + NUM + r"  ID_KV "
                                                           + NUM + r"  s_ID " + CIR + r"  interaction/ID_KV " + NUM)
    ft, w = PKF[f], word(l0)
    CL[f, int(l0)] = dict(idk=(ik, ikl, ikh), idv=iv, sid=(s_, sl, sh))
    smac3(f"PkSid{ft}{w}", "S6", i, (s_, sl, sh), f"s_ID(f, l0) = ID_K/(ID_K+ID_V) of the natural clamp at the state word from block l0={l0}, {f}", GX6B)
    smac3(f"PkIDK{ft}{w}", "S6", i, (ik, ikl, ikh), f"ID_K (nats) of the natural clamp from l0={l0}, {f}", GX6B)
    smac(f"PkIDV{ft}{w}", "S6", i, iv, f"ID_V (nats) of the natural clamp from l0={l0}, {f}", GX6B)
# sweeps (IIA / Phi by block)
SW = {}
for i in find(S6, "   sweep ", start=B6, many=True):
    mm = re.match(r"\s+sweep (BIND|ID)\s+(\S+)\s+IIA/Phi: (.*)$", S6[i])
    arm, f = mm[1], mm[2]
    pts = [x.split(":") for x in mm[3].split()]
    SW[arm, f] = {int(k): tuple(map(fl, v.split("/"))) for k, v in pts}
    for k, v in pts:
        ii, ph = v.split("/")
        smac(f"PkSweepIIA{arm.capitalize()}{PKF[f]}{word(k)}", "S6", i, ii, f"IIA at block {k}, {arm} sweep, {f}", GX6B)
        smac(f"PkSweepPhi{arm.capitalize()}{PKF[f]}{word(k)}", "S6", i, ph, f"Phi (nats) at block {k}, {arm} sweep, {f}", GX6B)

# recomputed part (b) quantities, through the committed scorer's Model class and the raw sweep files
PKD = ROOT / "results/gpu_stage6/prakash/Qwen2.5-14B-Instruct"
PKSRC = "results/gpu_stage6/prakash/Qwen2.5-14B-Instruct"
PKM = PKS.Model(PKD, False, lambda *a: None)
INV = {v: k for k, v in PKF.items()}
for (arm, dep, f), e in PKM.E.items():
    a_, ft = PKA[arm, dep], PKF[f]
    nm(f"PkNflip{a_}{ft}", str(int(e.ok1.sum())), f"number of the {e.n} pairs whose answer the full patch flips, {arm}@{dep} {f} [recomputed]", f"{PKSRC}/exchange.json", G6B)
    rmac(f"PkMargin{a_}{ft}", float(e.mB.mean()), f"base margin mean m(B) = log p(target) - log p(s_q) without patch (nats), {arm}@{dep} {f}", f"{PKSRC}/exchange.json", GX6B, "{:.2f}")
    if hasattr(e, "pvw"):
        rmac(f"PkPsiKw{a_}{ft}", float(e.pkw), f"words-only psi_K (punctuation positions excluded), {arm}@{dep} {f}", f"{PKSRC}/exchange.json", GX6B)
        rmac(f"PkPsiVw{a_}{ft}", float(e.pvw), f"words-only psi_V (punctuation positions excluded), {arm}@{dep} {f}", f"{PKSRC}/exchange.json", GX6B)
    PK[a_, ft]["nflip"] = int(e.ok1.sum())
e = PKM.ex("ID", 28, "LETTERS-AFTER")
rmac("PkDropIdTwentyeightLetter", 100 * e.drop, "percent of resamples failing the kappa rule in ID@28 LETTERS-AFTER (cell not evaluable)",
     f"{PKSRC}/exchange.json", GX6B, "{:.1f}")


def clamp_kappa(f, l0):
    """one-sided natural-clamp analogue of kappa: m = log p(S) - log p(s_q), psi from the K_S / V_S rows against ID."""
    cs = _CK[f, l0]
    mm_ = lambda c, r: c["lp"][r]["S"] - c["lp"][r]["s_q"]  # noqa: E731
    k = np.mean([mm_(c, "K_S") - mm_(c, "ID") for c in cs]); v = np.mean([mm_(c, "V_S") - mm_(c, "ID") for c in cs])
    return k / (k + v)


_CK = {}
_clamp_cells = json.load(open(PKD / "clamp.json"))["cells"]
for c in _clamp_cells:
    _CK.setdefault((c["format"], c["l0"]), []).append(c)
for f in ("NO-MENTION", "QNAMES", "OPTIONS-AFTER", "LETTERS-AFTER"):
    ft = PKF[f]
    for l0 in (1, 29):
        rmac(f"PkClampKappa{ft}{word(l0)}", clamp_kappa(f, l0), f"one-sided natural-clamp kappa (the exchange's statistic applied to the clamp) from l0={l0}, {f}",
             f"{PKSRC}/clamp.json", GX6B)
    kid = PKM.ex("ID", 0, f).kappa
    rmac(f"PkClampAgree{ft}", abs(kid - clamp_kappa(f, 1)), f"|kappa_ID@0 - one-sided clamp kappa from l0=1| (exchange vs clamp), {f}", f"{PKSRC}/exchange.json, clamp.json", GX6B)
    if f != "LETTERS-AFTER":
        rmac(f"PkClampStatGap{ft}", abs(clamp_kappa(f, 1) - PKM.sid(f, 1)), f"|one-sided clamp kappa - s_ID| from l0=1 (difference between the statistics), {f}",
             f"{PKSRC}/clamp.json", GX6B)
# replication of the released per-layer IIA (BIND, NO-MENTION)
THEIRS = {24: 0.69, 26: 0.72, 28: 1.00, 30: 1.00, 36: 0.06}   # Prakash et al.'s released Qwen2.5-14B per-layer IIA (80 validation pairs),
# as transcribed in docs/PREREGISTRATION.md, outcome of P-2026-10-05-H, replication table (not in results/)
filt = json.load(open(PKD / "filter.json"))
passing = [x["i"] for x in filt["pairs"] if x["ok"]]
shared = set(passing[80:160]) & set(PKM.pop)
sw = json.load(open(PKD / "sweep_BIND_NO-MENTION.json"))
REPL = {}
for blk, th in THEIRS.items():
    rows = sw["rows"][str(blk)]
    ours = np.mean([r["ok"] for r in rows]); sh_ = np.mean([r["ok"] for r in rows if r["i"] in shared])
    nm(f"PkTheirsIIA{word(blk)}", f"{th:.2f}", f"Prakash et al.'s released per-layer IIA at block {blk} (their 80 validation pairs)",
       "docs/PREREGISTRATION.md, outcome of P-2026-10-05-H, replication table", GX6B)
    rmac(f"PkOursIIA{word(blk)}", ours, f"our BIND IIA at block {blk}, NO-MENTION, 150 pairs", f"{PKSRC}/sweep_BIND_NO-MENTION.json", GX6B)
    rmac(f"PkSharedIIA{word(blk)}", sh_, f"our BIND IIA at block {blk} on the {len(shared)} pairs shared with their validation split", f"{PKSRC}/sweep_BIND_NO-MENTION.json", GX6B)
    REPL[blk] = (th, ours, sh_)
REPL_ALL = all(np.mean([r["ok"] for r in sw["rows"][str(b)]]) == 1.0 for b in range(30, 35))
assert REPL_ALL, "blocks 30-34 are no longer all 1.00"
nm("PkSharedN", str(len(shared)), "pairs shared between our population and their validation split [recomputed]", f"{PKSRC}/filter.json", GX6B)
# FP32 re-check (exploratory; the scorer skips labelled files)
fp = json.load(open(PKD / "sweep_BIND_NO-MENTION_fp32.json"))
FP = {}
for blk in sorted(map(int, fp["rows"])):
    a = {r["i"]: r for r in fp["rows"][str(blk)]}; b = {r["i"]: r for r in sw["rows"][str(blk)]}
    FP[blk] = (np.mean([r["ok"] for r in a.values()]), np.mean([b[i]["ok"] for i in a]))
    rmac(f"PkFpIIA{word(blk)}", FP[blk][0], f"FP32 re-check: BIND IIA at block {blk}, NO-MENTION", f"{PKSRC}/sweep_BIND_NO-MENTION_fp32.json", GX6B)
    rmac(f"PkBfIIA{word(blk)}", FP[blk][1], f"BF16 BIND IIA at block {blk} on the FP32 re-check pairs", f"{PKSRC}/sweep_BIND_NO-MENTION.json", GX6B)
a = {r["i"]: r for r in fp["rows"]["28"]}; b = {r["i"]: r for r in sw["rows"]["28"]}
rmac("PkFpAgree", np.mean([a[i]["ok"] == b[i]["ok"] for i in a]), "FP32 re-check: per-pair agreement of the flip at block 28", f"{PKSRC}/sweep_BIND_NO-MENTION_fp32.json", GX6B)
rmac("PkFpPhiDiff", abs(np.mean([a[i]["m_patch"] - a[i]["m_self"] for i in a]) - np.mean([b[i]["m_patch"] - b[i]["m_self"] for i in a])),
     "FP32 re-check: |Phi(FP32) - Phi(BF16)| at block 28 (nats)", f"{PKSRC}/sweep_BIND_NO-MENTION_fp32.json", GX6B, "{:.2f}")
fpi = json.load(open(PKD / "sweep_ID_NO-MENTION_fp32.json"))
rmac("PkFpIdIIAMin", min(np.mean([r["ok"] for r in v]) for v in fpi["rows"].values()), "FP32 re-check: smallest ID IIA over the re-checked blocks",
     f"{PKSRC}/sweep_ID_NO-MENTION_fp32.json", GX6B)
nm("PkFpIdBlocks", "--".join(str(x) for x in (min(map(int, fpi["rows"])), max(map(int, fpi["rows"])))), "blocks of the ID FP32 re-check",
   f"{PKSRC}/sweep_ID_NO-MENTION_fp32.json", GX6B)

# ---------------------------------------------------------------- stage 5 (preregistration G), parsed from STAGE5_SCORE.txt
# Part (a) depends on the .npz attention sidecars, which are not in the repository, so its numbers come from the score
# file only; parts (b)-(e) are reproduced byte for byte by re-running analysis/stage5_score.py on the committed raw files.
G5A = "c.4 sec:role (stage 5 part a, G1-G4: re-mention attention)"
G5B = "c.4 sec:readers rows (stage 5 part b, G5-G8: attention knockout)"
G5C = "c.4 sec:forms / sec:readers (stage 5 part c, G9-G12: membership and dose)"
G5D = "c.4 sec:forms (stage 5 part d, G13-G17: non-identical re-mentions)"
G5E = "c.4 sec:general (stage 5 part e, G18-G22: IOI)"
A5, B5, C5, D5, E5 = (sec(S5, f"######## PART ({p})") for p in "abcde")
ARM5 = {"POST": "Post", "P1": "Opt", "AFTER": "After", "NONE": "None"}
G5 = {}
AM = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"]
for m in AM:
    t = MT[m]
    h = find(S5, f"## {m} (", start=A5, end=B5)
    for arm in ("POST", "P1"):
        at = ARM5[arm]
        i = find(S5, f"  {arm:4s} n=150", start=h, end=h + 24)
        H_ = grab(S5, i, r"H\*=(\[.*?\])")[0]
        e_ = grab(S5, i + 1, r"E " + CIR); f_ = grab(S5, i + 1, r"  F " + CIR); fe = grab(S5, i + 1, r"F/E " + CIR); fb = grab(S5, i + 1, r"F_b " + CIR)
        smac3(f"GaE{at}{t}", "S5", i + 1, e_, f"E: excess attention of the later mention of B to the writing token at H*, {at}, {SH5[m]}", G5A)
        smac3(f"GaF{at}{t}", "S5", i + 1, f_, f"F: the same attention moved by the key clamp K_S, {at}, {SH5[m]}", G5A)
        smac3(f"GaFE{at}{t}", "S5", i + 1, fe, f"F/E, {at}, {SH5[m]}", G5A)
        smac3(f"GaFb{at}{t}", "S5", i + 1, fb, f"F_b (attention lost by B's mention under K_S), {at}, {SH5[m]}", G5A)
        g_ = grab(S5, i + 3, r"  G " + CIR)
        rs = grab(S5, i + 3, r"d\(ans->r_s\) " + CIR); rb = grab(S5, i + 3, r"d\(ans->r_b\) " + CIR)
        smac3(f"GaG{at}{t}", "S5", i + 3, g_, f"G: hop-2 attention of the answer position to the mention of B at H2*, {at}, {SH5[m]}", G5A)
        smac3(f"GaDansS{at}{t}", "S5", i + 3, rs, f"hop 2 under K_S: change of answer->r_s attention, {at}, {SH5[m]}", G5A)
        smac3(f"GaDansB{at}{t}", "S5", i + 3, rb, f"hop 2 under K_S: change of answer->r_b attention, {at}, {SH5[m]}", G5A)
        for j, cs in ((i + 4, "Low"), (i + 5, "Cap")):
            ik = grab(S5, j, r"ID_K " + CIR); dk = grab(S5, j, r"d_K " + CIR)
            em = "<- emitted" in S5[j]
            smac3(f"GaIDK{cs}{at}{t}", "S5", j, ik, f"ID_K (nats, eager run), {'lowercase' if cs == 'Low' else 'capitalised'} ids{' (emitted)' if em else ''}, {at}, {SH5[m]}", G5A)
            smac3(f"GaDK{cs}{at}{t}", "S5", j, dk, f"d_K (nats, non-specific key effect), {'lowercase' if cs == 'Low' else 'capitalised'} ids, {at}, {SH5[m]}", G5A)
            if em:
                G5[m, arm, "emit"] = cs
                G5[m, arm, "IDK"] = ik
        smac(f"GaCompetent{at}{t}", "S5", i + 6, grab(S5, i + 6, r"competent (\S+)")[0], f"fraction of competent cores, {at}, {SH5[m]}", G5A)
        G5[m, arm] = dict(E=e_, FE=fe, G=g_, H=H_)
    i = find(S5, "H*(POST) layers", start=h, end=h + 24)
    jac = grab(S5, i, r"Jaccard (\S+),")[0]
    smac(f"GaJaccard{t}", "S5", i, jac, f"Jaccard(H*(SENTENCE-AFTER), H*(OPTIONS-AFTER)), {SH5[m]}", G5A)
    smac3(f"GaRA{t}", "S5", i, grab(S5, i, r"R_A = E\(POST\)/E\(P1\) " + CIR), f"R_A = E(SENTENCE-AFTER)/E(OPTIONS-AFTER), {SH5[m]}", G5A)
    G5[m, "jac"] = jac
VG = find(S5, "== Verdicts (G1-G4", start=A5, end=B5)
for m in AM[:2]:
    t = MT[m]
    i = find(S5, "  G1 anchor", start=VG, end=VG + 12)
    e_, fe = grab(S5, i, re.escape(m) + r" E " + CIR + r" F/E " + CIR)[:3], grab(S5, i, re.escape(m) + r" E " + CIR + r" F/E " + CIR)[3:]
    smac3(f"GoneE{t}", "S5", i, e_, f"G1: E under SENTENCE-AFTER at the anchor {SH5[m]}", G5A)
    smac3(f"GoneFE{t}", "S5", i, fe, f"G1: F/E under SENTENCE-AFTER at the anchor {SH5[m]}", G5A)
    i = find(S5, "  G4a hop-2 anchor", start=VG, end=VG + 12)
    smac3(f"GfouraG{t}", "S5", i, grab(S5, i, re.escape(m) + r" G " + CIR), f"G4a: hop-2 attention G under SENTENCE-AFTER, anchor {SH5[m]}", G5A)
for m in AM[2:]:
    t = MT[m]
    i = find(S5, f"  G2 {m}:", start=VG, end=VG + 12)
    e_ = grab(S5, i, r"E " + CIR); fe = grab(S5, i, r"F/E " + CIR)
    smac3(f"GtwoE{t}", "S5", i, e_, f"G2: E under SENTENCE-AFTER at {SH5[m]}", G5A)
    smac3(f"GtwoFE{t}", "S5", i, fe, f"G2: F/E under SENTENCE-AFTER at {SH5[m]}", G5A)
    smac3(f"GtwoDKLow{t}", "S5", i, grab(S5, i, r"d_K lower " + CIR), f"G2: d_K (lowercase) beside ID_K, {SH5[m]}", G5A)
    smac3(f"GtwoDKCap{t}", "S5", i, grab(S5, i, r" cap " + CIR + r" beside"), f"G2: d_K (capitalised), {SH5[m]}", G5A)
    smac3(f"GtwoIDKLow{t}", "S5", i, grab(S5, i, r"ID_K lower " + CIR), f"G2: ID_K (lowercase) under SENTENCE-AFTER (within +-1 nat), {SH5[m]}", G5A)
    smac3(f"GtwoIDKCap{t}", "S5", i, grab(S5, i, r"ID_K lower " + CIR + r" cap " + CIR)[3:], f"G2: ID_K (capitalised) under SENTENCE-AFTER, {SH5[m]}", G5A)
    i = find(S5, "  G3 magnitude", start=VG, end=VG + 12)
    ra, er = grab(S5, i, re.escape(m) + r" R_A " + CIR + r" E\(m\)/min anchors " + CIR)[:3], grab(S5, i, re.escape(m) + r" R_A " + CIR + r" E\(m\)/min anchors " + CIR)[3:]
    smac3(f"GthreeRA{t}", "S5", i, ra, f"G3: R_A at {SH5[m]}", G5A)
    smac3(f"GthreeEratio{t}", "S5", i, er, f"G3: E(m)/min anchor E under SENTENCE-AFTER, {SH5[m]}", G5A)
    i = find(S5, "  G4b hop-2 cross-scale", start=VG, end=VG + 12)
    q_, g_ = grab(S5, i, re.escape(m) + r" Q " + CIR + r" \(G " + CIR + r"\)")[:3], grab(S5, i, re.escape(m) + r" Q " + CIR + r" \(G " + CIR + r"\)")[3:]
    smac3(f"GfourbQ{t}", "S5", i, q_, f"G4b: Q = G(m)/mean anchor G at {SH5[m]}", G5A)
    smac3(f"GfourbG{t}", "S5", i, g_, f"G4b: hop-2 attention G under SENTENCE-AFTER at {SH5[m]}", G5A)
# exploratory part (a)
for m in AM:
    t = MT[m]
    i = find(S5, f"  {m} cross-format:", start=VG, end=B5)
    a, b = grab(S5, i, r"P1 heads on POST E_cross " + CIR), grab(S5, i, r"POST heads on P1 E_cross " + CIR)
    smac(f"GaEcrossPost{t}", "S5", i, a[0], f"E of the OPTIONS-AFTER heads measured on SENTENCE-AFTER, {SH5[m]}", G5A)
    smac(f"GaEcrossOpt{t}", "S5", i, b[0], f"E of the SENTENCE-AFTER heads measured on OPTIONS-AFTER, {SH5[m]}", G5A)
    i = find(S5, f"  {m} stage-1 lowercase ID_K", start=VG, end=B5)
    a, b = grab(S5, i, r"\(sdpa, batched\): POST " + CIR + r" P1 " + CIR)[:3], grab(S5, i, r"\(sdpa, batched\): POST " + CIR + r" P1 " + CIR)[3:]
    smac(f"GaSdpaIDKPost{t}", "S5", i, a[0], f"stage-1 (sdpa) lowercase ID_K under SENTENCE-AFTER, {SH5[m]}", G5A)
    smac(f"GaSdpaIDKOpt{t}", "S5", i, b[0], f"stage-1 (sdpa) lowercase ID_K under OPTIONS-AFTER, {SH5[m]}", G5A)
    i = find(S5, f"  {m} POST: layer profile", start=VG, end=B5)
    top = grab(S5, i + 5, r"top-5 key columns at H2\* .*?: (\S+)=(\S+),")
    if top[0] == "p":
        smac(f"GaAnsToP{t}", "S5", i + 5, top[1], f"attention of the answer row at the hop-2 heads to the writing token itself, SENTENCE-AFTER, {SH5[m]}", G5A)
for m in AM[2:]:
    i = find(S5, f"  {m} d_K vs F_b co-variation", start=VG, end=B5)
    smac3(f"GaCovPearson{MT[m]}", "S5", i, grab(S5, i, r"lowercase Pearson " + CIR), f"Pearson(d_K, F_b) under SENTENCE-AFTER (lowercase), {SH5[m]}", G5A)

# part (b): attention knockout
KM = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3"]
MASKW = {"M0": "Mzero", "M1": "Mone", "M2": "Mtwo", "M2b": "Mtwob", "M3": "Mthree", "M4": "Mfour", "M5": "Mfive", "M6": "Msix",
         "M7": "Mseven", "M8": "Meight", "Mpost": "Mpost", "Mq": "Mq"}
KO5 = {}
KOR = (r"^\s+(\S+)\s+(AFTER|P1|POST|NONE)\s+(M\w+)\s+n=(\d+) ID_K " + CIR + r" r_K " + CIR + r"\s+ID_V " + CIR + r" span\s+" + NUM
       + r" q_V " + CIR + r"\s+acc (\S+)/(\S+) on (\S+)/(\S+) loc (\S+)")
for i in range(B5, C5):
    mm = re.match(KOR, S5[i])
    if not mm:
        continue
    g = mm.groups()
    m, arm, mask = g[0], g[1], g[2]
    KO5[m, arm, mask] = dict(IDK=g[4:7], RK=g[7:10], IDV=g[10:13], span=g[13], QV=g[14:17], accB=g[17], accS=g[18], onB=g[19], onS=g[20], loc=g[21])
    if mask in ("M2b", "M8", "Mq", "Mpost"):
        continue
    t, at, mw = MT[m], ARM5[arm], MASKW[mask]
    smac3(f"GbIDK{at}{mw}{t}", "S5", i, g[4:7], f"knockout {mask}: ID_K (nats), {at}, {SH5[m]}", G5B)
    smac3(f"GbRK{at}{mw}{t}", "S5", i, g[7:10], f"knockout {mask}: r_K = ID_K/ID_K(M0), {at}, {SH5[m]}", G5B)
    smac3(f"GbIDV{at}{mw}{t}", "S5", i, g[10:13], f"knockout {mask}: ID_V (nats), {at}, {SH5[m]}", G5B)
    smac3(f"GbQV{at}{mw}{t}", "S5", i, g[14:17], f"knockout {mask}: q_V = ID_V / span, {at}, {SH5[m]}", G5B)
    smac(f"GbSpan{at}{mw}{t}", "S5", i, g[13], f"knockout {mask}: span = mean d_KV (nats), {at}, {SH5[m]}", G5B)
    smac(f"GbAccB{at}{mw}{t}", "S5", i, g[17], f"knockout {mask}: acc_B (restricted argmax is B in the base run), {at}, {SH5[m]}", G5B)
    smac(f"GbOnB{at}{mw}{t}", "S5", i, g[19], f"knockout {mask}: on_B, {at}, {SH5[m]}", G5B)
assert len(KO5) == 3 * (12 * 3 + 5), len(KO5)
# G5
i0 = find(S5, "== G5 necessity", start=B5, end=C5)
rk5 = []
for m in KM:
    i = find(S5, f"   {m} ", start=i0, end=i0 + 5)
    for arm in ("AFTER", "P1", "POST"):
        v = grab(S5, i, rf" {arm} " + CIR)
        smac3(f"GfiveRK{ARM5[arm]}{MT[m]}", "S5", i, v, f"G5: r_K(M1) (candidate rows cannot attend to the writing token), {ARM5[arm]}, {SH5[m]}", G5B)
        rk5.append((fl(v[0]), v[0], i, fl(v[2]), arm))
amac("GfiveRKMin", min(x[0] for x in rk5), "G5: smallest r_K(M1) over the 9 model x format cells", f"{SNAME['S5']}:{i0 + 2}-{i0 + 4}", G5B, "{:.3f}")
amac("GfiveRKMax", max(x[0] for x in rk5), "G5: largest r_K(M1) over the 9 cells", f"{SNAME['S5']}:{i0 + 2}-{i0 + 4}", G5B, "{:.3f}")
amac("GfiveListUpperMax", max(x[3] for x in rk5 if x[4] != "POST"), "G5: largest upper bound of r_K(M1) in the list formats", f"{SNAME['S5']}:{i0 + 2}-{i0 + 4}", G5B, "{:.3f}")
# G6
i0 = find(S5, "== G6 matched control", start=B5, end=C5)
for m in KM:
    t = MT[m]
    i = find(S5, f"   {m} ", "r_K(M2)", start=i0, end=i0 + 12)
    smac3(f"GsixRK{t}", "S5", i, grab(S5, i, r"r_K\(M2\) " + CIR), f"G6: r_K(M2) (initial-location column cut, matched control), {SH5[m]}", G5B)
    a, b = grab(S5, i, r"\|dID_V\| (\S+) <= (\S+);")
    smac(f"GsixDIDV{t}", "S5", i, a, f"G6: |change of ID_V| under M2 (nats), {SH5[m]}", G5B)
    smac(f"GsixLim{t}", "S5", i, b, f"G6: limit for |change of ID_V| (nats), {SH5[m]}", G5B)
    smac3(f"GsixPaired{t}", "S5", i, grab(S5, i, r"paired ID_K M1-M2 " + CIR), f"G6: paired ID_K M1 - M2 (nats), {SH5[m]}", G5B)
    smac(f"GsixRKtwoinit{t}", "S5", i + 2, grab(S5, i + 2, r"r_K\(M2\) " + CIR)[0], f"G6: r_K(M2) on the 40 cores with two initial-location columns, {SH5[m]}", G5B)
# G7
i0 = find(S5, "== G7 the copy takes over", start=B5, end=C5)
for m in KM:
    t = MT[m]
    i = find(S5, f"   {m} ", "q_V^M0(NONE)", start=i0, end=i0 + 20)
    smac3(f"GsevenQVnone{t}", "S5", i, grab(S5, i, r"q_V\^M0\(NONE\) " + CIR), f"G7: q_V under NO-MENTION (M0), {SH5[m]}", G5B)
    for k, arm in ((1, "AFTER"), (3, "P1")):
        at = ARM5[arm]
        a = i + k; b = i + k + 1
        smac3(f"GsevenQV{at}{t}", "S5", a, grab(S5, a, r"q_V\^M1 " + CIR), f"G7(a): q_V under M1, {at}, {SH5[m]}", G5B)
        smac3(f"GsevenDIDV{at}{t}", "S5", a, grab(S5, a, r"paired ID_V M1-M0 " + CIR), f"G7(a): paired ID_V M1 - M0 (nats), {at}, {SH5[m]}", G5B)
        smac(f"GsevenRhoV{at}{t}", "S5", a, grab(S5, a, r"rho_V " + CIR)[0], f"G7: rho_V = ID_V(M1)/ID_V(M0), {at}, {SH5[m]}", G5B)
        ab, as_, ob, os_ = grab(S5, b, r"acc_B/S (\S+)/(\S+) on_B/S (\S+)/(\S+) ")
        for nmx, v in (("AccB", ab), ("AccS", as_), ("OnB", ob), ("OnS", os_)):
            smac(f"Gseven{nmx}{at}{t}", "S5", b, v, f"G7(b): {nmx} under M1, {at}, {SH5[m]}", G5B)
        smac(f"GsevenLoc{at}{t}", "S5", b, grab(S5, b, r"loc ratio " + CIR)[0], f"G7(b): loc_mass ratio M1/M0, {at}, {SH5[m]}", G5B)
        smac(f"GsevenSpan{at}{t}", "S5", b, grab(S5, b, r"span ratio " + CIR)[0], f"G7: span ratio M1/M0, {at}, {SH5[m]}", G5B)
# G8
i0 = find(S5, "== G8 routes", start=B5, end=C5)
for m in KM:
    t = MT[m]
    i = find(S5, f"   {m} ", "(a) NONE M3", start=i0, end=i0 + 14)
    smac3(f"GeightaRatio{t}", "S5", i, grab(S5, i, r"ID_V\^M3/ID_V\^M0 " + CIR), f"G8(a): ID_V(M3)/ID_V(M0) under NO-MENTION (answer cannot attend to the writing token), {SH5[m]}", G5B)
    smac3(f"GeightaPaired{t}", "S5", i, grab(S5, i, r"paired " + CIR), f"G8(a): paired ID_V M3 - M0 (nats), {SH5[m]}", G5B)
    i = find(S5, f"   {m} ", "(b) AFTER M3", start=i0, end=i0 + 14)
    smac3(f"GeightbRK{t}", "S5", i, grab(S5, i, r"r_K\(M3\) " + CIR), f"G8(b): r_K(M3) under LIST-AFTER, {SH5[m]}", G5B)
    smac(f"GeightbAccB{t}", "S5", i, grab(S5, i, r"acc_B (\S+) on_B")[0], f"G8(b): acc_B under M3, {SH5[m]}", G5B)
    for arm, key in (("AFTER", "(c) AFTER M4"), ("P1", "(c) P1    M4")):
        i = find(S5, f"   {m} ", key, start=i0, end=i0 + 14)
        smac3(f"GeightcRatio{ARM5[arm]}{t}", "S5", i, grab(S5, i, r"ID_V\^M4/ID_V\^M1 " + CIR), f"G8(c): ID_V(M4)/ID_V(M1), {ARM5[arm]}, {SH5[m]}", G5B)
        smac(f"GeightcOnB{ARM5[arm]}{t}", "S5", i, grab(S5, i, r"on_B\^M4 (\S+) ")[0], f"G8(c): on_B under M4, {ARM5[arm]}, {SH5[m]}", G5B)
# exploratory part (b)
X5 = find(S5, "== Exploratory", start=B5, end=C5)
for m in KM:
    t = MT[m]
    i = find(S5, f"   {m}: r_K / r_V", start=X5, end=C5)
    for arm in ("AFTER", "P1", "POST", "NONE"):
        j = find(S5, f"      {arm:5s} M", start=i + 1, end=i + 6)
        for mask, rk, rv in re.findall(r"(M\w+) " + NUM + "/" + NUM, S5[j]):
            if arm == "POST" and mask in ("M3", "M7"):
                smac(f"GbRKx{ARM5[arm]}{MASKW[mask]}{t}", "S5", j, rk, f"exploratory r_K under {mask}, {ARM5[arm]}, {SH5[m]}", G5B)
    j = find(S5, "row specificity AFTER", start=i, end=i + 8)
    smac3(f"GbRKfive{t}", "S5", j, grab(S5, j, r"r_K\(M5 trk\) " + CIR), f"exploratory r_K(M5): only the clamped candidates' rows cut, LIST-AFTER, {SH5[m]}", G5B)
    smac3(f"GbRKsix{t}", "S5", j, grab(S5, j, r"r_K\(M6 oth\) " + CIR), f"exploratory r_K(M6): the other candidates' rows cut, LIST-AFTER, {SH5[m]}", G5B)
    j = find(S5, "POST takeover", start=i, end=i + 8)
    smac3(f"GbQVpostMone{t}", "S5", j, grab(S5, j, r"q_V\(M1\) " + CIR), f"exploratory q_V under M1, SENTENCE-AFTER, {SH5[m]}", G5B)
    smac(f"GbQVpostMzero{t}", "S5", j, grab(S5, j, r"q_V\(M0\) " + CIR)[0], f"q_V under M0, SENTENCE-AFTER, {SH5[m]}", G5B)
i0 = find(S5, "== Gate b (sanity", start=B5, end=C5)
accs = [fl(grab(S5, i, r"acc_B (\S+) ")[0]) for i in range(i0 + 1, i0 + 13)]
amac("GbGateAccMax", max(accs), "Gate b: largest acc_B under M8 (all edges to the writing token cut) over 12 cells", f"{SNAME['S5']}:{i0 + 2}-{i0 + 13}", G5B)

# part (c): membership, dose and reader rows
CM = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
ARMC = {"NONE": "None", "S2": "Stwo", "S3": "Sthree", "S3out": "Sthreeout", "S3half": "Sthreehalf", "S4": "Sfour", "S4out": "Sfourout",
        "S6": "Ssix", "L2": "Ltwo", "L3": "Lthree", "L3out": "Lthreeout", "L4": "Lfour", "L4out": "Lfourout", "L6": "Lsix"}
ARMR = (r"^  (NONE|S2|S3|S3out|S3half|S4|S4out|S6|L2|L3|L3out|L4|L4out|L6)\s+" + CIR + r"\s+" + CIR + r"\s+" + NUM + r"\s+" + NUM
        + r"\s+" + NUM + r"\s+" + NUM + "$")
MC = {}
for m in CM:
    t = MT[m]
    h = find(S5, f"## {m} (", start=C5, end=D5)
    h1 = next(j for j in range(h + 1, D5) if S5[j].startswith("## ") or S5[j].startswith("=="))
    sd = find(S5, "seed-1 replication", start=h, end=h1, many=True)
    seed1 = sd[0] if sd else h1
    for i in range(h, h1):
        mm = re.match(ARMR, S5[i])
        if not mm:
            continue
        g = mm.groups()
        key = (m, g[0], 1 if i > seed1 else 0)
        MC[key] = dict(IDK=g[1:4], IDV=g[4:7], dK=g[7], dV=g[8], sK=g[9], sID=g[10])
        if i > seed1:
            continue
        aw = ARMC[g[0]]
        smac3(f"GcIDK{aw}{t}", "S5", i, g[1:4], f"ID_K (nats), arm {g[0]}, {SH5[m]}", G5C)
        smac3(f"GcIDV{aw}{t}", "S5", i, g[4:7], f"ID_V (nats), arm {g[0]}, {SH5[m]}", G5C)
        smac(f"GcDK{aw}{t}", "S5", i, g[7], f"d_K (nats), arm {g[0]}, {SH5[m]}", G5C)
        smac(f"GcDV{aw}{t}", "S5", i, g[8], f"d_V (nats), arm {g[0]}, {SH5[m]}", G5C)
    for fam in ("S", "L"):
        for s1, sfx in ((False, ""), (True, "Seedone")):
            lab = "(seed 1) " if s1 else ""
            if s1 and not find(S5, f"  G9 {lab}{fam} membership", start=h, end=h1, many=True):
                continue
            i = find(S5, f"  G9 {lab}{fam} membership", start=h, end=h1)
            a = grab(S5, i, rf"\(a\) {fam}3-NONE " + CIR); b = grab(S5, i, rf"{fam}3/{fam}6 " + CIR)
            c = grab(S5, i, rf"\(b\) {fam}3-{fam}3out " + CIR); d = grab(S5, i, rf"{fam}3out-NONE " + CIR)
            if True:
                smac3(f"Gnine{fam}InNone{sfx}{t}", "S5", i, a, f"G9 {lab}{fam}3 - NO-MENTION (paired ID_K, nats), {SH5[m]}", G5C)
                smac3(f"Gnine{fam}Ratio{sfx}{t}", "S5", i, b, f"G9 {lab}{fam}3/{fam}6, {SH5[m]}", G5C)
                smac3(f"Gnine{fam}InOut{sfx}{t}", "S5", i, c, f"G9 {lab}{fam}3 - {fam}3out (paired ID_K, nats), {SH5[m]}", G5C)
                smac3(f"Gnine{fam}OutNone{sfx}{t}", "S5", i, d, f"G9 {lab}{fam}3out - NO-MENTION (paired ID_K, nats), {SH5[m]}", G5C)
            i = find(S5, f"  G10 {lab}{fam} proportionality", start=h, end=h1)
            r2 = grab(S5, i, r"r_2 " + CIR); r3 = grab(S5, i, r"r_3 " + CIR); r4 = grab(S5, i, r"ladder r_2 \S+ r_3 \S+ r_4 (\S+)")[0]
            smac3(f"Gten{fam}Rtwo{sfx}{t}", "S5", i, r2, f"G10 {lab}r_2 = ID_K({fam}2)/ID_K({fam}6), {SH5[m]}", G5C)
            smac3(f"Gten{fam}Rthree{sfx}{t}", "S5", i, r3, f"G10 {lab}r_3, {SH5[m]}", G5C)
            smac(f"Gten{fam}Rfour{sfx}{t}", "S5", i, r4, f"G10 {lab}ladder r_4, {SH5[m]}", G5C)
            MC[m, fam, "r", s1] = (r2, r3, r4)
            if not s1:
                j = i + 1
                for step, nmx in (((3, 2), "ThreeTwo"), ((4, 3), "FourThree"), ((6, 4), "SixFour")):
                    v = grab(S5, j, rf"{fam}{step[0]}-{fam}{step[1]} " + CIR)
                    smac3(f"Gten{fam}Step{nmx}{t}", "S5", j, v, f"paired ID_K step {fam}{step[0]} - {fam}{step[1]} (nats, exploratory), {SH5[m]}", G5C)
    for s1, sfx in ((False, ""), (True, "Seedone")):
        if not find(S5, "  G11 (seed 1)" if s1 else "  G11 copy carries", start=h, end=h1, many=True):
            continue
        i = find(S5, "  G11 (seed 1)" if s1 else "  G11 copy carries", start=h, end=h1)
        d = grab(S5, i, r"ID_V\(L3out\)-ID_V\(L3\) " + CIR); rv = grab(S5, i, r"R_V " + CIR); gp = grab(S5, i, r"gap ID_V\(NONE\)-ID_V\(L3\) " + NUM)[0]
        smac3(f"GelevenDiff{sfx}{t}", "S5", i, d, f"G11 ID_V(L3out) - ID_V(L3) (nats){' seed 1' if s1 else ''}, {SH5[m]}", G5C)
        smac3(f"GelevenRV{sfx}{t}", "S5", i, rv, f"G11 R_V (share of the copy restored){' seed 1' if s1 else ''}, {SH5[m]}", G5C)
        smac(f"GelevenGap{sfx}{t}", "S5", i, gp, f"G11 evaluability gap ID_V(NONE) - ID_V(L3) (nats){' seed 1' if s1 else ''}, {SH5[m]}", G5C)
    for arm in ("S2", "S3", "L2", "L3", "S6", "L6", "S3out", "L3out"):
        i = find(S5, f"    {arm:5s} n=60 d(all)", start=h, end=h1)
        fw = grab(S5, i, r"f_words " + CIR); rest = grab(S5, i, r"rest_after_p (\S+) ")[0]; da = grab(S5, i, r"d\(all\) (\S+) ")[0]
        smac3(f"Gtwelve{ARMC[arm]}{t}", "S5", i, fw, f"G12 f_words: share of the splice effect d read by the named words' rows, {arm}, {SH5[m]}", G5C)
        smac(f"GtwelveRest{ARMC[arm]}{t}", "S5", i, rest, f"G12 share read by the rows after the question (rest_after_p), {arm}, {SH5[m]}", G5C)
        smac(f"GtwelveDall{ARMC[arm]}{t}", "S5", i, da, f"G12 d(all): total splice effect (nats), {arm}, {SH5[m]}", G5C)
        MC[m, arm, "fw"] = (fw, rest)
    i = find(S5, "S3half/S3", start=h, end=h1)
    smac3(f"GcHalf{t}", "S5", i, grab(S5, i, r"S3half/S3 " + CIR), f"S3half/S3 (one swapped candidate named, exploratory), {SH5[m]}", G5C)
    for fam in ("S2", "S3"):
        r_ = fl(MC[m, fam, 0]["IDK"][0]) / fl(MC[m, fam, 0]["dK"])
        amac(f"GcShare{ARMC[fam]}{t}", r_, f"ID_K/d_K in {fam} (share of the key-swap effect that is identity), {SH5[m]}", SNAME["S5"] + " part (c) arm table", G5C)
VC = find(S5, "== Verdicts (G9-G12", start=C5, end=D5)
i = find(S5, "  G12    reader rows", start=VC, end=D5)
smac("GtwelveCells", "S5", i, grab(S5, i, r"(\d+/\d+) cells")[0], "G12: cells meeting the criterion (of 12)", G5C)

# part (d): non-identical re-mentions
VARW = {"THE": "The", "MODIF": "Modif", "TITLE": "Title", "UPPER": "Upper", "PLURAL": "Plural", "SYN": "Syn", "FRMIX": "Frmix",
        "DEMIX": "Demix", "FR": "Fr", "DE": "De", "AFTER SYN": "AfterSyn", "AFTER FR": "AfterFr", "AFTER DE": "AfterDe"}
VD = {}
DR = (r"^    (THE|MODIF|TITLE|UPPER|PLURAL|SYN|FRMIX|DEMIX|FR|DE|AFTER SYN|AFTER FR|AFTER DE)\s+n=(\d+) r_K " + CIR + r" \(NONE-floored "
      + CIR + r"\)\s+rho_s " + CIR + r"\s+D " + CIR + r"\s+ID_K " + CIR + r" ID_V " + CIR)
DA = r"any-form \(n=(\d+)\): r_K\^any " + CIR + r"\s+rho_s\^any " + CIR + r"\s+r_K\^form " + CIR + r"\s+a_v (?:" + CIR + r" \(n=(\d+)\)|not evaluable)"
for m in CM:
    t = MT[m]
    h = find(S5, f"## {m}", start=D5, end=E5)
    h1 = next(j for j in range(h + 1, E5) if S5[j].startswith("## ") or S5[j].startswith("=="))
    for i in range(h, h1):
        mm = re.match(DR, S5[i])
        if not mm:
            continue
        g = mm.groups()
        v, w = g[0], VARW[g[0]]
        a = re.search(DA, S5[i + 1]).groups()
        VD[m, v] = dict(n=g[1], RK=g[2:5], RKnone=g[5:8], rho=g[8:11], IDK=g[14:17], IDV=g[17:20], nany=a[0], RKany=a[1:4], rhoany=a[4:7],
                        RKform=a[7:10], av=a[10:13], nav=a[13], ne="NOT EVALUABLE" in S5[i])
        smac3(f"GdRK{w}{t}", "S5", i, g[2:5], f"r_K of the {v} re-mention relative to the exact repeat (OTHER-floored, English measure), {SH5[m]}", G5D)
        smac3(f"GdRho{w}{t}", "S5", i, g[8:11], f"rho_s of the {v} re-mention, {SH5[m]}", G5D)
        smac(f"GdIDK{w}{t}", "S5", i, g[14], f"ID_K (nats) under {v}, {SH5[m]}", G5D)
        smac(f"GdIDV{w}{t}", "S5", i, g[17], f"ID_V (nats) under {v}, {SH5[m]}", G5D)
        smac(f"GdN{w}{t}", "S5", i, g[1], f"cores in the {v} cell, {SH5[m]}", G5D)
        smac3(f"GdRKany{w}{t}", "S5", i + 1, a[1:4], f"r_K^any (scored in any form) for {v}, {SH5[m]}", G5D)
        smac(f"GdRKform{w}{t}", "S5", i + 1, a[7], f"r_K^form (scored in the variant's own form) for {v}, {SH5[m]}", G5D)
        if a[10] is not None:
            smac3(f"GdAv{w}{t}", "S5", i + 1, a[10:13], f"a_v: attention of the {v} re-mention to the writing token relative to the exact repeat (span-summed), {SH5[m]}", G5D)
            smac(f"GdAvN{w}{t}", "S5", i + 1, a[13], f"cores in the a_v probe for {v}, {SH5[m]}", G5D)
VV = find(S5, "== Verdicts (G13-G17", start=D5, end=E5)


def rng(name, vars_, field, desc, models=CM):
    vals = [(fl(VD[m, v][field][0]), VD[m, v][field][0]) for m in models for v in vars_]
    lo, hi = min(vals), max(vals)
    nm(name + "Min", sv(lo[1]), desc + " (smallest over models)", SNAME["S5"] + " part (d) cells", G5D)
    nm(name + "Max", sv(hi[1]), desc + " (largest over models)", SNAME["S5"] + " part (d) cells", G5D)


rng("GthirteenThe", ["THE"], "RK", "G13 r_K(THE)")
rng("GthirteenModif", ["MODIF"], "RK", "G13 r_K(MODIF)")
rng("GfourteenCase", ["TITLE", "UPPER"], "RK", "G14 r_K of the case variants (TITLE, UPPER)")
rng("GfourteenPlural", ["PLURAL"], "RK", "G14 r_K(PLURAL)")
rng("GfifteenRK", ["SYN", "FRMIX", "DEMIX"], "RK", "G15 r_K of SYN, FRMIX, DEMIX")
rng("GfifteenRho", ["SYN", "FRMIX", "DEMIX"], "rho", "G15 rho_s of SYN, FRMIX, DEMIX")
for v in ("FRMIX", "DEMIX"):
    i = find(S5, f"  G17 {v} paired", start=VV, end=E5)
    for m in CM:
        x = grab(S5, i, SH5[m].replace(".", r"\.") + r":" + CIR)
        smac3(f"Gseventeen{VARW[v]}{MT[m]}", "S5", i, x, f"G17 paired ID_V({v}) - ID_V(SENTENCE-AFTER) (nats){' (not evaluable)' if 'Qwen2.5-14B' in SH5[m] else ''}, {SH5[m]}", G5D)
# G17 compensation size: the structure-matched floor sentence (POST_OTHER) against SENTENCE-AFTER, paired over all cores,
# through the committed part-(d) scorer's Model class (outcome record docs/PREREGISTRATION.md:545; not printed by the scorer)
sys.path.insert(0, str(ROOT))
from stage5_parts import variants as S5V  # noqa: E402
for m in ("Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"):
    _a = S5V.Model(ROOT / "results/gpu_stage5", m, lambda *a, **k: None, False).arms
    _ids = sorted(set(_a["POST_OTHER"]) & set(_a["POST"]))
    rmac(f"GdOtherComp{MT[m]}", float(np.mean(S5V.vec(_a["POST_OTHER"], _ids, "idV") - S5V.vec(_a["POST"], _ids, "idV"))),
         f"paired ID_V(POST_OTHER) - ID_V(SENTENCE-AFTER) (nats; the floor sentence's value compensation), {SH5[m]}",
         f"results/gpu_stage5/factorial/{m}_s0.json", G5D, "{:.2f}")
G16 = {}
for v in ("SYN", "FRMIX", "DEMIX"):
    i = find(S5, f"  G16 {v} ", start=VV, end=E5)
    for m in CM:
        G16[m, v] = grab(S5, i, SH5[m].replace(".", r"\.") + r":r\^any=(\w+),a=(\w+)")

# part (e): IOI
IM = ["gpt2", "Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Qwen2.5-7B", "Qwen2.5-14B-Instruct", "gpt2-xl"]
IARM = {"PLAIN": "Plain", "AFTER": "After", "BEFORE": "Before", "QUESTION": "Question", "INLINE": "Inline", "INLINE_BEFORE": "InlineBefore"}
IO5 = {}
IR = (r"^  (PLAIN|AFTER|BEFORE|QUESTION|INLINE|INLINE_BEFORE)\s+n=(\d+) sign\(ID_K\) (\S)\s+ID_K " + CIR + r"\s+ID_V " + CIR + r"\s+ID_KV "
      + CIR + r"\s+f_K " + CIR + r"\s+f_V " + CIR + r"\s+s_ID " + CIR)
for m in IM:
    t = MT[m]
    h = find(S5, f"## {m} (", start=E5)
    h1 = next((j for j in range(h + 1, len(S5)) if S5[j].startswith("## ") or S5[j].startswith("== ")), len(S5))
    for i in range(h, h1):
        mm = re.match(r"^  gate e (\S+)\s+n=(\d+) two-way (\S+) .* four-way (\S+) .* -> (passed|FAILED)", S5[i])
        if mm:
            IO5.setdefault((m, mm[1]), {}).update(two=mm[3], four=mm[4], gate=mm[5])
            smac(f"GeTwoway{IARM[mm[1]]}{t}", "S5", i, mm[3], f"Gate e two-way accuracy, {mm[1]}, {SH5[m]}", G5E)
            smac(f"GeFourway{IARM[mm[1]]}{t}", "S5", i, mm[4], f"Gate e four-way accuracy, {mm[1]}, {SH5[m]}", G5E)
            continue
        mm = re.match(IR, S5[i])
        if mm:
            g = mm.groups()
            IO5[m, g[0]].update(IDK=g[3:6], IDV=g[6:9], IDKV=g[9:12], FK=g[12:15], FV=g[15:18], SID=g[18:21])
            aw = IARM[g[0]]
            smac3(f"GeIDK{aw}{t}", "S5", i, g[3:6], f"IOI ID_K (nats), {g[0]}, {SH5[m]}", G5E)
            smac3(f"GeIDV{aw}{t}", "S5", i, g[6:9], f"IOI ID_V (nats), {g[0]}, {SH5[m]}", G5E)
            smac3(f"GeIDKV{aw}{t}", "S5", i, g[9:12], f"IOI ID_KV (nats), {g[0]}, {SH5[m]}", G5E)
            smac3(f"GeFK{aw}{t}", "S5", i, g[12:15], f"IOI f_K = ID_K/ID_KV, {g[0]}, {SH5[m]}", G5E)
            smac3(f"GeFV{aw}{t}", "S5", i, g[15:18], f"IOI f_V = ID_V/ID_KV, {g[0]}, {SH5[m]}", G5E)
    for key, pat, nmx, dsc in (("G19a", r"signed contrast sgn x \[ID_K\(AFTER\) - ID_K\(QUESTION\)\] " + CIR, "GnineteenContrast", "G19a: signed contrast ID_K(AFTER) - ID_K(QUESTION) (nats)"),
                               ("G21a", r"signed contrast sgn x \[ID_K\(AFTER\) - ID_K\(BEFORE\)\] " + CIR, "GtwentyoneaContrast", "G21a: signed contrast ID_K(AFTER) - ID_K(BEFORE) (nats)"),
                               ("G20", r"paired f_V\(QUESTION\) - f_V\(AFTER\) " + CIR, "GtwentyPaired", "G20: paired f_V(QUESTION) - f_V(AFTER)"),
                               ("G21b", r"\|ID_K\(Q\)\|/\|ID_K\(A\)\| " + CIR, "GtwentyonebRatio", "G21b: |ID_K(QUESTION)|/|ID_K(AFTER)|"),
                               ("G22a", r"vs PLAIN " + CIR, "GtwentytwoPlain", "G22a: signed contrast INLINE vs PLAIN (nats)"),
                               ("G22a", r"vs INLINE_BEFORE " + CIR, "GtwentytwoInlineBefore", "G22a: signed contrast INLINE vs INLINE_BEFORE (nats)")):
        hits = find(S5, f"  {key} ", start=h, end=h1, many=True)
        hits = [j for j in hits if re.search(pat, S5[j])]
        if hits:
            smac3(f"{nmx}{t}", "S5", hits[0], grab(S5, hits[0], pat), f"{dsc}, {SH5[m]}", G5E)
    hits = find(S5, "row splice (exploratory", start=h, end=h1, many=True)
    if hits:
        j = hits[0] + 1
        for g_, w in (("options_sx", "OptionsSX"), ("options", "Options")):
            smac3(f"GeSplice{w}{t}", "S5", j, grab(S5, j, rf" {g_} " + CIR), f"IOI row splice (exploratory, AFTER, n=60): share of the key effect read by the {g_} rows, {SH5[m]}", G5E)
VE = find(S5, "== Verdicts (G18-G22", start=E5)
for key, nmx in (("G18 ", "Geighteen"), ("G19a ", "Gnineteena"), ("G19b ", "Gnineteenb"), ("G20 ", "Gtwenty"), ("G21a ", "Gtwentyonea"), ("G21b ", "Gtwentyoneb")):
    i = find(S5, f"  {key}", start=VE)
    smac(f"{nmx}Count", "S5", i, grab(S5, i, r"(\d+/\d+) ->")[0], f"{key.strip()}: models meeting the prediction", G5E)
fks = [(fl(IO5[m, "PLAIN"]["FK"][0]), IO5[m, "PLAIN"]["FK"][0]) for m in IM[:3]]
nm("GeighteenFKMin", sv(min(fks)[1]), "G18: smallest f_K(PLAIN) over GPT-2 small, Qwen2.5-7B, Mistral-7B", SNAME["S5"] + " part (e)", G5E)
nm("GeighteenFKMax", sv(max(fks)[1]), "G18: largest f_K(PLAIN) over the three models", SNAME["S5"] + " part (e)", G5E)
inl = [(fl(IO5[m, "INLINE"]["IDK"][0]), IO5[m, "INLINE"]["IDK"][0]) for m in IM]
nm("GeInlineMin", sv(min(inl)[1]), "INLINE ID_K, most negative over the six IOI models (nats)", SNAME["S5"] + " part (e)", G5E)
nm("GeInlineMax", sv(max(inl)[1]), "INLINE ID_K, least negative over the six IOI models (nats)", SNAME["S5"] + " part (e)", G5E)
# stage-5 verdict strings and the summary
VG5 = {}
for i in range(find(S5, "VERDICTS (G1-G22"), find(S5, "PROVENANCE (")):
    mm = re.match(r"  (G\d+)\s+(.*?)\s{2,}(.*) -> (MET|NOT MET|NOT EVALUABLE)\s*(\[.*\])?$", S5[i])
    if mm:
        VG5[mm[1]] = (mm[3], mm[4], mm[5] or "")
assert len(VG5) == 22, len(VG5)
i = find(S5, "SUMMARY: ")
a, b = grab(S5, i, r"(\d+) MET, (\d+) NOT MET")
smac("GcountMet", "S5", i, a, "stage 5: predictions MET (of 22)", "c.9 tab:prereg-g")
smac("GcountNotMet", "S5", i, b, "stage 5: predictions NOT MET (of 22)", "c.9 tab:prereg-g")

# ---------------------------------------------------------------- v3 tables (generated; captions and labels live in the section files)
def tx(s, sign=False):
    """printed score-file number -> LaTeX: negatives in math mode, '+' kept only when sign=True."""
    s = s.strip()
    if s.lstrip("+-") in ("None", "nan"):
        return "--"
    neg = s.startswith("-")
    s = s.lstrip("+-")
    return f"$-${s}" if neg else (f"$+${s}" if sign else s)


def tci(t, sign=False):
    return f"{tx(t[0], sign)} [{tx(t[1])}, {tx(t[2])}]"


def tcell(t, sign=False):
    """estimate over its interval, as cell() does, from printed strings."""
    return (f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{tx(t[0], sign)}\\\\[-1pt]{{\\scriptsize[{tx(t[1])}, {tx(t[2])}]}}\\end{{tabular}}")


def two(a, b):
    return f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{a}\\\\[-1pt]{{\\scriptsize {b}}}\\end{{tabular}}"


def verd(v):
    v = v.strip()
    return {"MET": r"\textbf{met}", "NOT MET": r"\textbf{not met}", "NOT EVALUABLE": r"\textbf{not evaluable}", "NOT RUN": r"\textbf{not run}"}.get(v, v)


def slash(*xs):
    return " / ".join(xs)


def wtab(name, lines):
    (TAB / name).write_text("\n".join(lines) + "\n")


Q7, Q14, MI7, OL7, Q15, Q3 = ("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct",
                              "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct")
CK = r"$\checkmark$"
XX = r"$\times$"
NA = "--"
PH = "(ph)"

# ---- tab_accounts.tex: accounts vs observations (main text, single column, seven rows; c.7)
acc_rows = [
    (r"Hop~1: a sparse set of heads, whose core is duplicate-token heads, attends from the re-mentions to the writing token, key-matched, at every scale, also at 1.5B and 3B where $\mathrm{ID}_K \approx 0$ (G1--G3, H1--H4)",
     CK + "/" + XX, NA, NA, CK + "/" + PH),
    (r"The same words placed before the writing token give no key read (E1)", CK, XX, NA, CK),
    (r"The answer reads the option words, not the writing token (H5, G8b)", NA, XX, NA, CK),
    (r"Without a later mention the identity is copied through the answer's edge to the writing token (G8a, E4)", NA, CK, NA, CK),
    (r"Removing the readers raises the value copy (G7a, H3, I5)", NA, PH, NA, CK),
    (r"A remap refit without later mentions shows the same crossover (F1--F4)", NA, NA, XX, CK),
    (rf"Prakash et al.'s binding swap is key-dominant in every evaluable format ($\kappa$ {tx(PK['Bind', 'None']['kappa'][0])}, $\psi_V$ {tx(PK['Bind', 'None']['pv'])} without a later mention); later mentions shift $\kappa$ by {tx(macros['HnineDiff'], True)} (H7--H10)",
     NA, PH, NA, XX),
    (r"At 24B the remap's key read runs through the natural reader heads; ablating them removes most of its key-only effect (I1--I4)", NA, NA, NA, CK),
    (r"Without its readers, the remap keeps its effect on the margin (I6)", NA, NA, NA, XX),
]
lines = [r"\begin{tabular}{@{}p{0.54\columnwidth}cccc@{}}", r"\toprule", r"Observation (evidence) & DT & LB & FF & LM \\", r"\midrule"]
for r in acc_rows:
    lines.append(" & ".join(r) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_accounts.tex", lines)

# ---- tab_heads.tex: H1-H5 per model (OPTIONS-AFTER)
hq, hm = h6[Q7], h6[MI7]
rows = [
    ("", r"heads $k^*$ (of all heads)", f"{hq['kstar']} ({hq['nheads']})", f"{hm['kstar']} ({hm['nheads']})"),
    ("H1", r"$R(k^*)$, top-$k^*$ by $a_3$ see $K_S$", tcell(hq["H1"]), tcell(hm["H1"])),
    ("", r"$k_{80}$ by $a_3$ / by $f^+$ / by $d^-$", slash(*[x.replace("None", "none") for x in hq["k80"]]), slash(*[x.replace("None", "none") for x in hm["k80"]])),
    ("H2", r"$\mathrm{KO}(k^*)$, top-$k^*$ blinded", tcell(hq["H2"][0]), tcell(hm["H2"][0])),
    ("", r"random sets: $R$ / $\mathrm{KO}$ (mean of 3)", slash(tx(hq["H2"][1][0]), tx(hq["H2"][2][0])), slash(tx(hm["H2"][1][0]), tx(hm["H2"][2][0]))),
    ("H3", r"$\rho_K$ under mean-ablation of the top-$k^*$", tcell(hq["H3"]["rho"]), tcell(hm["H3"]["rho"])),
    ("", r"$\rho_K$, control sets (min--max)", f"{min(hq['H3']['rc'], key=fl).lstrip('+')}--{max(hq['H3']['rc'], key=fl).lstrip('+')}",
     f"{min(hm['H3']['rc'], key=fl).lstrip('+')}--{max(hm['H3']['rc'], key=fl).lstrip('+')}"),
    ("", r"rise of $\mathrm{ID}_V$ (nats)", tcell(hq["H3"]["dv"], True), tcell(hm["H3"]["dv"], True)),
    ("", r"answer kept (base argmax)", tx(hq["H3"]["base"]), tx(hm["H3"]["base"])),
    ("H4", r"$|C|$ (top heads by $a_3$ carrying 80\,\%)", hq["H4"]["kc"], hm["H4"]["kc"]),
    ("", r"median duplicate score $D$ over $C$", tcell(hq["H4"]["D"]), tcell(hm["H4"]["D"])),
    ("", r"median induction score $I$ over $C$", tx(hq["H4"]["I"][0]), tx(hm["H4"]["I"][0])),
    ("", r"median in-task $T_{\mathrm{dup}}$ over $C$", tx(hq["H4"]["T"][0]), tx(hm["H4"]["T"][0])),
    ("", r"top-10 by $a_3$ $\cap$ top-10 by $D$", hq["H4"]["ov"], hm["H4"]["ov"]),
    ("H5", r"$r_{\mathrm{ans}}$: only the answer row restored", tcell(hq["H5"]["ans"]), tcell(hm["H5"]["ans"])),
    ("", r"$r_{\mathrm{other}}$ / $r_{\mathrm{all}}$", slash(tx(hq["H5"]["other"][0]), tx(hq["H5"]["all"][0])), slash(tx(hm["H5"]["other"][0]), tx(hm["H5"]["all"][0]))),
    ("", r"$r_{\mathrm{ans}}(K)$ / $r_{\mathrm{ans}}(V)$ (secondary, not scored)", slash(tx(hq["H5"]["K"]), tx(hq["H5"]["V"])), slash(tx(hm["H5"]["K"]), tx(hm["H5"]["V"]))),
]
lines = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule", r" & Quantity (\fmtOpt{}) & Qwen2.5-7B & Mistral-7B \\", r"\midrule"]
for k, r in enumerate(rows):
    if r[0] and k:
        lines.append(r"\midrule")
    lines.append(" & ".join(r) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_heads.tex", lines)

# ---- tab_prakash.tex: the exchange on Prakash et al.'s binding swap (appendix; c.7)
PKCOL = ["None", "Qnames", "Opt", "Letter", "Qnamestwo"]
PKHEAD = [r"\fmtNone{}", r"\textsc{qnames}", r"\fmtOpt{}", r"\fmtLetter{}$^\dagger$", r"\textsc{qnames2}$^\dagger$"]


def pkcell(a_, ft):
    c = PK.get((a_, ft))
    if c is None:
        return NA
    top = (f"$\\kappa$ {tx(c['kappa'][0])} ({tx(c['pk'])}, {tx(c['pv'])})" if c["kappa"] else
           f"n/e ({tx(c['pk'])}, {tx(c['pv'])})")
    bot = f"flips {c['nflip']}/{c['n']}, $\\Phi$ {tx(c['phi'][0])}" + ("$^\\ddagger$" if c["b2"] != "MET" else "")
    return two(top, bot)


lines = [r"\begin{tabular}{@{}l" + "c" * 5 + r"@{}}", r"\toprule", " & " + " & ".join(PKHEAD) + r" \\", r"\midrule"]
for a_, lab in (("Bind", r"BIND@28"), ("IdZero", r"ID@0"), ("IdTwentyeight", r"ID@28")):
    lines.append(f"{lab} & " + " & ".join(pkcell(a_, ft) for ft in PKCOL) + r" \\")
lines.append(r"\midrule")
for l0, lab in ((29, r"$s_{\mathrm{ID}}(f, 29)$"), (1, r"$s_{\mathrm{ID}}(f, 1)$")):
    cells = []
    for f in ("NO-MENTION", "QNAMES", "OPTIONS-AFTER", "LETTERS-AFTER", "QNAMES2"):
        c = CL.get((f, l0))
        cells.append(NA if c is None else two(tx(c["sid"][0]), f"[{tx(c['sid'][1])}, {tx(c['sid'][2])}]"))
    lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_prakash.tex", lines)

# ---- tab_exchange.tex: all 13 exchange cells (appendix)
lines = [r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
         r"Edit & Format & $\Phi$ (nats) & flips & $\psi_K$ & $\psi_V$ & interaction & $\kappa$ & $\kappa_w$ & $\chi_K$ & $\chi_V$ \\", r"\midrule"]
FMTX = dict(zip(PKCOL, [r"\fmtNone{}", r"\textsc{qnames}", r"\fmtOpt{}", r"\fmtLetter{}", r"\textsc{qnames2}"]))
for a_, lab in (("Bind", r"BIND@28"), ("IdZero", r"ID@0"), ("IdTwentyeight", r"ID@28")):
    first = True
    for ft in PKCOL:
        c = PK.get((a_, ft))
        if c is None:
            continue
        kap = tci(c["kappa"]) if c["kappa"] else r"n/e"
        lines.append(f"{lab if first else ''} & {FMTX[ft]} & {tci(c['phi'])} & {c['nflip']}/{c['n']} & {tx(c['pk'])} & {tx(c['pv'])} & {tx(c['inter'])} & "
                     f"{kap} & {tx(c['kw'])} & {tx(c['rk'])} & {tx(c['rv'])} \\\\")
        first = False
    lines.append(r"\midrule" if a_ != "IdTwentyeight" else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_exchange.tex", lines)

# ---- tab_clamp.tex: s_ID of the natural clamp by format and onset (appendix; E-b)
L0S = sorted({l0 for (_, l0) in CL})
lines = [r"\begin{tabular}{@{}l" + "c" * len(L0S) + r"@{}}", r"\toprule", r"Format & " + " & ".join(f"$\\ell_0{{=}}{l}$" for l in L0S) + r" \\", r"\midrule"]
for f in ("NO-MENTION", "QNAMES", "OPTIONS-AFTER", "LETTERS-AFTER"):
    lines.append(f"{FMTX[PKF[f]]} & " + " & ".join(tx(CL[f, l]["sid"][0]) if (f, l) in CL else NA for l in L0S) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_clamp.tex", lines)

# ---- tab_replication.tex: released vs our per-layer IIA (appendix; recomputed)
lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Block & Prakash et al.\ (80 pairs) & ours (" + macros["PkN"] + r" pairs) & ours, " + macros["PkSharedN"] + r" shared pairs \\", r"\midrule"]
for blk, (th, ours, sh_) in REPL.items():
    lines.append(f"{'30--34' if blk == 30 else blk} & {th:.2f} & {ours:.3f} & {sh_:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_replication.tex", lines)

# ---- tab_ablation.tex: every ablation condition (appendix)
lines = [r"\begin{tabular}{@{}l" + "ccc" * 4 + r"@{}}", r"\toprule",
         r"Condition & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{SHORT[m]}, {ARM_TEX[a]}}}" for m in HM for a in ("P1", "POST")) + r" \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
         r" & " + " & ".join([r"$\rho_K$ & $\Delta\mathrm{ID}_V$ & kept"] * 4) + r" \\", r"\midrule"]
CLAB = {"none": "none", "top_kstar_mean": r"top-$k^*$ (mean)", "top_10_mean": "top-10 (mean)", "top_20_mean": "top-20 (mean)",
        "top_kstar_zero": r"top-$k^*$ (zero)", "rand0_kstar_mean": "random set 1", "rand1_kstar_mean": "random set 2",
        "rand2_kstar_mean": "random set 3", "active_kstar_mean": "most active at $G$", "next_kstar_mean": r"next $k^*$ by $a_3$"}
for c, lab in CLAB.items():
    cells = []
    for m in HM:
        for a in ("P1", "POST"):
            r_, d_, ms, ba = ABL[m, a, c]
            cells += [tx(r_[0]), tx(d_[0], True), tx(ba)]
    lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_ablation.tex", lines)

# ---- tab_prereg_g.tex: preregistration G (stage 5), every verdict as scored
def tri(*ms, f=lambda m: "", sep=" / "):
    return sep.join(f(m) for m in ms)


def M_(name):
    return macros[name]


K3 = [Q7, Q14, MI7]
gr = []
gr.append(("G1", r"Anchors attend: $E \geq 0.20$ (lower $> 0.10$) and $F/E \geq 0.5$ under \fmtPost{} at Qwen2.5-7B and 14B",
           f"$E$ {tci(G5[Q7, 'POST']['E'])}, $F/E$ {tx(G5[Q7, 'POST']['FE'][0])} (7B); $E$ {tci(G5[Q14, 'POST']['E'])}, $F/E$ {tx(G5[Q14, 'POST']['FE'][0])} (14B)", VG5["G1"][1]))
gr.append(("G2", r"Dissociation at 1.5B and 3B ($H_{\mathrm{diss}}$): the same attention while $\mathrm{ID}_K(\text{\fmtPost{}})$ stays within $\pm 1$ nat",
           f"$E$ {tci(G5[Q15, 'POST']['E'])}, $F/E$ {tx(G5[Q15, 'POST']['FE'][0])}, $\\mathrm{{ID}}_K$ (lowercase ids) {tx(M_('GtwoIDKLowQwenOnefive'))} (1.5B); "
           f"$E$ {tci(G5[Q3, 'POST']['E'])}, $F/E$ {tx(G5[Q3, 'POST']['FE'][0])}, $\\mathrm{{ID}}_K$ (lowercase ids) {tx(M_('GtwoIDKLowQwenThree'))} (3B; {tx(M_('GtwoIDKCapQwenThree'))} in the emitted casing)", VG5["G2"][1]))
gr.append(("G3", r"Magnitude under $H_{\mathrm{diss}}$: $R_A \geq 0.5$ and $E(m) \geq 0.5\times$ the smaller anchor $E$",
           f"$R_A$ {tx(M_('GthreeRAQwenOnefive'))} / {tx(M_('GthreeRAQwenThree'))}; $E(m)$/anchor {tx(M_('GthreeEratioQwenOnefive'))} / {tx(M_('GthreeEratioQwenThree'))} (1.5B / 3B)", VG5["G3"][1]))
gr.append(("G4", r"Hop 2 (attention): (a) $G \geq 0.10$ (lower $> 0.05$) at the anchors; (b) $Q \leq 0.5$ (upper $< 1$) at 1.5B and 3B",
           f"(a) $G$ {tx(M_('GfouraGQwenSeven'))} / {tx(M_('GfouraGQwenFourteen'))}; (b) $Q$ {tx(M_('GfourbQQwenOnefive'))} [{tx(M_('GfourbQQwenOnefiveLo'))}, {tx(M_('GfourbQQwenOnefiveHi'))}] / "
           f"{tx(M_('GfourbQQwenThree'))} [{tx(M_('GfourbQQwenThreeLo'))}, {tx(M_('GfourbQQwenThreeHi'))}]", VG5["G4"][1]))
gr.append(("G5", r"Cutting the candidate rows' attention to the writing token (M1): $r_K \leq 0.20$ (upper $\leq 0.25$) in the lists, $\leq 0.40$ in \fmtPost{}",
           f"\\fmtListA{{}} {tri(*K3, f=lambda m: tx(M_('GfiveRKAfter' + MT[m])))}; \\fmtOpt{{}} {tri(*K3, f=lambda m: tx(M_('GfiveRKOpt' + MT[m])))}; "
           f"\\fmtPost{{}} {tri(*K3, f=lambda m: tx(M_('GfiveRKPost' + MT[m])))}", VG5["G5"][1]))
gr.append(("G6", r"Matched control column (M2, \fmtListA{}): $r_K$ within $[0.8, 1.2]$, small $|\Delta\mathrm{ID}_V|$, $\mathrm{ID}_K$(M1)$-$(M2) $< 0$",
           f"$r_K$(M2) {tri(*K3, f=lambda m: tx(M_('GsixRK' + MT[m])))}", VG5["G6"][1]))
gr.append(("G7", r"$H_{\mathrm{redundant}}$: (a) the copy takes over under M1 in $\geq 2/3$ models in both list formats; (b) the answer is kept (acc$_B$, on$_B \geq 0.90$) in 3/3",
           f"(a) 3/3: $q_V$(M1) {tri(*K3, f=lambda m: tx(M_('GsevenQVAfter' + MT[m])))} (\\fmtListA{{}}) against {tri(*K3, f=lambda m: tx(M_('GsevenQVnone' + MT[m])))} under \\fmtNone{{}}; "
           f"(b) 1/3: acc$_B$ {tri(*K3, f=lambda m: tx(M_('GsevenAccBAfter' + MT[m])))} (\\fmtListA{{}}), {tri(*K3, f=lambda m: tx(M_('GsevenAccBOpt' + MT[m])))} (\\fmtOpt{{}})",
           VG5["G7"][1] + r" (partial takeover)"))
gr.append(("G8", r"Routes at the answer: (a) cutting answer$\to$writing token (M3) under \fmtNone{} leaves $\leq 0.6$ of $\mathrm{ID}_V$; (b) $r_K$(M3) $\geq 0.8$ under \fmtListA{}; (c) M4 against M1",
           f"(a) {tri(*K3, f=lambda m: tx(M_('GeightaRatio' + MT[m])))}; (b) {tri(*K3, f=lambda m: tx(M_('GeightbRK' + MT[m])))}; (c) {tri(*K3, f=lambda m: tx(M_('GeightcRatioAfter' + MT[m])))}", VG5["G8"][1]))
gr.append(("G9", r"Membership: naming the swapped candidates (S3, L3) reads the key; naming B with two others (S3out, L3out) does not",
           f"S3$-$\\fmtNone{{}} {tri(*K3, f=lambda m: tx(M_('GnineSInNone' + MT[m]), True))}; L3$-$\\fmtNone{{}} {tri(*K3, f=lambda m: tx(M_('GnineLInNone' + MT[m]), True))}; "
           f"S3out$-$\\fmtNone{{}} {tri(*K3, f=lambda m: tx(M_('GnineSOutNone' + MT[m]), True))}", VG5["G9"][1]))
gr.append(("G10", r"Proportionality refuted, per family: lower bound of $r_2 > 1/3$ and of $r_3 > 1/2$",
           f"sentences $r_2$ {tx(M_('GtenSRtwoQwenSeven'))} [{tx(M_('GtenSRtwoQwenSevenLo'))}, {tx(M_('GtenSRtwoQwenSevenHi'))}] / {tx(M_('GtenSRtwoQwenFourteen'))} / {tx(M_('GtenSRtwoMistralSeven'))}; "
           f"lists $r_2$ {tri(*K3, f=lambda m: tx(M_('GtenLRtwo' + MT[m])))}", VG5["G10"][1] + r" (lists met; sentences 2/3)"))
gr.append(("G11", r"The copy returns when S and X leave the list: $\mathrm{ID}_V$(L3out)$-\mathrm{ID}_V$(L3) $> 0$ and $R_V \geq 0.5$",
           f"$R_V$ {tri(*K3, f=lambda m: tx(M_('GelevenRV' + MT[m])))}", VG5["G11"][1] + " (0/3)"))
gr.append(("G12", r"Reader rows (splice): $f_{\mathrm{words}} \geq 0.5$ (lower $> 0.25$) in S2, S3, L2, L3",
           f"S2 {tri(*K3, f=lambda m: tx(M_('GtwelveStwo' + MT[m])))}; S3 {tri(*K3, f=lambda m: tx(M_('GtwelveSthree' + MT[m])))}; "
           f"L2 {tri(*K3, f=lambda m: tx(M_('GtwelveLtwo' + MT[m])))}; L3 {tri(*K3, f=lambda m: tx(M_('GtwelveLthree' + MT[m])))}", VG5["G12"][1] + f" ({M_('GtwelveCells')})"))
gr.append(("G13", r"Wrappers keep the read: $r_K \geq 0.75$ (lower $> 0.5$) for THE and MODIF, $\geq 3$ of 4 models",
           f"THE {tx(M_('GthirteenTheMin'))}--{tx(M_('GthirteenTheMax'))}; MODIF {tx(M_('GthirteenModifMin'))}--{tx(M_('GthirteenModifMax'))}", VG5["G13"][1] + " (THE met; MODIF not met)"))
gr.append(("G14", r"An exact repeat reads most: $r_K \leq 0.75$ (upper $< 1$) for each of six variants",
           f"TITLE, UPPER {tx(M_('GfourteenCaseMin'))}--{tx(M_('GfourteenCaseMax'))} (0/4 each); PLURAL {tx(M_('GfourteenPluralMin'))}--{tx(M_('GfourteenPluralMax'))}; SYN, FRMIX, DEMIX "
           f"$\\leq$ {tx(M_('GfifteenRKMax'))} (4/4 each)", VG5["G14"][1] + " (4/6 sub-verdicts)"))
gr.append(("G15", r"Token-level on the English measure: $r_K \leq 1/3$ and $\rho_s \leq 0.5$ for SYN, FRMIX, DEMIX",
           f"$r_K$ {tx(M_('GfifteenRKMin'))} to {tx(M_('GfifteenRKMax'))}; $\\rho_s$ {tx(M_('GfifteenRhoMin'))}--{tx(M_('GfifteenRhoMax'))}", VG5["G15"][1]))
g16 = "; ".join(f"{v} " + ", ".join(f"{SH5[m].replace('Qwen2.5-', 'Q').replace('Mistral-', 'M').replace('OLMo-2-', 'O')} {G16[m, v][0][0]}/{G16[m, v][1][0]}" for m in CM)
                for v in ("SYN", "FRMIX", "DEMIX"))
gr.append(("G16", r"Token pattern on $r_K^{\mathrm{any}}$ and $a_v$ in $\geq 3$ models, per variant",
           g16 + r" (t token, c concept, g graded; $r^{\mathrm{any}}$/$a_v$)", VG5["G16"][1] + " (graded)"))
gr.append(("G17", r"Value compensation: paired $\mathrm{ID}_V(v) - \mathrm{ID}_V(\text{\fmtPost{}}) > 0$ where the read is lost",
           f"FRMIX {tri(Q7, MI7, OL7, f=lambda m: tx(M_('GseventeenFrmix' + MT[m]), True))}; DEMIX {tri(Q7, MI7, OL7, f=lambda m: tx(M_('GseventeenDemix' + MT[m]), True))} "
           r"(Qwen2.5-7B / Mistral-7B / OLMo-2-7B)", VG5["G17"][1]))
gr.append(("G18", r"No key read in plain IOI: $f_K$(PLAIN) within $\pm 0.10$ (CI within $\pm 0.20$), 3/3",
           f"{tri('gpt2', Q7, MI7, f=lambda m: tx(M_('GeFKPlain' + MT[m]), True))} (GPT-2 small / Qwen2.5-7B / Mistral-7B)", VG5["G18"][1]))
gr.append(("G19", r"A later list opens a key read on IOI: (a) existence, (b) positive sign, 2/2",
           f"Qwen2.5-7B $\\mathrm{{ID}}_K$(AFTER) {tci((M_('GeIDKAfterQwenSeven'), M_('GeIDKAfterQwenSevenLo'), M_('GeIDKAfterQwenSevenHi')), True)}; Mistral-7B not evaluable (Gate e)",
           VG5["G19"][1] + f" ({M_('GnineteenaCount')})"))
gr.append(("G20", r"The lookup replaces the copy on IOI: $f_V$(AFTER) $\leq 0.5$, $f_V$(QUESTION) $\geq 0.75$",
           f"Qwen2.5-7B $f_V$ {tx(M_('GeFVAfterQwenSeven'))} (AFTER), {tx(M_('GeFVQuestionQwenSeven'))} (QUESTION); Mistral-7B not evaluable", VG5["G20"][1] + f" ({M_('GtwentyCount')})"))
gr.append(("G21", r"Controls: (a) BEFORE $|\mathrm{ID}_K| \leq 0.5$; (b) QUESTION; (c) GPT-2 small INLINE\_BEFORE $|\mathrm{ID}_K| \leq 0.5$",
           f"(a) {tx(M_('GeIDKBeforeQwenSeven'))} (Qwen2.5-7B); (b) met at Qwen2.5-7B, Mistral-7B not evaluable; (c) {tx(M_('GeIDKInlineBeforeGptTwo'))}", VG5["G21"][1] + " ((c) met)"))
gr.append(("G22", r"GPT-2 small in-sentence re-mention: (a) a key read, (b) inhibitory (replication of a disclosed pilot)",
           f"$\\mathrm{{ID}}_K$(INLINE) {tci((M_('GeIDKInlineGptTwo'), M_('GeIDKInlineGptTwoLo'), M_('GeIDKInlineGptTwoHi')))}", VG5["G22"][1]))
lines = [r"\begin{tabular}{p{0.04\textwidth}p{0.36\textwidth}p{0.42\textwidth}p{0.08\textwidth}}", r"\toprule", r" & Prediction & Observed & Verdict \\", r"\midrule"]
for k, (gid, pr, ob, v) in enumerate(gr):
    if gid in ("G5", "G9", "G13", "G18"):
        lines.append(r"\midrule")
    vv, _, rest = v.partition(" (")
    lines.append(f"{gid} & {pr} & {ob}{('; ' + rest[:-1]) if rest else ''} & {verd(vv)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_prereg_g.tex", lines)

# ---- tab_prereg_h.tex: preregistration H (stage 6), gates and every verdict as scored
hr = []
hr.append(("a1", r"FP32 exactness of the head and hop splices (unit tests) before any model", "7 passed, 0 failed", "MET"))
hr.append(("a2", r"BF16 floor per model and format: $|\text{none}-\text{clean}|$ and $|\text{all}_T-\text{full}| \leq \max(0.5, 0.02\,\bar d_{\mathrm{full}})$; hop exactness $\leq 0.1$",
           f"Qwen2.5-7B \\fmtOpt{{}} {tx(M_('HatwoNoneCleanOptQwenSeven'))}, {tx(M_('HatwoAllFullOptQwenSeven'))} $\\leq$ {tx(M_('HatwoBoundOptQwenSeven'))}; "
           f"\\fmtPost{{}} {tx(M_('HatwoNoneCleanPostQwenSeven'))} $>$ {tx(M_('HatwoBoundPostQwenSeven'))}; Mistral-7B $\\leq$ {tx(M_('HatwoAllFullPostMistralSeven'))} in both; hop 0.0000",
           "MET except Qwen2.5-7B \\fmtPost{}"))
hr.append(("a3", r"$\bar d_{\mathrm{full}} \geq 10$ (\fmtOpt{}), $> 0$ (\fmtPost{}); $d_G/d_{\mathrm{full}} \geq 0.8$",
           f"{tx(M_('HathreeDfullOptQwenSeven'))} / {tx(M_('HathreeDfullOptMistralSeven'))}; {tx(M_('HathreeDfullPostQwenSeven'))} / {tx(M_('HathreeDfullPostMistralSeven'))}; "
           f"{tx(M_('HathreeRatioQwenSeven'))} / {tx(M_('HathreeRatioMistralSeven'))}", "MET"))
hr.append(("H1", r"Sparsity: $R(k^*) \geq 0.8$, lower bound $\geq 0.7$ (top 5\,\% by $a_3$)",
           f"{tci(hq['H1'])} / {tci(hm['H1'])}", VH["H1"]))
hr.append(("H2", r"Necessity and specificity: $\mathrm{KO}(k^*) \geq 0.8$; random sets $R, \mathrm{KO} \leq 0.25$",
           f"KO {tx(hq['H2'][0][0])} / {tx(hm['H2'][0][0])}; random $R$ {tx(hq['H2'][1][0])} / {tx(hm['H2'][1][0])}, KO {tx(hq['H2'][2][0])} / {tx(hm['H2'][2][0])}", VH["H2"]))
hr.append(("H3", r"Ablation: $\rho_K \leq 0.5$; candidate mass kept; control sets $\rho_K \geq 0.75$; $\mathrm{ID}_V$ rises above the floor; answer kept $\geq 0.8$",
           f"$\\rho_K$ {tx(hq['H3']['rho'][0])} / {tx(hm['H3']['rho'][0])}; $\\Delta\\mathrm{{ID}}_V$ {tx(hq['H3']['dv'][0], True)} / {tx(hm['H3']['dv'][0], True)}; kept {tx(hq['H3']['base'])} / {tx(hm['H3']['base'])}", VH["H3"]))
hr.append(("H4", r"Canonical duplicate-token heads over $C$: median $D \geq 0.2$, median $I \leq 0.1$; median $T_{\mathrm{dup}} \geq 0.2$",
           f"$D$ {tx(hq['H4']['D'][0])} / {tx(hm['H4']['D'][0])}; $I$ {tx(hq['H4']['I'][0])} / {tx(hm['H4']['I'][0])}; $T_{{\\mathrm{{dup}}}}$ {tx(hq['H4']['T'][0])} / {tx(hm['H4']['T'][0])}", VH["H4"]))
hr.append(("H5", r"Second hop: $r_{\mathrm{ans}}(KV) \geq 0.5$; $r_{\mathrm{ans}} > r_{\mathrm{other}}$; $r_{\mathrm{all}} \geq 0.8$",
           f"{tx(hq['H5']['ans'][0])} / {tx(hm['H5']['ans'][0])}; $r_{{\\mathrm{{other}}}}$ {tx(hq['H5']['other'][0])} / {tx(hm['H5']['other'][0])}; secondary (not scored) $r_{{\\mathrm{{ans}}}}(K)$ {tx(hq['H5']['K'])} / {tx(hm['H5']['K'])}, "
           f"$r_{{\\mathrm{{ans}}}}(V)$ {tx(hq['H5']['V'])} / {tx(hm['H5']['V'])}; \\fmtPost{{}} line not evaluable", VH["H5"]))
hr.append(("H6", r"The same readers for list and sentence: top-20 overlap $\geq 10$",
           f"{hq['H6'][0]} (Gate a2 failed) / {hm['H6'][0]}", VH["H6"]))
hr.append(("b0", r"Exchange exactness per cell: $|m(r_4)-m(r_1)|$, $|m(r_0)-m(B)| \leq 0.3$ nats", f"{M_('PkBzeroAMin')}--{M_('PkBzeroAMax')}; {M_('PkBzeroBMin')}--{M_('PkBzeroBMax')} (13 cells)", "MET"))
hr.append(("b1", r"Reproduction (\fmtNone{} sweep): IIA $\geq 0.7$ at $\ell^*$ (BIND) and $\ell^*_{\mathrm{ID}}$ (ID)",
           f"BIND {M_('PkGatebOneBind')} at $\\ell^* = {M_('PkLstar')}$; ID {M_('PkGatebOneId')} at $\\ell^*_{{\\mathrm{{ID}}}} = {M_('PkLstarId')}$", "MET"))
hr.append(("b2", r"Effect size $\Phi \geq 3$ nats per cell", f"met in 11 of 13 cells; \\fmtLetter{{}} at depth 28: {tx(M_('PkPhiBindLetter'))}, {tx(M_('PkPhiIdTwentyeightLetter'))}", "MET except \\fmtLetter{} at 28"))
hr.append(("b3", r"For H10 as a dissociation: identity key-read route at $\ell^*+1$ ($s_{\mathrm{ID}} \geq 0.5$) and $\kappa_{\mathrm{ID}}(\ell^*) \geq 0.5$ under \fmtOpt{}",
           f"$s_{{\\mathrm{{ID}}}}$ {tx(M_('PkGatebThreeSid'))}; $\\kappa_{{\\mathrm{{ID}}}}$ {tx(M_('PkGatebThreeKappa'))}", "NOT MET"))
hr.append(("H7", r"The law on the binding swap: $|\kappa(f) - s_{\mathrm{ID}}(f, 29)| \leq 0.25$ for each evaluable $f$ and $r \geq 0.9$",
           f"gaps {M_('HsevenGapNone')} / {M_('HsevenGapQnames')} / {M_('HsevenGapOpt')}; $r = {M_('HsevenR')}$; secondary, against $s_{{\\mathrm{{ID}}}}(f, 0)$: gaps {M_('HsevenGapZeroNone')} / {M_('HsevenGapZeroQnames')} / {M_('HsevenGapZeroOpt')}", VH["H7"]))
hr.append(("H8", r"Under \fmtNone{}: $\psi_V \geq 0.5$, $\psi_K \leq 0.25$",
           f"$\\psi_V$ {tx(M_('HeightPsiV'))}, $\\psi_K$ {tx(M_('HeightPsiK'))}; $\\psi_V-\\psi_K$ {tci((M_('HeightDiff'), M_('HeightDiffLo'), M_('HeightDiffHi')))}", VH["H8"]))
hr.append(("H9", r"Crossover: $\kappa(\text{\fmtOpt{}}) - \kappa(\text{\fmtNone{}}) \geq 0.4$; \textsc{qnames} between",
           f"{tci((M_('HnineDiff'), M_('HnineDiffLo'), M_('HnineDiffHi')), True)}; $\\kappa$(\\textsc{{qnames}}) {tx(M_('PkKappaBindQnames'))} not between", VH["H9"]))
hr.append(("H10", r"The rival: $\kappa(f) \leq 0.25$ in every evaluable format",
           f"$\\kappa$ {tx(M_('PkKappaBindNone'))} / {tx(M_('PkKappaBindQnames'))} / {tx(M_('PkKappaBindOpt'))}", VH["H10"]))
hr.append(("H11", r"Identity edit: Part 1 gaps to $s_{\mathrm{ID}}(f, 1) \leq 0.25$ and $\kappa_{\mathrm{ID}}$ crossover $\geq 0.4$; Part 2 gaps at depth 28",
           f"Part 1 gaps {M_('HelevenGapNone')} / {M_('HelevenGapQnames')} / {M_('HelevenGapOpt')}, crossover {tci((M_('HelevenDiff'), M_('HelevenDiffLo'), M_('HelevenDiffHi')), True)}; "
           f"Part 2 gaps {M_('HelevenGapPartTwoNone')} / {M_('HelevenGapPartTwoQnames')} / {M_('HelevenGapPartTwoOpt')}", VH["H11"]))
hr.append(("H12", r"Optional replication at Llama-3-70B", "not run (Gate b1 passed at 14B)", VH["H12"]))
lines = [r"\begin{tabular}{p{0.04\textwidth}p{0.36\textwidth}p{0.42\textwidth}p{0.08\textwidth}}", r"\toprule", r" & Gate or prediction & Observed & Verdict \\", r"\midrule"]
for gid, pr, ob, v in hr:
    if gid in ("H1", "b0", "H7"):
        lines.append(r"\midrule")
    vv, sep_, rest = v.partition(" except ")
    lines.append(f"{gid} & {pr} & {ob}{('; not met: ' + rest) if sep_ else ''} & {verd(vv)}{'$^*$' if sep_ else ''} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_prereg_h.tex", lines)

# ---- stage-5 appendix tables
# re-mention attention (part a)
lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule", r"Model & Format & $E$ & $F/E$ & $G$ (hop 2) & $\mathrm{ID}_K$ (nats) & $d_K$ (nats) \\", r"\midrule"]
for m in (Q15, Q3, Q7, Q14):
    for k, arm in enumerate(("POST", "P1")):
        cs = G5[m, arm, "emit"]
        lines.append(f"{SHORT[m] if k == 0 else ''} & {ARM_TEX[arm]} & {tcell(G5[m, arm]['E'])} & {tx(G5[m, arm]['FE'][0])} & {tcell(G5[m, arm]['G'])} & "
                     f"{tcell(G5[m, arm, 'IDK'], True)} & {tx(M_(f'GaDK{cs}{ARM5[arm]}{MT[m]}'), True)} \\\\")
    lines.append(r"\midrule" if m != Q14 else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_attention.tex", lines)

# knockout masks (part b)
MASKS = ["M0", "M1", "M2", "M2b", "M3", "M4", "M5", "M6", "M7", "M8", "Mq", "Mpost"]
lines = [r"\begin{tabular}{@{}ll" + "ccc" * 3 + r"@{}}", r"\toprule",
         r"Format & Mask & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{SHORT[m]}}}" for m in K3) + r" \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}", r" & & " + " & ".join([r"$r_K$ & $q_V$ & acc$_B$"] * 3) + r" \\", r"\midrule"]
for arm in ("AFTER", "P1", "POST", "NONE"):
    first = True
    for mk in MASKS:
        if (K3[0], arm, mk) not in KO5:
            continue
        cells = []
        for m in K3:
            c = KO5[m, arm, mk]
            cells += [tx(c["RK"][0]), tx(c["QV"][0]), tx(c["accB"])]
        lines.append(f"{ARM_TEX[arm] if first else ''} & {mk} & " + " & ".join(cells) + r" \\")
        first = False
    lines.append(r"\midrule" if arm != "NONE" else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_knockout.tex", lines)

# membership ladders (part c)
CARMS = ["NONE", "S2", "S3", "S3half", "S3out", "S4", "S4out", "S6", "L2", "L3", "L3out", "L4", "L4out", "L6"]
lines = [r"\begin{tabular}{@{}l" + "cc" * 4 + r"@{}}", r"\toprule", r"Arm & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{SHORT[m]}}}" for m in CM) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}", r" & " + " & ".join([r"$\mathrm{ID}_K$ & $\mathrm{ID}_V$"] * 4) + r" \\", r"\midrule"]
for a in CARMS:
    if a == "L2":
        lines.append(r"\midrule")
    lines.append(f"{a.replace('NONE', chr(92) + 'fmtNone{}')} & " + " & ".join(f"{tx(MC[m, a, 0]['IDK'][0], True)} & {tx(MC[m, a, 0]['IDV'][0], True)}" for m in CM) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{9}{@{}l}{\emph{Reader rows (splice, $n = 60$): $f_{\mathrm{words}}$ / share read by the rows after the question}} \\")
for a in ("S2", "S3", "S6", "S3out", "L2", "L3", "L6", "L3out"):
    lines.append(f"{a} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{tx(MC[m, a, 'fw'][0][0])} / {tx(MC[m, a, 'fw'][1])}}}" for m in CM) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_membership.tex", lines)

# variants (part d)
VARS = ["THE", "MODIF", "TITLE", "UPPER", "PLURAL", "SYN", "FRMIX", "DEMIX", "FR", "DE", "AFTER SYN"]
lines = [r"\begin{tabular}{@{}l" + "ccc" * 4 + r"@{}}", r"\toprule", r"Variant & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{SHORT[m]}}}" for m in CM) + r" \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
         r" & " + " & ".join([r"$r_K$ & $r_K^{\mathrm{any}}$ & $a_v$"] * 4) + r" \\", r"\midrule"]
for v in VARS:
    cells = []
    for m in CM:
        c = VD[m, v]
        cells += [tx(c["RK"][0]), tx(c["RKany"][0]), tx(c["av"][0]) if c["av"][0] else NA]
    lines.append(f"{v.replace('AFTER ', 'list ')} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_variants.tex", lines)

# IOI by model and arm (part e)
IARMS = ["PLAIN", "AFTER", "BEFORE", "QUESTION", "INLINE", "INLINE_BEFORE"]
lines = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule", r"Model & Arm & Gate e (2-way / 4-way) & $\mathrm{ID}_K$ (nats) & $\mathrm{ID}_V$ & $\mathrm{ID}_{KV}$ & $f_K$ & $f_V$ \\", r"\midrule"]
for m in IM:
    for k, a in enumerate(IARMS):
        c = IO5[m, a]
        g_ = f"{c['two']} / {c['four']}" + ("" if c["gate"] == "passed" else r" (failed)")
        lines.append(f"{SH5[m] + (' (expl.)' if m in IM[3:] else '') if k == 0 else ''} & {a.replace('_', chr(92) + '_')} & {g_} & {tci(c['IDK'], True)} & {tx(c['IDV'][0], True)} & "
                     f"{tx(c['IDKV'][0], True)} & {tx(c['FK'][0], True)} & {tx(c['FV'][0], True)} \\\\")
    lines.append(r"\midrule" if m != IM[-1] else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_ioi.tex", lines)

# ---------------------------------------------------------------- v3 figures
GREY = "#9a9893"
RCOL = {"a3": INK, "fplus": PURPLE, "dminus": BROWN}
RLAB = {"a3": "$a_3$ (attention change)", "fplus": "$f^+$ (single-head gain)", "dminus": "$d^-$ (single-head loss)"}
HC = {}
for m in HM:
    M = HMOD[m]
    for s in ("a3", "fplus", "dminus", "rand0", "rand1", "rand2"):
        for arm in ("P1", "POST"):
            HC[m, arm, s] = (np.array([M.R(arm, s, i) for i in range(len(M.KS))]), np.array([M.KO(arm, s, i) for i in range(len(M.KS))]))
    # the plotted curves are the scorer's: every point equals the score file at its 2-decimal precision
    for (arm, s, w), pts in [((a, s_, w_), CURV[m, a, s_, w_]) for a in ("P1", "POST") for s_ in ("a3", "fplus", "dminus", "rand0", "rand1", "rand2") for w_ in ("R", "KO")]:
        got = HC[m, arm, s][0 if w == "R" else 1][:, 0]
        for k, v in pts.items():
            assert f"{got[M.KS.index(k)]:+.2f}" == f"{v:+.2f}" or abs(got[M.KS.index(k)] - v) < 0.0051, (m, arm, s, w, k, got[M.KS.index(k)], v)


def readers_figure():
    fig = plt.figure(figsize=(6.8, 2.35))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1, 1, 1.05], hspace=0.55, wspace=0.42, left=0.105, right=0.995, bottom=0.17, top=0.86)
    # (a) the v2 row splice: which later rows read the swapped key
    big_ = [m for m, _ in LOCM[1:]]
    for r_, (arm, title) in enumerate((("P1", "options-after"), ("POST", "sentence-after"))):
        ax = fig.add_subplot(gs[r_, 0])
        for k, (m, col) in enumerate(zip(big_, (BLUE, ORANGE, AQUA))):
            v = [locv[(m, arm, g)][0] for g, _ in GR]
            ax.barh(np.arange(len(GR)) + (k - 1) * 0.26, v, 0.24, color=col, label=SHORT[m], zorder=3)
        ax.set_yticks(range(len(GR)))
        ax.set_yticklabels(["mentions", "question", "story tail", "other later"], fontsize=5.6)
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=5.6)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.text(1.08, 3.45, title, ha="right", va="bottom", fontsize=5.8, color=INK2)
        if r_ == 0:
            ax.set_title("(a) rows that read the key", fontsize=7, color=INK, loc="left", x=-0.35)
            ax.legend(frameon=False, fontsize=6, loc="center right", handlelength=0.8, borderaxespad=0.1, labelspacing=0.15, bbox_to_anchor=(1.04, 0.47))
        else:
            ax.set_xlabel("fraction of the key effect", fontsize=5.8, labelpad=1)
    # (b) heads: R(k) and KO(k)
    for c_, m in enumerate(HM):
        ax = fig.add_subplot(gs[:, 1 + c_])
        M = HMOD[m]
        ks = np.array(M.KS)
        for s in ("rand0", "rand1", "rand2"):
            R_, K_ = HC[m, "P1", s]
            ax.plot(ks, R_[:, 0], color=GREY, lw=0.7, zorder=2, label="random sets" if s == "rand0" else None)
            ax.plot(ks, K_[:, 0], color=GREY, lw=0.7, ls="--", zorder=2)
        for s in ("dminus", "fplus", "a3"):
            R_, K_ = HC[m, "P1", s]
            if s == "a3":
                ax.fill_between(ks, R_[:, 1], R_[:, 2], color=RCOL[s], alpha=0.18, lw=0)
            ax.plot(ks, R_[:, 0], color=RCOL[s], lw=1.3 if s == "a3" else 0.9, marker="o" if s == "a3" else None, ms=2, zorder=4, label=RLAB[s])
            ax.plot(ks, K_[:, 0], color=RCOL[s], lw=1.3 if s == "a3" else 0.9, ls="--", zorder=4)
        ax.axvline(M.kstar, color=INK, lw=0.7, ls=":", zorder=1)
        ax.text(M.kstar * 0.92, 0.12, f"$k^*$ = {M.kstar}", fontsize=5.6, color=INK, ha="right")
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.05, 1.42)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(0.9, 140)
        ax.tick_params(labelsize=5.8)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("heads $k$ (log scale)", fontsize=6, labelpad=1)
        ax.set_title(("(b) heads: " if c_ == 0 else "") + SHORT[m], fontsize=7, color=INK, loc="left")
        if c_ == 0:
            ax.set_ylabel("fraction of the option-row read", fontsize=6, labelpad=1)
            h_, l_ = ax.get_legend_handles_labels()
            h_ += [matplotlib.lines.Line2D([], [], color=INK2, lw=0.9), matplotlib.lines.Line2D([], [], color=INK2, lw=0.9, ls="--")]
            l_ = [{"random sets": "random"}.get(x, x.split(" (")[0]) for x in l_] + ["$R(k)$", "KO$(k)$"]
            ax.legend(h_[::-1], l_[::-1], frameon=False, fontsize=6, loc="upper left", ncol=3, handlelength=1.3, borderaxespad=0.1,
                      labelspacing=0.15, columnspacing=0.5, bbox_to_anchor=(0.0, 1.0))
        else:
            ax.set_yticklabels([])
    # (c) the second hop
    ax = fig.add_subplot(gs[:, 3])
    w = 0.16
    for c_, m in enumerate(HM):
        H5_ = h6[m]["H5"]
        for j, (key, col, lab) in enumerate((("ans", INK, r"$r_{\mathrm{ans}}$: answer row"), ("other", LGREY, r"$r_{\mathrm{other}}$: other rows"),
                                              ("all", INK2, r"$r_{\mathrm{all}}$: every row"))):
            t_ = H5_[key]
            x_ = c_ + (j - 2) * w
            ax.bar(x_, fl(t_[0]), w * 0.92, color=col, zorder=3, label=lab if c_ == 0 else None,
                   yerr=[[fl(t_[0]) - fl(t_[1])], [fl(t_[2]) - fl(t_[0])]], error_kw=dict(lw=0.6, ecolor=INK))
        for j, (key, lab) in enumerate((("K", r"$r_{\mathrm{ans}}(K)$"), ("V", r"$r_{\mathrm{ans}}(V)$"))):
            x_ = c_ + (j + 1) * w
            ax.bar(x_, fl(H5_[key]), w * 0.92, color="white", edgecolor=INK, hatch="////" if key == "K" else "..", lw=0.6, zorder=3,
                   label=(lab + " (unscored)" if key == "K" else lab) if c_ == 0 else None)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(HM)))
    ax.set_xticklabels([SHORT[m] for m in HM], fontsize=6)
    ax.set_ylim(-0.05, 1.62)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=5.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylabel("fraction of the key-clamp effect removed", fontsize=5.8, labelpad=1)
    ax.set_title("(c) the answer's read", fontsize=7, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=1, handlelength=1.1, borderaxespad=0.1, labelspacing=0.15)
    fig.savefig(FIG / "fig_readers.pdf")
    plt.close(fig)


readers_figure()


def heads_appendix_figure():
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.6))
    for r_, m in enumerate(HM):
        M = HMOD[m]
        ks = np.array(M.KS)
        for c_, arm in enumerate(("P1", "POST")):
            ax = axes[r_, c_]
            for s in ("rand0", "rand1", "rand2"):
                R_, K_ = HC[m, arm, s]
                ax.plot(ks, R_[:, 0], color=GREY, lw=0.6, label="random" if s == "rand0" else None)
                ax.plot(ks, K_[:, 0], color=GREY, lw=0.6, ls="--")
            for s in ("dminus", "fplus", "a3"):
                R_, K_ = HC[m, arm, s]
                ax.plot(ks, R_[:, 0], color=RCOL[s], lw=1.1, label=RLAB[s])
                ax.plot(ks, K_[:, 0], color=RCOL[s], lw=1.1, ls="--")
            ax.axvline(M.kstar, color=INK, lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_xticks([1, 4, 16, 64])
            ax.set_xticklabels(["1", "4", "16", "64"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(color=GRID, lw=0.6)
            ax.set_title(f"{SHORT[m]}, {ARM_LABEL[arm].replace(chr(10), '')}", fontsize=7)
            ax.set_xlabel("heads $k$", fontsize=6.5)
            if c_ == 0:
                ax.set_ylabel("$R(k)$ solid, KO$(k)$ dashed", fontsize=6.5)
        ax = axes[r_, 2]
        for arm, col in (("P1", BLUE), ("POST", ORANGE)):
            prof = CURV[m, arm, "layer"]
            ax.bar(np.arange(len(prof)) + (0.2 if arm == "POST" else -0.2), prof, 0.4, color=col, label=ARM_LABEL[arm].replace("\n", ""))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("layer", fontsize=6.5)
        ax.set_ylabel("share of $d_G$, one layer", fontsize=6.5)
        ax.set_title(f"{SHORT[m]}, layer profile", fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.6)
        if r_ == 0:
            ax.legend(frameon=False, fontsize=5.5)
    axes[0, 0].legend(frameon=False, fontsize=5.0, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_heads_app.pdf")
    plt.close(fig)


heads_appendix_figure()


def sweeps_figure():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.4), sharex=True)
    FC = {"NO-MENTION": INK, "QNAMES": ORANGE, "OPTIONS-AFTER": BLUE, "LETTERS-AFTER": AQUA}
    for c_, arm in enumerate(("BIND", "ID")):
        for f, col in FC.items():
            if (arm, f) not in SW:
                continue
            ks = sorted(SW[arm, f])
            axes[0, c_].plot(ks, [SW[arm, f][k][0] for k in ks], color=col, lw=1.1, marker="o", ms=1.8, label=f.lower())
            axes[1, c_].plot(ks, [SW[arm, f][k][1] for k in ks], color=col, lw=1.1, marker="o", ms=1.8)
        for r_ in (0, 1):
            axes[r_, c_].axvline(28, color=INK2, lw=0.6, ls=":")
            axes[r_, c_].grid(color=GRID, lw=0.6)
        axes[0, c_].set_title(f"{arm} sweep, Qwen2.5-14B", fontsize=7.5)
        axes[1, c_].set_xlabel("patched block", fontsize=6.5)
    axes[0, 0].set_ylabel("IIA (answers flipped)", fontsize=6.5)
    axes[1, 0].set_ylabel(r"$\Phi$ (nats)", fontsize=6.5)
    axes[0, 1].legend(frameon=False, fontsize=5.8, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sweeps.pdf")
    plt.close(fig)


sweeps_figure()

# fig_frames panel (c) with Prakash et al.'s cells (H7: BIND@28 vs s_ID(f, 29); H11: ID@0 vs s_ID(f, 1))
PKPTS = []
for a_, l0, lab, mk, col in (("Bind", 29, "BIND@28, 14B (H7)", "D", PURPLE),
                             ("IdZero", 1, "ID@0, 14B (H11)", "P", PURPLE)):
    pts = []
    for f in ("NO-MENTION", "QNAMES", "OPTIONS-AFTER"):
        k_ = tuple(map(fl, PK[a_, PKF[f]]["kappa"]))
        s_ = tuple(map(fl, CL[f, l0]["sid"]))
        pts.append((s_, k_, f))
    PKPTS.append((lab, mk, col, pts))
draw_frames(pk=PKPTS)

# ================================================================ stage 7 (preregistration I): the reader heads at Mistral-Small-24B
# Parsed from STAGE7_SCORE.txt as stages 5 and 6 are (smac: printed string, verified against its line); arithmetic on
# printed values is registered with amac. Formats: Opt = OPTIONS-AFTER (P1), Letter = LETTERS-AFTER, Post = SENTENCE-AFTER.
G7I = "stage 7 (preregistration I): reader heads blocked under the released remap at Mistral-Small-24B"
S7F = ROOT / "results/gpu_stage7/STAGE7_SCORE.txt"
S7 = S7F.read_text().splitlines()
SCORE["S7"] = S7
SNAME["S7"] = "results/gpu_stage7/STAGE7_SCORE.txt"
F7 = {"Opt": "OPTIONS-AFTER", "Letter": "LETTERS-AFTER", "Post": "SENTENCE-AFTER"}
V7 = sec(S7, "VERDICTS (I1-I7")
R7 = sec(S7, "REPORTED (not scored")
X7 = sec(S7, "######## EXPLORATORY")
i7 = {}   # parsed printed strings for the tables
for ft, f in F7.items():
    i = find(S7, "  I-G2 ", f)
    d = grab(S7, i, r"d_full " + CIR)
    smac3(f"IgtwoDfull{ft}", "S7", i, d, f"I-G2 (a) mean d_full (natural clamp from layer 5), {f}", G7I)
    r_ = grab(S7, i, r"d_G/d_full " + CIR)
    smac3(f"IgtwoRatio{ft}", "S7", i, r_, f"I-G2 (b) d_G/d_full, {f}", G7I)
    ms = grab(S7, i, r"mention-specific d_G/\(d_full - d_full\(NONE\)\) " + CIR)
    smac(f"IgtwoRatioMs{ft}", "S7", i, ms[0], f"I-G2 (b) mention-specific d_G/(d_full - d_full(NONE)), printed, {f}", G7I)
    R_ = grab(S7, i, r"R\(64\) " + CIR)
    smac3(f"IgtwoR{ft}", "S7", i, R_, f"I-G2 (c) R(64): share of the natural key read at G recovered by H* alone, {f}", G7I)
    K_ = grab(S7, i, r"KO\(64\) " + CIR)
    smac3(f"IgtwoKO{ft}", "S7", i, K_, f"I-G2 (c) KO(64): share of the natural key read at G removed by blocking H*, {f}", G7I)
    rr = grab(S7, i, r"R_rand " + CIR)
    smac(f"IgtwoRrand{ft}", "S7", i, rr[0], f"I-G2 (c) R(64) of random sets (mean of three), {f}", G7I)
    kr = grab(S7, i, r"KO_rand " + CIR)
    smac(f"IgtwoKOrand{ft}", "S7", i, kr[0], f"I-G2 (c) KO(64) of random sets (mean of three), {f}", G7I)
    i7[ft, "G2"] = dict(d=d, ratio=r_, R=R_, KO=K_, Rr=rr, KOr=kr, v=S7[i].rsplit("-> ", 1)[1].strip())
    i = find(S7, "  I-G3 ", f)
    i7[ft, "D"] = grab(S7, i, r"\(1\) D " + NUM)[0]
    smac(f"IgthreeD{ft}", "S7", i, i7[ft, "D"], f"I-G3 (1) D: M - P effect in nats, unblocked, {f}", G7I)
i = find(S7, "  I-G3 ", "NO-MENTION")
smac("IgthreeDNone", "S7", i, grab(S7, i, r"\(1\) D " + NUM)[0], "I-G3 (1) D, NO-MENTION", G7I)
i = find(S7, "  I-G0 ")
i7["G0"] = grab(S7, i, r"(\d+) passed, (\d+) failed, (\d+) skipped")
smac("IgzeroPassed", "S7", i, i7["G0"][0], "I-G0: tests of tests/test_stage7_link.py passed (FP32, before any model)", G7I)
i = find(S7, "   OPTIONS-AFTER: H* = top-", start=X7)
smac("Ikstar", "S7", i, grab(S7, i, r"top-(\d+) by a3")[0], "k* = ceil(0.05 x 1280 heads) at Mistral-Small-24B", G7I)
i7["G1a"] = [find(S7, "  I-G1a ", f) for f in ("OPTIONS-AFTER", "LETTERS-AFTER", "SENTENCE-AFTER", "NO-MENTION", "LIST-BEFORE")]
_d1 = [abs(float(x)) for j in i7["G1a"] for x in re.findall(r"diff ([+-]\d+\.\d+)", S7[j])]
assert len(_d1) == 15, _d1
i7["G1a_max"] = max(_d1)
amac("IgoneaMaxDiff", i7["G1a_max"], "I-G1 (a) largest |difference| of phi, psi_K, psi_V from stage 3b over the five formats", SNAME["S7"], G7I, "{:.3f}")
i = find(S7, "  I-G4 ")
t4, dv4 = grab(S7, i, r"t " + CIR), grab(S7, i, r"dV " + CIR)
smac3("IgfourT", "S7", i, t4, "I-G4: t under A+(H*) (H* and layer 4 at G ablated), OPTIONS-AFTER", G7I)
smac3("IgfourDV", "S7", i, dv4, "I-G4: Delta_V under A+(H*), OPTIONS-AFTER", G7I)
i7["G4"] = (t4, dv4)
# verdicts
VI = {}
for p in range(1, 8):
    i = find(S7, f"  I{p}  ", start=V7, end=R7)
    VI[f"I{p}"] = S7[i].rsplit("-> ", 1)[1].strip()
for ft in ("Opt", "Letter"):
    f = F7[ft]
    i = find(S7, f"{f}: g_K", start=V7, end=R7)
    i7[ft, "I1"] = grab(S7, i, r"g_K " + CIR)
    smac3(f"IoneG{ft}", "S7", i, i7[ft, "I1"], f"I1 g_K: share of the remap's key-only effect read directly at the option rows, {f}", G7I)
    i = find(S7, f"{f}: KO_x(H*)", start=V7, end=R7)
    i7[ft, "I2"] = grab(S7, i, r"KO_x\(H\*\) " + CIR)
    smac3(f"ItwoKO{ft}", "S7", i, i7[ft, "I2"], f"I2 KO_x(H*): share of that read that passes through H*, {f}", G7I)
    i7[ft, "I2r"] = grab(S7, i, r"r_K\(H\*\) " + NUM)[0]
    smac(f"ItwoRK{ft}", "S7", i, i7[ft, "I2r"], f"I2 r_K(H*): key-only effect left with H* blinded, as a share of the unblinded, {f}", G7I)
    i = find(S7, f"{f}: KO_x rand0", start=V7, end=R7)
    i7[ft, "I3"] = [float(x) for x in re.findall(r"(?:rand\d|active) " + NUM, S7[i])]
    assert len(i7[ft, "I3"]) == 4, S7[i]
    i = find(S7, f"{f}: rand0: |d kappa|", start=V7, end=R7)
    i7[ft, "I7"] = [float(x) for x in re.findall(r"\| " + NUM, S7[i])]
    assert len(i7[ft, "I7"]) == 12, S7[i]
amac("IthreeMax", max(max(i7[ft, "I3"]) for ft in ("Opt", "Letter")), "I3: largest KO_x of the three random sets and the active set, both confirmatory formats", SNAME["S7"], G7I, "{:.3f}")
amac("IsevenMax", max(max(i7[ft, "I7"]) for ft in ("Opt", "Letter")), "I7: largest |d kappa|, |d psi~_V| or |t - 1| under A of the random and active sets, both formats", SNAME["S7"], G7I, "{:.3f}")
i = find(S7, "OPTIONS-AFTER: c_kappa", start=V7, end=R7)
i7["I4"] = grab(S7, i, r"c_kappa " + CIR)
smac3("IfourC", "S7", i, i7["I4"], "I4 c_kappa under A(H*): closure of the key share toward NO-MENTION, OPTIONS-AFTER", G7I)
i7["I4kA"] = grab(S7, i, r"kappa\^A " + CIR)
smac3("IfourKappaA", "S7", i, i7["I4kA"], "I4 kappa (key share) of the remap under A(H*), OPTIONS-AFTER", G7I)
smac("IfourKappaU", "S7", i, grab(S7, i, r"vs " + NUM)[0], "I4 kappa of the remap unblocked, OPTIONS-AFTER (in-run)", G7I)
smac("IfourKappaNone", "S7", i, grab(S7, i, r"NO-MENTION " + NUM)[0], "I4 kappa of the remap under NO-MENTION (in-run)", G7I)
i = find(S7, "LETTERS-AFTER: c~_K", start=V7, end=R7)
i7["I4L"] = grab(S7, i, r"c~_K " + CIR)
smac3("IfourCtLetter", "S7", i, i7["I4L"], "I4 c~_K under A(H*) (key closure on the unblocked scale), LETTERS-AFTER", G7I)
smac("IfourKappaALetter", "S7", i, grab(S7, i, r"kappa\^A " + CIR)[0], "I4 kappa under A(H*), LETTERS-AFTER (two-sided)", G7I)
i = find(S7, "OPTIONS-AFTER: dV", start=V7, end=R7)
i7["I5"] = grab(S7, i, r"dV " + CIR)
smac3("IfiveDV", "S7", i, i7["I5"], "I5 Delta_V under A(H*): value-only effect gained, nats, OPTIONS-AFTER", G7I)
smac("IfiveFloor", "S7", i, grab(S7, i, r"F_V " + NUM)[0], "I5 floor F_V = 0.25 (V(NONE) - V(OPTIONS-AFTER)), nats", G7I)
smac("IfiveRatio", "S7", i, grab(S7, i, r"dV / \(K - K\^A\) " + NUM)[0], "I5 Delta_V / lost key effect, OPTIONS-AFTER", G7I)
i = find(S7, "OPTIONS-AFTER: t ", start=V7, end=R7)
i7["I6"] = grab(S7, i, r"t " + CIR)
smac3("IsixT", "S7", i, i7["I6"], "I6 t = D under A(H*) / D unblocked, OPTIONS-AFTER", G7I)
rates = re.findall(r"M->T (\d\.\d\d)", S7[i])
smac("IsixMT", "S7", i, rates[0], "I6 argmax rate of M's target T, unblocked, OPTIONS-AFTER", G7I)
smac("IsixMTA", "S7", i, rates[1], "I6 argmax rate of M's target T under A(H*), OPTIONS-AFTER", G7I)
# reported lines
for ft, key, pat, nm_, desc in (("Post", "I1 SENTENCE-AFTER", r"g_K " + CIR, "IoneGPost", "I1 g_K, SENTENCE-AFTER (reported)"),
                                ("Post", "I2 SENTENCE-AFTER", r"KO_x\(H\*\) " + CIR, "ItwoKOPost", "I2 KO_x(H*), SENTENCE-AFTER (reported)"),
                                ("Post", "I6 SENTENCE-AFTER", r"t " + CIR, "IsixTPost", "I6 t under A(H*), SENTENCE-AFTER (reported)"),
                                ("Letter", "I6 LETTERS-AFTER", r"t " + CIR, "IsixTLetter", "I6 t under A(H*), LETTERS-AFTER (two-sided)"),
                                ("Letter", "I5 LETTERS-AFTER", r"dV " + CIR, "IfiveDVLetter", "I5 Delta_V under A(H*), LETTERS-AFTER (two-sided)")):
    i = find(S7, "  " + key, start=R7, end=X7)
    i7[nm_] = grab(S7, i, pat)
    smac3(nm_, "S7", i, i7[nm_], desc, G7I)
for ft, f in F7.items():
    i = find(S7, f"  N(H*) {f}: t", start=R7, end=X7)
    i7["N", ft] = grab(S7, i, r"t " + CIR)
    smac3(f"InT{ft}", "S7", i, i7["N", ft], f"N(H*) (knockout of H*'s edges from G to p, reported): t, {f}", G7I)
    i = find(S7, f"  A(H*) {f}: t", start=R7, end=X7)
    nats = grab(S7, i, r"nats D " + NUM + " K " + NUM + " V " + NUM + r" \(unblocked " + NUM + " / " + NUM + " / " + NUM + r"\)")
    i7["nats", ft] = nats
    for k_, s_ in zip(("DA", "KA", "VA", "DU", "KU", "VU"), nats):
        smac(f"Inats{k_}{ft}", "S7", i, s_, f"nats under A(H*) ({k_[0]}, {'ablated' if k_[1] == 'A' else 'unblocked'}), {f}", G7I)
# exploratory: the remap ranking and the cross-format overlap of H*
for ft, f in F7.items():
    i = find(S7, f"   {f}: |H_rem & H*| = ", start=X7)
    smac(f"IremOverlap{ft}", "S7", i, grab(S7, i, r"= (\d+) of 64")[0], f"exploratory: |H_rem & H*| (heads ranked on the remap itself vs H*), {f}", G7I)
i = find(S7, "overlap of H* across formats: OPTIONS-AFTER & LETTERS-AFTER", start=X7)
smac("IxOverlapOptLetter", "S7", i, grab(S7, i, r"= (\d+) of 64")[0], "exploratory: |H*(OPTIONS-AFTER) & H*(LETTERS-AFTER)|", G7I)
i = find(S7, "overlap of H* across formats: OPTIONS-AFTER & SENTENCE-AFTER", start=X7)
smac("IxOverlapOptPost", "S7", i, grab(S7, i, r"= (\d+) of 64")[0], "exploratory: |H*(OPTIONS-AFTER) & H*(SENTENCE-AFTER)|", G7I)
i = find(S7, "SUMMARY:")
i7["sum"] = grab(S7, i, r"(\d+) MET, (\d+) NOT MET, (\d+) NOT EVALUABLE of (\d+)")
smac("IcountMet", "S7", i, i7["sum"][0], "stage 7: predictions MET (of 7)", G7I)
smac("IcountNotMet", "S7", i, i7["sum"][1], "stage 7: predictions NOT MET (of 7)", G7I)
smac("IcountNotEval", "S7", i, i7["sum"][2], "stage 7: predictions NOT EVALUABLE (of 7)", G7I)
assert int(macros["GcountMet"]) + int(macros["GcountNotMet"]) == 22 and int(macros["HcountMet"]) + int(macros["HcountNotMet"]) + int(macros["HcountNotEval"]) + int(macros["HcountNotRun"]) == 12
amac("GHIcountPred", 22 + 12 + int(i7["sum"][3]), "predictions of G, H and I (22 + 12 + 7)", SNAME["S7"], G7I, "{:d}")
amac("GHIcountFailed", int(macros["GcountNotMet"]) + int(macros["HcountNotMet"]) + int(i7["sum"][1]),
     "failed predictions of G, H and I (NOT MET; H6 not evaluable and H12 not run are not counted)", SNAME["S7"], G7I, "{:d}")
i7["G1b"] = [float(x) for x in re.findall(r"D [\d.]+ vs [\d.]+ \(([+-]\d+\.\d) %", "\n".join(S7[j] for j in find(S7, "  I-G1b ", many=True)))]
assert len(i7["G1b"]) == 5, i7["G1b"]
amac("IgonebMaxPct", max(abs(x) for x in i7["G1b"]), "I-G1 (b) largest |difference| of the in-run D from the family's, per cent", SNAME["S7"], G7I, "{:.1f}")
assert VI == {"I1": "MET", "I2": "MET", "I3": "MET", "I4": "MET", "I5": "MET", "I6": "NOT MET", "I7": "MET"}, VI
# argmax rates under LETTERS-AFTER (two-sided lines), from the N(H*) block's rate line
i = find(S7, "  N(H*) LETTERS-AFTER: t", start=R7, end=X7)
rl = re.findall(r"(0|A:H|A\+:H|N:H): M->T (\d\.\d\d) P->S (\d\.\d\d) B->B (\d\.\d\d)", S7[i + 1])
rl = {k: v for k, *v in rl}
smac("IsixMTLetter", "S7", i + 1, rl["0"][0], "argmax rate of M's target, unblocked, LETTERS-AFTER", G7I)
smac("IsixMTALetter", "S7", i + 1, rl["A:H"][0], "argmax rate of M's target under A(H*), LETTERS-AFTER", G7I)
smac("IsixBBALetter", "S7", i + 1, rl["A:H"][2], "argmax rate of the natural answer under A(H*), LETTERS-AFTER", G7I)
# recomputed from the raw files (exploratory / post hoc): R(64) by story type; the non-additive part and row margins under A(H*)
G7J = json.loads((ROOT / "results/gpu_stage7/heads/gate.json").read_text())
for ft, arm in (("Opt", "P1"), ("Letter", "LETTER"), ("Post", "POST")):
    ev = G7J["arms"][arm]["eval"]
    dist = np.array([e["core"]["distractor_location"] in (e["core"]["base"], e["core"]["source"]) for e in ev])
    su = np.array([e["curves"]["H"]["suff"][G7J["provenance"]["KS"].index(64)] - e["curves"]["H"]["none"] for e in ev])
    al = np.array([e["curves"]["H"]["allG"] - e["curves"]["H"]["none"] for e in ev])
    assert abs(su.mean() / al.mean() - float(i7[ft, "G2"]["R"][0])) < 0.0006, (ft, su.mean() / al.mean())
    rmac(f"IrSixtyfourRest{ft}", su[~dist].mean() / al[~dist].mean(), f"R(64) on the E cores whose distractor is not B or S, {F7[ft]}", "results/gpu_stage7/heads/gate.json", G7I)
    rmac(f"IrSixtyfourDist{ft}", su[dist].mean() / al[dist].mean(), f"R(64) on the E cores whose distractor is B or S, {F7[ft]}", "results/gpu_stage7/heads/gate.json", G7I)
rmac("InDist", int(dist.sum()), "E cores whose distractor location is B or S", "results/gpu_stage7/heads/gate.json", G7I, "{:d}")
L7 = json.loads((ROOT / "results/gpu_stage7/link/mistral.json").read_text())["arms"]["P1"]


def m7(r, key, a="iT", b="iS"):
    c = r["runs"][key]["cand"]
    return c[r[a]] - c[r[b]]


SD7 = (101, 102, 103)
for cond, tg in (("0", "U"), ("A:H", "A")):
    D = np.array([np.mean([m7(r, f"{cond}/M_{s}") - m7(r, f"{cond}/P_{s}") for s in SD7]) for r in L7])
    K = np.array([np.mean([m7(r, f"{cond}/PK_{s}") - m7(r, f"{cond}/P_{s}") for s in SD7]) for r in L7])
    V = np.array([np.mean([m7(r, f"{cond}/PV_{s}") - m7(r, f"{cond}/P_{s}") for s in SD7]) for r in L7])
    assert abs(D.mean() - float(i7["nats", "Opt"][0 if tg == "A" else 3])) < 0.006, (cond, D.mean())
    rmac(f"InonAdd{tg}", (D - K - V).mean(), f"non-additive part D - K - V of M's effect, nats, OPTIONS-AFTER, {'A(H*)' if tg == 'A' else 'unblocked'}", "results/gpu_stage7/link/mistral.json", G7I, "{:.2f}")
    rmac(f"ImP{tg}", np.mean([np.mean([m7(r, f"{cond}/P_{s}") for s in SD7]) for r in L7]), f"mean margin log p(T) - log p(S) of the P rows, OPTIONS-AFTER, {cond}", "results/gpu_stage7/link/mistral.json", G7I, "{:.2f}")
    rmac(f"ImM{tg}", np.mean([np.mean([m7(r, f"{cond}/M_{s}") for s in SD7]) for r in L7]), f"mean margin log p(T) - log p(S) of the M rows, OPTIONS-AFTER, {cond}", "results/gpu_stage7/link/mistral.json", G7I, "{:.2f}")
    rmac(f"IbMargin{tg}", np.mean([m7(r, f"{cond}/B", "iB", "iS") for r in L7]), f"mean margin log p(B) - log p(S) of the natural B row, OPTIONS-AFTER, {cond}", "results/gpu_stage7/link/mistral.json", G7I, "{:.2f}")

# ---- tab_prereg_i.tex: preregistration I (stage 7), gates and every verdict as scored (OPTIONS-AFTER / LETTERS-AFTER)
OL = ("Opt", "Letter")


def _gv(p):
    ls = [S7[j] for j in find(S7, p, many=True)]
    assert ls, p
    return "MET" if all(l.rstrip().endswith("-> MET") for l in ls) else "NOT MET"


GV = {g: _gv("  " + g + " ") for g in ("I-G0", "I-G2")}
_g3 = [S7[j] for j in find(S7, "  I-G3 ", many=True)]   # its lines end ': MET' per clause, without '->'
assert len(_g3) == 4 and all(l.rstrip().endswith(": MET") or l.rstrip().endswith(": NOT MET") for l in _g3), _g3
GV["I-G3"] = "NOT MET" if any("NOT MET" in l for l in _g3) else "MET"
GV["I-G1"] = "MET" if all(_gv(f"  I-G1{x} ") == "MET" for x in "abcd") else "NOT MET"
_g4 = S7[find(S7, "  I-G4 ")].rstrip()
assert _g4.endswith("takeover: no (not a verdict gate)") or _g4.endswith("takeover: yes (not a verdict gate)"), _g4
GV["I-G4"] = "NOT MET" if _g4.endswith("takeover: no (not a verdict gate)") else "MET"
_sc = grab(S7, find(S7, "  I-G1d "), r"x_all\| (\d+(?:\.\d+)?), max \|N\(H\*_OPTIONS-AFTER\) - unblocked\| (\d+(?:\.\d+)?) nats")
ir = []
ir.append(("I-G0", r"FP32 exactness of the blinding, ablation and knockout code (unit tests), before any model",
           f"{i7['G0'][0]} passed, {i7['G0'][1]} failed, {i7['G0'][2]} skipped", GV["I-G0"]))
ir.append(("I-G1", r"Reproduction of stage 3b ($|\Delta\varphi|$, $|\Delta\psi_K|$, $|\Delta\psi_V| \leq 0.03$, five formats); in-run rows; knockout floor; no effect at rows before the writing token",
           f"largest difference {M_('IgoneaMaxDiff')}; in-run $D$ within {M_('IgonebMaxPct')}\\,\\%; structural check {max(_sc, key=float)} nats", GV["I-G1"]))
ir.append(("I-G2", r"The readers at 24B: $\bar d_{\mathrm{full}} \geq 3$; $d_G/d_{\mathrm{full}} \geq 0.7$; $R(64) \geq 0.7$, $\mathrm{KO}(64) \geq 0.8$; random sets $\leq 0.25$; BF16 floor",
           f"$\\bar d_{{\\mathrm{{full}}}}$ {slash(*(tx(i7[f, 'G2']['d'][0]) for f in OL))}; $d_G/d_{{\\mathrm{{full}}}}$ {slash(*(tx(i7[f, 'G2']['ratio'][0]) for f in OL))}; $R$ {slash(*(tx(i7[f, 'G2']['R'][0]) for f in OL))}; "
           f"KO {slash(*(tx(i7[f, 'G2']['KO'][0]) for f in OL))}; random $R$ {slash(*(tx(i7[f, 'G2']['Rr'][0]) for f in OL))}, KO {slash(*(tx(i7[f, 'G2']['KOr'][0]) for f in OL))}", GV["I-G2"]))
ir.append(("I-G3", r"The remap acts: $D \geq 3$ nats; direct read at $G$ $\geq 1$ nat", f"$D$ {slash(*(tx(i7[f, 'D']) for f in OL))}", GV["I-G3"]))
ir.append(("I-G4", r"For reading I5/I6 as a value takeover (not a verdict gate): under A$^+$, $t \geq 0.6$ and $\Delta_V > 0$ (CI excluding 0)",
           f"$t$ {tci(i7['G4'][0])}; $\\Delta_V$ {tci(i7['G4'][1], True)}", GV["I-G4"]))
ir.append(("I1", r"The remap's key is read at the option words: $g_K \geq 0.7$, lower bound $\geq 0.6$", slash(*(tci(i7[f, 'I1']) for f in OL)), VI["I1"]))
ir.append(("I2", r"Through the natural readers: $\mathrm{KO}_x(H^*) \geq 0.8$, lower bound $\geq 0.7$", slash(*(tci(i7[f, 'I2']) for f in OL)), VI["I2"]))
ir.append(("I3", r"Specificity: $\mathrm{KO}_x \leq 0.25$ for three random sets and the active set", f"at most {M_('IthreeMax')}", VI["I3"]))
ir.append(("I4", r"Under A($H^*$) the key share falls toward \fmtNone{}: $c_\kappa$ (\fmtOpt{}), $\tilde c_K$ (\fmtLetter{}) $\geq 0.5$, lower bound $\geq 0.3$",
           f"{tci(i7['I4'])} ($\\kappa$ {tx(M_('IfourKappaU'))} $\\to$ {tx(M_('IfourKappaA'))}; \\fmtNone{{}} {tx(M_('IfourKappaNone'))}) / {tci(i7['I4L'])}", VI["I4"]))
ir.append(("I5", r"The value channel carries more (\fmtOpt{}): $\Delta_V > 0$ (CI excluding 0) and $\geq F_V$",
           f"{tci(i7['I5'], True)} nats, $F_V$ = {M_('IfiveFloor')}; {M_('IfiveRatio')} of the lost key effect", VI["I5"]))
ir.append(("I6", r"The behaviour is kept (\fmtOpt{}): $t \geq 0.75$, lower bound $\geq 0.6$",
           f"{tci(i7['I6'])}; M's target is the argmax in {M_('IsixMTA')} of (core, fit) items ({M_('IsixMT')} unblocked)", VI["I6"]))
ir.append(("I7", r"Random and active sets change nothing: $|\Delta\kappa|$, $|\Delta\tilde\psi_V|$, $|t-1| \leq 0.10$", f"at most {M_('IsevenMax')}", VI["I7"]))
lines = [r"\begin{tabular}{p{0.05\textwidth}p{0.36\textwidth}p{0.41\textwidth}p{0.08\textwidth}}", r"\toprule", r" & Gate or prediction & Observed (\fmtOpt{} / \fmtLetter{}) & Verdict \\", r"\midrule"]
for gid, pr, ob, v in ir:
    if gid == "I1":
        lines.append(r"\midrule")
    lines.append(f"{gid} & {pr} & {ob} & {verd(v)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_prereg_i.tex", lines)

# ---- tab_link.tex: the remap under each blocking, per format (appendix app:stage7); every cell a printed score-file string
def cond7(line):
    d = {"t": grab_s(line, r"t " + CIR), "pk": grab_s(line, r"psi~_K " + NUM)[0], "pv": grab_s(line, r"psi~_V " + NUM)[0],
         "k": grab_s(line, r"kappa " + CIR), "ku": grab_s(line, r"\(unblocked " + NUM + r"\)")[0],
         "dv": grab_s(line, r"dV " + CIR), "drop": re.search(r"kappa [^;]*?\((\d+\.\d) % of resamples dropped\)", line),
         "nats": grab_s(line, r"nats D " + NUM + " K " + NUM + " V " + NUM + r" \(unblocked " + NUM + " / " + NUM + " / " + NUM + r"\)")}
    return d


def grab_s(line, pat):
    m_ = re.search(pat, line)
    assert m_, (pat, line[:160])
    return m_.groups()


TL = []
for ft, f in F7.items():
    i = find(S7, f"  I-G1b {f}")
    psu = grab_s(S7[i], r"unblocked batch psi_K " + NUM + r" vs [^,]+, psi_V " + NUM)
    i = find(S7, f"  N(H*) {f}: t", start=R7, end=X7)
    am = dict(re.findall(r"(0|A:H|A\+:H|N:H): M->T (\d\.\d\d)", S7[i + 1]))
    assert set(am) == {"0", "A:H", "A+:H", "N:H"}, S7[i + 1][:200]
    rows = []
    a_ = cond7(S7[find(S7, f"  A(H*) {f}: t", start=R7, end=X7)])
    rows.append(("unblocked", ("1", None, None), (a_["ku"], None, None), psu[0], psu[1], None, a_["nats"][3:], am["0"]))
    for lab, key, amk in ((r"A($H^*$)", f"  A(H*) {f}: t", "A:H"), (r"A$^+$($H^*$)", f"  A+(H*) {f}: t", "A+:H"), (r"N($H^*$)", f"  N(H*) {f}: t", "N:H")):
        c = cond7(S7[find(S7, key, start=R7, end=X7)])
        assert not c["drop"] or float(c["drop"].group(1)) <= 5, (lab, f)
        rows.append((lab, c["t"], c["k"], c["pk"], c["pv"], c["dv"], c["nats"][:3], am[amk]))
    j = find(S7, f"   {f}:", start=find(S7, "-- exploratory conditions (step 2)"), end=find(S7, "-- remap ranking"))
    for lab, key in ((r"N(all at $G$)$^\dagger$", "N:allG"), (r"A(block 5)$^\dagger$", "A:L4"), (r"N(rand0)$^\dagger$", "N:rand0"), (r"A($H^*_{\text{\fmtOpt{}}}$)$^\dagger$", "A:H_P1")):
        hits = [x for x in range(j + 1, j + 12) if S7[x].strip().startswith(key + " ")]
        if not hits:
            assert key == "A:H_P1" and ft != "Letter", (key, f)
            continue
        c = cond7(S7[hits[0]])
        kk = ("nan", None, None) if c["drop"] and float(c["drop"].group(1)) > 5 else (c["k"][0], None, None)
        rows.append((lab, c["t"], kk, c["pk"], c["pv"], c["dv"], c["nats"][:3], None))
    TL.append((ft, rows))
FTEX = {"Opt": r"\fmtOpt{}", "Letter": r"\fmtLetter{}", "Post": r"\fmtPost{}"}
lines = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r"Format & Condition & $t$ & $\kappa$ & $\tilde\psi_K$ & $\tilde\psi_V$ & $\Delta_V$ (nats) & $D$ / $K$ / $V$ (nats) & M$\to$T \\", r"\midrule"]
for ft, rows in TL:
    for n_, (lab, t, k, pk, pv, dv, nats, mt) in enumerate(rows):
        tt = tx(t[0]) if t[1] is None else tci(t)
        kk = tx(k[0]) if k[1] is None else tci(k)
        lines.append(f"{FTEX[ft] if n_ == 0 else ''} & {lab} & {tt} & {kk} & {tx(pk)} & {tx(pv)} & {'--' if dv is None else tci(dv, True)} & "
                     f"{' / '.join(tx(x) for x in nats)} & {'--' if mt is None else mt} \\\\")
    lines.append(r"\midrule" if ft != "Post" else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_link.tex", lines)


# ================================================================ stage 8 (preregistration J, paper v5): parts A, B, C and D
# Parsed from STAGE8A_SCORE.txt, STAGE8B_SCORE.txt, STAGE8C_SCORE.txt and STAGE8D_SCORE.txt as stages 5-7 are: smac/smac3 take a printed
# string at its printed precision and assert that it is on the cited line; a range over models or cells (...Min, ...Max)
# selects printed strings (nm); counts and sums of printed values are registered with amac; values the scorers do not
# print are recomputed from the raw results through the committed scorers' own classes (rmac: label them exploratory or
# post hoc), and where docs/PREREGISTRATION.md records them the recomputation reproduces the record (asserted below).
# Part B's cell table results/gpu_stage8b/tab_invariance.tex is the part-B scorer's own output, written in the same run
# as its score file, and is cited the same way (tag S8BINV). Part D's pytest session of record (results/gpu_stage8d/logs/
# pytest.log) prints the J-D-G0 total of the outcome record, which the score line does not (tag S8DPY).
# Groups: S8A (part A, natural passages), S8B (part B, fresh samples and the emitted-form score), S8C (part C, independently
# obtained identity edits and the channel-ratio law; its own conventions are at the head of its section), S8D (part D, the
# flag), S8X (counts across parts).
# Names: J + part (Ja, Jb, Jc, Jd, Jx) + statistic + format or arm + model tag; an estimate with an interval also has ...Lo and
# ...Hi. Model tags: LlamaEight, GemmaNine, GemmaTwo, QwenOnefive, QwenThree, QwenSeven, QwenFourteen, MistralSeven,
# MistralTwentyfour, OlmoSeven, PhiFour, FalconSeven. Part A formats: Opta, Leta, Mena, Nom, Optb, Menb. Part B arms:
# After, Post, PostNull, None, Pre, Before (Opt = P1 on S0). Part D cells: Qin, Qout, Inline, InlineChat, After.
PRE8 = dict(macros)   # every macro defined before stage 8 (asserted unchanged at the end)
N_PRE8 = len(REG)
G8A, G8B, G8D, G8X = "S8A", "S8B", "S8D", "S8X"
for _tag, _f in (("S8A", "results/gpu_stage8a/STAGE8A_SCORE.txt"), ("S8B", "results/gpu_stage8b/STAGE8B_SCORE.txt"),
                 ("S8D", "results/gpu_stage8d/STAGE8D_SCORE.txt"), ("S8BINV", "results/gpu_stage8b/tab_invariance.tex")):
    SCORE[_tag] = (ROOT / _f).read_text().splitlines()
    SNAME[_tag] = _f
S8A, S8B, S8D = SCORE["S8A"], SCORE["S8B"], SCORE["S8D"]
K8 = {"llama8": "LlamaEight", "gemma9": "GemmaNine", "gemma2b": "GemmaTwo", "qwen1.5": "QwenOnefive", "qwen3b": "QwenThree",
      "qwen7": "QwenSeven", "qwen14": "QwenFourteen", "mistral7": "MistralSeven", "mistral24": "MistralTwentyfour",
      "olmo7": "OlmoSeven", "phi4": "PhiFour", "falcon7": "FalconSeven"}
SH8 = {"llama8": "Llama-3.1-8B", "gemma9": "Gemma-2-9B", "gemma2b": "Gemma-2-2B", "qwen1.5": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B",
       "qwen7": "Qwen2.5-7B", "qwen14": "Qwen2.5-14B", "mistral7": "Mistral-7B", "mistral24": "Mistral-Small-24B",
       "olmo7": "OLMo-2-7B", "phi4": "Phi-4", "falcon7": "Falcon3-7B"}
FAM8 = {"llama8": "Llama", "gemma9": "Gemma", "gemma2b": "Gemma", "qwen1.5": "Qwen", "qwen3b": "Qwen", "qwen7": "Qwen",
        "qwen14": "Qwen", "mistral7": "Mistral", "mistral24": "Mistral", "olmo7": "OLMo", "phi4": "Phi", "falcon7": "Falcon"}
VRD = r"(MET|NOT MET|NOT EVALUABLE)"
CLS8 = (r"(\d+) lines: (\d+) MET, (\d+) NOT MET, (\d+) MET IN PART, (\d+) NOT EVALUABLE; observed (\d+) against expected "
        r"(\d+\.\d+); Brier (\d\.\d+|nan)")   # nan: a class with no line (part C, class L)


def sm8(name, tag, i, pat, desc, group, g=0):
    """one printed string (group g of pat on line i of score file tag) -> macro; returns the string."""
    s = grab(SCORE[tag], i, pat)[g]
    smac(name, tag, i, s, desc, group)
    return s


def sm83(name, tag, i, pat, desc, group, off=0):
    """estimate and interval (groups off..off+2 of pat on line i) -> name, nameLo, nameHi; returns the strings."""
    t_ = grab(SCORE[tag], i, pat)[off:off + 3]
    smac3(name, tag, i, t_, desc, group)
    return t_


def rng8(name, strs, desc, src, group):
    """the smallest and the largest of printed strings (by value) -> nameMin, nameMax, each a printed string."""
    v = sorted(strs, key=fl)
    nm(name + "Min", sv(v[0]), desc + " (smallest)", src, group)
    nm(name + "Max", sv(v[-1]), desc + " (largest)", src, group)


def rmac3(name, e3, desc, src, group, fmt="{:.3f}"):
    """recomputed estimate and 95 % interval -> name, nameLo, nameHi."""
    for suf, v, d in (("", e3[0], ""), ("Lo", e3[1], " (lower 95 % bound)"), ("Hi", e3[2], " (upper 95 % bound)")):
        rmac(name + suf, v, desc + d, src, group, fmt)


def classes8(L, a, b, pre, group, kinds=("L", "M", "R")):
    """the SUMMARY class lines: lines, MET, NOT MET, NOT EVALUABLE, observed against expected, Brier."""
    for c in kinds:
        lab = f"  account lines, class {c}: " if c != "Val" else "  measurement-validity lines, class R: "
        i = find(L, lab, start=a, end=b)
        tag = next(t_ for t_ in ("S8A", "S8B", "S8C", "S8D") if SCORE.get(t_) is L)
        what = "measurement-validity lines (class R)" if c == "Val" else f"account lines of class {c}"
        for j, w in enumerate(("Lines", "Met", "NotMet", "MetInPart", "NotEval", "Observed", "Exp", "Brier")):
            if w in ("MetInPart", "Observed") or (w == "Brier" and grab(L, i, CLS8)[j] == "nan"):
                continue
            sm8(f"{pre}Class{c}{w}", tag, i, CLS8, f"{what}: {dict(Lines='lines', Met='MET', NotMet='NOT MET', NotEval='NOT EVALUABLE', Exp='expected MET (sum of the recorded priors of the lines with a verdict)', Brier='Brier score')[w]}", group, j)


# ---------------------------------------------------------------- stage 8, part A: the format law on natural passages
MA8 = ["llama8", "gemma9", "qwen7", "mistral7"]
FA8 = {"OPTA": "Opta", "LETA": "Leta", "MENA": "Mena", "NOM": "Nom", "OPTB": "Optb", "MENB": "Menb"}   # figure order
FLAB8 = {"OPTA": "options after", "LETA": "letters after", "MENA": "sentence after", "NOM": "free form",
         "OPTB": "options before", "MENB": "sentence before"}
LWA = {"J-A1": "One", "J-A1b": "Oneb", "J-A1c": "Onec", "J-A2": "Two", "J-A3": "Three", "J-A4": "Four", "J-A5": "Five",
       "J-A5B": "FiveB", "J-A6a": "Sixa", "J-A6b": "Sixb", "J-A6c": "Sixc", "J-A6d": "Sixd", "J-A6e": "Sixe", "J-A7": "Seven",
       "J-A7b": "Sevenb", "J-A8": "Eight", "J-A8d": "Eightd", "J-A-HA1": "HaOne", "J-A-HA2": "HaTwo", "J-A-HA3a": "HaThreea",
       "J-A-HA3b": "HaThreeb"}
GA8, PA8, RA8, SA8, XA8 = (sec(S8A, s) for s in ("GATES (J-A-G0", "PREDICTIONS (code", "REPORTED (no verdicts)", "SUMMARY (G4",
                                                  "######## EXPLORATORY"))
SRCA = SNAME["S8A"]
# gates per model and format; NE8A: the cells a gate makes not evaluable
NE8A = {}
_lp, _p3, _g13 = [], [], []
for k in MA8:
    t = K8[k]
    for f, fw in FA8.items():
        for g in ("G1", "G2", "G3"):
            i = find(S8A, f"  {k:8s} {f:4s} J-A-{g} ", start=GA8, end=PA8)
            if not S8A[i].rstrip().endswith("-> MET"):
                NE8A.setdefault((k, f), []).append(g)
            if g == "G1":
                _lp += list(grab(S8A, i, r"lp\(S run\)\| (\d\.\d+), mean max\|lp\(ID\) - lp\(B run\)\| (\d\.\d+)"))
                sm8(f"JaGoneAgree{fw}{t}", "S8A", i, r"in (\d\.\d+) \(>= 0\.97\)",
                    f"J-A-G1: share of items whose KV_S generation equals the S run's (>= 0.97 required), {f}, {SH8[k]}", G8A)
            elif g == "G2":
                sm8(f"JaNcomp{fw}{t}", "S8A", i, r"n_comp (\d+)",
                    f"J-A-G2 n_comp: items competent in {f} (ID answers B, KV_S answers S, KV_X answers X), {SH8[k]}", G8A)
            else:
                _p3.append(grab(S8A, i, r"median p\(dec_B \| ID\) (\d\.\d+)")[0])
                _g13 += list(grab(S8A, i, r"g1 = dec in ID (\d\.\d+), KV_S (\d\.\d+), KV_X (\d\.\d+)"))
assert NE8A == {("llama8", "OPTA"): ["G1"], ("llama8", "MENA"): ["G2"], ("llama8", "MENB"): ["G2"], ("mistral7", "NOM"): ["G2"],
                ("mistral7", "MENA"): ["G2"], ("mistral7", "MENB"): ["G2"]}, NE8A
rng8("JaGoneLp", _lp, "J-A-G1 mean largest log-probability differences (KV_S vs the S run, ID vs the B run) over the 24 model x format cells, nats",
     SRCA + " GATES", G8A)
rng8("JaGthreeP", _p3, "J-A-G3 median p(dec_B | ID) over the 24 cells", SRCA + " GATES", G8A)
rng8("JaGthreeGone", _g13, "J-A-G3 share of competent items whose first generated token is the decision token (ID, KV_S, KV_X rows) over the 24 cells",
     SRCA + " GATES", G8A)
for k in ("qwen7", "mistral7"):
    t = K8[k]
    i = find(S8A, f"  {k:8s} J-A-HA-G1 ", start=GA8, end=PA8)
    for j, (w, d) in enumerate((("NoneClean", "mean |m(none) - m(clean)|"), ("AllFull", "mean |m(all_T) - m(full clamp)|"),
                                ("Bound", "bound max(0.5, 0.02 d_full)"))):
        sm8(f"JaHaGone{w}{t}", "S8A", i, r"\|m\(none\) - m\(clean\)\| (\d\.\d+), mean \|m\(all_T\) - m\(full clamp\)\| (\d\.\d+) "
            r"\(<= max\(0\.5, 0\.02 x d_full\) = (\d\.\d+)\)", f"J-A-HA-G1 heads BF16 floor: {d} (nats), {SH8[k]}", G8A, j)
    i = find(S8A, f"  {k:8s} J-A-HA-G2 ", start=GA8, end=PA8)
    sm83(f"JaHaGtwoDfull{t}", "S8A", i, r"d_full " + CIR, f"J-A-HA-G2 d_full: the key clamp's effect read at the option rows (nats), OPTA, {SH8[k]}", G8A)
    sm83(f"JaHaGtwoRatio{t}", "S8A", i, r"d_G/d_full " + CIR, f"J-A-HA-G2 d_G/d_full, OPTA, {SH8[k]}", G8A)
# the lines: header (code, class, prior, title, verdict, models met / evaluable / run) and one result line per model
LA8 = {}
for i in range(PA8 + 1, RA8):
    mm = re.match(r"  (J-A[\w-]+)\s+\[(\w), prior (\d\.\d\d)\] (.*?) -> " + VRD +
                  r"(?: \(derived; not counted\))? \((\d+) of (\d+) evaluable models met; (\d+) run\)$", S8A[i])
    if mm:
        cur = mm[1]
        LA8[cur] = dict(i=i, cls=mm[2], prior=mm[3], title=mm[4], v=mm[5], met=mm[6], ev=mm[7], run=mm[8], m={})
        continue
    mm = re.match(r"      (\w+): (.*) -> " + VRD + r"(?:  \[not evaluable: (.*)\])?$", S8A[i])
    if mm:
        LA8[cur]["m"][mm[1]] = dict(i=i, v=mm[3], why=mm[4] or "")
assert list(LA8) == list(LWA), list(LA8)
assert all(set(L["m"]) == set(MA8) for c, L in LA8.items() if not c.startswith("J-A-HA")), "part A: a model line is missing"
for code, L in LA8.items():
    smac(f"Ja{LWA[code]}Met", "S8A", L["i"], L["met"], f"{code}: evaluable models that meet the line", G8A)
    smac(f"Ja{LWA[code]}Eval", "S8A", L["i"], L["ev"], f"{code}: evaluable models (of {L['run']} run)", G8A)


def ia8(code, k):
    return LA8[code]["m"][k]["i"]


A8P = {}   # (model, format) -> printed (s_ID, ID_K, ID_V) triples on C(f), for the figure and the table
for k in MA8:
    t, S = K8[k], SH8[k]
    for code, f in (("J-A1", "OPTA"), ("J-A1c", "LETA"), ("J-A2", "NOM")):
        i, fw = ia8(code, k), FA8[f]
        for st, lab in (("Sid", "s_ID"), ("IDK", "ID_K"), ("IDV", "ID_V")):
            A8P[k, f, st] = sm83(f"Ja{st}{fw}{t}", "S8A", i, lab + " " + CIR,
                                 f"{lab}{'' if st == 'Sid' else ' (nats)'} on C({f}), the items competent in {FLAB8[f]} ({f}); line {code}, {S}", G8A)
        sm8(f"JaNart{fw}{t}", "S8A", i, r"\((\d+) articles\)", f"articles (clusters) of C({f}), {S}", G8A)
        assert grab(S8A, i, r"n=(\d+)")[0] == macros[f"JaNcomp{fw}{t}"], (k, f)
    i = ia8("J-A1b", k)
    sm83(f"JaOnebSid{t}", "S8A", i, r"s_ID " + CIR, f"J-A1b s_ID on the OPTA items whose answer has |P| >= 3 tokens, {S}", G8A)
    sm8(f"JaOnebN{t}", "S8A", i, r"n=(\d+)", f"J-A1b items with |P| >= 3, {S}", G8A)
    i = ia8("J-A3", k)
    sm83(f"JaThreeDiff{t}", "S8A", i, r"s_ID\(OPTA\) - s_ID\(NOM\) " + CIR, f"J-A3 (derived) s_ID(OPTA) - s_ID(NOM) on C(OPTA, NOM), {S}", G8A)
    sm8(f"JaThreeN{t}", "S8A", i, r"n=(\d+)", f"J-A3 items competent in OPTA and NOM, {S}", G8A)
    i = ia8("J-A4", k)
    sm83(f"JaFourRatio{t}", "S8A", i, r"ratio " + CIR, f"J-A4 ID_K(OPTB) / ID_K(OPTA) on C(OPTA, OPTB), {S}", G8A)
    sm83(f"JaFourDiff{t}", "S8A", i, r"ID_K\(OPTA\) - ID_K\(OPTB\) " + CIR, f"J-A4 ID_K(OPTA) - ID_K(OPTB) (nats), {S}", G8A)
    sm83(f"JaFourIDKOptb{t}", "S8A", i, r"; ID_K\(OPTB\) " + CIR, f"J-A4 ID_K(OPTB) on C(OPTA, OPTB) (nats), {S}", G8A)
    sm8(f"JaFourN{t}", "S8A", i, r"n=(\d+)", f"J-A4 items competent in OPTA and OPTB, {S}", G8A)
    i = ia8("J-A5", k)
    sm83(f"JaFiveDiff{t}", "S8A", i, r"s_ID\(MENA\) - s_ID\(NOM\) " + CIR, f"J-A5 s_ID(MENA) - s_ID(NOM) on C(MENA, NOM), {S}", G8A)
    sm83(f"JaFiveIDKMena{t}", "S8A", i, r"; ID_K\(MENA\) " + CIR, f"J-A5 ID_K(MENA) on C(MENA, NOM) (nats), {S}", G8A)
    sm83(f"JaFiveSidMena{t}", "S8A", i, r"; s_ID\(MENA\) " + CIR, f"J-A5 s_ID(MENA) on C(MENA, NOM), {S}", G8A)
    sm8(f"JaFiveN{t}", "S8A", i, r"n=(\d+)", f"J-A5 items competent in MENA and NOM, {S}", G8A)
    i = ia8("J-A5B", k)
    if "not both run" not in S8A[i]:
        sm83(f"JaFiveBDiff{t}", "S8A", i, r"ID_K\(MENA\) - ID_K\(MENB\) " + CIR, f"J-A5B ID_K(MENA) - ID_K(MENB) on C(MENA, MENB) (nats), {S}", G8A)
        sm83(f"JaFiveBIDKMenb{t}", "S8A", i, r"; ID_K\(MENB\) " + CIR, f"J-A5B ID_K(MENB) on C(MENA, MENB) (nats), {S}", G8A)
        sm8(f"JaFiveBN{t}", "S8A", i, r"n=(\d+)", f"J-A5B items competent in MENA and MENB, {S}", G8A)
    for code, f in (("J-A6a", "OPTA"), ("J-A6b", "LETA"), ("J-A6c", "NOM")):
        i, fw = ia8(code, k), FA8[f]
        cc = f"{code} cue conflict (key from one counterfactual entity, value from the other; KS_VX and KX_VS rows), {f}, {S}"
        A8P[k, f, "key"] = sm83(f"JaKeySrc{fw}{t}", "S8A", i, r"key-source " + CIR, "rate of answering the key source's entity, " + cc, G8A)
        A8P[k, f, "val"] = sm83(f"JaValSrc{fw}{t}", "S8A", i, r"value-source " + CIR, "rate of answering the value source's entity, " + cc, G8A)
        sm83(f"JaSrcDiff{fw}{t}", "S8A", i, r"difference " + CIR,
             ("value-source minus key-source rate, " if f == "NOM" else "key-source minus value-source rate, ") + cc, G8A)
        sm8(f"JaOtherB{fw}{t}", "S8A", i, r"other answers B (\d\.\d+)", "rate of answering B (the passage's own entity), " + cc, G8A)
        sm8(f"JaOther{fw}{t}", "S8A", i, r", other (\d\.\d+) ->", "rate of answers naming none of B, S, X, Z, D ('other'), " + cc, G8A)
    i = ia8("J-A6d", k)
    sm83(f"JaSixdOpta{t}", "S8A", i, r"OPTA n=\d+ S-rate " + CIR, f"J-A6d flag only (K_S, V_Z): rate of answering S, OPTA, {S}", G8A)
    sm8(f"JaSixdBOpta{t}", "S8A", i, r"\(B-rate (\d\.\d+)\)", f"J-A6d flag only: rate of answering B, OPTA, {S}", G8A)
    sm83(f"JaSixdNom{t}", "S8A", i, r"NOM n=\d+ S-rate " + CIR, f"J-A6d flag only (K_S, V_Z): rate of answering S, NOM, {S}", G8A)
    sm8(f"JaSixdZNom{t}", "S8A", i, r"\(Z-rate (\d\.\d+)\)", f"J-A6d flag only: rate of answering Z (the value's entity), NOM, {S}", G8A)
    i = ia8("J-A6e", k)
    sm83(f"JaSixeS{t}", "S8A", i, r"S-rate " + CIR, f"J-A6e copy fallback (K_Z, V_S): rate of answering S, OPTA, {S}", G8A)
    sm8(f"JaSixeB{t}", "S8A", i, r"; B (\d\.\d+),", f"J-A6e copy fallback: rate of answering B, OPTA, {S}", G8A)
    i = ia8("J-A7", k)
    sm8(f"JaSevenN{t}", "S8A", i, r"prior-free n=(\d+)", f"J-A7 prior-free items (closed-book argmax not B), {S}", G8A)
    sm83(f"JaSevenDiD{t}", "S8A", i, r"drop_OPTA\(K\)\] " + CIR,
         f"J-A7 KIVI 2-bit DiD [drop_NOM(V) - drop_OPTA(V)] - [drop_NOM(K) - drop_OPTA(K)] (faithful accuracy), {S}", G8A)
    for w, p in (("NomK", r"NOM\(K\)"), ("NomV", r"NOM\(V\)"), ("OptaK", r"OPTA\(K\)"), ("OptaV", r"OPTA\(V\)")):
        sm8(f"JaSevenDrop{w}{t}", "S8A", i, r"drops .*?" + p + " " + NUM, f"J-A7 accuracy drop under 2-bit {w[-1]} quantization, {w[:-1].upper()}, {S}", G8A)
    i = ia8("J-A7b", k)
    sm83(f"JaSevenbDiff{t}", "S8A", i, r"drop_NOM\(V\) - drop_OPTA\(V\) " + CIR, f"J-A7b drop_NOM(V) - drop_OPTA(V), {S}", G8A)
    i = ia8("J-A8", k)
    sm8(f"JaEightN{t}", "S8A", i, r"n=(\d+)", f"J-A8 items (leak-free PERSON/PLACE with a continuation token differing from c_B), {S}", G8A)
    A8P[k, "Icont"] = sm83(f"JaIcont{t}", "S8A", i, r"I_cont\(NOM\) " + CIR, f"J-A8 I_cont(NOM): K-V interaction on the continuation, as a share of d_KV^cont, {S}", G8A)
    A8P[k, "Idec"] = sm83(f"JaIdec{t}", "S8A", i, r"I_dec\(NOM\) " + CIR, f"J-A8 I_dec(NOM): K-V interaction at the decision token, {S}", G8A)
    sm83(f"JaContRatio{t}", "S8A", i, r"d_KV\^cont OPTA/NOM " + CIR, f"J-A8 d_KV^cont(OPTA) / d_KV^cont(NOM), {S}", G8A)
    for j, w in enumerate(("K", "V", "KV")):
        sm8(f"JaDcont{w}{t}", "S8A", i, r"d\^cont\(NOM\) K " + NUM + " V " + NUM + " KV " + NUM, f"J-A8 d_{w}^cont(NOM): effect on the continuation (nats), {S}", G8A, j)
    i = ia8("J-A8d", k)
    sm8(f"JaEightdNNom{t}", "S8A", i, r"NOM (\d+), OPTA (\d+)", f"J-A8d items with g1(K_Z) = dec_B, NOM (>= 40 needed), {S}", G8A, 0)
    sm8(f"JaEightdNOpta{t}", "S8A", i, r"NOM (\d+), OPTA (\d+)", f"J-A8d items with g1(K_Z) = dec_B, OPTA, {S}", G8A, 1)
    sm83(f"JaEightdNom{t}", "S8A", i, r"h\(K_Z, NOM\) " + CIR, f"J-A8d hybrid rate h(K_Z, NOM), {S}", G8A)
    sm8(f"JaEightdOpta{t}", "S8A", i, r"h\(K_Z, OPTA\) " + NUM, f"J-A8d hybrid rate h(K_Z, OPTA), {S}", G8A)
for k in ("qwen7", "mistral7"):
    t, S = K8[k], SH8[k]
    i = ia8("J-A-HA1", k)
    sm8(f"JaHaKstar{t}", "S8A", i, r"k\*=(\d+)", f"k* of the natural reader set N* (5 % of the heads), {S}", G8A)
    for w, p in (("R", "R_N "), ("KO", "KO_N "), ("Rrand", "R_rand "), ("KOrand", "KO_rand ")):
        sm83(f"JaHaOne{w}{t}", "S8A", i, p + CIR, f"J-A-HA1 {p.strip()}(k*) of the natural readers N* (rand: mean of 3 draws){' (not evaluable: J-A-HA-G1)' if k == 'qwen7' else ''}, {S}", G8A)
    i = ia8("J-A-HA2", k)
    sm83(f"JaHaTwoR{t}", "S8A", i, r"R_T " + CIR, f"J-A-HA2 R_T(k*): the template readers T* on natural text, {S}", G8A)
    sm83(f"JaHaTwoKO{t}", "S8A", i, r"KO_T " + CIR, f"J-A-HA2 KO_T(k*), {S}", G8A)
    ov, P_ = grab(S8A, i, r"\|T\* & N\*\| = (\d+) \(hypergeometric P (\S+)\)")
    smac(f"JaHaTwoOverlap{t}", "S8A", i, ov, f"J-A-HA2 |T* & N*| (template and natural reader sets), {S}", G8A)
    mant, ex_ = P_.split("e")
    nm(f"JaHaTwoP{t}", f"{mant}\\times 10^{{{int(ex_)}}}", f"J-A-HA2 hypergeometric P of that overlap (math mode; printed {P_}), {S}", f"{SRCA}:{i + 1}", G8A)
    i = ia8("J-A-HA3a", k)
    sm8(f"JaHaThreeaN{t}", "S8A", i, r"OPTA n=(\d+)", f"J-A-HA3a prior-free competent OPTA items, {S}", G8A)
    sm8(f"JaHaThreeaAcc{t}", "S8A", i, r"acc\(none\) (\d\.\d+)", f"J-A-HA3a faithful accuracy without ablation, {S}", G8A)
    sm83(f"JaHaThreeaDrop{t}", "S8A", i, r"drop\(N\*\) " + CIR, f"J-A-HA3a accuracy drop when N* is ablated at the options, {S}", G8A)
    for j, w in enumerate(("RandZero", "RandOne", "RandTwo", "Tstar")):
        sm8(f"JaHaThreea{w}{t}", "S8A", i, r"rand0 " + NUM + ", rand1 " + NUM + ", rand2 " + NUM + r"; drop\(T\*\) " + NUM,
            f"J-A-HA3a accuracy drop when {('random set 0', 'random set 1', 'random set 2', 'T*')[j]} is ablated, {S}", G8A, j)
    i = ia8("J-A-HA3b", k)
    sm83(f"JaHaThreebDec{t}", "S8A", i, r"decision accuracy change " + CIR, f"J-A-HA3b NOM decision-token accuracy change under N* ablation, {S}", G8A)
    sm83(f"JaHaThreebCont{t}", "S8A", i, r"continuation accuracy change " + CIR, f"J-A-HA3b NOM continuation accuracy change under N* ablation, {S}", G8A)
# reported (no verdicts): closed-book prior, accuracies, decision-token cue conflict, all valid items, KIVI, d_K^cont
for k in MA8:
    t, S = K8[k], SH8[k]
    h = find(S8A, f"  [{k}] n valid ", start=RA8, end=SA8)
    e = next(j for j in range(h + 1, SA8) if S8A[j].startswith("  ["))
    for cb, cw in (("CBOPT", "Opt"), ("CBLET", "Let")):
        i = find(S8A, f"    P_CB(B) {cb}:", start=h, end=e)
        sm8(f"JaPrior{cw}{t}", "S8A", i, cb + r": (\d\.\d+) of", f"closed-book prior: share of items whose argmax over the options is B ({cb}), {S}", G8A)
        for ty in ("PERSON", "PLACE", "NUMBER"):
            sm8(f"JaPrior{cw}{ty.capitalize()}{t}", "S8A", i, ty + r" (\d\.\d+) \(n", f"closed-book prior ({cb}), {ty} items, {S}", G8A)
    for f in ("NOM", "OPTA", "LETA"):
        fw = FA8[f]
        i = find(S8A, f"    {f} competent n=", start=h, end=e)
        for w, p in (("VZ", r"under V_Z "), ("KZ", r"; K_Z "), ("KVZ", r"KV_Z ")):
            sm83(f"JaAcc{w}{fw}{t}", "S8A", i, p + CIR, f"accuracy for B with {w[:-1]}_Z (the {w[:-1]} of a third entity Z) on C({f}), {S}", G8A)
        i = find(S8A, f"    {f} cue conflict at the decision token", start=h, end=e)
        A8P[k, f, "keyDec"] = sm83(f"JaKeySrcDec{fw}{t}", "S8A", i, r"key-source " + CIR, f"cue conflict at the decision token (g1): key-source rate, {f}, {S}", G8A)
        A8P[k, f, "valDec"] = sm83(f"JaValSrcDec{fw}{t}", "S8A", i, r"value-source " + CIR, f"cue conflict at the decision token (g1): value-source rate, {f}, {S}", G8A)
        i = find(S8A, f"    KIVI {f} competent n=", start=h, end=e)
        for j, w in enumerate(("ErrK", "ErrV", "None", "Ktwo", "Vtwo")):
            sm8(f"JaKivi{w}{fw}{t}", "S8A", i, r"relative error K (\d\.\d+), V (\d\.\d+); faithful \(S\) accuracy none (\d\.\d+), K2 (\d\.\d+), V2 (\d\.\d+)",
                f"KIVI on C({f}): {('relative reconstruction error of the keys (per channel)', 'relative reconstruction error of the values (per token)', 'faithful accuracy, no quantization', 'faithful accuracy, 2-bit keys', 'faithful accuracy, 2-bit values')[j]}, {S}", G8A, j)
    for f, fw in FA8.items():
        i = find(S8A, f"    {f} all valid items n=", start=h, end=e)
        sm83(f"JaSidAll{fw}{t}", "S8A", i, r"s_ID " + CIR, f"s_ID on all valid items (185), {f}, {S}", G8A)
        sm8(f"JaMassAll{fw}{t}", "S8A", i, r"option mass \(ID\) (\d\.\d+)", f"mass of the four options at the decision token (ID row), all valid items, {f}, {S}", G8A)
    i = find(S8A, "    d_K^cont(NOM) on the J-A8 population", start=h, end=e)
    sm83(f"JaDKcont{t}", "S8A", i, r"population " + CIR, f"d_K^cont(NOM): the key alone's effect on the continuation, J-A8 population (nats), {S}", G8A)
for k in ("qwen7", "mistral7"):
    t, S = K8[k], SH8[k]
    h = find(S8A, f"  [{k}] k*=", start=RA8, end=SA8)
    for f in ("OPTA", "NOM"):
        i = find(S8A, f"    {f} every ablated item", start=h, end=h + 4)
        cond = dict((c, a) for c, a in re.findall(r"(\w+): faithful\(S\) (\d\.\d+), clean-B chain", S8A[i]))
        assert set(cond) == {"none", "N", "T", "rand0", "rand1", "rand2", "C"}, cond
        for c, w in (("none", "None"), ("N", "Nstar"), ("T", "Tstar"), ("rand0", "RandZero"), ("rand1", "RandOne"), ("rand2", "RandTwo"), ("C", "Cstar")):
            smac(f"JaAblFaith{w}{FA8[f]}{t}", "S8A", i, cond[c],
                 f"faithful accuracy on every item with heads {c} ablated (N*: natural readers, T*: template readers, C*: DLA copy set), {f}, {S}", G8A)
# summary: classes, the R tally (S8X), the overall count, Holm
classes8(S8A, SA8, XA8, "Ja", G8A)
i = find(S8A, "  met rate among R account lines with a verdict:", start=SA8, end=XA8)
for j, (w, d) in enumerate((("Met", "met"), ("Verd", "with a verdict"), ("Exp", "expected met"))):
    sm8(f"JxRisky{w}A", "S8A", i, r"(\d+) of (\d+) \(expected (\d+\.\d+)\)", f"part A: risky (class R) account lines {d}", G8X, j)
i = find(S8A, "  overall: ", start=SA8, end=XA8)
for j, w in enumerate(("Met", "NotMet", "NotEval", "Lines")):
    sm8(f"JaCount{w}", "S8A", i, r"(\d+) MET, (\d+) NOT MET, (\d+) NOT EVALUABLE of (\d+) counted lines",
        f"part A: lines {('MET', 'NOT MET', 'NOT EVALUABLE', 'counted (J-A3 derived, not counted)')[j]}", G8A, j)
i = find(S8A, "    J-A5: 1 component", start=SA8, end=XA8)
sm8("JaHolmP", "S8A", i, r"\(p (\S+), Holm threshold (\S+);", "Holm sensitivity: one-sided p of J-A5's Qwen2.5-7B component s_ID(MENA) - s_ID(NOM) > 0", G8A, 0)
sm8("JaHolmThr", "S8A", i, r"\(p (\S+), Holm threshold (\S+);", "Holm sensitivity: its Holm threshold (the one decision that changes; no verdict changes)", G8A, 1)
# exploratory (not scored): onset and pieces (E1/E2), YEAR (E4), the LEAK stratum, full strings (E6), heads
for k in MA8:
    t, S = K8[k], SH8[k]
    h = find(S8A, f"  [{k}] explore parts:", start=XA8)
    e = next(j for j in range(h + 1, len(S8A)) if S8A[j].startswith("  [") or S8A[j].startswith("  heads ["))
    i = find(S8A, "    E1/E2 OPTA", start=h, end=e)
    for j, w in enumerate(("DKOpta", "OnOpta", "FirstOpta", "RestOpta")):
        sm8(f"JaEone{w}{t}", "S8A", i, r"d_K = m\(K_S\) - m\(ID\) " + NUM + r" nats; fractions K_S@on " + NUM + r", K_S\^1 " + NUM + r", K_S\^r " + NUM,
            f"E1/E2 OPTA (exploratory): {('d_K = m(K_S) - m(ID) (nats)', 'fraction of d_K kept with the clamp from 30 % depth', 'fraction carried by the first span piece', 'fraction carried by the later span pieces')[j]}, {S}", G8A, j)
    i = find(S8A, "    E1/E2 NOM", start=h, end=e)
    for j, w in enumerate(("DVNom", "OnNom", "FirstNom", "RestNom")):
        sm8(f"JaEone{w}{t}", "S8A", i, r"d_V = m\(V_S\) - m\(ID\) " + NUM + r" nats; fractions V_S@on " + NUM + r", V_S\^1 " + NUM + r", V_S\^r " + NUM,
            f"E1/E2 NOM (exploratory): {('d_V = m(V_S) - m(ID) (nats)', 'fraction of d_V kept with the clamp from 30 % depth', 'fraction carried by the first span piece', 'fraction carried by the later span pieces')[j]}, {S}", G8A, j)
    for f in ("NOM", "OPTA"):
        fw = FA8[f]
        i = find(S8A, f"    E4 YEAR {f} n=", start=h, end=e)
        for j, w in enumerate(("DK", "DV", "DKV", "Idec")):
            sm8(f"JaEfour{w}{fw}{t}", "S8A", i, r"d_K " + NUM + ", d_V " + NUM + ", d_KV " + NUM + ", I_dec " + NUM,
                f"E4 YEAR items (exploratory), at the first differing digit: {('d_K (nats)', 'd_V (nats)', 'd_KV (nats)', 'I_dec')[j]}, {f}, {S}", G8A, j)
        i = find(S8A, f"    LEAK {f} competent", start=h, end=e)
        sm8(f"JaLeakSid{fw}{t}", "S8A", i, r"s_ID " + NUM, f"LEAK stratum (partial re-mention of e_B in the passage; exploratory): s_ID, {f}, {S}", G8A)
        sm8(f"JaLeakN{fw}{t}", "S8A", i, r"competent n=(\d+) of", f"LEAK stratum: competent items, {f}, {S}", G8A)
        i = find(S8A, f"    E6 {f} full string", start=h, end=e)
        for j, w in enumerate(("IDK", "IDV", "Sid")):
            sm8(f"JaEsix{w}{fw}{t}", "S8A", i, r"ID_K " + NUM + ", ID_V " + NUM + r", s_ID\^full " + NUM,
                f"E6 full-string score (exploratory): {('ID_K (nats)', 'ID_V (nats)', 's_ID^full')[j]}, {f}, {S}", G8A, j)
for k in ("qwen7", "mistral7"):
    t, S = K8[k], SH8[k]
    h = find(S8A, f"  heads [{k}]", start=XA8)
    e = next((j for j in range(h + 1, len(S8A)) if not S8A[j].startswith("    ")), len(S8A))
    i = find(S8A, "    ID_K / ID_V with N* ablated at Q+", start=h, end=e)
    sm8(f"JaHeadsRhoK{t}", "S8A", i, r"\(rho_K " + NUM, f"heads (exploratory): rho_K with N* ablated at the options, OPTA, {S}", G8A)
    sm8(f"JaHeadsDV{t}", "S8A", i, r"ID_V change " + NUM, f"heads (exploratory): change of ID_V with N* ablated (nats), OPTA, {S}", G8A)
i = find(S8A, "  s_ID(OPTA) per model:", start=XA8)
sm8("JaSidOptaMean", "S8A", i, r"unweighted mean " + NUM, "unweighted mean of s_ID(OPTA) over the four models (gated cells included)", G8A)
# ranges over models that the outcome record cites
_tA = lambda base, fs=("",), ms=MA8: [macros[f"{base}{f}{K8[k]}"] for k in ms for f in fs]  # noqa: E731
rng8("JaFourIDKOptb", _tA("JaFourIDKOptb"), "J-A4 ID_K(OPTB) over the four models (nats)", SRCA + " J-A4", G8A)
rng8("JaFourRatio", _tA("JaFourRatio"), "J-A4 ID_K(OPTB)/ID_K(OPTA) over the four models", SRCA + " J-A4", G8A)
rng8("JaIcont", _tA("JaIcont"), "J-A8 I_cont(NOM) over the four models", SRCA + " J-A8", G8A)
rng8("JaIdec", _tA("JaIdec"), "J-A8 I_dec(NOM) over the four models", SRCA + " J-A8", G8A)
rng8("JaContRatio", _tA("JaContRatio"), "J-A8 continuation ratio OPTA/NOM over the four models", SRCA + " J-A8", G8A)
rng8("JaDKcont", _tA("JaDKcont"), "d_K^cont(NOM) over the four models (nats)", SRCA + " REPORTED", G8A)
rng8("JaSidAllBefore", _tA("JaSidAll", ("Optb", "Menb")), "s_ID on all valid items, OPTB and MENB, over the four models", SRCA + " REPORTED", G8A)
rng8("JaSidAllMena", _tA("JaSidAll", ("Mena",)), "s_ID on all valid items, MENA, over the four models", SRCA + " REPORTED", G8A)
rng8("JaMassAllList", _tA("JaMassAll", ("Opta", "Optb", "Leta")), "option mass with the options listed (OPTA, OPTB, LETA), all valid items", SRCA + " REPORTED", G8A)
rng8("JaMassAllNoList", _tA("JaMassAll", ("Nom", "Mena", "Menb")), "option mass without the options listed (NOM, MENA, MENB), all valid items", SRCA + " REPORTED", G8A)
for ty in ("Person", "Place", "Number"):
    rng8(f"JaPriorOpt{ty}", _tA(f"JaPriorOpt{ty}"), f"closed-book prior (CBOPT), {ty.upper()} items, over the four models", SRCA + " REPORTED", G8A)
rng8("JaPriorOpt", _tA("JaPriorOpt"), "closed-book prior (CBOPT), all items, over the four models", SRCA + " REPORTED", G8A)
rng8("JaKiviErrK", _tA("JaKiviErrK", ("Nom", "Opta", "Leta")), "KIVI relative error of the keys over models and formats", SRCA + " REPORTED", G8A)
rng8("JaKiviErrV", _tA("JaKiviErrV", ("Nom", "Opta", "Leta")), "KIVI relative error of the values over models and formats", SRCA + " REPORTED", G8A)
rng8("JaKiviMcq", _tA("JaKiviKtwo", ("Opta", "Leta")) + _tA("JaKiviVtwo", ("Opta", "Leta")), "KIVI faithful MCQ accuracy (OPTA, LETA) under 2-bit keys or values", SRCA + " REPORTED", G8A)
EONE8 = {"OnNom": "fraction of d_V = m(V_S) - m(ID) kept with the clamp from 30 % depth, NOM", "FirstNom": "fraction of d_V carried by the first span piece, NOM",
         "OnOpta": "fraction of d_K = m(K_S) - m(ID) kept with the clamp from 30 % depth, OPTA", "FirstOpta": "fraction of d_K carried by the first span piece, OPTA",
         "RestOpta": "fraction of d_K carried by the later span pieces, OPTA"}
for w, d in EONE8.items():
    rng8(f"JaEone{w}", _tA(f"JaEone{w}"), f"E1/E2 (exploratory): {d}, over the four models", SRCA + " EXPLORATORY", G8A)
EFOUR8 = {"DKNom": "d_K (nats), NOM", "DVNom": "d_V (nats), NOM", "IdecNom": "I_dec, NOM", "DKOpta": "d_K (nats), OPTA", "IdecOpta": "I_dec, OPTA"}
for w, d in EFOUR8.items():
    rng8(f"JaEfour{w}", _tA(f"JaEfour{w}"), f"E4 YEAR items (exploratory), at the first differing digit: {d}, over the four models", SRCA + " EXPLORATORY", G8A)
for w in ("Nom", "Opta"):
    rng8(f"JaLeakSid{w}", _tA(f"JaLeakSid{w}"), f"LEAK stratum s_ID, {w.upper()}, over the four models", SRCA + " EXPLORATORY", G8A)
rng8("JaEsixSidNom", _tA("JaEsixSidNom"), "E6 s_ID^full(NOM) over the four models", SRCA + " EXPLORATORY", G8A)
rng8("JaDcontK", _tA("JaDcontK"), "J-A8 d_K^cont(NOM) over the four models (nats)", SRCA + " J-A8", G8A)
rng8("JaValSrcDecNom", _tA("JaValSrcDecNom"), "NOM cue conflict at the decision token, value-source rate, over the four models", SRCA + " REPORTED", G8A)
# recomputed (not printed): s_ID, ID_K and ID_V on C(f) for MENA, OPTB and MENB, through the committed part-A scorer's own
# Model (Model.C, Model.idkv) and its two-stage article bootstrap (common.est, seed 20261010); the same code reproduces
# the printed OPTA, LETA and NOM values to the last printed digit (asserted).
from stage8a_parts import common as CA8  # noqa: E402
from stage8a_parts import factorial as FA8M  # noqa: E402
for k in MA8:
    t, S = K8[k], SH8[k]
    srcf = f"results/gpu_stage8a/factorial/{k}.json"
    M8 = FA8M.Model(k, json.load(open(ROOT / srcf)), None, False, True)
    for f, fw in FA8.items():
        ids = M8.C(f)
        assert len(ids) == int(macros[f"JaNcomp{fw}{t}"]), (k, f)
        kk, vv = M8.idkv(f, ids)
        arts = M8.arts(ids)
        for st, e8 in (("Sid", CA8.est(arts, CA8.sid, kk, vv)), ("IDK", CA8.est(arts, CA8.mean, kk)), ("IDV", CA8.est(arts, CA8.mean, vv))):
            if f in ("OPTA", "LETA", "NOM"):
                assert tuple(f"{x:+.3f}" for x in (e8.pt, e8.lo, e8.hi)) == A8P[k, f, st], (k, f, st, e8, A8P[k, f, st])
            else:
                lab = {"Sid": "s_ID", "IDK": "ID_K (nats)", "IDV": "ID_V (nats)"}[st]
                rmac3(f"Ja{st}{fw}{t}", (e8.pt, e8.lo, e8.hi),
                      f"{lab} on C({f}), the items competent in {FLAB8[f]} ({f}); scorer's Model.C/idkv and two-stage article bootstrap, {S}", srcf, G8A)
                A8P[k, f, st] = tuple(f"{x:+.3f}" for x in (e8.pt, e8.lo, e8.hi))

# ---------------------------------------------------------------- stage 8, part B: fresh samples and the emitted-form score
MB8P = ["qwen7", "qwen14", "mistral7", "olmo7"]
MB8N = ["llama8", "gemma9", "phi4", "falcon7"]
MB8 = MB8P + MB8N + ["gemma2b"]
AB8 = {"AFTER": "After", "POST": "Post", "POST-NULL": "PostNull", "NONE": "None", "PRE": "Pre", "BEFORE": "Before"}   # figure order
AS08 = {"P1": "Opt", "AFTER": "After", "BEFORE": "Before", "POST": "Post", "PRE": "Pre", "NONE": "None"}
ALAB8 = {"AFTER": "list after", "POST": "sentence after", "POST-NULL": "null sentence after", "NONE": "no mention",
         "PRE": "sentence before", "BEFORE": "list before", "P1": "options after"}
LWB = {"J-B1-P4f": "OnePf", "J-B1-N4": "OneNf", "J-B2-P4f": "TwoPf", "J-B2-N4": "TwoNf", "J-B3-P4f": "ThreePf", "J-B3-N4": "ThreeNf",
       "J-B4-P4f": "FourPf", "J-B4-N4": "FourNf", "J-B5-P4f": "FivePf", "J-B5-N4": "FiveNf", "J-B5b-P4f": "FivebPf", "J-B5b-N4": "FivebNf",
       "J-B-NULL-P4f": "NullPf", "J-B-NULL-N4": "NullNf", "J-B-LB-N4": "LbNf", "J-B-SMALL": "Small", "J-B6a": "Sixa", "J-B6b": "Sixb",
       "J-B6c": "Sixc", "J-B7": "Seven", "J-B8": "Eight"}
GB8, PB8, RB8, SB8, XB8 = (sec(S8B, s) for s in ("GATES (J-B-G0", "PREDICTIONS (code", "REPORTED (no verdicts)", "SUMMARY (G4",
                                                  "######## EXPLORATORY"))
SRCB = SNAME["S8B"]
# REPORTED: per model and arm on F, r, s_ID, beta_K, beta_V with 95 % cluster intervals, ID_K^E, ID_V^E and the flip rate
B8 = {}
_pb = (r"^    (\S+)\s+(AFTER|BEFORE|NONE|POST-NULL|POST|PRE)\s+\[(E|L)\] r " + CIR + r"  s_ID " + CIR + r"  beta_K " + CIR +
       r"  beta_V " + CIR + r"  ID_K\^E " + NUM + r"  ID_V\^E " + NUM + r"  flip (\d\.\d+)$")
for i in range(RB8, SB8):
    mm = re.match(_pb, S8B[i])
    if mm:
        g_ = mm.groups()
        B8[mm[1], mm[2]] = dict(i=i, sc=mm[3], R=g_[3:6], Sid=g_[6:9], BetaK=g_[9:12], BetaV=g_[12:15], IDK=g_[15], IDV=g_[16], Flip=g_[17])
assert len(B8) == 54 and all(d["sc"] == "E" for d in B8.values()), (len(B8), {d["sc"] for d in B8.values()})
for k in MB8:
    t, S = K8[k], SH8[k]
    ne = " (not evaluable: J-B-G5 competence)" if k == "gemma2b" else ""
    for arm, aw in AB8.items():
        d = B8[k, arm]
        for st, lab in (("R", "r = ID_K^E(arm)/ID_K^E(AFTER)"), ("Sid", "s_ID^E"), ("BetaK", "beta_K: share of generated answers flipped by the key-only clamp"),
                        ("BetaV", "beta_V: share flipped by the value-only clamp")):
            smac3(f"Jb{st}{aw}{t}", "S8B", d["i"], d[st], f"{lab}, {ALAB8[arm]} ({arm}), F, 95 % cluster interval, {S}{ne}", G8B)
        smac(f"JbIDK{aw}{t}", "S8B", d["i"], d["IDK"], f"ID_K^E (nats), {ALAB8[arm]} ({arm}), F, {S}{ne}", G8B)
        smac(f"JbIDV{aw}{t}", "S8B", d["i"], d["IDV"], f"ID_V^E (nats), {ALAB8[arm]} ({arm}), F, {S}{ne}", G8B)
        smac(f"JbFlip{aw}{t}", "S8B", d["i"], d["Flip"], f"flip rate of the key-only clamp, {ALAB8[arm]} ({arm}), F, {S}{ne}", G8B)
# the scorer's fig_crossover.csv carries the same beta values (to the printed precision)
import csv  # noqa: E402
for row in csv.DictReader(open(ROOT / "results/gpu_stage8b/fig_crossover.csv")):
    d = B8[row["model"], row["arm"]]
    for st, c in (("BetaK", "beta_K"), ("BetaV", "beta_V")):
        for v, s in zip((row[c], row[c + "_lo"], row[c + "_hi"]), d[st]):
            assert abs(float(v) - float(s)) <= 0.0005 + 1e-9, (row, st, d[st])
# ranges over models of every reported statistic, per arm (P4, N4, all eight P4 and N4 models; Gemma-2-2B excluded)
STB8 = {"R": "r^E = ID_K^E(arm)/ID_K^E(AFTER)", "Sid": "s_ID^E", "BetaK": "beta_K (share of generated answers flipped by the key-only clamp)",
        "BetaV": "beta_V (share flipped by the value-only clamp)", "Flip": "flip rate of the key-only clamp", "IDK": "ID_K^E (nats)", "IDV": "ID_V^E (nats)"}
GRB8 = {"Pfour": "four P4 models (Qwen2.5-7B, Qwen2.5-14B, Mistral-7B, OLMo-2-7B)", "Nfour": "four N4 models (Llama-3.1-8B, Gemma-2-9B, Phi-4, Falcon3-7B)",
        "Eight": "eight P4 and N4 models"}
for arm, aw in AB8.items():
    for st in ("R", "Sid", "BetaK", "BetaV", "Flip", "IDK", "IDV"):
        if st == "R" and arm == "AFTER":
            continue
        for grp, ms in (("Pfour", MB8P), ("Nfour", MB8N), ("Eight", MB8P + MB8N)):
            vals = [B8[k, arm][st][0] if isinstance(B8[k, arm][st], tuple) else B8[k, arm][st] for k in ms]
            rng8(f"JbRange{st}{aw}{grp}", vals, f"{STB8[st]}, {ALAB8[arm]} ({arm}), F, emitted-form scoring: point estimates over the {GRB8[grp]}",
                 SRCB + " REPORTED", G8B)
# the lines: header and per-model blocks with their component lines
LB8 = {}
for i in range(PB8 + 1, RB8):
    mm = re.match(r"  (J-B[\w-]+)\s+\[(\w), (\w), prior (\d\.\d\d)\] (.*?) -> " + VRD + r" \((\d+) of (\d+) evaluable met; (\d+) slots\)$", S8B[i])
    if mm:
        cur = mm[1]
        LB8[cur] = dict(i=i, cls=mm[2], kind=mm[3], prior=mm[4], title=mm[5], v=mm[6], met=mm[7], ev=mm[8], run=mm[9], m={})
        continue
    mm = re.match(r"      (?:P4 |N4 )?([\w.]+): " + VRD + r"\s*(.*)$", S8B[i])
    if mm:
        last = mm[1]
        LB8[cur]["m"][last] = dict(i=i, v=mm[2], txt=mm[3], comps=[])
    elif S8B[i].startswith("          "):
        LB8[cur]["m"][last]["comps"].append(i)
assert list(LB8) == list(LWB), list(LB8)
for code, L in LB8.items():
    smac(f"Jb{LWB[code]}Met", "S8B", L["i"], L["met"], f"{code}: evaluable models (slots) that meet the line", G8B)
    smac(f"Jb{LWB[code]}Eval", "S8B", L["i"], L["ev"], f"{code}: evaluable models (of {L['run']} slots)", G8B)


def cb8(code, k, sub):
    """the unique component line of model k on line code that contains sub."""
    hits = [j for j in LB8[code]["m"][k]["comps"] if sub in S8B[j]]
    assert len(hits) == 1, (code, k, sub, hits)
    return hits[0]


BND = r"(?:lower|upper) bound ([+-]\d+\.\d+)"
CBP = {}   # (name stem, model) -> printed bound, for ranges
for code, ms, lvl in (("J-B1-P4f", MB8P, "95 %"), ("J-B1-N4", MB8N, "98.75 %"), ("J-B2-P4f", MB8P, "95 %"), ("J-B2-N4", MB8N, "98.75 %"),
                      ("J-B3-P4f", MB8P, "95 %"), ("J-B3-N4", MB8N, "98.75 %"), ("J-B4-P4f", MB8P, "95 %"), ("J-B4-N4", MB8N, "98.75 %"),
                      ("J-B5-P4f", MB8P, "95 %"), ("J-B5-N4", MB8N, "98.75 %"), ("J-B5b-P4f", MB8P, "95 %"), ("J-B5b-N4", MB8N, "98.75 %"),
                      ("J-B-NULL-P4f", MB8P, "95 %"), ("J-B-NULL-N4", MB8N, "98.75 %"), ("J-B-LB-N4", MB8N, "98.75 %")):
    stem = {"J-B1": "One", "J-B2": "Two", "J-B3": "Three", "J-B4": "Four", "J-B5": "Five", "J-B5b": "Fiveb", "J-B-NULL": "Null", "J-B-LB": "Lb"}[code.rsplit("-", 1)[0]]
    comps = {"One": (("SidLb", "H0 s_ID^E(AFTER) <= 0.4", "lower bound of s_ID^E(AFTER)"), ("IDKLb", "H0 ID_K^E(AFTER) <= 0", "lower bound of ID_K^E(AFTER) (nats)")),
             "Two": (("RBeforeUb", "H0 r^E(BEFORE) - r^E(NONE)", "upper bound of r^E(BEFORE) - r^E(NONE)"), ("RPreUb", "H0 r^E(PRE) - r^E(NONE)", "upper bound of r^E(PRE) - r^E(NONE)"),
                     ("SidBeforeUb", "H0 s_ID^E(BEFORE) >= 0.1", "upper bound of s_ID^E(BEFORE)"), ("SidPreUb", "H0 s_ID^E(PRE) >= 0.1", "upper bound of s_ID^E(PRE)")),
             "Three": (("RLb", "H0 r^E(POST) <= 0.1", "lower bound of r^E(POST)"), ("RUb", "H0 r^E(POST) >= 1", "upper bound of r^E(POST)"),
                       ("DeltaPreLb", "H0 delta^E(POST - PRE)", "lower bound of delta^E(POST - PRE)"), ("DeltaNoneLb", "H0 delta^E(POST - NONE)", "lower bound of delta^E(POST - NONE)"),
                       ("SidLb", "H0 s_ID^E(POST) <= 0.1", "lower bound of s_ID^E(POST)"), ("SidDiffLb", "H0 s_ID^E(POST) - s_ID^E(NONE)", "lower bound of s_ID^E(POST) - s_ID^E(NONE)")),
             "Four": (("SidUb", "H0 s_ID^E(NONE) >= 0.15", "upper bound of s_ID^E(NONE)"), ("IDVLb", "H0 ID_V^E(NONE) <= 0", "lower bound of ID_V^E(NONE) (nats)")),
             "Five": (("BetaKLb", "H0 beta_K(AFTER) <= 0.4", "lower bound of beta_K(AFTER)"), ("BetaVLb", "H0 beta_V(NONE) <= 0.4", "lower bound of beta_V(NONE)"),
                      ("BetaKNoneUb", "H0 beta_K(NONE) >= 0.1", "upper bound of beta_K(NONE)"), ("BetaKBeforeUb", "H0 beta_K(BEFORE) >= 0.1", "upper bound of beta_K(BEFORE)"),
                      ("BetaKPreUb", "H0 beta_K(PRE) >= 0.1", "upper bound of beta_K(PRE)"), ("DiffLb", "H0 beta_K(AFTER) - beta_V(AFTER)", "lower bound of beta_K(AFTER) - beta_V(AFTER)")),
             "Fiveb": (("Lb", "H0 beta_K(POST) - beta_K(PRE)", "lower bound of beta_K(POST) - beta_K(PRE)"),),
             "Null": (("RUb", "H0 r^E(POST-NULL) >= 0.1", "upper bound of r^E(POST-NULL)"), ("SidUb", "H0 s_ID^E(POST-NULL) >= 0.1", "upper bound of s_ID^E(POST-NULL)"),
                      ("DeltaLb", "H0 delta^E(POST - POST-NULL)", "lower bound of delta^E(POST - POST-NULL)")),
             "Lb": (("Ub", "H0 ID_K^E(BEFORE) - ID_K^E(NONE)", "upper bound of ID_K^E(BEFORE) - ID_K^E(NONE) (nats)"),)}[stem]
    for k in ms:
        t, S = K8[k], SH8[k]
        for w, sub, d in comps:
            i = cb8(code, k, sub)
            CBP[stem + w, k] = sm8(f"Jb{stem}{w}{t}", "S8B", i, BND, f"{code}: {d} at {lvl} (the bound the line tests), {S}", G8B)
        if stem == "Three":
            i = next(j for j in LB8[code]["m"][k]["comps"] if "competent-only (n=" in S8B[j])
            sm8(f"JbThreeCompN{t}", "S8B", i, r"\(n=(\d+)\)", f"{code}: items competent in every arm (the competent-only check), {S}", G8B)
            for w, sub in (("DeltaPre", "delta^E(POST - PRE)"), ("DeltaNone", "delta^E(POST - NONE)"), ("SidDiff", "s_ID^E(POST) - s_ID^E(NONE)")):
                sm8(f"JbThreeComp{w}{t}", "S8B", cb8(code, k, "competent-only (n=" + macros[f"JbThreeCompN{t}"] + ") " + sub),
                    r"\) " + re.escape(sub) + " " + NUM, f"{code}: competent-only point estimate of {sub}, {S}", G8B)
        if stem == "Fiveb":
            i = LB8[code]["m"][k]["i"]
            sm8(f"JbFivebFlipPost{t}", "S8B", i, r"flip rate POST (\d\.\d+), PRE (\d\.\d+)", f"{code}: flip rate of the key-only clamp under POST, {S}", G8B, 0)
            sm8(f"JbFivebFlipPre{t}", "S8B", i, r"flip rate POST (\d\.\d+), PRE (\d\.\d+)", f"{code}: flip rate of the key-only clamp under PRE, {S}", G8B, 1)
for grp, ms, sx, lvl in (("Pfour", MB8P, "P4f", "95 %"), ("Nfour", MB8N, "N4", "98.75 %")):
    wh = f"over the {GRB8[grp]}, F, emitted-form scoring"
    rng8(f"JbOneSidLb{grp}", [CBP["OneSidLb", k] for k in ms], f"J-B1-{sx}: {lvl} lower bounds of s_ID^E(AFTER) (the bound the line tests) {wh}", SRCB + " J-B1", G8B)
    rng8(f"JbTwoRUb{grp}", [CBP[w, k] for k in ms for w in ("TwoRBeforeUb", "TwoRPreUb")],
         f"J-B2-{sx}: {lvl} upper bounds of r^E(BEFORE) - r^E(NONE) and r^E(PRE) - r^E(NONE) (the bounds the line tests) {wh}", SRCB + " J-B2", G8B)
    rng8(f"JbTwoSidUb{grp}", [CBP[w, k] for k in ms for w in ("TwoSidBeforeUb", "TwoSidPreUb")],
         f"J-B2-{sx}: {lvl} upper bounds of s_ID^E(BEFORE) and s_ID^E(PRE) (the bounds the line tests) {wh}", SRCB + " J-B2", G8B)
    rng8(f"JbNullRUb{grp}", [CBP["NullRUb", k] for k in ms], f"J-B-NULL-{sx}: {lvl} upper bounds of r^E(POST-NULL) (the bound the line tests) {wh}", SRCB + " J-B-NULL", G8B)
    rng8(f"JbNullDeltaLb{grp}", [CBP["NullDeltaLb", k] for k in ms],
         f"J-B-NULL-{sx}: {lvl} lower bounds of delta^E(POST - POST-NULL) (the bound the line tests) {wh}", SRCB + " J-B-NULL", G8B)
_fm = [k for c in ("J-B5b-P4f", "J-B5b-N4") for k, r_ in LB8[c]["m"].items() if r_["v"] == "MET"]
rng8("JbFivebFlipPostMet", [macros[f"JbFivebFlipPost{K8[k]}"] for k in _fm],
     f"J-B5b flip rate under POST over the models that meet J-B5b ({len(_fm)}: OLMo-2-7B excluded by its per-model verdict)", SRCB + " J-B5b", G8B)
amac("JbFivebNMet", len(_fm), "J-B5b: models (of the eight) whose per-model component is met", SRCB + " J-B5b", G8B, "{:d}")
# J-B-SMALL (Gemma-2-2B): its not-evaluable reason; J-B6a coverage; J-B6b cells (tab_invariance.tex, checked against the score file)
i = LB8["J-B-SMALL"]["m"]["gemma2b"]["i"]
assert "not evaluable: competence" in S8B[i], S8B[i]
_cov = []
for k in MB8P:
    for arm, aw in AS08.items():
        i = cb8("J-B6a", k, f"{arm}: coverage ")
        _cov.append(sm8(f"JbSixaCov{aw}{K8[k]}", "S8B", i, r"coverage (\d\.\d+)", f"J-B6a S0 coverage of the emitted forms (lowest row), {ALAB8[arm]} ({arm}), {SH8[k]}", G8B))
rng8("JbSixaCov", _cov, "J-B6a S0 coverage over the 24 cells of the four P4 models", SRCB + " J-B6a", G8B)
INV8 = {}
_inv_name = {SH8[k]: k for k in MB8P}
_c3 = r"^(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]$"
for i, line in enumerate(SCORE["S8BINV"]):
    cells = [c.strip() for c in line.rstrip(" \\").split("&")]
    if len(cells) != 7 or cells[0] not in _inv_name:
        continue
    k, arm = _inv_name[cells[0]], cells[1]
    t = K8[k]
    smac(f"JbSixbML{AS08[arm]}{t}", "S8BINV", i, cells[2], f"J-B6b S0 M^L: mass of the lower-case clean-B form, {ALAB8[arm]} ({arm}), {SH8[k]}", G8B)
    INV8[k, arm, "Sid"] = re.match(_c3, cells[5]).groups()
    smac3(f"JbSixbSid{AS08[arm]}{t}", "S8BINV", i, INV8[k, arm, "Sid"], f"J-B6b/S0: s_ID^E - s_ID^L (emitted-form minus lower-case score), {ALAB8[arm]} ({arm}), 95 % interval, {SH8[k]}", G8B)
    if cells[6] != "--":
        INV8[k, arm, "R"] = re.match(_c3, cells[6]).groups()
        smac3(f"JbSixbR{AS08[arm]}{t}", "S8BINV", i, INV8[k, arm, "R"], f"J-B6b/S0: r^E - r^L, {ALAB8[arm]} ({arm}), 95 % interval, {SH8[k]}", G8B)
assert len([x for x in INV8 if x[2] == "Sid"]) == 24, len(INV8)
_n6b = 0
for k in MB8P:
    for j in LB8["J-B6b"]["m"][k]["comps"]:
        mm = re.search(r"H0 (s_ID|r)\^E\((\w+)\) - (?:s_ID|r)\^L\(\w+\)(?: \[M\^L (\d\.\d+)\])? outside \(-0\.05, 0\.05\) rejected at 95% "
                       r"\(interval \[([+-]\d+\.\d+),([+-]\d+\.\d+)\]\)", S8B[j])
        assert mm, S8B[j]
        tb = INV8[k, mm[2], "Sid" if mm[1] == "s_ID" else "R"]
        assert (float(mm[4]), float(mm[5])) == (float(tb[1]), float(tb[2])), (k, mm.groups(), tb)
        if mm[3]:
            assert mm[3] == macros[f"JbSixbML{AS08[mm[2]]}{K8[k]}"], (k, mm[3])
        _n6b += 1
assert _n6b == 22, _n6b   # 6 (Qwen2.5-7B) + 8 + 8; OLMo-2-7B has no low-L-mass cell
# J-B7 and J-B8
_a7, _o7, _c7 = [], [], []
for k in MB8P + MB8N:
    for j in LB8["J-B7"]["m"][k]["comps"]:
        a, o, c = grab(S8B, j, r"A (\d\.\d+) >= 0\.95 \(n=\d+\), other\(B\) (\d\.\d+) <= 0\.10, coverage (\d\.\d+)")
        _a7.append(a), _o7.append(o), _c7.append(c)
assert len(_a7) == 48
rng8("JbSevenA", _a7, "J-B7 agreement A of the emitted-form argmax with generation over the 48 F cells", SRCB + " J-B7", G8B)
rng8("JbSevenOther", _o7, "J-B7 clean-B 'other' rate over the 48 F cells", SRCB + " J-B7", G8B)
rng8("JbSevenCov", _c7, "J-B7 coverage over the 48 F cells", SRCB + " J-B7", G8B)
L8m = LB8["J-B8"]["m"]["mistral24"]
sm8("JbEightN", "S8B", L8m["i"], r"(\d+) cores", "J-B8: native cores of the Mistral-Small-24B intervention frames", G8B)
_lo8, _hi8, _me8 = [], [], []
for j in L8m["comps"]:
    mm = re.search(r"interval \[([+-]\d+\.\d+),([+-]\d+\.\d+)\]", S8B[j])
    if mm:
        _lo8.append(mm[1]), _hi8.append(mm[2])
    mm = re.search(r"^          (\w+): M\^E\(natural B\) (\d\.\d+)", S8B[j])
    if mm:
        _me8.append(mm[2])
        smac(f"JbEightME{dict(P1='Opt', NONE='None', BEFORE='Before', POST='Post')[mm[1]]}", "S8B", j, mm[2], f"J-B8 M^E(natural B), {mm[1]}, Mistral-Small-24B", G8B)
assert len(_lo8) == 12 and len(_me8) == 4
rng8("JbEightLo", _lo8, "J-B8 lower bounds of the 12 E - L differences of phi, psi_K, psi_V (P1, NONE, BEFORE, POST)", SRCB + " J-B8", G8B)
rng8("JbEightHi", _hi8, "J-B8 upper bounds of the 12 E - L differences", SRCB + " J-B8", G8B)
rng8("JbEightME", _me8, "J-B8 M^E(natural B) over the four frames", SRCB + " J-B8", G8B)
i = find(S8B, "  J-B8 detail (L -> E):", start=RB8, end=SB8)
for arm, aw in (("P1", "Opt"), ("NONE", "None"), ("BEFORE", "Before"), ("POST", "Post"), ("LETTER", "Letter")):
    for j, (w, d) in enumerate((("PhiL", "phi under L"), ("PhiE", "phi under E"), ("PsiKL", "psi_K under L"), ("PsiKE", "psi_K under E"),
                                ("PsiVL", "psi_V under L"), ("PsiVE", "psi_V under E"))):
        sm8(f"JbEight{w}{aw}", "S8B", i, re.escape(arm) + r": phi " + NUM + " -> " + NUM + r", psi_K " + NUM + " -> " + NUM + r", psi_V " + NUM + " -> " + NUM,
            f"J-B8 detail: {d}, {arm}, Mistral-Small-24B frames", G8B, j)
# gates: J-B-G0 tests, J-B-G3 trie floor, J-B-G4 coverage and floor, J-B-G5 competence
i = find(S8B, "  J-B-G0  FP32 exactness", start=GB8, end=PB8)
amac("JbGzeroPassed", sum(int(x) for x in re.findall(r" (\d+) passed", S8B[i])), "J-B-G0: unit tests passed (sum over the eight files)", f"{SRCB}:{i + 1}", G8B, "{:d}")
i = find(S8B, "  N4 slots: ", start=GB8, end=PB8)
amac("JbNfourFamilies", len(re.findall(r"'(\w+)': '\w+'", S8B[i])), "N4: model families never examined before in this work (Llama, Gemma, Phi, Falcon)", f"{SRCB}:{i + 1}", G8B, "{:d}")
_ds, _dl, _bd = [], [], []
for j in find(S8B, "J-B-G3 (first 30 F cores)", start=GB8, end=PB8, many=True):
    for a, b, c in re.findall(r"\|ds\| (\d\.\d+) \(<= 0\.02\), mean\|dL\| (\d\.\d+) \(<= (\d\.\d+)\)", S8B[j]):
        _ds.append(a), _dl.append(b), _bd.append(c)
assert len(_ds) == 33, len(_ds)
rng8("JbGthreeDs", _ds, "J-B-G3 trie floor |delta s_ID| over 11 models x 3 arms", SRCB + " GATES", G8B)
rng8("JbGthreeDL", _dl, "J-B-G3 trie floor mean |delta L| (nats) over 11 models x 3 arms", SRCB + " GATES", G8B)
rng8("JbGthreeBound", _bd, "J-B-G3 bounds of the mean |delta L| (nats)", SRCB + " GATES", G8B)
_cv, _fl, _ac = [], [], []
for k in MB8:
    for arm, aw in AB8.items():
        i = find(S8B, f"  {k:9s} {arm:9s} J-B-G4 coverage", start=GB8, end=PB8)
        c, f_, a = grab(S8B, i, r"J-B-G4 coverage (\d\.\d+) .*? floor (\d\.\d+) .*?J-B-G5 acc_B (\d\.\d+)")
        _cv.append(c), _fl.append(f_)
        if k == "gemma2b":
            smac(f"JbGfiveAcc{aw}GemmaTwo", "S8B", i, a, f"J-B-G5 generated accuracy acc_B (>= 0.8 needed), {ALAB8[arm]} ({arm}), Gemma-2-2B", G8B)
        else:
            _ac.append(a)
rng8("JbGfourCov", _cv, "J-B-G4 coverage over the 54 F cells of the nine models", SRCB + " GATES", G8B)
rng8("JbGfourFloor", _fl, "J-B-G4 floor (nats) over the 54 F cells", SRCB + " GATES", G8B)
rng8("JbGfiveAcc", _ac, "J-B-G5 generated accuracy acc_B over the 48 F cells of the eight P4 and N4 models", SRCB + " GATES", G8B)
# reported: S0 relative changes of ID_K and ID_V, E against L
for i in range(RB8, SB8):
    mm = re.match(r"^    (\S+)\s+(P1|AFTER|BEFORE|POST|PRE|NONE)\s+M\^L\(B\) (\d\.\d+)  ID_K " + NUM + " -> " + NUM + r" \(([+-]\d+\.\d)%, (?:within|beyond) 15%\)  ID_V "
                  + NUM + " -> " + NUM + r" \(([+-]\d+\.\d)%", S8B[i])
    if mm:
        t, aw = K8[mm[1]], AS08[mm[2]]
        smac(f"JbSzeroPctIDK{aw}{t}", "S8B", i, mm[6], f"S0: relative change of ID_K, E against L (per cent), {ALAB8[mm[2]]} ({mm[2]}), {SH8[mm[1]]}", G8B)
        smac(f"JbSzeroPctIDV{aw}{t}", "S8B", i, mm[9], f"S0: relative change of ID_V, E against L (per cent), {ALAB8[mm[2]]} ({mm[2]}), {SH8[mm[1]]}", G8B)
# summary
classes8(S8B, SB8, XB8, "Jb", G8B, ("L", "M", "R", "Val"))
i = find(S8B, "  met rate among R account lines with a verdict:", start=SB8, end=XB8)
for j, (w, d) in enumerate((("Met", "met"), ("Verd", "with a verdict"), ("Exp", "expected met"))):
    sm8(f"JxRisky{w}B", "S8B", i, r"(\d+) of (\d+) \(expected (\d+\.\d+)\)", f"part B: risky (class R) account lines {d}", G8X, j)
i = find(S8B, "  overall: ", start=SB8, end=XB8)
for j, w in enumerate(("Met", "NotMet", "NotEval", "Lines")):
    sm8(f"JbCount{w}", "S8B", i, r"(\d+) MET, (\d+) NOT MET, (\d+) NOT EVALUABLE of (\d+) lines", f"part B: lines {('MET', 'NOT MET', 'NOT EVALUABLE', 'in all')[j]}", G8B, j)
# exploratory: the list-before sign, per-sentence and per-lexicon r(POST), X1 (attention and precision), X2 (1.5B and 3B on S0)
_lev, _lev2 = [], []
for k in MB8:
    h = find(S8B, f"  [{k} F] Sigma", start=XB8)
    e = next(j for j in range(h + 1, len(S8B)) if not S8B[j].startswith("    "))
    i = find(S8B, "    LIST-BEFORE sign:", start=h, end=e)
    sm8(f"JbLbDiff{K8[k]}", "S8B", i, r"ID_K\^E\(NONE\) " + NUM, f"ID_K^E(BEFORE) - ID_K^E(NONE) (nats; exploratory point estimate), {SH8[k]}", G8B)
    sm8(f"JbBidPost{K8[k]}", "S8B", i, r"b_ID\(POST\) " + NUM, f"b_ID(POST) (exploratory), {SH8[k]}", G8B)
    for j in find(S8B, "    per level ", start=h, end=e, many=True):
        (_lev2 if k == "gemma2b" else _lev).append(grab(S8B, j, r"r\^E\(POST\) " + NUM)[0])
assert len(_lev) == 80 and len(_lev2) == 10, (len(_lev), len(_lev2))
rng8("JbLevelRPost", _lev, "r^E(POST) per sentence (8) and per lexicon (2) over the eight P4 and N4 models (exploratory)", SRCB + " EXPLORATORY", G8B)
rng8("JbLevelRPostGemmaTwo", _lev2, "r^E(POST) per sentence and lexicon, Gemma-2-2B (exploratory)", SRCB + " EXPLORATORY", G8B)
for k in ("qwen1.5", "qwen3b"):
    h = find(S8B, f"  [{k} S0] Sigma", start=XB8)
    for arm in ("AFTER", "POST"):
        i = find(S8B, f"    {arm:9s} s_ID L/Sigma/E", start=h, end=h + 6)
        sm8(f"JbXtwoSid{AB8[arm]}{K8[k]}", "S8B", i, r"s_ID L/Sigma/E " + NUM + "/" + NUM + "/" + NUM, f"X2 (exploratory): s_ID^E on S0, {arm}, {SH8[k]}", G8B, 2)
        sm8(f"JbXtwoComp{AB8[arm]}{K8[k]}", "S8B", i, r"competent n=(\d+)", f"X2: competent S0 items, {arm}, {SH8[k]}", G8B)
for tag8, w in (("x1eager", "Eager"), ("x1fp32", "Fp")):
    i = find(S8B, f"  eval/qwen7_{tag8}_S0.json", start=XB8)
    for arm in ("AFTER", "POST", "NONE"):
        sm8(f"JbXone{w}{AB8[arm]}", "S8B", i, re.escape(arm) + r" s_ID\^E " + NUM,
            f"X1 (exploratory): s_ID^E on the first 60 S0 cores, {arm}, Qwen2.5-7B, {'BF16 eager attention' if w == 'Eager' else 'FP32 sdpa'}", G8B)

# ---------------------------------------------------------------- stage 8, part C: independently obtained identity edits and the channel-ratio law
# Parsed from STAGE8C_SCORE.txt in the style of parts A, B and D (group S8C, prefix Jc). Part C's intervals: a lambda
# statistic and a pooled lambda-bar carry the 90 % interval of the TOST (smac390 / rmac390: their ...Lo and ...Hi
# descriptions say 90 %); every other interval is the 95 % hierarchical cluster-bootstrap interval (smac3). The pytest
# session of record (results/gpu_stage8c/logs/pytest.log, one session) prints the 95 tests of the outcome record (tag
# S8CPY). Values the scorer does not print are recomputed through the committed part-C scorer's own classes
# (analysis/stage8c_parts: law.Model, readers.Readers, the hierarchical bootstrap stats.Boot, seed 20261012) and registered
# with rmac; the same code first reproduces every printed lambda statistic, pooled lambda-bar, J-C-READa row, lexical-code
# reading and identity-carrying phi to the printed digit (asserted), and where the outcome record (docs/PREREGISTRATION.md,
# "Outcome of P-2026-10-10-J, part C") prints a value marked "(recomputed)", the recomputation reproduces it (asserted).
# Names: Jc + statistic + instance + format or statistic code + depth + model tag (...Lo / ...Hi: the interval bounds).
# Instances: Nat (the natural write), T (the equivalence control), Eone, Etwo, Ethree, Efour (E4 at the seed level),
# Efoura, Efourb (seeds 101, 102), EfiveFr, EfiveDe, EfiveSyn, EfiveNl; components Par / Perp / Lex / Nonlex + instance.
# Statistic codes: Wopt = W(P1, l), Wpost = W(POST, l), Aletter = A_L(l), Aopt = A_P(l); depths Three, Seven, Eleven,
# Fifteen; formats Letter (LETTER), Opt (P1), Post (POST), None (NONE). Line words: One ... Five, Bound, Reada, Readb, Six,
# Screen, Win.
G8C = "S8C"
for _tag, _f in (("S8C", "results/gpu_stage8c/STAGE8C_SCORE.txt"), ("S8CPY", "results/gpu_stage8c/logs/pytest.log")):
    SCORE[_tag] = (ROOT / _f).read_text().splitlines()
    SNAME[_tag] = _f
S8C = SCORE["S8C"]
SRCC = SNAME["S8C"]
MC8 = ["qwen7", "mistral7", "llama8"]
DW8 = {3: "Three", 7: "Seven", 11: "Eleven", 15: "Fifteen"}
FWC8 = {"LETTER": "Letter", "P1": "Opt", "POST": "Post", "NONE": "None"}
ZW8 = {"nat": "Nat", "T": "T", "E1": "Eone", "E2": "Etwo", "E3": "Ethree", "E4": "Efour", "E4a": "Efoura", "E4b": "Efourb",
       "E5FR": "EfiveFr", "E5DE": "EfiveDe", "E5SYN": "EfiveSyn", "E5NL": "EfiveNl"}
FAMC8 = {"qwen7": ("E1", "E2", "E3", "E4", "E5FR", "E5DE", "E5SYN", "E5NL"), "mistral7": ("E1", "E2", "E4", "E5FR", "E5DE", "E5SYN", "E5NL"),
         "llama8": ("E1", "E2", "E5FR", "E5DE", "E5SYN", "E5NL")}
LWC8 = {"J-C1": "One", "J-C2": "Two", "J-C3": "Three", "J-C4": "Four", "J-C5": "Five", "J-C-BOUND": "Bound", "J-C-READa": "Reada",
        "J-C-READb": "Readb", "J-C6": "Six", "J-C-SCREEN": "Screen", "J-C-WIN": "Win"}
ZLAB8 = {"E1": "E1 (in-task steering vector)", "E2": "E2 (steering vector from neutral sentences)", "E3": "E3 (sparse-autoencoder features)",
         "E4": "E4 (DAS remap at the writing token, seed level)", "E4a": "E4 seed 101", "E4b": "E4 seed 102", "E5FR": "E5 French form",
         "E5DE": "E5 German form", "E5SYN": "E5 synonym", "E5NL": "E5 NONLEX of the in-task vector"}


def zw8(Z):
    """instance -> name piece: E1 -> Eone, PERP:E4 -> PerpEfour, NONLEX:nat -> NonlexNat."""
    head, _, base = Z.rpartition(":")
    return (head.capitalize() if head else "") + ZW8[base]


def zlab8(Z):
    head, _, base = Z.rpartition(":")
    return f"{head} component of {ZLAB8.get(base, base)}" if head else ZLAB8.get(Z, Z)


def sw8(code):
    """statistic code -> name piece: W(P1,3) -> WoptThree, W(POST,7) -> WpostSeven, A_L(7) -> AletterSeven, A_P(11) -> AoptEleven."""
    mm = re.match(r"(W|A_L|A_P)\((?:(P1|POST),)?(\d+)\)$", code)
    assert mm, code
    return {"W": "Wopt" if mm[2] == "P1" else "Wpost", "A_L": "Aletter", "A_P": "Aopt"}[mm[1]] + DW8[int(mm[3])]


def smac390(name, tag, idx, t, desc, group):
    """estimate and its 90 % interval (the TOST level of part C's lambda statistics) -> name, nameLo, nameHi."""
    smac(name, tag, idx, t[0], desc + " (point; ...Lo and ...Hi are its 90 % bounds)", group)
    smac(name + "Lo", tag, idx, t[1], desc + " (lower 90 % bound)", group)
    smac(name + "Hi", tag, idx, t[2], desc + " (upper 90 % bound)", group)


def sm8390(name, tag, i, pat, desc, group, off=0):
    t_ = grab(SCORE[tag], i, pat)[off:off + 3]
    smac390(name, tag, i, t_, desc, group)
    return t_


def rmac390(name, e3, desc, src, group, fmt="{:.3f}"):
    """recomputed estimate and 90 % interval -> name, nameLo, nameHi."""
    for suf, v, d in (("", e3[0], " (point; ...Lo and ...Hi are its 90 % bounds)"), ("Lo", e3[1], " (lower 90 % bound)"), ("Hi", e3[2], " (upper 90 % bound)")):
        rmac(name + suf, v, desc + d, src, group, fmt)


GC8, PC8, HC8, RC8 = (sec(S8C, s) for s in ("GATES (J-C-G0", "PREDICTIONS (code, class", "HEADLINE (C-6", "REPORTED (no verdict counted)"))
SC8, XC8 = S8C.index("SUMMARY"), sec(S8C, "######## EXPLORATORY")
# gates: J-C-G0 (the six files the score line names; the session of record ran four shared files too)
i = find(S8C, "  J-C-G0  FP32 unit tests", start=GC8, end=PC8)
assert S8C[i].endswith("failures in the run 0 -> MET"), S8C[i]
amac("JcGzeroPassed", sum(int(x) for x in re.findall(r" (\d+) passed", S8C[i])),
     "J-C-G0: unit tests passed, summed over the six files the score line names (15, 7, 19, 5, 5 and 7; the session of record also ran four shared files: JcGzeroPassedRun)",
     f"{SRCC}:{i + 1}", G8C, "{:d}")
PYC8 = SCORE["S8CPY"]
assert sum("test session starts" in x for x in PYC8) == 1
i = find(PYC8, " passed, ", " warning in ")
sm8("JcGzeroPassedRun", "S8CPY", i, r"=+ (\d+) passed, \d+ warnings? in [\d.]+s",
    "J-C-G0: unit tests passed in the pytest session of record (ten files: the six of the gate and the shared test_surface, test_clamp, test_head_splice and test_prakash; 0 failed, 0 skipped)", G8C)
assert not re.search(r"\b(failed|skipped|error)", PYC8[i]), PYC8[i]
_pyc8 = [x.split("::")[0] for x in PYC8 if re.match(r"^tests/\w+\.py::\S+ PASSED", x)]
assert len(_pyc8) == int(macros["JcGzeroPassedRun"]), len(_pyc8)
amac("JcGzeroFiles", len(set(_pyc8)), "J-C-G0: test files run in that session", SNAME["S8CPY"], G8C, "{:d}")
for k in MC8:
    t, S = K8[k], SH8[k]
    i = find(S8C, f"  J-C-G1 {k:8s} ", start=GC8, end=PC8)
    assert S8C[i].endswith("all within tolerance -> MET"), S8C[i]
    sm8(f"JcGoneCells{t}", "S8C", i, r"(\d+) evaluable cells", f"J-C-G1: evaluable cells (format x depth) in which T's tables and rows equal the natural ones (all within tolerance), {S}", G8C)
    i = find(S8C, f"  J-C-G2 {k:8s} ", start=GC8, end=PC8)
    assert S8C[i].endswith("-> MET"), S8C[i]
    sm83(f"JcGtwo{t}", "S8C", i, r"sigma\(LETTER, 3\) - sigma\(NONE, 3\) " + CIR, f"J-C-G2: sigma(LETTER, 3) - sigma(NONE, 3), the natural format effect on the key share (>= 0.5 needed), {S}", G8C)
    i = find(S8C, f"  J-C-G8 {k:8s} sensitivity -> MET", start=GC8, end=PC8)
    for row, rw, d in (("T", "T", "the equivalence control T (the natural tables)"),
                       ("SK50", "SkFifty", "synthetic rows: the natural key table interpolated halfway toward B (half-key)"),
                       ("SV50", "SvFifty", "synthetic rows: the natural value table interpolated halfway toward B (half-value)"),
                       ("SK80", "SkEighty", "synthetic rows: the key table at 0.8 of the natural displacement"),
                       ("SV80", "SvEighty", "synthetic rows: the value table at 0.8 of the natural displacement")):
        j = find(S8C, f"           {row}: ", start=i + 1, end=i + 6)
        assert ("old rule fails" in S8C[j]) == (row == "SK50"), S8C[j]   # the old kappa rule fails only the half-key rows
        sm8(f"JcGeight{rw}N{t}", "S8C", j, r": (\d+) statistics", f"J-C-G8 row {row}, {d}: lambda statistics, {S}", G8C)
        sm8390(f"JcGeight{rw}{t}", "S8C", j, r"pooled lambda " + CIR, f"J-C-G8 row {row}, {d}: pooled lambda-bar, {S}", G8C)
        sm83(f"JcGeight{rw}Mad{t}", "S8C", j, r"MAD " + CIR, f"J-C-G8 row {row}: the design's old kappa rule, mean |kappa - sigma| over the evaluable cells (<= 0.12 needed), {S}", G8C)
        sm8(f"JcGeight{rw}MadMax{t}", "S8C", j, r"max (\d\.\d+) over", f"J-C-G8 row {row}: largest |kappa - sigma| over the cells (old rule: <= 0.25), {S}", G8C)
# J-C-G4: every failing cell (the LETTER cells at l = 11 and 15 in every model) with its reason
_g4f = []
for k in MC8:
    t, S = K8[k], SH8[k]
    for l, lw in DW8.items():
        i = find(S8C, f"  J-C-G4 {k:8s} l={l:2d} ", start=GC8, end=PC8)
        cells = re.findall(r"(LETTER|P1|POST|NONE):(ok|x\(([^)]*)\))", S8C[i])
        assert len(cells) == 4, S8C[i]
        for f, ok, why in cells:
            if ok == "ok":
                continue
            _g4f.append((k, f, l))
            mm = re.search(r"ID_KV (\d+\.\d\d) < 2\.0", why)
            if mm:
                smac(f"JcGfourIDKV{FWC8[f]}{lw}{t}", "S8C", i, mm[1], f"J-C-G4: natural ID_KV (nats) of the failing {f} cell at l = {l} (>= 2 needed), {S}", G8C)
            mm = re.search(r"iota (\d+\.\d\d) > 0\.5", why)
            if mm:
                smac(f"JcGfourIota{FWC8[f]}{lw}{t}", "S8C", i, mm[1], f"J-C-G4: natural iota of the failing {f} cell at l = {l} (<= 0.5 needed){' (a ratio over a near-zero ID_KV)' if k == 'llama8' else ''}, {S}", G8C)
assert sorted(_g4f) == sorted((k, "LETTER", l) for k in MC8 for l in (11, 15)), _g4f
amac("JcGfourFail", len(_g4f), "J-C-G4: cells that fail, of the 48 (3 models x 4 depths x 4 formats): LETTER at l = 11 and 15 in every model", SRCC + " GATES", G8C, "{:d}")
# J-C-G6 (specificity of the flag direction: control R)
_g6 = []
for k in MC8:
    t, S = K8[k], SH8[k]
    for l, lw in DW8.items():
        i = find(S8C, f"  J-C-G6 {k:8s} l={l:2d} ", start=GC8, end=PC8)
        assert S8C[i].endswith("-> passed"), S8C[i]
        _g6.append(sm8(f"JcGsixR{lw}{t}", "S8C", i, r"\|ID_KV\^R\| (\d\.\d+) vs", f"J-C-G6: |ID_KV| of the random-direction control R under NONE (nats), l = {l}, {S}", G8C))
        sm8(f"JcGsixBound{lw}{t}", "S8C", i, r"vs 0\.1 ID_KV\^nat (\d\.\d+) ->", f"J-C-G6: its bound, 0.1 x the natural ID_KV under NONE (nats), l = {l}, {S}", G8C)
rng8("JcGsixR", _g6, "J-C-G6: |ID_KV^R| over the 12 model x depth cells (nats)", SRCC + " GATES", G8C)
# J-C-G5 (efficacy under NONE): phi(NONE) per model, instance and depth; the flip rates of E1-E4
EFF8C = {}
_g5 = {"near": [], "E3": [], "E4": [], "E4f": []}
for k in MC8:
    t, S = K8[k], SH8[k]
    for Z in FAMC8[k]:
        for l, lw in DW8.items():
            i = find(S8C, f"  J-C-G5 {k:8s} {Z:6s} l={l:2d} ", start=GC8, end=PC8)
            eff = S8C[i].endswith("-> effective")
            assert eff or S8C[i].endswith("-> ineffective"), S8C[i]
            EFF8C[k, Z, l] = eff
            rule = "effective: phi >= 0.3 with lower bound > 0" if Z.startswith("E5") else "effective: flip rate >= 0.8 x natural, phi >= 0.5 with lower bound > 0.3"
            q = sm83(f"JcGfivePhi{zw8(Z)}{lw}{t}", "S8C", i, r"phi\(NONE\) " + CIR,
                     f"J-C-G5: phi(NONE) = psi_KV, the edit's identity effect under NONE as a share of the natural one, {zlab8(Z)}, l = {l} "
                     f"({'effective' if eff else 'ineffective'}; {rule}), {S}", G8C)
            if Z.startswith("E5"):
                continue
            fz, fn = grab(S8C, i, r"flip (\d\.\d\d) vs natural (\d\.\d\d)")
            assert fn == "1.00", S8C[i]
            smac(f"JcGfiveFlip{zw8(Z)}{lw}{t}", "S8C", i, fz, f"J-C-G5: flip rate under NONE (share of (story, target) whose E-argmax in the KV row is the target), {zlab8(Z)}, l = {l}, {S}", G8C)
            if Z == "E1" and l == 3:
                smac(f"JcGfiveFlipNat{t}", "S8C", i, fn, f"J-C-G5: the natural flip rate under NONE (the same at every depth), {S}", G8C)
            if Z in ("E1", "E2"):
                assert eff, (k, Z, l)
                _g5["near"].append(q[0])
            elif Z == "E3":
                assert eff, (k, Z, l)
                _g5["E3"].append(q[0])
            elif eff:
                _g5["E4"].append(q[0]), _g5["E4f"].append(fz)
rng8("JcGfivePhiNear", _g5["near"], "J-C-G5: phi(NONE) of E1 and E2 over every depth of the three models (all effective, flip rate 1.00)", SRCC + " GATES", G8C)
rng8("JcGfivePhiEthree", _g5["E3"], "J-C-G5: phi(NONE) of E3 over the four layers, Qwen2.5-7B (all effective, flip rate 1.00)", SRCC + " GATES", G8C)
rng8("JcGfivePhiEfourEff", _g5["E4"], "J-C-G5: phi(NONE) of E4 at the depths where it is effective (l = 3, 7 at Qwen2.5-7B; 3, 7, 11 at Mistral-7B)", SRCC + " GATES", G8C)
rng8("JcGfiveFlipEfourEff", _g5["E4f"], "J-C-G5: flip rate of E4 at the depths where it is effective", SRCC + " GATES", G8C)
for k in ("qwen7", "mistral7"):
    amac(f"JcGfiveEffEfour{K8[k]}", sum(EFF8C[k, "E4", l] for l in DW8), f"J-C-G5: depths (of 4) at which E4 is effective, {SH8[k]}", SRCC + " GATES", G8C, "{:d}")
for k in MC8:
    amac(f"JcGfiveEffEfive{K8[k]}", sum(EFF8C[k, Z, l] for Z in FAMC8[k] if Z.startswith("E5") for l in DW8),
         f"J-C-G5: E5 sub-family x depth cells (of 16) gated in (phi >= 0.3 with lower bound > 0), {SH8[k]}", SRCC + " GATES", G8C, "{:d}")
amac("JcGfiveEffEfiveDe", sum(EFF8C[k, "E5DE", l] for k in MC8 for l in DW8), "J-C-G5: E5-DE (German form) cells gated in, over the three models and four depths", SRCC + " GATES", G8C, "{:d}")
assert EFF8C["mistral7", "E4", 11] and not EFF8C["qwen7", "E4", 11]
# J-C-G3 (the dictionary's FVE at its site; not met at any layer)
for l, lw in DW8.items():
    i = find(S8C, f"  J-C-G3 qwen7    l={l:2d} ", start=GC8, end=PC8)
    assert S8C[i].endswith("-> NOT MET"), S8C[i]
    a_, b_, c_, pub = grab(S8C, i, r"FVE " + NUM + " / " + NUM + " / " + NUM + r" at l-1 / l / l\+1 \(published (\d\.\d+);")
    for w, s_, d in (("Prev", a_, "at block l - 1"), ("", b_, "at block l, the dictionary's site (the gate's statistic)"), ("Next", c_, "at block l + 1")):
        smac(f"JcGthreeFve{w}{lw}", "S8C", i, s_, f"J-C-G3: FVE of the layer-{l} dictionary on the block outputs over every prompt position >= 1, {d}, Qwen2.5-7B", G8C)
    amac(f"JcGthreeFveRound{lw}", f"{round(float(b_)):,}".replace(",", "{,}"), f"J-C-G3: that FVE at block l rounded to an integer, with a thousands separator (math mode), layer {l}",
         f"{SRCC}:{i + 1}", G8C)
    smac(f"JcGthreePub{lw}", "S8C", i, pub, f"J-C-G3: the dictionary's published FVE (eval_results.json), layer {l}", G8C)
    amac(f"JcGthreePubRound{lw}", float(pub), f"J-C-G3: the published FVE rounded to 3 decimals, layer {l}", f"{SRCC}:{i + 1}", G8C, "{:.3f}")
    sm8(f"JcGthreeFveP{lw}", "S8C", i, r"FVE at p (\d\.\d+);", f"J-C-G3 (reported): FVE at the writing position p only, layer {l}", G8C)
    sm8(f"JcGthreeKf{lw}", "S8C", i, r"k_F (\d+) beta", f"E3 calibration: k_F, the number of selective latents written, layer {l}", G8C)
# the lines: header (code, class, kind, prior, verdict, title) and the detail lines below it
LC8 = {}
for i in range(PC8 + 1, HC8):
    mm = re.match(r"  (J-C[\w-]+)\s+(\w) (\w) prior (\d\.\d\d)\s+" + VRD + r"\s+(.*)$", S8C[i])
    if mm:
        cur = mm[1]
        LC8[cur] = dict(i=i, cls=mm[2], kind=mm[3], prior=mm[4], v=mm[5], title=mm[6], det=[])
    elif S8C[i].strip():
        LC8[cur]["det"].append(i)
assert list(LC8) == list(LWC8), list(LC8)
STC8 = r"(W\((?:P1|POST),\d+\)|A_L\(\d+\)|A_P\(\d+\)) ([+-]\d\.\d\d) \[([+-]\d\.\d\d),([+-]\d\.\d\d)\](\*)?(\(iota\))?"
PAT8 = r"(equivalent|R1 copy-only|R3 key-flat|graded departure(?: \(pooled lambda within the margin; some statistic not equivalent\))?)"
COMB8 = {}   # (line, model, instance, depths) -> the combo as printed
LAMP8 = {}   # (model, instance, statistic code) -> dict(t=(point, lo, hi) printed or recomputed strings, eq, inter, line, rec)


def lam8(code_line, k, Z, sts, i, rec=False, src=None):
    """register the lambda statistics of instance Z (printed strings on line i, or recomputed Q objects with rec=True)."""
    for s in sts:
        if rec:
            code, q, eq, inter = s["code"], s["q"], s["equivalent"], bool(s["interaction"])
            t3 = tuple(f"{x:.2f}" for x in (q.pt, q.lower(0.9), q.upper(0.9)))
        else:
            code, pt, lo, hi, star, iota = s
            t3, eq, inter = (sv(pt), sv(lo), sv(hi)), not star, bool(iota)
        nm_ = f"JcLam{zw8(Z)}{sw8(code)}{K8[k]}"
        d = (f"{code_line}: lambda = log psi_K - log psi_V of {zlab8(Z)}, statistic {code}; {'equivalent' if eq else 'not equivalent'}"
             f"{' (iota > 0.5 in a cell: a departure by the interaction rule)' if inter else ''}, {SH8[k]}")
        if rec:
            rmac390(nm_, (s["q"].pt, s["q"].lower(0.9), s["q"].upper(0.9)), d, src, G8C, "{:.2f}")
        else:
            smac390(nm_, "S8C", i, t3, d, G8C)
        assert (k, Z, code) not in LAMP8, (k, Z, code)
        LAMP8[k, Z, code] = dict(t=tuple(macros[nm_ + s_] for s_ in ("", "Lo", "Hi")), eq=eq, inter=inter, line=code_line, rec=rec)


for code in ("J-C1", "J-C2", "J-C3", "J-C4"):
    for i in LC8[code]["det"]:
        mm = re.match(r"    (\(seed-level, reported\) )?(qwen7|mistral7|llama8) (E\w+)(?: l in \[([\d, ]*)\])?: (.*) -> " + VRD + "$", S8C[i])
        if not mm:
            assert S8C[i] == "    E3 depths passing J-C-G3: []", S8C[i]
            continue
        seed, k, Z, dep, body, v = mm.groups()
        dep = tuple(int(x) for x in dep.split(",")) if dep else ((3, 7, 11, 15) if seed else ())
        t, S = K8[k], SH8[k]
        lw = LWC8[code]
        lab = f"{code}{' (seed level, reported)' if seed else ''}: {zlab8(Z)}, l in {list(dep)}"
        c = dict(i=i, v=v, seed=bool(seed), k=k, Z=Z, dep=dep)
        COMB8[code, k, Z, dep] = c
        if v == "NOT EVALUABLE":
            c["n"] = int(sm8(f"Jc{lw}Nstat{zw8(Z)}{t}", "S8C", i, r": (\d+) statistics \(< 3\)", f"{lab}: lambda statistics (< 3: the combo is not evaluable), {S}", G8C))
            continue
        m2 = re.match(r"(\d+)/(\d+) equivalent; pooled lambda " + CIR + r" \(90 %\) -> " + PAT8 + "; (.*)$", body)
        assert m2, body
        c.update(neq=int(m2[1]), n=int(m2[2]), pat=m2[6], pool=m2.groups()[2:5])
        smac(f"Jc{lw}Neq{zw8(Z)}{t}", "S8C", i, m2[1], f"{lab}: lambda statistics that are equivalent (90 % interval inside +-log 1.25), {S}", G8C)
        smac(f"Jc{lw}Nstat{zw8(Z)}{t}", "S8C", i, m2[2], f"{lab}: lambda statistics, {S}", G8C)
        smac390(f"Jc{lw}Pool{zw8(Z)}{t}", "S8C", i, c["pool"], f"{lab}: pooled lambda-bar (mean of the statistics; pattern {m2[6].split(' (')[0]}), {S}", G8C)
        sts = re.findall(STC8, m2[7])
        assert len(sts) == c["n"] and sum(not s[4] for s in sts) == c["neq"], (code, k, Z)
        lam8(code + (" (seed level)" if seed else ""), k, Z, sts, i)
        c["codes"] = [s[0] for s in sts]
amac("JcThreeDepths", 0, "J-C3: E3 layers passing J-C-G3 (none: J-C3 not evaluable)", f"{SRCC}:{LC8['J-C3']['i'] + 1}", G8C, "{:d}")
# J-C5: the identity-carrying PERP / NONLEX components and their statistics
CAR8 = []
for i in LC8["J-C5"]["det"]:
    mm = re.match(r"    (\w+) (PERP|NONLEX):(E\d) l=(\d+) carries identity \(phi\(NONE\) " + CIR + r" \(>= 0\.3, lower > 0\)\): (.*)$", S8C[i])
    assert mm, S8C[i]
    k, inst, l = mm[1], f"{mm[2]}:{mm[3]}", int(mm[4])
    q = mm.groups()[4:7]
    smac3(f"JcFivePhi{zw8(inst)}{DW8[l]}{K8[k]}", "S8C", i, q, f"J-C5: phi(NONE) of the identity-carrying {zlab8(inst)} at l = {l}, {SH8[k]}", G8C)
    sts = [] if mm[8].startswith("no statistic") else re.findall(STC8, mm[8])
    assert mm[8].startswith("no statistic") or sts, S8C[i]
    lam8("J-C5", k, inst, sts, i)
    CAR8.append(dict(k=k, Z=inst, l=l, phi=q, n=len(sts)))
amac("JcFiveCarriers", len(CAR8), "J-C5: identity-carrying (phi(NONE) >= 0.3, lower bound > 0) PERP or NONLEX components of an effective E1-E4 edit", SRCC + " J-C5", G8C, "{:d}")
amac("JcFiveCarriersStat", sum(c["n"] > 0 for c in CAR8), "J-C5: those with a lambda statistic", SRCC + " J-C5", G8C, "{:d}")
amac("JcFiveNstat", sum(c["n"] for c in CAR8), "J-C5: lambda statistics of the carrying components (one pooled test)", SRCC + " J-C5", G8C, "{:d}")
amac("JcFiveNeq", sum(LAMP8[x]["eq"] for x in LAMP8 if LAMP8[x]["line"] == "J-C5"), "J-C5: of those, equivalent", SRCC + " J-C5", G8C, "{:d}")
rng8("JcFivePhi", [c["phi"][0] for c in CAR8], "J-C5: phi(NONE) of the 12 identity-carrying components", SRCC + " J-C5", G8C)
rng8("JcFiveLam", [LAMP8[x]["t"][0] for x in LAMP8 if LAMP8[x]["line"] == "J-C5"], "J-C5: lambda of the 12 statistics of the carrying components", SRCC + " J-C5", G8C)
# J-C-BOUND: the pooled lambda-bar of each gated-in E5 sub-family and of E2 at the same cells
BND8 = {}
for i in LC8["J-C-BOUND"]["det"]:
    mm = re.match(r"    (\w+) (E5\w+): pooled lambda_E5 " + CIR + r" \(<= log 0\.5 = -0\.693; H0: >= log 0\.8 rejected, upper < -0\.223\) over (\d+) statistics; "
                  r"E2 at the same cells " + CIR + r" \(>= log 0\.8\) -> (met|not met)$", S8C[i])
    if not mm:
        mm = re.match(r"    (\w+) (E5\w+): no lambda statistic at a gated-in depth", S8C[i])
        assert mm, S8C[i]
        BND8[mm[1], mm[2]] = None
        continue
    k, Z = mm[1], mm[2]
    sw_ = ZW8[Z][5:]
    BND8[k, Z] = dict(i=i, pool=mm.groups()[2:5], n=int(mm[6]), e2=mm.groups()[6:9], v=mm[10])
    smac3(f"JcBound{sw_}{K8[k]}", "S8C", i, BND8[k, Z]["pool"], f"J-C-BOUND: pooled lambda-bar_E5 of {zlab8(Z)} over its statistics at the gated-in depths (<= log 0.5 point, upper 95 % bound < log 0.8 needed), {SH8[k]}", G8C)
    smac(f"JcBoundN{sw_}{K8[k]}", "S8C", i, mm[6], f"J-C-BOUND: lambda statistics of {zlab8(Z)}, {SH8[k]}", G8C)
    smac3(f"JcBoundEtwo{sw_}{K8[k]}", "S8C", i, BND8[k, Z]["e2"], f"J-C-BOUND: E2's pooled lambda-bar over the same statistics (>= log 0.8 point needed), the control of {zlab8(Z)}, {SH8[k]}", G8C)
_bv = [b["v"] for b in BND8.values() if b]
amac("JcBoundMet", _bv.count("met"), "J-C-BOUND: sub-families (model x E5 sub-family) met (Mistral-7B E5-NL, Llama-3.1-8B E5-SYN)", SRCC + " J-C-BOUND", G8C, "{:d}")
amac("JcBoundVerd", len(_bv), "J-C-BOUND: sub-families with a verdict", SRCC + " J-C-BOUND", G8C, "{:d}")
amac("JcBoundNoStat", sum(b is None for b in BND8.values()), "J-C-BOUND: sub-families with no lambda statistic at a gated-in depth (no verdict)", SRCC + " J-C-BOUND", G8C, "{:d}")
# J-C-READa: the gate, KO(H*), the random-set mean and ID_K per counted (model, family); J-C-READb's E3 and E4a rows
RDP8 = {}
for i in LC8["J-C-READa"]["det"]:
    s_ = S8C[i]
    mm = re.match(r"    (\w+): gate natural KO\(H\*\) " + CIR + r" \(>= 0\.7\); ID_K\^nat " + CIR + r" \(>= 2\.0\) -> MET$", s_)
    if mm:
        k = mm[1]
        smac3(f"JcReadaKONat{K8[k]}", "S8C", i, mm.groups()[1:4], f"J-C-READa gate: KO(H*) of the natural donor at P1, l = 7 (>= 0.7 needed), {SH8[k]}", G8C)
        RDP8[k, "nat", "H"] = mm.groups()[1:4]
        smac3(f"JcReadaIDKNat{K8[k]}", "S8C", i, mm.groups()[4:7], f"J-C-READa gate: the natural key read ID_K^nat[all] (nats; >= 2 needed), {SH8[k]}", G8C)
        continue
    mm = re.match(r"    (\w+) (E[12]): KO\(H\*\) " + CIR + r" \(>= 0\.7, H0: <= 0\.5 rejected\); random mean " + CIR + r" \(<= 0\.25\); ID_K ([+-]\d+\.\d\d) -> NOT MET$", s_)
    if mm:
        k, Z = mm[1], mm[2]
        smac3(f"JcReadaKO{zw8(Z)}{K8[k]}", "S8C", i, mm.groups()[2:5], f"J-C-READa: KO(H*) of {zlab8(Z)}: share of its key read removed when the reader heads H* see B's key at the options (>= 0.7 and lower bound > 0.5 needed), {SH8[k]}", G8C)
        smac3(f"JcReadaRand{zw8(Z)}{K8[k]}", "S8C", i, mm.groups()[5:8], f"J-C-READa: mean KO over the three size-matched random head sets of blocks > 7 (<= 0.25 needed: the line fails here), {zlab8(Z)}, {SH8[k]}", G8C)
        smac(f"JcReadaIDK{zw8(Z)}{K8[k]}", "S8C", i, mm[9], f"J-C-READa: the edit's key read ID_K[all] at P1, l = 7 (nats), {zlab8(Z)}, {SH8[k]}", G8C)
        RDP8[k, Z, "H"], RDP8[k, Z, "rand"] = mm.groups()[2:5], mm.groups()[5:8]
        continue
    mm = re.match(r"    (\w+) E4a: ID_K\^E4a " + CIR + r" < 2\.0 nats \(denominator gate\) -> NOT EVALUABLE$", s_)
    if mm:
        smac3(f"JcReadbIDKEfoura{K8[mm[1]]}", "S8C", i, mm.groups()[1:4], f"J-C-READb: the key read of E4 seed 101, ID_K[all] at P1, l = 7 (nats; < 2: not counted by the line's text), {SH8[mm[1]]}", G8C)
        continue
    assert s_ == "    qwen7 E3: layer 7 fails J-C-G3: not evaluable", s_
rng8("JcReadaKO", [RDP8[k, Z, "H"][0] for k in ("qwen7", "mistral7") for Z in ("E1", "E2")], "J-C-READa: KO(H*) of E1 and E2 over both models", SRCC + " J-C-READa", G8C)
rng8("JcReadaRand", [RDP8[k, Z, "rand"][0] for k in ("qwen7", "mistral7") for Z in ("E1", "E2")], "J-C-READa: random-set mean KO of E1 and E2 over both models", SRCC + " J-C-READa", G8C)
# J-C6 (Qwen2.5-14B, block 28): the anchor J-C-G7, the CAA flip rate and nu, every arm's exchange statistics, the differences
FW6 = {"NO-MENTION": "NoMention", "QNAMES": "Qnames", "OPTIONS-AFTER": "OptAfter"}
J6 = LC8["J-C6"]["det"]
i = J6[0]
mm = re.match(r"    J-C-G7 anchor: BIND IIA\(NO-MENTION\) (\d\.\d+) \(>= 0\.95\); kappa_BIND NO-MENTION ([+-]\d\.\d+) vs (\d\.\d+), QNAMES ([+-]\d\.\d+) vs (\d\.\d+), "
              r"OPTIONS-AFTER ([+-]\d\.\d+) vs (\d\.\d+) \(within 0\.05\) -> MET$", S8C[i])
assert mm, S8C[i]
smac("JcGsevenIIA", "S8C", i, mm[1], "J-C-G7 (the J-C6 anchor): BIND IIA under NO-MENTION at block 28 (>= 0.95 needed), Qwen2.5-14B", G8C)
for j, f in enumerate(FW6):
    smac(f"JcGsevenKappa{FW6[f]}", "S8C", i, mm[2 + 2 * j], f"J-C-G7: kappa_BIND in this run, {f}, Qwen2.5-14B", G8C)
    smac(f"JcGsevenKappaRef{FW6[f]}", "S8C", i, mm[3 + 2 * j], f"J-C-G7: kappa_BIND of stage 6 (within 0.05 needed), {f}, Qwen2.5-14B", G8C)
i = J6[1]
mm = re.match(r"    CAA flip rate under NO-MENTION (\d\.\d+) \(gate >= 0\.5\); nu\(CAA vs ID@28\) NO-MENTION (\d\.\d+), QNAMES (\d\.\d+), OPTIONS-AFTER (\d\.\d+)$", S8C[i])
assert mm, S8C[i]
smac("JcSixFlip", "S8C", i, mm[1], "J-C6: the CAA edit's flip rate under NO-MENTION (its IIA; >= 0.5 needed), Qwen2.5-14B", G8C)
for j, f in enumerate(FW6):
    smac(f"JcSixNu{FW6[f]}", "S8C", i, mm[2 + j], f"J-C6: nu, the distance of the CAA edit's K/V tables from the identity patch ID@28's, {f}, Qwen2.5-14B", G8C)
J6F = {}
for i in J6[2:]:
    mm = re.match(r"      (NO-MENTION|QNAMES|OPTIONS-AFTER): ", S8C[i])
    if mm:
        f = mm[1]
        arms = re.findall(r"(CAA|BIND|ID) kappa " + CIR + r" psi_K " + NUM + r" psi_V " + NUM + r" Phi " + NUM + r" IIA (\d\.\d\d) b0 (\d\.\d+)/(\d\.\d+)", S8C[i])
        assert [a[0] for a in arms] == ["CAA", "BIND", "ID"], S8C[i]
        for a in arms:
            aw, d = {"CAA": ("Caa", "the CAA identity edit"), "BIND": ("Bind", "the binding swap BIND"), "ID": ("Id", "the identity patch ID@28")}[a[0]]
            smac3(f"JcSixKappa{aw}{FW6[f]}", "S8C", i, a[1:4], f"J-C6: kappa = psi_K / (psi_K + psi_V) of {d} at block 28, {f}, Qwen2.5-14B", G8C)
            for w, s_, dd in (("PsiK", a[4], "psi_K"), ("PsiV", a[5], "psi_V"), ("Phi", a[6], "Phi = mean margin shift of the patched run (nats)"), ("IIA", a[7], "IIA (share of pairs whose top token is the target)"),
                              ("BzeroFour", a[8], "Gate b0: mean |m(r4) - m(r1)| (nats)"), ("BzeroZero", a[9], "Gate b0: mean |m(r0) - m(B)| (nats)")):
                smac(f"JcSix{w}{aw}{FW6[f]}", "S8C", i, s_, f"J-C6: {dd} of {d}, {f}, Qwen2.5-14B", G8C)
            J6F[f, a[0]] = a[1:4]
        continue
    mm = re.match(r"        kappa_CAA ([+-]\d\.\d+) \(<= 0\.3\); kappa_BIND - kappa_CAA " + CIR + r" \(>= 0\.3; H0: <= 0 rejected\) -> met$", S8C[i])
    assert mm, S8C[i]
    J6F[f, "diff"] = mm.groups()[1:4]
    smac3(f"JcSixDiff{FW6[f]}", "S8C", i, J6F[f, "diff"], f"J-C6: kappa_BIND - kappa_CAA (paired; >= 0.3 with H0 <= 0 rejected needed), {f}, Qwen2.5-14B", G8C)
assert len(J6F) == 12, len(J6F)
# J-C-SCREEN (Qwen2.5-7B, Llama-3.1-8B): the binding sweep, the key-route onsets, the working blocks W_B, no window
SCR8 = {}
for i in LC8["J-C-SCREEN"]["det"]:
    s_ = S8C[i]
    mm = re.match(r"    (\w+): population (\d+) pairs; BIND max IIA (\d\.\d+) at block (\d+) \(screen evaluable: max IIA >= 0\.7 and n >= 50\) -> yes$", s_)
    if mm:
        k = mm[1]
        t, S = K8[k], SH8[k]
        SCR8[k] = dict(i=i)
        smac(f"JcScreenPairs{t}", "S8C", i, mm[2], f"J-C-SCREEN: pairs of Prakash et al.'s material passing the LM filter under NO-MENTION (the population, at most 150), {S}", G8C)
        smac(f"JcScreenMaxIIA{t}", "S8C", i, mm[3], f"J-C-SCREEN: the largest IIA of the binding swap BIND over blocks (NO-MENTION), {S}", G8C)
        smac(f"JcScreenMaxBlock{t}", "S8C", i, mm[4], f"J-C-SCREEN: the block of that largest IIA, {S}", G8C)
        continue
    if s_.startswith("        BIND IIA/Phi per block: "):
        blk = re.findall(r"(\d+):(\d\.\d\d)/([+-]\d+\.\d)", s_)
        assert [int(b[0]) for b in blk] == list(range(len(blk))), s_
        SCR8[k]["blk"] = blk
        amac(f"JcScreenBlocks{t}", len(blk), f"J-C-SCREEN: blocks of the model (the BIND sweep covers every block), {S}", f"{SRCC}:{i + 1}", G8C, "{:d}")
        for b, iia, phi in blk:
            smac(f"JcScreenIIA{word(b)}{t}", "S8C", i, iia, f"J-C-SCREEN: BIND IIA when patched at block {b} (NO-MENTION), {S}", G8C)
            smac(f"JcScreenPhi{word(b)}{t}", "S8C", i, phi, f"J-C-SCREEN: BIND Phi (nats) when patched at block {b} (NO-MENTION; a working swap needs Phi >= 3 and IIA >= 0.5), {S}", G8C)
        continue
    if s_.startswith("        s_ID(OPTIONS-AFTER, l0): "):
        sid = re.findall(r"(\d+):([+-](?:\d\.\d+|nan))", s_)
        SCR8[k]["sid"] = {int(o): v for o, v in sid}
        for o, v in sid:
            if v.endswith("nan"):
                continue
            smac(f"JcScreenSid{word(o)}{t}", "S8C", i, v, f"J-C-SCREEN: s_ID(OPTIONS-AFTER) of the natural clamp from block {o} (the identity key route's share), {S}", G8C)
        continue
    mm = re.match(r"        W_B \(Phi >= 3\.0, IIA >= 0\.5\) \[([\d, ]*)\]; window \(s_ID\(l\+1\) >= 0\.4\) \[\]; l_w None$", s_)
    assert mm, s_
    wb = [int(x) for x in mm[1].split(",")]
    assert wb == [int(b) for b, iia, phi in SCR8[k]["blk"] if float(iia) >= 0.5 and float(phi) >= 3.0], (k, wb)
    for w, v, d in (("First", wb[0], "first"), ("Last", wb[-1], "last"), ("N", len(wb), "number of")):
        amac(f"JcScreenWb{w}{t}", v, f"J-C-SCREEN: {d} block(s) where the binding swap works (W_B: Phi >= 3 nats, IIA >= 0.5), {S}", f"{SRCC}:{i + 1}", G8C, "{:d}")
    _sw = [SCR8[k]["sid"][b + 1] for b in wb]
    nm(f"JcScreenSidWbMax{t}", sv(max(_sw, key=fl)), f"J-C-SCREEN: the largest s_ID(OPTIONS-AFTER, l + 1) over the working blocks l in W_B (a window needs >= 0.4), {S}", f"{SRCC}:{SCR8[k]['i'] + 3}", G8C)
    SCR8[k]["wb"] = wb
assert set(SCR8) == {"qwen7", "llama8"} and all("wb" in v for v in SCR8.values())
assert [S8C[i] for i in LC8["J-C-WIN"]["det"]] == ["    no window in any evaluable screened model"]
# the lines' unit counts (for the table): sub-combos met / with a verdict, by the line's combination rule
_u8 = {}
for code in ("J-C1", "J-C2", "J-C3", "J-C4"):
    vs = [c["v"] for (cd, k, Z, dep), c in COMB8.items() if cd == code and not c["seed"]]
    _u8[code] = (vs.count("MET"), sum(v != "NOT EVALUABLE" for v in vs), "combos (model x edit x depths)")
_u8["J-C5"] = (0, 1, "one pooled test over every statistic of every carrying component")
_u8["J-C-BOUND"] = (_bv.count("met"), len(_bv), "sub-families (model x E5 sub-family)")
_u8["J-C-READa"] = (0, sum(1 for x in RDP8 if x[1] in ("E1", "E2") and x[2] == "H"), "counted (model, family) pairs")
_u8["J-C-READb"] = (0, 0, "counted (model, family) pairs (E3: J-C-G3; E4a: below the 2-nat denominator)")
_u8["J-C6"] = (sum(1 for f in FW6 if (f, "diff") in J6F), len(FW6), "formats")
_u8["J-C-SCREEN"] = (len(SCR8), len(SCR8), "evaluable screened models without an overlap window")   # every window list is [] (asserted above)
_u8["J-C-WIN"] = (0, 0, "screened models with a window")
for code, L in LC8.items():
    met_, ev_, unit = _u8[code]
    if f"Jc{LWC8[code]}Met" in macros:   # J-C-BOUND: registered above (JcBoundMet)
        assert macros[f"Jc{LWC8[code]}Met"] == str(met_), code
    else:
        amac(f"Jc{LWC8[code]}Met", met_, f"{code}: {unit} met", f"{SRCC}:{L['i'] + 1}", G8C, "{:d}")
    amac(f"Jc{LWC8[code]}Eval", ev_, f"{code}: {unit} with a verdict", f"{SRCC}:{L['i'] + 1}", G8C, "{:d}")
assert {c: _u8[c][:2] for c in _u8} == {"J-C1": (0, 4), "J-C2": (1, 2), "J-C3": (0, 0), "J-C4": (0, 2), "J-C5": (0, 1), "J-C-BOUND": (2, 6), "J-C-READa": (0, 4),
                                        "J-C-READb": (0, 0), "J-C6": (3, 3), "J-C-SCREEN": (2, 2), "J-C-WIN": (0, 0)}, _u8
# the HEADLINE (C-6): the Section-5 outcome letter
sm8("JcHeadline", "S8C", HC8, r"outcome \((\w)\): ", "HEADLINE (C-6): the Section-5 outcome letter of part C, (e) 'Failed': even near-natural steering vectors depart from the natural channel ratio", G8C)
sm8("JcHeadlineWord", "S8C", HC8, r"outcome \(\w\): (\w+) \(", "HEADLINE: the outcome's word as printed", G8C)
assert macros["JcHeadline"] == "e"
# reported: the pi(t) diagnostic, the lexical-code reading, JC4 depth tracking
for k in ("qwen7", "mistral7"):
    for l, lw in DW8.items():
        i = find(S8C, f"  pi(t) diagnostic {k} l={l} ", start=RC8, end=SC8)
        sm83(f"JcPit{lw}{K8[k]}", "S8C", i, r"mean d_pi\(t\) - d_t " + CIR,
             f"pi(t) diagnostic (reported): mean d_pi(t) - d_t in E4's key-only rows under LETTER, l = {l}{' (the LETTER cell fails J-C-G4: do not cite)' if l >= 11 else ''}, {SH8[k]}", G8C)
LEXP8 = {}
for k in MC8:
    for Z in ("E1", "E2", "E3", "E4"):
        for l, lw in DW8.items():
            hits = find(S8C, f"  lexical-code reading {k} {Z} l={l}:", start=RC8, end=SC8, many=True)
            assert len(hits) == (1 if Z in FAMC8[k] and EFF8C[k, Z, l] else 0), (k, Z, l)
            if not hits:
                continue
            i = hits[0]
            g_ = grab(S8C, i, r"phi ([+-]\d\.\d+), PAR ([+-]\d\.\d+), LEX ([+-]\d\.\d+) -> ")
            for w, s_, d in zip(("Phi", "Par", "Lex"), g_, ("phi of the edit", "phi of its PAR component (the projection on the story's natural displacement)",
                                                         "phi of its LEX component (the projection on the 5-dimensional English lexical span)")):
                smac(f"JcLex{w}{zw8(Z)}{lw}{K8[k]}", "S8C", i, s_, f"lexical-code reading (reported): {d}, under NONE, {zlab8(Z)}, l = {l}, {SH8[k]}", G8C)
            LEXP8[k, Z, l] = dict(i=i, phi=g_[0], par=g_[1], lex=g_[2], code="acts through the natural lexical code" in S8C[i], counted="not counted" not in S8C[i])
assert all(v["code"] == (Z != "E4") for (k, Z, l), v in LEXP8.items())
_e4lex = [v for (k, Z, l), v in LEXP8.items() if Z == "E4"]
rng8("JcLexParEfour", [v["par"] for v in _e4lex], "lexical-code reading: phi of E4's PAR component at the depths where E4 is effective", SRCC + " REPORTED", G8C)
rng8("JcLexLexEfour", [v["lex"] for v in _e4lex], "lexical-code reading: phi of E4's LEX component at the depths where E4 is effective", SRCC + " REPORTED", G8C)
amac("JcLexParShareEfourMax", max(fl(v["par"]) / fl(v["phi"]) for v in _e4lex), "lexical-code reading: the largest phi(PAR)/phi of E4 (the rule needs >= 0.8)", SRCC + " REPORTED", G8C, "{:.2f}")
amac("JcLexLexShareEfourMax", max(fl(v["lex"]) / fl(v["phi"]) for v in _e4lex), "lexical-code reading: the largest phi(LEX)/phi of E4", SRCC + " REPORTED", G8C, "{:.2f}")
amac("JcLexCodeShareNearMin", min(max(fl(v["par"]), fl(v["lex"])) / fl(v["phi"]) for (k, Z, l), v in LEXP8.items() if Z in ("E1", "E2")),
     "lexical-code reading: the smallest max(phi(PAR), phi(LEX))/phi of E1 and E2 over models and depths (>= 0.8: they act through the lexical code)", SRCC + " REPORTED", G8C, "{:.2f}")
JC4P8 = []
for i in range(find(S8C, "  JC4 depth tracking", start=RC8, end=SC8) + 1, SC8):
    mm = re.match(r"    (\w+) (E\d) (\w+): delta_sigma ([+-]\d\.\d+) delta_kappa ([+-]\d\.\d+) -> (holds|does not hold)$", S8C[i])
    if not mm:
        break
    k, Z, f = mm[1], mm[2], mm[3]
    smac(f"JcJcfourDs{zw8(Z)}{FWC8[f]}{K8[k]}", "S8C", i, mm[4], f"JC4 depth tracking (reported): delta_sigma = sigma({f}, 3) - sigma({f}, 15), {zlab8(Z)} pair, {SH8[k]}", G8C)
    smac(f"JcJcfourDk{zw8(Z)}{FWC8[f]}{K8[k]}", "S8C", i, mm[5], f"JC4 depth tracking (reported): delta_kappa = kappa({f}, 3) - kappa({f}, 15) of {zlab8(Z)}, {SH8[k]}", G8C)
    JC4P8.append((mm[4], mm[5], mm[6]))
sm8("JcJcfourHold", "S8C", i, r"JC4: (\d+) of (\d+) qualifying", "JC4: qualifying (model, edit, format) pairs whose delta_kappa tracks delta_sigma", G8C, 0)
sm8("JcJcfourN", "S8C", i, r"JC4: (\d+) of (\d+) qualifying", "JC4: qualifying pairs (delta_sigma >= 0.25, both cells evaluable, the edit effective at l = 3 and 15)", G8C, 1)
assert len(JC4P8) == int(macros["JcJcfourN"]) and all(h == "holds" for _, _, h in JC4P8)
amac("JcJcfourAbsMax", max(abs(fl(a) - fl(b)) for a, b, _ in JC4P8), "JC4: the largest |delta_kappa - delta_sigma| over the qualifying pairs", SRCC + " REPORTED", G8C, "{:.3f}")
# summary: classes (part C has no class-L line: its Brier is nan and not registered), the R tally (S8X), the counts, Holm
classes8(S8C, SC8, XC8, "Jc", G8C)
i = find(S8C, "  met rate among R account lines with a verdict:", start=SC8, end=XC8)
for j, (w, d) in enumerate((("Met", "met"), ("Verd", "with a verdict"), ("Exp", "expected met"))):
    sm8(f"JxRisky{w}C", "S8C", i, r"(\d+) of (\d+) \(expected (\d+\.\d+)\)", f"part C: risky (class R) account lines {d}", G8X, j)
_vc = [L["v"] for L in LC8.values()]
for w, v in (("Met", "MET"), ("NotMet", "NOT MET"), ("NotEval", "NOT EVALUABLE")):
    amac(f"JcCount{w}", _vc.count(v), f"part C: lines {v} (counted from the verdict lines)", SRCC + " PREDICTIONS", G8C, "{:d}")
amac("JcCountLines", len(_vc), "part C: lines in all (every one an account line)", SRCC + " PREDICTIONS", G8C, "{:d}")
assert (macros["JcCountMet"], macros["JcCountNotMet"], macros["JcCountNotEval"], macros["JcCountLines"]) == ("2", "6", "3", "11")
assert sum(int(macros[f"JcClass{c}Met"]) for c in "LMR") == int(macros["JcCountMet"])
i = find(S8C, "  Holm sensitivity (analysis/stage8_holm.py", start=SC8, end=XC8)
sm8("JcHolmN", "S8C", i, r"over the (\d+) interval components", "Holm sensitivity: interval components of part C's R-class lines with a verdict", G8C)
_hch = 0
for code in ("J-C3", "J-C4", "J-C5", "J-C-BOUND", "J-C-READb", "J-C-SCREEN", "J-C-WIN"):
    i = find(S8C, f"    {code}: ", start=SC8, end=XC8)
    sm8(f"JcHolmN{LWC8[code]}", "S8C", i, r": (\d+) components;", f"Holm sensitivity: components of {code}", G8C)
    mm = re.search(r"(\d+) decision\(s\) change under Holm", S8C[i])
    _hch += int(mm[1]) if mm else 0
    for j, p_ in enumerate(re.findall(r"\(p (\S+?);", S8C[i])):
        smac(f"JcHolmP{LWC8[code]}{word(j + 1)}", "S8C", i, p_, f"Holm sensitivity: one-sided p of the {('first', 'second', 'third')[j]} {code} component whose decision changes (the interval rule rejects, Holm does not)", G8C)
    assert "verdict unchanged" in S8C[i], S8C[i]
amac("JcHolmChanged", _hch, "Holm sensitivity: decisions that change (no verdict changes)", SRCC + " SUMMARY", G8C, "{:d}")
# exploratory: kappa against sigma at l = 3 and 7, nu and the cosines, O4 flip rates, L-scored lambda, the E4 fits
KS8 = {}
for k in MC8:
    t, S = K8[k], SH8[k]
    h = find(S8C, f"-- {k}: kappa against sigma per cell", start=XC8)
    e = next(j for j in range(h + 1, len(S8C)) if S8C[j].startswith("--"))
    for l, lw in DW8.items():
        for f in FWC8:
            hits = find(S8C, f"   {f}@{l}: sigma ", start=h, end=e, many=True)
            if not hits:
                assert f == "LETTER" and l >= 11, (k, f, l)
                continue
            i = hits[0]
            sg = grab(S8C, i, r"sigma ([+-]\d\.\d+); kappa")[0]
            kp = dict(re.findall(r"(E\w+) ([+-]\d+\.\d\d)", S8C[i].split("; kappa ")[1]))
            KS8[k, f, l] = (sg, kp)
            if l not in (3, 7):
                continue
            smac(f"JcSigma{FWC8[f]}{lw}{t}", "S8C", i, sg, f"sigma = kappa^nat, the natural key share of the identity effect (descriptive), {f}, l = {l}, {S}", G8C)
            for Z, s_ in kp.items():
                smac(f"JcKappa{zw8(Z)}{FWC8[f]}{lw}{t}", "S8C", i, s_, f"kappa = mean ID_K / (mean ID_K + mean ID_V) of {zlab8(Z)} (descriptive; compare sigma), {f}, l = {l}, {S}", G8C)
        i = find(S8C, f"   l={l}: ", start=h, end=e)
        for Z, nu, ck, cv, a_ in re.findall(r"([\w:]+) nu (\d\.\d\d) cosK (-?\d\.\d\d) cosV (-?\d\.\d\d) a (-?\d\.\d\d)", S8C[i]):
            for w, s_, d in (("Nu", nu, "nu, the distance of its K/V tables from the target's natural ones (relative to the natural displacement)"),
                             ("CosK", ck, "cosine of its key displacement with the natural one"), ("CosV", cv, "cosine of its value displacement with the natural one"),
                             ("CosResid", a_, "cosine of its residual displacement at (p, l) with the natural one")):
                smac(f"Jc{w}{zw8(Z)}{lw}{t}", "S8C", i, s_, f"table statistics (exploratory; means over stories and targets): {d}, {zlab8(Z)}, l = {l}, {S}", G8C)
    h = find(S8C, f"-- {k}: greedy-generation flip rates", start=XC8)
    for f in ("NONE", "P1"):
        i = find(S8C, f"   {f}: nat ", start=h, end=h + 3)
        for Z, s_ in re.findall(r"(\w+) (\d\.\d\d)", S8C[i].split(": ", 1)[1]):
            smac(f"JcOfour{FWC8[f]}{zw8(Z)}{t}", "S8C", i, s_, f"O4 (exploratory): greedy-generation flip rate of the KV row at l = 7, first 40 stories, {f}, {'the natural write' if Z == 'nat' else zlab8(Z)}, {S}", G8C)
    h = find(S8C, f"-- {k}: L-scored lambda", start=XC8)
    for Z in ("E1", "E2", "E3", "E4"):
        if Z not in FAMC8[k]:
            continue
        i = find(S8C, f"   {Z}: mean lambda under L", start=h, end=h + 5)
        sm8(f"JcLscore{zw8(Z)}{t}", "S8C", i, r"under L ([+-]\d\.\d+)", f"L-scored lambda (exploratory): mean over the law statistics of {zlab8(Z)}, {S}", G8C)
        sm8(f"JcEscore{zw8(Z)}{t}", "S8C", i, r"vs E ([+-]\d\.\d+)", f"the same mean under the preregistered E score, {zlab8(Z)}, {S}", G8C)
_fit = {"first": [], "last": [], "flip": []}
for k in ("qwen7", "mistral7"):
    i = find(S8C, f"-- {k}: E4 fits ", start=XC8)
    fits = re.findall(r"(\d+)_(\d+) (\w+) loss (\d+\.\d\d)->(\d+\.\d\d) THOLD flip (\d\.\d\d)", S8C[i])
    assert len(fits) == 8, S8C[i]
    for x in fits:
        _fit["first"].append(x[3]), _fit["last"].append(x[4]), _fit["flip"].append(x[5])
rng8("JcDasLossFirst", _fit["first"], "E4 fits (16: 2 models x 4 depths x 2 seeds): mean loss over the first 50 steps", SRCC + " EXPLORATORY", G8C)
rng8("JcDasLossLast", _fit["last"], "E4 fits: mean loss over the last 50 steps", SRCC + " EXPLORATORY", G8C)
rng8("JcDasFlip", _fit["flip"], "E4 fits: held-out (THOLD) flip rate under NONE", SRCC + " EXPLORATORY", G8C)
# recomputed (not printed): through the committed part-C scorer's own classes. The same code first reproduces what the
# scorer prints (asserted: every combo's text, every carrying component's phi and statistics, every J-C-BOUND pooled value,
# the J-C-READa rows, the lexical-code reading), then computes the values the outcome record marks "(recomputed)" and the
# per-statistic lambdas the score file does not print (E3, E5, Qwen2.5-7B E2 at l = 11), each registered with rmac.
from stage8c_parts import law as C8LW  # noqa: E402
from stage8c_parts import readers as C8RD  # noqa: E402
from stage8c_parts.stats import log_ratio as c8lr, mean_q as c8mq, tost as c8tost  # noqa: E402
_c8mean = C8LW.Model.mean


def _c8mean_memo(self, f, l, Z, C, field="E"):
    """Model.mean is a pure function of its arguments (the per-story contrasts and the model's fixed bootstrap weights); a
    memo of its results changes no number and saves the repeated 10,000-resample means (about 2 minutes)."""
    memo = self.__dict__.setdefault("_mean_memo", {})
    if (f, l, Z, C, field) not in memo:
        memo[f, l, Z, C, field] = _c8mean(self, f, l, Z, C, field)
    return memo[f, l, Z, C, field]


C8LW.Model.mean = _c8mean_memo
C8M = {k: C8LW.Model(json.load(open(ROOT / f"results/gpu_stage8c/eval/{k}.json"))) for k in MC8}
SRCE8 = {k: f"results/gpu_stage8c/eval/{k}.json" for k in MC8}


def c8cells(code):
    """(K cell format, V cell format) of a lambda statistic."""
    kind = code.split("(")[0]
    if kind == "W":
        f = code[2:-1].split(",")[0]
        return f, f
    return {"A_L": ("LETTER", "NONE"), "A_P": ("P1", "NONE")}[kind]


def c8tab(m, l, Z, key):
    """a stored table statistic of instance Z at depth l: the mean over stories and both targets (E4: and both seeds)."""
    head, _, base = Z.rpartition(":")
    zs = [f"{head + ':' if head else ''}{b}" for b in (("E4a", "E4b") if base == "E4" else (base,))]
    v = [np.mean([m.S[s]["stats"][str(l)][f"{z}|{t_}"][key] for t_ in "SX"]) for z in zs for s in range(m.n)]
    return float(np.mean(v))


# (1) the law lines: every printed combo is reproduced by law.law (its printed text); the statistics of the combos without a
# verdict (Qwen2.5-7B E2 at l = 11) and of E3 (post hoc: J-C-G3 failed) are registered as recomputed
for (code, k, Z, dep), c in COMB8.items():
    met, txt, st, pat = C8LW.law(C8M[k], Z, list(dep), "edit")
    if c["v"] == "NOT EVALUABLE":
        assert met is None and len(st) == c["n"], (code, k, Z)
        if st:
            lam8(f"{code} (combo not evaluable: fewer than 3 statistics; recomputed)", k, Z, st, None, rec=True, src=SRCE8[k])
        continue
    assert f": {txt} -> {c['v']}" in S8C[c["i"]], (code, k, Z, txt[:80])
for dep, w in (([3], "LayerThree"), ([3, 7, 11, 15], "All")):
    met, txt, st, pat = C8LW.law(C8M["qwen7"], "E3", dep, "edit")
    assert met is False, (dep, txt)
    pool = c8mq([s["q"] for s in st])
    d = f"post hoc (J-C3 is not evaluable: J-C-G3): the law for E3 at Qwen2.5-7B admitting {'layer 3 only (the one layer whose FVE at p is >= published - 0.05)' if dep == [3] else 'all four layers'}"
    rmac(f"JcThreePosthoc{w}Neq", sum(s["equivalent"] for s in st), d + ": equivalent statistics", SRCE8["qwen7"], G8C, "{:d}")
    rmac(f"JcThreePosthoc{w}Nstat", len(st), d + ": lambda statistics", SRCE8["qwen7"], G8C, "{:d}")
    rmac390(f"JcThreePosthoc{w}Pool", (pool.pt, pool.lower(0.9), pool.upper(0.9)), d + ": pooled lambda-bar", SRCE8["qwen7"], G8C)
    if dep != [3]:
        lam8("post hoc E3 (J-C-G3 not met; recomputed)", "qwen7", "E3", st, None, rec=True, src=SRCE8["qwen7"])
_e3c = [(l, c_) for l in DW8 for c_ in ("PERP", "NONLEX") if C8M["qwen7"].effective(l, f"{c_}:E3", "carry")[0]]
rmac("JcThreePosthocCarriers", len(_e3c), "post hoc: PERP or NONLEX components of E3 that carry identity (phi(NONE) >= 0.3, lower bound > 0) at any layer", SRCE8["qwen7"], G8C, "{:d}")
# the outcome record's numbers (Sensitivities, J-C-G3) are reproduced
assert (macros["JcThreePosthocLayerThreeNeq"], macros["JcThreePosthocLayerThreeNstat"], macros["JcThreePosthocLayerThreePool"], macros["JcThreePosthocLayerThreePoolLo"],
        macros["JcThreePosthocLayerThreePoolHi"]) == ("1", "4", "-0.170", "-0.218", "-0.129")
assert (macros["JcThreePosthocAllNeq"], macros["JcThreePosthocAllNstat"], macros["JcThreePosthocAllPool"], macros["JcThreePosthocAllPoolLo"],
        macros["JcThreePosthocAllPoolHi"], macros["JcThreePosthocCarriers"]) == ("1", "10", "-0.640", "-0.772", "-0.542", "0")
# (2) J-C5: every carrying component's phi and statistics are reproduced
JC5INTER8 = []   # J-C5 statistics with an interaction cell (iota > 0.5), from the scorer's own stats_of
for c in CAR8:
    m = C8M[c["k"]]
    ok, ctxt = m.effective(c["l"], c["Z"], "carry")
    st, skip = m.stats_of(c["Z"], [c["l"]], gate=None, formats=C8LW.COMP_FORMATS)
    i = find(S8C, f"    {c['k']} {c['Z']} l={c['l']} carries identity", start=LC8["J-C5"]["i"], end=LC8["J-C-BOUND"]["i"])
    assert ok and f"({ctxt})" in S8C[i] and len(st) == c["n"], (c, ctxt)
    assert all(f"{s['code']} {s['q'].pt:+.2f} [{s['q'].lower(0.9):+.2f},{s['q'].upper(0.9):+.2f}]" + ("" if s["equivalent"] else "*") in S8C[i] for s in st), S8C[i]
    # the scorer's J-C5 lines print '*' but not '(iota)': the interaction flag of each statistic comes from the same scorer
    # call (a statistic with an interaction cell is never equivalent, so it carries '*' too)
    for s in st:
        x = LAMP8[c["k"], c["Z"], s["code"]]
        assert not x["inter"] and (s["interaction"] == [] or not x["eq"]), (c, s["code"])
        x["inter"] = bool(s["interaction"])
        if x["inter"]:
            JC5INTER8.append((c["k"], c["Z"], s["code"]))
            # the index description of the statistic's macros says so too (their values are unchanged)
            _nm5 = f"JcLam{zw8(c['Z'])}{sw8(s['code'])}{K8[c['k']]}"
            for i_, r in enumerate(REG):
                if r[1] in (_nm5, _nm5 + "Lo", _nm5 + "Hi"):
                    assert "; not equivalent, " in r[3], r
                    REG[i_] = (r[0], r[1], r[2], r[3].replace("; not equivalent, ", "; not equivalent (iota > 0.5 in a cell: a departure by the interaction rule), "), r[4])
# (3) J-C-BOUND: each sub-family's statistics (recomputed; their pooled value and E2's reproduce the printed ones)
for (k, Z), b in BND8.items():
    m = C8M[k]
    st, skip = m.stats_of(Z, m.depths, gate="carry")
    assert bool(st) == (b is not None), (k, Z)
    if not st:
        continue
    codes = {(s["code"], s["l"]) for s in st}
    st2 = [s for s in m.stats_of("E2", sorted({s["l"] for s in st}), gate="edit")[0] if (s["code"], s["l"]) in codes]
    assert c8mq([s["q"] for s in st]).txt() == "{} [{},{}]".format(*b["pool"]), (k, Z)   # printed strings (signs kept)
    assert c8mq([s["q"] for s in st2]).txt() == "{} [{},{}]".format(*b["e2"]), (k, Z)
    lam8("J-C-BOUND (statistics of the pooled lambda-bar_E5; recomputed)", k, Z, st, None, rec=True, src=SRCE8[k])
# (4) the lexical-code reading: phi, PAR and LEX reproduce the printed values; PERP and NONLEX (not printed) are recomputed
LEXC8 = {}
for (k, Z, l), v in LEXP8.items():
    m, lw = C8M[k], DW8[l]
    for c_, w in (("", "phi"), ("PAR:", "par"), ("LEX:", "lex")):
        assert f"{m.psi('NONE', l, c_ + Z, 'KV').pt:+.3f}" == v[w], (k, Z, l, c_)   # printed strings (signs kept)
    for c_ in ("PERP", "NONLEX"):
        LEXC8[k, Z, l, c_] = m.psi("NONE", l, f"{c_}:{Z}", "KV").pt
        rmac(f"JcLex{c_.capitalize()}{zw8(Z)}{lw}{K8[k]}", LEXC8[k, Z, l, c_], f"lexical-code reading: phi of the {c_} component of {zlab8(Z)} under NONE (not printed), l = {l}, {SH8[k]}",
             SRCE8[k], G8C)
for c in CAR8:
    assert f"{LEXC8[c['k'], c['Z'].split(':')[1], c['l'], c['Z'].split(':')[0]]:+.3f}" == c["phi"][0], c
# (5) J-C-READa and J-C-READb: the readers rows (the scorer's Readers); the natural donor's random-set KO, E4a and E3
RDR8 = {}
for k in ("qwen7", "mistral7"):
    t, S = K8[k], SH8[k]
    srcf = f"results/gpu_stage8c/readers/{k}.json"
    J_ = json.load(open(ROOT / srcf))
    R_ = C8RD.Readers(J_)
    a_, b_ = LC8["J-C-READa"]["i"], LC8["J-C-READb"]["i"]
    assert R_.gate()[1] in S8C[find(S8C, f"    {k}: gate ", start=a_, end=b_)]
    for D in ("E1", "E2"):
        assert R_.family(D)[1] in S8C[find(S8C, f"    {k} {D}: KO(H*) ", start=a_, end=b_)], (k, D)
    kn = R_.ko("nat", "H")
    rn = c8mq([R_.ko("nat", f"r{j}") for j in range(3)])
    rmac3(f"JcReadaRandNat{t}", (rn.pt, *rn.ci()), f"J-C-READa: the natural donor's mean KO over the three random head sets (the line's random-set control applied to the natural value; not printed), {S}", srcf, G8C)
    for j in range(3):
        rmac(f"JcReadaSet{word(j)}Nat{t}", R_.ko("nat", f"r{j}").pt, f"J-C-READa: KO of random head set r{j} for the natural donor, {S}", srcf, G8C)
    for D in [x for x in J_["donors"] if x in ("E3", "E4a")]:
        kq = R_.ko(D, "H")
        rq = c8mq([R_.ko(D, f"r{j}") for j in range(3)])
        dd = "E4 seed 101 (not counted: its key read is below 2 nats)" if D == "E4a" else "E3 (not evaluable: J-C-G3; post hoc)"
        rmac3(f"JcReadbKO{zw8(D)}{t}", (kq.pt, *kq.ci()), f"J-C-READb: KO(H*) of {dd}; under the common part's evaluability rule this counts against the line (< 0.7: NOT MET), {S}", srcf, G8C)
        rmac3(f"JcReadbRand{zw8(D)}{t}", (rq.pt, *rq.ci()), f"J-C-READb: random-set mean KO of {dd}, {S}", srcf, G8C)
        if D == "E3":
            iq = R_.m("E3", "all")
            rmac3(f"JcReadbIDK{zw8(D)}{t}", (iq.pt, *iq.ci()), f"J-C-READb: the key read ID_K[all] of {dd} (nats), {S}", srcf, G8C)
        RDR8[k, D] = (kq, rq)
    RDR8[k, "nat"] = (kn, rn)
    pair = {}
    for D in ("E1", "E2"):
        rz = c8mq([R_.ko(D, f"r{j}") for j in range(3)])
        pair[D, "H"], pair[D, "rand"] = R_.ko(D, "H").pt - kn.pt, rz.pt - rn.pt
    RDR8[k, "pair"] = pair
    p_ = J_["provenance"]
    n_h = len(p_["H_star"])
    assert all(len(r) == n_h for r in p_["rand"]) and all(b >= 8 for r in p_["rand"] for b, h in r), k
    rmac(f"JcReadaSetSize{t}", n_h, f"J-C-READ: heads in H*_{{>7}} (the stage-6 top-k* readers in blocks > 7) and in each size-matched random set, {S}", srcf, G8C, "{:d}")
    rmac(f"JcReadaPool{t}", (p_["n_layers"] - 8) * p_["n_heads"], f"J-C-READ: heads of blocks > 7, the pool the random sets are drawn from (not disjoint from H*), {S}", srcf, G8C, "{:d}")
    if k == "qwen7":
        assert p_["H_star"][0] in p_["rand"][0]   # the top-ranked reader is in random set r0
_pm = [abs(v) for k in ("qwen7", "mistral7") for (D, w), v in RDR8[k, "pair"].items() if (k, D, w) != ("mistral7", "E2", "H")]
rmac("JcReadaPairMax", max(_pm), "J-C-READa: the largest |edit - natural| difference of KO(H*) or of the random-set mean KO over E1 and E2 at both models, except Mistral-7B E2 on H* (JcReadaPairEtwoHMistralSeven)",
     "results/gpu_stage8c/readers/", G8C)
rmac("JcReadaPairEtwoHMistralSeven", RDR8["mistral7", "pair"]["E2", "H"], "J-C-READa: KO(H*) of E2 minus the natural donor's, Mistral-7B", "results/gpu_stage8c/readers/mistral7.json", G8C)
# the outcome record's numbers (Sensitivities, J-C-READa and J-C-READb) are reproduced
assert tuple(macros[f"JcReadaRandNatQwenSeven{s_}"] for s_ in ("", "Lo", "Hi")) == ("0.307", "0.281", "0.334")
assert tuple(macros[f"JcReadaRandNatMistralSeven{s_}"] for s_ in ("", "Lo", "Hi")) == ("0.283", "0.209", "0.391")
assert (macros["JcReadbKOEfouraQwenSeven"], macros["JcReadbKOEfouraMistralSeven"]) == ("0.590", "0.678")
assert f"{fl(macros['JcReadaSetZeroNatQwenSeven']):.2f}" == "0.66" and f"{fl(macros['JcReadaPairMax']):.2f}" == "0.01" and f"{fl(macros['JcReadaPairEtwoHMistralSeven']):.2f}" == "-0.03"
assert [macros[f"JcReada{w}{t}"] for w in ("SetSize", "Pool") for t in ("QwenSeven", "MistralSeven")] == ["32", "43", "560", "768"]
# (6) where J-C1 fails: the natural key channel behind every non-equivalent E1 / E2 statistic of J-C1 and J-C2 (both lines'
# combos, the not-evaluable Qwen2.5-7B E2 {11, 15} included), the depth-3 statistics, a channel floor, the L score
NE8, L38 = [], []
for (code, k, Z, dep), c in COMB8.items():
    if code not in ("J-C1", "J-C2") or c["seed"]:
        continue
    m = C8M[k]
    for s in m.stats_of(Z, list(dep), "edit")[0]:
        if s["l"] == 3:
            L38.append(abs(s["q"].pt))
        if s["equivalent"]:
            continue
        fk, fv = c8cells(s["code"])
        l = s["l"]
        Kn, KVn = m.mean(fk, l, "nat", "K").pt, m.mean(fk, l, "nat", "KV").pt
        NE8.append(dict(k=k, Z=Z, code=s["code"], l=l, fk=fk, psiK=m.psi(fk, l, Z, "K").pt, psiV=m.psi(fv, l, Z, "V").pt, idK=Kn, share=Kn / KVn,
                        idV=m.mean(fv, l, "nat", "V").pt))
_src1 = "results/gpu_stage8c/eval/ (J-C1 and J-C2 combos)"
rmac("JcOneNeN", len(NE8), "J-C1 / J-C2: non-equivalent lambda statistics of E1 and E2 (in the combos of both lines; 2 of them in the not-evaluable Qwen2.5-7B E2 {11, 15} combo)", _src1, G8C, "{:d}")
for w, key, d, f_ in (("PsiK", "psiK", "psi_K of the edit in the statistic's K cell", "{:.3f}"), ("PsiV", "psiV", "psi_V of the edit in its V cell", "{:.3f}"),
                      ("IDK", "idK", "natural ID_K in the K cell (nats)", "{:.2f}"), ("IDV", "idV", "natural ID_V in the V cell (nats; the smallest, 3.3, is Llama-3.1-8B E2 W(P1, 7): the outcome record's '5.8-17 nats' leaves it out)", "{:.2f}"),
                      ("Share", "share", "natural key share ID_K / ID_KV of the K cell (the record's '0.15-0.20' holds for the W(POST, 7) and l = 11 statistics only: JcOneNeShareSmall...)", "{:.3f}")):
    vals = [r[key] for r in NE8]
    rmac(f"JcOneNe{w}Min", min(vals), f"J-C1 / J-C2, the non-equivalent E1 / E2 statistics: {d} (smallest)", _src1, G8C, f_)
    rmac(f"JcOneNe{w}Max", max(vals), f"J-C1 / J-C2, the non-equivalent E1 / E2 statistics: {d} (largest)", _src1, G8C, f_)
_sm = [r["share"] for r in NE8 if r["fk"] == "POST" or r["l"] == 11]
rmac("JcOneNeShareSmallMin", min(_sm), "J-C1 / J-C2: natural key share ID_K / ID_KV of the K cell over the non-equivalent W(POST, 7) and l = 11 statistics (smallest; the A statistics at l = 7 sit on cells with shares 0.47-0.81)", _src1, G8C)
rmac("JcOneNeShareSmallMax", max(_sm), "the same (largest)", _src1, G8C)
rmac("JcOneLamThreeAbsMax", max(L38), "J-C1 / J-C2: the largest |lambda| of E1 and E2 at l = 3 over the three models (every statistic there is equivalent)", _src1, G8C)
m = C8M["qwen7"]
_s = next(s for s in m.stats_of("E2", [7], "edit")[0] if s["code"] == "A_L(7)")
rmac("JcOneEtwoAletterSevenLamQwenSeven", _s["q"].pt, "J-C1: lambda of E2's A_L(7) at Qwen2.5-7B, the key under-read inside the margin where the natural key effect is large", SRCE8["qwen7"], G8C)
rmac("JcOneEtwoAletterSevenPsiKQwenSeven", m.psi("LETTER", 7, "E2", "K").pt, "J-C1: psi_K of E2 under LETTER at l = 7, Qwen2.5-7B", SRCE8["qwen7"], G8C)
rmac("JcOneEtwoAletterSevenIDKQwenSeven", m.mean("LETTER", 7, "nat", "K").pt, "J-C1: the natural ID_K under LETTER at l = 7 (nats), Qwen2.5-7B", SRCE8["qwen7"], G8C, "{:.1f}")
_fl8 = {}
for fl_ in (3.0, 4.0, 4.8):
    C8LW.MIN_ID = fl_   # the channel floor of the 'channel used' rule (preregistered: 2 nats); restored below
    _fl8[fl_] = [C8LW.law(C8M[k], Z, list(dep), "edit")[0] for (code, k, Z, dep), c in COMB8.items() if code == "J-C1"]
C8LW.MIN_ID = 2.0
for fl_, w in ((3.0, "Three"), (4.0, "Four"), (4.8, "FourEight")):
    rmac(f"JcOneFloor{w}NotMet", _fl8[fl_].count(False), f"post hoc: J-C1 combos NOT MET with a channel floor of {fl_} nats in place of the preregistered 2", _src1, G8C, "{:d}")
rmac("JcOneFloorFourEightMet", _fl8[4.8].count(True), "post hoc: J-C1 combos MET with a channel floor of 4.8 nats (all four: J-C1 would be MET)", _src1, G8C, "{:d}")
_ls8 = []
for (code, k, Z, dep), c in COMB8.items():
    if code != "J-C1":
        continue
    m, ok = C8M[k], True
    for s in m.stats_of(Z, list(dep), "edit")[0]:
        fk, fv = c8cells(s["code"])
        qL = c8lr(m.psi(fk, s["l"], Z, "K", "L"), m.psi(fv, s["l"], Z, "V", "L"))
        inter = any(1 - (np.nanmean(m.ID(f, s["l"], Z, "K", "L")) + np.nanmean(m.ID(f, s["l"], Z, "V", "L"))) / np.nanmean(m.ID(f, s["l"], Z, "KV", "L")) > 0.5 for f in {fk, fv})
        ok &= bool(c8tost(qL) and not inter)
    _ls8.append((k, Z, ok))
rmac("JcOneLscoreNotMet", sum(not o for _, _, o in _ls8), "post hoc: J-C1 combos NOT MET when every statistic is scored with L (the robustness score) in place of E (the one that passes: Mistral-7B E1, at the margin)",
     _src1, G8C, "{:d}")
# the outcome record's numbers (Sensitivities, where J-C1 fails) are reproduced
assert (f"{fl(macros['JcOneNePsiKMin']):.2f}", f"{fl(macros['JcOneNePsiKMax']):.2f}", f"{fl(macros['JcOneNePsiVMin']):.2f}", f"{fl(macros['JcOneNePsiVMax']):.2f}") == ("0.75", "0.87", "0.96", "1.09")
# the record's "natural value effects of 5.8-17 nats" leaves out one statistic: Llama-3.1-8B E2 W(P1, 7), whose V cell (P1, l = 7)
# has a natural ID_V of 3.3 nats; JcOneNeIDVMin is the minimum over all 12 (3.32)
assert (f"{fl(macros['JcOneNeIDKMin']):.1f}", f"{fl(macros['JcOneNeIDKMax']):.1f}", f"{fl(macros['JcOneNeIDVMax']):.0f}") == ("2.6", "4.8", "17")
assert f"{min(r['idV'] for r in NE8 if (r['k'], r['Z'], r['code']) != ('llama8', 'E2', 'W(P1,7)')):.1f}" == "5.8" and macros["JcOneNeIDVMin"] == "3.32"
assert (f"{fl(macros['JcOneNeShareSmallMin']):.2f}", f"{fl(macros['JcOneNeShareSmallMax']):.2f}", macros["JcOneLamThreeAbsMax"]) == ("0.15", "0.20", "0.053")
assert (macros["JcOneEtwoAletterSevenLamQwenSeven"], macros["JcOneEtwoAletterSevenPsiKQwenSeven"], macros["JcOneEtwoAletterSevenIDKQwenSeven"]) == ("-0.154", "0.875", "14.2")
assert [macros[f"JcOneFloor{w}NotMet"] for w in ("Three", "Four", "FourEight")][2] == "0" and all(int(macros[f"JcOneFloor{w}NotMet"]) > 0 for w in ("Three", "Four"))
assert macros["JcOneFloorFourEightMet"] == "4" and macros["JcOneLscoreNotMet"] == "3" and [(k, Z) for k, Z, o in _ls8 if o] == [("mistral7", "E1")]
# (6b) v5 round 2: how the core test (J-C1) fails, as counts over its printed statistics: how many of its statistics are not
# equivalent, at which depths, and how many of those have a 90 % interval wholly outside the margin (a shown departure, not an
# inconclusive TOST); and the same count for Llama-3.1-8B's E2 combo of J-C2. Each count is cross-checked against the printed
# strings (equivalence flag; interval bounds at two decimals against log 1.25 = 0.223).
_L125 = float(np.log(1.25))
_jc1 = dict(n=0, ne=0, out=0, dep=[])
_jc2l = dict(n=0, ne=0, out=0)
for (code, k, Z, dep), c in COMB8.items():
    if c["seed"] or c["v"] == "NOT EVALUABLE" or not (code == "J-C1" or (code == "J-C2" and k == "llama8" and Z == "E2")):
        continue
    acc = _jc1 if code == "J-C1" else _jc2l
    for s in C8M[k].stats_of(Z, list(dep), "edit")[0]:
        p_ = LAMP8[k, Z, s["code"]]
        assert p_["eq"] == s["equivalent"], (k, Z, s["code"])
        acc["n"] += 1
        if s["equivalent"]:
            continue
        acc["ne"] += 1
        acc.setdefault("dep", []).append(s["l"])
        lo_, hi_ = s["q"].ci(0.90)
        out_ = hi_ < -_L125 or lo_ > _L125
        assert out_ == (fl(p_["t"][2]) < -0.223 or fl(p_["t"][1]) > 0.223), (k, Z, s["code"], p_["t"])
        acc["out"] += int(out_)
_jc1src = "results/gpu_stage8c/STAGE8C_SCORE.txt (J-C1 and J-C2 combos)"
amac("JcOneCoreN", _jc1["n"], "J-C1 (the core test: E1 and E2 at Qwen2.5-7B and Mistral-7B): lambda statistics in its four combos", _jc1src, G8C, "{:d}")
amac("JcOneCoreNe", _jc1["ne"], "J-C1: lambda statistics that are not equivalent (90 % interval not inside +-log 1.25)", _jc1src, G8C, "{:d}")
amac("JcOneCoreNeOut", _jc1["out"], "J-C1: non-equivalent lambda statistics whose 90 % interval lies wholly outside +-log 1.25 (a shown departure); the rest are inconclusive", _jc1src, G8C, "{:d}")
amac("JcOneCoreNeDepthMin", min(_jc1["dep"]), "J-C1: the shallowest block of a non-equivalent lambda statistic (every statistic at block 3 is equivalent)", _jc1src, G8C, "{:d}")
amac("JcTwoLlamaEtwoNe", _jc2l["ne"], "J-C2, Llama-3.1-8B E2 (vectors from unrelated sentences): non-equivalent lambda statistics", _jc1src, G8C, "{:d}")
amac("JcTwoLlamaEtwoNeOut", _jc2l["out"], "J-C2, Llama-3.1-8B E2: non-equivalent lambda statistics whose 90 % interval lies wholly outside +-log 1.25", _jc1src, G8C, "{:d}")
# the outcome record: Qwen E1 8/10, Mistral E1 5/6, Qwen E2 7/8, Mistral E2 3/6 equivalent (7 of 30 not, all at block 7 or deeper,
# none wholly outside); Llama E2 4/7 (A_L(7) -0.32 [-0.42, -0.24] wholly outside)
assert (macros["JcOneCoreN"], macros["JcOneCoreNe"], macros["JcOneCoreNeOut"], macros["JcOneCoreNeDepthMin"]) == ("30", "7", "0", "7")
assert (macros["JcTwoLlamaEtwoNe"], macros["JcTwoLlamaEtwoNeOut"]) == ("3", "1")
# (7) the nu pattern (post hoc): every lambda statistic of every identity-carrying instance (phi(NONE) >= 0.3, point) against
# the stored table statistics; Spearman over the statistics, the equivalence by nu band, the collinearity over the
# (model, instance, depth) units, the first-order O1 prediction
NUP8 = []
for k in MC8:
    m = C8M[k]
    insts = list(FAMC8[k]) + [f"{c_}:{z}" for z in ("E1", "E2", "E3", "E4") if z in FAMC8[k] for c_ in ("PERP", "NONLEX")] + ["NONLEX:nat"]
    for Z in insts:
        for l in m.depths:
            if not m.has("NONE", l, C8LW.probe(Z) + "|KV|S") or not m.psi("NONE", l, Z, "KV").pt >= 0.3:
                continue
            st, _ = m.stats_of(Z, [l], gate=None, formats=C8LW.COMP_FORMATS if ":" in Z else C8LW.FORMATS)
            x = {v: c8tab(m, l, Z, v) for v in ("nu", "cos_k", "cos_v", "cos_resid", "proj_k", "proj_v")}
            NUP8 += [dict(k=k, Z=Z, l=l, code=s["code"], lam=s["q"].pt, eq=s["equivalent"], q=s["q"], **x) for s in st]
_lam = [p["lam"] for p in NUP8]
_src7 = "results/gpu_stage8c/eval/ (lambda statistics and table statistics)"
rmac("JcNuN", len(NUP8), "post hoc nu pattern: lambda statistics of the identity-carrying instances (phi(NONE) >= 0.3; every family, E5 and the components; E3 included)", _src7, G8C, "{:d}")
for w, v in (("Nu", "nu"), ("CosK", "cos_k"), ("ProjK", "proj_k")):
    rmac(f"JcNuRho{w}", spearmanr([p[v] for p in NUP8], _lam)[0], f"post hoc nu pattern: Spearman correlation of lambda with {v} over those statistics", _src7, G8C, "{:.2f}")
for w, lo, hi, d in (("Low", 0, 0.1, "nu < 0.1"), ("Mid", 0.1, 0.3, "0.1 <= nu < 0.3"), ("High", 0.3, 9, "nu >= 0.3"), ("HighBelowOne", 0.3, 1.01, "0.3 <= nu < 1.01 (the outcome record's count of 53 leaves out the 10 statistics with nu > 1.01)")):
    B_ = [p for p in NUP8 if lo <= p["nu"] < hi]
    rmac(f"JcNuEq{w}", sum(p["eq"] for p in B_), f"post hoc nu pattern: equivalent statistics with {d}", _src7, G8C, "{:d}")
    rmac(f"JcNuN{w}", len(B_), f"post hoc nu pattern: statistics with {d}", _src7, G8C, "{:d}")
_un = {}
for p in NUP8:
    _un[p["k"], p["Z"], p["l"]] = p
_v5 = ("nu", "cos_k", "cos_v", "cos_resid", "proj_k")
_cl = [abs(spearmanr([u[a] for u in _un.values()], [u[b] for u in _un.values()])[0]) for j, a in enumerate(_v5) for b in _v5[j + 1:]]
rmac("JcNuUnits", len(_un), "post hoc nu pattern: (model, instance, depth) units behind the statistics", _src7, G8C, "{:d}")
rmac("JcNuCollMin", min(_cl), "post hoc nu pattern: smallest pairwise |Spearman| among nu, the K, V and residual cosines and the O1 key projection over the units (collinear)", _src7, G8C, "{:.2f}")
rmac("JcNuCollMax", max(_cl), "post hoc nu pattern: largest pairwise |Spearman| among those table statistics", _src7, G8C, "{:.2f}")
_o1 = [np.log(max(p["proj_k"], 0.01)) - np.log(max(p["proj_v"], 0.01)) for p in NUP8]
rmac("JcOoneMiss", np.mean(np.array(_lam) - np.array(_o1)), "post hoc: mean of lambda minus the O1 first-order prediction log(proj_k / proj_v) over those statistics (the far edits are under-predicted)", _src7, G8C, "{:.2f}")
rmac("JcOoneRho", spearmanr(_o1, _lam)[0], "post hoc: Spearman correlation of lambda with the O1 prediction (weak)", _src7, G8C, "{:.2f}")
# the outcome record's numbers (Sensitivities, the departures follow nu) are reproduced
assert (macros["JcNuN"], macros["JcNuRhoNu"], macros["JcNuRhoCosK"]) == ("107", "-0.75", "0.80")
assert [(macros[f"JcNuEq{w}"], macros[f"JcNuN{w}"]) for w in ("Low", "Mid", "HighBelowOne")] == [("11", "11"), ("24", "33"), ("0", "53")]
assert (macros["JcNuEqHigh"], macros["JcNuNHigh"]) == ("0", "63") and (macros["JcNuCollMin"], macros["JcNuCollMax"]) == ("0.84", "0.98")
assert f"{fl(macros['JcOoneMiss']):.1f}" == "-0.7"
# (8) J-C4 by depth (the pooled lambda-bar of each depth's statistics) and the psi floor of W(POST, 7)
for k in ("qwen7", "mistral7"):
    m, t = C8M[k], K8[k]
    st = m.stats_of("E4", m.depths, "edit")[0]
    for l in (3, 7):
        rmac(f"JcFourDepth{DW8[l]}{t}", np.mean([s["q"].pt for s in st if s["l"] == l]), f"J-C4: E4's pooled lambda-bar over its statistics at l = {l} alone, {SH8[k]}", SRCE8[k], G8C, "{:.2f}")
    for C in ("K", "V"):
        rmac(f"JcFourPsi{C}WpostSeven{t}", m.psi("POST", 7, "E4", C).pt, f"J-C4: psi_{C} of E4 under POST at l = 7 (the W(POST, 7) statistic sits at the 0.01 psi floor), {SH8[k]}", SRCE8[k], G8C)
assert (macros["JcFourDepthThreeQwenSeven"], macros["JcFourDepthThreeMistralSeven"]) == ("-1.03", "-0.67")
assert 2.05 <= -fl(macros["JcFourDepthSevenQwenSeven"]) < 2.85 and 2.05 <= -fl(macros["JcFourDepthSevenMistralSeven"]) < 2.85
assert (macros["JcFourPsiKWpostSevenQwenSeven"], macros["JcFourPsiKWpostSevenMistralSeven"]) == ("0.026", "0.012")
# (9) J-C-BOUND: how far the gated-in E5 edits write from the natural write; FR, DE and SYN against the natural displacement
_n5 = [c8tab(C8M[k], l, Z, "nu") for (k, Z, code), v in LAMP8.items() if Z.startswith("E5") for l in [int(re.search(r"(\d+)\)$", code)[1])]]
rmac("JcBoundNuMin", min(_n5), "J-C-BOUND: nu of the gated-in E5 edits that have lambda statistics (smallest)", "results/gpu_stage8c/eval/", G8C, "{:.2f}")
rmac("JcBoundNuMax", max(_n5), "J-C-BOUND: nu of the gated-in E5 edits that have lambda statistics (largest)", "results/gpu_stage8c/eval/", G8C, "{:.2f}")
_rr = [(c8tab(C8M[k], l, Z, "ratio_resid"), c8tab(C8M[k], l, Z, "cos_resid")) for k in MC8 for Z in ("E5FR", "E5DE", "E5SYN") for l in DW8 if not EFF8C[k, Z, l]]
for w, j, d in (("Norm", 0, "norm of the edit's residual displacement at (p, l) over the natural displacement's"), ("Cos", 1, "cosine of the two displacements")):
    for x_, f_ in (("Min", min), ("Max", max)):
        rmac(f"JcBoundFrDeSyn{w}{x_}", f_(x[j] for x in _rr), f"J-C-BOUND: {d}, E5 FR, DE and SYN at the (model, depth) cells where they are not gated in (alpha = 2), "
             f"{'smallest' if x_ == 'Min' else 'largest'} (the gated-in FR cells of Llama-3.1-8B, alpha = 1, write 1.3x)", "results/gpu_stage8c/eval/", G8C, "{:.2f}")
assert (macros["JcBoundNuMin"], macros["JcBoundNuMax"]) == ("0.65", "1.29")
assert 2.0 <= fl(macros["JcBoundFrDeSynNormMin"]) and fl(macros["JcBoundFrDeSynNormMax"]) < 3.5 and fl(macros["JcBoundFrDeSynCosMax"]) < 0.35

# ---------------------------------------------------------------- stage 8, part D: the flag the reader heads write
MD8 = ["qwen7", "mistral7"]
MD8S = ["qwen1.5", "qwen3b", "qwen7"]
CELLD = {"Q_IN": "Qin", "Q_OUT": "Qout", "INLINE": "Inline", "INLINE_CHAT": "InlineChat", "AFTER": "After"}
LWD = {"J-D1": "One", "J-D2": "Two", "J-D3": "Three", "J-D4": "Four", "J-D-ADDR": "Addr", "J-D-KN": "Kn", "J-D5": "Five", "J-D7": "Seven",
       "J-D-SIGN-Q": "SignQ", "J-D-SIGN-IOI": "SignIoi", "J-D8-Q": "EightQ", "J-D8-IOI": "EightIoi", "J-D-ROUTE-IOI": "Route",
       "J-D-HOP2": "HopTwo", "J-D6a": "Sixa", "J-D6": "Six", "J-D6-ROUTE": "SixRoute"}
GD8, PD8, DD8, RD8 = (sec(S8D, s) for s in ("GATES (J-D-G0", "PREDICTIONS (code, class", "D6 DECISION TABLE", "REPORTED (no verdict counted)"))
SD8, XD8 = S8D.index("SUMMARY"), sec(S8D, "######## EXPLORATORY")
SRCD = SNAME["S8D"]
i = find(S8D, "  J-D-G0  FP32 unit tests")
amac("JdGzeroPassed", sum(int(x) for x in re.findall(r" (\d+) passed", S8D[i])), "J-D-G0: unit tests passed, summed over the five files the score line prints (the outcome record's 61 also counts tests/test_head_splice.py, which the line does not print: JdGzeroPassedRun)", f"{SRCD}:{i + 1}", G8D, "{:d}")
# the pytest session of record (results/gpu_stage8d/logs/pytest.log, one session) prints the 61 of the outcome record: the five
# files above and tests/test_head_splice.py (7, counted from the session's PASSED lines)
SCORE["S8DPY"] = (ROOT / "results/gpu_stage8d/logs/pytest.log").read_text().splitlines()
SNAME["S8DPY"] = "results/gpu_stage8d/logs/pytest.log"
assert sum("test session starts" in x for x in SCORE["S8DPY"]) == 1
i = find(SCORE["S8DPY"], " passed in ")
sm8("JdGzeroPassedRun", "S8DPY", i, r"=+ (\d+) passed in [\d.]+s", "J-D-G0: unit tests passed in the pytest session of record (all six files, tests/test_head_splice.py included; 0 failed, 0 skipped)", G8D)
assert not re.search(r"\b(failed|skipped|error)", SCORE["S8DPY"][i]), SCORE["S8DPY"][i]
_py8 = [x.split("::")[0] for x in SCORE["S8DPY"] if re.match(r"^tests/\w+\.py::\S+ PASSED", x)]
amac("JdGzeroHeadSplice", _py8.count("tests/test_head_splice.py"), "J-D-G0: tests of tests/test_head_splice.py passed in that session (PASSED lines)", SNAME["S8DPY"], G8D, "{:d}")
assert len(_py8) == int(macros["JdGzeroPassedRun"]) == int(macros["JdGzeroPassed"]) + int(macros["JdGzeroHeadSplice"]), (len(_py8), macros["JdGzeroPassedRun"])
for k in MD8:
    t, S = K8[k], SH8[k]
    i = find(S8D, f"  J-D-G1 {k:8s} N_X", start=GD8, end=PD8)
    sm83(f"JdGoneNX{t}", "S8D", i, r"N_X \(P1, E8\) " + CIR, f"J-D-G1 N_X: the natural key read's effect on X under P1 on E8 (nats), {S}", G8D)
    i = find(S8D, f"  J-D-G5 {k:8s} accuracy", start=GD8, end=PD8)
    sm8(f"JdGfiveOther{t}", "S8D", i, r"other_agent (\d\.\d+), irrelevant_object (\d\.\d+)", f"J-D-G5 binding-query accuracy, other_agent, {S}", G8D, 0)
    sm8(f"JdGfiveIrr{t}", "S8D", i, r"other_agent (\d\.\d+), irrelevant_object (\d\.\d+)", f"J-D-G5 binding-query accuracy, irrelevant_object, {S}", G8D, 1)
    for cell, cw in CELLD.items():
        if not find(S8D, f"  J-D-G3 {k:8s} {cell:11s} ", start=GD8, end=PD8, many=True):
            assert (k, cell) == ("mistral7", "AFTER"), (k, cell)   # the IOI AFTER cell is run at Qwen2.5-7B only
            continue
        i = find(S8D, f"  J-D-G3 {k:8s} {cell:11s} ", start=GD8, end=PD8)
        if cell.startswith("Q_"):
            sm8(f"JdGthreeArg{cw}{t}", "S8D", i, r"argmax (?:un)?mentioned (\d\.\d+)", f"J-D-G3 competence, {cell}: argmax rate on the correct class, {S}", G8D)
        else:
            for j, w in enumerate(("Two", "Four")):
                sm8(f"JdGthree{w}{cw}{t}", "S8D", i, r"two-way (\d\.\d+) \(>= 0\.75\), four-way (\d\.\d+)", f"J-D-G3 competence, {cell}: {w.lower()}-way accuracy, {S}", G8D, j)
        i = find(S8D, f"  J-D-G4 {k:8s} {cell:11s} ", start=GD8, end=PD8)
        sm83(f"JdGfourDG{cw}{t}", "S8D", i, r"d_G " + CIR, f"J-D-G4 d_G (nats): the re-mention read at the list or name rows, {cell}, {S}", G8D)
        sm83(f"JdGfourIDK{cw}{t}", "S8D", i, r"ID_K\(allT\) " + CIR, f"J-D-G4 ID_K(allT) (nats), {cell}, {S}", G8D)
        sm8(f"JdGfourRatio{cw}{t}", "S8D", i, r"d_G/ID_K " + NUM, f"J-D-G4 d_G/ID_K (>= 0.5 for the transfer to be evaluable), {cell}, {S}", G8D)
for k in MD8S:
    t, S = K8[k], SH8[k]
    i = find(S8D, f"  J-D-G6 {k:8s} competent cores", start=GD8, end=PD8)
    sm8(f"JdGsixComp{t}", "S8D", i, r"competent cores (\d+) of (\d+)", f"J-D-G6: competent dissociation cores (>= 40 needed), {S}", G8D, 0)
    sm8(f"JdGsixN{t}", "S8D", i, r"competent cores (\d+) of (\d+)", f"J-D-G6: dissociation cores run, {S}", G8D, 1)
    sm83(f"JdGsixDlx{t}", "S8D", i, r"P1 injection dl_X " + CIR, f"J-D-G6: P1 injection's dl_X (nats), {S}", G8D)
    if "R(k*) of the in-run H*" in S8D[i]:
        sm83(f"JdGsixR{t}", "S8D", i, r"R\(k\*\) of the in-run H\* " + CIR, f"J-D-G6: R(k*) of the heads ranked in the run, {S}", G8D)
# the lines
LD8 = {}
for i in range(PD8 + 1, DD8):
    mm = re.match(r"  (J-D[\w-]+)\s+(\w) (\w) prior (\d\.\d\d)\s+" + VRD + r"\s+(.*)$", S8D[i])
    if mm:
        cur = mm[1]
        LD8[cur] = dict(i=i, cls=mm[2], kind=mm[3], prior=mm[4], v=mm[5], title=mm[6], m={})
        continue
    mm = re.match(r"    ([\w.]+)\s+" + VRD + r"\s+(.*)$", S8D[i])
    if mm:
        LD8[cur]["m"][mm[1]] = dict(i=i, v=mm[2], txt=mm[3])
assert list(LD8) == list(LWD), list(LD8)
for code, L in LD8.items():
    vs = [r_["v"] for r_ in L["m"].values()]
    amac(f"Jd{LWD[code]}Met", sum(v == "MET" for v in vs), f"{code}: models that meet the line", f"{SRCD}:{L['i'] + 1}", G8D, "{:d}")
    amac(f"Jd{LWD[code]}Eval", sum(v != "NOT EVALUABLE" for v in vs), f"{code}: evaluable models (of {len(vs)} run)", f"{SRCD}:{L['i'] + 1}", G8D, "{:d}")


def id8(code, k):
    return LD8[code]["m"][k]["i"]


for k in MD8:
    t, S = K8[k], SH8[k]
    i = id8("J-D1", k)
    sm83(f"JdOneRatio{t}", "S8D", i, r"ID_inj\^LOO/ID_K " + CIR, f"J-D1 ID_inj^LOO/ID_K: the leave-one-word-out flag injected at an absent option's row, as a share of the key read's identity effect, {S}", G8D)
    sm8(f"JdOnePiX{t}", "S8D", i, r"pi_X\(move, LOO\) (\d\.\d+)", f"J-D1 pi_X: share of stories whose answer moves to the absent option, {S}", G8D)
    sm83(f"JdOneIDK{t}", "S8D", i, r"; ID_K " + CIR, f"J-D1 ID_K (nats) of the natural key read on the inject population, {S}", G8D)
    i = id8("J-D2", k)
    sm83(f"JdTwoIotaMove{t}", "S8D", i, r"iota\(move\) " + CIR, f"J-D2 iota(move Delta): the flag moved from B's row to X's, as a share of N_X, {S}", G8D)
    for c, cw in (("orth", "Orth"), ("meanH", "MeanH"), ("active", "Active")):
        pat = c + r": iota " + CIR + r", difference " + CIR + r" \(>= 0\.4; H0: <= 0\), ratio " + NUM
        cd = {"orth": "the story's own write orthogonal to the flag, norm-matched", "meanH": "the mean H* output", "active": "the active-set flag"}[c]
        sm83(f"JdTwoIota{cw}{t}", "S8D", i, pat, f"J-D2 iota of the control ({cd}), {S}", G8D, 0)
        sm83(f"JdTwoDiff{cw}{t}", "S8D", i, pat, f"J-D2 iota(move) - iota(control: {cd}), {S}", G8D, 3)
        sm8(f"JdTwoRatio{cw}{t}", "S8D", i, pat, f"J-D2 iota(control) / iota(move) ({cd}), {S}", G8D, 6)
    i = id8("J-D3", k)
    for w, p, d in (("Flag", r"rho_K flag ", "the flag direction"), ("Pcone", r"pc1 ", "the first principal direction (pc1)"), ("MeanH", r"mean-H\* output ", "the mean H* output direction")):
        sm83(f"JdThreeRho{w}{t}", "S8D", i, p + CIR, f"J-D3 rho_K: key read left when {d} is ablated per layer at the option rows, {S}", G8D)
    i = id8("J-D4", k)
    for w, p, d in (("KV", r"r_ans\^inj\(KV\) ", "r_ans^inj(KV)"), ("K", r", K ", "r_ans^inj(K): the answer row sees the base keys only"),
                    ("V", r", V ", "r_ans^inj(V): base values only"), ("KmV", r"K - V ", "r_ans^inj(K) - r_ans^inj(V)"), ("Den", r"denominator ", "denominator (nats)")):
        sm83(f"JdFour{w}{t}", "S8D", i, p + CIR, f"J-D4 the injected flag read at hop 2: {d}, {S}", G8D)
    i = id8("J-D-ADDR", k)
    sm8(f"JdAddrCos{t}", "S8D", i, r"weighted cos\(Delta\^K, Delta\^KV\) " + NUM, f"J-D-ADDR weighted cosine of the key-only and the K-and-V flags, {S}", G8D)
    sm83(f"JdAddrRatio{t}", "S8D", i, r"ID_inj\(KV\)/ID_inj\(K\) " + CIR, f"J-D-ADDR ID_inj(KV)/ID_inj(K), {S}", G8D)
    i = id8("J-D-KN", k)
    sm83(f"JdKnPearson{t}", "S8D", i, r"mean per-story Pearson " + CIR, f"J-D-KN mean per-story Pearson correlation, {S}", G8D)
    sm83(f"JdKnBeta{t}", "S8D", i, r"B-loss ratio " + CIR, f"J-D-KN B-loss ratio beta (non-candidate key against removing B's flag), {S}", G8D)
    sm83(f"JdKnDiff{t}", "S8D", i, r"mean\(dl_S - dl_X\) " + CIR, f"J-D-KN mean(dl_S - dl_X) (nats), {S}", G8D)
    sm8(f"JdKnTol{t}", "S8D", i, r"inside \+-(\d+\.\d+) =", f"J-D-KN tolerance 0.1 x ID_K (nats), {S}", G8D)
    i = id8("J-D5", k)
    sm83(f"JdFivePooled{t}", "S8D", i, r"Psi_bind pooled " + CIR, f"J-D5 Psi_bind pooled (nats), {S}", G8D)
    for w, p in (("Obj", "object-first "), ("Dis", "distractor-first ")):
        sm83(f"JdFive{w}{t}", "S8D", i, p + CIR, f"J-D5 Psi_bind, {p.strip()} stories (nats), {S}", G8D)
        sm8(f"JdFiveN{w}{t}", "S8D", i, p + CIR + r" \(n (\d+)\)", f"J-D5 {p.strip()} stories, {S}", G8D, 3)
    sm83(f"JdFiveDiff{t}", "S8D", i, r"stratum difference " + CIR, f"J-D5 stratum difference (nats; its CI must include 0), {S}", G8D)
    i = id8("J-D7", k)
    for w, p, d in (("Qout", r"ID_K\(Q_OUT\) ", "ID_K under Q_OUT (which option is not mentioned?)"), ("Qin", r"ID_K\(Q_IN\) ", "ID_K under Q_IN (which option is mentioned?)"),
                    ("Diff", r"paired difference ", "ID_K(Q_IN) - ID_K(Q_OUT), paired")):
        sm83(f"JdSeven{w}{t}", "S8D", i, p + CIR, f"J-D7 {d} (nats), {S}", G8D)
    sm83(f"JdSignQIDV{t}", "S8D", id8("J-D-SIGN-Q", k), r"ID_V " + CIR, f"J-D-SIGN-Q ID_V under Q_OUT (nats), {S}", G8D)
    i = id8("J-D-SIGN-IOI", k)
    sm83(f"JdSignIoiIDK{t}", "S8D", i, r"INLINE ID_K " + CIR, f"J-D-SIGN-IOI ID_K under IOI INLINE (in-sentence re-mention) (nats), {S}", G8D)
    sm83(f"JdSignIoiIDV{t}", "S8D", i, r"ID_V " + CIR, f"J-D-SIGN-IOI ID_V under IOI INLINE (nats), {S}", G8D)
    i = id8("J-D8-Q", k)
    for cell, cw in (("Q_IN", "Qin"), ("Q_OUT", "Qout")):
        pre = (r"Q_IN: " if cell == "Q_IN" else r"\| Q_OUT: ") + r".*?"
        sm83(f"JdEight{cw}R{t}", "S8D", i, pre + r"R\(H\*\) " + CIR, f"J-D8-Q R(H*): the frozen reader heads recover the {cell} key read, {S}", G8D)
        sm83(f"JdEight{cw}KO{t}", "S8D", i, pre + r"KO\(H\*\) " + CIR, f"J-D8-Q KO(H*): blocking them removes it, {cell}, {S}", G8D)
        for j, w in enumerate(("Rrand", "KOrand")):
            sm8(f"JdEight{cw}{w}{t}", "S8D", i, pre + r"random R " + NUM + " KO " + NUM, f"J-D8-Q random sets' {w[:-4]}, {cell}, {S}", G8D, j)
        sm8(f"JdEight{cw}DGc{t}", "S8D", i, pre + r"d_Gc/ID_K " + NUM, f"J-D8-Q d_Gc/ID_K, {cell}, {S}", G8D)
    i = id8("J-D-ROUTE-IOI", k)
    for w, p, d in (("KV", r"r_ans\(KV\) ", "r_ans(KV)"), ("K", r", K ", "r_ans(K)"), ("V", r", V ", "r_ans(V)"), ("KmV", r"K - V ", "r_ans(K) - r_ans(V)"),
                    ("Other", r"r_other ", "r_other (S2 or tail rows)"), ("Dm", r"dm\(K_S\) ", "dm(K_S): the denominator (nats)")):
        sm83(f"JdRoute{w}{t}", "S8D", i, p + CIR, f"J-D-ROUTE-IOI the negative IOI read at the answer row's key read of the listed names: {d}, {S}", G8D)
    i = id8("J-D-HOP2", k)
    for cell, cw in (("Q_OUT", "Qout"), ("INLINE", "Inline")):
        pre = (r"Q_OUT: " if cell == "Q_OUT" else r"; INLINE: ") + r".*?"
        sm83(f"JdHopTwo{cw}Carry{t}", "S8D", i, pre + r"carry\(top-10\) " + CIR, f"J-D-HOP2 share of the injected flag's {cell} effect carried by the top-10 hop-2 heads, {S}", G8D)
        sm8(f"JdHopTwo{cw}All{t}", "S8D", i, pre + r"all heads " + NUM, f"J-D-HOP2 carried by all heads, {cell}, {S}", G8D)
        sm8(f"JdHopTwo{cw}Rand{t}", "S8D", i, pre + r"random 10 " + NUM, f"J-D-HOP2 carried by 10 random heads, {cell}, {S}", G8D)
        sm83(f"JdHopTwo{cw}Dlx{t}", "S8D", i, pre + r"dl_X\(inj\) " + CIR, f"J-D-HOP2 dl_X of the injection (nats), {cell}, {S}", G8D)
i = id8("J-D8-IOI", "qwen7")
for w, p in (("R", r"R\(H\*\) "), ("KO", r"KO\(H\*\) ")):
    sm83(f"JdEightIoi{w}QwenSeven", "S8D", i, p + CIR, f"J-D8-IOI {w}(H*) under IOI INLINE, Qwen2.5-7B (Mistral-7B not evaluable: J-D-G4)", G8D)
for j, w in enumerate(("Rrand", "KOrand")):
    sm8(f"JdEightIoi{w}QwenSeven", "S8D", i, r"random R " + NUM + " KO " + NUM, f"J-D8-IOI random sets' {w[:-4]}, Qwen2.5-7B", G8D, j)
sm8("JdEightIoiDGcQwenSeven", "S8D", i, r"d_Gc/ID_K " + NUM, "J-D8-IOI d_Gc/ID_K, Qwen2.5-7B", G8D)
for k in MD8S:
    t, S = K8[k], SH8[k]
    i = id8("J-D6a", k)
    sm83(f"JdSixaOmega{t}", "S8D", i, r"omega " + CIR, f"J-D6a omega: the flag written in the sentence (POST) against the list (P1), {S}", G8D)
    sm8(f"JdSixaKappa{t}", "S8D", i, r"kappa " + NUM, f"J-D6a kappa (cosine of the sentence and list flags), {S}", G8D)
    i = id8("J-D6", k)
    if LD8["J-D6"]["m"][k]["v"] == "NOT EVALUABLE":
        continue
    for w, p, d in (("Rho", r"rho ", "rho = dl_X(inj, POST)/dl_X(inj, P1)"), ("RhoNat", r"rho_nat ", "rho_nat = N_X(POST)/N_X(P1)"), ("Diff", r"rho - rho_nat ", "rho - rho_nat")):
        sm83(f"JdSix{w}{t}", "S8D", i, p + CIR, f"J-D6 {d}, {S}", G8D)
    sm8(f"JdSixTol{t}", "S8D", i, r"inside \+-(\d\.\d+)\)", f"J-D6 tolerance max(0.1, 0.5 |rho_nat|), {S}", G8D)
    for j, w in enumerate(("InjOpt", "InjPost")):
        sm83(f"JdSix{w}{t}", "S8D", i, r"dl_X\(inj\) P1 " + CIR + r", POST " + CIR, f"J-D6 dl_X of the injected P1 flag (nats), {('P1', 'POST')[j]}, {S}", G8D, 3 * j)
    for j, w in enumerate(("NXOpt", "NXPost")):
        sm83(f"JdSix{w}{t}", "S8D", i, r"N_X P1 " + CIR + r", POST " + CIR, f"J-D6 N_X (nats), {('P1', 'POST')[j]}, {S}", G8D, 3 * j)
i = id8("J-D6-ROUTE", "qwen1.5")
for w, p, d in (("V", r"r_ans\^inj\(V\) ", "r_ans^inj(V)"), ("K", r", K ", "r_ans^inj(K)"), ("VmK", r"V - K ", "r_ans^inj(V) - r_ans^inj(K)"), ("KV", r"; KV ", "r_ans^inj(KV)"),
                ("Dlx", r"dl_X\(inj\) ", "dl_X of the injection (nats)")):
    sm83(f"JdSixRoute{w}QwenOnefive", "S8D", i, p + CIR, f"J-D6-ROUTE at 1.5B under POST: {d}, Qwen2.5-1.5B", G8D)
sm8("JdSixTable", "S8D", DD8, r"\((\w)\) MIXED", "the D6 decision table's outcome letter (fixed in the entry): (d) mixed", G8D)
# reported
for k in MD8:
    t, S = K8[k], SH8[k]
    i = find(S8D, f"  {k}: iota add 0.5 / 1 / 2", start=RD8, end=SD8)
    dose = grab(S8D, i, r"iota add 0\.5 / 1 / 2 " + NUM + " / " + NUM + " / " + NUM)
    add1 = grab(S8D, i, r"add-only " + CIR)
    assert add1[0] == dose[1], (k, dose, add1)
    smac(f"JdDoseHalf{t}", "S8D", i, dose[0], f"dose: iota of adding 0.5 Delta at X's row, {S}", G8D)
    smac3(f"JdDoseOne{t}", "S8D", i, add1, f"dose: iota of adding 1 Delta at X's row (the add-only row, with its interval), {S}", G8D)
    smac(f"JdDoseTwo{t}", "S8D", i, dose[2], f"dose: iota of adding 2 Delta at X's row, {S}", G8D)
    sm8(f"JdPiXMove{t}", "S8D", i, r"pi_X\(move\) (\d\.\d+)", f"pi_X(move): share of stories whose answer moves to X when the full flag is moved, {S}", G8D)
    sm83(f"JdFullFlag{t}", "S8D", i, r"ID_inj\(full flag\)/ID_K " + CIR, f"ID_inj(full flag)/ID_K, {S}", G8D)
    i = find(S8D, f"  {k}: sanity rows:", start=RD8, end=SD8)
    for c, w in (("iso0", "IsoZero"), ("iso1", "IsoOne"), ("iso2", "IsoTwo"), ("hspan0", "HspanZero"), ("hspan1", "HspanOne"), ("hspan2", "HspanTwo"),
                 ("perm", "Perm"), ("choices", "Choices"), ("question", "Question")):
        sm8(f"JdSanity{w}{t}", "S8D", i, r"\b" + c + " " + NUM, f"iota of the sanity row {c}, {S}", G8D)
    sm83(f"JdRankOne{t}", "S8D", i, r"rank-1 capture " + CIR, f"rank-1 capture of the flag, {S}", G8D)
    sm83(f"JdFourOther{t}", "S8D", find(S8D, f"  {k}: D4 r_other", start=RD8, end=SD8), r"r_other " + CIR, f"J-D4 (reported) r_other, {S}", G8D)
    i = find(S8D, f"  {k}: B-identity share", start=RD8, end=SD8)
    rng8(f"JdBidShare{t}", re.findall(r"l\d+ (\d\.\d+)", S8D[i]), f"B-identity share of the per-story flag's variance over layers, {S}", f"{SRCD}:{i + 1}", G8D)
    if k == "qwen7":
        sm8("JdBidShareNull", "S8D", i, r"\(null (\d\.\d+)\)", "null B-identity share", G8D)
    i = find(S8D, f"  {k}: logit lens", start=RD8, end=SD8)
    lens = re.findall(r"l(\d+) (\d\.\d+)/(\d\.\d+)", S8D[i])
    top = max(lens, key=lambda x: float(x[1]))
    nm(f"JdLensMax{t}", top[1], f"largest |cos(Delta, W_U[location])| over layers, {S}", f"{SRCD}:{i + 1}", G8D)
    nm(f"JdLensLayer{t}", top[0], f"layer of that largest cosine, {S}", f"{SRCD}:{i + 1}", G8D)
    nm(f"JdLensRandMax{t}", max((x[2] for x in lens), key=float), f"largest cosine with random token rows over layers, {S}", f"{SRCD}:{i + 1}", G8D)
    i = find(S8D, f"  {k}: rho_K isotropic", start=RD8, end=SD8)
    for c, w in (("iso0", "Zero"), ("iso1", "One"), ("iso2", "Two")):
        sm83(f"JdRhoIso{w}{t}", "S8D", i, c + " " + CIR, f"rho_K of the isotropic direction {c}, {S}", G8D)
    sm83(f"JdPsiEv{t}", "S8D", find(S8D, f"  {k}: event vs initial", start=RD8, end=SD8), r"Psi_ev " + CIR, f"Psi_ev: event against initial flag (role confounded with order) (nats), {S}", G8D)
    for cell, cw in CELLD.items():
        if (k, cell) == ("mistral7", "AFTER"):
            continue
        h = find(S8D, f"  {k} {cell}: ID_K", start=RD8, end=SD8)
        for w, p in (("IDK", r"ID_K "), ("IDV", r"ID_V "), ("R", r"R\(H\*\) "), ("KO", r"KO\(H\*\) "), ("DG", r"d_G/ID_K "), ("DGc", r"d_Gc/ID_K ")):
            sm83(f"JdCell{cw}{w}{t}", "S8D", h, p + CIR, f"reported, {cell}: {p.replace(chr(92), '').strip()}{' (nats)' if w in ('IDK', 'IDV') else ''}, {S}", G8D)
        for j in range(h + 1, h + 4):
            if not S8D[j].startswith("     "):
                break   # the next cell's header (or the next block): only the indented sub-rows below a header belong to its cell
            sub = next(w for p, w in (("     behavioural correlate", "DlB"), ("     D9 rows", "D9"), ("     route of the injected IOI flag", "Route"))
                       if S8D[j].startswith(p))
            # the dl_B and D9 rows exist for the question cells only, the route row for the IOI in-sentence cell only
            assert (sub in ("DlB", "D9")) == (cell in ("Q_IN", "Q_OUT")) and (sub == "Route") == (cell == "INLINE"), (k, cell, sub, j + 1)
            if S8D[j].startswith("     behavioural correlate"):
                sm83(f"JdCell{cw}DlB{t}", "S8D", j, r"dl_B under K_S " + CIR, f"reported, {cell}: dl_B under K_S (nats), {S}", G8D)
            elif S8D[j].startswith("     D9 rows"):
                sm83(f"JdNine{cw}Dlx{t}", "S8D", j, r"dl_X\(\+Delta\^P1 at r_X\) " + CIR, f"D9, {cell}: dl_X of the P1 flag injected at X's row (nats), {S}", G8D)
                sm83(f"JdNine{cw}NX{t}", "S8D", j, r"N_X\(cfg\) " + CIR, f"D9, {cell}: N_X (nats), {S}", G8D)
                sm8(f"JdNine{cw}Ratio{t}", "S8D", j, r"ratio " + NUM, f"D9, {cell}: the P1 flag's effect as a share of N_X, {S}", G8D)
            elif S8D[j].startswith("     route of the injected IOI flag"):
                for w, p in (("KV", r"r_ans\(KV\) "), ("K", r", K "), ("V", r", V "), ("Other", r"other "), ("Dm", r"dm\(inj\) ")):
                    sm83(f"JdRouteInj{w}{t}", "S8D", j, p + CIR, f"route of the injected IOI flag: {p.replace(chr(92), '').strip(', ')}, {S}", G8D)
    i = find(S8D, f"  {k}: hop-2 carry under P1", start=RD8, end=SD8)
    for w, p in (("Top", r"top10 "), ("Rand", r"rand10 "), ("All", r"all ")):
        sm83(f"JdHopOpt{w}{t}", "S8D", i, p + CIR, f"hop-2 carry under P1 (the ranking format): {p.strip()}, {S}", G8D)
_sub8 = sorted(n for n in macros if n not in PRE8 and re.match(r"Jd(Cell\w+DlB|Nine|RouteInj)", n))
assert all(re.match(r"Jd(Cell(Qin|Qout)DlB|Nine(Qin|Qout)(Dlx|NX|Ratio))(QwenSeven|MistralSeven)(Lo|Hi)?$", n) or
           re.match(r"JdRouteInj\w+(QwenSeven|MistralSeven)(Lo|Hi)?$", n) for n in _sub8), _sub8
assert len(_sub8) == 2 * (2 * 3 + 2 * 7) + 2 * 5 * 3, len(_sub8)   # per model: dl_B and D9 under Q_IN and Q_OUT; the INLINE route row
for k in MD8S:
    sm83(f"JdSixb{K8[k]}", "S8D", find(S8D, f"  {k}: D6b Delta^POST", start=RD8, end=SD8), r"there " + CIR,
         f"D6b: the POST flag at the P1 list row as a share of the P1 flag there, {SH8[k]}", G8D)
amac("JdSanityAbsMax", max(abs(fl(macros[n])) for n in macros if n.startswith("JdSanity") and n not in PRE8), "largest |iota| of the sanity rows over both models",
     SRCD + " REPORTED", G8D, "{:.3f}")
rng8("JdRhoIso", [macros[f"JdRhoIso{w}{K8[k]}"] for k in MD8 for w in ("Zero", "One", "Two")], "rho_K of the isotropic directions over both models", SRCD + " REPORTED", G8D)
rng8("JdBidShare", [macros[f"JdBidShare{K8[k]}{x}"] for k in MD8 for x in ("Min", "Max")], "B-identity share over layers and both models", SRCD + " REPORTED", G8D)
# exploratory
for k in MD8:
    t, S = K8[k], SH8[k]
    for task in ("paint", "schedule"):
        sm8(f"JdXtask{task.capitalize()}{t}", "S8D", find(S8D, f"-- {k}: cross-task {task}:", start=XD8), r"ratio " + NUM,
            f"cross-task (exploratory): the belief flag in the {task} task as a share of the task's own flag, {S}", G8D)
    sm8(f"JdQoutKS{t}", "S8D", find(S8D, f"-- {k}: Q_OUT behaviour", start=XD8), r"K_S (\d\.\d+)", f"Q_OUT behaviour: share answering the story's own word B under K_S, {S}", G8D)
# summary
classes8(S8D, SD8, XD8, "Jd", G8D)
i = find(S8D, "  met rate among R account lines with a verdict:", start=SD8, end=XD8)
for j, (w, d) in enumerate((("Met", "met"), ("Verd", "with a verdict"), ("Exp", "expected met"))):
    sm8(f"JxRisky{w}D", "S8D", i, r"(\d+) of (\d+) \(expected (\d+\.\d+)\)", f"part D: risky (class R) account lines {d}", G8X, j)
_vd = [L["v"] for L in LD8.values()]
for w, v in (("Met", "MET"), ("NotMet", "NOT MET"), ("NotEval", "NOT EVALUABLE")):
    amac(f"JdCount{w}", _vd.count(v), f"part D: lines {v} (counted from the verdict lines)", SRCD + " PREDICTIONS", G8D, "{:d}")
amac("JdCountLines", len(_vd), "part D: lines in all", SRCD + " PREDICTIONS", G8D, "{:d}")
assert sum(int(macros[f"JdClass{c}{w}"]) for c in "LMR" for w in ("Met",)) == int(macros["JdCountMet"])
# recomputed (not printed; docs/PREREGISTRATION.md, outcome of part D): the 3B values of rho and rho_nat on its 35 competent
# cores, and the post hoc natural ratios on the X-specific contrast dl_X(K_X) - dl_X(K_S) and on ID_K, with rho minus
# each, through the committed scorer's own Diss class (competent cores, story bootstrap, seed 20261010). The same code
# reproduces the printed 1.5B and 7B rho, rho_nat and rho - rho_nat (asserted).
from stage8d_parts.lines import Diss as D8Diss, pick as d8pick, renorm as d8renorm  # noqa: E402
from stage8d_parts.stats import ratio as d8ratio  # noqa: E402
D8R = {}
for k in MD8S:
    t, S = K8[k], SH8[k]
    srcf = f"results/gpu_stage8d/diss/{k}.json"
    D_ = D8Diss(json.load(open(ROOT / srcf)))
    assert len(D_.ci) == int(macros[f"JdGsixComp{t}"]), (k, len(D_.ci))
    r_ = D_.rho()
    trio = {w: (r_[x].pt, *r_[x].ci()) for w, x in (("Rho", "rho"), ("RhoNat", "rho_nat"), ("Diff", "d"))}
    if k == "qwen3b":
        for w, d in (("Rho", "rho"), ("RhoNat", "rho_nat"), ("Diff", "rho - rho_nat")):
            rmac3(f"JdSix{w}{t}", trio[w], f"J-D6 {d} on the 35 competent cores (the line is not evaluable at 3B: J-D-G6), {S}", srcf, G8D)
    else:
        for w in ("Rho", "RhoNat", "Diff"):
            assert tuple(f"{x:+.3f}" for x in trio[w]) == tuple(("+" if not macros[f"JdSix{w}{t}{s}"].startswith("-") else "") + macros[f"JdSix{w}{t}{s}"]
                                                              for s in ("", "Lo", "Hi")), (k, w, trio[w])

        def _idk(f, D=D_):
            def l_(name, w):
                a = d8renorm(np.array([D.S[j]["fmt"][f]["rows"][name] for j in D.ci], float))
                return d8pick(a, D.ix[w][D.ci])
            c_ = lambda nm_: l_(nm_, "S") - l_(nm_, "X")  # noqa: E731
            return D.bc.mean(0.5 * (c_("K_S") - c_("K_X")))
        xs = d8ratio(*(D_.bc.mean(D_.dlx(f, "K_X") - D_.dlx(f, "K_S")) for f in ("POST", "P1")))
        ik = d8ratio(_idk("POST"), _idk("P1"))
        for w, q, d in (("NatX", xs, "natural ratio on the X-specific contrast [dl_X(K_X) - dl_X(K_S)](POST) / [same](P1)"),
                        ("DiffX", r_["rho"] - xs, "rho minus the X-specific natural ratio"),
                        ("NatIDK", ik, "natural ratio on ID_K: ID_K(POST)/ID_K(P1)"), ("DiffIDK", r_["rho"] - ik, "rho minus the ID_K natural ratio")):
            rmac3(f"JdSix{w}{t}", (q.pt, *q.ci()), f"J-D6 post hoc: {d}, competent cores, {S}", srcf, G8D)
            D8R[k, w] = q
# the outcome record's numbers (docs/PREREGISTRATION.md, part D sensitivities) are reproduced
assert (macros["JdSixRhoQwenThree"], macros["JdSixRhoNatQwenThree"]) == ("-0.025", "0.266")
assert (macros["JdSixNatXQwenOnefive"], macros["JdSixNatXQwenSeven"], macros["JdSixDiffXQwenOnefive"], macros["JdSixDiffXQwenSeven"]) == ("-0.113", "0.271", "-0.013", "-0.014")
assert (macros["JdSixNatIDKQwenOnefive"], macros["JdSixNatIDKQwenSeven"], macros["JdSixDiffIDKQwenOnefive"], macros["JdSixDiffIDKQwenSeven"]) == ("-0.115", "0.290", "-0.011", "-0.033")

# ---------------------------------------------------------------- stage 8, across parts A, B and D (kept as registered before part C; the ABCD totals follow)
_vs = []   # (part, line, model, verdict) of every counted line's per-model result
_vs += [("A", c, k, r_["v"]) for c, L in LA8.items() if L["cls"] != "D" for k, r_ in L["m"].items()]
_vs += [("B", c, k, r_["v"]) for c, L in LB8.items() for k, r_ in L["m"].items() if k in FAM8]
_vs += [("D", c, k, r_["v"]) for c, L in LD8.items() for k, r_ in L["m"].items()]
_fam = sorted({FAM8[k] for _, _, k, v in _vs if v in ("MET", "NOT MET")})
_famm = sorted({FAM8[k] for _, _, k, v in _vs if v == "MET"})
_mods = sorted({k for _, _, k, v in _vs if v in ("MET", "NOT MET")})
amac("JxFamilies", len(_fam), f"model families with a confirmatory stage-8 result (a per-model verdict, met or not met, on a counted line of parts A, B or D): {', '.join(_fam)}",
     "STAGE8A/8B/8D_SCORE.txt PREDICTIONS", G8X, "{:d}")
amac("JxFamiliesMet", len(_famm), f"model families with at least one per-model MET on a counted line of parts A, B or D: {', '.join(_famm)}", "STAGE8A/8B/8D_SCORE.txt PREDICTIONS", G8X, "{:d}")
amac("JxModels", len(_mods), f"models with a per-model verdict on a counted stage-8 line (parts A, B, D): {', '.join(SH8[k] for k in _mods)}",
     "STAGE8A/8B/8D_SCORE.txt PREDICTIONS", G8X, "{:d}")
for w, fmt in (("Met", "{:d}"), ("Verd", "{:d}"), ("Exp", "{:.2f}")):
    vals = [macros[f"JxRisky{w}{p}"] for p in "ABD"]
    amac(f"JxRisky{w}ABD", sum((float if w == "Exp" else int)(v) for v in vals),
         f"parts A, B and D: risky (class R) account lines {dict(Met='met', Verd='with a verdict', Exp='expected met')[w]} ({' + '.join(vals)}; part C excluded: JxRisky...ABCD include it)",
         "STAGE8A/8B/8D_SCORE.txt SUMMARY", G8X, fmt)
for w in ("Met", "NotMet", "NotEval", "Lines"):
    vals = [macros[f"J{p}Count{w}"] for p in "abd"]
    amac(f"Jx{w}ABD" if w != "Lines" else "JxLinesABD", sum(int(v) for v in vals),
         f"parts A, B and D: lines {dict(Met='MET', NotMet='NOT MET', NotEval='NOT EVALUABLE', Lines='counted')[w]} ({' + '.join(vals)}; part C excluded: the ...ABCD macros include it)",
         "STAGE8A/8B/8D_SCORE.txt SUMMARY", G8X, "{:d}")
for c in ("L", "M"):
    for w, fmt in (("Met", "{:d}"), ("Exp", "{:.2f}")):
        vals = [macros[f"J{p}Class{c}{w}"] for p in "abd"]
        amac(f"JxClass{c}{w}ABD", sum((float if w == "Exp" else int)(v) for v in vals), f"parts A, B and D: class-{c} account lines {'met' if w == 'Met' else 'expected met'} ({' + '.join(vals)})",
             "STAGE8A/8B/8D_SCORE.txt SUMMARY", G8X, fmt)
    amac(f"JxClass{c}VerdABD", sum(int(macros[f"J{p}Class{c}Lines"]) - int(macros[f"J{p}Class{c}NotEval"]) for p in "abd"),
         f"parts A, B and D: class-{c} account lines with a verdict", "STAGE8A/8B/8D_SCORE.txt SUMMARY", G8X, "{:d}")
assert int(macros["JxRiskyVerdA"]) == int(macros["JaClassRLines"]) - int(macros["JaClassRNotEval"])

# ---------------------------------------------------------------- stage 8, across all four parts (A-D): the entry's totals
# The same counts with part C added (macros ending in ABCD; the ABD macros above are kept unchanged). Part C's per-model
# verdicts: its law combos (J-C1, J-C2, J-C4), J-C-BOUND's sub-families, J-C-READa's (model, family) pairs, J-C6 (Qwen2.5-14B)
# and the screened models; J-C5 is one pooled test over models and has no per-model verdict. The alternative reading of
# J-C-READb (the common part's evaluability rule: E4a's KO(H*) 0.590 / 0.678 < 0.7 counts against the line) is registered
# with the suffix Alt.
_vsc = [("C", c, k, c_["v"]) for (c, k, Z, dep), c_ in COMB8.items() if not c_["seed"]]
_vsc += [("C", "J-C-BOUND", k, b["v"].upper()) for (k, Z), b in BND8.items() if b]
_vsc += [("C", "J-C-READa", k, "NOT MET") for (k, D, w) in RDP8 if D in ("E1", "E2") and w == "H"]
_vsc += [("C", "J-C6", "qwen14", LC8["J-C6"]["v"])] + [("C", "J-C-SCREEN", k, LC8["J-C-SCREEN"]["v"]) for k in SCR8]
_vs4 = _vs + _vsc
_fam4 = sorted({FAM8[k] for _, _, k, v in _vs4 if v in ("MET", "NOT MET")})
_famm4 = sorted({FAM8[k] for _, _, k, v in _vs4 if v == "MET"})
_mods4 = sorted({k for _, _, k, v in _vs4 if v in ("MET", "NOT MET")})
_src4 = "STAGE8A/8B/8C/8D_SCORE.txt"
amac("JxFamiliesABCD", len(_fam4), f"model families with a confirmatory stage-8 result (a per-model verdict, met or not met) in any of the four parts: {', '.join(_fam4)}", _src4 + " PREDICTIONS", G8X, "{:d}")
amac("JxFamiliesMetABCD", len(_famm4), f"model families with at least one per-model MET in any of the four parts: {', '.join(_famm4)}", _src4 + " PREDICTIONS", G8X, "{:d}")
amac("JxModelsABCD", len(_mods4), f"models with a per-model verdict in any of the four parts: {', '.join(SH8[k] for k in _mods4)}", _src4 + " PREDICTIONS", G8X, "{:d}")
for w, fmt in (("Met", "{:d}"), ("Verd", "{:d}"), ("Exp", "{:.2f}")):
    vals = [macros[f"JxRisky{w}{p}"] for p in "ABCD"]
    amac(f"JxRisky{w}ABCD", sum((float if w == "Exp" else int)(v) for v in vals),
         f"the entry in total (parts A-D): risky (class R) account lines {dict(Met='met', Verd='with a verdict', Exp='expected met')[w]} ({' + '.join(vals)})", _src4 + " SUMMARY", G8X, fmt)
for w in ("Met", "NotMet", "NotEval", "Lines"):
    vals = [macros[f"J{p}Count{w}"] for p in "abcd"]
    amac(f"Jx{w}ABCD", sum(int(v) for v in vals), f"the entry in total (parts A-D): lines {dict(Met='MET', NotMet='NOT MET', NotEval='NOT EVALUABLE', Lines='counted')[w]} ({' + '.join(vals)})",
         _src4 + " SUMMARY", G8X, "{:d}")
amac("JxAccountLinesABCD", int(macros["JxLinesABCD"]) - int(macros["JbClassValLines"]), "the entry in total: account lines (every counted line but part B's measurement-validity lines)", _src4 + " SUMMARY", G8X, "{:d}")
for c in ("L", "M"):
    for w, fmt in (("Met", "{:d}"), ("Exp", "{:.2f}")):
        vals = [macros[f"J{p}Class{c}{w}"] for p in "abcd"]
        amac(f"JxClass{c}{w}ABCD", sum((float if w == "Exp" else int)(v) for v in vals), f"the entry in total: class-{c} account lines {'met' if w == 'Met' else 'expected met'} ({' + '.join(vals)})",
             _src4 + " SUMMARY", G8X, fmt)
    amac(f"JxClass{c}VerdABCD", sum(int(macros[f"J{p}Class{c}Lines"]) - int(macros[f"J{p}Class{c}NotEval"]) for p in "abcd"),
         f"the entry in total: class-{c} account lines with a verdict", _src4 + " SUMMARY", G8X, "{:d}")
for w, fmt, d in (("Met", "{:d}", "met"), ("Exp", "{:.2f}", "expected met")):
    amac(f"JxClassVal{w}ABCD", (float if w == "Exp" else int)(macros[f"JbClassVal{w}"]), f"the entry in total: measurement-validity lines (all class R, all in part B) {d}", _src4 + " SUMMARY", G8X, fmt)
amac("JxClassValVerdABCD", int(macros["JbClassValLines"]) - int(macros["JbClassValNotEval"]), "the entry in total: measurement-validity lines with a verdict", _src4 + " SUMMARY", G8X, "{:d}")
_pb8 = float(LC8["J-C-READb"]["prior"])
for p, base in (("C", "C"), ("ABCD", "ABCD")):
    amac(f"JxRiskyMet{p}Alt", int(macros[f"JxRiskyMet{base}"]), f"{'part C' if p == 'C' else 'the entry in total'}: risky account lines met, under the common-part reading of J-C-READb (NOT MET; no change)",
         _src4 + " SUMMARY", G8X, "{:d}")
    amac(f"JxRiskyVerd{p}Alt", int(macros[f"JxRiskyVerd{base}"]) + 1, f"{'part C' if p == 'C' else 'the entry in total'}: risky account lines with a verdict under that reading (J-C-READb added)",
         _src4 + " SUMMARY", G8X, "{:d}")
    amac(f"JxRiskyExp{p}Alt", float(macros[f"JxRiskyExp{base}"]) + _pb8, f"{'part C' if p == 'C' else 'the entry in total'}: risky account lines expected met under that reading (+ J-C-READb's prior {_pb8:.2f})",
         _src4 + " SUMMARY", G8X, "{:.2f}")
# the outcome record's totals (docs/PREREGISTRATION.md, "The entry in total (parts A-D)") are reproduced
assert [macros[f"Jx{w}ABCD"] for w in ("Met", "NotMet", "NotEval", "Lines")] == ["36", "21", "12", "69"] and macros["JxAccountLinesABCD"] == "64"
assert [macros[f"JxRisky{w}ABCD"] for w in ("Met", "Verd", "Exp")] == ["10", "27", "12.05"]
assert [macros[f"JxClass{c}{w}ABCD"] for c in ("L", "M", "Val") for w in ("Met", "Verd", "Exp")] == ["6", "6", "5.40", "16", "19", "15.80", "4", "5", "2.60"]
assert [macros[f"JxRisky{w}{p}Alt"] for p in ("C", "ABCD") for w in ("Met", "Verd", "Exp")] == ["1", "5", "2.30", "10", "28", "12.65"]


# ---------------------------------------------------------------- stage 8 tables (captions and labels live in the section files)
def mf8(name):
    return fl(macros[name])


def mci8(name, sign=False):
    """'est [lo, hi]' of a registered estimate (printed strings; negatives in math mode)."""
    return tci((macros[name], macros[name + "Lo"], macros[name + "Hi"]), sign)


VW8 = {"MET": r"\textbf{met}", "NOT MET": r"\textbf{not met}", "NOT EVALUABLE": r"\textbf{not evaluable}"}
# Part B's three tables are the part-B scorer's own output: copied unchanged (read and re-written; the bytes are asserted equal)
# v5 round 1: the arm label P1 (the discovery-sample cores' options format) is printed as the paper's format macro; nothing
# else changes, which the assertion checks
for _t in ("tab_fresh.tex", "tab_mass.tex", "tab_invariance.tex"):
    _src = (ROOT / "results/gpu_stage8b" / _t).read_text()
    (TAB / _t).write_text(_src.replace(" & P1 & ", r" & \fmtOpt{} & "))
    assert (TAB / _t).read_text().replace(r" & \fmtOpt{} & ", " & P1 & ") == _src, _t

# ---- tab_prereg_j.tex: every stage-8 line (code, class, kind, prior, met / evaluable, verdict): parts A, B and D by models,
# part C by its lines' units (combos, sub-families, (model, family) pairs, formats, screened models). Two halves side by
# side: parts A and B on the left, C and D on the right; the entry's totals below both.
_left, _right = [], []
# a row spanning one half: the left half keeps the 1.5em gap to the right half that the column spec puts after column 6
LHALF = r"\multicolumn{6}{@{}p{0.47\textwidth}@{\hspace{1.5em}}}"
HALF = r"\multicolumn{6}{@{}p{0.47\textwidth}@{}}"


def _jrow(code, cls, kind, prior, met, ev, v, note=""):
    return f"{code} & {cls} & {kind} & {prior} & {met}/{ev} & {VW8[v]}{note}"


assert "all part-A lines are account lines" in S8A[SA8]
_left.append(LHALF + r"{\textbf{Part A}: natural passages (Llama-3.1-8B, Gemma-2-9B, Qwen2.5-7B, Mistral-7B)}")
for code, L in LA8.items():
    _left.append(_jrow(code, L["cls"], "--" if L["cls"] == "D" else "A", L["prior"], L["met"], L["ev"], L["v"],
                       r"$^\ddagger$" if L["cls"] == "D" else ""))
_left.append(LHALF + r"{\footnotesize \JaCountMet{} met, \JaCountNotMet{} not met, \JaCountNotEval{} not evaluable of \JaCountLines{} counted lines; "
             r"risky account lines met: \JxRiskyMetA{} of \JxRiskyVerdA{} with a verdict (expected \JxRiskyExpA{}).}")
_left.append(r"\addlinespace")
_left.append(LHALF + r"{\textbf{Part B}: fresh samples (P4: Qwen2.5-7B, Qwen2.5-14B, Mistral-7B, OLMo-2-7B; N4: Llama-3.1-8B, Gemma-2-9B, Phi-4, Falcon3-7B)}")
for code, L in LB8.items():
    _left.append(_jrow(code, L["cls"], L["kind"], L["prior"], L["met"], L["ev"], L["v"]))
_left.append(LHALF + r"{\footnotesize \JbCountMet{} met, \JbCountNotMet{} not met, \JbCountNotEval{} not evaluable of \JbCountLines{} lines; "
             r"risky account lines met: \JxRiskyMetB{} of \JxRiskyVerdB{} with a verdict (expected \JxRiskyExpB{}).}")
_right.append(HALF + r"{\textbf{Part C}: independently obtained identity edits (Qwen2.5-7B, Mistral-7B, Llama-3.1-8B; J-C6 at Qwen2.5-14B; the screen at "
              r"Qwen2.5-7B and Llama-3.1-8B)}")
for code, L in LC8.items():
    _right.append(_jrow(code, L["cls"], L["kind"], L["prior"], macros[f"Jc{LWC8[code]}Met"], macros[f"Jc{LWC8[code]}Eval"], L["v"], r"$^\S$" if code == "J-C-READb" else ""))
_right.append(HALF + r"{\footnotesize \JcCountMet{} met, \JcCountNotMet{} not met, \JcCountNotEval{} not evaluable of \JcCountLines{} lines; "
              r"risky account lines met: \JxRiskyMetC{} of \JxRiskyVerdC{} with a verdict (expected \JxRiskyExpC{}); headline: outcome (\JcHeadline{}). "
              r"Met: combos (J-C1--J-C4), sub-families (J-C-BOUND), (model, family) pairs (J-C-READ), formats (J-C6) or screened models (J-C-SCREEN, J-C-WIN) "
              r"met / with a verdict; J-C5 is one pooled test. $^\S$Not met under the common part's evaluability rule (KO($H^*$) of E4, seed 101, "
              r"\JcReadbKOEfouraQwenSeven{} and \JcReadbKOEfouraMistralSeven{} $< 0.7$, recomputed): risky lines met \JxRiskyMetCAlt{} of \JxRiskyVerdCAlt{} "
              r"(expected \JxRiskyExpCAlt{}).}")
_right.append(r"\addlinespace")
_right.append(HALF + r"{\textbf{Part D}: the flag (Qwen2.5-7B, Mistral-7B; J-D6, J-D6a, J-D6-ROUTE at Qwen2.5-1.5B, 3B, 7B)}")
for code, L in LD8.items():
    _right.append(_jrow(code, L["cls"], L["kind"], L["prior"], macros[f"Jd{LWD[code]}Met"], macros[f"Jd{LWD[code]}Eval"], L["v"]))
_right.append(HALF + r"{\footnotesize \JdCountMet{} met, \JdCountNotMet{} not met, \JdCountNotEval{} not evaluable of \JdCountLines{} lines; "
              r"risky account lines met: \JxRiskyMetD{} of \JxRiskyVerdD{} with a verdict (expected \JxRiskyExpD{}).}")
_n8 = max(len(_left), len(_right))
_left += [" & & & & & "] * (_n8 - len(_left))
_right += [" & & & & & "] * (_n8 - len(_right))
lines = [r"% two halves of 0.47\textwidth side by side: set this table in a table* float (full page width)",
         r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccll@{\hspace{1.5em}}lcccll@{}}", r"\toprule",
         r"Line & Cl. & Kind & Prior & Met & Verdict & Line & Cl. & Kind & Prior & Met & Verdict \\", r"\midrule"]
for a, b in zip(_left, _right):
    if a == r"\addlinespace" or b == r"\addlinespace":
        a = " & & & & & " if a == r"\addlinespace" else a
        b = " & & & & & " if b == r"\addlinespace" else b
    lines.append(f"{a} & {b} \\\\")
# the entry's totals, full width below both halves
lines += [r"\midrule",
          r"\multicolumn{12}{@{}p{0.94\textwidth}@{}}{\textbf{Parts A--D}: \JxMetABCD{} met, \JxNotMetABCD{} not met, \JxNotEvalABCD{} not evaluable of "
          r"\JxLinesABCD{} lines (\JxAccountLinesABCD{} account lines and \JbClassValLines{} measurement-validity lines); risky account lines met: \JxRiskyMetABCD{} of "
          r"\JxRiskyVerdABCD{} with a verdict (expected \JxRiskyExpABCD{}; \JxRiskyMetABCDAlt{} of \JxRiskyVerdABCDAlt{}, expected \JxRiskyExpABCDAlt{}, under the "
          r"common-part reading of J-C-READb).} \\",
          r"\bottomrule", r"\end{tabular}"]
wtab("tab_prereg_j.tex", lines)

# ---- tab_natural8.tex: part A, per model and format: n_comp, s_ID, ID_K and ID_V on C(f), and the gate that fails
lines = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrcccl@{}}", r"\toprule",
         r"Model & Format & $n_{\mathrm{comp}}$ & $s_{\mathrm{ID}}$ & $\mathrm{ID}_K$ (nats) & $\mathrm{ID}_V$ (nats) & Gate \\", r"\midrule"]
for n_, k in enumerate(MA8):
    for j, (f, fw) in enumerate(FA8.items()):
        dag = r"$^\dagger$" if f in ("MENA", "OPTB", "MENB") else ""
        gate = ", ".join(f"J-A-{g}" for g in NE8A.get((k, f), [])) or "--"
        lines.append(f"{SH8[k] if j == 0 else ''} & {FLAB8[f]}{dag} & {macros[f'JaNcomp{fw}{K8[k]}']} & {tcell(A8P[k, f, 'Sid'])} & "
                     f"{tcell(A8P[k, f, 'IDK'])} & {tcell(A8P[k, f, 'IDV'])} & {gate} \\\\")
    lines.append(r"\midrule" if n_ < len(MA8) - 1 else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_natural8.tex", lines)

# ---- tab_flag.tex: part D, every line's numbers per model (95 % story-bootstrap CIs); then the scale lines (1.5B, 3B, 7B)
FR8 = [("J-D1", [(r"$\mathrm{ID}^{\mathrm{LOO}}_{\mathrm{inj}}/\mathrm{ID}_K$", "JdOneRatio", True), (r"$\pi_X$", "JdOnePiX", False)]),
       ("J-D2", [(r"$\iota$(move $\Delta$)", "JdTwoIotaMove", True), (r"$\iota$(own write $\perp$ flag)", "JdTwoIotaOrth", True),
                 (r"$\iota$(mean $H^*$ output)", "JdTwoIotaMeanH", True), (r"$\iota$(active-set flag)", "JdTwoIotaActive", True)]),
       ("J-D3", [(r"$\rho_K$(flag)", "JdThreeRhoFlag", True), (r"$\rho_K$(pc1)", "JdThreeRhoPcone", True), (r"$\rho_K$(mean $H^*$ output)", "JdThreeRhoMeanH", True)]),
       ("J-D4", [(r"$r^{\mathrm{inj}}_{\mathrm{ans}}(KV)$", "JdFourKV", True), (r"$r^{\mathrm{inj}}_{\mathrm{ans}}(K)$", "JdFourK", True),
                 (r"$r^{\mathrm{inj}}_{\mathrm{ans}}(V)$", "JdFourV", True), (r"$r^{\mathrm{inj}}_{\mathrm{ans}}(K) - r^{\mathrm{inj}}_{\mathrm{ans}}(V)$", "JdFourKmV", True)]),
       ("J-D-ADDR", [(r"weighted $\cos(\Delta^K, \Delta^{KV})$", "JdAddrCos", False), (r"$\mathrm{ID}_{\mathrm{inj}}(KV)/\mathrm{ID}_{\mathrm{inj}}(K)$", "JdAddrRatio", True)]),
       ("J-D-KN", [("Pearson (per story)", "JdKnPearson", True), (r"B-loss ratio $\beta$", "JdKnBeta", True),
                   (r"mean$(\Delta\ell_S - \Delta\ell_X)$ (nats)", "JdKnDiff", True)]),
       ("J-D5", [(r"$\Psi_{\mathrm{bind}}$ pooled (nats)", "JdFivePooled", True), ("object-first", "JdFiveObj", True), ("distractor-first", "JdFiveDis", True),
                 ("stratum difference", "JdFiveDiff", True)]),
       ("J-D7", [(r"$\mathrm{ID}_K(Q_{\mathrm{OUT}})$ (nats)", "JdSevenQout", True), (r"$\mathrm{ID}_K(Q_{\mathrm{IN}})$ (nats)", "JdSevenQin", True),
                 ("paired difference (nats)", "JdSevenDiff", True)]),
       ("J-D-SIGN-Q", [(r"$\mathrm{ID}_V(Q_{\mathrm{OUT}})$ (nats)", "JdSignQIDV", True)]),
       ("J-D-SIGN-IOI", [(r"$\mathrm{ID}_K$, IOI in-sentence (nats)", "JdSignIoiIDK", True), (r"$\mathrm{ID}_V$, IOI in-sentence (nats)", "JdSignIoiIDV", True)]),
       ("J-D8-Q", [(r"$R(H^*)$, $Q_{\mathrm{IN}}$", "JdEightQinR", True), (r"$\mathrm{KO}(H^*)$, $Q_{\mathrm{IN}}$", "JdEightQinKO", True),
                   (r"$R(H^*)$, $Q_{\mathrm{OUT}}$", "JdEightQoutR", True), (r"$\mathrm{KO}(H^*)$, $Q_{\mathrm{OUT}}$", "JdEightQoutKO", True)]),
       ("J-D8-IOI", [(r"$R(H^*)$, IOI in-sentence", "JdEightIoiR", True), (r"$\mathrm{KO}(H^*)$, IOI in-sentence", "JdEightIoiKO", True)]),
       ("J-D-ROUTE-IOI", [(r"$r_{\mathrm{ans}}(KV)$", "JdRouteKV", True), (r"$r_{\mathrm{ans}}(K)$", "JdRouteK", True), (r"$r_{\mathrm{ans}}(V)$", "JdRouteV", True),
                          (r"$r_{\mathrm{ans}}(K) - r_{\mathrm{ans}}(V)$", "JdRouteKmV", True), (r"$r_{\mathrm{other}}$", "JdRouteOther", True)]),
       ("J-D-HOP2", [(r"carry, top-10 hop-2 heads, $Q_{\mathrm{OUT}}$", "JdHopTwoQoutCarry", True),
                     (r"carry, top-10 hop-2 heads, IOI", "JdHopTwoInlineCarry", True)])]


def _dcell(stem, k, ci):
    n_ = stem + K8[k]
    if n_ not in macros:
        why = LD8[next(c for c, rs in FR8 if any(s == stem for _, s, _ in rs))]["m"][k]
        assert why["v"] == "NOT EVALUABLE", (stem, k)
        return r"n.e.\ (" + why["txt"].replace(" not passed", "") + ")"
    return mci8(n_) if ci else tx(macros[n_])


lines = [r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lllll@{}}", r"\toprule", r"Line & Statistic & Qwen2.5-7B & Mistral-7B & Verdict \\", r"\midrule"]
for code, rows in FR8:
    for j, (lab, stem, ci) in enumerate(rows):
        lines.append(f"{code if j == 0 else ''} & {lab} & {_dcell(stem, 'qwen7', ci)} & {_dcell(stem, 'mistral7', ci)} & {VW8[LD8[code]['v']] if j == 0 else ''} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"", r"\medskip", r"",
          r"\begin{tabular}{@{}llcccl@{}}", r"\toprule", r"Line & Statistic & Qwen2.5-1.5B & Qwen2.5-3B & Qwen2.5-7B & Verdict \\", r"\midrule"]
FS8 = [("J-D6a", [(r"$\omega$", "JdSixaOmega", True), (r"$c_{\mathrm{fl}}$", "JdSixaKappa", False)]),
       ("J-D6", [(r"$\rho$", "JdSixRho", True), (r"$\rho_{\mathrm{nat}}$", "JdSixRhoNat", True), (r"$\rho - \rho_{\mathrm{nat}}$", "JdSixDiff", True)]),
       ("J-D6-ROUTE", [(r"$r^{\mathrm{inj}}_{\mathrm{ans}}(V)$", "JdSixRouteV", True), (r"$r^{\mathrm{inj}}_{\mathrm{ans}}(K)$", "JdSixRouteK", True),
                       (r"$r^{\mathrm{inj}}_{\mathrm{ans}}(V) - r^{\mathrm{inj}}_{\mathrm{ans}}(K)$", "JdSixRouteVmK", True)]),
       ("post hoc", [(r"natural ratio, $X$-specific contrast$^\dagger$", "JdSixNatX", True), (r"$\rho$ minus it$^\dagger$", "JdSixDiffX", True),
                     (r"natural ratio on $\mathrm{ID}_K$$^\dagger$", "JdSixNatIDK", True), (r"$\rho$ minus it$^\dagger$", "JdSixDiffIDK", True)]),
       ("D6b", [(r"$\Delta^{\mathrm{sentence}}/\Delta^{\mathrm{list}}$ at the list row", "JdSixb", True)])]
for code, rows in FS8:
    for j, (lab, stem, ci) in enumerate(rows):
        cells = []
        for k in MD8S:
            n_ = stem + K8[k]
            if n_ not in macros:
                cells.append("--")
            else:
                t3 = (macros[n_], macros[n_ + "Lo"], macros[n_ + "Hi"]) if ci else None
                c_ = tcell(t3) if ci else tx(macros[n_])
                if k == "qwen3b" and code == "J-D6":
                    c_ = two(tx(t3[0]) + r"$^\dagger$ (n.e.)", f"[{tx(t3[1])}, {tx(t3[2])}]")
                cells.append(c_)
        v = (VW8[LD8[code]["v"]] if code in LD8 else "--" if code == "post hoc" else "reported") if j == 0 else ""
        lines.append(f"{({'post hoc': 'J-D6, post hoc'}.get(code, code)) if j == 0 else ''} & {lab} & {' & '.join(cells)} & {v} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
wtab("tab_flag.tex", lines)

# ---- tab_law.tex: part C, every lambda statistic with its 90 % interval per model, edit and depth (appendix). * not
# equivalent (the 90 % interval is not inside +-log 1.25); iota: a departure by the interaction rule; dagger: recomputed
# through the scorer's classes (no printed value: E3, whose layers fail J-C-G3; E5, whose line prints only the pooled value;
# Qwen2.5-7B E2 at l = 11, in a combo with fewer than 3 statistics: its dagger is on the depth). The J-C5 statistics take
# their iota mark from the scorer's stats_of (the J-C5 lines print '*' only).
DAG8 = r"$^\dagger$"
SCOL8 = ["W(P1,{})", "W(POST,{})", "A_L({})", "A_P({})"]
ZORD8 = ["E1", "E2", "E3", "E4", "E4a", "E4b", "PERP:E2", "PERP:E4", "NONLEX:E4", "E5FR", "E5SYN", "E5NL"]
ZTEX8 = {"E1": "E1", "E2": "E2", "E3": "E3", "E4": "E4", "E4a": "E4, seed 101", "E4b": "E4, seed 102", "PERP:E2": "PERP of E2", "PERP:E4": "PERP of E4",
         "NONLEX:E4": "NONLEX of E4", "E5FR": "E5-FR", "E5SYN": "E5-SYN", "E5NL": "E5-NL"}


def lline8(s_):
    """the line a statistic belongs to, for the table."""
    for pre, lab in (("J-C1", "J-C1"), ("J-C2 (combo not evaluable", "J-C2 (n.e.)"), ("J-C2", "J-C2"), ("J-C4 (seed level)", "J-C4 (seed)"), ("J-C4", "J-C4"),
                     ("J-C5", "J-C5"), ("J-C-BOUND", "J-C-BOUND"), ("post hoc E3", "post hoc")):
        if s_.startswith(pre):
            return lab
    raise AssertionError(s_)


def lcell8(x):
    """lambda [90 % interval] (a printed -0.00 is set unsigned); * not equivalent; iota: the interaction rule."""
    if x is None:
        return "--"
    z = [v.lstrip("-") if fl(v) == 0 else v for v in x["t"]]
    return tci(z) + ("" if x["eq"] else ("$^{*\iota}$" if x["inter"] else "$^{*}$"))


_lr8 = []
for k in MC8:
    for Z in ZORD8:
        for l in DW8:
            cells = [LAMP8.get((k, Z, c.format(l))) for c in SCOL8]
            if not any(cells):
                continue
            got = [x for x in cells if x]
            assert len({x["line"] for x in got}) == 1 and len({x["rec"] for x in got}) == 1, (k, Z, l)
            _lr8.append((k, Z, l, cells, lline8(got[0]["line"]), got[0]["rec"]))
assert sum(sum(x is not None for x in r[3]) for r in _lr8) == len(LAMP8), (len(LAMP8), sum(sum(x is not None for x in r[3]) for r in _lr8))
lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lllcccc l@{}}", r"\toprule",
         r"Model & Edit & $l$ & $W(l)$, \fmtOpt{} & $W(l)$, \fmtPost{} & $A_L(l)$ & $A_P(l)$ & Line \\", r"\midrule"]
for n_, k in enumerate(MC8):
    rk = [r for r in _lr8 if r[0] == k]
    prev = None
    for j, (k_, Z, l, cells, line, rec) in enumerate(rk):
        # the dagger sits on the edit when all its rows are recomputed (E3, E5), else on the depth of each recomputed row
        # (Qwen2.5-7B E2 at l = 11)
        allrec = all(r[5] for r in rk if r[1] == Z)
        zlab_ = (ZTEX8[Z] + (DAG8 if allrec else "")) if Z != prev else ""
        lab_l = f"{l}{DAG8 if rec and not allrec else ''}"
        lines.append(f"{SH8[k] if j == 0 else ''} & {zlab_} & {lab_l} & " + " & ".join(lcell8(x) for x in cells) + f" & {line} \\\\")
        prev = Z
    lines.append(r"\midrule" if n_ < len(MC8) - 1 else r"\bottomrule")
lines.append(r"\end{tabular}")
wtab("tab_law.tex", lines)

# ---------------------------------------------------------------- stage 8 figures (no titles in the PDFs; captions come from LaTeX)
MD8META = {"CreationDate": None}   # reproducible PDFs
COLA8 = {"llama8": BLUE, "gemma9": ORANGE, "qwen7": AQUA, "mistral7": PURPLE}


def fresh_figure():
    """part B: s_ID^E per arm and model with its 95 % interval; P4 (blue) and N4 (orange) grouped; Gemma-2-2B grey and hollow."""
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.985)
    slots = [(k, BLUE, mk, "P4") for k, mk in zip(MB8P, "osD^")] + [(k, ORANGE, mk, "N4") for k, mk in zip(MB8N, "osD^")] + [("gemma2b", GREY, "v", "NE")]
    pos = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6, 9.0])
    pos = (pos - pos.mean()) * 0.083
    for j, arm in enumerate(AB8):
        ax.axvspan(j - 0.46, j + 0.46, color="#f4f3f0", lw=0, zorder=0)
        for (k, col, mk, grp), dx in zip(slots, pos):
            y, lo, hi = (mf8(f"JbSid{AB8[arm]}{K8[k]}{s}") for s in ("", "Lo", "Hi"))
            ax.plot([j + dx, j + dx], [lo, hi], color=col, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot(j + dx, y, marker=mk, ms=2.9, ls="none", color=col, mfc="white" if grp == "NE" else col, mew=0.6, zorder=3,
                    label=(SH8[k] + (" (not evaluable)" if grp == "NE" else "")) if j == 0 else None)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.set_xticks(range(len(AB8)))
    ax.set_xticklabels(["list\nafter", "sentence\nafter", "null\nsentence", "no\nmention", "sentence\nbefore", "list\nbefore"], fontsize=5.6)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlim(-0.5, len(AB8) - 0.5)
    ax.set_ylim(-0.1, 0.95)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_ylabel(r"$s_{\mathrm{ID}}^E$: key share of the identity", fontsize=6.3, labelpad=1)
    h_, l_ = ax.get_legend_handles_labels()
    hdr = matplotlib.lines.Line2D([], [], ls="none")
    lg1 = ax.legend([hdr] + h_[:4] + [hdr] + h_[4:8], [r"$\bf{discovery\ sample}$"] + l_[:4] + [r"$\bf{new\ families}$"] + l_[4:8],
                    ncol=2, frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.25, labelspacing=0.18,
                    columnspacing=0.6, handlelength=0.9, borderaxespad=0.15)
    ax.add_artist(lg1)
    ax.legend(h_[8:], l_[8:], frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.70), handletextpad=0.25, handlelength=0.9,
              borderaxespad=0.15)
    fig.text(0.005, 0.995, "(b)", fontsize=7, fontweight="bold", color=INK, ha="left", va="top")   # panel tag of Figure 1
    fig.savefig(FIG / "fig_fresh.pdf", metadata=MD8META)
    plt.close(fig)


fresh_figure()
ST8 = {"AFTER": (BLUE, "o"), "POST": (ORANGE, "D"), "POST-NULL": (AQUA, "^"), "NONE": (INK2, "s"), "PRE": (PURPLE, "v"), "BEFORE": (BROWN, "<")}


def crossover_figure():
    """part B behaviour: beta_K against beta_V per arm and model (eight P4 and N4 models), with the diagonal. Four arms have
    beta_K = 0.000 [0.000, 0.000] in every model and coincide on the beta_V axis; the strip below spreads them, one row per arm
    (P4 above N4 within a row), on a zoomed beta_V axis. The strip is a second view of the same points, not an offset."""
    Z8 = [a for a in ST8 if all(mf8(f"JbBetaK{AB8[a]}{K8[k]}{s_}") == 0 for k in MB8P + MB8N for s_ in ("", "Lo", "Hi"))]
    assert Z8 == ["POST-NULL", "NONE", "PRE", "BEFORE"], Z8
    W, H, R_ = 3.25, 3.3, 0.06
    zb, zh, mb, ML, ZL = 0.29, 0.66, 1.31, 0.40, 0.86   # inches: strip bottom and height, main bottom, left edges
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ML / W, mb / H, (W - ML - R_) / W, (H - mb - 0.04) / H])
    zx = fig.add_axes([ZL / W, zb / H, (W - ZL - R_) / W, zh / H])
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for arm, (col, mk) in ST8.items():
        for k in MB8P + MB8N:
            bk = [mf8(f"JbBetaK{AB8[arm]}{K8[k]}{s_}") for s_ in ("", "Lo", "Hi")]
            bv = [mf8(f"JbBetaV{AB8[arm]}{K8[k]}{s_}") for s_ in ("", "Lo", "Hi")]
            ax.plot([bv[1], bv[2]], [bk[0], bk[0]], color=col, lw=0.5, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.plot([bv[0], bv[0]], [bk[1], bk[2]], color=col, lw=0.5, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.plot(bv[0], bk[0], marker=mk, ms=3.6, ls="none", color=col, mfc=col if k in MB8P else "white", mew=0.7, zorder=3)
            if arm in Z8:
                y0 = len(Z8) - 1 - Z8.index(arm) + (0.2 if k in MB8P else -0.2)
                zx.plot([bv[1], bv[2]], [y0, y0], color=col, lw=0.6, alpha=0.6, zorder=2, solid_capstyle="butt")
                zx.plot(bv[0], y0, marker=mk, ms=3.0, ls="none", color=col, mfc=col if k in MB8P else "white", mew=0.6, zorder=3)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(labelsize=6)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel(r"$\beta_V$: answers moved by the value-only clamp", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel(r"$\beta_K$: answers moved by the key-only clamp", fontsize=6.5, labelpad=2)
    ax.text(0.635, 0.0, r"four arms at $\beta_K = 0$ (spread below)", fontsize=5.3, color=INK2, ha="right", va="center", zorder=4)
    # legends in the empty regions: the arms above the diagonal, the model groups below it (nothing crosses either)
    hs = [matplotlib.lines.Line2D([], [], marker=mk, ls="none", color=col, ms=3.6) for col, mk in ST8.values()]
    lg = ax.legend(hs, [ALAB8[a] for a in ST8], frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(0.27, 1.0), handletextpad=0.3,
                   labelspacing=0.25, borderaxespad=0.2)
    ax.add_artist(lg)
    hs = [matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, ms=3.6),
          matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=3.6, mew=0.7)]
    ax.legend(hs, ["discovery-sample models", "new families"], frameon=False, fontsize=5.6, loc="lower left", bbox_to_anchor=(0.30, 0.09),
              handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)
    lo = min(mf8(f"JbBetaV{AB8[a]}{K8[k]}Lo") for a in Z8 for k in MB8P + MB8N)
    zx.set_xlim(np.floor((lo - 0.02) * 20) / 20, 1.02)
    zx.set_xticks(np.arange(np.floor((lo - 0.02) * 20) / 20, 1.001, 0.1))
    zx.set_ylim(-0.6, len(Z8) - 0.4)
    zx.set_yticks(range(len(Z8)))
    zx.set_yticklabels([ALAB8[a] for a in reversed(Z8)], fontsize=5.6)
    for tl, a in zip(zx.get_yticklabels(), reversed(Z8)):
        tl.set_color(ST8[a][0])
    zx.tick_params(axis="y", length=0, pad=2)
    zx.tick_params(axis="x", labelsize=6)
    zx.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for y_ in np.arange(0.5, len(Z8) - 1, 1.0):
        zx.axhline(y_, color=GRID, lw=0.5, zorder=0)
    zx.set_xlabel(r"$\beta_V$ of these four arms (zoom; filled: discovery sample, hollow: new families)", fontsize=6.3, labelpad=1.5)
    fig.savefig(FIG / "fig_crossover.pdf", metadata=MD8META)
    plt.close(fig)


crossover_figure()


def crossover_main_figure():
    """Figure 1c: the scatter of fig_crossover without the zoom strip (the four arms at beta_K = 0 are spread out in the
    appendix figure), at the size of fig_fresh, with the panel tag (c)."""
    W, H, R_, ML, MB = 3.25, 2.3, 0.06, 0.40, 0.33
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ML / W, MB / H, (W - ML - R_) / W, (H - MB - 0.04) / H])
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for arm, (col, mk) in ST8.items():
        for k in MB8P + MB8N:
            bk = [mf8(f"JbBetaK{AB8[arm]}{K8[k]}{s_}") for s_ in ("", "Lo", "Hi")]
            bv = [mf8(f"JbBetaV{AB8[arm]}{K8[k]}{s_}") for s_ in ("", "Lo", "Hi")]
            ax.plot([bv[1], bv[2]], [bk[0], bk[0]], color=col, lw=0.5, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.plot([bv[0], bv[0]], [bk[1], bk[2]], color=col, lw=0.5, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.plot(bv[0], bk[0], marker=mk, ms=3.6, ls="none", color=col, mfc=col if k in MB8P else "white", mew=0.7, zorder=3)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(labelsize=6)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel(r"$\beta_V$: answers moved by the value-only clamp", fontsize=6.5, labelpad=1.5)
    # a shorter y label than the appendix figure's, so that the panel tag (c) in the top left corner clears it (v5 round 1)
    ax.set_ylabel(r"$\beta_K$: moved by the key-only clamp", fontsize=6.5, labelpad=2)
    ax.text(0.635, 0.0, r"four formats at $\beta_K = 0$ (Appendix D.4)", fontsize=5.3, color=INK2, ha="right", va="center", zorder=4)
    hs = [matplotlib.lines.Line2D([], [], marker=mk, ls="none", color=col, ms=3.6) for col, mk in ST8.values()]
    lg = ax.legend(hs, [ALAB8[a] for a in ST8], frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(0.27, 1.0), handletextpad=0.3,
                   labelspacing=0.25, borderaxespad=0.2)
    ax.add_artist(lg)
    hs = [matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, ms=3.6),
          matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=3.6, mew=0.7)]
    ax.legend(hs, ["discovery-sample models", "new families"], frameon=False, fontsize=5.6, loc="lower left", bbox_to_anchor=(0.30, 0.09),
              handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)
    fig.text(0.005, 0.995, "(c)", fontsize=7, fontweight="bold", color=INK, ha="left", va="top")
    fig.savefig(FIG / "fig_crossover_main.pdf", metadata=MD8META)
    plt.close(fig)


crossover_main_figure()


def natural_figure():
    """part A: (a) s_ID per format and model with its 95 % interval (hollow: the cell is not evaluable; a dagger on a format:
    its values are recomputed, as in tab_natural8.tex); (b) cue conflict, key-source against value-source rate (whole answer;
    the crosses are the rates at the decision token)."""
    fig = plt.figure(figsize=(6.75, 2.05))
    # v5 round 2: panel (b) gets more width and its legend sits outside the data (to the right of the square axes)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.35], wspace=0.22, left=0.065, right=0.99, bottom=0.15, top=0.91)
    ax = fig.add_subplot(gs[0, 0])
    offs = dict(zip(MA8, (-0.27, -0.09, 0.09, 0.27)))
    for j, f in enumerate(FA8):
        for k in MA8:
            y, lo, hi = (fl(s) for s in A8P[k, f, "Sid"])
            ne = (k, f) in NE8A
            x = j + offs[k]
            ax.plot([x, x], [lo, hi], color=COLA8[k], lw=0.9, zorder=2, solid_capstyle="butt")
            ax.plot(x, y, marker="o", ms=3.6, ls="none", color=COLA8[k], mfc="white" if ne else COLA8[k], mew=0.8, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    for xv in (2.5, 3.5):
        ax.axvline(xv, color=INK2, lw=0.5, ls=":", zorder=1)
    ax.set_xticks(range(len(FA8)))
    # the dagger marks the formats whose s_ID is recomputed from the raw results (not printed by the part-A scorer; rmac)
    rec = [f for f in FA8 if not any(f"JaSid{FA8[f]}{K8[k]}" in SRCLINE for k in MA8)]
    assert rec == ["MENA", "OPTB", "MENB"] and all(f"JaSid{FA8[f]}{K8[k]}" in SRCLINE for f in FA8 if f not in rec for k in MA8), rec
    ax.set_xticklabels([FLAB8[f].replace(" ", "\n") + (r"$^\dagger$" if f in rec else "") for f in FA8], fontsize=6.3)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlim(-0.5, len(FA8) - 0.5)
    ax.set_ylim(-0.1, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_ylabel(r"$s_{\mathrm{ID}}$: key share of the identity", fontsize=6.5)
    ax.set_title(r"(a) $s_{\mathrm{ID}}$ by format, counterfactual SQuAD passages", fontsize=7, color=INK, loc="left")
    # explicit handles: a model's key is its filled marker (its first plotted cell may be a hollow, not-evaluable one)
    h_ = [matplotlib.lines.Line2D([], [], marker="o", ls="none", color=COLA8[k], ms=3.6, mew=0.8) for k in MA8]
    l_ = [SH8[k] for k in MA8]
    h_.append(matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=3.6, mew=0.8))
    l_.append("not evaluable (gate)")
    ax.legend(h_, l_, frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, labelspacing=0.25, borderaxespad=0.3)
    bx = fig.add_subplot(gs[0, 1])
    bx.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    FMK = {"OPTA": "o", "LETA": "s", "NOM": "^"}
    for f, mk in FMK.items():
        for k in MA8:
            ky, kl, kh = (fl(s) for s in A8P[k, f, "key"])
            vy, vl, vh = (fl(s) for s in A8P[k, f, "val"])
            kd, vd = fl(A8P[k, f, "keyDec"][0]), fl(A8P[k, f, "valDec"][0])
            ne = (k, f) in NE8A
            bx.plot([vy, vd], [ky, kd], color=COLA8[k], lw=0.5, ls=":", zorder=2)
            bx.plot(vd, kd, marker="x", ms=2.6, mew=0.6, ls="none", color=COLA8[k], zorder=4)
            bx.plot([vl, vh], [ky, ky], color=COLA8[k], lw=0.5, alpha=0.4, zorder=2, solid_capstyle="butt")
            bx.plot([vy, vy], [kl, kh], color=COLA8[k], lw=0.5, alpha=0.4, zorder=2, solid_capstyle="butt")
            bx.plot(vy, ky, marker=mk, ms=4.2, ls="none", color=COLA8[k], mfc="white" if ne else COLA8[k], mew=0.8, zorder=3)
    bx.set_xlim(-0.03, 1.03)
    bx.set_ylim(-0.03, 1.03)
    bx.set_aspect("equal")
    bx.set_anchor("W")   # the square axes sit at the left of their column; the legend fills the right
    bx.set_xticks([0, 0.5, 1])
    bx.set_yticks([0, 0.5, 1])
    bx.tick_params(labelsize=6)
    bx.grid(color=GRID, lw=0.6, zorder=0)
    bx.set_xlabel("answers the value's entity", fontsize=6.3, labelpad=1)
    bx.set_ylabel("answers the key's entity", fontsize=6.3, labelpad=1)
    bx.set_title("(b) cue conflict", fontsize=7, color=INK, loc="left")
    hs = [matplotlib.lines.Line2D([], [], marker=mk, ls="none", color=INK2, ms=3.6) for mk in FMK.values()]
    hs.append(matplotlib.lines.Line2D([], [], marker="x", ls=":", color=INK2, ms=2.6, mew=0.6, lw=0.5))
    bx.legend(hs, [FLAB8[f] for f in FMK] + ["first token only"], frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5),
              handletextpad=0.3, labelspacing=0.35, borderaxespad=0.2)
    fig.savefig(FIG / "fig_natural.pdf", metadata=MD8META)
    plt.close(fig)


natural_figure()
COLD8 = {"qwen7": BLUE, "mistral7": ORANGE}
MKD8 = {"qwen7": "o", "mistral7": "s"}


def flag_figure():
    """part D: (a) iota of the dose, the move and the structured and isotropic controls; (b) rho_K; (c) the sign of the
    key and value reads under Q_IN, Q_OUT and IOI in-sentence re-mentions. The scale panel is fig_flag_scale."""
    # v5 round 1: panel (d), the scale lines, moved to its own appendix figure (fig_flag_scale), so the main figure has
    # three panels and the main text fewer symbols
    fig = plt.figure(figsize=(6.75, 1.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 0.9, 1.0], wspace=0.24, left=0.055, right=0.995, bottom=0.17, top=0.9)
    off = {"qwen7": -0.14, "mistral7": 0.14}

    def dot(ax, x, k, stem, hollow=False, ms=3.4):
        y = mf8(stem)
        if stem + "Lo" in macros:
            ax.plot([x, x], [mf8(stem + "Lo"), mf8(stem + "Hi")], color=COLD8[k], lw=0.8, zorder=2, solid_capstyle="butt")
        ax.plot(x, y, marker=MKD8[k], ms=ms, ls="none", color=COLD8[k], mfc="white" if hollow else COLD8[k], mew=0.7, zorder=3)

    # (a) iota
    ax = fig.add_subplot(gs[0, 0])
    cats = [("JdDoseHalf", "add\n" + r"0.5$\Delta$"), ("JdDoseOne", "add\n" + r"1$\Delta$"), ("JdDoseTwo", "add\n" + r"2$\Delta$"),
            ("JdTwoIotaMove", "move\n" + r"$\Delta$"), ("JdTwoIotaOrth", "orth."), ("JdTwoIotaActive", "active"),
            ("JdTwoIotaMeanH", "mean\n" + r"$H^*$"), (None, "iso.")]
    for j, (stem, _) in enumerate(cats):
        for k in MD8:
            if stem:
                dot(ax, j + off[k], k, stem + K8[k])
            else:
                for q, w in enumerate(("IsoZero", "IsoOne", "IsoTwo")):
                    ax.plot(j + off[k] + (q - 1) * 0.06, mf8(f"JdSanity{w}{K8[k]}"), marker=MKD8[k], ms=2.2, ls="none", color=COLD8[k], zorder=3)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axhline(1, color=INK2, lw=0.5, ls=":", zorder=1)
    ax.axvline(3.5, color=INK2, lw=0.5, ls=":", zorder=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[1] for c in cats], fontsize=6)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_ylabel(r"$\iota$: share of the natural effect on X", fontsize=6.3, labelpad=1)
    ax.set_title(r"(a) injecting the flag: $\iota$", fontsize=7, color=INK, loc="left")
    hs = [matplotlib.lines.Line2D([], [], marker=MKD8[k], ls="none", color=COLD8[k], ms=3.4) for k in MD8]
    ax.legend(hs, [SH8[k] for k in MD8], frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.9), handletextpad=0.2,
              labelspacing=0.2, borderaxespad=0.2)
    # (b) rho_K
    bx = fig.add_subplot(gs[0, 1])
    cats = [("JdThreeRhoFlag", "flag"), ("JdThreeRhoPcone", "pc1"), ("JdThreeRhoMeanH", "mean\n" + r"$H^*$"), (None, "iso.")]
    for j, (stem, _) in enumerate(cats):
        for k in MD8:
            if stem:
                dot(bx, j + off[k], k, stem + K8[k])
            else:
                for q, w in enumerate(("Zero", "One", "Two")):
                    dot(bx, j + off[k] + (q - 1) * 0.06, k, f"JdRhoIso{w}{K8[k]}", ms=2.2)
    bx.axhline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    bx.axhline(0.7, color=INK2, lw=0.5, ls=":", zorder=1)
    bx.text(3.48, 0.5, "0.5", fontsize=6, color=INK2, ha="right", va="bottom")
    bx.text(3.48, 0.7, "0.7", fontsize=6, color=INK2, ha="right", va="bottom")
    bx.set_xticks(range(len(cats)))
    bx.set_xticklabels([c[1] for c in cats], fontsize=6)
    bx.tick_params(axis="x", length=0, pad=2)
    bx.tick_params(axis="y", labelsize=6)
    bx.set_xlim(-0.5, len(cats) - 0.5)
    bx.set_ylim(0.3, 1.06)
    bx.set_yticks([0.4, 0.6, 0.8, 1.0])
    bx.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    bx.set_ylabel(r"$\rho_K$: key read left", fontsize=6.3, labelpad=1)
    bx.set_title(r"(b) ablating it: $\rho_K$", fontsize=7, color=INK, loc="left")
    # (c) sign
    cx = fig.add_subplot(gs[0, 2])
    cells = [("Qin", "mentioned"), ("Qout", "not\nmentioned"), ("Inline", "IOI")]
    w_ = 0.18
    for j, (cw, _) in enumerate(cells):
        for q, (k, ch) in enumerate([(k, ch) for k in MD8 for ch in ("IDK", "IDV")]):
            stem = f"JdCell{cw}{ch}{K8[k]}"
            x = j + (q - 1.5) * w_
            y = mf8(stem)
            cx.bar(x, y, w_ * 0.9, color=COLD8[k] if ch == "IDK" else "white", edgecolor=COLD8[k], lw=0.6, hatch="////" if ch == "IDV" else None, zorder=3,
                   yerr=[[y - mf8(stem + "Lo")], [mf8(stem + "Hi") - y]], error_kw=dict(lw=0.6, ecolor=INK, capsize=0))
    cx.axhline(0, color=INK2, lw=0.6, zorder=4)
    cx.set_xticks(range(len(cells)))
    cx.set_xticklabels([c[1] for c in cells], fontsize=6)
    cx.tick_params(axis="x", length=0, pad=2)
    cx.tick_params(axis="y", labelsize=6)
    cx.set_xlim(-0.5, len(cells) - 0.5)
    cx.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    cx.set_ylabel("identity read (nats)", fontsize=6.3, labelpad=1)
    cx.set_title("(c) sign of the reads", fontsize=7, color=INK, loc="left")
    hs = [matplotlib.patches.Patch(facecolor=INK2, edgecolor=INK2, lw=0.6), matplotlib.patches.Patch(facecolor="white", edgecolor=INK2, hatch="////", lw=0.6)]
    cx.legend(hs, [r"$\mathrm{ID}_K$", r"$\mathrm{ID}_V$"], frameon=False, fontsize=6, loc="lower left", handlelength=1.2, handletextpad=0.3,
              labelspacing=0.2, borderaxespad=0.2)
    fig.savefig(FIG / "fig_flag.pdf", metadata=MD8META)
    plt.close(fig)


def flag_scale_figure():
    """part D, appendix (v5 round 1; the former panel (d) of fig_flag): omega, rho and rho_nat at Qwen2.5-1.5B, 3B and 7B."""
    fig, dx = plt.subplots(figsize=(3.25, 2.1))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.11, top=0.98)
    # hollow: J-D6 is not evaluable at 3B (J-D-G6), and its rho and rho_nat there are recomputed (dagger); omega (J-D6a) is
    # evaluable at every scale. The post hoc X-specific ratio (recomputed; dagger) exists at 1.5B and 7B only: no line through 3B.
    SC8 = [("JdSixaOmega", INK, "o", r"$\omega$"), ("JdSixRho", AQUA, "D", r"$\rho$"), ("JdSixRhoNat", PURPLE, "s", r"$\rho_{\mathrm{nat}}$"),
           ("JdSixNatX", BROWN, "^", r"$\rho^X_{\mathrm{nat}}$ (post hoc)$^\dagger$")]
    hs, ls = [], []
    for q, (stem, col, mk, lab) in enumerate(SC8):
        xs_, ys_ = [], []
        for j, k in enumerate(MD8S):
            n_ = stem + K8[k]
            if n_ not in macros:
                continue
            x = j + (q - 1.5) * 0.09
            ne = k == "qwen3b" and stem != "JdSixaOmega"
            assert ne == (n_ not in SRCLINE) or stem == "JdSixNatX", n_   # hollow exactly where the value is recomputed
            dx.plot([x, x], [mf8(n_ + "Lo"), mf8(n_ + "Hi")], color=col, lw=0.8, zorder=2, solid_capstyle="butt")
            dx.plot(x, mf8(n_), marker=mk, ms=3.2, ls="none", color=col, mfc="white" if ne else col, mew=0.7, zorder=3)
            xs_.append(x), ys_.append(mf8(n_))
        if stem != "JdSixNatX":
            dx.plot(xs_, ys_, color=col, lw=0.6, ls="-", zorder=1)
        hs.append(matplotlib.lines.Line2D([], [], marker=mk, ls="none", color=col, ms=3.2, mew=0.7)), ls.append(lab)
    hs.append(matplotlib.lines.Line2D([], [], marker="s", ls="none", color=INK2, mfc="white", ms=3.2, mew=0.7))
    ls.append(r"not evaluable$^\dagger$")
    dx.axhline(0, color=INK2, lw=0.6, zorder=1)
    dx.set_xticks(range(len(MD8S)))
    dx.set_xticklabels(["1.5B", "3B", "7B"], fontsize=6)
    dx.tick_params(axis="x", length=0, pad=2)
    dx.tick_params(axis="y", labelsize=6)
    dx.set_xlim(-1.2, len(MD8S) - 0.5)   # a margin at the left for the legend, clear of the 1.5B column and of the 7B points
    dx.set_ylim(-0.25, 1.05)
    dx.set_yticks([0, 0.5, 1])
    dx.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    dx.set_ylabel(r"$\omega$; $\rho$ and $\rho_{\mathrm{nat}}$ (ratios)", fontsize=6.3, labelpad=1)
    # in the empty band under omega and above rho_nat (one column: two would run past the panel)
    dx.legend(hs, ls, frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.935), handletextpad=0.2, labelspacing=0.1,
              borderaxespad=0.1)
    fig.savefig(FIG / "fig_flag_scale.pdf", metadata=MD8META)
    plt.close(fig)


flag_figure()
flag_scale_figure()
COLC8 = {"nat": INK2, "E1": BLUE, "E2": AQUA, "E3": PURPLE, "E4": ORANGE, "E5": BROWN}
MKC8 = {"qwen7": "o", "mistral7": "s", "llama8": "^"}
LAMSY8 = dict(linthresh=0.25, linscale=0.9)       # the lambda axis: linear inside +-0.25, logarithmic beyond
LAMTK8 = ([-4, -2, -1, -0.5, -0.2, 0, 0.2], ["$-$4", "$-$2", "$-$1", "$-$0.5", "$-$0.2", "0", "0.2"])


def lam_axis8(ax, ylo=-4.6, yhi=0.42):
    """the lambda axis of fig_law (a) and fig_nu: symlog, the equivalence band +-log 1.25 and the copy-only bound log 0.5."""
    ax.set_yscale("symlog", **LAMSY8)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(LAMTK8[0])
    ax.set_yticklabels(LAMTK8[1])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhspan(-np.log(1.25), np.log(1.25), color="#e9eef7", lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.5, zorder=1)
    ax.axhline(np.log(0.5), color=INK2, lw=0.5, ls=":", zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)


def law_figure():
    """part C: (a) lambda per edit family, depth and statistic with 90 % intervals and the +-log 1.25 band (E3 hollow: not
    evaluable, post hoc; hollow too: Qwen2.5-7B E2 at l = 11, in a combo without a verdict); (b) kappa against sigma per cell
    at l = 3 and 7 (descriptive); (c) phi under NONE of the PAR, LEX, PERP and NONLEX components beside the full edit; (d)
    KO(H*) and the random-set mean KO per donor at P1, l = 7 (the natural donor's random-set mean, E3 and E4a recomputed).
    Marker shape is the model in every panel (circle Qwen2.5-7B, square Mistral-7B, triangle Llama-3.1-8B) and colour the
    edit family; fill is explained per panel ((a) verdict, (b) depth, (d) H* or random sets)."""
    fig = plt.figure(figsize=(6.75, 2.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 1.0, 1.3, 1.0], wspace=0.42, left=0.068, right=0.99, bottom=0.2, top=0.9)
    # (a) lambda
    ax = fig.add_subplot(gs[0, 0])
    lam_axis8(ax)
    fams = [("E1", (3, 7, 11)), ("E2", (3, 7, 11)), ("E3", (3, 7, 11)), ("E4", (3, 7))]
    x0, ticks, tlab, fx = 0.0, [], [], []
    for F, deps in fams:
        xs_ = []
        for l in deps:
            items = [(k, c.format(l)) for k in MC8 for c in SCOL8 if (k, F, c.format(l)) in LAMP8]
            offs = np.linspace(-0.38, 0.38, len(items)) if len(items) > 1 else [0.0]
            for (k, code), dx in zip(items, offs):
                x = LAMP8[k, F, code]
                y, lo, hi = (fl(v) for v in x["t"])
                ax.plot([x0 + dx] * 2, [lo, hi], color=COLC8[F], lw=0.6, alpha=0.75, zorder=2, solid_capstyle="butt")
                ax.plot(x0 + dx, y, marker=MKC8[k], ms=2.6, ls="none", color=COLC8[F], mfc="white" if x["rec"] else COLC8[F], mew=0.6, zorder=3)
            ticks.append(x0), tlab.append(str(l)), xs_.append(x0)
            x0 += 1.0
        fx.append((F, np.mean(xs_)))
        x0 += 0.45
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab, fontsize=6)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlim(-0.6, x0 - 0.45 - 0.4)
    for F, xm in fx:
        ax.text(xm, -0.085, {"E1": "E1", "E2": "E2", "E3": r"E3$^\dagger$", "E4": "E4"}[F], transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="top", fontsize=6.3, color=COLC8[F], fontweight="bold")
    ax.text(-0.6, -0.03, "$l$:", transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes), ha="right", va="top", fontsize=6, color=INK2)
    ax.set_ylabel(r"$\lambda = \log \psi_K - \log \psi_V$", fontsize=6.3, labelpad=1)
    ax.set_title(r"(a) channel ratio of the edits, $\lambda$", fontsize=7, color=INK, loc="left")
    ax.text(x0 - 0.45 - 0.45, np.log(1.25), r"$\pm\log 1.25$", fontsize=6, color=INK2, ha="right", va="bottom")
    ax.text(-0.45, np.log(0.5), r"$\log 0.5$", fontsize=6, color=INK2, ha="left", va="bottom")
    hs = [matplotlib.lines.Line2D([], [], marker=MKC8[k], ls="none", color=INK2, ms=2.8, mew=0.6) for k in MC8]
    hs.append(matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=2.8, mew=0.6))
    ax.legend(hs, [SH8[k] for k in MC8] + ["no verdict"], frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.2,
              labelspacing=0.15, borderaxespad=0.3, ncol=2, columnspacing=0.8)
    # (b) kappa against sigma: the model by shape as in (a), the depth by fill (filled l = 3, open l = 7)
    bx = fig.add_subplot(gs[0, 1])
    bx.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for (k, f, l), (sg, kp) in KS8.items():
        if l not in (3, 7):
            continue
        for Z in ("E1", "E2", "E4a", "E4b"):
            if Z in kp:
                F = Z[:2]
                bx.plot(fl(sg), fl(kp[Z]), marker=MKC8[k], ms=2.6, ls="none", color=COLC8[F], mfc=COLC8[F] if l == 3 else "white", mew=0.6,
                        zorder=3 if F != "E4" else 4, alpha=0.9)
    bx.set_xlim(-0.05, 1.05)
    bx.set_ylim(-0.05, 1.05)
    bx.set_xticks([0, 0.5, 1])
    bx.set_yticks([0, 0.5, 1])
    bx.tick_params(labelsize=6)
    bx.grid(color=GRID, lw=0.5, zorder=0)
    bx.set_xlabel(r"$\sigma$: natural key share", fontsize=6.3, labelpad=1)
    bx.set_ylabel(r"$\kappa$: the edit's key share", fontsize=6.3, labelpad=1)
    bx.set_title(r"(b) $\kappa$ against $\sigma$", fontsize=7, color=INK, loc="left")
    hs = [matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, ms=2.8, mew=0.6),
          matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=2.8, mew=0.6)]
    bx.legend(hs, ["filled: $l = 3$", "open: $l = 7$"], frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.03, 0.0), handletextpad=0.1,
              labelspacing=0.1, borderaxespad=0.1)
    # (c) the components
    cx = fig.add_subplot(gs[0, 2])
    comps = [("edit", "edit"), ("PAR", "PAR"), ("LEX", "LEX"), ("PERP", r"PERP$^\dagger$"), ("NONLEX", r"NONLEX$^\dagger$")]
    foff = {"E1": -0.25, "E2": 0.0, "E4": 0.25}
    moff = {"qwen7": -0.06, "mistral7": 0.0, "llama8": 0.06}
    for j, (c_, _) in enumerate(comps):
        for (k, Z, l), v in LEXP8.items():
            if Z == "E3":
                continue
            y = {"edit": fl(v["phi"]), "PAR": fl(v["par"]), "LEX": fl(v["lex"])}.get(c_)
            if y is None:
                y = LEXC8[k, Z, l, c_]
            cx.plot(j + foff[Z] + moff[k], y, marker=MKC8[k], ms=2.2, ls="none", color=COLC8[Z], mew=0.5, alpha=0.85, zorder=3)
    cx.axhline(0.3, color=INK2, lw=0.5, ls=":", zorder=1)
    cx.axhline(0, color=INK2, lw=0.5, zorder=1)
    cx.text(-0.45, 0.31, "0.3", fontsize=6, color=INK2, ha="left", va="bottom")
    for xv in np.arange(0.5, len(comps) - 1, 1.0):
        cx.axvline(xv, color=GRID, lw=0.5, zorder=0)
    cx.set_xticks(range(len(comps)))
    cx.set_xticklabels([c[1] for c in comps], fontsize=6, rotation=30, ha="right", rotation_mode="anchor")
    cx.tick_params(axis="x", length=0, pad=1.5)
    cx.tick_params(axis="y", labelsize=6)
    cx.set_xlim(-0.5, len(comps) - 0.5)
    cx.set_ylim(-0.12, 1.12)
    cx.set_yticks([0, 0.5, 1])
    cx.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    cx.set_ylabel(r"$\varphi$ without a mention", fontsize=6.3, labelpad=1)
    cx.set_title("(c) components of the edits", fontsize=7, color=INK, loc="left")
    # the families by colour patches (the marker shapes are the models, as in (a))
    hs = [matplotlib.patches.Patch(facecolor=COLC8[F], edgecolor="none") for F in ("E1", "E2", "E4")]
    cx.legend(hs, ["E1", "E2", "E4"], frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.02, 0.86), handlelength=0.9, handleheight=0.8,
              handletextpad=0.3, labelspacing=0.15, borderaxespad=0.1)
    # (d) the reader heads
    dx = fig.add_subplot(gs[0, 3])
    donors = [("nat", r"nat.$^\dagger$"), ("E1", "E1"), ("E2", "E2"), ("E3", r"E3$^\dagger$"), ("E4a", r"E4$^\dagger$")]
    mo = {"qwen7": -0.14, "mistral7": 0.14}
    YD8 = (-0.2, 1.2)
    for j, (D, _) in enumerate(donors):
        F = "nat" if D == "nat" else D[:2]
        for k in ("qwen7", "mistral7"):
            if D in ("E1", "E2"):
                kh = tuple(fl(x) for x in RDP8[k, D, "H"])
                kr = tuple(fl(x) for x in RDP8[k, D, "rand"])
            elif D == "nat":
                kh = tuple(fl(x) for x in RDP8[k, "nat", "H"])
                kr = tuple(fl(macros[f"JcReadaRandNat{K8[k]}{s_}"]) for s_ in ("", "Lo", "Hi"))
            elif (k, D) in RDR8:
                kq, rq = RDR8[k, D]
                kh, kr = (kq.pt, *kq.ci()), (rq.pt, *rq.ci())
            else:
                continue
            x = j + mo[k]
            for (y, lo, hi), filled in ((kh, True), (kr, False)):
                # every interval is drawn; one that runs past the axis is cut at it and marked with a caret there (E4a at
                # Mistral-7B, whose key read is 0.18 nats: its random-set interval runs past both ends)
                dx.plot([x, x], [max(lo, YD8[0]), min(hi, YD8[1])], color=COLC8[F], lw=0.6, alpha=0.75, zorder=2, solid_capstyle="butt")
                for cut, end, mk_ in ((lo < YD8[0], YD8[0] + 0.035, "v"), (hi > YD8[1], YD8[1] - 0.035, "^")):
                    if cut:
                        dx.plot(x, end, marker=mk_, ms=2.4, ls="none", color=COLC8[F], mew=0, alpha=0.75, zorder=2)
                dx.plot(x, y, marker=MKC8[k], ms=2.8, ls="none", color=COLC8[F], mfc=COLC8[F] if filled else "white", mew=0.6, zorder=3)
    dx.axhline(0.7, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    dx.axhline(0.25, color=INK2, lw=0.5, ls=":", zorder=1)
    dx.axhline(0, color=INK2, lw=0.5, zorder=1)
    dx.text(3.5, 0.71, "0.7", fontsize=6, color=INK2, ha="center", va="bottom")
    dx.text(3.5, 0.26, "0.25", fontsize=6, color=INK2, ha="center", va="bottom")
    dx.set_xticks(range(len(donors)))
    dx.set_xticklabels([d[1] for d in donors], fontsize=6)
    dx.tick_params(axis="x", length=0, pad=1.5)
    dx.tick_params(axis="y", labelsize=6)
    dx.set_xlim(-0.5, len(donors) - 0.5)
    dx.set_ylim(*YD8)
    dx.set_yticks([0, 0.5, 1])
    dx.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    dx.set_ylabel("KO: share of the key read removed", fontsize=6.0, labelpad=1)
    dx.set_title("(d) reader heads", fontsize=7, color=INK, loc="left")
    hs = [matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, ms=2.8, mew=0.6), matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=2.8, mew=0.6)]
    dx.legend(hs, [r"$H^*$", "random sets"], frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(-0.02, 0.02), handletextpad=0.1, labelspacing=0.1, borderaxespad=0.1)
    fig.savefig(FIG / "fig_law.pdf", metadata=MD8META)
    plt.close(fig)


def nu_figure():
    """part C, post hoc: lambda against nu (the distance of the edit's K/V tables from the natural ones) for every statistic of
    every identity-carrying instance (phi(NONE) >= 0.3); colour by family; circles full edits, triangles PERP, squares
    NONLEX components; filled = equivalent; the band +-log 1.25, the copy-only bound log 0.5 and the bands of nu with their
    equivalent counts."""
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.9)
    lam_axis8(ax, -4.6, 0.42)
    for p in sorted(NUP8, key=lambda p: p["eq"]):
        head, _, base = p["Z"].rpartition(":")
        F = "E5" if base.startswith("E5") else base[:2]
        mk = {"": "o", "PERP": "^", "NONLEX": "s"}[head]
        ax.plot(p["nu"], p["lam"], marker=mk, ms=3.0, ls="none", color=COLC8[F], mfc=COLC8[F] if p["eq"] else "white", mew=0.6, alpha=0.9, zorder=3)
    for xv in (0.1, 0.3):
        ax.axvline(xv, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for (lo, hi), w in (((0, 0.1), "Low"), ((0.1, 0.3), "Mid"), ((0.3, 1.42), "High")):
        ax.text((lo + hi) / 2, 1.005, f"{macros[f'JcNuEq{w}']}/{macros[f'JcNuN{w}']}" + (" equivalent" if w == "High" else ""),
                transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes), ha="center", va="bottom", fontsize=6, color=INK2)
    ax.text(0.015, np.log(0.5), r"$\log 0.5$", fontsize=6, color=INK2, ha="left", va="bottom", zorder=4)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.1, 0.3, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.1", "0.3", "0.5", "1"])
    ax.tick_params(labelsize=6)
    ax.set_xlabel(r"$\nu$: distance of the edit's K/V tables from the natural ones", fontsize=6.3, labelpad=1.5)
    ax.set_ylabel(r"$\lambda = \log \psi_K - \log \psi_V$", fontsize=6.3, labelpad=1)
    # shape: full edit or component (grey half-tone handles); fill: equivalent (filled) or not (open); at the top, clear of
    # every point. Colour: the family, in the empty lower left.
    hs = [matplotlib.lines.Line2D([], [], marker=m_, ls="none", color=INK2, mfc=LGREY, ms=3.0, mew=0.6) for m_ in ("o", "^", "s")]
    hs += [matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, ms=3.0, mew=0.6),
           matplotlib.lines.Line2D([], [], marker="o", ls="none", color=INK2, mfc="white", ms=3.0, mew=0.6)]
    lg = ax.legend(hs, ["full edit", "PERP comp.", "NONLEX comp.", "filled: equivalent", "open: not equivalent"], frameon=False, fontsize=6,
                   loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.25, labelspacing=0.12, borderaxespad=0.2, ncol=2, columnspacing=0.8,
                   handlelength=0.9)
    ax.add_artist(lg)
    hs = [matplotlib.patches.Patch(facecolor=COLC8[F], edgecolor="none") for F in ("E1", "E2", "E3", "E4", "E5")]
    lg = ax.legend(hs, ["E1", "E2", "E3", "E4", "E5"], frameon=True, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.25,
                   labelspacing=0.12, borderaxespad=0.15, borderpad=0.25, ncol=3, columnspacing=0.6, handlelength=0.9, handleheight=0.7)
    lg.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0)   # the nu lines pass behind it
    lg.set_zorder(5)
    fig.savefig(FIG / "fig_nu.pdf", metadata=MD8META)
    plt.close(fig)
law_figure()
nu_figure()


# ---------------------------------------------------------------- v5 revision, round 1: post hoc summaries the reviews asked for
# All of these are computed after the data, from printed verdicts and recorded priors or from the stored statistics; the
# paper labels them post hoc. (a) Poisson-binomial tails of the class tallies under the recorded priors (independent lines).
def _pb_pmf(ps):
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for p_ in ps:
        pmf[1:] = pmf[1:] * (1 - p_) + pmf[:-1] * p_
        pmf[0] *= 1 - p_
    return pmf


_LINES8 = ([("A", c, L["cls"], "A", float(L["prior"]), L["v"]) for c, L in LA8.items() if L["cls"] != "D"]
           + [("B", c, L["cls"], L["kind"], float(L["prior"]), L["v"]) for c, L in LB8.items()]
           + [("C", c, L["cls"], L["kind"], float(L["prior"]), L["v"]) for c, L in LC8.items()]
           + [("D", c, L["cls"], L["kind"], float(L["prior"]), L["v"]) for c, L in LD8.items()])
assert len(_LINES8) == 69, len(_LINES8)
_VD8 = ("MET", "NOT MET")
_R8 = [x for x in _LINES8 if x[2] == "R" and x[3] == "A" and x[5] in _VD8]
assert (len(_R8), sum(x[5] == "MET" for x in _R8), f"{sum(x[4] for x in _R8):.2f}") == (27, 10, "12.05")
_src_pb = "docs/PREREGISTRATION.md (recorded priors) and the four STAGE8*_SCORE.txt verdicts"
_pm = _pb_pmf([x[4] for x in _R8])
rmac("JxRiskyTailP", _pm[:11].sum(), "post hoc: P(at most 10 of the 27 risky account lines met) under the recorded priors (Poisson binomial, lines independent)", _src_pb, G8X, "{:.2f}")
for P_, w, tail in (("A", "le", 3), ("B", "ge", 4), ("C", "le", 1), ("D", "le", 2)):
    _ps = [x[4] for x in _R8 if x[0] == P_]
    _m = _pb_pmf(_ps)
    rmac(f"JxRiskyTailP{P_}", _m[:tail + 1].sum() if w == "le" else _m[tail:].sum(),
         f"post hoc: P(at {'most' if w == 'le' else 'least'} {tail} of part {P_}'s {len(_ps)} risky account lines met) under the recorded priors", _src_pb, G8X, "{:.2f}")
_MC8 = [x for x in _LINES8 if x[2] == "M" and x[0] == "C" and x[5] in _VD8]
assert (len(_MC8), sum(x[5] == "MET" for x in _MC8), f"{sum(x[4] for x in _MC8):.2f}") == (4, 1, "3.30")
rmac("JcClassMTailP", _pb_pmf([x[4] for x in _MC8])[:2].sum(), "post hoc: P(at most 1 of part C's 4 extrapolation lines met) under the recorded priors", _src_pb, G8X, "{:.3f}")
_MABD = [x for x in _LINES8 if x[2] == "M" and x[0] != "C" and x[5] in _VD8]
assert (macros["JxClassMMetABD"], macros["JxClassMVerdABD"], macros["JxClassMExpABD"]) == ("15", "15", "12.50")   # defined above
assert (sum(x[5] == "MET" for x in _MABD), len(_MABD), f"{sum(x[4] for x in _MABD):.2f}") == (15, 15, "12.50")
rmac("JxClassMTailPABD", _pb_pmf([x[4] for x in _MABD])[15:].sum(), "post hoc: P(all 15 extrapolation lines of parts A, B and D met) under the recorded priors", _src_pb, G8X, "{:.2f}")
# (b) the risky account lines that a gate made not evaluable, resolved as the outcome records' sensitivity notes say they
# would have come out had the gate passed or been read otherwise (post hoc; J-C-WIN has no window and cannot be resolved)
_CF8 = {"J-A6d": False,      # part A: had Llama's options-after cell passed J-A-G1 by one item (Llama S-rate 0.541 < 0.6)
        "J-A-HA1": True, "J-A-HA2": True, "J-A-HA3a": False, "J-A-HA3b": False,   # part A: with Qwen's heads gate waived
        "J-B-SMALL": False,  # part B: had Gemma-2-2B passed competence
        "J-C3": False,       # part C: under either post hoc dictionary gate
        "J-C-READb": False,  # part C: under the common part's evaluability rule
        "J-D8-IOI": True}    # part D: had Mistral's INLINE cell passed J-D-G4 (missed by 0.021)
_RNE8 = [x for x in _LINES8 if x[2] == "R" and x[3] == "A" and x[5] == "NOT EVALUABLE"]
assert {x[1] for x in _RNE8} == set(_CF8) | {"J-C-WIN"}, sorted(x[1] for x in _RNE8)
_cf = [(x[4], _CF8[x[1]]) for x in _RNE8 if x[1] in _CF8]
amac("JxRiskyNotEvalABCD", len(_RNE8), "risky account lines not evaluable (parts A-D)", _src_pb, G8X, "{:d}")
amac("JxRiskySensResolved", len(_cf), "post hoc: risky account lines not evaluable whose outcome the outcome records resolve (J-C-WIN cannot be)", _src_pb, G8X, "{:d}")
amac("JaGateFailCells", len(NE8A), "part A: model x format cells that failed a gate (J-A-G1 or J-A-G2), of 24", SRCA + " GATES", G8X, "{:d}")
amac("JxRiskySensHeld", sum(v for _, v in _cf), "post hoc: risky account lines not evaluable through a gate that would have been met (the outcome records' sensitivity notes)", _src_pb, G8X, "{:d}")
amac("JxRiskySensNotHeld", sum(not v for _, v in _cf), "post hoc: risky account lines not evaluable through a gate that would not have been met (the outcome records' sensitivity notes)", _src_pb, G8X, "{:d}")
amac("JxRiskySensMet", 10 + sum(v for _, v in _cf), "post hoc: risky account lines met with those lines resolved", _src_pb, G8X, "{:d}")
amac("JxRiskySensVerd", 27 + len(_cf), "post hoc: risky account lines with a verdict with those lines resolved", _src_pb, G8X, "{:d}")
amac("JxRiskySensExp", 12.05 + sum(p_ for p_, _ in _cf), "post hoc: risky account lines expected met with those lines resolved", _src_pb, G8X, "{:.2f}")
rmac("JxRiskySensTailP", _pb_pmf([x[4] for x in _R8] + [p_ for p_, _ in _cf])[:10 + sum(v for _, v in _cf) + 1].sum(),
     "post hoc: P(at most that many met) under the recorded priors, with those lines resolved", _src_pb, G8X, "{:.2f}")
assert (macros["JxRiskySensHeld"], macros["JxRiskySensNotHeld"], macros["JxRiskySensMet"], macros["JxRiskySensVerd"], macros["JxRiskySensExp"]) == ("3", "6", "13", "36", "16.70")
# (c) the discovery entries G-I counted over the predictions with a verdict (H6 not evaluable and H12 not run left out)
amac("GHIcountVerd", int(macros["GcountMet"]) + int(macros["GcountNotMet"]) + int(macros["HcountMet"]) + int(macros["HcountNotMet"])
     + int(macros["IcountMet"]) + int(macros["IcountNotMet"]), "predictions of G, H and I with a verdict (H6 not evaluable and H12 not run left out)", SNAME["S7"], G8X, "{:d}")
assert macros["GHIcountVerd"] == "39"
# (d) the nu pattern by edit family: steering vectors (full E1 and E2 edits), E3 (gate failed) and the rest (the remap, the
# non-lexical vectors and the PERP and NONLEX components); within-family Spearman correlations; the statistics with nu >= 0.3
# split into those whose 90 % interval lies wholly outside the margin and the inconclusive rest; a bootstrap over the units
def _fam8(p):
    return "steer" if p["Z"] in ("E1", "E2") else "E3" if p["Z"] == "E3" else "other"


_LOW8 = [p for p in NUP8 if p["nu"] < 0.3]
_HIGH8 = [p for p in NUP8 if p["nu"] >= 0.3]
for w, B_ in (("Below", _LOW8), ("Above", _HIGH8)):
    for f_, fw in (("steer", "Steer"), ("E3", "Ethree"), ("other", "Other")):
        rmac(f"JcNu{fw}{w}", sum(_fam8(p) == f_ for p in B_), f"post hoc nu pattern: statistics of family {f_} with nu {'< 0.3' if w == 'Below' else '>= 0.3'}", _src7, G8C, "{:d}")
rmac("JcNuNBelow", len(_LOW8), "post hoc nu pattern: statistics with nu < 0.3", _src7, G8C, "{:d}")
for f_, fw in (("steer", "Steer"), ("other", "Other")):
    _B = [p for p in NUP8 if _fam8(p) == f_]
    rmac(f"JcNuRho{fw}", spearmanr([p["nu"] for p in _B], [p["lam"] for p in _B])[0], f"post hoc nu pattern: Spearman of lambda with nu within family {f_} ({len(_B)} statistics)", _src7, G8C, "{:.2f}")
    rmac(f"JcNuN{fw}", len(_B), f"post hoc nu pattern: statistics of family {f_}", _src7, G8C, "{:d}")
_out8 = [p for p in _HIGH8 if (lambda lo, hi: hi < -np.log(1.25) or lo > np.log(1.25))(*p["q"].ci(0.90))]
rmac("JcNuOutHigh", len(_out8), "post hoc nu pattern: statistics with nu >= 0.3 whose 90 % interval lies wholly outside +-log 1.25 (a shown departure; the rest are inconclusive)", _src7, G8C, "{:d}")
_uk8 = sorted({(p["k"], p["Z"], p["l"]) for p in NUP8})
_ix8 = {u: [i for i, p in enumerate(NUP8) if (p["k"], p["Z"], p["l"]) == u] for u in _uk8}
_rs8 = np.random.default_rng(20261014)
_bo8 = []
for _ in range(2000):
    _ii = [i for u in _rs8.choice(len(_uk8), len(_uk8)) for i in _ix8[_uk8[u]]]
    _bo8.append(spearmanr([NUP8[i]["nu"] for i in _ii], [NUP8[i]["lam"] for i in _ii])[0])
rmac("JcNuRhoNuLo", np.percentile(_bo8, 2.5), "post hoc nu pattern: Spearman of lambda with nu, lower 95 % bound of a bootstrap over the units (model, instance, depth)", _src7, G8C, "{:.2f}")
rmac("JcNuRhoNuHi", np.percentile(_bo8, 97.5), "post hoc nu pattern: Spearman of lambda with nu, upper 95 % bound of a bootstrap over the units (model, instance, depth)", _src7, G8C, "{:.2f}")
print("v5 round 1:", {n: macros[n] for n in ("JxRiskyTailP", "JcClassMTailP", "JxClassMTailPABD", "JxRiskySensTailP", "JcNuSteerBelow", "JcNuEthreeBelow", "JcNuOtherBelow",
      "JcNuSteerAbove", "JcNuEthreeAbove", "JcNuOtherAbove", "JcNuRhoSteer", "JcNuRhoOther", "JcNuOutHigh", "JcNuRhoNuLo", "JcNuRhoNuHi",
      "JxRiskyTailPA", "JxRiskyTailPB", "JxRiskyTailPC", "JxRiskyTailPD")})


# ---------------------------------------------------------------- checks
# (1) every score-file macro still equals its line at the printed precision
for name, (tag, idx, s) in SRCLINE.items():
    assert s in SCORE[tag][idx] and macros[name] == sv(s), (name, s)
# (2) the decisive stage-6 values, re-derived from the raw JSON by the committed scorer classes, reproduce the score file
_buf = []
H6S.score(ROOT / "results/gpu_stage6/heads", out=_buf.append)
PKS.score(ROOT / "results/gpu_stage6/prakash", out=_buf.append)
_S6set = set(S6)
_buf = [y for x in _buf for y in x.split("\n") if y.strip()]
_miss = [x for x in _buf if x not in _S6set and "root " not in x]
assert not _miss, _miss[:5]
_dec = [n for n in SRCLINE if n.startswith(("Hone", "Htwo", "Hthree", "Hfour", "Hfive", "Hsix", "Hseven", "Height", "Hnine", "Heleven", "PkKappa", "PkPsi", "PkSid"))]
print(f"check: {len(SRCLINE)} score-file macros match their lines; stage-6 scorer output re-derived from raw ({len(_buf)} lines) matches "
      f"STAGE6_SCORE.txt line for line ({len(_dec)} decisive H/kappa/psi/s_ID macros covered)")
# (3) optional: re-run the stage-5 scorer (parts b-e reproduce byte for byte; part a needs the .npz sidecars, not in the repo)
if os.environ.get("VERIFY_STAGE5"):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, str(ROOT / "analysis/stage5_score.py"), "--out", f"{td}/s5.txt"], cwd=ROOT, check=False,
                       capture_output=True)
        new = Path(f"{td}/s5.txt").read_text().splitlines()
    a_ = new[new.index(next(x for x in new if x.startswith("######## PART (b)"))):]
    b_ = S5[B5:]
    assert a_ == b_, "stage-5 parts (b)-(e) do not reproduce"
    print("check: stage-5 scorer re-run reproduces parts (b)-(e) of STAGE5_SCORE.txt line for line")
# (4) v2 macros are untouched (nm() refuses to overwrite; this guards any later edit)
assert V2_MACROS <= set(macros)
print(f"v3: {len(REG)} new macros ({sum(1 for r in REG if 'recomputed' in r[3])} recomputed, {sum(1 for r in REG if 'arithmetic' in r[3])} arithmetic)")
# (5) stage 8: every macro defined before stage 8 keeps its value (every stage-8 score-file macro equals its line: (1));
# the new macros by group
assert all(macros[n] == v for n, v in PRE8.items()), [n for n, v in PRE8.items() if macros[n] != v][:5]
_s8 = REG[N_PRE8:]
assert all(r[0] in (G8A, G8B, G8C, G8D, G8X) for r in _s8)
_s8n = {r[1] for r in _s8}
print(f"stage 8: {len(_s8)} new macros (" + ", ".join(f"{g} {sum(1 for r in _s8 if r[0] == g)}" for g in (G8A, G8B, G8C, G8D, G8X)) +
      f"; {sum(1 for n in SRCLINE if n in _s8n)} printed strings checked against their score-file lines, {sum(1 for r in _s8 if 'recomputed' in r[3])} recomputed, "
      f"{sum(1 for r in _s8 if 'arithmetic' in r[3])} arithmetic, the rest ranges selecting printed strings); the {len(PRE8)} earlier macros are unchanged")

# ---------------------------------------------------------------- index of the new macros (for the writers)
# MACRO_INDEX=<file> lists every registered macro by group, as before stage 8. MACRO_INDEX_GROUPS=<g1,g2,...> (optional)
# lists only those groups in full and counts the others, e.g. MACRO_INDEX_GROUPS=S8A,S8B,S8C,S8D,S8X for docs/V5_MACROS.md.
if os.environ.get("MACRO_INDEX"):
    from collections import OrderedDict
    by = OrderedDict()
    for g, n, v, d, s in REG:
        by.setdefault(g, []).append((n, v, d, s))
    only = [g for g in os.environ.get("MACRO_INDEX_GROUPS", "").split(",") if g]
    assert all(g in by for g in only), [g for g in only if g not in by]
    out = ["# Index of the v3 macros (generated by paper/make_figures.py)", "",
           f"{len(REG)} new macros. Group = section of docs/V3_PLAN.md. Value = as in numbers.tex. "
           "'[recomputed]' = not printed by a scorer: label exploratory or post hoc in the paper.", ""]
    if len(REG) > N_PRE8:
        out += [f"Stage 8 (preregistration J; {len(REG) - N_PRE8} of the macros) has groups by part instead: S8A = part A (natural passages), "
                "S8B = part B (fresh samples, the emitted-form score), S8C = part C (independently obtained identity edits, the channel-ratio "
                "law), S8D = part D (the flag), S8X = counts across parts (...ABD: parts A, B and D, as registered before part C; ...ABCD: the "
                "entry in total). Names: J + part (Ja, Jb, Jc, Jd, Jx) + statistic + format or arm + model tag; "
                "...Lo and ...Hi are the interval bounds (95 %; the N4 verdict bounds of part B are 98.75 %; part C's lambda statistics and "
                "pooled lambda-bars have 90 % bounds, the TOST level; each description says which). "
                "Negative values are in math mode. '[arithmetic on printed values]' = a count or sum of printed values; a name ending "
                "in Min or Max selects a printed string. A value of -0.000 (or -0.0) is the scorer's printed string, kept verbatim; it is "
                "marked NEGATIVE ZERO below: in running text quote its interval or range instead.", ""]
    if only:
        out += [f"Generated with MACRO_INDEX_GROUPS={','.join(only)}: groups {', '.join(only)} are listed in full; the others are only counted "
                "(unset MACRO_INDEX_GROUPS to list every group).", ""]
        out += [f"- {g}: {len(r_)} macros" for g, r_ in by.items() if g not in only] + [""]
    negz = re.compile(r"-0(\.0+)?")
    for g, rows_ in by.items():
        if only and g not in only:
            continue
        out += [f"## {g}", "", "| macro | value | meaning | source |", "|---|---|---|---|"]
        out += [f"| `\\{n}` | `{v}` | {(f'**NEGATIVE ZERO: prints as {v} (a minus sign on zero); quote the interval or range instead.** ' if g in (G8A, G8B, G8C, G8D, G8X) and negz.fullmatch(v) else '') + d.replace('|', '/')} | {s} |"
                for n, v, d, s in rows_]
        out.append("")
    Path(os.environ["MACRO_INDEX"]).write_text("\n".join(out) + "\n")
    print("wrote macro index to", os.environ["MACRO_INDEX"])

with open(ROOT / "paper" / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by paper/make_figures.py from saved results. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and figures to {FIG}")
