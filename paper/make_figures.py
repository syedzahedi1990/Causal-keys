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
         r"Remap & Format & $\varphi$ & $\psi_K$ & $\rho_K$ & $\psi_V$ & $\rho_V$ \\", r"\midrule"]
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
    """recomputed from raw results (not printed by a scorer)."""
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
    (r"Removing the readers raises the value copy (G7a, H3)", NA, PH, NA, CK),
    (r"A remap refit without later mentions shows the same crossover (F1--F4)", NA, NA, XX, CK),
    (rf"Prakash et al.'s binding swap is key-dominant in every evaluable format ($\kappa$ {tx(PK['Bind', 'None']['kappa'][0])}, $\psi_V$ {tx(PK['Bind', 'None']['pv'])} without a later mention); later mentions shift $\kappa$ by {tx(macros['HnineDiff'], True)} (H7--H10)",
     NA, PH, NA, XX),
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
         r"Edit & Format & $\Phi$ (nats) & flips & $\psi_K$ & $\psi_V$ & interaction & $\kappa$ & $\kappa_w$ & $\rho_K$ & $\rho_V$ \\", r"\midrule"]
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
           f"$E$ {tci(G5[Q15, 'POST']['E'])}, $F/E$ {tx(G5[Q15, 'POST']['FE'][0])}, $\\mathrm{{ID}}_K$ {tx(M_('GtwoIDKLowQwenOnefive'))} (1.5B); "
           f"$E$ {tci(G5[Q3, 'POST']['E'])}, $F/E$ {tx(G5[Q3, 'POST']['FE'][0])}, $\\mathrm{{ID}}_K$ {tx(M_('GtwoIDKLowQwenThree'))} (3B)", VG5["G2"][1]))
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
           f"gaps {M_('HsevenGapNone')} / {M_('HsevenGapQnames')} / {M_('HsevenGapOpt')}; $r = {M_('HsevenR')}$", VH["H7"]))
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

# ---------------------------------------------------------------- index of the new macros (for the writers)
if os.environ.get("MACRO_INDEX"):
    from collections import OrderedDict
    by = OrderedDict()
    for g, n, v, d, s in REG:
        by.setdefault(g, []).append((n, v, d, s))
    out = ["# Index of the v3 macros (generated by paper/make_figures.py)", "",
           f"{len(REG)} new macros. Group = section of docs/V3_PLAN.md. Value = as in numbers.tex. "
           "'[recomputed]' = not printed by a scorer: label exploratory or post hoc in the paper.", ""]
    for g, rows_ in by.items():
        out += [f"## {g}", "", "| macro | value | meaning | source |", "|---|---|---|---|"]
        out += [f"| `\\{n}` | `{v}` | {d.replace('|', '/')} | {s} |" for n, v, d, s in rows_]
        out.append("")
    Path(os.environ["MACRO_INDEX"]).write_text("\n".join(out) + "\n")
    print("wrote macro index to", os.environ["MACRO_INDEX"])

with open(ROOT / "paper" / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by paper/make_figures.py from saved results. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and figures to {FIG}")
