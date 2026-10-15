"""The zero-GPU stage-9 pre-analyses (analysis/stage9_pre.py; docs/stage9/SPEC.md sections 13.5 and 15): every number that
docs/STAGE9_PREDICTIONS.md section 1 ("Seen before this note") quotes is reproduced from the committed results files at its
printed precision, and so are the in-hand analogues of section 2 that the same rows give. Each expected string is also
asserted to occur in the note (or, for the extra values that SPEC 15.2 lists, in the spec), so a later edit of either file
is caught. The pure helpers (the l_c bracket, the dose curve, the printed-line parsers) are tested on hand cases.

This file is one of the FP32 gates of scripts/gpu_stage9.sh and is frozen with the note. The note rounds every number
half up from the recomputed value (its section 1 says so once), so the values at an exact tie are matched by that rule
(``printed``: the printed number is the value rounded half up at the printed precision): row 4's E4b flips at Mistral
(39/40 and 1/40: 0.98 and 0.03) and P2's brackets (4/32 and 12/32: 0.13 and 0.38). The two numbers that differ from a
difference or ratio of other printed values are the recomputed ones (section 1 row 5's Spearman difference: 0.0076
unrounded, printed 0.008, while 0.862 - 0.855 = 0.007; section 2's P5b tau(3) at Mistral: 0.563 unrounded, printed 0.56,
while 0.56 / 0.99 = 0.57). The recomputed values themselves are pinned exactly. paper/numbers.tex is checked by
tests/test_paper_s9p.py (not a gate: the paper's macros may be regenerated after the freeze; they format the binary float,
which differs from half up only at an exact tie: \\NpEditFlipEfourbNoneMistralSeven prints 0.97 for 39/40). No model is
loaded; the module fixture reads the results once (about 30 s)."""
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import stage9_pre as P  # noqa: E402

NOTE = (ROOT / "docs/STAGE9_PREDICTIONS.md").read_text()
SPEC = (ROOT / "docs/stage9/SPEC.md").read_text()
SEEN = NOTE[NOTE.index("## 1. Seen before this note"):NOTE.index("## 2. Predictions")]
PRED = NOTE[NOTE.index("## 2. Predictions"):NOTE.index("## 3. Falsifiers")]
S152 = SPEC[SPEC.index("### 15.2 Items"):SPEC.index("### 15.3 Figure")]


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def ci2(e):
    return f"{f2(e[0])} [{f2(e[1])}, {f2(e[2])}]"


def row(n):
    """The text of row n of the note's section-1 table."""
    m = re.search(rf"^\| {n} \|.*$", SEEN, re.M)
    assert m, n
    return m.group(0)


def in_note(s, where):
    assert s in where, (s, where[:200])


def num(s):
    """a printed number of the note (a Unicode minus allowed) -> Decimal."""
    return Decimal(s.replace("\u2212", "-"))


def half_up(x, d):
    """x rounded half up to d decimals (the note's rule); x is first rounded to 12 decimals, so a value at an exact tie
    that the binary float holds just below it (39/40 = 0.97499...) counts as the tie."""
    return str(Decimal(f"{x:.12f}").quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def printed(s, x):
    """the printed number s is x rounded half up at the precision s is printed with."""
    d = -num(s).as_tuple().exponent
    return num(s) == Decimal(half_up(x, d))


def grab(pattern, where):
    mm = re.search(pattern, where)
    assert mm, (pattern, where[:200])
    return mm.groups()


@pytest.fixture(scope="module")
def A():
    return P.all_items(ROOT)


# ---------------------------------------------------------------- row 1: stage 8b, 150 F cores, KV patch from round(0.3 L)
def test_row1_stale_after(A):
    fg, r1 = A["format_gap"], row(1)
    want = {"mistral7": "Mistral 0.90 [0.83, 0.96]", "llama8": "Llama 0.82 [0.74, 0.90]", "phi4": "Phi-4 0.67",
            "olmo7": "OLMo-2 0.61", "falcon7": "Falcon3 0.25", "gemma9": "Gemma-2-9B 0.19", "qwen7": "Qwen-7B 0.09 [0.03, 0.17]",
            "qwen14": "Qwen-14B 0.01"}
    for k, s in want.items():
        in_note(s, r1)
        e = fg[k]["arms"]["AFTER"]["stale"]
        got = ci2(e) if "[" in s else f2(e[0])
        assert s.endswith(got), (k, s, e)
        assert fg[k]["n"] == 150 and fg[k]["l0"] == round(0.3 * fg[k]["L"])
    # the intervals SPEC 15.2 item 1 adds for the other five models
    for k, s in {"phi4": "Phi-4 0.67 [0.56, 0.78]", "olmo7": "OLMo 0.61 [0.49, 0.73]", "falcon7": "Falcon 0.25 [0.15, 0.36]",
                 "gemma9": "Gemma-2-9B 0.19 [0.11, 0.29]", "qwen14": "Qwen-14B 0.01 [0.00, 0.04]"}.items():
        in_note(s, " ".join(S152.split()))
        assert s.endswith(ci2(fg[k]["arms"]["AFTER"]["stale"])), (k, fg[k]["arms"]["AFTER"]["stale"])


def test_row1_free_form_and_full_patch(A):
    fg, r1 = A["format_gap"], row(1)
    in_note("NONE, BEFORE, POST-NULL, PRE flip 0.98-1.00", r1)
    free = [fg[k]["arms"][a]["flip"][0] for k in fg for a in P.FREE_ARMS]
    assert (f2(min(free)), f2(max(free))) == ("0.98", "1.00")
    in_note("POST 0.87 (Mistral), 0.99-1.00 otherwise", r1)
    assert f2(fg["mistral7"]["arms"]["POST"]["flip"][0]) == "0.87"
    other = [fg[k]["arms"]["POST"]["flip"][0] for k in fg if k != "mistral7"]
    assert (f2(min(other)), f2(max(other))) == ("0.99", "1.00")
    in_note("full patch 0.98-1.00", r1)
    full = [fg[k]["arms"][a]["flip0"][0] for k in fg for a in P.ARMS_B]
    assert (f2(min(full)), f2(max(full))) == ("0.98", "1.00")


def test_row1_silent_failure(A):
    fg, r1 = A["format_gap"], row(1)
    in_note("Of the non-flips, B: 0.985 / 0.992 / 0.61 (Mistral / Llama / Qwen-7B)", r1)
    a = {k: fg[k]["after"] for k in fg}
    assert (f3(a["mistral7"]["nonflip_b"]), f3(a["llama8"]["nonflip_b"]), f2(a["qwen7"]["nonflip_b"])) == ("0.985", "0.992", "0.61")
    in_note("Median absolute p(B) of stale answers: Mistral 0.999, Llama 0.846", r1)
    assert (f3(a["mistral7"]["med_abs_pb"]), f3(a["llama8"]["med_abs_pb"])) == ("0.999", "0.846")
    # SPEC 15.2 item 1: every model's absolute and six-way medians, and the AFTER value-only generated flip
    for s, key, k in (("Mistral 0.999", "med_abs_pb", "mistral7"), ("Phi-4 0.997", "med_abs_pb", "phi4"), ("Qwen-7B 0.997", "med_abs_pb", "qwen7"),
                      ("Gemma 0.994", "med_abs_pb", "gemma9"), ("Falcon 0.981", "med_abs_pb", "falcon7"), ("OLMo 0.914", "med_abs_pb", "olmo7"),
                      ("Qwen-14B 0.908", "med_abs_pb", "qwen14"), ("Llama 0.846", "med_abs_pb", "llama8"), ("Mistral 1.000", "med_six_pb", "mistral7"),
                      ("Phi-4 0.998", "med_six_pb", "phi4"), ("Gemma 0.996", "med_six_pb", "gemma9"), ("OLMo 0.918", "med_six_pb", "olmo7"),
                      ("Llama 0.855", "med_six_pb", "llama8")):
        in_note(s, " ".join(S152.split()))
        assert s.endswith(f3(a[k][key])), (s, k, a[k][key])
    in_note("AFTER generated V_S@0 flip (0.00-0.15)", " ".join(S152.split()))
    g = [a[k]["gen_v0"][0] for k in a]
    assert (f2(min(g)), f2(max(g))) == ("0.00", "0.15")


# ---------------------------------------------------------------- row 2: stage 8c, natural patch from blocks 4, 8, 12, 16
def test_row2_onset_curves(A):
    oc, r2 = A["onset_curves"], row(2)
    want = {("P1", "qwen7"): "Qwen 0.98 / 0.94 / 0.69 / 0.00", ("P1", "mistral7"): "Mistral 1.00 / 0.36 / 0.01 / 0.00",
            ("P1", "llama8"): "Llama 1.00 / 0.68 / 0.01 / 0.00", ("LETTER", "qwen7"): "Qwen 0.86 / 0.73 / 0.30 / 0.00",
            ("LETTER", "mistral7"): "Mistral 0.97 / 0.29 / 0.00 / 0.00", ("LETTER", "llama8"): "Llama 1.00 / 0.22 / 0.00 / 0.00"}
    for (f, k), s in want.items():
        in_note(s, r2)
        assert oc[k]["depths"] == [3, 7, 11, 15] and oc[k]["onsets"] == [4, 8, 12, 16] and oc[k]["n"] == 80
        assert s.endswith(" / ".join(f2(x[0]) for x in oc[k]["flips"]["KV"][f])), (f, k)
    in_note("NONE 1.00 at all four", r2)
    assert all(f2(x[0]) == "1.00" for k in oc for x in oc[k]["flips"]["KV"]["NONE"])
    in_note("Value-only: NONE 0.84-0.99, P1 at most 0.01", r2)
    vn = [x[0] for k in oc for x in oc[k]["flips"]["V"]["NONE"]]
    assert (f2(min(vn)), f2(max(vn))) == ("0.84", "0.99")
    assert f2(max(x[0] for k in oc for x in oc[k]["flips"]["V"]["P1"])) == "0.01"


def test_row2_brackets(A):
    oc, r2 = A["onset_curves"], row(2)
    in_note("Brackets for l_c(P1): Mistral (4, 8], Llama (8, 12], Qwen (12, 16] blocks; l_c(NONE) > 16 in all three", r2)
    for k, (lo, hi) in {"mistral7": (4, 8), "llama8": (8, 12), "qwen7": (12, 16)}.items():
        b = oc[k]["lc"]["P1"]
        assert (b["lo"], b["hi"], b["censored"]) == (lo, hi, False)
        assert oc[k]["lc"]["NONE"]["censored"] and oc[k]["lc"]["NONE"]["lo"] == 16


# ---------------------------------------------------------------- row 3: stage 8a explore, 185 E passages
def test_row3_passages(A):
    po, r3 = A["passage_onsets"], row(3)
    for k, s in {"qwen7": "0.65 / 0.19 (Qwen)", "mistral7": "0.41 / 0.01 (Mistral)", "llama8": "0.25 / 0.04 (Llama)"}.items():
        in_note(s, r3)
        sh = po[k]["share"]["OPTA"]
        assert s.startswith(f"{f2(sh['K_S'][0])} / {f2(sh['K_S@on'][0])}"), (k, sh)
    in_note("NOM value: 0.93 / 0.92, 0.88 / 0.87, 0.93 / 0.94", r3)
    assert ", ".join(f"{f2(po[k]['share']['NOM']['V_S'][0])} / {f2(po[k]['share']['NOM']['V_S@on'][0])}" for k in ("qwen7", "mistral7", "llama8")) == \
        "0.93 / 0.92, 0.88 / 0.87, 0.93 / 0.94"
    s152 = " ".join(S152.split())
    in_note("NOM V 0.93/0.88/0.93/0.95 from 0 and 0.92/0.87/0.94/0.94 from 0.3 L; OPTA K 0.65/0.41/0.25/0.60 from 0 and 0.19/0.01/0.04/0.03 from 0.3 L", s152)
    order = ("qwen7", "mistral7", "llama8", "gemma9")
    for f, r_, s in (("NOM", "V_S", "0.93/0.88/0.93/0.95"), ("NOM", "V_S@on", "0.92/0.87/0.94/0.94"), ("OPTA", "K_S", "0.65/0.41/0.25/0.60"),
                     ("OPTA", "K_S@on", "0.19/0.01/0.04/0.03")):
        assert "/".join(f2(po[k]["share"][f][r_][0]) for k in order) == s
    in_note("NOM V 0.987-1.000, OPTA K 0.516/0.014/0.273/0.116", s152)
    nom = [po[k]["ratio"]["NOM"][0] for k in order]
    assert (f3(min(nom)), f3(max(nom))) == ("0.987", "1.000")
    assert "/".join(f3(po[k]["ratio"]["OPTA"][0]) for k in order) == "0.516/0.014/0.273/0.116"
    assert all(po[k]["n"] == {"NOM": 185, "OPTA": 185} and po[k]["onset"] == round(0.3 * po[k]["L"]) for k in order)


def test_row3_ratio_equals_printed_fraction(A):
    """The recomputed margin-shift fractions equal the part-A scorer's printed 'fractions K_S@on / V_S@on'."""
    S8A = (ROOT / "results/gpu_stage8a/STAGE8A_SCORE.txt").read_text().splitlines()
    for k, r in A["passage_onsets"].items():
        i0 = next(i for i, x in enumerate(S8A) if x.startswith(f"  [{k}] explore parts"))
        for f, ch in (("NOM", "V"), ("OPTA", "K")):
            line = next(x for x in S8A[i0:i0 + 16] if f"E1/E2 {f} (" in x)
            assert f"fractions {ch}_S@on {r['ratio'][f][0]:+.3f}," in line, (k, f, line)


# ---------------------------------------------------------------- row 4: stage 8c explore, block 7, 40 stories, generation
def test_row4_edit_transfer(A):
    et, r4 = A["edit_transfer"], row(4)
    m = et["models"]
    in_note("E1 flips NONE 1.00 vs P1 0.35 (Mistral), 0.95 (Qwen), 0.65 (Llama)", r4)
    assert all(f2(m[k]["flip"]["E1"]["NONE"][0]) == "1.00" for k in m)
    assert tuple(f2(m[k]["flip"]["E1"]["P1"][0]) for k in ("mistral7", "qwen7", "llama8")) == ("0.35", "0.95", "0.65")
    # the Mistral pair are exact ties at 2 decimals (39/40 and 1/40): half up, 0.98 / 0.03
    g = grab(r"E4b (\d\.\d+) vs (\d\.\d+) \(Mistral\), (\d\.\d+) vs (\d\.\d+) \(Qwen\)", r4)
    e4b = [m[k]["flip"]["E4b"][f][0] for k in ("mistral7", "qwen7") for f in ("NONE", "P1")]
    assert all(printed(s, x) for s, x in zip(g, e4b)), (g, e4b)
    assert e4b == pytest.approx([39 / 40, 1 / 40, 79 / 80, 1 / 4], abs=1e-12)
    assert g == ("0.98", "0.03", "0.99", "0.25")
    assert all(m[k]["n"] == 40 and m[k]["layer"] == 7 for k in m)
    in_note("At Mistral 99-100 % of the P1 failures are B", r4)
    fb = [m["mistral7"]["fail_b"][z][0] for z in ("nat", "E1", "E4b")]
    assert (f"{100 * min(fb):.0f}", f"{100 * max(fb):.0f}") == ("99", "100")
    in_note("Spearman 0.92 (9 edit x model pairs)", r4)
    assert len(et["pairs"]) == 9 and f2(et["rho"]) == "0.92"
    assert {(p["k"], p["z"]) for p in et["pairs"]} == {("qwen7", z) for z in ("E1", "E2", "E3", "E4a", "E4b")} | {("mistral7", z) for z in ("E1", "E2", "E4a", "E4b")}
    in_note("Qwen E3 1.00 -> 0.50", " ".join(S152.split()))
    assert (f2(m["qwen7"]["flip"]["E3"]["NONE"][0]), f2(m["qwen7"]["flip"]["E3"]["P1"][0])) == ("1.00", "0.50")


def test_row4_kappa_from_the_printed_line():
    S8C = (ROOT / "results/gpu_stage8c/STAGE8C_SCORE.txt").read_text().splitlines()
    i, sig, kap = P.kappa_line(S8C, "qwen7")
    assert S8C[i].strip().startswith("P1@7: sigma") and (sig, kap["E1"], kap["E4b"]) == ("+0.657", "+0.64", "+0.34")
    i, sig, kap = P.kappa_line(S8C, "mistral7")
    assert (sig, kap["E1"], kap["E4a"]) == ("+0.683", "+0.67", "+0.18") and "E3" not in kap


# ---------------------------------------------------------------- row 5: stage 8c dose rows, block 3, and the lambda law
def test_row5_dose(A):
    d, r5 = A["dose"], row(5)
    order = ("qwen7", "mistral7", "llama8")
    in_note("Half key, full value: P1 flips 0.31 / 0.32 / 0.66 (Qwen / Mistral / Llama)", r5)
    assert " / ".join(f2(d[k]["flip"]["SK50", "P1", 3][0]) for k in order) == "0.31 / 0.32 / 0.66"
    in_note("Half value, full key: NONE flips 0.93 / 0.92 / 0.94", r5)
    assert " / ".join(f2(d[k]["flip"]["SV50", "NONE", 3][0]) for k in order) == "0.93 / 0.92 / 0.94"
    in_note("ψ_K(0.5) in P1 0.10 / 0.18 / 0.13 against ψ_V(0.5) in NONE 0.60 / 0.58 / 0.56", r5)
    assert " / ".join(f2(d[k]["psi"]["K", "P1", 3, 50][0]) for k in order) == "0.10 / 0.18 / 0.13"
    assert " / ".join(f2(d[k]["psi"]["V", "NONE", 3, 50][0]) for k in order) == "0.60 / 0.58 / 0.56"
    assert all(d[k]["used"]["P1", 3, "K"] and d[k]["used"]["NONE", 3, "V"] for k in order)
    in_note("Per-core ψ_K(0.5) at Qwen: median 0.06 (10th-90th percentile 0.02-0.20)", r5)
    st = d["qwen7"]["story"]
    assert (f2(st["p50"]), f2(st["p10"]), f2(st["p90"])) == ("0.06", "0.02", "0.20")
    in_note("(SK80 median 0.48)", " ".join(S152.split()))
    assert f2(st["med80"]) == "0.48"


def test_row5_lambda_law(A):
    L, r5 = A["law"], row(5)
    in_note("λ ranks: calibrated curve Spearman 0.862, log proj_k alone 0.855 (difference ", r5)
    assert (f3(L["rho_curve"]), f3(L["rho_logpk"])) == ("0.862", "0.855")
    # the unrounded difference is 0.0076 (printed 0.008, = \NpLawRhoDiff), not the difference of the two printed values
    # (0.862 - 0.855 = 0.007)
    assert f3(L["diff"]) == "0.008" and f3(round(L["rho_curve"], 3) - round(L["rho_logpk"], 3)) == "0.007"
    d, lo, hi = grab(r"\(difference\D*?(\d\.\d+)[^,\[]*, 95 % \[([−-]?\d\.\d+), ([−-]?\d\.\d+)\]\)", r5)
    assert printed(d, L["diff"]) and d == "0.008", d
    assert printed(lo, L["diff_lo"]) and printed(hi, L["diff_hi"]), (lo, hi)
    assert (f3(L["diff_lo"]), f3(L["diff_hi"])) == ("-0.060", "0.090")
    s152 = " ".join(S152.split())
    in_note("138 λ statistics in 49 units", s152)
    assert (L["n_stats"], L["n_units"]) == (138, 49)
    in_note("MAE 0.369 (curve) against 0.868 (O1)", s152)
    assert (f3(L["mae_curve"]), f3(L["mae_o1"])) == ("0.369", "0.868")
    in_note("with a resample mean of 0.012", s152)
    assert f3(L["diff_mean"]) == "0.012"
    in_note("The partial rank correlation of the curve with λ given log proj_k is 0.53", s152)
    in_note("Spearman(p_k, p_v) is 0.92", s152)
    assert (f2(L["partial"]), f2(L["rho_pk_pv"])) == ("0.53", "0.92")


# ---------------------------------------------------------------- row 6: stage 5 knockout
def test_row6_knockout(A):
    ko, r6 = A["knockout"], row(6)
    in_note("Cutting the answer's attention to the writing token keeps 1.00 of listed answers", r6)
    assert all(f2(ko[k]["keep_b"][f, "M3"][0]) == "1.00" for k in ko for f in ("AFTER", "P1"))
    in_note("Under NONE it sends 0.79-1.00 of answers to the initial or distractor location", r6)
    v = {k: f2(ko[k]["init"][0]) for k in ko}
    assert v == {"qwen7": "1.00", "mistral7": "0.85", "qwen14": "0.79"}
    in_note("(Qwen-7B 1.00, Mistral 0.85, Qwen-14B 0.79)", " ".join(S152.split()))
    in_note("(1.00 under M3; 0.97-1.00 under M7)", " ".join(S152.split()))
    m7 = [ko[k]["keep_b"][f, "M7"][0] for k in ko for f in ("AFTER", "P1")]
    assert (f2(min(m7)), f2(max(m7))) == ("0.97", "1.00")
    assert all(ko[k]["n"][f, m] == 150 for k in ko for (f, m) in ko[k]["n"])


# ---------------------------------------------------------------- item 5 (SPEC 15.2): the binding swap by format
def test_item5_binding_swap(A):
    b = A["binding_swap"]
    in_note("(0.993 at block 28 under NO-MENTION; at most 0.01 at every block 18-40 under QNAMES, OPTIONS-AFTER and LETTERS)", " ".join(S152.split()))
    assert b["blocks"] == (18, 40) and max(float(v[1]) for v in b["named"].values()) == 0.01
    # the anchor is the printed IIA at the preregistered l* = 28 (0.99; 0.993 at 3 decimals), not the best block (1.00 at 29)
    assert b["none_lstar"][1:3] == ("0.99", 28) and b["none_lstar"][0] == b["none"][0] and b["none"][1:] == ("1.00", 29)
    assert b["id_none_one_to"][1] == 35 and b["id_opt_zero_from"][1] == 20
    S8C = (ROOT / "results/gpu_stage8c/STAGE8C_SCORE.txt").read_text()
    assert "BIND IIA(NO-MENTION) 0.993" in S8C   # the 0.993 is the stage-8c J-C-G7 anchor's printed value (\JcGsevenIIA)


# ---------------------------------------------------------------- item 8: the one-token copy (design review, section 6)
def test_item8_one_token_copy(A):
    c = A["one_token_copy"]
    v = {k: f2(c[k]["V_S"]["est"][0]) for k in c}
    j = {k: f2(c[k]["KV_S"]["est"][0]) for k in c}
    # the review's sentence: the value-only clamp completes the multi-token answer in 4-11 % of cases, the joint clamp in
    # 92-98 % (Qwen2.5-7B, Mistral-7B, Llama-3.1-8B); Gemma-2-9B completes 61 % under the value-only clamp
    assert (v["mistral7"], v["llama8"], v["qwen7"]) == ("0.04", "0.08", "0.11") and v["gemma9"] == "0.61"
    assert (j["qwen7"], j["llama8"], j["mistral7"]) == ("0.92", "0.93", "0.98") and j["gemma9"] == "0.96"
    assert {k: c[k]["V_S"]["n"] for k in c} == {"qwen7": 35, "mistral7": 23, "llama8": 24, "gemma9": 62}


# ---------------------------------------------------------------- section 2: in-hand analogues of the predictions
def test_section2_analogues(A):
    oc, fg = A["onset_curves"], A["format_gap"]
    pr = " ".join(PRED.split())
    # the brackets are exact fractions of L (4/32 = 0.125 and 12/32 = 0.375 are ties at 2 decimals): half up, 0.13 and 0.38
    g = grab(r"P1 brackets \(2\): Mistral \((\d\.\d+), (\d\.\d+)\] L, Llama \((\d\.\d+), (\d\.\d+)\] L, Qwen \((\d\.\d+), (\d\.\d+)\] L", pr)
    fr = [oc[k]["lc"]["P1"][e] / oc[k]["L"] for k in ("mistral7", "llama8", "qwen7") for e in ("lo", "hi")]
    assert all(printed(s, x) for s, x in zip(g, fr)), (g, fr)
    assert fr == [4 / 32, 8 / 32, 8 / 32, 12 / 32, 12 / 28, 16 / 28]
    in_note("l_c(NONE) > 0.5 L (Qwen > 0.57 L)", pr)
    assert {k: f2(oc[k]["lc"]["NONE"]["lo"] / oc[k]["L"]) for k in oc} == {"qwen7": "0.57", "mistral7": "0.50", "llama8": "0.50"}
    in_note("largest gap 0.11 (Qwen, P1, block 11; E-argmax)", pr)
    g = max(oc, key=lambda k: oc[k]["eone_gap"]["max"])
    assert (g, f2(oc[g]["eone_gap"]["max"]), oc[g]["eone_gap"]["cell"]) == ("qwen7", "0.11", ("P1", 11))
    in_note("block 7: 0.25 / 0.03 (generation)", pr)
    m = A["edit_transfer"]["models"]
    assert tuple(f2(m[k]["flip"]["E4b"]["P1"][0] / m[k]["flip"]["E4b"]["NONE"][0]) for k in ("qwen7", "mistral7")) == ("0.25", "0.03")
    in_note("Mistral 0.87 at 0.31 L; 0.88-0.89 at blocks 8-16 (2)", pr)
    assert f2(fg["mistral7"]["arms"]["POST"]["flip"][0]) == "0.87" and f2(fg["mistral7"]["l0"] / fg["mistral7"]["L"]) == "0.31"
    post = [x[0] for x in oc["mistral7"]["flips"]["KV"]["POST"][1:]]
    assert (f2(min(post)), f2(max(post))) == ("0.88", "0.89")
    in_note("0.985 / 0.992 / 0.61 and 0.999, at 0.3 L (1)", pr)


def test_section2_tau_pinned(A):
    """P5b's tau(3) of E4b at Mistral is 0.563 (P1 0.556, NONE 0.988; printed 0.56 = \\NpTauEfourbThreeMistralSeven), not
    the ratio of the rounded flips (0.56 / 0.99 = 0.57); the recomputed values are pinned (Qwen's 0.59 reproduces). The
    other number that differs from a computation on printed values, section 1 row 5's Spearman difference, is pinned in
    test_row5_lambda_law."""
    pr = " ".join(PRED.split())
    tq, tm = grab(r"P1: τ\(3\) (\d\.\d+) / (\d\.\d+) \(E-argmax\)", pr)
    oc = A["onset_curves"]
    q, m = oc["qwen7"]["tau_e4b_3"], oc["mistral7"]["tau_e4b_3"]
    assert (q["P1"], q["NONE"], m["P1"], m["NONE"]) == pytest.approx((19 / 32, 1.0, 89 / 160, 79 / 80), abs=1e-12)
    assert (f2(q["tau"]), f2(m["tau"])) == ("0.59", "0.56") and f2(round(m["P1"], 2) / round(m["NONE"], 2)) == "0.57"
    assert printed(tq, q["tau"]) and printed(tm, m["tau"]) and (tq, tm) == ("0.59", "0.56"), (tq, tm)
    assert "tau_e4b_3" not in oc["llama8"]   # no stored DAS basis at Llama


# ---------------------------------------------------------------- helpers on hand cases
def test_bracket():
    assert P.bracket([4, 8, 12, 16], [1.0, 0.36, 0.01, 0.0], 32) == {"lo": 4, "hi": 8, "censored": False}
    assert P.bracket([4, 8, 12, 16], [0.4, 0.3, 0.0, 0.0], 32) == {"lo": 0, "hi": 4, "censored": False}
    assert P.bracket([4, 8, 12, 16], [1.0, 1.0, 0.99, 0.5], 28) == {"lo": 16, "hi": 28, "censored": True}   # 0.5 is not below


def test_curve_and_floor():
    assert P.curve(-0.3, 0.1, 0.5) == 0.0 and P.curve(0.0, 0.1, 0.5) == 0.0
    assert P.curve(0.5, 0.1, 0.5) == pytest.approx(0.1) and P.curve(0.8, 0.1, 0.5) == pytest.approx(0.5)
    assert P.curve(0.25, 0.1, 0.5) == pytest.approx(0.05) and P.curve(0.9, 0.1, 0.5) == pytest.approx(0.75)
    assert P.curve(1.0, 0.1, 0.5) == pytest.approx(1.0) and P.curve(1.2, 0.1, 0.5) == pytest.approx(1 + 0.2 * 0.5 / 0.2)
    assert P.lg(0.001) == pytest.approx(np.log(0.01)) and P.lg(-2) == pytest.approx(np.log(0.01)) and P.lg(0.5) == pytest.approx(np.log(0.5))


def test_parsers_on_hand_lines():
    lines = ["-- qwen7: kappa against sigma per cell (descriptive)", "   LETTER@7: sigma +0.987; kappa E1 +0.98",
             "   P1@7: sigma +0.657; kappa E1 +0.64, E4b +0.34", "-- mistral7: kappa against sigma per cell (descriptive)",
             "   P1@3: sigma +0.881; kappa E1 +0.88", "-- next"]
    assert P.kappa_line(lines, "qwen7") == (2, "+0.657", {"E1": "+0.64", "E4b": "+0.34"})
    with pytest.raises(AssertionError):
        P.kappa_line(lines, "mistral7")     # no P1@7 line before the next header
    sw = P.sweeps(["   sweep BIND QNAMES        IIA/Phi: 18:0.00/-0.3 19:0.01/+0.9", "   sweep ID   NO-MENTION IIA/Phi: 0:1.00/+46.0"])
    assert sw["BIND", "QNAMES"] == (0, {18: ("0.00", "-0.3"), 19: ("0.01", "+0.9")}) and sw["ID", "NO-MENTION"][1][0] == ("1.00", "+46.0")
    assert P._argdec({"B": -1.0, "S": -0.5, "X": -3.0, "Z": -2.0, "D": -0.5}) == "S"   # a tie goes to the first in B, S, X, Z, D
    with pytest.raises(AssertionError):
        P._argdec({"B": 0.0, "S": -1.0})
    s6 = ["   l* = 28, l*_ID = 0", "   l* = 28 (lstar.json 28), l*_ID = 0 (lstar.json 0): re-derived from the sweep files, equal"]
    assert P.lstar_line(s6) == (1, 28)
    with pytest.raises(AssertionError):
        P.lstar_line(["   l* = 28 (lstar.json 27), l*_ID = 0 (lstar.json 0)"])   # the re-derived and stored l* differ
    with pytest.raises(AssertionError):
        P.lstar_line(s6 + s6[1:])                                               # not a unique line


def test_printed_helper():
    """the note-matching helper: a printed number is the value rounded half up at its own precision (exact ties up, also
    when the binary float sits just below the tie)."""
    assert printed("0.98", 39 / 40) and not printed("0.97", 39 / 40) and printed("0.975", 39 / 40) and printed("0.03", 1 / 40)
    assert printed("0.13", 4 / 32) and not printed("0.12", 4 / 32) and printed("0.125", 4 / 32) and printed("0.38", 12 / 32)
    assert not printed("0.57", 0.5633) and printed("0.56", 0.5633) and not printed("0.007", 0.0076) and printed("0.008", 0.0076)
    assert printed("−0.060", -0.0601) and not printed("0.060", -0.0601) and not printed("0.974", 39 / 40)
    assert printed("0.11", 0.11249999999999993) and printed("0.113", 0.11249999999999993) and half_up(0.975, 2) == "0.98"

