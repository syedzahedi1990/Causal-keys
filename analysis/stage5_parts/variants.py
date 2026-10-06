"""Part (d) of P-2026-10-05-G (docs/PREREGISTRATION.md): Gates d1-d3 and predictions G13-G17 (with the G16 decision table)
on the stage-5 factorial ({root}/factorial/{model}_s0.json, --arm-modules ckeys.variants), the competence prompts
({root}/competence/{model}.json, experiments/form_competence.py) and the attention probe ({root}/form_attention/
{model}.json, experiments/form_attention.py), exactly as drafted. Called by analysis/stage5_score.py (``score(root)``) or
standalone. Gate 0 is the shared gate of analysis/stage5_parts/subsets.py.

Per core and arm v (lp of the English tokens S, X, B and of the arm's form tokens Sf, Xf, Bf against the self-clamp row):
  ID_K, ID_V as in the paper (English scoring); ID_K^form with the form token; ID_K^any, ID_V^any with log[p(y) + p(f)]
  (= the English measure when f = y, so POST^any = POST).
  r_K(v) = [ID_K(v) - ID_K(FLOOR_v)] / [ID_K(ANCHOR_v) - ID_K(FLOOR_v)] (ratio of means, paired cores, floor POST_OTHER /
  POST_FR_OTHER / POST_DE_OTHER / AFTER_OTHER, anchor POST or AFTER; NONE-floored r reported beside it);
  rho_s(v) = s_ID(v)/s_ID(ANCHOR_v) (POST; AFTER for the list family), the shares computed inside every resample;
  r_K^any(v) = [ID_K^any(v) - ID_K(FLOOR_v)] / [ID_K(ANCHOR_v) - ID_K(FLOOR_v)] and rho_s^any likewise: only v's
  numerator is any-form, floor and anchor are scored on the English tokens (the floor arms carry other words' forms),
  on the same cores; D(v) = ID_K(ANCHOR_v) - ID_K(v) paired (reported);
  A_v = layer mean of 1/2 [att_S(K_S) - att_S(K_X) + att_X(K_X) - att_X(K_S)] (span-summed, head-averaged attention of
  the form's tokens to p), a_v = A_v / A_POST (ratio of means over the probe cores).
Populations: primary = all cores; SYN/FRMIX/DEMIX/FR/DE and AFTER_* cells = the per-word competence-gated subset
  (cores whose S and X words both map correctly; TITLE/UPPER/PLURAL cells stay on all cores, the entry names the
  per-word subset for the SYN/FRMIX/DEMIX/FR/DE cells only), applied to every statistic of the cell, a_v on the probe cores
  included; any-form, form-scored and attention measures exclude cores with a first-token collision among f_B, f_S, f_X
  (DE/DEMIX in Qwen/OLMo and Mistral, UPPER in Mistral). Evaluability follows
  the entry only (Gates 0, d1-d3 and the exclusions; no minimum n beyond a non-empty cell); n is printed beside every
  cell. 10,000 resamples with one fixed index set per n.
Gates: Gate 0; d1 per model x arm: clean B and S accuracy >= 0.95 (a cell needs v, its floor and its anchor); d2 per
  model x family: >= 5/6 words map correctly (every family of experiments/form_competence.py, TITLE/UPPER/PLURAL
  included; the per-word subset only for SYN/FR/DE); d3 per model: A_POST > 0 with CI excluding 0 (else a_v not evaluable).
Verdict rule: a prediction is MET if >= 3 evaluable models (of Qwen2.5-7B, 14B, Mistral-7B, OLMo-2-7B) meet it, NOT MET
  if >= 3 are evaluable and fewer than 3 meet it, otherwise NOT EVALUABLE.
G13 r_K(POST_THE) >= 0.75 with lower bound > 0.5; r_K(POST_MODIF) likewise (separately). THE failing flags G14-G17 as
    frame-confounded.
G14 for each of TITLE, UPPER, PLURAL, SYN, FRMIX, DEMIX: r_K(v) <= 0.75 with upper bound < 1.0.
G15 for each of SYN, FRMIX, DEMIX: r_K(v) <= 1/3 (upper < 0.5) and rho_s(v) <= 0.5 (upper < 0.75).
G16 for each of SYN, FRMIX, DEMIX on the collision-excluded population: token pattern = r_K^any <= 1/3 (upper < 0.5) and
    a_v <= 1/3 (upper < 0.5); concept pattern = r_K^any >= 2/3 (lower > 0.5), rho_s^any >= 0.5 (lower > 0.25) and a_v >= 2/3
    (lower > 0.5); else graded. Decision table: G15 met + token on both (within the same model in >= 3 models, the G16 line)
    = token-level lookup; concept on a_v with (concept
    on r^any or G15 not met) = concept-level lookup; concept on a_v with G15 met and token on r^any = lookup without readout.
G17 for FRMIX and DEMIX, cells where G15 is met in Qwen2.5-7B, Mistral-7B, OLMo-2-7B: paired ID_V(v) - ID_V(POST) > 0 with
    CI excluding 0; met per v if >= 2 evaluable cells and >= 2 of them meet; not evaluable with < 2 evaluable cells.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage5_parts.subsets import FACTORIAL, IDX, NAN, boot, fmt, gate0, ratio, verdict  # noqa: E402

from ckeys.story import LOCATIONS, pick_x  # noqa: E402
from ckeys.variants import FAMILY, FLOOR, collision_pairs, excluded  # noqa: E402

COMPETENCE, PROBE = "competence", "form_attention"
GATED = {"SYN", "FR", "DE"}  # families whose cells take the per-word competence-gated subset (the entry's SYN/FRMIX/DEMIX/FR/DE)
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
GAP_MODELS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]  # stage-3b ID_V gap > 2 nats
SENT = ["POST_THE", "POST_MODIF", "POST_TITLE", "POST_UPPER", "POST_PLURAL", "POST_SYN", "POST_FRMIX", "POST_DEMIX", "POST_FR", "POST_DE"]
LISTS = ["AFTER_SYN", "AFTER_FR", "AFTER_DE"]
FLOORS = ["POST_OTHER", "POST_FR_OTHER", "POST_DE_OTHER", "AFTER_OTHER"]
G14 = ["POST_TITLE", "POST_UPPER", "POST_PLURAL", "POST_SYN", "POST_FRMIX", "POST_DEMIX"]
G15 = ["POST_SYN", "POST_FRMIX", "POST_DEMIX"]
ALL = ["NONE", "POST"] + SENT + FLOORS[:3] + ["AFTER"] + LISTS + FLOORS[3:]


def short(v):
    return v.replace("POST_", "").replace("AFTER_", "AFTER ")


def per_core(results, arm):
    """Per core: ID_K, ID_V, the form-scored and any-form versions, clean accuracies and masses."""
    out = {}
    for r in results:
        if r["arm"] != arm or r["view"] != "direct":
            continue
        m, idr = r["m"], r["m"]["ID@0"]["lp"]
        has = "Sf" in idr

        def dl(k, t, mode):
            a, b = m[k]["lp"], idr
            if mode == "plain" or not has:
                return a[t] - b[t]
            if mode == "form":
                return a[t + "f"] - b[t + "f"]
            return np.logaddexp(a[t], a[t + "f"]) - np.logaddexp(b[t], b[t + "f"])

        def ident(ch, mode):
            s, x = (f"{ch}_S@0", f"{ch}_X@0")
            return 0.5 * ((dl(s, "S", mode) - dl(x, "S", mode)) + (dl(x, "X", mode) - dl(s, "X", mode)))

        c = r["clean"]
        out[json.dumps(r["core"], sort_keys=True)] = {
            "idK": ident("K", "plain"), "idV": ident("V", "plain"), "idKf": ident("K", "form"), "idKa": ident("K", "any"),
            "idVa": ident("V", "any"), "accB": c["B"]["argmax_cand"] == r["core"]["base"], "accS": c["S"]["argmax_cand"] == r["core"]["source"],
            "mass": c["B"]["mass"], "sum": ident("K", "plain") + ident("V", "plain"), "core": r["core"], "X": r["X"], "has_form": has,
            "dK": m["K_S@0"]["m"] - m["ID@0"]["m"], "dV": m["V_S@0"]["m"] - m["ID@0"]["m"]}
    return out


def vec(A, ids, key="idK"):
    return np.array([A[i][key] for i in ids], float)


def boot_fn(fn, *arrays):
    """``fn`` of the means, recomputed in each resample over the same cores."""
    arrays = [np.asarray(a, float) for a in arrays]
    n = len(arrays[0])
    if n == 0:
        return NAN
    idx = IDX(n)
    r = fn(*[a[idx].mean(1) for a in arrays])
    return fn(*[a.mean() for a in arrays]), np.percentile(r, 2.5), np.percentile(r, 97.5)


def r_K(arms, v, ids, floor, anchor, key="idK"):
    return ratio(vec(arms[v], ids, key) - vec(arms[floor], ids), vec(arms[anchor], ids) - vec(arms[floor], ids))


def rho(arms, v, ids, anchor="POST", key="idK", keyv="idV"):
    return boot_fn(lambda k, vv, kp, vp: (k / (k + vv)) / (kp / (kp + vp)), vec(arms[v], ids, key), vec(arms[v], ids, keyv),
                   vec(arms[anchor], ids), vec(arms[anchor], ids, "idV"))


def load_probe(root, model):
    f = Path(root) / PROBE / f"{model}.json"
    if not f.exists():
        return None
    items = json.load(open(f))["items"]
    A = {}
    for it in items:
        a = it["att"]
        S, X = (np.array(a["K_S"]["span"]), np.array(a["K_X"]["span"]))
        s, x = it["s"], it["x"]
        A.setdefault(it["arm"], {})[json.dumps(it["core"], sort_keys=True)] = {
            "A": 0.5 * ((S[s] - X[s]) + (X[x] - S[x])).mean(), "layers": 0.5 * ((S[s] - X[s]) + (X[x] - S[x])),
            "last": 0.5 * ((np.array(a["K_S"]["last"])[s] - np.array(a["K_X"]["last"])[s]) + (np.array(a["K_X"]["last"])[x] - np.array(a["K_S"]["last"])[x])).mean(),
            "B": (S[it["b"]] - X[it["b"]]).mean()}
    return A


def load_competence(root, model):
    f = Path(root) / COMPETENCE / f"{model}.json"
    return json.load(open(f))["families"] if f.exists() else None


class Model:
    def __init__(self, root, m, out, test):
        self.m, self.out = m, out
        f = Path(root) / FACTORIAL / f"{m}_s0.json"
        self.ok = f.exists()
        if not self.ok:
            return
        d = json.load(open(f))
        self.prov, res = d.get("provenance", {}), d["results"]
        self.arms = {a: per_core(res, a) for a in ALL}
        self.arms = {a: v for a, v in self.arms.items() if v}
        self.comp, self.probe = load_competence(root, m), load_probe(root, m)
        self.g0 = gate0(self.prov, self.arms, m, out, test)
        self.d1 = {a: (np.mean([v["accB"] for v in A.values()]), np.mean([v["accS"] for v in A.values()])) for a, A in self.arms.items()}
        for a, (b, s) in self.d1.items():
            out(f"  gate d1 [{a:14s}] clean accuracy B {b:.3f} S {s:.3f} -> {'pass' if min(b, s) >= 0.95 else 'FAIL'}")
        self.d2 = {}
        if self.comp is None:
            out("  gate d2: competence file MISSING (gated cells not evaluable)")
        for fam, words in (self.comp or {}).items():
            k = sum(w["correct"] for w in words.values())
            self.d2[fam] = {w: v["correct"] for w, v in words.items()}
            out(f"  gate d2 [{fam:7s}] {k}/6 map correctly ({', '.join(w for w, v in words.items() if not v['correct']) or 'none failing'}) -> "
                f"{'pass' if k >= 5 else 'FAIL'}")
        self.d3 = None
        if self.probe is None:
            out("  gate d3: form_attention file MISSING (a_v not evaluable)")
        elif "POST" in self.probe:
            Ap = boot([v["A"] for v in self.probe["POST"].values()])
            self.d3 = Ap[0] > 0 and Ap[1] > 0
            out(f"  gate d3 A_POST {fmt(Ap, 4)} > 0 with CI excl. 0 (n={len(self.probe['POST'])}) -> {'pass' if self.d3 else 'FAIL'}")

    def d1_ok(self, *arms):
        return all(a in self.d1 and min(self.d1[a]) >= 0.95 for a in arms)

    def population(self, v):
        """(ids of the primary population of v, reason if not evaluable; the ids are kept when a gate fails so the
        values are still reported)."""
        floor, anchor = FLOOR[v]
        need = [v, floor, anchor, "POST", "NONE"]
        if any(a not in self.arms for a in need):
            return [], f"arm missing ({[a for a in need if a not in self.arms]})"
        ids, why = sorted(set.intersection(*(set(self.arms[a]) for a in need))), []
        fam = FAMILY.get(v)
        if fam and fam not in self.d2:
            why.append(f"competence of {fam} not measured")
        elif fam and sum(self.d2[fam].values()) < 5:
            why.append(f"gate d2 failed for {fam} ({sum(self.d2[fam].values())}/6)")
        elif fam in GATED:
            ids = [i for i in ids if self.d2[fam][self.arms[v][i]["core"]["source"]] and self.d2[fam][self.arms[v][i]["X"]]]
        if not self.d1_ok(v, floor, anchor):
            why.append("gate d1 failed")
        return ids, "; ".join(why)

    def no_collision(self, v, ids):
        pairs = collision_pairs(self.m, v)
        return [i for i in ids if not excluded(self.arms[v][i]["core"], self.arms[v][i]["X"], pairs)]

    def a_v(self, v):
        """(a_v ratio, n) on the probe cores without collisions, per-word gated like the cell's other statistics (SYN/FR/DE
        families: cores whose S and X words both map correctly); None if not evaluable."""
        if self.probe is None or v not in self.probe or "POST" not in self.probe or not self.d3:
            return None, 0
        pairs, fam = collision_pairs(self.m, v), FAMILY.get(v)
        ok = (lambda c: self.d2[fam][c["source"]] and self.d2[fam][pick_x(c)]) if fam in GATED and fam in self.d2 else (lambda c: True)
        ids = [i for i in sorted(set(self.probe[v]) & set(self.probe["POST"])) if ok(json.loads(i)) and not excluded(json.loads(i), pick_x(json.loads(i)), pairs)]
        return (ratio([self.probe[v][i]["A"] for i in ids], [self.probe["POST"][i]["A"] for i in ids]) if ids else None), len(ids)

    def cell(self, v):
        """All statistics of one variant cell: None when not evaluable (with the reason)."""
        ids, why = self.population(v)
        if self.g0 is not True:
            why = "Gate 0 not passed" + (f"; {why}" if why else "")
        floor, anchor = FLOOR[v]
        if not ids and not why:
            why = "no core left in the population"
        c = {"v": v, "n": len(ids), "why": why, "evaluable": not why}
        if not ids:
            return c
        a = self.arms
        c["r"], c["r_none"] = r_K(a, v, ids, floor, anchor), r_K(a, v, ids, "NONE", anchor)
        c["rho"], c["D"] = rho(a, v, ids, anchor), boot(vec(a[anchor], ids) - vec(a[v], ids))
        c["idK"], c["idV"], c["sum"] = boot(vec(a[v], ids)), boot(vec(a[v], ids, "idV")), boot(vec(a[v], ids, "sum"))
        c["dV"] = boot(vec(a[v], ids, "idV") - vec(a["POST"], ids, "idV"))
        c["sID"] = boot_fn(lambda k, vv: k / (k + vv), vec(a[v], ids), vec(a[v], ids, "idV"))
        nc = self.no_collision(v, ids)
        c["n_any"] = len(nc)
        if nc:
            c["r_any"], c["r_form"] = r_K(a, v, nc, floor, anchor, "idKa"), r_K(a, v, nc, floor, anchor, "idKf")
            c["rho_any"] = rho(a, v, nc, anchor, "idKa", "idVa")
        c["a"], c["n_probe"] = self.a_v(v)
        return c


def pattern_r(c):
    if not c.get("evaluable") or "r_any" not in c:
        return None
    if c["r_any"][0] <= 1 / 3 and c["r_any"][2] < 0.5:
        return "token"
    if c["r_any"][0] >= 2 / 3 and c["r_any"][1] > 0.5 and c["rho_any"][0] >= 0.5 and c["rho_any"][1] > 0.25:
        return "concept"
    return "graded"


def pattern_a(c):
    if not c.get("evaluable") or c.get("a") is None:
        return None
    if c["a"][0] <= 1 / 3 and c["a"][2] < 0.5:
        return "token"
    if c["a"][0] >= 2 / 3 and c["a"][1] > 0.5:
        return "concept"
    return "graded"


def rule(cells):
    """The >= 3 evaluable models rule over {model: True / False / None}."""
    ev = [bool(v) for v in cells.values() if v is not None]
    k = sum(ev)
    return (k >= 3) if len(ev) >= 3 else None, k, len(ev)


def mshort(m):
    return m.replace("-Instruct", "").replace("-1124", "").replace("-v0.3", "")


def line(name, cells, out, note=""):
    v, k, n = rule(cells)
    out(f"  {name}: {k}/{n} evaluable models" + (f" of {len(cells)}" if n < len(cells) else "") + f" -> "
        f"{'NOT EVALUABLE (< 3 evaluable models)' if v is None else verdict(v)}" + note)
    return v


def report_cell(md, c, out):
    v = c["v"]
    if not c.get("r"):
        out(f"    {short(v):10s} NOT EVALUABLE: {c['why']}")
        return
    tag = "" if c["evaluable"] else f"  [NOT EVALUABLE: {c['why']}]"
    out(f"    {short(v):10s} n={c['n']:3d} r_K {fmt(c['r'])} (NONE-floored {fmt(c['r_none'])})  rho_s {fmt(c['rho'])}  D {fmt(c['D'])}  "
        f"ID_K {fmt(c['idK'])} ID_V {fmt(c['idV'])} K+V {c['sum'][0]:+.2f} s_ID {c['sID'][0]:.2f}{tag}")
    if "r_any" in c:
        out(f"    {'':10s} any-form (n={c['n_any']}): r_K^any {fmt(c['r_any'])}  rho_s^any {fmt(c['rho_any'])}  r_K^form {fmt(c['r_form'])}"
            + (f"  a_v {fmt(c['a'])} (n={c['n_probe']})" if c.get("a") else "  a_v not evaluable"))


def score(root, models=None, test=False, out=print):
    root = Path(root)
    if test:
        out("TEST MODE: verdict lines are not preregistered results; every model with a factorial file is scored")
        models = sorted(f.stem[:-3] for f in (root / FACTORIAL).glob("*_s0.json"))
    models = MODELS if models is None else models
    out("== Part (d): non-identical re-mentions (sentence variants of POST; floors POST_OTHER / FR_OTHER / DE_OTHER; list family secondary)")
    M, cells = {}, {}
    for m in models:
        out(f"\n## {m}")
        md = Model(root, m, out, test)
        M[m] = md
        if not md.ok:
            out("  factorial MISSING")
            continue
        cells[m] = {v: md.cell(v) for v in SENT + LISTS}
        out("  cells (primary r_K: floor OTHER, anchor POST / AFTER; gated population for SYN/FR/DE families):")
        for v in SENT + LISTS:
            report_cell(md, cells[m][v], out)
        n_all = len(set.intersection(*(set(md.arms[a]) for a in md.arms)))
        out(f"  floors and anchors (n={n_all}): " + "  ".join(f"{a} ID_K {fmt(boot([x['idK'] for x in md.arms[a].values()]))}"
                                                            for a in ("NONE", "POST", "POST_OTHER", "POST_FR_OTHER", "POST_DE_OTHER", "AFTER", "AFTER_OTHER") if a in md.arms))
    ev = [m for m in models if m in cells]

    def cmap(f):
        def g(c):
            x = f(c)
            return bool(x) if isinstance(x, (bool, np.bool_)) else x
        return {m: (g(cells[m][v]) if cells[m][v].get("evaluable") else None) for m in ev} | {m: None for m in models if m not in ev}

    out(f"\n== Verdicts (G13-G17; models {models}; rule: met if >= 3 evaluable models meet, not met if >= 3 evaluable and < 3 meet)")
    final, decisions = {}, {}
    for v in ("POST_THE", "POST_MODIF"):
        c = cmap(lambda c: c["r"][0] >= 0.75 and c["r"][1] > 0.5)
        final[f"G13 {short(v)}"] = line(f"G13 {short(v):5s} r_K >= 0.75, lower > 0.5    " + " ".join(f"{mshort(m)}:{'met' if c[m] else 'not met' if c[m] is False else 'n/e'}" for m in models), c, out)
    frame = final["G13 THE"] is not True
    if frame:
        out("  G13 THE not met or not evaluable: G14-G17 are confounded with the sentence frame and reported as such")
    for v in G14:
        c = cmap(lambda c: c["r"][0] <= 0.75 and c["r"][2] < 1.0)
        final[f"G14 {short(v)}"] = line(f"G14 {short(v):6s} r_K <= 0.75, upper < 1.0    " + " ".join(f"{mshort(m)}:{'met' if c[m] else 'not met' if c[m] is False else 'n/e'}" for m in models), c, out)
    out(f"  G14 summary 'the read is strongest for an exact repeat': {'printed' if all(final[f'G14 {short(v)}'] for v in G14) else 'NOT printed'} "
        f"({sum(bool(final[f'G14 {short(v)}']) for v in G14)}/6 sub-verdicts met)")
    g15 = {}
    for v in G15 + ["POST_FR", "POST_DE"] + LISTS:
        c = cmap(lambda c: c["r"][0] <= 1 / 3 and c["r"][2] < 0.5 and c["rho"][0] <= 0.5 and c["rho"][2] < 0.75)
        lab, sec = (f"G15 {short(v):9s}", "") if v in G15 else (f"  secondary {short(v):9s}", " [the G15 rule on a secondary arm; exploratory, not a preregistered verdict]")
        res = line(f"{lab} r_K <= 1/3 (upper < 0.5) and rho_s <= 0.5 (upper < 0.75)    "
                   + " ".join(f"{mshort(m)}:{'met' if c[m] else 'not met' if c[m] is False else 'n/e'}" for m in models), c, out, sec)
        if v in G15:
            g15[v], final[f"G15 {short(v)}"] = c, res
    out(f"  G15 headline 'token-level on the answer-position measure': {'MET' if all(final[f'G15 {short(v)}'] for v in G15) else 'NOT MET'}")
    for v in G15:
        pr, pa = cmap(pattern_r), cmap(pattern_a)
        tok_r, con_r = rule({m: (x == "token") if x else None for m, x in pr.items()}), rule({m: (x == "concept") if x else None for m, x in pr.items()})
        tok_a, con_a = rule({m: (x == "token") if x else None for m, x in pa.items()}), rule({m: (x == "concept") if x else None for m, x in pa.items()})
        both = {m: ((pr[m] == "token" and pa[m] == "token") if pr[m] and pa[m] else None) for m in models}
        tok_both = rule(both)[0]  # token on both statistics within the same model (the G16 line); the decision table uses it
        final[f"G16 {short(v)}"] = line(f"G16 {short(v):6s} token pattern on r_K^any and a_v    "
                                        + " ".join(f"{mshort(m)}:r^any={pr[m] or 'n/e'},a={pa[m] or 'n/e'}" for m in models), both, out)
        out(f"      concept pattern: r^any {con_r[1]}/{con_r[2]}, a_v {con_a[1]}/{con_a[2]}; token pattern: r^any {tok_r[1]}/{tok_r[2]}, a_v {tok_a[1]}/{tok_a[2]}")
        g15v = final[f"G15 {short(v)}"]
        if g15v and tok_both:
            dec = "token-level lookup"
        elif con_a[0] and (con_r[0] or g15v is False):
            dec = "concept-level lookup"
        elif con_a[0] and g15v and tok_r[0]:
            dec = "lookup without readout"
        elif tok_a[0] and con_r[0]:
            dec = "reported as is (attention token pattern, any-form concept pattern; per-layer and per-head profiles examined)"
        elif tok_a[0] is None and tok_r[0] is None:
            dec = "not evaluable"
        else:
            dec = "graded (per-model profile reported verbatim, phrased as near-token-level)"
        out(f"      decision table {short(v)}: {dec}")
        decisions[short(v)] = dec
    for v in ("POST_FRMIX", "POST_DEMIX"):
        c = {m: (bool(cells[m][v]["dV"][0] > 0 and cells[m][v]["dV"][1] > 0) if m in ev and m in GAP_MODELS and g15[v].get(m) else None) for m in models}
        k, n = sum(bool(x) for x in c.values()), sum(x is not None for x in c.values())
        final[f"G17 {short(v)}"] = (k >= 2) if n >= 2 else None
        out(f"  G17 {short(v):5s} paired ID_V(v) - ID_V(POST) > 0 (CI excl. 0), cells with G15 met in {[mshort(g) for g in GAP_MODELS]}: "
            + " ".join(f"{mshort(m)}:{fmt(cells[m][v]['dV']) if m in ev and 'dV' in cells[m][v] else 'n/a'}{'' if c[m] is not None else '(n/e)'}" for m in models)
            + f" -> {k}/{n} evaluable cells -> {'NOT EVALUABLE (< 2 evaluable cells)' if n < 2 else verdict(k >= 2)}")
    if frame:
        out("  (G14-G17 above are frame-confounded: G13 THE not met)")
    explore(M, cells, models, out)
    return {"cells": cells, "final": final, "gate0": {m: M[m].g0 for m in M if M[m].ok}, "models": M, "decisions": decisions}


def explore(M, cells, models, out):
    out("\n== Exploratory (not scored)")
    for m in models:
        if m not in cells:
            continue
        md, C = M[m], cells[m]
        order = sorted((v for v in SENT if C[v].get("r")), key=lambda v: -C[v]["r"][0])
        out(f"  {m} r_K ordering (OTHER-floored): " + " > ".join(f"{short(v)} {C[v]['r'][0]:.2f}" for v in order))
        out(f"  {m} NONE-floored: " + "  ".join(f"{short(v)} {C[v]['r_none'][0]:.2f}" for v in order))
        fx = [f"r({short(a)}) {C[a]['r'][0]:.2f} vs r({short(b)}) {C[b]['r'][0]:.2f}" for a, b in (("POST_FR", "POST_FRMIX"), ("POST_DE", "POST_DEMIX"))
              if C[a].get("r") and C[b].get("r")]
        if fx:
            out(f"  {m} frame effect (sentence language vs word form): " + "; ".join(fx))
        ids = sorted(set.intersection(*(set(md.arms[a]) for a in md.arms)))
        out(f"  {m} l0 rows (ID_K-style d_K / d_V per arm, all cores): " + "  ".join(
            f"{short(a)} dK {np.mean(vec(md.arms[a], ids, 'dK')):+.1f} dV {np.mean(vec(md.arms[a], ids, 'dV')):+.1f}" for a in ("POST", "POST_SYN", "POST_FRMIX", "POST_DEMIX", "POST_OTHER") if a in md.arms))
        for v in ("POST_SYN", "POST_DE", "POST_DEMIX"):
            if v in md.arms:
                pair = [i for i in ids if {md.arms[v][i]["core"]["source"], md.arms[v][i]["X"]} == {"cabinet", "closet"}]
                rest = [i for i in ids if i not in pair]
                pair and out(f"  {m} {short(v)} per-pair: {{S,X}} = {{cabinet, closet}} ID_K {fmt(boot(vec(md.arms[v], pair)))} (n={len(pair)}) vs other pairs {fmt(boot(vec(md.arms[v], rest)))}")
        forms = [v for v in SENT + LISTS if C[v].get("r_form")]
        if forms:
            out(f"  {m} form-scored r_K^form: " + "  ".join(f"{short(v)} {C[v]['r_form'][0]:.2f}" for v in forms))
        if md.probe:
            for v in ("POST_THE", "POST_MODIF", "POST_TITLE", "POST_UPPER", "POST_PLURAL", "POST_SYN", "POST_FRMIX", "POST_DEMIX", "POST_FR", "POST_DE"):
                a, n = md.a_v(v)
                if a:
                    prof = np.mean([x["layers"] for x in md.probe[v].values()], 0)
                    out(f"  {m} probe {short(v):6s} a_v {fmt(a)} (n={n}); A_v {np.mean([x['A'] for x in md.probe[v].values()]):+.4f}, last-token "
                        f"{np.mean([x['last'] for x in md.probe[v].values()]):+.4f}, B-form {np.mean([x['B'] for x in md.probe[v].values()]):+.4f}; "
                        f"per-layer max {prof.max():+.4f} at L{int(prof.argmax())}")
        out(f"  {m} clean candidate mass under the floors/lists: " + "  ".join(
            f"{short(a)} {np.mean([x['mass'] for x in md.arms[a].values()]):.3f}" for a in ("POST", "POST_FR", "AFTER", "AFTER_FR", "AFTER_DE") if a in md.arms))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/gpu_stage5")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--test", action="store_true", help="score every <model>_s0.json found; banner")
    a = ap.parse_args(argv)
    score(a.root, list(filter(None, a.models.split(","))), a.test)


if __name__ == "__main__":
    main()
