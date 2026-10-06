"""Part (e) of P-2026-10-05-G (docs/PREREGISTRATION.md): Gate e and predictions G18-G22 on the IOI factorial
({root}/ioi/{model}_s1.json from experiments/ioi_factorial.py; seed 1 = ckeys.ioi.SEED, the confirmatory cores; seed 0 is the disclosed GPT-2 pilot), exactly as drafted; the row splice
({root}/row_restricted/{model}_ioi.json) and the GPT-2 attention probe ({root}/ioi_attention/{model}.json) are
exploratory. Called by analysis/stage5_score.py (``score(root)``) or standalone.

Per core (ckeys.ioi.identity_measures): ID_K, ID_V, ID_KV against the batched self-clamp row (keys/values clamped from
layer 0), the clean-run competence and the batch-noise floors. Means over cores; 95 % percentile CIs from 10,000
core-bootstrap resamples (numpy seed 20261006, one index set per n, shared by every arm of a model); ratios of means
(f_K = ID_K/ID_KV, f_V, s_ID = ID_K/(ID_K + ID_V), |ID_K(Q)|/|ID_K(A)|) and the fraction difference f_V(QUESTION) -
f_V(AFTER) are recomputed within each resample on the intersected cores (dropped cores printed, expected 0).
Signed contrasts: sgn_arm = sign of the full-sample mean ID_K of the manipulated arm (AFTER at 7B, INLINE at GPT-2),
fixed before resampling; the contrast is sgn_arm x mean[ID_K(arm) - ID_K(control)].
Gate e (per model x arm, clean B run): two-way >= 0.75, four-way >= 0.50, mean LD > 0 with CI excluding 0.
G18  f_K(PLAIN) in [-0.10, +0.10] with CI within [-0.20, +0.20], 3/3 of GPT-2 small, Qwen2.5-7B-Instruct, Mistral-7B
     (precondition mean ID_KV(PLAIN) >= 3 nats, else not evaluable).
G19  7B pair: (a) CI of ID_K(AFTER) excludes 0, |mean| >= 2.0, signed contrast vs QUESTION > 0 (CI excl. 0), 2/2;
     (b) the sign is positive, 2/2, where (a) holds.
G20  given G19a and b: f_V(AFTER) <= 0.50, f_V(QUESTION) >= 0.75, paired f_V(QUESTION) - f_V(AFTER) > 0 (CI excl. 0),
     s_ID(AFTER) >= 0.50, 2/2.
G21  (a) |ID_K(BEFORE)| <= 0.5 and signed contrast sgn_AFTER x [ID_K(AFTER) - ID_K(BEFORE)] > 0 (CI excl. 0), evaluated
     where G19a holds (else not evaluable), 2/2;
     (b) |f_K(QUESTION)| <= 0.15 and |ID_K(QUESTION)| <= 0.20 x |ID_K(AFTER)|, 2/2; (c) GPT-2 |ID_K(INLINE_BEFORE)| <= 0.5.
G22  GPT-2 small: (a) CI of ID_K(INLINE) excludes 0, |mean| >= 0.5, signed contrast vs PLAIN >= 0.5 (CI excl. 0) and
     vs INLINE_BEFORE > 0 (CI excl. 0); (b) ID_K(INLINE) < 0 with CI excluding 0 (scored on its own, as written).
Alternatives printed per 7B model: "inhibitory key read" (G19a met, G19b not; G20 descriptive), "additive mixture"
(f_V(AFTER) >= 0.5). A failing gate makes the cell not evaluable, which counts as not met in every k/k line. Only the
conditions the entry states are applied (G19b and G21a where (a) holds, G20 given G19); the row splice and the attention probe
are exploratory and carry no preregistered criterion.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ckeys.ioi import ARMS, SEED as IOI_SEED, identity_measures  # noqa: E402

IOI, ROWS, ATT = "ioi", "row_restricted", "ioi_attention"
GPT2 = "gpt2"
PAIR = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"]
EXPLORATORY = ["Qwen2.5-7B", "Qwen2.5-14B-Instruct", "gpt2-xl"]
SEED, B = 20261006, 10000
NAN = (float("nan"),) * 3
_idx = {}


def IDX(n):
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    return _idx[n]


def boot(x):
    x = np.asarray(x, float)
    if not len(x):
        return NAN
    bs = x[IDX(len(x))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ratio(a, b, absolute=False):
    """Ratio of means over the same cores, recomputed in each resample (``absolute``: of the absolute means)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return NAN
    idx = IDX(len(a))
    ma, mb = a[idx].mean(1), b[idx].mean(1)
    r = (np.abs(ma) / np.abs(mb)) if absolute else ma / mb
    p = abs(a.mean()) / abs(b.mean()) if absolute else a.mean() / b.mean()
    return p, np.percentile(r, 2.5), np.percentile(r, 97.5)


def frac_diff(aV, aKV, bV, bKV):
    """f_V(a) - f_V(b) = mean aV/mean aKV - mean bV/mean bKV over the same cores, recomputed in each resample."""
    aV, aKV, bV, bKV = (np.asarray(x, float) for x in (aV, aKV, bV, bKV))
    if not len(aV):
        return NAN
    idx = IDX(len(aV))
    r = aV[idx].mean(1) / aKV[idx].mean(1) - bV[idx].mean(1) / bKV[idx].mean(1)
    return aV.mean() / aKV.mean() - bV.mean() / bKV.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def fmt(t, d=2):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"


def verdict(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def excl0(t):
    return t[1] > 0 or t[2] < 0


def ioi_dir(root):
    return Path(root) / IOI if (Path(root) / IOI).is_dir() else Path(root)  # a bare pilot directory holds the files itself


def load(root, model, seed=IOI_SEED):
    f = ioi_dir(root) / f"{model}_s{seed}.json"
    if not f.exists():
        return {}, {}
    d = json.load(open(f))
    arms = {}
    for r in d["results"]:
        arms.setdefault(r["arm"], {})[json.dumps(r["core"], sort_keys=True)] = identity_measures(r)
    return d["provenance"], arms


def vec(cell, ids, key):
    return np.array([cell[i][key] for i in ids], float)


def paired(A, Bc, key="idK"):
    ids = sorted(set(A) & set(Bc))
    return [A[i][key] - Bc[i][key] for i in ids], len(A) + len(Bc) - 2 * len(ids)


def signed(A, Bc, sgn):
    d, dropped = paired(A, Bc)
    t = boot(d)
    return (sgn * t[0], min(sgn * t[1], sgn * t[2]), max(sgn * t[1], sgn * t[2])), dropped


def gate(cell, out, label):
    ids = sorted(cell)
    two, four = vec(cell, ids, "two_B").mean(), vec(cell, ids, "four_B").mean()
    ld = boot(vec(cell, ids, "LD_B"))
    ok = bool(two >= 0.75 and four >= 0.50 and ld[1] > 0)
    out(f"  gate e {label:14s} n={len(ids):3d} two-way {two:.2f} (>= 0.75) four-way {four:.2f} (>= 0.50) LD {fmt(ld)} (> 0, CI excl. 0) "
        f"cand-mass {vec(cell, ids, 'mass_B').mean():.2f} -> {'passed' if ok else 'FAILED (not evaluable)'}")
    return ok


def table(arms, out, floors=True):
    for arm in ARMS:
        if arm not in arms:
            continue
        c, ids = arms[arm], sorted(arms[arm])
        K, V, KV = (vec(c, ids, k) for k in ("idK", "idV", "idKV"))
        d0 = {ch: np.array([c[i]["d"][f"{ch}_S@0"] for i in ids]) for ch in ("K", "V", "KV")}
        fl = (vec(c, ids, "floor_B").mean(), vec(c, ids, "floor_S").mean())
        noisy = " NOISY" if max(fl) > 1 else ""
        out(f"  {arm:14s} n={len(ids):3d} sign(ID_K) {'+' if K.mean() > 0 else '-'}  ID_K {fmt(boot(K))}  ID_V {fmt(boot(V))}  ID_KV {fmt(boot(KV))}  "
            f"f_K {fmt(ratio(K, KV))}  f_V {fmt(ratio(V, KV))}  s_ID {fmt(ratio(K, K + V))}  s_K {fmt(ratio(d0['K'], d0['K'] + d0['V']))}  "
            f"interaction {fmt(boot(d0['KV'] - d0['K'] - d0['V']))}" + (f"  floors B {fl[0]:.3f} S {fl[1]:.3f}{noisy}" if floors else ""))


def exploratory(arms, out):
    out("  exploratory: four-way renormalised ID, LD-based d_C, onset 0.3 L key share, ABBA/BABA, competent subset, clean log-probs")
    for arm in ARMS:
        if arm not in arms:
            continue
        c, ids = arms[arm], sorted(arms[arm])
        L = c[ids[0]]["L"]
        l0 = [int(k.split("@")[1]) for k in c[ids[0]]["d"] if k.startswith("KV_S") and not k.endswith("@0")]
        K4, V4, KV4 = (vec(c, ids, k) for k in ("id4K", "id4V", "id4KV"))
        dld = {ch: np.array([c[i]["dLD"][f"{ch}_S@0"] for i in ids]) for ch in ("K", "V", "KV")}
        line = f"    {arm:14s} ID4_K {fmt(boot(K4))} ID4_V {fmt(boot(V4))} ID4_KV {fmt(boot(KV4))} f4_K {fmt(ratio(K4, KV4))} | d^LD K {boot(dld['K'])[0]:+.2f} V {boot(dld['V'])[0]:+.2f} KV {boot(dld['KV'])[0]:+.2f}"
        if l0:
            dm = {ch: np.array([c[i]["d"][f"{ch}_S@{l0[0]}"] for i in ids]) for ch in ("K", "V")}
            line += f" | l0={l0[0]}/{L} dK {dm['K'].mean():+.2f} dV {dm['V'].mean():+.2f} s_K {fmt(ratio(dm['K'], dm['K'] + dm['V']))}"
        out(line)
        for pat in ("ABBA", "BABA"):
            sub = [i for i in ids if c[i]["pattern"] == pat]
            if sub:
                K, V, KV = (vec(c, sub, k) for k in ("idK", "idV", "idKV"))
                out(f"      {pat} n={len(sub):3d} ID_K {fmt(boot(K))} ID_V {fmt(boot(V))} ID_KV {fmt(boot(KV))} f_K {fmt(ratio(K, KV))} two-way {vec(c, sub, 'two_B').mean():.2f}")
        comp = [i for i in ids if c[i]["two_B"] and c[i]["two_S"]]
        if comp:
            K, V, KV = (vec(c, comp, k) for k in ("idK", "idV", "idKV"))
            out(f"      competent (two-way correct in B and S) n={len(comp):3d} ID_K {fmt(boot(K))} f_K {fmt(ratio(K, KV))} f_V {fmt(ratio(V, KV))} s_ID {fmt(ratio(K, K + V))}")
        lp = {t: np.array([c[i]["lpB"][t] for i in ids]) for t in ("B", "S", "X", "Subj")}
        out("      clean-B lp mean [q1, median, q3]: " + "  ".join(f"{t} {v.mean():+.2f} [{np.percentile(v, 25):+.1f},{np.median(v):+.1f},{np.percentile(v, 75):+.1f}]" for t, v in lp.items()))
    if "AFTER" in arms and "QUESTION" in arms:
        ids = sorted(set(arms["AFTER"]) & set(arms["QUESTION"]))
        out(f"    nats ratio ID_V(AFTER)/ID_V(QUESTION) {fmt(ratio(vec(arms['AFTER'], ids, 'idV'), vec(arms['QUESTION'], ids, 'idV')))} (descriptive, floor-confounded)")


def splice(root, model, out):
    f = Path(root) / ROWS / f"{model}_ioi.json"
    if not f.exists():
        return
    d = json.load(open(f))
    res, prov = (d["results"], d.get("provenance", {})) if isinstance(d, dict) else (d, {})
    out(f"  row splice (exploratory, {f.name}; skipped_items {prov.get('skipped_items', '-')}): fraction of the full key effect (ratio of mean log-odds changes) seen by each row group")
    for arm in sorted({r["arm"] for r in res}):
        R = [r for r in res if r["arm"] == arm]
        full = np.array([r["m"]["all"] - r["m_B"] for r in R])
        none = np.array([r["m"]["none"] - r["m_B"] for r in R])
        gs = [g for g in ("options_sx", "options", "choices", "tail", "sentence_tail", "rest_after_p", "self") if g in R[0]["m"]]
        out(f"    {arm} n={len(R)} full {full.mean():+.2f} none {none.mean():+.3f} | " + "  ".join(
            f"{g} {fmt(ratio(np.array([r['m'][g] - r['m_B'] for r in R]), full))}" for g in gs) + "   (exploratory, no preregistered criterion; stage-3 reference under the belief task: options rows about 0.7-1.0)")


def probe(root, model, out):
    f = Path(root) / ATT / f"{model}.json"
    if f.exists():
        out(f"  attention probe (exploratory, {f.name}):")
        for line in json.load(open(f)).get("summary", "").splitlines():
            out("    " + line)


def score(root, pair=None, gpt2=GPT2, exploratory_models=None, test=False, out=print, seed=IOI_SEED):
    root = Path(root)
    found = sorted(f.stem[:-3] for f in ioi_dir(root).glob(f"*_s{seed}.json"))
    if test:
        out("TEST MODE: verdict lines are not preregistered results; every non-GPT-2 model found stands in for the 7B pair")
        pair, exploratory_models = [m for m in found if not m.startswith("gpt2")], []
    pair = PAIR if pair is None else pair
    exploratory_models = EXPLORATORY if exploratory_models is None else exploratory_models
    plain3 = [gpt2] + pair
    out(f"== Part (e): IOI, K/V clamps at the IO mention (Gate e, G18-G22); GPT-2 small = {gpt2}; 7B pair = {pair}")
    G, V = {}, {}   # G[model][arm] gate; V[model][pred] True / False / None
    for m in plain3 + exploratory_models:
        prov, arms = load(root, m, seed)
        role = "GPT-2 small" if m == gpt2 else "7B pair" if m in pair else "exploratory"
        out(f"\n## {m} ({role})")
        V[m], G[m] = {}, {}
        if not arms:
            out("  ioi factorial MISSING (every cell not evaluable)")
            continue
        out(f"  skipped_items {prov.get('skipped_items')}, chat {prov.get('chat')}, bos {prov.get('bos')}, dtype {prov.get('args', {}).get('dtype')}, "
            f"assert_exact {prov.get('assert_exact')}, exact_violations {prov.get('exact_violations', '-')} (floor > {prov.get('exact_threshold', 1e-3):g}; FP32 only), "
            f"max floor {'-' if prov.get('max_floor') is None else format(prov['max_floor'], '.2e')}, arms {[a for a in ARMS if a in arms]}"
            + (f", label: {prov['label']}" if prov.get("label") else ""))
        for arm in ARMS:
            if arm in arms:
                G[m][arm] = gate(arms[arm], out, arm)
        table(arms, out)
        g = G[m]
        A = lambda arm: arms[arm] if g.get(arm) else None  # noqa: E731  evaluable cell or None
        # G18
        c = A("PLAIN")
        if c is not None:
            ids = sorted(c)
            K, KV = vec(c, ids, "idK"), vec(c, ids, "idKV")
            fk, kv = ratio(K, KV), boot(KV)
            pre = kv[0] >= 3.0
            V[m]["G18"] = (abs(fk[0]) <= 0.10 and fk[1] >= -0.20 and fk[2] <= 0.20) if pre else None
            out(f"  G18 PLAIN f_K {fmt(fk)} (|.| <= 0.10, CI within [-0.20, +0.20]); precondition ID_KV {fmt(kv)} >= 3 -> {'ok' if pre else 'FAILS'}; "
                f"ID_K {fmt(boot(K))} -> {verdict(V[m]['G18'])}")
        else:
            V[m]["G18"] = None
            out("  G18 PLAIN: not evaluable (cell missing or gate failed)")
        # G19-G21 (7B pair quantities; printed for every model, scored on the pair)
        a, q, bf = A("AFTER"), A("QUESTION"), A("BEFORE")
        sgn = 0
        if a is not None:
            ia = sorted(a)
            Ka = vec(a, ia, "idK")
            ka = boot(Ka)
            sgn = 1 if ka[0] >= 0 else -1
            out(f"  AFTER ID_K {fmt(ka)} sign {'+' if sgn > 0 else '-'} |f_K| {fmt(ratio(Ka, vec(a, ia, 'idKV'), absolute=True))} (reported, no preregistered criterion)")
        if a is not None and q is not None:
            con, dropped = signed(a, q, sgn)
            V[m]["G19a"] = excl0(ka) and abs(ka[0]) >= 2.0 and con[1] > 0
            out(f"  G19a CI excl. 0 {excl0(ka)}, |mean| {abs(ka[0]):.2f} >= 2.0, signed contrast sgn x [ID_K(AFTER) - ID_K(QUESTION)] {fmt(con)} > 0 "
                f"(dropped cores {dropped}) -> {verdict(V[m]['G19a'])}")
            V[m]["G19b"] = (ka[0] > 0 and ka[1] > 0) if V[m]["G19a"] else None
            out(f"  G19b sign positive (selection): mean {ka[0]:+.2f}, lower {ka[1]:+.2f} > 0 -> {verdict(V[m]['G19b'])}"
                + ("   ALTERNATIVE: inhibitory key read (G19a met with a negative sign; G20 descriptive)" if V[m]["G19a"] and V[m]["G19b"] is not None and not V[m]["G19b"] else ""))
            iq = sorted(q)
            fva, fvq = ratio(vec(a, ia, "idV"), vec(a, ia, "idKV")), ratio(vec(q, iq, "idV"), vec(q, iq, "idKV"))
            sid = ratio(Ka, Ka + vec(a, ia, "idV"))
            ids = sorted(set(a) & set(q))
            fd = frac_diff(vec(q, ids, "idV"), vec(q, ids, "idKV"), vec(a, ids, "idV"), vec(a, ids, "idKV"))
            ok20 = fva[0] <= 0.50 and fvq[0] >= 0.75 and fd[1] > 0 and sid[0] >= 0.50
            V[m]["G20"] = ok20 if (V[m]["G19a"] and V[m]["G19b"]) else None
            out(f"  G20 f_V(AFTER) {fmt(fva)} <= 0.50, f_V(QUESTION) {fmt(fvq)} >= 0.75, paired f_V(QUESTION) - f_V(AFTER) {fmt(fd)} > 0 (n={len(ids)}), "
                f"s_ID(AFTER) {fmt(sid)} >= 0.50 -> {verdict(V[m]['G20'])}" + ("   ALTERNATIVE: additive mixture (f_V(AFTER) >= 0.5)" if fva[0] >= 0.5 and V[m]["G19a"] else ""))
            Kq = vec(q, iq, "idK")
            fkq, rq = ratio(Kq, vec(q, iq, "idKV")), ratio(vec(q, ids, "idK"), vec(a, ids, "idK"), absolute=True)
            V[m]["G21b"] = abs(fkq[0]) <= 0.15 and rq[0] <= 0.20
            out(f"  G21b QUESTION ID_K {fmt(boot(Kq))}, |f_K| {abs(fkq[0]):.2f} <= 0.15, |ID_K(Q)|/|ID_K(A)| {fmt(rq)} <= 0.20 -> {verdict(V[m]['G21b'])}")
        else:
            V[m]["G19a"] = V[m]["G19b"] = V[m]["G20"] = V[m]["G21b"] = None
            out("  G19a/G19b/G20/G21b: not evaluable (AFTER or QUESTION cell missing or gate failed)")
        if bf is not None:
            kb = boot(vec(bf, sorted(bf), "idK"))
            mag = abs(kb[0]) <= 0.5
            if a is not None:  # sgn from AFTER's point estimate; the contrast is evaluated where G19a holds (else not evaluable: the sign is noise)
                con, dropped = signed(a, bf, sgn)
                V[m]["G21a"] = (mag and con[1] > 0) if V[m]["G19a"] else None
                out(f"  G21a BEFORE ID_K {fmt(kb)} |mean| <= 0.5 {mag}; signed contrast sgn x [ID_K(AFTER) - ID_K(BEFORE)] {fmt(con)} > 0 (dropped {dropped})"
                    + ("" if V[m]["G19a"] else "; G19a not met: the contrast is not evaluable") + f" -> {verdict(V[m]['G21a'])}")
            else:
                V[m]["G21a"] = None
                out(f"  G21a BEFORE ID_K {fmt(kb)} |mean| <= 0.5 {mag}; contrast not evaluable (AFTER cell missing or gate failed) -> NOT EVALUABLE")
        else:
            V[m]["G21a"] = None
            out("  G21a BEFORE: not evaluable (cell missing or gate failed)")
        # G21c / G22 (GPT-2 small quantities; printed for every model with the arms, scored on GPT-2 small)
        inl, ib, pl = A("INLINE"), A("INLINE_BEFORE"), A("PLAIN")
        if ib is not None:
            kib = boot(vec(ib, sorted(ib), "idK"))
            V[m]["G21c"] = abs(kib[0]) <= 0.5
            out(f"  G21c INLINE_BEFORE ID_K {fmt(kib)} |mean| <= 0.5 -> {verdict(V[m]['G21c'])}")
        else:
            V[m]["G21c"] = None
            out("  G21c INLINE_BEFORE: not evaluable (cell missing or gate failed)")
        if inl is not None and ib is not None and pl is not None:
            ki = boot(vec(inl, sorted(inl), "idK"))
            s = 1 if ki[0] >= 0 else -1
            cp, dp = signed(inl, pl, s)
            cb, db = signed(inl, ib, s)
            V[m]["G22a"] = excl0(ki) and abs(ki[0]) >= 0.5 and cp[0] >= 0.5 and cp[1] > 0 and cb[1] > 0
            out(f"  G22a INLINE ID_K {fmt(ki)} sign {'+' if s > 0 else '-'}: CI excl. 0 {excl0(ki)}, |mean| >= 0.5; signed contrast vs PLAIN {fmt(cp)} (>= 0.5, CI excl. 0; dropped {dp}), "
                f"vs INLINE_BEFORE {fmt(cb)} (> 0, CI excl. 0; dropped {db}) -> {verdict(V[m]['G22a'])}")
            V[m]["G22b"] = ki[0] < 0 and ki[2] < 0
            out(f"  G22b INLINE ID_K < 0 with CI excluding 0 (inhibitory): upper {ki[2]:+.2f} -> {verdict(V[m]['G22b'])}")
        else:
            V[m]["G22a"] = V[m]["G22b"] = None
            out("  G22a/G22b: not evaluable (INLINE, INLINE_BEFORE or PLAIN cell missing or gate failed)")
        exploratory(arms, out)
        splice(root, m, out)
        probe(root, m, out)

    V = {m: {k: (None if v is None else bool(v)) for k, v in d.items()} for m, d in V.items()}

    def k_of(pred, models):
        return sum(V.get(m, {}).get(pred) is True for m in models), len(models)

    out(f"\n== Verdicts (G18-G22; not evaluable counts as not met; GPT-2 small = {gpt2}, 7B pair = {pair})")
    out("  Gate e: " + "; ".join(f"{m} " + ",".join(f"{a}:{'ok' if ok else 'FAIL'}" for a, ok in G.get(m, {}).items()) for m in plain3))
    final = {}
    for pred, name, models in (("G18", "no key read in plain IOI", plain3), ("G19a", "a later list opens a key read (existence)", pair),
                               ("G19b", "the sign is positive (selection)", pair), ("G20", "the lookup replaces the copy", pair),
                               ("G21a", "position control (BEFORE)", pair), ("G21b", "wrapper control (QUESTION)", pair)):
        k, n = k_of(pred, models)
        final[pred] = k == n
        out(f"  {pred:5s} {name:42s} {k}/{n} -> {verdict(final[pred])}   per model: {[(m, V.get(m, {}).get(pred)) for m in models]}")
    for pred, name in (("G21c", "GPT-2 small INLINE_BEFORE |ID_K| <= 0.5"), ("G22a", "GPT-2 small in-sentence re-mention opens a key read"),
                       ("G22b", "GPT-2 small in-sentence read is inhibitory")):
        v = V.get(gpt2, {}).get(pred)
        final[pred] = v is True
        out(f"  {pred:5s} {name:42s} -> {verdict(v)}")
    alt = [m for m in pair if V.get(m, {}).get("G19a") and V[m].get("G19b") is False]
    if alt:
        out(f"  ALTERNATIVE met: inhibitory key read at {alt} (G19a met with a negative sign; G20 descriptive there)")
    scored = {p: plain3 if p == "G18" else [gpt2] if p in ("G21c", "G22a", "G22b") else pair for p in final}
    return {"gates": G, "pred": V, "final": final, "models": scored}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/gpu_stage5")
    ap.add_argument("--pair", default=",".join(PAIR))
    ap.add_argument("--gpt2", default=GPT2)
    ap.add_argument("--exploratory", default=",".join(EXPLORATORY))
    ap.add_argument("--test", action="store_true", help="every non-GPT-2 model found stands in for the 7B pair; banner")
    a = ap.parse_args(argv)
    score(a.root, list(filter(None, a.pair.split(","))), a.gpt2, list(filter(None, a.exploratory.split(","))), a.test)


if __name__ == "__main__":
    main()
