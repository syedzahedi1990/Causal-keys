"""Score part D of preregistration P-2026-10-10-J (docs/PREREGISTRATION.md, GPU stage 8): what the reader heads write (the
identity flag), the sign of the key read, and the 1.5B / 3B route. Gates J-D-G0 to J-D-G7, the confirmatory lines J-D1 ...
J-D6-ROUTE, the D6 decision table, the reported lines, the summary by risk class (with the Holm sensitivity analysis over
the interval components of the R lines, computed by the shared helper analysis/stage8_holm.py) and the exploratory report.

Inputs under --results (default results/gpu_stage8d), as written by scripts/gpu_stage8d.sh: <stage>/<tag>.json for the
stages preflight, sets, fit, inject, ablate, bind, sign, diss, before, xtask of experiments/stage8_flag.py (tags qwen7,
mistral7, qwen1.5, qwen3b; TEST_<key> with --test), logs/pytest.log (J-D-G0: its last pytest run), COMMIT.txt, ENV.txt,
REVISIONS.txt, SKIPPED.txt. Output STAGE8D_SCORE.txt (also --out) with the sections PROVENANCE, POPULATION, GATES,
PREDICTIONS, REPORTED, SUMMARY, EXPLORATORY. Exit status 1 if a part of the scorer raised (its lines NOT EVALUABLE, the
traceback printed); 2 if, outside --test, the provenance or population check reports MISMATCH, or results exist without
a passing J-D-G0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage8d_parts import lines as LN  # noqa: E402
from stage8d_parts.stats import Tests, V, combine, f3  # noqa: E402

BIG, SMALL, DISS = ("qwen7", "mistral7"), ("qwen1.5", "qwen3b"), ("qwen1.5", "qwen3b", "qwen7")
KEYS = ("qwen7", "mistral7", "qwen1.5", "qwen3b")
STAGES = ("preflight", "sets", "fit", "inject", "ablate", "bind", "sign", "diss", "before", "xtask")
STEPS_OF = {"qwen7": ("preflight", "sets", "fit", "inject", "ablate", "bind", "sign", "diss"),
            "mistral7": ("preflight", "sets", "fit", "inject", "ablate", "bind", "sign"),
            "qwen1.5": ("preflight", "sets", "fit", "diss"), "qwen3b": ("preflight", "sets", "fit", "diss")}
SIZES = {"inject": 100, "ablate": 60, "bind": 100, "Q": 100, "IOI": 100, "diss": {"qwen1.5": 100, "qwen3b": 60, "qwen7": 100}}
G0_FILES = {"tests/test_flag.py": 13, "tests/test_questions.py": 11, "tests/test_stage8d_score.py": 15}
POP_SHA = {  # = experiments/stage8_flag.POP_SHA (tests/test_stage8d_score.py checks the copy)
    "R": "9036af1a838a12d58a7a7eb40f70dd800dd659bfd6e95ed1ab5a2ce10862f936",
    "R'": "fe348ba40a19182a67cee382a3533c1af009273acb740c01d418662a570122a2",
    "E8": "2c198705c595d8068467a188d46e5c22793e079f837ba39bc1e0b43fb14b0f13",
    "BIND": "495c14620cd22e38af0ac67c376441442c40264996db98872d2bae4009a7ee92",
    "F_IOI_CAND": "d41e791a42cc3a833b32220c54f6aa5b129bfda79e904fb8bf5c382ebf0f8b9a",
    "E_IOI_CAND": "2b165365fc60e69af9bbe0d943bb9ea7fc30ec65a2747cd6f04630e58ddd0334",
    "XFIT_paint": "9b9a320f4a58309b388517d63b9d31b6ba58d29546f13e73bdc8258c01cf2189",
    "XEVAL_paint": "be234aaf7083406fa87ee16d896feae675ff473b461694673a9027cffb36970c",
    "XFIT_schedule": "4a56ae029499e3d93b7adcb375febca35394c7f234fc459d32af7336041433c4",
    "XEVAL_schedule": "8b4c7c05fdbe82e9cf5b947024e8dc4a904d07b0f6d955e0e5bf155752396edc"}
SETS_SHA = {"qwen7": "dea841d0d49899bac7ad35c71a851ca7ec5d212d8ecd5310407de54d5c77e72a",
            "mistral7": "55a1b17b1ba16b6afc4e0ae94b15818b8b5c11f02d985deea214aa0d49152220"}
FLOOR = 0.05                         # J-D-G2: mean |duplicated none row - none row| (nats)
# code -> (class, kind, prior, models, title); the priors were recorded in the entry before any stage-8 output. Class
# follows the prior (decision D1): L = implied by data in hand on the same models and material, prior >= 0.9;
# M = prior >= 0.8; R = prior < 0.8.
LINES = {
    "J-D1": ("R", "A", 0.55, BIG, "sufficiency: the leave-one-word-out flag moves the answer to an absent option (ID_inj^LOO / ID_K)"),
    "J-D2": ("R", "A", 0.55, BIG, "specificity against structured controls (own write orthogonal to the flag, mean H* output, active-set flag)"),
    "J-D3": ("R", "A", 0.50, BIG, "necessity: ablating one direction per layer at the option rows removes the key read (rho_K)"),
    "J-D4": ("M", "A", 0.85, BIG, "the injected flag is read at hop 2 by key (r_ans^inj)"),
    "J-D-ADDR": ("M", "A", 0.80, BIG, "the flag is identity-free: the key-only and the natural (K and V) writes give the same flag"),
    "J-D-KN": ("R", "A", 0.40, BIG, "a non-candidate key acts as removing B's flag, with no specific beneficiary"),
    "J-D5": ("R", "A", 0.25, BIG, "the flag of an initial-state sentence carries its binding (Psi_bind > 0 in both sentence orders)"),
    "J-D7": ("M", "A", 0.85, BIG, "consistency: the key read is negative under Q_OUT and positive under Q_IN"),
    "J-D-SIGN-Q": ("M", "A", 0.85, BIG, "Q_OUT: the key and the value reads are both negative (task semantics)"),
    "J-D-SIGN-IOI": ("L", "A", 0.90, BIG, "IOI INLINE: the key read is negative while the value read is positive"),
    "J-D8-Q": ("M", "A", 0.85, BIG, "consistency: the frozen H* carry the Q_IN and Q_OUT key reads"),
    "J-D8-IOI": ("R", "A", 0.55, BIG, "the frozen H* carry the negative IOI INLINE key read at the listed names"),
    "J-D-ROUTE-IOI": ("R", "A", 0.40, BIG, "IOI INLINE: the negative read is applied at the answer row's key read of the listed names"),
    "J-D-HOP2": ("R", "A", 0.25, BIG, "the same top-10 hop-2 heads carry the injected flag's Q_OUT and IOI effects (same reader, opposite sign)"),
    "J-D6a": ("L", "A", 0.90, DISS, "consistency: the flag is written in sentences as in lists at every scale (omega, kappa)"),
    "J-D6": ("R", "A", 0.35, DISS, "mediation: the sentence-row flag carries the natural POST/P1 read ratio at 1.5B, 3B and 7B"),
    "J-D6-ROUTE": ("R", "A", 0.35, ("qwen1.5",), "at 1.5B the answer reads the sentence-row flag by value (r_ans^inj(V) > r_ans^inj(K))"),
}
NEED = {code: (1 if len(v[3]) == 1 else 2) for code, v in LINES.items()}
COMPONENTS = []                      # (line code, Holm component dict, rejected by the line's own rule): Holm sensitivity


def canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # noqa: BLE001  (a truncated file is reported, not fatal)
        return {"error": repr(ex)}


def tagof(key, test):
    return ("TEST_" if test else "") + key


# --------------------------------------------------------------------------- inputs
class Inputs:
    def __init__(self, root: Path, test: bool):
        self.root, self.test, self.F = root, test, {}
        for st in STAGES:
            for k in KEYS:
                f = root / st / f"{tagof(k, test)}.json"
                self.F[(st, k)] = load_json(f) if f.exists() else None

    def get(self, st, k):
        J = self.F.get((st, k))
        return J if J and "error" not in J and not (J.get("provenance") or {}).get("skipped") else None


def manifest_revisions():
    try:
        J = json.load(open(HERE.parent / "scripts" / "stage8_models.json"))
        return {k: v["revision"] for k, v in J["models"].items()}
    except Exception:  # noqa: BLE001
        return {}


def provenance(I: Inputs, out):
    out("PROVENANCE (the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt, SKIPPED.txt; every results file)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt", "SKIPPED.txt"):
        f = I.root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines()[-30:] if l.strip()) if f.exists() else " absent"))
    revs, ok, commits = manifest_revisions(), True, set()
    for (st, k), J in sorted(I.F.items()):
        if J is None:
            continue
        if "error" in J or "provenance" not in J:
            out(f"  {st}/{k}: " + (f"UNREADABLE {J.get('error')}" if "error" in J else "NO PROVENANCE") + "  MISMATCH")
            ok = False
            continue
        p = J["provenance"]
        if p.get("skipped"):
            out(f"  {st}/{k}: skipped ({p['skipped']})")
            continue
        commits.add(p.get("git_commit"))
        flag = []
        if not I.test:
            ver = p.get("verified") or {}
            if st != "preflight" and revs.get(k) and ver.get("revision") != revs[k]:
                flag.append(f"revision {ver.get('revision')} != the manifest's {revs[k]}")
            if "dtype" in p and not str(p["dtype"]).endswith("bfloat16"):
                flag.append("dtype is not BF16")
            want = "eager" if st == "sets" else "sdpa"
            if "attn_implementation" in p and p["attn_implementation"] != want:
                flag.append(f"attention is not {want}")
        S = I.get("sets", k)
        if st not in ("preflight", "sets") and S and p.get("sets_sha256") not in (None, S["sets_sha256"]):
            flag.append("sets hash differs from the sets file")
        Fj = I.get("fit", k)
        if st not in ("preflight", "sets", "fit", "before") and Fj and p.get("flags_sha256") != Fj["pt_sha256"]:
            flag.append("flags hash differs from the fit file")
        if st == "sets" and k in BIG and not I.test:
            sub = {x: J["sets"].get(x) for x in ("H", "rand", "active", "kstar")}
            if canon(sub) != SETS_SHA[k] or (p.get("stage6") or {}).get("sets_sha256") != SETS_SHA[k] or p.get("source") != "stage6":
                flag.append("H* / random / active sets differ from stage 6")
        ok &= not flag
        out(f"  {st}/{k}: commit {str(p.get('git_commit'))[:10]}, {p.get('model')} rev {str((p.get('verified') or {}).get('revision'))[:10]}, "
            f"dtype {p.get('dtype', '-')}, attn {p.get('attn_implementation', '-')}, device {p.get('device', '-')}, "
            f"test_mode {p.get('test_mode')}" + "".join(f"  MISMATCH: {x}" for x in flag))
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else ""))
    return ok and len(commits) <= 1


def population(I: Inputs, out):
    out("POPULATION (pinned hashes of R, R', E8, BIND, the IOI candidate lists and the task lists; sizes"
        + (" not checked in TEST MODE)" if I.test else ")"))
    ok = True
    for k in KEYS:
        pf = I.get("preflight", k)
        if pf is None:
            continue
        diff = {n: v for n, v in pf.get("sha", {}).items() if POP_SHA.get(n) != v}
        ov = {n: v for n, v in pf.get("overlap", {}).items() if v}
        good = not diff and not ov and set(pf.get("sha", {})) == set(POP_SHA)
        ok &= good
        out(f"  preflight {k}: hashes {'OK' if not diff else 'MISMATCH ' + str(sorted(diff))}; overlaps {ov or 'none'}; "
            f"IOI E_ioi {len(pf.get('ioi', {}).get('E_idx', []))} valid, F_ioi {len(pf.get('ioi', {}).get('F_idx', []))}; "
            f"prompts checked {pf.get('prompts_checked')}")
        for st in STAGES[1:]:
            J = I.get(st, k)
            if J is None:
                continue
            pp = J["provenance"].get("populations") or {}
            if pp != pf.get("sha"):
                ok = False
                out(f"  {st}/{k}: population hashes differ from the preflight: MISMATCH")
    if not I.test:
        for k in BIG:
            for st, want in (("inject", SIZES["inject"]), ("ablate", SIZES["ablate"]), ("bind", SIZES["bind"])):
                J = I.get(st, k)
                if J is not None and len(J["stories"]) != want:
                    ok = False
                    out(f"  {st}/{k}: {len(J['stories'])} stories != {want}: MISMATCH")
            J = I.get("sign", k)
            if J is not None:
                for arm, want in (("Q_IN", 100), ("Q_OUT", 100), ("P1", 100), ("INLINE", 100), ("INLINE_CHAT", 100)):
                    if len(J["results"].get(arm, [])) != want:
                        ok = False
                        out(f"  sign/{k} {arm}: {len(J['results'].get(arm, []))} != {want}: MISMATCH")
        for k in DISS:
            J = I.get("diss", k)
            if J is not None and len(J["stories"]) != SIZES["diss"][k]:
                ok = False
                out(f"  diss/{k}: {len(J['stories'])} stories != {SIZES['diss'][k]}: MISMATCH")
    out(f"  population: {'OK' if ok else 'MISMATCH'}")
    return ok


def gate_g0(root, out):
    f = root / "logs" / "pytest.log"
    if not f.exists():
        out("  J-D-G0  FP32 unit tests: no logs/pytest.log -> NOT EVALUABLE (not run)")
        return None
    t = f.read_text()
    hd = list(re.finditer(r"^==== (\S+) .*-m pytest.*$", t, re.M))
    t = t[hd[-1].end():] if hd else t
    res, ok = [], True
    for fn, need in G0_FILES.items():
        p, x, s = (len(re.findall(rf"^{re.escape(fn)}::\S+ {w}", t, re.M)) for w in ("PASSED", "(?:FAILED|ERROR)", "SKIPPED"))
        good = p >= need and x == 0 and s == 0
        ok &= good
        res.append(f"{fn} {p} passed, {x} failed, {s} skipped")
    other = len(re.findall(r"^tests/\S+::\S+ (?:FAILED|ERROR)", t, re.M))
    ok &= other == 0
    out(f"  J-D-G0  FP32 unit tests (last of {len(hd)} run(s)): {'; '.join(res)}; failures in the run {other} -> {V(ok)}")
    return ok


# --------------------------------------------------------------------------- the gates per model
class Gates:
    def __init__(self, I: Inputs, out, g0):
        self.I, self.g0 = I, g0
        self.inj, self.abl, self.bind, self.cell, self.diss = {}, {}, {}, {}, {}
        self.G1, self.G2, self.G3, self.G4, self.G5, self.G6 = {}, {}, {}, {}, {}, {}
        out("GATES (J-D-G0 before any model; J-D-G1 natural effect; J-D-G2 batch floor; J-D-G3 competence; J-D-G4 transfer "
            "evaluable; J-D-G5 binding queries; J-D-G6 dissociation population; J-D-G7 ratio denominators, per line)")
        for k in BIG:
            J = I.get("inject", k)
            if J:
                x = self.inj[k] = LN.Inject(J)
                NX = x.NX()
                self.G1[k] = bool(NX.pt >= 3 and NX.lo() > 0)
                out(f"  J-D-G1 {k:8s} N_X (P1, E8) {NX.txt()} (>= 3 nats, CI excl. 0) -> {V(self.G1[k])}")
                fl = x.floor()
                self.G2[(k, "inject")] = fl <= FLOOR
                out(f"  J-D-G2 {k:8s} inject batch floor {fl:.4f} (<= {FLOOR}) -> {V(fl <= FLOOR)}")
            else:
                out(f"  {k}: no inject results")
            if I.get("ablate", k):
                self.abl[k] = LN.Ablate(I.get("ablate", k))
            if I.get("bind", k):
                b = self.bind[k] = LN.Bind(I.get("bind", k))
                accs = {v: b.acc(v) for v in ("direct", "other_agent", "irrelevant_object")}
                self.G5[k] = bool(accs["other_agent"] >= 0.8 and accs["irrelevant_object"] >= 0.8)
                out(f"  J-D-G5 {k:8s} accuracy other_agent {accs['other_agent']:.2f}, irrelevant_object {accs['irrelevant_object']:.2f} "
                    f"(>= 0.80; direct {accs['direct']:.2f}) -> {V(self.G5[k])}")
            S = I.get("sign", k)
            if S:
                for arm, recs in S["results"].items():
                    if not recs:
                        continue
                    c = self.cell[(k, arm)] = LN.Cell(recs, arm)
                    if arm == "P1":
                        continue
                    ok, txt = c.competence()
                    self.G3[(k, arm)] = ok
                    out(f"  J-D-G3 {k:8s} {arm:11s} {txt} -> {V(ok)}")
                    if "rows" in recs[0]:
                        fl = c.floor()
                        self.G2[(k, arm)] = fl <= FLOOR
                        out(f"  J-D-G2 {k:8s} {arm:11s} batch floor {fl:.4f} -> {V(fl <= FLOOR)}")
                    t = c.transfer()
                    dG, idT = t["dG"], t["idT"]
                    g4 = bool(abs(dG.pt) >= 1 and (dG.lo() > 0 or dG.hi() < 0) and np.sign(dG.pt) == np.sign(idT.pt)
                              and abs(dG.pt) / max(abs(idT.pt), 1e-12) >= 0.5)
                    self.G4[(k, arm)] = g4
                    out(f"  J-D-G4 {k:8s} {arm:11s} d_G {dG.txt()}, ID_K(allT) {idT.txt()}, d_G/ID_K {dG.pt / idT.pt if idT.pt else float('nan'):+.3f} "
                        f"(|d_G| >= 1, CI excl. 0, same sign, >= 0.5) -> {V(g4)}")
        for k in DISS:
            J = I.get("diss", k)
            if not J:
                continue
            d = self.diss[k] = LN.Diss(J)
            ncomp = len(d.ci)
            fl = d.floor()
            self.G2[(k, "diss")] = fl <= FLOOR
            parts, ok = [f"competent cores {ncomp} of {d.n} (>= 40)"], ncomp >= 40 or I.test
            if ncomp >= 2:
                inj = d.rho()["inj"]["P1"]
                parts.append(f"P1 injection dl_X {inj.txt()} (> 0, CI excl. 0)")
                ok &= bool(inj.pt > 0 and inj.lo() > 0)
            else:
                ok = False
            if k in SMALL:
                R = d.gate_R()
                parts.append(f"R(k*) of the in-run H* {R.txt() if R is not None else 'missing'} (>= 0.6)")
                ok &= bool(R is not None and R.pt >= 0.6)
            self.G6[k] = ok
            out(f"  J-D-G6 {k:8s} " + "; ".join(parts) + f"; batch floor {fl:.4f} -> {V(ok)}")

    def base(self, k):
        """G0 and G1 and the inject floor: the D1-D4 family in model k."""
        if self.g0 is not True or k not in self.inj:
            return False
        return bool(self.G1.get(k)) and bool(self.G2.get((k, "inject")))


def denom_ok(q, need):
    return bool(abs(q.pt) >= need and (q.lo() > 0 or q.hi() < 0))


# --------------------------------------------------------------------------- the lines
def lines_all(G: Gates, I: Inputs):
    R = {}
    T = lambda code: Tests(code, COMPONENTS)  # noqa: E731

    def run(code, fn, models):
        per, det = {}, []
        for k in models:
            n0 = len(COMPONENTS)
            try:
                v, txt = fn(k)
            except Exception:  # noqa: BLE001
                v, txt = None, "scorer error:\n" + traceback.format_exc()
            if v is None:          # the Holm family holds the components of the models where the line is evaluable
                del COMPONENTS[n0:]
            per[k] = v
            det.append(f"    {k:8s} {V(v):13s} {txt}")
        R[code] = (combine(per, NEED[code]), per, det)

    def d1(k):
        if not G.base(k):
            return None, "gates (J-D-G0, J-D-G1, J-D-G2) not passed or no results"
        x = G.inj[k].d1()
        t = T("J-D1")
        a = t.lower_gt(x["ratio"], 0.35, f"{k} ID_inj^LOO/ID_K")
        ok = bool(x["ratio"].pt >= 0.5 and a and x["pi_x"] >= 0.5)
        return ok, f"ID_inj^LOO/ID_K {x['ratio'].txt()} (>= 0.5; H0: <= 0.35), pi_X(move, LOO) {x['pi_x']:.2f} (>= 0.5); ID_K {x['idk'].txt()}"
    run("J-D1", d1, BIG)

    def d2(k):
        if not G.base(k):
            return None, "gates not passed or no results"
        x = G.inj[k].d2()
        t, ok, parts = T("J-D2"), True, [f"iota(move) {x['move'].txt()}"]
        for c, s in x["controls"].items():
            a = t.lower_gt(s["diff"], 0.0, f"{k} iota(flag)-iota({c})")
            ok &= bool(s["diff"].pt >= 0.4 and a and s["frac"].pt <= 0.3)
            parts.append(f"{c}: iota {s['iota'].txt()}, difference {s['diff'].txt()} (>= 0.4; H0: <= 0), ratio {f3(s['frac'].pt)} (<= 0.3)")
        return ok, "; ".join(parts)
    run("J-D2", d2, BIG)

    def d3(k):
        if not G.base(k) or k not in G.abl:
            return None, "gates not passed or no ablation results"
        a = G.abl[k]
        base = a.b.mean(a.v("none", "idK"))
        if not (base.pt >= 1 and base.lo() > 0):
            return None, f"ID_K(none) {base.txt()} < 1 nat (J-D-G7)"
        rf, rp, rm = a.rho("flag"), a.rho("pc1"), a.rho("meanH")
        t = T("J-D3")
        a = t.upper_lt(rf, 0.6, f"{k} rho_K(flag)")
        ok = bool(rf.pt <= 0.5 and a and rp.pt >= 0.7 and rm.pt >= 0.7)
        return ok, f"rho_K flag {rf.txt()} (<= 0.5; H0: >= 0.6), pc1 {rp.txt()} (>= 0.7), mean-H* output {rm.txt()} (>= 0.7)"
    run("J-D3", d3, BIG)

    def d4(k):
        if not G.base(k):
            return None, "gates not passed or no results"
        x = G.inj[k].d4()
        if not denom_ok(x["denom"], 1.0):
            return None, f"route denominator dl_X(move) {x['denom'].txt()} (J-D-G7: >= 1 nat, CI excl. 0)"
        t = T("J-D4")
        a = t.lower_gt(x["r"]["KV"], 0.45, f"{k} r_ans(KV)")
        b = t.lower_gt(x["KmV"], 0.0, f"{k} r_ans(K)-r_ans(V)")
        ok = bool(x["r"]["KV"].pt >= 0.6 and a and b)
        return ok, (f"r_ans^inj(KV) {x['r']['KV'].txt()} (>= 0.6; H0: <= 0.45), K {x['r']['K'].txt()}, V {x['r']['V'].txt()}, "
                    f"K - V {x['KmV'].txt()} (H0: <= 0); denominator {x['denom'].txt()}")
    run("J-D4", d4, BIG)

    def addr(k):
        if not G.base(k) or not I.get("fit", k):
            return None, "gates not passed or no results"
        wc = I.get("fit", k)["summary"]["wcos_K_KV"]
        r = G.inj[k].addr()
        t = T("J-D-ADDR")
        a = t.inside(r, 0.7, 1.4, f"{k} ID_inj(KV)/ID_inj(K)")
        ok = bool(wc >= 0.9 and 0.8 <= r.pt <= 1.25 and a)
        return ok, f"weighted cos(Delta^K, Delta^KV) {wc:+.3f} (>= 0.9); ID_inj(KV)/ID_inj(K) {r.txt()} (in [0.8, 1.25]; 95 % CI inside [0.7, 1.4])"
    run("J-D-ADDR", addr, BIG)

    def kn(k):
        if not G.base(k):
            return None, "gates not passed or no results"
        x = G.inj[k].kn()
        tol = 0.1 * abs(x["idk"].pt)
        t = T("J-D-KN")
        a = t.inside(x["spec"], -tol, tol, f"{k} mean(dl_S - dl_X | K_N)")
        ok = bool(x["r"].pt >= 0.8 and 0.7 <= x["beta"].pt <= 1.3 and a)
        return ok, (f"mean per-story Pearson {x['r'].txt()} (>= 0.8; {x['r_nan']} undefined), B-loss ratio {x['beta'].txt()} (in [0.7, 1.3]), "
                    f"mean(dl_S - dl_X) {x['spec'].txt()} (95 % CI inside +-{tol:.3f} = 0.1 x ID_K)")
    run("J-D-KN", kn, BIG)

    def d5(k):
        if G.g0 is not True or k not in G.bind:
            return None, "J-D-G0 not passed or no binding results"
        if not G.G5.get(k):
            return None, "J-D-G5 (query accuracy) not passed"
        x = G.bind[k].psi()
        t = T("J-D5")
        a = t.lower_gt(x["pooled"], 0.0, f"{k} Psi_bind")
        lo, hi = x["diff"].ci()
        incl = bool(np.isfinite(lo) and lo <= 0 <= hi)
        return bool(a and incl), (f"Psi_bind pooled {x['pooled'].txt()} (H0: <= 0), object-first {x['obj_first'].txt()} (n {x['n_obj']}), "
                                  f"distractor-first {x['dis_first'].txt()} (n {x['n_dis']}), stratum difference {x['diff'].txt()} (CI must include 0)")
    run("J-D5", d5, BIG)

    def cells_ok(k, arms, floor=True):
        for arm in arms:
            if (k, arm) not in G.cell:
                return f"no {arm} results"
            if not G.G3.get((k, arm)):
                return f"J-D-G3 {arm} not passed"
            if floor and not G.G2.get((k, arm), True):
                return f"J-D-G2 {arm} not passed"
        return None

    def d7(k):
        why = "J-D-G0 not passed" if G.g0 is not True else cells_ok(k, ("Q_IN", "Q_OUT"))
        if why:
            return None, why
        ki, _ = G.cell[(k, "Q_IN")].ids()
        ko, _ = G.cell[(k, "Q_OUT")].ids()
        t = T("J-D7")
        a = t.upper_lt(ko, 0.0, f"{k} ID_K(Q_OUT)")
        b = t.lower_gt(ki, 0.0, f"{k} ID_K(Q_IN)")
        c = t.lower_gt(ki - ko, 0.0, f"{k} ID_K(Q_IN)-ID_K(Q_OUT)")
        ok = bool(ko.pt <= -1 and a and ki.pt >= 1 and b and c)
        return ok, f"ID_K(Q_OUT) {ko.txt()} (<= -1; H0: >= 0), ID_K(Q_IN) {ki.txt()} (>= 1; H0: <= 0), paired difference {(ki - ko).txt()} (H0: <= 0)"
    run("J-D7", d7, BIG)

    def signq(k):
        why = "J-D-G0 not passed" if G.g0 is not True else cells_ok(k, ("Q_OUT",))
        if why:
            return None, why
        ik, iv = G.cell[(k, "Q_OUT")].ids()
        t = T("J-D-SIGN-Q")
        a = t.upper_lt(ik, 0.0, f"{k} ID_K(Q_OUT)")
        b = t.upper_lt(iv, 0.0, f"{k} ID_V(Q_OUT)")
        ok = bool(a and b)
        return ok, f"Q_OUT ID_K {ik.txt()} (H0: >= 0), ID_V {iv.txt()} (H0: >= 0)"
    run("J-D-SIGN-Q", signq, BIG)

    def signioi(k):
        why = "J-D-G0 not passed" if G.g0 is not True else cells_ok(k, ("INLINE",))
        if why:
            return None, why
        ik, iv = G.cell[(k, "INLINE")].ids()
        t = T("J-D-SIGN-IOI")
        a = t.upper_lt(ik, 0.0, f"{k} ID_K(INLINE)")
        b = t.lower_gt(iv, 0.0, f"{k} ID_V(INLINE)")
        ok = bool(a and b)
        return ok, f"INLINE ID_K {ik.txt()} (H0: >= 0), ID_V {iv.txt()} (H0: <= 0)"
    run("J-D-SIGN-IOI", signioi, BIG)

    def d8cell(k, arm, code):
        why = "J-D-G0 not passed" if G.g0 is not True else cells_ok(k, (arm,), floor=False)
        if why:
            return None, why
        if not G.G4.get((k, arm)):
            return None, f"J-D-G4 {arm} not passed"
        x = G.cell[(k, arm)].transfer()
        t = T(code)
        a = t.lower_gt(x["R"], 0.45, f"{k} {arm} R(H*)")
        b = t.lower_gt(x["KO"], 0.45, f"{k} {arm} KO(H*)")
        ok = bool(x["R"].pt >= 0.6 and a and x["KO"].pt >= 0.6 and b and x["Rrand"].pt <= 0.15 and x["KOrand"].pt <= 0.15)
        return ok, (f"{arm}: R(H*) {x['R'].txt()} (>= 0.6; H0: <= 0.45), KO(H*) {x['KO'].txt()} (>= 0.6; H0: <= 0.45), "
                    f"random R {f3(x['Rrand'].pt)} KO {f3(x['KOrand'].pt)} (<= 0.15); d_Gc/ID_K {f3(x['dGc'].pt)}")

    def d8q(k):
        res = [d8cell(k, arm, "J-D8-Q") for arm in ("Q_IN", "Q_OUT")]
        vs = [v for v, _ in res]
        v = False if False in vs else (True if all(x is True for x in vs) else None)
        return v, " | ".join(t for _, t in res)
    run("J-D8-Q", d8q, BIG)
    run("J-D8-IOI", lambda k: d8cell(k, "INLINE", "J-D8-IOI"), BIG)

    def route(k):
        why = "J-D-G0 not passed" if G.g0 is not True else cells_ok(k, ("INLINE",))
        if why:
            return None, why
        x = G.cell[(k, "INLINE")].route()["K_S"]
        if not denom_ok(x["denom"], 1.0):
            return None, f"denominator dm(K_S) {x['denom'].txt()} (J-D-G7: |mean| >= 1 nat, CI excl. 0)"
        t = T("J-D-ROUTE-IOI")
        a = t.lower_gt(x["KmV"], 0.0, f"{k} r_ans(K)-r_ans(V) IOI")
        ok = bool(x["r"]["KV"].pt >= 0.6 and a)
        return ok, (f"K_S rows: r_ans(KV) {x['r']['KV'].txt()} (>= 0.6), K {x['r']['K'].txt()}, V {x['r']['V'].txt()}, K - V {x['KmV'].txt()} "
                    f"(H0: <= 0); r_other {x['r']['other'].txt()} (>= 0.5 would mean S2 or tail rows carry it); dm(K_S) {x['denom'].txt()}")
    run("J-D-ROUTE-IOI", route, BIG)

    def hop2(k):
        why = "J-D-G0 not passed" if G.g0 is not True else cells_ok(k, ("Q_OUT", "INLINE"))
        if why:
            return None, why
        t, parts, ok, diff = T("J-D-HOP2"), [], True, True
        X = {arm: G.cell[(k, arm)].hop2() for arm in ("Q_OUT", "INLINE")}
        for arm, x in X.items():
            if not denom_ok(x["dinj"], 0.5):
                return None, f"{arm}: injection effect {x['dinj'].txt()} (J-D-G7: |mean| >= 0.5 nat, CI excl. 0)"
        for arm, x in X.items():
            c = x["carry"]["top10"]
            a = t.lower_gt(c, 0.2, f"{k} {arm} carry(top10)")
            ok &= bool(c.pt >= 0.5 and a)
            diff &= bool(c.pt <= 0.2 and np.isfinite(c.hi()) and c.hi() < 0.5)
            parts.append(f"{arm}: carry(top-10) {c.txt()} (>= 0.5; H0: <= 0.2), all heads {f3(x['carry']['all'].pt)}, "
                         f"random 10 {f3(x['carry']['rand10'].pt)}; dl_X(inj) {x['dinj'].txt()}")
        reading = "same reader, opposite sign" if ok else "different reader" if diff else "intermediate"
        return bool(ok), f"[{reading}] " + "; ".join(parts)
    run("J-D-HOP2", hop2, BIG)

    def d6a(k):
        if G.g0 is not True or k not in G.diss or not I.get("fit", k):
            return None, "J-D-G0 not passed or no results"
        if not G.G2.get((k, "diss"), False):
            return None, "J-D-G2 (diss batch floor) not passed"
        om = G.diss[k].omega()
        kap = I.get("fit", k)["summary"]["kappa_POST_P1"]
        t = T("J-D6a")
        a = t.lower_gt(om, 0.4, f"{k} omega")
        ok = bool(om.pt >= 0.6 and a and kap >= 0.7)
        return ok, f"omega {om.txt()} (>= 0.6; H0: <= 0.4), kappa {kap:+.3f} (>= 0.7)"
    run("J-D6a", d6a, DISS)

    RHO = {}

    def d6(k):
        if G.g0 is not True or k not in G.diss:
            return None, "J-D-G0 not passed or no results"
        if not G.G6.get(k) or not G.G2.get((k, "diss"), False):
            return None, "J-D-G6 or J-D-G2 not passed"
        x = G.diss[k].rho()
        RHO[k] = x
        tol = max(0.1, 0.5 * abs(x["rho_nat"].pt))
        t = T("J-D6")
        same = np.sign(x["rho"].pt) == np.sign(x["rho_nat"].pt)
        a = t.inside(x["d"], -tol, tol, f"{k} rho - rho_nat")
        ok = bool(same and a)
        return ok, (f"rho {x['rho'].txt()}, rho_nat {x['rho_nat'].txt()} (same sign: {bool(same)}), rho - rho_nat {x['d'].txt()} "
                    f"(95 % CI inside +-{tol:.3f}); dl_X(inj) P1 {x['inj']['P1'].txt()}, POST {x['inj']['POST'].txt()}; "
                    f"N_X P1 {x['NX']['P1'].txt()}, POST {x['NX']['POST'].txt()}; {len(G.diss[k].ci)} competent cores")
    run("J-D6", d6, DISS)

    def d6r(k):
        if G.g0 is not True or k not in G.diss:
            return None, "J-D-G0 not passed or no results"
        if not G.G6.get(k):
            return None, "J-D-G6 not passed"
        x = G.diss[k].route()
        if not denom_ok(x["denom"], 0.2):
            return None, f"denominator dl_X(inj, POST) {x['denom'].txt()} (J-D-G7: |mean| >= 0.2 nat, CI excl. 0)"
        t = T("J-D6-ROUTE")
        ok = t.lower_gt(x["VmK"], 0.0, f"{k} r_ans(V)-r_ans(K) POST")
        return ok, (f"POST: r_ans^inj(V) {x['r']['V'].txt()}, K {x['r']['K'].txt()}, V - K {x['VmK'].txt()} (H0: <= 0); KV {x['r']['KV'].txt()}; "
                    f"dl_X(inj) {x['denom'].txt()}")
    run("J-D6-ROUTE", d6r, ("qwen1.5",))
    return R, RHO


def d6_table(R, RHO):
    """The D6 decision table (fixed in the entry)."""
    v6, vr = R["J-D6"][0], R["J-D6-ROUTE"][0]
    notmed = [k for k, x in RHO.items() if x["rho_nat"].hi() < 0 and abs(x["rho"].pt) <= 0.1 and x["rho"].lo() > -0.2 and x["rho"].hi() < 0.2]
    if v6 is True and vr is True:
        return "(a) MEDIATED, READ BY VALUE AT 1.5B: the sentence-row flag carries the natural read ratio at every scale, and at 1.5B the answer reads it by value"
    if v6 is True:
        return "(b) MEDIATED: the sentence-row flag carries the natural read ratio at every scale; the 1.5B route is not shown to be by value"
    if notmed:
        return f"(c) NOT MEDIATED in {notmed}: the natural negative read (rho_nat < 0) is not carried by the sentence-row flag (rho ~ 0)"
    return "(d) MIXED: none of (a)-(c); reported as such"


# --------------------------------------------------------------------------- reported and exploratory
def reported(G: Gates, I: Inputs, out):
    out("\nREPORTED (no verdict counted)")
    for k, x in G.inj.items():
        d = x.d1()
        io = d["iota"]
        mono = io["add0.5"].pt < io["add1"].pt < io["add2"].pt
        out(f"  {k}: iota add 0.5 / 1 / 2 {f3(io['add0.5'].pt)} / {f3(io['add1'].pt)} / {f3(io['add2'].pt)} (monotone: {mono}); "
            f"add-only {io['add1'].txt()} (>= 0.25, LB > 0: {io['add1'].pt >= 0.25 and io['add1'].lo() > 0}); move {io['move'].txt()}; "
            f"pi_X(move) {d['pi_move']:.2f}; ID_inj(full flag)/ID_K {d['idinj_full'].txt()}; N_X {d['NX'].txt()}")
        s = x.d2()
        out(f"  {k}: sanity rows: " + ", ".join(f"{n} {f3(q.pt)}" for n, q in s["sanity"].items())
            + f"; pi_X shift choices {s['pi_shift']['choices']:+.2f}, question {s['pi_shift']['question']:+.2f}; rank-1 capture {s['capture'].txt()}")
        r = x.d4()
        out(f"  {k}: D4 r_other {r['r']['other'].txt()}")
        Fj = I.get("fit", k)
        if Fj:
            sm = Fj["summary"]
            sh = sm["identity_share"]
            out(f"  {k}: B-identity share of the per-story flag (null {sh['null']:.2f}): " + ", ".join(f"l{l} {v:.2f}" for l, v in sh["share"].items()))
            out(f"  {k}: logit lens max|cos(Delta, W_U[loc])| vs random: " + ", ".join(f"l{l} {a:.3f}/{b:.3f}" for l, (a, b) in sm["logit_lens"].items()))
    for k, a in G.abl.items():
        out(f"  {k}: rho_K isotropic " + ", ".join(f"{c} {a.rho(c).txt()}" for c in ("iso0", "iso1", "iso2")) + " (sanity >= 0.85)")
    for k, b in G.bind.items():
        out(f"  {k}: event vs initial (role confounded with order) Psi_ev {b.psi_ev().txt()}")
    for (k, arm), c in G.cell.items():
        if arm == "P1":
            out(f"  {k}: hop-2 carry under P1 (the ranking format): " + ", ".join(f"{h} {q.txt()}" for h, q in c.hop2()["carry"].items()))
            continue
        ik, iv = c.ids()
        t = c.transfer()
        line = (f"  {k} {arm}: ID_K {ik.txt()}, ID_V {iv.txt()}; R(H*) {t['R'].txt()}, KO(H*) {t['KO'].txt()}, d_G/ID_K {t['dG_over_idT'].txt()}, "
                f"d_Gc/ID_K {t['dGc'].txt()} (a re-mention read only if >= 0.5)")
        out(line)
        if arm in ("Q_IN", "Q_OUT") and "rows" in c.S[0]:
            bi = np.array([s["ix"]["B"] for s in c.S])
            dB = c.b.mean(LN.pick(c.scores("rows", "K_S"), bi) - LN.pick(c.scores("rows", "none"), bi))
            out(f"     behavioural correlate: dl_B under K_S {dB.txt()} (Q_OUT: > 0 expected, the written word becomes a 'not mentioned' answer)")
            dinj, nx = c.b.mean(c.dlx("rows", "inj")), c.b.mean(c.dlx("rows", "K_X"))
            out(f"     D9 rows: dl_X(+Delta^P1 at r_X) {dinj.txt()}, N_X(cfg) {nx.txt()}, ratio {f3(dinj.pt / nx.pt if nx.pt else float('nan'))}; "
                f"isotropic " + ", ".join(f3(c.b.mean(c.dlx('rows', f'iso{i}')).pt) for i in range(3)))
        if arm == "INLINE":
            r = c.route()["inj"]
            out(f"     route of the injected IOI flag: r_ans(KV) {r['r']['KV'].txt()}, K {r['r']['K'].txt()}, V {r['r']['V'].txt()}, "
                f"other {r['r']['other'].txt()}; dm(inj) {r['denom'].txt()}")
    for k, d in G.diss.items():
        if len(d.ci) >= 2:
            x = d.rho()
            out(f"  {k}: D6b Delta^POST at the P1 list row / Delta^P1 there {x['d6b'].txt()} (>= 0.6 expected)")


def exploratory(G: Gates, I: Inputs, out):
    out("\n######## EXPLORATORY (not scored)")
    for k in KEYS:
        S = I.get("sets", k)
        if S:
            out(f"-- {k}: sets ({S['provenance'].get('source')}): k* {S['sets']['kstar']}, L* {S['layers']}; hop-2 top-10 {S['sets']['hop_top']}; "
                f"in-run a3 overlap with H* {S['a3_inrun_overlap_H']}")
        Fj = I.get("fit", k)
        if Fj:
            sm = Fj["summary"]
            geo = sm["geometry"]
            cc = np.array(geo["cross_cos"])
            off = (cc.sum() - np.trace(cc)) / max(cc.size - len(cc), 1)
            out(f"-- {k}: flag geometry: consistency " + ", ".join(f"l{l} {v:.2f}" for l, v in geo["c"].items())
                + f"; mean cross-layer cos {off:+.3f}; flag cosines " + "; ".join(f"{p} {np.mean(list(v.values())):+.2f}" for p, v in sm["cos"].items()))
    for k, x in G.inj.items():
        out(f"-- {k}: E6, iota on the raw margin m_X = lp(X) - lp(B): " + ", ".join(f"{n} {x.iota_m(n).txt()}" for n in ("add1", "move", "looX")))
    for k, a in G.abl.items():
        conds = [c for c in a.conds if "@" in c] + ["none", "flag"]
        out(f"-- {k}: H3/G7b reconciliation (base answer rate, initial-location rate): "
            + ", ".join(f"{c} {a.rates(c)[0]:.2f}/{a.rates(c)[1]:.2f}" for c in conds))
    for (k, arm), c in G.cell.items():
        if arm == "Q_OUT" and "rows" in c.S[0]:
            out(f"-- {k}: Q_OUT behaviour, answer = the story's own word B: " + ", ".join(f"{n} {v:.2f}" for n, v in c.behaviour().items()))
        if arm == "INLINE" and "rows" in c.S[0]:
            out(f"-- {k}: IOI injections at the listed IO_X row (dl, four-way): " + ", ".join(
                f"{n} {c.b.mean(c.dlx('rows', n)).txt()}" for n in ("injIOI", "moveIOI", "injP1", "iso", "K_X")))
    B = I.get("before", "qwen7")
    if B and "knockout" in B:
        ko = B["knockout"]
        base = np.mean([x["ko"]["none"]["idK"] for x in ko])
        for v in ("names", "list"):
            f = 1 - np.mean([x["ko"][v]["idK"] for x in ko]) / base if base else float("nan")
            read = "p-as-re-mention" if f >= 0.5 else "second-order read elsewhere" if f <= 0.2 else "intermediate"
            out(f"-- qwen7: IOI BEFORE knockout of p -> {v} rows: ID_K {base:+.3f} -> fraction removed {f:+.3f} ({read}; two-sided rule)")
        rs = B.get("rowsplice") or []
        if rs:
            full = np.mean([r["m"]["all"] - r["m_B"] for r in rs])
            out("-- qwen7: IOI BEFORE RowSplice fractions of the full key effect: " + ", ".join(
                f"{g} {np.mean([r['m'][g] - r['m_B'] for r in rs]) / full:+.2f}" for g in rs[0]["m"] if g not in ("all", "none")))
    for k in BIG:
        X = I.get("xtask", k)
        if not X:
            continue
        for t, v in X["tasks"].items():
            if "stories" not in v:
                out(f"-- {k}: cross-task {t}: {v}")
                continue
            S = v["stories"]
            xi = np.array([s["ix"]["X"] for s in S])

            def dl(n):
                return LN.pick(LN.renorm(np.array([s["rows"][n] for s in S])), xi) - LN.pick(LN.renorm(np.array([s["rows"]["none"] for s in S])), xi)
            own, bel = dl("own").mean(), dl("belief").mean()
            rr = bel / own if own else float("nan")
            read = "shared pointer" if rr >= 0.6 else "task-specific" if rr <= 0.3 else "intermediate"
            out(f"-- {k}: cross-task {t}: own flag dl_X {own:+.3f}, belief flag {bel:+.3f}, ratio {rr:+.3f} ({read}); "
                f"weighted cos {v['wcos_task_belief']:+.3f}")


def holm_sensitivity(R, out):
    """Decision D2 (common part): Holm's step-down at familywise one-sided 0.025 over the interval components of this
    part's R-class account lines that have a verdict (the components of the models where the line is evaluable), by the
    shared helper analysis/stage8_holm.py (one-sided p from the bootstrap SE: Phi(-(est - bound)/se) for '>',
    Phi((est - bound)/se) for '<'). Per line: the components whose decision changes, and the verdict with Holm's
    decisions in place of the interval decisions (a MET line with a component no longer rejected becomes NOT MET; a
    component Holm rejects but the interval rule does not is listed and changes no verdict). Reported; no verdict uses
    it. Returns {code: (a decision changed, the verdict changed)}."""
    fam = [(c, d, own) for c, d, own in COMPONENTS if LINES.get(c, ("",))[0] == "R" and R.get(c, (None,))[0] is not None]
    bad = [d["name"] for _, d, _ in fam if not (np.isfinite(d["est"]) and np.isfinite(d["se"]))]
    fam = [x for x in fam if np.isfinite(x[1]["est"]) and np.isfinite(x[1]["se"])]
    try:
        from stage8_holm import holm as holm_shared   # analysis/stage8_holm.py, identical in every part (D2)
        keys = [(d["line"], d["name"]) for _, d, _ in fam]
        assert len(set(keys)) == len(keys), "component names must be unique"
        res_ = holm_shared([dict(d) for _, d, _ in fam]) if fam else []
        by = {(g["line"], g["name"]): g for g in res_}
        got = [by[k_] for k_ in keys]          # matched by (line, name), whatever order the helper returns
    except Exception as ex:  # noqa: BLE001  (the sensitivity analysis never stops the report)
        out(f"  Holm sensitivity NOT COMPUTED: analysis/stage8_holm.py failed ({type(ex).__name__}: {ex})")
        return {}
    out(f"  Holm sensitivity (analysis/stage8_holm.py; reported, no verdict uses it): step-down at familywise one-sided 0.025 "
        f"over the {len(fam)} interval components of the R-class account lines with a verdict (normal p from the bootstrap "
        f"SE)" + (f"; {len(bad)} undefined components left out" if bad else ""))
    res = {}
    for code, (cls, *_rest) in LINES.items():
        if cls != "R":
            continue
        v = R.get(code, (None,))[0]
        rows = [(d, own, h) for (c, d, own), h in zip(fam, got) if c == code]
        flips = [f"{d['name']} (p {h['p']:.2g}; interval rule {'rejects' if own else 'does not reject'}, Holm "
                 f"{'rejects' if h['reject'] else 'does not reject'})" for d, own, h in rows if bool(h["reject"]) != own]
        lost = any(own and not bool(h["reject"]) for d, own, h in rows)
        nv = False if (v is True and lost) else v
        res[code] = (bool(flips), nv != v)
        out(f"    {code}: {len(rows)} components; " + (f"{len(flips)} decision(s) change under Holm: " + "; ".join(flips[:12])
                                                       + (" ..." if len(flips) > 12 else "") if flips else "no decision changes under Holm")
            + (f"; verdict {V(v)} -> {V(nv)} under Holm" if nv != v else f"; verdict unchanged ({V(v)})"))
    return res


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gpu_stage8d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true", help="TEST_ tags: sizes not checked, verdicts are plumbing checks")
    a = ap.parse_args(argv)
    root = Path(a.results)
    COMPONENTS.clear()
    lines, errors = [], []
    out = lines.append
    out(f"Stage 8 part D scoring, preregistration P-2026-10-10-J; results {root}" + ("; TEST MODE (verdicts are plumbing checks)" if a.test else ""))
    I = Inputs(root, a.test)
    out("")
    prov_ok = provenance(I, out)
    out("")
    pop_ok = population(I, out)
    out("")
    g0 = gate_g0(root, out)
    R, RHO, G = {}, {}, None
    try:
        G = Gates(I, out, g0)
        R, RHO = lines_all(G, I)
    except Exception:  # noqa: BLE001
        errors.append("gates/lines")
        out("  SCORER ERROR; every line NOT EVALUABLE\n" + traceback.format_exc())
    out("\nPREDICTIONS (code, class, kind, prior, verdict; the numbers it was decided on; 2/2 lines need two evaluable models)")
    for code, (cls, kind, prior, models, title) in LINES.items():
        v, per, det = R.get(code, (None, {}, ["    not computed"]))
        out(f"  {code:14s} {cls} {kind} prior {prior:.2f}  {V(v):13s} {title}")
        for x in det:
            out(x)
    if R:
        out("\nD6 DECISION TABLE (fixed in the entry): " + d6_table(R, RHO))
    try:
        if G is not None:
            reported(G, I, out)
    except Exception:  # noqa: BLE001
        out("  reported lines failed:\n" + traceback.format_exc())
    out("\nSUMMARY")
    rl = []
    for cls in ("L", "M", "R"):
        codes = [c for c, x in LINES.items() if x[0] == cls]
        vs = [R.get(c, (None,))[0] for c in codes]
        ev = [(c, v) for c, v in zip(codes, vs) if v is not None]
        exp = sum(LINES[c][2] for c, _ in ev)
        met = sum(bool(v) for _, v in ev)
        brier = np.mean([(LINES[c][2] - bool(v)) ** 2 for c, v in ev]) if ev else float("nan")
        ne = [c for c, v in zip(codes, vs) if v is None]
        if cls == "R":
            rl = ev
        out(f"  account lines, class {cls}: {len(codes)} lines: {met} MET, {len(ev) - met} NOT MET, 0 MET IN PART, "
            f"{len(ne)} NOT EVALUABLE; observed {met} against expected {exp:.2f}; Brier {brier:.3f}"
            + (f"; not evaluable: {', '.join(ne)}" if ne else ""))
    out("  measurement-validity lines: none in part D (the batch floors and competence checks are gates)")
    out(f"  met rate among R account lines with a verdict: {sum(bool(v) for _, v in rl)} of {len(rl)} "
        f"(expected {sum(LINES[c][2] for c, _ in rl):.2f})")
    holm_sensitivity(R, out)
    out(f"  provenance {'OK' if prov_ok else 'MISMATCH'}; population {'OK' if pop_ok else 'MISMATCH'}; J-D-G0 {V(g0)}"
        + (f"; SCORER ERROR in {errors}" if errors else ""))
    try:
        if G is not None:
            exploratory(G, I, out)
    except Exception:  # noqa: BLE001
        out("  exploratory report failed:\n" + traceback.format_exc())
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE8D_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    has_results = any(v is not None for (st, _), v in I.F.items() if st != "preflight")
    return 1 if errors else 2 if not a.test and not (prov_ok and pop_ok and (g0 or not has_results)) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
