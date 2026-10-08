"""analysis/stage7_score.py on synthetic stage-7 roots at the preregistered n (96 E cores, 60 R cores, 600 family
results): the all-MET scenario; the kappa rule on every kappa and the dropped-resample rule; an undefined kappa after
blocking is NOT MET with the reason printed (not NOT EVALUABLE); the evaluability split (I1 evaluable without I-G2 and
without I-G3 clause 2; I4/I5 NOT EVALUABLE when NO-MENTION fails I-G1 (b) or kappa(NO-MENTION) is undefined); I5 on a
file where only the denominator falls (NOT MET); I-G1 (d) at 0.1 nats (0.09 passes, 0.11 fails); the "both formats"
combination and OPTIONS-AFTER-only I5/I6; threshold boundaries (equal counts as met where the entry says >=); a
provenance mismatch exits 2; Gate I-G0 from the last pytest run; a TEST_ tag skips the size checks; I-G1 (c) with the
kappa waiver; I3 and I7 NOT MET; the remap-ranking report and its deadline stub; population, release, frames and dtype
mismatches (exit 2) and a scorer error (exit 1); I-G0 with a skipped test or too few tests; partial exploratory batches;
unblocked-kappa resample drops (I4 NOT EVALUABLE); SENTENCE-AFTER reported without a verdict."""
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ckeys.story import LOCATIONS, PAIR_SWAP, make_cores

ROOT = Path(__file__).resolve().parents[1]
REV = "9527884be6e5616bdd54de542f9ae13384489724"
RELEASE = "2dea297d508e51f07b927e9eb0571f40e99d25d996ccd7ad3342cb46942d0416"
SEEDS = (101, 102, 103)
ARMS = ("P1", "LETTER", "POST", "NONE", "BEFORE")
I = [f"I{i}" for i in range(1, 8)]
KS, KSTAR, NL, H = [8, 16, 32, 64, 128], 64, 40, 32


def pins():
    sys.path.insert(0, str(ROOT))
    from experiments.stage7_link import BASES_SHA, NATIVE_SHA
    return BASES_SHA, NATIVE_SHA


# unblocked family means (D, K, V) per format, the stage-3b values
BASE = {"P1": (13.7, 7.24, 3.14), "LETTER": (12.87, 10.0, 0.23), "POST": (12.17, 2.38, 6.36), "NONE": (14.34, 0.99, 11.51), "BEFORE": (12.13, -0.02, 11.78)}


def scenario():
    """Per format and condition (D, K, V); B_x: a(all) = K, g_K, KO_x(H*), KO_x of the controls; gate: d_full, ratio, R, KO."""
    S = {}
    for arm, (D, K, V) in BASE.items():
        c = {"0": (D, K, V), "N:null": (D, K, V)}
        if arm in ("P1", "LETTER", "POST"):
            c |= {f"A:{s}": (D, K, V) for s in ("rand0", "rand1", "rand2", "active")}
            c |= {"A:H": (0.88 * D, 0.1 * K, V + 0.45 * K), "A+:H": (0.85 * D, 0.08 * K, V + 0.45 * K), "N:H": (0.6 * D, 0.2 * K, V),
                  "A:L4": (D, K, V), "N:rand0": (D, K, V), "N:allG": (0.5 * D, 0.1 * K, V)}
            if arm == "LETTER":
                c["A:H_P1"] = (D, 0.5 * K, V)
        if arm == "BEFORE":
            c["N:H_P1"] = (D, K, V)
        S[arm] = dict(fam=c, gK=0.85, KO=0.92, KOc=0.05, before_gap=0.0, frames_scale=1.0,
                      gate=dict(dfull={"P1": 17.0, "LETTER": 18.0, "POST": 8.0}.get(arm, 2.0), ratio=0.9, R=0.95, KO=0.97, Rr=0.01, KOr=0.01))
    return S


def cand(m, iS, iT, rng, noise=0.0):
    c = [-8.0] * 6
    c[iS] = 0.0
    c[iT] = m + (rng.normal(0, noise) if noise else 0.0)
    return c


def run(m, r, rng, noise):
    c = cand(m, r["iS"], r["iT"], rng, noise)
    return {"cand": c, "argmax": int(np.argmax(c))}


def ecores():
    out, rng = [], random.Random(5)
    for i in range(96):
        b, s = rng.sample(LOCATIONS, 2)
        while PAIR_SWAP[s] == b:
            b, s = rng.sample(LOCATIONS, 2)
        rest = [x for x in LOCATIONS if x not in (b, s, PAIR_SWAP[s])]
        out.append(dict(id=hashlib.sha256(str(i).encode()).hexdigest(), agent="Alice", other="Bob", object=f"widget{i}", distractor="pen",
                        initial=rest[0], distractor_location=(b if i % 3 == 0 else rest[1]), base=b, source=s, target=PAIR_SWAP[s]))
    return out


def write(root, S, noise=0.3, commit="abc123", tag="mistral", seed=0, log=None):
    rng = np.random.default_rng(seed)
    BASES_SHA, NATIVE_SHA = pins()
    root = Path(root)
    for d in ("heads", "link", "frames"):
        (root / d).mkdir(parents=True, exist_ok=True)
    E = ecores()
    prov = lambda attn: {"git_commit": commit, "model": "mistralai/Mistral-Small-24B-Instruct-2501", "revision": REV, "dtype": "torch.bfloat16",  # noqa: E731
                         "attn_implementation": attn, "device": "NVIDIA A100-SXM4-80GB", "transformers": "5.18.0", "torch": "2.x", "test_mode": False,
                         "n_layers": NL, "heads_per_layer": H, "n_heads": NL * H, "n_eligible": (NL - 5) * H, "kstar": KSTAR, "KS": KS}
    rel = {"release_sha256": RELEASE, "bases_sha256": BASES_SHA, "native_sha256": NATIVE_SHA, "bases_used": "released original_1000", "status": "PASS"}
    json.dump({"provenance": prov(None) | {"attn_implementation": None}, "release": rel}, open(root / "preflight.json", "w"))
    elig = [(l, h) for l in range(5, NL) for h in range(H)]
    order = list(np.random.default_rng(1).permutation(len(elig)))
    Hs = {arm: [list(elig[i]) for i in order[j * KSTAR:(j + 1) * KSTAR]] for j, arm in enumerate(("P1", "LETTER", "POST"))}
    sets = {"arms": {a: {"H": Hs[a], "active": Hs[a][::-1], "next": Hs[a]} for a in Hs}, "rand": [[list(elig[i]) for i in order[-KSTAR * (r + 1):len(elig) - KSTAR * r]] for r in range(3)],
            "L4": [[4, h] for h in range(H)], "kstar": KSTAR}
    sh = hashlib.sha256(json.dumps(sets, sort_keys=True).encode()).hexdigest()
    R = make_cores(60, random.Random(0))
    rank = {"provenance": prov("eager"), "sets": sets, "sets_sha256": sh, "mu_file": "mu.pt", "mu_sha256": "f" * 64,
            "dup": {"D": np.zeros((NL, H)).tolist(), "I": np.zeros((NL, H)).tolist()},
            "arms": {a: {"rank": [{"core": c} for c in R], "ranking_l0": [list(c) for c in elig]} for a in Hs}}
    json.dump(rank, open(root / "heads/rank.json", "w"))
    gate = {"provenance": prov("sdpa") | {"sets_sha256": sh, "mu_sha256": "f" * 64}, "arms": {}, "none_clamp": {"eval": []}}
    for arm in ("P1", "LETTER", "POST"):
        g, ev = S[arm]["gate"], []
        for c in E:
            df = g["dfull"] + rng.normal(0, noise)
            dG = g["ratio"] * df
            curves = {"H": dict(suff=[g["R"] * dG * min(1, k / KSTAR) for k in KS], none=0.0, allG=dG, allT=df,
                                ko=[(1 - g["KO"]) * dG * min(1, KSTAR / k) for k in KS], ko_none=0.0, ko_allG=dG)}
            for s in ("rand0", "rand1", "rand2", "active"):
                curves[s] = dict(suff=[g["Rr"] * dG], none=0.0, allG=dG, ko=[(1 - g["KOr"]) * dG], ko_none=0.0, ko_allG=dG)
            ev.append(dict(core=c, mB=0.0, mF=df, curves=curves))
        gate["arms"][arm] = {"eval": ev, "skipped_items": 0}
    gate["none_clamp"]["eval"] = [dict(core=c, mB=0.0, mF=2.0 + rng.normal(0, noise)) for c in E]
    json.dump(gate, open(root / "heads/gate.json", "w"))
    link = {"provenance": prov("sdpa") | {"sets_sha256": sh, "mu_sha256": "f" * 64, "release": rel, "skipped_items": {a: 0 for a in ARMS}}, "arms": {}}
    for arm in ARMS:
        A, recs = S[arm], []
        for c in E:
            iS, iT, iB = LOCATIONS.index(c["source"]), LOCATIONS.index(c["target"]), LOCATIONS.index(c["base"])
            r = dict(core=c, iS=iS, iT=iT, iB=iB, init=LOCATIONS.index(c["initial"]), runs={})
            for cond, (D, K, V) in A["fam"].items():
                if cond == "N:null" and A.get("null_exact", True):
                    continue
                sgn = 1 if len(recs) % 2 == 0 else -1
                sp = A.get("spread", {}).get(cond, 0.0) * sgn                                  # zero-sum spread across cores
                spK = A.get("spreadK", {}).get(cond, 0.0) * sgn
                for s in SEEDS:
                    for n, m in (("P", 0.0), ("M", D + sp), ("PK", K + spK), ("PV", V), ("MK", D - K), ("MV", D - V)):
                        r["runs"][f"{cond}/{n}_{s}"] = run(m, r, rng, noise if n != "P" else 0.0)
                for n, m in (("B", -5.0), ("S", 0.0), ("T", 14.0)):
                    r["runs"][f"{cond}/{n}"] = {"cand": cand(m, iS, iT, rng), "argmax": iB if n == "B" else iT if n == "T" else iS}
            for k in [k for k in r["runs"] if k.startswith("0/") and A.get("null_exact", True)]:
                r["runs"]["N:null/" + k[2:]] = copy.deepcopy(r["runs"][k])     # the null knockout is exact
            if arm in ("P1", "LETTER", "POST"):
                D, K, _ = A["fam"]["0"]
                aa, an = K, (1 - A["gK"]) * K
                vals = {"P": 0.0, "M": D, "x_all": aa, "x_notG": an, "x_H": aa - A["KO"] * (aa - an), "s_H": 0.9 * (aa - an), "s_G": aa - an,
                        "rem_H": D - 0.8 * K} | {f"x_{s}": aa - A["KOc"] * (aa - an) for s in ("rand0", "rand1", "rand2", "active")}
                for s in SEEDS:
                    for n, m in vals.items():
                        r["runs"][f"x/{n}_{s}"] = run(m, r, rng, noise if n != "P" else 0.0)
                    for n, m in {"P": 0.0, "x_all": aa, "x_k8": aa - 0.3 * (aa - an), "x_k128": an}.items():
                        r["runs"][f"curve/{n}_{s}"] = run(m, r, rng, noise if n != "P" else 0.0)
            if arm == "BEFORE":
                for s in SEEDS:
                    r["runs"][f"before/P_{s}"] = run(0.0, r, rng, 0.0)
                    xa = run(-0.02, r, rng, 0.0)
                    r["runs"][f"before/x_all_{s}"] = xa
                    r["runs"][f"before/x_HP1_{s}"] = {"cand": [x + A["before_gap"] for x in xa["cand"]], "argmax": xa["argmax"]}
                for k in list(r["runs"]):
                    if k.startswith("0/"):
                        r["runs"]["N:H_P1/" + k[2:]] = copy.deepcopy(r["runs"][k])
            recs.append(r)
        link["arms"][arm] = recs
    json.dump(link, open(root / f"link/{tag}.json", "w"))
    fr = []
    for arm in ARMS:
        D, K, V = S[arm]["fam"]["0"]
        D = D / S[arm]["frames_scale"]
        for i, c in enumerate(E + [dict(e, id="x" + e["id"], target=e["base"]) for e in E[:24]]):
            iS, iT = LOCATIONS.index(c["source"]), LOCATIONS.index(c["target"])
            r = dict(core=c, iS=iS, iT=iT)
            runs = {"B": run(-5, r, rng, 0), "S": run(0, r, rng, 0), "T": run(D / 0.7, r, rng, 0)}
            for s in SEEDS:
                runs |= {f"m3_{s}": run(D, r, rng, 0), f"pca_{s}": run(0, r, rng, 0), f"addition_{s}": run(K, r, rng, 0), f"removal_{s}": run(D - K, r, rng, 0),
                         f"addition_v_{s}": run(V, r, rng, 0), f"removal_v_{s}": run(D - V, r, rng, 0)}
            fr.append(dict(core=c, arm=arm, runs=runs, core_index=i))
    bprov = {k: {"sha256": v} for k, v in BASES_SHA.items()}
    json.dump({"provenance": {"repo": "mistralai/Mistral-Small-24B-Instruct-2501", "revision": REV, "bases": bprov, "transformers": "5.18.0"}, "results": fr},
              open(root / f"frames/{tag}.json", "w"))
    (root / "ref.json").write_text((root / f"frames/{tag}.json").read_text())
    (root / "log_pytest_stage7.txt").write_text(log if log is not None else
                                                "==== 2026-10-08T00:00:00Z python -m pytest tests/test_stage7_link.py\n"
                                                + "".join(f"tests/test_stage7_link.py::test_{i} PASSED\n" for i in range(17)))
    (root / "COMMIT.txt").write_text(f"==== a run\n{commit}\n")
    return root


def remaprank(root, stub=False, commit="abc123"):
    """heads/remaprank.json: the exploratory remap ranking (H_rem = the first 64 of H*, KO_x rows from the link B_x rows),
    or the stub stage_remaprank writes once the deadline has passed."""
    root = Path(root)
    sh = json.load(open(root / "heads/rank.json"))["sets_sha256"]
    if stub:
        J = {"provenance": {"skipped": "deadline passed", "sets_sha256": sh}, "arms": {}}
    else:
        rank, link = json.load(open(root / "heads/rank.json")), json.load(open(root / "link/mistral.json"))
        prov = dict(rank["provenance"], sets_sha256=sh)
        J = {"provenance": prov, "arms": {}}
        for arm in ("P1", "LETTER", "POST"):
            ko = [{"core": r["core"], "iS": r["iS"], "iT": r["iT"],
                   "runs": {f"{n}_{s}": r["runs"][f"x/{x}_{s}"] for s in SEEDS for n, x in (("P", "P"), ("x_all", "x_all"), ("x_notG", "x_notG"), ("x_Hrem", "x_H"))}}
                  for r in link["arms"][arm]]
            J["arms"][arm] = {"H_rem": rank["sets"]["arms"][arm]["H"][:32] + rank["sets"]["rand"][0][:32], "ko": ko}
    json.dump(J, open(root / "heads/remaprank.json", "w"))
    return root


def mutate(root, lab, fn):
    """Load a results file of the root, apply fn to it, write it back."""
    J = json.load(open(Path(root) / lab))
    fn(J)
    json.dump(J, open(Path(root) / lab, "w"))


def score(root, tag="mistral", rc=0):
    r = subprocess.run([sys.executable, "analysis/stage7_score.py", "--root", str(root), "--tag", tag, "--ref", str(Path(root) / "ref.json")],
                       cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == rc, (r.returncode, r.stderr[-3000:], r.stdout[-4000:])
    text = (Path(root) / "STAGE7_SCORE.txt").read_text()
    assert text == r.stdout
    order = [text.index(s) for s in ("PROVENANCE", "POPULATION", "GATES", "VERDICTS", "REPORTED", "SUMMARY", "EXPLORATORY")]
    assert order == sorted(order), order
    verd = text[text.index("\nVERDICTS"):text.index("\nREPORTED")]
    lines = re.findall(r"^  (I\d) +.*?-> (MET|NOT MET|NOT EVALUABLE)$", verd, re.M)
    assert [h for h, _ in lines] == I, lines
    return text, dict(lines)


def per_format(text, pred, fmt):
    """The verdict of one format's line under prediction ``pred``."""
    verd = text[text.index("\nVERDICTS"):text.index("\nREPORTED")]
    blk = verd[verd.index(f"\n  {pred} "):]
    m = re.search(rf"^         {fmt}: .*?-> (MET|NOT MET|NOT EVALUABLE)(  \[.*\])?$", blk, re.M)
    return m.group(1), m.group(0)


def test_all_met(tmp_path):
    text, v = score(write(tmp_path, scenario()))
    assert all(v[i] == "MET" for i in I), v
    assert "provenance OK" in text and "population: OK" in text and "SUMMARY: 7 MET, 0 NOT MET, 0 NOT EVALUABLE of 7" in text
    assert re.search(r"I-G0 .* 17 passed, 0 failed, 0 skipped .*-> MET", text) and "I5/I6 readable as value takeover: yes" in text
    assert "H1/H2 pattern at 24B (OPTIONS-AFTER): replicated" in text and "consistency check" in text
    # SENTENCE-AFTER is reported without a verdict word, with the format's gate status; the knockout contrast prints the
    # two t values with their intervals and no interpretive clause
    rep = text[text.index("\nREPORTED"):text.index("\nSUMMARY")]
    assert re.search(r"^  I1 SENTENCE-AFTER: .* -> meets the threshold \(no verdict\) \[gates: passed\]$", rep, re.M), rep[:600]
    assert not re.search(r"SENTENCE-AFTER: .*-> (MET|NOT MET)", rep)
    assert re.search(r"t under A \+0\.\d+ \[.*\] vs under N \+0\.\d+ \[", rep) and "recurs at the head level" not in rep
    assert "-- exploratory parts skipped at the deadline: none" in text and "KO_x(k) 8:" in text and "exploratory report failed" not in text
    assert "-- remap ranking: no results file" in text


def test_kappa_undefined_after_blocking_is_not_met(tmp_path):
    S = scenario()
    S["P1"]["fam"]["A:H"] = (1.0, 0.1, 0.2)        # behaviour lost: D^A < 3, kappa undefined
    text, v = score(write(tmp_path, S))
    assert v["I4"] == "NOT MET" and v["I6"] == "NOT MET"
    w, line = per_format(text, "I4", "OPTIONS-AFTER")
    assert w == "NOT MET" and "kappa undefined after blocking" in line


def test_dropped_resamples_rule(tmp_path):
    S = scenario()
    S["P1"]["fam"]["A:H"] = (3.02, 0.4, 2.65)       # D^A just above the rule's 3 nats, with a large spread across cores:
    S["P1"]["spread"] = {"A:H": 2.0}                  # the point is defined but many resamples fall below 3 nats
    text, v = score(write(tmp_path, S, noise=0.0))
    w, line = per_format(text, "I4", "OPTIONS-AFTER")
    assert w == "NOT MET" and "% of resamples dropped by the kappa rule (> 5 %)" in line, line


def test_evaluability_split(tmp_path):
    S = scenario()
    S["P1"]["gate"]["KO"] = 0.5                       # I-G2 (c) fails in OPTIONS-AFTER
    text, v = score(write(tmp_path, S))
    assert per_format(text, "I1", "OPTIONS-AFTER")[0] == "MET"
    for p in ("I2", "I3", "I4", "I5", "I6", "I7"):
        assert per_format(text, p, "OPTIONS-AFTER")[0] == "NOT EVALUABLE", p
    assert v["I1"] == "MET" and v["I2"] == "NOT EVALUABLE" and v["I5"] == "NOT EVALUABLE"
    S = scenario()
    S["P1"]["gK"] = 0.05                              # I-G3 clause 2 fails (a(all) - a(all@G) < 1): I1 is still scored
    text, v = score(write(tmp_path, S))
    assert per_format(text, "I1", "OPTIONS-AFTER")[0] == "NOT MET" and per_format(text, "I2", "OPTIONS-AFTER")[0] == "NOT EVALUABLE"
    assert per_format(text, "I4", "OPTIONS-AFTER")[0] == "MET"


def test_no_mention_gates(tmp_path):
    S = scenario()
    S["NONE"]["frames_scale"] = 1.1                       # the NO-MENTION unblocked batch misses the family D by 10 %
    text, v = score(write(tmp_path, S))
    assert v["I1"] == "MET" and v["I4"] == "NOT EVALUABLE" and v["I5"] == "NOT EVALUABLE" and v["I6"] == "MET"
    assert "NO-MENTION I-G1b" in per_format(text, "I4", "OPTIONS-AFTER")[1]
    S = scenario()
    S["NONE"]["fam"]["0"] = (14.34, -2.0, 4.0)            # kappa(NO-MENTION) undefined: psi_K < -0.1
    text, v = score(write(tmp_path, S))
    w, line = per_format(text, "I4", "OPTIONS-AFTER")
    assert w == "NOT EVALUABLE" and "kappa(NO-MENTION) defined" in line, line
    assert per_format(text, "I5", "OPTIONS-AFTER")[0] == "MET"


def test_i5_denominator_only_is_not_met(tmp_path):
    S = scenario()
    D, K, V = BASE["P1"]
    S["P1"]["fam"]["A:H"] = (D / 2, 0.1 * K, V)        # V in nats unchanged, D halved: psi_V doubles, I5 must not be met
    text, v = score(write(tmp_path, S))
    assert v["I5"] == "NOT MET"


def test_g1d_at_0p1_nats(tmp_path):
    S = scenario()
    S["BEFORE"]["before_gap"] = 0.09
    text, v = score(write(tmp_path, S))
    assert re.search(r"I-G1d .*-> MET", text) and v["I1"] == "MET"
    S["BEFORE"]["before_gap"] = 0.11
    text, v = score(write(tmp_path, S))
    assert re.search(r"I-G1d .*-> NOT MET", text) and all(v[i] == "NOT EVALUABLE" for i in I)


def test_both_formats_and_p1_only(tmp_path):
    S = scenario()
    D, K, V = BASE["LETTER"]
    S["LETTER"]["fam"]["A:H"] = (0.3 * D, 0.9 * K, V)    # LETTERS-AFTER: no key closure, behaviour lost
    text, v = score(write(tmp_path, S))
    assert v["I4"] == "NOT MET" and v["I6"] == "MET" and v["I5"] == "MET"
    assert "LETTERS-AFTER (two-sided, no verdict)" in text


def test_boundaries(tmp_path):
    S = scenario()
    S["P1"]["fam"]["0"] = (12.0, 7.24, 3.14)
    S["P1"]["fam"]["A:H"] = (9.0, 0.724, 6.5)             # t = 9 / 12 = 0.75 exactly (no noise: the interval is the point)
    text, v = score(write(tmp_path, S, noise=0.0))
    assert v["I6"] == "MET", per_format(text, "I6", "OPTIONS-AFTER")
    S["P1"]["fam"]["A:H"] = (8.99, 0.724, 6.5)
    text, v = score(write(tmp_path, S, noise=0.0))
    assert v["I6"] == "NOT MET"


def test_provenance_mismatch_and_i0(tmp_path):
    root = write(tmp_path, scenario())
    J = json.load(open(root / "heads/gate.json"))
    J["provenance"]["git_commit"] = "other"
    json.dump(J, open(root / "heads/gate.json", "w"))
    text, _ = score(root, rc=2)
    assert "MISMATCH: not one commit" in text
    root = write(tmp_path, scenario(), log="==== 2026-10-08T00:00:00Z python -m pytest x\ntests/test_stage7_link.py::test_0 PASSED\n"
                 "==== 2026-10-08T01:00:00Z python -m pytest x\ntests/test_stage7_link.py::test_0 FAILED\n")
    text, v = score(root, rc=2)
    assert re.search(r"I-G0 .*last of 2 run\(s\).*-> NOT MET", text) and all(v[i] == "NOT EVALUABLE" for i in I)


def test_test_tag_skips_sizes(tmp_path):
    root = write(tmp_path, scenario(), tag="TEST_Qwen2.5-0.5B-Instruct")
    J = json.load(open(root / "link/TEST_Qwen2.5-0.5B-Instruct.json"))
    for a in J["arms"]:
        J["arms"][a] = J["arms"][a][:2]
    json.dump(J, open(root / "link/TEST_Qwen2.5-0.5B-Instruct.json", "w"))
    G = json.load(open(root / "heads/gate.json"))
    for a in G["arms"]:
        G["arms"][a]["eval"] = G["arms"][a]["eval"][:2]
    G["none_clamp"]["eval"] = G["none_clamp"]["eval"][:2]
    json.dump(G, open(root / "heads/gate.json", "w"))
    text, v = score(root, tag="TEST_Qwen2.5-0.5B-Instruct", rc=0)
    assert "TEST MODE" in text and "population: OK" in text


def test_g1c_kappa_waiver_and_failure(tmp_path):
    """I-G1 (c): kappa undefined in both the null and the unblocked batch waives the kappa part (MET); undefined in only one
    fails; a null knockout that moves psi~_V by more than 0.02 fails, and the N lines carry the note. The remap-ranking stub
    of a deadline is reported as skipped."""
    S = scenario()
    S["POST"]["fam"]["0"] = S["POST"]["fam"]["N:null"] = (12.17, 1.0, 1.0)            # kappa undefined (psi_K + psi_V < 0.3), exact null
    S["P1"]["fam"]["0"] = (10.0, 2.85, 0.2)                                            # kappa defined (psi sum 0.305) ...
    S["P1"]["fam"]["N:null"], S["P1"]["null_exact"] = (10.0, 2.75, 0.2), False         # ... null undefined (0.295); psi~_V, t equal
    S["LETTER"]["fam"]["N:null"], S["LETTER"]["null_exact"] = (12.87, 10.0, 0.73), False   # psi~_V moves by 0.039
    root = remaprank(write(tmp_path, S, noise=0.0), stub=True)
    text, v = score(root)
    g = {f: re.search(rf"I-G1c {f} .*", text).group(0) for f in ("OPTIONS-AFTER", "LETTERS-AFTER", "SENTENCE-AFTER")}
    assert "undefined in both" in g["SENTENCE-AFTER"] and g["SENTENCE-AFTER"].endswith("-> MET"), g
    assert "|kappa_null - kappa| nan" in g["OPTIONS-AFTER"] and g["OPTIONS-AFTER"].endswith("-> NOT MET"), g
    assert "|psi~_V null - psi_V| 0.0389" in g["LETTERS-AFTER"] and g["LETTERS-AFTER"].endswith("-> NOT MET"), g
    assert re.search(r"N\(H\*\) OPTIONS-AFTER: .*\[I-G1 \(c\) not passed", text) and not re.search(r"N\(H\*\) SENTENCE-AFTER: .*\[I-G1 \(c\)", text)
    assert "heads/remaprank.json: skipped (deadline passed)" in text and "-- remap ranking: deadline passed" in text


def test_i3_i7_not_met_and_remap_ranking(tmp_path):
    S = scenario()
    S["P1"]["KOc"] = 0.5                                       # a random set removes half of the G-row part: I3 NOT MET
    D, K, V = BASE["P1"]
    S["P1"]["fam"]["A:rand0"] = (D, K, 1.5 * V)               # |d psi~_V| = 0.115 > 0.10: I7 NOT MET
    root = remaprank(write(tmp_path, S))
    text, v = score(root)
    assert v["I3"] == "NOT MET" and v["I7"] == "NOT MET" and v["I2"] == "MET", v
    assert per_format(text, "I3", "LETTERS-AFTER")[0] == "MET" and per_format(text, "I7", "LETTERS-AFTER")[0] == "MET"
    assert re.search(r"^   OPTIONS-AFTER: \|H_rem & H\*\| = 32 of 64, Jaccard 0\.333, .*KO_x\(H_rem\) \+0\.9", text, re.M), text[-1500:]


def test_population_provenance_and_scorer_error(tmp_path):
    root = write(tmp_path / "a", scenario())
    mutate(root, "link/mistral.json", lambda J: [J["arms"]["P1"][3]["runs"].pop(k) for k in list(J["arms"]["P1"][3]["runs"]) if k.startswith("A:rand1/")])
    text, _ = score(root, rc=2)
    assert "missing ['A:rand1'] MISMATCH" in text and "population: MISMATCH" in text
    root = write(tmp_path / "b", scenario())
    mutate(root, "link/mistral.json", lambda J: J["provenance"]["release"]["bases_sha256"].__setitem__("m3_101", "0" * 64))
    mutate(root, "frames/mistral.json", lambda J: J["provenance"].update(revision="main"))
    mutate(root, "heads/gate.json", lambda J: J["provenance"].update(dtype="torch.float16"))
    text, _ = score(root, rc=2)
    assert "the predecessor's release (link/mistral.json): RELEASE.json 2dea297d508e, bases and stories file MISMATCH against the pins" in text
    assert re.search(r"frames/mistral.json: .*MISMATCH: revision main", text)
    assert re.search(r"heads/gate.json: .*MISMATCH: the entry fixes BF16 and sdpa attention", text)
    root = write(tmp_path / "c", scenario())
    mutate(root, "frames/mistral.json", lambda J: J["provenance"]["bases"].__setitem__("pca_102", {"sha256": "0" * 64}))
    text, _ = score(root, rc=2)
    assert re.search(r"frames/mistral.json: .*MISMATCH: bases sha256", text)
    root = write(tmp_path / "d", scenario())
    mutate(root, "link/mistral.json", lambda J: J["arms"]["P1"][5]["runs"].pop("x/x_notG_101"))
    r = subprocess.run([sys.executable, "analysis/stage7_score.py", "--root", str(root), "--ref", str(root / "ref.json")], cwd=ROOT,
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == 1 and "SCORER ERROR; the predictions are NOT EVALUABLE" in r.stdout and "SCORER ERROR in ['link']" in r.stdout, r.stdout[-2000:]


def test_i0_skipped_or_too_few(tmp_path):
    head = "==== 2026-10-08T00:00:00Z python -m pytest tests/test_stage7_link.py\n"
    for log, why in ((head + "".join(f"tests/test_stage7_link.py::test_{i} PASSED\n" for i in range(16)) + "tests/test_stage7_link.py::test_16 SKIPPED\n", "16 passed, 0 failed, 1 skipped"),
                     (head + "".join(f"tests/test_stage7_link.py::test_{i} PASSED\n" for i in range(16)), "16 passed, 0 failed, 0 skipped")):
        text, v = score(write(tmp_path, scenario(), log=log), rc=2)
        assert re.search(rf"I-G0 .*{why} .*-> NOT MET", text) and all(v[i] == "NOT EVALUABLE" for i in I), why


def test_partial_exploratory_batches(tmp_path):
    """A batch present in only some cores (as a deadline inside a format used to leave it) is reported as such, and the rest
    of the exploratory report (later formats, breakdowns, the skipped list) is still written."""
    root = write(tmp_path, scenario())

    def cut(J):
        for r in J["arms"]["P1"][60:]:
            for k in [k for k in r["runs"] if k.startswith(("curve/", "A:L4/"))]:
                r["runs"].pop(k)
        J["provenance"]["explo_skipped"] = ["POST/curve", "POST/A:L4"]
    mutate(root, "link/mistral.json", cut)
    text, v = score(root)
    ex = text[text.index("######## EXPLORATORY"):]
    assert "exploratory report failed" not in ex and all(v[i] == "MET" for i in I), ex[:2000]
    assert "KO_x(k): the curve batch is present in only some cores: not reported" in ex and "A:L4     present in only some cores" in ex
    assert "-- exploratory parts skipped at the deadline: 2: POST/curve, POST/A:L4" in ex
    blk = ex[ex.index("   LETTERS-AFTER:"):]
    assert "KO_x(k) 8:" in blk and "seed 103:" in blk and "distractor at B or S (n=32)" in blk


def test_unblocked_kappa_drops_make_i4_not_evaluable(tmp_path):
    """kappa(NO-MENTION) is defined at the point estimate but fails the kappa rule in more than 5 % of the resamples
    (psi_K just above -0.1, spread across cores): I4 under OPTIONS-AFTER is NOT EVALUABLE, not NOT MET."""
    S = scenario()
    S["NONE"]["fam"]["0"] = (14.34, -1.40, 11.51)
    S["NONE"]["spreadK"] = {"0": 1.0}
    text, v = score(write(tmp_path, S, noise=0.0))
    w, line = per_format(text, "I4", "OPTIONS-AFTER")
    assert w == "NOT EVALUABLE" and "unblocked kappas defined in >= 95 % of resamples" in line, line
    assert v["I5"] == "MET" and v["I6"] == "MET"
