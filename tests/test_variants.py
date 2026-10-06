"""ckeys.variants (prereg G, part d): sentences, lists, forms and collisions; tokenisation and span checks on 300 cores
(seeds 0 and 1) x the Qwen2.5 / Mistral-7B / OLMo-2 tokenizers; the registered arms through format_factorial.run_item
(form log-probs, self-clamp exactness), experiments/form_attention.py against remention_attention.item_2a and
experiments/form_competence.py at Qwen2.5-0.5B FP32; analysis/stage5_parts/variants.py on synthetic data."""
import io
import re
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ckeys.variants as var
import experiments.form_attention as fa
import experiments.form_competence as fc
from analysis.stage5_parts import variants as sc
from analysis.stage5_parts.subsets import REF3B
from ckeys.encoding import ALL_ARMS, ARM_BUILDERS, ROOM, arm_span, candidate_ids, encode, raw_prompt
from ckeys.story import LOCATIONS, OBJECTS, make_cores, pick_x, record
from experiments.format_factorial import run_item
from experiments.remention_attention import cased_ids, item_2a

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
TOKS = {"qwen": (QWEN, 55), "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", 54), "olmo": ("allenai/OLMo-2-1124-7B-Instruct", 58)}
# seed-0 collision counts of the entry: (cores with f_S = f_X, cores with any pairwise coincidence among f_B, f_S, f_X)
COUNTS = {"qwen": {"POST_DE": (20, 61), "POST_DEMIX": (20, 61), "POST_UPPER": (0, 0)}, "olmo": {"POST_DE": (20, 61), "POST_DEMIX": (20, 61), "POST_UPPER": (0, 0)},
          "mistral": {"POST_DE": (12, 35), "POST_DEMIX": (12, 35), "POST_UPPER": (10, 28)}}
torch.set_grad_enabled(False)


def test_sentences_lists_and_registry():
    assert var.SENTENCES["POST"] == ROOM and var.variant_prompt("AFTER", "s", "q") == raw_prompt("AFTER", "s", "q")
    assert len(var.VARIANT_ARMS) == 17 and all(a in ARM_BUILDERS and not ARM_BUILDERS[a].needs_core for a in var.VARIANT_ARMS)
    assert [a for a in ALL_ARMS if a in var.VARIANT_ARMS] == list(var.VARIANT_ARMS)
    assert all(w not in OBJECTS and w not in LOCATIONS for w in var.FORMS["POST_OTHER"])
    assert raw_prompt("POST_DEMIX", "S.", "Q?") == "Read the story and answer the question.\n\nStory: S. The room has a Kiste, a Korb, a Regal, a Schublade, a Schrank and a Kammer.\nQuestion: Q?\nAnswer with one word.\nAnswer:"
    assert raw_prompt("AFTER_FR", "S.", "Q?") == "Read the story and answer the question.\n\nStory: S.\nQuestion: Q?\nChoices: boîte, panier, étagère, tiroir, armoire, placard\nAnswer with one word.\nAnswer:"
    assert arm_span("POST_THE", "S.", "Q?") == var.SENTENCES["POST_THE"] and arm_span("AFTER_SYN", "S.", "Q?") == "Choices: crate, hamper, ledge, compartment, cupboard, wardrobe"
    assert var.FLOOR["POST_SYN"] == ("POST_OTHER", "POST") and var.FLOOR["POST_FR"] == ("POST_FR_OTHER", "POST") and var.FLOOR["AFTER_DE"] == ("AFTER_OTHER", "AFTER")
    assert var.FAMILY["POST_FRMIX"] == "FR" and "POST_THE" not in var.FAMILY
    for v in var.VARIANT_ARMS:  # every variant sentence or list follows the story / question exactly like POST / AFTER
        raw = raw_prompt(v, "S.", "Q?")
        assert raw.startswith("Read the story and answer the question.\n\nStory: S." + (" " if v in var.SENTENCES else "\nQuestion: Q?\nChoices: "))
        assert raw.endswith("\nAnswer with one word.\nAnswer:")


@pytest.mark.parametrize("fam", list(TOKS))
def test_forms_collisions_and_spans(fam):
    name, p_ref = TOKS[fam]
    tok = AutoTokenizer.from_pretrained(name)
    cid = candidate_ids(tok, "POST")
    assert var.form_ids(tok, "LETTER") == candidate_ids(tok, "LETTER") and var.form_ids(tok, "POST_MODIF") == cid == var.form_ids(tok, "POST_THE")
    assert var.form_ids(tok, "S3") == cid and all(len(s) >= 2 for s in var.form_spans(tok, "POST_MODIF"))
    for arm in ("POST_DE", "POST_DEMIX", "POST_UPPER", "POST_SYN", "POST_FRMIX", "POST_TITLE", "POST_PLURAL", "AFTER_DE"):
        assert var.collisions(tok, arm) == var.collision_pairs(name, arm), arm
        f = var.form_ids(tok, arm)  # no form's first token equals another location's exact token
        assert not any(f[i] == cid[j] for i in range(6) for j in range(6) if i != j), arm
    cores0 = make_cores(150, random.Random(0))
    for arm, (n_sx, n_any) in COUNTS[fam].items():
        pairs = var.collisions(tok, arm)
        assert sum(var.excluded(c, pick_x(c), pairs) for c in cores0) == n_any, arm
        assert sum(any({c["source"], pick_x(c)} == {a, b} for a, b in pairs) for c in cores0) == n_sx, arm
    bad, lens = [], {}
    arms = ["NONE", "POST", "AFTER"] + list(var.VARIANT_ARMS)
    for ci, core in enumerate(cores0 + make_cores(150, random.Random(1))):
        X = pick_x(core)
        for arm in arms:
            ids = {}
            for k, loc in (("B", core["base"]), ("S", core["source"]), ("X", X)):
                r = record(core, "direct", loc)
                ids[k] = encode(tok, raw_prompt(arm, r["story"], r["query"]))[0].tolist()
            if len({len(v) for v in ids.values()}) != 1:
                bad.append((ci, arm, "len")); continue
            d = [i for i in range(len(ids["B"])) if ids["B"][i] != ids["S"][i]]
            dx = [i for i in range(len(ids["B"])) if ids["B"][i] != ids["X"][i]]
            if d != [p_ref] or dx != [p_ref]:
                bad.append((ci, arm, "p", d, dx)); continue
            if ci == 0:
                lens[arm] = len(ids["B"])
            if arm == "NONE":
                continue
            spans = fa.locate_spans(ids["B"], p_ref, var.form_spans(tok, arm))
            if spans is None:
                bad.append((ci, arm, "span")); continue
            fid = var.form_ids(tok, arm)  # the form token heads the span (MODIF: the exact noun ends the phrase)
            if any(ids["B"][sp[-1 if arm == "POST_MODIF" else 0]] != f for sp, f in zip(spans, fid)):
                bad.append((ci, arm, "first-token"))
    assert not bad, bad[:5]
    if fam == "qwen":
        assert lens == {"NONE": 88, "POST": 109, "POST_THE": 109, "POST_MODIF": 117, "POST_TITLE": 109, "POST_UPPER": 116, "POST_PLURAL": 103,
                        "POST_SYN": 110, "POST_FRMIX": 118, "POST_DEMIX": 117, "POST_FR": 119, "POST_DE": 117, "POST_OTHER": 109,
                        "POST_FR_OTHER": 114, "POST_DE_OTHER": 116, "AFTER": 102, "AFTER_SYN": 103, "AFTER_FR": 111, "AFTER_DE": 110, "AFTER_OTHER": 102}


@pytest.fixture(scope="module")
def mt():
    return (AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32, attn_implementation="eager").eval(),
            AutoTokenizer.from_pretrained(QWEN))


def test_run_item_forms_and_exactness(mt):
    model, tok = mt
    core = make_cores(1, random.Random(0))[0]
    the, syn, post = (run_item(model, tok, core, a, "direct", "cpu") for a in ("POST_THE", "POST_SYN", "POST"))
    assert the["pos"] == syn["pos"] == post["pos"] == 55 and the["len"] == 109 and syn["len"] == 110
    assert set(the["m"]["ID@0"]["lp"]) == {"S", "B", "X", "init", "Sf", "Bf", "Xf"} and set(post["m"]["ID@0"]["lp"]) == {"S", "B", "X", "init"}
    for row in the["m"].values():  # exact forms: the form token is the English token
        assert all(row["lp"][t + "f"] == row["lp"][t] for t in ("S", "B", "X"))
    assert all(syn["m"]["ID@0"]["lp"][t + "f"] != syn["m"]["ID@0"]["lp"][t] for t in ("S", "B", "X"))
    for r in (the, syn):  # self-clamp row == clean B (FP32 exactness of the clamp through the registry)
        assert abs(r["m"]["ID@0"]["m"] - r["clean"]["B"]["m"]) < 1e-4
        assert all(abs(r["m"]["ID@0"]["lp"][t] - r["clean"]["B"]["lp"][t]) < 1e-4 for t in r["clean"]["B"]["lp"])
    pc = sc.per_core([the, syn, post], "POST_THE")
    v = next(iter(pc.values()))
    assert v["has_form"] and abs(v["idKa"] - v["idK"]) < 1e-9 and abs(v["idKf"] - v["idK"]) < 1e-9 and abs(v["idVa"] - v["idV"]) < 1e-9
    assert not next(iter(sc.per_core([post], "POST").values()))["has_form"]


def test_form_attention_matches_item_2a(mt):
    model, tok = mt
    core = make_cores(1, random.Random(0))[0]
    it = fa.item(model, tok, core, "POST", candidate_ids(tok, "POST"))
    assert it["p"] == 55 and [sp[0] for sp in it["spans"]] == [68, 71, 74, 77, 80, 83] and all(len(sp) == 1 for sp in it["spans"])
    _, arrays, _ = item_2a(model, tok, core, "POST", cased_ids(tok))
    s = it["s"]
    for cond, key in (("K_S", "A_KS_rs_p"), ("B", "A_B_rs_p")):  # span sum of a single-token span == the head mean of 2a's array
        assert np.allclose(np.array(it["att"][cond]["span"][s]), arrays[key].astype(np.float32).mean(1), atol=2e-3)
        assert np.allclose(np.array(it["att"][cond]["last"][s]), np.array(it["att"][cond]["span"][s]))
    syn = fa.item(model, tok, core, "POST_SYN", candidate_ids(tok, "POST"))
    assert syn["spans"][1] == [71, 72] and len(syn["att"]["K_X"]["span"]) == 6 and len(syn["att"]["K_X"]["span"][0]) == 24
    res = fc.run(model, tok, families=("FR",))
    assert set(res["FR"]) == set(LOCATIONS) and {"form", "pick", "correct", "top1", "lp"} <= set(res["FR"]["box"]) and res["FR"]["box"]["form"] == "boîte"


# ---- scorer on synthetic files
def synth_item(core, arm, idK, idV, form, rng, acc=True):
    X, rows = pick_x(core), {}
    for k in ("ID@0", "K_S@0", "V_S@0", "KV_S@0", "K_X@0", "V_X@0", "KV_X@0"):
        lp, e = {t: -3.0 for t in ("S", "B", "X", "init", "Sf", "Bf", "Xf")}, rng.normal(0, 0.3)
        lp["S"] += (idK + e) * (k == "K_S@0") + (idV + e) * (k == "V_S@0")
        lp["X"] += (idK + e) * (k == "K_X@0") + (idV + e) * (k == "V_X@0")
        lp["Sf"] += (form + e) * (k == "K_S@0")
        lp["Xf"] += (form + e) * (k == "K_X@0")
        rows[k] = {"m": lp["S"] - lp["B"], "lp": lp}
    clean = {r: {"lp": {}, "m": 0.0, "mass": 0.5, "argmax_cand": (core["source"] if r == "S" else core["base"]) if acc else "shelf"} for r in ("B", "S", "X")}
    return {"core": core, "X": X, "arm": arm, "view": "direct", "pos": 55, "len": 100, "n_layers": 24, "clean": clean, "m": rows}


def write_synth(root, model, pattern="token", n=150, post=6.0, none=1.0, after=20.0, acc=None, fails=None, skipped=0):
    """pattern: token (synonyms/translations at the floor on every measure), concept (any-form and attention carry the
    read), noread (attention yes, readout no)."""
    rng, cores = np.random.default_rng(1), make_cores(n, random.Random(0))
    spec = {"NONE": (none, 16.0, 0), "POST": (post, 12.0, 0), "AFTER": (after, 6.0, 0), "POST_THE": (0.95 * post, 12.0, 0), "POST_MODIF": (0.9 * post, 12.0, 0),
            "POST_TITLE": (0.5 * post, 13.0, 0), "POST_UPPER": (0.3 * post, 14.0, 0), "POST_PLURAL": (0.55 * post, 13.0, 0), "POST_OTHER": (0.3, 15.0, 0),
            "POST_FR_OTHER": (0.2, 15.0, 0), "POST_DE_OTHER": (0.2, 15.0, 0), "AFTER_OTHER": (0.5, 15.0, 0)}
    f = post if pattern == "concept" else 0.0
    for a in ("POST_SYN", "POST_FRMIX", "POST_DEMIX", "POST_FR", "POST_DE"):
        spec[a] = (0.6, 15.0, f)
    for a in ("AFTER_SYN", "AFTER_FR", "AFTER_DE"):
        spec[a] = (1.5, 14.0, f)
    Path(root, "factorial").mkdir(parents=True, exist_ok=True)
    res = [synth_item(c, a, k, v, fm, rng, (acc or {}).get(a, True)) for a, (k, v, fm) in spec.items() for c in cores]
    json.dump({"provenance": {"skipped_items": skipped}, "results": res}, open(Path(root, "factorial", f"{model}_s0.json"), "w"))
    fails = fails or {}
    fams = {fam: {w: {"form": var.FORMS["POST_" + fam][i], "pick": "shelf" if w in fails.get(fam, ()) else w, "correct": w not in fails.get(fam, ()), "top1": w}
                  for i, w in enumerate(LOCATIONS)} for fam in fc.FAMILIES}
    Path(root, "competence").mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": {}, "families": fams}, open(Path(root, "competence", f"{model}.json"), "w"))
    att = {"POST": 0.05, "POST_THE": 0.045, "POST_MODIF": 0.04, "POST_TITLE": 0.02, "POST_UPPER": 0.01, "POST_PLURAL": 0.02, "POST_FR": 0.001, "POST_DE": 0.001}
    att |= {a: (0.04 if pattern in ("concept", "noread") else 0.002) for a in ("POST_SYN", "POST_FRMIX", "POST_DEMIX")}
    items = []
    for arm, A in att.items():
        for i, c in enumerate(cores[:60]):
            s, x, b = LOCATIONS.index(c["source"]), LOCATIONS.index(pick_x(c)), LOCATIONS.index(c["base"])

            def mat(own):
                M = rng.uniform(0.01, 0.03, (6, 24))
                for j in own:
                    M[j] += A + rng.normal(0, 0.002, 24)
                return M.tolist()

            a = {cond: {"span": mat(own), "last": mat(own), "head_max": mat(own)} for cond, own in (("B", []), ("K_S", [s]), ("K_X", [x]))}
            items.append({"core": c, "X": pick_x(c), "arm": arm, "p": 55, "T": 100, "spans": [], "b": b, "s": s, "x": x, "att": a, "lp": {}, "i": i})
    Path(root, "form_attention").mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": {"skipped_items": 0}, "items": items}, open(Path(root, "form_attention", f"{model}.json"), "w"))


def run_score(root, **kw):
    buf = io.StringIO()
    res = sc.score(root, out=lambda s: buf.write(s + "\n"), **kw)
    return res, buf.getvalue()


def test_scorer_synthetic(tmp_path):
    refs = {m: REF3B[m] for m in sc.MODELS}
    for m, r in refs.items():
        write_synth(tmp_path, m, "token", post=r["POST"], none=r["NONE"], after=r["AFTER"], fails={"DE": ("drawer",)} if "Mistral" in m else None)
    res, txt = run_score(tmp_path)
    f = res["final"]
    assert all(f[k] for k in ("G13 THE", "G13 MODIF")) and all(f[f"G14 {v}"] for v in ("TITLE", "UPPER", "PLURAL", "SYN", "FRMIX", "DEMIX"))
    assert all(f[f"G15 {v}"] for v in ("SYN", "FRMIX", "DEMIX")) and all(f[f"G16 {v}"] for v in ("SYN", "FRMIX", "DEMIX"))
    assert f["G17 FRMIX"] and f["G17 DEMIX"] and txt.count("decision table") == 3 and "token-level lookup" in txt and "confounded" not in txt
    assert "Gate 0 -> passed" in txt and "gate d2 [DE     ] 5/6 map correctly (drawer) -> pass" in txt and "gate d3 A_POST" in txt
    assert "DEMIX      n=150" in txt and "any-form (n=89)" in txt and "DEMIX      n= 90" in txt  # collision exclusions; Mistral gated on drawer
    assert "secondary FR " in txt and "secondary AFTER SYN" in txt and "G15 FR " not in txt and "G15 DE " not in txt   # secondary arms carry no G15 label
    for line in ("G13 THE", "G13 MODIF", "G14 TITLE", "G14 UPPER", "G14 PLURAL", "G14 SYN", "G14 FRMIX", "G14 DEMIX", "G15 SYN", "G15 FRMIX",
                 "G15 DEMIX", "secondary FR", "secondary DE", "secondary AFTER SYN", "G16 SYN", "G16 FRMIX", "G16 DEMIX", "G17 FRMIX", "G17 DEMIX"):
        assert any(l.strip().startswith(line) and "MET" in l for l in txt.splitlines()), line
    # concept pattern in every model -> G15 not met, concept-level lookup; noread -> lookup without readout
    for pat, label in (("concept", "concept-level lookup"), ("noread", "lookup without readout")):
        for m, r in refs.items():
            write_synth(tmp_path, m, pat, post=r["POST"], none=r["NONE"], after=r["AFTER"])
        res, txt = run_score(tmp_path)
        assert txt.count(label) == 3, (pat, [l for l in txt.splitlines() if "decision table" in l])
        assert res["final"]["G15 SYN"] is True and res["final"]["G17 FRMIX"] is True and res["final"]["G16 SYN"] is False
    # gates: a failed Gate 0 (one model) leaves 3 evaluable; a failed d1 and a failed d2 make cells not evaluable
    m0, r0 = sc.MODELS[0], refs[sc.MODELS[0]]
    write_synth(tmp_path, m0, "token", post=r0["POST"] + 3, none=r0["NONE"], after=r0["AFTER"])
    for m in sc.MODELS[1:]:
        write_synth(tmp_path, m, "token", post=refs[m]["POST"], none=refs[m]["NONE"], after=refs[m]["AFTER"], acc={"POST_UPPER": False},
                    fails={"SYN": ("box", "shelf")} if "OLMo" in m else None)
    res, txt = run_score(tmp_path)
    assert "Gate 0 -> FAILED" in txt and res["final"]["G13 THE"] is True and res["final"]["G14 UPPER"] is None and res["final"]["G15 SYN"] is None
    assert "NOT EVALUABLE: gate d1 failed" in txt and "gate d2 failed for SYN (4/6)" in txt and "G14 UPPER" in txt
    # the per-word competence subset applies to the SYN/FR/DE families only: a failing TITLE word (5/6, gate passed) leaves POST_TITLE on all cores
    m1, r1 = sc.MODELS[1], refs[sc.MODELS[1]]
    write_synth(tmp_path, m1, "token", post=r1["POST"], none=r1["NONE"], after=r1["AFTER"], fails={"TITLE": ("box",), "SYN": ("box",)})
    res, txt = run_score(tmp_path)
    blk = txt.split(f"## {m1}")[1].split("\n## ")[0]
    assert "gate d2 [TITLE  ] 5/6 map correctly (box) -> pass" in blk and re.search(r"TITLE\s+n=150 ", blk) and int(re.search(r"SYN\s+n=\s*(\d+) ", blk).group(1)) < 150
    # --test mode with a 0.5B-like file (n = 3, no reference): every cell not evaluable, every verdict line printed
    for m in sc.MODELS:
        for f in ("factorial", "competence", "form_attention"):
            (tmp_path / f / (m + ("_s0.json" if f == "factorial" else ".json"))).unlink()
    write_synth(tmp_path, "Qwen2.5-0.5B-Instruct", "token", n=3, post=-0.5, none=0.1, after=0.3)
    res, txt = run_score(tmp_path, test=True)
    assert "TEST MODE" in txt and "no stage-3b reference" in txt and "n=  3" in txt and "< 60" not in txt and txt.count("NOT EVALUABLE (< 3 evaluable models)") >= 19
    assert all(v is None for v in res["final"].values()) and "decision table SYN: not evaluable" in txt
