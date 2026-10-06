"""analysis/stage5_score.py: the entry point prints the gates, one verdict line per prediction G1-G22 in order, the
provenance and population checks, and writes STAGE5_SCORE.txt: on an empty root (every line NOT EVALUABLE), and on a
synthetic root holding every part (the writers of the part tests) under the preregistered tag, where the combination
rules are checked: G7 is H_redundant with H_replaced beside it, a Gate b failure makes G5-G8 NOT EVALUABLE, G4b without
H_diss (H_track, mixed, or no gated-in small model) is NOT APPLICABLE and left out, G2 under H_track is NOT MET, a missing
knockout file is MISSING in Gate b (k/k semantics) and not an M8 failure, part (e) evaluability is judged on the scored
models only, a corrupt file is a SCORER ERROR (exit 1), and the provenance / population checks report MISMATCH (exit 2
outside a TEST_ tag; a smaller part must hold the first k cores of the larger one)."""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from test_ioi import _fixture as ioi_fixture
from test_knockout import synth_knockout
from test_remention_attention import synth as attention_synth
from test_subsets import good_spec, write_synth as subsets_synth
from test_variants import write_synth as variants_synth

from analysis.stage5_parts import attention, ioi, knockout, subsets, variants  # noqa: F401
from ckeys.story import make_cores

ROOT = Path(__file__).resolve().parents[1]
G = [f"G{i}" for i in range(1, 23)]
ATT = {"Qwen2.5-7B-Instruct": (0.5, 0.8, 0.3, 5.5, 20.0), "Qwen2.5-14B-Instruct": (0.6, 0.8, 0.4, 12.0, 35.0),
       "Qwen2.5-1.5B-Instruct": (0.5, 0.8, 0.05, -0.4, 2.8), "Qwen2.5-3B-Instruct": (0.45, 0.7, 0.08, 0.0, 10.0)}  # H_diss, every line met
IOI = {"PLAIN": (0.05, 8.0, 8.0), "QUESTION": (0.1, 9.0, 10.0), "BEFORE": (0.1, 2.0, 2.0), "AFTER": (4.0, 1.0, 5.0)}


def run(root, tag, rc=0):
    r = subprocess.run([sys.executable, "analysis/stage5_score.py", "--root", str(root), "--tag", tag], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == rc, (r.returncode, r.stderr[-2000:])
    text = (root / "STAGE5_SCORE.txt").read_text()
    assert text == r.stdout
    return text


def verdict_lines(text):
    block = text.split("\nVERDICTS")[1].split("\nPROVENANCE")[0]
    lines = [l for l in block.splitlines() if re.match(r"  G\d+ ", l)]
    assert [l.split()[0] for l in lines] == G
    return dict(zip(G, lines))


def real_cores(f):
    """The attention writer repeats one core; the population check wants the seed-0 cores in item order."""
    d, cores = json.load(open(f)), make_cores(150, random.Random(0))
    for it in d["items"]:
        it["core"] = cores[it["i"]]
    json.dump(d, open(f, "w"))


def full_root(root):
    """Every part on the preregistered models and sizes: attention (H_diss), knockout (H_redundant), factorial seeds 0
    and 1 with the subset and variant arms, splice, competence, form probe, IOI (selection read at the pair)."""
    (root / "attention").mkdir(parents=True)
    for m, (E, F, Gh, post, p1) in ATT.items():
        attention_synth(root / "attention", m, 150, E=E, F=F, G=Gh, idk_post=post, idk_p1=p1, cap="7B" in m or "14B" in m or "3B" in m)
        real_cores(root / "attention" / f"{m}.json")
    for m in knockout.MODELS:
        synth_knockout(root / "knockout", m)
    for m in subsets.PRIMARY:
        r = subsets.REF3B[m]
        subsets_synth(root, m, good_spec(r["POST"], r["NONE"], r["AFTER"]))
        subsets_synth(root, m, good_spec(r["POST"], r["NONE"], r["AFTER"]), seed=1, splice=False)
    for m in variants.MODELS:
        r = subsets.REF3B[m]
        d = json.load(open(root / "factorial" / f"{m}_s0.json")) if (root / "factorial" / f"{m}_s0.json").exists() else None
        variants_synth(root, m, "token", post=r["POST"], none=r["NONE"], after=r["AFTER"])
        if d:  # one seed-0 factorial per model: the subset arms join the variant arms
            f = root / "factorial" / f"{m}_s0.json"
            v = json.load(open(f))
            v["results"] += [it for it in d["results"] if it["arm"] not in {"NONE", "POST", "AFTER"}]
            json.dump(v, open(f, "w"))
    for m in ioi.PAIR:
        ioi_fixture(root, m, IOI, n=200)
    ioi_fixture(root, "gpt2", {"PLAIN": (0.05, 8.0, 8.0), "INLINE": (-1.5, 6.0, 5.0), "INLINE_BEFORE": (0.1, 6.0, 6.0)}, n=200)
    (root / "COMMIT.txt").write_text("synthetic\n")


def test_empty_root(tmp_path):
    text = run(tmp_path, "TEST_empty")
    for head in ("GATES", "VERDICTS", "PROVENANCE", "POPULATION", "SUMMARY:"):
        assert head in text
    v = verdict_lines(text)
    assert all(l.rstrip().endswith("not scored]") and "-> NOT EVALUABLE" in l for l in v.values())
    assert "22 NOT EVALUABLE of 22" in text and "TEST MODE" in text
    for p in "abcde":
        assert f"######## PART ({p})" in text


def test_empty_knockout_dir(tmp_path):
    """A knockout directory without any *_s0.json under a TEST_ tag (the state after a knockout step FAILED and its file
    was moved aside): Gate b NOT EVALUATED and G5-G8 NOT EVALUABLE by the k/k semantics, not 'Gate b FAILED'."""
    for d in ("attention", "knockout", "factorial", "ioi"):
        (tmp_path / d).mkdir()
    text = run(tmp_path, "TEST_x")
    v = verdict_lines(text)
    assert "Gate b (part b, M8 sanity): NOT EVALUATED (no knockout results)" in text
    assert all("Gate b FAILED" not in v[g] and "-> NOT EVALUABLE" in v[g] for g in ("G5", "G6", "G7", "G8")), v["G5"]
    assert "22 NOT EVALUABLE of 22" in text


def test_full_root(tmp_path):
    full_root(tmp_path)
    text = run(tmp_path, "stage5")
    v = verdict_lines(text)
    assert "TEST MODE" not in text
    for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20", "G21", "G22"):
        assert v[g].split("->")[-1].strip().startswith("MET"), v[g]
    assert "-> NOT MET" in v["G12"]                                                             # the subsets writer's L2 cell
    assert "H_redundant: MET -> MET  [H_replaced NOT MET]" in v["G7"]
    assert "(a): MET (2/2 anchors); (b): MET (2/2 gated-in small models) -> MET" in v["G4"]
    assert "SUMMARY: 21 MET, 1 NOT MET, 0 NOT EVALUABLE of 22; provenance OK; population OK" in text
    assert "Gate b (part b, M8 sanity)" in text and "-> MET" in text.split("Gate b (part b")[1].split("\n")[0]
    for d, n in (("attention/Qwen2.5-7B-Instruct.json: n = 150", 150), ("knockout/Qwen2.5-7B-Instruct_s0.json: n = 150", 150),
                 ("row_restricted/Qwen2.5-7B-Instruct_direct.json: n = 60", 60), ("ioi/gpt2_s1.json: n = 200", 200)):
        assert d in text and f"(expected {n}): OK" in text.split(d)[1].split("\n")[0]
    assert "Qwen2.5-7B-Instruct: cores across parts: attention same, knockout same, factorial same, row_restricted the first 60 of 150, form_attention the first 60 of 150" in text
    assert "gpt2 (IOI): cores across parts: ioi same" in text
    # the H_replaced alternative and the partial takeover are reported beside G7, never as its verdict
    for m in knockout.MODELS:
        synth_knockout(tmp_path / "knockout", m, replaced=True)
    v = verdict_lines(run(tmp_path, "stage5"))
    assert "H_redundant: NOT MET -> NOT MET  [H_replaced MET]" in v["G7"]
    for m in knockout.MODELS:
        synth_knockout(tmp_path / "knockout", m, takeover=False)
    v = verdict_lines(run(tmp_path, "stage5"))
    assert "H_redundant: NOT MET -> NOT MET  [H_replaced NOT MET; partial takeover]" in v["G7"]
    # a Gate b failure in one model makes G5-G8 NOT EVALUABLE (the entry: uninterpretable until fixed and re-run)
    synth_knockout(tmp_path / "knockout", knockout.MODELS[1], gate_ok=False)
    text = run(tmp_path, "stage5")
    v = verdict_lines(text)
    for g in ("G5", "G6", "G7", "G8"):
        assert "-> NOT EVALUABLE  [Gate b FAILED: NOT EVALUABLE until the knockout is fixed and re-run]" in v[g], v[g]
    assert "SUMMARY: 17 MET, 1 NOT MET, 4 NOT EVALUABLE of 22" in text
    for m in knockout.MODELS:
        synth_knockout(tmp_path / "knockout", m)
    # a missing knockout file is MISSING in Gate b, not an M8 failure: the k/k lines count it as not met (G5 sentence 2/3 still MET)
    (tmp_path / "knockout" / f"{knockout.MODELS[2]}_s0.json").unlink()
    text = run(tmp_path, "stage5")
    v = verdict_lines(text)
    assert "Mistral-7B-Instruct-v0.3 MISSING -> NOT EVALUABLE" in text.split("Gate b (part b")[1].split("\n")[0]
    assert all("Gate b FAILED" not in v[g] for g in ("G5", "G6", "G7", "G8")) and "-> NOT MET" in v["G5"] and "-> NOT MET" in v["G6"]
    assert "G5 list (AFTER and P1) 2/3 -> NOT MET;  G5 sentence (POST) 2/3 -> MET;  G5 -> NOT MET" in text
    synth_knockout(tmp_path / "knockout", knockout.MODELS[2])
    # H_track at the small models: G4b is NOT APPLICABLE (its precondition H_diss failed) and G4 follows (a)
    attention_synth(tmp_path / "attention", "Qwen2.5-1.5B-Instruct", 150, E=0.0, F=0.0, G=0.0, idk_post=-0.4, idk_p1=2.8)
    attention_synth(tmp_path / "attention", "Qwen2.5-3B-Instruct", 150, E=0.02, F=0.0, G=0.0, idk_post=0.0, idk_p1=10.0, cap=True)
    real_cores(tmp_path / "attention" / "Qwen2.5-1.5B-Instruct.json"), real_cores(tmp_path / "attention" / "Qwen2.5-3B-Instruct.json")
    text = run(tmp_path, "stage5")
    v = verdict_lines(text)
    assert "(a): MET (2/2 anchors); (b): NOT APPLICABLE (G2 did not declare H_diss) -> MET" in v["G4"]
    assert "alternative H_track declared" in v["G2"] and "-> NOT MET" in v["G2"] and "(under H_track, 2/2) -> MET" in v["G3"]
    # no gated-in small model (ID_K(P1) < 1 at both): G2 NOT EVALUABLE, G4b NOT APPLICABLE (no H_diss declared, as under mixed) and G4 follows (a)
    attention_synth(tmp_path / "attention", "Qwen2.5-1.5B-Instruct", 150, E=0.5, F=0.8, G=0.05, idk_post=-0.4, idk_p1=0.5)
    attention_synth(tmp_path / "attention", "Qwen2.5-3B-Instruct", 150, E=0.45, F=0.7, G=0.08, idk_post=0.0, idk_p1=0.5, cap=True)
    real_cores(tmp_path / "attention" / "Qwen2.5-1.5B-Instruct.json"), real_cores(tmp_path / "attention" / "Qwen2.5-3B-Instruct.json")
    text = run(tmp_path, "stage5")
    v = verdict_lines(text)
    assert "Qwen2.5-1.5B-Instruct (small) GATED OUT, Qwen2.5-3B-Instruct (small) GATED OUT" in text
    assert "NOT EVALUABLE (no gated-in small model)" in v["G2"] and "(a): MET (2/2 anchors); (b): NOT APPLICABLE (no gated-in small model, G2 did not declare H_diss) -> MET" in v["G4"]
    for m in ("Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"):
        attention_synth(tmp_path / "attention", m, 150, E=ATT[m][0], F=ATT[m][1], G=ATT[m][2], idk_post=ATT[m][3], idk_p1=ATT[m][4], cap="3B" in m)
        real_cores(tmp_path / "attention" / f"{m}.json")
    # part (e): an exploratory model with a value does not make a pair line whose cells are all not evaluable print NOT MET
    ioi_fixture(tmp_path, "Qwen2.5-14B-Instruct", IOI, n=200)
    for m in ioi.PAIR:
        f = tmp_path / "ioi" / f"{m}_s1.json"
        d = json.load(open(f))
        for it in d["results"]:
            if it["arm"] == "AFTER":
                it["clean"]["B"]["lp"]["Subj"] = 0.0   # the AFTER gate fails at both pair models
        json.dump(d, open(f, "w"))
    text = run(tmp_path, "stage5")
    v = verdict_lines(text)
    assert v["G20"].rstrip().endswith("NOT EVALUABLE -> NOT EVALUABLE") and "(a): NOT EVALUABLE; (b): NOT EVALUABLE -> NOT EVALUABLE" in v["G19"]
    assert "(a): NOT EVALUABLE; (b): NOT EVALUABLE; (c): MET -> NOT MET" in v["G21"] and "## Qwen2.5-14B-Instruct (exploratory)" in text
    assert "-> MET" in text.split("## Qwen2.5-14B-Instruct (exploratory)")[1].split("G20 ")[1].split("\n")[0]
    # a corrupt knockout file is a SCORER ERROR in part (b) (exit 1), the other parts are still scored
    (tmp_path / "knockout" / "Qwen2.5-7B-Instruct_s0.json").write_text('{"provenance": {}, "results": "x"}')
    text = run(tmp_path, "stage5", rc=1)
    assert "SCORER ERROR in part (b)" in text and "SCORER ERROR in parts ['b']" in text and "-> MET" in verdict_lines(text)["G1"]
    synth_knockout(tmp_path / "knockout", "Qwen2.5-7B-Instruct")
    text = run(tmp_path, "stage5")
    assert "provenance OK; population OK" in text and "SCORER ERROR" not in text
    # provenance: two commits or a results file without a provenance block are MISMATCH outside TEST tags (exit 2, the score still written)
    f = tmp_path / "ioi" / "gpt2_s1.json"
    d = json.load(open(f)); d["provenance"]["git_commit"] = "other"; json.dump(d, open(f, "w"))
    text = run(tmp_path, "stage5", rc=2)
    assert "MISMATCH: not one commit" in text and "provenance MISMATCH; population OK" in text and "SCORER ERROR" not in text
    d["provenance"]["git_commit"] = json.load(open(tmp_path / "ioi" / "Qwen2.5-7B-Instruct_s1.json"))["provenance"].get("git_commit")
    json.dump(d, open(f, "w"))
    g = tmp_path / "ioi" / "Qwen2.5-7B-Instruct_s1.json"   # one model, files at two revisions
    e = json.load(open(g)); ea = e["provenance"].setdefault("args", {}); rev = ea.get("revision"); ea["revision"] = "rev-other"; json.dump(e, open(g, "w"))
    text = run(tmp_path, "stage5", rc=2)
    assert "Qwen2.5-7B-Instruct: MISMATCH: files with different revisions" in text and "revisions MISMATCH in ['Qwen2.5-7B-Instruct']" in text
    ea["revision"] = rev; json.dump(e, open(g, "w"))
    (tmp_path / "ENV.txt").write_text("".join(f"m{i} revision {i:040d}\n" for i in range(12)))
    text = run(tmp_path, "stage5")
    assert "    m11 revision " + "0" * 38 + "11" in text and "revisions: one per model" in text   # every ENV.txt line echoed, none cut
    f = tmp_path / "row_restricted" / "Qwen2.5-7B-Instruct_direct.json"
    json.dump(json.load(open(f))["results"], open(f, "w"))
    text = run(tmp_path, "stage5", rc=2)
    assert "Qwen2.5-7B-Instruct_direct.json: NO PROVENANCE (list of 480 items)  MISMATCH" in text
    assert "NO PROVENANCE" in run(tmp_path, "TEST_x") and "MISMATCH" not in run(tmp_path, "TEST_x").split("Qwen2.5-7B-Instruct_direct.json")[1].split("\n")[0]
    # population: a smaller part on 60 seed-0 cores that are not the first 60, and a dropped core, are MISMATCH
    f = tmp_path / "form_attention" / "Qwen2.5-7B-Instruct.json"
    d, cores = json.load(open(f)), make_cores(150, random.Random(0))
    for it in d["items"]:
        it["core"] = cores[it["i"] + 1]
    json.dump(d, open(f, "w"))
    text = run(tmp_path, "stage5", rc=2)
    assert "form_attention a subset of 150, NOT the first 60: MISMATCH" in text and "population MISMATCH" in text
    d["items"] = d["items"][1:]; json.dump(d, open(f, "w"))
    text = run(tmp_path, "stage5", rc=2)
    assert "population MISMATCH" in text and "form_attention/Qwen2.5-7B-Instruct.json: n per arm" in text
