"""analysis/stage6_score.py: the entry point prints the provenance, the population checks, the gates and one verdict line
per prediction H1-H12 in order, and writes STAGE6_SCORE.txt: on an empty root (every line NOT EVALUABLE, H12 NOT RUN), on a
synthetic part-(b) root at the preregistered n (the H_read fixture of tests/test_prakash.py: H7-H9 and H11 met, H10 not;
Gate a1 read off the pipeline's pytest log), with a Llama-3-70B directory (H12 scored) or its SKIPPED.txt (NOT RUN), a
corrupt file (SCORER ERROR, exit 1) and a population short of the preregistered n (MISMATCH, exit 2 outside a TEST_ tag);
Gate a1 from the last pytest run of an appended log, H1-H6 NOT EVALUABLE without a passing Gate a1 outside a TEST_ tag, and
the provenance MISMATCH paths (dtype, attention, commit, revision, release hash; exit 2)."""
import json
import re
import subprocess
import sys
from pathlib import Path

from test_prakash import write_synth

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

ROOT = Path(__file__).resolve().parents[1]
H = [f"H{i}" for i in range(1, 13)]


def run(root, tag="stage6", rc=0):
    r = subprocess.run([sys.executable, "analysis/stage6_score.py", "--root", str(root), "--tag", tag], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == rc, (r.returncode, r.stderr[-2000:], r.stdout[-3000:])
    text = (root / "STAGE6_SCORE.txt").read_text()
    assert text == r.stdout
    order = [text.index(s) for s in ("PROVENANCE", "POPULATION", "GATES", "VERDICTS", "SUMMARY")]
    assert order == sorted(order), order
    verd = text[text.index("\nVERDICTS"):text.index("\nSUMMARY")]
    lines = re.findall(r"^  (H\d+) +.*?-> (MET|NOT MET|NOT EVALUABLE|NOT RUN)\b", verd, re.M)
    assert [h for h, _ in lines] == H, lines
    return text, dict(lines)


def test_empty_root(tmp_path):
    text, v = run(tmp_path)
    assert all(v[h] == "NOT EVALUABLE" for h in H[:11]) and v["H12"] == "NOT RUN"
    assert "Gate a1" in text and "population: OK" in text


def test_part_b_synthetic(tmp_path):
    (tmp_path / "heads").mkdir()
    write_synth(tmp_path / "prakash", n=150)
    (tmp_path / "log_pytest_stage6.txt").write_text("".join(f"tests/test_head_splice.py::test_{i} PASSED\n" for i in range(7)))
    (tmp_path / "COMMIT.txt").write_text("==== a run\nabc123\n")
    text, v = run(tmp_path)
    assert [v[h] for h in H[6:]] == ["MET", "MET", "MET", "NOT MET", "MET", "NOT RUN"], v
    assert all(v[h] == "NOT EVALUABLE" for h in H[:6])
    assert re.search(r"Gate a1 .* 7 passed, 0 failed .*-> MET", text) and "Gate b3" in text and "Gate b0" in text
    assert "    abc123" in text and "pool sha256" in text and "OK (ckeys.causaltom pins)" in text and "population: OK" in text
    (tmp_path / "log_pytest_stage6.txt").write_text("tests/test_head_splice.py::test_0 FAILED\n")
    text, _ = run(tmp_path)
    assert re.search(r"Gate a1 .*-> NOT MET", text) and "Gate a1 FAILED" in text
    d = tmp_path / "prakash" / "Meta-Llama-3-70B-Instruct"
    d.mkdir()
    (d / "SKIPPED.txt").write_text("no access\n")
    assert run(tmp_path)[1]["H12"] == "NOT RUN"
    d.joinpath("SKIPPED.txt").unlink()
    write_synth(tmp_path / "prakash", model="Meta-Llama-3-70B-Instruct", n=150, nL=80, ls=34, li=6)
    assert run(tmp_path)[1]["H12"] == "MET"


def test_errors_and_mismatch(tmp_path):
    write_synth(tmp_path / "a" / "prakash", n=40)          # 40 of the 150 preregistered pairs
    text, _ = run(tmp_path / "a", rc=2)
    assert "population: MISMATCH" in text
    run(tmp_path / "a", tag="TEST_Qwen2.5-0.5B-Instruct")    # a TEST_ tag does not check the sizes
    d = write_synth(tmp_path / "b" / "prakash", n=150)
    (d / "exchange.json").write_text("{trunc")
    text, v = run(tmp_path / "b", rc=1)
    assert "SCORER ERROR in part (b)" in text and "UNREADABLE" in text and v["H7"] == "NOT EVALUABLE"
    j = json.load(open(d / "lstar.json"))
    assert j["population"][:3] == [0, 1, 2]


def test_gate_a1_last_run(tmp_path):
    (tmp_path / "COMMIT.txt").write_text("==== 2026-10-01T10:00:00Z PART=all MODEL= TEST_MODE=0 FORCE=0\naaaaaaaaaa11\n"
                                         "==== 2026-10-01T12:00:00Z PART=all MODEL= TEST_MODE=0 FORCE=0\nbbbbbbbbbb22\n")
    hdr = "==== 2026-10-01T{}Z python -m pytest tests/test_head_splice.py -v\n"
    block = lambda s: "".join(f"tests/test_head_splice.py::test_{i} {s}\n" for i in range(7))  # noqa: E731
    f = tmp_path / "log_pytest_stage6.txt"
    f.write_text(hdr.format("10:01:00") + block("ERROR") + "==== 2026-10-01T10:02:00Z FAILED exit 1\n" + hdr.format("12:01:00") + block("PASSED"))
    text, _ = run(tmp_path)
    assert re.search(r"Gate a1 .*last of 2 run\(s\), 2026-10-01T12:01:00Z, commit bbbbbbbbbb\): 7 passed, 0 failed .*-> MET", text), text
    f.write_text(hdr.format("10:01:00") + block("PASSED") + hdr.format("12:01:00") + block("PASSED").replace("test_3 PASSED", "test_3 FAILED"))
    text, _ = run(tmp_path)
    assert re.search(r"Gate a1 .*commit bbbbbbbbbb\): 6 passed, 1 failed .*-> NOT MET", text), text


def test_gate_a1_gates_part_a():
    import stage6_score as s6
    per = {"Qwen2.5-7B-Instruct": True, "Mistral-7B-Instruct-v0.3": True}
    ra = {"verdicts": {h: True for h in H[:6]} | {"H5s": True}, "per_model": {h: per for h in H[:6] + ["H5s"]},
          "gates": {m: {} for m in per}}
    R = {"a": ([], ra), "b": ([], None)}
    for a1, test, want in ((True, False, True), (False, False, None), (None, False, None), (False, True, True)):
        lines = []
        F = s6.verdicts(R, a1, False, lines.append, test)
        assert all(F[h] is want for h in H[:6]), (a1, test, F)
        assert ("H1-H6 are NOT EVALUABLE" in "\n".join(lines)) == (want is None)


def test_provenance_mismatch(tmp_path):
    d = write_synth(tmp_path / "prakash", n=150)
    for f in d.glob("*.json"):
        j = json.load(open(f))
        j["provenance"] |= {"dtype": "torch.bfloat16", "attn_implementation": "sdpa", "revision": "r1"}
        json.dump(j, open(f, "w"))
    text, _ = run(tmp_path)
    assert "MISMATCH" not in text[:text.index("\nGATES")]
    for key, val, msg in (("dtype", "torch.float32", "the entry fixes BF16"), ("attn_implementation", "eager", "the entry fixes BF16"),
                          ("git_commit", "other", "not one commit"), ("revision", "r2", "different revisions")):
        j = json.load(open(d / "exchange.json"))
        old = j["provenance"][key]
        j["provenance"][key] = val
        json.dump(j, open(d / "exchange.json", "w"))
        text, _ = run(tmp_path, rc=2)
        assert msg in text and "SUMMARY" in text and "provenance MISMATCH" in text, key
        j["provenance"][key] = old
        json.dump(j, open(d / "exchange.json", "w"))
    j = json.load(open(d / "preflight.json"))
    k = next(iter(j["release"]["files"]))
    j["release"]["files"][k] = "0" * 64
    json.dump(j, open(d / "preflight.json", "w"))
    text, _ = run(tmp_path, rc=2)
    assert "MISMATCH against ckeys.causaltom" in text
