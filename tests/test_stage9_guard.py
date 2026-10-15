"""The stage-9 guard (scripts/stage9_common.sh, s9_git_checks; docs/stage9/SPEC.md sections 12.1 and 13.4), run in
temporary git repositories without CUDA: it refuses a tree with modified tracked files, untracked (not ignored) files in
the code directories, a history without the commit "Freeze stage-9 predictions", a prediction note or spec absent at HEAD
or changed after that commit, and code (ckeys experiments analysis scripts tests data) changed after it; it passes on a
clean frozen clone (also when later commits touch only other files, and it takes the most recent freeze commit), and
prints the freeze commit, its date and the sha256 of the note and the spec. s9_guard also refuses results under OUT that
git tracks and results files in OUT written at another commit than HEAD (the scorer wants one commit). Also: s9_init in
TEST_MODE (no guard) sets the variables the stage-8 helpers need; s9_lb_stale (a stories file that read another lb file
than the current one) and its use before the st step.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "Freeze stage-9 predictions"


def sh(cmd, cwd, env=None, check=True):
    r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise AssertionError(f"{cmd}: {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="none", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="none", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    """A small repository with the stage-9 library, a note, a spec and one file in each code directory."""
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "stage9_common.sh", r / "scripts" / "stage9_common.sh")
    shutil.copy(ROOT / "scripts" / "stage8_common.sh", r / "scripts" / "stage8_common.sh")
    (r / "docs" / "stage9").mkdir(parents=True)
    (r / "docs" / "STAGE9_PREDICTIONS.md").write_text("# predictions\nP1 ...\n")
    (r / "docs" / "stage9" / "SPEC.md").write_text("# spec\n")
    for d in ("ckeys", "experiments", "analysis", "tests", "data"):
        (r / d).mkdir()
        (r / d / "x.py").write_text(f"# {d}\n")
    (r / "paper").mkdir()
    (r / "paper" / "main.tex").write_text("draft\n")
    git(r, "init", "-q")
    commit(r, "Initial")
    return r


def checks(repo):
    return sh("source scripts/stage9_common.sh && s9_git_checks", repo, check=False)


def test_refuses_without_freeze_commit(repo):
    r = checks(repo)
    assert r.returncode == 1 and "no commit with the subject 'Freeze stage-9 predictions'" in r.stderr
    commit(repo, "Freeze stage-9 predictions (draft)")            # the subject must match exactly
    assert checks(repo).returncode == 1


def test_passes_on_a_clean_frozen_clone(repo):
    F = commit(repo, SUBJECT)
    r = checks(repo)
    assert r.returncode == 0, r.stderr
    f, date, note, spec, head = r.stdout.strip().split("\t")
    assert f == F == head and date.startswith("20")
    assert note == hashlib.sha256((repo / "docs/STAGE9_PREDICTIONS.md").read_bytes()).hexdigest()
    assert spec == hashlib.sha256((repo / "docs/stage9/SPEC.md").read_bytes()).hexdigest()
    # later commits that touch only other files (the paper, results) keep F; an untracked results file is not a change
    (repo / "paper" / "main.tex").write_text("revised\n")
    H = commit(repo, "Paper edits")
    (repo / "results").mkdir()
    (repo / "results" / "out.json").write_text("{}")
    r = checks(repo)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\t")[0] == F and r.stdout.strip().split("\t")[4] == H
    # a temporary clone of the frozen repository passes too
    clone = repo.parent / "clone"
    git(repo.parent, "clone", "-q", str(repo), str(clone))
    shutil.copy(repo / "scripts" / "stage9_common.sh", clone / "scripts" / "stage9_common.sh")
    assert checks(clone).returncode == 0


def test_refuses_a_dirty_tree(repo):
    commit(repo, SUBJECT)
    (repo / "paper" / "main.tex").write_text("uncommitted\n")
    r = checks(repo)
    assert r.returncode == 1 and "modified tracked files" in r.stderr


@pytest.mark.parametrize("path", ["docs/STAGE9_PREDICTIONS.md", "docs/stage9/SPEC.md"])
def test_refuses_a_note_or_spec_changed_after_the_freeze(repo, path):
    commit(repo, SUBJECT)
    (repo / path).write_text((repo / path).read_text() + "changed after the freeze\n")
    commit(repo, "Edit after the freeze")
    r = checks(repo)
    assert r.returncode == 1 and f"{path} differs from its text at the freeze commit" in r.stderr


def test_refuses_a_note_absent_at_head(repo):
    commit(repo, SUBJECT)
    (repo / "docs" / "STAGE9_PREDICTIONS.md").unlink()
    commit(repo, "Remove the note")
    r = checks(repo)
    assert r.returncode == 1 and "docs/STAGE9_PREDICTIONS.md is absent at HEAD" in r.stderr


def test_refuses_a_note_added_after_the_freeze(tmp_path):
    r = tmp_path / "late"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "stage9_common.sh", r / "scripts" / "stage9_common.sh")
    (r / "ckeys").mkdir()
    (r / "ckeys" / "x.py").write_text("x\n")
    git(r, "init", "-q")
    commit(r, SUBJECT)                                             # frozen before the note existed
    (r / "docs" / "stage9").mkdir(parents=True)
    (r / "docs" / "STAGE9_PREDICTIONS.md").write_text("p\n")
    (r / "docs" / "stage9" / "SPEC.md").write_text("s\n")
    commit(r, "Add the note")
    res = checks(r)
    assert res.returncode == 1 and "differs from its text at the freeze commit" in res.stderr


@pytest.mark.parametrize("d", ["ckeys", "experiments", "analysis", "scripts", "tests", "data"])
def test_refuses_code_changed_after_the_freeze(repo, d):
    commit(repo, SUBJECT)
    (repo / d / "new.py").write_text("# added after the freeze\n")
    commit(repo, "Code after the freeze")
    r = checks(repo)
    assert r.returncode == 1 and "differs from the freeze commit" in r.stderr


def test_refuses_untracked_code_but_not_ignored_or_other_files(repo):
    """An untracked file in a code directory (code a run would import that is not in the freeze commit) is refused; an
    ignored one (__pycache__) and an untracked file outside the code directories (results) are not."""
    (repo / ".gitignore").write_text("__pycache__/\n")
    commit(repo, SUBJECT)
    (repo / "ckeys" / "__pycache__").mkdir()
    (repo / "ckeys" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (repo / "results").mkdir()
    (repo / "results" / "out.json").write_text("{}")
    assert checks(repo).returncode == 0
    (repo / "ckeys" / "shadow.py").write_text("# not committed\n")
    r = checks(repo)
    assert r.returncode == 1 and "untracked files in the code directories" in r.stderr and "ckeys/shadow.py" in r.stderr
    (repo / "ckeys" / "shadow.py").unlink()
    (repo / "tests" / "sub").mkdir()
    (repo / "tests" / "sub" / "t.py").write_text("# nested\n")
    r = checks(repo)
    assert r.returncode == 1 and "tests/sub/t.py" in r.stderr


def test_the_most_recent_freeze_commit_is_used(repo):
    commit(repo, SUBJECT)
    (repo / "ckeys" / "x.py").write_text("# fixed\n")
    (repo / "docs" / "STAGE9_PREDICTIONS.md").write_text("# predictions, refrozen\n")
    F2 = commit(repo, SUBJECT)
    r = checks(repo)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\t")[0] == F2


def test_s9_init_test_mode_sets_the_helpers_variables(tmp_path):
    """TEST_MODE: no guard, no pip; OUT, TGZ, DEV, DTYPE, PART and the deadline are set, COMMIT.txt, ENV.txt,
    PIP_FREEZE.txt and REVISIONS.txt are written, and STAGE9_FREEZE is unset."""
    out = tmp_path / "out"
    env = dict(os.environ, TEST_MODE="1", OUT=str(out), PY=str(Path(os.sys.executable)), STAGE9_FREEZE="stale",
               TGZ=str(tmp_path / "t.tgz"), TMPDIR=str(tmp_path))
    r = sh("source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_init > /dev/null && "
           "echo \"$OUT|$TGZ|$DEV|$DTYPE|$PART|$PART_UC|${STAGE9_FREEZE:-unset}|$KEEP_CACHE|$TEST_MODE\" && "
           "[ \"$STAGE8_DEADLINE\" -gt \"$(date +%s)\" ] && echo deadline-ok", ROOT, env=env)
    vals = r.stdout.strip().splitlines()
    assert vals[-2] == f"{out}|{tmp_path / 't.tgz'}|cpu|float32|9|9|unset|1|1"
    assert vals[-1] == "deadline-ok"
    for f in ("COMMIT.txt", "ENV.txt", "PIP_FREEZE.txt", "REVISIONS.txt"):
        assert (out / f).exists(), f
    assert "stage 9 TEST_MODE=1" in (out / "COMMIT.txt").read_text()
    assert "frozen at" not in (out / "COMMIT.txt").read_text()
    bad = sh("source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_init", ROOT,
             env=dict(env, FORCE="yes"), check=False)
    assert bad.returncode == 2 and "bad FORCE='yes'" in bad.stdout


def test_s9_finish_prints_the_named_sections(tmp_path):
    """s9_finish scores into STAGE9_SCORE.txt and prints its CHECKS, PREDICTIONS, FALSIFIERS and SUMMARY sections."""
    out = tmp_path / "out"
    fake = tmp_path / "scorer.py"
    fake.write_text("import sys, pathlib\n"
                    "root = pathlib.Path(sys.argv[sys.argv.index('--results') + 1])\n"
                    "(root / 'STAGE9_SCORE.txt').write_text('#### PROVENANCE\\nprov-line\\n#### CHECKS (x)\\ncheck-line\\n'\n"
                    "  '#### CONTROLS\\nctl-line\\n#### PREDICTIONS\\npred-line\\n#### FALSIFIERS\\nfal-line\\n'\n"
                    "  '#### DIRECTION B\\nb-line\\n#### SUMMARY\\nsum-line\\n')\n")
    env = dict(os.environ, TEST_MODE="1", OUT=str(out), PY=str(Path(os.sys.executable)), TGZ=str(tmp_path / "t.tgz"),
               TMPDIR=str(tmp_path))
    r = sh(f"source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_init > /dev/null && s9_finish {fake}",
           ROOT, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    for x in ("check-line", "pred-line", "fal-line", "sum-line"):
        assert x in r.stdout
    for x in ("prov-line", "ctl-line", "b-line"):
        assert x not in r.stdout
    assert (out / "MANIFEST.sha256").exists() and (tmp_path / "t.tgz").exists()
    assert "STAGE9_SCORE.txt" in (out / "MANIFEST.sha256").read_text()


def _stub_python(tmp_path, cuda_rc=0):
    """A stand-in for $PY in s9_guard: the transformers check passes; the CUDA check exits cuda_rc; the read of the
    results files' commits (s9_other_commits) runs in the real Python."""
    p = tmp_path / "stubpy"
    p.write_text(f'#!/usr/bin/env bash\ncase "$*" in *cuda*) exit {cuda_rc};; *git_commit*) exec "{os.sys.executable}" "$@";; '
                 f'*) exit 0;; esac\n')
    p.chmod(0o755)
    return p


def test_s9_guard_records_the_freeze_the_scorer_reads(repo, tmp_path):
    """s9_guard (checks 1-5) on a frozen repository: the COMMIT.txt line it appends is the record the scorer parses,
    and it exports STAGE9_FREEZE, STAGE9_NOTE_SHA256 and STAGE9_SPEC_SHA256."""
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    from stage9_parts.checks import Freeze
    F = commit(repo, SUBJECT)
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    env = dict(os.environ, OUT=str(out), S8_STATE=str(state), PY=str(_stub_python(tmp_path)), TGZ=str(tmp_path / "g.tgz"))
    r = sh("source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_guard && "
           "echo \"$STAGE9_FREEZE|$STAGE9_NOTE_SHA256|$STAGE9_SPEC_SHA256\"", repo, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    f, note, spec = r.stdout.strip().splitlines()[-1].split("|")
    assert f == F
    assert note == hashlib.sha256((repo / "docs/STAGE9_PREDICTIONS.md").read_bytes()).hexdigest()
    assert spec == hashlib.sha256((repo / "docs/stage9/SPEC.md").read_bytes()).hexdigest()
    fr = Freeze(out)
    assert fr.unique and (fr.F, fr.note, fr.spec) == (F, note, spec) and fr.heads == [F]
    # a refusal dies (exit 1, FAILED.txt) without a freeze record
    out2 = tmp_path / "out2"
    (out2 / "logs").mkdir(parents=True)
    (tmp_path / "nocuda").mkdir()
    env2 = dict(env, OUT=str(out2), PY=str(_stub_python(tmp_path / "nocuda", cuda_rc=1)), TGZ=str(tmp_path / "g2.tgz"))
    r = sh("source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_guard", repo, env=env2, check=False)
    assert r.returncode == 1 and "no CUDA device with >= 75 GiB" in (out2 / "FAILED.txt").read_text()
    (repo / "ckeys" / "x.py").write_text("# changed\n")                     # a dirty tree: refused before CUDA
    out3 = tmp_path / "out3"
    (out3 / "logs").mkdir(parents=True)
    r = sh("source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_guard", repo,
           env=dict(env, OUT=str(out3), TGZ=str(tmp_path / "g3.tgz")), check=False)
    assert r.returncode == 1 and "stage-9 guard: modified tracked files" in (out3 / "FAILED.txt").read_text()
    assert not Freeze(out3).recs


def test_s9_guard_refuses_results_at_another_commit_or_tracked(repo, tmp_path):
    """A later session must run at the HEAD of the results already in OUT (the scorer wants one commit across the
    files): results written at another commit are refused before anything runs, results at HEAD are not; and results
    that git tracks (committed between sessions) are refused."""
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    from stage9_parts.checks import Freeze
    F = commit(repo, SUBJECT)
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    (out / "stories").mkdir()
    state = tmp_path / "state"
    state.mkdir()
    env = dict(os.environ, OUT=str(out), S8_STATE=str(state), PY=str(_stub_python(tmp_path)), TGZ=str(tmp_path / "g.tgz"))
    run = "source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_guard"
    (out / "stories" / "mistral7.json").write_text(json.dumps({"provenance": {"git_commit": "e" * 40}, "items": []}))
    r = sh(run, repo, env=env, check=False)
    assert r.returncode == 1 and "written at another commit than HEAD" in (out / "FAILED.txt").read_text()
    assert "e" * 40 + " stories/mistral7.json" in (out / "FAILED.txt").read_text() and not Freeze(out).recs
    (out / "FAILED.txt").unlink()
    (out / "stories" / "mistral7.json").write_text(json.dumps({"provenance": {"git_commit": F}, "items": []}))
    (out / "stories" / "llama8.json.failed.20261015T000000Z").write_text("{}")     # moved aside: not read
    r = sh(run, repo, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert Freeze(out).heads == [F]
    # results committed into the repository: refused with its own reason
    res = repo / "results" / "gpu_stage9"
    (res / "logs").mkdir(parents=True)
    (res / "COMMIT.txt").write_text("==== session 1\n")
    commit(repo, "Partial stage-9 results")
    r = sh(run, repo, env=dict(env, OUT=str(res)), check=False)
    assert r.returncode == 1 and "are tracked by git" in (res / "FAILED.txt").read_text()


def test_s9_lb_stale_and_the_st_step(tmp_path):
    """s9_lb_stale: true when the stories file recorded another lb sha256 than the lb file has (cal reran), or cannot be
    read; false without a stories file or when they agree. gpu_stage9.sh redoes a done st step when it is true."""
    out = tmp_path / "out"
    (out / "lb").mkdir(parents=True)
    (out / "stories").mkdir()
    lb = out / "lb" / "TEST_qwen7.json"
    lb.write_text(json.dumps({"provenance": {"finished_utc": "2026-10-15T10:00:00Z"}, "l_b": 12}))
    sha = hashlib.sha256(lb.read_bytes()).hexdigest()
    env = dict(os.environ, OUT=str(out), PY=str(Path(os.sys.executable)))

    def stale():
        r = sh("source scripts/stage8_common.sh && source scripts/stage9_common.sh && s9_lb_stale TEST_qwen7", ROOT,
               env=env, check=False)
        return r.returncode == 0
    assert not stale()                                                         # no stories file
    st = out / "stories" / "TEST_qwen7.json"
    st.write_text(json.dumps({"provenance": {"lb_sha256": sha}, "items": []}))
    assert not stale()
    lb.write_text(json.dumps({"provenance": {"finished_utc": "2026-10-15T11:00:00Z"}, "l_b": 12}))   # cal reran
    assert stale()
    st.write_text("{truncated")
    assert stale()
    sh_text = (ROOT / "scripts" / "gpu_stage9.sh").read_text()
    i, j = sh_text.index('s9_lb_stale "$t"'), sh_text.index('s8_step "st_$key"')
    assert i < j and 'mv "$OUT/steps/st_$key.done"' in sh_text[i:j]
    assert '! s9_lb_stale "${TAGP}$1"' in sh_text                             # core_done is false while it is stale
