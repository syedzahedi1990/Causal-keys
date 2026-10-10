"""Gate JB-G0 (preregistration J, part B), items 7 and the tokenizer constraints: the fresh populations F and C (disjoint
from U = every core of stages 1-7 and from each other, deterministic, of the stated sizes, hashes pinned), S0 and U
hashes, the per-core attributes (lexicon, sentence, order), the prompt builders (byte-identical to ckeys.encoding on S0;
B, S, X differ at one word; the same order in list and sentence; POST and PRE share the sentence; POST-NULL names no
candidate), answer parsing and frames with either lexicon, and the length and single-token constraints in every Part-B
tokenizer that is cached locally (the per-model tokenizer check of experiments/fresh_factorial.py repeats them on the
verified files before each model loads)."""
import collections
import os
import random

import pytest

from ckeys import fresh
from ckeys.encoding import raw_prompt as enc_raw_prompt
from ckeys.story import LOCATIONS, OBJECTS, make_cores, record

TOKENIZERS = {"qwen7": "Qwen/Qwen2.5-7B-Instruct", "qwen14": "Qwen/Qwen2.5-14B-Instruct",
              "mistral7": "mistralai/Mistral-7B-Instruct-v0.3", "olmo7": "allenai/OLMo-2-1124-7B-Instruct",
              "llama8": "unsloth/Meta-Llama-3.1-8B-Instruct", "gemma9": "unsloth/gemma-2-9b-it",
              "gemma2b": "unsloth/gemma-2-9b-it",   # gemma-2-2b-it's tokenizer files are those of gemma-2-9b-it (manifest)
              "phi4": "microsoft/phi-4", "falcon7": "tiiuae/Falcon3-7B-Instruct", "yi9": "01-ai/Yi-1.5-9B-Chat"}


def test_u_and_s0_hashes():
    U = fresh.used_cores()
    assert len(U) == 3981 and fresh.u_hash(U) == fresh.U_SHA256
    S0 = fresh.population("S0")
    assert [it.core for it in S0] == make_cores(150, random.Random(0))
    import hashlib
    import json
    assert hashlib.sha256(json.dumps([list(fresh.core_tuple(it.core)) for it in S0]).encode()).hexdigest() == fresh.S0_SHA256
    assert all(it.lex == 1 and it.order == tuple(range(6)) and it.sent is None for it in S0)
    assert fresh.pop_hash(S0) == fresh.POP_SHA256["S0"]


def test_fresh_populations_disjoint_deterministic_pinned():
    U = fresh.used_cores()
    F, C = fresh.population("F"), fresh.population("C")
    assert len(F) == 150 and len(C) == 30
    tF, tC = [fresh.core_tuple(i.core) for i in F], [fresh.core_tuple(i.core) for i in C]
    assert len(set(tF)) == 150 and len(set(tC)) == 30
    assert not set(tF) & U and not set(tC) & U and not set(tC) & set(tF)
    assert fresh.pop_hash(F) == fresh.POP_SHA256["F"] and fresh.pop_hash(C) == fresh.POP_SHA256["C"]
    fresh.population.cache_clear()   # determinism: rebuilt from scratch, the same items
    assert fresh.population("F") == F and fresh.population("C") == C
    # the stream: F is the first 150 of make_cores(., Random(20261013)) outside U (and not repeated)
    stream = [c for c in make_cores(400, random.Random(20261013)) if fresh.core_tuple(c) not in U]
    seen, first = set(), []
    for c in stream:
        if fresh.core_tuple(c) not in seen:
            seen.add(fresh.core_tuple(c))
            first.append(c)
    assert [i.core for i in F] == first[:150]
    assert sum(fresh.core_tuple(c) in U for c in make_cores(400, random.Random(20261013))) == 4


def test_attributes_and_clusters():
    for name, n in (("F", 150), ("C", 30)):
        P = fresh.population(name)
        assert collections.Counter(i.lex for i in P) == {1: n // 2, 2: n // 2}
        assert set(i.sent for i in P) == set(range(8))
        assert all(sorted(i.order) == list(range(6)) for i in P)
        hs = sorted(P, key=lambda i: fresh.core_hash(i.core))
        assert all(i.lex == 2 for i in hs[:n // 2]) and all(i.sent == r % 8 for r, i in enumerate(hs))
    F = fresh.population("F")
    assert len({(i.core["base"], i.core["source"]) for i in F}) == 30 and len({i.core["object"] for i in F}) == 20
    assert all(i.cluster == (i.lex, i.core["base"], i.core["source"]) for i in F)


def test_s0_prompts_equal_the_published_builders():
    for it in fresh.population("S0")[:40]:
        for arm in ("P1", "AFTER", "BEFORE", "POST", "PRE", "NONE", "LETTER"):
            for loc in (it.core["base"], it.core["source"], it.x_canon):
                rec = record(it.core, "direct", loc)
                assert fresh.raw_prompt(it, arm, loc) == enc_raw_prompt(arm, rec["story"], rec["query"])


def test_fresh_prompts_render_lexicon_order_and_sentences():
    F = fresh.population("F")
    for it in F:
        W = it.words
        assert set(it.track.values()) <= set(W) and len({it.track["S"], it.track["B"], it.track["X"]}) == 3
        assert it.track["X"] not in {it.w(it.core[f]) for f in ("base", "source", "initial", "distractor_location")}
        ordered = [W[i] for i in it.order]
        p = {a: fresh.raw_prompt(it, a, it.core["base"]) for a in fresh.ARMS_F}
        assert "Choices: " + ", ".join(ordered) in p["AFTER"] and "Choices: " + ", ".join(ordered) in p["BEFORE"]
        sent = fresh.SENTENCES[it.sent].format(*ordered)
        assert sent in p["POST"] and sent in p["PRE"]
        assert p["POST"].index(sent) > p["POST"].index(" watches as ") and p["PRE"].index(sent) < p["PRE"].index(" watches as ")
        null = fresh.SENTENCES[it.sent].format(*[fresh.NULL_NOUNS[i] for i in it.order])
        assert null in p["POST-NULL"]
        story_words = set(p["NONE"].replace(".", " ").replace("?", " ").split())
        for w in fresh.LEX1 + fresh.LEX2:
            assert w not in null.replace(".", " ").replace(",", " ").split()
        other = fresh.LEX2 if it.lex == 1 else fresh.LEX1
        assert not any(w in story_words for w in other)       # the other lexicon never appears
        for a in fresh.ARMS_F:   # B, S, X prompts differ only in the moved-to word
            pb, ps = fresh.raw_prompt(it, a, it.core["base"]), fresh.raw_prompt(it, a, it.core["source"])
            assert pb.replace(f"moved to the {it.track['B']}.", "@") == ps.replace(f"moved to the {it.track['S']}.", "@")
    assert not set(fresh.NULL_NOUNS) & (set(fresh.LEX1) | set(fresh.LEX2) | set(OBJECTS))


def test_parse_stop_and_frames_per_lexicon():
    W2 = fresh.LEX2
    assert fresh.parse(" The crate", W2) == "crate" and fresh.parse("**Jar**", W2) == "jar"
    assert fresh.parse(" In the buckets.", W2) == "bucket" and fresh.parse(" The box", W2) == "other"
    assert fresh.parse(" On the shelf", LOCATIONS) == "shelf" and fresh.parse(" I don't know", W2) == "other"
    names = {"a": "Alice", "b": "Bob", "o": "candle", "d": "map"}
    assert fresh.extract_frame(" Alice thinks the candle is in the tray", names, W2) == " {a} thinks the {o} is in the "
    assert fresh.extract_frame(" the box", names, W2) is None and fresh.extract_frame(" the box", names, LOCATIONS) == " the "

    class T:
        def decode(self, ids):
            return "".join(ids)
    stop = fresh.stop_fn(T(), W2)
    assert not stop(0, [" In", " the", " cr"]) and stop(0, [" In", " the", " crate"]) and stop(0, [" jar", "."])
    assert set(fresh.frame_sets([" {a} put it in the ", " "])["E"]) >= set(fresh.FRAMES_E_FIXED)
    assert fresh.frame_sets([" "])["E"].count(" ") == 1


def _tok(name):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError as ex:
        pytest.skip(f"tokenizer {name} not cached: {type(ex).__name__}")


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_lengths_and_single_tokens_cached_tokenizer(key):
    tok = _tok(TOKENIZERS[key])
    for w in fresh.LEX1 + fresh.LEX2 + fresh.NULL_NOUNS:
        x = tok(" " + w, add_special_tokens=False).input_ids
        assert len(x) == 1, (key, w)
        for p in (",", ".", " and"):
            assert x[0] in tok(f"the {w}{p}", add_special_tokens=False).input_ids, (key, w, p)
    n = lambda s: len(tok(" " + s, add_special_tokens=False).input_ids)  # noqa: E731
    room = n(fresh.ROOM_TEMPLATE.format(*fresh.LEX1))
    rng = random.Random(1)
    for t in fresh.SENTENCES:
        lens = {n(t.format(*[nouns[i] for i in o])) for nouns in (fresh.LEX1, fresh.LEX2, fresh.NULL_NOUNS)
                for o in [tuple(range(6))] + [tuple(rng.sample(range(6), 6)) for _ in range(4)]}
        assert room - 3 <= min(lens) and max(lens) <= room + 3, (key, t, lens, room)
        assert len(lens) == 1, (key, t, lens)   # the null sentence has the after-sentence's length


@pytest.mark.parametrize("key", ["qwen7", "mistral7", "llama8", "gemma9", "phi4", "falcon7", "olmo7", "yi9"])
def test_items_valid_and_formsets_cached_tokenizer(key):
    from experiments.fresh_factorial import Invalid, encode_item
    tok = _tok(TOKENIZERS[key])
    for name, arms in (("F", fresh.ARMS_F), ("C", fresh.ARMS_CAL), ("S0", fresh.ARMS_S0)):
        P = fresh.population(name)
        for it in P[:50] if name != "C" else P:
            for a in arms:
                try:
                    encode_item(tok, it, a)
                except Invalid as ex:
                    raise AssertionError((key, it.key, a, str(ex)))
        for lex in (1, 2):
            it = next((i for i in P if i.lex == lex), None)
            if it is None:
                continue
            fs, dropped = fresh.form_set(tok, it, [" {a} believes the {o} is in the "])
            assert not dropped and not fs.check_decode(), (key, fs.check_decode()[:3])
            for w in it.words:
                assert len(fs.sets["E"][w]) >= len(fs.sets["sigma"][w]) >= 4
