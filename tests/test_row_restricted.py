"""experiments/row_restricted_keys.py plumbing: registered arms get mention / mention_words groups with the exactness
checks (none == clean, all == full K_S swap), and a raw-prompt RowTask on GPT-2 small (fused c_attn) is exact."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import experiments.row_restricted_keys as rr
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import PREFIX, register_arm
from ckeys.interventions import blocks

torch.set_grad_enabled(False)
SENT = "The room has a box, a drawer and a cabinet."


def test_registered_arm_groups():
    register_arm("T_ROOM3", lambda story, query: PREFIX + story + " " + SENT + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:",
                 span=lambda story, query: SENT)
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tok = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval(), AutoTokenizer.from_pretrained(name)
    res = rr.run(model, tok, rr.get_task("belief", view="direct"), ["T_ROOM3", "POST"], 1, log=lambda s: None)
    r3, rp = res
    assert r3["sizes"]["mention_words"] == 3 and r3["sizes"]["mention"] >= 10 and "choices" not in r3["sizes"]
    assert rp["sizes"]["remention_words"] == 6 and "mention" not in rp["sizes"]
    for r in res:
        assert abs(r["m"]["none"] - r["m_B"]) < 1e-4 and abs(r["m"]["all"] - r["m_S"]) > 1e-3
        assert set(r["sizes"]) | {"none"} == set(r["m"])
    assert "rows=mention_words" in rr.summarize(res, ["T_ROOM3"]) and "rows=remention_words" in rr.summarize(res, ["POST"])


class NameTask:  # a raw-prompt task: the IO name of an IOI sentence is the critical token
    chat, bos = False, True

    def cores(self, n, seed=0):
        return [{"io_b": "Ruth", "io_s": "Erik", "subj": "Charles"}][:n]

    def prompts(self, core, arm):
        return tuple(f"When {io} and Charles got a ticket at the beach, Charles decided to give it to\nChoices: Ruth, Erik, Charles\nAnswer:"
                     for io in (core["io_b"], core["io_s"]))

    def targets(self, tok, core, arm):
        return tuple(tok.encode(" " + core[k], add_special_tokens=False)[0] for k in ("io_s", "io_b"))

    def groups(self, tok, text, ids, off, p, core, arm):
        c0 = text.index("Choices: ")
        return {"choices": rr.rows_in(off, c0, text.index("\n", c0))}


def test_gpt2_rowsplice_exact():
    rr.register_task("names", NameTask)
    model, tok = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval(), AutoTokenizer.from_pretrained("gpt2")
    task = rr.get_task("names")
    res = rr.run(model, tok, task, ["AFTER"], 1, log=lambda s: None)[0]
    raw_b, raw_s = task.prompts(task.cores(1)[0], "AFTER")
    _, ib, _ = rr.encode_with_offsets(tok, raw_b, chat=False, bos=True)
    _, is_, _ = rr.encode_with_offsets(tok, raw_s, chat=False, bos=True)
    assert ib[0, 0] == tok.bos_token_id and res["p"] == (ib[0] != is_[0]).nonzero().item() == 2
    L = range(len(blocks(model)))
    with capture_kv(model, [res["p"]], L, "k") as K:
        model(is_, use_cache=False)
    iS, iB = task.targets(tok, task.cores(1)[0], "AFTER")
    with clamp_kv(model, [res["p"]], K, L, "k"):
        lp = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
    assert abs(res["m"]["all"] - (lp[iS] - lp[iB]).item()) < 1e-4 and abs(res["m"]["none"] - res["m_B"]) < 1e-4
    assert res["sizes"]["choices"] >= 5 and abs(res["m"]["all"] - res["m_B"]) > 1e-3
