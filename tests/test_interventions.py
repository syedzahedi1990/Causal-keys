import torch, pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from ckeys.interventions import capture, edit, edits, RotatedSubspace, pca_basis, signed_permutation_null

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOCS = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]

@pytest.fixture(scope="module")
def mt():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    return model, tok

def enc(tok, loc):
    story = f"The apple is in the box. Ben moves the apple to the {loc}. Where is the apple now?"
    msgs = [{"role": "user", "content": story + " Answer with one word from: " + ", ".join(LOCS) + "."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = ids if torch.is_tensor(ids) else ids["input_ids"]
    loc_tok = tok.encode(" " + loc, add_special_tokens=False)[0]
    pos = (ids[0] == loc_tok).nonzero()[0].item()
    return ids, pos

def answer(model, tok, ids):
    with torch.no_grad():
        lg = model(ids).logits[0, -1]
    cand = [tok.encode(l, add_special_tokens=False)[0] for l in LOCS]
    return LOCS[int(lg[cand].argmax())], lg

def test_identity_edit_is_noop(mt):
    model, tok = mt
    ids, pos = enc(tok, "shelf")
    a0, lg0 = answer(model, tok, ids)
    with edit(model, 4, "resid", [pos], lambda h: h):
        a1, lg1 = answer(model, tok, ids)
    assert torch.allclose(lg0, lg1, atol=1e-4) and a0 == "shelf"

def test_full_resid_patch_transfers_source(mt):
    model, tok = mt
    ib, pb = enc(tok, "shelf"); is_, ps = enc(tok, "drawer")
    assert ib.shape == is_.shape and pb == ps
    with capture(model, [4], "resid") as st:
        model(is_)
    hs = st[4][:, ps].clone()
    with edit(model, 4, "resid", [pb], lambda h: hs):
        a, _ = answer(model, tok, ib)
    assert a == "drawer"

def test_self_key_exchange_noop_and_null_norm(mt):
    model, tok = mt
    ids, pos = enc(tok, "shelf")
    L = range(2, model.config.num_hidden_layers)
    with capture(model, L, "k") as st:
        _, lg0 = answer(model, tok, ids)
    specs = [(l, "k", [pos], (lambda h, l=l: st[l][:, [pos]])) for l in L]
    with edits(model, specs):
        _, lg1 = answer(model, tok, ids)
    assert torch.allclose(lg0, lg1, atol=1e-4)
    r, d = torch.randn(1, 128), torch.randn(1, 128)
    g = torch.Generator().manual_seed(0)
    nl = signed_permutation_null(r, d, n_kv=2, generator=g)
    for k in range(2):
        sl = slice(64 * k, 64 * (k + 1))
        assert torch.allclose((nl - r)[:, sl].norm(), (d - r)[:, sl].norm(), atol=1e-5)

def test_rotated_subspace_orthonormal():
    s = RotatedSubspace(32, 4, init=pca_basis(torch.randn(50, 32), 4))
    assert torch.allclose(s.U @ s.U.T, torch.eye(4), atol=1e-5)
