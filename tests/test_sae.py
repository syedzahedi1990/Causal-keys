"""Part of Gate J-C-G0 of P-2026-10-10-J part C (docs/PREREGISTRATION.md): the BatchTopK dictionary code of family E3
(ckeys/sae.py) against direct references on small random dictionaries (no model, no download): the encoder with the
threshold mask, the decoder orientation (d_j = decoder.weight[:, j]), loading the published dictionary_learning state-dict
layout with its sha256, the streaming FVE, the selectivity ranking, the E3 edit (latents of F_t set to beta abar, those of
F_B outside F_t set to 0, everything else and the SAE error kept at B), and an edit over every differing latent with
zero reconstruction error reproducing h_t. The SAE pins are checked for form."""
import hashlib

import pytest
import torch

from ckeys import sae

torch.manual_seed(0)


def small(d=16, m=64, seed=1, threshold=0.1):
    g = torch.Generator().manual_seed(seed)
    return sae.BatchTopK(torch.randn(m, d, generator=g), torch.randn(m, generator=g) * 0.1, torch.randn(d, m, generator=g),
                         torch.randn(d, generator=g) * 0.1, threshold)


def test_encode_decode_reference():
    S = small()
    h = torch.randn(5, 16)
    pre = torch.relu((h - S.b_dec) @ S.W_enc.T + S.b_enc)
    ref = torch.where(pre > S.threshold, pre, torch.zeros_like(pre))
    a = S.encode(h)
    assert torch.allclose(a, ref, atol=1e-6)
    assert (a[(pre > 0) & (pre <= S.threshold)] == 0).all() and ((pre > 0) & (pre <= S.threshold)).any(), "threshold mask not exercised"
    # decoder orientation: decode(e_j) - b_dec is column j of decoder.weight [d, m]
    for j in (0, 7, 63):
        e = torch.zeros(64)
        e[j] = 1.0
        assert torch.allclose(S.decode(e) - S.b_dec, S.W_dec[:, j], atol=1e-6)
        assert torch.equal(S.col(j), S.W_dec[:, j])
    assert torch.allclose(S.decode(a), a @ S.W_dec.T + S.b_dec, atol=1e-6)


def test_load_state_dict_layout_and_hash(tmp_path):
    S = small()
    sd = {"encoder.weight": S.W_enc, "encoder.bias": S.b_enc, "decoder.weight": S.W_dec, "b_dec": S.b_dec,
          "threshold": torch.tensor(S.threshold), "k": torch.tensor(8)}
    f = tmp_path / "ae.pt"
    torch.save(sd, f)
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    L = sae.BatchTopK.load(f, sha256=h)
    x = torch.randn(3, 16)
    assert torch.allclose(L.encode(x), S.encode(x)) and L.k == 8 and abs(L.threshold - S.threshold) < 1e-7
    with pytest.raises(AssertionError):
        sae.BatchTopK.load(f, sha256="0" * 64)


def test_fve_streaming_equals_direct():
    S = small()
    h = torch.randn(40, 16)
    hh = S.decode(S.encode(h))
    hd, rd = h.double(), (h - hh).double()
    direct = 1 - rd.var(0, unbiased=False).sum() / hd.var(0, unbiased=False).sum()
    f = sae.FVE()
    f.add(h[:13], hh[:13])
    f.add(h[13:], hh[13:])
    assert abs(f.value() - float(direct)) < 1e-6 * max(1.0, abs(float(direct)))
    assert abs(sae.fve(S, h) - float(direct)) < 1e-6 * max(1.0, abs(float(direct)))


def test_selectivity_and_select():
    abar = torch.tensor([[1.0, 0.0, 0.5, 0.2], [0.2, 0.3, 0.6, 0.0], [0.0, 0.1, 0.0, 0.1]])
    s = sae.selectivity(abar)
    assert torch.allclose(s[0], torch.tensor([0.8, -0.3, -0.1, 0.1]))
    assert torch.allclose(s[1], torch.tensor([-0.8, 0.2, 0.1, -0.2]))
    F = sae.select(abar, 2)
    assert F[0].tolist() == [0, 3] and F[1].tolist() == [1, 2] and F[2].tolist() == []   # only s > 0 are kept


def test_edit_reference_and_kept_latents():
    S = small(threshold=0.0)
    hB = torch.randn(16)
    abar = torch.rand(6, 64)
    F = [torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5])] + [torch.tensor([], dtype=torch.long)] * 4
    t, b, beta = 0, 1, 2.0
    out = sae.edit(S, hB, t, b, F, abar, beta)
    a = S.encode(hB)
    coef = torch.zeros(64)
    coef[[1, 2, 3]] = beta * abar[0, [1, 2, 3]] - a[[1, 2, 3]]
    coef[[4, 5]] = -a[[4, 5]]                                   # F_B minus F_t; latent 3 is in both and set to beta abar
    assert torch.allclose(out, hB + S.W_dec @ coef, atol=1e-5)
    # the edit changes the reconstruction by exactly the target code minus B's code; the SAE error stays at B's
    tgt = sae.edited_latents(S, hB, t, b, F, abar, beta)
    err_B = hB - S.decode(a)
    assert torch.allclose(out - S.decode(tgt), err_B, atol=1e-5)
    # batched rows give the same as single rows
    o2 = sae.edit(S, torch.stack([hB, hB]), t, b, F, abar, beta)
    assert torch.allclose(o2[0], out, atol=1e-6) and torch.allclose(o2[1], out, atol=1e-6)


def test_full_latent_edit_reproduces_target_with_zero_error():
    """An orthonormal dictionary (W_enc = W_dec^T, no encoder bias, threshold 0) reconstructs any code a >= 0 exactly;
    editing every differing latent of h_B toward h_t's code gives h_t."""
    d, m = 64, 16
    W = torch.linalg.qr(torch.randn(d, m))[0]
    S = sae.BatchTopK(W.T.contiguous(), torch.zeros(m), W, torch.randn(d) * 0.1, 0.0)
    aB, aT = torch.relu(torch.randn(m)), torch.relu(torch.randn(m))
    hB, hT = S.decode(aB), S.decode(aT)
    assert torch.allclose(S.encode(hB), aB, atol=1e-5) and torch.allclose(S.encode(hT), aT, atol=1e-5)
    Ft = torch.nonzero(aT).flatten()
    Fb = torch.nonzero(aB).flatten()
    abar = torch.stack([aT, aB] + [torch.zeros(m)] * 4)
    out = sae.edit(S, hB, 0, 1, [Ft, Fb] + [torch.tensor([], dtype=torch.long)] * 4, abar, 1.0)
    assert torch.allclose(out, hT, atol=1e-5)


def test_random_and_pins():
    R = sae.BatchTopK.random(32, 128, seed=3)
    assert R.W_dec.shape == (32, 128) and torch.allclose(R.W_dec.norm(dim=0), torch.ones(128), atol=1e-5)
    assert set(sae.PINS) == set(sae.LAYERS) == {3, 7, 11, 15}
    for l, p in sae.PINS.items():
        assert all(len(p[k]) == 64 and int(p[k], 16) >= 0 for k in ("ae.pt", "config.json", "eval_results.json"))
        assert 0.7 < p["fve"] < 1.0
    assert len(sae.REVISION) == 40 and sae.TRAINER == "trainer_1"
