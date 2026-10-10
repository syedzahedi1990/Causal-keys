"""The third-party sparse autoencoder of stage 8, part C (preregistration J, family E3): BatchTopK dictionaries of
andyrdt/saes-qwen2.5-7b-instruct (the dictionary_learning ``BatchTopKSAE`` layout), trained on the output of 0-based
block l of Qwen2.5-7B-Instruct (config: io = "out", submodule resid_post_layer_<l>), trainer_1 (k = 64, 131,072 latents).

State dict of ae.pt (checked on the pinned file's pickle header): encoder.weight [m, d], encoder.bias [m],
decoder.weight [d, m], b_dec [d], threshold [] (one global threshold), k [] (int).
  encode a(h) = relu(W_enc (h - b_dec) + b_enc) * [relu(.) > threshold]   (the inference path, use_threshold=True)
  decode h_hat = W_dec a + b_dec,  d_j = decoder.weight[:, j]
FVE (dictionary_learning's frac_variance_explained) = 1 - sum_dim var(h - h_hat) / sum_dim var(h) over a set of vectors.

Selection and the edit (entry, family E3):
  abar_j(x) = mean a_j(h_x,l(p)) over the fit cores (all six values written at p); s_j(x) = abar_j(x) - max_{y != x} abar_j(y);
  F_x = the top-k_F latents by s_j(x) among those with s_j(x) > 0.
  edit toward t from the base B:  h <- h_B + sum_{j in F_t} (beta abar_j(t) - a_j(h_B)) d_j - sum_{j in F_B \\ F_t} a_j(h_B) d_j,
  so the latents of F_t are set to beta abar_j(t), those of F_B outside F_t to 0, and every other latent and the SAE error
  stay at B's.
The files are fetched at the pinned revision (REPO, REVISION) and every file's sha256 is asserted (PINS).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

REPO = "andyrdt/saes-qwen2.5-7b-instruct"
REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"     # the repository head on 2026-10-10 (last modified 2025-05-26)
TRAINER = "trainer_1"
LAYERS = (3, 7, 11, 15)
# per layer: sha256 of ae.pt (the LFS oid on the Hub), config.json and eval_results.json at REVISION; the published FVE
PINS = {
    3: {"ae.pt": "93f70d8aac4976fa9193b3c108b85fe94b9792643fe16c43d2be160a2098ed4f",
        "config.json": "5667fd2bd9c188927eba6c034cddf474c7ae84a1f33df485de398847d120c382",
        "eval_results.json": "1986b2b392dff5a064ea6c45f03eb4be8faffac53a1714096af1a97753aca259", "fve": 0.93087890625},
    7: {"ae.pt": "94be36b5ba215103b6cad7e05edaed097d677c70ff886d170a628744d6204092",
        "config.json": "19b6c408283063c15e1d1d44b68d4bc618fead3cd8526e828f2b29a2476be161",
        "eval_results.json": "4892e73fb50f093976915ff5b9b3f37b389d4df1e3d55d0b8355be0a34eddc54", "fve": 0.862109375},
    11: {"ae.pt": "36bddbd229d59c11ed61f77a95a89da3ed141214e891677bb5d7facf64b57100",
         "config.json": "bd21d02ae9d119d2f8536dab177e50ad10fbf8f5c66977dbb611351e60e9f83d",
         "eval_results.json": "9eb49624a999891a47c5f3e05d8312e5b58652c23007e715b7096ae7aa859ed1", "fve": 0.82669921875},
    15: {"ae.pt": "2efefadd8d85ad1a2bfcf3cb5eceb2976590dc88b4873e56b864c42c387f5cc5",
         "config.json": "9fd2d432683dfb847220a00262195dd2c324e3050bd701c9bfb05bd9fde1a116",
         "eval_results.json": "176ad5c35b2e098e88da0a1fcb14aa51ae457ea2202f68ad67d05775f4993e39", "fve": 0.8069140625},
}
FVE_MARGIN = 0.05            # Gate J-C-G3: FVE >= published - 0.05
KF_GRID = (4, 8, 16, 32, 64)
BETA_GRID = (2, 4)


def sha_file(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(layer: int, dest) -> dict:
    """Download the three files of layer ``layer`` at REVISION into ``dest`` (huggingface_hub, resumable) and assert
    each sha256 against PINS. Returns {file: local path} plus the hashes. A file already present is re-hashed."""
    from huggingface_hub import hf_hub_download
    rec = {}
    for fn in ("config.json", "eval_results.json", "ae.pt"):
        path = hf_hub_download(REPO, f"resid_post_layer_{layer}/{TRAINER}/{fn}", revision=REVISION, local_dir=str(dest))
        h = sha_file(path)
        assert h == PINS[layer][fn], f"SAE layer {layer} {fn}: sha256 {h} != pinned {PINS[layer][fn]}"
        rec[fn] = {"path": str(path), "sha256": h}
    cfg = json.loads(Path(rec["config.json"]["path"]).read_text())
    assert cfg["buffer"]["io"] == "out" and cfg["trainer"]["layer"] == layer, cfg
    ev = json.loads(Path(rec["eval_results.json"]["path"]).read_text())
    rec["published_fve"] = ev["frac_variance_explained"]
    assert abs(rec["published_fve"] - PINS[layer]["fve"]) < 1e-9
    return rec


class BatchTopK:
    """A BatchTopK dictionary at its inference path (fixed threshold). Tensors in FP32 on ``device``."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold, k=None):
        self.W_enc, self.b_enc, self.W_dec, self.b_dec = W_enc, b_enc, W_dec, b_dec   # [m, d], [m], [d, m], [d]
        self.threshold = float(threshold)
        self.k = None if k is None else int(k)
        self.m, self.d = W_enc.shape
        assert W_dec.shape == (self.d, self.m) and b_enc.shape == (self.m,) and b_dec.shape == (self.d,)

    @classmethod
    def load(cls, path, sha256: str | None = None, device="cpu"):
        if sha256 is not None:
            h = sha_file(path)
            assert h == sha256, f"{path}: sha256 {h} != {sha256}"
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        t = lambda k: sd[k].to(device=device, dtype=torch.float32)  # noqa: E731
        return cls(t("encoder.weight"), t("encoder.bias"), t("decoder.weight"), t("b_dec"), float(sd["threshold"]),
                   int(sd["k"]) if "k" in sd else None)

    @classmethod
    def random(cls, d, m, seed=0, threshold=0.05, device="cpu"):
        """A random dictionary (unit-norm decoder columns, W_enc = W_dec^T) for TEST_MODE and the unit tests."""
        g = torch.Generator().manual_seed(seed)
        W_dec = torch.randn(d, m, generator=g)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        return cls(W_dec.T.contiguous().to(device), (0.01 * torch.randn(m, generator=g)).to(device), W_dec.to(device),
                   (0.01 * torch.randn(d, generator=g)).to(device), threshold)

    def pre(self, h):
        return torch.relu((h.float() - self.b_dec) @ self.W_enc.T + self.b_enc)

    def encode(self, h):
        a = self.pre(h)
        return a * (a > self.threshold)

    def decode(self, a):
        return a.float() @ self.W_dec.T + self.b_dec

    def col(self, j):
        return self.W_dec[:, j]


class FVE:
    """Streaming FVE over batches of vectors (dictionary_learning's definition, variances about the set's own mean)."""

    def __init__(self):
        self.n, self.s1, self.s2, self.e1, self.e2 = 0, 0.0, 0.0, 0.0, 0.0

    def add(self, h, h_hat):
        h, r = h.double(), (h - h_hat).double()
        self.n += h.shape[0]
        self.s1, self.s2 = self.s1 + h.sum(0), self.s2 + (h * h).sum(0)
        self.e1, self.e2 = self.e1 + r.sum(0), self.e2 + (r * r).sum(0)

    def value(self) -> float:
        var = lambda s1, s2: (s2 / self.n - (s1 / self.n) ** 2).sum()  # noqa: E731
        return float(1.0 - var(self.e1, self.e2) / var(self.s1, self.s2))


def fve(sae: BatchTopK, h) -> float:
    f = FVE()
    f.add(h, sae.decode(sae.encode(h)))
    return f.value()


def selectivity(abar):
    """abar [6, m] (mean activation per value) -> s [6, m], s[x, j] = abar[x, j] - max_{y != x} abar[y, j]."""
    out = torch.empty_like(abar)
    for x in range(abar.shape[0]):
        others = torch.cat([abar[:x], abar[x + 1:]])
        out[x] = abar[x] - others.max(0).values
    return out


def select(abar, kF: int):
    """The latent sets F_x (top-kF by selectivity with s > 0), one LongTensor per value."""
    s = selectivity(abar)
    F = []
    for x in range(abar.shape[0]):
        v, i = s[x].topk(min(kF, s.shape[1]))
        F.append(i[v > 0].clone())
    return F


def edit(sae: BatchTopK, hB, t: int, b: int, F, abar, beta: float = 1.0):
    """The E3 edit of the residual hB [d] (or [R, d] with one (t, b) for every row) toward value t from base value b:
    returns hB + sum_{j in F_t} (beta abar_j(t) - a_j(hB)) d_j - sum_{j in F_b \\ F_t} a_j(hB) d_j (FP32, cast back)."""
    a = sae.encode(hB)
    Ft, Fb = F[t], F[b]
    drop = Fb[~torch.isin(Fb, Ft)]
    coef = torch.zeros_like(a)
    coef[..., Ft] = beta * abar[t, Ft].to(a) - a[..., Ft]
    coef[..., drop] = -a[..., drop]
    return (hB.float() + coef @ sae.W_dec.T).to(hB.dtype)


def edited_latents(sae: BatchTopK, hB, t: int, b: int, F, abar, beta: float = 1.0):
    """The target code of the edit (the latents the edit writes, every other latent of hB unchanged) - for the tests."""
    a = sae.encode(hB).clone()
    Ft, Fb = F[t], F[b]
    drop = Fb[~torch.isin(Fb, Ft)]
    a[..., Ft] = beta * abar[t, Ft].to(a)
    a[..., drop] = 0.0
    return a
