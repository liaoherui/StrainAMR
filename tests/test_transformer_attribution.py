"""Planted-signal tests for library/transformer_attribution.py.

Ground truth
------------
resistant  <=>  SNV token A present   OR   (PC tokens B AND C both present)
k-mer tokens K1, K2, K3 are perfect tags of A (present iff A present), so they are
mutually redundant.

Expected
--------
* IG ranks A top in the SNV modality and the K tags top in the k-mer modality.
* Repacked occlusion flags both B and C (the trained model partly encodes B through
  the *position* of C, which in-place masking cannot see).
* Occlusion interaction for (B, C) is positive (synergy / AND).
* Interaction among K1/K2/K3 is <= 0 (redundancy / OR).
* The linear fusion head gives an exactly additive modality decomposition.
* IG completeness residual is small relative to the logit range.

Run:  python -m pytest tests/test_transformer_attribution.py -q   (or python tests/...)
"""

import os
import random
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import Transformer_without_pos_multimodal_add_attn as TM  # noqa: E402
from library import transformer_attribution as TA  # noqa: E402

A, B, C = 5, 7, 9
K = (11, 12, 13)
L1, L2, L3 = 24, 24, 30


def make_data(n, seed):
    rng = random.Random(seed)
    x1 = np.zeros((n, L1), int)
    x2 = np.zeros((n, L2), int)
    x3 = np.zeros((n, L3), int)
    y = np.zeros(n, int)
    for i in range(n):
        has_a = rng.random() < 0.3
        has_b = rng.random() < 0.5
        has_c = rng.random() < 0.5
        s1 = [t for t in range(20, 60) if rng.random() < 0.25]
        s2 = [t for t in range(20, 60) if rng.random() < 0.25]
        if has_a:
            s1.append(A)
        if has_b:
            s2.append(B)
        if has_c:
            s2.append(C)
        s1 = sorted(s1)[:L1]
        s2 = sorted(s2)[:L2]
        x1[i, :len(s1)] = s1
        x2[i, :len(s2)] = s2
        # k-mer modality uses fixed positions (token id at its own slot)
        for t in range(1, L3):
            if t in K:
                present = has_a
            else:
                present = rng.random() < 0.4
            x3[i, t] = t if present else 0
        y[i] = int(has_a or (has_b and has_c))
    return [torch.tensor(x1), torch.tensor(x2), torch.tensor(x3)], torch.tensor(y)


def train_model(seed=0):
    torch.manual_seed(seed)
    TM.device = torch.device("cpu")
    model = TM.Transformer(src_vocab_size_1=64, src_vocab_size_2=64, src_vocab_size_3=40, src_pad_idx=0,
                           embed_size=32, heads=4, dropout=0.1, device="cpu",
                           max_length_1=L1, max_length_2=L2, max_length_3=L3)
    xs, y = make_data(600, seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(30):
        model.train()
        perm = torch.randperm(len(y))
        for s in range(0, len(y), 32):
            idx = perm[s:s + 32]
            out = model(*[x[idx] for x in xs])[0].squeeze(1)
            loss = lossf(out, y[idx].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = train_model()
    return _MODEL


ENC = ["encoder1", "encoder2", "encoder3"]


def _engine(xs, target="logit"):
    eng = TA.TransformerAttributor(get_model(), ENC, device=torch.device("cpu"), target=target)
    eng.set_layout(xs)
    return eng


def test_layout_detection():
    xs, _ = make_data(100, 1)
    assert [TA.detect_packed(x) for x in xs] == [True, True, False]


def test_accuracy_and_ig_rankings():
    xs, y = make_data(300, 123)
    eng = _engine(xs)
    logits = eng.logits(xs).numpy()
    acc = float(((logits > 0).astype(int) == y.numpy()).mean())
    assert acc > 0.95, acc
    res = eng.token_attributions(xs, batch_size=16, steps=24)
    s1 = TA.summarise_tokens(res["ig"][0], y.numpy())
    s2 = TA.summarise_tokens(res["ig"][1], y.numpy())
    s3 = TA.summarise_tokens(res["ig"][2], y.numpy())
    assert TA.rank_tokens(s1)[0] == A
    assert s1[A]["MeanAttr_present"] > 0
    assert C in TA.rank_tokens(s2)[:3]
    assert set(TA.rank_tokens(s3)[:3]) == set(K)
    spread = np.ptp(res["logit"])
    assert np.median(np.abs(res["completeness_residual"])) < 0.05 * spread
    eng.close()
    return acc


def test_occlusion_recovers_both_and_partners():
    """Repacked occlusion must flag both B and C; in-place masking can miss B."""
    xs, y = make_data(300, 123)
    eng = _engine(xs)
    occ = eng.occlusion(xs, 1, [B, C, 20, 21, 22, 23])
    top2 = sorted(occ, key=lambda t: -occ[t]["LOO_flip"])[:2]
    assert set(top2) == {B, C}, occ
    # the pitfall: in-place masking (hole) on the same model
    sel = [i for i in range(300) if {B, C} <= set(xs[1][i].tolist()) and A not in xs[0][i].tolist()]
    base = eng.logits([x[sel] for x in xs]).numpy()
    rows = torch.stack([eng.remove_tokens(xs[1][i], [B], repack=True) for i in sel])
    after = eng.logits([xs[0][sel], rows, xs[2][sel]]).numpy()
    flip = float(np.mean((base > 0) != (after > 0)))
    assert flip > 0.9, flip
    eng.close()


def test_modality_additivity():
    xs, _ = make_data(64, 7)
    eng = _engine(xs)
    dec = eng.modality_decomposition(xs)
    assert np.max(np.abs(dec["additivity_residual"])) < 1e-3
    # the per-modality fast path must reproduce full forward passes exactly
    full = eng.logits(xs).numpy()
    rows = torch.stack([eng.remove(xs[1][i], [B, 21], 1) for i in range(64)])
    fast = eng.replaced_logits(xs, 1, list(range(64)), rows)
    slow = eng.logits([xs[0], rows, xs[2]]).numpy()
    assert np.max(np.abs(fast - slow)) < 1e-3 * max(1.0, np.abs(full).max())
    eng.close()


def test_pair_interactions():
    xs, _ = make_data(300, 99)
    eng = _engine(xs)
    pc = eng.pair_interactions(xs, 1, [B, C, 21, 22, 23], max_samples=150)
    bc = pc[(B, C)]["prob_mean"]
    others = [v["prob_mean"] for k, v in pc.items() if k != (B, C)]
    assert bc > 0.2 and bc > max(abs(o) for o in others), (bc, others)
    km = eng.pair_interactions(xs, 2, list(K), max_samples=150)
    assert all(v["logit_mean"] <= 0.05 for v in km.values()), km
    eng.close()


def test_faithfulness_ordering():
    xs, y = make_data(200, 5)
    eng = _engine(xs)
    res = eng.token_attributions(xs, batch_size=16, steps=16)
    rows = TA.deletion_curve(eng, xs, y.numpy(), 0,
                             {"ig_local": res["ig"][0].per_sample, "random": "random"}, ks=(1, 2))
    # A's k-mer tags keep the prediction resistant, so decisions rarely flip
    # (cross-modality redundancy); the loss of logit support must still be larger.
    d = {(r["ranking"], r["k"]): r["support_drop_logit"] for r in rows}
    assert d[("ig_local", 1)] > 3 * d[("random", 1)], d
    eng.close()


if __name__ == "__main__":
    test_layout_detection()
    print("accuracy", test_accuracy_and_ig_rankings())
    test_occlusion_recovers_both_and_partners()
    test_modality_additivity()
    test_pair_interactions()
    test_faithfulness_ordering()
    print("all planted-signal tests passed")


def test_pipeline_end_to_end_all_model_variants(tmp_path=None):
    """Smoke test of library/interpret_pipeline.run for 1-, 2- and 3-encoder models."""
    import tempfile
    from library import interpret_pipeline as IP

    xs, y = make_data(80, 3)
    xe, ye = make_data(40, 4)
    arrays = [x.numpy() for x in xs]
    earr = [x.numpy() for x in xe]
    lengths = [L1, L2, L3]
    for labels in (["snv"], ["snv", "kmer"], ["snv", "pc", "kmer"]):
        idx = [["snv", "pc", "kmer"].index(l) for l in labels]
        tr = [arrays[i] for i in idx]
        ev = [IP.remove_unseen(arrays[i], earr[i]) for i in idx]
        vocab = [int(a.max()) + 2 for a in tr]
        model, enc = IP.build_model(labels, vocab, lengths, torch.device("cpu"), embed_size=16, heads=4)
        model.eval()
        out = tempfile.mkdtemp() if tmp_path is None else str(tmp_path / "_".join(labels))
        opts = IP.InterpretOptions(steps=4, batch_size=8, occlusion_top=5, pair_top=4, pair_samples=10,
                                   ks=(1, 2), random_repeats=1, sanity=True, rf_shap=False,
                                   log=lambda m: None)
        summ = IP.run(model, enc, labels, tr, y.numpy(), [f"s{i}" for i in range(80)], out, opts,
                      eval_arrays=ev, y_eval=ye.numpy(), ids_eval=[f"e{i}" for i in range(40)],
                      device=torch.device("cpu"))
        for lab in labels:
            assert os.path.exists(os.path.join(out, f"{lab}_token_attribution.tsv"))
            assert os.path.exists(os.path.join(out, f"{lab}_pair_interaction.tsv"))
        assert os.path.exists(os.path.join(out, "faithfulness_deletion.tsv"))
        assert summ["max_additivity_residual"]["train"] < 1e-3
