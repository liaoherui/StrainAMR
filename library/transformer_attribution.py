"""Attribution computed directly on the StrainAMR Transformer classifier.

Why this module exists
----------------------
The SHAP rankings produced during feature building explain an *auxiliary random
forest*, not the neural classifier that makes StrainAMR's predictions.  This module
closes that gap.  Every quantity below is computed on the trained Transformer itself
and is expressed in the units of its resistance logit, so values are comparable
across tokens, samples and modalities.

What is computed
----------------
1. Token attribution (per sample, every token, signed)
   Integrated Gradients (Sundararajan et al., 2017) on the token embeddings, with
   the padding-token embedding as baseline.  Positive values push the prediction
   towards "resistant".  ``method="gxi"`` gives the single-step (gradient x delta)
   approximation for very large runs.  A completeness residual is reported for
   every sample so the quality of the integral can be checked.
2. Exact modality decomposition
   The fusion head of all StrainAMR models is ``Linear -> BatchNorm -> Linear`` with
   no non-linearity.  In eval mode the logit is therefore exactly additive over
   modalities: ``f(x) = sum_m g_m(x_m) + b``.  The contribution of modality ``m`` is
   ``f(x) - f(x with modality m fully masked)`` and the contributions sum exactly
   to ``f(x) - f(all masked)``.  A corollary: cross-modality interactions are
   identically zero in this architecture, so pair analysis is within-modality.
3. Pair interaction (within modality, by occlusion)
   ``I(i, j) = f(x) - f(x\\i) - f(x\\j) + f(x\\ij)`` where ``x\\i`` removes every
   occurrence of token ``i`` (token -> padding id, which is also excluded as an
   attention key).  ``I > 0``: synergy (both needed, AND-like, e.g. epistasis);
   ``I < 0``: redundancy (either suffices, OR-like, e.g. several k-mers that tag
   one allele).  Head-averaged attention between the two tokens is reported
   alongside, so attention-based and effect-based pair scores can be compared.
4. Attention received
   Mean attention each token receives (last layer, averaged over heads and valid
   query positions).  Provided as a *baseline explanation* for the faithfulness
   test, not as a recommended importance measure.

How a token is "removed"
------------------------
The classifier flattens the encoder output position-by-position into ``fc1``, so it
is position-aware even without positional embeddings.  SNV-graph and protein-cluster
tokens are *packed* (left-aligned, padding at the end), so the presence of a token
is also encoded by the positions of every token after it.  Masking a token in place
(leaving a hole) therefore does NOT reproduce the input of a genome that lacks the
feature, and can miss features entirely (see tests/test_transformer_attribution.py).
Removal here re-packs packed modalities (delete, shift left, pad at the end) and
zeroes in place only for fixed-slot modalities such as the k-mer panel.  The layout
of each modality is auto-detected (``detect_packed``).  Integrated Gradients, being
an in-place embedding interpolation, cannot see this positional channel; use it as
a fast screen and rely on occlusion for effect sizes.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

PAD_ID = 0


# --------------------------------------------------------------------------- utils
def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip the ``module.`` prefix that DataParallel adds to checkpoint keys."""
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _logit(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.reshape(output.shape[0])


@dataclass
class TokenTable:
    """Per-sample token attributions for one modality (token_id -> score)."""

    per_sample: List[Dict[int, float]] = field(default_factory=list)


# ------------------------------------------------------------------------- engine
class TransformerAttributor:
    """Attribution engine for the StrainAMR single/dual/triple-encoder models."""

    def __init__(
        self,
        model: nn.Module,
        encoder_names: Sequence[str],
        device: Optional[torch.device] = None,
        pad_id: int = PAD_ID,
        max_rows: int = 32,
        target: str = "logit",
    ) -> None:
        if target not in ("logit", "prob"):
            raise ValueError("target must be 'logit' or 'prob'")
        self.target = target
        self.model = unwrap(model)
        self.encoder_names = list(encoder_names)
        self.encoders = [getattr(self.model, n) for n in self.encoder_names]
        self.device = device or next(self.model.parameters()).device
        self.pad_id = pad_id
        self.max_rows = max(1, int(max_rows))
        self.model.eval()
        # gradients are only needed w.r.t. the (leaf) interpolated embeddings
        self._req_grad = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._alpha: Optional[torch.Tensor] = None
        self._captured: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.encoders)
        self._attn_mode: Optional[str] = None  # None | "received" | "full"
        self._attn_out: List[Optional[torch.Tensor]] = [None] * len(self.encoders)
        self._handles = []
        for m, enc in enumerate(self.encoders):
            self._handles.append(enc.word_embedding.register_forward_hook(self._emb_hook(m)))
            layers = list(getattr(enc, "layers", []))
            if layers:
                self._handles.append(layers[-1].attention.register_forward_hook(self._attn_hook(m)))

    # ---------------------------------------------------------------- hooks --
    def _emb_hook(self, m: int):
        def hook(module, inputs, out):
            if self._alpha is None:
                return None
            base = module.weight[self.pad_id].detach()
            delta = (out - base).detach()
            a = self._alpha.to(out.dtype).view(-1, 1, 1)
            new = (base + a * delta).detach().requires_grad_(True)
            self._captured[m] = (new, delta)
            return new

        return hook

    def _attn_hook(self, m: int):
        def hook(module, inputs, out):
            if self._attn_mode is None:
                return
            values, keys, query, mask = inputs
            n = query.shape[0]
            h, d = module.heads, module.head_dim
            with torch.no_grad():
                k = module.keys(keys.reshape(n, -1, h, d))
                q = module.queries(query.reshape(n, -1, h, d))
                energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
                if mask is not None:
                    energy = energy.masked_fill(mask == 0, float("-1e20"))
                att = torch.softmax(energy / (module.embed_size ** 0.5), dim=3).mean(1)  # (n, q, k)
                if self._attn_mode == "full":
                    self._attn_out[m] = att.detach()
                else:
                    valid = (mask.reshape(n, -1) != 0).to(att.dtype) if mask is not None else torch.ones(
                        att.shape[:2], device=att.device)
                    recv = (att * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
                    self._attn_out[m] = recv.detach()

        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for p, r in zip(self.model.parameters(), getattr(self, "_req_grad", [])):
            p.requires_grad_(r)

    # --------------------------------------------------------------- basics --
    def _f(self, logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit) if self.target == "prob" else logit

    def _to_dev(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [x.to(self.device).long() for x in inputs]

    @torch.no_grad()
    def logits(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Batched logits, chunked by ``max_rows``."""
        n = inputs[0].shape[0]
        out = []
        for s in range(0, n, self.max_rows):
            chunk = self._to_dev([x[s:s + self.max_rows] for x in inputs])
            out.append(_logit(self.model(*chunk)).float().cpu())
        return torch.cat(out) if out else torch.zeros(0)

    # ------------------------------------------------ exact per-modality forward --
    def _head(self):
        """Collapse fc1 -> BatchNorm(eval) -> out into one weight vector per modality."""
        if getattr(self, "_head_cache", None) is not None:
            return self._head_cache
        mdl = self.model
        fc1, bn, out = mdl.fc1, mdl.bn, mdl.out
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = (out.weight.view(-1) * scale) @ fc1.weight  # (D_total,)
        c = out.weight.view(-1) @ (scale * (fc1.bias - bn.running_mean) + bn.bias) + out.bias.view(-1)
        sizes = [e.word_embedding.embedding_dim for e in self.encoders]
        self._head_cache = (w.detach(), float(c), sizes)
        return self._head_cache

    @torch.no_grad()
    def modality_score(self, x_m: torch.Tensor, m: int) -> torch.Tensor:
        """Additive term g_m(x_m) of the logit (exact; only encoder m is run)."""
        w, _, sizes = self._head()
        lens = getattr(self, "_lens", None)
        if lens is None:
            raise RuntimeError("call set_layout() first")
        offs, start = [], 0
        for j, L in enumerate(lens):
            offs.append((start, start + L * sizes[j]))
            start += L * sizes[j]
        a, b = offs[m]
        wm = w[a:b]
        out = []
        for s0 in range(0, x_m.shape[0], self.max_rows):
            x = x_m[s0:s0 + self.max_rows].to(self.device).long()
            enc = self.encoders[m](x, self.model.make_src_mask(x))
            if isinstance(enc, (tuple, list)):
                enc = enc[0]
            out.append((enc.reshape(enc.shape[0], -1) @ wm).float().cpu())
        return torch.cat(out) if out else torch.zeros(0)

    def replaced_logits(self, tensors: Sequence[torch.Tensor], m: int, idx: Sequence[int],
                        new_rows: torch.Tensor, base_logits: Optional[np.ndarray] = None,
                        base_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Logits of samples ``idx`` after replacing modality ``m`` by ``new_rows``."""
        idx = list(idx)
        if base_logits is None:
            bl = self.logits([t[idx] for t in tensors]).numpy()
        else:
            bl = np.asarray(base_logits)[idx]
        if base_scores is None:
            bs = self.modality_score(tensors[m][idx], m).numpy()
        else:
            bs = np.asarray(base_scores)[idx]
        return bl - bs + self.modality_score(new_rows, m).numpy()

    @staticmethod
    def remove_tokens(x: torch.Tensor, token_ids: Sequence[int], repack: bool = False) -> torch.Tensor:
        """Copy of 1-D token row ``x`` without ``token_ids``.

        ``repack=False`` sets them to PAD in place (fixed-slot layout);
        ``repack=True`` deletes them and shifts later tokens left (packed layout).
        """
        if not len(token_ids):
            return x.clone()
        ids = torch.as_tensor(list(token_ids), dtype=x.dtype)
        hit = torch.isin(x, ids)
        if not repack:
            y = x.clone()
            y[hit] = PAD_ID
            return y
        keep = x[(~hit) & (x != PAD_ID)]
        y = torch.zeros_like(x)
        y[: keep.numel()] = keep
        return y

    def set_layout(self, tensors: Sequence[torch.Tensor]) -> List[bool]:
        """Detect packed vs fixed-slot layout for each modality from data."""
        self.packed = [detect_packed(t) for t in tensors]
        self._lens = [int(t.shape[1]) for t in tensors]
        return self.packed

    def remove(self, x: torch.Tensor, token_ids: Sequence[int], modality: int) -> torch.Tensor:
        packed = getattr(self, "packed", None)
        repack = bool(packed[modality]) if packed is not None else True
        return self.remove_tokens(x, token_ids, repack=repack)

    # ---------------------------------------------------- integrated gradients --
    def integrated_gradients(
        self,
        inputs: Sequence[torch.Tensor],
        steps: int = 16,
        method: str = "ig",
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Position-level IG for one batch.

        Returns ``(attr, logit_x, logit_baseline)`` where ``attr[m]`` has shape
        ``(B, L_m)`` and sums (over positions and modalities) to approximately
        ``logit_x - logit_baseline``.
        """
        inputs = self._to_dev(inputs)
        b = inputs[0].shape[0]
        if method == "gxi":
            alphas = torch.ones(1)
        else:
            steps = max(1, int(steps))
            alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps  # midpoint rule
        s_tot = alphas.numel()
        grad_sum: List[Optional[torch.Tensor]] = [None] * len(inputs)
        deltas: List[Optional[torch.Tensor]] = [None] * len(inputs)

        steps_per_chunk = max(1, self.max_rows // max(1, b))
        try:
            for s0 in range(0, s_tot, steps_per_chunk):
                a = alphas[s0:s0 + steps_per_chunk]
                k = a.numel()
                rep = [x.repeat_interleave(k, dim=0) for x in inputs]  # sample-major
                self._alpha = a.repeat(b).to(self.device)
                with torch.enable_grad():
                    out = self._f(_logit(self.model(*rep)))
                    caps = [c[0] for c in self._captured]
                    grads = torch.autograd.grad(out.sum(), caps)
                for m, g in enumerate(grads):
                    g = g.float().view(b, k, *g.shape[1:]).sum(1)
                    grad_sum[m] = g if grad_sum[m] is None else grad_sum[m] + g
                    if deltas[m] is None:
                        deltas[m] = self._captured[m][1].float().view(b, k, *g.shape[1:])[:, 0]
                self._captured = [None] * len(self.encoders)
        finally:
            self._alpha = None

        attr = [((grad_sum[m] / s_tot) * deltas[m]).sum(-1).cpu() for m in range(len(inputs))]
        with torch.no_grad():
            fx = self._f(_logit(self.model(*inputs))).float().cpu()
            self._alpha = torch.zeros(b, device=self.device)
            try:
                f0 = self._f(_logit(self.model(*inputs))).float().cpu()
            finally:
                self._alpha = None
                self._captured = [None] * len(self.encoders)
        return attr, fx, f0

    # ------------------------------------------------------ attention received --
    @torch.no_grad()
    def attention_received(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        inputs = self._to_dev(inputs)
        self._attn_mode = "received"
        try:
            self.model(*inputs)
            res = [a.float().cpu() if a is not None else None for a in self._attn_out]
        finally:
            self._attn_mode = None
            self._attn_out = [None] * len(self.encoders)
        return res

    @torch.no_grad()
    def attention_full(self, inputs: Sequence[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        inputs = self._to_dev(inputs)
        self._attn_mode = "full"
        try:
            self.model(*inputs)
            res = [a.float().cpu() if a is not None else None for a in self._attn_out]
        finally:
            self._attn_mode = None
            self._attn_out = [None] * len(self.encoders)
        return res

    # ----------------------------------------------------- dataset-level runs --
    def token_attributions(
        self,
        tensors: Sequence[torch.Tensor],
        batch_size: int = 4,
        steps: int = 16,
        method: str = "ig",
        with_attention: bool = True,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, object]:
        """Run IG (and attention-received) over a dataset.

        Returns a dict with per-modality ``TokenTable`` for IG and attention, plus
        per-sample ``logit``, ``logit_baseline`` and ``completeness_residual``.
        """
        n = tensors[0].shape[0]
        m_count = len(tensors)
        ig_tables = [TokenTable() for _ in range(m_count)]
        att_tables = [TokenTable() for _ in range(m_count)]
        logits, base_logits, resid = [], [], []
        for s in range(0, n, batch_size):
            batch = [t[s:s + batch_size] for t in tensors]
            attr, fx, f0 = self.integrated_gradients(batch, steps=steps, method=method)
            att = self.attention_received(batch) if with_attention else [None] * m_count
            total = sum(a.sum(1) for a in attr)
            logits.append(fx)
            base_logits.append(f0)
            resid.append(fx - f0 - total)
            for m in range(m_count):
                toks = batch[m].long()
                for i in range(toks.shape[0]):
                    ig_tables[m].per_sample.append(_scatter(toks[i], attr[m][i]))
                    if att[m] is not None:
                        att_tables[m].per_sample.append(_scatter(toks[i], att[m][i], reduce="sum"))
                    else:
                        att_tables[m].per_sample.append({})
            if progress:
                progress(f"  attributed {min(s + batch_size, n)}/{n} samples")
        return {
            "ig": ig_tables,
            "attention": att_tables,
            "logit": torch.cat(logits).numpy(),
            "logit_baseline": torch.cat(base_logits).numpy(),
            "completeness_residual": torch.cat(resid).numpy(),
        }

    def modality_decomposition(self, tensors: Sequence[torch.Tensor]) -> Dict[str, np.ndarray]:
        """Exact additive decomposition of the logit over modalities (see module docstring)."""
        full = self.logits(tensors)
        contrib = []
        for m in range(len(tensors)):
            masked = [t if j != m else torch.zeros_like(t) for j, t in enumerate(tensors)]
            contrib.append((full - self.logits(masked)).numpy())
        all_masked = self.logits([torch.zeros_like(t) for t in tensors]).numpy()
        contrib = np.stack(contrib, 1)
        return {
            "logit": full.numpy(),
            "contrib": contrib,
            "logit_all_masked": all_masked,
            "additivity_residual": full.numpy() - all_masked - contrib.sum(1),
        }

    def pair_interactions(
        self,
        tensors: Sequence[torch.Tensor],
        modality: int,
        candidate_tokens: Sequence[int],
        max_samples: int = 200,
        with_attention: bool = True,
        seed: int = 0,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Occlusion interaction for all pairs among ``candidate_tokens``."""
        cand = [int(t) for t in candidate_tokens]
        cand_set = set(cand)
        n = tensors[0].shape[0]
        idx = list(range(n))
        rng = random.Random(seed)
        # prefer samples that contain at least two candidates
        present = [sorted(cand_set.intersection(tensors[modality][i].tolist())) for i in idx]
        idx = [i for i in idx if len(present[i]) >= 2]
        if max_samples and len(idx) > max_samples:
            idx = sorted(rng.sample(idx, max_samples))
        base_logits = self.logits(tensors).numpy()
        acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        att_acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        pacc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for c, i in enumerate(idx):
            toks = present[i]
            base = [t[i] for t in tensors]
            variants: List[Tuple] = [()] + [(a,) for a in toks] + list(combinations(toks, 2))
            rows = [[b.clone() for b in base] for _ in variants]
            for r, v in zip(rows, variants):
                if v:
                    r[modality] = self.remove(r[modality], v, modality)
            sc = self.modality_score(torch.stack([r[modality] for r in rows]), modality).numpy()
            f = sc - sc[0] + base_logits[i]
            val = {v: f[k] for k, v in enumerate(variants)}
            pval = {v: 1.0 / (1.0 + math.exp(-float(np.clip(f[k], -60, 60)))) for k, v in enumerate(variants)}
            att = None
            if with_attention:
                full = self.attention_full([b.unsqueeze(0) for b in base])[modality]
                att = full[0] if full is not None else None
            row = base[modality].tolist()
            for a, bb in combinations(toks, 2):
                key = (a, bb) if a < bb else (bb, a)
                acc[key].append(float(val[()] - val[(a,)] - val[(bb,)] + val[(a, bb)]))
                pacc[key].append(float(pval[()] - pval[(a,)] - pval[(bb,)] + pval[(a, bb)]))
                if att is not None:
                    pa = [p for p, t in enumerate(row) if t == a]
                    pb = [p for p, t in enumerate(row) if t == bb]
                    s_att = 0.5 * (att[pa][:, pb].mean() + att[pb][:, pa].mean())
                    att_acc[key].append(float(s_att))
            if progress and (c + 1) % 25 == 0:
                progress(f"  pair interactions: {c + 1}/{len(idx)} samples")
        out = {}
        for key, vals in acc.items():
            v = np.asarray(vals)
            pv = np.asarray(pacc[key])
            out[key] = {
                "n": int(v.size),
                "logit_mean": float(v.mean()),
                "logit_z": _z(v),
                "prob_mean": float(pv.mean()),
                "prob_z": _z(pv),
                "attention": float(np.mean(att_acc[key])) if att_acc.get(key) else float("nan"),
            }
        return out

    def occlusion(
        self,
        tensors: Sequence[torch.Tensor],
        modality: int,
        candidate_tokens: Sequence[int],
    ) -> Dict[int, Dict[str, float]]:
        """Exact leave-one-out effect of removing each candidate token.

        Returns, per token, the mean drop in logit and in P(resistant) over the
        samples that contain it, plus the fraction of those samples whose predicted
        class flips.  Positive = the token supports resistance.
        """
        cand = [int(t) for t in candidate_tokens]
        base = self.logits(tensors).numpy()
        score = self.modality_score(tensors[modality], modality).numpy()
        sets = [set(r.tolist()) for r in tensors[modality]]
        out = {}
        for t in cand:
            idx = [i for i, st in enumerate(sets) if t in st]
            if not idx:
                continue
            rows = torch.stack([self.remove(tensors[modality][i], [t], modality) for i in idx])
            after = self.replaced_logits(tensors, modality, idx, rows, base, score)
            b = base[idx]
            out[t] = {
                "LOO_logit": float(np.mean(b - after)),
                "LOO_prob": float(np.mean(_sig(b) - _sig(after))),
                "LOO_flip": float(np.mean((b > 0) != (after > 0))),
                "LOO_n": len(idx),
            }
        return out


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _z(v: np.ndarray) -> float:
    if v.size < 2:
        return float("nan")
    sd = float(v.std(ddof=1))
    return float(v.mean() / (sd / math.sqrt(v.size))) if sd > 0 else float("nan")


def detect_packed(t: torch.Tensor) -> bool:
    """True if tokens are left-packed (a token id can occupy different positions).

    Fixed-slot layout (k-mer panel): every position holds at most one distinct
    non-zero token id across samples.
    """
    t = t.long()
    nz = t != PAD_ID
    if not bool(nz.any()):
        return True
    cols = torch.nonzero(nz, as_tuple=True)[1]
    vals = t[nz]
    pairs = torch.unique(torch.stack([cols, vals]), dim=1)
    return pairs.shape[1] > torch.unique(cols).numel()


# ------------------------------------------------------------------ aggregation
def _scatter(tokens: torch.Tensor, values: torch.Tensor, reduce: str = "sum") -> Dict[int, float]:
    """Sum position-level values into token ids (ignoring padding)."""
    tokens = tokens.long()
    keep = tokens != PAD_ID
    if not bool(keep.any()):
        return {}
    uniq, inv = torch.unique(tokens[keep], return_inverse=True)
    agg = torch.zeros(uniq.numel(), dtype=torch.float64)
    agg.index_add_(0, inv, values[keep].double())
    return {int(t): float(v) for t, v in zip(uniq.tolist(), agg.tolist())}


def summarise_tokens(
    table: TokenTable,
    labels: Sequence[int],
    n_total: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Global per-token statistics.

    ``Importance`` = mean |attribution| over *all* samples (absent -> 0), the same
    convention as mean |SHAP|, so the two rankings are directly comparable.
    """
    labels = np.asarray(labels)
    n = n_total or len(table.per_sample)
    n_r = max(1, int((labels == 1).sum()))
    n_s = max(1, int((labels == 0).sum()))
    stats: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, d in enumerate(table.per_sample):
        y = int(labels[i]) if i < len(labels) else -1
        for t, v in d.items():
            s = stats[t]
            s["n"] += 1
            s["abs_sum"] += abs(v)
            s["sum"] += v
            if y == 1:
                s["n_R"] += 1
                s["sum_R"] += v
            elif y == 0:
                s["n_S"] += 1
                s["sum_S"] += v
    out = {}
    for t, s in stats.items():
        out[t] = {
            "Importance": s["abs_sum"] / n,
            "MeanAttr_present": s["sum"] / s["n"],
            "MeanAbs_present": s["abs_sum"] / s["n"],
            "MeanAttr_R": s["sum_R"] / s["n_R"] if s["n_R"] else float("nan"),
            "MeanAttr_S": s["sum_S"] / s["n_S"] if s["n_S"] else float("nan"),
            "Prev_R": s["n_R"] / n_r,
            "Prev_S": s["n_S"] / n_s,
            "N_present": int(s["n"]),
        }
    return out


def rank_tokens(summary: Dict[int, Dict[str, float]], key: str = "Importance") -> List[int]:
    return [t for t, _ in sorted(summary.items(), key=lambda kv: -kv[1][key])]


# -------------------------------------------------------------- faithfulness
def deletion_curve(
    engine: TransformerAttributor,
    tensors: Sequence[torch.Tensor],
    labels: Sequence[int],
    modality: int,
    rankings: Dict[str, object],
    ks: Sequence[int] = (1, 2, 5, 10, 20, 50),
    random_repeats: int = 3,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Remove the top-k tokens of one modality and measure the effect on the model.

    ``rankings`` maps a name to either
      * a list of token ids (global ranking; the top-k *present* tokens are removed),
      * a list of per-sample dicts ``{token: signed attribution}`` (local ranking;
        tokens supporting the predicted class are removed first), or
      * the string ``"random"``.
    Effect is ``sign * (f(x) - f(x_del))`` with ``sign`` = +1 for predicted
    resistant and -1 for predicted susceptible, i.e. the loss of support for the
    model's own decision.  Larger is more faithful.
    """
    labels = np.asarray(labels)
    n = tensors[0].shape[0]
    base = engine.logits(tensors).numpy()
    base_score = engine.modality_score(tensors[modality], modality).numpy()
    all_idx = list(range(n))
    pred = (base > 0).astype(int)
    sign = np.where(pred == 1, 1.0, -1.0)
    present = [sorted(set(tensors[modality][i].tolist()) - {PAD_ID}) for i in range(n)]
    rows_out = []
    for name, rk in rankings.items():
        reps = random_repeats if isinstance(rk, str) and rk == "random" else 1
        for k in ks:
            eff_all, peff_all, flip_all, f1_all, auc_all = [], [], [], [], []
            for r in range(reps):
                rng = random.Random(seed + 1000 * r + k)
                new_rows = []
                for i in range(n):
                    toks = present[i]
                    if isinstance(rk, str):
                        chosen = rng.sample(toks, min(k, len(toks)))
                    elif isinstance(rk, list) and rk and isinstance(rk[0], dict):
                        d = rk[i]
                        chosen = sorted(toks, key=lambda t: -sign[i] * d.get(t, 0.0))[:k]
                    else:
                        order = {t: j for j, t in enumerate(rk)}
                        chosen = sorted([t for t in toks if t in order], key=order.get)[:k]
                    new_rows.append(engine.remove(tensors[modality][i], chosen, modality))
                after = engine.replaced_logits(tensors, modality, all_idx, torch.stack(new_rows), base, base_score)
                eff_all.append(float(np.mean(sign * (base - after))))
                peff_all.append(float(np.mean(sign * (_sig(base) - _sig(after)))))
                flip_all.append(float(np.mean((after > 0).astype(int) != pred)))
                f1_all.append(_f1(labels, (after > 0).astype(int)))
                auc_all.append(_auc(labels, after))
            rows_out.append({
                "ranking": name,
                "k": k,
                "support_drop_logit": float(np.mean(eff_all)),
                "support_drop_prob": float(np.mean(peff_all)),
                "flip_rate": float(np.mean(flip_all)),
                "F1_after": float(np.mean(f1_all)),
                "AUC_after": float(np.mean(auc_all)),
                "F1_before": _f1(labels, pred),
                "AUC_before": _auc(labels, base),
            })
    return rows_out


def _f1(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isin(y, [0, 1])
    y, p = y[ok], p[ok]
    tp = float(((p == 1) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isin(y, [0, 1])
    y, s = y[ok], s[ok]
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --------------------------------------------------------------- sanity check
def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def randomize_parameters(model: nn.Module, scope: str, seed: int = 0) -> nn.Module:
    """Re-initialise parameters (Adebayo et al., 2018 model-randomisation test).

    ``scope='head'`` re-initialises the fusion head (fc1, bn, out);
    ``scope='all'`` re-initialises every layer.
    """
    torch.manual_seed(seed)
    model = unwrap(model)
    for name, mod in model.named_modules():
        is_head = name in {"fc1", "bn", "out"}
        if scope == "all" or is_head:
            if hasattr(mod, "reset_parameters") and not isinstance(mod, nn.BatchNorm1d):
                mod.reset_parameters()
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()
                mod.reset_parameters()
    return model
