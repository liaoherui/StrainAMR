"""End-to-end interpretation of a trained StrainAMR Transformer.

Used by ``StrainAMR_interpret.py`` (standalone, on existing checkpoints) and by
``StrainAMR_model_train.py`` / ``StrainAMR_model_predict.py``.

Leakage rule: every *global* ranking (neural, RF-SHAP, chi-squared, attention) is
computed on the training split only.  The held-out split is used for local
(per-genome) explanations and for the faithfulness test.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from itertools import combinations  # noqa: F401  (kept for users extending the pipeline)
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from library import transformer_attribution as TA

# ------------------------------------------------------------------ constants
FILE_STEM = {"snv": "sentence_fs", "pc": "pc_token_fs", "kmer": "kmer_token"}
LEN_COL = {"snv": 0, "pc": 1, "kmer": 2}
ANNOT_FILES = {
    "snv": ["feature_remain_graph.txt", "node_token_match.txt"],
    "pc": ["feature_remain_pc.txt", "pc_matches.txt"],
    "kmer": ["kmer_token_id.txt"],
}
SHAP_FILES = {
    "snv": "strains_train_sentence_fs_shap.txt",
    "pc": "strains_train_pc_token_fs_shap.txt",
    "kmer": "strains_train_kmer_token_shap.txt",
}


@dataclass
class InterpretOptions:
    method: str = "ig"                 # ig | gxi
    steps: int = 16                    # IG steps
    target: str = "logit"              # logit | prob (IG output scale)
    batch_size: int = 4                # samples per IG batch
    max_rows: int = 32                 # rows per forward pass
    max_train: int = 0                 # subsample training genomes for global ranking (0 = all)
    max_eval: int = 0                  # subsample held-out genomes (0 = all)
    occlusion_top: int = 50            # candidates per modality for exact leave-one-out
    pair_top: int = 15                 # candidates per modality for pair interactions
    pair_samples: int = 200            # genomes per modality for pair interactions
    faithfulness: bool = True
    ks: Sequence[int] = (1, 2, 5, 10, 20, 50)
    random_repeats: int = 3
    rf_shap: bool = False              # compute RF-SHAP baseline if no SHAP file is found
    sanity: bool = False
    export_topk: Sequence[int] = ()
    sample_top: int = 20               # tokens per genome in the per-sample table
    seed: int = 0
    log: Optional[callable] = None
    shap_files: Dict[str, str] = field(default_factory=dict)
    annotation_files: Dict[str, List[str]] = field(default_factory=dict)


# ----------------------------------------------------------------- data I/O
def read_token_file(path: str, length: int):
    """Mirror of ``process_intsv`` in the training script."""
    ids, labels, rows = [], [], []
    with open(path) as fh:
        fh.readline()
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ele = line.split()
            toks = [int(t) for t in re.split(",", ele[-1]) if t != ""]
            ids.append(ele[0])
            labels.append(int(ele[1]))
            rows.append(toks)
    width = length if length else max(len(r) for r in rows)
    mat = np.zeros((len(rows), width), dtype=np.int64)
    mx = 0
    for i, r in enumerate(rows):
        r = r[:width]
        mat[i, : len(r)] = r
        if r:
            mx = max(mx, max(r))
    return mat, np.asarray(labels), ids, mx


def load_lengths(indir: str):
    with open(os.path.join(indir, "longest_len_fs.txt")) as fh:
        fh.readline()
        ele = fh.readline().strip().split("\t")
    return [int(e) for e in ele[:3]]


def canonical_labels(fused: Optional[str]) -> List[str]:
    """Modality order exactly as in StrainAMR_model_train.py."""
    if not fused or fused == "all":
        return ["snv", "pc", "kmer"]
    parts = [p.strip() for p in fused.split(",") if p.strip()]
    if len(parts) == 1:
        return [parts[0] if parts[0] in ("pc", "kmer") else "snv"]
    if len(parts) == 2:
        if "pc" in parts and "kmer" in parts:
            return ["kmer", "pc"]
        if "snv" in parts and "kmer" in parts:
            return ["snv", "kmer"]
        return ["snv", "pc"]
    return ["snv", "pc", "kmer"]


def remove_unseen(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Mirror of ``remove_new_ele``: tokens unseen in training become 0 in place."""
    seen = np.unique(train)
    out = test.copy()
    out[~np.isin(out, seen)] = 0
    return out


def load_split(indir: str, split: str, labels: Sequence[str], lengths: Sequence[int]):
    arrays, maxtok, y, ids = [], [], None, None
    for lab in labels:
        path = os.path.join(indir, f"strains_{split}_{FILE_STEM[lab]}.txt")
        mat, yy, ii, mx = read_token_file(path, lengths[LEN_COL[lab]])
        arrays.append(mat)
        maxtok.append(mx)
        y, ids = yy, ii
    return arrays, y, ids, maxtok


def build_model(labels, vocab, lengths, device, embed_size=512, heads=8):
    from library import (Transformer_without_pos, Transformer_without_pos_multimodal_add_attn,
                         Transformer_without_pos_multimodal_add_attn_only2)
    L = [lengths[LEN_COL[l]] for l in labels]
    kw = dict(src_pad_idx=0, dropout=0.1, embed_size=embed_size, heads=heads, device=device)
    if len(labels) == 1:
        return Transformer_without_pos.Transformer(src_vocab_size=vocab[0], max_length=L[0], **kw), ["encoder"]
    if len(labels) == 2:
        return (Transformer_without_pos_multimodal_add_attn_only2.Transformer(
            src_vocab_size_1=vocab[0], src_vocab_size_2=vocab[1], max_length_1=L[0], max_length_2=L[1], **kw),
            ["encoder1", "encoder2"])
    return (Transformer_without_pos_multimodal_add_attn.Transformer(
        src_vocab_size_1=vocab[0], src_vocab_size_2=vocab[1], src_vocab_size_3=vocab[2],
        max_length_1=L[0], max_length_2=L[1], max_length_3=L[2], **kw),
        ["encoder1", "encoder2", "encoder3"])


def resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    for cand in ("models/best_model_f1_score.pt", "best_model_f1_score.pt", "models/checkpoint_best.pt",
                 "checkpoint_best.pt"):
        p = os.path.join(path, cand)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No checkpoint found under {path}")


def find_default_files(indir: str, labels: Sequence[str]):
    shap, annot = {}, {}
    for lab in labels:
        for root in (os.path.join(indir, "shap"), indir):
            p = os.path.join(root, SHAP_FILES[lab])
            if os.path.exists(p):
                shap[lab] = p
                break
        annot[lab] = [os.path.join(indir, f) for f in ANNOT_FILES[lab] if os.path.exists(os.path.join(indir, f))]
    return shap, annot


# ----------------------------------------------------------- baselines (train)
def presence_matrix(arr: np.ndarray):
    toks = np.unique(arr[arr != 0])
    index = {t: j for j, t in enumerate(toks)}
    X = np.zeros((arr.shape[0], len(toks)), dtype=np.uint8)
    for i, row in enumerate(arr):
        for t in set(row[row != 0].tolist()):
            X[i, index[t]] = 1
    return X, toks


def chi2_ranking(arr: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    from sklearn.feature_selection import chi2
    X, toks = presence_matrix(arr)
    stat, _ = chi2(X, y)
    stat = np.nan_to_num(stat)
    return {int(t): float(s) for t, s in zip(toks, stat)}


def rf_shap_ranking(arr: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[int, float]:
    """Same recipe as library/shap_feature_select.py (RF 500 trees, Tree SHAP, class 1)."""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    X, toks = presence_matrix(arr)
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(X, y)
    sv = shap.TreeExplainer(rf).shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(0)
    return {int(t): float(v) for t, v in zip(toks, imp)}


def read_shap_file(path: str) -> Dict[int, float]:
    out = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ti = header.index("Token_ID") if "Token_ID" in header else 1
        vi = header.index("Shap") if "Shap" in header else len(header) - 1
        for n, line in enumerate(fh):
            p = line.rstrip("\n").split("\t")
            try:
                out[int(p[ti])] = float(p[vi]) if vi < len(p) else float(-n)
            except (ValueError, IndexError):
                continue
    return out


def load_annotations(files: Sequence[str]):
    try:
        from library.token_contribution import load_additional_annotations
        return load_additional_annotations(list(files))
    except Exception:
        return {}, []


# ------------------------------------------------------------------- writing
def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "NA"
        return f"{v:.6g}"
    return str(v)


def _write_tsv(path: str, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(_fmt(v) for v in r) + "\n")


def write_topk_token_files(indir, outdir, label, ranking, ks, splits=("train", "test")):
    """Filtered token files keeping only the top-k neural tokens (same format as
    select_topx_shap.py) so the existing retraining workflow can be reused."""
    os.makedirs(outdir, exist_ok=True)
    for k in ks:
        keep = set(ranking[:k])
        for split in splits:
            src = os.path.join(indir, f"strains_{split}_{FILE_STEM[label]}.txt")
            if not os.path.exists(src):
                continue
            dst = os.path.join(outdir, f"strains_{split}_{FILE_STEM[label]}_neural_top{k}.txt")
            with open(src) as fi, open(dst, "w") as fo:
                fo.write(fi.readline())
                for line in fi:
                    ele = line.rstrip("\n").split("\t")
                    if len(ele) < 2:
                        continue
                    toks = [t for t in ele[-1].split(",") if t == "0" or (t and int(t) in keep)]
                    fo.write(f"{ele[0]}\t{ele[1]}\t{len(toks)}\t{','.join(toks)}\n")


# ------------------------------------------------------------------ pipeline
def run(
    model,
    encoder_names: Sequence[str],
    labels: Sequence[str],
    train_arrays: Sequence[np.ndarray],
    y_train: np.ndarray,
    ids_train: Sequence[str],
    outdir: str,
    opts: InterpretOptions,
    eval_arrays: Optional[Sequence[np.ndarray]] = None,
    y_eval: Optional[np.ndarray] = None,
    ids_eval: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    indir: Optional[str] = None,
) -> Dict[str, object]:
    log = opts.log or (lambda m: print(m, flush=True))
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rng = np.random.RandomState(opts.seed)
    device = device or next(model.parameters()).device
    summary: Dict[str, object] = {"modalities": list(labels), "options": {
        k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in opts.__dict__.items()
        if k not in ("log", "shap_files", "annotation_files")}}

    def sub(arrays, y, ids, cap):
        if cap and len(ids) > cap:
            from sklearn.model_selection import train_test_split
            idx, _ = train_test_split(np.arange(len(ids)), train_size=cap, stratify=y, random_state=opts.seed)
            idx = np.sort(idx)
            return [a[idx] for a in arrays], y[idx], [ids[i] for i in idx]
        return list(arrays), y, list(ids)

    tr_arrays, tr_y, tr_ids = sub(train_arrays, np.asarray(y_train), ids_train, opts.max_train)
    tr_t = [torch.from_numpy(np.asarray(a, dtype=np.int64)) for a in tr_arrays]
    has_eval = eval_arrays is not None and len(eval_arrays) and len(eval_arrays[0])
    if has_eval:
        ev_arrays, ev_y, ev_ids = sub(eval_arrays, np.asarray(y_eval), ids_eval, opts.max_eval)
        ev_t = [torch.from_numpy(np.asarray(a, dtype=np.int64)) for a in ev_arrays]
    labelled_eval = bool(has_eval and np.isin(ev_y, [0, 1]).all() and len(np.unique(ev_y)) == 2)

    eng = TA.TransformerAttributor(model, encoder_names, device=device, max_rows=opts.max_rows, target=opts.target)
    packed = eng.set_layout(tr_t)
    summary["packed_layout"] = dict(zip(labels, packed))
    log(f"[interpret] layout (True = packed, removal re-packs): {summary['packed_layout']}")

    annotations, annot_cols = {}, {}
    for lab in labels:
        annotations[lab], annot_cols[lab] = load_annotations(opts.annotation_files.get(lab, []))

    # ---------------------------------------------------------- 1. IG (train)
    log(f"[interpret] {opts.method.upper()} on {len(tr_ids)} training genomes (global ranking)")
    tr_res = eng.token_attributions(tr_t, batch_size=opts.batch_size, steps=opts.steps, method=opts.method,
                                    progress=log)
    resid = np.abs(tr_res["completeness_residual"])
    spread = float(np.ptp(tr_res["logit"])) or 1.0
    summary["ig_completeness"] = {"median_abs_residual": float(np.median(resid)),
                                  "p95_abs_residual": float(np.percentile(resid, 95)),
                                  "output_range": spread}

    tables, rankings = {}, {}
    for m, lab in enumerate(labels):
        s_ig = TA.summarise_tokens(tr_res["ig"][m], tr_y)
        s_att = TA.summarise_tokens(tr_res["attention"][m], tr_y)
        chi = chi2_ranking(tr_arrays[m], tr_y)
        shap_scores = {}
        if opts.shap_files.get(lab) and os.path.exists(opts.shap_files[lab]):
            shap_scores = read_shap_file(opts.shap_files[lab])
            summary.setdefault("rf_shap_source", {})[lab] = opts.shap_files[lab]
        elif opts.rf_shap:
            log(f"[interpret] computing RF-SHAP baseline for {lab} (training split only)")
            shap_scores = rf_shap_ranking(tr_arrays[m], tr_y, seed=opts.seed)
            summary.setdefault("rf_shap_source", {})[lab] = "computed"
        ig_rank = TA.rank_tokens(s_ig)
        att_rank = [t for t, _ in sorted(s_att.items(), key=lambda kv: -kv[1]["MeanAttr_present"])]
        chi_rank = [t for t, _ in sorted(chi.items(), key=lambda kv: -kv[1])]
        shap_rank = [t for t, _ in sorted(shap_scores.items(), key=lambda kv: -kv[1])] if shap_scores else []

        # ------------------------------------------- 2. exact occlusion (train)
        cand = []
        for r in (ig_rank, chi_rank, shap_rank):
            for t in r[: opts.occlusion_top]:
                if t not in cand:
                    cand.append(t)
        occ = {}
        if opts.occlusion_top > 0 and cand:
            log(f"[interpret] leave-one-out occlusion for {len(cand)} {lab} candidates")
            occ = eng.occlusion(tr_t, m, cand)
        n_tr = len(tr_ids)
        occ_imp = {t: abs(v["LOO_logit"]) * v["LOO_n"] / n_tr for t, v in occ.items()}
        occ_rank = [t for t, _ in sorted(occ_imp.items(), key=lambda kv: -kv[1])]
        occ_rank += [t for t in ig_rank if t not in occ_imp]  # tail: IG order

        rankings[lab] = {"ig_global": ig_rank, "occlusion_global": occ_rank, "attention_global": att_rank,
                         "chi2_train": chi_rank}
        if shap_rank:
            rankings[lab]["rf_shap"] = shap_rank

        # -------------------------------------------------- token table (train)
        rank_pos = {t: i + 1 for i, t in enumerate(ig_rank)}
        occ_pos = {t: i + 1 for i, t in enumerate(occ_rank[: len(occ_imp)])}
        shap_pos = {t: i + 1 for i, t in enumerate(shap_rank)}
        cols = annot_cols.get(lab, [])
        header = ["Token_ID", "Rank_IG", "IG_Importance", "IG_MeanAttr_present", "IG_MeanAttr_R", "IG_MeanAttr_S",
                  "Prev_R", "Prev_S", "N_present", "Attn_received", "Rank_Occlusion", "LOO_logit", "LOO_prob",
                  "LOO_flip", "Chi2", "Rank_RF_SHAP", "RF_SHAP"] + cols
        rows = []
        for t in ig_rank:
            s = s_ig[t]
            o = occ.get(t, {})
            meta = annotations.get(lab, {}).get(t, {})
            rows.append([t, rank_pos[t], s["Importance"], s["MeanAttr_present"], s["MeanAttr_R"], s["MeanAttr_S"],
                         s["Prev_R"], s["Prev_S"], s["N_present"], s_att.get(t, {}).get("MeanAttr_present", float("nan")),
                         occ_pos.get(t, "NA"), o.get("LOO_logit", float("nan")), o.get("LOO_prob", float("nan")),
                         o.get("LOO_flip", float("nan")), chi.get(t, float("nan")), shap_pos.get(t, "NA"),
                         shap_scores.get(t, float("nan"))] + [meta.get(c, "") for c in cols])
        path = os.path.join(outdir, f"{lab}_token_attribution.tsv")
        _write_tsv(path, header, rows)
        tables[lab] = path
        # SHAP-format ranking files (ID, Token_ID, Shap) -> reusable by select_topx_shap.py
        for name, rk, score in (("ig", ig_rank, lambda t: s_ig[t]["Importance"]),
                                ("occlusion", occ_rank[: len(occ_imp)], lambda t: occ_imp[t])):
            _write_tsv(os.path.join(outdir, f"{lab}_neural_rank_{name}.txt"), ["ID", "Token_ID", "Shap"],
                       [[i + 1, t, score(t)] for i, t in enumerate(rk)])
        if opts.export_topk and indir:
            write_topk_token_files(indir, os.path.join(outdir, "topk_token_files"), lab, ig_rank, opts.export_topk)

        # agreement between rankings (train)
        agree = {}
        for other in ("rf_shap", "chi2_train", "attention_global", "occlusion_global"):
            if other not in rankings[lab]:
                continue
            o_rank = rankings[lab][other]
            for k in (10, 50, 100):
                a, b = set(ig_rank[:k]), set(o_rank[:k])
                agree[f"ig_vs_{other}_jaccard@{k}"] = len(a & b) / max(1, len(a | b))
            common = [t for t in ig_rank if t in set(o_rank)]
            if len(common) > 2:
                pos = {t: i for i, t in enumerate(o_rank)}
                agree[f"ig_vs_{other}_spearman"] = TA.spearman(list(range(len(common))), [pos[t] for t in common])
        summary.setdefault("ranking_agreement", {})[lab] = agree

        # ------------------------------------------------ 3. pair interactions
        if opts.pair_top > 1:
            pc = occ_rank[: opts.pair_top]
            log(f"[interpret] pair interactions among top {len(pc)} {lab} tokens")
            pairs = eng.pair_interactions(tr_t, m, pc, max_samples=opts.pair_samples, seed=opts.seed, progress=log)
            prow = []
            for (a, b), v in sorted(pairs.items(), key=lambda kv: -abs(kv[1]["prob_mean"])):
                ma = annotations.get(lab, {}).get(a, {})
                mb = annotations.get(lab, {}).get(b, {})
                prow.append([a, b, v["n"], v["prob_mean"], v["prob_z"], v["logit_mean"], v["logit_z"],
                             "synergy" if v["prob_mean"] > 0 else "redundancy", v["attention"]]
                            + [ma.get(c, "") for c in cols] + [mb.get(c, "") for c in cols])
            _write_tsv(os.path.join(outdir, f"{lab}_pair_interaction.tsv"),
                       ["Token_A", "Token_B", "N", "Interaction_prob", "Z_prob", "Interaction_logit", "Z_logit",
                        "Type", "Attention_AB"] + [f"A_{c}" for c in cols] + [f"B_{c}" for c in cols], prow)
            if prow:
                att = np.array([r[8] for r in prow], float)
                inter = np.array([abs(r[3]) for r in prow], float)
                ok = ~np.isnan(att)
                summary.setdefault("pair_attention_vs_interaction_spearman", {})[lab] = (
                    TA.spearman(att[ok], inter[ok]) if ok.sum() > 2 else float("nan"))

    # ------------------------------------------- 4. modality decomposition
    for split, tens, yy, ids in [("train", tr_t, tr_y, tr_ids)] + ([("eval", ev_t, ev_y, ev_ids)] if has_eval else []):
        dec = eng.modality_decomposition(tens)
        rows = [[ids[i], int(yy[i]), dec["logit"][i]] + list(dec["contrib"][i]) +
                [dec["logit_all_masked"][i], dec["additivity_residual"][i]] for i in range(len(ids))]
        _write_tsv(os.path.join(outdir, f"modality_contribution_{split}.tsv"),
                   ["Sample_ID", "Label", "Logit"] + [f"Contrib_{l}" for l in labels] + ["Logit_all_masked", "Residual"],
                   rows)
        c = dec["contrib"]
        share = np.abs(c) / np.clip(np.abs(c).sum(1, keepdims=True), 1e-9, None)
        summary.setdefault("modality_share", {})[split] = {
            l: {"mean_abs_share": float(share[:, m].mean()),
                "mean_contrib_R": float(c[yy == 1, m].mean()) if (yy == 1).any() else float("nan"),
                "mean_contrib_S": float(c[yy == 0, m].mean()) if (yy == 0).any() else float("nan")}
            for m, l in enumerate(labels)}
        summary.setdefault("max_additivity_residual", {})[split] = float(np.max(np.abs(dec["additivity_residual"])))

    # ---------------------------------------------- 5. local explanations (eval)
    if has_eval:
        log(f"[interpret] per-genome attributions for {len(ev_ids)} held-out genomes")
        ev_res = eng.token_attributions(ev_t, batch_size=opts.batch_size, steps=opts.steps, method=opts.method,
                                        with_attention=False, progress=log)
        rows = []
        for m, lab in enumerate(labels):
            cols = annot_cols.get(lab, [])
            for i, sid in enumerate(ev_ids):
                d = ev_res["ig"][m].per_sample[i]
                for t, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))[: opts.sample_top]:
                    meta = annotations.get(lab, {}).get(t, {})
                    rows.append([sid, int(ev_y[i]), ev_res["logit"][i], lab, t, v,
                                 "->R" if v > 0 else "->S", "; ".join(f"{c}={meta.get(c, '')}" for c in cols if meta.get(c))])
        _write_tsv(os.path.join(outdir, "sample_token_attribution_eval.tsv"),
                   ["Sample_ID", "Label", "Output", "Modality", "Token_ID", "Attribution", "Direction", "Annotation"],
                   rows)

        # ---------------------------------------------- 6. faithfulness (eval)
        if opts.faithfulness:
            frows = []
            for m, lab in enumerate(labels):
                rk = {"ig_local": ev_res["ig"][m].per_sample}
                rk.update(rankings[lab])
                rk["random"] = "random"
                log(f"[interpret] deletion test for {lab}: {', '.join(rk)}")
                for r in TA.deletion_curve(eng, ev_t, ev_y, m, rk, ks=opts.ks, random_repeats=opts.random_repeats,
                                           seed=opts.seed):
                    frows.append([lab] + [r[k] for k in ("ranking", "k", "support_drop_logit", "support_drop_prob",
                                                         "flip_rate", "F1_after", "AUC_after", "F1_before",
                                                         "AUC_before")])
            if not labelled_eval:
                for r in frows:
                    r[6:10] = [float("nan")] * 4
            _write_tsv(os.path.join(outdir, "faithfulness_deletion.tsv"),
                       ["Modality", "Ranking", "k", "Support_drop_logit", "Support_drop_prob", "Flip_rate", "F1_after",
                        "AUC_after", "F1_before", "AUC_before"], frows)
            aopc = {}
            for lab in labels:
                for name in {r[1] for r in frows if r[0] == lab}:
                    vals = [r[3] for r in frows if r[0] == lab and r[1] == name]
                    aopc.setdefault(lab, {})[name] = float(np.mean(vals))
            summary["aopc_logit"] = aopc

        # ------------------------------------- 6b. token-order sensitivity (eval)
        # Packed modalities: shuffle the order of the non-pad tokens within each
        # genome (same token set, different positions).  A position-free model would
        # be unaffected; the drop measures how much the classifier reads position.
        orows = []
        base = eng.logits(ev_t).numpy()
        for m, lab in enumerate(labels):
            if not packed[m]:
                continue
            g = torch.Generator().manual_seed(opts.seed)
            shuf = ev_t[m].clone()
            for i in range(shuf.shape[0]):
                nz = int((shuf[i] != 0).sum())
                if nz > 1:
                    row = shuf[i][shuf[i] != 0]
                    shuf[i, :nz] = row[torch.randperm(nz, generator=g)]
            after = eng.logits([t if j != m else shuf for j, t in enumerate(ev_t)]).numpy()
            orows.append([lab, float(np.mean(np.abs(base - after))), float(np.mean((base > 0) != (after > 0))),
                          TA._f1(ev_y, (base > 0).astype(int)) if labelled_eval else float("nan"),
                          TA._f1(ev_y, (after > 0).astype(int)) if labelled_eval else float("nan"),
                          TA._auc(ev_y, base) if labelled_eval else float("nan"),
                          TA._auc(ev_y, after) if labelled_eval else float("nan")])
        if orows:
            _write_tsv(os.path.join(outdir, "order_sensitivity_eval.tsv"),
                       ["Modality", "Mean_abs_dlogit", "Flip_rate", "F1_original", "F1_shuffled", "AUC_original",
                        "AUC_shuffled"], orows)
            summary["order_sensitivity"] = {r[0]: {"mean_abs_dlogit": r[1], "flip_rate": r[2]} for r in orows}

    # ------------------------------------------------------------ 7. sanity
    if opts.sanity:
        log("[interpret] model-randomisation sanity check")
        import copy
        n_s = min(len(tr_ids), 64)
        sub_t = [t[:n_s] for t in tr_t]
        ref = {lab: TA.summarise_tokens(TA.TokenTable(tr_res["ig"][m].per_sample[:n_s]), tr_y[:n_s])
               for m, lab in enumerate(labels)}
        srows = []
        base_model = TA.unwrap(model)
        for scope in ("head", "all"):
            rmodel = TA.randomize_parameters(copy.deepcopy(base_model), scope, seed=opts.seed + 1).to(device)
            reng = TA.TransformerAttributor(rmodel, encoder_names, device=device, max_rows=opts.max_rows,
                                            target=opts.target)
            rres = reng.token_attributions(sub_t, batch_size=opts.batch_size, steps=opts.steps, method=opts.method,
                                           with_attention=False)
            reng.close()
            for m, lab in enumerate(labels):
                rs = TA.summarise_tokens(rres["ig"][m], tr_y[:n_s])
                toks = sorted(set(ref[lab]) & set(rs))
                # prevalence-free comparisons: mean |attr| and signed mean among genomes carrying the token
                rho_abs = TA.spearman([ref[lab][t]["MeanAbs_present"] for t in toks],
                                      [rs[t]["MeanAbs_present"] for t in toks])
                rho_sgn = TA.spearman([ref[lab][t]["MeanAttr_present"] for t in toks],
                                      [rs[t]["MeanAttr_present"] for t in toks])
                rho_imp = TA.spearman([ref[lab][t]["Importance"] for t in toks], [rs[t]["Importance"] for t in toks])
                prev = TA.spearman([ref[lab][t]["Importance"] for t in toks], [ref[lab][t]["N_present"] for t in toks])
                key = lambda d: sorted(d, key=lambda t: -d[t]["MeanAbs_present"])[:20]
                overlap = len(set(key(ref[lab])) & set(key(rs))) / 20.0
                srows.append([scope, lab, len(toks), rho_abs, rho_sgn, overlap, rho_imp, prev])
        _write_tsv(os.path.join(outdir, "sanity_randomization.tsv"),
                   ["Randomized", "Modality", "N_tokens", "Spearman_meanabs_present", "Spearman_signed_present",
                    "Top20_overlap_meanabs", "Spearman_importance", "Trained_importance_vs_prevalence"], srows)
        summary["sanity"] = [dict(zip(["randomized", "modality", "n", "spearman_meanabs_present",
                                       "spearman_signed_present", "top20_overlap", "spearman_importance",
                                       "trained_importance_vs_prevalence"], r)) for r in srows]

    eng.close()
    summary["runtime_sec"] = round(time.time() - t0, 1)
    summary["tables"] = tables
    with open(os.path.join(outdir, "interpret_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    log(f"[interpret] done in {summary['runtime_sec']} s -> {outdir}")
    return summary
