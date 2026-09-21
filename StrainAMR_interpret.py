#!/usr/bin/env python
"""StrainAMR_interpret - attribution on the trained Transformer classifier.

Runs on an existing checkpoint, so models trained with ``-a 0`` (e.g. the 3-fold
benchmark) can be interpreted without retraining.

Example
-------
python StrainAMR_interpret.py \
    -i Benchmark_features/Ecoli_ciprofloxacin_3fold/Fold1 \
    -m Model_3fold_batch_best/Ecoli_ciprofloxacin_3fold/Fold1 \
    -o Interpret/Ecoli_ciprofloxacin/Fold1 --rf_shap --sanity --export_topk 10,50,100

Outputs (in -o)
---------------
<mod>_token_attribution.tsv    global token table (train split): IG, occlusion, attention,
                               chi2, RF-SHAP ranks side by side + annotations
<mod>_neural_rank_{ig,occlusion}.txt   rankings in SHAP-file format (select_topx_shap.py)
<mod>_pair_interaction.tsv     within-modality synergy/redundancy (occlusion) vs attention
modality_contribution_*.tsv    exact per-genome logit decomposition over modalities
sample_token_attribution_eval.tsv      per-genome top tokens for the held-out split
faithfulness_deletion.tsv      deletion test on held-out genomes for every ranking
sanity_randomization.tsv       model-randomisation check (--sanity)
topk_token_files/              filtered token files for retraining (--export_topk)
interpret_summary.json         completeness, additivity, agreement, AOPC, runtime
"""

import argparse
import os
import sys

import numpy as np
import torch

from library import interpret_pipeline as IP
from library import transformer_attribution as TA


def parse_ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip()) if s else ()


def main():
    ap = argparse.ArgumentParser(prog="StrainAMR_interpret.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input_file", required=True,
                    help="Feature directory (output of StrainAMR_build_train/test; contains strains_train_*.txt)")
    ap.add_argument("-m", "--model_PATH", required=True, help="Checkpoint (.pt) or training output directory")
    ap.add_argument("-o", "--outdir", default="StrainAMR_interpret_res")
    ap.add_argument("-f", "--feature_used", default="all", help="Same value as used for training (default: all)")
    ap.add_argument("--test_dir", default=None,
                    help="Directory holding strains_test_*.txt if different from -i (e.g. prediction features)")
    ap.add_argument("--no_eval", action="store_true", help="Skip held-out analyses even if test files exist")
    ap.add_argument("--embed_size", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--method", choices=["ig", "gxi"], default="ig")
    ap.add_argument("--steps", type=int, default=16, help="Integrated-gradient steps (default 16)")
    ap.add_argument("--target", choices=["logit", "prob"], default="logit")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_rows", type=int, default=32, help="Rows per forward pass (GPU memory knob)")
    ap.add_argument("--max_train", type=int, default=0, help="Subsample training genomes (stratified; 0 = all)")
    ap.add_argument("--max_eval", type=int, default=0, help="Subsample held-out genomes (stratified; 0 = all)")
    ap.add_argument("--occlusion_top", type=int, default=50)
    ap.add_argument("--pair_top", type=int, default=15)
    ap.add_argument("--pair_samples", type=int, default=200)
    ap.add_argument("--no_faithfulness", action="store_true")
    ap.add_argument("--ks", default="1,2,5,10,20,50")
    ap.add_argument("--rf_shap", action="store_true",
                    help="Compute an RF-SHAP baseline on the training split when no SHAP file is found")
    ap.add_argument("--sanity", action="store_true", help="Model-parameter randomisation check")
    ap.add_argument("--export_topk", default="", help="e.g. 10,50,100 -> filtered token files for retraining")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    indir = a.input_file
    labels = IP.canonical_labels(a.feature_used)
    lengths = IP.load_lengths(indir)
    tr_arrays, tr_y, tr_ids, maxtok = IP.load_split(indir, "train", labels, lengths)
    vocab = [m + 2 for m in maxtok]

    test_dir = a.test_dir or indir
    ev_arrays = ev_y = ev_ids = None
    if not a.no_eval and all(os.path.exists(os.path.join(test_dir, f"strains_test_{IP.FILE_STEM[l]}.txt"))
                             for l in labels):
        ev_arrays, ev_y, ev_ids, _ = IP.load_split(test_dir, "test", labels, lengths)
        ev_arrays = [IP.remove_unseen(tr, ev) for tr, ev in zip(tr_arrays, ev_arrays)]

    model, enc_names = IP.build_model(labels, vocab, lengths, device, embed_size=a.embed_size, heads=a.heads)
    ckpt = IP.resolve_checkpoint(a.model_PATH)
    state = TA.clean_state_dict(torch.load(ckpt, map_location=device))
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[interpret] checkpoint: {ckpt}", flush=True)
    print(f"[interpret] modalities: {labels}; train n={len(tr_ids)}; eval n={0 if ev_ids is None else len(ev_ids)}",
          flush=True)

    shap_files, annot_files = IP.find_default_files(indir, labels)
    opts = IP.InterpretOptions(
        method=a.method, steps=a.steps, target=a.target, batch_size=a.batch_size, max_rows=a.max_rows,
        max_train=a.max_train, max_eval=a.max_eval, occlusion_top=a.occlusion_top, pair_top=a.pair_top,
        pair_samples=a.pair_samples, faithfulness=not a.no_faithfulness, ks=parse_ints(a.ks),
        rf_shap=a.rf_shap, sanity=a.sanity, export_topk=parse_ints(a.export_topk), seed=a.seed,
        shap_files=shap_files, annotation_files=annot_files)
    IP.run(model, enc_names, labels, tr_arrays, tr_y, tr_ids, a.outdir, opts,
           eval_arrays=ev_arrays, y_eval=ev_y, ids_eval=ev_ids, device=device, indir=indir)


if __name__ == "__main__":
    sys.exit(main())
