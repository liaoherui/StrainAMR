#!/usr/bin/env python3
"""Compact per-fold completeness table. Read-only."""
import argparse
import glob
import os
import sys

WORKROOT = "/data2/hmy/StrainAMR-optimize"
FEATURE_ROOT = os.path.join(WORKROOT, "features-3fold")
MODEL_ROOTS = [
    os.path.join(WORKROOT, "models-3fold"),
    os.path.join(WORKROOT, "models-mtb-schemeA"),
    os.path.join(WORKROOT, "models-3fold-ablation"),
]
FOLDS = ["Fold1", "Fold2", "Fold3"]


def has_model(dataset, fold):
    for root in MODEL_ROOTS:
        base = os.path.join(root, dataset, fold)
        for md in sorted(glob.glob(os.path.join(base, "model-gpu[0-9]"))):
            ck = os.path.join(md, "models", "best_model_f1_score.pt")
            if os.path.exists(ck):
                return os.path.getsize(ck) / 1e6, os.path.basename(md)
    return None, None


def has_pred(dataset, fold):
    for root in MODEL_ROOTS:
        base = os.path.join(root, dataset, fold)
        for pd in sorted(glob.glob(os.path.join(base, "prediction-gpu[0-9]"))):
            if os.path.exists(os.path.join(pd, "logs", "output_sample_prob_predict.txt")):
                return True, os.path.basename(pd)
    return False, None


def has_analysis(dataset, fold):
    for root in MODEL_ROOTS:
        base = os.path.join(root, dataset, fold)
        for d in sorted(glob.glob(os.path.join(base, "*gpu[0-9]"))):
            if os.path.isdir(os.path.join(d, "analysis")):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    datasets = sorted(d for d in os.listdir(FEATURE_ROOT)
                      if os.path.isdir(os.path.join(FEATURE_ROOT, d)) and d.endswith("_3fold"))
    rows = []
    for ds in datasets:
        for fold in FOLDS:
            feat = os.path.join(FEATURE_ROOT, ds, fold)
            exists = os.path.isdir(feat)
            bt = os.path.exists(os.path.join(feat, ".build-train.complete"))
            bv = os.path.exists(os.path.join(feat, ".build-val.complete"))
            db = os.path.isdir(os.path.join(feat, "db"))
            vf = os.path.isdir(os.path.join(feat, "val_features"))
            size, mdir = has_model(ds, fold)
            pred, pdir = has_pred(ds, fold)
            analysis = has_analysis(ds, fold)

            feat_s = "NONE" if not exists else ("OK" if bt and bv else "PART")
            notes = []
            if not exists:
                notes.append("无特征目录")
            else:
                if not bt:
                    notes.append("无.build-train")
                if not bv:
                    notes.append("无.build-val")
                if not db:
                    notes.append("无db/")
                if not vf:
                    notes.append("无val_features/")
            if size is None:
                notes.append("无模型")
            if not pred:
                notes.append("无预测")
            if not analysis:
                notes.append("无analysis/")
            rows.append({
                "dataset": ds, "fold": fold, "feat": feat_s,
                "db": "Y" if db else "-", "vf": "Y" if vf else "-",
                "model": f"{size:.0f}" if size else "NONE", "mdir": mdir or "-",
                "pred": "Y" if pred else "-", "analysis": "Y" if analysis else "-",
                "notes": ", ".join(notes),
            })

    header = (f"{'dataset':34s} {'fold':5s} {'feat':4s} {'db':2s} {'vf':2s} "
              f"{'modelMB':>7s} {'gpu':6s} {'pred':4s} {'analysis':8s} notes")
    if not args.quiet:
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['dataset']:34s} {r['fold']:5s} {r['feat']:4s} {r['db']:2s} {r['vf']:2s} "
                  f"{r['model']:>7s} {r['mdir']:6s} {r['pred']:4s} {r['analysis']:8s} {r['notes']}")
        print("-" * len(header))

    bad = [r for r in rows if r["notes"]]
    print(f"\n共 {len(rows)} 折，完整 {len(rows) - len(bad)} 折，有缺失 {len(bad)} 折")
    if bad:
        print("\n== 异常清单 ==")
        for r in bad:
            print(f"  {r['dataset']}/{r['fold']}: {r['notes']}")
    for key, label in (("feat", "特征标记"), ("db", "db/"), ("vf", "val_features/"),
                       ("pred", "预测概率"), ("analysis", "analysis/")):
        n = sum(1 for r in rows if r[key] in ("OK", "Y"))
        print(f"  {label:14s} {n}/{len(rows)}")
    print(f"  {'best checkpoint':14s} {sum(1 for r in rows if r['model'] != 'NONE')}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
