#!/usr/bin/env python3
"""三折 F1 汇总：每折 F1@0.5 + 三折均值/标准差，输出 CSV + Markdown。只读。"""
import csv, glob, os, re, statistics, sys
from datetime import datetime

WORKROOT = "/data2/hmy/StrainAMR-optimize"
ROOTS = [os.path.join(WORKROOT, "models-3fold")]          # 想加 Scheme A 就把另一行打开
# ROOTS.append(os.path.join(WORKROOT, "models-mtb-schemeA"))
SPECIES = {"Aba": "A. baumannii", "Cdi": "C. difficile", "Ecoli": "E. coli",
           "Kcp": "K. pneumoniae", "Mtb": "M. tuberculosis", "Sau": "S. aureus"}
ORDER = ["Aba", "Cdi", "Ecoli", "Kcp", "Mtb", "Sau"]

def read_probs(path):
    rows = []
    with open(path) as fh:
        next(fh, None)
        for line in fh:
            p = line.replace("\t", " ").split()
            if len(p) < 4:
                continue
            try:
                rows.append((int(p[1]), float(p[3])))
            except ValueError:
                continue
    return rows

def f1_at(y, s, thr=0.5):
    tp = sum(1 for a, b in zip(y, s) if a == 1 and b >= thr)
    fp = sum(1 for a, b in zip(y, s) if a == 0 and b >= thr)
    fn = sum(1 for a, b in zip(y, s) if a == 1 and b < thr)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

def find_file(ds, fold):
    for root in ROOTS:
        pats = [f"{root}/{ds}/{fold}/model-gpu[0-9]/logs/output_sample_prob_val_best_f1.txt",
                f"{root}/{ds}/{fold}/prediction-gpu[0-9]/logs/output_sample_prob_predict.txt"]
        for pat in pats:
            hit = sorted(glob.glob(pat))
            if hit:
                return hit[0]
    return None

def main():
    datasets = sorted(d for d in os.listdir(f"{WORKROOT}/models-3fold") if d.endswith("_3fold"))
    rows = []
    for ds in datasets:
        m = re.match(r"^([A-Za-z]+)_([a-z0-9]+)_3fold$", ds)
        if not m:
            continue
        prefix, drug = m.group(1), m.group(2)
        f1s = []
        for fold in ("Fold1", "Fold2", "Fold3"):
            path = find_file(ds, fold)
            if not path:
                continue
            recs = read_probs(path)
            if recs:
                f1s.append(f1_at([r[0] for r in recs], [r[1] for r in recs]))
        if not f1s:
            continue
        rows.append({
            "species": SPECIES.get(prefix, prefix), "drug": drug, "dataset": ds,
            "f1": f1s, "mean": statistics.mean(f1s),
            "sd": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        })

    rows.sort(key=lambda r: (ORDER.index([k for k, v in SPECIES.items() if v == r["species"]][0])
                             if r["species"] in SPECIES.values() else 99, r["drug"]))

    out = f"{WORKROOT}/f1-summary-{datetime.now():%Y%m%d-%H%M}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "f1_summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "drug", "fold1", "fold2", "fold3", "F1_mean", "F1_sd", "folds"])
        for r in rows:
            f = [round(v, 4) for v in r["f1"]] + [""] * (3 - len(r["f1"]))
            w.writerow([r["species"], r["drug"]] + f + [round(r["mean"], 4), round(r["sd"], 4), len(r["f1"])])

    md = ["| species | drug | Fold1 | Fold2 | Fold3 | mean | sd |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = [f"{v:.3f}" for v in r["f1"]] + ["-"] * (3 - len(r["f1"]))
        md.append(f"| {r['species']} | {r['drug']} | {f[0]} | {f[1]} | {f[2]} | **{r['mean']:.3f}** | {r['sd']:.3f} |")
    with open(os.path.join(out, "f1_slide.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n输出目录: {out}")
    print("  f1_summary.csv   f1_slide.md")

if __name__ == "__main__":
    sys.exit(main())
