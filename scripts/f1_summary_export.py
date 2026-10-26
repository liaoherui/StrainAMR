#!/usr/bin/env python3
"""Export three-fold averaged F1 results for slides / Excel. Read-only on models.

Writes into --out-dir (default: f1-summary-<YYYYmmdd-HHMM>):
    f1_by_fold.csv   one row per fold (n, pos, F1@0.5, AUC, AP)
    f1_summary.csv   one row per dataset: F1 of each fold, mean, sd, AUC/AP mean
    f1_slide.md      markdown table sorted like the paper figure
With --compare also:
    f1_vs_paper.csv  our mean next to the numbers in the reference figure

Usage:
    python3 scripts/f1_summary_export.py
    python3 scripts/f1_summary_export.py --compare --schemeA
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import eval_threshold_metrics as ev  # noqa: E402

WORKROOT = "/data2/hmy/StrainAMR-optimize"
ROOTS = [
    os.path.join(WORKROOT, "models-3fold"),
    os.path.join(WORKROOT, "models-mtb-schemeA"),
    os.path.join(WORKROOT, "models-3fold-ablation"),
]
RUN_DIR_RE = re.compile(r"^(?:model|prediction)-gpu\d+$")
FOLD_RE = re.compile(r"^Fold(\d+)")

SPECIES = {
    "Aba": "A. baumannii",
    "Cdi": "C. difficile",
    "Ecoli": "E. coli",
    "Kcp": "K. pneumoniae",
    "Mtb": "M. tuberculosis",
    "Sau": "S. aureus",
}
SPECIES_ORDER = ["Aba", "Cdi", "Ecoli", "Kcp", "Mtb", "Sau"]
DRUG_ORDER = {
    "Aba": ["amikacin", "ampicillin", "avycaz", "imipenem"],
    "Cdi": ["azithromycin", "clindamycin", "clarithromycin", "moxifloxacin"],
    "Ecoli": ["amoxicillin", "ampicillin", "ciprofloxacin", "levofloxacin"],
    "Kcp": ["amoxicillin", "avycaz", "ciprofloxacin", "levofloxacin"],
    "Mtb": ["amoxicillin", "ciprofloxacin", "ethionamide", "ofloxacin"],
    "Sau": ["clindamycin", "ciprofloxacin", "erythromycin", "levofloxacin"],
}
# numbers read off the reference figure (StrainAMR bars)
PAPER = {
    ("Aba", "amikacin"): 0.925, ("Aba", "ampicillin"): 0.900,
    ("Aba", "avycaz"): 0.980, ("Aba", "imipenem"): 0.918,
    ("Cdi", "azithromycin"): 0.967, ("Cdi", "clindamycin"): 0.979,
    ("Cdi", "clarithromycin"): 0.962, ("Cdi", "moxifloxacin"): 0.977,
    ("Sau", "clindamycin"): 0.966, ("Sau", "ciprofloxacin"): 0.981,
    ("Sau", "erythromycin"): 0.960, ("Sau", "levofloxacin"): 1.000,
    ("Ecoli", "amoxicillin"): 0.914, ("Ecoli", "ampicillin"): 0.930,
    ("Ecoli", "ciprofloxacin"): 0.964, ("Ecoli", "levofloxacin"): 0.965,
    ("Mtb", "amoxicillin"): 0.520, ("Mtb", "ciprofloxacin"): 0.905,
    ("Mtb", "ethionamide"): 0.641, ("Mtb", "ofloxacin"): 0.918,
    ("Kcp", "amoxicillin"): 0.940, ("Kcp", "avycaz"): 0.944,
    ("Kcp", "ciprofloxacin"): 0.974, ("Kcp", "levofloxacin"): 0.980,
}


def discover(roots):
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        src_dir = os.path.basename(os.path.normpath(root))
        source = ev.DEFAULT_ALIASES.get(src_dir, src_dir)
        pattern = os.path.join(root, "*", "Fold*", "*", "logs", "output_sample_prob_*.txt")
        for path in glob.glob(pattern):
            rel = os.path.relpath(path, root).split(os.sep)
            if len(rel) != 5:
                continue
            dataset, fold_dir, run_dir, _logs, fname = rel
            if not RUN_DIR_RE.match(run_dir):
                continue
            fm = FOLD_RE.match(fold_dir)
            if not fm:
                continue
            stage = "val" if "val_best_f1" in fname else "predict"
            rank = (0 if stage == "val" else 1,
                    0 if stage == "predict" else 1,
                    0 if src_dir == "models-3fold" else 1)
            key = (source, dataset, int(fm.group(1)))
            if key not in found or rank < found[key][0]:
                found[key] = (rank, {"source": source, "dataset": dataset,
                                     "fold": int(fm.group(1)), "stage": stage,
                                     "file": path})
    return [v[1] for v in found.values()]


def parse_dataset(dataset):
    m = re.match(r"^([A-Za-z]+)_([a-z0-9]+)_3fold$", dataset)
    if not m:
        return None, None, None
    prefix, drug = m.group(1), m.group(2)
    return prefix, SPECIES.get(prefix, prefix), drug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--compare", action="store_true", help="加一列参考图的数字和差值")
    ap.add_argument("--schemeA", action="store_true", help="把 Scheme A 的折也列进汇总")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        WORKROOT, "f1-summary-" + datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for r in discover(ROOTS):
        if r["source"] != "baseline" and not args.schemeA:
            continue
        prefix, species, drug = parse_dataset(r["dataset"])
        if prefix is None:
            continue
        recs = ev.read_probs(r["file"])
        if not recs:
            continue
        y = [x[1] for x in recs]
        s = [x[2] for x in recs]
        m = ev.confusion(y, s, 0.5)
        rows.append({
            "species": species, "prefix": prefix, "drug": drug,
            "dataset": r["dataset"], "source": r["source"], "fold": r["fold"],
            "n": len(y), "n_pos": sum(y),
            "f1": m["f1"], "sens": m["sensitivity"], "spec": m["specificity"],
            "auc": ev.roc_auc(y, s), "ap": ev.average_precision(y, s),
        })

    rows.sort(key=lambda r: (SPECIES_ORDER.index(r["prefix"])
                             if r["prefix"] in SPECIES_ORDER else 99,
                             DRUG_ORDER.get(r["prefix"], []).index(r["drug"])
                             if r["drug"] in DRUG_ORDER.get(r["prefix"], []) else 99,
                             r["source"], r["fold"]))

    def write_csv(name, header, body):
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(body)
        return path

    write_csv(
        "f1_by_fold.csv",
        ["species", "drug", "dataset", "source", "fold", "n", "n_pos",
         "F1@0.5", "sensitivity", "specificity", "AUC", "AP"],
        [[r["species"], r["drug"], r["dataset"], r["source"], r["fold"], r["n"], r["n_pos"],
          round(r["f1"], 4), round(r["sens"], 4), round(r["spec"], 4),
          round(r["auc"], 4) if r["auc"] is not None else "",
          round(r["ap"], 4) if r["ap"] is not None else ""] for r in rows],
    )

    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["source"]), []).append(r)

    summary = []
    for (dataset, source), recs in sorted(groups.items()):
        recs.sort(key=lambda r: r["fold"])
        prefix, species, drug = parse_dataset(dataset)
        f1s = [r["f1"] for r in recs]
        aucs = [r["auc"] for r in recs if r["auc"] is not None]
        aps = [r["ap"] for r in recs if r["ap"] is not None]
        entry = {
            "species": species, "prefix": prefix, "drug": drug,
            "dataset": dataset, "source": source, "folds": len(recs),
            "n": statistics.mean(r["n"] for r in recs),
            "n_pos": statistics.mean(r["n_pos"] for r in recs),
            "f1": f1s,
            "f1_mean": statistics.mean(f1s),
            "f1_sd": statistics.stdev(f1s) if len(f1s) > 1 else None,
            "auc_mean": statistics.mean(aucs) if aucs else None,
            "ap_mean": statistics.mean(aps) if aps else None,
        }
        if args.compare:
            entry["paper"] = PAPER.get((prefix, drug))
            entry["delta"] = (entry["f1_mean"] - entry["paper"]) if entry["paper"] else None
        summary.append(entry)

    summary.sort(key=lambda e: (
        SPECIES_ORDER.index(e["prefix"]) if e["prefix"] in SPECIES_ORDER else 99,
        DRUG_ORDER.get(e["prefix"], []).index(e["drug"])
        if e["drug"] in DRUG_ORDER.get(e["prefix"], []) else 99,
        e["source"]))

    header = ["species", "drug", "dataset", "source", "folds", "n", "n_pos",
              "F1_Fold1", "F1_Fold2", "F1_Fold3", "F1_mean", "F1_sd", "AUC_mean", "AP_mean"]
    if args.compare:
        header += ["paper_F1", "delta"]
    body = []
    for e in summary:
        f1s = [round(v, 4) for v in e["f1"]] + [""] * (3 - len(e["f1"]))
        row = [e["species"], e["drug"], e["dataset"], e["source"], e["folds"],
               int(round(e["n"])), int(round(e["n_pos"]))] + f1s + [
            round(e["f1_mean"], 4),
            round(e["f1_sd"], 4) if e["f1_sd"] is not None else "",
            round(e["auc_mean"], 4) if e["auc_mean"] is not None else "",
            round(e["ap_mean"], 4) if e["ap_mean"] is not None else ""]
        if args.compare:
            row += [round(e["paper"], 3) if e["paper"] else "",
                    round(e["delta"], 3) if e["delta"] is not None else ""]
        body.append(row)
    write_csv("f1_summary.csv", header, body)

    md = ["| species | drug | F1 Fold1 | F1 Fold2 | F1 Fold3 | F1 mean | F1 sd | AUC |"
          + (" paper | Δ |" if args.compare else ""),
          "|---|---|---|---|---|---|---|---|" + ("---|---|" if args.compare else "")]
    for e in summary:
        f1s = [f"{v:.3f}" for v in e["f1"]] + ["-"] * (3 - len(e["f1"]))
        sd = f"{e['f1_sd']:.3f}" if e["f1_sd"] is not None else "NA"
        auc = f"{e['auc_mean']:.3f}" if e["auc_mean"] is not None else "NA"
        line = (f"| {e['species']} | {e['drug']} | {f1s[0]} | {f1s[1]} | {f1s[2]} | "
                f"**{e['f1_mean']:.3f}** | {sd} | {auc} |")
        if args.compare:
            paper = f"{e['paper']:.3f}" if e["paper"] else "-"
            delta = f"{e['delta']:+.3f}" if e["delta"] is not None else "-"
            line += f" {paper} | {delta} |"
        md.append(line)
    with open(os.path.join(out_dir, "f1_slide.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n文件写到: {out_dir}")
    for name in sorted(os.listdir(out_dir)):
        print("  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
