#!/usr/bin/env python3
"""Threshold-aware evaluation of StrainAMR prediction files.

Reads existing output_sample_prob_*.txt files (read-only) and reports:
  * F1 / sensitivity / specificity at the default 0.5 threshold
  * F1 / sensitivity / specificity at the per-fold oracle threshold (best F1)
  * F1 at a cross-fold threshold: median of the other folds' oracle thresholds
  * ROC-AUC and average precision (threshold-free)

The oracle threshold is optimistic by construction: it is chosen on the same
fold it is scored on.  The cross-fold column is the honest one and is what a
deployed per-drug threshold would look like.  Nothing is written back into the
model directories.

Usage:
    python3 eval_threshold_metrics.py --dry-run
    python3 eval_threshold_metrics.py --out /data2/.../eval-thresholds-<date>
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

FOLD_RE = re.compile(r"^Fold(\d+)(.*)$")
DATASET_RE = re.compile(r"^(?P<species>[A-Za-z]+)_(?P<drug>[a-z0-9]+)_3fold$")

DEFAULT_ALIASES = {
    "models-3fold": "baseline",
    "models-baseline-extra-3fold": "baseline",
    "models-3fold-ablation": "ablation",
    "models-mtb-schemeA": "schemeA",
}

LAYOUTS = [
    ("val", "*/Fold*/model-gpu[0-9]/logs/output_sample_prob_val_best_f1.txt"),
    ("predict", "*/Fold*/prediction-gpu[0-9]/logs/output_sample_prob_predict.txt"),
    ("flat", "*/Fold*/logs/output_sample_prob_val_best_f1.txt"),
]


def read_probs(path):
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if len(header) < 4:
            header = fh.readline().rstrip("\n").split()
        for line in fh:
            parts = line.replace("\t", " ").split()
            if len(parts) < 4:
                continue
            try:
                rows.append((parts[0], int(parts[1]), float(parts[3])))
            except ValueError:
                continue
    return rows


def roc_auc(y, s):
    npos = sum(y)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(s)) if y[i] == 1)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def average_precision(y, s):
    npos = sum(y)
    if npos == 0:
        return None
    order = sorted(range(len(s)), key=lambda i: -s[i])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(order):
        j = i
        score = s[order[i]]
        while j + 1 < len(order) and s[order[j + 1]] == score:
            j += 1
        for k in range(i, j + 1):
            if y[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        recall = tp / npos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return ap


def confusion(y, s, thr):
    tp = sum(1 for a, b in zip(y, s) if a == 1 and b >= thr)
    fp = sum(1 for a, b in zip(y, s) if a == 0 and b >= thr)
    fn = sum(1 for a, b in zip(y, s) if a == 1 and b < thr)
    tn = sum(1 for a, b in zip(y, s) if a == 0 and b < thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "threshold": round(thr, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": round(sens, 6),
        "specificity": round(spec, 6),
        "precision": round(prec, 6),
        "f1": round(f1, 6),
    }


def best_f1(y, s):
    order = sorted(range(len(s)), key=lambda i: -s[i])
    tp = fp = 0
    best = (-1.0, 1.0, None)
    i = 0
    while i < len(order):
        j = i
        score = s[order[i]]
        while j + 1 < len(order) and s[order[j + 1]] == score:
            j += 1
        for k in range(i, j + 1):
            if y[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        fn = sum(y) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, score, (tp, fp, fn))
        i = j + 1
    return best


def discover(roots, aliases):
    runs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for stage, pattern in LAYOUTS:
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                rel = os.path.relpath(path, root)
                parts = rel.split(os.sep)
                dataset = parts[0]
                m = DATASET_RE.match(dataset)
                if not m:
                    continue
                fold_dir = next((p for p in parts if FOLD_RE.match(p)), "")
                fm = FOLD_RE.match(fold_dir or "")
                fold = int(fm.group(1)) if fm else 0
                source_dir = os.path.basename(os.path.normpath(root))
                runs.append({
                    "source": aliases.get(source_dir, source_dir),
                    "source_dir": source_dir,
                    "dataset": dataset,
                    "species": m.group("species"),
                    "drug": m.group("drug"),
                    "fold": fold,
                    "fold_dir": fold_dir,
                    "stage": stage,
                    "file": path,
                })
    # one record per (source, dataset, fold): prefer the val file when both exist
    chosen = {}
    priority = {"val": 0, "flat": 1, "predict": 2}
    dir_priority = {"models-3fold": 0, "models-baseline-extra-3fold": 1}
    for r in runs:
        key = (r["source"], r["dataset"], r["fold_dir"])
        better = key not in chosen
        if not better:
            cur = chosen[key]
            if priority[r["stage"]] < priority[cur["stage"]]:
                better = True
            elif priority[r["stage"]] == priority[cur["stage"]]:
                better = dir_priority.get(r["source_dir"], 9) < dir_priority.get(cur["source_dir"], 9)
        if better:
            chosen[key] = r
    return sorted(chosen.values(), key=lambda r: (r["dataset"], r["source"], r["fold"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="Directory containing <Dataset>/Fold<n>/... (repeatable)")
    ap.add_argument("--out", default=None, help="Output directory (must not already contain results)")
    ap.add_argument("--alias", action="append", default=None,
                    help="Map a source directory to a group, e.g. models-baseline-extra-3fold=baseline")
    ap.add_argument("--dry-run", action="store_true", help="Only list discovered files")
    args = ap.parse_args()

    roots = args.root or [
        "/data2/hmy/StrainAMR-optimize/models-3fold",
        "/data2/hmy/StrainAMR-optimize/models-3fold-ablation",
        "/data2/hmy/StrainAMR-optimize/models-baseline-extra-3fold",
        "/data2/hmy/StrainAMR-optimize/models-mtb-schemeA",
    ]
    aliases = dict(DEFAULT_ALIASES)
    for item in args.alias or []:
        if "=" not in item:
            ap.error(f"--alias expects FROM=TO, got {item!r}")
        k, v = item.split("=", 1)
        aliases[k] = v
    runs = discover(roots, aliases)
    if not runs:
        print("no prediction files found under:", ", ".join(roots), file=sys.stderr)
        return 2
    if args.dry_run:
        for r in runs:
            print(f"{r['source']:12s}{r['source_dir']:32s}{r['dataset']:34s}{r['fold_dir']:8s}{r['stage']:8s}{r['file']}")
        print(f"\n{len(runs)} run(s)")
        return 0
    if not args.out:
        ap.error("--out is required unless --dry-run is used")
    if os.path.exists(os.path.join(args.out, "summary.tsv")):
        ap.error(f"{args.out} already has summary.tsv; choose a fresh directory")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)

    results = []
    for r in runs:
        rows = read_probs(r["file"])
        if not rows:
            print(f"skip (empty): {r['file']}", file=sys.stderr)
            continue
        y = [x[1] for x in rows]
        s = [x[2] for x in rows]
        oracle_f1, oracle_thr, _ = best_f1(y, s)
        rec = dict(r)
        rec.update({
            "n": len(y),
            "n_pos": sum(y),
            "n_neg": len(y) - sum(y),
            "auc": None if roc_auc(y, s) is None else round(roc_auc(y, s), 6),
            "average_precision": None if average_precision(y, s) is None else round(average_precision(y, s), 6),
            "at_0.5": confusion(y, s, 0.5),
            "oracle": confusion(y, s, oracle_thr),
            "oracle_threshold": round(oracle_thr, 6),
        })
        rec["_y"] = y
        rec["_s"] = s
        results.append(rec)

    # cross-fold threshold: median oracle threshold of the other folds
    groups = {}
    for rec in results:
        groups.setdefault((rec["source"], rec["dataset"]), []).append(rec)
    for (_source, _dataset), recs in groups.items():
        thr = {r["fold_dir"]: r["oracle_threshold"] for r in recs}
        for rec in recs:
            others = [t for f, t in thr.items() if f != rec["fold_dir"]]
            if len(recs) >= 2 and others:
                t = statistics.median(others)
                rec["crossfold"] = confusion(rec["_y"], rec["_s"], t)
                rec["crossfold"]["source_folds"] = sorted(f for f in thr if f != rec["fold_dir"])
            else:
                rec["crossfold"] = None

    for rec in results:
        rec.pop("_y", None)
        rec.pop("_s", None)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{rec['source']}__{rec['dataset']}__{rec['fold_dir']}__{rec['stage']}")
        with open(os.path.join(args.out, "runs", safe + ".json"), "w") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)

    def fmt(v, nd=4):
        return "NA" if v is None else f"{v:.{nd}f}"

    columns = ["dataset", "source", "stage", "fold", "n", "n_pos", "auc", "ap",
               "f1@0.5", "sens@0.5", "spec@0.5",
               "f1@oracle", "thr_oracle", "sens@oracle", "spec@oracle",
               "thr_crossfold", "f1@crossfold", "sens@crossfold", "spec@crossfold"]
    lines = ["\t".join(columns)]
    for rec in results:
        cf = rec["crossfold"]
        lines.append("\t".join([
            rec["dataset"], rec["source"], rec["stage"], str(rec["fold"]), str(rec["n"]), str(rec["n_pos"]),
            fmt(rec["auc"]), fmt(rec["average_precision"]),
            fmt(rec["at_0.5"]["f1"]), fmt(rec["at_0.5"]["sensitivity"]), fmt(rec["at_0.5"]["specificity"]),
            fmt(rec["oracle"]["f1"]), fmt(rec["oracle_threshold"]),
            fmt(rec["oracle"]["sensitivity"]), fmt(rec["oracle"]["specificity"]),
            fmt(cf["threshold"]) if cf else "NA",
            fmt(cf["f1"]) if cf else "NA",
            fmt(cf["sensitivity"]) if cf else "NA",
            fmt(cf["specificity"]) if cf else "NA",
        ]))
    with open(os.path.join(args.out, "summary.tsv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # markdown
    md = ["# Threshold-aware evaluation", "",
          "`F1@0.5` is the tool default. `F1@oracle` picks the best threshold on the same",
          "fold and is therefore optimistic. `F1@crossfold` uses the median threshold of",
          "the *other* folds, which is the honest estimate of a deployed per-drug threshold.",
          "AUC and AP do not depend on any threshold.", "",
          "| dataset | source | stage | fold | n | pos | AUC | AP | F1@0.5 | Sens@0.5 | Spec@0.5 | F1@oracle | thr* | F1@crossfold | thr_cf |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec in results:
        cf = rec["crossfold"]
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            rec["dataset"], rec["source"], rec["stage"], rec["fold"], rec["n"], rec["n_pos"],
            fmt(rec["auc"], 3), fmt(rec["average_precision"], 3),
            fmt(rec["at_0.5"]["f1"], 3), fmt(rec["at_0.5"]["sensitivity"], 3), fmt(rec["at_0.5"]["specificity"], 3),
            fmt(rec["oracle"]["f1"], 3), fmt(rec["oracle_threshold"], 3),
            fmt(cf["f1"], 3) if cf else "NA",
            fmt(cf["threshold"], 3) if cf else "NA"))
    md += ["", "## Per-dataset means", "",
           "| dataset | source | folds | F1@0.5 | F1@oracle | F1@crossfold | AUC | AP |",
           "|---|---|---|---|---|---|---|---|"]
    for (source, dataset), recs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        def mean(key):
            vals = []
            for r in recs:
                if key == "auc":
                    vals.append(r["auc"])
                elif key == "ap":
                    vals.append(r["average_precision"])
                elif key == "f1_0.5":
                    vals.append(r["at_0.5"]["f1"])
                elif key == "f1_oracle":
                    vals.append(r["oracle"]["f1"])
                else:
                    vals.append(r["crossfold"]["f1"] if r["crossfold"] else None)
            vals = [v for v in vals if v is not None]
            return statistics.mean(vals) if vals else None
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            dataset, source, len(recs), fmt(mean("f1_0.5"), 3), fmt(mean("f1_oracle"), 3),
            fmt(mean("f1_cf"), 3), fmt(mean("auc"), 3), fmt(mean("ap"), 3)))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"evaluated {len(results)} run(s) -> {args.out}")
    print(os.path.join(args.out, "summary.tsv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
