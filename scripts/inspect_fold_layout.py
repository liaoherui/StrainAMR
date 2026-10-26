#!/usr/bin/env python3
"""Inspect the per-fold directory layout and completeness. Read-only.

Section 1: full tree of one representative fold (features / model / prediction),
           with file sizes and per-directory file counts.
Section 2: coverage table - every relative path found under any fold of the same
           kind, and in how many of the 72 folds it appears.  Anything below
           72/72 is a "not every fold has this" signal.
Section 3: hard checklist - markers, best checkpoint, selected_model.txt,
           token_lengths.txt, prediction probabilities; lists what is missing.

Usage:
    python3 scripts/inspect_fold_layout.py                    # defaults
    python3 scripts/inspect_fold_layout.py --dataset Mtb_amoxicillin_3fold --fold Fold1
    python3 scripts/inspect_fold_layout.py --depth 2 > fold-structure.txt
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

WORKROOT = "/data2/hmy/StrainAMR-optimize"
FEATURE_ROOT = os.path.join(WORKROOT, "features-3fold")
MODEL_ROOTS = [
    os.path.join(WORKROOT, "models-3fold"),
    os.path.join(WORKROOT, "models-mtb-schemeA"),
    os.path.join(WORKROOT, "models-3fold-ablation"),
]
DATASETS = ["Fold1", "Fold2", "Fold3"]
CPU_DIR_RE = re.compile(r"^(model|prediction)-gpu(\d+)$")


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0


def count_files(path, cap=200000):
    total = 0
    for _dirpath, _dirnames, filenames in os.walk(path):
        total += len(filenames)
        if total >= cap:
            return total, True
    return total, False


def norm_rel(rel):
    parts = rel.split(os.sep)
    if parts and CPU_DIR_RE.match(parts[0]):
        parts[0] = CPU_DIR_RE.sub(r"\1-gpuN", parts[0])
    return "/".join(parts)


def collect(fold_path, max_depth):
    """Return set of normalised relative paths (files and dirs) up to max_depth."""
    found = set()
    if not os.path.isdir(fold_path):
        return found
    base_depth = fold_path.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(fold_path):
        depth = dirpath.count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        for d in dirnames:
            found.add(norm_rel(os.path.relpath(os.path.join(dirpath, d), fold_path)) + "/")
        for f in filenames:
            found.add(norm_rel(os.path.relpath(os.path.join(dirpath, f), fold_path)))
    return found


def print_tree(path, max_depth, indent=0, show_counts=True, prefix=""):
    if not os.path.isdir(path):
        print(f"{'  ' * indent}[缺少] {path}")
        return
    pad = "  " * indent
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        print(f"{pad}[无法读取] {path} ({exc})")
        return
    for name in entries:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            n_files, capped = count_files(full) if show_counts else (0, False)
            cap = "+" if capped else ""
            print(f"{pad}{name}/  ({n_files}{cap} files)")
            if indent + 1 < max_depth:
                print_tree(full, max_depth, indent + 1, show_counts)
        else:
            try:
                size = os.path.getsize(full)
            except OSError:
                size = -1
            print(f"{pad}{name}  ({human(size) if size >= 0 else '?'})")


def gpu_dirs(fold_path, kind):
    if not os.path.isdir(fold_path):
        return []
    return sorted(
        os.path.join(fold_path, d) for d in os.listdir(fold_path)
        if os.path.isdir(os.path.join(fold_path, d)) and re.match(rf"^{kind}-gpu\d+$", d)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Mtb_ofloxacin_3fold")
    ap.add_argument("--fold", default="Fold1")
    ap.add_argument("--depth", type=int, default=3, help="depth for the coverage scan")
    ap.add_argument("--tree-depth", type=int, default=3)
    args = ap.parse_args()

    # ---------------------------------------------------------------- section 1
    print("=" * 78)
    print(f"1. 代表性折的目录树: {args.dataset}/{args.fold}")
    print("=" * 78)

    feat_fold = os.path.join(FEATURE_ROOT, args.dataset, args.fold)
    print(f"\n--- 特征 (features-3fold/{args.dataset}/{args.fold}) ---")
    print_tree(feat_fold, args.tree_depth)
    vf = os.path.join(feat_fold, "val_features")
    if os.path.isdir(vf):
        print(f"\n    val_features 各子项的文件数:")
        for name in sorted(os.listdir(vf)):
            full = os.path.join(vf, name)
            if os.path.isdir(full):
                n, capped = count_files(full)
                print(f"      {name}/  ({n}{'+' if capped else ''} files)")
            else:
                print(f"      {name}  ({human(os.path.getsize(full))})")

    for root in MODEL_ROOTS:
        fold_path = os.path.join(root, args.fold and os.path.join(root, args.dataset, args.fold))
        fold_path = os.path.join(root, args.dataset, args.fold)
        if not os.path.isdir(fold_path):
            continue
        print(f"\n--- 模型/预测 ({os.path.basename(root)}/{args.dataset}/{args.fold}) ---")
        print_tree(fold_path, args.tree_depth)

    # ---------------------------------------------------------------- section 2
    print()
    print("=" * 78)
    print(f"2. 全部折的产物覆盖率 (depth<={args.depth})")
    print("=" * 78)

    kinds = [("features", [(FEATURE_ROOT, "")] )]
    model_pairs = []
    for root in MODEL_ROOTS:
        model_pairs.append((root, os.path.basename(root)))
    kinds = [("features", [FEATURE_ROOT])]
    kinds.append(("models", MODEL_ROOTS))

    for kind, roots in kinds:
        counter = Counter()
        total = 0
        for root in roots:
            if not os.path.isdir(root):
                continue
            for ds in sorted(os.listdir(root)):
                ds_path = os.path.join(root, ds)
                if not os.path.isdir(ds_path):
                    continue
                for fold in DATASETS:
                    fold_path = os.path.join(ds_path, fold)
                    if not os.path.isdir(fold_path):
                        continue
                    total += 1
                    for rel in collect(fold_path, args.depth):
                        counter[rel] += 1
        print(f"\n--- {kind}: {total} 个折目录 ---")
        partial = [(rel, n) for rel, n in counter.items() if n < total]
        full = [(rel, n) for rel, n in counter.items() if n == total]
        print(f"  只在部分折出现 ({len(partial)} 项):")
        for rel, n in sorted(partial, key=lambda kv: (kv[1], kv[0])):
            print(f"    {n:>3}/{total}  {rel}")
        print(f"  每个折都有 ({len(full)} 项):")
        for rel, n in sorted(full):
            print(f"    {n:>3}/{total}  {rel}")

    # ---------------------------------------------------------------- section 3
    print()
    print("=" * 78)
    print("3. 关键产物硬检查")
    print("=" * 78)

    missing = defaultdict(list)
    checked = 0
    for ds in sorted(os.listdir(FEATURE_ROOT)):
        ds_path = os.path.join(FEATURE_ROOT, ds)
        if not os.path.isdir(ds_path):
            continue
        for fold in DATASETS:
            feat = os.path.join(ds_path, fold)
            if not os.path.isdir(feat):
                missing["features: 整个折目录"].append(f"{ds}/{fold}")
                continue
            checked += 1
            for marker in (".build-train.complete", ".build-val.complete"):
                if not os.path.exists(os.path.join(feat, marker)):
                    missing[f"features: {marker}"].append(f"{ds}/{fold}")
            for sub in ("db", "val_features"):
                if not os.path.isdir(os.path.join(feat, sub)):
                    missing[f"features: {sub}/"].append(f"{ds}/{fold}")

            model_hits = []
            pred_hits = []
            for root in MODEL_ROOTS:
                base = os.path.join(root, ds, fold)
                for md in gpu_dirs(base, "model"):
                    if os.path.exists(os.path.join(md, "models", "best_model_f1_score.pt")):
                        model_hits.append(md)
                    for rel in ("selected_model.txt", "token_lengths.txt"):
                        if not os.path.exists(os.path.join(md, rel)):
                            missing[f"model: {rel}"].append(f"{ds}/{fold}/{os.path.basename(md)}")
                for pd in gpu_dirs(base, "prediction"):
                    if os.path.exists(os.path.join(pd, "logs", "output_sample_prob_predict.txt")):
                        pred_hits.append(pd)
            if not model_hits:
                missing["model: best_model_f1_score.pt"].append(f"{ds}/{fold}")
            if not pred_hits:
                missing["prediction: output_sample_prob_predict.txt"].append(f"{ds}/{fold}")

    print(f"检查了 {checked} 个 baseline 折目录")
    if not missing:
        print("全部通过：无缺失项")
    for key in sorted(missing):
        items = missing[key]
        print(f"\n[缺] {key}  ({len(items)} 处)")
        for it in items:
            print(f"    {it}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
