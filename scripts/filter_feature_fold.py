#!/usr/bin/env python3
"""Mirror a StrainAMR feature fold while removing selected sample IDs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


TEXT_SUFFIXES = (".txt", ".tsv")


def filter_text_file(src: str, dst: str, excluded: set[str]) -> tuple[int, bool]:
    """Write dst without excluded rows; return (removed, was_filtered)."""
    fd, tmp_path = tempfile.mkstemp(prefix=".filter-", dir=os.path.dirname(dst))
    removed = 0
    try:
        with os.fdopen(fd, "w") as out, open(src) as source:
            header = source.readline()
            if not header:
                os.unlink(tmp_path)
                return 0, False
            out.write(header)
            for line in source:
                sample_id = line.split("\t", 1)[0].strip()
                if sample_id in excluded:
                    removed += 1
                else:
                    out.write(line)
        if removed:
            os.replace(tmp_path, dst)
            return removed, True
        os.unlink(tmp_path)
        return 0, False
    except UnicodeDecodeError:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return 0, False
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def link_file(src: str, dst: str) -> None:
    os.symlink(os.path.abspath(src), dst)


def mirror_tree(src_root: str, dst_root: str, excluded: set[str]) -> dict:
    stats = {"files": 0, "symlinks": 0, "filtered_files": 0, "removed_rows": 0}
    for current, dirs, files in os.walk(src_root):
        dirs.sort()
        files.sort()
        relative = os.path.relpath(current, src_root)
        target_dir = dst_root if relative == "." else os.path.join(dst_root, relative)
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            src = os.path.join(current, name)
            dst = os.path.join(target_dir, name)
            stats["files"] += 1
            removed = 0
            filtered = False
            if name.endswith(TEXT_SUFFIXES):
                removed, filtered = filter_text_file(src, dst, excluded)
            if filtered:
                stats["filtered_files"] += 1
                stats["removed_rows"] += removed
            else:
                link_file(src, dst)
                stats["symlinks"] += 1
    return stats


def row_ids(path: str) -> list[str]:
    with open(path) as handle:
        handle.readline()
        return [line.split("\t", 1)[0].strip() for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-sample", action="append", required=True)
    args = parser.parse_args()

    src_fold = os.path.abspath(args.fold_dir)
    out_dir = os.path.abspath(args.out_dir)
    excluded = set(args.exclude_sample)
    if os.path.exists(out_dir) and any(os.scandir(out_dir)):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"source_fold": src_fold, "excluded_samples": sorted(excluded), "trees": {}}
    for name in ("db", "val_features"):
        src = os.path.join(src_fold, name)
        if not os.path.isdir(src):
            raise FileNotFoundError(src)
        summary["trees"][name] = mirror_tree(src, os.path.join(out_dir, name), excluded)

    checks = {
        "train": [
            ("db/strains_train_sentence_fs.txt", "db/strains_train_pc_token_fs.txt"),
            ("db/strains_train_sentence_fs.txt", "db/strains_train_kmer_token.txt"),
        ],
        "val": [
            ("val_features/strains_test_sentence_fs.txt", "val_features/strains_test_pc_token_fs.txt"),
            ("val_features/strains_test_sentence_fs.txt", "val_features/strains_test_kmer_token.txt"),
        ],
    }
    summary["channel_checks"] = {}
    for split, pairs in checks.items():
        counts = []
        reference_ids = None
        for left, right in pairs:
            left_ids = row_ids(os.path.join(out_dir, left))
            right_ids = row_ids(os.path.join(out_dir, right))
            if left_ids != right_ids:
                raise ValueError(f"{split} channel IDs differ: {left} vs {right}")
            reference_ids = left_ids
            counts.extend([len(left_ids), len(right_ids)])
        if any(sample_id in excluded for sample_id in reference_ids):
            raise ValueError(f"excluded sample remains in {split} features")
        summary["channel_checks"][split] = {"rows": len(reference_ids), "ids_identical": True}

    with open(os.path.join(out_dir, "filter_manifest.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
