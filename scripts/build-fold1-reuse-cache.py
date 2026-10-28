#!/usr/bin/env python3
"""Build a per-sample cache for Fold2/Fold3 from Fold1 feature outputs.

Only fold-independent, per-genome results are reused: gene calling, protein
prediction, RGI output and AMRFinderPlus hits. Fold-dependent artifacts
(variation graphs, k-mer/PC/SNV token vocabularies, feature selection) are
still rebuilt by `strainamr build-train` for each fold.
"""

import argparse
import csv
import gzip
import json
import os
import shutil
from pathlib import Path

MODALITIES = (
    ("Genes", ".fa"),
    ("Genes_info", ".genes"),
    ("Proteins", ".faa"),
    ("RGI_raw", ".txt"),
)

TRAIN_DIRS = {
    "Genes": "Genes",
    "Genes_info": "Genes_info",
    "Proteins": "Proteins",
    "RGI_raw": "RGI_raw",
}

TEST_DIRS = {
    "Genes": "Genes_ts",
    "Genes_info": "Genes_info_ts",
    "Proteins": "Proteins_ts",
    "RGI_raw": "RGI_raw_ts",
}


def read_samples(paths):
    samples = set()
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "Strain_ID" not in (reader.fieldnames or []):
                raise SystemExit(f"{path}: missing Strain_ID column")
            for row in reader:
                sample = (row.get("Strain_ID") or "").strip()
                if sample:
                    samples.add(sample)
    return sorted(samples)


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 0:
        return "existing"

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def build_links(fold1_dir, samples, cache_dir):
    source_split = {}
    counts = {"train": 0, "val": 0, "linked": 0, "copied": 0, "existing": 0}

    for sample in samples:
        candidates = (
            ("train", fold1_dir / "db", TRAIN_DIRS),
            ("val", fold1_dir / "val_features", TEST_DIRS),
        )
        chosen = None
        for split, base, dirs in candidates:
            if (base / dirs["Genes"] / f"{sample}.fa").is_file():
                chosen = (split, base, dirs)
                break

        if chosen is None:
            raise SystemExit(f"{sample}: no Fold1 Genes output found")

        split, base, dirs = chosen
        source_split[sample] = split
        counts[split] += 1

        for logical, suffix in MODALITIES:
            source = base / dirs[logical] / f"{sample}{suffix}"
            if not source.is_file() or source.stat().st_size == 0:
                raise SystemExit(f"{sample}: missing or empty {source}")
            status = link_or_copy(
                source,
                cache_dir / logical / f"{sample}{suffix}",
            )
            counts[status] += 1

    return source_split, counts


def split_amrfinder(paths, samples, output_dir):
    sample_set = set(samples)
    header = None
    grouped = {sample: [] for sample in samples}

    for path in paths:
        if not path.is_file():
            continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if header is None:
                header = reader.fieldnames
            elif reader.fieldnames != header:
                raise SystemExit(f"{path}: AMRFinder header mismatch")
            for row in reader:
                genome = (row.get("Genome") or "").strip()
                if genome not in sample_set:
                    raise SystemExit(f"{path}: unexpected Genome={genome!r}")
                grouped[genome].append(row)

    if header is None:
        raise SystemExit("no AMRFinder source files found")

    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        destination = output_dir / f"{sample}.tsv.gz"
        with gzip.open(destination, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=header,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in grouped[sample]:
                writer.writerow(row)

    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold1-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--amrfinder-out", required=True, type=Path)
    parser.add_argument("--skip-amrfinder", action="store_true")
    args = parser.parse_args()

    samples = read_samples(args.manifests)
    source_split, counts = build_links(args.fold1_dir, samples, args.cache_dir)

    if args.skip_amrfinder:
        grouped = {}
    else:
        grouped = split_amrfinder(
            (
                args.fold1_dir / "db" / "mechanism" / "amrfinder_hits.tsv.gz",
                args.fold1_dir / "val_features" / "mechanism" / "amrfinder_hits.tsv.gz",
            ),
            samples,
            args.amrfinder_out,
        )

    manifest = {
        "fold1_dir": str(args.fold1_dir),
        "cache_dir": str(args.cache_dir),
        "samples": len(samples),
        "samples_from_fold1_train": counts["train"],
        "samples_from_fold1_val": counts["val"],
        "amrfinder_skipped": args.skip_amrfinder,
        "amrfinder_samples": len(grouped),
        "amrfinder_samples_with_hits": sum(1 for rows in grouped.values() if rows),
        "amrfinder_hits": sum(len(rows) for rows in grouped.values()),
        "linked_files": counts["linked"],
        "copied_files": counts["copied"],
        "existing_files": counts["existing"],
    }

    manifest_path = args.cache_dir / "cache-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
