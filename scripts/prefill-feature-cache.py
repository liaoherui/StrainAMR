#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import os
import shutil
from pathlib import Path


def read_samples(path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty label file")

        id_column = (
            "Strain_ID"
            if "Strain_ID" in reader.fieldnames
            else reader.fieldnames[0]
        )

        return sorted({
            row[id_column].strip()
            for row in reader
            if row.get(id_column, "").strip()
        })


def resolve_cache_file(cache_dir, sample, suffix, cache_prefix):
    candidates = [sample]

    if cache_prefix and not sample.startswith(cache_prefix):
        candidates.append(f"{cache_prefix}{sample}")

    for cache_sample in candidates:
        source = cache_dir / f"{cache_sample}{suffix}"

        if source.is_file() and source.stat().st_size > 0:
            return source, cache_sample

    return None, None


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 0:
        return "existing"

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)

    return "linked"


def prefill_files(
    samples,
    cache_root,
    work_dir,
    layout,
    cache_prefix,
):
    if layout == "train":
        mapping = [
            ("Genes", ".fa", "Genes"),
            ("Genes_info", ".genes", "Genes_info"),
            ("Proteins", ".faa", "Proteins"),
            ("RGI_raw", ".txt", "RGI_raw"),
        ]
    else:
        mapping = [
            ("Genes", ".fa", "Genes_ts"),
            ("Genes_info", ".genes", "Genes_info_ts"),
            ("Proteins", ".faa", "Proteins_ts"),
            ("RGI_raw", ".txt", "RGI_raw_ts"),
        ]

    summary = {}

    for cache_name, suffix, output_name in mapping:
        source_dir = cache_root / cache_name
        output_dir = work_dir / output_name

        linked = 0
        existing = 0
        missing = []

        for sample in samples:
            source, cache_sample = resolve_cache_file(
                source_dir,
                sample,
                suffix,
                cache_prefix,
            )
            destination = output_dir / f"{sample}{suffix}"

            if source is None:
                missing.append(sample)
                continue

            status = link_or_copy(source, destination)

            if status == "linked":
                linked += 1
            else:
                existing += 1

        summary[output_name] = {
            "linked": linked,
            "existing": existing,
            "missing": missing,
        }

    return summary


def rebuild_amrfinder(
    samples,
    amr_cache,
    work_dir,
    organism,
    mode,
    software_version,
    database_version,
    cache_prefix,
):
    mechanism_dir = work_dir / "mechanism"
    resolved_sources = {}

    for sample in samples:
        source, cache_sample = resolve_cache_file(
            amr_cache,
            sample,
            ".tsv.gz",
            cache_prefix,
        )

        if source is not None:
            resolved_sources[sample] = (source, cache_sample)

    missing = [
        sample
        for sample in samples
        if sample not in resolved_sources
    ]

    marker_path = mechanism_dir / ".amrfinder-cache-complete.json"

    if missing:
        if marker_path.exists():
            marker_path.unlink()
        return {
            "ready": False,
            "missing": missing,
            "hits": 0,
        }

    header = None
    rows = []

    for sample in samples:
        source, cache_sample = resolved_sources[sample]

        with gzip.open(
            source,
            "rt",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")

            if header is None:
                header = reader.fieldnames

            if reader.fieldnames != header:
                raise RuntimeError(f"AMRFinder header mismatch: {source}")

            for row in reader:
                cached_genome = row.get("Genome", "").strip()

                if cached_genome != cache_sample:
                    raise RuntimeError(
                        f"{source}: unexpected Genome={cached_genome!r}; "
                        f"expected cache sample {cache_sample!r}"
                    )

                normalized = dict(row)
                normalized["Genome"] = sample
                rows.append(normalized)

    if header is None:
        raise RuntimeError("No AMRFinder cache headers found")

    rows.sort(
        key=lambda row: (
            row.get("Genome", ""),
            row.get("Element", ""),
            row.get("Protein", ""),
        )
    )

    mechanism_dir.mkdir(parents=True, exist_ok=True)

    hits_tmp = mechanism_dir / ".amrfinder_hits.tsv.gz.tmp"
    hits_final = mechanism_dir / "amrfinder_hits.tsv.gz"

    with gzip.open(
        hits_tmp,
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=header,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(hits_tmp, hits_final)

    screened_tmp = mechanism_dir / ".screened_genomes.txt.tmp"
    screened_final = mechanism_dir / "screened_genomes.txt"

    screened_tmp.write_text(
        "\n".join(samples) + "\n",
        encoding="utf-8",
    )
    os.replace(screened_tmp, screened_final)

    marker = {
        "organism": organism,
        "mode": mode,
        "software_version": software_version,
        "database_version": database_version,
        "samples": samples,
        "sample_count": len(samples),
        "hit_count": len(rows),
    }

    marker_tmp = mechanism_dir / ".amrfinder-cache-complete.json.tmp"
    marker_tmp.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker_path)

    return {
        "ready": True,
        "missing": [],
        "hits": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--sample-cache", required=True, type=Path)
    parser.add_argument("--amrfinder-cache", required=True, type=Path)
    parser.add_argument(
        "--layout",
        choices=["train", "test"],
        required=True,
    )
    parser.add_argument("--organism", required=True)
    parser.add_argument("--mode", default="protein")
    parser.add_argument("--software-version", required=True)
    parser.add_argument("--database-version", required=True)
    parser.add_argument("--cache-prefix", default="")
    parser.add_argument("--skip-amrfinder", action="store_true")
    args = parser.parse_args()

    samples = read_samples(args.labels)

    summary = prefill_files(
        samples,
        args.sample_cache,
        args.work_dir,
        args.layout,
        args.cache_prefix,
    )

    if args.skip_amrfinder:
        amr = None
    else:
        amr = rebuild_amrfinder(
            samples,
            args.amrfinder_cache,
            args.work_dir,
            args.organism,
            args.mode,
            args.software_version,
            args.database_version,
            args.cache_prefix,
        )

    print("Samples:", len(samples))

    for directory, result in summary.items():
        print(
            f"{directory}: linked={result['linked']} "
            f"existing={result['existing']} "
            f"missing={len(result['missing'])}"
        )

    if amr is None:
        print("AMRFinder cache: SKIPPED")
    else:
        print(
            "AMRFinder cache:",
            "READY" if amr["ready"] else "INCOMPLETE",
            f"hits={amr['hits']}",
            f"missing={len(amr['missing'])}",
        )

        if amr["missing"]:
            print(
                "AMRFinder missing examples:",
                ", ".join(amr["missing"][:10]),
            )


if __name__ == "__main__":
    main()
