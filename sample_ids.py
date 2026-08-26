"""Utilities for matching genome filenames to sample identifiers."""

import os
import re


NCBI_ACCESSION = re.compile(r"^(GC[AF]_\d+)")


def sample_id_from_filename(filename):
    """Return the sample ID used by the pipeline for a genome filename.

    NCBI assembly filenames commonly append a version and assembly name to the
    accession (for example, ``GCF_900125395.1_name_genomic.fna``), while
    phenotype files use the unversioned accession.  Collapse that filename to
    ``GCF_900125395`` and leave non-NCBI basenames unchanged.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    match = NCBI_ACCESSION.match(basename)
    return match.group(1) if match else basename


def index_genomes(genome_dir):
    """Build a sample ID -> genome path mapping, rejecting ambiguous IDs."""
    genomes = {}
    for filename in sorted(os.listdir(genome_dir)):
        path = os.path.join(genome_dir, filename)
        if not os.path.isfile(path):
            continue
        sample_id = sample_id_from_filename(filename)
        if sample_id in genomes:
            raise ValueError(
                f"Multiple genome files resolve to sample ID {sample_id!r}: "
                f"{genomes[sample_id]} and {path}"
            )
        genomes[sample_id] = path
    return genomes


def read_label_ids(label_file):
    """Read first-column sample IDs after the label file's header."""
    ids = []
    with open(label_file, "r") as handle:
        next(handle, None)
        for line_number, line in enumerate(handle, start=2):
            fields = line.split()
            if not fields:
                continue
            sample_id = fields[0]
            if sample_id in ids:
                raise ValueError(
                    f"Duplicate sample ID {sample_id!r} in {label_file} "
                    f"(line {line_number})"
                )
            ids.append(sample_id)
    return ids


def validate_label_genomes(label_file, genomes):
    """Require a one-to-one match between label IDs and normalized genomes."""
    label_ids = read_label_ids(label_file)
    label_set = set(label_ids)
    genome_set = set(genomes)
    missing = sorted(label_set - genome_set)
    unlabeled = sorted(genome_set - label_set)
    if missing or unlabeled:
        details = []
        if missing:
            details.append("label IDs without genome files: " + ", ".join(missing))
        if unlabeled:
            details.append("genome IDs without labels: " + ", ".join(unlabeled))
        raise ValueError(
            "Genome filenames and label IDs do not match after NCBI accession "
            "normalization; " + "; ".join(details)
        )
    return label_ids
