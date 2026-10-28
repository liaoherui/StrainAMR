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


def _preview(ids, n=8):
    ids = sorted(ids)
    return ", ".join(ids[:n]) + (f" ... (+{len(ids) - n} more)" if len(ids) > n else "")


def validate_label_genomes(label_file, genomes, require_all_genomes_labeled=False):
    """Match label IDs to genome files, and say clearly which side is short.

    The two mismatches are not equally serious:

    * a label with no genome file is fatal -- the file is missing or misnamed, and silently
      training on fewer genomes than the label file describes would hide that;
    * a genome file with no label is skipped with a warning. Extra assemblies sitting in the input
      directory are common (the bundled Sau example has 61 genomes for 50 labels), and there is
      nothing to train on for them anyway.

    Pass require_all_genomes_labeled=True to make the second case fatal as well.
    """
    label_ids = read_label_ids(label_file)
    label_set = set(label_ids)
    genome_set = set(genomes)
    missing = label_set - genome_set
    unlabeled = genome_set - label_set
    if missing:
        raise ValueError(
            f"{len(missing)} label IDs have no genome file (after NCBI accession normalization): "
            + _preview(missing)
            + f"\n  {len(genome_set)} genome files found in the input directory."
            + "\n  Check that the label file's first column matches the genome filenames."
        )
    if unlabeled:
        msg = (f"{len(unlabeled)} genome files have no label and will be skipped: "
               + _preview(unlabeled)
               + f"\n  ({len(label_set)} of {len(genome_set)} genomes will be used.)")
        if require_all_genomes_labeled:
            raise ValueError(msg)
        print("WARNING: " + msg, flush=True)
    return label_ids
