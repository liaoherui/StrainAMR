#!/usr/bin/env python3
"""Build Mtb Scheme-A features without modifying an existing StrainAMR fold.

Scheme A keeps the existing three-channel Transformer and replaces only the
SNV/graph channel with tokens derived from an H37Rv reference region.  PC and
k-mer features are linked read-only from an existing baseline fold.

All outputs go to a new directory.  The script refuses to write into a
non-empty output directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict


DRUG_REGIONS = {
    "ofloxacin": "gyrBA",
    "ciprofloxacin": "gyrBA",
    # These are not enabled in the current experiment, but the region is the
    # same and keeping the mapping explicit avoids accidental cipro/oflox
    # mismatches later.
    "levofloxacin": "gyrBA",
    "moxifloxacin": "gyrBA",
}

# The paper's Table 3 coordinates are 0-based, half-open: end - start equals
# the reported locus length.  Keep that convention here rather than silently
# adding one base at either boundary.
REGIONS = {
    "gyrBA": (4997, 9818),
}

SNV_TOPLEVEL_SKIP = {
    "strains_train_sentence.txt",
    "strains_train_sentence_fs.txt",
    "strains_test_sentence.txt",
    "strains_test_sentence_fs.txt",
    "node_token_match.txt",
    "feature_remain_graph.txt",
    "longest_len_fs.txt",
}

SNV_SHAP_PREFIXES = (
    "strains_train_sentence",
    "strains_test_sentence",
)


def log(msg: str) -> None:
    print(f"[mtb-schemeA] {msg}", flush=True)


def read_fasta(path: str) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in seqs:
                    raise ValueError(f"duplicate FASTA record {name!r} in {path}")
                seqs[name] = []
            else:
                if name is None:
                    raise ValueError(f"sequence before first FASTA header in {path}")
                seqs[name].append(line.upper())
    if not seqs:
        raise ValueError(f"no FASTA records in {path}")
    return {k: "".join(v) for k, v in seqs.items()}


def choose_reference(seqs: dict[str, str], requested: str | None) -> tuple[str, str]:
    if requested:
        if requested in seqs:
            return requested, seqs[requested]
        matches = [k for k in seqs if k.startswith(requested)]
        if len(matches) == 1:
            return matches[0], seqs[matches[0]]
        raise ValueError(
            f"reference record {requested!r} not found; available: {', '.join(sorted(seqs))}"
        )
    preferred = [k for k in seqs if k == "NC_000962.3" or k.startswith("NC_000962.3 ")]
    if preferred:
        return preferred[0], seqs[preferred[0]]
    if len(seqs) == 1:
        key = next(iter(seqs))
        return key, seqs[key]
    raise ValueError(
        "reference FASTA has multiple records; pass --reference-name. "
        f"Available: {', '.join(sorted(seqs))}"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def read_labels(path: str) -> tuple[list[str], list[tuple[str, str]]]:
    with open(path) as fh:
        header_line = fh.readline().rstrip("\n")
        header = header_line.split("\t")
        if len(header) < 2:
            header = header_line.split()
        rows = []
        seen = set()
        for lineno, raw in enumerate(fh, 2):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"label line {lineno} in {path} has fewer than two columns")
            sid = parts[0]
            if sid in seen:
                raise ValueError(f"duplicate sample ID {sid!r} in {path}")
            seen.add(sid)
            rows.append((sid, parts[1]))
        if not rows:
            raise ValueError(f"no labeled samples in {path}")
        if len(header) < 2:
            header = ["Strain_ID", "Label"]
        return header[:2], rows


def index_genomes(path: str) -> dict[str, str]:
    out = {}
    for entry in os.scandir(path):
        if not (entry.is_file() or entry.is_symlink()):
            continue
        name = entry.name
        if name.endswith((".fai", ".gzi")):
            continue
        sid = os.path.splitext(name)[0]
        if sid in out:
            raise ValueError(f"duplicate genome prefix {sid!r} in {path}")
        out[sid] = entry.path
    if not out:
        raise ValueError(f"no genome files in {path}")
    return out


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def token_id_for_event(pos: int, kind: str, allele: str, length: int) -> int:
    """Stable, train/test-independent token ID.

    Eleven IDs are reserved for every reference position: four SNP alleles,
    three insertion-size bins (1, 2, >=3) and three deletion-size bins.
    Token ID 0 remains the model's padding token.
    """
    base = (int(pos) - 1) * 11
    if kind == "snp":
        offset = {"A": 1, "C": 2, "G": 3, "T": 4}[allele]
    elif kind == "ins":
        offset = 5 + min(max(int(length), 1), 3) - 1
    elif kind == "del":
        offset = 8 + min(max(int(length), 1), 3) - 1
    else:
        raise ValueError(kind)
    return base + offset


def event_label(region: str, start0: int, pos: int, kind: str, ref_base: str,
                allele: str, length: int) -> str:
    # pos is 1-based inside the extracted region; start0 is 0-based.
    h37rv_pos = start0 + pos
    if kind == "snp":
        return f"{region}:H37Rv_{h37rv_pos}:{ref_base}>{allele}"
    if kind == "ins":
        return f"{region}:H37Rv_{h37rv_pos}:ins_len_{min(length, 3)}{'+' if length >= 3 else ''}"
    return f"{region}:H37Rv_{h37rv_pos}:del_len_{min(length, 3)}{'+' if length >= 3 else ''}"


def write_token_mapping(path: str, region: str, start0: int, ref_region: str) -> None:
    with open(path, "w") as out:
        out.write("GeneID\tNodeID\tTokenID\n")
        for idx, ref_base in enumerate(ref_region, 1):
            if ref_base not in "ACGT":
                continue
            for alt in "ACGT":
                if alt == ref_base:
                    continue
                tid = token_id_for_event(idx, "snp", alt, 1)
                name = event_label(region, start0, idx, "snp", ref_base, alt, 1)
                out.write(f"{name}\t{ref_base}>{alt}\t{tid}\n")
            for length in (1, 2, 3):
                tid = token_id_for_event(idx, "ins", "", length)
                name = event_label(region, start0, idx, "ins", ref_base, "", length)
                out.write(f"{name}\tins_{length}\t{tid}\n")
                tid = token_id_for_event(idx, "del", "", length)
                name = event_label(region, start0, idx, "del", ref_base, "", length)
                out.write(f"{name}\tdel_{length}\t{tid}\n")


def cigar_events(qseq: str, ref_region: str, tstart: int, cigar: str, mapq: int):
    events: dict[tuple[int, str, str, int], list[int]] = {}
    covered: set[int] = set()
    qpos = 0
    rpos = int(tstart)

    def add(pos: int, kind: str, allele: str, length: int) -> None:
        pos = max(1, min(int(pos), len(ref_region)))
        key = (pos, kind, allele, int(length))
        rec = events.setdefault(key, [0, 0])
        rec[0] += 1
        rec[1] = max(rec[1], int(mapq))

    for raw_len, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        n = int(raw_len)
        if n <= 0:
            continue
        if op in ("M", "=", "X"):
            for i in range(n):
                ri = rpos + i
                qi = qpos + i
                if 0 <= ri < len(ref_region):
                    covered.add(ri + 1)
                    if 0 <= qi < len(qseq):
                        rb = ref_region[ri]
                        qb = qseq[qi]
                        if rb in "ACGT" and qb in "ACGT" and rb != qb:
                            add(ri + 1, "snp", qb, 1)
            rpos += n
            qpos += n
        elif op == "I":
            anchor = rpos if rpos > 0 else int(tstart) + 1
            add(anchor, "ins", "", n)
            qpos += n
        elif op == "D":
            add(rpos + 1, "del", "", n)
            rpos += n
        elif op == "N":
            rpos += n
        elif op == "S":
            qpos += n
        elif op in ("H", "P"):
            continue
    return events, covered


def minimap2_path(requested: str | None) -> str:
    if requested:
        if not os.path.exists(requested):
            raise FileNotFoundError(f"minimap2 not found: {requested}")
        return requested
    found = shutil.which("minimap2")
    if found:
        return found
    env_path = "/home/lhr/miniconda3/envs/strainamr/bin/minimap2"
    if os.path.exists(env_path):
        return env_path
    raise FileNotFoundError("minimap2 not found; pass --minimap2 /path/to/minimap2")


def sample_events(genome_path: str, ref_fa: str, ref_region: str, minimap2: str,
                  threads: int, min_mapq: int, min_alen: int):
    cmd = [
        minimap2, "-x", "asm5", "-c", "--secondary=no",
        "-t", str(max(1, int(threads))), ref_fa, genome_path,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"minimap2 failed for {genome_path} (exit {proc.returncode}): "
            f"{proc.stderr[-1000:]}"
        )

    genome = read_fasta(genome_path)
    votes: Counter = Counter()
    best_mapq: dict[tuple[int, str, str, int], int] = defaultdict(int)
    covered: set[int] = set()
    alignments = 0

    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        qname, _qlen, qstart, qend, strand = fields[0], fields[1], fields[2], fields[3], fields[4]
        tstart, _tend, _nmatch, alen, mapq = fields[7], fields[8], fields[9], fields[10], fields[11]
        if int(alen) < int(min_alen) or int(mapq) < int(min_mapq):
            continue
        cigar = None
        for tag in fields[12:]:
            if tag.startswith("cg:Z:"):
                cigar = tag[5:]
                break
        if not cigar or qname not in genome:
            continue
        qseq = genome[qname][int(qstart):int(qend)].upper()
        if strand == "-":
            qseq = reverse_complement(qseq)
        ev, cov = cigar_events(qseq, ref_region, int(tstart), cigar, int(mapq))
        covered.update(cov)
        for key, rec in ev.items():
            votes[key] += rec[0]
            best_mapq[key] = max(best_mapq[key], rec[1])
        alignments += 1

    # Keep at most one event per reference position.  This prevents a
    # duplicated/overlapping alignment from emitting two alleles at one site.
    by_pos: dict[int, list[tuple[int, int, tuple[int, str, str, int]]]] = defaultdict(list)
    for key, count in votes.items():
        by_pos[key[0]].append((count, best_mapq[key], key))
    selected = []
    for _pos, candidates in by_pos.items():
        candidates.sort(key=lambda x: (x[0], x[1], -x[2][0]), reverse=True)
        selected.append(candidates[0][2])

    selected.sort()
    coverage_fraction = len(covered) / float(len(ref_region))
    return selected, coverage_fraction, alignments


def write_sentence(path: str, header: list[str], label_rows: list[tuple[str, str]],
                   tokens_by_id: dict[str, list[int]]) -> int:
    width = 1
    with open(path, "w") as out:
        out.write(f"{header[0]}\t{header[1]}\tTokens_Num\tTokens\n")
        for sid, label in label_rows:
            tokens = sorted(set(int(x) for x in tokens_by_id.get(sid, [])))
            width = max(width, len(tokens))
            field = ",".join(str(x) for x in tokens) if tokens else "0"
            out.write(f"{sid}\t{label}\t{len(tokens)}\t{field}\n")
    return width


def link_or_copy(src: str, dst: str, copy_file: bool = False) -> None:
    if os.path.lexists(dst):
        raise FileExistsError(dst)
    if copy_file:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_top_level(src_dir: str, dst_dir: str, copy_baseline: bool) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for entry in sorted(os.scandir(src_dir), key=lambda e: e.name):
        if not (entry.is_file() or entry.is_symlink()):
            continue
        if entry.name in SNV_TOPLEVEL_SKIP or entry.name.startswith(".build-"):
            continue
        if entry.name == "build_params.json":
            shutil.copy2(entry.path, os.path.join(dst_dir, entry.name))
        else:
            link_or_copy(entry.path, os.path.join(dst_dir, entry.name), copy_baseline)


def mirror_shap(src_dir: str, dst_dir: str, copy_baseline: bool) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        return
    for entry in sorted(os.scandir(src_dir), key=lambda e: e.name):
        if not (entry.is_file() or entry.is_symlink()):
            continue
        if entry.name.startswith(SNV_SHAP_PREFIXES):
            continue
        if entry.name == "longest_len_fs_shap.txt":
            continue
        link_or_copy(entry.path, os.path.join(dst_dir, entry.name), copy_baseline)


def filter_feature_file(path: str, excluded: set[str]) -> int:
    """Remove rows for excluded sample IDs while preserving file order.

    A mirrored baseline file may be a symlink, so write a temporary file in the
    destination directory and atomically replace the link with a filtered copy.
    """
    if not os.path.isfile(path):
        return 0
    fd, tmp_path = tempfile.mkstemp(prefix=".exclude-", dir=os.path.dirname(path))
    removed = 0
    try:
        with os.fdopen(fd, "w") as out, open(path) as src:
            header = src.readline()
            if not header:
                return 0
            out.write(header)
            for line in src:
                sample_id = line.split("\t", 1)[0].strip()
                if sample_id in excluded:
                    removed += 1
                else:
                    out.write(line)
        if removed:
            os.replace(tmp_path, path)
        else:
            os.unlink(tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return removed


def write_lengths(path: str, graph_width: int, pc_width: int, kmer_width: int) -> None:
    with open(path, "w") as out:
        out.write("Graph\tPC\tKmer\n")
        out.write(f"{int(graph_width)}\t{int(pc_width)}\t{int(kmer_width)}\n")


def read_lengths(path: str) -> tuple[int, int, int]:
    with open(path) as fh:
        fh.readline()
        vals = fh.readline().strip().split("\t")
    return int(vals[0]), int(vals[1]), int(vals[2])


def feature_rows(path: str) -> int:
    with open(path) as fh:
        return max(0, sum(1 for _ in fh) - 1)


def write_all_features_fallback(raw_sentence: str, mapping_path: str, out_path: str) -> int:
    """Write every observed training token when chi-square selects none.

    `feature_selection_sp.sef` falls back to scanning all tokens internally, but
    leaves `feature_remain_graph.txt` with only a header.  build-test uses that
    file as its whitelist, so without this fallback every validation genome
    would silently become all zeroes.  This mirrors the train-side fallback and
    makes the situation explicit in the log.
    """
    token_to_name = {}
    with open(mapping_path) as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                token_to_name[str(parts[-1])] = parts[0]
    observed = set()
    with open(raw_sentence) as fh:
        fh.readline()
        for line in fh:
            field = line.rstrip("\n").split("\t")[-1]
            observed.update(t for t in field.split(",") if t and t != "0")
    ordered = sorted(observed, key=lambda x: int(x))
    with open(out_path, "w") as out:
        out.write("ID\tFeature_ID\tFeature\tAMR_Gene_Family\tP-value\tChi2-statistic\n")
        for i, token in enumerate(ordered, 1):
            out.write(f"{i}\t{token}\t{token_to_name.get(token, token)}\tNA\tNA\tNA\n")
    return len(ordered)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-dir", required=True,
                    help="Existing baseline fold directory, e.g. features-3fold/Mtb_ofloxacin_3fold/Fold1")
    ap.add_argument("--out-dir", required=True,
                    help="New, empty output directory. Existing non-empty directories are refused.")
    ap.add_argument("--reference", required=True, help="H37Rv reference FASTA (whole genome).")
    ap.add_argument("--reference-name", default="NC_000962.3",
                    help="Reference FASTA record name (default NC_000962.3).")
    ap.add_argument("--drug", required=True, choices=sorted(DRUG_REGIONS),
                    help="Drug name; must match the fold's labels.")
    ap.add_argument("--region", default=None,
                    help="Region name; default is selected from --drug (gyrBA for ofloxacin/ciprofloxacin).")
    ap.add_argument("--train-genomes", default=None,
                    help="Training genome directory (default: FOLD/input/genomes-train).")
    ap.add_argument("--val-genomes", default=None,
                    help="Validation genome directory (default: FOLD/input/genomes-val).")
    ap.add_argument("--train-label", default=None,
                    help="Training label file (default: FOLD/input/labels/train_label.tsv).")
    ap.add_argument("--val-label", default=None,
                    help="Validation label file (default: FOLD/input/labels/val_label.tsv).")
    ap.add_argument("--code-dir", default="/data2/hmy/StrainAMR-optimize/code",
                    help="StrainAMR code directory used for the existing chi2/SHAP utilities.")
    ap.add_argument("--minimap2", default=None, help="Path to minimap2 (default: search PATH).")
    ap.add_argument("--threads", type=int, default=4, help="Threads per minimap2 process.")
    ap.add_argument("--jobs", type=int, default=1, help="Genomes processed in parallel.")
    ap.add_argument("--min-mapq", type=int, default=20, help="Minimum PAF mapping quality.")
    ap.add_argument("--min-alen", type=int, default=200, help="Minimum PAF alignment length.")
    ap.add_argument("--exclude-sample", action="append", default=[],
                    help="Sample ID to remove from both splits and from mirrored PC/k-mer/SHAP rows. "
                         "May be supplied more than once.")
    ap.add_argument("--copy-baseline", action="store_true",
                    help="Copy baseline PC/k-mer/SHAP files instead of read-only symlinks.")
    ap.add_argument("--skip-shap", action="store_true",
                    help="Skip SHAP ranking. Only use this for a fast pilot, and then train with -a 0.")
    ap.add_argument("--no-selection", action="store_true",
                    help="Write raw tokens only; skip chi2 and SHAP. Useful for a quick syntax/QC run.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan only.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fold = os.path.abspath(args.fold_dir)
    out = os.path.abspath(args.out_dir)
    region = args.region or DRUG_REGIONS[args.drug]
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; available: {', '.join(REGIONS)}")
    start0, end0 = REGIONS[region]

    if not os.path.isdir(fold):
        raise FileNotFoundError(f"fold directory not found: {fold}")
    train_genomes = args.train_genomes or os.path.join(fold, "input", "genomes-train")
    val_genomes = args.val_genomes or os.path.join(fold, "input", "genomes-val")
    train_label = args.train_label or os.path.join(fold, "input", "labels", "train_label.tsv")
    val_label = args.val_label or os.path.join(fold, "input", "labels", "val_label.tsv")

    for path in (train_genomes, val_genomes, train_label, val_label):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    required_baseline = [
        os.path.join(fold, "db", "strains_train_pc_token_fs.txt"),
        os.path.join(fold, "db", "strains_train_kmer_token.txt"),
        os.path.join(fold, "db", "longest_len_fs.txt"),
        os.path.join(fold, "val_features", "strains_test_pc_token_fs.txt"),
        os.path.join(fold, "val_features", "strains_test_kmer_token.txt"),
    ]
    missing_baseline = [p for p in required_baseline if not os.path.exists(p)]
    if missing_baseline:
        raise FileNotFoundError(
            "baseline fold is incomplete; missing:\n  " + "\n  ".join(missing_baseline)
        )
    if os.path.exists(out) and any(os.scandir(out)):
        raise FileExistsError(
            f"output directory is not empty: {out}\n"
            "Choose a new --out-dir; Scheme A never overwrites an existing run."
        )

    ref_seqs = read_fasta(args.reference)
    ref_name, ref_seq = choose_reference(ref_seqs, args.reference_name)
    if end0 > len(ref_seq):
        raise ValueError(f"region {region}:{start0}-{end0} exceeds reference length {len(ref_seq)}")
    ref_region = ref_seq[start0:end0].upper()
    ref_region_id = f"{ref_name}:{start0}-{end0} (0-based half-open)"

    train_header, train_rows = read_labels(train_label)
    val_header, val_rows = read_labels(val_label)
    excluded = set(args.exclude_sample)
    if excluded:
        before_train, before_val = len(train_rows), len(val_rows)
        train_rows = [row for row in train_rows if row[0] not in excluded]
        val_rows = [row for row in val_rows if row[0] not in excluded]
        log(f"excluding {len(excluded)} sample ID(s): {', '.join(sorted(excluded))}; "
            f"train {before_train}->{len(train_rows)}, val {before_val}->{len(val_rows)}")
        if not train_rows or not val_rows:
            raise ValueError("exclusion removed all training or validation samples")
    train_files = index_genomes(train_genomes)
    val_files = index_genomes(val_genomes)
    for name, files in (("training", train_files), ("validation", val_files)):
        missing = [sid for sid, _ in (train_rows if name == "training" else val_rows) if sid not in files]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} labeled {name} sample(s) have no genome file: {', '.join(missing[:10])}"
            )

    plan = {
        "fold": fold,
        "out": out,
        "drug": args.drug,
        "region": region,
        "reference": os.path.abspath(args.reference),
        "reference_record": ref_name,
        "coordinates_0based_half_open": [start0, end0],
        "coordinates_1based_inclusive": [start0 + 1, end0],
        "coordinate_convention": "0-based half-open, matching the paper table",
        "region_length": len(ref_region),
        "region_sha256": sha256_text(ref_region),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "excluded_samples": sorted(excluded),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    os.makedirs(out, exist_ok=True)
    db = os.path.join(out, "db")
    val = os.path.join(out, "val_features")
    work = os.path.join(out, "work")
    os.makedirs(db, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    os.makedirs(work, exist_ok=True)

    mirror_top_level(os.path.join(fold, "db"), db, args.copy_baseline)
    mirror_top_level(os.path.join(fold, "val_features"), val, args.copy_baseline)
    mirror_shap(os.path.join(fold, "db", "shap"), os.path.join(db, "shap"), args.copy_baseline)
    mirror_shap(os.path.join(fold, "val_features", "shap"), os.path.join(val, "shap"),
                args.copy_baseline)

    if excluded:
        feature_files = [
            os.path.join(db, "strains_train_pc_token.txt"),
            os.path.join(db, "strains_train_pc_token_fs.txt"),
            os.path.join(db, "strains_train_kmer_token.txt"),
            os.path.join(val, "strains_test_pc_token.txt"),
            os.path.join(val, "strains_test_pc_token_fs.txt"),
            os.path.join(val, "strains_test_kmer_token.txt"),
            os.path.join(db, "shap", "train_samples.tsv"),
            os.path.join(val, "shap", "test_samples.tsv"),
        ]
        for feature_dir in (os.path.join(db, "shap"), os.path.join(val, "shap")):
            if os.path.isdir(feature_dir):
                feature_files.extend(
                    entry.path for entry in os.scandir(feature_dir)
                    if entry.is_file() and entry.name.startswith("strains_")
                    and entry.name.endswith(".txt")
                )
        removed_rows = sum(filter_feature_file(path, excluded) for path in feature_files)
        log(f"removed {removed_rows} mirrored feature row(s) for excluded samples")

    ref_fa = os.path.join(work, "h37rv_region.fa")
    with open(ref_fa, "w") as fh:
        fh.write(f">{region}_{ref_name}_{start0}_{end0}\n")
        for i in range(0, len(ref_region), 80):
            fh.write(ref_region[i:i + 80] + "\n")

    mapping_path = os.path.join(db, "node_token_match.txt")
    write_token_mapping(mapping_path, region, start0, ref_region)

    mm2 = minimap2_path(args.minimap2)

    def process_one(item):
        sid, genome_path = item
        events, coverage, n_align = sample_events(
            genome_path, ref_fa, ref_region, mm2, args.threads,
            args.min_mapq, args.min_alen,
        )
        tokens = [token_id_for_event(pos, kind, allele, length)
                  for pos, kind, allele, length in events]
        counts = Counter(kind for _pos, kind, _allele, _length in events)
        return sid, tokens, coverage, n_align, counts

    def run_group(rows, files, group_name):
        results = {}
        items = [(sid, files[sid]) for sid, _ in rows]
        if args.jobs <= 1:
            iterator = map(process_one, items)
            for idx, result in enumerate(iterator, 1):
                results[result[0]] = result
                if idx % 25 == 0 or idx == len(items):
                    log(f"{group_name}: {idx}/{len(items)}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(process_one, item) for item in items]
                for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                    result = fut.result()
                    results[result[0]] = result
                    if idx % 25 == 0 or idx == len(items):
                        log(f"{group_name}: {idx}/{len(items)}")
        return results

    log(f"tokenizing {len(train_rows)} training genomes")
    train_results = run_group(train_rows, train_files, "train")
    log(f"tokenizing {len(val_rows)} validation genomes")
    val_results = run_group(val_rows, val_files, "validation")

    no_alignment = [sid for sid, res in list(train_results.items()) + list(val_results.items())
                    if res[3] == 0]
    if no_alignment:
        raise RuntimeError(
            f"{len(no_alignment)} genome(s) had no alignment in {ref_region_id}: "
            f"{', '.join(no_alignment[:10])}"
        )

    train_raw = os.path.join(db, "strains_train_sentence.txt")
    val_raw = os.path.join(val, "strains_test_sentence.txt")
    train_width = write_sentence(
        train_raw, train_header, train_rows,
        {sid: res[1] for sid, res in train_results.items()},
    )
    val_width = write_sentence(
        val_raw, val_header, val_rows,
        {sid: res[1] for sid, res in val_results.items()},
    )

    with open(os.path.join(out, "mtb_feature_qc.tsv"), "w") as qc:
        qc.write("Split\tSample\tAlignments\tCoverage\tSNP\tInsertion\tDeletion\tTokens\n")
        for split, rows, resmap in (("train", train_rows, train_results),
                                    ("val", val_rows, val_results)):
            for sid, _ in rows:
                _sid, tokens, coverage, n_align, counts = resmap[sid]
                qc.write(f"{split}\t{sid}\t{n_align}\t{coverage:.4f}\t"
                         f"{counts.get('snp', 0)}\t{counts.get('ins', 0)}\t"
                         f"{counts.get('del', 0)}\t{len(tokens)}\n")

    # Reuse baseline PC/k-mer widths; build-train later takes max(db, val).
    db_g, db_p, db_k = read_lengths(os.path.join(fold, "db", "longest_len_fs.txt"))
    if os.path.exists(os.path.join(fold, "val_features", "longest_len_fs.txt")):
        va_g, va_p, va_k = read_lengths(os.path.join(fold, "val_features", "longest_len_fs.txt"))
    else:
        va_g, va_p, va_k = db_g, db_p, db_k
    pc_width = max(db_p, va_p)
    kmer_width = max(db_k, va_k)
    write_lengths(os.path.join(db, "longest_len_fs.txt"),
                  max(train_width, val_width), pc_width, kmer_width)
    write_lengths(os.path.join(val, "longest_len_fs.txt"),
                  max(train_width, val_width), pc_width, kmer_width)

    params_path = os.path.join(db, "build_params.json")
    if os.path.exists(params_path):
        with open(params_path) as fh:
            params = json.load(fh)
    else:
        params = {}
    params["snv_mode"] = "mtb_region_tokens_replace"
    params["mtb_feature"] = plan
    with open(params_path, "w") as fh:
        json.dump(params, fh, indent=1)

    with open(os.path.join(out, "mtb_feature_manifest.json"), "w") as fh:
        json.dump(plan, fh, indent=2)

    min_cov = min(res[2] for res in list(train_results.values()) + list(val_results.values()))
    log(f"raw tokens written; min region coverage={min_cov:.3f}; "
        f"train max width={train_width}; val max width={val_width}")
    if min_cov < 0.5:
        log("WARNING: at least one genome covers less than 50% of the selected region; "
            "inspect mtb_feature_qc.tsv before interpreting the model.")

    if args.no_selection:
        log("--no-selection set: raw token files only")
    else:
        if not os.path.isdir(args.code_dir):
            raise FileNotFoundError(args.code_dir)
        sys.path.insert(0, args.code_dir)
        from feature_selection_sp import sef
        from feature_selection_sp_test import sef_test

        log("running chi-square selection")
        sef(
            train_raw,
            os.path.join(db, "feature_remain_graph.txt"),
            os.path.join(db, "strains_train_sentence_fs.txt"),
            mapping_files=[mapping_path],
            rgi_dir=None,
        )
        if feature_rows(os.path.join(db, "feature_remain_graph.txt")) == 0:
            n = write_all_features_fallback(
                train_raw, mapping_path, os.path.join(db, "feature_remain_graph.txt")
            )
            log(f"WARNING: chi-square selected zero features; kept all {n} observed "
                "training tokens so validation is not silently zeroed. Treat this run as exploratory.")
        sef_test(
            val_raw,
            os.path.join(db, "feature_remain_graph.txt"),
            os.path.join(val, "strains_test_sentence_fs.txt"),
        )

        if not args.skip_shap:
            os.environ.setdefault("MPLBACKEND", "Agg")
            from library.shap_feature_select_withcls import shap_select
            log("running SHAP ranking for the Mtb SNV channel")
            shap_select(
                os.path.join(db, "strains_train_sentence_fs.txt"),
                os.path.join(db, "shap", "strains_train_sentence_fs_shap_filter.txt"),
                [mapping_path],
                rgi_dir=None,
            )
            write_lengths(os.path.join(db, "shap", "longest_len_fs_shap.txt"),
                          max(train_width, val_width), pc_width, kmer_width)

    log(f"done -> {out}")
    log("Next: train with the original StrainAMR command, using -i OUT/db "
        "--val-dir OUT/val_features and a new model output directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
