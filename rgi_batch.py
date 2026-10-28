"""Run CARD RGI over many genomes without paying for it once per genome.

StrainAMR ran `rgi main` separately for every genome, so the CARD database and the DIAMOND/BLAST
index were loaded from scratch 1393 times for a 1393-genome dataset, and every gene that several
genomes share was searched again for each of them.

Two changes, neither of which alters what RGI is asked about any individual sequence:

* **Deduplication.**  Genomes of one species share most of their genes verbatim.  Measured on
  S. aureus assemblies the duplication factor grows with the dataset -- 1.2x at 5 genomes, 2.8x at
  20, 3.8x at 61, and it keeps climbing as the pangenome saturates.  Each distinct gene sequence is
  searched once and its hits are written back to every genome that carries it.  Two identical
  sequences cannot get different RGI results, so this is exact.
* **Batching.**  The distinct sequences are searched in a few large runs instead of one run per
  genome, which amortises RGI's start-up over the whole dataset.

The per-genome output files are byte-for-byte what the old code produced, except that hits are
listed in the order the batch produced them.  `--rgi-batch 1` restores the original one-run-per-genome
behaviour for comparison.

A caveat worth stating: batching assumes RGI's verdict on a sequence does not depend on which other
sequences share its input file.  That is how BLAST/DIAMOND e-values work (they depend on query length
and database size, not on the rest of the query set), but it has not been verified against a real RGI
install here, so compare one small dataset both ways before trusting a large run -- see check_batches().
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

ORF_PREFIX = "u"                      # headers written into the batch FASTAs: >u<uid>
# RGI names an ORF after its FASTA header and appends prodigal's own description, so a field looks
# like "u1234_1 # 1 # 1392 # 1 # ID=2582_1;partial=10;..." -- match the placeholder and keep the rest.
_UID = re.compile(r"^" + ORF_PREFIX + r"(\d+)(_\d+)?(\s.*)?$", re.S)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [RGI] {msg}", flush=True)


def read_fasta(path):
    """(header_id, sequence) for each record; the id is the header up to the first whitespace."""
    name, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def collect_unique(gene_files):
    """Distinct gene sequences across all genomes, and where each one occurs.

    Returns (uniq, occurrences, n_total) where uniq[uid] = sequence and
    occurrences[uid] = [(sample, gene_id), ...].
    """
    by_hash, uniq, occ = {}, [], []
    n_total = 0
    for sample, path in gene_files.items():
        for gid, seq in read_fasta(path):
            n_total += 1
            h = hashlib.blake2b(seq.encode(), digest_size=16).digest()
            uid = by_hash.get(h)
            if uid is None:
                uid = len(uniq)
                by_hash[h] = uid
                uniq.append(seq)
                occ.append([])
            occ[uid].append((sample, gid))
    return uniq, occ, n_total


def _write_batches(uniq, tmpdir, per_batch):
    paths = []
    for start in range(0, len(uniq), per_batch):
        path = os.path.join(tmpdir, f"batch_{start // per_batch}.fa")
        with open(path, "w") as o:
            for uid in range(start, min(start + per_batch, len(uniq))):
                o.write(f">{ORF_PREFIX}{uid}\n{uniq[uid]}\n")
        paths.append(path)
    return paths


def _rgi(input_fa, output_prefix, threads, extra=""):
    cmd = (f"rgi main --input_sequence {input_fa} --output_file {output_prefix} --local --clean "
           f"-n {threads} {extra}").strip()
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"rgi main failed (exit code {rc}) on {input_fa}")
    if not os.path.exists(output_prefix + ".txt"):
        raise RuntimeError(f"rgi main produced no {output_prefix}.txt")
    return output_prefix + ".txt"


def _split_row(fields, occ, id_columns=(0, 1)):
    """One output row per genome carrying this sequence, with the batch id swapped back.

    RGI names an ORF after the FASTA header it came from, so the columns holding that name carry the
    batch's 'u<uid>' placeholder plus RGI's own '_<n>' ORF suffix. Both are rewritten to the gene id
    the genome actually uses, which is what extract_seq_for_graph.extract() parses.
    """
    # Match the field exactly as RGI wrote it, never a stripped copy: RGI pads some of these columns
    # with a trailing space, and rebuilding from the stripped form would silently drop it, leaving
    # output that differs from the per-genome run by whitespace alone.
    uid = None
    for col in id_columns:
        if col < len(fields):
            m = _UID.match(fields[col])
            if m:
                uid = int(m.group(1))
                break
    if uid is None or uid >= len(occ):
        return []
    rows = []
    for sample, gid in occ[uid]:
        row = list(fields)
        for col in id_columns:
            if col >= len(row):
                continue
            m = _UID.match(row[col])
            if m:
                # gene id + RGI's ORF suffix + the rest of the field (description, trailing space)
                row[col] = gid + (m.group(2) or "") + (m.group(3) or "")
        rows.append((sample, row))
    return rows


def auto_batch_size(n_unique, target_batches=10, minimum=5000):
    """Batch size giving roughly ``target_batches`` runs, so progress is visible.

    One huge batch is the fewest RGI start-ups, but it also means no output at all until the whole
    dataset is done -- which looks like a hang on a run that takes hours. RGI's start-up is seconds,
    so a handful of batches costs almost nothing and reports progress along the way.
    """
    if n_unique <= minimum:
        return max(1, n_unique)
    return max(minimum, -(-n_unique // target_batches))


def run_rgi(gene_files, out_dir, threads=1, per_batch=0, tmpdir=None, extra=""):
    """Search every distinct gene sequence once and write per-genome RGI tables into out_dir.

    ``per_batch=0`` sizes the batches automatically so progress is reported about ten times.
    """
    os.makedirs(out_dir, exist_ok=True)
    pending = {s: p for s, p in gene_files.items()
               if not (os.path.exists(os.path.join(out_dir, s + ".txt"))
                       and os.path.getsize(os.path.join(out_dir, s + ".txt")) > 0)}
    if not pending:
        _log(f"all {len(gene_files)} genomes already have RGI output; nothing to do")
        return
    if len(pending) < len(gene_files):
        _log(f"{len(gene_files) - len(pending)} genomes already done, {len(pending)} to go")

    t0 = time.time()
    uniq, occ, n_total = collect_unique(pending)
    _log(f"{n_total} genes in {len(pending)} genomes -> {len(uniq)} distinct sequences to search "
         f"({n_total / max(1, len(uniq)):.1f}x duplication)")

    tmpdir = tmpdir or os.path.join(out_dir, "_rgi_batches")
    os.makedirs(tmpdir, exist_ok=True)
    per_batch = per_batch or auto_batch_size(len(uniq))
    batches = _write_batches(uniq, tmpdir, per_batch)
    _log(f"searching in {len(batches)} batch(es) of up to {per_batch} sequences, {threads} threads"
         + (f", extra RGI options: {extra}" if extra else ""))
    _log("this is the CARD search itself and is the slowest step; RGI is quiet while a batch runs")

    header = None
    rows_by_sample = {s: [] for s in pending}
    for i, fa in enumerate(batches, 1):
        out = _rgi(fa, fa[:-3], threads, extra)
        with open(out) as f:
            head = f.readline().rstrip("\n")
            header = header or head
            n = 0
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                for sample, row in _split_row(line.split("\t"), occ):
                    rows_by_sample.setdefault(sample, []).append("\t".join(row))
                    n += 1
        el = time.time() - t0
        left = el / i * (len(batches) - i)
        _log(f"batch {i}/{len(batches)}: {n} hits mapped back; {el / 60:.1f} min elapsed"
             + (f", ~{left / 60:.0f} min left" if i < len(batches) else ""))

    if header is None:
        raise RuntimeError("no RGI output produced")
    for sample in pending:
        with open(os.path.join(out_dir, sample + ".txt"), "w") as o:
            o.write(header + "\n")
            for row in rows_by_sample.get(sample, []):
                o.write(row + "\n")
    shutil.rmtree(tmpdir, ignore_errors=True)
    hits = sum(len(v) for v in rows_by_sample.values())
    empty = sum(1 for s in pending if not rows_by_sample.get(s))
    _log(f"done in {time.time() - t0:.0f} s: {hits} hits over {len(pending)} genomes"
         + (f" ({empty} with none)" if empty else ""))


def check_batches(gene_files, out_dir, threads=1, n=3):
    """Run n genomes the old way and the batched way, and report whether the tables agree.

    Rows are matched by ORF name and ARO, then compared column by column, so a disagreement names
    the column and shows both values instead of printing two nearly identical truncated lines.
    """
    import tempfile
    samples = list(gene_files)[:n]
    sel = {s: gene_files[s] for s in samples}
    with tempfile.TemporaryDirectory() as tmp:
        one = os.path.join(tmp, "per_genome")
        many = os.path.join(tmp, "batched")
        os.makedirs(one)
        for s, p in sel.items():
            _rgi(p, os.path.join(one, s), threads)
        run_rgi(sel, many, threads)
        ok = True
        for s in samples:
            ha, ra = _rows(os.path.join(one, s + ".txt"))
            hb, rb = _rows(os.path.join(many, s + ".txt"))
            header = ha or hb
            same = ra == rb
            ok &= same
            print(f"  {s}: {'identical' if same else 'DIFFERS'} ({len(ra)} vs {len(rb)} hits)")
            if same:
                continue
            for key in sorted(set(ra) - set(rb))[:3]:
                print(f"    only in the per-genome run: {key}")
            for key in sorted(set(rb) - set(ra))[:3]:
                print(f"    only in the batched run   : {key}")
            shown = 0
            for key in sorted(set(ra) & set(rb)):
                fa, fb = ra[key], rb[key]
                diffs = [i for i in range(max(len(fa), len(fb)))
                         if (fa[i] if i < len(fa) else None) != (fb[i] if i < len(fb) else None)]
                if not diffs or shown >= 3:
                    continue
                shown += 1
                print(f"    {key}: {len(diffs)} column(s) differ")
                for i in diffs[:4]:
                    name = header[i] if i < len(header) else f"column {i}"
                    va = fa[i] if i < len(fa) else "<missing>"
                    vb = fb[i] if i < len(fb) else "<missing>"
                    print(f"      {name}:")
                    print(f"        per-genome: {va[:70]}")
                    print(f"        batched   : {vb[:70]}")
    print("batched RGI reproduces per-genome RGI" if ok
          else "batched RGI DIFFERS - rerun that dataset with --rgi-batch 1")
    return ok


def _rows(path, normalise=True):
    """(header, {key: fields}) for an RGI table, keyed by ORF name and ARO."""
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        out = {}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if normalise:
                line = _PRODIGAL_ID.sub(r"ID=<n>_\1", line)
            fields = line.split("\t")
            orf = fields[0].split(" # ")[0] if fields else ""
            aro = fields[10] if len(fields) > 10 else ""
            out[f"{orf} (ARO {aro})"] = fields
    return header, out


_PRODIGAL_ID = re.compile(r"ID=\d+_(\d+)")


def _table(path, normalise=True):
    """Rows of an RGI table, optionally with the one field batching cannot reproduce removed.

    RGI copies prodigal's description into ORF_ID, and that description contains `ID=<n>_<m>` where
    <n> counts ORFs from the start of the *input file*. Batched and per-genome runs necessarily number
    them differently. Nothing downstream reads it -- extract_seq_for_graph() parses the ORF name and
    the ARO column -- so the comparison ignores it.
    """
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        next(f, None)
        rows = [line.rstrip("\n") for line in f if line.strip()]
    if normalise:
        rows = [_PRODIGAL_ID.sub(r"ID=<n>_\1", r) for r in rows]
    return set(rows)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog=(sys.argv[0] if sys.argv[0].startswith("strainamr") else "rgi_batch.py"),
        description="Batched, deduplicated CARD RGI over a directory of prodigal gene FASTAs "
                    "(<sample>.fa, as build-train writes into <outdir>/Genes)")
    ap.add_argument("genes_dir")
    ap.add_argument("-o", "--out", required=True, help="directory for the per-genome RGI tables")
    ap.add_argument("-t", "--threads", type=int, default=1)
    ap.add_argument("--per-batch", type=int, default=0,
                    help="sequences per RGI run (0 = automatic, about ten batches)")
    ap.add_argument("--rgi-extra", default="",
                    help="extra options passed to `rgi main`, e.g. '-a DIAMOND' (much faster than "
                         "RGI's default BLAST, but the hits are not identical)")
    ap.add_argument("--check", type=int, default=0,
                    help="instead of running, compare N genomes batched vs one-at-a-time")
    a = ap.parse_args(argv)
    files = {f[:-3]: os.path.join(a.genes_dir, f) for f in sorted(os.listdir(a.genes_dir))
             if f.endswith(".fa")}
    if a.check:
        return 0 if check_batches(files, a.out, a.threads, a.check) else 1
    run_rgi(files, a.out, a.threads, a.per_batch, extra=a.rgi_extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
