"""Drop-in replacement for the PhenotypeSeeker step of StrainAMR v1.

v1 uses PhenotypeSeeker only as a k-mer selector: it reads <drug>_MLdf.csv, takes the column names
(the selected k-mers) and the 0/non-0 presence pattern, and throws the rest away.  This module does
the same selection without writing a k-mer list per genome to disk.

The statistic is the one PhenotypeSeeker uses, reproduced deliberately, quirks included:
  * weighted 2x2 table (phenotype x k-mer presence), weights default to 1;
  * scipy.stats.chisquare over the four cells against the expected distribution, with ddof=1, i.e.
    df=2 rather than the df=1 a 2x2 table would normally use;
  * a k-mer is kept when it is present in >= min_samples and <= max_samples genomes, absent in >= 2,
    and its p-value passes the (Bonferroni-corrected) cut-off;
  * the lowest-p-value n_kmers are reported.

Memory is bounded by the k-mer space, not by the data: for k=13 the 4^13 canonical space is indexed
directly with three arrays (1.3 GB total), so no per-genome intermediate files are produced.  Each
scanning process additionally holds a 4^k byte bitmap (64 MB at k=13) and the k-mer list of the
genome it is on, so `threads` costs roughly 0.1 GB each on top and is capped at 16.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

CODE = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate(b"ACGT"):
    CODE[b] = i
    CODE[b + 32] = i          # lowercase


def read_fasta(path: str):
    """Sequences of a (optionally gzipped) FASTA file, newlines and spaces removed."""
    opener = open
    with open(path, "rb") as fh:
        if fh.read(2) == b"\x1f\x8b":
            import gzip
            opener = gzip.open
    with opener(path, "rb") as fh:
        raw = fh.read()
    return [r[r.find(b"\n") + 1:].translate(None, b"\n\r \t")
            for r in raw.split(b">")[1:]]


def encode_contig(piece: np.ndarray, k: int) -> np.ndarray:
    """Canonical codes of every k-mer of one gap-free stretch of ACGT (given as 0..3 codes).

    The forward and reverse-complement codes are built by shifting in one base at a time, which
    touches each base twice, rather than by multiplying a (n, k) sliding window by a vector of
    powers of four; same numbers, a third of the work and no int64 intermediates.
    """
    n = len(piece) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.int32 if k <= 15 else np.int64)
    dt = np.int32 if k <= 15 else np.int64          # 2 bits per base, plus a sign bit to spare
    p = piece.astype(dt)
    q = 3 - p                                        # complement; the reverse comes from the shifts
    fwd = np.zeros(n, dtype=dt)
    rev = np.zeros(n, dtype=dt)
    for j in range(k):
        fwd <<= 2
        fwd |= p[j:j + n]
        rev |= q[j:j + n] << (2 * j)
    return np.minimum(fwd, rev)


# One scratch bitmap per process, reused across genomes: allocating and first-touching 4^13 bytes
# for every genome would cost more than the counting does.
_SEEN = {}


def _bitmap(size: int) -> np.ndarray:
    b = _SEEN.get(size)
    if b is None:
        b = np.zeros(size, dtype=bool)
        _SEEN[size] = b
    return b


def genome_kmers(path: str, k: int) -> np.ndarray:
    """Sorted unique canonical k-mer codes of one genome (presence only, counts are not needed).

    Deduplication is done by marking a bitmap over the whole canonical k-mer space and reading the
    set positions back, which is sorted and unique by construction.  Sorting the ~10 million codes
    of a bacterial genome with np.unique instead costs about thirty times as much, and that sort was
    over 90% of the k-mer step's run time.  Above k=15 the bitmap no longer fits, so np.unique is
    kept for that case.
    """
    size = 4 ** k
    pieces = []
    for seq in read_fasta(path):
        a = CODE[np.frombuffer(seq, dtype=np.uint8)]
        # split on anything that is not ACGT (N, IUPAC codes): k-mers may not span them
        bad = np.flatnonzero(a == 255)
        for piece in np.split(a, bad) if len(bad) else [a]:
            if len(piece) and piece[0] == 255:
                piece = piece[1:]
            if len(piece) >= k:
                pieces.append(piece)
    if not pieces:
        return np.empty(0, dtype=np.int32 if k <= 15 else np.int64)
    if k > 15:
        return np.unique(np.concatenate([encode_contig(p, k) for p in pieces]))
    seen = _bitmap(size)
    seen[:] = False
    for piece in pieces:
        seen[encode_contig(piece, k)] = True
    return np.flatnonzero(seen).astype(np.int32)


def _init_worker(k, codes=None):
    global _WK, _WCODES
    _WK = k
    _WCODES = codes


def _scan(path):
    return genome_kmers(path, _WK)


def _scan_hits(path):
    """Presence of the selected k-mers only: one bit per selected k-mer, so almost nothing to send
    back to the parent."""
    return _hits(genome_kmers(path, _WK), _WCODES)


def _hits(km, sorted_codes):
    pos = np.searchsorted(sorted_codes, km)
    inside = pos < len(sorted_codes)
    pos, km_in = pos[inside], km[inside]
    return pos[sorted_codes[pos] == km_in]


def _scan_all(samples, k, threads, what, codes=None):
    """Run `genome_kmers` over every genome, in worker processes when asked, yielding results in
    input order.  Each worker holds its own scratch bitmap (4^k bytes, 64 MB at k=13), so the
    process count is capped rather than following the thread budget blindly."""
    fn = _scan if codes is None else _scan_hits
    n = len(samples)
    paths = [p for _n, p, _y, _w in samples]
    t0 = time.time()
    procs = max(1, min(int(threads or 1), 16, n))
    sys.stderr.write(f"  scanning {n} genomes for {what} with {procs} process(es)\n")
    if procs == 1:
        _init_worker(k, codes)
        for i, p in enumerate(paths):
            yield i, fn(p)
            sys.stderr.write(f"\r  {i + 1}/{n} genomes ({time.time() - t0:.0f} s)")
            sys.stderr.flush()
        sys.stderr.write("\n")
        return
    import multiprocessing as mp
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    pool = ctx.Pool(procs, initializer=_init_worker, initargs=(k, codes))
    # One block at a time, rather than pool.imap over the whole list.  The counting caller is slower
    # than the workers, and imap does not throttle: finished k-mer lists (20 MB each) would pile up
    # in the result queue faster than they are consumed, which for a large dataset is many GB of
    # memory held for nothing.  A block bounds that at block_size lists.
    block = 2 * procs
    try:
        for s in range(0, n, block):
            for j, res in enumerate(pool.map(fn, paths[s:s + block])):
                yield s + j, res
            sys.stderr.write(f"\r  {min(s + block, n)}/{n} genomes ({time.time() - t0:.0f} s)")
            sys.stderr.flush()
    finally:
        pool.terminate()
        pool.join()
    sys.stderr.write("\n")


def count(samples, k, threads=1):
    """Weighted presence totals per canonical k-mer, over the whole k-mer space."""
    size = 4 ** k
    w_pheno_w_kmer = np.zeros(size, dtype=np.float64)     # sum of weights of R genomes carrying it
    wo_pheno_w_kmer = np.zeros(size, dtype=np.float64)    # same for S genomes
    n_w_kmer = np.zeros(size, dtype=np.int32)             # number of genomes carrying it
    # genome_kmers returns each k-mer once, so no index repeats within a genome and the buffered
    # `arr[idx] += v` is equivalent to np.add.at -- same numbers, about a third less time.
    for i, km in _scan_all(samples, k, threads, "k-mer counting"):
        _name, _path, y, w = samples[i]
        n_w_kmer[km] += 1
        if y == 1:
            w_pheno_w_kmer[km] += w
        else:
            wo_pheno_w_kmer[km] += w
    return w_pheno_w_kmer, wo_pheno_w_kmer, n_w_kmer


def select(samples, k=13, min_samples=0, max_samples=0, pvalue=0.05, n_kmers=1000, omit_B=False,
           bonferroni_tested=False, threads=1):
    """Return (kmer_string, chi2, pvalue, n_carriers) for the selected k-mers, best p-value first."""
    n = len(samples)
    min_samples = 2 if min_samples == 0 else min_samples
    max_samples = n - 2 if max_samples == 0 else max_samples
    wR, wS, cnt = count(samples, k, threads)

    tot_R = sum(w for _, _, y, w in samples if y == 1)
    tot_S = sum(w for _, _, y, w in samples if y == 0)

    # PhenotypeSeeker's Bonferroni denominator is the size of its union k-mer list -- every k-mer seen
    # in at least one genome, including the ones the frequency filter then rejects -- not the number of
    # tests actually conducted.  Reproduced here so the selection matches; --bonferroni-tested divides
    # by the tests really performed instead, which is the statistically defensible choice.
    n_union = int((cnt >= 1).sum())
    keep = (cnt >= min_samples) & (cnt <= max_samples) & ((n - cnt) >= 2)
    idx = np.flatnonzero(keep)
    sys.stderr.write(f"  {len(idx)} k-mers pass the frequency filter, {n_union} in the union "
                     f"(of {4 ** k} possible)\n")
    if not len(idx):
        return []

    a = wR[idx]                  # phenotype 1, k-mer present
    b = tot_R - a                # phenotype 1, k-mer absent
    c = wS[idx]                  # phenotype 0, k-mer present
    d = tot_S - c
    w_pheno, wo_pheno = a + b, c + d
    w_kmer, wo_kmer = a + c, b + d
    total = w_pheno + wo_pheno
    exp = np.stack([w_pheno * w_kmer, w_pheno * wo_kmer, wo_pheno * w_kmer, wo_pheno * wo_kmer]) / total
    obs = np.stack([a, b, c, d])
    # PhenotypeSeeker calls scipy.stats.chisquare(obs, exp, 1): ddof=1 over four cells, so df=2
    chi2, p = stats.chisquare(obs, exp, 1, axis=0)

    cutoff = pvalue if omit_B else pvalue / (len(idx) if bonferroni_tested else n_union)
    ok = np.flatnonzero(p < cutoff)
    sys.stderr.write(f"  {len(ok)} k-mers below the p-value cut-off {cutoff:.3e}\n")
    order = ok[np.argsort(p[ok], kind="stable")]
    if n_kmers:
        order = order[:n_kmers]
    return [(decode(int(idx[j]), k), float(chi2[j]), float(p[j]), int(cnt[idx[j]])) for j in order]


def decode(code: int, k: int) -> str:
    s = []
    for _ in range(k):
        s.append("ACGT"[code & 3])
        code >>= 2
    return "".join(reversed(s))


def presence_matrix(samples, kmers, k, threads=1):
    """0/1 presence of the selected k-mers in each genome -- what v1 reads out of MLdf.csv."""
    codes = np.array([encode(s) for s in kmers], dtype=np.int64)
    order = np.argsort(codes)
    sorted_codes = codes[order]
    X = np.zeros((len(samples), len(kmers)), dtype=np.uint8)
    for i, hit in _scan_all(samples, k, threads, "presence of the selected k-mers", sorted_codes):
        X[i, order[hit]] = 1
    return X


def encode(s: str) -> int:
    v = 0
    for ch in s:
        v = (v << 2) | "ACGT".index(ch)
    return v


def gsc_weights(samples, threads=1, workdir="."):
    """Gerstein-Sonnhammer-Chothia weights from mash distances, as PhenotypeSeeker's -w computes them.

    Genomes in a large clone each get a small weight, so one over-sampled lineage cannot decide which
    k-mers look associated with the phenotype.  Requires `mash` on PATH; without it, returns None and
    the caller falls back to unweighted selection.

    NOTE: PhenotypeSeeker 1.2.3's own -w produces the right multiset of weights but attaches them to
    the wrong genomes.  It fills the distance matrix in the order `mash paste reference.msh
    K-mer_lists/*.msh` emits (shell glob order, i.e. alphabetical) but labels the rows with the sample
    order of the phenotype file (modeling.py:405-429).  Measured on 50 genomes: its weights correlate
    -0.008 with correctly assigned ones, and 0.999999 once re-attached in mash's order.  This function
    keeps the pairing, so its output differs from PhenotypeSeeker -w on purpose.
    """
    import shutil as _sh
    import subprocess
    import tempfile
    if _sh.which("mash") is None:
        sys.stderr.write("  mash not found on PATH: k-mer selection will be unweighted\n")
        return None
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        sketches = []
        for name, path, _y, _w in samples:
            out = os.path.join(tmp, name)
            subprocess.run(["mash", "sketch", "-r", path, "-o", out], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sketches.append(out + ".msh")
        ref = os.path.join(tmp, "reference.msh")
        subprocess.run(["mash", "paste", ref] + sketches, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dist = subprocess.run(["mash", "dist", ref, ref], check=True, capture_output=True, text=True).stdout
    path_to_name = {p: n for n, p, _y, _w in samples}
    names = [n for n, _p, _y, _w in samples]
    pos = {n: i for i, n in enumerate(names)}
    D = np.zeros((len(names), len(names)))
    for line in dist.strip().split("\n"):
        f = line.split("\t")
        a, b = path_to_name.get(f[0]), path_to_name.get(f[1])
        if a is None or b is None:
            continue
        D[pos[a], pos[b]] = float(f[2])
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    try:
        from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
    except ImportError:
        sys.stderr.write("  biopython not installed: k-mer selection will be unweighted\n")
        return None
    lower = [[float(D[i][j]) for j in range(i + 1)] for i in range(len(names))]
    tree = DistanceTreeConstructor().nj(_DistanceMatrix(list(names), lower))
    w = _gsc_from_tree(tree)
    m = np.mean([w[n] for n in names]) or 1.0
    return {n: w[n] / m for n in names}          # normalised to mean 1, as PhenotypeSeeker does


def _gsc_from_tree(tree):
    """Gerstein-Sonnhammer-Chothia leaf weights of a Bio.Phylo tree."""
    def clip(c):
        c.branch_length = min(max(c.branch_length or 0.0, 1e-9), 1e9)
        for ch in c.clades:
            clip(ch)

    def branch_sum(c):
        c._bsum = sum(branch_sum(ch) + ch.branch_length for ch in c.clades)
        return c._bsum

    def node_weight(c, parent=None):
        if parent is None:
            c._w = 1.0
        else:
            c._w = parent._w * (c.branch_length + c._bsum) / parent._bsum if parent._bsum else parent._w
        for ch in c.clades:
            node_weight(ch, c)

    root = tree.root
    clip(root)
    root.branch_length = root.branch_length or 0.0
    branch_sum(root)
    node_weight(root)
    return {leaf.name: leaf._w for leaf in tree.get_terminals()}


def write_mldf(samples, kmers, k, path, phenotype_col="phenotype", threads=1):
    """Write a PhenotypeSeeker-style <drug>_MLdf.csv so the rest of v1 needs no changes.

    Columns: an unnamed sample column, one column per selected k-mer, then 'weights' and 'phenotype',
    which is exactly what generate_token_from_ps.generate_tps() expects.
    """
    X = presence_matrix(samples, kmers, k, threads)
    with open(path, "w") as o:
        o.write("," + ",".join(kmers) + ",weights,phenotype\n")
        for i, (name, _p, y, w) in enumerate(samples):
            o.write(name + "," + ",".join("1" if v else "0" for v in X[i]) + f",{w},{y}\n")
    return X


def write_token_file(samples, kmers, k, token_ids, labels, path, threads=1):
    """Token file for new genomes: the selected k-mers looked up directly, no k-mer lists on disk.

    Replaces `phenotypeseeker prediction` plus generate_token_from_ps_predict.generate_tpsp().
    """
    X = presence_matrix(samples, kmers, k, threads)
    with open(path, "w") as o:
        o.write("Strain_ID\tLabel\tTokens_Num\tTokens\n")
        for i, (name, _p, _y, _w) in enumerate(samples):
            toks = [token_ids[j] if X[i, j] else "0" for j in range(len(kmers))]
            o.write(f"{name}\t{labels.get(name, '')}\t{len(toks)}\t{','.join(toks)}\n")


def read_pheno(path, weights=None):
    samples = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[2] == "":
                continue
            w = 1.0 if weights is None else weights.get(p[0], 1.0)
            samples.append((p[0], p[1], int(float(p[2])), w))
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pheno")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-l", "--kmer_length", type=int, default=13)
    ap.add_argument("--weights", default=None, help="TSV of sample<TAB>weight (default: all 1)")
    ap.add_argument("--gsc-weights", action="store_true",
                    help="compute GSC weights from mash distances (PhenotypeSeeker's -w)")
    ap.add_argument("--pvalue", type=float, default=0.05)
    ap.add_argument("--n_kmers", type=int, default=1000)
    ap.add_argument("--min", dest="min_samples", type=int, default=0)
    ap.add_argument("--max", dest="max_samples", type=int, default=0)
    ap.add_argument("--omit_B_correction", action="store_true")
    ap.add_argument("-t", "--threads", type=int, default=1,
                    help="worker processes for the genome scan (capped at 16; each holds a 4^k "
                         "byte bitmap, 64 MB at k=13)")
    ap.add_argument("--bonferroni-tested", action="store_true",
                    help="divide the p-value cut-off by the number of tests actually conducted rather "
                         "than by the size of the union k-mer list (PhenotypeSeeker does the latter)")
    a = ap.parse_args()
    weights = None
    if a.weights:
        weights = {}
        for line in open(a.weights):
            p = line.split()
            weights[p[0]] = float(p[1])
    samples = read_pheno(a.pheno, weights)
    if a.gsc_weights and weights is None:
        w = gsc_weights(samples)
        if w:
            samples = [(n, p, y, w.get(n, 1.0)) for n, p, y, _ in samples]
            sys.stderr.write(f"  GSC weights: {min(w.values()):.3f} - {max(w.values()):.3f} (mean 1)\n")
    sys.stderr.write(f"{len(samples)} samples, k={a.kmer_length}\n")
    t0 = time.time()
    sel = select(samples, a.kmer_length, a.min_samples, a.max_samples, a.pvalue, a.n_kmers,
                 a.omit_B_correction, a.bonferroni_tested, a.threads)
    with open(a.out, "w") as o:
        o.write("kmer\tchi2\tpvalue\tn_carriers\n")
        for km, c, p, ncar in sel:
            o.write(f"{km}\t{c:.2f}\t{p:.2E}\t{ncar}\n")
    sys.stderr.write(f"{len(sel)} k-mers written to {a.out} in {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
