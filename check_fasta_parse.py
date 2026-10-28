#!/usr/bin/env python3
"""Does the new FASTA reader in kmer_select.py see the same sequence as the old one?

    python check_fasta_parse.py <genome_dir> [n_genomes]

Reports, per genome, whether the old line-by-line parser and the new whole-file parser produce the
same sequences and the same canonical 13-mer set.  Any DIFFERS line means the k-mer channel changed
when kmer_select.py was rewritten, and that is enough to change the training matrix.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import kmer_select as K


def old_parse(path):
    seqs = []
    with open(path, "rb") as f:
        cur = []
        for line in f:
            if line.startswith(b">"):
                if cur:
                    seqs.append(b"".join(cur)); cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.append(b"".join(cur))
    return seqs


def old_kmers(path, k=13):
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    out = []
    for seq in old_parse(path):
        a = K.CODE[np.frombuffer(seq, dtype=np.uint8)]
        bad = np.flatnonzero(a == 255)
        for piece in (np.split(a, bad) if len(bad) else [a]):
            if len(piece) and piece[0] == 255:
                piece = piece[1:]
            if len(piece) < k:
                continue
            p = piece.astype(np.int64)
            fwd = np.lib.stride_tricks.sliding_window_view(p, k) @ powers
            rc = (np.lib.stride_tricks.sliding_window_view((3 - p)[::-1], k) @ powers)[::-1]
            out.append(np.minimum(fwd, rc))
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(out))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))[:limit]
    if not files:
        sys.exit("no files in " + d)
    bad = 0
    for fn in files:
        p = os.path.join(d, fn)
        so, sn = old_parse(p), K.read_fasta(p)
        ko, kn = old_kmers(p), K.genome_kmers(p, 13)
        same_seq = so == sn
        same_km = len(ko) == len(kn) and np.array_equal(ko, kn.astype(np.int64))
        tag = "ok     " if (same_seq and same_km) else "DIFFERS"
        print(f"{tag} {fn}: {len(so)}/{len(sn)} records, {len(ko)}/{len(kn)} distinct 13-mers"
              + ("" if same_km else f"   <-- {abs(len(ko) - len(kn))} k-mer difference"))
        if not same_seq:
            print(f"        record count or content differs: old {len(so)} vs new {len(sn)}")
            for i, (a, b) in enumerate(zip(so, sn)):
                if a != b:
                    print(f"        first differing record #{i}: old {len(a)} bp, new {len(b)} bp")
                    break
        bad += not (same_seq and same_km)
    print(f"\n{len(files) - bad}/{len(files)} genomes identical between the two parsers")
    if bad:
        print("The FASTA reader rewrite changed the k-mer content of these genomes. That is a real "
              "regression in the k-mer channel and has to be fixed whatever else is going on.")
    else:
        print("The two parsers agree on these genomes, so the FASTA reader is not the cause; the "
              "k-mer content is unchanged and the SHAP failure comes from somewhere else.")


if __name__ == "__main__":
    main()
