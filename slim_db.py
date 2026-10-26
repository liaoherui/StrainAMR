"""Shrink build directories to what the later steps read.

    strainamr slim-db db-Kpn-imipenem/                     # dry run: shows what would be removed
    strainamr slim-db db-Kpn-imipenem/ --apply             # does it
    strainamr slim-db test-Kpn-imipenem/ model/val_features --apply   # build-test outputs too

Works on a training database (build-train output) and on feature directories made by build-test
(a test set, or the val_features/ that `train --val-input` creates); the kind is detected.

Removes an explicit list of build intermediates (per-genome genes, proteins, raw RGI output, graph
inputs, clustering intermediates, unused SHAP matrices) and keeps everything else, so a file this
list does not know about is never deleted.  Before removing anything it

  * replaces the rgi_train/ symlinks with copies of the files they point to (RGI/ is removed, and
    prediction reads rgi_train/ for gene-family annotation);
  * checks that every file the later stages require is present;
  * refuses if the known-mechanism layer is enabled but its results are missing, because Proteins/
    is the only way to recover them (`strainamr mechanism-calibrate --run-amrfinder`).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys

# Files build-test, train and predict require (raise if missing), traced through the code.
REQUIRED = [
    'train_label.txt',
    'feature_remain_graph.txt', 'feature_remain_pc.txt',
    'strains_train_sentence.txt', 'strains_train_pc_token.txt', 'strains_train_kmer_token.txt',
    'strains_train_sentence_fs.txt', 'strains_train_pc_token_fs.txt',
    'node_token_match.txt', 'pc_matches.txt', 'kmer_token_id.txt', 'longest_len_fs.txt',
    'merged_proteins_train.fa', 'merge_train_cdhit.clstr',
    'shap/strains_train_sentence_fs_shap_rmf.txt', 'shap/strains_train_pc_token_fs_shap_rmf.txt',
    'shap/strains_train_kmer_token_shap_rmf.txt',
    'shap/strains_train_sentence_fs_shap_filter.txt', 'shap/strains_train_pc_token_fs_shap_filter.txt',
    'shap/strains_train_kmer_token_shap_filter.txt',
    'GFA_train_Minimap2',
]
MECHANISM_REQUIRED = ['mechanism/amrfinder_hits.tsv.gz', 'mechanism/screened_genomes.txt',
                      'mechanism/determinant_table.tsv', 'mechanism/calibration.json']

# Build intermediates nothing downstream reads.  Directories are removed whole.
REMOVE_DIRS = ['Genes', 'Proteins', 'Genes_info', 'RGI_raw', 'RGI', 'genes_train', 'proteins_train',
               'Genomes_train', 'K-mer_lists', 'mmseqs_tmp', 'mechanism/_amrfinder_tmp']
REMOVE_FILES = ['merge_train_cdhit',                 # cd-hit representative FASTA (only .clstr is read)
                'mmseqs_pc_all_seqs.fasta', 'mmseqs_pc_rep_seq.fasta', 'mmseqs_pc_cluster.tsv',
                'shap/*_shap_local_matrix.txt', 'shap/*_cls_info.txt', 'shap/*_cls*_feature.txt']
# build-test output (test set or val_features/): predict, train --val-dir and interpret read only the
# token files, the length file, shap/ and mechanism/.
TEST_REQUIRED = ['strains_test_sentence_fs.txt', 'strains_test_pc_token_fs.txt',
                 'strains_test_kmer_token.txt', 'longest_len_fs.txt']
TEST_REMOVE_DIRS = ['Genes_ts', 'Proteins_ts', 'Genes_info_ts', 'RGI_raw_ts', 'RGI_ts', 'genes_test',
                    'proteins_test', 'rgi_test', 'Genomes_test', 'Align_test_res', 'PS_out/K-mer_lists',
                    'mechanism/_amrfinder_tmp']
TEST_REMOVE_FILES = ['merged_proteins_test.fa', 'merge_test_cdhit', 'merge_test_cdhit.clstr']

# PS_out/: keep only what is read later (the PhenotypeSeeker model and the MLdf table used to infer
# the drug name); drop the rest.
PS_KEEP = ['*.pkl', '*_MLdf.csv']


def size_of(path):
    if os.path.islink(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dp, _dn, fns in os.walk(path):
        for f in fns:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1024.0


def kind_of(d):
    if os.path.exists(os.path.join(d, 'train_label.txt')):
        return 'database'
    if os.path.exists(os.path.join(d, 'strains_test_kmer_token.txt')):
        return 'features'
    return None


def plan_features(d):
    """Same as plan(), for a build-test output directory.  The fourth element lists fix-ups to run
    before removing anything (saving the CARD hits from RGI_raw_ts/ if they were not saved)."""
    remove, problems, fixups = [], [], []
    for x in TEST_REMOVE_DIRS:
        p = os.path.join(d, x)
        if os.path.lexists(p):
            remove.append(p)
    for x in TEST_REMOVE_FILES:
        p = os.path.join(d, x)
        if os.path.isfile(p):
            remove.append(p)
    for r in TEST_REQUIRED:
        if not os.path.exists(os.path.join(d, r)):
            problems.append(f'required file missing (build-test did not finish?): {r}')
    mech = os.path.join(d, 'mechanism')
    has_amr = os.path.exists(os.path.join(mech, 'amrfinder_hits.tsv.gz'))
    has_card = os.path.exists(os.path.join(mech, 'card_hits.tsv.gz'))
    rgi_raw = os.path.join(d, 'RGI_raw_ts')
    if has_amr and not has_card and os.path.isdir(rgi_raw):
        fixups.append(('save the CARD/RGI hits from RGI_raw_ts/ into mechanism/', d))
    if not has_amr and os.path.isdir(os.path.join(d, 'Proteins_ts')):
        problems.append('no known-mechanism results (mechanism/amrfinder_hits.tsv.gz): predictions '
                        'for these genomes would be model-only, and Proteins_ts/ is about to be '
                        'removed. Pass --force if that is intended (e.g. built with --no-mechanism)')
    return remove, problems, [], fixups


def _save_card(d):
    import mechanism_layer as ML
    with open(os.path.join(d, 'mechanism', 'screened_genomes.txt')) as fh:
        genomes = [x.strip() for x in fh if x.strip()]
    ML.dump_card_hits(os.path.join(d, 'RGI_raw_ts'), d, genomes, gene_dir=os.path.join(d, 'Genes_ts'))


def plan(db):
    """(paths to remove, problems that block slimming, rgi_train links to materialise, fix-ups)."""
    if kind_of(db) == 'features':
        return plan_features(db)
    remove, problems = [], []
    for d in REMOVE_DIRS:
        p = os.path.join(db, d)
        if os.path.lexists(p):
            remove.append(p)
    for pat in REMOVE_FILES:
        sub, base = os.path.split(pat)
        root = os.path.join(db, sub)
        if os.path.isdir(root):
            remove += [os.path.join(root, f) for f in sorted(os.listdir(root))
                       if fnmatch.fnmatch(f, base) and os.path.isfile(os.path.join(root, f))]
    ps = os.path.join(db, 'PS_out')
    if os.path.isdir(ps):
        for f in sorted(os.listdir(ps)):
            if not any(fnmatch.fnmatch(f, k) for k in PS_KEEP):
                remove.append(os.path.join(ps, f))
    for r in REQUIRED:
        if not os.path.exists(os.path.join(db, r)):
            problems.append(f'required file missing (the database is incomplete): {r}')
    params = {}
    pp = os.path.join(db, 'build_params.json')
    if os.path.exists(pp):
        with open(pp) as fh:
            params = json.load(fh)
    mech = params.get('mechanism') or {}
    if mech.get('enabled'):
        miss = [m for m in MECHANISM_REQUIRED if not os.path.exists(os.path.join(db, m))]
        if miss:
            problems.append('known-mechanism layer enabled but missing ' + ', '.join(miss)
                            + '; recover it with `strainamr mechanism-calibrate --db ' + db
                            + ' --run-amrfinder` while Proteins/ still exists')
    if mech.get('enabled') and os.path.isdir(os.path.join(db, 'RGI_raw')) \
            and not os.path.exists(os.path.join(db, 'mechanism', 'card_hits.tsv.gz')):
        problems.append('the CARD/RGI hits are not saved yet and RGI_raw/ is about to be removed; '
                        'run `strainamr mechanism-calibrate --db ' + db + '` first (it saves them '
                        'in seconds)')
    if not mech.get('enabled') and mech.get('error'):
        problems.append('the known-mechanism layer failed during the build (' + mech['error'][:120]
                        + '); fix it with `strainamr mechanism-calibrate --db ' + db
                        + ' --run-amrfinder` first, or pass --force to slim without it')
    links = []
    rt = os.path.join(db, 'rgi_train')
    if os.path.isdir(rt):
        for f in sorted(os.listdir(rt)):
            fp = os.path.join(rt, f)
            if os.path.islink(fp):
                target = os.path.realpath(fp)
                if not os.path.exists(target):
                    problems.append(f'rgi_train/{f} points to a missing file ({target})')
                else:
                    links.append((fp, target))
    return remove, problems, links, []


def slim_one(path, apply, force):
    d = os.path.abspath(path)
    if not os.path.isdir(d):
        print(f'not a directory: {d}')
        return 1
    kind = kind_of(d)
    if kind is None:
        print(f'{d}: neither a training database (no train_label.txt) nor a build-test output '
              f'(no strains_test_kmer_token.txt); not touching it')
        return 1
    remove, problems, links, fixups = plan(d)
    soft = ('failed during the build', 'would be model-only')
    blocking = [p for p in problems if not (force and any(k in p for k in soft))]
    before = size_of(d)
    freed = sum(size_of(p) for p in remove) - sum(size_of(t) for _l, t in links)  # links become copies
    print(f'{kind}: {d}')
    print(f'current size: {human(before)}')
    for what, _ in fixups:
        print(f'  first   {what}')
    for p in remove:
        print(f'  remove  {human(size_of(p)):>10}  {os.path.relpath(p, d)}')
    if links:
        print(f'  copy    rgi_train/: {len(links)} symlinks replaced by the files they point to')
    print(f'size after: about {human(before - freed)} (frees {human(freed)})')
    if blocking:
        print('Not slimming:')
        for p in blocking:
            print('  - ' + p)
        return 1
    if not apply:
        print('Dry run. Pass --apply to do it.')
        return 0
    for _what, arg in fixups:
        _save_card(arg)
    for fp, target in links:
        tmp = fp + '.tmp'
        shutil.copy2(target, tmp)
        os.replace(tmp, fp)                   # replaces the symlink itself, not its target
    for p in remove:
        if os.path.islink(p) or os.path.isfile(p):
            os.remove(p)
        else:
            shutil.rmtree(p)
    _r, problems2, links2, _f = plan(d)
    still = [p for p in problems2 if p.startswith('required')]
    if still or links2:
        print('ERROR after slimming: ' + '; '.join(still + [f'{len(links2)} links left']))
        return 1
    print(f'done: {human(size_of(d))}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='strainamr slim-db', description=__doc__.split('\n\n')[0])
    ap.add_argument('dirs', nargs='+',
                    help='training database(s) and/or build-test output directories')
    ap.add_argument('--apply', action='store_true', help='actually remove (default: dry run)')
    ap.add_argument('--force', action='store_true',
                    help='slim even without known-mechanism results (built with --no-mechanism, '
                         'or the layer failed)')
    a = ap.parse_args(argv)
    rc = 0
    for i, d in enumerate(a.dirs):
        if i:
            print()
        rc = max(rc, slim_one(d, a.apply, a.force))
    return rc


if __name__ == '__main__':
    sys.exit(main())
