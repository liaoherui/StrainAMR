"""Known-mechanism layer: curated resistance determinants, calibrated on the training genomes.

The learned model only knows what its training genomes contained.  A determinant the training set
never held -- NDM in a KPC-dominated Klebsiella collection, say -- is invisible to it, while curated
knowledge recognises it at once.  This module supplies that knowledge as a separate, auditable layer:

1. AMRFinderPlus is run on every genome (build-train and build-test).  Its Subclass field is
   drug-specific (CARBAPENEM means carbapenemase activity, not merely "a beta-lactamase"), and with
   --organism it reports curated point mutations as well as acquired genes.
2. On the training genomes each determinant relevant to the drug is calibrated against the labels:
   how many resistant and how many susceptible genomes carry it.
3. A genome is called resistant by this layer when it carries a determinant that is either
   supported by the training data, or curated but too rare in the training set to judge.  A
   determinant that the training data contradicts (common in susceptible genomes) is reported but
   not counted.  No determinant means "no known mechanism", not "susceptible".
4. CARD (the RGI output the feature pipeline produces anyway) is a second source, held to a stricter
   rule.  Its drug annotation is by class and broad -- OXA-1, OXA-9 and every SHV allele are labelled
   "carbapenem" -- so a CARD determinant is never counted on curation alone.  It is counted only when
   it predicts resistance among the training genomes AMRFinderPlus does not already explain: this
   keeps what CARD adds (e.g. a regulatory mutation AMRFinderPlus does not curate) and drops what
   merely duplicates or travels with an AMRFinderPlus determinant (KPC itself; OXA-9 on the KPC
   plasmid), which would otherwise turn linkage into false resistance calls.

Files (all small):
    <db>/mechanism/amrfinder_hits.tsv.gz    every AMRFinderPlus hit, one row per genome and hit
    <db>/mechanism/screened_genomes.txt     genomes that were screened (absence of hits is meaningful)
    <db>/mechanism/card_hits.tsv.gz         every CARD/RGI hit (all drug classes), same columns
    <db>/mechanism/determinant_table.tsv    calibration of each drug-relevant determinant
    <db>/mechanism/calibration.json         settings, versions and the training-set summary
    <test>/mechanism/amrfinder_hits.tsv.gz  + screened_genomes.txt, for genomes being predicted
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(HERE, 'drug_to_amrfinder.txt')
SUBDIR = 'mechanism'
HITS = 'amrfinder_hits.tsv.gz'
SCREENED = 'screened_genomes.txt'
CARD_HITS = 'card_hits.tsv.gz'
CARD_SCREENED = 'card_screened_genomes.txt'
CLASS_FILE = os.path.join(HERE, 'drug_to_class.txt')
TABLE = 'determinant_table.tsv'
CALIB = 'calibration.json'

DEFAULT_MIN_CARRIERS = 5
DEFAULT_MIN_PPV = 0.8

# AMRFinderPlus report columns, by our name -> accepted header names (v3.10+ first, older second)
_COLS = {
    'protein': ('Protein id', 'Protein identifier'),
    'contig': ('Contig id',),
    'start': ('Start',),
    'stop': ('Stop',),
    'strand': ('Strand',),
    'element': ('Element symbol', 'Gene symbol'),
    'name': ('Element name', 'Sequence name'),
    'scope': ('Scope',),
    'type': ('Type', 'Element type'),
    'subtype': ('Subtype', 'Element subtype'),
    'class': ('Class',),
    'subclass': ('Subclass',),
    'method': ('Method',),
    'coverage': ('% Coverage of reference', '% Coverage of reference sequence'),
    'identity': ('% Identity to reference', '% Identity to reference sequence'),
    'closest': ('Closest reference accession', 'Accession of closest sequence'),
}
HIT_FIELDS = ['Genome', 'Source', 'Protein', 'Contig', 'Start', 'Stop', 'Strand', 'Element',
              'Element_name', 'Scope', 'Type', 'Subtype', 'Class', 'Subclass', 'Method', 'Coverage',
              'Identity', 'Closest_reference', 'Mechanism', 'Gene_family', 'SNPs']

# Methods that mean "the whole determinant is there": exact/allele/BLAST matches over >90% of the
# reference, and point mutations.  PARTIAL*, HMM-only and INTERNAL_STOP hits are reported but a
# determinant the training set cannot vouch for is only counted on a complete hit.
_COMPLETE = re.compile(r'^(EXACT|ALLELE|BLAST|POINT)[XPN]?$')


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [mechanism] {msg}", flush=True)


# --------------------------------------------------------------------------- drug -> terms
def drug_terms(drug, mapping_file=MAPPING_FILE):
    """AMRFinderPlus Class/Subclass terms relevant to `drug`.

    Returns (terms, source): terms is a frozenset (empty when AMRFinderPlus has no curated knowledge
    for the drug) and source says where they came from.
    """
    d = (drug or '').strip().lower()
    if os.path.exists(mapping_file):
        with open(mapping_file) as fh:
            for line in fh:
                if not line.strip() or line.startswith('#'):
                    continue
                f = line.rstrip('\n').split('\t')
                if f[0] == 'Drug_Name' or len(f) < 2:
                    continue
                if f[0].strip().lower() == d:
                    t = f[1].strip()
                    terms = frozenset() if t in ('', '-') else frozenset(
                        x.strip().upper() for x in t.split(',') if x.strip())
                    return terms, os.path.basename(mapping_file)
    # not listed: fall back to the drug's own name ("gentamicin" -> GENTAMICIN), which is how
    # AMRFinderPlus names drug-level subclasses
    parts = [p for p in re.split(r'[-/ ]+', d.upper()) if p]
    return frozenset(parts), 'drug name (not in %s)' % os.path.basename(mapping_file)


def drug_classes(drug, class_file=CLASS_FILE):
    """CARD drug classes for `drug`, from drug_to_class.txt (the same table the RGI filter uses)."""
    d = (drug or '').strip().lower()
    if os.path.exists(class_file):
        with open(class_file) as fh:
            for line in fh:
                f = line.rstrip('\n').split('\t')
                if len(f) >= 3 and f[1].strip().lower() == d:
                    return frozenset(c.strip().lower() for c in f[-1].split(';') if c.strip())
    return frozenset()


def _terms_of(hit):
    s = (hit.get('Class', '') + '/' + hit.get('Subclass', '')).upper()
    return {t.strip() for t in re.split(r'[/;,]', s) if t.strip()}


def relevant(hit, terms, classes=frozenset()):
    """Is this hit about the drug?  AMRFinderPlus hits by Class/Subclass term, CARD hits by drug class."""
    if hit.get('Source') == 'CARD':
        hc = {c.strip().lower() for c in hit.get('Class', '').split(';') if c.strip()}
        return bool(classes) and bool(hc & classes)
    return bool(terms) and bool(_terms_of(hit) & terms) and hit.get('Type', 'AMR').upper() == 'AMR'


def det_key(hit):
    """Determinant identity used in the calibration table.  CARD keys are prefixed so the two sources
    never collide, and CARD variant models carry the mutations they found (one gyrA model, several
    distinct substitutions)."""
    if hit.get('Source') == 'CARD':
        snps = hit.get('SNPs', '')
        return 'CARD:' + hit['Element'] + (f' [{snps}]' if snps and snps.lower() != 'n/a' else '')
    return hit['Element']


def complete(hit):
    return bool(_COMPLETE.match(hit.get('Method', '').strip().upper()))


# --------------------------------------------------------------------------- running AMRFinderPlus
def amrfinder_versions(exe='amrfinder', db=None):
    """(software version, database version) as reported by amrfinder, or (None, None)."""
    if shutil.which(exe) is None:
        return None, None
    cmd = [exe, '--database_version'] + (['--database', db] if db else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None, None
    text = out.stdout + out.stderr
    sw = re.search(r'Software version:\s*(\S+)', text)
    dbv = re.search(r'Database version:\s*(\S+)', text)
    return (sw.group(1) if sw else None), (dbv.group(1) if dbv else None)


def amrfinder_organisms(exe='amrfinder', db=None):
    """Organism names accepted by --organism (from `amrfinder -l`), or None if it cannot be asked."""
    try:
        out = subprocess.run([exe, '-l'] + (['--database', db] if db else []),
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout + out.stderr
    m = re.search(r'Available --organism options:\s*(.*)', text, re.S)
    body = m.group(1) if m else text
    names = [x.strip() for x in re.split(r'[,\n]', body) if re.fullmatch(r'[A-Za-z_]+', x.strip() or '-')]
    return names or None


def check_amrfinder(organism, exe='amrfinder', db=None):
    """Problems that would stop the mechanism layer, as a list of lines (empty = ready).

    Called before any genome is processed: a missing tool or database must not surface hours into
    a build.
    """
    problems = []
    if shutil.which(exe) is None:
        return ['  amrfinder     needed for the known-mechanism layer: conda install -c conda-forge '
                '-c bioconda ncbi-amrfinderplus && amrfinder -u ; or pass --no-mechanism']
    sw, dbv = amrfinder_versions(exe, db)
    if dbv is None:
        problems.append('  amrfinder     installed but no database found: run `amrfinder -u` (or pass '
                        '--amrfinder-db), or pass --no-mechanism')
        return problems
    if organism and organism.lower() != 'none':
        orgs = amrfinder_organisms(exe, db)
        if orgs and organism not in orgs:
            problems.append(f"  amrfinder     --amr-organism '{organism}' is not one of: "
                            + ', '.join(orgs))
    return problems


def _read_proteins(path):
    """(protein id, contig, start, stop, strand, sequence) from a prodigal/pyrodigal .faa."""
    pid = head = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if pid is not None:
                    yield _protein_record(pid, head, chunks)
                head = line[1:]
                pid = head.split()[0]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if pid is not None:
        yield _protein_record(pid, head, chunks)


def _protein_record(pid, head, chunks):
    seq = ''.join(chunks).rstrip('*')
    parts = [p.strip() for p in head.split('#')]
    start = stop = strand = ''
    if len(parts) >= 4:                       # prodigal header: id # start # end # strand # attrs
        start, stop, strand = parts[1], parts[2], parts[3]
        strand = {'1': '+', '-1': '-'}.get(strand, strand)
    contig = pid.rsplit('_', 1)[0] if '_' in pid else pid
    return pid, contig, start, stop, strand, seq


def read_report(path):
    """Rows of an AMRFinderPlus report as dicts keyed by our column names."""
    rows = []
    with open(path) as fh:
        rd = csv.reader(fh, delimiter='\t')
        header = next(rd, None)
        if not header:
            return rows
        idx = {}
        for key, names in _COLS.items():
            for n in names:
                if n in header:
                    idx[key] = header.index(n)
                    break
        missing = [k for k in ('element', 'class', 'subclass', 'method') if k not in idx]
        if missing:
            raise ValueError(f"{path}: not an AMRFinderPlus report (no column for {missing}); "
                             f"header was: {header}")
        for f in rd:
            if not f or not any(x.strip() for x in f):
                continue
            rows.append({k: (f[i] if i < len(f) else '') for k, i in idx.items()})
    return rows


def _hit_row(genome, rep, protein='', contig='', start='', stop='', strand=''):
    return {
        'Genome': genome,
        'Source': 'AMRFinderPlus',
        'Mechanism': '', 'Gene_family': '', 'SNPs': '',
        'Protein': protein or (rep.get('protein', '') if rep.get('protein', 'NA') != 'NA' else ''),
        'Contig': contig or rep.get('contig', ''),
        'Start': start or rep.get('start', ''),
        'Stop': stop or rep.get('stop', ''),
        'Strand': strand or rep.get('strand', ''),
        'Element': rep.get('element', ''),
        'Element_name': rep.get('name', ''),
        'Scope': rep.get('scope', ''),
        'Type': rep.get('type', ''),
        'Subtype': rep.get('subtype', ''),
        'Class': rep.get('class', ''),
        'Subclass': rep.get('subclass', ''),
        'Method': rep.get('method', ''),
        'Coverage': rep.get('coverage', ''),
        'Identity': rep.get('identity', ''),
        'Closest_reference': rep.get('closest', ''),
    }


def _run(cmd, what):
    _log(' '.join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        tail = '\n'.join((out.stderr or out.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"AMRFinderPlus failed on {what} (exit {out.returncode}):\n{tail}")
    return out


def run_amrfinder_proteins(protein_files, out_dir, organism=None, threads=1, db=None,
                           exe='amrfinder', keep_tmp=False):
    """Screen the proteins of many genomes with one AMRFinderPlus run over the distinct sequences.

    Identical protein sequences get identical results (same organism, same database), and genomes of
    one species share most of their proteins, so this is one run over ~2-5% of the input instead of
    one run per genome, each reloading the database.  Results are expanded back to every genome.
    """
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, '_amrfinder_tmp')
    os.makedirs(tmp, exist_ok=True)
    uniq = {}            # sequence hash -> index
    occ = []             # index -> [(genome, protein id, contig, start, stop, strand)]
    n_total = 0
    distinct = os.path.join(tmp, 'distinct_proteins.faa')
    with open(distinct, 'w') as out:
        for genome in sorted(protein_files):
            for pid, contig, start, stop, strand, seq in _read_proteins(protein_files[genome]):
                if not seq:
                    continue
                n_total += 1
                h = hashlib.blake2b(seq.encode(), digest_size=16).digest()
                i = uniq.get(h)
                if i is None:
                    i = uniq[h] = len(occ)
                    occ.append([])
                    out.write(f'>u{i}\n{seq}\n')
                occ[i].append((genome, pid, contig, start, stop, strand))
    _log(f"{n_total} proteins in {len(protein_files)} genomes -> {len(occ)} distinct sequences "
         f"({n_total / max(1, len(occ)):.1f}x duplication)")
    report = os.path.join(tmp, 'distinct_report.tsv')
    cmd = [exe, '-p', distinct, '-o', report, '--threads', str(max(1, int(threads)))]
    if organism and organism.lower() != 'none':
        cmd += ['--organism', organism]
    if db:
        cmd += ['--database', db]
    t0 = time.time()
    _run(cmd, f'{len(occ)} distinct proteins')
    rows = read_report(report)
    _log(f"AMRFinderPlus: {len(rows)} hits among distinct proteins in {time.time() - t0:.0f}s")
    hits = []
    for r in rows:
        m = re.fullmatch(r'u(\d+)', r.get('protein', ''))
        if not m:
            continue
        for genome, pid, contig, start, stop, strand in occ[int(m.group(1))]:
            hits.append(_hit_row(genome, r, pid, contig, start, stop, strand))
    _write_hits(out_dir, hits, sorted(protein_files))
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(hits)


def _nuc_worker(args):
    genome, path, tmp, organism, db, exe = args
    rep = os.path.join(tmp, genome + '.tsv')
    cmd = [exe, '-n', path, '-o', rep, '--threads', '1']
    if organism and organism.lower() != 'none':
        cmd += ['--organism', organism]
    if db:
        cmd += ['--database', db]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return genome, None, (out.stderr or out.stdout).strip().splitlines()[-3:]
    return genome, rep, None


def run_amrfinder_nucleotide(genome_files, out_dir, organism=None, threads=1, db=None,
                             exe='amrfinder', keep_tmp=False):
    """Screen whole assemblies (translated search): adds rRNA mutations and internal stops, slower."""
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, '_amrfinder_tmp')
    os.makedirs(tmp, exist_ok=True)
    jobs = [(g, p, tmp, organism, db, exe) for g, p in sorted(genome_files.items())]
    hits, failed = [], []
    with multiprocessing.Pool(max(1, min(int(threads), len(jobs) or 1))) as pool:
        for k, (genome, rep, err) in enumerate(pool.imap_unordered(_nuc_worker, jobs), 1):
            if rep is None:
                failed.append((genome, err))
                continue
            hits += [_hit_row(genome, r) for r in read_report(rep)]
            if k % max(1, len(jobs) // 10) == 0:
                _log(f"AMRFinderPlus (nucleotide): {k}/{len(jobs)} genomes")
    if failed:
        raise RuntimeError('AMRFinderPlus failed on %d genome(s), e.g. %s: %s'
                           % (len(failed), failed[0][0], ' | '.join(failed[0][1] or [])))
    _write_hits(out_dir, hits, sorted(genome_files))
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(hits)


def _write_hits(out_dir, hits, genomes):
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, HITS), 'wt', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=HIT_FIELDS, delimiter='\t', lineterminator='\n')
        w.writeheader()
        for h in sorted(hits, key=lambda x: (x['Genome'], x['Element'], x['Protein'])):
            w.writerow(h)
    with open(os.path.join(out_dir, SCREENED), 'w') as fh:
        fh.write('\n'.join(genomes) + '\n')
    _log(f"wrote {len(hits)} hits for {len(genomes)} genomes to {os.path.join(out_dir, HITS)}")


AMRFINDER_CACHE_MARKER = '.amrfinder-cache-complete.json'


def _reuse_cached_amrfinder(out_dir, genomes, organism, mode, exe='amrfinder', db=None):
    """Return cached hit count when a prefilled AMRFinder result is complete and compatible.

    Reuse is deliberately opt-in: the prefill program must create
    ``.amrfinder-cache-complete.json``. Existing output from an interrupted or differently
    configured run is never accepted merely because the two result files happen to exist.
    """
    marker_path = os.path.join(out_dir, AMRFINDER_CACHE_MARKER)
    hits_path = os.path.join(out_dir, HITS)
    screened_path = os.path.join(out_dir, SCREENED)

    if not all(os.path.isfile(path) for path in (marker_path, hits_path, screened_path)):
        return None

    try:
        with open(marker_path, encoding='utf-8') as handle:
            marker = json.load(handle)

        expected = sorted(set(genomes))
        with open(screened_path, encoding='utf-8') as handle:
            screened = sorted({
                line.strip() for line in handle
                if line.strip()
            })

        software_version, database_version = amrfinder_versions(exe, db)

        compatible = (
            marker.get('organism') == organism
            and marker.get('mode') == mode
            and marker.get('software_version') == software_version
            and marker.get('database_version') == database_version
            and marker.get('samples') == expected
            and screened == expected
        )
        if not compatible:
            _log('existing AMRFinder cache marker is incompatible; running AMRFinderPlus')
            return None

        expected_set = set(expected)
        count = 0

        with gzip.open(hits_path, 'rt', encoding='utf-8', errors='replace') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if reader.fieldnames != HIT_FIELDS:
                _log('existing AMRFinder cache has an unexpected header; running AMRFinderPlus')
                return None

            for row in reader:
                genome = row.get('Genome', '').strip()
                if genome not in expected_set:
                    _log('existing AMRFinder cache contains an unexpected genome; '
                         'running AMRFinderPlus')
                    return None
                count += 1

        _log(f'reusing cached AMRFinderPlus results: {count} hits for '
             f'{len(expected)} genomes')
        return count

    except (OSError, ValueError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
        _log(f'existing AMRFinder cache could not be validated ({type(exc).__name__}: '
             f'{str(exc)[:160]}); running AMRFinderPlus')
        return None


def screen(work_dir, genomes, organism=None, threads=1, db=None, mode='protein',
           protein_dir=None, genome_files=None, exe='amrfinder'):
    """Run AMRFinderPlus for `genomes` and write <work_dir>/mechanism/ hits + screened list."""
    out_dir = os.path.join(work_dir, SUBDIR)

    cached_count = _reuse_cached_amrfinder(
        out_dir,
        genomes,
        organism,
        mode,
        exe=exe,
        db=db,
    )
    if cached_count is not None:
        return cached_count

    if mode == 'nucleotide':
        files = {g: genome_files[g] for g in genomes if g in genome_files}
        return run_amrfinder_nucleotide(files, out_dir, organism, threads, db, exe)
    files = {g: os.path.join(protein_dir, g + '.faa') for g in genomes
             if os.path.exists(os.path.join(protein_dir, g + '.faa'))}
    missing = sorted(set(genomes) - set(files))
    if missing:
        raise FileNotFoundError(f"no protein file for {len(missing)} genome(s), e.g. "
                                f"{os.path.join(protein_dir, missing[0] + '.faa')}")
    return run_amrfinder_proteins(files, out_dir, organism, threads, db, exe)


# --------------------------------------------------------------------------- CARD/RGI hits
_RGI_SUBTYPE = {'protein homolog model': 'AMR', 'protein variant model': 'POINT',
                'protein overexpression model': 'POINT', 'rRNA gene variant model': 'POINT'}


def _gene_coords(gene_fa):
    """gene id -> (contig, start, stop, strand) from the prodigal-style headers of a genes .fa."""
    coords = {}
    if gene_fa and os.path.exists(gene_fa):
        with open(gene_fa) as fh:
            for line in fh:
                if line.startswith('>'):
                    head = line[1:].rstrip('\n')
                    gid, contig, start, stop, strand, _seq = _protein_record(head.split()[0], head, [])
                    coords[gid] = (contig, start, stop, strand)
    return coords


def read_rgi(path, genome, coords=None):
    """Rows of one RGI main .txt report, in the hit-table columns.

    StrainAMR runs RGI on the called genes, and RGI calls genes again inside each of them, so its
    ORF_ID is '<gene id>_1' and its Start/Stop are relative to the gene.  The gene id is the protein
    id AMRFinderPlus sees; genome coordinates come from the genes file (`coords`) when given.
    """
    coords = coords or {}
    rows = []
    with open(path) as fh:
        rd = csv.reader(fh, delimiter='\t')
        header = next(rd, None)
        if not header:
            return rows
        ix = {k: i for i, k in enumerate(header)}
        need = ('ORF_ID', 'Best_Hit_ARO', 'Drug Class', 'Cut_Off')
        if any(k not in ix for k in need):
            raise ValueError(f"{path}: not an RGI report (missing {[k for k in need if k not in ix]})")
        g = lambda f, k: (f[ix[k]] if k in ix and ix[k] < len(f) else '').strip()
        for f in rd:
            if not f or not any(x.strip() for x in f):
                continue
            orf = g(f, 'ORF_ID').split()[0] if g(f, 'ORF_ID') else ''
            gene = orf.rsplit('_', 1)[0] if re.search(r'_\d+$', orf) else orf
            contig, start, stop, strand = coords.get(gene, (gene.rsplit('_', 1)[0], '', '', ''))
            rows.append({
                'Genome': genome, 'Source': 'CARD', 'Protein': gene,
                'Contig': contig, 'Start': start, 'Stop': stop, 'Strand': strand,
                'Element': g(f, 'Best_Hit_ARO'), 'Element_name': g(f, 'Best_Hit_ARO'),
                'Scope': '', 'Type': 'AMR',
                'Subtype': _RGI_SUBTYPE.get(g(f, 'Model_type'), g(f, 'Model_type')),
                'Class': g(f, 'Drug Class'), 'Subclass': g(f, 'Antibiotic'),
                'Method': g(f, 'Cut_Off'),
                'Coverage': g(f, 'Percentage Length of Reference Sequence'),
                'Identity': g(f, 'Best_Identities'), 'Closest_reference': 'ARO:' + g(f, 'ARO'),
                'Mechanism': g(f, 'Resistance Mechanism'), 'Gene_family': g(f, 'AMR Gene Family'),
                'SNPs': g(f, 'SNPs_in_Best_Hit_ARO'),
            })
    return rows


def dump_card_hits(rgi_dir, work_dir, genomes, gene_dir=None):
    """Save every CARD/RGI hit of `genomes` (all drug classes) into <work_dir>/mechanism/.

    The per-genome RGI reports are large and slim-db removes them; this keeps the part the
    mechanism layer reads, so it can be recalibrated later without re-running RGI.
    """
    out_dir = os.path.join(work_dir, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    hits, screened, missing = [], [], []
    for g in sorted(genomes):
        path = os.path.join(rgi_dir, g + '.txt')
        if not os.path.exists(path):
            missing.append(g)
            continue
        hits += read_rgi(path, g, _gene_coords(os.path.join(gene_dir, g + '.fa')) if gene_dir else None)
        screened.append(g)
    with gzip.open(os.path.join(out_dir, CARD_HITS), 'wt', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=HIT_FIELDS, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(hits)
    with open(os.path.join(out_dir, CARD_SCREENED), 'w') as fh:
        fh.write('\n'.join(screened) + '\n')
    _log(f"saved {len(hits)} CARD/RGI hits for {len(screened)} genomes"
         + (f" ({len(missing)} without an RGI report, e.g. {missing[0]})" if missing else ''))
    return len(hits)


# --------------------------------------------------------------------------- reading hits
def load_hits(mech_dir, sources=('AMRFinderPlus', 'CARD')):
    """(hits by genome, genomes screened by AMRFinderPlus) from a mechanism/ directory.

    CARD hits are added when card_hits.tsv.gz exists; a database built before the CARD source was
    added simply has none.
    """
    by_genome = {}
    files = []
    if 'AMRFinderPlus' in sources:
        files.append((os.path.join(mech_dir, HITS), 'AMRFinderPlus'))
    if 'CARD' in sources and os.path.exists(os.path.join(mech_dir, CARD_HITS)):
        files.append((os.path.join(mech_dir, CARD_HITS), 'CARD'))
    for path, src in files:
        with gzip.open(path, 'rt') as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                row.setdefault('Source', src)
                if not row.get('Source'):
                    row['Source'] = src
                for k in HIT_FIELDS:
                    row.setdefault(k, '')
                by_genome.setdefault(row['Genome'], []).append(row)
    with open(os.path.join(mech_dir, SCREENED)) as fh:
        screened = {x.strip() for x in fh if x.strip()}
    return by_genome, screened


def has_card(work_dir):
    return os.path.exists(os.path.join(work_dir, SUBDIR, CARD_HITS))


def has_hits(work_dir):
    m = os.path.join(work_dir, SUBDIR)
    return os.path.exists(os.path.join(m, HITS)) and os.path.exists(os.path.join(m, SCREENED))


def read_labels(path):
    labels = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            f = line.split()
            if len(f) >= 2 and f[1] in ('0', '1'):
                labels[f[0]] = int(f[1])
    return labels


# --------------------------------------------------------------------------- calibration
TABLE_FIELDS = ['Determinant', 'Source', 'Element_name', 'Type', 'Subtype', 'Class', 'Subclass',
                'Mechanism', 'Carriers_R', 'Carriers_S', 'PPV', 'Unexplained_R', 'Unexplained_S',
                'Unexplained_PPV', 'Status', 'Counted', 'Methods_seen']


def _collect(genomes, by_genome, labels, keep):
    """determinant key -> stats over the training genomes, for hits that pass `keep`."""
    stats = {}
    for g in genomes:
        seen = set()
        for h in by_genome.get(g, []):
            if not keep(h):
                continue
            k = det_key(h)
            st = stats.setdefault(k, {'R': set(), 'S': set(), 'hit': h, 'methods': set()})
            st['methods'].add(h['Method'])
            if k not in seen:
                (st['R'] if labels[g] else st['S']).add(g)
                seen.add(k)
    return stats


def _row(k, st, status, counted, unexplained=None):
    h = st['hit']
    r, s_ = len(st['R']), len(st['S'])
    row = {'Determinant': k, 'Source': h.get('Source', 'AMRFinderPlus'), 'Element_name': h['Element_name'],
           'Type': h['Type'], 'Subtype': h['Subtype'], 'Class': h['Class'], 'Subclass': h['Subclass'],
           'Mechanism': h.get('Mechanism', ''), 'Carriers_R': r, 'Carriers_S': s_,
           'PPV': f'{r / (r + s_):.3f}' if r + s_ else 'NA', 'Unexplained_R': '', 'Unexplained_S': '',
           'Unexplained_PPV': '', 'Status': status, 'Counted': counted,
           'Methods_seen': ','.join(sorted(st['methods']))}
    if unexplained is not None:
        ur, us = unexplained
        row.update({'Unexplained_R': ur, 'Unexplained_S': us,
                    'Unexplained_PPV': f'{ur / (ur + us):.3f}' if ur + us else 'NA'})
    return row


def calibrate(db_dir, drug, labels_file=None, min_carriers=DEFAULT_MIN_CARRIERS,
              min_ppv=DEFAULT_MIN_PPV, mapping_file=MAPPING_FILE, settings=None):
    """Write determinant_table.tsv and calibration.json for the training genomes in db_dir.

    Uses the training labels only, so it is safe inside any train/validation/test design as long as
    db_dir was built from the training genomes alone.

    AMRFinderPlus determinant, >= min_carriers training carriers:
        supported (PPV >= min_ppv, counted) | not_predictive (reported only)
    AMRFinderPlus determinant, fewer carriers: rare_in_training (counted on a complete hit)
    CARD determinant: judged only on the training genomes that no counted AMRFinderPlus
        determinant explains; >= min_carriers of those and PPV >= min_ppv -> supported (counted);
        fewer -> redundant (it duplicates or travels with an AMRFinderPlus determinant);
        otherwise not_predictive.  Never counted on curation alone.
    """
    mech = os.path.join(db_dir, SUBDIR)
    labels = read_labels(labels_file or os.path.join(db_dir, 'train_label.txt'))
    by_genome, screened = load_hits(mech)
    genomes = sorted(g for g in screened if g in labels)
    terms, terms_src = drug_terms(drug, mapping_file)
    classes = drug_classes(drug)
    n_r = sum(labels[g] for g in genomes)
    n_s = len(genomes) - n_r

    # 1. AMRFinderPlus
    amr = _collect(genomes, by_genome, labels,
                   lambda h: h.get('Source') != 'CARD' and relevant(h, terms))
    rows = []
    for k, st in amr.items():
        n = len(st['R']) + len(st['S'])
        if n >= min_carriers:
            status = 'supported' if len(st['R']) / n >= min_ppv else 'not_predictive'
        else:
            status = 'rare_in_training'
        rows.append(_row(k, st, status, {'supported': 'yes', 'not_predictive': 'no',
                                         'rare_in_training': 'if complete hit'}[status]))
    amr_table = {r['Determinant']: r for r in rows}
    stems = supported_stems(amr_table)
    explained = {g for g in genomes
                 if call_genome([h for h in by_genome.get(g, []) if h.get('Source') != 'CARD'],
                                amr_table, terms, stems)['call'] == 'R'}

    # 2. CARD, on what AMRFinderPlus leaves unexplained
    card = _collect(genomes, by_genome, labels,
                    lambda h: h.get('Source') == 'CARD' and h.get('Method') != 'Loose'
                    and relevant(h, terms, classes))
    for k, st in card.items():
        ur = len(st['R'] - explained)
        us = len(st['S'] - explained)
        if ur + us >= min_carriers:
            status = 'supported' if ur / (ur + us) >= min_ppv else 'not_predictive'
        elif len(st['R']) + len(st['S']) >= min_carriers:
            status = 'redundant'
        else:
            status = 'rare_in_training'
        rows.append(_row(k, st, status, 'yes' if status == 'supported' else 'no', (ur, us)))

    order = {'supported': 0, 'rare_in_training': 1, 'redundant': 2, 'not_predictive': 3}
    rows.sort(key=lambda x: (x['Source'] != 'AMRFinderPlus', order[x['Status']],
                             -(x['Carriers_R'] + x['Carriers_S']), x['Determinant']))
    with open(os.path.join(mech, TABLE), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE_FIELDS, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    count = lambda src, st: sum(r['Source'] == src and r['Status'] == st for r in rows)
    calib = {'drug': drug, 'terms': sorted(terms), 'terms_source': terms_src,
             'card_classes': sorted(classes), 'card_available': has_card(db_dir),
             'available': bool(terms) or bool(classes and has_card(db_dir)),
             'min_carriers': min_carriers, 'min_ppv': min_ppv,
             'n_genomes': len(genomes), 'n_resistant': n_r, 'n_susceptible': n_s,
             'n_screened_unlabelled': len(screened) - len(genomes),
             'amrfinder': {st: count('AMRFinderPlus', st) for st in
                           ('supported', 'rare_in_training', 'not_predictive')},
             'card': {st: count('CARD', st) for st in
                      ('supported', 'redundant', 'rare_in_training', 'not_predictive')},
             'n_explained_by_amrfinder': len(explained)}
    calib.update(settings or {})
    # apparent (in-sample) performance on the training genomes: optimistic, but makes an obviously
    # broken calibration visible at build time
    table = {r['Determinant']: r for r in rows}
    perf = {}
    for name, use in (('amrfinder_only', ('AMRFinderPlus',)), ('amrfinder_plus_card', ('AMRFinderPlus', 'CARD'))):
        tp = fp = fn = tn = 0
        for g in genomes:
            hits = [h for h in by_genome.get(g, []) if h.get('Source', 'AMRFinderPlus') in use]
            call = call_genome(hits, table, terms, stems, classes)['call']
            y = labels[g]
            if call == 'R':
                tp += y; fp += 1 - y
            else:
                fn += y; tn += 1 - y
        perf[name] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                      'sensitivity': round(tp / max(1, tp + fn), 4),
                      'specificity': round(tn / max(1, tn + fp), 4)}
    calib['train_apparent'] = perf['amrfinder_plus_card']
    calib['train_apparent_amrfinder_only'] = perf['amrfinder_only']
    with open(os.path.join(mech, CALIB), 'w') as fh:
        json.dump(calib, fh, indent=1)
    if not calib['available']:
        _log(f"no curated knowledge for '{drug}' ({terms_src}); the layer will report "
             f"'unavailable' and leave every call to the model")
    else:
        _log(f"'{drug}': AMRFinderPlus terms {sorted(terms)}, CARD classes {sorted(classes)}; "
             f"{len(genomes)} training genomes ({n_r} R / {n_s} S)")
        _log(f"  AMRFinderPlus determinants: {calib['amrfinder']}")
        _log(f"  CARD determinants: {calib['card']} "
             f"(judged on the {len(genomes) - len(explained)} genomes AMRFinderPlus does not explain)")
        for name in ('amrfinder_only', 'amrfinder_plus_card'):
            a_ = perf[name]
            _log(f"  in-sample, {name.replace('_', ' ')}: sensitivity {a_['sensitivity']:.3f}, "
                 f"specificity {a_['specificity']:.3f} (TP {a_['TP']}, FP {a_['FP']}, "
                 f"FN {a_['FN']}, TN {a_['TN']})")
    return calib


def load_table(db_dir):
    path = os.path.join(db_dir, SUBDIR, TABLE)
    with open(path) as fh:
        return {r['Determinant']: r for r in csv.DictReader(fh, delimiter='\t')}


def load_calibration(db_dir):
    with open(os.path.join(db_dir, SUBDIR, CALIB)) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- calling
def gene_stem(symbol):
    """Gene family of an AMRFinderPlus symbol: blaKPC-2 -> blaKPC.  Allele-level hits carry the
    allele name, but partial and novel-allele hits are reported under the family name."""
    return re.sub(r'-\d+[A-Za-z]?$', '', symbol or '')


def supported_stems(table):
    return {gene_stem(k) for k, v in table.items()
            if v.get('Status') == 'supported' and v.get('Source', 'AMRFinderPlus') != 'CARD'}


def judge(hit, table, stems=None):
    """(status, counted) for one drug-relevant hit, given the calibration table.

    AMRFinderPlus hits:
      supported                  training carriers mostly resistant -> counted
      not_predictive             training carriers often susceptible -> reported, not counted
      catalogue_rare / _unseen   too rare in training to judge -> counted on a complete hit
      *_incomplete               partial / HMM-only / internal-stop hit -> counted only when a
                                 supported determinant of the same gene family exists
                                 ('supported_family_incomplete'), e.g. KPC split across contigs
    CARD hits: counted only when 'supported' (beyond AMRFinderPlus); otherwise card_redundant,
      card_not_predictive, card_rare or card_unseen, all reported, none counted.
    """
    if hit.get('Source') == 'CARD':
        row = table.get(det_key(hit))
        if row is None:
            return 'card_unseen', False
        if row['Status'] == 'supported':
            return 'card_supported', True
        return {'redundant': 'card_redundant', 'not_predictive': 'card_not_predictive'}.get(
            row['Status'], 'card_rare'), False
    row = table.get(hit['Element'])
    if row is not None and row['Status'] == 'supported':
        return ('supported', True) if complete(hit) else ('supported_incomplete', True)
    if row is not None and row['Status'] == 'not_predictive':
        return 'not_predictive', False
    status = 'catalogue_rare' if row is not None else 'catalogue_unseen'
    if complete(hit):
        return status, True
    if stems is None:
        stems = supported_stems(table)
    if gene_stem(hit['Element']) in stems:
        return 'supported_family_incomplete', True
    return status + '_incomplete', False


def call_genome(hits, table, terms, stems=None, classes=frozenset()):
    """Mechanism-layer call for one genome: {'call': 'R'|'none'|'unavailable', 'evidence': [...]}

    'none' means no counted determinant was found -- no known mechanism -- not "susceptible".
    """
    if not terms and not classes:
        return {'call': 'unavailable', 'evidence': []}
    if stems is None:
        stems = supported_stems(table)
    ev = []
    for h in hits:
        if h.get('Source') == 'CARD' and h.get('Method') == 'Loose':
            continue
        if not relevant(h, terms, classes):
            continue
        status, counted = judge(h, table, stems)
        ev.append(dict(h, Status=status, Counted='yes' if counted else 'no'))
    return {'call': 'R' if any(e['Counted'] == 'yes' for e in ev) else 'none', 'evidence': ev}


# --------------------------------------------------------------------------- hybrid predictions
PRED_FILE = 'predictions.tsv'
EVID_FILE = 'mechanism_evidence.tsv'
SUMMARY_FILE = 'mechanism_summary.tsv'
THRESHOLD_FILE = 'mechanism_threshold.json'
TAU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05 .. 0.95


def read_probs(path):
    """id -> (label or None, probability) from a StrainAMR probability table
    (Sample_ID/Sample_Id, Label, Pred, Prob)."""
    out = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 4 or not f[0]:
                continue
            try:
                lab = int(float(f[1]))
                lab = lab if lab in (0, 1) else None
            except ValueError:
                lab = None
            out[f[0]] = (lab, float(f[3]))
    return out


def read_feature_labels(feat_dir):
    """id -> label (or None) from a feature directory's token file (mechanism-only mode)."""
    out = {}
    path = os.path.join(feat_dir, 'strains_test_kmer_token.txt')
    if not os.path.exists(path):
        path = os.path.join(feat_dir, 'strains_train_kmer_token.txt')
    with open(path) as fh:
        fh.readline()
        for line in fh:
            f = line.split('\t')
            if len(f) >= 2:
                try:
                    lab = int(float(f[1]))
                    out[f[0]] = lab if lab in (0, 1) else None
                except ValueError:
                    out[f[0]] = None
    return out


def calls_for(feat_dir, db_dir, genomes=None):
    """(id -> call_genome() result, calibration) for the genomes screened in feat_dir/mechanism,
    judged with db_dir's calibration.  (None, calibration) if feat_dir was never screened."""
    calib = load_calibration(db_dir)
    if not has_hits(feat_dir):
        return None, calib
    table = load_table(db_dir)
    terms = frozenset(calib.get('terms') or [])
    classes = frozenset(calib.get('card_classes') or []) if calib.get('card_available') else frozenset()
    stems = supported_stems(table)
    by_genome, screened = load_hits(os.path.join(feat_dir, SUBDIR))
    ids = screened if genomes is None else [g for g in genomes if g in screened]
    return {g: call_genome(by_genome.get(g, []), table, terms, stems, classes) for g in ids}, calib


def _metrics(y, pred, score=None):
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    sens = tp / (tp + fn) if tp + fn else 0.0
    m = {'n': len(y), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
         'F1': round(2 * prec * sens / (prec + sens), 4) if prec + sens else 0.0,
         'sensitivity': round(sens, 4), 'specificity': round(tn / (tn + fp), 4) if tn + fp else 0.0,
         'precision': round(prec, 4), 'accuracy': round((tp + tn) / max(1, len(y)), 4), 'AUC': ''}
    if score is not None and 0 < sum(y) < len(y):
        try:
            from sklearn.metrics import roc_auc_score
            m['AUC'] = round(float(roc_auc_score(y, score)), 4)
        except Exception:          # noqa: BLE001 - AUC is optional
            pass
    return m


def _best_tau(y, score_fn):
    """Threshold on the grid with the best F1; ties go to the one closest to 0.5."""
    best = None
    for t in TAU_GRID:
        f1 = _metrics(y, score_fn(t))['F1']
        key = (f1, -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def hybrid(mcall, prob, tau):
    return 1 if mcall == 'R' or (prob is not None and prob >= tau) else 0


def choose_threshold(db_dir, val_dir, val_probs_file, model_dir, note=None):
    """Pick tau on the validation genomes and save it next to the model (mechanism_threshold.json).

    tau is the model probability above which a genome with no counted known determinant is called
    resistant.  The model-only threshold is tuned on the same genomes too, so the hybrid is compared
    against an equally tuned model rather than against the default 0.5.
    """
    probs = read_probs(val_probs_file)
    ids = [g for g, (lab, _p) in probs.items() if lab is not None]
    info = {'val_probs': os.path.abspath(val_probs_file), 'val_features': os.path.abspath(val_dir),
            'n_val': len(ids), 'grid': [TAU_GRID[0], TAU_GRID[-1], 0.05]}
    if note:
        info['caveat'] = note
    calls, calib = calls_for(val_dir, db_dir, ids)
    y = [probs[g][0] for g in ids]
    p = [probs[g][1] for g in ids]
    if not ids or not 0 < sum(y) < len(y):
        info.update({'tau': 0.5, 'tau_model_only': 0.5,
                     'note': 'validation set lacks one of the classes: default thresholds'})
    elif calls is None or not calib.get('available'):
        tm = _best_tau(y, lambda t: [int(q >= t) for q in p])
        info.update({'tau': tm, 'tau_model_only': tm,
                     'note': 'no known-mechanism results for the validation genomes: hybrid = model'})
    else:
        missing = [g for g in ids if g not in calls]
        mc = [calls.get(g, {'call': 'not_screened'})['call'] for g in ids]
        info['tau'] = _best_tau(y, lambda t: [hybrid(c, q, t) for c, q in zip(mc, p)])
        info['tau_model_only'] = _best_tau(y, lambda t: [int(q >= t) for q in p])
        if missing:
            info['note'] = f'{len(missing)} validation genomes were not screened (model-only for them)'
        info['val_metrics'] = {
            'model@0.5': _metrics(y, [int(q >= 0.5) for q in p], p),
            f"model@{info['tau_model_only']}": _metrics(y, [int(q >= info['tau_model_only']) for q in p], p),
            'mechanism': _metrics(y, [int(c == 'R') for c in mc]),
            f"hybrid@{info['tau']}": _metrics(y, [hybrid(c, q, info['tau']) for c, q in zip(mc, p)]),
        }
    with open(os.path.join(model_dir, THRESHOLD_FILE), 'w') as fh:
        json.dump(info, fh, indent=1)
    _log(f"threshold for genomes without a counted known determinant: tau = {info['tau']} "
         f"(model-only threshold {info['tau_model_only']}), chosen on {len(ids)} validation genomes")
    for k, m in (info.get('val_metrics') or {}).items():
        _log(f"  validation {k:<14s} F1 {m['F1']:.3f}  sens {m['sensitivity']:.3f}  "
             f"spec {m['specificity']:.3f}  (TP {m['TP']} FP {m['FP']} FN {m['FN']} TN {m['TN']})")
    return info


def load_threshold(model_dir):
    path = os.path.join(model_dir, THRESHOLD_FILE) if model_dir else ''
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def val_probs_file(model_dir):
    """The validation probability table of the checkpoint train kept."""
    sel = os.path.join(model_dir, 'selected_model.txt')
    loss = os.path.join(model_dir, 'output_sample_prob_val_loss.txt')
    f1 = os.path.join(model_dir, 'logs', 'output_sample_prob_val_best_f1.txt')
    if os.path.exists(sel) and 'checkpoint_best' in open(sel).read() and os.path.exists(loss):
        return loss
    return f1 if os.path.exists(f1) else (loss if os.path.exists(loss) else None)


def write_predictions(db_dir, feat_dir, out_dir, probs=None, tau=None, mode='hybrid',
                      tau_model_only=None, labels=None):
    """predictions.tsv + mechanism_evidence.tsv + mechanism_summary.tsv for one set of genomes.

    probs: id -> (label, probability) from the model, or None (mechanism-only).
    Returns the metrics dict (empty without labels).
    """
    os.makedirs(out_dir, exist_ok=True)
    if probs is None:
        lab = labels if labels is not None else read_feature_labels(feat_dir)
        probs = {g: (l, None) for g, l in lab.items()}
    ids = list(probs)
    calls, calib = calls_for(feat_dir, db_dir, ids)
    table = load_table(db_dir) if calls is not None else {}
    tau = 0.5 if tau is None else tau
    if calls is None:
        _log(f'WARNING: {feat_dir} has no known-mechanism results (built with --no-mechanism, or by an '
             f'older StrainAMR): calls are model-only')
    rows, ev_rows = [], []
    for g in ids:
        lab, p = probs[g]
        res = (calls or {}).get(g)
        if res is None:
            mcall = 'not_screened'
            ev = []
        else:
            mcall = res['call']
            ev = res['evidence']
        counted = [e for e in ev if e['Counted'] == 'yes']
        model_r = p is not None and p >= tau
        if mode == 'mechanism-only':
            call = int(mcall == 'R')
        elif mode == 'model-only':
            call = int(p is not None and p >= 0.5)
        else:
            call = hybrid(mcall, p, tau)
        if mcall == 'R' and model_r:
            basis = 'mechanism+model'
        elif mcall == 'R':
            basis = 'mechanism'
        elif model_r:
            basis = 'model'
        else:
            basis = 'none'
        if mode != 'hybrid':
            basis = {'mechanism-only': 'mechanism' if call else 'none',
                     'model-only': 'model' if call else 'none'}[mode]
        flags = []
        statuses = {e['Status'] for e in counted}
        if counted and statuses <= {'catalogue_unseen', 'catalogue_rare'}:
            flags.append('novel_to_training')
        if counted and statuses <= {'supported_incomplete', 'supported_family_incomplete'}:
            flags.append('incomplete_hit_only')
        if mcall == 'R' and p is not None and p < 0.5 and 'novel_to_training' not in flags:
            flags.append('model_disagrees')
        if call and mcall != 'R' and mode != 'model-only':
            flags.append('no_known_determinant')
        if mcall in ('unavailable', 'not_screened'):
            flags.append('mechanism_' + mcall)
        det = '; '.join(sorted({f"{det_key(e)}[{e['Status']}]" for e in counted}))
        other = '; '.join(sorted({f"{e['Element']}[{e['Status']}]" for e in ev
                                  if e['Counted'] != 'yes' and e.get('Source') != 'CARD'}))
        rows.append({'Sample_ID': g, 'Label': '' if lab is None else lab,
                     'Call': 'R' if call else 'S', 'Basis': basis,
                     'Model_prob': '' if p is None else f'{p:.4f}',
                     'Model_call': '' if p is None else ('R' if p >= 0.5 else 'S'),
                     'Mechanism_call': mcall, 'Determinants': det, 'Not_counted': other,
                     'Flags': ';'.join(flags)})
        for e in ev:
            t = table.get(det_key(e), {})
            ev_rows.append({'Sample_ID': g, 'Source': e.get('Source', ''), 'Determinant': det_key(e),
                            'Element_name': e.get('Element_name', ''), 'Status': e['Status'],
                            'Counted': e['Counted'], 'Method': e.get('Method', ''),
                            'Identity': e.get('Identity', ''), 'Coverage': e.get('Coverage', ''),
                            'Protein': e.get('Protein', ''), 'Contig': e.get('Contig', ''),
                            'Start': e.get('Start', ''), 'Stop': e.get('Stop', ''),
                            'Strand': e.get('Strand', ''), 'Class': e.get('Class', ''),
                            'Subclass': e.get('Subclass', ''), 'Mechanism': e.get('Mechanism', ''),
                            'Train_carriers_R': t.get('Carriers_R', 0),
                            'Train_carriers_S': t.get('Carriers_S', 0)})
    for name, rs in ((PRED_FILE, rows), (EVID_FILE, ev_rows)):
        with open(os.path.join(out_dir, name), 'w', newline='') as fh:
            if rs:
                w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()), delimiter='\t', lineterminator='\n')
                w.writeheader()
                w.writerows(rs)
            else:
                fh.write('Sample_ID\n')
    _log(f"wrote {os.path.join(out_dir, PRED_FILE)} ({len(rows)} genomes) and {EVID_FILE} "
         f"({len(ev_rows)} drug-relevant hits)")
    # metrics, where labels exist
    lab_ids = [r for r in rows if r['Label'] != '']
    metrics = {}
    if lab_ids and 0 < sum(r['Label'] for r in lab_ids) < len(lab_ids):
        y = [r['Label'] for r in lab_ids]
        p = [probs[r['Sample_ID']][1] for r in lab_ids]
        mc = [r['Mechanism_call'] for r in lab_ids]
        if p[0] is not None:
            metrics['model@0.5'] = _metrics(y, [int(q >= 0.5) for q in p], p)
            if tau_model_only is not None and tau_model_only != 0.5:
                metrics[f'model@{tau_model_only}'] = _metrics(y, [int(q >= tau_model_only) for q in p], p)
        if calls is not None:
            metrics['mechanism'] = _metrics(y, [int(c == 'R') for c in mc])
            if p[0] is not None:
                metrics[f'hybrid@{tau}'] = _metrics(y, [hybrid(c, q, tau) for c, q in zip(mc, p)])
        with open(os.path.join(out_dir, SUMMARY_FILE), 'w') as fh:
            cols = ['n', 'F1', 'sensitivity', 'specificity', 'precision', 'accuracy', 'AUC',
                    'TP', 'FP', 'FN', 'TN']
            fh.write('Method\t' + '\t'.join(cols) + '\n')
            for k, m in metrics.items():
                fh.write(k + '\t' + '\t'.join(str(m[c]) for c in cols) + '\n')
        _log('                 F1     sens   spec   AUC     (TP/FP/FN/TN)')
        for k, m in metrics.items():
            auc = f"{m['AUC']:.3f}" if m['AUC'] != '' else '  -  '
            _log(f"  {k:<14s} {m['F1']:.3f}  {m['sensitivity']:.3f}  {m['specificity']:.3f}  {auc}   "
                 f"({m['TP']}/{m['FP']}/{m['FN']}/{m['TN']})")
        disc = sum(1 for r in lab_ids if r['Mechanism_call'] == 'R' and r['Model_call'] == 'S')
        rest = sum(1 for r in lab_ids if 'no_known_determinant' in r['Flags'])
        _log(f"  known determinant but model says S: {disc}; called R without any known determinant: {rest}")
    return metrics


def eval_main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog='strainamr mechanism-eval',
        description="Add the known-mechanism layer to predictions that already exist: writes "
                    "predictions.tsv, mechanism_evidence.tsv and the model / mechanism / hybrid "
                    "comparison, without re-running training or prediction.")
    p.add_argument('--db', required=True, help='training database')
    p.add_argument('--features', required=True, help='build-test output of the genomes predicted')
    p.add_argument('--pred', help='predict output directory (or its output_sample_prob_predict.txt); '
                                  'omit for mechanism-only calls')
    p.add_argument('--model', help='training output directory: its mechanism_threshold.json gives tau')
    p.add_argument('--val-features', help='if the model has no mechanism_threshold.json yet (trained '
                                          'before this layer existed), choose tau now on these '
                                          'validation features and save it into --model')
    p.add_argument('--tau', type=float, help='use this threshold instead')
    p.add_argument('--mode', choices=['hybrid', 'model-only', 'mechanism-only'], default='hybrid')
    p.add_argument('-o', '--outdir', help='where to write (default: --pred, else --features)')
    a = p.parse_args(argv)
    probs = None
    if a.pred:
        f = a.pred if os.path.isfile(a.pred) else os.path.join(a.pred, 'logs', 'output_sample_prob_predict.txt')
        probs = read_probs(f)
    tau, tau_m = a.tau, None
    if tau is None and a.model:
        info = load_threshold(a.model)
        if info is None and a.val_features:
            vf = val_probs_file(a.model)
            if not vf:
                sys.exit(f'no validation probability table in {a.model}')
            info = choose_threshold(a.db, a.val_features, vf, a.model)
        if info is None:
            _log(f'no {THRESHOLD_FILE} in {a.model}: using tau 0.5 (pass --val-features to choose it)')
        else:
            tau, tau_m = info['tau'], info.get('tau_model_only')
    if probs is None and a.mode == 'hybrid':
        a.mode = 'mechanism-only'
    out = a.outdir or (a.pred if a.pred and os.path.isdir(a.pred) else a.features)
    write_predictions(a.db, a.features, out, probs, tau, a.mode, tau_m)
    return 0


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog='strainamr mechanism-calibrate',
        description="(Re)calibrate the known-mechanism layer of a training database. Without "
                    "--run-amrfinder it re-reads the saved AMRFinderPlus and CARD hits in mechanism/, "
                    "so thresholds can be changed in seconds; with it, AMRFinderPlus is run first on "
                    "the database's Proteins/ and the CARD hits are re-read from RGI_raw/ (only "
                    "possible before `slim-db` removed them).")
    p.add_argument('--db', required=True, help='training database (build-train output)')
    p.add_argument('--drug', help='drug name (default: from build_params.json)')
    p.add_argument('--labels', help='label file (default: <db>/train_label.txt)')
    p.add_argument('--min-carriers', type=int, default=None,
                   help=f'training carriers needed to judge a determinant (default {DEFAULT_MIN_CARRIERS})')
    p.add_argument('--min-ppv', type=float, default=None,
                   help=f'resistant fraction among carriers to count it (default {DEFAULT_MIN_PPV})')
    p.add_argument('--run-amrfinder', action='store_true',
                   help='run AMRFinderPlus on <db>/Proteins first (databases built before this layer)')
    p.add_argument('--amr-organism', help="AMRFinderPlus --organism (see `amrfinder -l`), or 'none'")
    p.add_argument('--amrfinder-db', help='AMRFinderPlus database directory (default: its own)')
    p.add_argument('-t', '--threads', type=int, default=4)
    a = p.parse_args(argv)
    db = os.path.abspath(a.db)
    params_path = os.path.join(db, 'build_params.json')
    params = json.load(open(params_path)) if os.path.exists(params_path) else {}
    mparams = params.get('mechanism', {}) or {}
    drug = a.drug or params.get('drug')
    if not drug:
        sys.exit('no drug name: pass --drug')
    min_carriers = a.min_carriers if a.min_carriers is not None else mparams.get('min_carriers', DEFAULT_MIN_CARRIERS)
    min_ppv = a.min_ppv if a.min_ppv is not None else mparams.get('min_ppv', DEFAULT_MIN_PPV)
    organism = a.amr_organism or mparams.get('organism')
    adb = a.amrfinder_db or mparams.get('amrfinder_db')
    if a.run_amrfinder:
        if not organism:
            sys.exit("--run-amrfinder needs --amr-organism (e.g. Klebsiella_pneumoniae, see "
                     "`amrfinder -l`), or 'none' to skip point mutations")
        problems = check_amrfinder(organism, db=adb)
        if problems:
            sys.exit('\n'.join(['Cannot run AMRFinderPlus:'] + problems))
        pdir = os.path.join(db, 'Proteins')
        if not os.path.isdir(pdir):
            sys.exit(f"{pdir} not found (removed by slim-db?); the database must be rebuilt")
        genomes = sorted(read_labels(a.labels or os.path.join(db, 'train_label.txt')))
        screen(db, genomes, organism, a.threads, adb, 'protein', protein_dir=pdir)
        sw, dbv = amrfinder_versions(db=adb)
        mparams.update({'enabled': True, 'organism': organism, 'mode': 'protein',
                        'amrfinder_version': sw, 'database_version': dbv, 'amrfinder_db': adb})
    rgi_raw = os.path.join(db, 'RGI_raw')
    if os.path.isdir(rgi_raw) and (a.run_amrfinder or not has_card(db)):
        dump_card_hits(rgi_raw, db, sorted(read_labels(a.labels or os.path.join(db, 'train_label.txt'))),
                       gene_dir=os.path.join(db, 'Genes'))
    elif not has_card(db):
        _log('no CARD hits saved and no RGI_raw/ to read them from: calibrating on AMRFinderPlus only')
    if not has_hits(db):
        sys.exit(f"no {SUBDIR}/{HITS} in {db}: pass --run-amrfinder (needs <db>/Proteins)")
    mparams.update({'enabled': True, 'min_carriers': min_carriers, 'min_ppv': min_ppv})
    calibrate(db, drug, a.labels, min_carriers, min_ppv,
              settings={k: mparams.get(k) for k in ('organism', 'mode', 'amrfinder_version',
                                                    'database_version')})
    if params:
        params['mechanism'] = mparams
        with open(params_path, 'w') as fh:
            json.dump(params, fh, indent=1)
    _log(f"wrote {os.path.join(db, SUBDIR, TABLE)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
