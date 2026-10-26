import re
import os
import sys
import argparse
import numpy as np
import shutil
import pc_clusters as PC
from sample_ids import index_genomes, sample_id_from_filename, validate_label_genomes
from extract_seq_for_graph import extract
from build_graph_batch_minimap2 import build
from align_genome_to_graph import align
from generate_token_from_graph import generate_fg
from generate_token_from_alignment import generate_at
from generate_token_from_ps import generate_tps
from generate_token_from_ps_predict import generate_tpsp
from feature_selection_sp import sef
from cal_length_fs import scan_length_fs,scan_length_fs_shap
from library import shap_feature_select_withcls
from steplog import log, step, skipped, summary
script_dir = os.path.dirname(os.path.abspath(__file__))

def build_dir(idir):
    if not os.path.exists(idir):
        os.makedirs(idir)


import multiprocessing


def check_tools(gene_caller='pyrodigal', cluster_tool='mmseqs', kmer_tool='builtin',
                kmer_weights=True, snv_c=0, pc_c=0, kmer_c=0, fatal=True):
    """Check the external tools this run will need, before hours of work depend on them.

    Missing tools used to surface only when the pipeline reached them -- after gene calling and RGI
    had already run. Each entry is (command, what it is for, what to do without it).
    """
    import shutil as _sh
    problems = []
    if gene_caller == 'pyrodigal':
        # a Python package, not a command, so `which` never finds it
        try:
            import pyrodigal
            if not hasattr(pyrodigal, 'GeneFinder') and not hasattr(pyrodigal, 'OrfFinder'):
                problems.append('  pyrodigal     installed but has neither GeneFinder nor OrfFinder; '
                                'install pyrodigal>=3, or use --gene-caller prodigal')
            else:
                log('[tools] pyrodigal ' + getattr(pyrodigal, '__version__', '?'), flush=True)
        except ImportError:
            problems.append('  pyrodigal     needed for gene calling (a Python package, not a '
                            'command): conda install -c conda-forge -c bioconda "pyrodigal>=3", '
                            'or use --gene-caller prodigal')
    needed = [('rgi', 'resistance-gene annotation', None)]
    if gene_caller == 'prodigal':
        needed.append(('prodigal', 'gene calling', 'use --gene-caller pyrodigal'))
    if not snv_c:
        needed += [('minimap2', 'pangenome graph', None), ('seqwish', 'pangenome graph', None)]
    if not pc_c:
        if cluster_tool == 'mmseqs':
            needed.append(('mmseqs', 'protein clustering', 'use --cluster-tool cd-hit'))
        else:
            needed.append(('cd-hit', 'protein clustering', 'use --cluster-tool mmseqs'))
    if not kmer_c and kmer_tool == 'builtin' and kmer_weights:
        needed.append(('mash', 'genome distances for k-mer sample weighting',
                       'use --no-kmer-weights (selection will not be corrected for population '
                       'structure)'))
    if not kmer_c and kmer_tool == 'phenotypeseeker':
        needed.append(('phenotypeseeker', 'k-mer selection', 'use --kmer-tool builtin'))
    missing = [(c, w, f) for c, w, f in needed if _sh.which(c) is None]
    if not missing and not problems:
        return True
    lines = ['Missing dependencies:'] + problems
    for cmd, what, fix in missing:
        lines.append(f'  {cmd:<12}  needed for {what}' + (f'; otherwise {fix}' if fix else ''))
    if missing:
        lines.append('Install the commands with: conda install -c conda-forge -c bioconda '
                     + ' '.join({'mmseqs': 'mmseqs2'}.get(c, c) for c, _, _ in missing))
    msg = '\n'.join(lines)
    if fatal:
        raise RuntimeError(msg)
    print(msg, flush=True)
    return False


def call_genes_pyrodigal(genome_path, genes_fa, proteins_faa, info_file=None):
    """Prodigal gene calling in-process via pyrodigal, writing the same .fa/.faa prodigal wrote.

    Same algorithm and same output format (headers '>contig_i # start # end # strand # attrs'), but
    no subprocess per genome and no temporary files.  The whole assembly trains the model once, as
    prodigal's single mode does; tiny assemblies fall back to metagenomic mode.
    """
    import pyrodigal
    # pyrodigal 3.x calls it GeneFinder, 2.x called it OrfFinder
    Finder = getattr(pyrodigal, 'GeneFinder', None) or getattr(pyrodigal, 'OrfFinder')
    contigs = []
    name = None
    chunks = []
    with open(genome_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if name is not None:
                    contigs.append((name, ''.join(chunks)))
                name = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if name is not None:
        contigs.append((name, ''.join(chunks)))
    total = sum(len(s) for _, s in contigs)
    if total < 100000:
        finder = Finder(meta=True)
    else:
        finder = Finder()
        finder.train(*[s.encode() for _, s in contigs])
    n_genes = 0
    with open(genes_fa, 'w') as fa, open(proteins_faa, 'w') as faa:
        for cname, seq in contigs:
            genes = finder.find_genes(seq.encode())
            if hasattr(genes, 'write_genes'):
                genes.write_genes(fa, sequence_id=cname)
                genes.write_translations(faa, sequence_id=cname)
            else:                                   # very old pyrodigal: write the records ourselves
                for i, g in enumerate(genes, 1):
                    head = (f"{cname}_{i} # {g.begin} # {g.end} # {g.strand} # ID={i};"
                            f"partial={int(bool(g.partial_begin))}{int(bool(g.partial_end))}")
                    fa.write(f">{head}\n{g.sequence()}\n")
                    faa.write(f">{head}\n{g.translate()}\n")
            n_genes += len(genes)
    if info_file:
        with open(info_file, 'w') as fh:
            fh.write('# pyrodigal: %d genes in %d contigs (%d bp)\n' % (n_genes, len(contigs), total))
    return n_genes


def _prodigal_rgi_worker(args):
    s, dr, gdir, ginfo, pdir, rgi, rgi_threads, gene_caller, rgi_extra = args
    proc = multiprocessing.current_process()
    log(f"[Prodigal/RGI] start {s} -- {proc.name}", flush=True)
    if os.path.exists(rgi+'/'+s+'.txt') and os.path.getsize(rgi+'/'+s+'.txt') != 0:
        log(f"[Prodigal/RGI] skip {s} -- {proc.name}", flush=True)
        return s, 0
    if not os.path.exists(pdir+'/'+s+'.faa') or os.path.getsize(pdir+'/'+s+'.faa') == 0:
        if gene_caller=='pyrodigal':
            try:
                call_genes_pyrodigal(dr[s], gdir+'/'+s+'.fa', pdir+'/'+s+'.faa', ginfo+'/'+s+'.genes')
            except Exception as exc:                       # noqa: BLE001 - reported per genome below
                log(f"[Prodigal/RGI] pyrodigal failed on {s}: {exc}", flush=True)
                return s, 1
        else:
            rc=os.system('prodigal -i '+dr[s]+' -o '+ginfo+'/'+s+'.genes -d '+gdir+'/'+s+'.fa -a '+pdir+'/'+s+'.faa')
            if rc!=0:
                return s, rc
    if rgi_threads:                      # 0 means RGI runs later, batched over all genomes at once
        rc=os.system('rgi main --input_sequence '+gdir+'/'+s+'.fa --output_file '+rgi+'/'+s+
                     ' --local --clean  -n '+str(rgi_threads)+(' '+rgi_extra if rgi_extra else ''))
    else:
        rc=0
    log(f"[Prodigal/RGI] done {s} -- {proc.name}", flush=True)
    return s, rc


def run_prodigal_rgi(dr, odir, threads=1, gene_caller='pyrodigal', rgi_batch=0, rgi_extra=''):
    gdir=odir+'/Genes'
    ginfo=odir+'/Genes_info'
    pdir=odir+'/Proteins'
    rgi=odir+'/RGI_raw'

    build_dir(gdir)
    build_dir(ginfo)
    build_dir(pdir)
    build_dir(rgi)

    # One process per genome, each of which used to start RGI with a hard-coded "-n 10": with -t 32 that
    # is 320 threads competing for the machine, which is slower than running 32.  Spend the budget once.
    threads=max(1,int(threads))
    workers=min(threads,max(1,len(dr)))
    batched = rgi_batch != 1                # rgi_batch == 1 keeps the original one-RGI-run-per-genome
    rgi_threads=0 if batched else max(1,threads//workers)
    if batched:
        log(f"[Prodigal/RGI] calling genes for {len(dr)} genomes with {workers} processes; "
              f"RGI runs afterwards over all of them at once",flush=True)
    else:
        log(f"[Prodigal/RGI] {len(dr)} genomes, {workers} parallel processes x {rgi_threads} "
              f"RGI threads",flush=True)

    params = [(s, dr, gdir, ginfo, pdir, rgi, rgi_threads, gene_caller, rgi_extra) for s in dr]
    pool = multiprocessing.Pool(processes=workers)
    # apply_async without collecting results swallowed every failure: a genome whose RGI crashed simply
    # produced no output and the pipeline carried on with an incomplete dataset.
    failed=[s for s,rc in pool.map(_prodigal_rgi_worker, params) if rc]
    pool.close()
    pool.join()
    if failed:
        raise RuntimeError('prodigal/RGI failed for %d genome(s): %s%s'
                           %(len(failed),', '.join(failed[:10]),' ...' if len(failed)>10 else ''))
    if batched:
        import rgi_batch as RB          # aliased: a plain import would shadow the rgi_batch argument
        gene_files={s: gdir+'/'+s+'.fa' for s in dr if os.path.exists(gdir+'/'+s+'.fa')}
        RB.run_rgi(gene_files, rgi, threads,
                   per_batch=rgi_batch_size(rgi_batch, len(gene_files)), extra=rgi_extra)
    return gdir,pdir


def rgi_batch_size(batch_arg, n_genomes):
    """Sequences per RGI run; 0 (the default) lets rgi_batch size them for visible progress."""
    return int(batch_arg) if batch_arg and int(batch_arg) > 1 else 0

def link_or_copy(src,odir):
    """Symlink a file into the work directory, falling back to a copy where symlinks are unavailable.

    These directories used to be full copies of the gene, protein and RGI output of every genome, so a
    run held three extra copies of the whole dataset on disk (plus a fourth in the merged protein FASTA).
    Nothing downstream writes to them, so links are equivalent.
    """
    dst=os.path.join(odir,os.path.basename(src))
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src),dst)
    except (OSError,NotImplementedError):
        shutil.copy2(src,dst)

def copy_genome(gdir,index,odir,t):
    bfix=''
    if t=='gene':
        bfix='fa'
    else:
        bfix='txt'
    for i in index:
        src = os.path.join(gdir, i + '.' + bfix)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"{t.capitalize()} file not found for sample {i}: {src}. "
                "Check whether genome filenames match IDs in the label file."
            )
        link_or_copy(src, odir)

def copy_protein(pdir,index,odir):
    for i in index:
        src = os.path.join(pdir, i + '.faa')
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Protein file not found for sample {i}: {src}. "
                "Check whether genome filenames match IDs in the label file."
            )
        link_or_copy(src, odir)
def stat_rig(indir):
    arr=[]
    for filename in os.listdir(indir):
        if re.search('json', filename):continue
        f = open(indir + '/' + filename, 'r')
        line = f.readline()
        c=0
        while True:
            line=f.readline().strip()
            if not line:break
            c+=1
        arr.append(c)
    arr=np.array(arr)
    return np.median(arr)


def filter_rgi(indir,drug,mfile,odir):
    f=open(mfile,'r')
    d={}
    line=f.readline()
    if not os.path.exists(odir):
        os.makedirs(odir)
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        if ele[1]==drug:
            if drug not in d:d[drug]={}
            cl=re.split(';',ele[-1])
            for c in cl:
                d[drug][c]=''
    for filename in os.listdir(indir):
        if re.search('json',filename):
            os.system('cp '+indir+'/'+filename+' '+odir)
            continue
        o=open(odir+'/'+filename,'w+')
        f=open(indir+'/'+filename,'r')
        line=f.readline()
        o.write(line)
        while True:
            line=f.readline().strip()
            if not line:break
            for c in d[drug]:
                if re.search(c,line):
                    o.write(line+'\n')
                    break

def merge_all_proteins(indir,odir,t):
    """Concatenate the protein FASTAs, dropping exact duplicate sequences.

    Bacterial genomes of one species share the great majority of their proteins verbatim: on 1393
    K. pneumoniae genomes, 7.3 M proteins collapse to about 176 k distinct sequences.  Clustering the
    distinct set is ~40x less work and ~40x less memory, and the result is the same -- identical
    sequences always land in the same cluster at any identity threshold, so they are simply written
    back into their representative's cluster afterwards (see pc_clusters.cluster_tokens).
    """
    import hashlib
    merged=odir+'/merged_proteins_'+t+'.fa'
    dup_map=odir+'/duplicate_proteins_'+t+'.tsv'
    seen={}                      # sequence hash -> representative protein id
    n_all=0
    with open(merged,'w') as o, open(dup_map,'w') as od:
        for filename in sorted(os.listdir(indir)):
            path=os.path.join(indir,filename)
            if not os.path.isfile(path):continue
            pid=None
            chunks=[]
            def flush():
                nonlocal pid,chunks
                if pid is None:return
                seq=''.join(chunks)
                h=hashlib.blake2b(seq.encode(),digest_size=16).digest()
                rep=seen.get(h)
                if rep is None:
                    seen[h]=pid
                    o.write('>'+pid+'\n'+seq+'\n')
                else:
                    od.write(pid+'\t'+rep+'\n')
                pid=None;chunks=[]
            with open(path) as f:
                for line in f:
                    line=line.rstrip('\n')
                    if line.startswith('>'):
                        flush()
                        pid=line[1:].split()[0]
                        n_all+=1
                    elif line:
                        chunks.append(line.strip())
            flush()
    log(f'[proteins] {n_all} proteins -> {len(seen)} distinct sequences to cluster',flush=True)
    return merged

def run_cdhit(ptrain,work_dir,threads=0,tool='mmseqs',memory_gb=0):
    """Cluster proteins at 90% identity and write a CD-HIT style .clstr file.

    MMseqs2 (default) does this in a fraction of CD-HIT's time and memory on large protein sets;
    --cluster-tool cd-hit keeps the original behaviour.  Both write merge_train_cdhit.clstr, so
    everything downstream is unchanged.
    """
    cls1=work_dir+'/merge_train_cdhit.clstr'
    if tool=='cd-hit':
        mem='-M 0' if not memory_gb else '-M '+str(int(memory_gb*1024))
        rc=os.system('cd-hit -i '+ptrain+' -d 0 -o '+work_dir+'/merge_train_cdhit -c 0.9 -n 5 '+mem+
                     ' -T '+str(threads))
        if rc!=0:
            raise RuntimeError('cd-hit failed (exit code %d)'%rc)
        return cls1
    tmp=work_dir+'/mmseqs_tmp'
    build_dir(tmp)
    pref=work_dir+'/mmseqs_pc'
    cmd=('mmseqs easy-cluster '+ptrain+' '+pref+' '+tmp+' --min-seq-id 0.9 -c 0.8 --cov-mode 1'
         ' --cluster-mode 0 -v 1 --threads '+str(threads if threads else 1))
    if memory_gb:
        cmd+=' --split-memory-limit '+str(int(memory_gb))+'G'
    print(cmd,flush=True)
    rc=os.system(cmd)
    if rc!=0:
        raise RuntimeError('mmseqs easy-cluster failed (exit code %d); install MMseqs2 or pass '
                           '--cluster-tool cd-hit'%rc)
    mmseqs_tsv_to_clstr(pref+'_cluster.tsv',cls1)
    shutil.rmtree(tmp,ignore_errors=True)
    return cls1

def mmseqs_tsv_to_clstr(tsv,ofile):
    """MMseqs2 writes <representative>\t<member> per line; rewrite it as a CD-HIT .clstr file.

    Only the fields the parser downstream reads are reproduced: the ">Cluster n" header lines and,
    per member, a third whitespace-separated field ">id..." with "*" marking the representative.
    """
    members={}
    with open(tsv) as f:
        for line in f:
            rep,mem=line.rstrip('\n').split('\t')[:2]
            members.setdefault(rep,[]).append(mem)
    with open(ofile,'w') as o:
        for i,(rep,mem) in enumerate(members.items()):
            o.write('>Cluster %d\n'%i)
            ordered=[rep]+[m for m in mem if m!=rep]
            for k,m in enumerate(ordered):
                o.write('%d\t0aa, >%s... %s\n'%(k,m,'*' if k==0 else 'at 90.00%'))

def output_pc_token_file(d,pdir,label,ofile,idx):
    dr={} # Strain prefix -> Tokens string
    for filename in os.listdir(pdir):
        pre=sample_id_from_filename(filename)
        if pre not in dr:
            dr[pre]=[]
        f=open(pdir+'/'+filename,'r')
        dc={} # contigs -> token list
        while True:
            line=f.readline().strip()
            if not line:break
            if not re.search('>',line):continue
            ele=line.split()
            pid=re.sub('>','',ele[0])
            if pid not in d:continue
            ct=re.split('_',pid)
            ct='_'.join(ct[:-1])
            if ct not in dc:          # dc preserves insertion order, so a separate contig list is redundant
                dc[ct]=[]
            dc[ct].append(d[pid])
        '''
        if pre=='573_46831':
            print(dc)
            exit()
        '''
        tem=[] # all contig tokens
        for c in dc:
            tem.append(','.join(dc[c]))
        dr[pre]=',0,'.join(tem)
        #print(dr)
        #exit()
    f=open(label,'r')
    o=open(ofile,'w+')
    line=f.readline().strip()
    o.write(line+'\tTokens_Num\tTokens\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        if ele[0] not in idx:continue
        o.write(line+'\t')
        arr=re.split(',',dr[ele[0]])
        o.write(str(len(arr))+'\t'+dr[ele[0]]+'\n')
    o.close()

    

def generate_tokens_from_cdhit(work_dir,label,train):
    # The cluster-file parsing and the token numbering live in pc_clusters, which build-test uses too:
    # the two sides must produce the same protein -> token id mapping or the validation PC tokens mean
    # different clusters than the ones the model was trained on.
    d=PC.cluster_tokens(work_dir+'/merge_train_cdhit.clstr',
                        dup_map=work_dir+'/'+PC.DUP_MAP,
                        matches_file=work_dir+'/pc_matches.txt')
    log('[PC] '+PC.summarise(work_dir+'/merge_train_cdhit.clstr',work_dir+'/'+PC.DUP_MAP),flush=True)
    output_pc_token_file(d,work_dir+'/proteins_train',label,work_dir+'/strains_train_pc_token.txt',train)
    #output_pc_token_file(d2,work_dir+'/proteins_val',label,work_dir+'/strains_val_pc_token.txt',val)
    

def run_ps(train,ingenome,label,drug,work_dir,threads=1,keep_kmer_lists=False,
           kmer_tool='builtin',kmer_weights=True):
    o=open('tem.pheno','w+') 
    o.write('ID\tAddress\t'+drug+'\n')
    dtrain={} # Pre -> Genome dir
    #dval={}
    for filename in os.listdir(ingenome):
        pre=sample_id_from_filename(filename)
        if pre in train:
            dtrain[pre]=ingenome+'/'+filename
        '''
        elif pre in val:
            dval[pre]=ingenome+'/'+filename
        '''
    dl={}
    arr_train=[]
    #arr_val=[]
    f=open(label,'r')
    line=f.readline().strip()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        dl[ele[0]]=ele[1]
        if ele[0] in train:
            arr_train.append(ele[0])
        '''
        elif ele[0] in val:
            arr_val.append(ele[0])
        '''
    for a in arr_train:
        o.write(a+'\t'+dtrain[a]+'\t'+dl[a]+'\n')
    o.close()
    #exit()
    
    if kmer_tool=='builtin':
        run_kmer_selection(drug,work_dir,kmer_weights,threads)
    else:
        print(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker modeling tem.pheno',flush=True)
        ps_cmd=script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker modeling tem.pheno'
        if threads and int(threads)>1:
            ps_cmd+=' --num_threads '+str(int(threads))
        rc=os.system(ps_cmd)
        if rc!=0:
            raise RuntimeError('PhenotypeSeeker modeling failed (exit code %d)'%rc)
    #os.system('phenotypeseeker modeling tem.pheno')
    #print(script_dir+'/PhenotypeSeeker/phenotypeseeker modeling tem.pheno',flush=True)
    '''
    o2=open('ps_inf1.txt','w+')
    o3=open('ps_inf2.txt','w+')
    for a in arr_val:
        o2.write(a+'\t'+dval[a]+'\n')
    o2.close()
    o3.write(drug+'\tlog_reg_model_'+drug+'.pkl')
    o3.close()
    os.system(script_dir+'/PhenotypeSeeker/phenotypeseeker prediction ps_inf1.txt ps_inf2.txt')
    '''
    #exit()
    build_dir(work_dir+'/PS_out')
    if not os.path.exists(drug+'_MLdf.csv'):
        raise RuntimeError('PhenotypeSeeker produced no '+drug+'_MLdf.csv; the k-mer tokens cannot be built')
    generate_tps(drug+'_MLdf.csv',label,work_dir+'/strains_train_kmer_token.txt','tem_token_id.txt',train)
    #exit()
    #generate_tpsp('tem_token_id.txt','K-mer_lists',label,work_dir+'/strains_val_kmer_token.txt',val)
    # One "mv" over several globs failed as a whole if any one of them matched nothing, and os.system's
    # exit code was never checked, so missing outputs went unnoticed.  Move whatever exists, per pattern.
    import glob as _glob
    for pattern in ('tem.pheno','k-mers_and_coefficients*'+drug+'*','log*'+drug+'*','chi2*'+drug+'*',
                    drug+'_MLdf.csv','summary*'+drug+'*'):
        for src in _glob.glob(pattern):
            shutil.move(src,os.path.join(work_dir+'/PS_out',os.path.basename(src)))
    #os.system('rm tem.pheno')
    if os.path.exists('tem_token_id.txt'):
        shutil.move('tem_token_id.txt',work_dir+'/kmer_token_id.txt')
    # PhenotypeSeeker leaves one k-mer list per genome in K-mer_lists/ (tens of GB for a large dataset).
    # The feature matrix has been consumed by now, so it is only worth keeping to re-run this step.
    if os.path.isdir('K-mer_lists') and not keep_kmer_lists:
        size=sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk('K-mer_lists') for f in fs)
        shutil.rmtree('K-mer_lists',ignore_errors=True)
        log('[k-mer] removed K-mer_lists/ (%.1f GB); pass --keep-kmer-lists to keep it'%(size/1e9),
              flush=True)

def run_kmer_selection(drug,work_dir,use_weights=True,threads=1,k=13,n_kmers=1000):
    """Select phenotype-associated k-mers without PhenotypeSeeker.

    Reproduces PhenotypeSeeker's selection exactly (same weighted chi-square, same filters, same
    Bonferroni denominator) but counts k-mers in memory instead of writing one k-mer list per genome,
    which for a large dataset is tens of GB of intermediate files.  See kmer_select.py.

    Sample weighting is ON by default here: without it the selection is driven by whichever clone is
    most numerous in the training set.  PhenotypeSeeker leaves it off unless -w is passed (and its -w
    attaches the weights to the wrong genomes -- see kmer_select.gsc_weights).  Pass
    --no-kmer-weights to reproduce the old, uncorrected behaviour.
    """
    import kmer_select as KS
    samples=KS.read_pheno('tem.pheno')
    log(f'[k-mer] selecting phenotype-associated {k}-mers from {len(samples)} genomes',flush=True)
    if use_weights:
        w=KS.gsc_weights(samples,workdir=work_dir)
        if w:
            samples=[(n,p,y,w.get(n,1.0)) for n,p,y,_ in samples]
            log(f'[k-mer] GSC sample weights from mash distances: '
                  f'{min(w.values()):.3f} - {max(w.values()):.3f} (mean 1)',flush=True)
        else:
            log('[k-mer] WARNING: sample weighting unavailable, selection is uncorrected for '
                  'population structure',flush=True)
    sel=KS.select(samples,k=k,n_kmers=n_kmers,threads=threads)
    if not sel:
        raise RuntimeError('no k-mer passed the selection filters for '+drug)
    kmers=[x[0] for x in sel]
    KS.write_mldf(samples,kmers,k,drug+'_MLdf.csv',threads=threads)
    with open(work_dir+'/kmer_selection_'+drug+'.tsv','w') as o:
        o.write('kmer\tchi2\tpvalue\tn_carriers\n')
        for km,c,pv,ncar in sel:
            o.write(f'{km}\t{c:.2f}\t{pv:.2E}\t{ncar}\n')
    log(f'[k-mer] {len(kmers)} k-mers selected -> {drug}_MLdf.csv',flush=True)


def cal_len(infile1):
    f=open(infile1,'r')
    line=f.readline()
    ms=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        tokens_field = ele[-1].strip() if ele else ''
        if tokens_field:
            token_count = len(tokens_field.split(','))
        else:
            token_count = 0
        if token_count>ms:
            ms=token_count
    '''
    f=open(infile2,'r')
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        if int(ele[-2])>ms:
            ms=int(ele[-2])
    '''
    return ms

def scan_length(odir):
    o=open(odir+'/longest_len.txt','w+')
    o.write('Graph\tPC\tKmer\n')
    #for filename in os.listdir(odir):
    #if not re.search('Fold',filename):continue
    ls1=cal_len(odir+'/strains_train_sentence.txt')
    ls2=cal_len(odir+'/strains_train_pc_token.txt')
    ls3=cal_len(odir+'/strains_train_kmer_token.txt')
    o.write(str(ls1)+'\t'+str(ls2)+'\t'+str(ls3)+'\n')
    


def shap_rank(infile, ofile, mapping_files, rgi_dir=None):
    """SHAP feature ranking for one channel, without letting it abort the whole build.

    It is the last step of a run that has already spent hours on gene calling, clustering and k-mer
    selection, and its only products are the *_shap_filter.txt tables that `interpret` reads. A
    channel that cannot be explained is reported and skipped; the database is still written.
    """
    try:
        shap_feature_select_withcls.shap_select(infile, ofile, mapping_files, rgi_dir=rgi_dir)
        return True
    except Exception as exc:                    # noqa: BLE001 - reported, not swallowed
        log('WARNING: SHAP ranking failed for %s: %s: %s' % (os.path.basename(infile),
            type(exc).__name__, str(exc).split(chr(10))[0][:200]), flush=True)
        log('         the database is still usable for training and prediction; only the SHAP '
            'tables for this channel, and `interpret` on it, are missing.', flush=True)
        return False


def run(ingenome,label,odir,drug,pc_c,snv_c,kmer_c,mfile,threads=1,feature_limit=None,sentence_limit=None,
        cluster_tool='mmseqs',memory_gb=0,keep_kmer_lists=False,kmer_tool='builtin',
        kmer_weights=True,gene_caller='pyrodigal',rgi_batch=0,rgi_extra='',
        mechanism=True,amr_organism=None,amrfinder_mode='protein',amrfinder_db=None,
        mech_min_carriers=None,mech_min_ppv=None):
    import mechanism_layer as ML
    dr = index_genomes(ingenome)
    labeled = set(validate_label_genomes(label, dr))
    dr = {g: p for g, p in dr.items() if g in labeled}   # skip unlabelled extras
    check_tools(gene_caller, cluster_tool, kmer_tool, kmer_weights, snv_c, pc_c, kmer_c)
    if mechanism:
        # checked before any genome is touched: a missing tool or database must not surface hours in
        if not amr_organism:
            raise SystemExit("The known-mechanism layer needs --amr-organism, the AMRFinderPlus "
                             "--organism for point mutations (e.g. Klebsiella_pneumoniae, Escherichia, "
                             "Staphylococcus_aureus; list them with `amrfinder -l`). Pass "
                             "--amr-organism none to screen acquired genes only, or --no-mechanism "
                             "to build without the layer.")
        _problems = ML.check_amrfinder(amr_organism, db=amrfinder_db)
        if _problems:
            raise SystemExit('\n'.join(['Known-mechanism layer cannot run:'] + _problems))
    log(f"StrainAMR build-train: {len(dr)} genomes, drug '{drug}', {threads} threads -> {odir}")
    with step('gene calling + RGI', f'{len(dr)} genomes'):
        gdir,pdir=run_prodigal_rgi(dr,odir,threads,gene_caller,rgi_batch,rgi_extra)
    gdir=odir+'/Genes'
    pdir=odir+'/Proteins'
    argn=stat_rig(odir+'/RGI_raw')
    #print(argn)
    #exit()
    '''
    if argn<21:
        build_dir(odir+'/RGI')
        os.system('cp '+odir+'/RGI_raw/* '+odir+'/RGI')
    '''
    with step('filtering RGI hits to the drug class'):
        filter_rgi(odir+'/RGI_raw',drug,mfile,odir+'/RGI')
    mech = {'enabled': False}
    if mechanism:
        mech = {'enabled': True, 'organism': amr_organism, 'mode': amrfinder_mode,
                'amrfinder_db': amrfinder_db,
                'min_carriers': ML.DEFAULT_MIN_CARRIERS if mech_min_carriers is None else mech_min_carriers,
                'min_ppv': ML.DEFAULT_MIN_PPV if mech_min_ppv is None else mech_min_ppv}
        mech['amrfinder_version'], mech['database_version'] = ML.amrfinder_versions(db=amrfinder_db)
        try:
            with step('known mechanisms: saving CARD/RGI hits'):
                ML.dump_card_hits(odir+'/RGI_raw', odir, sorted(dr), gene_dir=odir+'/Genes')
        except Exception as exc:                # noqa: BLE001 - CARD is the secondary source
            log('WARNING: could not save the CARD/RGI hits (%s: %s); the layer will use '
                'AMRFinderPlus only' % (type(exc).__name__, str(exc)[:200]), flush=True)
        try:
            with step(f'known mechanisms: AMRFinderPlus ({amrfinder_mode}, organism {amr_organism})'):
                ML.screen(odir, sorted(dr), amr_organism, threads, amrfinder_db, amrfinder_mode,
                          protein_dir=odir+'/Proteins', genome_files=dr)
            with step('known mechanisms: calibration on the training labels'):
                ML.calibrate(odir, drug, label, mech['min_carriers'], mech['min_ppv'],
                             settings={k: mech[k] for k in ('organism', 'mode', 'amrfinder_version',
                                                            'database_version')})
        except Exception as exc:                # noqa: BLE001 - reported; the features still get built
            mech['enabled'] = False
            mech['error'] = f'{type(exc).__name__}: {exc}'[:500]
            log('WARNING: known-mechanism layer failed: ' + mech['error'], flush=True)
            log('         features are still built; rerun just this part with `strainamr '
                'mechanism-calibrate --db ' + odir + ' --run-amrfinder` BEFORE slim-db removes '
                'Proteins/.', flush=True)
    #exit()
    f=open(label,'r')
    line=f.readline()
    x=[]
    y=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        #print(ele)
        #exit()
        #ele[0]=re.split('\.',ele[0])[0]
        x.append(ele[0])
        y.append(ele[1])
    x=np.array(x)
    y=np.array(y)
    #splits=StratifiedKFold(n_splits=3,shuffle=True,random_state=1234)
    #datasets=splits.split(x,y)
    c=1
    #fold_arr=[] # for sef arr
    #for train_idx,val_idx in datasets:
    if True:

        #print(len(y[train_idx]),len(y[val_idx]))
        #exit()
        train=x
        #print(train)
        #exit()
        #val=x[val_idx]
        ########### Preprocess ###########
        work_dir=odir
        build_dir(work_dir)
        tem_gt=work_dir+'/genes_train'
        tem_pt=work_dir+'/proteins_train'
        tem_rt=work_dir+'/rgi_train'
        build_dir(tem_gt)
        build_dir(tem_pt)
        build_dir(tem_rt)
        '''
        tem_gv=work_dir+'/genes_val'
        tem_pv=work_dir+'/proteins_val'
        tem_rv=work_dir+'/rgi_val'
        build_dir(tem_gv)
        build_dir(tem_pv)
        build_dir(tem_rv)
        '''
           
        
        with step('linking genes, proteins and RGI output into the work directory'):
            copy_genome(odir+'/Genes',train,tem_gt,'gene')
            copy_genome(odir+'/RGI',train,tem_rt,'rgi')
            copy_protein(odir+'/Proteins',train,tem_pt)
        
        
        ########### Graph-based tokens ###########
        if not snv_c==1:
            gt=work_dir+'/Genomes_train'
            graph=work_dir+'/GFA_train_Minimap2'
            with step('graph tokens: extracting resistance-gene sequences'):
                extract(tem_rt,tem_gt,gt)
            with step('graph tokens: building variation graphs (minimap2 + seqwish)'):
                build(gt,graph,threads)
            with step('graph tokens: reading tokens off the graphs'):
                generate_fg(graph,tem_gt,work_dir+'/strains_train_sentence.txt',work_dir+'/node_token_match.txt',label,train)
        else:
            skipped('graph tokens')

        #generate_at(label,align_res,work_dir+'/node_token_match.txt',tem_gv,work_dir+'/strains_val_sentence.txt',val)
        #c+=1
        #continue
        #exit()
         
        ############### PC tokens ############
        if not pc_c==1:
            with step('PC tokens: merging and deduplicating proteins'):
                ptrain=merge_all_proteins(tem_pt,work_dir,'train')
            with step(f'PC tokens: clustering at 90% identity ({cluster_tool})'):
                run_cdhit(ptrain,work_dir,threads=threads,tool=cluster_tool,memory_gb=memory_gb)
            with step('PC tokens: writing the token files'):
                generate_tokens_from_cdhit(work_dir,label,train)
        else:
            skipped('PC tokens')
        
        ############### K-mer tokens ##########
        if not kmer_c==1:
            with step(f'k-mer tokens ({kmer_tool})'):
                run_ps(train,ingenome,label,drug,work_dir,threads,keep_kmer_lists,kmer_tool,kmer_weights)
        else:
            skipped('k-mer tokens')
    
        with step('chi-square feature selection'):
            sef(
                work_dir+'/strains_train_sentence.txt',
                work_dir+'/feature_remain_graph.txt',
                work_dir+'/strains_train_sentence_fs.txt',
                feature_limit,
                sentence_limit,
                mapping_files=[work_dir+'/node_token_match.txt'],
                rgi_dir=work_dir+'/rgi_train'
            )
            sef(
                work_dir+'/strains_train_pc_token.txt',
                work_dir+'/feature_remain_pc.txt',
                work_dir+'/strains_train_pc_token_fs.txt',
                feature_limit,
                sentence_limit,
                mapping_files=[work_dir+'/pc_matches.txt']
            )

    shap_dir = odir + '/shap'
    build_dir(shap_dir)
    with step('SHAP feature ranking'):
        ok = [shap_rank(work_dir+'/strains_train_sentence_fs.txt',
                        shap_dir+'/strains_train_sentence_fs_shap_filter.txt',
                        [work_dir+'/node_token_match.txt'], rgi_dir=work_dir+'/rgi_train'),
              shap_rank(work_dir+'/strains_train_pc_token_fs.txt',
                        shap_dir+'/strains_train_pc_token_fs_shap_filter.txt',
                        [work_dir+'/pc_matches.txt']),
              shap_rank(work_dir+'/strains_train_kmer_token.txt',
                        shap_dir+'/strains_train_kmer_token_shap_filter.txt',
                        [work_dir+'/kmer_token_id.txt'])]
        if not all(ok):
            log('[SHAP] %d of 3 channels ranked' % sum(ok), flush=True)

    # Record the options that decide the token-vector widths. build-test has to apply exactly the
    # same limits, and requiring the user to repeat them by hand is how the two sides drift apart.
    import json as _json
    with open(os.path.join(odir, 'build_params.json'), 'w') as _fh:
        # pc_tokens is the size of the PC vocabulary this database defines.  build-test recomputes it
        # from the same cluster file and refuses to continue if it differs, because a mismatch means
        # the two sides number the clusters differently and every PC token would name the wrong one.
        _pc_clstr = os.path.join(odir, PC.CLUSTER_FILE)
        _pc_n = len(set(PC.cluster_tokens(_pc_clstr).values())) if os.path.exists(_pc_clstr) else None
        _json.dump({'drug': drug, 'feature_limit': feature_limit, 'sentence_limit': sentence_limit,
                    'cluster_tool': cluster_tool, 'kmer_tool': kmer_tool,
                    'kmer_weights': bool(kmer_weights), 'gene_caller': gene_caller,
                    'rgi_extra': rgi_extra or '', 'pc_tokens': _pc_n, 'mechanism': mech},
                   _fh, indent=1)
    with step('token length statistics'):
        scan_length(odir)
        scan_length_fs(odir)
        scan_length_fs_shap(shap_dir)
    shutil.copy(label, odir+'/train_label.txt')
    summary(odir, 'build-train step timings')

def main():
    usage="StrainAMR_build_train - Takes strain genomes (training sets) as input and extracts graph-based, pc-based, k-mer-based features for antimicrobial resistance prediction."
    parser=argparse.ArgumentParser(prog=(sys.argv[0] if sys.argv[0].startswith("strainamr") else "StrainAMR_build_train.py"),description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input strain genomes (training set).")
    parser.add_argument('-l','--label_file',dest='lab_file',type=str,help="The directory of the input label files.")
    parser.add_argument('-d','--drug',dest='drug_name',type=str,help="The name of the predicted drug.")
    parser.add_argument('-p','--pc',dest='close_pc',type=int,help="If set to 1, then will skip pc tokens generation step. (Defaut: 0)" ,default=0)
    parser.add_argument('-s','--snv',dest='close_snv',type=int,help="If set to 1, then will skip snv tokens generation step. (Default: 0)",default=0)
    parser.add_argument('-k','--kmer',dest='close_kmer',type=int,help="If set to 1, then will skip k-mer tokens generation step. (Default:0)",default=0)
    parser.add_argument('-t','--threads',dest='threads',type=int,help="Number of parallel processes. (Default:1)",default=1)
    parser.add_argument(
        '--feature-limit',
        dest='feature_limit',
        type=int,
        help="Maximum number of features to keep in feature_remain files (default: keep all).",
        default=None
    )
    parser.add_argument(
        '--sentence-limit',
        dest='sentence_limit',
        type=int,
        help="Maximum number of tokens per strain in *_sentence_fs.txt outputs (default: keep all).",
        default=None
    )

    parser.add_argument(
        '--cluster-tool',
        dest='cluster_tool',
        choices=['mmseqs','cd-hit'],
        default='mmseqs',
        help="Tool used to cluster proteins into PC tokens at 90%% identity. mmseqs (default) is much "
             "faster and lighter on large protein sets; cd-hit reproduces the original behaviour."
    )
    parser.add_argument(
        '--memory',
        dest='memory_gb',
        type=int,
        default=0,
        help="Memory budget in GB for the clustering step (default: unlimited, as before)."
    )
    parser.add_argument(
        '--rgi-batch',
        dest='rgi_batch',
        type=int,
        default=0,
        help="Sequences per RGI run. RGI is run over all genomes at once by default, on each distinct "
             "gene sequence only (identical genes cannot get different RGI hits), instead of once per "
             "genome. Pass 1 to restore the original one-run-per-genome behaviour, or a number to set "
             "the batch size."
    )
    parser.add_argument(
        '--rgi-extra',
        dest='rgi_extra',
        type=str,
        default='',
        help="Extra options passed straight to `rgi main`, e.g. '-a DIAMOND'. RGI defaults to BLAST, "
             "which is much slower on a large search; DIAMOND is far quicker but its hits are not "
             "identical, so switch only if you are prepared to rebuild everything with it."
    )
    parser.add_argument(
        '--gene-caller',
        dest='gene_caller',
        choices=['pyrodigal','prodigal'],
        default='pyrodigal',
        help="Gene caller. pyrodigal (default) runs prodigal's algorithm in-process, without a "
             "subprocess per genome; prodigal calls the external binary."
    )
    parser.add_argument(
        '--kmer-tool',
        dest='kmer_tool',
        choices=['builtin','phenotypeseeker'],
        default='builtin',
        help="How phenotype-associated k-mers are selected. 'builtin' (default) reproduces "
             "PhenotypeSeeker's selection without writing per-genome k-mer lists to disk; "
             "'phenotypeseeker' calls the original tool."
    )
    parser.add_argument(
        '--no-kmer-weights',
        dest='kmer_weights',
        action='store_false',
        help="Do not weight genomes by how closely related they are when selecting k-mers. Weighting "
             "is on by default so that one over-represented clone cannot decide the selection; turn it "
             "off to reproduce the behaviour of earlier StrainAMR runs."
    )
    parser.add_argument(
        '--keep-kmer-lists',
        dest='keep_kmer_lists',
        action='store_true',
        help="Keep PhenotypeSeeker's K-mer_lists/ directory. It holds one k-mer list per genome (tens of "
             "GB for a large dataset) and is only needed to re-run the k-mer step, so it is deleted by "
             "default once the feature matrix has been written."
    )
    parser.add_argument(
        '--amr-organism',
        dest='amr_organism',
        default=None,
        help="AMRFinderPlus --organism for the known-mechanism layer, which enables its curated point "
             "mutations (e.g. Klebsiella_pneumoniae, Escherichia, Staphylococcus_aureus, "
             "Acinetobacter_baumannii, Clostridioides_difficile; `amrfinder -l` lists them). Pass "
             "'none' to screen acquired genes only. Required unless --no-mechanism."
    )
    parser.add_argument(
        '--no-mechanism',
        dest='mechanism',
        action='store_false',
        help="Build without the known-mechanism layer (no AMRFinderPlus needed)."
    )
    parser.add_argument(
        '--amrfinder-mode',
        dest='amrfinder_mode',
        choices=['protein', 'nucleotide'],
        default='protein',
        help="protein (default): screen the called proteins, run once over the distinct sequences "
             "of all genomes. nucleotide: screen whole assemblies per genome, which adds rRNA "
             "mutations and internal stop codons but is much slower."
    )
    parser.add_argument('--amrfinder-db', dest='amrfinder_db', default=None,
                        help="AMRFinderPlus database directory (default: the one `amrfinder -u` installed).")
    parser.add_argument('--mech-min-carriers', dest='mech_min_carriers', type=int, default=None,
                        help="Training genomes that must carry a determinant before the training data "
                             "judge it (default 5); rarer ones are trusted as curated.")
    parser.add_argument('--mech-min-ppv', dest='mech_min_ppv', type=float, default=None,
                        help="Resistant fraction among a determinant's training carriers needed to "
                             "count it (default 0.8).")
    #parser.add_argument('-m','--mfile',dest='map_file',type=str,help="The directory of the mapping file about the drug and its class.")
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of results. (Default: StrainAMR_res)")
    args=parser.parse_args()
    infile=args.input_file
    lab_file=args.lab_file
    drug=args.drug_name
    pc_c=args.close_pc
    snv_c=args.close_snv
    kmer_c=args.close_kmer
    #print(pc_c)
    #exit()
    mfile=script_dir+'/drug_to_class.txt'
    out=args.outdir
    if not out:
        out='StrainAMR_res'

    #run('/computenodes/node35/team3/herui/AMR_data/Phenotype_Seeker_data/Ref_Genome','cdi_label.txt','Cdi_3fold','azithromycin','drug_to_class.txt')
    threads=args.threads
    feature_limit=args.feature_limit
    if feature_limit is not None and feature_limit <= 0:
        feature_limit=None
    sentence_limit=args.sentence_limit
    if sentence_limit is not None and sentence_limit <= 0:
        sentence_limit=None
    run(infile,lab_file,out,drug,pc_c,snv_c,kmer_c,mfile,threads,feature_limit,sentence_limit,
        args.cluster_tool,args.memory_gb,args.keep_kmer_lists,args.kmer_tool,args.kmer_weights,
        args.gene_caller,args.rgi_batch,args.rgi_extra,
        args.mechanism,args.amr_organism,args.amrfinder_mode,args.amrfinder_db,
        args.mech_min_carriers,args.mech_min_ppv)

if __name__=="__main__":
    sys.exit(main())
    
