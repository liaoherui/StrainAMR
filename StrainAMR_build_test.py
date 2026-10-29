import re
import os
import sys
import argparse
import numpy as np
import subprocess
import multiprocessing
from extract_seq_for_graph import extract
from build_graph_batch_minimap2 import build
from align_genome_to_graph import align
from generate_token_from_graph import generate_fg
from generate_token_from_alignment import generate_at
from generate_token_from_ps import generate_tps
from generate_token_from_ps_predict import generate_tpsp
from feature_selection_sp_test import sef_test
from cal_length_test_fs import scan_length_fs,scan_length_fs_shap
from steplog import log, step, summary
import pc_clusters as PC
from sample_ids import index_genomes, sample_id_from_filename, validate_label_genomes
script_dir = os.path.dirname(os.path.abspath(__file__))

def inherit_build_settings(params, gene_caller=None, rgi_extra=None, mechanism=None, amrfinder_db=None,
                           check=True):
    """Resolve the annotation settings for build-test from the training database's build_params.

    Gene calling, RGI options and the known-mechanism layer must match the database, otherwise the
    same gene is called or annotated differently on the two sides.  None means "take the database's
    value"; an explicit value that differs is honoured but warned about.
    Returns (gene_caller, rgi_extra, mechanism, amrfinder_db, mechanism_settings).
    """
    import mechanism_layer as ML
    params = params or {}
    resolved = {}
    for name, val, default in (('gene_caller', gene_caller, 'pyrodigal'), ('rgi_extra', rgi_extra, '')):
        train_val = params.get(name)
        if val is None:
            val = train_val if train_val is not None else default
            if train_val is not None:
                log(f"using {name.replace('_', '-')} '{val}' from the training database")
            elif params:
                log(f"WARNING: the training database does not record {name.replace('_', '-')}; "
                    f"assuming '{val}'")
        elif train_val is not None and val != train_val:
            log(f"WARNING: --{name.replace('_', '-')} '{val}' differs from the '{train_val}' the "
                f"training database was built with; the same gene may be annotated differently")
        resolved[name] = val
    mech = dict(params.get('mechanism') or {})
    if mechanism is None:
        mechanism = bool(mech.get('enabled'))
    elif mechanism and not mech.get('enabled'):
        log('WARNING: the training database has no known-mechanism layer; skipping it here too')
        mechanism = False
    if mechanism:
        amrfinder_db = amrfinder_db or mech.get('amrfinder_db')
        if check:
            problems = ML.check_amrfinder(mech.get('organism'), db=amrfinder_db)
            if problems:
                raise SystemExit('\n'.join(['The training database uses the known-mechanism layer, '
                                            'but AMRFinderPlus cannot run here:'] + problems
                                           + ['  (or pass --no-mechanism for model-only predictions)']))
            _sw, dbv = ML.amrfinder_versions(db=amrfinder_db)
            if mech.get('database_version') and dbv != mech.get('database_version'):
                log(f"WARNING: AMRFinderPlus database {dbv} differs from the "
                    f"{mech['database_version']} the training database was calibrated with; "
                    f"determinant names can change between releases. Use the same database "
                    f"(--amrfinder-db) for comparable results.")
    return resolved['gene_caller'], resolved['rgi_extra'], mechanism, amrfinder_db, mech


def build_dir(idir):
    if not os.path.exists(idir):
        os.makedirs(idir)


def _prodigal_rgi_worker(args):
    s, dr, gdir, ginfo, pdir, rgi, rgi_threads, gene_caller, rgi_extra = args
    proc = multiprocessing.current_process()
    log(f"[Prodigal/RGI] start {s} -- {proc.name}", flush=True)
    if os.path.exists(rgi + '/' + s + '.txt') and os.path.getsize(rgi + '/' + s + '.txt') != 0:
        log(f"[Prodigal/RGI] skip {s} -- {proc.name}", flush=True)
        return s, 0
    if not os.path.exists(pdir + '/' + s + '.faa') or os.path.getsize(pdir + '/' + s + '.faa') == 0:
        if gene_caller == 'pyrodigal':
            from StrainAMR_build_train import call_genes_pyrodigal
            try:
                call_genes_pyrodigal(dr[s], gdir + '/' + s + '.fa', pdir + '/' + s + '.faa',
                                     ginfo + '/' + s + '.genes')
            except Exception as exc:                       # noqa: BLE001 - reported per genome below
                log(f"[Prodigal/RGI] pyrodigal failed on {s}: {exc}", flush=True)
                return s, 1
        else:
            rc = os.system('prodigal -i ' + dr[s] + ' -o ' + ginfo + '/' + s + '.genes -d ' + gdir + '/'
                           + s + '.fa -a ' + pdir + '/' + s + '.faa')
            if rc != 0:
                return s, rc
    if rgi_threads:                      # 0 means RGI runs later, batched over all genomes at once
        rc = os.system('rgi main --input_sequence ' + gdir + '/' + s + '.fa --output_file ' + rgi + '/'
                       + s + ' --local --clean  -n ' + str(rgi_threads)
                       + (' ' + rgi_extra if rgi_extra else ''))
    else:
        rc = 0
    log(f"[Prodigal/RGI] done {s} -- {proc.name}", flush=True)
    return s, rc


def run_prodigal_rgi(dr, odir, threads=1, gene_caller='pyrodigal', rgi_batch=0, rgi_extra=''):
    gdir=odir+'/Genes_ts'
    ginfo=odir+'/Genes_info_ts'
    pdir=odir+'/Proteins_ts'
    rgi=odir+'/RGI_raw_ts'

    build_dir(gdir)
    build_dir(ginfo)
    build_dir(pdir)
    build_dir(rgi)

    # spend one thread budget instead of (processes x hard-coded 10) threads, and surface failures:
    # apply_async without collecting results let a crashed genome pass silently.
    threads=max(1,int(threads))
    workers=min(threads,max(1,len(dr)))
    batched = rgi_batch != 1
    rgi_threads=0 if batched else max(1,threads//workers)
    log(f"[Prodigal/RGI] {len(dr)} genomes, {workers} parallel processes"
          + ("; RGI runs afterwards over all of them at once" if batched
             else f" x {rgi_threads} RGI threads"),flush=True)
    params = [(s, dr, gdir, ginfo, pdir, rgi, rgi_threads, gene_caller, rgi_extra) for s in dr]
    pool = multiprocessing.Pool(processes=workers)
    failed=[s for s,rc in pool.map(_prodigal_rgi_worker, params) if rc]
    pool.close()
    pool.join()
    if failed:
        raise RuntimeError('prodigal/RGI failed for %d genome(s): %s%s'
                           %(len(failed),', '.join(failed[:10]),' ...' if len(failed)>10 else ''))
    if batched:
        import rgi_batch as RB
        from StrainAMR_build_train import rgi_batch_size
        gene_files={s: gdir+'/'+s+'.fa' for s in dr if os.path.exists(gdir+'/'+s+'.fa')}
        RB.run_rgi(gene_files, rgi, threads, per_batch=rgi_batch_size(rgi_batch, len(gene_files)),
                   extra=rgi_extra)
    return gdir,pdir

def copy_genome(gdir,odir,t):
    bfix=''
    if t=='gene':
        bfix='fa'
    else:
        bfix='txt'
    #for i in index:
    os.system('cp '+gdir+'/*.'+bfix+' '+odir)

def copy_protein(pdir,odir):
    #for i in index:
    os.system('cp '+pdir+'/*.faa '+odir)
     
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
    os.system('cat '+indir+'/* >'+odir+'/merged_proteins_'+t+'.fa')
    return odir+'/merged_proteins_'+t+'.fa'
    
def run_cdhit(pval,work_dir,train_merged_proteins):
    # Run CD-Hit on test proteins against the training representatives
    os.system('cd-hit-2d -i '+train_merged_proteins+' -i2 '+pval+' -d 0 -o '+work_dir+'/merge_test_cdhit -c 0.9 -n 5 -M 0 -T 0')
    cls2=work_dir+'/merge_test_cdhit.clstr'
    return cls2

def output_pc_token_file(d,pdir,label2,ofile):
    dr={} # Strain prefix -> Tokens string
    for filename in os.listdir(pdir):
        pre=os.path.splitext(filename)[0]
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
    f=open(label2,'r')
    o=open(ofile,'w+')
    line=f.readline().strip()
    o.write(line+'\tTokens_Num\tTokens\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        #if ele[0] not in idx:continue
        o.write(line+'\t')
        arr=re.split(',',dr[ele[0]])
        o.write(str(len(arr))+'\t'+dr[ele[0]]+'\n')
    o.close()

    

def generate_tokens_from_cdhit(train_cluster_file,work_dir,label2):
    # Same protein -> token mapping the training run produced, from the same code (pc_clusters).
    # This used to be a second copy of the parser here, and it lacked the duplicate-protein step:
    # a cluster whose members were all identical sequences counted as a singleton on this side only,
    # was dropped, and every token id after it was shifted by one.  The validation PC tokens then
    # named different clusters than the training vocabulary the model had learned.
    d=PC.cluster_tokens(train_cluster_file)
    log('[PC] '+PC.summarise(train_cluster_file),flush=True)
    if not d:
        raise RuntimeError('no PC tokens from '+train_cluster_file+': the training database looks '
                           'incomplete')
    # Guard against the two sides numbering the clusters differently, which is silent and ruins every
    # PC token: build-train recorded how many tokens this database defines.
    n_here=len(set(d.values()))
    _bp=os.path.join(os.path.dirname(os.path.abspath(train_cluster_file)),'build_params.json')
    if os.path.exists(_bp):
        import json as _json
        try:
            n_train=_json.load(open(_bp)).get('pc_tokens')
        except (ValueError,OSError):
            n_train=None
        if n_train is not None and n_train!=n_here:
            raise RuntimeError(
                'PC token vocabulary mismatch: the training database defines %d PC tokens but reading '
                '%s here gives %d. The token ids would name different protein clusters than the model '
                'was trained on. Rebuild the database with this version of StrainAMR.'
                %(n_train,train_cluster_file,n_here))

    # cd-hit-2d lists the matched TRAINING representative first, marked '*', then the test proteins
    # that joined it; each of those inherits the representative's token.
    d2={}
    tid='NA'
    unmapped=0
    with open(work_dir+'/merge_test_cdhit.clstr') as f2:
        for line in f2:
            line=line.rstrip('\n')
            if not line.strip() or re.search('Cluster',line):
                continue
            pre=re.sub(r'\.\.\.','',re.sub('>','',line.split()[2]))
            if line.rstrip().endswith('*'):
                tid=d.get(pre,'NA')
                if tid=='NA':
                    unmapped+=1
            elif tid!='NA':
                d2[pre]=tid
    log(f'[PC] {len(d2)} test proteins mapped to training clusters '
        f'({unmapped} clusters had no token)',flush=True)

    #output_pc_token_file(d,work_dir+'/proteins_train',label,work_dir+'/strains_train_pc_token.txt',train)
    output_pc_token_file(d2,work_dir+'/proteins_test',label2,work_dir+'/strains_test_pc_token.txt')
    

def run_ps(intest,label,label2,drug,work_dir,train_dir,kmer_tool='builtin',threads=1):
    #o=open('tem.pheno','w+') 
    #o.write('ID\tAddress\t'+drug+'\n')
    #dtrain={} # Pre -> Genome dir
    dval={}
    '''
    for filename in os.listdir(ingenome):
        pre=os.path.splitext(filename)[0]
        if pre in train:
            dtrain[pre]=ingenome+'/'+filename
        elif pre in val:
            dval[pre]=ingenome+'/'+filename
    '''
    for filename in os.listdir(intest):
        pre=sample_id_from_filename(filename)
        #if pre in val:
        dval[pre]=intest+'/'+filename
    dl={}
    #arr_train=[]
    arr_val=[]
    '''
    f=open(label,'r')
    line=f.readline().strip()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        dl[ele[0]]=ele[1]
        if ele[0] in train:
            arr_train.append(ele[0])
    '''
    f=open(label2,'r')
    line=f.readline().strip()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        dl[ele[0]]=ele[1]
        #if ele[0] in val:
        arr_val.append(ele[0])
    '''
    for a in arr_train:
        o.write(a+'\t'+dtrain[a]+'\t'+dl[a]+'\n')
    o.close()
    '''
    #exit()
            
    #os.system(script_dir+'/PhenotypeSeeker/phenotypeseeker modeling tem.pheno')
    build_dir(work_dir+'/PS_out')

    token_id_path = os.path.join(train_dir, 'kmer_token_id.txt')
    if not os.path.exists(token_id_path):
        raise FileNotFoundError(f"k-mer token id file not found: {token_id_path}")

    if kmer_tool=='builtin':
        # Look the training k-mers up directly in the test genomes. PhenotypeSeeker's `prediction`
        # only ever served to produce K-mer_lists/ for exactly this lookup, and it additionally
        # required a logistic-regression model file whose predictions StrainAMR discards.
        import kmer_select as KS
        kmers=[];tokens=[]
        for line in open(token_id_path):
            f=line.rstrip('\n').split('\t')
            if len(f)>=2:
                kmers.append(f[0]);tokens.append(f[1])
        samples=[(pre,dval[pre],0,1.0) for pre in arr_val if pre in dval]
        missing=[pre for pre in arr_val if pre not in dval]
        if missing:
            raise FileNotFoundError('no genome file for: '+', '.join(missing[:5])
                                    +(' ...' if len(missing)>5 else ''))
        log(f'[k-mer] looking up {len(kmers)} training k-mers in {len(samples)} test genomes',flush=True)
        KS.write_token_file(samples,kmers,len(kmers[0]) if kmers else 13,tokens,dl,
                            work_dir+'/strains_test_kmer_token.txt',threads=threads)
        log('[k-mer] wrote '+work_dir+'/strains_test_kmer_token.txt',flush=True)
        return

    train_ps_dir = os.path.join(train_dir, 'PS_out')
    model_path = os.path.join(train_ps_dir, f'log_reg_model_{drug}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PhenotypeSeeker model not found: {model_path}")

    o2=open('ps_inf1.txt','w+')
    o3=open('ps_inf2.txt','w+')
    for a in arr_val:
        o2.write(a+'\t'+dval[a]+'\n')
    o2.close()
    o3.write(drug+'\t'+model_path+'\n')
    o3.close()
    os.system('mv ps_inf1.txt ps_inf2.txt '+work_dir+'/PS_out')
    print(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker prediction '+work_dir+'/PS_out/ps_inf1.txt '+work_dir+'/PS_out/ps_inf2.txt',flush=True)
    os.system(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker prediction '+work_dir+'/PS_out/ps_inf1.txt '+work_dir+'/PS_out/ps_inf2.txt')
    #exit()
    #exit()
    #generate_tps(drug+'_MLdf.csv',label,work_dir+'/strains_train_kmer_token.txt','tem_token_id.txt',train)
    #exit()
    if os.path.exists(work_dir+'/PS_out/K-mer_lists'):
        os.system('rm -rf '+work_dir+'/PS_out/K-mer_lists')
    os.system('mv K-mer_lists '+work_dir+'/PS_out')
    generate_tpsp(token_id_path,work_dir+'/PS_out/K-mer_lists',label2,work_dir+'/strains_test_kmer_token.txt')
    #os.system('rm -rf K-mer_lists')
    #os.system('rm ps_inf1.txt ps_inf2.txt ')
    os.system('mv predictions_'+drug+'.txt '+work_dir+'/PS_out')
    #os.system('mv tem_token_id.txt '+work_dir+'/kmer_token_id.txt')

def cal_len(infile1,infile2):
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
    f=open(infile2,'r')
    line=f.readline()
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
    return ms

def scan_length(test_dir,train_dir):
    o=open(test_dir+'/longest_len_test.txt','w+')
    o.write('Graph\tPC\tKmer\n')
    #for filename in os.listdir(odir):
    #if not re.search('Fold',filename):continue
    ls1=cal_len(train_dir+'/strains_train_sentence.txt',test_dir+'/strains_test_sentence.txt')
    ls2=cal_len(train_dir+'/strains_train_pc_token.txt',test_dir+'/strains_test_pc_token.txt')
    ls3=cal_len(train_dir+'/strains_train_kmer_token.txt',test_dir+'/strains_test_kmer_token.txt')
    o.write(str(ls1)+'\t'+str(ls2)+'\t'+str(ls3)+'\n')
    


#def run(ingenome,label,odir,drug,mfile,intest,label2):
def run(intest,label2,odir,drug,pc_c,snv_c,kmer_c,mfile,threads=1,feature_limit=None,
        sentence_limit=None,train_dir=None,kmer_tool='builtin',gene_caller=None,
        rgi_batch=0,rgi_extra=None,mechanism=None,amrfinder_db=None):
    # gene_caller / rgi_extra / mechanism default to None = "whatever the training database used":
    # the two sides must annotate genomes identically, or the same gene gets different tokens.
    work_dir = os.path.abspath(odir)
    build_dir(work_dir)
    train_dir = os.path.abspath(train_dir if train_dir else work_dir)
    dr = index_genomes(intest)
    val = list(dr)
    if label2:
        label2 = os.path.abspath(label2)
        if not os.path.exists(label2):
            raise FileNotFoundError(f"Provided label file not found: {label2}")
        labeled = set(validate_label_genomes(label2, dr))
        dr = {g: p for g, p in dr.items() if g in labeled}   # skip unlabelled extras
        val = list(dr)
    label=os.path.join(train_dir,'train_label.txt')
    if not os.path.exists(label):
        raise FileNotFoundError(f"Training label file not found: {label}")

    shap_dir = work_dir + '/shap'
    build_dir(shap_dir)
    train_shap_dir = os.path.join(train_dir, 'shap')
    if not os.path.isdir(train_shap_dir):
        raise FileNotFoundError(f"Training SHAP directory not found: {train_shap_dir}")
    required_shap_files = [
        'strains_train_sentence_fs_shap_rmf.txt',
        'strains_train_pc_token_fs_shap_rmf.txt',
        'strains_train_kmer_token_shap_rmf.txt',
        'strains_train_sentence_fs_shap_filter.txt',
        'strains_train_pc_token_fs_shap_filter.txt',
        'strains_train_kmer_token_shap_filter.txt',
    ]
    for rel_path in required_shap_files:
        candidate = os.path.join(train_shap_dir, rel_path)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Required SHAP reference file missing: {candidate}")
    required_feature_files = [
        os.path.join(train_dir, 'feature_remain_graph.txt'),
        os.path.join(train_dir, 'feature_remain_pc.txt'),
        os.path.join(train_dir, 'strains_train_sentence.txt'),
        os.path.join(train_dir, 'strains_train_pc_token.txt'),
        os.path.join(train_dir, 'strains_train_kmer_token.txt'),
        os.path.join(train_dir, 'node_token_match.txt'),
        os.path.join(train_dir, 'pc_matches.txt'),
    ]
    for feature_path in required_feature_files:
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Required training artifact missing: {feature_path}")
    shap_sample_file = os.path.join(shap_dir, 'test_samples.tsv')
    with open(shap_sample_file, 'w') as shap_fh:
        shap_fh.write('Sample_ID\n')
    with open(shap_sample_file, 'a') as shap_fh:
        for sid in sorted(val):
            shap_fh.write(f"{sid}\n")

    provided_label_file = bool(label2)
    if not provided_label_file:
        placeholder_label = os.path.join(work_dir, 'test_labels_placeholder.txt')
        header = 'ID\tLabel'
        if os.path.exists(label):
            with open(label, 'r') as train_label_fh:
                train_header = train_label_fh.readline().strip()
                if train_header:
                    header = train_header
        with open(placeholder_label, 'w') as placeholder_fh:
            placeholder_fh.write(header + '\n')
            for sid in sorted(val):
                placeholder_fh.write(f"{sid}\t-1\n")
        label2 = placeholder_label
        print(f"No test label file supplied. Created placeholder labels at {label2}.", flush=True)
    # Inherit the feature/sentence limits the training database was built with, unless overridden:
    # they set the token-vector widths, and a mismatch between the two sides shows up much later as a
    # shape error during training or prediction.
    _p_loaded = {}                      # the training database's build_params.json, if any
    if train_dir:
        params_path = os.path.join(train_dir, 'build_params.json')
        if os.path.exists(params_path):
            import json as _json
            with open(params_path) as _fh:
                _p = _json.load(_fh)
            _p_loaded = _p
            for _name, _val in (('feature_limit', feature_limit), ('sentence_limit', sentence_limit)):
                _train_val = _p.get(_name)
                if _val is None and _train_val is not None:
                    if _name == 'feature_limit':
                        feature_limit = _train_val
                    else:
                        sentence_limit = _train_val
                    log(f"using --{_name.replace('_', '-')} {_train_val} from the training database")
                elif _val is not None and _train_val is not None and _val != _train_val:
                    log(f"WARNING: --{_name.replace('_', '-')} {_val} differs from the {_train_val} the "
                        f"training database was built with; the token widths will not match")
        else:
            # Release bundles may ship only the calibrated mechanism layer, without the training
            # database's build_params.json.  Recover the settings that matter (drug, organism,
            # carrier and PPV thresholds) from mechanism/calibration.json so that new genomes are
            # annotated exactly like the training genomes.
            _cal = os.path.join(train_dir, 'mechanism', 'calibration.json')
            if os.path.exists(_cal):
                import json as _json
                with open(_cal) as _fh:
                    _c = _json.load(_fh)
                _p_loaded = {
                    'drug': _c.get('drug'),
                    'gene_caller': 'pyrodigal',
                    'rgi_extra': '',
                    'mechanism': {
                        'enabled': bool(_c.get('available', True)),
                        'organism': _c.get('organism'),
                        'mode': _c.get('mode', 'protein'),
                        'amrfinder_db': None,
                        'min_carriers': _c.get('min_carriers', 5),
                        'min_ppv': _c.get('min_ppv', 0.8),
                    },
                }
                log(f"WARNING: {params_path} is missing; annotation settings rebuilt from {_cal}")
    gene_caller, rgi_extra, mechanism, amrfinder_db, _mech = inherit_build_settings(
        _p_loaded, gene_caller, rgi_extra, mechanism, amrfinder_db)
    import mechanism_layer as ML

    from StrainAMR_build_train import check_tools
    check_tools(gene_caller, 'cd-hit', kmer_tool, False, snv_c, pc_c, kmer_c)
    # Run prodigal and rgi for all input genomes
    print('Run Prodigal and RGI for all input genomes!',flush=True)
    log(f"StrainAMR build-test: {len(dr)} genomes, drug '{drug}', {threads} threads -> {work_dir}")
    with step('gene calling + RGI', f'{len(dr)} genomes'):
        gdir,pdir=run_prodigal_rgi(dr,work_dir,threads,gene_caller,rgi_batch,rgi_extra)
    gdir=work_dir+'/Genes_ts'
    pdir=work_dir+'/Proteins_ts'
    #exit()
    with step('filtering RGI hits to the drug class'):
        filter_rgi(work_dir+'/RGI_raw_ts',drug,mfile,work_dir+'/RGI_ts')
    if mechanism and os.path.abspath(work_dir) == os.path.abspath(train_dir):
        log('WARNING: build-test writes into the training database itself (-o equals --db); skipping '
            'the known-mechanism screening so the database\'s own results are not overwritten. Use a '
            'separate -o.')
        mechanism = False
    if mechanism:
        try:
            with step('known mechanisms: saving CARD/RGI hits'):
                ML.dump_card_hits(work_dir+'/RGI_raw_ts', work_dir, sorted(dr), gene_dir=work_dir+'/Genes_ts')
        except Exception as exc:                # noqa: BLE001 - CARD is the secondary source
            log('WARNING: could not save the CARD/RGI hits (%s: %s)' % (type(exc).__name__,
                str(exc)[:200]), flush=True)
        try:
            with step(f"known mechanisms: AMRFinderPlus ({_mech.get('mode', 'protein')}, organism "
                      f"{_mech.get('organism')})"):
                ML.screen(work_dir, sorted(dr), _mech.get('organism'), threads, amrfinder_db,
                          _mech.get('mode', 'protein'), protein_dir=work_dir+'/Proteins_ts',
                          genome_files=dr)
        except Exception as exc:                # noqa: BLE001 - reported; features still get built
            log('WARNING: known-mechanism screening failed: %s: %s'
                % (type(exc).__name__, str(exc)[:300]), flush=True)
            log('         predictions for these genomes will be model-only.', flush=True)
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
    #x=np.array(x)
    #y=np.array(y)
    #splits=StratifiedKFold(n_splits=3,shuffle=True,random_state=1234)
    #datasets=splits.split(x,y)
    #train,val_new,y_train,y_val=train_test_split(x, y, stratify=y, random_state=42)
    #print(val)
    #exit()
    #c=1
    #fold_arr=[] # for sef arr
    if True:
        #print('Fold '+str(c)+' starts!',flush=True)
        #print(len(y[train_idx]),len(y[val_idx]))
        #exit()
        #train=x[train_idx]
        #val=x[val_idx]
        ########### Preprocess ###########
        #build_dir(work_dir)
        #tem_gt=work_dir+'/genes_train'
        #tem_pt=work_dir+'/proteins_train'
        #tem_rt=work_dir+'/rgi_train'
        #build_dir(tem_gt)
        #build_dir(tem_pt)
        #build_dir(tem_rt)
        tem_gv=work_dir+'/genes_test'
        tem_pv=work_dir+'/proteins_test'
        tem_rv=work_dir+'/rgi_test'
        build_dir(tem_gv)
        build_dir(tem_pv)
        build_dir(tem_rv)
        
        
        copy_genome(work_dir+'/Genes_ts',tem_gv,'gene')
        copy_genome(work_dir+'/RGI_ts',tem_rv,'rgi')
        

        copy_protein(work_dir+'/Proteins_ts',tem_pv)
        
        
        ########### Graph-based tokens ###########
        
        #gt=work_dir+'/Genomes_train'
        if not snv_c==1:
            gv=work_dir+'/Genomes_test'
            extract(tem_rv,tem_gv,gv)

            graph=os.path.join(train_dir,'GFA_train_Minimap2')
            if not os.path.exists(graph):
                raise FileNotFoundError(f"Training GFA directory not found: {graph}")

            align_res=work_dir+'/Align_test_res'
            align(gv,graph,align_res)


            generate_at(label2,align_res,os.path.join(train_dir,'node_token_match.txt'),tem_gv,work_dir+'/strains_test_sentence.txt')
        
        ############### PC tokens ############
        #ptrain=merge_all_proteins(tem_pt,work_dir,'train')
        if not pc_c==1:
            pval=merge_all_proteins(tem_pv,work_dir,'test')
            train_merged_proteins = os.path.join(train_dir, 'merged_proteins_train.fa')
            if not os.path.exists(train_merged_proteins):
                raise FileNotFoundError(f"Merged training proteins not found: {train_merged_proteins}")
            train_cluster_file = os.path.join(train_dir, 'merge_train_cdhit.clstr')
            if not os.path.exists(train_cluster_file):
                raise FileNotFoundError(f"Training CD-HIT cluster file not found: {train_cluster_file}")

            with step('PC tokens: mapping test proteins to the training clusters'):
                run_cdhit(pval,work_dir,train_merged_proteins)
                generate_tokens_from_cdhit(train_cluster_file,work_dir,label2)
        
        ############### K-mer tokens ##########
        #print(os.path.abspath(work_dir))
        #exit()
        if not kmer_c==1:
            with step(f'k-mer tokens ({kmer_tool})'):
                run_ps(intest,label,label2,drug,os.path.abspath(work_dir),train_dir,kmer_tool,threads)
        #exit()
        
        #train=x[train_idx]
        #val=x[val_idx]
        sef_test(
            work_dir+'/strains_test_sentence.txt',
            os.path.join(train_dir,'feature_remain_graph.txt'),
            work_dir+'/strains_test_sentence_fs.txt',
            feature_limit,
            sentence_limit
        )
        sef_test(
            work_dir+'/strains_test_pc_token.txt',
            os.path.join(train_dir,'feature_remain_pc.txt'),
            work_dir+'/strains_test_pc_token_fs.txt',
            feature_limit,
            sentence_limit
        )
        ### For shap
        sef_test(
            work_dir+'/strains_test_sentence_fs.txt',
            os.path.join(train_shap_dir,'strains_train_sentence_fs_shap_rmf.txt'),
            shap_dir+'/strains_test_sentence_fs_shap_filter.txt',
            None,
            sentence_limit
        )
        sef_test(
            work_dir+'/strains_test_pc_token_fs.txt',
            os.path.join(train_shap_dir,'strains_train_pc_token_fs_shap_rmf.txt'),
            shap_dir+'/strains_test_pc_token_fs_shap_filter.txt',
            None,
            sentence_limit
        )
        sef_test(
            work_dir+'/strains_test_kmer_token.txt',
            os.path.join(train_shap_dir,'strains_train_kmer_token_shap_rmf.txt'),
            shap_dir+'/strains_test_kmer_token_shap_filter.txt'
        )

        #c+=1
    with step('token length statistics'):
        scan_length(work_dir,train_dir)
        scan_length_fs(work_dir,train_dir)
        scan_length_fs_shap(shap_dir,train_shap_dir)
    summary(work_dir, 'build-test step timings')


def main():
    usage="StrainAMR_build_test - Takes strain genomes (test sets) as input and extracts graph-based, pc-based, k-mer-based features for antimicrobial resistance prediction."
    parser=argparse.ArgumentParser(prog=(sys.argv[0] if sys.argv[0].startswith("strainamr") else "StrainAMR_build_test.py"),description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input strain genomes (test set).")
    parser.add_argument('-l','--label_file',dest='lab_file',type=str,help="The directory of the input label file for your test data. If omitted, placeholder labels will be generated automatically.")
    parser.add_argument('-d','--drug',dest='drug_name',type=str,help="The name of the predicted drug. (Note: The drug name of your input test data must match the drug name of your training data.)")
    parser.add_argument('-p','--pc',dest='close_pc',type=int,help="If set to 1, then will skip pc tokens generation step. (Defaut: 0)" ,default=0)
    parser.add_argument('-s','--snv',dest='close_snv',type=int,help="If set to 1, then will skip snv tokens generation step. (Default: 0)",default=0)
    parser.add_argument('-k','--kmer',dest='close_kmer',type=int,help="If set to 1, then will skip k-mer tokens generation step. (Default:0)",default=0)
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory for generated test features and reports.")
    parser.add_argument('--rgi-batch',dest='rgi_batch',type=int,default=0,
                        help="Sequences per RGI run. RGI runs over all genomes at once by default, on "
                             "each distinct gene sequence only. Pass 1 for the original one-run-per-"
                             "genome behaviour.")
    parser.add_argument('--gene-caller',dest='gene_caller',choices=['pyrodigal','prodigal'],
                        default=None,
                        help="Gene caller. Default: the one the training database was built with "
                             "(recorded in build_params.json), else pyrodigal.")
    parser.add_argument('--rgi-extra',dest='rgi_extra',type=str,default=None,
                        help="Extra options for `rgi main`. Default: the ones the training database "
                             "was built with, so both sides annotate genes the same way.")
    parser.add_argument('--no-mechanism',dest='mechanism',action='store_const',const=False,
                        default=None,
                        help="Skip AMRFinderPlus screening even if the training database has the "
                             "known-mechanism layer (predictions will then be model-only).")
    parser.add_argument('--amrfinder-db',dest='amrfinder_db',default=None,
                        help="AMRFinderPlus database directory (default: the training database's).")
    parser.add_argument('--kmer-tool',dest='kmer_tool',choices=['builtin','phenotypeseeker'],default='builtin',
                        help="How the training k-mers are looked up in the test genomes. 'builtin' "
                             "(default) reads them straight from the genomes; 'phenotypeseeker' calls "
                             "the original tool, which needs its k-mer lists and model file.")
    parser.add_argument('--db',dest='db_dir',type=str,help="Directory containing the pre-built training database (default: use the output directory).")
    parser.add_argument('-t','--threads',dest='threads',type=int,help="Number of parallel processes. (Default:1)",default=1)
    parser.add_argument(
        '--feature-limit',
        dest='feature_limit',
        type=int,
        help="Maximum number of features to keep when filtering with feature_remain files (default: keep all).",
        default=None
    )
    parser.add_argument(
        '--sentence-limit',
        dest='sentence_limit',
        type=int,
        help="Maximum number of tokens per strain in *_sentence_fs.txt outputs (default: keep all).",
        default=None
    )
    args=parser.parse_args()
    infile=args.input_file
    lab_file=args.lab_file
    drug=args.drug_name
    pc_c=args.close_pc
    snv_c=args.close_snv
    kmer_c=args.close_kmer
    mfile=script_dir+'/drug_to_class.txt'
    out=args.outdir
    threads=args.threads
    if not out:
        print('Please provide an output directory for the generated test data!')
        exit()
    out=os.path.abspath(out)
    db_dir=args.db_dir
    if db_dir:
        db_dir=os.path.abspath(db_dir)
        if not os.path.isdir(db_dir):
            raise FileNotFoundError(f"Training database directory not found: {db_dir}")
    else:
        db_dir=out
    feature_limit=args.feature_limit
    if feature_limit is not None and feature_limit <= 0:
        feature_limit=None
    sentence_limit=args.sentence_limit
    if sentence_limit is not None and sentence_limit <= 0:
        sentence_limit=None
    run(infile,lab_file,out,drug,pc_c,snv_c,kmer_c,mfile,threads,feature_limit,sentence_limit,
        train_dir=db_dir,kmer_tool=args.kmer_tool,gene_caller=args.gene_caller,
        rgi_batch=args.rgi_batch,rgi_extra=args.rgi_extra,mechanism=args.mechanism,
        amrfinder_db=args.amrfinder_db)

if __name__=="__main__":
    sys.exit(main())
#run('../../Sau/Ref_Genome','sau_label.txt','Sau_split','levofloxacin','drug_to_class.txt','../../Sau/Ref_Genome_extra_used','sau_label_test.txt')


#run('../../Ecoli/Ref_Genome','ecoli_label.txt','Ecoli_3fold','levofloxacin','drug_to_class.txt')
#run('../../Kcp/Ref_Genome','kcp_label.txt','Kcp_3fold','ceftazidime-avibactam','drug_to_class.txt')
    
