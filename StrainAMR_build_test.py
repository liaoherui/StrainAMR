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
from sample_ids import index_genomes, sample_id_from_filename, validate_label_genomes
script_dir = os.path.dirname(os.path.abspath(__file__))

def build_dir(idir):
    if not os.path.exists(idir):
        os.makedirs(idir)


def _prodigal_rgi_worker(args):
    s, dr, gdir, ginfo, pdir, rgi = args
    proc = multiprocessing.current_process()
    print(f"[Prodigal/RGI] start {s} -- {proc.name}", flush=True)
    if os.path.exists(rgi + '/' + s + '.txt') and os.path.getsize(rgi + '/' + s + '.txt') != 0:
        print(f"[Prodigal/RGI] skip {s} -- {proc.name}", flush=True)
        return
    if not os.path.exists(pdir + '/' + s + '.faa') or os.path.getsize(pdir + '/' + s + '.faa') == 0:
        os.system('prodigal -i ' + dr[s] + ' -o ' + ginfo + '/' + s + '.genes -d ' + gdir + '/' + s + '.fa -a ' + pdir + '/' + s + '.faa')
    os.system('rgi main --input_sequence ' + gdir + '/' + s + '.fa --output_file ' + rgi + '/' + s + ' --local --clean  -n 10')
    print(f"[Prodigal/RGI] done {s} -- {proc.name}", flush=True)


def run_prodigal_rgi(dr, odir, threads=1):
    gdir=odir+'/Genes_ts'
    ginfo=odir+'/Genes_info_ts'
    pdir=odir+'/Proteins_ts'
    rgi=odir+'/RGI_raw_ts'

    build_dir(gdir)
    build_dir(ginfo)
    build_dir(pdir)
    build_dir(rgi)

    params = [(s, dr, gdir, ginfo, pdir, rgi) for s in dr]
    pool = multiprocessing.Pool(processes=int(threads))
    for p in params:
        pool.apply_async(_prodigal_rgi_worker, (p,))
    pool.close()
    pool.join()
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
        contigs=[]
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
            if ct not in contigs:
                contigs.append(ct)
            if ct not in dc:
                dc[ct]=[]
            dc[ct].append(d[pid])
        '''
        if pre=='573_46831':
            print(contigs)
            print(dc)
            exit()
        '''
        tem=[] # all contig tokens
        for c in contigs:
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
    f=open(train_cluster_file,'r')
    #o=open(work_dir+'/pc_matches.txt','w+')
    dcls={} # Cls_ID -> proteins
    arr=[]
    while True:
        line=f.readline().strip()
        if not line:break
        if re.search('Cluster',line):
            cls=re.sub('>','',line)
            cls=re.sub(' ','_',cls)
            if cls not in arr:
                arr.append(cls)
            if cls not in dcls:
                dcls[cls]={}
        else:
            pre=line.split()[2]
            pre=re.sub('>','',pre)
            pre=re.sub('\.\.\.','',pre)
            dcls[cls][pre]=''
    d={} # proteins -> token_ID | without single cluster
    i=1
    for c in arr:
        if len(dcls[c])==1:continue
        for p in dcls[c]:
            d[p]=str(i)
            #o.write(str(i)+'\t'+p+'\n')
        i+=1
    f2=open(work_dir+'/merge_test_cdhit.clstr','r')
    d2={}  # For val samples: proteins -> token_ID | without single cluster
    while True:
        line=f2.readline().strip()
        if not line:break
        if re.search('Cluster',line):continue
        pre=line.split()[2]
        pre=re.sub('>','',pre)
        pre=re.sub('\.\.\.','',pre)
        if line[-1]=='*':
            if pre in d:
                tid=d[pre]
            else:
                tid='NA'
        else:
            if not tid=='NA':
                d2[pre]=tid

    #output_pc_token_file(d,work_dir+'/proteins_train',label,work_dir+'/strains_train_pc_token.txt',train)
    output_pc_token_file(d2,work_dir+'/proteins_test',label2,work_dir+'/strains_test_pc_token.txt')
    

def run_ps(intest,label,label2,drug,work_dir,train_dir):
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

    train_ps_dir = os.path.join(train_dir, 'PS_out')
    model_path = os.path.join(train_ps_dir, f'log_reg_model_{drug}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PhenotypeSeeker model not found: {model_path}")

    token_id_path = os.path.join(train_dir, 'kmer_token_id.txt')
    if not os.path.exists(token_id_path):
        raise FileNotFoundError(f"k-mer token id file not found: {token_id_path}")

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
def run(intest,label2,odir,drug,pc_c,snv_c,kmer_c,mfile,threads=1,feature_limit=None,sentence_limit=None,train_dir=None):
    work_dir = os.path.abspath(odir)
    build_dir(work_dir)
    train_dir = os.path.abspath(train_dir if train_dir else work_dir)
    dr = index_genomes(intest)
    val = list(dr)
    if label2:
        label2 = os.path.abspath(label2)
        if not os.path.exists(label2):
            raise FileNotFoundError(f"Provided label file not found: {label2}")
        validate_label_genomes(label2, dr)
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
    # Run prodigal and rgi for all input genomes
    print('Run Prodigal and RGI for all input genomes!',flush=True)
    gdir,pdir=run_prodigal_rgi(dr,work_dir,threads)
    gdir=work_dir+'/Genes_ts'
    pdir=work_dir+'/Proteins_ts'
    #exit()
    filter_rgi(work_dir+'/RGI_raw_ts',drug,mfile,work_dir+'/RGI_ts')
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

            run_cdhit(pval,work_dir,train_merged_proteins)
            generate_tokens_from_cdhit(train_cluster_file,work_dir,label2)
        
        ############### K-mer tokens ##########
        #print(os.path.abspath(work_dir))
        #exit()
        if not kmer_c==1:
            run_ps(intest,label,label2,drug,os.path.abspath(work_dir),train_dir)
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
    scan_length(work_dir,train_dir)
    scan_length_fs(work_dir,train_dir)
    scan_length_fs_shap(shap_dir,train_shap_dir)


def main():
    usage="StrainAMR_build_test - Takes strain genomes (test sets) as input and extracts graph-based, pc-based, k-mer-based features for antimicrobial resistance prediction."
    parser=argparse.ArgumentParser(prog="StrainAMR_build_test.py",description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input strain genomes (test set).")
    parser.add_argument('-l','--label_file',dest='lab_file',type=str,help="The directory of the input label file for your test data. If omitted, placeholder labels will be generated automatically.")
    parser.add_argument('-d','--drug',dest='drug_name',type=str,help="The name of the predicted drug. (Note: The drug name of your input test data must match the drug name of your training data.)")
    parser.add_argument('-p','--pc',dest='close_pc',type=int,help="If set to 1, then will skip pc tokens generation step. (Defaut: 0)" ,default=0)
    parser.add_argument('-s','--snv',dest='close_snv',type=int,help="If set to 1, then will skip snv tokens generation step. (Default: 0)",default=0)
    parser.add_argument('-k','--kmer',dest='close_kmer',type=int,help="If set to 1, then will skip k-mer tokens generation step. (Default:0)",default=0)
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory for generated test features and reports.")
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
    run(infile,lab_file,out,drug,pc_c,snv_c,kmer_c,mfile,threads,feature_limit,sentence_limit,train_dir=db_dir)

if __name__=="__main__":
    sys.exit(main())
#run('../../Sau/Ref_Genome','sau_label.txt','Sau_split','levofloxacin','drug_to_class.txt','../../Sau/Ref_Genome_extra_used','sau_label_test.txt')


#run('../../Ecoli/Ref_Genome','ecoli_label.txt','Ecoli_3fold','levofloxacin','drug_to_class.txt')
#run('../../Kcp/Ref_Genome','kcp_label.txt','Kcp_3fold','ceftazidime-avibactam','drug_to_class.txt')
    
