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
    
def run_cdhit(pval,work_dir):
    # Run CD-Hit on all training proteins
    #os.system('cd-hit -i '+ptrain+' -d 0 -o '+work_dir+'/merge_train_cdhit -c 0.9 -n 5 -M 0')
    # Run CD-Hit on test proteins
    os.system('cd-hit-2d -i '+work_dir+'/merged_proteins_train.fa -i2 '+pval+' -d 0 -o '+work_dir+'/merge_test_cdhit -c 0.9 -n 5 -M 0 -T 0')
    cls1=work_dir+'/merge_train_cdhit.clstr'
    cls2=work_dir+'/merge_test_cdhit.clstr'
    return cls1,cls2

def output_pc_token_file(d,pdir,label2,ofile):
    dr={} # Strain prefix -> Tokens string
    for filename in os.listdir(pdir):
        pre=re.split('\.',filename)[0]
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

    

def generate_tokens_from_cdhit(work_dir,label2):
    f=open(work_dir+'/merge_train_cdhit.clstr','r')
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
    

def run_ps(intest,label,label2,drug,work_dir):
    #o=open('tem.pheno','w+') 
    #o.write('ID\tAddress\t'+drug+'\n')
    #dtrain={} # Pre -> Genome dir
    dval={}
    '''
    for filename in os.listdir(ingenome):
        pre=re.split('\.',filename)[0]
        if pre in train:
            dtrain[pre]=ingenome+'/'+filename
        elif pre in val:
            dval[pre]=ingenome+'/'+filename
    '''
    for filename in os.listdir(intest):
        pre=re.split('\.',filename)[0]
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
    o2=open('ps_inf1.txt','w+')
    o3=open('ps_inf2.txt','w+')
    for a in arr_val:
        o2.write(a+'\t'+dval[a]+'\n')
    o2.close()
    o3.write(drug+'\t'+work_dir+'/PS_out/log_reg_model_'+drug+'.pkl\n')
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
    generate_tpsp(work_dir+'/kmer_token_id.txt',work_dir+'/PS_out/K-mer_lists',label2,work_dir+'/strains_test_kmer_token.txt')
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
        if int(ele[-2])>ms:
            ms=int(ele[-2])
    f=open(infile2,'r')
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        if int(ele[-2])>ms:
            ms=int(ele[-2])
    return ms

def scan_length(odir):
    o=open(odir+'/longest_len_test.txt','w+')
    o.write('Graph\tPC\tKmer\n')
    #for filename in os.listdir(odir):
    #if not re.search('Fold',filename):continue
    ls1=cal_len(odir+'/strains_train_sentence.txt',odir+'/strains_test_sentence.txt')
    ls2=cal_len(odir+'/strains_train_pc_token.txt',odir+'/strains_test_pc_token.txt')
    ls3=cal_len(odir+'/strains_train_kmer_token.txt',odir+'/strains_test_kmer_token.txt')
    o.write(str(ls1)+'\t'+str(ls2)+'\t'+str(ls3)+'\n')
    


#def run(ingenome,label,odir,drug,mfile,intest,label2):
def run(intest,label2,odir,drug,pc_c,snv_c,kmer_c,mfile,threads=1):
    label=odir+'/train_label.txt'
    shap_dir = odir + '/shap'
    build_dir(shap_dir)
    dr={}
    val=[]
    for filename in os.listdir(intest):
        pre=re.split('\.',filename)[0]
        #print(filename)
        #print(pre)
        #exit()
        dr[pre]=intest+'/'+filename
        val.append(pre)
    # Run prodigal and rgi for all input genomes
    print('Run Prodigal and RGI for all input genomes!',flush=True)
    gdir,pdir=run_prodigal_rgi(dr,odir,threads)
    gdir=odir+'/Genes_ts'
    pdir=odir+'/Proteins_ts'
    #exit()
    filter_rgi(odir+'/RGI_raw_ts',drug,mfile,odir+'/RGI_ts')
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
        ele[0]=re.split('\.',ele[0])[0]
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
        work_dir=odir
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
        
        
        copy_genome(odir+'/Genes_ts',tem_gv,'gene')
        copy_genome(odir+'/RGI_ts',tem_rv,'rgi')
        

        copy_protein(odir+'/Proteins_ts',tem_pv)
        
        
        ########### Graph-based tokens ###########
        
        #gt=work_dir+'/Genomes_train'
        if not snv_c==1:
            gv=work_dir+'/Genomes_test'
            extract(tem_rv,tem_gv,gv)

            graph=work_dir+'/GFA_train_Minimap2'

            align_res=work_dir+'/Align_test_res'
            align(gv,graph,align_res)


            generate_at(label2,align_res,work_dir+'/node_token_match.txt',tem_gv,work_dir+'/strains_test_sentence.txt')
        
        ############### PC tokens ############
        #ptrain=merge_all_proteins(tem_pt,work_dir,'train')
        if not pc_c==1:
            pval=merge_all_proteins(tem_pv,work_dir,'test')

            cls1,cls2=run_cdhit(pval,work_dir)
            generate_tokens_from_cdhit(work_dir,label2) 
        
        ############### K-mer tokens ##########
        #print(os.path.abspath(work_dir))
        #exit()
        if not kmer_c==1:
            run_ps(intest,label,label2,drug,os.path.abspath(work_dir))
        #exit()
        
        #train=x[train_idx]
        #val=x[val_idx]
        sef_test(work_dir+'/strains_test_sentence.txt',work_dir+'/feature_remain_graph.txt',work_dir+'/strains_test_sentence_fs.txt')
        sef_test(work_dir+'/strains_test_pc_token.txt',work_dir+'/feature_remain_pc.txt',work_dir+'/strains_test_pc_token_fs.txt')
        ### For shap
        sef_test(work_dir+'/strains_test_sentence_fs.txt',shap_dir+'/strains_train_sentence_fs_shap_rmf.txt',shap_dir+'/strains_test_sentence_fs_shap_filter.txt')
        sef_test(work_dir+'/strains_test_pc_token_fs.txt',shap_dir+'/strains_train_pc_token_fs_shap_rmf.txt',shap_dir+'/strains_test_pc_token_fs_shap_filter.txt')
        sef_test(work_dir+'/strains_test_kmer_token.txt',shap_dir+'/strains_train_kmer_token_shap_rmf.txt',shap_dir+'/strains_test_kmer_token_shap_filter.txt')

        #c+=1
    scan_length(odir)
    scan_length_fs(odir)
    scan_length_fs_shap(shap_dir)
        #exit()


def main():
    usage="StrainAMR_build_test - Takes strain genomes (test sets) as input and extracts graph-based, pc-based, k-mer-based features for antimicrobial resistance prediction."
    parser=argparse.ArgumentParser(prog="StrainAMR_build_test.py",description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input strain genomes (test set).")
    parser.add_argument('-l','--label_file',dest='lab_file',type=str,help="The directory of the input label files for your test data.")
    parser.add_argument('-d','--drug',dest='drug_name',type=str,help="The name of the predicted drug. (Note: The drug name of your input test data must match the drug name of your training data.)")
    parser.add_argument('-p','--pc',dest='close_pc',type=int,help="If set to 1, then will skip pc tokens generation step. (Defaut: 0)" ,default=0)
    parser.add_argument('-s','--snv',dest='close_snv',type=int,help="If set to 1, then will skip snv tokens generation step. (Default: 0)",default=0)
    parser.add_argument('-k','--kmer',dest='close_kmer',type=int,help="If set to 1, then will skip k-mer tokens generation step. (Default:0)",default=0)
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of results. (Note: The output directory of your input test data must match the out directory of your training data.)")
    parser.add_argument('-t','--threads',dest='threads',type=int,help="Number of parallel processes. (Default:1)",default=1)
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
        print('Please provide the output directoy that matches the out directory of your training data!')
        exit()
    run(infile,lab_file,out,drug,pc_c,snv_c,kmer_c,mfile,threads)

if __name__=="__main__":
    sys.exit(main())
#run('../../Sau/Ref_Genome','sau_label.txt','Sau_split','levofloxacin','drug_to_class.txt','../../Sau/Ref_Genome_extra_used','sau_label_test.txt')


#run('../../Ecoli/Ref_Genome','ecoli_label.txt','Ecoli_3fold','levofloxacin','drug_to_class.txt')
#run('../../Kcp/Ref_Genome','kcp_label.txt','Kcp_3fold','ceftazidime-avibactam','drug_to_class.txt')
    
