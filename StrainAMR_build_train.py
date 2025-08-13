import re
import os
import sys
import argparse
import numpy as np
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
script_dir = os.path.dirname(os.path.abspath(__file__))

def build_dir(idir):
    if not os.path.exists(idir):
        os.makedirs(idir)


from concurrent.futures import ProcessPoolExecutor

def run_prodigal_rgi(dr, odir, threads=1):
    gdir=odir+'/Genes'
    ginfo=odir+'/Genes_info'
    pdir=odir+'/Proteins'
    rgi=odir+'/RGI_raw'

    build_dir(gdir)
    build_dir(ginfo)
    build_dir(pdir)
    build_dir(rgi)
    def worker(s):
        if os.path.exists(rgi+'/'+s+'.txt') and os.path.getsize(rgi+'/'+s+'.txt') != 0:
            return
        if not os.path.exists(pdir+'/'+s+'.faa') or os.path.getsize(pdir+'/'+s+'.faa') == 0:
            os.system('prodigal -i '+dr[s]+' -o '+ginfo+'/'+s+'.genes -d '+gdir+'/'+s+'.fa -a '+pdir+'/'+s+'.faa')
        os.system('rgi main --input_sequence '+gdir+'/'+s+'.fa --output_file '+rgi+'/'+s+' --local --clean  -n 10')

    with ProcessPoolExecutor(max_workers=threads) as exe:
        exe.map(worker, dr)
    return gdir,pdir

def copy_genome(gdir,index,odir,t):
    bfix=''
    if t=='gene':
        bfix='fa'
    else:
        bfix='txt'
    for i in index:
        os.system('cp '+gdir+'/'+i+'.'+bfix+' '+odir)

def copy_protein(pdir,index,odir):
    for i in index:
        os.system('cp '+pdir+'/'+i+'.faa '+odir)
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
    os.system('cat '+indir+'/* >'+odir+'/merged_proteins_'+t+'.fa')
    return odir+'/merged_proteins_'+t+'.fa'
    
def run_cdhit(ptrain,work_dir):
    # Run CD-Hit on all training proteins
    os.system('cd-hit -i '+ptrain+' -d 0 -o '+work_dir+'/merge_train_cdhit -c 0.9 -n 5 -M 0 -T 0')
    # Run CD-Hit on test proteins
    #os.system('cd-hit-2d -i '+work_dir+'/merge_train_cdhit  -i2 '+pval+' -d 0 -o '+work_dir+'/merge_val_cdhit -c 0.9 -n 5 -M 0')
    cls1=work_dir+'/merge_train_cdhit.clstr'
    #cls2=work_dir+'/merge_val_cdhit.clstr'
    return cls1

def output_pc_token_file(d,pdir,label,ofile,idx):
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
    f=open(work_dir+'/merge_train_cdhit.clstr','r')
    o=open(work_dir+'/pc_matches.txt','w+')
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
            o.write(str(i)+'\t'+p+'\n')
        i+=1
    '''
    f2=open(work_dir+'/merge_val_cdhit.clstr','r')
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
    '''

    output_pc_token_file(d,work_dir+'/proteins_train',label,work_dir+'/strains_train_pc_token.txt',train)
    #output_pc_token_file(d2,work_dir+'/proteins_val',label,work_dir+'/strains_val_pc_token.txt',val)
    

def run_ps(train,ingenome,label,drug,work_dir):
    o=open('tem.pheno','w+') 
    o.write('ID\tAddress\t'+drug+'\n')
    dtrain={} # Pre -> Genome dir
    #dval={}
    for filename in os.listdir(ingenome):
        pre=re.split('\.',filename)[0]
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
    
    print(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker modeling tem.pheno',flush=True)
    #print('phenotypeseeker modeling tem.pheno',flush=True)
    os.system(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker modeling tem.pheno')
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
    generate_tps(drug+'_MLdf.csv',label,work_dir+'/strains_train_kmer_token.txt','tem_token_id.txt',train)
    #exit()
    #generate_tpsp('tem_token_id.txt','K-mer_lists',label,work_dir+'/strains_val_kmer_token.txt',val)
    os.system('mv tem.pheno k-mers_and_coefficients*'+drug+'* log*'+drug+'* '+'chi2*'+drug+'*  '+drug+'_MLdf.csv summary*'+drug+'* '+work_dir+'/PS_out')
    #os.system('rm tem.pheno')
    os.system('mv tem_token_id.txt '+work_dir+'/kmer_token_id.txt')

def cal_len(infile1):
    f=open(infile1,'r')
    line=f.readline()
    ms=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        if int(ele[-2])>ms:
            ms=int(ele[-2])
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
    


def run(ingenome,label,odir,drug,pc_c,snv_c,kmer_c,mfile,threads=1):
    dr={}
    for filename in os.listdir(ingenome):
        pre=re.split('\.',filename)[0]
        #print(filename)
        #print(pre)
        #exit()
        dr[pre]=ingenome+'/'+filename
    # Run prodigal and rgi for all input genomes
    print('Run Prodigal and RGI for all input genomes!',flush=True)
    gdir,pdir=run_prodigal_rgi(dr,odir,threads)
    gdir=odir+'/Genes'
    pdir=odir+'/Proteins'
    #exit()
    argn=stat_rig(odir+'/RGI_raw')
    #print(argn)
    #exit()
    '''
    if argn<21:
        build_dir(odir+'/RGI')
        os.system('cp '+odir+'/RGI_raw/* '+odir+'/RGI')
    '''
    filter_rgi(odir+'/RGI_raw',drug,mfile,odir+'/RGI')
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
    x=np.array(x)
    y=np.array(y)
    #splits=StratifiedKFold(n_splits=3,shuffle=True,random_state=1234)
    #datasets=splits.split(x,y)
    c=1
    #fold_arr=[] # for sef arr
    #for train_idx,val_idx in datasets:
    if True:
        print('StrainAMR_build_train starts!',flush=True)
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
           
        
        copy_genome(odir+'/Genes',train,tem_gt,'gene')
        copy_genome(odir+'/RGI',train,tem_rt,'rgi')
        

        copy_protein(odir+'/Proteins',train,tem_pt)
        
        
        ########### Graph-based tokens ###########
        if not snv_c==1:
            gt=work_dir+'/Genomes_train'
            #gv=work_dir+'/Genomes_val'
            extract(tem_rt,tem_gt,gt)
            #exit()
            #extract(tem_rv,tem_gv,gv)

            graph=work_dir+'/GFA_train_Minimap2'
            build(gt,graph,threads)

            #align_res=work_dir+'/Align_val_res'
            #align(gv,graph,align_res)

            generate_fg(graph,tem_gt,work_dir+'/strains_train_sentence.txt',work_dir+'/node_token_match.txt',label,train)

        #generate_at(label,align_res,work_dir+'/node_token_match.txt',tem_gv,work_dir+'/strains_val_sentence.txt',val)
        #c+=1
        #continue
        #exit()
         
        ############### PC tokens ############
        if not pc_c==1:
            ptrain=merge_all_proteins(tem_pt,work_dir,'train')
            #pval=merge_all_proteins(tem_pv,work_dir,'val')
            #exit()

            cls1=run_cdhit(ptrain,work_dir)
            generate_tokens_from_cdhit(work_dir,label,train) 
        
        ############### K-mer tokens ##########
        if not kmer_c==1:
            run_ps(train,ingenome,label,drug,work_dir)
    
        sef(work_dir+'/strains_train_sentence.txt',work_dir+'/feature_remain_graph.txt',work_dir+'/strains_train_sentence_fs.txt')
        sef(work_dir+'/strains_train_pc_token.txt',work_dir+'/feature_remain_pc.txt',work_dir+'/strains_train_pc_token_fs.txt')

        #c+=1
    
    shap_feature_select_withcls.shap_select(work_dir+'/strains_train_sentence_fs.txt',work_dir+'/strains_train_sentence_fs_shap_filter.txt')
    shap_feature_select_withcls.shap_select(work_dir+'/strains_train_pc_token_fs.txt',work_dir+'/strains_train_pc_token_fs_shap_filter.txt')
    #exit()
    shap_feature_select_withcls.shap_select(work_dir+'/strains_train_kmer_token.txt',work_dir+'/strains_train_kmer_token_shap_filter.txt')
    
    #exit()
    scan_length(odir)
    scan_length_fs(odir)
    scan_length_fs_shap(odir)
    os.system('cp '+label+' '+odir+'/train_label.txt')
        #exit()

def main():
    usage="StrainAMR_build_train - Takes strain genomes (training sets) as input and extracts graph-based, pc-based, k-mer-based features for antimicrobial resistance prediction."
    parser=argparse.ArgumentParser(prog="StrainAMR_build_train.py",description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input strain genomes (training set).")
    parser.add_argument('-l','--label_file',dest='lab_file',type=str,help="The directory of the input label files.")
    parser.add_argument('-d','--drug',dest='drug_name',type=str,help="The name of the predicted drug.")
    parser.add_argument('-p','--pc',dest='close_pc',type=int,help="If set to 1, then will skip pc tokens generation step. (Defaut: 0)" ,default=0)
    parser.add_argument('-s','--snv',dest='close_snv',type=int,help="If set to 1, then will skip snv tokens generation step. (Default: 0)",default=0)
    parser.add_argument('-k','--kmer',dest='close_kmer',type=int,help="If set to 1, then will skip k-mer tokens generation step. (Default:0)",default=0)
    parser.add_argument('-t','--threads',dest='threads',type=int,help="Number of parallel processes. (Default:1)",default=1)

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
    run(infile,lab_file,out,drug,pc_c,snv_c,kmer_c,mfile,threads)

if __name__=="__main__":
    sys.exit(main())
    
