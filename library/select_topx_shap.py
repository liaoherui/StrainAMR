import re
import os

def select(infile,shap,ofile,n):
    f=open(shap,'r')
    line=f.readline()
    d={}
    c=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        d[ele[1]]=''
        c+=1
        if c==n:break
    #print(len(d))
    o=open(ofile,'w+')
    f=open(infile,'r')
    line=f.readline()
    o.write(line)
    at={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        o.write(ele[0]+'\t'+ele[1]+'\t')
        tk=re.split(',',ele[-1])
        tem=[]
        dx={}
        for t in tk:
            if t=='0':
                tem.append(t)
            elif t in d:
                dx[t]=''
                at[t]=''
                tem.append(t)
        o.write(str(len(tem))+'\t'+','.join(tem)+'\n')
    
    based=os.path.dirname(ofile)
    pre=os.path.splitext(os.path.basename(infile))[0]
    o=open(based+'/'+pre+'_shap_rmf_top100.txt','w+')
    o.write('ID\tToken_ID\n')
    c=1
    for t in at:
        o.write(str(c)+'\t'+t+'\n')
        c+=1
     



select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_sentence_fs.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_sentence_fs_shap.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_sentence_fs_shap_filter_top100.txt',100)

select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_pc_token_fs.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_pc_token_fs_shap.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_pc_token_fs_shap_filter_top100.txt',100)

select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_kmer_token.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_kmer_token_shap.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold3/strains_train_kmer_token_shap_filter_top100.txt',100)
