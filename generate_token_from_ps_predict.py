import re
import os

def generate_tpsp(intid,indir,label,ofile):
    f=open(intid,'r')
    d={}
    arr=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        d[ele[0]]=ele[1]
        arr.append(ele[0])
    dr={}
    for filename in os.listdir(indir):
        if re.search('_db_',filename):continue
        if re.search('filtered',filename):continue
        pre=re.split('_k-mer',filename)[0]
        #print(pre)
        #exit()
        f=open(indir+'/'+filename,'r')
        line=f.readline()
        dr[pre]=[]
        tem={}
        while True:
            line=f.readline().strip()
            if not line:break
            ele=line.split('\t')
            #print(ele)
            tem[ele[0]]=str(ele[-1])
            #exit()
        for a in arr:
            if tem[a]=='0':
                dr[pre].append('0')
            else:
                dr[pre].append(str(d[a]))
        #print(dr)
        #exit()
    f=open(label,'r')
    o=open(ofile,'w+')
    line=f.readline().strip()
    o.write(line+'\tTokens_Num\tTokens\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        #if ele[0] not in val:continue
        o.write(ele[0]+'\t'+ele[1]+'\t'+str(len(dr[ele[0]]))+'\t'+','.join(dr[ele[0]])+'\n')



#generate_tpsp('cdi_token_id.txt','../PhenotypeSeeker/example/Cdi_analysis/K-mer_lists','../Cdi_val_res/strains_sentences.txt','strains_setences_cdi_val.txt')
