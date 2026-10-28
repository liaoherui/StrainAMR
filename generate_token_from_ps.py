import re
import os

def generate_tps(inmatrix,label,ofile,otid,train):
    f=open(inmatrix,'r')
    line=f.readline().strip()
    ele=re.split('\,',line)
    ele=ele[1:-2]
    #print(ele[-1])
    #exit()
    o1=open(otid,'w+')
    arr=[]
    c=1
    for e in ele:
        o1.write(e+'\t'+str(c)+'\n')
        arr.append(str(c))
        c+=1
    o1.close()
    # One joined string per genome instead of a list of per-column strings.  The matrix is genomes x
    # selected k-mers (easily 1e5 columns), so the list version held ~1e8 separate Python str objects,
    # tens of GB, almost all of them '0'.  The written output is unchanged.
    d={}
    dn={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        name=ele[0]
        ele=ele[1:-2]
        tokens=[arr[c] if e!='0' else '0' for c,e in enumerate(ele)]
        d[name]=','.join(tokens)
        dn[name]=len(tokens)
    #print(train)
    #exit()
    f=open(label,'r')
    o=open(ofile,'w+')
    line=f.readline().strip()
    o.write(line+'\tTokens_Num\tTokens\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        #print(ele)
        #exit()
        if ele[0] not in train:continue
        #print(ele[0],'exits!')
        o.write(ele[0]+'\t'+ele[1]+'\t'+str(dn[ele[0]])+'\t'+d[ele[0]]+'\n')


#generate_tps('../PhenotypeSeeker/example/Cdi_analysis/Azithromycin_MLdf.csv','../Cdi_train_res/strains_sentences.txt','strains_setences_cdi_train.txt','cdi_token_id.txt')
