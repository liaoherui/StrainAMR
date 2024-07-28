import re
import os

def generate_tps(inmatrix,label,ofile,otid,train):
    f=open(inmatrix,'r')
    line=f.readline().strip()
    ele=re.split('\,',line)
    ele=ele[1:]
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
    d={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        d[ele[0]]=[]
        name=ele[0]
        ele=ele[1:]
        c=0
        for e in ele:
            if e=='0':
                d[name].append('0')
                c+=1
            else:
                d[name].append(arr[c])
                c+=1
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
        o.write(ele[0]+'\t'+ele[1]+'\t'+str(len(d[ele[0]]))+'\t'+','.join(d[ele[0]])+'\n')


#generate_tps('../PhenotypeSeeker/example/Cdi_analysis/Azithromycin_MLdf.csv','../Cdi_train_res/strains_sentences.txt','strains_setences_cdi_train.txt','cdi_token_id.txt')
