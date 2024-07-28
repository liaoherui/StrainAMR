import re
import os
import numpy as np
from sklearn.feature_selection import chi2


def scan_token(infile,ofile,d):
    o=open(ofile,'w+')
    f=open(infile,'r')
    line=f.readline().strip()
    o.write(line+'\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        tk=re.split(',',ele[-1])
        if len(tk)==1 and tk[0]=='0':
            o.write(line+'\n')
            continue
        tem=[]
        for t in tk:
            if t=='0':
                #tem.append(t)
                continue
            if t in d:
                tem.append(t)
        if len(tem)==0:
            o.write(ele[0]+'\t'+ele[1]+'\t'+str(len(tem))+'\t0\n')
        else:
            o.write(ele[0]+'\t'+ele[1]+'\t'+str(len(tem))+'\t'+','.join(tem)+'\n')
    

def sef_test(infile2,infile,ofile):
    '''
    f=open(infile,'r')
    line=f.readline()
    d={} # strain -> token_id -> frequency
    arr=[] # token_id list
    s=[] # strain_id list
    y=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        d[ele[0]]={}
        y.append(int(ele[1]))
        s.append(ele[0])
        tk=re.split(',',ele[-1])
        for e in tk:
            if e not in d[ele[0]]:
                d[ele[0]][e]=1
            else:
                d[ele[0]][e]+=1
            if e not in arr:
                arr.append(e)
    X=[]
    for x in s:
        tem=[]
        for a in arr:
            if a not in d[x]:
                tem.append(0)
            else:
                tem.append(d[x][a])
        X.append(tem)
    X=np.array(X)
    y=np.array(y)
    
    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #nX=sel.fit_transform(X)
    #print(nX.shape)
    scores, pvalues = chi2(X, y)
    o=open(ofile,'w+')
    o.write('ID\tFeature_ID\tP-value\tChi2-statistic\n')
    c=1
    dr={}
    di={}
    for i in range(len(scores)):
        if pvalues[i]>0.05:continue
        #print(f"Feature {arr[i]}: P-value = {pvalues[i]}, Chi2-statistic = {scores[i]}")
        dr[arr[i]]=pvalues[i]
        #o.write(str(c)+'\t'+str(arr[i])+'\t'+str(pvalues[i])+'\t'+str(scores[i])+'\n')
        di[arr[i]]=str(arr[i])+'\t'+str(pvalues[i])+'\t'+str(scores[i])+'\n'
        #c+=1
        #exit()
    res=sorted(dr.items(), key = lambda kv:(kv[1], kv[0]))
    '''
    dused={}

    f=open(infile,'r')
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        dused[str(ele[1])]=''

    scan_token(infile2,ofile,dused)
    '''
    o2=open(ofile2,'w+')
    f2=open(infile,'r')
    line=f2.readline()
    o2.write(line)
    while True:
        line=f2.readline().strip()
        if not line:break
        ele=line.split('\t')
        tk=re.split(',',ele[-1])
        tem=[]
        for t in tk:
            if t in dused:
                tem.append(t)
        #ele[-2]=str(len(tem))
        o2.write(ele[0]+'\t'+ele[1]+'\t'+ele[2]+'\t'+str(len(tem))+'\t'+','.join(tem)+'\n')
    '''
    #scan_token(infile2,ov,dused)

    

#sef('Build_multimodal_tokens/Kcp_3fold/Fold1/strains_train_sentence.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/strains_val_sentence.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/kcp_feature_remain_graph.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/strains_train_sentence_fs.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/strains_val_sentence_fs.txt')

#sef('Build_multimodal_tokens/Kcp_3fold/Fold1/strains_train_pc_token.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/strains_val_pc_token.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/kcp_feature_remain_pc.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/strains_train_pc_token_fs.txt','Build_multimodal_tokens/Kcp_3fold/Fold1/strains_val_pc_token_fs.txt')

#sef('Build_multimodal_tokens/Kcp_3fold/Fold2/strains_train_sentence.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/strains_val_sentence.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/kcp_feature_remain_graph.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/strains_train_sentence_fs.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/strains_val_sentence_fs.txt')

#sef('Build_multimodal_tokens/Kcp_3fold/Fold2/strains_train_pc_token.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/strains_val_pc_token.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/kcp_feature_remain_pc.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/strains_train_pc_token_fs.txt','Build_multimodal_tokens/Kcp_3fold/Fold2/strains_val_pc_token_fs.txt')

#sef('Build_multimodal_tokens/Kcp_3fold/Fold3/strains_train_sentence.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/strains_val_sentence.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/kcp_feature_remain_graph.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/strains_train_sentence_fs.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/strains_val_sentence_fs.txt')

#sef('Build_multimodal_tokens/Kcp_3fold/Fold3/strains_train_pc_token.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/strains_val_pc_token.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/kcp_feature_remain_pc.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/strains_train_pc_token_fs.txt','Build_multimodal_tokens/Kcp_3fold/Fold3/strains_val_pc_token_fs.txt')
