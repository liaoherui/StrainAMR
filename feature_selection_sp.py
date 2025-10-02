import re
import os
import numpy as np
from sklearn.feature_selection import chi2


def scan_token(infile,ofile,d,token_priority=None,max_tokens=None):
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
        for idx,t in enumerate(tk):
            if t=='0':continue
            if t in d:
                tem.append((idx,t))
        if len(tem)==0:
            o.write(ele[0]+'\t'+ele[1]+'\t'+str(len(tem))+'\t0\n')
            continue
        if max_tokens is not None and max_tokens>0:
            decorated=[]
            default_rank=len(token_priority) if token_priority else 0
            for original_idx,t in tem:
                rank=default_rank+original_idx
                if token_priority and t in token_priority:
                    rank=token_priority[t]
                decorated.append((rank,original_idx,t))
            decorated.sort()
            tem=[(orig_idx,tok) for _,orig_idx,tok in decorated[:max_tokens]]
            tem.sort(key=lambda x:(token_priority[x[1]] if token_priority and x[1] in token_priority else len(token_priority) if token_priority else x[0],x[0]))
        tokens=[t for _,t in tem]
        o.write(ele[0]+'\t'+ele[1]+'\t'+str(len(tokens))+'\t'+','.join(tokens)+'\n')
    

def sef(infile,ofile,ofile2,max_features=None,max_tokens_per_sentence=None):
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
    all_t={}
    for i in range(len(scores)):
        if pvalues[i]>0.05:
            all_t[arr[i]]=''
            continue
        #print(f"Feature {arr[i]}: P-value = {pvalues[i]}, Chi2-statistic = {scores[i]}")
        dr[arr[i]]=pvalues[i]
        #o.write(str(c)+'\t'+str(arr[i])+'\t'+str(pvalues[i])+'\t'+str(scores[i])+'\n')
        di[arr[i]]=str(arr[i])+'\t'+str(pvalues[i])+'\t'+str(scores[i])+'\n'
        #c+=1
        #exit()
    res=sorted(dr.items(), key = lambda kv:(kv[1], kv[0]))
    if max_features is not None and max_features > 0:
        res=res[:max_features]
    dused={}
    ordered_features=[r[0] for r in res]
    for r in res:
        o.write(str(c)+'\t'+di[r[0]])
        dused[r[0]]=''
        c+=1
    if len(res)==0:
        dused=all_t
        ordered_features=sorted(all_t.keys())

    token_priority={}
    for idx,token in enumerate(ordered_features):
        token_priority[token]=idx

    scan_token(infile,ofile2,dused,token_priority if token_priority else None,max_tokens_per_sentence)
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
