import re
import os
import numpy as np
from sklearn.feature_selection import chi2

from library import utils


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
            if t=='0':
                continue
            if t in d:
                tem.append((idx,t))
        selected=tem
        if max_tokens is not None and max_tokens>0 and len(tem)>max_tokens:
            decorated=[]
            default_rank=len(token_priority) if token_priority else 0
            for original_idx,t in tem:
                rank=default_rank+original_idx
                if token_priority and t in token_priority:
                    rank=token_priority[t]
                decorated.append((rank,original_idx,t))
            decorated.sort()
            selected=[(orig_idx,tok) for _,orig_idx,tok in decorated[:max_tokens]]
        if not selected:
            kept_indices=set()
        else:
            selected.sort(key=lambda x:x[0])
            kept_indices={idx for idx,_ in selected}
        tokens_out=[]
        for idx,t in enumerate(tk):
            if t=='0':
                if not tokens_out or tokens_out[-1] != '0':
                    tokens_out.append('0')
                continue
            if idx in kept_indices:
                tokens_out.append(t)
        if len(tokens_out)==0:
            tokens_out=['0']
        non_zero_count=sum(1 for t in tokens_out if t!='0')
        o.write(ele[0]+'\t'+ele[1]+'\t'+str(non_zero_count)+'\t'+','.join(tokens_out)+'\n')
    

def sef(
    infile,
    ofile,
    ofile2,
    max_features=None,
    max_tokens_per_sentence=None,
    mapping_files=None,
    rgi_dir=None,
):
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
    token_map = utils.load_token_mappings(mapping_files)
    rgi_map = utils.load_rgi_annotations(rgi_dir)

    o=open(ofile,'w+')
    header=['ID','Feature_ID','Feature','AMR_Gene_Family','P-value','Chi2-statistic']
    o.write('\t'.join(header)+'\n')
    c=1
    dr={}
    stats={}
    all_t={}
    for i in range(len(scores)):
        if pvalues[i]>0.05:
            all_t[arr[i]]=''
            continue
        #print(f"Feature {arr[i]}: P-value = {pvalues[i]}, Chi2-statistic = {scores[i]}")
        dr[arr[i]]=pvalues[i]
        stats[arr[i]]=(pvalues[i],scores[i])
    res=sorted(dr.items(), key = lambda kv:(kv[1], kv[0]))
    if max_features is not None and max_features > 0:
        res=res[:max_features]
    dused={}
    ordered_features=[r[0] for r in res]
    for r in res:
        feature_id=r[0]
        feature_name=utils.token_to_feature(feature_id,token_map)
        amr_family=rgi_map.get(feature_name,'NA') if rgi_map else 'NA'
        pval,chi2=stats.get(feature_id,(dr[feature_id],0.0))
        row=[str(c),str(feature_id),feature_name,amr_family,str(pval),str(chi2)]
        o.write('\t'.join(row)+'\n')
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
