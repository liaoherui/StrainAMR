import re
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap

current_dir = os.path.dirname(os.path.realpath(__file__))
import sys
# Add the 'system' directory to the Python path
#system_path = os.path.join(current_dir, '../../')
sys.path.append(current_dir)
import utils
from shapmat.PCA import customPCA
from shapmat.clustering import shap_clustering
from shapmat.clustering_plot import plot_cluster
from shapmat.clustering_plot import plot_elbow_method
import matplotlib

#exit()

def convert2arr(infile):
    f=open(infile,'r')
    line=f.readline()
    arr={}
    strains=[]
    y=[]
    d1={} # strain -> key -> 0 or 1
    #d2={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        key=re.split(',',ele[-1])
        d1[ele[0]]={}
        strains.append(ele[0])
        y.append(int(ele[1]))
        for k in key:
            if k=='0':continue
            arr[k]=''
            d1[ele[0]][k]=1
            #d2[]=
    feas=list(arr.keys())
    print('Shap::There are ',len(feas),' features in total.')
    X=[]
    for s in strains:
        tem=[]
        for e in feas:
            if e in d1[s]:
                tem.append(1)
            else:
                tem.append(0)
        X.append(tem)
    X=np.array(X)
    y=np.array(y)
    #X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

    #return X_train, X_test, y_train, y_test,feas
    return X,y,feas,strains

def regenerate(infile,ofile,arrs):
    f=open(infile,'r')
    o=open(ofile,'w+')
    line=f.readline()
    o.write(line)
    sid=0
    td={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        tk=re.split(',',ele[-1])
        o.write(ele[0]+'\t'+ele[1]+'\t')
        c=0
        sv=arrs[sid]
        ds={}
        tem=[]
        for t in tk:
            if t=='0':
                tem.append(t)
            elif not float(sv[t])==0.0:
                ds[t]=''
                td[t]=''
                tem.append(t)
        #ttem=np.array(tem)
        #arr = np.delete(ttem, np.where(ttem == 0))
        #o.write(str(len(arr))+'\t')
        o.write(str(len(tem))+'\t'+','.join(tem)+'\n')
        sid+=1
                        
    return td
            

def shap_select(infile, ofile, mapping_files=None, rgi_dir=None):
    """Run SHAP on tokens and write a table with feature names and optional RGI info."""
    based = os.path.dirname(ofile)
    pre = os.path.splitext(os.path.basename(infile))[0]
    X, y, feas, strains = convert2arr(infile)
    map_dict = utils.load_token_mappings(mapping_files)
    rgi_map = utils.load_rgi_annotations(rgi_dir)
    nX=pd.DataFrame(data = X,columns=feas,index=strains)
    #X_test,y_test,w=convert2arr(intest)
    clf = RandomForestClassifier(random_state=0,n_estimators=500)
    model=clf.fit(X,y)
    
    y_pred=clf.predict_proba(X)
    y_pred_rf = [pred[1] for pred in y_pred]
    #print(y_pred)
    #print(y_pred_rf)
    pred_y=[1 if item>0.5 else 0 for item in y_pred_rf]
    #exit()
    auc_test = roc_auc_score(y, y_pred_rf)
    print('Shap::Train: ',auc_test)
    #exit()

    explainer = shap.TreeExplainer(model)
    shap_values=explainer.shap_values(X)
    shap_obj=explainer(nX)
    shap_obj.values=shap_values[0]
    shap_obj.base_values=shap_obj.base_values[:,0]
    #print(shap_values[0],y_pred_rf)
    shap_df=pd.DataFrame(data = shap_values[0],columns=feas,index=strains)
    '''
    if len(shap_values) == 2:
        shap_df_1=pd.DataFrame(data = shap_values[1],columns=feas,index=strains)
    '''
    rs=[] # resistant strains
    ss=[] # sensitive strains
    yp_r=[]
    yp_s=[]
    c=0
    for s in strains:
        if not y[c]==int(pred_y[c]):
            c+=1
            continue
        if y[c]==0:
            ss.append(s)
            yp_s.append(y_pred_rf[c])
        else:
            rs.append(s)
            yp_r.append(y_pred_rf[c])
        c+=1
    #print(shap_df)
    plt.figure()
    #shap.summary_plot(shap_values, nX,max_display=10,class_names=["Susceptible","Resistant"])
    shap.summary_plot(shap_values, nX,max_display=10,class_names=["Resistant","Susceptible"])
    #shap.plots.beeswarm(shap_obj, max_display=10,class_names=["Susceptible","Resistant"])
    plt.savefig(based + '/' + pre + '_summary.png', dpi=400)
    shap.plots.beeswarm(shap_obj, max_display=20)
    plt.savefig(based + '/' + pre + '_beeswarm.png', dpi=400)
    # exit()
    # print(shap_obj)
    # print(len(shap_values))
    if len(shap_values) == 2:
        shap_values_0 = shap_values[0]
        shap_values = shap_values[1]
    arrs = []  # arr contains dict: key ->
    op = open(based + '/' + pre + '_shap_local_matrix.txt', 'w+')
    op.write('Samples,' + ','.join(feas) + '\n')
    c = 0
    for s in shap_values:
        tem = dict(zip(feas, s))
        st = []
        for e in s:
            st.append(str(e))
        op.write(strains[c] + ',' + ','.join(st) + '\n')
        arrs.append(tem)
        c += 1
    # shap.plots.beeswarm(shap_values)
    # plt.savefig("ecoli_pc_summary.png",dpi=400)
    # exit()

    # print(X[0],shap_values[0])
    # print(feas)
    # exit()

    shapm = np.abs(shap_values).mean(0)
    if len(shap_values) == 2:
        shapm_0 = np.abs(shap_values_0).mean(0)
    d = dict(zip(feas, shapm))
    if len(shap_values) == 2:
        ds0 = dict(zip(feas, shapm_0))
    res = sorted(d.items(), key=lambda x: x[1], reverse=True)
    c = 0
    o = open(based + '/' + pre + '_shap.txt', 'w+')
    extra = ''
    if rgi_map:

        extra = '\tAMR_Gene_Family'

    if len(shap_values) == 2:
        o.write('ID\tToken_ID\tFeature' + extra + '\tShap_0\tShap_1\n')
    else:
        o.write('ID\tToken_ID\tFeature' + extra + '\tShap\n')
    for r in res:
        if r[1] == 0:
            continue
        feat_name = utils.token_to_feature(r[0], map_dict)

        amr = rgi_map.get(feat_name, 'NA') if rgi_map else 'NA'
        if len(shap_values) == 2:
            if rgi_map:
                o.write(f"{c + 1}\t{r[0]}\t{feat_name}\t{amr}\t{r[1]}\t{ds0[r[0]]}\n")

            else:
                o.write(f"{c + 1}\t{r[0]}\t{feat_name}\t{r[1]}\t{ds0[r[0]]}\n")
        else:
            if rgi_map:

                o.write(f"{c + 1}\t{r[0]}\t{feat_name}\t{amr}\t{r[1]}\n")
                
            else:
                o.write(f"{c + 1}\t{r[0]}\t{feat_name}\t{r[1]}\n")
        c += 1

    td = regenerate(infile, ofile, arrs)
    o = open(based + '/' + pre + '_shap_rmf.txt', 'w+')
    o.write('ID\tToken_ID\n')
    c = 1
    for t in td:
        o.write(str(c) + '\t' + t + '\n')
        c += 1
    #exit()
    if len(rs)==0:
        print('No correctly predicted Resistant strains! Stop cluster analysis.')
        return
    shap_df_r=shap_df[shap_df.index.isin(rs)]
    #print(shap_df)
    #print(shap_df_r)
    #exit()
    PCA = customPCA(X=shap_df_r,crc_proba=yp_r,n_components=2)
    shap_PC = PCA.PCA_scores()
    cumulative_explained_var = PCA.cumulative_explained_variance()
    
    
    #y_correct = y.loc[shap_PC.index]
    #print(y_correct)
    #exit()
    y_df=pd.Series(data=y,index=strains)
    y_df_r=y_df[y_df.index.isin(rs)]
    #print(y_df_r)
    #exit()
    CRC_kmeans = shap_clustering(PC_scores=shap_PC,y=y_df_r)
    n_clust=3 # Default 3 clusters
    #nrows, ncols= 2, 2
    #fig = plt.figure(figsize=(15,10),dpi=70)

    CRC_kmeans_df = CRC_kmeans.kmeans(n_clusters=n_clust)

    #print(CRC_kmeans_df.head(197))
    #exit()
    plot_cluster(CRC_cluster_df=CRC_kmeans_df,n_cluster=n_clust,figsize=(10,10),title='clusters',savefig=True,output_path=based+'/'+pre+'_kmeans.png')
    plot_elbow_method(CRC_cluster_df=CRC_kmeans_df,savefig=True,output_path=based+'/'+pre+'_elbow.png')
    
    nrows, ncols= 1, n_clust
    fig = plt.figure(figsize=(15,10),dpi=70)
    cluster_assigned = CRC_kmeans_df[['cluster']]
    oc=open(based+'/'+pre+'_cls_info.txt','w+')
    #print(type(cluster_assigned),cluster_assigned.index,cluster_assigned['cluster'])
    #print(np.array(cluster_assigned))
    rs=np.array(cluster_assigned)
    #print(rs)
    #exit()
    i=0
    for c in cluster_assigned.index:
        #print(c)
        oc.write(c+'\t'+str(rs[i][0])+'\n')
        i+=1
    #print(type(cluster_assigned),cluster_assigned.index)
    #exit()
    for i in range(1,n_clust+1):
        ax = fig.add_subplot(nrows, ncols, i)
        cluster_name=i
        c_ids = cluster_assigned[cluster_assigned['cluster']==cluster_name].index.to_list()
        #print(c_ids,nX)
        #exit()
        shap_values_df_c = shap_df_r.loc[c_ids]
        shap_values_c = shap_values_df_c.values
        ot=open(based+'/'+pre+'_cls'+str(i)+'_feature.txt','w+')
        shapm_c = np.abs(shap_values_c).mean(0)     
        dm = dict(zip(feas, shapm_c))
        res = sorted(dm.items(), key=lambda x: x[1], reverse=True)
        cm=0
        ot.write('ID\tToken_ID\tShap\n')
        for r in res:
            if r[1]==0:continue
            ot.write(str(cm + 1) + '\t' + r[0] + '\t' + str(r[1]) + '\n')
            cm+=1



        X_c = nX[shap_values_df_c.columns].loc[c_ids]
        #print(X_c)
        #print(shap_values_c)
        #exit()
        shap.summary_plot(shap_values_c,X_c,show=False,plot_size=None, max_display=10,plot_type='bar')
        
        plt.xlim((0,0.065))
        plt.title(f'Cluster {cluster_name}',fontsize=20)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.1, hspace=0.3)
    plt.savefig(based+'/'+pre+'_cluster_feature_importance.png',bbox_inches='tight',dpi=400)

    #exit()
    #print(shap_PC.head(2))
    #exit()
    #exit()
    #print(shap_obj)
    #exit()
    #exit()
    #print(X.shape,len(feas))
    #exit()
    #nX=pd.DataFrame(data = X,columns=feas)
    #print(nX)
    #shap_obj=explainer(X)
    #print(c)
    #print(d)
    #print(shap_values)
    #print(shap_values.shape
    #y_pred=clf.predict_proba(X_test)
    #y_pred_rf = [pred[1] for pred in y_pred]

    #auc_test=roc_auc_score(y_test, y_pred_rf)

#shap_select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold1/strains_train_pc_token_fs.txt','./strains_train_pc_token_fs_shap_filter.txt')
#exit()
#shap_select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold1/strains_train_kmer_token.txt','./strains_train_kmer_token_shap_filter.txt')
#exit()
#exit()
#shap_select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold1/strains_train_sentence_fs.txt','./strains_train_sentence_fs_shap_filter.txt')

#shap_select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold1/strains_train_pc_token_fs.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold1/strains_train_pc_token_fs_shap_filter.txt')
    
#shap_select('../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold2/strains_train_kmer_token.txt','../Original_StrainAMR_res_for_shap/Ecoli_3fold/Fold2/strains_train_kmer_token_shap_filter.txt')
#exit()

def shap_interaction_select(infile, pair_out, top_n=100):
    """Compute SHAP interaction scores and output top interacting token pairs."""
    X, y, feats, strains = convert2arr(infile)
    clf = RandomForestClassifier(random_state=0, n_estimators=500)
    model = clf.fit(X, y)
    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(X)
    if isinstance(interactions, list):
        interactions = interactions[1]
    inter_mean = np.abs(interactions).mean(0)
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            pairs.append((feats[i], feats[j], inter_mean[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    with open(pair_out, 'w+') as o:
        o.write('Token_ID_1\tToken_ID_2\tInteraction\n')
        c = 0
        for a, b, v in pairs:
            if v == 0:
                continue
            o.write(f'{a}\t{b}\t{v}\n')
            c += 1
            if c >= top_n:
                break
