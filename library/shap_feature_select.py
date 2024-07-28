import re
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap

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
            

def shap_select(infile,ofile):
    based=os.path.dirname(ofile)
    pre=os.path.splitext(os.path.basename(infile))[0]
    #X_train, X_test, y_train, y_test,feas=convert2arr(infile)
    X,y,feas,strains=convert2arr(infile)
    #print(X,X.shape)
    #exit()
    nX=pd.DataFrame(data = X,columns=feas)
    #X_test,y_test,w=convert2arr(intest)
    clf = RandomForestClassifier(random_state=0,n_estimators=500)
    model=clf.fit(X,y)
    
    y_pred=clf.predict_proba(X)
    y_pred_rf = [pred[1] for pred in y_pred]
    auc_test = roc_auc_score(y, y_pred_rf)
    print('Shap::Train: ',auc_test)
    #exit()

    explainer = shap.TreeExplainer(model)
    try:
        shap_values=explainer.shap_values(X)
    except:
        shap_values=explainer.shap_values(X,check_additivity=False)
    try:
        shap_obj=explainer(nX)
    except:
        shap_obj=explainer(nX,check_additivity=False)
    shap_obj.values=shap_values[0]
    shap_obj.base_values=shap_obj.base_values[:,0]
    #print(shap_obj[0])
    #print(shap_obj)
    #exit()
    #exit()
    #print(X.shape,len(feas))
    #exit()
    #nX=pd.DataFrame(data = X,columns=feas)
    #print(nX)
    #shap_obj=explainer(X)
    shap.summary_plot(shap_values,nX)
    plt.savefig(based+'/'+pre+"_summary.png",dpi=400)
    shap.plots.beeswarm(shap_obj,max_display=20)
    plt.savefig(based+'/'+pre+"_beeswarm.png",dpi=400)
    #exit()
    #print(shap_obj)
    #print(len(shap_values))
    if len(shap_values)==2:
        shap_values=shap_values[1]
    arrs=[] # arr contains dict: key -> 
    op=open(based+'/'+pre+'_shap_local_matrix.txt','w+')
    op.write('Samples,'+','.join(feas)+'\n')
    c=0
    for s in shap_values:
        tem=dict(zip(feas,s))
        st=[]
        for e in s:
            st.append(str(e))
        op.write(strains[c]+','+','.join(st)+'\n')
        arrs.append(tem)
        c+=1
    #shap.plots.beeswarm(shap_values)
    #plt.savefig("ecoli_pc_summary.png",dpi=400)
    #exit()

    #print(X[0],shap_values[0])
    #print(feas)
    #exit()

    shapm=np.abs(shap_values).mean(0)
    #print(shapm.shape)
    #exit()
    d=dict(zip(feas,shapm))

    res=sorted(d.items(), key=lambda x: x[1],reverse=True)
    #print(res[:10])
    c=0
    o=open(based+'/'+pre+'_shap.txt','w+')
    o.write('ID\tToken_ID\tShap\n')
    for r in res:
        #print()
        if r[1]==0:continue
        o.write(str(c+1)+'\t'+r[0]+'\t'+str(r[1])+'\n')
        c+=1
    
    td=regenerate(infile,ofile,arrs)
    o=open(based+'/'+pre+'_shap_rmf.txt','w+')
    o.write('ID\tToken_ID\n')
    c=1
    for t in td:
        o.write(str(c)+'\t'+t+'\n')
        c+=1
    #print(c)
    #print(d)
    #print(shap_values)
    #print(shap_values.shape
    #y_pred=clf.predict_proba(X_test)
    #y_pred_rf = [pred[1] for pred in y_pred]

