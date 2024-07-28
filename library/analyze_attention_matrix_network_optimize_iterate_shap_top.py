import re
import os
import uuid
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import markov_clustering as mc
from networkx.algorithms.community.centrality import girvan_newman
from cdlib import algorithms
import random

def add_dc(i,dc):
    if i not in dc:
        dc[i]=1
    else:
        dc[i]+=1

def add_dcs(s1,s2,dcs,v,shap_top):
    if s1 in shap_top:
        if s1 not in dcs:
            dcs[s1]={}
        if s2 not in dcs[s1]:
            dcs[s1][s2]=v
        else:
             dcs[s1][s2]+=v
    if s2 in shap_top:
        if s2 not in dcs:
            dcs[s2]={}
        if s1 not in dcs[s2]:
            dcs[s2][s1]=v
        else:
            dcs[s2][s1]+=v


def build_net(matrix,sentence,d1,d2,k,dg,shap_top,dc,dcs):
    G=nx.Graph()
    matrix_z=matrix
    for i in range(len(matrix)):
        matrix_z[i][i]=0
    '''
    ne=(matrix_z<0)
    for s in ne:
        for e in s:
            if e:
                print(s)
                exit()
    print(ne)
    exit()
    '''
    exist=(matrix_z!=0)
    all_v=matrix_z[exist]
    cutoff=np.percentile(all_v, 75)
    matrix_z[matrix_z<cutoff]=0
    indices=np.where(matrix_z>0)
    indices=list(zip(indices[0],indices[1]))
    dcount={}
    dcst={}
    for i in indices:
        if not int(sentence[i[0]]) in shap_top and not int(sentence[i[1]]) in shap_top:continue
        s1=int(sentence[i[0]])
        s2=int(sentence[i[1]])
        if s1==0 or s2==0:continue
        if int(s1)==int(s2):continue
        add_dc(s1,dc)
        add_dc(s2,dc)
        add_dc(s1,dcount)
        add_dc(s2,dcount)
        add_dcs(s1,s2,dcst,matrix_z[i[0]][i[1]],shap_top)
        #add_dcs(s2,dcs,matrix_z[i[0]][i[1]])
        #s1=str(s1)
        #s2=str(s2)
        if dg.has_edge(s1,s2):
            dg[s1][s2]['value']+=matrix_z[i[0]][i[1]]
        else:
            dg.add_edge(s1,s2)
            dg[s1][s2]['value']=matrix_z[i[0]][i[1]]
            dg[s1][s2]['count']=0
            dg[s1][s2]['weight']=0
        G.add_edge(s1,s2)
    for u,v in G.edges:
        dg[u][v]['count']+=1
    for s in dcst:
        if s not in shap_top:continue
        if s not in dcs:
            dcs[s]={}
        for sd in dcst[s]:
            if sd not in dcs[s]:
                dcs[s][sd]=dcst[s][sd]/float(dcount[sd])
            else:
                dcs[s][sd]+=dcst[s][sd]/float(dcount[sd])
    #num=matrix_z.sum(axis=0)
    #den=exist.sum(axis=0)
    #print(num,num.shape)
    #print(den,den.shape)
    #exit()
    #row_means = np.mean(matrix_z, axis=1)

def lf(i,d):
    if int(i) not in d:
        d[int(i)]=1
    else:
        d[int(i)]+=1

def stat_sent_count(sentence_file):
    f=open(sentence_file,'r')
    line=f.readline()
    dpc={}
    dnc={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        st=re.split(',',ele[-1])
        if int(ele[1])==1:
            for s in st:
                lf(s,dpc)
        else:
            for s in st:
                lf(s,dnc)
    return dpc,dnc

def check_top10_attn(odir,dg,pre,shap_top,dc,dcs,dcn):
    d=nx.to_dict_of_dicts(dg)
    o=open(odir+'/'+pre+'_tokens_top_raw.txt','w+')
    o2=open(odir+'/'+pre+'_tokens_top_norm.txt','w+')
    o3=open(odir+'/'+pre+'_tokens_top_norm_sent.txt','w+')
    o4=open(odir+'/'+pre+'_tokens_top_norm_sent_m10_new_top.txt','w+')
    o5=open(odir+'/'+pre+'_tokens_top_norm_sent_m50_new_top.txt','w+')
    o.write('ID\tShap_token_ID\tImportant_token\tAttention_weight\n')
    o2.write('ID\tShap_token_ID\tImportant_token\tAttention_weight\n')
    o3.write('ID\tShap_token_ID\tImportant_token\tAttention_weight\n')
    o4.write('ID\tShap_token_ID\tImportant_token\tAttention_weight\n')
    o5.write('ID\tShap_token_ID\tImportant_token\tAttention_weight\n')
    c=1
    c2=1
    c3=1
    c4=1
    c5=1
    for s in shap_top:
        td=dg[s]
        tem={}
        tem2={}
        tem3={}
        tem4={}
        tem5={}
        for t in td:
            tem[t]=d[s][t]['value']
            tem2[t]=d[s][t]['value']/float(dc[t])
            tem3[t]=dcs[s][t]
            if float(dcn[t])>10:
                tem4[t]=d[s][t]['value']/float(dc[t])
            else:
                tem4[t]=0
            if float(dcn[t])>50:
                tem5[t]=d[s][t]['value']/float(dc[t])
            else:
                tem5[t]=0
        res=sorted(tem.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        for r in res[:10]:
            o.write(str(c)+'\t'+str(s)+'\t'+str(r[0])+'\t'+str(r[1])+'\n')
            c+=1
        res2=sorted(tem2.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        
        for r in res2[:10]:
            o2.write(str(c2)+'\t'+str(s)+'\t'+str(r[0])+'\t'+str(r[1])+'\n')
            c2+=1

        res3=sorted(tem3.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        for r in res3[:10]:
            o3.write(str(c3)+'\t'+str(s)+'\t'+str(r[0])+'\t'+str(r[1])+'\n')
            c3+=1
        res4=sorted(tem4.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        for r in res4[:10]:
            o4.write(str(c4)+'\t'+str(s)+'\t'+str(r[0])+'\t'+str(r[1])+'\n')
            c4+=1
        res5=sorted(tem5.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        for r in res5[:10]:
            o5.write(str(c5)+'\t'+str(s)+'\t'+str(r[0])+'\t'+str(r[1])+'\n')
            c5+=1

    


def write_out(da,db,ofile1,ofile2):
    o1=open(ofile1,'w+')
    o2=open(ofile2,'w+')
    res1=sorted(da.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    res2=sorted(db.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    c=1
    for r in res1:
        o1.write(str(c)+'\t'+r[0]+'\t'+str(r[1])+'\n')
        c+=1
    c=1
    for r in res2:
        o2.write(str(c)+'\t'+r[0]+'\t'+str(r[1])+'\n')
        c+=1


def filter_low_weight_edges(g):
    if True:
        for u,v in g.edges:
            g[u][v]['weight']=g[u][v]['value']/g[u][v]['count']
            #all_weight.append(d[i][u][v]['weight'])
        '''
        all_weight=np.array(all_weight)
        fv=np.percentile(all_weight,10) # Remove edges whose weight is smaller than 10-percentile value
        for u,v in d[i].edges:
            if d[i][u][v]['weight']<fv:
                d[i].remove_edge(u,v)
        '''

def plot_graph(in_arr,g,outg,labels):
    plt.figure(figsize=(12,12))
    comp=cm.rainbow(np.linspace(0, 1, len(in_arr))) 
    d={}
    s=0
    for a in in_arr:
        for e in a:
            d[e]=comp[s]
        s+=1
    color_map=[]
    for node in g:
        if node in d:
            color_map.append(d[node])
        else:
            color_map.append('gray')

    layout=nx.spring_layout(g) 
    nx.draw(g, layout, node_color=color_map)
    nx.draw_networkx_labels(g,layout,labels)
    plt.savefig(outg,dpi=300)


def scan_graphs(out,g,pre):
    if True:
        o=open(out+'/graph_'+pre+'_stat.txt','w+')
        o.write('Nodes_count\t'+str(g.number_of_nodes())+'\n') 
        o.write('Edges_count\t'+str(g.number_of_edges())+'\n')
        o.write('--------------\n')
        o.write('Top1000_Edges\tweight\n')
        res=sorted_edges = sorted(g.edges(data=True), key=lambda x: x[2]['weight'],reverse=True)
        ch=0
        for r in res:
            if float(r[0])==0 or float(r[1])==0:continue
            o.write(str(r[0])+'_'+str(r[1])+'\t'+str(r[2]['weight'])+'\n')
            ch+=1
            if ch==1000:break
        o1=open(out+'/graph_'+pre+'_node_degree.txt','w+')
        o1.write('Token_ID\tDegree_Centrality\n')
        labels={}
        #res=sorted(nx.degree_centrality(g).items(), key=lambda x : x[1], reverse=True)
        res=sorted(nx.eigenvector_centrality(g,weight = 'weight').items(), key=lambda x : x[1], reverse=True)
        top1=res[0][0]
         
        c=0
        for r in res:
            o1.write(str(r[0])+'\t'+str(r[1])+'\n')
            if c<10:
                labels[r[0]]=r[0]
            c+=1

        deg=nx.eigenvector_centrality(g,weight = 'weight')
        cent = np.fromiter(deg.values(), float)
        sizes = cent / np.max(cent) * 200
        normalize = mcolors.Normalize(vmin=cent.min(), vmax=cent.max())
        colormap = cm.viridis
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(cent)
        
        plt.figure(figsize=(12,12))
        layout=nx.spring_layout(g)
        plt.colorbar(scalarmappaple)
        #nx.draw(g, layout, node_size=sizes, node_color=sizes, cmap=colormap,edge_size=0.2,alpha=0.3)
        nx.draw(g, layout, node_size=sizes, node_color=sizes, cmap=colormap,alpha=0.3)
        nx.draw_networkx_edges(g, layout,alpha=0.3,width=0.2)
        nx.draw_networkx_labels(g,layout,labels)
        plt.savefig(out+'/graph'+'_'+pre+'_network.png',dpi=300)

        '''
        # MCL algorithm test
        spm=nx.to_scipy_sparse_matrix(g,weight='weight')
        #matrix = nx.to_scipy_sparse_matrix(spm)
        #print(spm.toarray())
        clusters=[]
        mq=0
        for inflation in [i / 10 for i in range(15, 26)]:
            result=mc.run_mcl(spm,inflation=inflation)
            clusters_m=mc.get_clusters(result)
            Q=mc.modularity(matrix=result, clusters=clusters_m)
            if Q>mq:
                mq=Q
                clusters=clusters_m
            print("inflation:", inflation, "modularity:", Q)


        o4=open(out+'/graph_mcl_'+pre+'.txt','w+')
        index_to_node = {i: node for i, node in enumerate(g.nodes)}
        #clusters=mc.get_clusters(result)
        s=1
        for c in clusters:
            for e in c:
                o4.write(str(index_to_node[e])+'\t'+str(s)+'\n')
            s+=1
        num=nx.number_of_nodes(g)
        positions = {i:(random.random() * 2 - 1, random.random() * 2 - 1) for i in range(num)}

        if not len(clusters)==0:
            plt.figure(figsize=(12,12))
            mc.draw_graph(spm, clusters, pos=positions, node_size=50, with_labels=False, edge_color="silver")
        
            plt.savefig(out+'/graph'+'_'+pre+'_network_mcl.png',dpi=300)
        '''
        
        
        # Community detection
        o5=open(out+'/community_'+pre+'_louvain_r2.txt','w+')
        #communities=list(nx.community.label_propagation_communities(g))
        #communities= list(nx.community.louvain_communities(g,seed=123))
        coms = algorithms.louvain(g,weight="weight",resolution=2.,randomize=False)

        s=1
        for n in coms.communities:
            for e in n:
                o5.write(str(e)+'\t'+str(s)+'\n')
            s+=1
        plot_graph(coms.communities,g,out+'/community_'+pre+'_louvain_r2.png',labels)
        #exit()
        #print(communities)
        
        #h = g.copy()
        # Iterate run louvain detection
        '''
        # Herui's annotate at 2023-12-18
        dused={}
        for i in range(4):
            tid=i+1
            temd={}
            ins=h.copy()
            o6=open(out+'/community_'+pre+'_louvain_'+str(tid)+'.txt','w+')
            coms = algorithms.louvain(h, weight='weight', resolution=1.,randomize=False)
            s=1
            tg=0 
            for n in coms.communities:
                for e in n:
                    o6.write(str(e)+'\t'+str(s)+'\n')
                    temd[e]=s
                    if e==top1:
                        tg=s
                s+=1
            go={}
            for s in labels:
                if s in dused:continue
                go[s]=labels[s]
            plot_graph(coms.communities,ins,out+'/community_'+pre+'_louvain'+str(tid)+'.png',go)
            for node in ins.nodes:
                if not temd[node]==tg:
                    h.remove_node(node)
                    if node in labels:
                        dused[node]=''
        '''



        '''
        o7=open(out+'/community_'+pre+'_eigenvector.txt','w+')
        coms=algorithms.eigenvector(g)
        s=1
        for n in coms.communities:
            for e in n:
                o7.write(str(e)+'\t'+str(s)+'\n')
            s+=1
        plot_graph(coms.communities,g,out+'/community_'+pre+'_eigenvector.png',labels)
        '''
        '''
        o8=open(out+'/community_'+pre+'_percomvc.txt','w+')
        coms=algorithms.percomvc(g)
        s=1
        for n in coms.communities:
            for e in n:
                o8.write(str(e)+'\t'+str(s)+'\n')
            s+=1
        plot_graph(coms.communities,g,out+'/community_'+pre+'_percomvc.png',labels)
        

        o9=open(out+'/community_'+pre+'_modm.txt','w+')
        coms=algorithms.mod_m(g, top1)
        s=1
        for n in coms.communities:
            for e in n:
                o9.write(str(e)+'\t'+str(s)+'\n')
            s+=1
        plot_graph(coms.communities,g,out+'/community_'+pre+'_modm.png',labels)
        
        
        o9=open(out+'/community_'+pre+'_lswl.txt','w+')
        coms=algorithms.lswl(g,top1)
        s=1
        for n in coms.communities:
            for e in n:
                o9.write(str(e)+'\t'+str(s)+'\n')
            s+=1
        plot_graph(coms.communities,g,out+'/community_'+pre+'_lswl.png',labels)
        

        o10=open(out+'/community_'+pre+'_pairs.txt','w+')
        coms=algorithms.paris(g)
        s=1
        for n in coms.communities:
            for e in n:
                o10.write(str(e)+'\t'+str(s)+'\n')
            s+=1
        plot_graph(coms.communities,g,out+'/community_'+pre+'_pairs.png',labels)
        '''



    

def obtain_important_tokens(matrix,sentence_file,odir,pre,shap_top_file):
    '''
    f=open(sentence_file,'r')
    sentence_new_file=uuid.uuid1().hex+'.txt'
    o=open(sentence_new_file,'w+')
    line=f.readline()
    o.write(line)
    c=0
    while True:
        line=f.readline().strip()
        if not line:break
        if index[c]:
            o.write(line+'\n')
        c+=1
    o.close()
    '''
    fp=open(shap_top_file,'r')
    line=fp.readline()
    shap_top={}
    c=0
    while True:
        line=fp.readline().strip()
        if not line:break
        ele=line.split('\t')
        shap_top[int(ele[1])]=''
        c+=1
        if c>10:break
    #exit()
    f=open(sentence_file,'r')
    line=f.readline()
    ss_p=[] # sentences arr
    ss_n=[]
    le=matrix.shape[-1]
    dl={} # label info
    sid=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        tk=re.split(',',ele[-1])
        tem=np.zeros(le)
        dl[sid]=int(ele[1])
        c=0
        for t in tk:
            tem[c]=int(t)
            c+=1 
        if int(ele[1])==1:
            ss_p.append(tem)
        else:
            ss_n.append(tem)
        sid+=1
    ss_p=np.array(ss_p)
    ss_n=np.array(ss_n)
    c=0
    d1={} # for token < positive >
    d2={} # for token - pairs <positive>
    d3={} # <negative>
    d4={} # <negative>
    dpc={} # first token pair importance cal method
    dnc={}
    dpcs={} # second token pair importance cal method
    dpns={}
    k=20
    cp=0
    cn=0
    # Before we start, we need to generate 2 graphs
    dgp=nx.Graph() # graphs for positive samples
    dgn=nx.Graph() # graphs for negative samples
    '''
    G=nx.Graph()
    dgp[i]=G
    G=nx.Graph()
    dgn[i]=G
    '''
    for e in matrix:
        '''
        if dl[c]==0:
            c+=1
            continue
        '''
        if dl[c]==1:
            sentence=ss_p[cp]
            cp+=1
        else:
            sentence=ss_n[cn]
            cn+=1
        #print(len(sentence))
        # Ther are 8 heads for each matrix
        tc=0 # Head count
        for m in e:
            #print(m)
            #m=np.array(m)
            #print(m.shape)
            #print('Contain negative value?: ',np.any(matrix<0))
            #tc+=1
            #continue
            #print(m.shape)
            if dl[c]==1:
                build_net(m,sentence,d1,d2,k,dgp,shap_top,dpc,dpcs)
                #print(d1)
                #exit()
            else:
                #continue
                build_net(m,sentence,d3,d4,k,dgn,shap_top,dnc,dpns)
            tc+=1
        #exit()
        c+=1
    if not os.path.exists(odir):
        os.makedirs(odir)
    #os.system('rm '+sentence_new_file)
    filter_low_weight_edges(dgp) 
    filter_low_weight_edges(dgn)
    dpc_sc,dnc_sc=stat_sent_count(sentence_file)
    check_top10_attn(odir,dgp,pre+'_positive',shap_top,dpc,dpcs,dpc_sc)
    check_top10_attn(odir,dgn,pre+'_negative',shap_top,dnc,dpns,dnc_sc)
    
    scan_graphs(odir,dgp,pre+'_positive')
    scan_graphs(odir,dgn,pre+'_negative')
    #write_out(d1,d2,odir+'/important_tokens_positive_'+pre+'.txt',odir+'/important_token_pairs_positive_'+pre+'.txt')
    #write_out(d3,d4,odir+'/important_tokens_negative_'+pre+'.txt',odir+'/important_token_pairs_negative_'+pre+'.txt')

    '''
    res1=sorted(d1.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    res2=sorted(d2.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    o1=open(odir+'/important_tokens_'+pre+'.txt','w+')
    o2=open(odir+'/important_token_pairs_'+pre+'.txt','w+')
    c=1
    for r in res1:
        o1.write(str(c)+'\t'+r[0]+'\t'+str(r[1])+'\n')
        c+=1
    c=1
    for r in res2:
        o2.write(str(c)+'\t'+r[0]+'\t'+str(r[1])+'\n')
        c+=1
    '''
