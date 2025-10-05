import re
import os

def generate_fg(ingraph,refdir,ofile,mfile,label,train):
    om=open(mfile,'w+')
    om.write('GeneID\tNodeID\tTokenID\n')
    dt={} # Strain_ID -> Contig_ID -> Tokens (with correct ID)
    dc={} # used for counting
    c=1
    raw_arr=[]
    for filename in os.listdir(ingraph):
        #pre=re.split('\.',filename)[0]
        pre=os.path.splitext(filename)[0]
        f=open(ingraph+'/'+filename,'r')
        dtem={} # Tem node id to token id
        while True:
            line=f.readline().strip()
            if not line:break
            ele=line.split()
            if line[0]=='S':
                pid=pre+'\t'+ele[1]
                if pid not in dc:
                    raw_arr.append(pid)
                    dc[pid]=c
                    dtem[ele[1]]=c
                    c+=1
            if line[0]=='P':
                sti=re.split('#',ele[1])
                sid=sti[0]
                cid=sti[1]
                if sid not in dt:
                    dt[sid]={cid:''}
                else:
                    dt[sid][cid]=''
                #cid=sti[1]
                tk=re.sub('\+','',ele[2])
                tk=re.sub('\-','',tk)
                tk=re.split(',',tk)
                #tk=re.sub('+','',tk)
                #tk=re.sub('-','',tk)
                tem=[]
                for t in tk:
                    #dt[sid][cid]
                    tem.append(str(dtem[t]))
                dt[sid][cid]=','.join(tem)
                #print(pre)
                #print(dt)
                #exit()
    
    for r in raw_arr:
        om.write(str(r)+'\t'+str(dc[r])+'\n')
    ds={} # strainID -> tokens array
    #dcc={} # strainID -> sentence length
    for filename in os.listdir(refdir):
        if re.search('fai',filename):continue
        #pre=re.split('\.',filename)[0]
        pre=os.path.splitext(filename)[0]
        if pre not in dt:continue
        if pre not in ds:
            ds[pre]=[]
            #dcc[pre]=0
        f=open(refdir+'/'+filename,'r')
        while True:
            line=f.readline().strip()
            if not line:break
            if re.search('>',line):
                cid=line.split()[0]
                cid=re.sub('>','',cid)
                if cid in dt[pre]:
                    ds[pre].append(dt[pre][cid])
                    #tem=re.split(',',dt[pre][cid])
    #print(len(train))
    o=open(ofile,'w+')               
    f=open(label,'r')
    line=f.readline().strip()
    o.write(line+'\tTokens_Num\tTokens\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        '''
        if ele[0] not in dt:
            #print(ele[0],' not in dt, please check! Graph part!')
            continue
        '''
        if ele[0] not in ds:
            if ele[0] in train:
                o.write(ele[0]+'\t'+ele[1]+'\t0\t0\n')
            continue
        all_t=',0,'.join(ds[ele[0]])
        ac=str(len(re.split(',',all_t)))
        o.write(ele[0]+'\t'+ele[1]+'\t'+ac+'\t'+all_t+'\n')



#generate_fg('GFA_train_Minimap2','../../../Sau_train_res/Genes','strains_sau_train_sentence.txt','node_token_match.txt','../../../Sau_train_res/strains_sentences.txt')
