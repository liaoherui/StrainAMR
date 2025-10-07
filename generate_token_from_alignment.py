import re
import os

def generate_at(label,inalign,mfile,genomes,ofile):
    f=open(mfile,'r')
    line=f.readline()
    dm={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        dm[ele[0]+'_'+ele[1]]=str(ele[2])
    #d={}
    dt={} # strain -> contig -> tokens
    for filename in os.listdir(inalign):
        #dtem={}
        #gid=re.split('\.',filename)[0]
        gid=os.path.splitext(filename)[0]
        #pre=sa[]

        #if pre not in d:d[pre]=[]
        f1=open(inalign+'/'+filename,'r')
        dused={}
        while True:
            line=f1.readline().strip()
            if not line:break
            ele=line.split('\t')
            sid=re.split('#',ele[0])[0]
            cid=re.split('#',ele[0])[1]
            if sid not in dt:
                dt[sid]={cid:''}
            else:
                if cid not in dt[sid]:
                    dt[sid][cid]=''
            #if pre not in d:d[pre]=[]
            #print(ele)
            if re.search('>',ele[5]) and re.search('<',ele[5]):
                nid=[]
                nid_tem=re.split('>',ele[5])[1:]
                for n in nid:
                    if re.search('<',n):
                        ele=re.split('<',n)
                        for e in ele:
                            nid.append(e)
                    else:
                        nid.append(n)
            elif re.search('>',ele[5]):
                nid=re.split('>',ele[5])[1:]
            elif re.search('<',ele[5]):
                nid=re.split('>',ele[5])[1:]
            '''
            if sid=='562_22426':
                print(nid)
            '''
            tem=[]
            for n in nid:
                tem.append(dm[gid+'_'+n])
            if ele[0] not in dused:
                dused[ele[0]]=''
                dt[sid][cid]=','.join(tem)
                '''
                if sid=='562_22426':
                    #print(ele[0])
                    print(dt[sid])
                    #exit()
                '''
            #print(nid)
            #exit()
            #nid.append('0')
            #print(dt['562_22426'])
            #exit()
        '''
        f2=open(genomes+'/'+pre+'.fa')
        while True:
            line=f2.readline().strip()
            if not line:break
            if re.search('>',line):
                cid=re.sub('>','',line)
                d[pre]=d[pre]+dtem[cid]
        d[pre]=d[pre][:-1]
        '''
        #print(d[pre])
        #exit()
    #print(dt['562_22426'])
    #print(dt)
    #exit()
    d={} # strain -> tokens
    for filename in os.listdir(genomes):
        if re.search('fai',filename):continue
        #pre=re.split('\.',filename)[0]
        pre=os.path.splitext(filename)[0]
        if pre not in dt:continue
        if pre not in d:d[pre]=[]
        f=open(genomes+'/'+filename,'r')
        #tem=[]
        while True:
            line=f.readline().strip()
            if not line:break
            if re.search('>',line):
                cid=line.split()[0]
                cid=re.sub('>','',cid)
                #tem=[]
                if cid in dt[pre]:
                    if not dt[pre][cid]=='':
                        d[pre].append(dt[pre][cid])
        #d[pre].append(tem)
        #print(d[pre])
        #exit()
    f=open(label,'r')
    o=open(ofile,'w+')
    line=f.readline().strip()
    o.write(line+'\tTokens_Num\tTokens\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        '''
        if ele[0] not in dt:
            print(ele[0],' not in dt, please check! Alignment part!')
            continue
        '''
        #print(ele[0],val)
        #exit()
        if ele[0] not in d:
            #print(ele[0],' not in d, please check! Alignment part!')
            #if ele[0] in val:
            #print(ele)
            o.write(ele[0]+'\t'+ele[1]+'\t0\t0\n')
            continue
        #print(d[ele[0]])
        #exit()
        tk=',0,'.join(d[ele[0]])
        tkn=str(len(re.split(',',tk)))
        o.write(ele[0]+'\t'+ele[1]+'\t'+tkn+'\t'+tk+'\n')


#generate_at('../../../Sau_val_res/strains_sentences.txt','Align_val_res','node_token_match.txt','../../../Sau_val_res/Genes','strains_sau_val_sentence.txt')
#generate_alignment_token('../../../Sau_test_res/strains_sentences.txt','Align_test_res','node_token_match.txt','../../../Sau_test_res/Genes','strains_sau_test_sentence.txt')
