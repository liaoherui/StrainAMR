import re
import os

def extract(arg_dir,refdir,odir):
    d={}
    dt={}
    for filename in os.listdir(arg_dir):
        #pre=re.split('\.',filename)[0]
        pre=os.path.splitext(filename)[0]
        if not re.search('txt',filename):continue
        f=open(arg_dir+'/'+filename,'r')
        line=f.readline()
        while True:
            line=f.readline().strip()
            if not line:break
            ele=line.split('\t')
            #print(ele)
            #exit()
            cid=re.split('_',ele[1])[:-1]
            cid='_'.join(cid)
            cid=pre+'#'+cid
            if ele[10] not in d:
                d[ele[10]]={}
            d[ele[10]][cid]=''
            dt[cid]=''
            '''
            if re.search('562_20499',filename):
                print(cid,ele[10])
            '''
            #print(d)
            #exit()
    #print(d['3002790']['CP021681_88'])
    #exit()
    d2={}
    dn={}
    for filename in os.listdir(refdir):
        if re.search('fai',filename):continue
        #pre=re.split('\.',filename)[0]
        pre=os.path.splitext(filename)[0]
        f=open(refdir+'/'+filename,'r')
        while True:
            line=f.readline().strip()
            if not line:break
            if re.search('>',line):
                #print(line)
                ele=line.split()
                cid=re.sub('>','',ele[0])
                cid=pre+'#'+cid
                #print(cid)
                if cid in dt:
                    dn[cid]=pre
                    if cid not in d2:
                        d2[cid]=''
                    go=True
                else:
                    go=False
            else:
                if go:
                    d2[cid]+=line
                #exit()
    if not os.path.exists(odir):
        os.makedirs(odir)
    for s in d:
        if len(d[s])==1:
            print(s,' only 1 contig, will skip!')
            continue
            #o=open(odir+'/'+s+'_single.fasta','w+')
        o=open(odir+'/'+s+'.fasta','w+')
        for e in d[s]:
            #print(e)
            #o.write('>'+dn[e]+'#'+e+'\n')
            o.write('>'+e+'\n')
            o.write(d2[e]+'\n')

#extract('rgi_train')
#build('../../Sau_Train_rgi_genes','../../../Sau_train_res/Genes','Genomes_train')
#build('../../Sau_Val_rgi_genes','../../../Sau_val_res/Genes','Genomes_val')
#build('../../Sau_Test_rgi_genes','../../../Sau_test_res/Genes','Genomes_test')

