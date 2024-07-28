import re
import os

def align(indir,gdir,odir):
    if not os.path.exists(odir):
        os.makedirs(odir)
    d={}
    for filename in os.listdir(gdir):
        pre=re.split('\.',filename)[0]
        d[pre]=gdir+'/'+filename
    #o=open(ofile,'w+')
    for filename in os.listdir(indir):
        pre=re.split('\.',filename)[0]
        if pre not in d:
            print(pre,' not in training data, will skip!')
            continue
        os.system('GraphAligner -g '+d[pre]+' -f '+indir+'/'+filename+' -a '+odir+'/'+pre+'.gaf -x vg -t 32 ')
        print('GraphAligner -g '+d[pre]+' -f '+indir+'/'+filename+' -a '+odir+'/'+pre+'.gaf -x vg -t 32 ')




#align('Genomes_val','GFA_train_Minimap2','run_galign.sh','Align_val_res')
#align('Genomes_test','GFA_train_Minimap2','run_galign.sh','Align_test_res')
