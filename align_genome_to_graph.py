import re
import os
from concurrent.futures import ProcessPoolExecutor

def align(indir, gdir, odir, threads=1):
    if not os.path.exists(odir):
        os.makedirs(odir)
    d={}
    for filename in os.listdir(gdir):
        pre=re.split('\.',filename)[0]
        d[pre]=gdir+'/'+filename
    #o=open(ofile,'w+')
    def worker(filename):
        pre=re.split('\.',filename)[0]
        if pre not in d:
            print(pre,' not in training data, will skip!')
            return
        os.system('GraphAligner -g '+d[pre]+' -f '+indir+'/'+filename+' -a '+odir+'/'+pre+'.gaf -x vg -t 32 ')
        print('GraphAligner -g '+d[pre]+' -f '+indir+'/'+filename+' -a '+odir+'/'+pre+'.gaf -x vg -t 32 ')

    with ProcessPoolExecutor(max_workers=threads) as exe:
        exe.map(worker, os.listdir(indir))




#align('Genomes_val','GFA_train_Minimap2','run_galign.sh','Align_val_res')
#align('Genomes_test','GFA_train_Minimap2','run_galign.sh','Align_test_res')
