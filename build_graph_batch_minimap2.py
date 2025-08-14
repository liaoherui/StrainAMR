import re
import os
from concurrent.futures import ProcessPoolExecutor

def build(infa, odir, threads=1):
    if not os.path.exists(odir):
        os.makedirs(odir)
    #o=open(sh,'w+')
    def worker(filename):
        pre=re.split('\.',filename)[0]
        print(f"[Graph] start {pre}", flush=True)
        os.system('minimap2 -cx asm20 -X -t 8 '+infa+'/'+filename+' '+infa+'/'+filename+'  > tem.paf')
        os.system('seqwish -s '+infa+'/'+filename+' -p tem.paf -g '+odir+'/'+pre+'.gfa')
        print(f"[Graph] done {pre}", flush=True)

    with ProcessPoolExecutor(max_workers=threads) as exe:
        list(exe.map(worker, os.listdir(infa)))


#build('Genomes_train','GFA_train_Minimap2')
