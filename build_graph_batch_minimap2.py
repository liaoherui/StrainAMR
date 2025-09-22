import re
import os
import multiprocessing


def _graph_worker(args):
    filename, infa, odir = args
    proc = multiprocessing.current_process()
    #pre=re.split('\.',filename)[0]
    pre=os.path.splitext(filename)[0]
    print(f"[Graph] start {pre} -- {proc.name}", flush=True)
    paf = os.path.join(odir, f"{pre}.paf")
    os.system('minimap2 -cx asm20 -X -t 8 '+infa+'/'+filename+' '+infa+'/'+filename+'  > '+paf)
    os.system('seqwish -s '+infa+'/'+filename+' -p '+paf+' -g '+odir+'/'+pre+'.gfa')
    os.remove(paf)
    print(f"[Graph] done {pre} -- {proc.name}", flush=True)


def build(infa, odir, threads=1):
    if not os.path.exists(odir):
        os.makedirs(odir)
    params = [(fn, infa, odir) for fn in os.listdir(infa)]
    pool = multiprocessing.Pool(processes=int(threads))
    for p in params:
        pool.apply_async(_graph_worker, (p,))
    pool.close()
    pool.join()


#build('Genomes_train','GFA_train_Minimap2')
