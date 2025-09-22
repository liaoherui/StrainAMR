import re
import os
import multiprocessing


def _align_worker(args):
    filename, indir, d, odir = args
    proc = multiprocessing.current_process()
    #pre=re.split('\.',filename)[0]
    pre=os.path.splitext(filename)[0]
    if pre not in d:
        print(pre,' not in training data, will skip!', flush=True)
        return
    print(f"[Align] start {pre} -- {proc.name}", flush=True)
    os.system('GraphAligner -g '+d[pre]+' -f '+indir+'/'+filename+' -a '+odir+'/'+pre+'.gaf -x vg -t 32 ')
    print(f"[Align] done {pre} -- {proc.name}", flush=True)


def align(indir, gdir, odir, threads=1):
    if not os.path.exists(odir):
        os.makedirs(odir)
    d={}
    for filename in os.listdir(gdir):
        #pre=re.split('\.',filename)[0]
        pre=os.path.splitext(filename)[0]
        d[pre]=gdir+'/'+filename
    params = [(fn, indir, d, odir) for fn in os.listdir(indir)]
    pool = multiprocessing.Pool(processes=int(threads))
    for p in params:
        pool.apply_async(_align_worker, (p,))
    pool.close()
    pool.join()




#align('Genomes_val','GFA_train_Minimap2','run_galign.sh','Align_val_res')
#align('Genomes_test','GFA_train_Minimap2','run_galign.sh','Align_test_res')
