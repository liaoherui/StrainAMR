import re
import os

def build(infa,odir):
    if not os.path.exists(odir):
        os.makedirs(odir)
    #o=open(sh,'w+')
    for filename in os.listdir(infa):
        pre=re.split('\.',filename)[0]
        os.system('minimap2 -cx asm20 -X -t 8 '+infa+'/'+filename+' '+infa+'/'+filename+'  > tem.paf')
        os.system('seqwish -s '+infa+'/'+filename+' -p tem.paf -g '+odir+'/'+pre+'.gfa')


#build('Genomes_train','GFA_train_Minimap2')
