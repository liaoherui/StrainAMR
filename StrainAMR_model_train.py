import re
import os
import sys
import shutil
import argparse
from typing import List
import numpy as np
from library import Transformer_without_pos_multimodal_add_attn,Transformer_without_pos,Transformer_without_pos_multimodal_add_attn_only2
from library import interpret_pipeline as IP
from library import transformer_attribution as TA
from library.transformer_attribution import clean_state_dict
import torch
from torch.nn import functional as F
from torch import optim,nn
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from steplog import log          # not `step`: the training loop below uses `step` as its own variable
import random

#accelerator = Accelerator()
#device = accelerator.device
lr=0.001
batch_size=20
epoch_num=100
dropout=0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'You are using {device}!',flush=True)
#device = torch.device("cpu")
print(torch.cuda.device_count())
#exit()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_best.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf   # np.Inf was removed in numpy 2.0; np.inf works in both
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model,odir,ol):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,odir,ol)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}',file=ol)
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,odir,ol)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model,odir,ol):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',file=ol)
        torch.save(model.state_dict(), odir+'/'+self.path)
        self.val_loss_min = val_loss
        return True

class subDataset(Dataset.Dataset):
    def __init__(self,feature1,feature2,feature3,label):
        self.feature1=feature1
        self.feature2=feature2
        self.feature3=feature3
        self.label=label
    def __len__(self):
        return len(self.label)
    def __getitem__(self,index):
        feature1=torch.Tensor(self.feature1[index]).to(device)
        feature2=torch.Tensor(self.feature2[index]).to(device)
        feature3=torch.Tensor(self.feature3[index]).to(device)
        label=torch.Tensor(self.label[index]).to(device)
        return feature1,feature2,feature3,label

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


def stat_matrix_shape(infile):
    f=open(infile,'r')
    x=0
    y=0
    line=f.readline().strip()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        x+=1
        tk=re.split(',',ele[-1])
        #print(ele[0],len(tk))
        if len(tk)>y:
            y=len(tk)
    return x,y

def process_intsv(infile,in_x):
    f=open(infile,'r')

    line=f.readline()
    #data=pd.read_table(infile)
    x,ls=stat_matrix_shape(infile)
    if in_x==0:
        matrix=np.zeros((x,ls))
    else:
        matrix=np.zeros((x,in_x))
    #print(matrix.shape)
    #exit()
    #print(data)
    #exit()
    dtoken={}
    c=0
    yp=[]
    mt=0
    samples=[]
    truncated=[0,0]     # genomes truncated, longest seen

    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        #print(ele)
        #exit()
        tk=re.split(',',ele[-1])
        width=matrix.shape[1]
        if len(tk)>width:
            truncated[0]+=1
            truncated[1]=max(truncated[1],len(tk))
            tk=tk[:width]
        tem=0
        for t in tk:
            if int(t) > mt:
                mt=int(t)
            matrix[c][tem]=int(t)
            tem+=1
            if not int(t)==0:
                dtoken[t]=''
        c+=1
        yp.append(int(ele[1]))
        samples.append(ele[0])
    if truncated[0]:
        log(f"WARNING: {truncated[0]} genome(s) in {os.path.basename(infile)} have more tokens than "
            f"the model's width {matrix.shape[1]} (longest {truncated[1]}); the extra tokens were "
            f"dropped. The width is fixed by the trained classifier and cannot be widened afterwards.")
    yp=np.array(yp)
    y=np.eye(2)[yp]
    return matrix,yp,y,mt,ls,samples

def return_batch(train_sentence1,train_sentence2,train_sentence3,label,flag):
    X_train1=torch.from_numpy(train_sentence1).to(device)
    X_train2=torch.from_numpy(train_sentence2).to(device)
    X_train3=torch.from_numpy(train_sentence3).to(device)
    y_train=torch.from_numpy(label).to(device)
    train_dataset=subDataset(X_train1,X_train2,X_train3, y_train)
    if torch.cuda.device_count()>1:
        training_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=flag,
            num_workers=0,
            drop_last = True,
        )
    else:
        training_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=flag,
            num_workers=0,
        )
    return training_loader
        
    
def remove_new_ele(in_train,in_test):
    d={}
    #print(in_train,in_test)
    for e in in_train:
        for x in e:
            d[str(x)]=''
    res=[]
    for r in in_test:
        #print(r)
        #exit()
        tem=[]
        for x in r:
            if str(x) in d:
                tem.append(int(x))
            else:
                tem.append(0)
        res.append(tem)
    res=np.array(res)
    return res

def load_token_len(indir, *extra_dirs):
    """Padding width per channel: the longest token vector over every directory given.

    A validation or test set built separately can hold a genome with more tokens than any training
    genome, and padding to the training width alone overruns the matrix. build-test already writes
    the combined maximum into its own longest_len_fs.txt, but taking the maximum here as well keeps
    this correct whichever directory is passed.
    """
    lens = None
    for d in (indir,) + tuple(x for x in extra_dirs if x):
        path = os.path.join(d, 'longest_len_fs.txt')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            f.readline()
            ele = f.readline().strip().split('\t')
        vals = tuple(int(x) for x in ele[:3])
        lens = vals if lens is None else tuple(max(a, b) for a, b in zip(lens, vals))
    if lens is None:
        raise FileNotFoundError(f"no longest_len_fs.txt in {indir}"
                                + (f" or {extra_dirs}" if extra_dirs else ""))
    return lens

def parsef(ins):
    res=re.split(',',ins)
    return len(res)


def infer_drug(build_dir):
    """Drug name recorded by build-train, from the files its k-mer step leaves behind."""
    import glob
    names = set()
    for pat, cut in ((build_dir + '/kmer_selection_*.tsv', ('kmer_selection_', '.tsv')),
                     (build_dir + '/PS_out/*_MLdf.csv', ('', '_MLdf.csv'))):
        for path in glob.glob(pat):
            base = os.path.basename(path)
            if cut[0]:
                base = base[len(cut[0]):]
            names.add(base[:-len(cut[1])])
    return names.pop() if len(names) == 1 else None


def build_validation_features(val_input, val_label, build_dir, outdir, drug=None, threads=1):
    """Extract features for a separate set of validation genomes against an existing training database.

    This is the v1 equivalent of running build-test on the validation genomes: the database built from
    the training genomes is reused as-is, so the validation genomes never influenced the vocabulary,
    the protein clusters or the k-mer selection. Results are cached, so re-running training does not
    redo the extraction.
    """
    import StrainAMR_build_test as BT
    drug = drug or infer_drug(build_dir)
    if not drug:
        raise ValueError("could not work out the drug name from " + build_dir + "; pass --drug")
    val_out = os.path.join(outdir, 'val_features')
    done = os.path.join(val_out, 'strains_test_kmer_token.txt')
    if os.path.exists(done) and os.path.getsize(done) > 0:
        print(f"Reusing validation features in {val_out}", flush=True)
        return val_out
    print(f"Extracting validation features for '{drug}' from {val_input} against the database in "
          f"{build_dir} (the database is not rebuilt)", flush=True)
    os.makedirs(val_out, exist_ok=True)
    mfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drug_to_class.txt')
    BT.run(val_input, val_label, val_out, drug, 0, 0, 0, mfile, threads, train_dir=build_dir)
    return val_out


def main():
    usage = "StrainAMR_model_train - Takes output folder of StrainAMR_build_train and StrainAMR_build_test as input, and finishes both train and prediction."
    parser = argparse.ArgumentParser(prog=(sys.argv[0] if sys.argv[0].startswith("strainamr") else "StrainAMR_fold_run.py"), description=usage)
    parser.add_argument('-i', '--input_file', dest='input_file', type=str,
                        help="The directory of the input files (output folder of StrainAMR_build_train and StrainAMR_build_test).")
    parser.add_argument('-f', '--feature_used', dest='fused', type=str,
                        help="Choose the feature you wanna use to train the model (e.g. -f kmer,snv means the model will only use these two features for learning). If not set, all three kinds of features will be used. (Default:all)")
    parser.add_argument('-t', '--train_mode', dest='train_mode', type=str,
                        help="If set to 1, then it means there are no test data and only training data exists in the input. (Default: 0).")
    parser.add_argument('--val-input', dest='val_input', type=str, default=None,
                        help="Directory of validation genomes. Their features are extracted against "
                             "the training database in -i (the database is not rebuilt), then one "
                             "model is trained on the -i genomes and selected on these. Needs "
                             "--val-label.")
    parser.add_argument('--val-label', dest='val_label', type=str, default=None,
                        help="Label file for the --val-input genomes.")
    parser.add_argument('--val-dir', dest='val_dir', type=str, default=None,
                        help="Use an existing build-test output directory as the validation set "
                             "instead of extracting one (skips the work if you already ran build-test "
                             "on your validation genomes).")
    parser.add_argument('--drug', dest='drug', type=str, default=None,
                        help="Drug name, only needed with --val-input when it cannot be inferred from -i.")
    parser.add_argument('--val_ids', dest='val_ids', type=str, default=None,
                        help="File listing the genome IDs to use as the validation set (one per line, "
                             "or a label file whose first column is the ID). Trains a single model on "
                             "the remaining genomes and selects it on these. Unlike '-t 1' the split "
                             "is given rather than random, and unlike '-t 0' the test set is not used "
                             "for validation.")
    parser.add_argument('--val_ratio', dest='val_ratio', type=float, default=0.2,
                        help="When '-t 1' is set, split the training data into train/validation using this fraction for validation (Default: 0.2).")
    parser.add_argument('-s', '--save_mode', dest='save_mode', type=str,
                        help="If set to 0, then the model with minimum val loss will be saved, otherwise, the model with best performance on val data will be saved in the output dir. (Default: 1).")
    parser.add_argument('-a', '--attention_weight', dest='attn_weight', type=str,
                        help="If set to 0, then will not output the attention weight between tokens. (Default: 1).")
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        help="Output directory of results. (Default: StrainAMR_fold_res)")
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help="Batch size for training and evaluation. (Default: 20).")
    parser.add_argument('--attr_method', dest='attr_method', choices=['ig', 'gxi'], default='ig',
                        help="Transformer attribution: integrated gradients (ig, default) or single-step gxi (faster).")
    parser.add_argument('--attr_steps', dest='attr_steps', type=int, default=16,
                        help="Integrated-gradient steps (default: 16).")
    parser.add_argument('--attr_max_train', dest='attr_max_train', type=int, default=0,
                        help="Subsample training genomes used for global attribution ranking (0 = all).")
    parser.add_argument('--attr_sanity', dest='attr_sanity', type=int, default=0,
                        help="Set to 1 to run the model-randomisation sanity check after training.")
    parser.add_argument('--lr', dest='lr', type=float, default=None,
                        help="Adam learning rate (Default: 0.001). The pooled head has ~900x fewer "
                             "weights in fc1 than the flatten head, so the rate that suits one will "
                             "usually not suit the other.")
    parser.add_argument('--patience', dest='patience', type=int, default=None,
                        help="Stop when the selected metric has not improved for this many epochs: "
                             "validation F1 with -s 1 (the default), validation loss with -s 0. "
                             "0 disables it and runs every epoch. (Default: 20)")
    parser.add_argument('--head', dest='head', choices=['flatten','pooled'], default='flatten',
                        help="Classifier head. 'flatten' (default) is the original: one weight per "
                             "encoder slot, which only works where slot position is meaningful. "
                             "'pooled' summarises each channel with attention instead, so packed "
                             "channels (graph, PC) can contribute; see library/pooled_head.py.")
    parser.add_argument('--threads-build', dest='threads_build', type=int, default=4,
                        help="Threads for the --val-input feature extraction (Default: 4).")
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help="Number of training epochs. (Default: 100).")
    args = parser.parse_args()
    indir = args.input_file
    tm=args.train_mode
    val_ids_file=args.val_ids
    val_ratio = args.val_ratio
    fused=args.fused
    sm=args.save_mode
    atw=args.attn_weight
    odir= args.outdir
    global batch_size, epoch_num
    if not atw:
        atw=1
    else:
        atw=int(atw)
    if not sm:
        sm=1
    else:
        sm=int(sm)
    if not tm:
        tm=0
    else:
        tm=int(tm)
    if args.batch_size:
        batch_size = args.batch_size
    if args.epochs:
        epoch_num = args.epochs
    global lr
    if args.lr:
        lr = args.lr
    patience = 20 if args.patience is None else max(0, int(args.patience))
    if not fused:
        fused='all'
        fnum=3
    else:
        fnum=parsef(fused)
    if not odir:
        odir='StrainAMR_fold_res'
    os.makedirs(odir, exist_ok=True)
    models_dir = os.path.join(odir, 'models')
    logs_dir = os.path.join(odir, 'logs')
    analysis_dir = os.path.join(odir, 'analysis')
    shap_dir = os.path.join(odir, 'shap')
    for d in (models_dir, logs_dir, analysis_dir, shap_dir):
        os.makedirs(d, exist_ok=True)
    val_dir = args.val_dir
    if args.val_input:
        if not args.val_label:
            raise ValueError("--val-input needs --val-label (the labels of those genomes)")
        val_dir = build_validation_features(args.val_input, args.val_label, indir, odir,
                                            args.drug, args.threads_build)
    if val_dir:
        tm = 0                       # the given validation set takes the place of the '-t 0' test set

    # copy SHAP tables generated during feature building into the training output
    build_shap_dir = os.path.join(indir, 'shap')
    if os.path.isdir(build_shap_dir):
        for fname in (
            'strains_train_sentence_fs_shap_filter.txt',
            'strains_train_pc_token_fs_shap_filter.txt',
            'strains_train_kmer_token_shap_filter.txt',
        ):
            src = os.path.join(build_shap_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, shap_dir)
    # buffering=1 (line-buffered): this handle is held for the whole run and never closed, so an
    # unbuffered-until-exit log loses its tail whenever the process is killed -- e.g. by the OOM
    # killer -- which is exactly when the last lines matter most.
    ol = open(os.path.join(logs_dir, 'train_pred_log.txt'), 'w', buffering=1)
    #lss1=765
    #lss2=536
    #lss3=1000
    lss1,lss2,lss3=load_token_len(indir, val_dir)
    # The classifier flattens the whole sequence (fc1 is Linear((L1+L2+L3)*embed, 64)), so these
    # widths are baked into the weights. Record them next to the model: prediction has to use exactly
    # the same ones, and cannot work them out from the token files it happens to be given.
    with open(os.path.join(odir, 'token_lengths.txt'), 'w') as _fh:
        _fh.write('Graph\tPC\tKmer\n%d\t%d\t%d\n' % (lss1, lss2, lss3))
    _emb, _nch = 512, fnum
    _head_in = _nch * _emb if args.head == 'pooled' else ([lss1, lss1 + lss2,
                                                           lss1 + lss2 + lss3][fnum - 1]) * _emb
    log(f"classifier head: {args.head} -- fc1 takes {_head_in} inputs ({_head_in * 64 / 1e6:.2f}M weights)")
    log(f"token padding widths: graph {lss1}, PC {lss2}, k-mer {lss3}"
        + (" (max over training and validation)" if val_dir else ""))

    x_train1,y_train,yl_train,token_size1,ls,sid_train=process_intsv(indir+'/strains_train_sentence_fs.txt',lss1)
    x_train2,y_train,yl_train,token_size2,ls,sid_train=process_intsv(indir+'/strains_train_pc_token_fs.txt',lss2)
    x_train3,y_train,yl_train,token_size3,ls,sid_train=process_intsv(indir+'/strains_train_kmer_token.txt',lss3)

    has_validation = False
    create_validation_from_train = tm == 1
    if val_ids_file:
        # A given validation set: hold out exactly these genomes, train one model on the rest.
        # Unlike '-t 1' the split is not random, and unlike '-t 0' it does not consume the test set,
        # so the held-out genomes stay a genuine validation set and the test set stays untouched.
        wanted = set()
        with open(val_ids_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                first = line.split()[0]
                if first.lower() in ('strain_id', 'id', 'genome'):
                    continue
                wanted.add(first)
        sid_array = np.array(sid_train)
        mask = np.array([sid in wanted for sid in sid_array])
        missing = wanted - set(sid_array.tolist())
        if not mask.any():
            raise ValueError(f"--val_ids: none of the {len(wanted)} listed genomes are in the training "
                             f"features; are these the right IDs?")
        if missing:
            print(f"--val_ids: {len(missing)} listed genomes are not in the training features "
                  f"(ignored): {', '.join(sorted(missing)[:5])}"
                  + (' ...' if len(missing) > 5 else ''), flush=True)
        x_val1, x_train1 = x_train1[mask], x_train1[~mask]
        x_val2, x_train2 = x_train2[mask], x_train2[~mask]
        x_val3, x_train3 = x_train3[mask], x_train3[~mask]
        y_val, y_train = y_train[mask], y_train[~mask]
        sid_val = sid_array[mask].tolist()
        sid_train = sid_array[~mask].tolist()
        has_validation = True
        create_validation_from_train = False
        import collections as _c
        print(f"Given train/validation split: {len(sid_train)} train / {len(sid_val)} validation "
              f"genomes (validation labels: {dict(_c.Counter(np.asarray(y_val).tolist()))})", flush=True)
        if len(sid_val) < 20:
            print("  WARNING: fewer than 20 validation genomes; early stopping and model selection "
                  "will be noisy", flush=True)
    elif create_validation_from_train:
        if not 0 < val_ratio < 1:
            raise ValueError("'--val_ratio' must be within (0, 1) when '-t 1' is specified.")
        sid_train_array = np.array(sid_train)
        (
            x_train1,
            x_val1,
            x_train2,
            x_val2,
            x_train3,
            x_val3,
            y_train,
            y_val,
            sid_train,
            sid_val,
        ) = train_test_split(
            x_train1,
            x_train2,
            x_train3,
            y_train,
            sid_train_array,
            test_size=val_ratio,
            stratify=y_train,
            random_state=42,
        )
        sid_train = sid_train.tolist()
        sid_val = sid_val.tolist()
        has_validation = True
        print(
            f"Train/validation split enabled (-t 1). Validation ratio: {val_ratio:.2f}.",
            flush=True,
        )
    elif tm == 0:
        src = val_dir or indir
        x_val1,y_val,yl_val,token_size_val1,ls_val,sid_val=process_intsv(src+'/strains_test_sentence_fs.txt',lss1)
        x_val2,y_val,yl_val,token_size_val2,ls_val,sid_val=process_intsv(src+'/strains_test_pc_token_fs.txt',lss2)
        x_val3,y_val,yl_val,token_size_val3,ls_val,sid_val=process_intsv(src+'/strains_test_kmer_token.txt',lss3)
        has_validation = True
        if val_dir:
            import collections as _c
            print(f"Given train/validation split: {len(sid_train)} training genomes, "
                  f"{len(sid_val)} validation genomes from {val_dir} "
                  f"(validation labels: {dict(_c.Counter(np.asarray(y_val).astype(int).tolist()))})",
                  flush=True)
            if len(sid_val) < 20:
                print("  WARNING: fewer than 20 validation genomes; early stopping and model "
                      "selection will be noisy", flush=True)
    else:
        y_val = None
        sid_val = []

    tsize1=token_size1+2
    print('Token count_type1:',tsize1)
    tsize2=token_size2+2
    print('Token count_type2:',tsize2)
    tsize3=token_size3+2
    print('Token count_type1:',tsize3)
    setup_seed(10)
    if fnum == 1:
        encoder_names = ['encoder']
        if fused == 'pc':
            x_train1, token_size1 = x_train2, token_size2
            x_val1, token_size_val1 = x_val2, token_size_val2
            tsize1 = tsize2
            lss1 = lss2
            feature_labels = ['pc']
        elif fused == 'kmer':
            x_train1, token_size1 = x_train3, token_size3
            x_val1, token_size_val1 = x_val3, token_size_val3
            tsize1 = tsize3
            lss1 = lss3
            feature_labels = ['kmer']
        model = Transformer_without_pos.Transformer(src_vocab_size=tsize1, max_length=lss1, device=device, src_pad_idx=0, dropout=0.1, head=args.head)
        if fused == 'snv' or fused not in {'pc', 'kmer'}:
            feature_labels = ['snv']
    elif fnum==2:
        encoder_names = ['encoder1', 'encoder2']
        if re.search('pc',fused) and  re.search('kmer',fused):
            x_train1, token_size1 = x_train3,  token_size3
            x_val1,  token_size_val1 = x_val3, token_size_val3
            tsize1 = tsize3
            lss1=lss3
            feature_labels = ['kmer', 'pc']
        elif re.search('snv', fused) and re.search('kmer', fused):
            x_train2, token_size2 = x_train3,token_size3
            x_val2,token_size_val2 = x_val3, token_size_val3
            tsize2 = tsize3
            lss2=lss3
            feature_labels = ['snv', 'kmer']
        else:
            feature_labels = ['snv', 'pc']
        model = Transformer_without_pos_multimodal_add_attn_only2.Transformer(src_vocab_size_1=tsize1,src_vocab_size_2=tsize2,max_length_1=lss1,max_length_2=lss2,device=device,src_pad_idx=0,dropout=0.1,head=args.head)
    else:
        encoder_names = ['encoder1', 'encoder2', 'encoder3']
        feature_labels = ['snv', 'pc', 'kmer']
        model=Transformer_without_pos_multimodal_add_attn.Transformer(src_vocab_size_1=tsize1,src_vocab_size_2=tsize2,src_vocab_size_3=tsize3,max_length_1=lss1,max_length_2=lss2,max_length_3=lss3,device=device,src_pad_idx=0,dropout=0.1,head=args.head)
    optimizer=optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCEWithLogitsLoss()


    x_train1=x_train1.astype(int)
    #x_val1=x_val1.astype(int)
    x_train2=x_train2.astype(int)
    #x_val2=x_val2.astype(int)
    x_train3=x_train3.astype(int)
    if has_validation:
        x_val1=x_val1.astype(int)
        x_val2=x_val2.astype(int)
        x_val3=x_val3.astype(int)



    max_f1=0
    max_auc=0
    #f1_test=0
    model.to(device)

    if has_validation:
        x_val1=remove_new_ele(x_train1,x_val1)
        x_val2=remove_new_ele(x_train2,x_val2)
        x_val3=remove_new_ele(x_train3,x_val3)

    train_loader=return_batch(x_train1,x_train2,x_train3,y_train,flag=True)
    if has_validation:
        test_loader=return_batch(x_val1,x_val2,x_val3,y_val,flag=False)
    valid_losses = []
    if torch.cuda.device_count() > 1:
        print(f" Use {torch.cuda.device_count()} GPUs!\n",flush=True)
        model=nn.DataParallel(model)
        model.to(device)
    # Early stopping used to be created (and checked) only under -s 0, so with the default -s 1 it
    # never ran and every training went the full --epochs. Both modes stop now, each on the quantity
    # it selects the checkpoint by: validation loss under -s 0, validation F1 under -s 1.
    early_stopping = EarlyStopping(patience=patience or 10**9, verbose=True) if sm == 0 else None
    f1_stall = 0
    log(f"learning rate {lr}, up to {epoch_num} epochs, early stopping on "
        + ("validation loss" if sm == 0 else "validation F1")
        + (f" with patience {patience}" if patience else " disabled"))
    for epoch in range(epoch_num):
        _=model.train()
        #exit()
        all_pred_train=[]
        all_pred=[]
        test_label=[]
        train_label=[]
        all_logit=[]
        running_loss = 0.0
        for step, (batch_x1,batch_x2,batch_x3, batch_y) in enumerate(train_loader):
            #optimizer.zero_grad()
            #batch_y=batch_y

            sentence1=batch_x1.int()
            sentence2=batch_x2.int()
            sentence3=batch_x3.int()
            #print(model.device,sentence1.device,sentence2.device,sentence3.device)
            #exit()
            if fnum == 1:
                predictions = model(sentence1)
            elif fnum==2:
                predictions, as1, as2= model(sentence1, sentence2)
            else:
                predictions,as1,as2,as3=model(sentence1,sentence2,sentence3)

            #print(predictions.squeeze(1))
            #exit()
            loss=loss_func(predictions.squeeze(1),batch_y.float())
            optimizer.zero_grad()
            loss.backward()
            #accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item()
            #exit()
            #print(as1.grad.shape,as1.shape)
            #cal_attr(as1,as1.grad)
            #exit()
            '''
            for name, module in model.named_modules():
                print(name)
            exit()
            '''
            logit=torch.sigmoid(predictions.squeeze(1)).cpu().detach().numpy()
            pred=[1 if item>0.5 else 0 for item in logit]
            #all_pred+=pred
            #all_logit+=[i for i in logit]
            all_pred_train+=pred
            all_logit += [i for i in logit]
            train_label+=batch_y.tolist()

        acc=accuracy_score(train_label,all_pred_train)
        precision=precision_score(train_label,all_pred_train)
        recall=recall_score(train_label,all_pred_train)
        fscore= 2 * precision * recall / (precision + recall)
        fpr, tpr, thresholds = roc_curve(train_label, all_logit)
        roc = auc(fpr, tpr)
        print(f'Training set || epoch no. {epoch} || Train Loss: {running_loss / (len(train_loader)):.4f} || accuracy: {acc} || precision: {precision} || recall: {recall} || fscore: {fscore} || AUC: {roc}',flush=True)
        print(f'Training set || epoch no. {epoch} || Train Loss: {running_loss / (len(train_loader)):.4f} || accuracy: {acc} || precision: {precision} || recall: {recall} || fscore: {fscore} || AUC: {roc}',file=ol,flush=True)

        _=model.eval()
        '''
        for name, module in model.named_modules():
            print(name)
        exit()
        '''
        eval_fscore = None
        eval_all_pred = []
        eval_all_logit = []
        eval_test_label = []
        es_out = False
        if has_validation:
            with torch.no_grad():
                all_pred=[]
                test_label=[]
                all_logit=[]
                for step, (batch_x1,batch_x2,batch_x3, batch_y) in enumerate(test_loader):
                    sentence1=batch_x1.int()
                    sentence2=batch_x2.int()
                    sentence3=batch_x3.int()
                    if fnum == 1:
                        predictions = model(sentence1)
                    elif fnum == 2:
                        predictions, as1, as2 = model(sentence1, sentence2)
                    else:
                        predictions,as1,as2,as3 = model(sentence1,sentence2,sentence3)
                    #if epoch==epoch_num-1:
                    #print('before_acc:',predictions,flush=True)
                    #predictions = accelerator.gather(predictions)
                    #print('after_acc:',predictions,flush=True)
                    batch_y = batch_y.to(device)
                    loss = loss_func(predictions.squeeze(1), batch_y.float())
                    valid_losses.append(loss.item())
                    logit=torch.sigmoid(predictions.squeeze(1)).cpu().detach().numpy()
                    #print('logit:',logit,flush=True)
                    pred  = [1 if item > 0.5 else 0 for item in logit]
                    #print('pred:',pred,flush=True)
                    all_pred+=pred
                    all_logit+=[i for i in logit]
                    test_label+=batch_y.tolist()
            acc=accuracy_score(test_label,all_pred)
            precision=precision_score(test_label,all_pred)
            recall=recall_score(test_label,all_pred)
            fscore= 2 * precision * recall / (precision + recall)
            fpr, tpr, thresholds = roc_curve(test_label, all_logit)
            roc = auc(fpr, tpr)
            valid_loss = np.average(valid_losses)
            print(f'Validation set || epoch no. {epoch} || Validation Loss: {valid_loss:.4f} || accuracy: {acc} || precision: {precision} || recall: {recall} || fscore: {fscore} || AUC: {roc}',flush=True)
            print(f'Validation set || epoch no. {epoch} || Validation Loss: {valid_loss:.4f} || accuracy: {acc} || precision: {precision} || recall: {recall} || fscore: {fscore} || AUC: {roc}',file=ol,flush=True)
            eval_fscore = fscore
            eval_all_pred = all_pred
            eval_all_logit = all_logit
            eval_test_label = test_label
            if sm==0:
                es_out = early_stopping(valid_loss, model, models_dir, ol)
                if early_stopping.early_stop:
                    log(f"early stopping: validation loss has not improved for {patience} epochs")
                    break
            elif patience:
                # -s 1 keeps the best-F1 checkpoint, so stall on F1 rather than on loss: the loss of
                # an overconfident model climbs while its decisions are unchanged, and stopping on
                # that would cut training off while the F1 is still improving.
                if eval_fscore > max_f1 + 1e-6:
                    f1_stall = 0
                else:
                    f1_stall += 1
                    log(f"validation F1 has not improved for {f1_stall} epoch(s) "
                        f"(best {max_f1:.4f}, patience {patience})")
                    if f1_stall >= patience:
                        log(f"early stopping: validation F1 has not improved for {patience} epochs")
                        break

        #if fscore>max_f1:
        if has_validation and sm==0 and es_out:
            #max_f1=fscore
            with open(odir + '/output_sample_prob_val_loss.txt', 'w') as o2:
                o2.write('Sample_Id\tLabel\tPred\tProb\n')
                c=0
                for e in eval_test_label:
                    o2.write(sid_val[c]+'\t'+str(e)+'\t'+str(eval_all_pred[c])+'\t'+str(eval_all_logit[c])+'\n')
                    c+=1
        if has_validation and eval_fscore is not None and eval_fscore>max_f1:
            max_f1=eval_fscore
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model_f1_score.pt"))
            with open(os.path.join(logs_dir, 'output_sample_prob_val_best_f1.txt'), 'w') as o3:
                o3.write('Sample_Id\tLabel\tPred\tProb\n')
                c=0
                for e in eval_test_label:
                    o3.write(sid_val[c]+'\t'+str(e)+'\t'+str(eval_all_pred[c])+'\t'+str(eval_all_logit[c])+'\n')
                    c+=1
        if acc==1 and epoch>9:
            break
        #if roc>max_auc:
        '''
        if es_out:
            max_auc=roc
            if tm==0:
                with open(os.path.join(logs_dir, 'output_sample_prob_auc.txt'), 'w') as o3:
                    c=0
                    for e in test_label:
                        o3.write(sid_val[c]+'\t'+str(e)+'\t'+str(all_logit[c])+'\n')
                        c+=1
                # if sm == 1:
                #     torch.save(model.state_dict(), odir + "/best_model_auc.pth")
        #break
        '''

        '''
        if  fscore>max_f1:
            max_f1=fscore
            torch.save(model.state_dict(),"Sau_transformer_res_graph_val/best_model.pth")
        '''

        #exit()
    # The checkpoint that is actually kept, not the last-epoch weights.
    ckpt = os.path.join(models_dir, "best_model_f1_score.pt")
    if sm == 0 and os.path.exists(os.path.join(models_dir, "checkpoint_best.pt")):
        ckpt = os.path.join(models_dir, "checkpoint_best.pt")
    # Record which checkpoint was kept, so `predict -m <this directory>` loads the same one instead of
    # the caller having to know the file name and the save-mode rule.  (Written before the '-a 0'
    # exit below, which used to skip it.)
    if os.path.exists(ckpt):
        with open(os.path.join(odir, 'selected_model.txt'), 'w') as _fh:
            _fh.write(os.path.relpath(ckpt, odir) + '\n')

    # Known-mechanism layer: choose, on the validation genomes, the model probability above which a
    # genome with no counted known determinant is called resistant (mechanism_threshold.json).
    if has_validation and os.path.exists(os.path.join(indir, 'mechanism', 'calibration.json')):
        try:
            import mechanism_layer as ML
            _vprobs = ML.val_probs_file(odir)
            _vdir = val_dir or indir
            if _vprobs:
                ML.choose_threshold(indir, _vdir, _vprobs, odir,
                                    note=None if val_dir else
                                    'validation genomes were split from the training database, which '
                                    'was also used to calibrate the determinants')
            else:
                log('known-mechanism layer: no validation probability table; tau left at 0.5')
        except Exception as exc:            # noqa: BLE001 - the model itself is trained and saved
            log(f'WARNING: could not choose the known-mechanism threshold: {type(exc).__name__}: {exc}')

    if atw == 0:
        print("Detect '-a 0'! -> Skip the attribution calculation step...")
        return

    if not has_validation:
        print('No validation/test data available: global attribution only (no held-out analyses).', flush=True)

    if os.path.exists(ckpt):
        TA.unwrap(model).load_state_dict(clean_state_dict(torch.load(ckpt, map_location=device)))
        print(f'Attribution uses checkpoint: {ckpt}', flush=True)
    model.eval()

    train_list = [x_train1, x_train2, x_train3][:fnum]
    eval_list = [x_val1, x_val2, x_val3][:fnum] if has_validation else None
    shap_files, annot_files = IP.find_default_files(indir, feature_labels)
    opts = IP.InterpretOptions(
        method=args.attr_method, steps=args.attr_steps, batch_size=max(1, min(batch_size, 8)),
        max_train=args.attr_max_train, faithfulness=has_validation, sanity=bool(args.attr_sanity),
        shap_files=shap_files, annotation_files=annot_files,
    )
    IP.run(model, encoder_names, feature_labels, train_list, np.asarray(y_train), list(sid_train),
           os.path.join(analysis_dir, 'transformer_attribution'), opts,
           eval_arrays=eval_list, y_eval=np.asarray(y_val) if has_validation else None,
           ids_eval=list(sid_val) if has_validation else None, device=device, indir=indir)


if __name__=="__main__":
    sys.exit(main())
