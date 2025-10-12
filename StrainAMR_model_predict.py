import re
import os
import sys
import argparse
import numpy as np
from library import Transformer_without_pos_multimodal_add_attn,analyze_attention_matrix_network_optimize_iterate_shap,Transformer_without_pos,Transformer_without_pos_multimodal_add_attn_only2,analyze_attention_matrix_network_optimize_iterate_shap_top
from library import utils
from library.token_contribution import export_token_contributions
import torch
from torch.nn import functional as F
from torch import optim,nn
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
import random
import shap

#accelerator = Accelerator()
#device = accelerator.device
lr=0.001
batch_size=20
epoch_num=50
dropout=0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model,odir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,odir)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,odir)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model,odir):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
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
    #line=f.readline()
    c=0
    yp=[]
    mt=0
    samples=[]

    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        #print(ele)
        #exit()
        tk=re.split(',',ele[-1])
        tem=0
        for t in tk:
            #print(t,ele[0])
            if int(t) > mt:
                mt=int(t)
            matrix[c][tem]=int(t)
            tem+=1
            if not int(t)==0:
                dtoken[t]=''
        c+=1
        yp.append(int(ele[1]))
        samples.append(ele[0])
    #print(len(yp))
    yp=np.array(yp)
    y=np.eye(2)[yp]
    return matrix,yp,y,mt,ls,samples

def return_batch(train_sentence1,train_sentence2,train_sentence3,label,flag):
    X_train1=torch.from_numpy(train_sentence1).to(device)
    X_train2=torch.from_numpy(train_sentence2).to(device)
    X_train3=torch.from_numpy(train_sentence3).to(device)
    y_train=torch.from_numpy(label).to(device)
    train_dataset=subDataset(X_train1,X_train2,X_train3, y_train)
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

def load_token_len(indir):
    f=open(indir+'/longest_len_fs.txt','r')
    line=f.readline()
    line=f.readline().strip()
    ele=line.split('\t')
    return int(ele[0]),int(ele[1]),int(ele[2])

def parsef(ins):
    res=re.split(',',ins)
    return len(res)


def load_token_presence_matrix(path):
    samples = []
    sample_tokens = []
    feature_index = {}
    feature_order = []

    with open(path, 'r') as handle:
        header = handle.readline()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if not parts:
                continue
            sample_id = parts[0]
            tokens_field = parts[-1] if len(parts) >= 3 else ''
            tokens = [tok for tok in tokens_field.split(',') if tok and tok != '0']
            samples.append(sample_id)
            sample_tokens.append(tokens)
            for token in tokens:
                if token not in feature_index:
                    feature_index[token] = len(feature_order)
                    feature_order.append(token)

    if not feature_order:
        return np.zeros((len(samples), 0), dtype=np.float32), samples, feature_order

    matrix = np.zeros((len(samples), len(feature_order)), dtype=np.float32)
    for row_idx, tokens in enumerate(sample_tokens):
        for token in tokens:
            col_idx = feature_index[token]
            matrix[row_idx, col_idx] += 1.0

    return matrix, samples, feature_order


def train_regression_explainer(feature_matrix, targets):
    if feature_matrix.size == 0 or len(targets) == 0:
        return None, None

    model = RandomForestRegressor(n_estimators=500, random_state=0)
    model.fit(feature_matrix, targets)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=np.float32)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = float(np.asarray(expected_value).mean())
    else:
        expected_value = float(expected_value)
    return shap_values, expected_value


def write_shap_local_matrix(path, sample_ids, feature_order, shap_values):
    with open(path, 'w', encoding='utf-8') as handle:
        header = ['Samples'] + [str(token) for token in feature_order]
        handle.write(','.join(header) + '\n')
        for idx, sample_id in enumerate(sample_ids):
            row = [sample_id]
            if shap_values.size:
                row.extend(f"{float(val):.6f}" for val in shap_values[idx])
            handle.write(','.join(row) + '\n')


def write_sample_directional_features(
    path,
    sample_ids,
    feature_order,
    shap_values,
    mapping,
    top_k=10,
    positive=True,
    gene_family_map=None,
):
    limit = top_k if top_k and top_k > 0 else None
    with open(path, 'w', encoding='utf-8') as handle:
        header = ['Sample_ID', 'Rank', 'Token_ID', 'Feature']
        include_gene_family = gene_family_map is not None
        if include_gene_family:
            header.append('AMR_Gene_Family')
        header.append('ShapValue')
        handle.write('\t'.join(header) + '\n')
        for idx, sample_id in enumerate(sample_ids):
            values = shap_values[idx]
            order = np.argsort(-values) if positive else np.argsort(values)
            written = 0
            for position in order:
                value = values[position]
                if positive and value <= 0:
                    continue
                if not positive and value >= 0:
                    continue
                token = feature_order[position]
                feature_name = utils.token_to_feature(token, mapping)
                row = [
                    sample_id,
                    str(written + 1),
                    str(token),
                    feature_name,
                ]
                if include_gene_family:
                    row.append(gene_family_map.get(feature_name, 'NA'))
                row.append(f"{float(value):.6f}")
                handle.write('\t'.join(row) + '\n')
                written += 1
                if limit is not None and written >= limit:
                    break
            if written == 0:
                row = [sample_id, '0', 'NA', 'NA']
                if include_gene_family:
                    row.append('NA')
                row.append('0')
                handle.write('\t'.join(row) + '\n')


def write_global_ranking(path, feature_order, shap_values, mapping, gene_family_map=None):
    mean_abs = np.abs(shap_values).mean(axis=0) if shap_values.size else np.array([])
    mean_pos = np.where(shap_values > 0, shap_values, 0).mean(axis=0) if shap_values.size else np.array([])
    mean_neg = np.where(shap_values < 0, shap_values, 0).mean(axis=0) if shap_values.size else np.array([])
    order = np.argsort(-mean_abs)
    include_gene_family = gene_family_map is not None
    with open(path, 'w', encoding='utf-8') as handle:
        header = ['ID', 'Token_ID', 'Feature']
        if include_gene_family:
            header.append('AMR_Gene_Family')
        header.extend(['Shap', 'MeanPositiveContribution', 'MeanNegativeContribution'])
        handle.write('\t'.join(header) + '\n')
        for rank, pos in enumerate(order, start=1):
            token = feature_order[pos]
            feature_name = utils.token_to_feature(token, mapping)
            row = [str(rank), str(token), feature_name]
            if include_gene_family:
                row.append(gene_family_map.get(feature_name, 'NA'))
            row.extend([
                f"{float(mean_abs[pos]):.6f}",
                f"{float(mean_pos[pos]):.6f}",
                f"{float(mean_neg[pos]):.6f}",
            ])
            handle.write('\t'.join(row) + '\n')


def generate_test_shap_reports(
    feature_files,
    feature_labels,
    probabilities,
    sample_ids,
    shap_dir,
    output_dir,
    top_k,
    mapping_files,
    rgi_dir=None,
):
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    prob_map = {sid: float(prob) for sid, prob in zip(sample_ids, probabilities)}

    rgi_map = utils.load_rgi_annotations(rgi_dir) if rgi_dir else {}

    shap_name_map = {
        'snv': 'strains_test_sentence_fs_shap.txt',
        'pc': 'strains_test_pc_token_fs_shap.txt',
        'kmer': 'strains_test_kmer_token_shap.txt',
    }
    shap_local_name_map = {
        'snv': 'strains_test_sentence_fs_shap_local_matrix.txt',
        'pc': 'strains_test_pc_token_fs_shap_local_matrix.txt',
        'kmer': 'strains_test_kmer_token_shap_local_matrix.txt',
    }

    shap_outputs = {}

    for label in feature_labels:
        feature_path = feature_files.get(label)
        if not feature_path or not os.path.exists(feature_path):
            continue

        matrix, matrix_samples, feature_order = load_token_presence_matrix(feature_path)
        if matrix.shape[0] == 0:
            continue

        sample_probs = []
        valid_indices = []
        for idx, sid in enumerate(matrix_samples):
            if sid in prob_map:
                sample_probs.append(prob_map[sid])
                valid_indices.append(idx)

        if not sample_probs:
            continue

        if len(valid_indices) != matrix.shape[0]:
            matrix = matrix[valid_indices]
            matrix_samples = [matrix_samples[idx] for idx in valid_indices]

        shap_values, _ = train_regression_explainer(matrix, sample_probs)
        if shap_values is None:
            continue

        mapping = utils.load_token_mappings(mapping_files.get(label)) if mapping_files else {}
        gene_family_map = rgi_map if label == 'snv' else None

        local_matrix_path = os.path.join(output_dir, f'{label}_test_shap_local_matrix.csv')
        write_shap_local_matrix(local_matrix_path, matrix_samples, feature_order, shap_values)

        pos_path = os.path.join(output_dir, f'{label}_test_top_positive_features.tsv')
        neg_path = os.path.join(output_dir, f'{label}_test_top_negative_features.tsv')
        write_sample_directional_features(
            pos_path,
            matrix_samples,
            feature_order,
            shap_values,
            mapping,
            top_k,
            positive=True,
            gene_family_map=gene_family_map,
        )
        write_sample_directional_features(
            neg_path,
            matrix_samples,
            feature_order,
            shap_values,
            mapping,
            top_k,
            positive=False,
            gene_family_map=gene_family_map,
        )

        global_path = os.path.join(output_dir, f'{label}_test_shap.txt')
        write_global_ranking(global_path, feature_order, shap_values, mapping, gene_family_map=gene_family_map)

        shap_filename = shap_name_map.get(label)
        if shap_filename:
            shap_target = os.path.join(shap_dir, shap_filename)
            if os.path.abspath(global_path) != os.path.abspath(shap_target):
                with open(global_path, 'r', encoding='utf-8') as src, open(shap_target, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            shap_outputs[label] = shap_target
        
        local_filename = shap_local_name_map.get(label)
        if local_filename:
            shap_local_target = os.path.join(shap_dir, local_filename)
            if os.path.abspath(local_matrix_path) != os.path.abspath(shap_local_target):
                with open(local_matrix_path, 'r', encoding='utf-8') as src, open(shap_local_target, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())

        print(
            f'Computed SHAP explanations for {label} test features: '
            f'{len(matrix_samples)} samples, {len(feature_order)} unique tokens.',
            flush=True,
        )

    return shap_outputs


def main():
    usage = "StrainAMR_model_predict - Takes output folder of StrainAMR_build_train and StrainAMR_build_test as input, and finishes prediction of test genomes."
    parser = argparse.ArgumentParser(prog="StrainAMR_fold_run.py", description=usage)
    parser.add_argument('-i', '--input_file', dest='input_file', type=str,
                        help="The directory of the input files (output folder of StrainAMR_build_train and StrainAMR_build_test).")
    parser.add_argument('-f', '--feature_used', dest='fused', type=str,
                        help="Choose the feature you wanna use to train the model (e.g. -f kmer,snv means the model will only use these two features for learning). If not set, all three kinds of features will be used. (Default:all)")
    # parser.add_argument('-t', '--train_mode', dest='train_mode', type=str,
    #                     help="If set to 1, then it means there are no test data and only training data exists in the input. (Default: 0).")
    parser.add_argument('-m', '--model_PATH', dest='model_PATH', type=str,
                        help="The dir of pre-trained models.")
    # parser.add_argument('-a', '--attention_weight', dest='attn_weight', type=str,
    #                     help="If set to 0, then will not output the attention weight between tokens. (Default: 1).")
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        help="Output directory of results. (Default: StrainAMR_fold_res)")
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help="Batch size for evaluation and interpretability export. (Default: 20).")
    parser.add_argument('--shap-top-k', dest='shap_top_k', type=int,
                        help="Number of top positive/negative features to record per sample for SHAP outputs (default: 10).",
                        default=10)
    parser.add_argument('--db', dest='db_dir', type=str,
                        help="Directory containing the training database artifacts (default: use the input directory).")
    args = parser.parse_args()
    if not args.input_file:
        raise ValueError('Please provide the directory with test features via -i/--input_file.')
    indir = os.path.abspath(args.input_file)
    #tm=args.train_mode
    fused=args.fused
    #sm=args.save_model
    #atw=args.attn_weight
    odir= args.outdir
    model_PATH=args.model_PATH
    train_dir = args.db_dir if args.db_dir else indir
    train_dir = os.path.abspath(train_dir)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training database directory not found: {train_dir}")
    if not os.path.isdir(indir):
        raise FileNotFoundError(f"Input feature directory not found: {indir}")
    global batch_size
    # if not atw:
    #     atw=1
    # else:
    #     atw=int(atw)
    # if not sm:
    #     sm=0
    # else:
    #     sm=int(sm)
    # if not tm:
    #     tm=0
    # else:
    #     tm=int(tm)
    if not fused:
        fused='all'
        fnum=3
    else:
        fnum=parsef(fused)
    if not odir:
        odir='StrainAMR_fold_res'
    odir = os.path.abspath(odir)

    if not os.path.exists(odir):
        os.makedirs(odir)

    shap_dir = os.path.join(odir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)
    input_shap_dir = os.path.join(indir, 'shap')
    if not os.path.isdir(input_shap_dir):
        input_shap_dir = ""
    train_shap_dir = os.path.join(train_dir, 'shap')
    if not os.path.isdir(train_shap_dir):
        train_shap_dir = ""

    logs_dir = os.path.join(odir, 'logs')
    analysis_dir = os.path.join(odir, 'analysis')

    for d in (logs_dir,analysis_dir):
        os.makedirs(d, exist_ok=True)
        
    ol=open(logs_dir +'/samples_pred_log.txt','w')

    if args.batch_size:
        batch_size = args.batch_size
    shap_top_k = args.shap_top_k
    if shap_top_k is not None and shap_top_k <= 0:
        shap_top_k = None

    def resolve_shap_file(*relative_paths):
        search_roots = [shap_dir]
        if input_shap_dir:
            search_roots.append(input_shap_dir)
        if train_shap_dir:
            search_roots.append(train_shap_dir)
        search_roots.append(indir)
        if train_dir not in search_roots:
            search_roots.append(train_dir)
        for rel_path in relative_paths:
            if not rel_path:
                continue
            for root_path in search_roots:
                candidate = os.path.join(root_path, rel_path)
                if os.path.exists(candidate):
                    return candidate
        return ""
    #lss1=765
    #lss2=536
    #lss3=1000
    lss1,lss2,lss3=load_token_len(train_dir)

    x_train1,y_train,yl_train,token_size1,ls,sid_train=process_intsv(train_dir+'/strains_train_sentence_fs.txt',lss1)
    x_train2,y_train,yl_train,token_size2,ls,sid_train=process_intsv(train_dir+'/strains_train_pc_token_fs.txt',lss2)
    x_train3,y_train,yl_train,token_size3,ls,sid_train=process_intsv(train_dir+'/strains_train_kmer_token.txt',lss3)

    x_val1,y_val,yl_val,token_size_val1,ls_val,sid_val=process_intsv(indir+'/strains_test_sentence_fs.txt',lss1)
    x_val2,y_val,yl_val,token_size_val2,ls_val,sid_val=process_intsv(indir+'/strains_test_pc_token_fs.txt',lss2)
    x_val3,y_val,yl_val,token_size_val3,ls_val,sid_val=process_intsv(indir+'/strains_test_kmer_token.txt',lss3)
    test_labels_available = np.isin(y_val, [0, 1]).all()

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
        else:
            feature_labels = ['snv']
        model = Transformer_without_pos.Transformer(src_vocab_size=tsize1, max_length=lss1, device=device, src_pad_idx=0, dropout=0.1)
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
        model = Transformer_without_pos_multimodal_add_attn_only2.Transformer(src_vocab_size_1=tsize1,src_vocab_size_2=tsize2,max_length_1=lss1,max_length_2=lss2,device=device,src_pad_idx=0,dropout=0.1)
    else:
        encoder_names = ['encoder1', 'encoder2', 'encoder3']
        feature_labels = ['snv', 'pc', 'kmer']
        model=Transformer_without_pos_multimodal_add_attn.Transformer(src_vocab_size_1=tsize1,src_vocab_size_2=tsize2,src_vocab_size_3=tsize3,max_length_1=lss1,max_length_2=lss2,max_length_3=lss3,device=device,src_pad_idx=0,dropout=0.1)
    #optimizer=optim.Adam(model.parameters(), lr=lr)
    #loss_func = nn.BCEWithLogitsLoss()
    # Load the model
    model.load_state_dict(torch.load(model_PATH, map_location=device))


    x_train1=x_train1.astype(int)
    #x_val1=x_val1.astype(int)
    x_train2=x_train2.astype(int)
    #x_val2=x_val2.astype(int)
    x_train3=x_train3.astype(int)

    x_val1=x_val1.astype(int)
    x_val2=x_val2.astype(int)
    x_val3=x_val3.astype(int)



    max_f1=0
    max_auc=0
    #f1_test=0
    #check=9


    x_val1=remove_new_ele(x_train1,x_val1)
    x_val2=remove_new_ele(x_train2,x_val2)
    x_val3=remove_new_ele(x_train3,x_val3)

    train_loader=return_batch(x_train1,x_train2,x_train3,y_train,flag=True)

    test_loader=return_batch(x_val1,x_val2,x_val3,y_val,flag=False)
    at1_train=[]
    at2_train=[]
    at3_train=[]
    at1_test=[]
    at2_test=[]
    at3_test=[]
    valid_losses = []
    #early_stopping = EarlyStopping(patience=20, verbose=True)
    if True:
        at1_train_tem=[]
        at2_train_tem=[]
        at3_train_tem=[]
        at1_test_tem=[]
        at2_test_tem=[]
        at3_test_tem=[]
        model.to(device)
        all_pred_train=[]
        all_pred=[]
        test_label=[]
        train_label=[]
        all_logit=[]
        running_loss = 0.0
        _=model.eval()
        '''
        for name, module in model.named_modules():
            print(name)
        exit()
        '''
        if True:
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
                        at1_test_tem.append(as1.detach().cpu().numpy())
                        at2_test_tem.append(as2.detach().cpu().numpy())
                    else:
                        predictions,as1,as2,as3 = model(sentence1,sentence2,sentence3)
                        at1_test_tem.append(as1.detach().cpu().numpy())
                        at2_test_tem.append(as2.detach().cpu().numpy())
                        at3_test_tem.append(as3.detach().cpu().numpy())

                    batch_y = batch_y.to(device)
                    logit=torch.sigmoid(predictions.squeeze(1)).cpu().detach().numpy()
                    pred  = [1 if item > 0.5 else 0 for item in logit]
                    all_pred+=pred
                    all_logit+=[i for i in logit]
                    test_label+=batch_y.tolist()
            if test_labels_available:
                try:
                    test_label_arr = np.array(test_label, dtype=int)
                    acc=accuracy_score(test_label_arr,all_pred)
                    precision=precision_score(test_label_arr,all_pred)
                    recall=recall_score(test_label_arr,all_pred)
                    fscore= 2 * precision * recall / (precision + recall)
                    fpr, tpr, thresholds = roc_curve(test_label_arr, all_logit)
                    roc = auc(fpr, tpr)
                    print(f'Test set || accuracy: {acc} || precision: {precision} || recall: {recall} || fscore: {fscore} || AUC: {roc}',flush=True)
                    print(f'Test set || accuracy: {acc} || precision: {precision} || recall: {recall} || fscore: {fscore} || AUC: {roc}',file=ol)
                except ValueError as exc:
                    msg = f'Unable to compute evaluation metrics: {exc}'
                    print(msg, flush=True)
                    print(msg, file=ol)
            else:
                msg = 'Test labels are missing or outside the expected range. Skipping evaluation metrics.'
                print(msg, flush=True)
                print(msg, file=ol)

        if fnum == 3:
            test_at1 = np.vstack(at1_test_tem)
            test_at2 = np.vstack(at2_test_tem)
            test_at3 = np.vstack(at3_test_tem)
            pc_file = os.path.join(indir, 'strains_test_pc_token_fs.txt')
            snv_file = os.path.join(indir, 'strains_test_sentence_fs.txt')
            kmer_file = os.path.join(indir, 'strains_test_kmer_token.txt')
            pc_shap = os.path.join(shap_dir, 'strains_test_pc_token_fs_shap.txt')
            snv_shap = os.path.join(shap_dir, 'strains_test_sentence_fs_shap.txt')
            kmer_shap = os.path.join(shap_dir, 'strains_test_kmer_token_shap.txt')
            pair_pc = os.path.join(shap_dir, 'strains_test_pc_interaction.txt')
            pair_snv = os.path.join(shap_dir, 'strains_test_snv_interaction.txt')
            pair_kmer = os.path.join(shap_dir, 'strains_test_kmer_interaction.txt')
            '''
            shap_feature_select_withcls.shap_select(
                pc_file, pc_shap, [os.path.join(train_dir, 'pc_matches.txt')]
            )
            shap_feature_select_withcls.shap_select(
                snv_file, snv_shap, [os.path.join(train_dir, 'node_token_match.txt')],
                rgi_dir=os.path.join(train_dir, 'rgi_train')
            )
            shap_feature_select_withcls.shap_select(
                kmer_file, kmer_shap, [os.path.join(train_dir, 'kmer_token_id.txt')]
            )
            shap_feature_select_withcls.shap_interaction_select(
                pc_file,
                pair_pc,
                map_files=[os.path.join(train_dir, 'pc_matches.txt')],
            )
            shap_feature_select_withcls.shap_interaction_select(
                snv_file,
                pair_snv,
                map_files=[os.path.join(train_dir, 'node_token_match.txt')],
                rgi_dir=os.path.join(train_dir, 'rgi_train'),
            )
            shap_feature_select_withcls.shap_interaction_select(
                kmer_file,
                pair_kmer,
                map_files=[os.path.join(train_dir, 'kmer_token_id.txt')],
            )

            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                test_at2,
                pc_file,
                analysis_dir,
                'pc_predict',
                pc_shap,
                pair_pc,
                map_files=[os.path.join(train_dir, 'pc_matches.txt')],

            )
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                test_at1,
                snv_file,
                analysis_dir,
                'graph_predict',
                snv_shap,
                pair_snv,
                map_files=[os.path.join(train_dir, 'node_token_match.txt')],
                rgi_dir=os.path.join(train_dir, 'rgi_train'),

            )
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                test_at3,
                kmer_file,
                analysis_dir,
                'kmer_predict',
                kmer_shap,
                pair_kmer,
                map_files=[os.path.join(train_dir, 'kmer_token_id.txt')],

            )
            '''

        feature_file_map = {
            'snv': os.path.join(indir, 'strains_test_sentence_fs.txt'),
            'pc': os.path.join(indir, 'strains_test_pc_token_fs.txt'),
            'kmer': os.path.join(indir, 'strains_test_kmer_token.txt'),
        }
        mapping_file_map = {
            'snv': [
                os.path.join(train_dir, 'node_token_match.txt'),
                os.path.join(train_dir, 'feature_remain_graph.txt'),
            ],
            'pc': [
                os.path.join(train_dir, 'pc_matches.txt'),
                os.path.join(train_dir, 'feature_remain_pc.txt'),
            ],
            'kmer': [os.path.join(train_dir, 'kmer_token_id.txt')],
        }
        shap_outputs = generate_test_shap_reports(
            feature_file_map,
            feature_labels,
            np.array(all_logit, dtype=float),
            sid_val,
            shap_dir,
            shap_dir,
            shap_top_k,
            mapping_file_map,
            rgi_dir=os.path.join(train_dir, 'rgi_train'),
        )

        o2 = open(os.path.join(logs_dir, 'output_sample_prob_predict.txt'), 'w+')
        o2.write('Sample_ID\tLabel\tPred\tProb\n')
        c=0
        for e in test_label:
            try:
                label_value = int(e)
            except (TypeError, ValueError):
                label_value = e
            o2.write(sid_val[c]+'\t'+str(label_value)+'\t'+str(all_pred[c])+'\t'+str(all_logit[c])+'\n')
            c+=1
        o2.close()
        '''
        shap_file_map = {
            'snv': resolve_shap_file('strains_test_sentence_fs_shap.txt', 'strains_train_sentence_fs_shap.txt'),
            'pc': resolve_shap_file('strains_test_pc_token_fs_shap.txt', 'strains_train_pc_token_fs_shap.txt'),
            'kmer': resolve_shap_file('strains_test_kmer_token_shap.txt', 'strains_train_kmer_token_shap.txt')
        }
        '''
        
        shap_file_map = {}
        shap_candidates = {
            'snv': [
                shap_outputs.get('snv'),
                os.path.join(shap_dir, 'strains_test_sentence_fs_shap.txt'),
                os.path.join(input_shap_dir, 'strains_test_sentence_fs_shap.txt') if input_shap_dir else "",
                os.path.join(train_shap_dir, 'strains_train_sentence_fs_shap.txt') if train_shap_dir else "",
                os.path.join(input_shap_dir, 'strains_train_sentence_fs_shap.txt') if input_shap_dir else "",
                os.path.join(train_dir, 'strains_train_sentence_fs_shap.txt'),
            ],
            'pc': [
                shap_outputs.get('pc'),
                os.path.join(shap_dir, 'strains_test_pc_token_fs_shap.txt'),
                os.path.join(input_shap_dir, 'strains_test_pc_token_fs_shap.txt') if input_shap_dir else "",
                os.path.join(train_shap_dir, 'strains_train_pc_token_fs_shap.txt') if train_shap_dir else "",
                os.path.join(input_shap_dir, 'strains_train_pc_token_fs_shap.txt') if input_shap_dir else "",
                os.path.join(train_dir, 'strains_train_pc_token_fs_shap.txt'),
            ],
            'kmer': [
                shap_outputs.get('kmer'),
                os.path.join(shap_dir, 'strains_test_kmer_token_shap.txt'),
                os.path.join(input_shap_dir, 'strains_test_kmer_token_shap.txt') if input_shap_dir else "",
                os.path.join(train_shap_dir, 'strains_train_kmer_token_shap.txt') if train_shap_dir else "",
                os.path.join(input_shap_dir, 'strains_train_kmer_token_shap.txt') if input_shap_dir else "",
                os.path.join(train_dir, 'strains_train_kmer_token_shap.txt'),
            ],
        }
        for key, candidates in shap_candidates.items():
            chosen = ""
            for path in candidates:
                if path and os.path.exists(path):
                    chosen = path
                    break
            shap_file_map[key] = chosen

        relevant_shap = {label: shap_file_map.get(label, "") for label in feature_labels}
        if any(os.path.exists(path) for path in relevant_shap.values() if path):
            feature_tensors = []
            if fnum >= 1:
                feature_tensors.append(torch.from_numpy(x_val1))
            if fnum >= 2:
                feature_tensors.append(torch.from_numpy(x_val2))
            if fnum == 3:
                feature_tensors.append(torch.from_numpy(x_val3))

            annotation_file_map = {
                'snv': [os.path.join(train_dir, 'feature_remain_graph.txt')],
                'pc': [os.path.join(train_dir, 'feature_remain_pc.txt')],
                'kmer': [os.path.join(train_dir, 'kmer_token_id.txt')],
            }
            relevant_annotations = {
                label: annotation_file_map.get(label, []) for label in feature_labels
            }
            try:
                export_token_contributions(
                    model,
                    feature_tensors,
                    feature_labels,
                    relevant_shap,
                    analysis_dir,
                    encoder_names,
                    device=device,
                    batch_size=batch_size,
                    top_k_pairs=20,
                    subset_sizes=(2, 3),
                    subset_sample_size=32,
                    sample_ids=sid_val,
                    annotation_file_map=relevant_annotations,
                )
                print('Saved interpretability summaries for test predictions.', flush=True)
            except Exception as exc:
                print(f'Failed to export interpretability summaries: {exc}', flush=True)
        else:
            print('No SHAP reference files found. Skipping interpretability export.', flush=True)



if __name__=="__main__":
    sys.exit(main())
