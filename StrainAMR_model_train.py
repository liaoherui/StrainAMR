import re
import os
import sys
import argparse
import shutil
import numpy as np

from library import (
    Transformer_without_pos_multimodal_add_attn,
    analyze_attention_matrix_network_optimize_iterate_shap,
    Transformer_without_pos,
    Transformer_without_pos_multimodal_add_attn_only2,
    analyze_attention_matrix_network_optimize_iterate_shap_top,
    shap_feature_select_withcls,
)

import torch
from torch.nn import functional as F
from torch import optim,nn
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_curve, auc
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
        self.val_loss_min = np.Inf
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

def load_token_len(indir):
    f=open(indir+'/longest_len_fs.txt','r')
    line=f.readline()
    line=f.readline().strip()
    ele=line.split('\t')
    return int(ele[0]),int(ele[1]),int(ele[2])

def parsef(ins):
    res=re.split(',',ins)
    return len(res)


def main():
    usage = "StrainAMR_model_train - Takes output folder of StrainAMR_build_train and StrainAMR_build_test as input, and finishes both train and prediction."
    parser = argparse.ArgumentParser(prog="StrainAMR_fold_run.py", description=usage)
    parser.add_argument('-i', '--input_file', dest='input_file', type=str,
                        help="The directory of the input files (output folder of StrainAMR_build_train and StrainAMR_build_test).")
    parser.add_argument('-f', '--feature_used', dest='fused', type=str,
                        help="Choose the feature you wanna use to train the model (e.g. -f kmer,snv means the model will only use these two features for learning). If not set, all three kinds of features will be used. (Default:all)")
    parser.add_argument('-t', '--train_mode', dest='train_mode', type=str,
                        help="If set to 1, then it means there are no test data and only training data exists in the input. (Default: 0).")
    parser.add_argument('-s', '--save_mode', dest='save_mode', type=str,
                        help="If set to 0, then the model with minimum val loss will be saved, otherwise, the model with best performance on val data will be saved in the output dir. (Default: 1).")
    parser.add_argument('-a', '--attention_weight', dest='attn_weight', type=str,
                        help="If set to 0, then will not output the attention weight between tokens. (Default: 1).")
    parser.add_argument('-o', '--outdir', dest='outdir', type=str,
                        help="Output directory of results. (Default: StrainAMR_fold_res)")
    args = parser.parse_args()
    indir = args.input_file
    tm=args.train_mode
    fused=args.fused
    sm=args.save_mode
    atw=args.attn_weight
    odir= args.outdir
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
    ol = open(os.path.join(logs_dir, 'train_pred_log.txt'), 'w')
    #lss1=765
    #lss2=536
    #lss3=1000
    lss1,lss2,lss3=load_token_len(indir)

    x_train1,y_train,yl_train,token_size1,ls,sid_train=process_intsv(indir+'/strains_train_sentence_fs.txt',lss1)
    x_train2,y_train,yl_train,token_size2,ls,sid_train=process_intsv(indir+'/strains_train_pc_token_fs.txt',lss2)
    x_train3,y_train,yl_train,token_size3,ls,sid_train=process_intsv(indir+'/strains_train_kmer_token.txt',lss3)
    if tm==0:
        x_val1,y_val,yl_val,token_size_val1,ls_val,sid_val=process_intsv(indir+'/strains_test_sentence_fs.txt',lss1)
        x_val2,y_val,yl_val,token_size_val2,ls_val,sid_val=process_intsv(indir+'/strains_test_pc_token_fs.txt',lss2)
        x_val3,y_val,yl_val,token_size_val3,ls_val,sid_val=process_intsv(indir+'/strains_test_kmer_token.txt',lss3)

    tsize1=token_size1+2
    print('Token count_type1:',tsize1)
    tsize2=token_size2+2
    print('Token count_type2:',tsize2)
    tsize3=token_size3+2
    print('Token count_type1:',tsize3)
    setup_seed(10)
    if fnum == 1:
        if fused == 'pc':
            x_train1, token_size1 = x_train2, token_size2
            x_val1, token_size_val1 = x_val2, token_size_val2
            tsize1 = tsize2
            lss1 = lss2
        elif fused == 'kmer':
            x_train1, token_size1 = x_train3, token_size3
            x_val1, token_size_val1 = x_val3, token_size_val3
            tsize1 = tsize3
            lss1 = lss3
        model = Transformer_without_pos.Transformer(src_vocab_size=tsize1, max_length=lss1, device=device, src_pad_idx=0, dropout=0.1)
    elif fnum==2:
        if re.search('pc',fused) and  re.search('kmer',fused):
            x_train1, token_size1 = x_train3,  token_size3
            x_val1,  token_size_val1 = x_val3, token_size_val3
            tsize1 = tsize3
            lss1=lss3
        elif re.search('snv', fused) and re.search('kmer', fused):
            x_train2, token_size2 = x_train3,token_size3
            x_val2,token_size_val2 = x_val3, token_size_val3
            tsize2 = tsize3
            lss2=lss3
        model = Transformer_without_pos_multimodal_add_attn_only2.Transformer(src_vocab_size_1=tsize1,src_vocab_size_2=tsize2,max_length_1=lss1,max_length_2=lss2,device=device,src_pad_idx=0,dropout=0.1)
    else:
        model=Transformer_without_pos_multimodal_add_attn.Transformer(src_vocab_size_1=tsize1,src_vocab_size_2=tsize2,src_vocab_size_3=tsize3,max_length_1=lss1,max_length_2=lss2,max_length_3=lss3,device=device,src_pad_idx=0,dropout=0.1)
    optimizer=optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCEWithLogitsLoss()


    x_train1=x_train1.astype(int)
    #x_val1=x_val1.astype(int)
    x_train2=x_train2.astype(int)
    #x_val2=x_val2.astype(int)
    x_train3=x_train3.astype(int)
    if tm==0:
        x_val1=x_val1.astype(int)
        x_val2=x_val2.astype(int)
        x_val3=x_val3.astype(int)



    max_f1=0
    max_auc=0
    #f1_test=0
    model.to(device)

    if tm==0:
        x_val1=remove_new_ele(x_train1,x_val1)
        x_val2=remove_new_ele(x_train2,x_val2)
        x_val3=remove_new_ele(x_train3,x_val3)

    train_loader=return_batch(x_train1,x_train2,x_train3,y_train,flag=True)
    if tm==0:
        test_loader=return_batch(x_val1,x_val2,x_val3,y_val,flag=False)
    at1_train=[]
    at2_train=[]
    at3_train=[]
    at1_test=[]
    at2_test=[]
    at3_test=[]
    valid_losses = []
    if torch.cuda.device_count() > 1:
        print(f" Use {torch.cuda.device_count()} GPUs!\n",flush=True)
        model=nn.DataParallel(model)
        model.to(device)
    if sm==0:
        early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(epoch_num):
        at1_train_tem=[]
        at2_train_tem=[]
        at3_train_tem=[]
        at1_test_tem=[]
        at2_test_tem=[]
        at3_test_tem=[]

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
            if fnum==1 and atw==1:
                at1_train_tem.append(as1.detach().cpu().numpy())
            if fnum==2 and atw==1:
                at1_train_tem.append(as1.detach().cpu().numpy())
                at2_train_tem.append(as2.detach().cpu().numpy())
            if fnum==3 and atw==1:
                at1_train_tem.append(as1.detach().cpu().numpy())
                at2_train_tem.append(as2.detach().cpu().numpy())
                at3_train_tem.append(as3.detach().cpu().numpy())
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
        if tm==0:
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
                    if fnum==1 and atw==1:
                        at1_test_tem.append(as1.detach().cpu().numpy())
                    if fnum==2 and atw==1:
                        at1_test_tem.append(as1.detach().cpu().numpy())
                        at2_test_tem.append(as2.detach().cpu().numpy())
                    if fnum==3 and atw==1:
                        at1_test_tem.append(as1.detach().cpu().numpy())
                        at2_test_tem.append(as2.detach().cpu().numpy())
                        at3_test_tem.append(as3.detach().cpu().numpy())
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
            if sm==0:
                es_out = early_stopping(valid_loss, model, models_dir, ol)
                if early_stopping.early_stop:
                    print("Early stopping!!!")
                    break

        #if fscore>max_f1:
        if sm==0:
            if es_out:
                #max_f1=fscore
                if fnum==1 and atw==1:
                    at1_train = at1_train_tem
                if fnum==2 and atw==1:
                    at1_train = at1_train_tem
                    at2_train = at2_train_tem
                if fnum==3 and atw==1:
                    at1_train=at1_train_tem
                    at2_train=at2_train_tem
                    at3_train=at3_train_tem
                if tm==0:
                    if fnum==1 and atw==1:
                        at1_test = at1_test_tem
                    if fnum==2 and atw==1:
                        at1_test = at1_test_tem
                        at2_test = at2_test_tem
                    if fnum==3 and atw==1:
                        at1_test=at1_test_tem
                        at2_test=at2_test_tem
                        at3_test=at3_test_tem
                    o2 = open(os.path.join(logs_dir, 'output_sample_prob_val_loss.txt'), 'w+')
                    o2.write('Sample_Id\tLabel\tPred\tProb\n')
                    c=0
                    for e in test_label:
                        o2.write(sid_val[c]+'\t'+str(e)+'\t'+str(all_pred[c])+'\t'+str(all_logit[c])+'\n')
                        c+=1
        if fscore>max_f1:
            max_f1=fscore
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model_f1_score.pt"))
            o3 = open(os.path.join(logs_dir, 'output_sample_prob_val_best_f1.txt'), 'w+')
            o3.write('Sample_Id\tLabel\tPred\tProb\n')
            c=0
            for e in test_label:
                o3.write(sid_val[c]+'\t'+str(e)+'\t'+str(all_pred[c])+'\t'+str(all_logit[c])+'\n')
                c+=1
            if sm==1:
                if fnum==1 and atw==1:
                    at1_train = at1_train_tem
                if fnum==2 and atw==1:
                    at1_train = at1_train_tem
                    at2_train = at2_train_tem
                if fnum==3 and atw==1:
                    at1_train=at1_train_tem
                    at2_train=at2_train_tem
                    at3_train=at3_train_tem
                if tm==0:
                    if fnum==1 and atw==1:
                        at1_test = at1_test_tem
                    if fnum==2 and atw==1:
                        at1_test = at1_test_tem
                        at2_test = at2_test_tem
                    if fnum==3 and atw==1:
                        at1_test=at1_test_tem
                        at2_test=at2_test_tem
                        at3_test=at3_test_tem
        if acc==1 and epoch>9:
            break
        #if roc>max_auc:
        '''
        if es_out:
            max_auc=roc
            if tm==0:
                o3 = open(os.path.join(logs_dir, 'output_sample_prob_auc.txt'), 'w+')
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
    ## Not test feature importance for now.
    if atw==0:
        print('Detect \'-a 1\'! -> Skip the attention weight calculation step...')
        exit()
    #cic_train = y_train == all_pred_train
    if atw==1:
        train_at1 = np.vstack(at1_train)[: len(x_train1)]
        train_at2 = np.vstack(at2_train)[: len(x_train1)]
        train_at3 = np.vstack(at3_train)[: len(x_train1)]

    #train_at1=train_at1[cic_train]
    #train_at2=train_at2[cic_train]
    #train_at3=train_at3[cic_train]

    #cic_test = y_val == all_pred
    if tm==0 and atw==1:
        test_at1 = np.vstack(at1_test)[: len(x_val1)]
        test_at2 = np.vstack(at2_test)[: len(x_val1)]
        test_at3 = np.vstack(at3_test)[: len(x_val1)]

    #test_at1=test_at1[cic_test]
    #test_at2=test_at2[cic_test]
    #test_at3=test_at3[cic_test]

    #print(train_at2)
    #exit()
    #shap_top=load_shap(indir+'/')
    pair_pc = os.path.join(shap_dir, 'strains_train_pc_interaction.txt')
    pair_snv = os.path.join(shap_dir, 'strains_train_snv_interaction.txt')
    pair_kmer = os.path.join(shap_dir, 'strains_train_kmer_interaction.txt')

    map_files_all = [
        os.path.join(indir, 'node_token_match.txt'),
        os.path.join(indir, 'pc_matches.txt'),
        os.path.join(indir, 'kmer_token_id.txt'),
    ]

    shap_feature_select_withcls.shap_interaction_select(
        indir + '/strains_train_pc_token_fs.txt',
        pair_pc,
        map_files=[indir + '/pc_matches.txt'],
    )
    shap_feature_select_withcls.shap_interaction_select(
        indir + '/strains_train_sentence_fs.txt',
        pair_snv,
        map_files=[indir + '/node_token_match.txt'],
        rgi_dir=os.path.join(indir, 'rgi_train'),
    )
    shap_feature_select_withcls.shap_interaction_select(
        indir + '/strains_train_kmer_token.txt',
        pair_kmer,
        map_files=[indir + '/kmer_token_id.txt'],
    )
    if fnum==1:
        if fused=='pc':
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                train_at2,
                indir + '/strains_train_pc_token_fs.txt',
                analysis_dir,
                'pc_train',
                os.path.join(indir, 'shap', 'strains_train_pc_token_fs_shap.txt'),
                pair_pc,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
        if fused=='snv':
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                train_at1,
                indir + '/strains_train_sentence_fs.txt',
                analysis_dir,
                'graph_train',
                os.path.join(indir, 'shap', 'strains_train_sentence_fs_shap.txt'),
                pair_snv,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )

        if fused=='kmer':
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                train_at3,
                indir + '/strains_train_kmer_token.txt',
                analysis_dir,
                'kmer_train',
                os.path.join(indir, 'shap', 'strains_train_kmer_token_shap.txt'),
                pair_kmer,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
    if fnum==2:
        if re.search('pc',fused):
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                train_at2,
                indir + '/strains_train_pc_token_fs.txt',
                analysis_dir,
                'pc_train',
                os.path.join(indir, 'shap', 'strains_train_pc_token_fs_shap.txt'),
                pair_pc,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                train_at1,
                indir + '/strains_train_sentence_fs.txt',
                analysis_dir,
                'graph_train',
                os.path.join(indir, 'shap', 'strains_train_sentence_fs_shap.txt'),
                pair_snv,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                train_at3,
                indir + '/strains_train_kmer_token.txt',
                analysis_dir,
                'kmer_train',
                os.path.join(indir, 'shap', 'strains_train_kmer_token_shap.txt'),
                pair_kmer,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
    elif fnum==3:
        analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
            train_at2,
            indir + '/strains_train_pc_token_fs.txt',
            analysis_dir,
            'pc_train',
            os.path.join(indir, 'shap', 'strains_train_pc_token_fs_shap.txt'),
            pair_pc,
            map_files_all,
            rgi_dir=os.path.join(indir, 'rgi_train')
        )
        analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
            train_at1,
            indir + '/strains_train_sentence_fs.txt',
            analysis_dir,
            'graph_train',
            os.path.join(indir, 'shap', 'strains_train_sentence_fs_shap.txt'),
            pair_snv,
            map_files_all,
            rgi_dir=os.path.join(indir, 'rgi_train')
        )
        analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
            train_at3,
            indir + '/strains_train_kmer_token.txt',
            analysis_dir,
            'kmer_train',
            os.path.join(indir, 'shap', 'strains_train_kmer_token_shap.txt'),
            pair_kmer,
            map_files_all,
            rgi_dir=os.path.join(indir, 'rgi_train')
        )
        if tm == 0:
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                test_at2,
                indir + '/strains_test_pc_token_fs.txt',
                analysis_dir,
                'pc_test',
                os.path.join(indir, 'shap', 'strains_train_pc_token_fs_shap.txt'),
                pair_pc,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                test_at1,
                indir + '/strains_test_sentence_fs.txt',
                analysis_dir,
                'graph_test',
                os.path.join(indir, 'shap', 'strains_train_sentence_fs_shap.txt'),
                pair_snv,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )
            analyze_attention_matrix_network_optimize_iterate_shap.obtain_important_tokens(
                test_at3,
                indir + '/strains_test_kmer_token.txt',
                analysis_dir,
                'kmer_test',
                os.path.join(indir, 'shap', 'strains_train_kmer_token_shap.txt'),
                pair_kmer,
                map_files_all,
                rgi_dir=os.path.join(indir, 'rgi_train')
            )


if __name__=="__main__":
    sys.exit(main())
