from fastai.datasets import untar_data 
import torch
import torch.nn as nn
from fastai.text import get_language_model, convert_weights
from tqdm import tqdm_notebook, tqdm
import pickle 
from fastai.text import AWD_LSTM
from itertools import chain
from fastai.core import even_mults
from fastai.callback import annealing_cos, annealing_exp, annealing_linear
from typing import Callable, Union
import numpy as np
import pandas as pd
from fastai.layers import CrossEntropyFlat
from preprocessing import *
from fastai.text import get_text_classifier
from pathlib import Path


class Databunch() :
    """
    Container of a Dataloaders for the validation and training datasets 

    Arguments:
        train_dl: The training dataloader (must implement __iter__ method)
        valid_dl: The validaation dataloader (must implement __iter__ method)

    """
    def __init__(self, train_dl, valid_dl, vocab) :
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.vocab = vocab
    
    @property
    def train_ds(self): return self.train_dl.dataset
        
    @property
    def valid_ds(self): return self.valid_dl.dataset  
    
    def save(self, path) :
        pickle.dump((self.train_dl, self.valid_dl, self.vocab), open(path, 'wb'))

class LMDatabunch(Databunch) :

    @classmethod
    def from_csv(cls, path, text_col, pctg, bs=64, bptt=70, vocab=None) :
        tl = TextList.from_csv(path, text_col)
        sd = SplitData.split_by_func(tl, partial(random_splitter, pctg=0.2))
        proc_tok,proc_num = TokenizeProcessor(max_workers=8),NumericalizeProcessor(vocab=vocab)
        ll = label_by_func(sd, lambda x: 0, proc_x = [proc_tok,proc_num])
        train_dl, valid_dl = get_lm_dls(ll.train, ll.valid, bs, bptt)
        return cls(train_dl, valid_dl, proc_num.vocab)

class ClasDatabunch(Databunch) :

    def save(self, path) :
        pickle.dump((self.train_dl.dataset, self.valid_dl.dataset, self.vocab, self.train_dl.batch_size), open(path, 'wb'))
    
    @classmethod
    def from_csv(cls, path, text_col, label_col, pctg, bs=64, vocab=None) :
        df = pd.read_csv(path)
        tl = TextList.from_df(df, text_col)
        sd = SplitData.split_by_func(tl, partial(random_splitter, pctg=0.2))
        tweet_to_label = {}
        it = tqdm_notebook(range(df.shape[0]), total=df.shape[0])
        for i in it : 
            tweet_to_label[df[text_col].iloc[i]] = df[label_col].iloc[i]
        proc_tok,proc_num = TokenizeProcessor(max_workers=8),NumericalizeProcessor(vocab=vocab)
        ll = label_by_func(sd, lambda x: tweet_to_label[x], proc_x = [proc_tok,proc_num])
        train_dl, valid_dl = get_clas_dls(ll.train, ll.valid, bs)
        return cls(train_dl, valid_dl, proc_num.vocab)

def load_data(path, data_type=LMDatabunch) :
    if data_type == LMDatabunch :
        train_dl, valid_dl, vocab = pickle.load(open(path, 'rb'))
        return data_type(train_dl, valid_dl, vocab)
    else : 
        train_ds, valid_ds, vocab, bs = pickle.load(open(path, 'rb'))
        train_dl, valid_dl = get_clas_dls(train_ds, valid_ds, bs)
        return data_type(train_dl, valid_dl, bs)


class Learner():
    """
    Container for a deep learning model, an optimizer, a loss function and the data containing the training and validation 
    with possibility to train the model

    Arguments :
        model (nn.Module): the pytorch model
        opt (torch.optim): the pytorch optimizer
        loss_func (Callable): the loss function 
        data (Databunch): the data containing the training and validation dataloaders
    """
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
    
    def freeze_to(self, n) :
        """
        Freezes the optimizer parameter group up to n
        """
        if n >= len(self.opt.param_groups) :
            raise ValueError(f'The optimizer only has {len(self.opt.param_groups)} parameter groups')
        
        for g in self.opt.param_groups[:n]:
            for l in g['params']:
                l.requires_grad=False
        for g in self.opt.param_groups[n:]: 
            for l in g['params']:
                l.requires_grad=True
    
    def unfreeze(self) :
        """
        Unfreezes the whole parameter groups 
        """
        self.freeze_to(0)

    def save(self, path) :
        state = {'model' : self.model.state_dict(), 'opt' : self.opt.state_dict()}
        torch.save(state, path)
    
    def load(self, path) :
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])


class TextLanguageLearner(Learner) :
    
    def save_encoder(self, path) :
        torch.save(self.model[0].state_dict(), path)
    
    def fit(self, epochs, **kwargs) :
        return fit(epochs, self, lm=True, **kwargs)

    def validate(self, cuda=True) :
        validate(self, cuda, lm=True)
    
class TextClassifierLearner(Learner) :
    
    def fit(self, epochs, **kwargs) :
        return fit(epochs, self, lm=False, **kwargs)
    
    def validate(self, cuda=True) :
        validate(self, cuda, lm=False)

    def add_test(self, test_dl) :
        self.test_dl = test_dl
    
    def predict_test(self) :
        if self.test_dl is None :
            return 
        preds = []
        self.model = self.model.cuda()
        batches = tqdm_notebook(self.test_dl, leave=False,
                    total=len(self.test_dl), desc=f'Predictions')
        for x, _ in batches :
            x = x.cuda()
            pred = self.model(x)[0]
            preds.append(torch.argmax(pred, dim=1))
        preds = torch.cat(preds, dim=0)
        return preds
    
    def make_submission(self, path) :
        preds = self.predict_test()
        preds = (preds - (preds == 0).type(torch.cuda.LongTensor)).tolist()
        sub = pd.DataFrame({'Id' : range(1, len(preds)+1), 'Prediction' : list(preds)})
        sub.to_csv(path)

def load_encoder_clas(model, enc_path):
        model[0].module.load_state_dict(torch.load(enc_path))

def load_pretrained_lm(vocab) :    
    """
    Load fastai's pretrained awd_lstm model
    """
    lm = get_language_model(AWD_LSTM, len(vocab))
    model_path = untar_data('https://s3.amazonaws.com/fast-ai-modelzoo/wt103-1', data=False)
    fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
    old_itos = pickle.load(open(fnames[1], 'rb'))
    old_stoi = {v:k for k,v in enumerate(old_itos)}
    wgts = torch.load(fnames[0], map_location=lambda storage, loc: storage)
    wgts = convert_weights(wgts, old_stoi, vocab)
    lm.load_state_dict(wgts)
    return lm

def get_lang_model_param_groups(language_model) :
    """
    Returns the parameter groups structured by the RNN layers of the language model
    """
    parameters = [] 
    for i in range(3) :
        layer = f'{i}'
        parameters.append({'params' :language_model._modules['0']._modules['rnns']._modules[layer].parameters()})
    modules = chain(language_model._modules['1'].parameters(), language_model._modules['0']._modules['encoder'].parameters())
    parameters.append({'params': modules})
    return parameters

def get_class_model_param_groups(classifier_model) :
    """
    Returns the parameter groups structured by the RNN layers of the classifier model
    """
    parameters = []
    parameters.append({'params' : chain(classifier_model[0].module.encoder.parameters(), classifier_model[0].module.encoder_dp.parameters())})
    for rnn in classifier_model[0].module.rnns :
        parameters.append({'params' : rnn.parameters()})
    parameters.append({'params' : classifier_model[1].parameters()})
    return parameters

def get_language_learner(data, opt_func=torch.optim.Adam, loss_func=CrossEntropyFlat(), lr=0.01) :
    model = load_pretrained_lm(data.vocab)
    opt = opt_func(get_lang_model_param_groups(model), lr=lr)
    return TextLanguageLearner(model, opt, loss_func, data)

def get_classifier_learner(data, enc_path, opt_func=torch.optim.Adam, loss_func=CrossEntropyFlat(), lr=0.01) :
    model = get_text_classifier(AWD_LSTM, len(data.vocab), 2)
    load_encoder_clas(model, enc_path)
    opt = opt_func(get_class_model_param_groups(model), lr=lr)
    return TextClassifierLearner(model, opt, loss_func, data)


def fit(epochs, learn, lm, cuda=True, show_info=True, grad_clip=0.1, alpha=2., beta=1., record=True, one_cycle=True, 
                 max_lr:Union[float,slice]=0.01,  div_factor:float=25., pct_start:float=0.3, final_div:float=None, moms=(0.95, 0.85),
                 annealing:Callable=annealing_cos, notebook=True):
    
    """
    Train the learner for a number of epochs

    Arguments :

        epochs : number of epochs
        learn : the Learner
        cuda : if we train on gpu or not
        show_info : show training and validation loss and accuracy
        grad_clip : use fro gradiant clipping 
        alpha :  activation regularization parameter 
        beta : temporal activation regularization parameter
        record : to record hyperparameters (learning rate and mometnum) and losses
        one_cycle : for using cycling learning rate and momentum
        max_lr : the max learning rate for the cycle (if max_lr is a slice then discriminative learning rate is applied)
        div_factor : factor to divide max_lr to get the starting learning rate for the cycle 
        pct_start : at which fraction of the cycle do we reach max_lr
        final_div : factor to divide max_lr to get the ending learning rate for the cycle
        moms : the maximum and lowest momentum for the cycle 
        annealing : the interpolation function for the learning rate and the momentum 
    """
     #number of batches in one epoch for validation and training data
    train_size = len(learn.data.train_dl)
    valid_size = len(learn.data.valid_dl)
    
    # total iterations and cut used for slanted_triangular learning rates (T and cut from paper)
    total_iterations = epochs*train_size

    if record:
        momentum = [[] for i in range(len(learn.opt.param_groups))]
        lrs_record = [[] for i in range(len(learn.opt.param_groups))]
        train_losses = []
        val_losses =[]
        train_accs = []
        valid_accs =[]
    
    #puts model on gpu
    if cuda :
        learn.model.cuda()
       
    #Start the epoch
    for epoch in range(epochs):
        
        if hasattr(learn.data.train_dl.dataset, "batchify"): learn.data.train_dl.dataset.batchify()

        #loss and accuracy 
        train_loss, valid_loss, train_acc, valid_acc = 0, 0, 0, 0

        #puts the model on training mode (activates dropout)
        learn.model.train()
        
        if notebook :
            batches_train = tqdm_notebook(learn.data.train_dl, leave=False,
                    total=len(learn.data.train_dl), desc=f'Epoch {epoch} training')
        else :
            batches_train = tqdm(learn.data.train_dl, leave=False,
                    total=len(learn.data.train_dl), desc=f'Epoch {epoch} training')

        
        #batch number counter
        batch_num = 0

        learn.model.reset()
       
        #starts sgd for each batches
        for x, y in batches_train:
            
            #cyclical learning rate and momentum
            if one_cycle :
                
                cut = int(total_iterations*pct_start)
                iteration = (epoch * train_size) + batch_num
                
                #next we compute the maximum lrs for each layer of our model, we can use either discriminative
                #learning rate or the same learning rate for each layer
                
                #if we use discriminative learning rates
                if isinstance(max_lr, slice) :
                    max_lrs = even_mults(max_lr.start, max_lr.stop, len(learn.opt.param_groups))
                
                #else we give the same max_lr to every layer of the model
                else :
                    max_lrs = [max_lr for i in range(len(learn.opt.param_groups))]
                
                #the final learning rate division factor
                if final_div is None: final_div = div_factor*1e4
                
                  
                if iteration < cut :
                    lrs = [annealing(lr/div_factor, lr, iteration/cut) for lr in max_lrs]
                    mom = annealing(moms[0], moms[1], iteration/cut) 
                else :
                    lrs = [annealing(lr, lr/final_div, (iteration-cut)/(total_iterations-cut)) for lr in max_lrs]
                    mom = annealing(moms[1], moms[0], (iteration-cut)/(total_iterations-cut))
                
                for i, param_group, lr in zip(range(len(learn.opt.param_groups)), learn.opt.param_groups, lrs) :
                    param_group['lr'] = lr
                    param_group['betas'] = (mom ,param_group['betas'][1])
                    if record :
                        lrs_record[i].append(lr)
                        momentum[i].append(mom)
            
            batch_num+=1

           #forward pass
            if cuda :
                x = x.cuda()
                y = y.cuda()
            pred, raw_out, out = learn.model(x)
            loss = learn.loss_func(pred, y)
            
            #activation regularization 
            if alpha != 0.:  loss += alpha * out[-1].float().pow(2).mean()
            
            #temporal activation regularization 
            if beta != 0.:
                h = raw_out[-1]
                if len(h)>1: loss += beta * (h[:,1:] - h[:,:-1]).float().pow(2).mean()
            
            train_loss += loss
            if lm :
                train_acc += (torch.argmax(pred, dim=2) == y).type(torch.FloatTensor).mean() 
            else :
                train_acc += (torch.argmax(pred, dim=1) == y).type(torch.FloatTensor).mean() 

            # compute gradients and updtape parameters
            loss.backward()
            
            #gradient clipping
            if grad_clip:  nn.utils.clip_grad_norm_(learn.model.parameters(), grad_clip)
            
            #optimizationm step
            learn.opt.step()
            learn.opt.zero_grad()

        train_loss = train_loss/train_size
        train_acc = train_acc/train_size
        

        # putting the model in eval mode so that dropout is not applied
        learn.model.eval()

        if notebook :
            batches_valid = tqdm_notebook(learn.data.valid_dl, leave=False,
                total=len(learn.data.valid_dl), desc=f'Epoch {epoch} validation')
        else : 
            batches_valid = tqdm(learn.data.valid_dl, leave=False,
                total=len(learn.data.valid_dl), desc=f'Epoch {epoch} validation')    
    
        with torch.no_grad():
            for x, y in batches_valid: 
                if cuda :
                    x = x.cuda()
                    y = y.cuda()
                pred = learn.model(x)[0]
                loss = learn.loss_func(pred, y)

                valid_loss += loss
                if lm :
                    valid_acc += (torch.argmax(pred, dim=2) == y).type(torch.FloatTensor).mean() 
                else :
                    valid_acc += (torch.argmax(pred, dim=1) == y).type(torch.FloatTensor).mean() 
                
        valid_loss = valid_loss/valid_size
        valid_acc = valid_acc/valid_size
        
        if show_info :
            print("Epoch {:.0f} training loss : {:.3f}, train accuracy : {:.3f}, validation loss : {:.3f}, valid accuracy : {:.3f}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))
        if record :
            val_losses.append(valid_loss)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
    
    if record :
        return {'train_loss' : train_losses, 'valid_loss' : val_losses, 'train_acc': train_acc, 'valid_acc' : valid_acc, 'lrs' : lrs_record, 'momentums' : momentum}    

  
def validate(learn, cuda=True, lm=True) :
    """
    Computes the validation loss and accuracy of the learner
    """
    valid_size = len(learn.data.valid_dl)
    
    #puts model on gpu
    if cuda :
        learn.model.cuda()
    else :
        learn.model.cpu()
    
    #loss and accuracy 
    valid_loss, valid_acc = 0, 0

    #puts the model on training mode (activates dropout)
    learn.model.train()
        
    # putting the model in eval mode so that dropout is not applied
    learn.model.eval()
    with torch.no_grad():
        batches = tqdm_notebook(learn.data.valid_dl, leave=False,
                total=len(learn.data.valid_dl), desc=f'Validation')
        for x, y in batches: 
            if cuda :
                x = x.cuda()
                y = y.cuda()
            pred = learn.model(x)[0]
            loss = learn.loss_func(pred, y)

            valid_loss += loss
            if lm :
                valid_acc += (torch.argmax(pred, dim=2) == y).type(torch.FloatTensor).mean() 
            else :
                valid_acc += (torch.argmax(pred, dim=1) == y).type(torch.FloatTensor).mean() 
                
    valid_loss = valid_loss/valid_size
    valid_acc = valid_acc/valid_size
        
    print("Loss : {:.3f}, Accuracy : {:.3f}".format(valid_loss, valid_acc))
   
