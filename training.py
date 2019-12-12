from fastai.datasets import untar_data 
import torch
import torch.nn as nn
from fastai.text import get_language_model, convert_weights
from tqdm import tqdm_notebook
import pickle 
from fastai.text import AWD_LSTM
from itertools import chain
from fastai.core import even_mults
from fastai.callback import annealing_cos, annealing_exp, annealing_linear
from typing import Callable, Union
import numpy as np



def load_pretrained_lm(vocab) :    
    lm = get_language_model(AWD_LSTM, len(vocab))
    model_path = untar_data('https://s3.amazonaws.com/fast-ai-modelzoo/wt103-1', data=False)
    fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
    old_itos = pickle.load(open(fnames[1], 'rb'))
    old_stoi = {v:k for k,v in enumerate(old_itos)}
    wgts = torch.load(fnames[0], map_location=lambda storage, loc: storage)
    wgts = convert_weights(wgts, old_stoi, vocab)
    lm.load_state_dict(wgts)
    return lm

class Databunch() :
    def __init__(self, train_dl, valid_dl) :
        self.train_dl = train_dl
        self.valid_dl = valid_dl
    @property
    def train_ds(self): return self.train_dl.dataset
        
    @property
    def valid_ds(self): return self.valid_dl.dataset    

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
    
    def freeze_to(self, n) :
        assert(n < len(self.opt.param_groups))
        for g in self.opt.param_groups[:n]:
            for l in g['params']:
                l.requires_grad=False
        for g in self.opt.param_groups[n:]: 
            for l in g['params']:
                l.requires_grad=True
    def unfreeze(self) :
        self.freeze_to(0)
    def freeze(self) :
        for g in self.opt.param_groups:
            for l in g['params']:
                l.requires_grad=False

def get_model_param_groups(model) :
    parameters = [] 
    for i in range(3) :
        layer = f'{i}'
        parameters.append({'params' :model._modules['0']._modules['rnns']._modules[layer].parameters()})
    modules = chain(model._modules['1'].parameters(), model._modules['0']._modules['encoder'].parameters())
    parameters.append({'params': modules})
    return parameters

def get_class_model_param_groups(model) :
    parameters = []
    parameters.append({'params' : chain(model[0].module.encoder.parameters(), model[0].module.encoder_dp.parameters())})
    for rnn in model[0].module.rnns :
        parameters.append({'params' : rnn.parameters()})
    parameters.append({'params' : model[1].parameters()})
    return parameters

def save_encoder_lm(language_model, path) :
    torch.save(language_model[0].state_dict(), path)

def load_encoder_clas(language_model, path):
    language_model[0].module.load_state_dict(torch.load(path))

def fit_awd_lstm(epochs, learn, cuda=True, show_info=True, grad_clip=0.1, alpha=2., beta=1., record=True, one_cycle=True, 
                 cut_frac = 0.1, n_max = 0.01, ratio=32, discr=True, discr_rate=2.6, lm=True):
    
    #number of batches in one epoch for validation and training data
    train_size = len(learn.data.train_dl)
    valid_size = len(learn.data.valid_dl)
    
    # total iterations and cut used for slanted_triangular learning rates (T and cut from paper)
    total_iterations = epochs*train_size
    cut = int(total_iterations*cut_frac)

    
    if record:
        lrs = []
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
        
        #iterator over all batches in training
        batches = tqdm_notebook(learn.data.train_dl, leave=False,
                        total=len(learn.data.train_dl), desc=f'Epoch {epoch} training')
        
        #batch number counter
        batch_num = 0

        learn.model.reset()
       
        #starts sgd for each batches
        for x, y in batches:
            
            #Slanted_triangular learning rates
            if one_cycle :
                iteration = (epoch * train_size) + batch_num
                assert(total_iterations >= iteration)

                if iteration < cut :
                    p=iteration/cut
                else :
                    p = 1-( (iteration-cut) / (cut*(1/cut_frac-1) ))
                    p = max(p, 0)
                new_lr = n_max*( (1 + p*(ratio-1)) / ratio )
                
                for p in learn.opt.param_groups :
                    p['lr'] = new_lr
                lrs.append(new_lr)
            batch_num+=1

            #disdcriminative learning rate 
            if discr :
                for i in range(3) :#all  3 layers starting from last one 
                    learn.opt.param_groups[-(i+2)]['lr'] = learn.opt.param_groups[-(i+2)]['lr']/ (discr_rate)**i

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
        with torch.no_grad():
            batches = tqdm_notebook(learn.data.valid_dl, leave=False,
                     total=len(learn.data.valid_dl), desc=f'Epoch {epoch} validation')
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
        
        if show_info :
            print("Epoch {:.0f} training loss : {:.3f}, train accuracy : {:.3f}, validation loss : {:.3f}, valid accuracy : {:.3f}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))
        if record :
            val_losses.append(valid_loss)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
    
    if record :
        return {'train_loss' : train_losses, 'valid_loss' : val_losses, 'train_acc': train_acc, 'valid_acc' : valid_acc, 'lrs' : lrs}    




    
def validate(learn, cuda=True, lm=True) :
    
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
   
def fit2(epochs, learn, lm, cuda=True, show_info=True, grad_clip=0.1, alpha=2., beta=1., record=True, one_cycle=True, 
                 max_lr:Union[float,slice]=0.01,  div_factor:float=25., pct_start:float=0.3, final_div:float=None, moms=(0.95, 0.85),
                 annealing:Callable=annealing_cos):
    
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
        
        #iterator over all batches in training
        batches = tqdm_notebook(learn.data.train_dl, leave=False,
                        total=len(learn.data.train_dl), desc=f'Epoch {epoch} training')
        
        #batch number counter
        batch_num = 0

        learn.model.reset()
       
        #starts sgd for each batches
        for x, y in batches:
            
            #cyclical learning rates and momentum
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
        with torch.no_grad():
            batches = tqdm_notebook(learn.data.valid_dl, leave=False,
                     total=len(learn.data.valid_dl), desc=f'Epoch {epoch} validation')
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
        
        if show_info :
            print("Epoch {:.0f} training loss : {:.3f}, train accuracy : {:.3f}, validation loss : {:.3f}, valid accuracy : {:.3f}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))
        if record :
            val_losses.append(valid_loss)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
    
    if record :
        return {'train_loss' : train_losses, 'valid_loss' : val_losses, 'train_acc': train_acc, 'valid_acc' : valid_acc, 'lrs' : lrs_record, 'momentums' : momentum}    
