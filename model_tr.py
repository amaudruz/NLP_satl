from training import *
from preprocessing import *
from fastai.layers import CrossEntropyFlat
import pickle
from fastai.text import get_text_classifier
import dill
from fastai.basic_data import load_data
import matplotlib.pyplot as plt 

def save_learner(learner, path) :
    state = {'model' : learner.model.state_dict(), 'opt' : learner.opt.state_dict()}
    torch.save(state, path)

def load_learner(learner, path) :
    state = torch.load(path)
    learn.model.load_state_dict(state['model'])
    learn.opt.load_state_dict(state['opt'])

if __name__ == "__main__":
    ll = pickle.load(open('data/ll_clas.pkl', 'rb'))
    vocab = pickle.load(open('data/vocab_lm.pkl', 'rb'))
    traind_dl, valid_dl = get_clas_dls(ll.train, ll.valid, 64)
    data = Databunch(traind_dl, valid_dl)
    model = get_text_classifier(AWD_LSTM, len(vocab), 2)
    load_encoder_clas(model, 'data/my_encoders/lm_1enco.pth')
    model = model.cuda()
    lr = 0.01
    opt = torch.optim.Adam(get_class_model_param_groups(model), lr=lr)
    loss_func = CrossEntropyFlat()
    learn = Learner(model, opt, loss_func, data)
    load_learner(learn, 'data/models/my_class_4epoch.pth')
    info = fit2(8, learn, max_lr= slice(2e-3/10, 2e-3), lm=False, notebook=False)
    save_learner(learn, 'data/models/my_class_4_8epoch.pth')
