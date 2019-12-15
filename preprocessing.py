from functools import partial
from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor
import spacy
import re
import collections
from tqdm import tqdm_notebook
from torch import *
import html
import pandas as pd
import numpy as np
from exp.nb_04 import *
from torch import *
from torch.utils.data import Sampler
import re 


def compose(x, funcs, *args, order_key='_order', **kwargs):
    """
    apllies (ordered) functions in funcs sqeuentially to x and return result
    """
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

class ListContainer():
    """
    At simple data structure for creating lists
    """
    def __init__(self, items): self.items = items
    def __getitem__(self, idx):
        try: return self.items[idx]
        except TypeError:
            if isinstance(idx[0],bool):
                assert len(idx)==len(self) # bool mask
                return [o for m,o in zip(idx,self.items) if m]
            return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

class ItemList(ListContainer):
    """
    A listContainer containing items that can be tranformed before accessed 
    """
    def __init__(self, items, path='.', tfms=None):
        super().__init__(items)
        self.path,self.tfms = Path(path),tfms

    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'
    
    def new(self, items, cls=None):
        if cls is None: cls=self.__class__
        return cls(items, self.path, tfms=self.tfms)
    
    def  get(self, i): return i
    def _get(self, i): return compose(self.get(i), self.tfms)
    
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res,list): return [self._get(o) for o in res]
        return self._get(res)

def split_by_func(items, f):
    """
    Splits an list of items into two lists with the spliting func f
    """
    mask = [f(o) for o in items]
    # `None` values will be filtered out
    f = [o for o,m in zip(items,mask) if m==False]
    t = [o for o,m in zip(items,mask) if m==True ]
    return f,t

class SplitData():
    """
    Contains a training and validation list of items
    """
    def __init__(self, train, valid): self.train,self.valid = train,valid
        
    def __getattr__(self,k): return getattr(self.train,k)
    #This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self,data): self.__dict__.update(data) 
    
    @classmethod
    def split_by_func(cls, il, f):
        lists = map(il.new, split_by_func(il.items, f))
        return cls(*lists)

    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'

class Processor():
    """
    Parent class for processors
    """
    def process(self, items): return items

def _label_by_func(ds, f, cls=ItemList): 
    """
    Returns a new ItemList containig the labels of evrey item in list ds using function f 
    """
    return cls([f(o) for o in ds.items], path=ds.path)


class LabeledData():
    """
    Contains label data that have been processed 
    """
    def process(self, il, proc): return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x, self.y = self.process(x, proc_x),self.process(y, proc_y)
        self.proc_x, self.proc_y = proc_x,proc_y
        
    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx): return self.x[idx],self.y[idx]
    def __len__(self): return len(self.x)
    
    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)
    
    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)

def label_by_func(sd, f, proc_x=None, proc_y=None):
    """
    Transform splitted data sd into splitted labled data using splitter f and processes proc_x and proc_y 
    """
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train,valid)

class TextList(ItemList) :
    @classmethod
    def from_df(cls, df, text_col):
        texts = df[text_col]
        texts = texts.values
        return cls(texts)
    @classmethod
    def from_csv(cls,  path, text_col):
        df = pd.read_csv(path)
        return cls.from_df(df, text_col)

def random_splitter(item, pctg=0.2):
    test = np.random.uniform(0, 1)
    if test < pctg :
        return True
    else :
        return False

#special tokens
UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()

def sub_br(t):
    "Replaces the <br /> by \n"
    re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
    return re_br.sub("\n", t)

def spec_add_spaces(t):
    "Add spaces around / and #"
    return re.sub(r'([/#])', r' \1 ', t)

def rm_useless_spaces(t):
    "Remove multiple spaces"
    return re.sub(' {2,}', ' ', t)

def replace_rep(t):
    "Replace repetitions at the character level: cccc -> TK_REP 4 c"
    def _replace_rep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)
    
def replace_wrep(t):
    "Replace word repetitions: word word word -> TK_WREP 3 word"
    def _replace_wrep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '
    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)

def fixup_text(x):
    "Various messy things we've seen in documents"
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
    
default_pre_rules = [fixup_text, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, sub_br]
default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ]

def replace_all_caps(x):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1: res.append(TK_UP); res.append(t.lower())
        else: res.append(t)
    return res

def deal_caps(x):
    "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` before."
    res = []
    for t in x:
        if t == '': continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower(): res.append(TK_MAJ)
        res.append(t.lower())
    return res

def add_eos_bos(x): return [BOS] + x + [EOS]

default_post_rules = [deal_caps, replace_all_caps, add_eos_bos]

def parallel(func, arr, max_workers=4):
    """
    Applies in parallel the func to the elements of arr
    """
    if max_workers<2: results = list(tqdm_notebook(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(tqdm_notebook(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results

class TokenizeProcessor(Processor):
    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4): 
        self.chunksize,self.max_workers = chunksize,max_workers
        self.tokenizer = spacy.blank(lang).tokenizer
        for w in default_spec_tok:
            self.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        i,chunk = args
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items): 
        toks = []
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])
    
    def proc1(self, item): return self.proc_chunk([item])[0]
    
    def deprocess(self, toks): return [self.deproc1(tok) for tok in toks]
    def deproc1(self, tok):    return " ".join(tok)

class NumericalizeProcessor(Processor):
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2): 
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq
    
    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]
            for o in reversed(default_spec_tok):
                if o in self.vocab: self.vocab.remove(o)
                self.vocab.insert(0, o)
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)}) 
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return [self.otoi[o] for o in item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return [self.vocab[i] for i in idx]

class LM_PreLoader():
    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data,self.bs,self.bptt,self.shuffle = data,bs,bptt,shuffle
        total_len = sum([len(t) for t in data.x])
        self.n_batch = total_len // bs
        self.batchify()
    
    def __len__(self): return ((self.n_batch-1) // self.bptt) * self.bs
    
    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return source[seq_idx:seq_idx+self.bptt],source[seq_idx+1:seq_idx+self.bptt+1]
    
    def batchify(self):
        texts = self.data.x
        if self.shuffle: texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([tensor(t) for t in texts])
        self.batched_data = stream[:self.n_batch * self.bs].view(self.bs, self.n_batch)

def get_lm_dls(train_ds, valid_ds, bs, bptt, **kwargs):
    """
    Returns the training and validation language model DataLoaders 
    """
    return (DataLoader(LM_PreLoader(train_ds, bs, bptt, shuffle=True), batch_size=bs, **kwargs),
            DataLoader(LM_PreLoader(valid_ds, bs, bptt, shuffle=False), batch_size=2*bs, **kwargs))


class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))

class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,
        batches[0],batches[max_idx] = batches[max_idx],batches[0]            # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)

def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])

def get_clas_dls(train_ds, valid_ds, bs, **kwargs):
    train_sampler = SortishSampler(train_ds.x, key=lambda t: len(train_ds.x[t]), bs=bs)
    valid_sampler = SortSampler(valid_ds.x, key=lambda t: len(valid_ds.x[t]))
    return (DataLoader(train_ds, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, sampler=valid_sampler, collate_fn=pad_collate, **kwargs))


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
    """
    Loads the databunch stored in path
    """
    if data_type == LMDatabunch :
        train_dl, valid_dl, vocab = pickle.load(open(path, 'rb'))
        return data_type(train_dl, valid_dl, vocab)
    else : 
        train_ds, valid_ds, vocab, bs = pickle.load(open(path, 'rb'))
        train_dl, valid_dl = get_clas_dls(train_ds, valid_ds, bs)
        return data_type(train_dl, valid_dl, bs)