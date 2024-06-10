# %%
from typing import Union,List,Any,Dict,Optional
import pickle as pkl
import pandas as pd
from torch import nn
import torch
from transformers import EsmModel, EsmConfig, EsmTokenizer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.optim import Adam
import random
import numpy as np
import logging
import time
import os
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import  Dataset
from .util import OrderManager
from copy import deepcopy
from math import ceil
import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# from collections import OrderedDict
from tqdm import tqdm
import sys
import networkx as nx
# from PyPDF2 import PdfReader, PdfWriter
from torch.utils.tensorboard import SummaryWriter

# %%
class ConcatProteinDataset(Dataset):
    def __init__(self,pkl_path:str,order_manager:OrderManager,
                 max_domain:int=15) -> None:
        '''
        for expedience, begin with df's pkl
        TODO fetch directly from neo4j
        '''
        super().__init__()
        self.pkl_path=pkl_path
        self.order_manager=order_manager
        self.data:pd.DataFrame=pd.read_pickle(pkl_path)
        self.data=self.data[~self.data['taxo'].isna()]
        self.max_domain=max_domain
    def __len__(self):
        '''
        BUG: some taxo is missing!
        '''
        return len(self.data)
    
    def __getitem__(self, idx) -> dict:
        name:str=self.data.iloc[idx].name
        taxos:List[str]=self.data.iloc[idx]['taxo'].split(' ;')
        seqs:List[str]=self.data.iloc[idx]['seq'].upper().replace('-','').split('#')
        ls=len(seqs)
        if ls<self.max_domain:
            seqs+=['']*(self.max_domain-ls)
            sentence_mask=[1]*ls+[0]*(self.max_domain-ls)
        elif ls>self.max_domain:
            r=np.random.choice(ls,self.max_domain)
            r.sort()
            seqs=[seqs[i] for i in r]
            sentence_mask=[1]*(self.max_domain)
        else:
            sentence_mask=[1]*(self.max_domain)
        return {'name':name,
                'seq':seqs,
                'taxo':self.order_manager.order_to_idx(taxos[2:9:2]),
                'sentence_mask':sentence_mask}

    def fetch_single(self,idx):
        '''
        get batched single entry
        '''
        entry=self[idx]
        return dict(
        name=(entry['name'],),
        seq=[(s,) for s in entry['seq']],
        taxo=[torch.tensor([i]).long() for i in entry['taxo']],
        sentence_mask=[torch.tensor([i]).long() for i in entry['sentence_mask']])
        
    def fetch_domain_name(self,idx):
        if 'family' not in self.data.columns:
            return ('',)
        else:
            return self.data.iloc[idx]['family']
        
    def name_to_idx(self,name:str):
        return np.where(self.data.index==name)[0][0]