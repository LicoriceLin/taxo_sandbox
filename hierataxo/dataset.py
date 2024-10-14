# %%
from typing import Union,List,Any,Dict,Optional,Literal
import pickle as pkl
import pandas as pd
from torch import nn
import torch
from transformers import EsmModel, EsmConfig, EsmTokenizer
# from torch.nn.modules.loss import _Loss
# import torch.nn.functional as F
# from torch.optim import Adam
# import random
import numpy as np
# import logging
# import time
# import os
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import  Dataset,random_split,DataLoader
from .util import OrderManager
# from copy import deepcopy
# from math import ceil
# import matplotlib.pyplot as plt
# # import matplotlib as mpl
# import matplotlib.colors as mcolors
# from matplotlib.axes import Axes
# from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
# # from collections import OrderedDict
# from tqdm import tqdm
# import sys
# import networkx as nx
# from PyPDF2 import PdfReader, PdfWriter
# from torch.utils.tensorboard import SummaryWriter
import lightning as L
from typing import Callable
# %%

class ConcatProteinDataModule(L.LightningDataModule):
    def __init__(self,
        # order_manager:OrderManager,
        pkl_path:str,
        model_name:str='facebook/esm2_t6_8M_UR50D',
        max_domain:int=15,
        max_length:int=500,
        split_ratio:List[float]=[0.9,0.09,0.01],
        train_bs:int=2,
        infer_bs:int=25,
        order_manager_kwargs:Dict[str,Any]={
                'hierarchical_labels':'taxo_data/hierarchy_order_Riboviria.pkl',
                'level_names':['Kingdom','Phylum','Class','Order'],
                'level_colors':['pinkish red','purply','ocean','peach'],
                'layout_prog':'dot',
                'layout_modification':None,
                },
        test_mode:Literal['all','test','train','valid']='all'
        ):
        super().__init__()
        self.order_manager_kwargs=order_manager_kwargs
        self.pkl_path=pkl_path
        self.model_name=model_name
        self.max_domain=max_domain
        self.max_length=max_length
        self.save_hyperparameters()
        
        self.order_manager=OrderManager(**self.order_manager_kwargs)
        assert len(split_ratio)==3 and sum(split_ratio)==1
        self.split_ratio=split_ratio
        self.train_bs,self.infer_bs=train_bs,infer_bs
        self.test_mode=test_mode
        # self.save_hyperparameters(
        #     'pkl_path','model_name','max_domain','max_length',
        #     'split_ratio','train_bs','infer_bs'
        # )
        # try:
        #     self.save_hyperparameters(ignore=['order_manager'])
        # except:
        #     pass
    # def prepare_data(self) -> None:
    #     pass

    def setup(self,stage:str):
        self.data:pd.DataFrame=pd.read_pickle(self.pkl_path)
        self.data=self.data[~self.data['taxo'].isna()]
        self.dataset=ConcatProteinDataset(self.pkl_path,
            self.order_manager,
            model_name=self.model_name,
            max_length=self.max_length,
            max_domain=self.max_domain)
        # if stage in ['fit','test'] :
        self.trainset,self.valset,self.testset=random_split(self.dataset,self.split_ratio)

    def train_dataloader(self):
        '''
        TODO loader with multiple length & domain padding size
        '''
        return DataLoader(self.trainset, batch_size=self.train_bs, shuffle=True, 
            num_workers=min(self.train_bs,16),collate_fn=self.dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.infer_bs, shuffle=False,
            num_workers=min(self.infer_bs,16),collate_fn=self.dataset.collate_fn)

    def test_dataloader(self):
        if self.test_mode=='all':
            return DataLoader(self.dataset, batch_size=self.infer_bs, shuffle=False,
                num_workers=min(self.infer_bs,16),collate_fn=self.dataset.collate_fn)
        elif self.test_mode=='test':
            return DataLoader(self.testset, batch_size=self.infer_bs, shuffle=False, 
                num_workers=min(self.infer_bs,16),collate_fn=self.dataset.collate_fn)
        elif self.test_mode=='train':
            return DataLoader(self.trainset, batch_size=self.infer_bs, shuffle=False, 
                num_workers=min(self.infer_bs,16),collate_fn=self.dataset.collate_fn)
        elif self.test_mode=='valid':
            return DataLoader(self.valset, batch_size=self.infer_bs, shuffle=False, 
                num_workers=min(self.infer_bs,16),collate_fn=self.dataset.collate_fn)
        
    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.infer_bs, shuffle=False, 
            num_workers=min(self.infer_bs,16),collate_fn=self.dataset.collate_fn)
    

class ConcatProteinDataset(Dataset):
    def __init__(self,pkl_path:str,
        order_manager:OrderManager,
        model_name:str='facebook/esm2_t6_8M_UR50D',
        max_domain:int=15,max_length:int=600,
        ) -> None:
        '''
        for expedience, begin with df's pkl
        TODO fetch directly from neo4j
        '''
        super().__init__()
        self.pkl_path=pkl_path
        self.order_manager=order_manager
        self.data:pd.DataFrame=pd.read_pickle(pkl_path)
        self.data=self.data[~self.data['taxo'].isna()]
        self.max_domain,self.max_length=max_domain,max_length
        self.tokenizer:EsmTokenizer = EsmTokenizer.from_pretrained(model_name)
        self.tokenization:Callable[[List[str]],Dict[str,torch.Tensor]]=lambda ipt:self.tokenizer(ipt,
            return_tensors="pt",padding='max_length',
            truncation=True,max_length=self.max_length)
        
    def __len__(self):
        '''
        BUG: some taxo is missing!
        '''
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str,str|torch.Tensor|List[str]]:
        '''
        dict contents:  
        `name` : str,  
        `seq`  : List[str], padded to max_domain by empty string; 
        `taxo_label` : List[str], so far fixed as taxos[2:9:2];  
        `sentence_mask` : Tensor in size of [max_domain,];  
        `input_ids` :  Tensor in size of [max_domain,max_length]; 1 = valid domain;  
        `attention_mask` : Tensor in size of [max_domain,max_length] 1= valid aa;  
        '''
        name:str=self.data.iloc[idx].name
        taxos:List[str]=self.data.iloc[idx]['taxo'].split(' ;')
        seqs:List[str]=self.data.iloc[idx]['seq'].upper().replace('-','').split('#')
        domain_labels:List[str]=list(self.data.iloc[idx]['family'])
        ls=len(seqs)
        if ls<self.max_domain:
            seqs+=['']*(self.max_domain-ls)
            domain_labels+=['']*(self.max_domain-ls)
            sentence_mask=[1]*ls+[0]*(self.max_domain-ls)
        elif ls>self.max_domain:
            r=np.random.choice(ls,self.max_domain)
            r.sort()
            seqs=[seqs[i] for i in r]
            domain_labels=[domain_labels[i] for i in r]
            sentence_mask=[1]*(self.max_domain)
        else:
            sentence_mask=[1]*(self.max_domain)
        o={'name':name,
            'seq':seqs,
            'taxo_label':taxos[2:9:2],
            'domain_label':domain_labels,
            'taxo':torch.tensor(self.order_manager.order_to_idx(taxos[2:9:2])),
            # 'sentence_embeddings':sentence_embeddings,#torch.hstack(sentence_embeddings)
            'sentence_mask':torch.tensor(sentence_mask)}
        o.update(self.tokenization(seqs))
        return o

    def fetch_single(self,idx):
        '''
        get batched single entry;  
        usually for debug-only usage
        '''
        entry=self[idx]
        for k,v in entry.items():
            if isinstance(v,torch.Tensor):
                entry[k]=torch.unsqueeze(v,0)
            else:
                entry[k]=[v]
        return entry
        # entry=self[idx]
        # return dict(
        # name=(entry['name'],),
        # seq=[(s,) for s in entry['seq']],
        # taxo=[torch.tensor([i]).long() for i in entry['taxo']],
        # sentence_mask=[torch.tensor([i]).long() for i in entry['sentence_mask']])
        
    def fetch_domain_name(self,idx):
        raise RuntimeError('Deprecation! `domain_name` now collected by `__getitem__`')
        if 'family' not in self.data.columns:
            return ('',)
        else:
            return self.data.iloc[idx]['family']
        
    def name_to_idx(self,name:str):
        return np.where(self.data.index==name)[0][0]
    
    @property
    def collate_fn(self):
        if not hasattr(self,'_collate_fn'):
            def c_fn(items:List[Dict[str,Union[str,torch.Tensor,List[str]]]])->Dict[str,torch.Tensor|List[str]]:
                o=dict()
                for k,v in items[0].items():
                    if isinstance(v,torch.Tensor):
                        o[k]=torch.stack([i[k] for i in items])
                    else:
                        o[k]=[i[k] for i in items]
                return o
            self._collate_fn=c_fn
            
        return self._collate_fn
        

    @property
    def collate_fn_single_domain(self):
        return None