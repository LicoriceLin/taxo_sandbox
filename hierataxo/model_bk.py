# %%
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler,random_split
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
from captum.attr import IntegratedGradients
from torch.nn.modules.activation import MultiheadAttention
import lightning as L

# TODO implement tensorboard
logger=logging.getLogger()
time_str=time.strftime("%y%m%d-%H%M%S", time.localtime())

class ClassificationHead(nn.Module):
    def __init__(
        self,input_dim:int,inner_dim:int,num_classes:int,activation_fn:str='gelu',pooler_dropout:float=0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = getattr(F,activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HierarESM(L.LightningModule):
    # initialization
    def __init__(self,order_manager:OrderManager,model_name:str='facebook/esm2_t6_8M_UR50D',
                 max_length:int=8000,max_domain:int=15,device:Union[int,str]=0,
                 nhead:int=4,ff_fold:int=4,num_block_layers:int=2,to_freeze:int=0):
        super().__init__()
        self.order_manager=order_manager
        # order_manager.device=device
        self.max_length=max_length
        self.max_domain=max_domain
        # self.device=device
        self.nhead=nhead
        self.ff_fold=ff_fold
        self.num_block_layers=num_block_layers
        
        self.tokenizer:EsmTokenizer = EsmTokenizer.from_pretrained(model_name)
        self.backbone:EsmModel = EsmModel.from_pretrained(model_name)#.to(self.device)
        self.hidden_size:int=self.backbone.config.hidden_size
        # self.dtype=self.backbone.config.torch_dtype
        #TODO a trainable embedding for Family (maybe from the Pfam MSA)
        #https://www.nature.com/articles/s41586-021-03819-2/figures/3
        #TODO positional embedding for "sentence" (maybe related to Hits' begin/end)
        assert to_freeze<self.backbone.config.num_hidden_layers,'no enough layer to freeze!'
        self.to_freeze=to_freeze
        if self.to_freeze>0:
            self.partial_freeze()
        self.make_transformer_hierar_layers()
    
    def partial_freeze(self):
        for name, param in self.backbone.named_parameters():
            if ('backbone' in name ):
                names=name.split('.')
                if name[1]=='embeddings' or (
                    name[1]=='encoder' and (int(names[3]))<self.to_freeze):
                    param.requires_grad = False
        
    def make_transformer_hierar_layers(self):
        #TODO use nn.TransformerEncoder with 2 layers
        self.opt_initiator=nn.TransformerEncoderLayer(
            d_model=self.hidden_size,nhead=self.nhead,
            dim_feedforward=self.hidden_size*self.ff_fold,
            activation='gelu',batch_first=True,
            device=self.device,dtype=self.dtype)
        decoder_layer=nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
            d_model=self.hidden_size,nhead=self.nhead,
            dim_feedforward=self.hidden_size*self.ff_fold,
            activation='gelu',batch_first=True,
            device=self.device,dtype=self.dtype),
            num_layers=self.num_block_layers,
            norm=None
            # norm=nn.LayerNorm(self.hidden_size,eps=1e-5,bias=True)
            )
        for i,l in enumerate(self.order_manager.levels):
            setattr(self,f'decoder_{i+1}',
                    deepcopy(decoder_layer))
            num_classes=len(l)
            # inner_dim=self.hidden_size+int((self.hidden_size*num_classes)**0.5)
            inner_dim=self.hidden_size*ceil(1+num_classes**0.5)
            setattr(self,f'head_{i+1}',ClassificationHead(
                self.hidden_size,inner_dim,num_classes).to(self.device))
            
    # training blocks            
    def parse_sentence(self,sentence:List[str],device:Optional[int]=None
            )->Dict[str,torch.Tensor]:
        if device is None: device=self.device 
        # TODO: when single domain length > max_length, selece a radom length 
        # random.choices with weights
        # pass the position_ids to backbone 
        # (see https://huggingface.co/transformers/v3.2.0/model_doc/bert.html?highlight=position_ids)
        # TODO: when training, random mask some of the amino acids.
        # from transformers import DataCollatorForLanguageModeling
        # (or reload `tokenize` in tokenizer)
        
        ipt=[i.upper().replace('-','').replace('#','<eos>') for i in sentence]
        ipt = self.tokenizer(ipt, return_tensors="pt",padding='max_length',truncation=True,max_length=self.max_length).to(device)
        return ipt
           
    def forward(self, 
            attention_mask:torch.Tensor,
            sentence_mask:torch.Tensor,
            input_ids:Optional[torch.Tensor]=None,
            inputs_embeds:Optional[torch.Tensor]=None,
            **kwargs):
        #ipt:Dict[str,torch.Tensor],sentence_mask:torch.Tensor
        '''
        'attention_mask':[batch_size,max_domain,hidden_size]
        'sentence_mask':[batch_size,max_domain], 
        'input_ids': usually from datasets;
        'inputs_embeds' : usually from integrated ingradient workflows
        
        `sentence_mask`:
            [batch_size*max_domain,]
        
        '''
        # device,dtype=self.device,self.dtype
        device=sentence_mask.device
        if input_ids is not None:
            ipt={'input_ids':input_ids.view(-1,self.max_length),
                'attention_mask':attention_mask.view(-1,self.max_length)
                }
        elif inputs_embeds is not None:
            ipt={'inputs_embeds':inputs_embeds.view(-1,self.max_length,self.hidden_size),
                 'attention_mask':attention_mask.view(-1,self.max_length)}
        else:
            raise ValueError('either `input_ids` or `inputs_embeds` is required')
        
        sentence_mask=sentence_mask.view(-1)
        
        ipt={k:v[sentence_mask.type(torch.bool)] for k,v in ipt.items()}
        ori_ebs:torch.Tensor = F.normalize(self.backbone(**ipt).pooler_output,dim=-1)
        ebs=torch.zeros(sentence_mask.shape[0],ori_ebs.shape[-1],
                        dtype=ori_ebs.dtype,device=device)
        ebs[sentence_mask.bool()]=ori_ebs
        ebs=ebs.view(-1,self.max_domain,ebs.shape[-1])
        sentence_mask=sentence_mask.view(-1,self.max_domain)
        
        #init opt
        bs,ss,es=ebs.shape
        order_count=len(self.order_manager.levels)
        opt_size=ss+order_count
        
        memory_key_padding_mask:torch.Tensor=(1-sentence_mask.float()).bool()

        src_key_padding_mask=torch.zeros(bs,opt_size,device=device).bool()
        src_key_padding_mask[:,:ss]=memory_key_padding_mask
        src_key_padding_mask[:,ss:]=True
        # src_key_padding_mask=src_key_padding_mask.type(torch.bool)
        output=torch.zeros(bs,order_count,es,dtype=ori_ebs.dtype,device=device)
        output=torch.concat((ebs,output),dim=1)
        output=self.opt_initiator(output,src_key_padding_mask=src_key_padding_mask)
        
        #auto-regression
        tgt_mask=torch.triu(torch.ones((opt_size, opt_size), device=device), diagonal=1).bool()
        tgt_key_padding_mask=torch.ones(bs,opt_size,device=device).bool()
        tgt_key_padding_mask[:,:ss]=False
        o:List[torch.Tensor]=[]
        for i,l in enumerate(self.order_manager.levels):
            tgt_key_padding_mask[:,ss+i]=False
            decoder=getattr(self,f'decoder_{i+1}')
            head=getattr(self,f'head_{i+1}')
            output=decoder(tgt=output,memory=ebs,tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)
            o.append(head(output[:,ss+i,:]))
        return o
          
    # inference functions  
    def report_eval(self,sample:Dict[str,Union[List[Union[torch.Tensor,str]],torch.Tensor]]
                    ,pdf:Optional[PdfPages]=None):
        '''
        
        '''
        prev_model_state = self.training
        # batch_size=len(sample['name'])
        self.eval()

        with torch.set_grad_enabled(False):
            batch_name,domains,batch_y,domains_mask=self.process_batch(sample)
            domains_mask=domains_mask.to(self.device)
            ipts=self.parse_sentence(domains)
            x:List[torch.Tensor]=self(ipts['input_ids'].view(-1,self.max_domain,self.max_length),
                                      ipts['attention_mask'].view(-1,self.max_domain,self.max_length),
                                      domains_mask.view(-1,self.max_domain))
            
            y=self.order_manager.idx_to_onehot(batch_y)
            # import pdb;pdb.set_trace()
            for i,name in enumerate(batch_name):
                fig,axes=plt.subplots(3,1,figsize=(16, 40),height_ratios=[16,16,8])
                axes:List[Axes]
                for i_,(d,label) in enumerate([(x,'Pred'),(y,'True')]):
                    self.order_manager.draw_classification_view(
                        color_dict=self.order_manager.distribution_to_color_dict(
                        distribution=d,batch_i=i),ax=axes[i_])
                    axes[i_].set_title(f'{name}-{label}',fontdict={'fontsize':48})
                    # hide_spline(axes[i_])
                self.order_manager.draw_true_probs(x,batch_y,axes[2],batch_i=i)
                axes[2].set_title(f'{name}-True-Probabs',fontdict={'fontsize':48})
                # fig.suptitle(name,fontsize='xx-large')
                fig.tight_layout()
                if pdf is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
                return fig,axes
                # break
            
        if prev_model_state:
            self.train()
            
    def _infer(self,sample:Dict[str,Union[List[Union[torch.Tensor,str]],
                                          torch.Tensor]]):
        '''
        sample: from dataloder or dataset.fetch_single
        check `process_batch` to know the shape of input
        ---
        output:
        x:prob distributes
        y:true indices
        '''
        prev_model_state = self.training
        # batch_size=len(sample['name'])
        self.eval()
            
        with torch.set_grad_enabled(False):
            batch_name,domains,batch_y,domains_mask=self.process_batch(sample)
            domains_mask=domains_mask.to(self.device)
            ipts=self.parse_sentence(domains)
            x:List[torch.Tensor]=[i.to('cpu') for i in self(ipts['input_ids'].view(-1,self.max_domain,self.max_length),
                            ipts['attention_mask'].view(-1,self.max_domain,self.max_length),
                            domains_mask.view(-1,self.max_domain))]
            y=batch_y
            
        if prev_model_state:
            self.train()
        return x,y

    def _embed(self,sample:Dict[str,Union[List[Union[torch.Tensor,str]],torch.Tensor]]):
        '''
        check `process_batch` to know the shape of input
        '''
        prev_model_state = self.training
        # batch_size=len(sample['name'])
        self.eval()
        device,dtype=self.device,self.dtype
        with torch.set_grad_enabled(False):
            batch_name,domains,batch_y,sentence_mask=self.process_batch(sample)
            sentence_mask=sentence_mask.to(self.device)
            ipt=self.parse_sentence(domains)
         
            ipt={k:v[sentence_mask.type(torch.bool)] for k,v in ipt.items()}
            ori_ebs:torch.Tensor = F.normalize(self.backbone(**ipt).pooler_output,dim=-1)
            ebs=torch.zeros(sentence_mask.shape[0],ori_ebs.shape[-1],
                            dtype=dtype,device=device)
            ebs[sentence_mask.bool()]=ori_ebs
            ebs=ebs.view(-1,self.max_domain,ebs.shape[-1])
            sentence_mask=sentence_mask.view(-1,self.max_domain)
            
            #init opt
            bs,ss,es=ebs.shape
            order_count=len(self.order_manager.levels)
            opt_size=ss+order_count
            
            memory_key_padding_mask:torch.Tensor=(1-sentence_mask).bool()

            src_key_padding_mask=torch.zeros(bs,opt_size,device=device).bool()
            src_key_padding_mask[:,:ss]=memory_key_padding_mask
            src_key_padding_mask[:,ss:]=True
            # src_key_padding_mask=src_key_padding_mask.type(torch.bool)
            output=torch.zeros(bs,order_count,es,dtype=dtype,device=device)
            output=torch.concat((ebs,output),dim=1)
            output=self.opt_initiator(output,src_key_padding_mask=src_key_padding_mask)
            
            #auto-regression
            tgt_mask=torch.triu(torch.ones((opt_size, opt_size), device=device), diagonal=1).bool()
            tgt_key_padding_mask=torch.ones(bs,opt_size,device=device).bool()
            tgt_key_padding_mask[:,:ss]=False
            o:List[torch.Tensor]=[]
            for i,l in enumerate(self.order_manager.levels):
                tgt_key_padding_mask[:,ss+i]=False
                decoder=getattr(self,f'decoder_{i+1}')
                # head=getattr(self,f'head_{i+1}')
                output=decoder(tgt=output,memory=ebs,tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask)
                o.append(output[:,ss+i,:])
        if prev_model_state:
            self.train()
        return batch_name,o,batch_y

    def _gradient(self,sample:Dict[str,Union[List[Union[torch.Tensor,str]],torch.Tensor]],
                  n_steps:int=100,internal_batch_size:int=5,bg_token='<mask>')->torch.Tensor:
        prev_model_state = self.training
        self.backbone.embeddings.token_dropout=False
        self.backbone.config.token_dropout=False
        embedding_layer = self.backbone.get_input_embeddings()
        
        input_ids,attention_mask,sentence_mask=self.process_sample(sample)
        fake_ids=torch.clone(input_ids).detach()
        fake_ids[fake_ids>3]=self.tokenizer.get_vocab()[bg_token]
        inputs_embeds=embedding_layer(input_ids)
        fake_inputs_embeds=embedding_layer(fake_ids)
        
        def forward_func(inputs_embeds:torch.Tensor,attention_mask:torch.Tensor,domains_mask:torch.Tensor):
            return self(**{'input_ids':None,'inputs_embeds':inputs_embeds,
                    'attention_mask':attention_mask,'sentence_mask':domains_mask})[-1]
        ig = IntegratedGradients(forward_func)
        
        o_attribute=ig.attribute(inputs_embeds,fake_inputs_embeds,
            target=sample['taxo'][-1].item(),
            internal_batch_size=internal_batch_size,
            n_steps=n_steps,
            additional_forward_args=(attention_mask,sentence_mask))
        
        mapping_gradients=o_attribute[attention_mask.bool()].detach().to('cpu')
        
        self.backbone.embeddings.token_dropout=True
        self.backbone.config.token_dropout=True
        if prev_model_state:
            self.train()
        return mapping_gradients
    
    def _attention(self,multihead_attn:MultiheadAttention,
            sample:Dict[str,Union[List[Union[torch.Tensor,str]],torch.Tensor]]):
        '''
        multihead_attn must be a layer in self
        e.g. self.decoder_4.layers[1].multihead_attn
             self.decoder_1.layers[0].self_attn
        '''
        keeper={}
        def input_hook(module, args,kargs,output):
            keeper['args']=args
            keeper['kargs']=kargs
        hook_handle = multihead_attn.register_forward_hook(input_hook,with_kwargs=True)
        with torch.set_grad_enabled(False):
            _=self(*self.process_sample(sample))
            kargs={}; kargs.update(keeper['kargs'])
            kargs['need_weights']=True
            attention_weight:torch.Tensor=multihead_attn(*keeper['args'],**kargs)[1]
            target_mask:torch.Tensor=~kargs['key_padding_mask']
            query_mask=torch.ones(target_mask.shape[0],attention_weight.shape[1],
                    device=attention_weight.device).bool()
            query_mask[:,:target_mask.shape[1]]=target_mask
            hook_handle.remove()
            return (attention_weight[0,query_mask[0]][:,target_mask[0]],
                    attention_weight,query_mask,target_mask)

    # helpers
    def process_batch(self,batch:Dict[str,Union[List[Union[torch.Tensor,str]],torch.Tensor]]):
        '''
        LEGACY (should be a `collate_fn`)
        process_sample for some inference functions.
        
        dtype:
        batch_name:List[str]
        domains:List[str]
        batch_y:List[torch.Tensor]
        domains_mask: torch.Tensor
        
        reform batch from dataloader into model inputs
        Note: domains_mask/batch_y need to be `.to()` later
        TODO use it alone as a collate_fn in `DataLoader`
        TODO convert domains_mask into onehot with order_manager
        '''
        max_domain=self.max_domain
        batch_name, batch_x, batch_y,sentence_mask = (
            batch['name'], batch['seq'], batch['taxo'],batch['sentence_mask'])
        batch_y:List[torch.Tensor] #dtype=torch.long
        batch_name:List[str]
        batch_size=len(batch_x[0])
        domains:List[str]=[]
        for i in range(batch_size):
            for j in range(max_domain):    
                domains.append(batch_x[j][i])
        domains_mask=torch.stack(sentence_mask).T.reshape(-1)
        return batch_name,domains,batch_y,domains_mask
    
    def process_sample(self,sample:Dict[str,Union[List[Union[torch.Tensor,str]],torch.Tensor]]):
        '''
        LEGACY `collate_fn`
        sample: from DataLoader or self.fetch_single
        output: the input of forward
            input_ids:torch.Tensor,
            attention_mask:torch.Tensor,
            sentence_mask:torch.Tensor
            the first dimension be the batch
        '''
        #
        max_domain,max_length=self.max_domain,self.max_length
        domains=[]
        batch_size=len(sample['seq'][0])
        for i in range(batch_size):
            for j in range(max_domain):    
                domains.append(sample['seq'][j][i])
        domains_mask=torch.stack(sample['sentence_mask']).T.reshape(-1).to(self.device)
        ipts=self.parse_sentence(domains)
        return (ipts['input_ids'].view(-1,max_domain,max_length),
                ipts['attention_mask'].view(-1,max_domain,max_length),
                domains_mask.view(-1,max_domain))
        
class HierarchicalLossNetwork(_Loss):
    '''Logics to calculate the loss of the model.
    '''
    def __init__(self, order_manager:OrderManager, device:Union[str,int]=0, #, hierarchical_labels:dict
                 alpha:float=0.5, beta:float=0.8, p_loss:float=3.,a_incremental:float=1.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        self.order_manager=order_manager
        self.a_incremental=a_incremental
        
    def check_hierarchy(self, current_level, previous_level,numeric_hierarchy:Dict[str,List[str]],rank:Optional[int]=None):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''
        if rank is None: rank=self.device 
        def single_check(c_l,p_l):
            if p_l==0 or c_l==0:
                return 0
            if p_l not in numeric_hierarchy:
                return 1
            if c_l in numeric_hierarchy[p_l]:
                return 1
            else:
                return 0
        bool_tensor = [single_check(current_level[i],previous_level[i]) for i in range(previous_level.size()[0])]
        return torch.FloatTensor(bool_tensor).to(rank)

    def calculate_lloss(self, predictions, true_labels,rank:Optional[int]=None):
        '''Calculates the layer loss.
        '''

        lloss = 0
        alpha=self.alpha
        for l in range(self.order_manager.total_level):
            lloss += alpha* nn.CrossEntropyLoss()(predictions[l], true_labels[l])
            alpha=alpha*self.a_incremental
        return lloss

    def calculate_dloss(self, predictions, true_labels,rank:Optional[int]=None):
        '''Calculate the dependence loss.
        '''
        if rank is None: rank=self.device 
        dloss = 0
        for l in range(1, self.order_manager.total_level):
            current_numeric_hierarchy=self.order_manager.peeled_numeric_hierarchy[l-1]
            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred,current_numeric_hierarchy,prev_lvl_pred.device)
            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(prev_lvl_pred.device), torch.FloatTensor([1]).to(prev_lvl_pred.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(prev_lvl_pred.device), torch.FloatTensor([1]).to(prev_lvl_pred.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
    
    def forward(self, predictions, true_labels,rank:Optional[int]=None):
        '''
        for true labels, please use indice (id) instead of onehots
        '''
        if rank is None: rank=self.device 
        dloss = self.calculate_dloss(predictions, true_labels)
        lloss=self.calculate_lloss(predictions, true_labels)
        return dloss+lloss


# %%
class wrapper_module(nn.Module):
    '''
    helper for tensor board graph logger
    '''
    def __init__(self, inner:nn.Module,*extra_args) -> None:
        super().__init__()
        self.inner=inner
        self.extra_args=extra_args
        
    def forward(self,args):
        return self.inner(args,*self.extra_args)