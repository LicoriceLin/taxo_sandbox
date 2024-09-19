from typing import Union,List,Any,Dict,Optional
import pickle as pkl
import pandas as pd
import numpy as np

from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import LambdaLR,ExponentialLR,ChainedScheduler
from transformers import EsmModel, EsmConfig, EsmTokenizer

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import logging
import time
from tqdm import tqdm
import sys
import networkx as nx
from copy import deepcopy
from math import ceil
from itertools import chain

from transformers import EsmModel, EsmConfig, EsmTokenizer
from captum.attr import IntegratedGradients
from torch.nn.modules.activation import MultiheadAttention
import lightning as L

from .util import OrderManager

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
    
class HierarchicalLossNetwork(nn.Module):
    '''Logics to calculate the loss of the model.
    '''
    def __init__(self, order_manager:OrderManager,
                 alpha:float=0.5, beta:float=0.8, 
                 p_loss:float=3.,a_incremental:float=1.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.order_manager=order_manager
        self.a_incremental=a_incremental
        self.crossentropy=nn.CrossEntropyLoss()

    def check_hierarchy(self, current_level, previous_level,numeric_hierarchy:Dict[str,List[str]]):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''
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
        return torch.FloatTensor(bool_tensor)

    def calculate_lloss(self, predictions:List[torch.Tensor], true_labels:torch.Tensor)->torch.Tensor:
        '''Calculates the layer loss.
        '''

        lloss = 0
        alpha=self.alpha
        for l in range(self.order_manager.total_level):
            lloss += alpha*self.crossentropy(predictions[l], true_labels[:,l])
            alpha=alpha*self.a_incremental
        return lloss

    def calculate_dloss(self, predictions:List[torch.Tensor], true_labels:torch.Tensor)->torch.Tensor:
        '''Calculate the dependence loss.
        '''
        dloss = 0
        for l in range(1, self.order_manager.total_level):
            current_numeric_hierarchy=self.order_manager.peeled_numeric_hierarchy[l-1]
            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)
            # import pdb;pdb.set_trace()
            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred,current_numeric_hierarchy).to(prev_lvl_pred.device)
            l_prev = torch.where(prev_lvl_pred == true_labels[:,l-1], torch.FloatTensor([0]).to(prev_lvl_pred.device), torch.FloatTensor([1]).to(prev_lvl_pred.device))
            l_curr = torch.where(current_lvl_pred == true_labels[:,l], torch.FloatTensor([0]).to(prev_lvl_pred.device), torch.FloatTensor([1]).to(prev_lvl_pred.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
    
    def forward(self, predictions:List[torch.Tensor], true_labels:torch.Tensor)->torch.Tensor:
        '''
        for true labels, please use indice (id) instead of onehots
        '''
        dloss = self.calculate_dloss(predictions, true_labels)
        lloss=self.calculate_lloss(predictions, true_labels)
        return dloss+lloss

class HierarESM(L.LightningModule):
    '''
    model
    ---
    `model_name`              : transformers' model name  
    `max_length,max_domain`   : parameters for truncation 
    `nhead ff_fold`           : parameters for classification head
    `num_block_layers`        : parameters for classification head

    loss
    ---
    `alpha`         : coeff. for layer-wise loss  
    `a_incremental` : exp. incremental for layer-wise loss 
    `beta`          : coeff. for dependence (affiliation) loss  
    `p_loss`        : base for dependence (affiliation) loss (no change usually)
    
    optimizers
    ---
    `optimizer_kwargs` & `scheduler_kwargs` :  
            dicts for overriding default args, check `configure_optimizers`
    '''
    
    def __init__(self,order_manager:OrderManager,
            model_name:str='facebook/esm2_t6_8M_UR50D',
            max_length:int=8000,
            max_domain:int=15,
            nhead:int=4,
            ff_fold:int=4,
            #
            num_block_layers:int=2,
            to_freeze:int=0,
            alpha:float=0.5, beta:float=0.8, 
            p_loss:float=3.,a_incremental:float=1.3,
            #
            optimizer_kwargs:Dict[str,Any]={'backbone_lr':1e-4,'head_lr':1e-3,'weight_decay':0.01},
            scheduler_kwargs:Dict[str,Any]={'warmup_iter_1':20,'warmup_iter_2':30,'warmup_lr':1e-10}
            ):

        super().__init__()
        self.order_manager=order_manager
        self.max_length=max_length
        self.max_domain=max_domain
        self.nhead=nhead
        self.ff_fold=ff_fold
        self.num_block_layers=num_block_layers
        self.model_name=model_name
        self.to_freeze=to_freeze
        self.alpha,self.beta,self.ploss,self.a_incremental=(
            alpha,beta,p_loss,a_incremental)
        self.optimizer_kwargs=optimizer_kwargs
        self.scheduler_kwargs=scheduler_kwargs
        self.save_hyperparameters(ignore=['order_manager'])

        self._configure_model()
        self.criterion=HierarchicalLossNetwork(self.order_manager,
                    alpha,beta,p_loss,a_incremental)
        
    def _configure_model(self):
        self.tokenizer:EsmTokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self.backbone:EsmModel = EsmModel.from_pretrained(self.model_name).train()
        self.hidden_size:int=self.backbone.config.hidden_size
        assert self.to_freeze<self.backbone.config.num_hidden_layers,'no enough layer to freeze!'
        if self.to_freeze>0:
            self._partial_freeze()
        self._make_transformer_hierar_layers()
        #TODO a trainable embedding for Family (maybe from the Pfam MSA)
        #https://www.nature.com/articles/s41586-021-03819-2/figures/3
        #TODO positional embedding for "sentence" (maybe related to Hits' begin/end)

    def _partial_freeze(self):
        for name, param in self.backbone.named_parameters():
            if ('backbone' in name ):
                names=name.split('.')
                if name[1]=='embeddings' or (
                    name[1]=='encoder' and (int(names[3]))<self.to_freeze):
                    param.requires_grad = False

    def _make_transformer_hierar_layers(self):
        #TODO use nn.TransformerEncoder with 2 layers
        self.opt_initiator=nn.TransformerEncoderLayer(
            d_model=self.hidden_size,nhead=self.nhead,
            dim_feedforward=self.hidden_size*self.ff_fold,
            activation='gelu',batch_first=True)
        decoder_layer=nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
            d_model=self.hidden_size,nhead=self.nhead,
            dim_feedforward=self.hidden_size*self.ff_fold,
            activation='gelu',batch_first=True,),
            num_layers=self.num_block_layers,
            norm=None
            )
        for i,l in enumerate(self.order_manager.levels):
            setattr(self,f'decoder_{i+1}',
                    deepcopy(decoder_layer))
            num_classes=len(l)
            # inner_dim=self.hidden_size+int((self.hidden_size*num_classes)**0.5)
            inner_dim=self.hidden_size*ceil(1+num_classes**0.5)
            setattr(self,f'head_{i+1}',ClassificationHead(
                self.hidden_size,inner_dim,num_classes))
            
    def forward(self, 
        attention_mask:torch.Tensor,
        sentence_mask:torch.Tensor,
        input_ids:Optional[torch.Tensor]=None,
        inputs_embeds:Optional[torch.Tensor]=None,
        **kwargs):
        '''
        'attention_mask':[batch_size,max_domain,hidden_size]
        'sentence_mask':[batch_size,max_domain], 
        'input_ids': usually from datasets;  
        'inputs_embeds' : usually from integrated ingradient workflows;  
        
        other parameters would be ignored, so feel free to use `module.forward(**batch)` 


        
        '''
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
    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        predictions=self.forward(**batch)
        true_labels:torch.Tensor=batch['taxo']
        loss=self.criterion(predictions,true_labels)
        # self.log('loss/train_sum',loss,reduce_fx=torch.sum) #.item()
        self.log('loss/train',loss,prog_bar=True) 
        self.log_dict(cal_accuracy(predictions,true_labels,
            self.order_manager.level_names,'train/'),prog_bar=True) # batch_size=true_labels.shape[0],prog_bar=True
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions=self.forward(**batch)
        true_labels:torch.Tensor=batch['taxo']
        loss=self.criterion(predictions,true_labels)
        self.log('loss/valid',loss.item())
        self.log('val_loss',loss.item())
        self.log_dict(cal_accuracy(predictions,true_labels,
            self.order_manager.level_names,'valid/'))
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        predictions=self.forward(**batch)
        true_labels:torch.Tensor=batch['taxo']
        loss=self.criterion(predictions,true_labels)
        self.log('test/loss',loss.item())
        self.log_dict(cal_accuracy(predictions,true_labels,
            self.order_manager.level_names,'test/'))
        return loss

    def predict_step(self, batch:dict,
            batch_idx, dataloader_idx=0):
        predictions=self.forward(**batch)
        predict_labels=self.order_manager.idx_to_order(predictions,need_argmax=True)
        for level_name,predict in zip(self.order_manager.level_names,predictions):
            batch[f'{level_name}_predict']=predict
        batch['predict_label']=predict_labels
        # if 'taxo' in batch:
        #     true_labels:torch.Tensor=batch['taxo']
        #     batch.update(cal_accuracy(batch['predictions'],true_labels,
        #     self.order_manager.level_names,''))

        # for i,level_name in enumerate(self.order_manager.level_names):
        #     predict = torch.argmax(predictions[i], dim=1)
        # TODO from lightning.pytorch.callbacks import BasePredictionWriter
        # save each epoch to 
        return batch

    def configure_optimizers(self):
        backbone_params=chain(
            self.backbone.embeddings.parameters(),
            self.backbone.encoder.parameters(),
            )
        head_params=chain(
            *[self.backbone.pooler.parameters(),
              self.backbone.contact_head.parameters(),
              self.opt_initiator.parameters()
            ],*[
                getattr(self,f'decoder_{i+1}').parameters() 
                for i in range(self.order_manager.total_level)
            ],*[
                getattr(self,f'head_{i+1}').parameters() 
                for i in range(self.order_manager.total_level)
            ])
        optimizer = Adam(params=[{'params': backbone_params,'lr':self.optimizer_kwargs.pop('backbone_lr',1e-4)},
                                  {'params':head_params,'lr':self.optimizer_kwargs.pop('head_lr',1e-3)}],
                          **self.optimizer_kwargs)
        # head_optimizer = AdamW(params=head_params,
        #                   **self.optimizer_kwargs)

        exp_gamma=self.scheduler_kwargs.get('exp_gamma',0.98)
        def warmup_exp_increase(current_step):
            warmup_lr = self.scheduler_kwargs.get('warmup_lr',1e-10)
            warmup_iter_1 = self.scheduler_kwargs.get('warmup_iter_1',20)
            warmup_iter_2 = self.scheduler_kwargs.get('warmup_iter_2',30)
            if current_step < warmup_iter_1:
                return warmup_lr* (exp_gamma**current_step)
            elif current_step < warmup_iter_1 + warmup_iter_2:
                progress = (current_step - warmup_iter_1) / warmup_iter_2
                return warmup_lr * (1 / warmup_lr) ** progress * (exp_gamma**current_step)
            else:
                return (exp_gamma**current_step)
            
        scheduler = LambdaLR(optimizer, lr_lambda=[warmup_exp_increase,lambda step: exp_gamma**step])
        # exp_scheduler = ExponentialLR(optimizer,gamma=self.scheduler_kwargs.get('exp_gamma',0.98))
        # scheduler = ChainedScheduler([warmup_scheduler, exp_scheduler], optimizer=optimizer)
        # return [optimizer],[warmup_scheduler,exp_scheduler]
        return {"optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'val_loss',
                "frequency":1,
                "interval": "epoch",
                "strict": False,
                "name": 'scheduler',
            }}

    def _gradient(self):
        pass

    def _embed(self):
        pass
        
    def _attention(self):
        pass
    
    def _infer(self):
        pass


def cal_accuracy(predictions:List[torch.Tensor],true_labels:torch.Tensor,
        level_names:List[str],prefix:str='')->Dict[str,float]:
    o={}
    bs=true_labels.shape[0]
    for i,level_name in enumerate(level_names):
        predict = torch.argmax(predictions[i], dim=1)
        correct_pred = torch.sum(predict == true_labels[:,i])#.item()
        o[f'{prefix}acc_{level_name}'] = correct_pred/bs
    return o