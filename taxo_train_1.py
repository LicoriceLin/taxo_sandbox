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
# TODO implement tensorboard
logger=logging.getLogger()
time_str=time.strftime("%y%m%d-%H%M%S", time.localtime())



# %%

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
# 
# %%
def hide_spline(ax:Axes):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
def circle(center:tuple,outer:tuple,**kwargs):
    '''
    kwargs: for `plt.Circle`
    '''
    radius = ((center[0]-outer[0])**2+
                (center[1]-outer[1])**2)**0.5
    return plt.Circle((center[0], center[1]), radius,**kwargs)
    
class OrderManager:
    def __init__(self,order_dict:dict,level_names:Optional[list]=None, #device:Union[str,int]='cpu',
                 color_levels:List[str]=['pinkish red','purply','ocean','peach'],
                 layout_prog="twopi"
                 ):
        '''
        
        # device: Only used in `idx_to_onehot`
        remove device,default to cpu, might be moved to GPU in later steps
        color levels: 
        check https://xkcd.com/color/rgb/ for viable colors. 
        '''
        # self.device=device
        self._parse_order(order_dict)
        
        if level_names is None:
            self.level_names=[str(i+1) for i in range(len(self.levels))]
        else:
            assert len(level_names)==len(self.levels)
            self.level_names=level_names
            
        assert len(color_levels)>=len(self.levels)
        self._gen_visual_props(color_levels=color_levels)
        self._gen_order_graph(layout_prog)
        
    def _parse_order(self,order_dict:dict):
        def smart_fetch_list(l:list,idx:int,init_l=[])->list:
            while len(l)<idx+1:
                l.append(init_l)
            return l[idx]
        def non_redundant_add(l:list,i):
            if i not in l:
                l.append(i)
        def recurse_add(l:list,d:Union[dict,list],level:int=0)->List[List[Any]]:
            cur_l=smart_fetch_list(l,level,['Null'])
            if isinstance(d,dict):
                for k,v in d.items():
                    non_redundant_add(cur_l,k)
                    recurse_add(l,v,level+1)
            elif isinstance(d,list):
                for k in d:
                    non_redundant_add(cur_l,k)
            else:
                raise ValueError(f'{d}')
            return l
        def recurse_w2i(l:List[List[str]],d:Union[dict,list],level:int=0):
            l_cur=l[level]
            if isinstance(d,dict):
                o={}
                for k,v in d.items():
                    o[l_cur.index(k)]=recurse_w2i(l,v,level+1)
            elif isinstance(d,list):
                o=[l_cur.index(i) for i in d]
            else:
                raise ValueError(f'{d}')
            return o

        def safe_update_dict(d1:dict,d2:dict):
            for k,v in d2.items():
                if k not in d1:
                    d1[k]=v
                else:
                    if isinstance(v,dict):
                        d1[k]=safe_update_dict(d1[k],v)
                    elif isinstance(v,list):
                        for l in v:
                            if l not in d1[k]:
                                d1[k].append(l)
                    else:
                        print(v)
            return d1

        def peel_dict(d:dict,o:list=[]):
            next_dict={}
            j=next(iter(d.values()))
            if isinstance(j,list):
                o_dict=d
                o.append(o_dict)
            elif isinstance(j,dict):
                o_dict={}
                next_dict={}
                for k,v in d.items():
                    o_dict[k]=list(v.keys())
                    safe_update_dict(next_dict,v)
                o.append(o_dict)
                # print(next_dict)
                peel_dict(next_dict,o)
            return o
        
        def parse_order(order_dict:dict):
            hierarchical_labels=order_dict
            # hierarchical_labels:dict=pkl.load(open(order_file,'rb'))
            levels=recurse_add([],hierarchical_labels)
            total_level = len(levels)
            numeric_hierarchy = recurse_w2i(levels,hierarchical_labels)
            peeled_numeric_hierarchy:List[Dict[str,List[str]]]=peel_dict(numeric_hierarchy)
            return hierarchical_labels,levels,total_level,numeric_hierarchy,peeled_numeric_hierarchy
        
        (self.hierarchical_labels,self.levels,
         self.total_level,self.numeric_hierarchy,
         self.peeled_numeric_hierarchy
        )=parse_order(order_dict)
        
    def _gen_visual_props(self,color_levels:List[str]):
        def recur_size_count(hierarchical_labels:dict,cur_key:str='root',null_level=0):
            o={}
            null_level_dict={}
            if cur_key=='root':
                hierarchical_labels_=deepcopy(hierarchical_labels)
            else:
                hierarchical_labels_=hierarchical_labels
                
            if 'Null' in hierarchical_labels_:
                hierarchical_labels_[f"{cur_key}'s Null"]=hierarchical_labels_.pop('Null')
                null_level_dict[f"{cur_key}'s Null"]=null_level
                
            for k,v in hierarchical_labels_.items():
                if isinstance(v,list):
                    if 'Null' in v:
                        v[v.index('Null')]=f"{k}'s Null"
                        null_level_dict[f"{k}'s Null"]=null_level+1
                    o[k]=len(v)
                elif isinstance(v,dict):
                    sub_o,_,sub_null_level_dict=recur_size_count(v,cur_key=k,null_level=null_level+1)
                    o[k]=sum(sub_o.values())
                    o.update(sub_o)
                    null_level_dict.update(sub_null_level_dict)
            size_count=o
            return size_count,hierarchical_labels_,null_level_dict
        (self.size_count,self.split_null_hierarchical_labels,
         self.null_level_dict)=recur_size_count(self.hierarchical_labels)
        max_colors=[mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{i}']) for i in color_levels]
        
        self.color_levels=[mcolors.LinearSegmentedColormap.from_list(
            name,[[1-(1-j)*0.05 for j in max_colors[i]],max_colors[i]]) for i,name in enumerate(self.level_names)]
        self.color_names:List[str]=color_levels
        
    def _gen_order_graph(self,layout_prog:str):
        '''
        if you want to edit the layout properties, add them to the graph property
        
        e.g.
        order_manager.order_graph.graph['rankdir']="LR"
        order_manager.order_graph.nodes['root']['root']=True
        '''
        self.order_graph=nx.DiGraph()
        def to_graph(G:nx.DiGraph,hierarchical_labels:dict,cur_key:str='root'):
            for k,v in hierarchical_labels.items():
                G.add_edge(cur_key,k)
                if isinstance(v,list):
                    for i in v:
                        G.add_edge(k,i)
                elif isinstance(v,dict):
                    for i,sub_d in v.items():
                        G.add_edge(k,i)
                        to_graph(G,sub_d,i)
        to_graph(self.order_graph,self.split_null_hierarchical_labels)
        self.graph_pos = nx.nx_agraph.graphviz_layout(self.order_graph, prog=layout_prog)
        
    def order_to_idx(self,order_list:List[str])->List[int]:
        o=[]
        for taxo,level in zip(order_list,self.levels):
            o.append(level.index(taxo))
        return o
    
    def idx_to_onehot(self,idx_list:Union[List[int],List[torch.Tensor]],device:Optional[str]='cpu')->List[torch.Tensor]:
        # if device is None: device=self.device
        o=[]
        if isinstance(idx_list[0],int):
            for idx,level in zip(idx_list,self.levels):
                h=nn.functional.one_hot(torch.tensor([idx]),num_classes=len(level)).to(device)
                o.append(h)
        else:
            for idx,level in zip(idx_list,self.levels):
                h=nn.functional.one_hot(idx,num_classes=len(level)).to(device)
                o.append(h)
        return o
    
    def order_to_onehot(self,order_list:List[str])->List[torch.Tensor]:
        return self.idx_to_onehot(self.order_to_idx(order_list))

    def distribution_to_color_dict(self,distribution:List[torch.Tensor],batch_i:int=0)->Dict[str,tuple]:
        '''
        distribution: output from models or gt label after idx_to_onehot
        for robustness, I put `.detach().to('cpu')` in this method
        '''
        null_level_dict=self.null_level_dict
        color_dict={'root':mcolors.to_rgb('dimgrey')}
        is_long= (distribution[0].dtype==torch.long)
        for level_id,level in enumerate(self.levels):
            dist:torch.Tensor=distribution[level_id][batch_i].float().detach().to('cpu')
            if not is_long:
                dist:torch.Tensor=F.softmax(dist,dim=-1)
            cmap=self.color_levels[level_id]
            level_color_dict={}
            for i,term in enumerate(level):
                level_color_dict[term]=cmap(dist[i])
            if 'Null' in level_color_dict:
                null_color=level_color_dict.pop('Null')
                split_nulls={k:null_color for k,v in null_level_dict.items() if v==level_id}
                level_color_dict.update(split_nulls)
            color_dict.update(level_color_dict)
        return color_dict
    
    def draw_classification_view(self,color_dict:Dict[str,tuple],ax:Axes,
            node_size=300,null_color=(0,0,0,0)):
        '''
        must work on a given ax
        '''
        to_label=lambda x:x if 'Null' not in x else 'Null'
        G,pos=self.order_graph,self.graph_pos
        nx.draw_networkx_nodes(G,pos,ax=ax,node_size=node_size
            ,node_color=[color_dict.get(n,null_color) for n in G],edgecolors='grey')
        nx.draw_networkx_edges(G,pos,ax=ax,arrowstyle='->',edge_color='grey')
        nx.draw_networkx_labels(G,pos,labels={n:to_label(n) for n in G},ax=ax,font_size=10,font_color='dimgrey')
        for i in range(len(self.levels)):
            circle_=circle(self.graph_pos['root'],self.graph_pos[self.levels[i][1]],
                        **dict(fill=False, edgecolor=mcolors.XKCD_COLORS[f'xkcd:{self.color_names[i]}'], 
                            linestyle='--', linewidth=2, alpha=0.5))
            ax.add_patch(circle_)
        hide_spline(ax)
        return ax
    
    def cal_true_probs(self,pred:List[torch.Tensor],true:List[torch.Tensor],batch_i:int=0):
        probabs=[]
        for level_id,level in enumerate(self.levels):
            dist=F.softmax(pred[level_id][batch_i].float().detach().to('cpu'),dim=-1)
            true_label=true[level_id][batch_i]
            probabs.append(dist[true_label].item())
        return probabs
    
    def draw_true_probs(self,pred:List[torch.Tensor],true:List[torch.Tensor],ax:Axes,
                        batch_i:int=0,fontsize:int=24):
        # fig, ax = plt.subplots()
        # ax:Axes
        # probabs=[]
        # for level_id,level in enumerate(self.levels):
        #     dist=F.softmax(pred[level_id][batch_i].float().detach().to('cpu'),dim=-1)
        #     true_label=true[level_id][batch_i]
        #     probabs.append(dist[true_label].item())
        probabs=self.cal_true_probs(pred,true,batch_i)
        x_ticks = list(range(len(self.level_names)))
        y_ticks=np.linspace(0,1,6)
        colors=[mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{i}']) for i in self.color_names]
        bars = ax.bar(x_ticks, probabs, color=colors)
        ax.set_xticks(x_ticks,self.level_names,fontsize=fontsize,color="dimgrey")
        ax.set_ylim(0,1.1)
        ax.set_yticks(y_ticks,labels=[f'{i:.1f}' for i in y_ticks], fontsize=fontsize*2/3)
        ax.set_ylabel("Probab of True Label",fontdict={'fontsize':fontsize, 'color': 'dimgrey'})
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                    ha='center', va='bottom',fontdict={'fontsize':fontsize,'color': 'dimgrey'})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        
# %%
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

class HierarESM(nn.Module):
    def __init__(self,order_manager:OrderManager,model_name:str='facebook/esm2_t6_8M_UR50D',
                 max_length:int=8000,max_domain:int=15,device:Union[int,str]=0,
                 nhead:int=4,ff_fold:int=4,num_block_layers:int=2,to_freeze:int=0):
        super().__init__()
        self.order_manager=order_manager
        # order_manager.device=device
        self.max_length=max_length
        self.max_domain=max_domain
        self.device=device
        self.nhead=nhead
        self.ff_fold=ff_fold
        self.num_block_layers=num_block_layers
        
        self.tokenizer:EsmTokenizer = EsmTokenizer.from_pretrained(model_name)
        self.backbone:EsmModel = EsmModel.from_pretrained(model_name).to(self.device)
        self.hidden_size:int=self.backbone.config.hidden_size
        self.dtype=self.backbone.config.torch_dtype
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
              
    def forward(self, input_ids:torch.Tensor,
            attention_mask:torch.Tensor,sentence_mask:torch.Tensor):
        #ipt:Dict[str,torch.Tensor],sentence_mask:torch.Tensor
        '''
        `ipt`: 
            ordered dict from parse_sentence
            {'input_ids':[batch_size*max_domain,hidden_size], 
            'attention_mask':[batch_size*max_domain,hidden_size]}
        
        `sentence_mask`:
            [batch_size*max_domain,]
        
        '''
        ipt={'input_ids':input_ids.view(-1,self.max_length),
             'attention_mask':attention_mask.view(-1,self.max_length)
             }
        sentence_mask=sentence_mask.view(-1)
        
        device,dtype=self.device,self.dtype
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
        
        memory_key_padding_mask:torch.Tensor=(1-sentence_mask.float()).bool()

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
            head=getattr(self,f'head_{i+1}')
            output=decoder(tgt=output,memory=ebs,tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)
            o.append(head(output[:,ss+i,:]))
        return o
            
    def report_eval(self,sample,pdf:PdfPages):
        prev_model_state = self.training
        # batch_size=len(sample['name'])
        self.eval()

        with torch.set_grad_enabled(False):
            batch_name,domains,batch_y,domains_mask=process_batch(sample,self.max_domain)
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
                pdf.savefig(fig)
                plt.close(fig)
                # break
            
        if prev_model_state:
            self.train()
            
    def _infer(self,sample):
        '''
        sample: from dataloder or dataset.fetch_single
        '''
        prev_model_state = self.training
        # batch_size=len(sample['name'])
        self.eval()
            
        with torch.set_grad_enabled(False):
            batch_name,domains,batch_y,domains_mask=process_batch(sample,self.max_domain)
            domains_mask=domains_mask.to(self.device)
            ipts=self.parse_sentence(domains)
            x:List[torch.Tensor]=self(ipts['input_ids'].view(-1,self.max_domain,self.max_length),
                            ipts['attention_mask'].view(-1,self.max_domain,self.max_length),
                            domains_mask.view(-1,self.max_domain))
            y=batch_y
            
        if prev_model_state:
            self.train()
        return x,y

    def _embed(self,sample):
        prev_model_state = self.training
        # batch_size=len(sample['name'])
        self.eval()
        device,dtype=self.device,self.dtype
        with torch.set_grad_enabled(False):
            batch_name,domains,batch_y,sentence_mask=process_batch(sample,self.max_domain)
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
        return batch_name,o,batch_y
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

def process_batch(batch:dict,max_domain:int):
    '''
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
    batch_name, batch_x, batch_y,sentence_mask = batch['name'], batch['seq'], batch['taxo'],batch['sentence_mask']
    batch_y:List[torch.Tensor] #dtype=torch.long
    batch_name:List[str]
    batch_size=len(batch_x[0])
    domains:List[str]=[]
    for i in range(batch_size):
        for j in range(max_domain):    
            domains.append(batch_x[j][i])
    domains_mask=torch.stack(sentence_mask).T.reshape(-1)
    return batch_name,domains,batch_y,domains_mask
    
# %%
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
        lloss=hierar_loss.calculate_lloss(predictions, true_labels)
        return dloss+lloss

# %%
def cal_accuracy(predictions, labels):
    '''Calculates the accuracy of the prediction.
    '''

    num_data = labels.size()[0]
    predicted = torch.argmax(predictions, dim=1)

    correct_pred = torch.sum(predicted == labels)

    accuracy = correct_pred*(100/num_data)

    return accuracy.item()

# %%

class wrapper_module(nn.Module):
    def __init__(self, inner:nn.Module,*extra_args) -> None:
        super().__init__()
        self.inner=inner
        self.extra_args=extra_args
        
    def forward(self,args):
        return self.inner(args,*self.extra_args)
    
def process_sample(sample:dict):
    domains=[]
    batch_size=len(sample['seq'][0])
    for i in range(batch_size):
        for j in range(max_domain):    
            domains.append(sample['seq'][j][i])
    domains_mask=torch.stack(sample['sentence_mask']).T.reshape(-1).to(device)
    ipts=hierar_esmmodel.parse_sentence(domains)
    return (ipts['input_ids'].view(-1,max_domain,max_length),
            ipts['attention_mask'].view(-1,max_domain,max_length),
            domains_mask.view(-1,max_domain))
    
# %%
if __name__=='__main__':
    seed=47
    device=1
    batch_size=2
    batch_size_val=30
    max_domain=15
    acc_step=20
    valid_step=500
    max_length=500
    to_freeze=3
    set_seed(seed)
    
    order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                            level_names=['Kingdom','Phylum','Class','Order'])
    hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=max_length,to_freeze=to_freeze,device=device)
    hierar_loss=HierarchicalLossNetwork(order_manager)
    optimizer = Adam(hierar_esmmodel.parameters(), lr=1e-4)

    dataset=ConcatProteinDataset('taxo_data/proseq_taxo_1.pkl',order_manager)
    trainset,valset=random_split(dataset,[0.9,0.1])
    train_generator = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_generator = DataLoader(valset, batch_size=batch_size_val, shuffle=True, num_workers=batch_size)
    
    odir=f'train/v1_{time_str}_seed{seed}'
    os.mkdir(odir)
    writer = SummaryWriter(log_dir=f'{odir}/log')
    # %% train
    hierar_esmmodel.train()

    def init_acc_logs():
        pred_dict={i:[] for i in order_manager.level_names}
        true_dict={i:[] for i in order_manager.level_names}
        return pred_dict,true_dict
    

        
        
    def train_step(sample):
        optimizer.zero_grad()
        batch_name, batch_x, batch_y,sentence_mask =sample['name'], sample['seq'], sample['taxo'],sample['sentence_mask']
        model_input=process_sample(sample)
        x:List[torch.Tensor]=hierar_esmmodel(*model_input)  
        y=[i.to(device) for i in batch_y]
        loss:torch.Tensor = hierar_loss(x,y,device)
        return [i.detach().cpu() for i in x],loss
    
    pred_dict,true_dict=init_acc_logs()
    cur_acc_step=0
    global_step=0
    best_val_acc=0
    
    for epoch in range(10):
        for step, sample in tqdm(enumerate(train_generator)):
            x,loss=train_step(sample)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss',loss.item(),global_step)
            writer.add_scalar('epoch',epoch,global_step)
            for i in range(len(x)):
                k=order_manager.level_names[i]
                pred_dict[k].append(x[i])
                true_dict[k].append(sample['taxo'][i])
                cur_acc_step+=1
        
            if cur_acc_step==acc_step:
                pred_dict={k:torch.concat(v,dim=0) for k,v in pred_dict.items()}
                true_dict={k:torch.concat(v,dim=0) for k,v in true_dict.items()}
                accuracies={k:cal_accuracy(pred_dict[k],true_dict[k]) for k,v in pred_dict.items()}
                writer.add_scalars('train/accuracy',accuracies,global_step)
                pred_dict,true_dict=init_acc_logs()
                cur_acc_step=0
                
            
            if global_step%valid_step==0:
                with torch.set_grad_enabled(False):
                    hierar_esmmodel.eval()
                    x_s,y_s,ebs,labels=[],[],[],[]
                    for step, sample in tqdm(enumerate(val_generator)):
                        # batch_y=sample['taxo']
                        x_s.append([i.detach().to('cpu') for i in hierar_esmmodel(*process_sample(sample))])
                        y_s.append(sample['taxo'])
                        
                        _names,_ebs,_labels=hierar_esmmodel._embed(sample)
                        ebs.append(torch.concat(_ebs,dim=1).detach().to('cpu'))
                        labels.extend([order_manager.levels[-1][i] for i in _labels[-1].tolist()])
                        del _names,_ebs,_labels
                    x_s=[torch.concat([j[i] for j in x_s],dim=0) for i in range(order_manager.total_level)]
                    y_s=[torch.concat([j[i] for j in y_s],dim=0) for i in range(order_manager.total_level)]
                    accuracies={k:cal_accuracy(i,j) for k,i,j in zip(order_manager.level_names,x_s,y_s)}
                    writer.add_scalars('valid/accuracy',accuracies,global_step)
                    writer.add_scalar('valid/loss',hierar_loss(x_s,y_s,'cpu').item(),global_step)
                    writer.add_embedding(mat=torch.concat(ebs,dim=0),
                                        metadata=labels,
                                        global_step=global_step,
                                        tag='valid/embeddings')
                    if best_val_acc<accuracies[order_manager.level_names[-1]]:
                        best_val_acc=accuracies[order_manager.level_names[-1]]
                    hierar_esmmodel.train()
                    
            global_step+=1
                # import pdb;pdb.set_trace()
                # writer.add_pr_curve('valid prc',labels=y_s[-1],predictions=x_s[-1],global_step=global_step)
                
        torch.save(hierar_esmmodel.state_dict(), f'{odir}/ep-{epoch}.pt')
        
    writer.add_hparams(hparam_dict=dict(max_domain=max_domain,max_length=max_length,
                    to_freeze=to_freeze,seed=seed,device=device,
                    batch_size=batch_size,acc_step=acc_step),
                       metric_dict={'best_val_acc':best_val_acc})
# %%

# if 0:
#     order_manager=OrderManager('dataset/taxo_data/hierarchy_order.pkl')
#     hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=500)
#     model='train/240426_v1/output/240426-130251-ep1.pt'#'train/240426_v1/output/240426-182439-ep2.pt'
#     para=0
#     d=torch.load(model)
#     if para:
#         d=OrderedDict({k.replace('module.',''):v for k,v in d.items()})

#     hierar_esmmodel.load_state_dict(d) #"train/240426-141811-ep1.pt"
#     hierar_esmmodel.eval()
#     dataset=ConcatProteinDataset('dataset/taxo_data/proseq_taxo.pkl',order_manager)
#     test_generator = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

#     from itertools import accumulate
#     level_indices=[0]+list(accumulate([len(i) for i in order_manager.levels]))

#     rank=0
#     with torch.set_grad_enabled(False):
#         predict_eb,true_eb=[],[]
#         predict_labels,true_labels=[],[]
#         for _,sample in enumerate(test_generator):
#             batch_x, batch_y = sample['seq'], sample['taxo']
#             batch_y:torch.Tensor
#             ipt=[i.upper().replace('#','<eos>') for i in batch_x]
#             ipt = hierar_esmmodel.tokenizer(ipt, return_tensors="pt",padding=True,truncation=True,max_length=hierar_esmmodel.max_length).to(rank)
#             x:List[torch.Tensor]=hierar_esmmodel(ipt)
#             p,t=[],[]
#             p_eb,t_eb=[],[]
#             predict_idx=[torch.argmax(i, dim=1).to('cpu') for i in x]
#             for i,l in enumerate(order_manager.levels):
#                 cur_predict=predict_idx[i].numpy()
#                 cur_true=batch_y[i].numpy()
#                 cur_plabel=np.vectorize(lambda x:l[x])(cur_predict)
#                 cur_tlabel=np.vectorize(lambda x:l[x])(cur_true)
#                 p.append(cur_plabel)
#                 t.append(cur_tlabel)
#                 cur_eb=x[i].to('cpu').numpy()
#                 p_eb.append(cur_eb)
#                 t_eb.append(nn.functional.one_hot(batch_y[i],len(l)).numpy())
#             predict_labels.append(np.stack(p).T)
#             true_labels.append(np.stack(t).T)
#             predict_eb.append(np.hstack(p_eb))
#             true_eb.append(np.hstack(t_eb))
#             if _>100:
#                 break
#         predict_labels=np.vstack(predict_labels)
#         true_labels=np.vstack(true_labels)
#         predict_eb=np.vstack(predict_eb)
#         true_eb=np.vstack(true_eb)
        
#     def fetch_level_eb(eb:np.ndarray,l:int):
#         return torch.tensor(eb[:,level_indices[l]:level_indices[l+1]])

 
#     plt.close()
#     taxo_hierars=['Realm', 'Kingdom','Phylum', 'Class','Order'] #TODO merge into manager
#     single_h,single_w=8,4
#     fig,axes=plt.subplots(2,len(order_manager.levels),figsize=[single_h*2,single_w*len(order_manager.levels)])
#     axes:List[List[Axes]]
#     for i in range(len(order_manager.levels)):
#         ax=axes[0][i]
#         data=F.softmax(fetch_level_eb(predict_eb,i),dim=-1)
#         sns.heatmap(data,cmap='Greens', ax=ax,cbar=False,
#             xticklabels=order_manager.levels[i])
#         ax.set_ylabel('sample id')
#         ax.set_title(f'Pred {taxo_hierars[i]}')
#         # if i!=len(order_manager.levels)
#         cbar = ax.collections[0].colorbar
        
#         ax=axes[1][i]
#         data=fetch_level_eb(true_eb,i)
#         sns.heatmap(data,cmap='Greens', ax=ax,cbar=False,
#             xticklabels=order_manager.levels[i])
#         ax.set_ylabel('sample id')
#         ax.set_title(f'Pred {taxo_hierars[i]}')

#     # %%
#     from collections import OrderedDict    
#     order_manager=OrderManager('dataset/taxo_data/hierarchy_order.pkl')
#     hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=500)
#     model='train/240426_v1/output/240426-130251-ep1.pt'#'train/240426_v1/output/240426-182439-ep2.pt'
#     para=0
#     d=torch.load(model)
#     if para:
#         d=OrderedDict({k.replace('module.',''):v for k,v in d.items()})

#     hierar_esmmodel.load_state_dict(d) #"train/240426-141811-ep1.pt"
#     hierar_esmmodel.eval()
#     dataset=ConcatProteinDataset('dataset/taxo_data/proseq_taxo.pkl',order_manager)
#     test_generator = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

#     from itertools import accumulate
#     level_indices=[0]+list(accumulate([len(i) for i in order_manager.levels]))

#     rank=0
#     with torch.set_grad_enabled(False):
#         predict_eb,true_eb=[],[]
#         predict_labels,true_labels=[],[]
#         for _,sample in enumerate(test_generator):
#             batch_x, batch_y = sample['seq'], sample['taxo']
#             batch_y:torch.Tensor
#             ipt=[i.upper().replace('#','<eos>') for i in batch_x]
#             ipt = hierar_esmmodel.tokenizer(ipt, return_tensors="pt",padding=True,truncation=True,max_length=hierar_esmmodel.max_length).to(rank)
#             x:List[torch.Tensor]=hierar_esmmodel(ipt)
#             p,t=[],[]
#             p_eb,t_eb=[],[]
#             predict_idx=[torch.argmax(i, dim=1).to('cpu') for i in x]
#             for i,l in enumerate(order_manager.levels):
#                 cur_predict=predict_idx[i].numpy()
#                 cur_true=batch_y[i].numpy()
#                 cur_plabel=np.vectorize(lambda x:l[x])(cur_predict)
#                 cur_tlabel=np.vectorize(lambda x:l[x])(cur_true)
#                 p.append(cur_plabel)
#                 t.append(cur_tlabel)
#                 cur_eb=x[i].to('cpu').numpy()
#                 p_eb.append(cur_eb)
#                 t_eb.append(nn.functional.one_hot(batch_y[i],len(l)).numpy())
#             predict_labels.append(np.stack(p).T)
#             true_labels.append(np.stack(t).T)
#             predict_eb.append(np.hstack(p_eb))
#             true_eb.append(np.hstack(t_eb))
#             if _>100:
#                 break
#         predict_labels=np.vstack(predict_labels)
#         true_labels=np.vstack(true_labels)
#         predict_eb=np.vstack(predict_eb)
#         true_eb=np.vstack(true_eb)
        
#     def fetch_level_eb(eb:np.ndarray,l:int):
#         return torch.tensor(eb[:,level_indices[l]:level_indices[l+1]])

#     from matplotlib.axes import Axes
#     plt.close()
#     taxo_hierars=['Realm', 'Kingdom','Phylum', 'Class','Order'] #TODO merge into manager
#     single_h,single_w=8,4
#     fig,axes=plt.subplots(2,len(order_manager.levels),figsize=[single_h*2,single_w*len(order_manager.levels)])
#     axes:List[List[Axes]]
#     for i in range(len(order_manager.levels)):
#         ax=axes[0][i]
#         data=F.softmax(fetch_level_eb(predict_eb,i),dim=-1)
#         sns.heatmap(data,cmap='Greens', ax=ax,cbar=False,
#             xticklabels=order_manager.levels[i])
#         ax.set_ylabel('sample id')
#         ax.set_title(f'Pred {taxo_hierars[i]}')
#         # if i!=len(order_manager.levels)
#         cbar = ax.collections[0].colorbar
        
#         ax=axes[1][i]
#         data=fetch_level_eb(true_eb,i)
#         sns.heatmap(data,cmap='Greens', ax=ax,cbar=False,
#             xticklabels=order_manager.levels[i])
#         ax.set_ylabel('sample id')
#         ax.set_title(f'Pred {taxo_hierars[i]}')


#     # %%
#     '''
#     [src/tgt/memory]_mask:  
#         ``False`` values will be unchanged;
#         FloatTensor: added to the attention weight.

#     [src/tgt/memory]_key_padding_mask:
#         specified elements in the key to be ignored by the attention.
#         ``True`` will be ignored
#         ``False`` will be unchanged
        
#     '''
#     class LearnedPositionalEmbedding(nn.Embedding):
#         """
#         This module learns positional embeddings up to a fixed maximum size.
#         Padding ids are ignored by either offsetting based on padding_idx
#         or by setting padding_idx to None and ensuring that the appropriate
#         position ids are passed to the forward function.
#         """

#         def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
#             if padding_idx is not None:
#                 num_embeddings_ = num_embeddings + padding_idx + 1
#             else:
#                 num_embeddings_ = num_embeddings
#             super().__init__(num_embeddings_, embedding_dim, padding_idx)
#             self.max_positions = num_embeddings

#         def forward(self, input: torch.Tensor):
#             """Input is expected to be of size [bsz x seqlen]."""
#             if input.size(1) > self.max_positions:
#                 raise ValueError(
#                     f"Sequence length {input.size(1)} above maximum "
#                     f" sequence length of {self.max_positions}"
#                 )
#             mask = input.ne(self.padding_idx).int()
#             positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
#             return F.embedding(
#                 positions,
#                 self.weight,
#                 self.padding_idx,
#                 self.max_norm,
#                 self.norm_type,
#                 self.scale_grad_by_freq,
#                 self.sparse,
#             )
#     # %%
#     for i, sample in enumerate(train_generator):
#         optimizer.zero_grad()
#         batch_x, batch_y,sentence_mask = sample['seq'], sample['taxo'],sample['sentence_mask']
#         ipts=hierar_esmmodel.parse_sentence(batch_x)
        
#         y=[i.to(device) for i in batch_y]
#         x=hierar_esmmodel(batch_x)
#         total_loss:torch.Tensor = hierar_loss(x,y)
#         total_loss.backward()
#         optimizer.step()
        
#         logger.info(f'loss: {total_loss.item():.3f}')
#         accuracies=[cal_accuracy(x[i],y[i]) for i in range(len(x))]
#         logger.info(f'level-wise acc: {'\t'.join([f'{i:2.2f}' for i in accuracies])}')

#     torch.save(hierar_esmmodel.state_dict(), f'train/{time_str}-ep1.pt')
        
#     order_manager=OrderManager('dataset/taxo_data/hierarchy_order.pkl')
#     hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=500)
#     model='train/240426_v1/output/240426-130251-ep1.pt'#'train/240426_v1/output/240426-182439-ep2.pt'
#     para=0
#     d=torch.load(model)
#     if para:
#         d=OrderedDict({k.replace('module.',''):v for k,v in d.items()})

#     hierar_esmmodel.load_state_dict(d) #"train/240426-141811-ep1.pt"
#     hierar_esmmodel.eval()
#     dataset=ConcatProteinDataset('dataset/taxo_data/proseq_taxo.pkl',order_manager)
#     test_generator = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

#     from itertools import accumulate
#     level_indices=[0]+list(accumulate([len(i) for i in order_manager.levels]))

#     rank=0
#     with torch.set_grad_enabled(False):
#         predict_eb,true_eb=[],[]
#         predict_labels,true_labels=[],[]
#         for _,sample in enumerate(test_generator):
#             batch_x, batch_y = sample['seq'], sample['taxo']
#             batch_y:torch.Tensor
#             ipt=[i.upper().replace('#','<eos>') for i in batch_x]
#             ipt = hierar_esmmodel.tokenizer(ipt, return_tensors="pt",padding=True,truncation=True,max_length=hierar_esmmodel.max_length).to(rank)
#             x:List[torch.Tensor]=hierar_esmmodel(ipt)
#             p,t=[],[]
#             p_eb,t_eb=[],[]
#             predict_idx=[torch.argmax(i, dim=1).to('cpu') for i in x]
#             for i,l in enumerate(order_manager.levels):
#                 cur_predict=predict_idx[i].numpy()
#                 cur_true=batch_y[i].numpy()
#                 cur_plabel=np.vectorize(lambda x:l[x])(cur_predict)
#                 cur_tlabel=np.vectorize(lambda x:l[x])(cur_true)
#                 p.append(cur_plabel)
#                 t.append(cur_tlabel)
#                 cur_eb=x[i].to('cpu').numpy()
#                 p_eb.append(cur_eb)
#                 t_eb.append(nn.functional.one_hot(batch_y[i],len(l)).numpy())
#             predict_labels.append(np.stack(p).T)
#             true_labels.append(np.stack(t).T)
#             predict_eb.append(np.hstack(p_eb))
#             true_eb.append(np.hstack(t_eb))
#             if _>100:
#                 break
#         predict_labels=np.vstack(predict_labels)
#         true_labels=np.vstack(true_labels)
#         predict_eb=np.vstack(predict_eb)
#         true_eb=np.vstack(true_eb)
        
#     def fetch_level_eb(eb:np.ndarray,l:int):
#         return torch.tensor(eb[:,level_indices[l]:level_indices[l+1]])

#     from matplotlib.axes import Axes
#     plt.close()
#     taxo_hierars=['Realm', 'Kingdom','Phylum', 'Class','Order'] #TODO merge into manager
#     single_h,single_w=8,4
#     fig,axes=plt.subplots(2,len(order_manager.levels),figsize=[single_h*2,single_w*len(order_manager.levels)])
#     axes:List[List[Axes]]
#     for i in range(len(order_manager.levels)):
#         ax=axes[0][i]
#         data=F.softmax(fetch_level_eb(predict_eb,i),dim=-1)
#         sns.heatmap(data,cmap='Greens', ax=ax,cbar=False,
#             xticklabels=order_manager.levels[i])
#         ax.set_ylabel('sample id')
#         ax.set_title(f'Pred {taxo_hierars[i]}')
#         # if i!=len(order_manager.levels)
#         cbar = ax.collections[0].colorbar
        
#         ax=axes[1][i]
#         data=fetch_level_eb(true_eb,i)
#         sns.heatmap(data,cmap='Greens', ax=ax,cbar=False,
#             xticklabels=order_manager.levels[i])
#         ax.set_ylabel('sample id')
#         ax.set_title(f'Pred {taxo_hierars[i]}')

#     # %%

#     # %%
#     world_size = 1
#     order_manager=OrderManager('dataset/taxo_data/hierarchy_order.pkl')
#     hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=500)
#     hierar_esmmodel.load_state_dict(torch.load('train/240426-130251-ep1.pt')) #"train/240426-141811-ep1.pt"
#     dataset=ConcatProteinDataset('dataset/taxo_data/proseq_taxo.pkl',order_manager)
#     train_generator = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=world_size)
#     hierar_loss=HierarchicalLossNetwork(order_manager)
#     optimizer = Adam(hierar_esmmodel.parameters(), lr=1e-4)

#     def setup(rank, world_size):
#         # print(f'my rank: {rank}')
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#         dist.init_process_group("gloo", rank=rank, world_size=world_size)
#     def cleanup():
#         dist.destroy_process_group()
        
#     def train(rank, world_size):
#         setup(rank, world_size)
#         # model & optimizer
#         ddp_model = DDP(hierar_esmmodel.to(rank), device_ids=[rank],find_unused_parameters=True)
#         ddp_model.train()
#         optimizer = Adam(hierar_esmmodel.parameters(), lr=1e-4)
#         #dataset & dataloader
#         sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#         loader = DataLoader(dataset, batch_size=2, sampler=sampler)
#         test_loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
#         #train
#         # epoch_loss = []
#         # epoch_superclass_accuracy = []
#         # epoch_subclass_accuracy = []
        
#         for epoch in range(20):
#             if rank == 0:
#                 ddp_model.eval()
#                 with torch.set_grad_enabled(False):
#                     predict_labels,true_labels=[],[]
#                     for _,sample in enumerate(test_loader):
#                         batch_x, batch_y = sample['seq'], sample['taxo']
#                         ipt=[i.upper().replace('#','<eos>') for i in batch_x]
#                         ipt = hierar_esmmodel.tokenizer(ipt, return_tensors="pt",padding=True,truncation=True,max_length=hierar_esmmodel.max_length).to(rank)
#                         x=ddp_model(ipt)
#                         p,t=[],[]
#                         predict_idx=[torch.argmax(i, dim=1).to('cpu') for i in x]
#                         for i,l in enumerate(order_manager.levels):
#                             cur_predict=predict_idx[i].numpy()
#                             cur_true=batch_y[i].numpy()
#                             cur_plabel=np.vectorize(lambda x:l[x])(cur_predict)
#                             cur_tlabel=np.vectorize(lambda x:l[x])(cur_true)
#                             p.append(cur_plabel)
#                             t.append(cur_tlabel)
#                         predict_labels.append(np.stack(p).T)
#                         true_labels.append(np.stack(t).T)
#                         if _>100:
#                             break
#                     predict_labels=np.vstack(predict_labels)
#                     true_labels=np.vstack(true_labels)
#                     fig,ax=plt.subplots(1,1,figsize=[8,24])
#                     sns.heatmap(predict_labels==true_labels, cmap=['grey', 'red'], ax=ax,
#                     cbar_kws={"ticks":[0.25, 0.75]},xticklabels=['Realm', 'Kingdom','Phylum', 'Class','Order'])
#                     ax.set_ylabel('sample id')
#                     cbar = ax.collections[0].colorbar
#                     cbar.set_ticklabels(['wrong', 'correct'])
#                     fig.savefig(f'train/{time_str}-ep{epoch+1}-performance.png')
#                 ddp_model.train()
                
#             for sample in loader:
#                 optimizer.zero_grad()
                
#                 batch_x, batch_y = sample['seq'], sample['taxo']
                
#                 ipt=[i.upper().replace('#','<eos>') for i in batch_x]
#                 ipt = hierar_esmmodel.tokenizer(ipt, return_tensors="pt",padding=True,truncation=True,max_length=hierar_esmmodel.max_length).to(rank)
#                 x=ddp_model(ipt)
                
#                 y=[i.to(rank) for i in batch_y]
                
#                 loss:torch.Tensor = hierar_loss(x,y,rank)
#                 loss.backward()
#                 optimizer.step()
                
#                 logger.info(f'rank:{rank} loss: {loss.item():.3f}')
#                 accuracies=[cal_accuracy(x[i],y[i]) for i in range(len(x))]
#                 logger.info(f'rank:{rank} level-wise acc: {'\t'.join([f'{i:2.2f}' for i in accuracies])}')
                
#             if rank == 0 and epoch%2==1:
#                 torch.save(hierar_esmmodel.state_dict(), f'train/{time_str}-ep{epoch+1}.pt')
                
            
        
#         cleanup()
    
#     # %%
#     if __name__ == "__main__":   
#         logging.basicConfig(filename=f'train/{time_str}.log', 
#                     filemode='a',           
#                     level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(module)s - %(filename)s - %(message)s')
#         torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
    

    