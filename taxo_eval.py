# %%
'''
%load_ext autoreload
%autoreload 1
%aimport taxo_train_1
%aimport

%load_ext autoreload
%autoreload 2
'''
from taxo_train_1 import OrderManager,HierarESM,ConcatProteinDataset,HierarchicalLossNetwork,process_batch
from typing import Union,List,Any,Dict,Optional
import pickle as pkl
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import EsmModel, EsmConfig, EsmTokenizer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.optim import Adam
import random
import numpy as np
import logging
import time
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from copy import deepcopy
from math import ceil
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
# %%
if __name__=='__main__':
    import sys
    ep=sys.argv[1]
    device=int(sys.argv[2])
    batch_size=25 # maximum valid batch size on 2080Ti: 100; train: 2 (or maybe 3?)
    max_domain=15
    acc_step=20
    max_length=500
    to_freeze=3
    model=f'/home/rnalab/zfdeng/pgg/Deep_Hierarchical_Classification/train/240430-223935/ep-{ep}.pt'
    order_manager=OrderManager(pkl.load(open('dataset/taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                            level_names=['Kingdom','Phylum','Class','Order'])
    hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=max_length,to_freeze=to_freeze,device=device)
    hierar_esmmodel.load_state_dict(torch.load(model))
    hierar_esmmodel.eval()
    dataset=ConcatProteinDataset('dataset/taxo_data/proseq_taxo.pkl',order_manager)
    train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    '''
    cuda utility:
    gpu_info = torch.cuda.get_device_properties(0)
    torch.cuda.memory_allocated(0)
    torch.cuda.memory_reserved(0)
    '''
    with torch.set_grad_enabled(False):
        with PdfPages(f'tmp-ep-{ep}.pdf') as pdf:
            for step, sample in tqdm(enumerate(train_generator)):
                hierar_esmmodel.report_eval(sample,pdf)
                if step>20:
                    break
# sys.exit(0)
        # names,batch_x, batch_y,sentence_mask = sample['name'], sample['seq'], sample['taxo'],sample['sentence_mask']
        # domains=[]
        # for i in range(batch_size):
        #     for j in range(max_domain):    
        #         domains.append(batch_x[j][i])
        # domains_mask=torch.stack(sentence_mask).T.reshape(-1).to(device)
        # batch_name,domains,batch_y,domains_mask=process_batch(sample)
        # ipts=hierar_esmmodel.parse_sentence(domains)
        # x=hierar_esmmodel(ipts,domains_mask)    
        # y=[i.to(device) for i in batch_y]
        # break
#%%
# def recur_size_count(hierarchical_labels:dict,cur_key:str='root',null_level=0):
#     o={}
#     null_level_dict={}
#     if cur_key=='root':
#         hierarchical_labels_=deepcopy(hierarchical_labels)
#     else:
#         hierarchical_labels_=hierarchical_labels
        
#     if 'Null' in hierarchical_labels_:
#         hierarchical_labels_[f"{cur_key}'s Null"]=hierarchical_labels_.pop('Null')
#         null_level_dict[f"{cur_key}'s Null"]=null_level
        
#     for k,v in hierarchical_labels_.items():
#         if isinstance(v,list):
#             if 'Null' in v:
#                 v[v.index('Null')]=f"{k}'s Null"
#                 null_level_dict[f"{k}'s Null"]=null_level+1
#             o[k]=len(v)
#         elif isinstance(v,dict):
#             sub_o,_,sub_null_level_dict=recur_size_count(v,cur_key=k,null_level=null_level+1)
#             o[k]=sum(sub_o.values())
#             o.update(sub_o)
#             null_level_dict.update(sub_null_level_dict)
#     return o,hierarchical_labels_,null_level_dict
# o,hierarchical_labels_,null_level_dict=recur_size_count(order_manager.hierarchical_labels)
# %%
# import networkx as nx
# G=nx.DiGraph()
# def to_graph(G:nx.DiGraph,hierarchical_labels:dict,cur_key:str='root'):
#     for k,v in hierarchical_labels.items():
#         G.add_edge(cur_key,k)
#         if isinstance(v,list):
#             for i in v:
#                 G.add_edge(k,i)
#         elif isinstance(v,dict):
#             for i,sub_d in v.items():
#                 G.add_edge(k,i)
#                 to_graph(G,sub_d,i)

# to_graph(G,order_manager.split_null_hierarchical_labels)

# # x=[i.to('cpu') for i in x]
# import matplotlib.colors as mcolors
# color_levels=[mcolors.LinearSegmentedColormap.from_list(
#     i,[[1,1,1],mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{i}'])]) for i in ['pinkish red','purply','ocean','peach']]
# def x_to_color_dict(x:List[torch.Tensor],
#                     order_manager:OrderManager,
#                     batch_i:int=0,
#                     color_levels=color_levels,
#                     ):
#     null_level_dict=order_manager.null_level_dict
#     color_dict={'root':mcolors.to_rgb('white')}
#     for level_id,level in enumerate(order_manager.levels):
#         pred:torch.Tensor=x[level_id][batch_i].to('cpu')
#         cmap=color_levels[level_id]
#         level_color_dict={}
#         pred:torch.Tensor
#         pred=F.softmax(pred.to('cpu'),dim=-1)
#         for i,term in enumerate(level):
#             level_color_dict[term]=cmap(pred[i])
#         if 'Null' in level_color_dict:
#             null_color=level_color_dict.pop('Null')
#             split_nulls={k:null_color for k,v in null_level_dict.items() if v==level_id}
#             level_color_dict.update(split_nulls)
#         color_dict.update(level_color_dict)
#     return color_dict
    
    
# def y_to_color_dict(y:List[torch.Tensor],
#                     order_manager:OrderManager,
#                     batch_i:int=0,
#                     color_levels=color_levels,
#                     ):
#     null_level_dict=order_manager.null_level_dict
#     color_dict={'root':mcolors.to_rgb('white')}
#     for level_id,level in enumerate(order_manager.levels):
#         # TODO:use order_manager's idx_to_onehot and merge with x_to_color_dict
#         true:torch.Tensor=F.one_hot(y[level_id].to('cpu'),len(level))[batch_i].float().numpy()
#         cmap=color_levels[level_id]
#         level_color_dict={}
#         for i,term in enumerate(level):
#             level_color_dict[term]=cmap(true[i])
#         if 'Null' in level_color_dict:
#             null_color=level_color_dict.pop('Null')
#             split_nulls={k:null_color for k,v in null_level_dict.items() if v==level_id}
#             level_color_dict.update(split_nulls)
#         color_dict.update(level_color_dict)
#     return color_dict

# color_dict=x_to_color_dict(x,order_manager,batch_i=0)
# fig,ax=plt.subplots(1,1,figsize=(16, 16))
# ax:Axes
# pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
# to_label=lambda x:x if 'Null' not in x else 'Null'
# nx.draw_networkx_nodes(G,pos,ax=ax,node_size=300
#                        ,node_color=[color_dict.get(n,(0,0,0,0)) for n in G],edgecolors='black')
# nx.draw_networkx_edges(G,pos)
# nx.draw_networkx_labels(G,pos,labels={n:to_label(n) for n in G})

        
        
        
    
# color_dict=
#%%
# peeled_name_hierarchy=[]
# levels=order_manager.levels
# peeled_numeric_hierarchy=order_manager.peeled_numeric_hierarchy

# for i in range(len(peeled_numeric_hierarchy)):
#     hierarchy=peeled_numeric_hierarchy[i]
#     level=levels[i+1]
#     higher_level=levels[i]
#     d={higher_level[k]:[level[i] for i in v] for k,v in hierarchy.items()}
#     peeled_name_hierarchy.append(d)
# for level,hierarchy in zip(order_manager.levels[1::-1], order_manager.peeled_numeric_hierarchy[::-1]):
#     d={k:v for k,v in hierarchy.items()}
# def init_box_position():


            


# x_grid=np.linspace(0,100,len(levels)+2)
# cur_x=1       
# for h in peeled_name_hierarchy:
#     x=x_grid[cur_x]
#     y_grid=np.linspace(0,100,len(levels[-1])+2)
#     next_y_grid=np.linspace(0,100,sum([len(i) for i in h.values()])+2)
    
    


    
