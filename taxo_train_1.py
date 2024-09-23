# %%
from typing import Union,List,Any,Dict,Optional
import pickle as pkl
import pandas as pd
from torch import nn
import torch
# from transformers import EsmModel, EsmConfig, EsmTokenizer
# from torch.nn.modules.loss import _Loss
# import torch.nn.functional as F
from torch.optim import Adam,AdamW
# import random
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
# import networkx as nx
# from PyPDF2 import PdfReader, PdfWriter
from torch.utils.tensorboard import SummaryWriter
from hierataxo import (
    OrderManager,ConcatProteinDataset,
    HierarESM,HierarchicalLossNetwork,
    cal_accuracy,set_seed)
# TODO implement tensorboard
logger=logging.getLogger()
time_str=time.strftime("%y%m%d-%H%M%S", time.localtime())    
# %%
if __name__=='__main__':
    seed=int(sys.argv[1])
    device=3
    batch_size=2
    batch_size_val=25
    max_domain=15
    acc_step=20
    valid_step=1000
    max_length=500
    to_freeze=3
    set_seed(seed)
    
    order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                            level_names=['Kingdom','Phylum','Class','Order'])
    hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=max_length,to_freeze=to_freeze,device=device)
    # torch.save(hierar_esmmodel.state_dict(), f'seed-{seed}-ep-0.pt')
    # sys.exit(0)
    hierar_loss=HierarchicalLossNetwork(order_manager)
    optimizer = AdamW(hierar_esmmodel.parameters(), lr=1e-4)

    dataset=ConcatProteinDataset('taxo_data/proseq_taxo_1.pkl',order_manager)
    trainset,valset=random_split(dataset,[0.9,0.1])
    train_generator = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_generator = DataLoader(valset, batch_size=batch_size_val, shuffle=True, num_workers=batch_size)
    
    odir=f'train/v1_{time_str}_seed{seed}'
    os.mkdir(odir)
    writer = SummaryWriter(log_dir=f'{odir}/log')
    # %% train
    #TODO protocol to Task object
    
    cur_acc_step=0
    global_step=0
    best_val_acc=0
    hierar_esmmodel.train()

    def init_acc_logs():
        pred_dict={i:[] for i in order_manager.level_names}
        true_dict={i:[] for i in order_manager.level_names}
        return pred_dict,true_dict
    
    pred_dict,true_dict=init_acc_logs()

    def train(sample):
        optimizer.zero_grad()
        batch_name, batch_x, batch_y,sentence_mask =sample['name'], sample['seq'], sample['taxo'],sample['sentence_mask']
        model_input=hierar_esmmodel.process_sample(sample)
        x:List[torch.Tensor]=hierar_esmmodel(*model_input)  
        y=[i.to(device) for i in batch_y]
        loss:torch.Tensor = hierar_loss(x,y,device)
        return [i.detach().cpu() for i in x],loss
    
    
    def valid(val_generator:DataLoader):
        global best_val_acc
        with torch.set_grad_enabled(False):
            hierar_esmmodel.eval()
            x_s,y_s,ebs,labels=[],[],[],[]
            for step, sample in tqdm(enumerate(val_generator)):
                # batch_y=sample['taxo']
                x_s.append([i.detach().to('cpu') for i in 
                            hierar_esmmodel(*hierar_esmmodel.process_sample(sample))])
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
            writer.add_scalar('val_loss',hierar_loss(x_s,y_s,'cpu').item(),global_step)
            writer.add_embedding(mat=torch.concat(ebs,dim=0),
                                metadata=labels,
                                global_step=global_step,
                                tag='valid/embeddings')
            if best_val_acc<accuracies[order_manager.level_names[-1]]:
                best_val_acc=accuracies[order_manager.level_names[-1]]
            hierar_esmmodel.train()
            
    
    
    for epoch in range(60):
        for step, sample in tqdm(enumerate(train_generator)):
            x,loss=train(sample)
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
                valid(val_generator)
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
    

    