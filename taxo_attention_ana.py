# %%
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
ep=6
device=2
batch_size=25 # maximum valid batch size on 2080Ti: 100; train: 2 (or maybe 3?)
max_domain=15
acc_step=20
max_length=500
to_freeze=3
model=f'/home/rnalab/zfdeng/pgg/Deep_Hierarchical_Classification/train/240430-223935/ep-{ep}.pt'
order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                        level_names=['Kingdom','Phylum','Class','Order'])
hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=max_length,to_freeze=to_freeze,device=device)
hierar_esmmodel.load_state_dict(torch.load(model))
hierar_esmmodel.eval()
dataset=ConcatProteinDataset('taxo_data/proseq_taxo.pkl',order_manager)
train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)


embedding_output = None
embedding_gradients = None

def forward_hook(module, input, output):
    global embedding_output
    embedding_output = output

def backward_hook(module, grad_input, grad_output):
    global embedding_gradients
    embedding_gradients = grad_output[0]

embedding_layer = hierar_esmmodel.backbone.embeddings
embedding_layer.register_forward_hook(forward_hook)
embedding_layer.register_full_backward_hook(backward_hook)


# with torch.set_grad_enabled(False):
batch_name,domains,batch_y,domains_mask=process_batch(dataset.fetch_single(0),hierar_esmmodel.max_domain)
domains_mask=domains_mask.to(hierar_esmmodel.device)
ipts=hierar_esmmodel.parse_sentence(domains)
x:List[torch.Tensor]=hierar_esmmodel(ipts,domains_mask)

# y=hierar_esmmodel.order_manager.idx_to_onehot(batch_y)
y=[i.to(device) for i in batch_y]
l=nn.CrossEntropyLoss()(x[-1],y[-1])
l.backward()

#TODO visualize on sequence/3D structure
squared_sum = torch.sum(embedding_gradients ** 2, dim=2)
l2_norm = torch.sqrt(squared_sum)
import seaborn as sns
sns.heatmap(l2_norm[:,ipts['attention_mask'][0]==1].tolist())