# %%
from taxo_train_1 import OrderManager,HierarESM,ConcatProteinDataset,HierarchicalLossNetwork,process_batch
from typing import Union,List,Any,Dict,Optional,Tuple
import pickle as pkl
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
# from transformers import EsmModel, EsmConfig, EsmTokenizer
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
# from torch.utils.data import DataLoader, Dataset, DistributedSampler
from copy import deepcopy
from math import ceil
import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler
from functools import reduce,partial
import random
import pandas as pd
# %%
# if __name__=='__main__':
    # import sys
    # ep=sys.argv[1]
    # device=int(sys.argv[2])
ep=6
device=0
batch_size=50 # maximum valid batch size on 2080Ti: 100; train: 2 (or maybe 3?)
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
train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(batch_size,16))

o=[]
with torch.set_grad_enabled(False):
    # i=0
    for step, sample in enumerate(train_generator):
        batch_name,batch_x,batch_y=hierar_esmmodel._embed(sample)
        # i+=1
        o.append((batch_name,[i.to('cpu') for i in batch_x],[i.to('cpu') for i in batch_y]))
        torch.cuda.empty_cache()
        # if i>5:
        #     break
    
    
        
# %%
# import numpy as np
# from sklearn.decomposition import PCA
embed_size=o[0][1][0].shape[-1]
def merge_batch_infer(eb_opts)->Tuple[List[str],np.ndarray,np.ndarray]:
    names=reduce(lambda x,y:x+y, [i[0] for i in eb_opts])
    embeds=torch.vstack([torch.hstack(i[1]) for i in eb_opts]).numpy()
    labels=torch.vstack([torch.vstack(i[2]).T for i in eb_opts]).numpy()
    return names,embeds,labels
#%%
names,embeds,labels=merge_batch_infer(o)
pkl.dump((names,embeds,labels),open('embeds.pkl','wb'))

# # l2_embed=o[2].to('cpu').numpy()
# # l2_label=[order_manager.levels[2][i] for i in batch_y[2].numpy()]

# embedding = StandardScaler().fit_transform(embeds[:,embed_size*level_to_check:embed_size*(level_to_check+1)])
# embedding = reducer.fit_transform(embedding) # toxic for pylance
reducer = umap.UMAP()
embedding_full = StandardScaler().fit_transform(embeds)
# embedding_full = reducer.fit_transform(embedding_full) # toxic for pylance

reducer = umap.UMAP()
embedding_last = StandardScaler().fit_transform(embeds[:,embed_size*3:embed_size*(3+1)])
# embedding_last = reducer.fit_transform(embedding_full) # toxic for pylance
# level_to_check=1

titles=order_manager.level_names
for level_to_check,title in enumerate(titles):
    # labels=[order_manager.levels[2][i] for i in _[2][:,level_to_check]]

    level_label=labels[:,level_to_check]
    unque_labels=pd.Series(level_label).unique()
    readable_labels={i:order_manager.levels[level_to_check][i] for i in unque_labels}

    # color_ids={k:i for i,k in enumerate(pd.Series(l2_label).value_counts().index)}

    colors=list(mcolors.CSS4_COLORS.keys())
    sampled_colors=random.sample(colors,len(pd.Series(level_label).value_counts()))
    # color_ids={k:random.randint(0,len(colors)-1) for i,k in enumerate(pd.Series(l2_label).value_counts().index)}
    color_ids={k:sampled_colors[i] for i,k in enumerate(unque_labels)}


    fig,ax=plt.subplots(1,1,figsize=(25,25))
    ax:Axes
    for label in unque_labels:
        ax.scatter(
            embedding_last[level_label==label,0],
            embedding_last[level_label==label,1],
            c=color_ids[label],
            label=readable_labels[label])
    ax.legend()
    ax.set_title(title,{'fontsize':28})
    plt.savefig(f'full-{title}.svg')
    plt.close(fig)
    
    fig,ax=plt.subplots(1,1,figsize=(25,25))
    ax:Axes
    for label in unque_labels:
        ax.scatter(
            embedding_full[level_label==label,0],
            embedding_full[level_label==label,1],
            c=color_ids[label],
            label=readable_labels[label])
    ax.legend()
    ax.set_title(title,{'fontsize':28})
    plt.savefig(f'last-{title}.svg')
    plt.close(fig)
    # plt.gca().set_aspect('equal', 'datalim')
    # scaled_penguin_data = StandardScaler().fit_transform(penguin_data)