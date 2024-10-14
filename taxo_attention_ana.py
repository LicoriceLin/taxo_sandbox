# %%
from hierataxo import OrderManager,HierarESM,ConcatProteinDataset,HierarchicalLossNetwork
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
import seaborn as sns

# %%
# ep=6
device=1
# batch_size=25 # maximum valid batch size on 2080Ti: 100; train: 2 (or maybe 3?)
max_domain=15
acc_step=20
max_length=500
to_freeze=3
n_steps=100

model='/home/rnalab/zfdeng/pgg/taxo_sandbox/train/v1_240521-210442_seed77/ep-28.pt'
order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                        level_names=['Kingdom','Phylum','Class','Order'])
hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=max_length,to_freeze=to_freeze,device=device)
hierar_esmmodel.load_state_dict(torch.load(model))
hierar_esmmodel.eval()
dataset=ConcatProteinDataset('taxo_data/proseq_taxo_1.pkl',order_manager)
# train_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

# %%
embedding_output = None
embedding_gradients = None

def forward_hook(module, input, output):
    global embedding_output
    embedding_output = output

def backward_hook(module, grad_input, grad_output):
    global embedding_gradients
    embedding_gradients = grad_output[0]
from captum.attr import IntegratedGradients
hierar_esmmodel.backbone.embeddings.token_dropout=False
hierar_esmmodel.backbone.config.token_dropout=False
embedding_layer = hierar_esmmodel.backbone.get_input_embeddings()

#hierar_esmmodel.backbone.embeddings.word_embeddings
embedding_layer.register_forward_hook(forward_hook)
embedding_layer.register_full_backward_hook(backward_hook)
# embedding_layer = hierar_esmmodel.backbone.embeddings

entry_id=1
# with torch.set_grad_enabled(False):
def name_to_idx(name:str):
    return np.where(dataset.data.index==name)[0][0]


sample=dataset.fetch_single(entry_id)

# batch_name,domains,batch_y,domains_mask=hierar_esmmodel.process_batch(sample)
input_ids,attention_mask,sentence_mask=hierar_esmmodel.process_sample(sample)
fake_ids=torch.clone(input_ids).detach()
fake_ids[fake_ids>3]=hierar_esmmodel.tokenizer.get_vocab()['<mask>']

inputs_embeds=embedding_layer(input_ids)
fake_inputs_embeds=embedding_layer(fake_ids)

def forward_func(inputs_embeds:torch.Tensor,attention_mask:torch.Tensor,domains_mask:torch.Tensor):
    return hierar_esmmodel(**{'input_ids':None,'inputs_embeds':inputs_embeds,'attention_mask':attention_mask,'sentence_mask':domains_mask})[-1]
ig = IntegratedGradients(forward_func)
o_attribute=ig.attribute(inputs_embeds,fake_inputs_embeds,target=sample['taxo'][-1].item(),internal_batch_size=5,
    n_steps=n_steps,additional_forward_args=(attention_mask,sentence_mask))
mapping_gradients=o_attribute[attention_mask.bool()].detach().to('cpu')

# %%
def hmp():
    sns.heatmap(mapping_gradients.detach().to('cpu'),cmap='RdBu')
    sns.heatmap(torch.sqrt(torch.sum(mapping_gradients** 2, dim=-1)).unsqueeze(0),cmap='Blues')

def map_to_pdb(mapping_gradients,idx,pdbfile):
    from pymol import cmd
    res_gradients=mapping_gradients[idx[0]:idx[1]]
    l2_res_gradients=torch.sqrt(torch.sum(res_gradients** 2, dim=-1))
    cmd.load(pdbfile, 'entry0')
    all_objects = cmd.get_object_list('all')
    residues = cmd.get_model(all_objects[0])
    for l,(b,e) in zip(l2_res_gradients.tolist(),residues.get_residues()):
        cmd.alter(f'index {b}-{e}',f"b={l}")
    cmd.save('o.pdb')
    cmd.delete('entry0')
    
# %%
from torch.nn.modules.activation import MultiheadAttention
multihead_attn:torch.nn.Module=hierar_esmmodel.decoder_1.layers[0].multihead_attn

def attention_ana(multihead_attn:MultiheadAttention,sample):
    keeper={}
    def input_hook(module, args,kargs,output):
        keeper['args']=args
        keeper['kargs']=kargs
    hook_handle = multihead_attn.register_forward_hook(input_hook,with_kwargs=True)
    with torch.set_grad_enabled(False):
        _=hierar_esmmodel(*hierar_esmmodel.process_sample(sample))
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
        
sns.heatmap(attention_ana(hierar_esmmodel.opt_initiator.self_attn,dataset.fetch_single(2))[0].tolist())
    
    


# %%
domains_mask=sentence_mask.to(hierar_esmmodel.device)
ipts=hierar_esmmodel.parse_sentence(domains)
mapping_domains=''.join([f'#{i}@' for i,j in zip(domains,domains_mask) if j])
mapping_domains_lens=[len(i)+2 for i,j in zip(domains,domains_mask) if j]
# ipts: 'input_ids', 'attention_mask'


# y=hierar_esmmodel.order_manager.idx_to_onehot(batch_y)



# %%
# Vmethyltransf, FTO_NTD, Viral_helicase1, RdRP_3
# fig,axes=plt.subplots(order_manager.total_level,1,figsize=(20,order_manager.total_level*4))
# axes:List[Axes]

fig,ax=plt.subplots(1,1,figsize=(len(mapping_domains)*0.07,order_manager.total_level*2)) # TODO
ax:Axes
o_grad=[]

for level_id in range(order_manager.total_level):
    x:List[torch.Tensor]=hierar_esmmodel(ipts,domains_mask)
    y=[i.to(device) for i in batch_y]
    l=nn.CrossEntropyLoss()(x[level_id],y[level_id])
    l.backward()
    mapping_gradients=embedding_gradients.view(-1,embedding_gradients.shape[-1])[ipts['attention_mask'][domains_mask==1].view(-1)==1]
    #TODO visualize on sequence/3D structure
    l2_norm = F.layer_norm(mapping_gradients,
            mapping_gradients.shape, eps=10e-10)
    # l2_norm = F.layer_norm(mapping_gradients[:,ipts['attention_mask'][0]==1],
    #         (mapping_gradients.shape[-1],), eps=10e-10)
    l2_norm = torch.sqrt(torch.sum(l2_norm ** 2, dim=-1))
    # l2_norm=
    o_grad.append(F.layer_norm(l2_norm,
            l2_norm.shape, eps=10e-10).tolist())
    # o_grad.append(((l2_norm-l2_norm.min(dim=-1).values)/(
    #     l2_norm.max(dim=-1).values-l2_norm.min(dim=-1).values)
    #     ).cpu().numpy().reshape(-1).tolist())

sns.heatmap(o_grad,yticklabels=order_manager.level_names,ax=ax)

ax.set_xticks(range(len(mapping_domains)),list(mapping_domains),fontsize=3,rotation=0)

# %%


def to_unk(l:int):
    if l>0: return ['<mask>']*l
    else: return ['']
fake_domains=[to_unk(len(i)) for i in domains]
fake_ipts=hierar_esmmodel.tokenizer(
    fake_domains, return_tensors="pt",padding='max_length',truncation=True,
    max_length=hierar_esmmodel.max_length,is_split_into_words=True).to(device)

inputs_embeds=embedding_layer(ipts['input_ids'])
fake_inputs_embeds=embedding_layer(fake_ipts['input_ids'])

y=[i.to(device) for i in batch_y]




ig = IntegratedGradients(forward_func)
o_attribute=ig.attribute(inputs_embeds,fake_inputs_embeds,target=y[-1].item(),internal_batch_size=5,
    n_steps=n_steps,additional_forward_args=(ipts['attention_mask'],domains_mask))

mapping_gradients=o_attribute[domains_mask==1].view(-1,o_attribute.shape[-1])[ipts['attention_mask'][domains_mask==1].view(-1)==1]
l2_norm = F.layer_norm(mapping_gradients,
            mapping_gradients.shape, eps=10e-10)
l2_norm = torch.sqrt(torch.sum(l2_norm ** 2, dim=-1))
fig,ax=plt.subplots(1,1,figsize=(len(mapping_domains)*0.07,15)) 
sns.heatmap([l2_norm.tolist()],ax=ax,yticklabels=False)
ax.set_xticks(np.cumsum(np.array([0]+mapping_domains_lens))[:-1],
              dataset.data.iloc[entry_id]['family']
              ,fontsize=3,rotation=0)
fig.savefig('AAV1.svg')
from functools import partial
def forward_func_cross(inputs_embeds:torch.Tensor,attention_mask:torch.Tensor,domains_mask:torch.Tensor,y:int,level_id:int):
    x=hierar_esmmodel({'inputs_embeds':inputs_embeds,'attention_mask':attention_mask},domains_mask)
    y=F.one_hot(torch.tensor([y]),x[level_id].shape[-1]).repeat(x[level_id].shape[0],1).type_as(x[level_id]) 
    return nn.CrossEntropyLoss(reduction='none')(x[level_id],y)
ig = IntegratedGradients(partial(forward_func_cross,y=y[3].item(),level_id=3))
n_steps=50
o_attribute=ig.attribute(inputs_embeds,fake_inputs_embeds,internal_batch_size=5,
    n_steps=n_steps,additional_forward_args=(ipts['attention_mask'],domains_mask))

# %%
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
entry_id=1
sample=dataset.fetch_single(entry_id)
seq_num=len([i for i in sample['seq'] if len(i[0])>0])
seq_names=dataset.data.iloc[entry_id]['family']
with torch.set_grad_enabled(False):
    with PdfPages(f'tmp-AAV1-modi2.pdf') as pdf:
        hierar_esmmodel.report_eval(sample,pdf)
        if seq_num>1:
            for mask_count in range(1,seq_num):
                for mask in combinations(range(seq_num),mask_count):
                    sample_modi=deepcopy(sample)
                    for m in mask:
                        sample_modi['sentence_mask'][m][0]=0                        
                    # lefts=[i for i in range(seq_num) if i not in mask]
                    sample_modi['name']=(sample_modi['name'][0]+'\n'+','.join(
                        [seq_names[i] for i in range(seq_num) if i not in mask]),)
                    hierar_esmmodel.report_eval(sample_modi,pdf)

# %%
from neomodel import db,config
from neomodel.integration.pandas import to_dataframe
config.DATABASE_URL = 'bolt://neo4j:WBrtpKCUW28e@52.4.3.233:7687'
def pandas_query(q:str,**kwargs):
    params=kwargs
    return to_dataframe(db.cypher_query(q,params,resolve_objects=True))
fasta_hits=pandas_query(
        '''
        MATCH (fasta:Fasta)-[hasreg:hasRegion]->(region:HitRegion)-[hasaff:hasAffiliate {representation:TRUE}]-(hit:Hit)--(f:HitFamily)
        RETURN fasta.name as name,hasreg.regid as regid,hit.aligned_seq as seq,fasta.taxonomy as taxonomy,f.name as family
        ''')
seqs={}
taxo={}
family={}
for name,subg in fasta_hits.groupby(by='name'):
    seqs[name]='#'.join(subg.sort_values(by='regid')['seq'])
    taxo[name]=subg['taxonomy'].iloc[0]
    family[name]=tuple(subg['family'].to_list())
o=pd.DataFrame([seqs,taxo,family]).T
o.columns=['seq','taxo','family']