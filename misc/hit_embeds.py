# %%
from transformers import EsmModel, EsmTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import Union,List,Any,Dict,Optional
# from neomodel import db,config
from neomodel.sync_.core import Database
from neomodel.integration.pandas import to_dataframe,to_series
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle as pkl
# %%
# config.DATABASE_URL = 'bolt://neo4j:WBrtpKCUW28e@52.4.3.233:7687'
# db.cypher_query
url='bolt://neo4j:WBrtpKCUW28e@52.4.3.233:7687'
model_name:str='facebook/esm2_t6_8M_UR50D'
device=0
max_length=500

db=Database()
db.set_connection(url)
tokenizer = EsmTokenizer.from_pretrained(model_name)
backbone:EsmModel = EsmModel.from_pretrained(
    model_name,add_pooling_layer=False).to(device)

def curate_seq(s:str):
    return s.replace('-','').upper()

def tokenize(s:pd.Series):
    return tokenizer(s, return_tensors="pt",padding='max_length',
            truncation=True,max_length=max_length).to(device)
    
df=to_dataframe(db.cypher_query('MATCH (h:Hit) RETURN h.name as name, h.aligned_seq as seq'))
s=df['seq'].apply(curate_seq)
ipt=tokenize(s.to_list())

batchsize=128 # max:128*5/4=160

embeds:List[np.ndarray]=[]
with torch.set_grad_enabled(False):
    # step=0
    for i in tqdm(range(0,len(df),batchsize)):
        s=df['seq'].iloc[i:i+batchsize].apply(curate_seq)
        # name=df['name'].iloc[i:i+batchsize]
        ipt=tokenize(s.to_list())
        opt=backbone(**ipt)
        opt=opt.last_hidden_state[:,0,:].to('cpu').numpy()
        embeds.append(opt)
        # step+=1
        # if step>3:
        #     break
embeds_array=np.vstack(embeds)
name=df['name'].to_numpy()


pkl.dump({'name':name,'embedding':embeds_array},open('eb.pkl','wb'))

# class HitDataset(Dataset):
#     def __init__(self,url:str) -> None:
#         '''
#         for expedience, begin with df's pkl
#         TODO fetch directly from neo4j
#         '''
#         super().__init__()
#         self.db=Database(url=url)
        
#     def __len__(self)

# %%
