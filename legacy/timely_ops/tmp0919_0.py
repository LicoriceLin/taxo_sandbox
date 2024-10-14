from __future__ import annotations
from hierataxo import  OrderManager
from hierataxo.dataset import ConcatProteinDataModule
import pickle as pkl
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hierataxo.util import xkcd_color,hide_spline
from hierataxo.util import named_taxo_palette
from hierataxo.plot_taxotree import layout_modification,manual_modify_graph_pos,hex_path,to_label,order_manager
import matplotlib.pyplot as plt
from typing import List,Dict

# order_manager=OrderManager(
#     pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
#     level_names=['Kingdom','Phylum','Class','Order'],
#     layout_prog='dot',
#     layout_modification=layout_modification,
#     )
def fetch_pred_dist(s:pd.Series):
    return [torch.tensor([s[f'{i}_predict']]) for i in order_manager.level_names]

def distribution_to_color_dict(s:pd.Series)->Dict[str,tuple]:
    '''
    `s`: `pd.Series` from infer results
    '''
    p=fetch_pred_dist(s)
    return order_manager.distribution_to_color_dict(p)

def fetch_gt_dist(s:pd.Series):
    return order_manager.order_to_onehot(s['taxo_label'])

def taxo_label_to_color_dict(s:pd.Series)->Dict[str,tuple]:
    return order_manager.distribution_to_color_dict(
        order_manager.order_to_onehot(s['taxo_label']))

def to_prob_tensors(pred:pd.DataFrame)->Dict[str,torch.Tensor]:
    o={f'{i}_gt':[] for i in order_manager.level_names}
    o.update({f'{i}_pred':[] for i in order_manager.level_names})
    for _,s in pred.iterrows():
        gts=order_manager.order_to_onehot(s['taxo_label'])
        preds=fetch_pred_dist(s)
        for level,gt,p in zip(order_manager.level_names,gts,preds):
            o[f'{level}_gt'].append(gt)
            o[f'{level}_pred'].append(p)
    for k,v in o.items():
        o[k]=torch.concat(v)
    return o

def cal_acc(pred:pd.DataFrame):
    prob_tensors=to_prob_tensors(pred)
    o={}
    for level in order_manager.level_names:
        o[level]=torch.sum(torch.argmax(prob_tensors[f'{level}_pred'],dim=1
            )==torch.argmax(prob_tensors[f'{level}_gt'],dim=1)).item()/len(pred)
    return o

