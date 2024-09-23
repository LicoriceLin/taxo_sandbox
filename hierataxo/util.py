# %%
from typing import Union,List,Any,Dict,Optional,Callable,Literal,Tuple
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
from pymol import cmd
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
import matplotlib.path as mpath
from logging import warning
def hide_spline(ax:Axes,hide:str='trlb'):
    if 't' in hide:
        ax.spines['top'].set_visible(False)
    if 'r' in hide:    
        ax.spines['right'].set_visible(False)
    if 'l' in hide:
        ax.spines['left'].set_visible(False)
    if 'b' in hide:
        ax.spines['bottom'].set_visible(False)
    
def circle(center:tuple,outer:tuple,**kwargs):
    '''
    kwargs: for `plt.Circle`
    '''
    radius = ((center[0]-outer[0])**2+
                (center[1]-outer[1])**2)**0.5
    return plt.Circle((center[0], center[1]), radius,**kwargs)
xkcd_color=lambda x:mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{x}'])
class OrderManager:
    '''
    `hierarchical_labels` : dict,{L1label:{L2label1:{...:[LastLabel1,LastLabel2]}}}  
    `level_names`         : list for each levels' name  
    `level_colors`        : str, color for each level, check https://xkcd.com/color/rgb/ for viable colors.  
    `layout_prog`         : str, engine for layout calculation, check https://graphviz.org/docs/layouts/  
    `layout_modification` : Callable,  
            if you want to edit the layout properties, add them to the graph property  e.g. :  
            order_manager.order_graph.graph['rankdir']="LR"  
            order_manager.order_graph.nodes['root']['root']=True  

    order properties:  
    - `levels`                   : [[L1label1,L1label2,...],[L2label1,L2label2,...]]  
    - `total_level`              : len(levels)  
    - `level_names`              : level_names  
    - `hierarchical_labels`      : {L1label1:{L2label1:{...:[LastLabel1,LastLabel2]}}}   
    - `numeric_hierarchy`        : {L1Idx1:{L2Idx1:{...:[LastIdx1,LastIdx2]}}}   
    - `peeled_numeric_hierarchy` : [ {L1Idx1:[L2Idx1,L2Idx2,...],L1Idx2:[L2Idx3,L2Idx4,...]},  
                                        {L2Idx1:[L3Idx1,L3Idx2,...],...},{}]  
    - `null_level_dict`          : {"root's Null": 0,"Orthornavirae's Null": 1,"Negarnaviricota's Null's Null": 3,...}  

    visual properties  
    - `max_colors`               : [(r_i,g_i,b_i) for i in levels]  
    - `color_levels`             : [`LinearSegmentedColormap`_i for i in levels]  
    - `color_names`              : level_colors input  
    - `order_graph`              :  `DiGraph`  
    - `graph_pos`                : calculated position for `order_graph`  

    TODO manage null values. 
    '''
    def __init__(self,
        hierarchical_labels:dict,
        level_names:Optional[list]=None, 
        level_colors:List[str]=['pinkish red','purply','ocean','peach'],
        layout_prog:str="twopi",
        layout_modification:Callable[[nx.DiGraph],None]=lambda x:None
                 ):

        self._parse_order(hierarchical_labels,level_names)
        self._parse_visual(level_colors=level_colors,layout_prog=layout_prog,
                layout_modification=layout_modification)
        
        
    def _parse_order(self,hierarchical_labels:dict,level_names):
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
        
        def parse_order(hierarchical_labels:dict):
            levels=recurse_add([],hierarchical_labels)
            total_level = len(levels)
            numeric_hierarchy = recurse_w2i(levels,hierarchical_labels)
            peeled_numeric_hierarchy:List[Dict[str,List[str]]]=peel_dict(numeric_hierarchy)
            return hierarchical_labels,levels,total_level,numeric_hierarchy,peeled_numeric_hierarchy
        
        (self.hierarchical_labels,self.levels,
         self.total_level,self.numeric_hierarchy,
         self.peeled_numeric_hierarchy
        )=parse_order(hierarchical_labels)
        
        if level_names is None:
            self.level_names=[str(i+1) for i in range(len(self.levels))]
        else:
            assert len(level_names)==len(self.levels)
            self.level_names=level_names
            
    def _parse_visual(self,level_colors:List[str],layout_prog:str,
                      layout_modification:Callable[[nx.DiGraph],None]=lambda x:None):
        assert len(level_colors)>=len(self.levels)
        self.layout_prog=layout_prog
        self.layout_modification=layout_modification
        self._gen_visual_props(level_colors=level_colors)
        self._gen_order_graph(layout_prog,layout_modification)
        
    def _gen_visual_props(self,level_colors:List[str]):
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
        if level_colors[0][0]!='#':
            max_colors=[xkcd_color(i) for i in level_colors] #mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{i}']
        else:
            max_colors=[mcolors.to_rgb(i) for i in level_colors]
        self.max_colors=max_colors
        self.color_levels=[mcolors.LinearSegmentedColormap.from_list(
            name,[[1-(1-j)*0.05 for j in max_colors[i]],max_colors[i]]) for i,name in enumerate(self.level_names)]
        self.color_names:List[str]=level_colors
        
    def _gen_order_graph(self,layout_prog:str,layout_modification:Callable[[nx.DiGraph],None]=lambda x:None):
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
        layout_modification(self.order_graph)
        self.graph_pos:Dict[str,Tuple[float,float]] = nx.nx_agraph.graphviz_layout(self.order_graph, prog=layout_prog)
        
    def order_to_idx(self,order_list:List[str])->List[int]:
        o=[]
        for taxo,level in zip(order_list,self.levels):
            o.append(level.index(taxo))
        return o
    
    def idx_to_onehot(self,idx_list:List[Union[int,torch.Tensor]],device:Optional[str]='cpu')->List[torch.Tensor]:
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
    
    def idx_to_order(self,idx_list:List[torch.Tensor],need_argmax:bool=False)->List[str]|List[List[str]]:
        '''
        `idx_list` : List[torch.Tensor]  
        `need_argmax`: <br>
            - False for legacy;  
            - True: process the direct output of `HierarESM`  
        '''
        if not need_argmax:
            warning('`need_argmax` : eprecation use! modify your dataflow!')
        o=[]
        for i,level in enumerate(self.levels):
            if need_argmax:
                idx_l=torch.argmax(idx_list[i],dim=1).tolist()
            else:
                idx_l=idx_list[i].tolist() if isinstance(
                    idx_list[i],torch.Tensor) else idx_list[i]
            o.append([level[j] for j in idx_l])
        if need_argmax:
            o=[list(row) for row in zip(*o)]
        return o
    
    def order_to_onehot(self,order_list:List[str])->List[torch.Tensor]:
        return self.idx_to_onehot(self.order_to_idx(order_list))

    #-- classification graph
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
            node_size=300,null_color=(0,0,0,0),
            to_label:Callable[[str],str]=lambda x:x if 'Null' not in x else 'Null',
            level_boundary:Literal['circle','vertical','none']='none',
            node_shape:str|mpath.Path='o',arrowstyle='->',connectionstyle='arc3'):
        '''
        must work on a given ax
        '''
        # to_label=lambda x:x if 'Null' not in x else 'Null'
        # def to_label(x:str):
        if level_boundary=='none':
            level_boundary={'twopi':'circle','dot':'vertical'}.get(self.layout_prog,'none')

        if level_boundary=='circle':
            for i in range(len(self.levels)):
                # edgecolor=self.color_names[i] if self.color_names[i][0
                #         ]=='#' else mcolors.XKCD_COLORS[f'xkcd:{self.color_names[i]}']
                circle_=circle(self.graph_pos['root'],self.graph_pos[self.levels[i][1]],
                            **dict(fill=False, edgecolor=self.max_colors[i], 
                                linestyle='--', linewidth=2, alpha=0.5))
                ax.add_patch(circle_)
        elif level_boundary=='vertical':
            graph_xs=list(set([i[0] for i in self.graph_pos.values()]))
            graph_xs.sort()
            graph_ys=[i[1] for i in self.graph_pos.values()]
            graph_ymin,graph_ymax=min(graph_ys),max(graph_ys)
            vext=0.05*(graph_ymax-graph_ymin)
            # for i,x,color in zip(range(self.total_level),graph_xs,self.max_colors):
            ax.vlines(graph_xs[1:],graph_ymin-vext,graph_ymax+vext,colors=self.max_colors,
                        linestyle='--', linewidth=1, alpha=0.5)
        elif level_boundary=='none':
            pass
        else:
            raise ValueError(f'level_boundary: {level_boundary}')

        dimgrey=xkcd_color('dark grey')

        G,pos=self.order_graph,self.graph_pos
        nx.draw_networkx_nodes(G,pos,ax=ax,node_size=node_size,node_shape=node_shape,
            node_color=[color_dict.get(n,null_color) for n in G],edgecolors=dimgrey)
        nx.draw_networkx_edges(G,pos,ax=ax,arrowstyle=arrowstyle,edge_color=dimgrey,connectionstyle=connectionstyle)
        nx.draw_networkx_labels(G,pos,labels={n:to_label(n) for n in G},ax=ax,font_size=11,font_color=dimgrey,font_weight='heavy')
        #{a numeric value in range 0-1000, 
        # 'ultralight', 'light', 'normal', 
        # 'regular', 'book', 'medium', 'roman', 
        # 'semibold', 'demibold', 'demi', 'bold', 
        # 'heavy', 'extra bold', 'black'}

        hide_spline(ax)
        return ax
    
    #-- bar plots
    def cal_true_probs(self,pred:List[torch.Tensor],true:List[torch.Tensor],batch_i:int=0):
        '''
        true: onehot tensor
        '''
        probabs=[]
        for level_id,level in enumerate(self.levels):
            dist=F.softmax(pred[level_id][batch_i].float().detach().to('cpu'),dim=-1)
            true_label=true[level_id][batch_i].argmax()
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
        # if self.color_names[0][0]!='#':
        #     colors=[mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{i}']) for i in self.color_names]
        # else:
        #     colors=[mcolors.to_rgb(i) for i in self.color_names]        
        # colors=[mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{i}']) for i in self.color_names]
        bars = ax.bar(x_ticks, probabs, color=self.max_colors)
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
def cal_accuracy(predictions:torch.Tensor, labels:torch.Tensor):
    '''Calculates the accuracy of the prediction.
       prediction: the layer output
       label: the [bz,1] tensor gt id instead of onehot 
    '''
    raise RuntimeError('`cal_accuracy` managed by model.py now')
    num_data = labels.size()[0]
    predicted = torch.argmax(predictions, dim=1)

    correct_pred = torch.sum(predicted == labels)

    accuracy:torch.Tensor = correct_pred*(100/num_data)

    return accuracy.item()

# %%
def map_to_pdb(mapping_gradients,idx,pdbfile):
    
    res_gradients=mapping_gradients[idx[0]:idx[1]]
    l2_res_gradients=torch.sqrt(torch.sum(res_gradients** 2, dim=-1))
    cmd.load(pdbfile, 'entry0')
    all_objects = cmd.get_object_list('all')
    residues = cmd.get_model(all_objects[0])
    for l,(b,e) in zip(l2_res_gradients.tolist(),residues.get_residues()):
        cmd.alter(f'index {b}-{e}',f"b={l}")
    cmd.save('o.pdb')
    cmd.delete('entry0')

#%%
taxo_palette = {
    'red_(crayola)': {0: '#f10c45', 100: '#31020e', 200: '#61051c', 300: '#92072a', 400: '#c30938', 500: '#f10c45', 600: '#f63c6b', 700: '#f86d90', 800: '#fa9eb5', 900: '#fdceda'}, 
    'magenta_dye': {0: '#c5267c', 100: '#270819', 200: '#4f0f31', 300: '#76174a', 400: '#9e1e62', 500: '#c5267c', 600: '#dc4696', 700: '#e574b0', 800: '#eda2ca', 900: '#f6d1e5'}, 
    'purpureus': {0: '#983fb2', 100: '#1e0c23', 200: '#3c1947', 300: '#5a256a', 400: '#78328e', 500: '#983fb2', 600: '#af5fc7', 700: '#c387d5', 800: '#d7afe3', 900: '#ebd7f1'}, 
    'yinmn_blue': {0: '#4d5da2', 100: '#0f1321', 200: '#1f2541', 300: '#2e3862', 400: '#3d4a82', 500: '#4d5da2', 600: '#6c7aba', 700: '#919bcb', 800: '#b5bddc', 900: '#dadeee'}, 
    'bice_blue': {0: '#276c9a', 100: '#08161f', 200: '#102b3e', 300: '#17415d', 400: '#1f577c', 500: '#276c9a', 600: '#3590cc', 700: '#67acd9', 800: '#9ac7e6', 900: '#cce3f2'}, 
    'cerulean': {0: '#017b92', 100: '#00191d', 200: '#00313b', 300: '#004a58', 400: '#016276', 500: '#017b92', 600: '#01b7db', 700: '#27dafe', 800: '#6fe6fe', 900: '#b7f3ff'}, 
    'cambridge_blue': {0: '#90a296', 100: '#1c211e', 200: '#38433b', 300: '#546459', 400: '#6f8577', 500: '#90a296', 600: '#a6b5ab', 700: '#bcc7c0', 800: '#d3dad5', 900: '#e9ecea'}, 
    'tan': {0: '#c8a989', 100: '#2e2216', 200: '#5c432b', 300: '#896541', 400: '#b2875b', 500: '#c8a989', 600: '#d3baa1', 700: '#decbb8', 800: '#e9dcd0', 900: '#f4eee7'}, 
    'sandy_brown': {0: '#ffb07c', 100: '#4b1e00', 200: '#973c00', 300: '#e25b00', 400: '#ff822f', 500: '#ffb07c', 600: '#ffbf95', 700: '#ffcfaf', 800: '#ffdfca', 900: '#ffefe4'}, 
    'battleship_gray': {0: '#8b8b8b', 100: '#1c1c1c', 200: '#383838', 300: '#545454', 400: '#707070', 500: '#8b8b8b', 600: '#a3a3a3', 700: '#bababa', 800: '#d1d1d1', 900: '#e8e8e8'}
}

named_taxo_palette={'Null': '#cccccc',
 'Orthornavirae': '#ffff43',
 'Pararnavirae': '#c6afe9',
 'Duplornaviricota': '#ff5555',
 'Kitrinoviricota': '#ffdd55',
 'Lenarviricota': '#ccff00',
 'Negarnaviricota': '#87deaa',
 'Pisuviricota': '#5c9dff',
 'Artverviricota': '#c6afe9',
 'Chrymotiviricetes': '#ff80b9',
 'Resentoviricetes': '#f9acd4',
 'Vidaverviricetes': '#ff8680',
 'Alsuviricetes': '#ff9955',
 'Flasuviricetes': '#ffff00',
 'Magsaviricetes': '#ffcc00',
 'Tolucaviricetes': '#decd87',
 'Amabiliviricetes': '#ecff00',
 'Howeltoviricetes': '#e0ff66',
 'Leviviricetes': '#dff4a4',
 'Miaviricetes': '#d3ff7a',
 'Chunqiuviricetes': '#b0e481',
 'Milneviricetes': '#91e77e',
 'Monjiviricetes': '#7fe68e',
 'Yunchangviricetes': '#6cef91',
 'Ellioviricetes': '#88f8bd',
 'Insthoviricetes': '#88f8bd',
 'Duplopiviricetes': '#19f5d0',
 'Pisoniviricetes': '#00ccff',
 'Stelpaviricetes': '#8080ff',
 'Revtraviricetes': '#c6afe9',
 'Ghabrivirales': '#ff80b9',
 'Reovirales': '#f9acd4',
 'Mindivirales': '#ff8080',
 'Hepelivirales': '#e79f8b',
 'Martellivirales': '#febaad',
 'Tymovirales': '#ffb380',
 'Amarillovirales': '#ffff00',
 'Nodamuvirales': '#ffcc00',
 'Tolivirales': '#decd87',
 'Wolframvirales': '#ecff00',
 'Cryppavirales': '#e0ff66',
 'Norzivirales': '#e9f683',
 'Timlovirales': '#d0ef7a',
 'Ourlivirales': '#d3ff7a',
 'Muvirales': '#b0e481',
 'Serpentovirales': '#91e77e',
 'Jingchuvirales': '#6cdc77',
 'Mononegavirales': '#2ec455',
 'Goujianvirales': '#6cef91',
 'Bunyavirales': '#88f8bd',
 'Articulavirales': '#79f7ce',
 'Durnavirales': '#19f5d0',
 'Nidovirales': '#00cfdd',
 'Picornavirales': '#98eaff',
 'Sobelivirales': '#72c6ff',
 'Patatavirales': '#84a8ff',
 'Stellavirales': '#999ce0',
 'Yadokarivirales': '#867fdf',
 'Blubervirales': '#ccaaff',
 'Ortervirales': '#eeaaff'}

def show_palette(palette:Dict[str,Dict[str,str]]):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax:Axes
    for i, (color_name, shades) in enumerate(palette.items()):
        for j, (shade, hex_color) in enumerate(shades.items()):
            ax.add_patch(plt.Rectangle((j, -i), 1, 1, color=hex_color))
            ax.text(j + 0.5, -i + 0.5, f"{shade}", color='white', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(-len(palette)+1, 1)
    ax.set_yticks([-i + 0.5 for i in range(len(palette))])
    ax.set_yticklabels(list(palette.keys()), fontsize=12)
    ax.set_xticks(np.arange(0.5,10.5,1))
    # ax.set_xticklabels(["DEFAULT", "100", "200", "300", "400", "500", "600", "700", "800", "900"], fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    return fig,ax

#%%  random
def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)
        
class LocalRandomGenerator:
    def __init__(self,seed:int):
        self.seed=seed
        self.random_generator=random.Random(seed)
        self.numpy_generator=np.random.default_rng(seed)
        self.torch_generator=torch.Generator('cpu')
        self.torch_generator.manual_seed(seed)