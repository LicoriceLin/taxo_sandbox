from hierataxo import OrderManager
import pickle as pkl
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager

from hierataxo.util import named_taxo_palette,xkcd_color
import matplotlib.colors as mcolors
import re
import networkx as nx
import numpy as np
import matplotlib.path as mpath
import numpy as np
from typing import Dict,Tuple


# fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

def layout_modification(G:nx.DiGraph):
    G.graph['rankdir']='LR'
    G.nodes['root']['root']=True

def manual_modify_graph_pos(graph_pos:Dict[str,Tuple[float,float]]):
    graph_xs=list(set([i[0] for i in graph_pos.values()]))
    graph_xs.sort()
    ideal_deltax=graph_xs[1]-graph_xs[0]
    ideal_graph_map={i:j for i,j in zip(
    graph_xs,
    np.linspace(graph_xs[0],graph_xs[0]+4*ideal_deltax,5).tolist()
        )} 
    graph_ys=[i[1] for i in graph_pos.values()]
    graph_ymin,graph_ymax=min(graph_ys),max(graph_ys)
    for k,v in graph_pos.items():
        graph_pos[k]=(ideal_graph_map.get(v[0],v[0]),graph_ymin+graph_ymax-v[1])

def _hex_path():
    s5=5**0.5
    vertices = [
        (-s5, -1/2),   
        (-s5, 1/2),    
        (s5, 1/2),     
        (s5, -1/2),    
        (-s5, -1/2),   
    ]
    codes = [
        mpath.Path.MOVETO,  
        mpath.Path.LINETO,  
        mpath.Path.LINETO, 
        mpath.Path.LINETO,  
        mpath.Path.CLOSEPOLY, 
    ]
    hex_path = mpath.Path(vertices, codes)
    return hex_path
hex_path=_hex_path()
_pattern = r'virae|viricota|viricetes|virales'

def to_label(x:str):
    if 'Null' in x:
        return 'Null'
    elif re.search(_pattern, x):
        return re.sub(_pattern, '-', x)
    elif x=='root':
        return 'Riboviria'
    else:
        return x


'''
a TMP standard `OrderManager` for CM meeting ana
'''
order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
    level_names=['Kingdom','Phylum','Class','Order'],
    layout_prog='dot',layout_modification=layout_modification)
manual_modify_graph_pos(order_manager.graph_pos)

def standard_classification_view(
        color_dict:Dict[str,tuple],
        ax:Axes,
        order_manager:OrderManager=order_manager,
        set_rcParams:bool=False):
    '''
    TMP function for CM meeting ana
    please ensure the ax takes up a 6*12 space
    '''
    if set_rcParams:
        c_rcParams=plt.rcParamsDefault
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({
            # "text.usetex": True,
            # "text.latex.preamble": r"\usepackage{amsmath}",
            'svg.fonttype':'none',
            'font.sans-serif':['Arial','Helvetica',
                'DejaVu Sans',
                'Bitstream Vera Sans',
                'Computer Modern Sans Serif',
                'Lucida Grande',
                'Verdana',
                'Geneva',
                'Lucid',
                'Avant Garde',
                'sans-serif'],
            "pdf.use14corefonts":False,
            'pdf.fonttype':42,
            'text.color':xkcd_color('dark grey'),
            'axes.labelweight':'heavy',
            'axes.titleweight':'extra bold'
                })

    order_manager.draw_classification_view(color_dict,ax,null_color=(0.8,0.8,0.8),
        to_label=to_label,node_size=4200,level_boundary='none',
        arrowstyle='-',node_shape=hex_path,
        connectionstyle="angle,angleA=-90,angleB=180,rad=0")
    
    if set_rcParams:
        plt.rcParams.update(c_rcParams)

    return ax

if __name__=='__main__':
    # plt.rcParams.update(plt.rcParamsDefault)
    # plt.rcParams.update({
    #     # "text.usetex": True,
    #     # "text.latex.preamble": r"\usepackage{amsmath}",
    #     'svg.fonttype':'none',
    #     'font.sans-serif':['Arial','Helvetica',
    #         'DejaVu Sans',
    #         'Bitstream Vera Sans',
    #         'Computer Modern Sans Serif',
    #         'Lucida Grande',
    #         'Verdana',
    #         'Geneva',
    #         'Lucid',
    #         'Avant Garde',
    #         'sans-serif'],
    #     "pdf.use14corefonts":False,
    #     'pdf.fonttype':42,
    #     'text.color':xkcd_color('dark grey'),
    #     'axes.labelweight':'heavy',
    #     'axes.titleweight':'extra bold'
    #         })
    

    
    plt.close()
    fig,ax=plt.subplots(1,1,figsize=(6,12))

    # order_manager.draw_classification_view(named_taxo_palette,ax,null_color=(0.8,0.8,0.8),
    #     to_label=to_label,node_size=4200,level_boundary='none',
    #     arrowstyle='-',node_shape=hex_path,
    #     connectionstyle="angle,angleA=-90,angleB=180,rad=0")
    standard_classification_view(named_taxo_palette,ax,set_rcParams=True)

    plt.tight_layout()
    # plt.show()
    fig.savefig('colormap.svg')