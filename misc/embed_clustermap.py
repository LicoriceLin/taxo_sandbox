# %%
'''
archive only
not for real run
'''
# %%
import pickle as pkl
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
from taxo_train_1 import OrderManager
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
if __name__=='__main__':
    order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                            level_names=['Kingdom','Phylum','Class','Order'])
    names,embeds,labels=pkl.load(open('../output/embeds.pkl','rb'))
    level_to_check=1
    level_label=labels[:,level_to_check]
    unque_labels=pd.Series(level_label).unique()
    readable_labels={i:order_manager.levels[level_to_check][i] for i in unque_labels}

    # %% random gen your color scheme
    colors=list(mcolors.CSS4_COLORS.keys())
    sampled_colors=random.sample(colors,len(pd.Series(level_label).value_counts()))
    dendro_color=[sampled_colors[i] for i in level_label]
    name_color_dict={order_manager.levels[level_to_check][i]:sampled_colors[i] for i in unque_labels}

    cmap = mcolors.ListedColormap(list(name_color_dict.values()))
    norm = mcolors.BoundaryNorm(np.arange(len(name_color_dict) + 1) - 0.5, len(name_color_dict))
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    cb = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(cb, cax=ax, orientation='horizontal')
    cbar.set_ticks(np.arange(len(name_color_dict)))
    cbar.set_ticklabels(list(name_color_dict.keys()))
    plt.show()

    # %%
    gs = gridspec.GridSpec(1, 4, width_ratios=[0.2,0.02,0.6,0.06], wspace=0.)
    cluster_grid = sns.clustermap(embeds,col_cluster=False, 
                row_colors=dendro_color,
                xticklabels=False,
                yticklabels=False,
                cmap='RdBu')

    fig=cluster_grid.figure
    fig.set_size_inches(24,18)
    cluster_grid.ax_row_dendrogram.set_position(gs[0].get_position(fig))
    cluster_grid.ax_heatmap.set_position(gs[2].get_position(fig))
    cluster_grid.ax_row_colors.set_position(gs[1].get_position(fig))
    cluster_grid.cax.set_visible(False)
    ax2 = fig.add_subplot(gs[3])
    cbar = fig.colorbar(cb, cax=ax2, orientation='vertical',location='right')
    ax2.set_aspect(200)
    ax2.tick_params(rotation=-45)
    cbar.set_ticks(np.arange(len(name_color_dict)))
    cbar.set_ticklabels(list(name_color_dict.keys()))
    fig.set_dpi(400)
    fig.savefig('cluster_.png')
    plt.close(fig)