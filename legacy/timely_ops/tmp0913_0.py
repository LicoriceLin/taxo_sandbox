# %%
import pickle as pkl
from hierataxo.util import (
    OrderManager)
order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                        level_names=['Kingdom','Phylum','Class','Order'])
# %%
