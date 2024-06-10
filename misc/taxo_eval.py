import numpy as np
import pickle as pkl
import torch
from hierataxo import (
    OrderManager,ConcatProteinDataset,
    HierarESM,HierarchicalLossNetwork,
    cal_accuracy,set_seed,taxo_palette)
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys


model= sys.argv[1]#'/train/v1_240521-210553_seed10043/ep-28.pt'
out_file=sys.argv[2]#'output/seed10043-ep28.pt'
device=int(sys.argv[3]) #3
batch_size=20 # maximum valid batch size on 2080Ti: 100; train: 2 (or maybe 3?)


max_domain=15
max_length=500
to_freeze=3

default_colors=lambda x:taxo_palette[x][500]
order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                            level_names=['Kingdom','Phylum','Class','Order'],color_levels=[default_colors(i) for i in ['red_(crayola)','purpureus','magenta_dye','sandy_brown']])
hierar_esmmodel=HierarESM(order_manager=order_manager,max_length=max_length,to_freeze=to_freeze,device=device)
hierar_esmmodel.load_state_dict(torch.load(model))
hierar_esmmodel.eval()
dataset=ConcatProteinDataset('taxo_data/proseq_taxo_1.pkl',order_manager)

val_generator = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
hierar_esmmodel.eval()
names,x_s,y_s,ebs=[],[],[],[]

def val_batch(sample):
        x_s.append([i.detach().to('cpu') for i in 
                    hierar_esmmodel(*hierar_esmmodel.process_sample(sample))])
        y_s.append(sample['taxo'])
        
def emb_batch(sample):
        _names,_ebs,_labels=hierar_esmmodel._embed(sample)
        ebs.append(torch.concat(_ebs,dim=1).detach().to('cpu'))
        names.extend(_names)
        
if __name__=='__main__':
    with torch.set_grad_enabled(False):
        # i=0
        for step, sample in tqdm(enumerate(val_generator)):
            val_batch(sample)
            emb_batch(sample) #TODO use hook to accelerate
            # i+=1
            # if i>5:
            #     break
            
    x_s=[torch.concat([j[i] for j in x_s],dim=0) for i in range(order_manager.total_level)]
    y_s=[torch.concat([j[i] for j in y_s],dim=0) for i in range(order_manager.total_level)]
    # final_accuracies={k:cal_accuracy(i,j) for k,i,j in zip(order_manager.level_names,x_s,y_s)}
    ebs=torch.concat(ebs,dim=0)    
    
    torch.save({'id':names,'pd':x_s,'gt':y_s,'eb':ebs},f=out_file)
