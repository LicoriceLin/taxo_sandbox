import pickle as pkl
from typing import List,Dict
from itertools import chain
import pandas as pd
import os
from glob import glob
from pathlib import Path

from hierataxo import  OrderManager
from hierataxo.dataset import ConcatProteinDataModule
from hierataxo.model import HierarESM
# from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, DistributedSampler,random_split
import lightning as L

import torch
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor,BasePredictionWriter
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger



class PredictWriter(BasePredictionWriter):
    def __init__(self, outname:str):
        super().__init__(write_interval='epoch')
        self.outname = outname

    def write_on_epoch_end(self, trainer, pl_module, 
        predictions:List[Dict[str,List[str]|torch.Tensor]], batch_indices):
        o={}
        dtypes={k:type(v) for k,v in predictions[0].items()}
        for k,dt in dtypes.items():
            if dt is list:
                o[k]=sum([i[k] for i in predictions],start=[])
            elif dt is torch.Tensor:
                o[k]=torch.concat([i[k] for i in predictions]).tolist()
        pd.DataFrame(o).to_pickle(trainer.default_root_dir+'/'+self.outname)


if __name__=='__main__':
    max_length=1000  #500
    max_domain= 1 #28 #15
    train_bs=2
    infer_bs=60 #30
    seed=42
    seed_everything(seed, workers=True)
    order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                            level_names=['Kingdom','Phylum','Class','Order'])
    model=HierarESM(order_manager,max_length=max_length,max_domain=max_domain,
        optimizer_kwargs={'backbone_lr':1e-4,'head_lr':1e-3,'weight_decay':0.01},
        scheduler_kwargs={'warmup_iter_1':20,'warmup_iter_2':30,'warmup_lr':1e-10})

    datamodule=ConcatProteinDataModule(order_manager,'taxo_data/proseq_taxo_single_domain.pkl',#'taxo_data/proseq_taxo_15more.pkl', #'taxo_data/proseq_taxo_1.pkl'
        max_length=max_length,max_domain=max_domain,train_bs=train_bs,infer_bs=infer_bs)

    default_root_dir='infer'
    exp_dir='reload_ckpt'
    suffix='-singledomain' #''
    # for i in Path('tmp_poster_ana/used_models/').iterdir():
    #     seed=i.stem.split('-')[1]
    #     model.load_state_dict(torch.load(i,map_location='cpu'),strict=False)
    #     trainer = L.Trainer(
    #         default_root_dir=default_root_dir, 
    #         accelerator='gpu',
    #         strategy='auto',
    #         max_epochs=100,
    #         check_val_every_n_epoch=1,
    #         log_every_n_steps=1,
    #         precision='32',
    #         max_steps=0,
    #         callbacks=PredictWriter(f'{exp_dir}/seed-{seed}{suffix}.pkl'),
    #         # model.on_save_checkpoint
    #         # limit_predict_batches=30,
    #         # limit_train_batches=100,
    #         # limit_val_batches=30,
    #         # callbacks=[ModelCheckpoint(dirpath=default_root_dir+'/'+exp_dir,monitor='val_loss',
    #         #                 filename='checkpoint-{step:06d}-{val_loss:.2f}',save_top_k=2),
    #         #            LearningRateMonitor('epoch',log_weight_decay=True)],
    #         # logger=TensorBoardLogger(save_dir=default_root_dir,name=exp_dir)
    #         )
    model.load_state_dict(torch.load('tmp_poster_ana/used_models/seed-77-ep-28.pt',map_location='cpu'),strict=False)
    trainer = L.Trainer(
        default_root_dir=default_root_dir, 
        accelerator='gpu',
        strategy='auto',
        max_epochs=100,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        precision='32',
        max_steps=0,
        limit_test_batches=2
        # callbacks=PredictWriter(f'{exp_dir}/seed-{seed}{suffix}.pkl'),
        )
    trainer.test(model,datamodule)
    trainer.save_checkpoint(f'{default_root_dir}/{exp_dir}/seed-{seed}-ep28.ckpt')
    # break