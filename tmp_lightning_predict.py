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
        # print(f'saving: {trainer.default_root_dir+'/'+self.outname}')
        pd.DataFrame(o).to_pickle(trainer.default_root_dir+'/'+self.outname) #


if __name__=='__main__':
    # max_length=500 #1000 #500  #
    # max_domain=15 #1 #28 #
    # infer_bs=30 #60 #60 #
    # dataset='taxo_data/proseq_taxo_single_domain.pkl'#, 'taxo_data/proseq_taxo_1.pkl', #'taxo_data/proseq_taxo_15more.pkl', #
    # ckpt_path='train/lightning_exp7_seed77/checkpoint-step=197901-val_loss=7.78.ckpt'
    # default_root_dir='infer'
    # exp_dir='exp7'
    # output_stem='seed-77' #f'seed-{seed}{suffix}'
    from argparse import ArgumentParser
    parser=ArgumentParser(
        prog='taxo-infer',
        description='place holder description',
        epilog='place holder epilog')
    parser.add_argument('--dataset')
    parser.add_argument('--ckpt_path')
    parser.add_argument('--exp_dir',default='predict_output')
    parser.add_argument('--output_stem',default='poc')
    parser.add_argument('--default_root_dir',default='infer')
    parser.add_argument('--mode',choices=['predict','test'],default='test')
    parser.add_argument('--limit_batches',type=int,default=-1)
    parser.add_argument('--seed',type=int,default=42,
        help='used for test, make sure the test set is the same as fitting stage.')
    parser.add_argument('--test_mode',choices=['all','test','train','valid'],default='all')
    args=parser.parse_args()
    dataset,ckpt_path,exp_dir,output_stem,default_root_dir=(
        args.dataset,args.ckpt_path,args.exp_dir,
        args.output_stem,args.default_root_dir
        )
    seed_everything(args.seed)
    if '_single_domain.' in dataset:
        max_length,max_domain,infer_bs=1000,1,60
        output_suffix='-singledomain'
    elif '_15more.' in dataset:
         max_length,max_domain,infer_bs=500,28,6
         output_suffix='-15more'
    elif '_1.' in dataset:
         max_length,max_domain,infer_bs=500,15,30
         output_suffix='-normal'
    else:
        raise ValueError
    output_name=output_stem+output_suffix
    # seed=42
    # seed_everything(seed, workers=True)
    # order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
    #                         level_names=['Kingdom','Phylum','Class','Order'])
    model=HierarESM(max_length=max_length,max_domain=max_domain,
        optimizer_kwargs={'backbone_lr':1e-4,'head_lr':1e-3,'weight_decay':0.01},
        scheduler_kwargs={'warmup_iter_1':20,'warmup_iter_2':30,'warmup_lr':1e-10})

    datamodule=ConcatProteinDataModule(dataset,
        max_length=max_length,max_domain=max_domain,infer_bs=infer_bs,test_mode=args.test_mode)
    limit_batches=None if args.limit_batches==-1 else args.limit_batches
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
    # model.load_state_dict(torch.load('train/lightning_exp7/checkpoint-step=165917-val_loss=8.15.ckpt',map_location='cpu'),strict=False)
    trainer = L.Trainer(
        default_root_dir=default_root_dir, 
        accelerator='gpu',
        strategy='auto',
        max_epochs=100,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        precision='16-mixed',
        logger=False,
        # max_steps=0,
        limit_test_batches=limit_batches,
        limit_predict_batches=limit_batches,
        callbacks=PredictWriter(f'{exp_dir}/{output_name}.pkl'),
        )
    # trainer.predict(model,datamodule,ckpt_path='train/lightning_exp7/checkpoint-step=165917-val_loss=8.15.ckpt')
    if args.mode=='test':
        trainer.test(model,datamodule,ckpt_path=ckpt_path)
    elif args.mode=='predict':
        trainer.predict(model,datamodule,ckpt_path=ckpt_path)
    # trainer.save_checkpoint(f'{default_root_dir}/{exp_dir}/seed-{seed}-ep28.ckpt')
    # break