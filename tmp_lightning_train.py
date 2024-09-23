# %%
import pickle as pkl
from hierataxo.util import (
    OrderManager)
from hierataxo import (
    OrderManager,ConcatProteinDataset,
    HierarchicalLossNetwork,
    cal_accuracy,set_seed)
from hierataxo.dataset import ConcatProteinDataModule
from hierataxo.model import HierarESM
# from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, DistributedSampler,random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor
order_manager=OrderManager(pkl.load(open('taxo_data/hierarchy_order.pkl','rb'))['Riboviria'],
                        level_names=['Kingdom','Phylum','Class','Order'])
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import torch
# L.Trainer()
max_length=500
max_domain=15
train_bs=3
infer_bs=20
seed=42
seed_everything(seed, workers=True)
model=HierarESM(order_manager,max_length=max_length,max_domain=max_domain,
    optimizer_kwargs={'backbone_lr':5e-5,'head_lr':1e-4,'weight_decay':0.01},
    scheduler_kwargs={'warmup_iter_1':1,'warmup_iter_2':49,'warmup_lr':1e-7,'exp_gamma':0.95})
datamodule=ConcatProteinDataModule(order_manager,'taxo_data/proseq_taxo_1.pkl',
    max_length=max_length,max_domain=max_domain,train_bs=train_bs,infer_bs=infer_bs)
datamodule.setup('fit')
# %%
default_root_dir='train'
exp_dir='lightning_exp7'

model.load_state_dict(torch.load('train/lightning_exp4/checkpoint-step=053973-val_loss=28.14.ckpt',map_location='cpu')['state_dict'])
trainer = L.Trainer(
    default_root_dir=default_root_dir, 
    accelerator='gpu',
    strategy='auto',
    max_epochs=100,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    precision='16-mixed',
    # limit_train_batches=100,
    # limit_val_batches=30,
    callbacks=[ModelCheckpoint(dirpath=default_root_dir+'/'+exp_dir,monitor='val_loss',
                    filename='checkpoint-{step:06d}-{val_loss:.2f}',save_top_k=2),
               LearningRateMonitor('epoch')],
    logger=TensorBoardLogger(save_dir=default_root_dir,name=exp_dir)
    )

trainer.fit(model,datamodule=datamodule,
    # ckpt_path='train/lightning_exp6/checkpoint-step=031984-val_loss=30.26.ckpt'
    )
# trainer.test(model,datamodule=datamodule,
#     # ckpt_path='infer/reload_ckpt/seed-77-last.ckpt'
#     )
#%%
