python tmp_lightning_predict.py \
--dataset taxo_data/proseq_taxo_1.pkl \
--ckpt_path train/lightning_exp8_seed7/checkpoint-step=149925-val_loss=5.02.ckpt \
--exp_dir infer/exp7 \
--output_stem seed-7 \
--mode test \
--test_mode test 


python tmp_lightning_predict.py \
--dataset taxo_data/proseq_taxo_1.pkl \
--ckpt_path train/lightning_exp8_seed7/checkpoint-step=149925-val_loss=5.02.ckpt \
--default_root_dir infer/ \
--exp_dir exp7 \
--output_stem seed-7 \
--mode predict 