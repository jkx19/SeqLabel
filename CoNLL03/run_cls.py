import os

lr = 1e-4
batch_size = 16
psl = 5
mid_dim = 512
bert_type = 'base'

lrs = [1e-4, 5e-5, 1e-5]
psls = [5,8]
mid_dims = [512, 1024]
for lr in lrs:
    for psl in psls:
        for mid_dim in mid_dims:
            COMMANDLINE = f"python cli.py \
                --lr={lr} \
                --batch_size={batch_size} \
                --pre_seq_len={psl} \
                --mid_dim={mid_dim} \
                --bert_type={bert_type}"

            os.system(COMMANDLINE)
        