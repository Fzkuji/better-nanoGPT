# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'better-nanoGPT'
wandb_run_name='999 W128-T1024-V16384-L24'

# these make the total batch size be ~0.5M
# 12 batch_size * 1024 context_length * 5 gradaccum * 4 GPUs = 245,760
data = {
    "train": {
        "datasets": [
            {
                "dataset": "openwebtext",   # 'openwebtext' or 'shakespeare' or 'shakespeare_char' or 'pg19'
                "batch_size": 12,           # must fit in GPU memory
                "context_length": 1024      # size of the input to the model
            }
        ]
    },
    "val": {
        "datasets": [
            {
                "dataset": "openwebtext",
                "batch_size": 1,       # must fit in GPU memory
                "context_length": 16384  # size of the input to the model
            },
            {
                "dataset": "pg19",
                "batch_size": 1,        # must fit in GPU memory
                "context_length": 16384  # size of the input to the model
            }
        ]
    }
}
gradient_accumulation_steps = 5 * 4  # accumulate gradients over N * batch_size samples

# model
block_size = 128
n_layer = 24
n_head = 12
n_embd = 768
position_embedding = 'alibi' # 'rope' or 'none' or 'alibi'
max_position_embeddings = 32768

# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10


# weight decay
weight_decay = 1e-1
