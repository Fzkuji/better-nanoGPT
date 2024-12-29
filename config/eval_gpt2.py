# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters

data = {
    "train": {
        "datasets": [
            {
                "dataset": "openwebtext",   # 'openwebtext' or 'shakespeare' or 'shakespeare_char' or 'pg19'
                "batch_size": 1,           # must fit in GPU memory
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

# eval stuff
eval_interval = 1000
eval_iters = 20 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
