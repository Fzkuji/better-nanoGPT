# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters

train_batch_size = 12
val_batch_size = 12
train_size = 512
val_size = 1024
eval_interval = 50
eval_iters = 100 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
