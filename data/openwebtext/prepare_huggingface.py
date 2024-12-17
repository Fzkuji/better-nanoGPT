import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
from transformers import GPT2TokenizerFast

# number of workers in .map() call
num_proc = 8

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

# 使用Hugging Face的GPT2分词器
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

if __name__ == '__main__':
    # 加载 openwebtext 数据集
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)

    # 将数据集拆分为train和val
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # test改为val

    # 定义处理函数，使用HuggingFace的GPT2分词器
    def process(example):
        # 不添加特殊token
        ids = tokenizer.encode(example['text'], add_special_tokens=False)
        # 添加eos
        ids.append(tokenizer.eos_token_id)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # 对数据集进行分词
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 将分词后的数据集转换为二进制文件用于训练
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # 因为gpt2词典大小 < 2**16，可用uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # 分批写入数据
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # 写入mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin大小约17GB, val.bin约8.5MB
    # train约9B tokens, val约4M tokens

    # 读取bin文件示例:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
