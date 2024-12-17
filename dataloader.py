import json
import os

import numpy as np
import torch

pretrain = ['openwebtext', 'pg19', 'shakespeare', 'shakespeare_char']

def get_batch(
        dataset: str=None,
        split: str=None,
        data_tensor=None,
        batch_size: int = 1,
        length: int = 256,
        pad_token_id: int = -1,
        device_type: str = 'cuda',
        device=None):
    if dataset in pretrain:
        return get_batch_pretrain(
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            length=length,
            device_type=device_type,
            device=device
        )
    elif data_tensor is not None:
        return get_batch_finetune(
            data_tensor=data_tensor,
            batch_size=batch_size,
            length=length,
            pad_token_id=pad_token_id,
            device_type=device_type,
            device=device
        )
    else:
        raise ValueError('Invalid dataset or data_tensor not provided')


def get_batch_pretrain(dataset, split, batch_size, length, device_type='cuda', device=None):
    # poor man's data loader
    data_dir = os.path.join('data', dataset)

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + length]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def load_data_from_json_lines(json_path, sequence_length, pad_token_id=0):
    """
    从多行 JSON 文件中加载数据并预处理。
    - json_path: JSON 文件路径。
    - sequence_length: 每个输入序列的长度。
    - pad_token_id: 用于填充的 token id，默认是 0。
    """
    sequences = []

    # 按行读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                data = json.loads(line)
                sequences.append(data['input_ids'])  # 假设每行 JSON 都有 input_ids

    # 过滤掉长度大于 sequence_length 的序列
    sequences = [seq for seq in sequences if len(seq) <= sequence_length]

    # 对每个序列进行 padding 到 sequence_length
    padded_sequences = [
        seq + [pad_token_id] * (sequence_length - len(seq)) for seq in sequences
    ]

    # 转换为 tensor 并返回
    data_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    return data_tensor


def get_batch_finetune(data_tensor, batch_size, length, pad_token_id=-1, device_type='cuda', device=None):
    """
    从预加载的数据中提取批量数据。
    - data_tensor: 已加载和预处理的完整数据 tensor。
    - batch_size: 批量大小。
    - length: 每个输入序列的长度。
    - pad_token_id: 用于填充的 token id，默认是 0。
    """
    # 随机选择批量数据索引
    indices = np.random.choice(len(data_tensor), batch_size, replace=False)
    selected_sequences = data_tensor[indices]

    # 构建输入和标签
    x = selected_sequences
    y = torch.tensor(
        [seq[1:].tolist() + [pad_token_id] for seq in selected_sequences], dtype=torch.long
    )

    # 确保 Y 的长度与 X 一致
    y = y[:, :length]

    # 将数据移到设备上
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


