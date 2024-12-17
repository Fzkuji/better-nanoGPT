import json
import torch
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch_from_json_lines(json_path, batch_size, sequence_length, pad_token_id=0):
    """
    从多行 JSON 文件中加载数据并生成批量数据。
    - json_path: JSON 文件路径。
    - split: 数据集划分（'train' 或 'val'）。
    - batch_size: 批量大小。
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

    # 随机选择批量数据
    indices = np.random.choice(len(padded_sequences), batch_size, replace=False)
    selected_sequences = [padded_sequences[i] for i in indices]

    # 构建输入和标签
    x = torch.tensor(selected_sequences, dtype=torch.long)
    y = torch.tensor(
        [seq[1:] + [pad_token_id] for seq in selected_sequences], dtype=torch.long
    )

    # 确保 Y 的长度与 X 一致
    y = y[:, :sequence_length]

    # 将数据移到设备上
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# 示例使用
json_path = "train.json"  # 替换为你的 JSON 文件路径
batch_size = 1
sequence_length = 256
pad_token_id = -1
# 获取一个训练批次
X, Y = get_batch_from_json_lines(json_path, batch_size, sequence_length, pad_token_id)

print("X shape:", X)  # 预期: [batch_size, sequence_length]
print("Y shape:", Y)  # 预期: [batch_size, sequence_length]
