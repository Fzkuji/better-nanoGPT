import json
import torch
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"


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


def get_batch(data_tensor, batch_size, sequence_length, pad_token_id=0):
    """
    从预加载的数据中提取批量数据。
    - data_tensor: 已加载和预处理的完整数据 tensor。
    - batch_size: 批量大小。
    - sequence_length: 每个输入序列的长度。
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
    y = y[:, :sequence_length]

    # 将数据移到设备上
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# 示例使用
json_path = "./val.json"  # 替换为你的 JSON 文件路径
sequence_length = 128
pad_token_id = -1

# 加载和预处理数据
data_tensor = load_data_from_json_lines(json_path, sequence_length, pad_token_id)

# 获取一个训练批次
batch_size = 8
X, Y = get_batch(data_tensor, batch_size, sequence_length, pad_token_id)

print("Total samples:", len(data_tensor))  # 数据集中样本总数
print("X shape:", X.shape)  # 预期: [batch_size, sequence_length]
print("Y shape:", Y.shape)  # 预期: [batch_size, sequence_length]
