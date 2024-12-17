import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from datasets import load_dataset
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.model_max_length = 9999999
tokenizer.pad_token = tokenizer.eos_token

# 处理函数
def generate_and_tokenize_prompt(data_point):
    """
    根据数据点生成完整的输入，并将其标记化为模型的输入格式。
    """
    # 构造用户提示
    system_prompt = data_point.get("system_prompt", "").strip()
    question = data_point["question"].strip()
    response = data_point["response"].strip()

    # 拼接提示
    if system_prompt:
        user_prompt = f"""Below is a conversation between a user and an AI assistant. The system provides context, and the AI responds to the user's question.

### System Prompt:
{system_prompt}

### User:
{question}

### AI:
"""
    else:
        user_prompt = f"""Below is a conversation between a user and an AI assistant. The AI responds to the user's question.

### User:
{question}

### AI:
"""

    # 计算用户提示的 token 数量
    len_user_prompt_tokens = len(tokenizer(user_prompt)["input_ids"])

    # 整体 token 化
    full_tokens = tokenizer(user_prompt + response)["input_ids"]

    # 构造 labels
    labels = [-1] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]
    eos_token_id = tokenizer.eos_token_id
    labels.append(eos_token_id)

    return {
        "input_ids": full_tokens,
        "labels": labels,  # 输出部分正常计算损失
        "attention_mask": [1] * len(full_tokens),  # 每个 token 都有效
    }


# 加载数据集
ds = load_dataset("Open-Orca/OpenOrca", split="train")

# 划分数据集为 train 和 val
split_ds = ds.train_test_split(test_size=0.1, seed=42)

# 设置并行处理的核数
num_proc = os.cpu_count() - 4  # 使用所有可用核数，保留 1 核给系统

# 处理 train 数据集（多核加速）
processed_train_ds = split_ds["train"].map(
    generate_and_tokenize_prompt,
    batched=False,
    num_proc=num_proc
)

# 处理 val 数据集（多核加速）
processed_val_ds = split_ds["test"].map(
    generate_and_tokenize_prompt,
    batched=False,
    num_proc=num_proc
)

# 保存处理后的数据集为 JSON 格式
processed_train_ds.to_json("train.json")
processed_val_ds.to_json("val.json")

print("数据已经处理并保存为 JSON 文件！")
