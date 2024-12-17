import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.model_max_length = 9999999
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(data_point):
    """
    根据数据点生成完整的输入，并将其标记化为模型的输入格式。
    """
    system_prompt = data_point.get("system_prompt", "").strip()
    question = data_point["question"].strip()
    response = data_point["response"].strip()

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

    len_user_prompt_tokens = len(tokenizer(user_prompt)["input_ids"])
    full_tokens = tokenizer(user_prompt + response)["input_ids"]

    # 构造labels
    labels = [-1] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]
    eos_token_id = tokenizer.eos_token_id
    labels.append(eos_token_id)

    return {
        "input_ids": full_tokens,
        "labels": labels,
        "attention_mask": [1] * len(full_tokens),
    }

# 加载数据集
ds = load_dataset("Open-Orca/OpenOrca", split="train")
df = ds.to_pandas()

# 删除response中包含大量重复 "NO_DIFF" 或 "No emotion" 的样本
def should_remove_response(resp):
    no_diff_count = resp.count("NO_DIFF")
    no_emotion_count = resp.count("No emotion")
    return no_diff_count > 1 or no_emotion_count > 1

df = df[~df["response"].apply(should_remove_response)]

# 添加长度列用于排序
df["response_length"] = df["response"].apply(lambda x: len(x.strip().split()))
df["question_length"] = df["question"].apply(lambda x: len(x.strip().split()))

# 按照要求排序：首先按response长度降序，然后按question长度降序
df = df.sort_values(by=["response_length", "question_length"], ascending=[False, False])

# 丢弃一些数据集(例如只保留排序后前一半的数据，可以根据需要修改)
# 假设数据集很大，这里仅示例保留前 50000 条数据，可根据实际需求调整
df = df.head(1000000)

# 转回Dataset
ds = Dataset.from_pandas(df, preserve_index=False)

# 划分数据集为train和val
split_ds = ds.train_test_split(test_size=0.1, seed=42)

num_proc = os.cpu_count() - 4 if os.cpu_count() > 4 else 1

processed_train_ds = split_ds["train"].map(
    generate_and_tokenize_prompt,
    batched=False,
    num_proc=num_proc
)

processed_val_ds = split_ds["test"].map(
    generate_and_tokenize_prompt,
    batched=False,
    num_proc=num_proc
)

processed_train_ds.to_json("train.json")
processed_val_ds.to_json("val.json")

print("数据已经处理并保存为 JSON 文件！")
