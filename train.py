import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# --- 1. 创建一个随机权重的语言模型 ---
# 这里我们使用 GPT-2 架构，因为它是一个自回归模型，非常适合这个任务。
# GPT2Config 会自动用随机权重初始化模型。conda activate sft_env
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("MPS 可用:", torch.backends.mps.is_available())
print("step 1/5: 创建模型")
config = GPT2Config(
    vocab_size=50257, 
    n_positions=1024,
    n_embd=128,      
    n_layer=4,
    n_head=4
)
model = AutoModelForCausalLM.from_config(config)

# --- 2. 加载数据集 ---
print("step 2/5: 加载数据集")
data_dir = "/Users/guoxinyue/repo/MathLM/data/"
data_name = "simple_9"
data_path = data_dir + f"{data_name}.jsonl"
import json
from datasets import Dataset

# --- 2. 加载数据集 ---
data_list = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

# 将 Python 字典列表转换为 Hugging Face Dataset
dataset = Dataset.from_list(data_list)

print("数据集加载成功，样本数量：", len(dataset))
print("第一个样本：", dataset[0])

# --- 3. 初始化分词器 ---
print("step 3/5: 初始化分词器")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 4. 设置训练参数 ---
print("step 4/5: 设置训练参数")
model_dir = "/Users/guoxinyue/repo/MathLM/models/"
log_dir = "/Users/guoxinyue/repo/MathLM/logs/"

training_args = SFTConfig(
          # 只计算完成部分的损失
    assistant_only_loss=True,
    max_seq_length=32,                 # 输入文本的最大长度
    output_dir=model_dir,             # 模型和检查点保存的目录
    per_device_train_batch_size=4,         # 每个设备（M1 Pro）的训练批次大小
    num_train_epochs=1000,                   # 训练的总轮数
    learning_rate=2e-5,                    # 学习率
    weight_decay=0.01,                     # 权重衰减
    logging_dir=log_dir,                  # 日志保存的目录
    logging_steps=10,                      # 每隔多少步打印一次日志
    save_steps=10000,                        # 每隔多少步保存一次检查点
    evaluation_strategy="no",              # 不进行评估，因为我们只有一个训练集
    report_to="none",                      # 不向任何在线平台汇报训练进度
    # 启用 MPS 加速（Apple Silicon）
    # 当 PyTorch 支持 MPS 时，这行会自动使用GPU
    dataloader_pin_memory=False,           # 在M1 Pro上，设置为False更稳定
)

# --- 5. 配置SFT训练器 ---
print("step 5/5: 配置SFT训练器")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    
    # training_args 用于配置训练过程
    # 你可以根据需要添加更多参数，比如 learning_rate, num_train_epochs 等。
    args=training_args
)

# --- 6. 开始训练 ---
print("开始训练...")
trainer.train()

print("训练完成！")