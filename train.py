import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from trl import SFTTrainer, SFTConfig
import json
from datasets import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())


print("step 1/5: 创建模型")
config = GPT2Config(
    vocab_size=50257, 
    n_positions=1024,
    n_embd=768,      
    n_layer=12,
    n_head=12
)
model = AutoModelForCausalLM.from_config(config)


print("step 2/5: 加载数据集")
data_path = "./data/simple_9.jsonl"

data_list = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))
dataset = Dataset.from_list(data_list)

print("数据集加载成功，样本数量：", len(dataset))
print("第一个样本：", dataset[0])


print("step 3/5: 初始化分词器")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("step 4/5: 设置训练参数")
model_dir = "./models"
log_dir = "./logs"


training_args = SFTConfig(
        completion_only_loss=True,
        max_length=32,
        output_dir=model_dir, 
        per_device_train_batch_size=8,
        num_train_epochs=1000, 
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=10,
        save_steps=100000,
        report_to="none",
        fp16=True if device.type == "cuda" else False,
    )


print("step 5/5: 配置SFT训练器")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args
)

print("开始训练...")
trainer.train()

print("训练完成！")