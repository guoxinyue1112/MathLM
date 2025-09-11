import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

def test_model(model_path="/Users/guoxinyue/repo/MathLM/models/checkpoint-3300"):
    """
    加载训练好的模型并进行测试。
    
    参数:
    model_path (str): 训练好的模型保存路径。
    """
    # 1. 加载分词器和模型
    print("正在加载分词器和模型...")
    # 这里我们加载训练时保存的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 将模型设置为评估模式
    model.eval()

    # 检查是否支持 MPS 加速并将其移动到相应的设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("已检测到 MPS，模型将被移动到 GPU 上进行推理。")
    else:
        device = torch.device("cpu")
        print("未检测到 MPS，模型将在 CPU 上进行推理。")
    
    model.to(device)
    
    # 2. 生成测试数据集
    # 为了测试方便，我们直接创建一个简单的测试数据集
    test_data = [
        {"prompt": "2+3=", "completion": "5"},
        {"prompt": "7+1=", "completion": "8"},
        {"prompt": "4+6=", "completion": "10"},
        {"prompt": "9+0=", "completion": "9"},
        {"prompt": "5+5=", "completion": "10"},
    ]
    test_dataset = Dataset.from_list(test_data)
    
    print("--- 开始测试 ---")
    with torch.no_grad():
        for sample in test_dataset:
            input_text = sample["prompt"]
            
            # 将输入文本转换为模型可以理解的ID
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            # 3. 使用模型进行文本生成
            # 这里我们使用模型的 generate() 方法来生成文本
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 5,  # 确保生成足够长的序列
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.8,                   # 控制生成结果的随机性
                do_sample=True,                    # 启用采样，与 temperature 配合使用
            )
            
            # 将生成的ID转换回文本
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 4. 打印结果
            print(f"输入: {input_text}\t | \t输出: {output_text}")

    print("--- 测试完成 ---")

if __name__ == "__main__":
    model_dir = "/Users/guoxinyue/repo/MathLM/models/checkpoint-3300"
    # 请确保这里的路径与你的训练脚本中 output_dir 相同
    test_model(model_path=model_dir)