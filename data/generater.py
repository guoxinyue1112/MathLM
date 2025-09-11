import json

dir_path = "/Users/guoxinyue/repo/MathLM/data/"
RANGE = 9
file_name = f"simple_{RANGE}.jsonl"

def generate_full_math_dataset(filename=dir_path + file_name):
    """
    生成一个包含所有 range 以内加减法算式的完整数据集。

    参数:
    filename (str): 数据集保存的文件路径。
    """
    dataset = []

    # 生成所有加法算式
    for a in range(RANGE + 1):
        for b in range(RANGE + 1):
            result = a + b
            prompt = f"{a}+{b}="
            completion = f"{result}"
            dataset.append({"prompt": prompt, "completion": completion})
            # full_text = f"{a} + {b} = {result}"
            # dataset.append({"text": full_text})
            # # 确保结果在十以内
            # if result <= 10:
            #     full_text = f"{a} + {b} = {result}"
            #     dataset.append({"text": full_text})

    # # 生成所有减法算式
    # for a in range(RANGE + 1):
    #     for b in range(RANGE + 1):
    #         # 确保结果非负
    #         if a >= b:
    #             result = a - b
    #             full_text = f"{a} - {b} = {result}"
    #             dataset.append({"text": full_text})
                
    # 将数据集写入JSON Lines文件
    with open(filename, 'w', encoding='utf-8') as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"成功生成 {len(dataset)} 个样本，并保存到 {filename}。")

if __name__ == "__main__":
    generate_full_math_dataset()