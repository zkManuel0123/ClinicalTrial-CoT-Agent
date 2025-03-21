import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_section_content(trial_id, section_id):
    # 构建临床试验文件路径
    trial_path = Path(r"\CT json") / f"{trial_id}.json"
    if not trial_path.exists():
        return None
    
    trial_data = read_json_file(trial_path)
    return trial_data.get(section_id, "")

def create_prompt(sample_id, sample_data):
    # 获取Primary试验的内容
    primary_content = get_section_content(sample_data["Primary_id"], sample_data["Section_id"])
    
    if sample_data["Type"] == "Comparison":
        # 获取Secondary试验的内容
        secondary_content = get_section_content(sample_data["Secondary_id"], sample_data["Section_id"])
        prompt_template = {
            sample_id: {
                "Type": "Comparison",
                "Trial_Content": {
                    "Primary_Trial": primary_content,
                    "Secondary_Trial": secondary_content
                },
                "Section_id": sample_data["Section_id"],
                "Primary_id": sample_data["Primary_id"],
                "Secondary_id": sample_data["Secondary_id"],
                "Statement": sample_data["Statement"]
            }
        }
    else:  # Single类型
        prompt_template = {
            sample_id: {
                "Type": "Single",
                "Trial_Content": {
                    "Primary_Trial": primary_content
                },
                "Section_id": sample_data["Section_id"],
                "Primary_id": sample_data["Primary_id"],
                "Statement": sample_data["Statement"]
            }
        }
    
    # task_prompt = "Task: Determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR).\n"
    # task_prompt += json.dumps(prompt_template, indent=2)
    # task_prompt += "\nIf the statement is true based on the section, return 'Entailment'.\nIf the statement is false, return 'Contradiction'.\nDo not return any text and content other than this. Only return 'Entailment' or 'Contradiction'."
    

    task_prompt = "Task: Determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR). Let's approach this step by step:\n\n"
    task_prompt += json.dumps(prompt_template, indent=2)
    task_prompt += "\n\nLet's think about this step by step:\n"
    task_prompt += "1. First, let's identify the key claim made in the statement.\n"
    task_prompt += "2. Next, let's examine the relevant information provided in the CTR section.\n"
    task_prompt += "3. Let's compare the statement with the CTR information:\n"
    task_prompt += "   - What specific evidence supports or contradicts the statement?\n"
    task_prompt += "   - Are there any important details or conditions mentioned in the CTR that affect our conclusion?\n"
    task_prompt += "4. Based on this analysis, we can conclude:\n"
    task_prompt += "\nFinal Answer: [Your reasoning should lead to either 'Entailment' or 'Contradiction']\nDo not include any other text."

    return task_prompt

def get_model_prediction(prompt):
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",  # 使用适当的模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        prediction = response.choices[0].message.content.strip()
        # 确保结果是 Entailment 或 Contradiction
        if prediction not in ["Entailment", "Contradiction"]:
            return "NAN"  # 默认返回
        return prediction
    except Exception as e:
        print(f"API error: {e}")
        return "Contradiction"  # 出错时默认返回

def main():
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始处理时间: {start_datetime}")

    # 读取测试文件
    test_file = r"\test.json"
    test_data = read_json_file(test_file)
    
    # 存储结果
    results = {}
    
    # 处理所有样本
    total_samples = len(test_data)
    for i, (sample_id, sample_data) in enumerate(test_data.items()):
     
        sample_start_time = time.time()
        
        print(f"Processing sample {i+1}/{total_samples}: {sample_id}")
        prompt = create_prompt(sample_id, sample_data)
        prediction = get_model_prediction(prompt)
        results[sample_id] = {"Prediction": prediction}
        
        # 计算并显示每个样本的处理时间
        sample_time = time.time() - sample_start_time
        print(f"Sample processing time: {sample_time:.2f} seconds")
        
        # 实时保存结果
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    
    # 计算总运行时间
    total_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nProcessing completed!")
    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Total running time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average processing time per sample: {total_time/total_samples:.2f} seconds")
    print("Results saved to predictions.json")


if __name__ == "__main__":
    main()




