import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime
import re

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端（使用阿里云接口）
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
    # task_prompt += "\nFinal Answer: [Your reasoning should lead to either 'Entailment' or 'Contradiction']\nDo not include any other text."
    task_prompt += "\nFinal Answer: [IMPORTANT: In the Final Answer, your response MUST end with 'Final Answer: ' followed by ONLY 'Entailment' or 'Contradiction'. No other format is acceptable.]"
    return task_prompt

def get_model_prediction(prompt):
    try:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",  
            messages=messages,
            temperature=0
        )
        prediction = response.choices[0].message.content.strip()
        
        # 打印原始输出
        print("\n=== Qwen-2.5 原始输出 ===")
        print(prediction)
        print("=====================\n")
        
        # 使用正则表达式提取 Final Answer 后的结果
        pattern = r"Final Answer:.*?(Contradiction|Entailment)"
        match = re.search(pattern, prediction, re.DOTALL)
        
        if match:
            result = match.group(1)
            return result
        return "NAN"
        
    except Exception as e:
        print(f"API error: {e}")
        
        if "Arrearage" in str(e):
            print("error")
            
            import sys
            sys.exit(1)
        return "NAN"

def main():
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始处理时间: {start_datetime}")

    # 读取测试文件
    test_file = r"\test.json"
    test_data = read_json_file(test_file)
    
    # 将字典转换为列表以便按索引访问
    samples_list = list(test_data.items())
    
    # 从第4540个样本开始处理
    start_index = 4539  # 因为索引从0开始，所以是4540-1
    
    # 创建新的结果文件名（包含起始样本编号）
    output_file = f'predictions_CoT_qwen2.5_from_{start_index + 1}.json'
    
    # 存储结果
    results = {}
    
    # 处理所有样本
    for i in range(start_index, len(samples_list)):
        sample_id, sample_data = samples_list[i]
        sample_start_time = time.time()
        
        print(f"Processing sample {i + 1}: {sample_id}")
        prompt = create_prompt(sample_id, sample_data)
        prediction = get_model_prediction(prompt)
        results[sample_id] = {"Prediction": prediction}
        
        # 每处理一个样本就保存一次结果，防止中断丢失数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {output_file}")
        
        # 计算并显示样本的处理时间
        sample_time = time.time() - sample_start_time
        print(f"Sample processing time: {sample_time:.2f} seconds")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nProcessing completed!")
    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Total running time: {total_time:.2f} seconds")
    print(f"Processed samples from {start_index + 1} to {len(samples_list)}")


if __name__ == "__main__":
    main()




