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
    trial_path = Path(r"D:\Master_Thesis\Task-2-SemEval-2024-main\training_data\CT json") / f"{trial_id}.json"
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
    
    # Baseline Prompt
    
    # task_prompt = "Task: Determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR).\n"
    # task_prompt += json.dumps(prompt_template, indent=2)
    # task_prompt += "\nIf the statement is true based on the section, return 'Entailment'.\nIf the statement is false, return 'Contradiction'.\nDo not return any text and content other than this. Only return 'Entailment' or 'Contradiction'."
    

    task_prompt = """Task: Using the ReAct framework, determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR).
Follow these ReAct steps:  
Thought 1: Let me first understand the key points in the statement and identify the relevant information in the trial content.  
Action 1: Extract and list key claims from the statement. 
Observation 1: [List extracted key claims]  
Thought 2: Now let me analyze each key claim against the trial content.  
Action 2: Compare each claim with the corresponding information in the trial content. 
Observation 2: [Document evidence for/against each claim]  
Thought 3: Based on the evidence, let me determine if there's a complete logical entailment or contradiction.  Action 3: Make final judgment. 
Observation 3: Based on systematic analysis: 
- If ALL key claims are supported by the trial content: Return 'Entailment' 
- If ANY key claim contradicts or cannot be supported by the trial content: Return 'Contradiction'  
 
"""
    task_prompt += json.dumps(prompt_template, indent=2)
    task_prompt += "\nFinal Output: Based on the above analysis, provide ONLY ONE WORD as your final answer: either 'Entailment' or 'Contradiction'. Do not include any other text."

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
    test_file = r"D:\Master_Thesis\Task-2-SemEval-2024-main\test.json"
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



# # 只处理前10个样本
#     for i, (sample_id, sample_data) in enumerate(test_data.items()):
#         if i >= 20:  # 达到3个样本后停止
#             break
            
#         print(f"处理样本 {i+1}/10: {sample_id}")
        
#         # 创建prompt
#         prompt = create_prompt(sample_id, sample_data)
#         print("打印prompt结果：",prompt)
#         # 获取预测结果
#         prediction = get_model_prediction(prompt)
#         print("打印prediction结果：",prediction)
#         # 保存结果
#         results[sample_id] = {"Prediction": prediction}
#         print("打印results结果：",results)
#         # 实时保存结果到文件
#         with open('predictions_test.json', 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=4)
        
#     print("测试完成！结果已保存到 predictions_test.json")

# if __name__ == "__main__":
#     main()