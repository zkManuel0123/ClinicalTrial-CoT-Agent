import os
import json
import anthropic
from dotenv import load_dotenv
from pathlib import Path
import time

# 加载环境变量
load_dotenv()

# 初始化 Anthropic 客户端
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
    
    task_prompt = "Task: Determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR).\n"
    task_prompt += json.dumps(prompt_template, indent=2)
    task_prompt += "\nIf the statement is true based on the section, return 'Entailment'.\nIf the statement is false, return 'Contradiction'.\nDo not return any text and content other than this. Only return 'Entailment' or 'Contradiction'."
    

#     task_prompt = """Task: Using the ReAct framework, determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR).
# Follow these ReAct steps:  
# Thought 1: Let me first understand the key points in the statement and identify the relevant information in the trial content.  
# Action 1: Extract and list key claims from the statement. 
# Observation 1: [List extracted key claims]  
# Thought 2: Now let me analyze each key claim against the trial content.  
# Action 2: Compare each claim with the corresponding information in the trial content. 
# Observation 2: [Document evidence for/against each claim]  
# Thought 3: Based on the evidence, let me determine if there's a complete logical entailment or contradiction.  Action 3: Make final judgment. 
# Observation 3: Based on systematic analysis: 
# - If ALL key claims are supported by the trial content: Return 'Entailment' 
# - If ANY key claim contradicts or cannot be supported by the trial content: Return 'Contradiction'  
# Final Answer: [Entailment/Contradiction]  
# """
    # task_prompt += json.dumps(prompt_template, indent=2)
    # task_prompt += "Remember: Only return 'Entailment' or 'Contradiction' as your final answer to output for the user."

    return task_prompt

def get_model_prediction(prompt):
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0
        )
        prediction = response.content[0].text.strip()
        # 确保结果是 Entailment 或 Contradiction
        if prediction not in ["Entailment", "Contradiction"]:
            return "NAN"  
        return prediction
    except Exception as e:
        print(f"API error: {e}")
        return "NAN"  

def main():
    # 读取测试文件
    test_file = r"\test.json"
    test_data = read_json_file(test_file)
    
    # 存储结果
    results = {}
    
    # 添加时间检测
    start_time = time.time()
    
    # 获取总样本数
    total_samples = len(test_data)
    
    for i, (sample_id, sample_data) in enumerate(test_data.items(), 1):
        print(f"Processing sample {i}/{total_samples}: {sample_id}")
        
        # 记录每个样本的处理开始时间
        sample_start_time = time.time()
        
        # 创建prompt
        prompt = create_prompt(sample_id, sample_data)
        
        # 获取预测结果
        prediction = get_model_prediction(prompt)
        
        # 计算样本处理时间
        sample_processing_time = time.time() - sample_start_time
        
        # 保存结果
        results[sample_id] = {
            "Prediction": prediction,
            "Processing_Time": f"{sample_processing_time:.2f} seconds"
        }
        
        # 实时保存结果到文件
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        print(f"Sample {sample_id} processed in {sample_processing_time:.2f} seconds")
        print(f"Prediction: {prediction}\n")
        
    # 计算总处理时间
    total_time = time.time() - start_time
    print(f"\nTest completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/total_samples:.2f} seconds")
    print("Results saved to predictions.json")


if __name__ == "__main__":
    main()



