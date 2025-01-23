import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import Graph, StateGraph

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

def create_base_prompt_template(sample_id, sample_data):
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
    
    task_prompt = """"""
    task_prompt += json.dumps(prompt_template, indent=2)
    

    return task_prompt

def create_reasoning_prompt(sample_id, sample_data):
    task_prompt = """Task: As the primary reviewer, analyze whether the given statement is logically entailed by the clinical trial report (CTR) section. Please:
1. Carefully examine the evidence from the trial content
2. Provide a step-by-step reasoning process
3. Draw a conclusion (Entailment/Contradiction)
4. Explain your rationale
Input:
"""
    task_prompt += json.dumps(create_base_prompt_template(sample_id, sample_data), indent=2)
    task_prompt += "\n\nProvide your analysis in the following format:\nReasoning Process:\n[Your step-by-step analysis]\n\nConclusion:\n[Entailment/Contradiction]\n\nRationale:\n[Your explanation]"
    return task_prompt

def create_verification_prompt(sample_id, sample_data, primary_analysis):
    task_prompt = """Task: As the secondary reviewer, verify the reasoning and conclusion provided by the primary reviewer. Please:
5. Review the original evidence and statement
6. Analyze the primary reviewer's reasoning process
7. Identify any logical flaws or inconsistencies or 
8. Confirm or challenge the conclusion
9. Provide your final judgment
Original Case:
"""
    task_prompt += json.dumps(create_base_prompt_template(sample_id, sample_data), indent=2)
    task_prompt += "\n\nPrimary Reviewer's Analysis:\n"
    task_prompt += primary_analysis
    task_prompt += "\n\nProvide your verification in the following format:\nVerification Analysis:\n[Your analysis of the primary review]\n\nIdentified Issues (if any):\n[List any logical flaws or inconsistencies]\n\nJustification:\n[Your explanation]\n\nFinal Judgment:\n[MUST output ONLY 'Entailment' or 'Contradiction']"
    return task_prompt

def final_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    verification_content = state["final_verification"]
    
    try:
        import re
        # 先找到 Final Judgment 部分
        pattern = r"Final Judgment:.*?(Contradiction|Entailment)"
        match = re.search(pattern, verification_content, re.DOTALL)
        
        if match:
            # 提取到的 Contradiction 或 Entailment
            result = match.group(1)
            state["final_prediction"] = result
        else:
            state["final_prediction"] = "NAN"
    except:
        state["final_prediction"] = "NAN"
    
    return state

def get_model_prediction(prompt, is_verification=False):
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        prediction = response.choices[0].message.content.strip()
        return prediction
    except Exception as e:
        print(f"API error: {e}")
        return "Error: " + str(e)

def primary_reviewer(state: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = state["sample_id"]
    sample_data = state["sample_data"]
    prompt = create_reasoning_prompt(sample_id, sample_data)
    analysis = get_model_prediction(prompt)
    state["primary_analysis"] = analysis
    return state

def secondary_reviewer(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = create_verification_prompt(
        state["sample_id"], 
        state["sample_data"], 
        state["primary_analysis"]
    )
    verification = get_model_prediction(prompt, is_verification=True)
    state["final_verification"] = verification
    return state

# 定义状态类型
class WorkflowState(TypedDict):
    sample_id: str
    sample_data: dict
    primary_analysis: Optional[str]
    final_verification: Optional[str]
    final_prediction: Optional[str]

def create_workflow() -> Graph:
    # 创建工作流
    workflow = StateGraph(WorkflowState)
    
    # 添加节点
    workflow.add_node("primary_review", primary_reviewer)
    workflow.add_node("secondary_review", secondary_reviewer)
    workflow.add_node("final_extraction", final_extractor)
    
    # 设置工作流程
    workflow.set_entry_point("primary_review")
    workflow.add_edge("primary_review", "secondary_review")
    workflow.add_edge("secondary_review", "final_extraction")
    workflow.set_finish_point("final_extraction")
    
    return workflow.compile()

def main():
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Processing Start Time: {start_datetime}")

    # Read test file
    test_file = r"D:\Master_Thesis\Task-2-SemEval-2024-main\test.json"
    test_data = read_json_file(test_file)
    total_samples = len(test_data)
    
    # Create workflow
    workflow = create_workflow()
    results = {}
    
    # Process all samples
    for idx, (sample_id, sample_data) in enumerate(test_data.items(), 1):
        sample_start_time = time.time()
        print(f"\nProcessing sample {idx}/{total_samples}: {sample_id}")
        
        # Initialize state
        state = {
            "sample_id": sample_id,
            "sample_data": sample_data,
            "primary_analysis": None,
            "final_verification": None,
            "final_prediction": None
        }
        
        # Run workflow
        final_state = workflow.invoke(state)
        results[sample_id] = {
            "Prediction": final_state["final_prediction"],
            "Primary_Analysis": final_state["primary_analysis"],
            "Verification": final_state["final_verification"]
        }
        
        # Calculate and display processing time for each sample
        sample_time = time.time() - sample_start_time
        print(f"Prediction Result: {final_state['final_prediction']}")
        print(f"Sample Processing Time: {sample_time:.2f} seconds")
        
        # Save results in real-time
        with open('predictions_test_all.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        # Display progress
        if idx % 10 == 0:
            print(f"\nCompleted: {idx}/{total_samples} samples")
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_sample = elapsed_time / idx
            estimated_remaining_time = avg_time_per_sample * (total_samples - idx)
            print(f"Estimated Time Remaining: {estimated_remaining_time/60:.2f} minutes")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nProcessing Complete!")
    print(f"Start Time: {start_datetime}")
    print(f"End Time: {end_datetime}")
    print(f"Total Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Processing Time per Sample: {total_time/total_samples:.2f} seconds")
    print(f"Results saved to predictions_test_all.json")

if __name__ == "__main__":
    main()




