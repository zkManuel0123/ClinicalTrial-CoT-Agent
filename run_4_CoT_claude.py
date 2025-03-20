import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
import time
from datetime import datetime
import re


load_dotenv()


client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_section_content(trial_id, section_id):
    
    trial_path = Path(r"\CT json") / f"{trial_id}.json"
    if not trial_path.exists():
        return None
    
    trial_data = read_json_file(trial_path)
    return trial_data.get(section_id, "")

def create_prompt(sample_id, sample_data):
    
    primary_content = get_section_content(sample_data["Primary_id"], sample_data["Section_id"])
    
    if sample_data["Type"] == "Comparison":
        
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
    else:  
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
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,  
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        prediction = response.content[0].text.strip()
        
       
        print("\n=== Claude 3.5 Sonnet  ===")
        print(prediction)
        print("=====================\n")
        
       
        pattern = r"Final Answer:.*?(Contradiction|Entailment)"
        match = re.search(pattern, prediction, re.DOTALL)
        
        if match:
            result = match.group(1)
            return result
        return "NAN"
        
    except Exception as e:
        print(f"API error: {e}")
        return "Contradiction"

def main():
    
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"time start: {start_datetime}")

   
    test_file = r"\test.json"
    test_data = read_json_file(test_file)
    
    
    results = {}
    
   
    total_samples = len(test_data)
    for i, (sample_id, sample_data) in enumerate(test_data.items()):  
        sample_start_time = time.time()
        
        print(f"Processing sample {i+1}/{total_samples}: {sample_id}")
        prompt = create_prompt(sample_id, sample_data)
        prediction = get_model_prediction(prompt)
        results[sample_id] = {"Prediction": prediction}
        
        
        sample_time = time.time() - sample_start_time
        print(f"Sample processing time: {sample_time:.2f} seconds")
        
        
        with open('predictions_CoT_claude.json', 'w', encoding='utf-8') as f:  
            json.dump(results, f, indent=4)
    
    
    total_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\nProcessing completed!")
    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")
    print(f"Total running time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average processing time per sample: {total_time/total_samples:.2f} seconds")
    print("Results saved to predictions_CoT_claude.json")  


if __name__ == "__main__":
    main()




