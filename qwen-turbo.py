import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    
    task_prompt = "Task: Determine whether the following statement is logically entailed by the specified section of the clinical trial report (CTR).\n"
    task_prompt += json.dumps(prompt_template, indent=2)
    task_prompt += "\nIf the statement is true based on the section, return 'Entailment'.\nIf the statement is false, return 'Contradiction'.\nDo not return any text and content other than this. Only return 'Entailment' or 'Contradiction'."
    

    return task_prompt

def get_model_prediction(prompt):
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        prediction = response.choices[0].message.content.strip()
        
        if prediction not in ["Entailment", "Contradiction"]:
            return "NAN"  
        return prediction
    except Exception as e:
        print(f"API error: {e}")
        return "Contradiction"  

def main():
   
    test_file = r"\test.json"
    test_data = read_json_file(test_file)
    
    
    results = {}
    
    
    total_samples = len(test_data)
    for i, (sample_id, sample_data) in enumerate(test_data.items()):
        print(f"Processing sample {i+1}/{total_samples}: {sample_id}")
        
        
        prompt = create_prompt(sample_id, sample_data)
        
        
        prediction = get_model_prediction(prompt)
        
        
        results[sample_id] = {"Prediction": prediction}
        
        
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
    print("Prediction completed! Results saved to predictions.json")


if __name__ == "__main__":
    main()

