# Inspecting Clinical Trial Inference through Chain-of-Thought and Dual-Agent reasoning with Large Language Models


## Comments and Guidance:
(23.01.2025 updates)
1. **Dual-Agent + CoT**: Please see the code file: run_6_DualAgent_Cot_qwenturbo.py
2. The API Key of Model Qwen-turbo has already been writen into the file ".env"
3. **Evaluation**: Please see the original official evaluation file in: Task-2-SemEval-2024-main/evaluation.py, but notes that the file name you want to evaluate needs to be changed in the code file. And Start evaluation.py by using the command line: `python evaluate.py \Task-2-SemEval-2024-main D:\Master_Thesis\output`
4. **Evaluation Files**: The prediction file this time which produced by "run_6_DualAgent_Cot_qwenturbo.py" is placed firstly in the root directory, see: predictions_test_all.json. You can see the detail information of the analysis and conclusions and prediction lables of the two agents in this file. Afterwards, I extracted all the prediction labels only to fit the evaluation file, and must place it in: Task-2-SemEval-2024-main\res\predictions_DualAgent_CoT_qwenturbo.json
5. **Final Results**: All the txt files of results are generated into the folder: "output"


