# Master Thesis Project
This repository contains code and resources for the master's thesis project on AI-assisted evaluation of clinical trial analysis.

## Comments and Guidance:
(23.01.2025 updates)
1. **Dual-Agent + CoT**: Please see the code file: run_6_DualAgent_Cot_qwenturbo.py
2. The API Key of Model Qwen-turbo has already been writen into the file ".env"
3. **Evaluation**: Please see the original official evaluation file in: Task-2-SemEval-2024-main/evaluation.py, but notes that the file name you want to evaluate needs to be changed in the code file. And Start evaluation.py by using the command line: `python evaluate.py \Task-2-SemEval-2024-main D:\Master_Thesis\output`
4. **Evaluation Files**: The prediction file this time which produced by "run_6_DualAgent_Cot_qwenturbo.py" is placed firstly in the root directory, see: predictions_test_all.json. You can see the detail information of the analysis and conclusions and prediction lables of the two agents in this file. Afterwards, I extracted all the prediction labels only to fit the evaluation file, and must place it in: Task-2-SemEval-2024-main\res\predictions_DualAgent_CoT_qwenturbo.json
5. **Final Results**: All the txt files of results are generated into the folder: "output"

**Other Notes:**
Why didn't I use a third agent to extract the lables but used REGEX instead?

At first, I attempted to use three agents, with the third agent responsible for summarizing the conclusions of the second agent and returning only "Entailment" or "Contradiction." However, I found that the runtime was too long—it took 30 hours. Using three agents (30 hours) doubled the time compared to using two agents (18 hours).

After reconsidering, I realized that having a third agent solely to extract the conclusions from the second agent was not very necessary. Therefore, I switched to using regular expressions instead. After debugging and testing with 30 samples, I found that the regular expressions could accurately extract the keywords—either "entailment" or "contradiction"—from the second agent's conclusions. This approach was not only faster but also reduced the time and cost associated with using an additional agent just for information extraction.

Furthermore, I do want to test whether adding more agents for evaluation (three agents, four agents, five agents...but test time will multiply, I guess. I considered using groq and sambanova instead.) will lead to an improvement or decrease in the performance of the results. This is what I want do next.
