#!/usr/bin/env python3

import json
import os
import os.path
import sys
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.simplefilter('ignore')

def extract_control_set(predictions, gold):
    control_predicitons = {}
    for key in gold.keys():
        if "Causal_type" not in gold[key].keys():
            control_predicitons[key] = predictions[key]
    return control_predicitons


def extract_by_intervention(predictions, gold):
    para_predictions = {}
    cont_predictions = {}
    numerical_para_predictions = {}
    numerical_cont_predictions = {}
    definitions_predictions = {}
    for key in predictions.keys():
        if "Intervention" not in gold[key].keys():
            continue
        if gold[key]["Intervention"] == "Paraphrase":
            para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Contradiction":
            cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_paraphrase":
            numerical_para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_contradiction":
            numerical_cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Text_appended":
            definitions_predictions[key] = predictions[key]
    print("Number of contradiction samples:", len(cont_predictions))
    return para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions


def extract_by_causal_type(predictions, gold):
    predictions_preserving = {}
    predictions_altering = {}
    for key in predictions.keys():
        if "Causal_type" not in gold[key].keys():
            continue
        if gold[key]["Causal_type"][0] == "Preserving":
            predictions_preserving[key] = predictions[key]
        elif gold[key]["Causal_type"][0] == "Altering":
            predictions_altering[key] = predictions[key]
    return predictions_preserving, predictions_altering


def faithfulness(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] != gold[gold[key]["Causal_type"][1]]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Faithfulness = sum(results) / N
    return Faithfulness


def consistency(predictions_preserving, predictions, gold):
    uuid_list = list(predictions_preserving.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions_preserving[key]["Prediction"] == predictions[gold[key]["Causal_type"][1]]["Prediction"]:
            results.append(1)
        else:
            results.append(0)
    Consistency = sum(results) / N
    return Consistency


def extract_contrast_set(predictions, gold):
    contrast_predicitons = {}
    for key in predictions.keys():
        if "Causal_type" in gold[key].keys():
            contrast_predicitons[key] = predictions[key]
    return contrast_predicitons


def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        if gold[key]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)
    F1 = f1_score(gold_labels, pred_labels)
    Recall = recall_score(gold_labels, pred_labels)
    Precision = precision_score(gold_labels, pred_labels)  
    return F1, Recall, Precision

# def F1_Recall_Precision(predictions, gold):
#     pred_labels = []
#     gold_labels = []
    
#     # 添加计数器来追踪每种标签的数量
#     pred_counts = {"Contradiction": 0, "Entailment": 0}
#     gold_counts = {"Contradiction": 0, "Entailment": 0}
    
#     for key in predictions.keys():
#         # 统计预测标签的分布
#         pred_label = predictions[key]["Prediction"]
#         pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
        
#         # 统计真实标签的分布
#         gold_label = gold[key]["Label"]
#         gold_counts[gold_label] = gold_counts.get(gold_label, 0) + 1
        
#         # 转换为二值标签
#         pred_labels.append(1 if pred_label == "Entailment" else 0)
#         gold_labels.append(1 if gold_label == "Entailment" else 0)
    
#     # 打印标签分布
#     print(f"预测标签分布: {pred_counts}")
#     print(f"真实标签分布: {gold_counts}")
    
#     # 检查是否有足够的不同类别
#     if len(set(pred_labels)) < 2 or len(set(gold_labels)) < 2:
#         print(f"警告: 数据分布不均衡 - 预测类别数: {len(set(pred_labels))}, 真实类别数: {len(set(gold_labels))}")
#         # 返回 -1 而不是 0，表示这是一个无效的计算场景
#         return -1, -1, -1
        
#     return f1_score(gold_labels, pred_labels), precision_score(gold_labels, pred_labels), recall_score(gold_labels, pred_labels)


def main():

    # Load files
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    pred_dir = os.path.join(input_dir, 'res')
    gold_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(pred_dir):
        raise RuntimeError('{} does not exist'.format(pred_dir))

    if not os.path.isdir(gold_dir):
        raise RuntimeError('{} does not exist'.format(gold_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gold_filename = os.path.join(gold_dir, 'gold_test.json')
    pred_filename = os.path.join(pred_dir, 'predictions.json')

    with open(pred_filename) as json_file:
        predictions = json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)

    



    # Control Test Set F1, Recall, Precision PUBLIC
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(extract_control_set(predictions, gold), gold)


    # Contrast Consistency & Faithfullness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)
    Faithfulness = faithfulness(predictions_altering, gold)
    Consistency = consistency(predictions_preserving, predictions, gold)


    # Intervention-wise Consistency & Faithfullness HIDDEN
    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(predictions, gold)
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)
    definitions_preserving = extract_by_causal_type(definitions_predictions, gold)[0]
    para_Consistency = consistency(para_preserving, predictions, gold)
    cont_Faithfulness = faithfulness(cont_altering, gold)
    cont_Consistency = consistency(cont_preserving, predictions, gold)
    numerical_para_Consistency = consistency(numerical_para_preserving, predictions, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    numerical_cont_Consistency = consistency(numerical_cont_preserving, predictions, gold)
    definitions_Consistency = consistency(definitions_preserving, predictions, gold)


    # Intervention-wise F1, Recall, Precision HIDDEN
    Contrast_F1, Contrast_Rec, Contrast_Prec = F1_Recall_Precision(contrast_predictions, gold)
    para_F1, para_Rec, para_Prec = F1_Recall_Precision(para_predictions, gold)
    cont_F1, cont_Rec, cont_Prec = F1_Recall_Precision(cont_predictions, gold)
    numerical_para_F1, numerical_para_Rec, numerical_para_Prec = F1_Recall_Precision(numerical_para_predictions, gold)
    numerical_cont_F1, numerical_cont_Rec, numerical_cont_Prec = F1_Recall_Precision(numerical_cont_predictions, gold)
    definitions_F1, definitions_Rec, definitions_Prec = F1_Recall_Precision(definitions_predictions, gold)

    # Output results

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as f:
        print('Control_F1: ', Control_F1, file=f)
        print('Control_Recall: ', Control_Rec, file=f)
        print('Control_Precision: ', Control_Prec, file=f)
        print('Contrast_F1: ', Contrast_F1, file=f)
        print('Contrast_Recall: ', Contrast_Rec, file=f)
        print('Contrast_Precision: ', Contrast_Prec, file=f)
        print('Faithfulness: ', Faithfulness, file=f)
        print('Consistency: ', Consistency, file=f)
        print('Para_Consistency: ', para_Consistency, file=f)
        print('Cont_Faithfulness: ', cont_Faithfulness, file=f)
        print('Cont_Consistency: ', cont_Consistency, file=f)
        print('Numerical_Para_Consistency: ', numerical_para_Consistency, file=f)
        print('Numerical_Cont_Faithfulness: ', numerical_cont_Faithfulness, file=f)
        print('Numerical_Cont_Consistency: ', numerical_cont_Consistency, file=f)
        print('Definitions_Consistency: ', definitions_Consistency, file=f)
        print('Para_F1: ', para_F1, file=f)
        print('Para_Recall: ', para_Rec, file=f)
        print('Para_Precision: ', para_Prec, file=f)
        print('Cont_F1: ', cont_F1, file=f)
        print('Cont_Recall: ', cont_Rec, file=f)
        print('Cont_Precision: ', cont_Prec, file=f)
        print('Numerical_Para_F1: ', numerical_para_F1, file=f)
        print('Numerical_Para_Recall: ', numerical_para_Rec, file=f)
        print('Numerical_Para_Precision: ', numerical_para_Prec, file=f)
        print('Numerical_Cont_F1: ', numerical_cont_F1, file=f)
        print('Numerical_Cont_Recall: ', numerical_cont_Rec, file=f)
        print('Numerical_Cont_Precision: ', numerical_cont_Prec, file=f)
        print('Definitions_F1: ', definitions_F1, file=f)
        print('Definitions_Recall: ', definitions_Rec, file=f)
        print('Definitions_Precision: ', definitions_Prec, file=f)


if '__main__' == __name__:
    main()










