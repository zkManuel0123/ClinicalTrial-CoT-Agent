import json

def extract_predictions(file_path='predictions_test_all.json'):
    """
    从JSON文件中提取prediction键值对
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        dict: 只包含prediction的字典
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建新字典只保存prediction
        predictions = {}
        for key, value in data.items():
            predictions[key] = {'Prediction': value['Prediction']}
            
        return predictions
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"错误: '{file_path}' 不是有效的JSON文件")
        return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def save_predictions(predictions, output_file='predictions_only.json'):
    """
    将提取的predictions保存到新的JSON文件
    
    Args:
        predictions: 提取的predictions字典
        output_file: 输出文件名
    """
    if predictions:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=4)
            print(f"成功保存到 {output_file}")
        except Exception as e:
            print(f"保存文件时发生错误: {str(e)}")

def main():
    # 提取predictions
    predictions = extract_predictions()
    
    # 保存结果
    if predictions:
        save_predictions(predictions)
        print(f"共处理 {len(predictions)} 条数据")

if __name__ == "__main__":
    main()
