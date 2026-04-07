import json

def calculate_pope_metrics(file_path):
    results = [json.loads(l) for l in open(file_path, "r")]
    
    total = len(results)
    yes_count = 0
    correct = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for res in results:
        pred = res["model_answer"].lower()
        label = res["label"].lower()

        if pred == "yes":
            yes_count += 1
        
        if pred == label:
            correct += 1
            
        # 计算 F1 指标
        if pred == "yes" and label == "yes": tp += 1
        elif pred == "yes" and label == "no": fp += 1
        elif pred == "no" and label == "no": tn += 1
        elif pred == "no" and label == "yes": fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 30)
    print(f"Yes Ratio: {yes_count / total * 100:.2f}%")
    print(f"Accuracy:  {correct / total * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # 运行此脚本计算最终结果
    calculate_pope_metrics("results.jsonl")
