import json
import glob
import os
from sklearn.metrics import precision_recall_fscore_support

def remove_role(name):
    if "(" in name and ")" in name:
        return name.split("(")[0].strip()
    return name

def get_predictions(predictions_file):
    with open(predictions_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_expected(articles_folder):
    expected_all = {}
    article_paths = glob.glob(os.path.join(articles_folder, "*.json"))
    
    for article_path in article_paths:
        article_name = os.path.basename(article_path)
        with open(article_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        expected_all[article_name] = data
    
    return expected_all

def evaluate_task2_by_alignment(predictions, expected, alignment):
    
    y_true = []
    y_pred = []
    
    for article_name, pred_data in predictions.items():
        if article_name not in expected:
            continue
        
        expected_task1 = expected[article_name].get('task1', {}) # use ground truth classification
        expected_task2 = expected[article_name].get('task2', {})
        predicted_task2 = pred_data.get('task2_prediction', {})
        predicted_task1 = pred_data.get('task1_prediction', {})

        
        alignment_used = predicted_task1
        entities = set()
        for entity in alignment_used.get(alignment, []):
            # if we use the model's output remove roles
            cleaned = remove_role(entity).lower() if remove_role(entity) else entity.lower()  
            entities.add(cleaned)
        
        if not entities:
            continue
            
        all_paras = set(expected_task2.keys()) | set(predicted_task2.keys())
        
        # binary labels for specified-alignment entities
        for para in all_paras:
            expected_entities = [e.lower() for e in expected_task2.get(para, [])]
            predicted_entities = [e.lower() for e in predicted_task2.get(para, [])]
            
            for entity in entities:
                y_true.append(1 if entity in expected_entities else 0)
                y_pred.append(1 if entity in predicted_entities else 0)
    
    if y_true and any(y_true):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": len(y_true),  # num_entities x num_paras
            "actual_num_assigments": sum(y_true) # total preds - change this
        }
    return None

def main():
    PREDICTIONS_FILE = "results/2025-06-17_00-46_predictions_evaluation_set.json"
    ARTICLES_FOLDER = "./data/evaluation_dataset/" 

    predictions = get_predictions(PREDICTIONS_FILE)
    expected_all = get_expected(ARTICLES_FOLDER)
    
    print("Using model's alignment classification for task 1")
    # Victim-aligned evaluation
    victim_results = evaluate_task2_by_alignment(predictions, expected_all, "Victim-aligned")
    if victim_results:
        print("Victim-aligned evaluation results:")
        print(json.dumps(victim_results, indent=2))
    else:
        print("No victim-aligned results found")
    
    # Police-aligned evaluation
    police_results = evaluate_task2_by_alignment(predictions, expected_all, "Police-aligned")
    if police_results:
        print("Police-aligned evaluation results:")
        print(json.dumps(police_results, indent=2))
    else:
        print("No police-aligned results found")
    
    # Save results
    # if victim_results or police_results:
    #     output_file = "results/alignment_breakdown.json"
    #     with open(output_file, 'w') as f:
    #         json.dump({
    #             "victim_aligned": victim_results,
    #             "police_aligned": police_results
    #         }, f, indent=2)
    #     print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()