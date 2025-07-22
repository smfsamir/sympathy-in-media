import os
import re
import json
from extended.task1 import preprocess_article_with_unidecode, extract_and_classify_entities
from task2 import get_paragraph_perspectives
from task2_eval import evaluate_task2, task2_evaluation_report
from task1_eval import evaluate_task1, task1_evaluation_report


ARTICLES_FOLDER = "./data/evaluation_dataset"
# this should be implemented for partitioning based on yang2021
# PEOPLE_TERMS_FILE = "./resources/textbook_analysis/people_terms.csv"
# PB_WORDS_FILE = "./regex/police.txt"
# VA_WORDS_FILE = "./regex/victim.txt"
# HUMAN_NOUNS = set(pd.read_csv(PEOPLE_TERMS_FILE, names=['noun', 'race/gender', 'category'])['noun'].values)

def get_victim_name(json_file):
    match = re.search(r'_(.*?)_', json_file)
    return match.group(1) if match else None

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def evaluate_task(predictions_file, expected_file, task_name, evaluate_func, report_func):
    predictions = load_json(predictions_file)
    expected = load_json(expected_file)

    results = []

    for article_name, prediction in predictions.items():
        if article_name not in expected:
            print(f"Skipping {article_name}, no gold labels.")
            continue

        pred = prediction[task_name]
        expected_dict = expected[article_name]

        evaluation_dict, y_true, y_pred = evaluate_func(pred, expected_dict)
        results.append({"y_true": y_true, "y_pred": y_pred})

    report_func(results)


if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)

    json_files = sorted(
        [f for f in os.listdir(ARTICLES_FOLDER) if f.endswith(".json")],
        key=lambda f: int(f.split('_')[0])
    )

    task1_output = {}

    for json_file in json_files:
        data = load_json(os.path.join(ARTICLES_FOLDER, json_file))
        victim_name = get_victim_name(json_file)

        text = preprocess_article_with_unidecode(data['article'])
        victim_tokens, officer_tokens = extract_and_classify_entities(text, victim_name.lower())

        combined_tokens = list(victim_tokens + officer_tokens)
        # print(combined_tokens)

        task1_output[json_file] = {
            "task1": {
                "Victim-aligned": victim_tokens,
                "Police-aligned": officer_tokens
            },
            "task2": get_paragraph_perspectives(data['article'], combined_tokens)
        }

    predictions_file = "./output/predictions.json"
    write_json(predictions_file, task1_output)

    evaluate_task(
        predictions_file=predictions_file,
        expected_file="./eval/task1_eval.json",
        task_name="task1",
        evaluate_func=evaluate_task1,
        report_func=task1_evaluation_report
    )

    evaluate_task(
        predictions_file=predictions_file,
        expected_file="./eval/task2_eval.json",
        task_name="task2",
        evaluate_func=evaluate_task2,
        report_func=task2_evaluation_report
    )
