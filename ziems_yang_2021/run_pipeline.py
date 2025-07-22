import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import sys
from tqdm import tqdm

import task1
import task2
from task1_evaluation import evaluate_task1, task1_evaluation_report
from task2_evaluation import evaluate_task2, evaluate_task2_by_alignment, remove_role


def load_ff(py_path: Path):
    spec = importlib.util.spec_from_file_location("ff", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_victim_name(json_file):
    m = re.search(r"_(.*?)_", json_file)
    return m.group(1) if m else None


def save_output(file_name, data, save_folder):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = timestamp + "_" + file_name + ".json"
    save_path = os.path.join(save_folder, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_task1(json_files, articles_folder, ff):
    output_pred = {}
    for json_file in tqdm(json_files, desc="Task 1", unit="article"):
        article_path = f"{articles_folder}/{json_file}"
        data = json.loads(Path(article_path).read_text())
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of sentences (strings).")
        name = get_victim_name(article_path)
        result = task1.process_article(data, ff, name, "ignore_gender", "ignore_race")
        output_pred[json_file] = result
    save_output("task1_predictions", output_pred, "predictions")


def eval_task1(gold_path, predictions_dir="predictions"):
    pred_path = max(
        Path(predictions_dir).glob("*task1_predictions.json"),
        key=lambda p: p.stat().st_mtime,
    )
    print(f"Using {pred_path.name}")
    pred_task1 = json.load(open(pred_path))
    gold = json.load(open(gold_path))

    results = []
    for art, block in pred_task1.items():
        cleaned_task1 = {
            k: [remove_role(n) for n in v] for k, v in gold[art]["task1"].items()
        }
        prfs, y_true, y_pred = evaluate_task1(block["task1"], cleaned_task1)
        results.append({"article": art, "y_true": y_true, "y_pred": y_pred})

    task1_evaluation_report(results)


def run_task2(json_files, articles_folder, ff, task1_gold):
    task1 = json.loads(Path(task1_gold).read_text())
    output_pred_task2 = {}

    for json_file in tqdm(json_files, desc="Task 2", unit="article"):
        article_path = Path(articles_folder) / json_file
        paragraphs = json.loads(article_path.read_text())
        if not isinstance(paragraphs, list):
            raise ValueError("article_json must be a list of strings (paragraphs).")

        art_gold = task1[json_file]["task1"]
        all_names = [
            remove_role(n)
            for n in (art_gold["Victim-aligned"] + art_gold["Police-aligned"])
        ]

        result = task2.process_article(paragraphs, ff, all_names)
        output_pred_task2[json_file] = result

    save_output("task2_predictions", output_pred_task2, "predictions")


def eval_task2(gold_path, predictions_dir="predictions"):
    pred_path = max(
        Path(predictions_dir).glob("*task2_predictions.json"),
        key=lambda p: p.stat().st_mtime,
    )
    print(f"Using {pred_path.name}")
    pred_raw = json.load(open(pred_path))
    gold = json.load(open(gold_path))

    wrapped_preds = {
        art: {
            "task2_prediction": blk["task2"],
            "task1_prediction": {
                k: [remove_role(n) for n in v] for k, v in gold[art]["task1"].items()
            },
        }
        for art, blk in pred_raw.items()
        if art in gold
    }

    victim_metrics = evaluate_task2_by_alignment(wrapped_preds, gold, "Victim-aligned")
    police_metrics = evaluate_task2_by_alignment(wrapped_preds, gold, "Police-aligned")

    all_y_true, all_y_pred = [], []
    for art, blk in pred_raw.items():
        if art not in gold or "task2" not in gold[art]:
            continue
        prfs_tuple, y_t, y_p = evaluate_task2(gold[art]["task2"], blk["task2"])
        if y_t and y_p:
            all_y_true.extend(y_t)
            all_y_pred.extend(y_p)

    P, R, F1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average="binary", zero_division=0
    )

    overall_metrics = {
        "precision": round(P, 3),
        "recall": round(R, 3),
        "f1": round(F1, 3),
        "support": len(all_y_true),
    }

    report = {
        "overall": overall_metrics,
        "victim-aligned": victim_metrics,
        "police-aligned": police_metrics,
    }

    save_output("task2_evaluation", report, "results")


def main():
    articles_folder = "../data/articles/"
    gold = "../data/training_data.json"
    ff = load_ff(Path("framing_functions.py"))

    json_files = sorted(
        [f for f in os.listdir(articles_folder) if f.endswith(".json")],
        key=lambda f: int(f.split("_")[0]),
    )

    run_task1(json_files, articles_folder, ff)
    run_task2(json_files, articles_folder, ff, gold)

    eval_task1(gold)
    eval_task2(gold)


if __name__ == "__main__":
    main()
