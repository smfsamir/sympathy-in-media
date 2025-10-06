#!/usr/bin/env python3
from openai import OpenAI
from dotenv import dotenv_values
from collections import defaultdict
import ast
import os
import json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import glob
from datetime import datetime
from splits import TRAIN_DEV, TRAIN_FILES, VAL_FILES, TEST_FILES
import argparse
from statistics import median
import time
import random
from tqdm import tqdm
import openai  # needed for RateLimitError


def load_prompt(filename):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        return f.read()

TASK1_PROMPT = load_prompt("v1/task1_prompt.txt")
TASK2_PROMPT = load_prompt("v1/task2_prompt.txt")
ARTICLES_FOLDER = "data/articles"


def call_with_retries(client, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            wait = (2 ** attempt) + random.random()
            print(f"Rate limit hit, retrying in {wait:.2f}s...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


def load_client():
    config = dotenv_values(".env")
    key = config["OPENAI_API_KEY"]
    client = OpenAI(api_key=key)
    return client


def resolve_article_paths(file_list=None, base_dir=ARTICLES_FOLDER):
    if file_list:
        paths = []
        search_root = os.path.dirname(base_dir.rstrip(os.sep)) or base_dir
        for name in file_list:
            direct = os.path.join(base_dir, name)
            if os.path.exists(direct):
                paths.append(direct)
                continue
            hits = glob.glob(os.path.join(search_root, "**", name), recursive=True)
            if hits:
                paths.append(hits[0])
            else:
                print("WARNING: Missing file " + name)
        return paths
    return glob.glob(os.path.join(base_dir, "*.json"))


# HELPER: cleans backticks that gpt-4o sometimes includes
def clean_response(raw):
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().endswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw

def task1(client, article_path):
    article_name = os.path.basename(article_path)

    try:
        with open(article_path, "r") as f:
            article_data = json.load(f)

        if isinstance(article_data, dict):
            article_content = article_data["article"]
        elif isinstance(article_data, list):
            article_content = article_data
        else:
            raise ValueError("Unexpected article format in", article_path)

        if isinstance(article_content, list):
            article_content = "\n".join(article_content)

        message = TASK1_PROMPT + article_content

        response = call_with_retries(
            client,
            model="gpt-4o",
            temperature=0,
            max_tokens=1500,
            messages=[{"role": "user", "content": message}],
        )

        usage = response.usage
        cleaned = clean_response(response.choices[0].message.content)

        try:
            response_as_dict = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return {"article": article_name, "error": "Failed to parse response"}

        result = {
            "article": article_name,
            "article_data": article_data,
            "response": cleaned,
            "response_as_dict": response_as_dict,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
        return result

    except Exception as e:
        print("ERROR: Failed to process ", article_name, ": ", str(e))
        return {"article": article_name, "error": str(e)}


# Task2: assign paragraphs to individuals
def task2(client, article_data, task1_response):
    if isinstance(article_data, dict):
        article_text = article_data.get("article", "")
    else:
        article_text = article_data

    numbered_article = []
    if isinstance(article_text, list):
        for i, para in enumerate(article_text, start=1):
            line = "Paragraph " + str(i) + ": " + para
            numbered_article.append(line)
    article_text_prompt = "\n\n".join(numbered_article)

    messages = [
        {"role": "user", "content": TASK1_PROMPT + article_text_prompt},
        {"role": "assistant", "content": task1_response},
        {"role": "user", "content": TASK2_PROMPT},
    ]

    response = call_with_retries(
        client,
        model="gpt-4o",
        temperature=0,
        max_tokens=1500,
        messages=messages,
    )

    usage = response.usage
    cleaned = clean_response(response.choices[0].message.content)

    try:
        response_as_dict = ast.literal_eval(cleaned)
    except Exception as e:
        print("Failed to parse Task 2 output:", cleaned)
        return {"error": str(e), "raw": cleaned}

    result = {
        "response_as_dict": response_as_dict,
        "response": cleaned,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }
    return result


def prune_empty_paragraphs(paragraphs):
    if not isinstance(paragraphs, dict):
        return None
    return {para: entity for para, entity in paragraphs.items() if entity}


def save_predictions(task1_results, task2_results):
    combined = {}

    for t1_res, t2_res in zip(task1_results, task2_results):
        prediction = {
            "task1_prediction": t1_res.get("response_as_dict", None),
            "task2_prediction": prune_empty_paragraphs(
                t2_res.get("response_as_dict", None)
            ),
        }
        article = t1_res.get("article", None)
        combined[article] = prediction

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = timestamp + "_predictions.json"
    save_path = os.path.join("results", filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print("Predictions saved to " + save_path)


def summarize_usage(task1_results, task2_results):
    usage_summary = {
        "n_requests": 0,
        "median_prompt_tokens": 0,
        "median_completion_tokens": 0,
        "median_total_tokens": 0,
        "per_task": {"task1": {}, "task2": {}},
    }

    all_prompt, all_completion, all_total = [], [], []
    task1_prompt, task1_completion, task1_total = [], [], []
    task2_prompt, task2_completion, task2_total = [], [], []

    for res in task1_results:
        if "usage" in res:
            task1_prompt.append(res["usage"]["prompt_tokens"])
            task1_completion.append(res["usage"]["completion_tokens"])
            task1_total.append(res["usage"]["total_tokens"])
            all_prompt.append(res["usage"]["prompt_tokens"])
            all_completion.append(res["usage"]["completion_tokens"])
            all_total.append(res["usage"]["total_tokens"])
            usage_summary["n_requests"] += 1

    for res in task2_results:
        if "usage" in res:
            task2_prompt.append(res["usage"]["prompt_tokens"])
            task2_completion.append(res["usage"]["completion_tokens"])
            task2_total.append(res["usage"]["total_tokens"])
            all_prompt.append(res["usage"]["prompt_tokens"])
            all_completion.append(res["usage"]["completion_tokens"])
            all_total.append(res["usage"]["total_tokens"])
            usage_summary["n_requests"] += 1

    if all_prompt:
        usage_summary["median_prompt_tokens"] = median(all_prompt)
        usage_summary["median_completion_tokens"] = median(all_completion)
        usage_summary["median_total_tokens"] = median(all_total)

    usage_summary["per_task"]["task1"] = {
        "median_prompt_tokens": median(task1_prompt) if task1_prompt else 0,
        "median_completion_tokens": median(task1_completion) if task1_completion else 0,
        "median_total_tokens": median(task1_total) if task1_total else 0,
    }

    usage_summary["per_task"]["task2"] = {
        "median_prompt_tokens": median(task2_prompt) if task2_prompt else 0,
        "median_completion_tokens": median(task2_completion) if task2_completion else 0,
        "median_total_tokens": median(task2_total) if task2_total else 0,
    }

    return usage_summary


def main(file_list=None):
    article_paths = resolve_article_paths(file_list, ARTICLES_FOLDER)

    if not article_paths:
        print("No articles found in " + ARTICLES_FOLDER)
        return

    client = load_client()

    # --- Task 1 ---
    task1_results = []
    print(f"Starting Task 1 on {len(article_paths)} articles...")
    for article_path in tqdm(article_paths, desc="Task 1"):
        task1_result = task1(client, article_path)
        task1_results.append(task1_result)
        time.sleep(1)  # avoid token burst
    print("Task 1 Complete")

    # --- Task 2 ---
    task2_results = []
    print(f"Starting Task 2 on {len(task1_results)} articles...")
    for task1_result in tqdm(task1_results, desc="Task 2"):
        if "error" in task1_result:
            task2_results.append({"error": "skipped due to task1 error"})
            continue
        task2_result = task2(
            client, task1_result["article_data"], task1_result["response"]
        )
        task2_results.append(task2_result)
        time.sleep(1)
    print("Task 2 Complete")

    save_predictions(task1_results, task2_results)

    summary = summarize_usage(task1_results, task2_results)
    out_file = f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M')}_token_usage_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print("Token usage summary saved to", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "train", "val", "test"])
    args = parser.parse_args()

    split_map = {
        "dev": TRAIN_DEV,
        "train": TRAIN_FILES,
        "val": VAL_FILES,
        "test": TEST_FILES,
    }
    chosen = split_map.get(args.split) if args.split else None
    main(chosen)
