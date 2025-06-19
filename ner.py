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

def load_prompt(filename):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        return f.read()

TASK1_PROMPT = load_prompt("task1_prompt.txt")
TASK2_PROMPT = load_prompt("task2_prompt.txt")
ARTICLES_FOLDER = "data/articles/"

def load_client():
    config = dotenv_values(".env")
    key = config["OPENAI_API_KEY"] 
    client = OpenAI(
        api_key=key
    )
    return client


# TASK 1 HELPERS 

# remove roles 
def clean_name(name):
    if "(" in name and ")" in name:
        return name.split("(")[0].strip()

# cleans backticks that gpt-4o sometimes includes
def clean_response(raw):
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]  
        if lines and lines[-1].strip().endswith("```"):
            lines = lines[:-1] 
        raw = "\n".join(lines).strip()
    return raw

# evaluates task 1 for a single article 
def evaluate_task1(pred, expected_dict):    
    pred_victim = set()
    for name in pred["Victim-aligned"]:
        cleaned_name = clean_name(name)
        if cleaned_name:
            pred_victim.add(clean_name(name))
        else:
            pred_victim.add(name.lower())
    
    pred_police = set()
    for name in pred["Police-aligned"]:
        cleaned_name = clean_name(name)
        if cleaned_name:
            pred_police.add(clean_name(name))
        else:
            pred_police.add(name.lower())
    
    # DEBUG LOG: Uncomment to easily see how Task 1 predictions compare to expected results
    # print("------", article_name, "-----TASK 1 EVALUATION----")
    # print("\nPredicted Victim-aligned: ", pred_victim)
    # print("\nActual Victim-aligned: ", set(expected_dict["Victim-aligned"]))
    # print("\nPredicted Police-aligned: ", pred_police)
    # print("\nActual Police-aligned: ", set(expected_dict["Police-aligned"]))

    people = (
        pred_victim |
        pred_police |
        set(expected_dict["Victim-aligned"]) |
        set(expected_dict["Police-aligned"])
    )

    y_true = []
    y_pred = []


    for person in people:
        if person in expected_dict["Victim-aligned"]:
            y_true.append("Victim-aligned")
        elif person in expected_dict["Police-aligned"]:
            y_true.append("Police-aligned")
        else:
            y_true.append("Unknown")

        if person in pred_victim:
            y_pred.append("Victim-aligned")
        elif person in pred_police:
            y_pred.append("Police-aligned")
        else:
            y_pred.append("Unknown")

    evaluation_dict = classification_report(
        y_true,
        y_pred,
        labels=["Victim-aligned", "Police-aligned"],
        digits=3,
        zero_division=0,
        output_dict=True
    )

    return evaluation_dict, y_true, y_pred

# returns a classification report for Task 1
def task1_evaluation_report(results):
    all_y_true = []
    all_y_pred = []
    
    for result in results:
        if result.get('y_true') and result.get('y_pred'):
            all_y_true.extend(result['y_true'])
            all_y_pred.extend(result['y_pred'])
    # DEBUG
    # prints results to terminal
    # if all_y_true and all_y_pred:
    #     overall_report = classification_report(
    #         all_y_true, 
    #         all_y_pred, 
    #         labels=["Victim-aligned", "Police-aligned"], 
    #         digits=3, 
    #         zero_division=0
    #     )
    #     print(overall_report)
    if all_y_true and all_y_pred:    
        report_dict = classification_report(
            all_y_true, 
            all_y_pred, 
            labels=["Victim-aligned", "Police-aligned"], 
            digits=3, 
            zero_division=0,
            output_dict=True,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = timestamp + "_task1_evaluation.json"
        save_path = os.path.join("results", filename)
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=2)
    else:
        print("WARNING: No valid Task 1 predictions found for overall report")


# Task1: identifies individuals and their affiliations in an article
def task1(client, article_path):
    article_name = os.path.basename(article_path)
    
    try:
        with open(article_path, "r") as f:
            article_data = json.load(f)
        
        # Extract article content
        # Evaluation dataset articles will be in dict format
        if isinstance(article_data, dict):
            article_content = article_data["article"]
        # Otherwise we should be given an array
        elif isinstance(article_data, list):
            article_content = article_data
        else:
            raise ValueError("Unexpected article format in", article_path)

        if isinstance(article_content, list):
            article_content = "\n".join(article_content)
        
        message = TASK1_PROMPT + article_content
        response = client.chat.completions.create(
                model="gpt-4o", 
                temperature=0,
                max_tokens = 1500, 
                messages = [
                    {"role": "user", "content" : message}
                ]
        )

        response = clean_response(response.choices[0].message.content)


        try:
            response_as_dict = ast.literal_eval(response)
        except (ValueError, SyntaxError) as e:
            # DEBUG LOG
            # print("DEBUG: Error parsing response for:", article_name)
            # print("---- RAW RESPONSE START ----")
            # print(raw_response)
            # print("---- RAW RESPONSE END ----")
            # print("Parse error:", str(e))
            return {"article": article_name, "error": "Failed to parse response"}

        # Get expected results from the same file
        if isinstance(article_data, list):
            return {
            "article": article_name,
            "article_data": article_data,
            "response": response,
            "response_as_dict": response_as_dict
            }
        elif isinstance(article_data, dict) and article_data.get("task1", {}):

            expected_task1 = article_data.get("task1", {})
            evaluation_result, y_true, y_pred = evaluate_task1(response_as_dict, expected_task1)
            
            return {
                "article": article_name,
                "article_data": article_data,
                "response": response,
                "response_as_dict": response_as_dict,
                "evaluation": evaluation_result,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        else:
            return{"WARNING: Task1 formatting or expected is wrong for ", article_name}

        
    except Exception as e:
        print("ERROR: Failed to process ", article_name, ": ", str(e))
        return {"article": article_name, "error": str(e)}

# TASK 2 HELPERS 

# Sort paragraphs, in case they are returned out of order
def sort_paragraph_nums(all_paras):
    paragraph_numbers = []
    for para in all_paras:
        try:
            num = int(para.lower().replace("paragraph", "").strip())
            paragraph_numbers.append(num)
        except:
            print("WARNING: Could not parse paragraph number from " + {para})
            continue
    return sorted(paragraph_numbers)

def evaluate_task2(expected, predicted):
    expected_paras = set(expected.keys())
    predicted_paras = set(predicted.keys())
    all_paras = expected_paras | predicted_paras

    # DEBUG LOG
    # if expected_paras != predicted_paras:
    #     print("PARAGRAPH COVERAGE:")
    #     print(f"Expected paragraphs: {sorted(expected_paras)}")
    #     print(f"Predicted paragraphs: {sorted(predicted_paras)}")
    #     print(f"Missing from prediction: {sorted(expected_paras - predicted_paras)}")
    #     print(f"Extra in prediction: {sorted(predicted_paras - expected_paras)}")
    #     print(f"Evaluating on all {len(all_paras)} paragraphs")
    
    paragraph_numbers = sort_paragraph_nums(all_paras)

    y_true = []
    y_pred = []

    all_entities = set()
    for entities in expected.values():
        all_entities.update(entities)
    for entities in predicted.values():
        all_entities.update(entities)
    
    if not all_entities:
        print("WARNING: No entities found in either expected or predicted results")
        return (None, None, None)

    # For each paragraph, create a binary label for every entity to rep. whether or not is it tagged
    for num in paragraph_numbers:
        para_key = f"paragraph {num}"
        expected_entities = expected.get(para_key, [])
        predicted_entities = predicted.get(para_key, [])
        
        for entity in all_entities:
            y_true.append(1 if entity in expected_entities else 0)
            y_pred.append(1 if entity in predicted_entities else 0)

    return precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    ), y_true, y_pred


def task2_evaluation_report(results):
    all_y_true = []
    all_y_pred = []

    for result in results:
        if result.get("y_true") and result.get("y_pred"):
            all_y_true.extend(result["y_true"])
            all_y_pred.extend(result["y_pred"])

    if all_y_true and all_y_pred:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_y_true,
            all_y_pred,
            average='binary',
            zero_division=0
        )
        report_dict = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": len(all_y_true)
        }

        # DEBUG
        # print("OVERALL TASK 2 REPORT:")
        # print(json.dumps(report_dict))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = timestamp + "_task2_evaluation.json"
        save_path = os.path.join("results", filename)
        with open(save_path, "w") as f:
            json.dump(report_dict, f)

    else:
        print("No valid Task 2 predictions found.")


# Task 2: assign paragraphs to individuals    
def task2(client, article_data, task1_response):
    if isinstance(article_data, dict):
        article_text = article_data.get("article", "")
    else:
        article_text = article_data

    numbered_article = []
    if isinstance(article_text, list):
        # article_text = "\n\n-PARAGRAPH BREAK-\n\n".join(article_text)
        for i, para in enumerate(article_text, start=1):
            line = "Paragraph " + str(i) + ": " + para
            numbered_article.append(line)
    article_text_prompt = "\n\n".join(numbered_article)
    
    messages = [
        {"role": "user", "content": TASK1_PROMPT + article_text_prompt},
        {"role": "assistant", "content": task1_response},
        {"role": "user", "content": TASK2_PROMPT}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=1500,
        messages=messages
    )

    response = clean_response(response.choices[0].message.content)

    try:
        response_as_dict = ast.literal_eval(response)
    except Exception as e:
        print("Failed to parse Task 2 output:", response)
        return {"error": str(e), "raw": response}

    if isinstance(article_data, list):
        return {
            "response_as_dict": response_as_dict,
            "response": response,
        }
    elif isinstance(article_data, dict) and not article_data.get("task2", {}):
        print("WARNING: No expected task2 results found")
        return {
            "response_as_dict": response_as_dict,
            "response": response,
        }
    else:
        expected_task2 = article_data.get("task2", {})
        eval_metrics, y_true, y_pred = evaluate_task2(expected_task2, response_as_dict)

        if eval_metrics is None:
            print("WARNING: Skipping Task 2 eval")
            return {
                "response_as_dict": response_as_dict,
                "response": response,
                "error": "Unknown",
            }

        return {
            "response_as_dict": response_as_dict,
            "response": response,
            "evaluation": {
                "precision": eval_metrics[0],
                "recall": eval_metrics[1],
                "f1": eval_metrics[2],
                "support": eval_metrics[3],
            },
            "y_true": y_true,
            "y_pred": y_pred
        }

def prune_empty_paragraphs(paragraphs):
    if not isinstance(paragraphs, dict):
        return None
    return {para: entity for para, entity in paragraphs.items() if entity}

def save_predictions(task1_results, task2_results):
    combined = {}

    for t1_res, t2_res in zip(task1_results, task2_results):
        prediction = {
            "task1_prediction": t1_res.get("response_as_dict", None),
            "task2_prediction": prune_empty_paragraphs(t2_res.get("response_as_dict", None))
        }
        article = t1_res.get("article", None)
        combined[article] = prediction
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = timestamp + "_predictions.json"
    save_path = os.path.join("results", filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print("Predictions saved to " + save_path)

    return

def evaluate_predictions(task1_results, task2_results):
    print("Starting Evaluation...")
    task1_evaluation_report(task1_results) 
    task2_evaluation_report(task2_results)
    print("Evaluation complete")

def main():
    articles_dir = ARTICLES_FOLDER
    article_paths = glob.glob(os.path.join(articles_dir, "*.json"))
    
    if not article_paths:
        print("No articles found in ", articles_dir)
        return
        
    client = load_client()
    
    task1_results = []
    print("Starting Task 1...")
    for article_path in article_paths:
        task1_result = task1(client, article_path)
        task1_results.append(task1_result)
    print("Task 1 Complete")

    task2_results = []
    print("Starting Task 2...")
    for task1_result in task1_results:
        task2_result = task2(client, task1_result["article_data"], task1_result["response"])
        task2_results.append(task2_result)
    print("Task 2 Complete")


    # Evaluate against your dataset
    evaluate_predictions(task1_results, task2_results)

    save_predictions(task1_results, task2_results)

    return    
    
    

if __name__ == "__main__":
    main()