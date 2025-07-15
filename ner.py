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
ARTICLES_FOLDER = "data/articles/batch5"

def load_client():
    config = dotenv_values(".env")
    key = config["OPENAI_API_KEY"] 
    client = OpenAI(
        api_key=key
    )
    return client


# TASK 1 HELPERS 
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
            
            return {
                "article": article_name,
                "article_data": article_data,
                "response": response,
                "response_as_dict": response_as_dict,
            }
        else:
            return{"WARNING: Task1 formatting or expected is wrong for ", article_name}

        
    except Exception as e:
        print("ERROR: Failed to process ", article_name, ": ", str(e))
        return {"article": article_name, "error": str(e)}



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


        return {
            "response_as_dict": response_as_dict,
            "response": response,
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

    save_predictions(task1_results, task2_results)

    return    
    
    

if __name__ == "__main__":
    main()