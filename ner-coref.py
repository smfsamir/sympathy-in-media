import openai
from dotenv import dotenv_values
from collections import defaultdict
import ast
import os
import json
from sklearn.metrics import classification_report
import glob


AFFILIATION_MAP = {
    "Victim-aligned": "Victim-aligned",
    "Government officials": "Police-aligned",
    "Policing institution officials": "Police-aligned"
}

'''

===========================      PROMPTS      =============================================

# NOTE
## Questions:
## I said "dominant entity" as the person whose perspective is in that paragraph since the paragraph could mention several entities
## for part b (testing) do I just do that manually on a subset of the articles we have so far
## TLDR i dont think my prompt is precise or robust enough so results are not great

# make task 2 its own prompt and then 

# task 2 could be a list of lists 
# so one paragraph = 1 list 

'''
# Task 1: idenitfy individuals and their affiliations
task1_prompt = """
Read the provided article on an incident of police violence to complete the following two tasks. Task 1: Identify each person mentioned in the article and determine their affiliation. The same person could be mentioned many times by different referring expressions.

For instance, the terms "he", "Bob Dylan", "the singer", and "Dylan" all refer to the same entity.

Use the article context to recognize these mentions as co-references and list the person only once using their most complete name (e.g. "Bob Dylan"). Do not create separate entities for pronouns, mentions, and names (e.g. "he", "the singer", "Bob Dylan", etc). A person can either be on the side of the police, or the civilian victim.

For example, government officials including police chiefs, mayors, police oversight officials, police unions officials. Government officials should be assigned into the police-aligned category, even if their statements are ostensibly sympathetic. There may also be bystanders who take the side of the police. For these bystanders, it is important to take the quotes into account in determining whether they side with the police or not.  

People who side with the civilian victim, on the other hand, tend to be, of course, the victim's family or friends, their attorney, community members who knew the victim, witnesses, civil rights organization leaders, and more.

People mentioned from previous cases (whether officers or victims) should also be recorded, even if they are not part of the current case itself. 

Following each person's name, indicate their role in parentheses, such as (victim), (police chief), (witness), (mayor), (victim's mother), (police officer) etc. 

Return the results in the following format:

{

"Victim-aligned": ["Person1 (role)", "Person2 (role)", "Person3 (role)"],

"Government officials": ["Person (role)", "Person5 (role)", "Person6 (role)"],

"Policing institution officials": ["Person7 (role)", "Person8 (role)"]

}

Here is the article:
"""


def load_client():
    config = dotenv_values(".env")
    key = config["OPENAI_API_KEY"] 
    client = openai.AzureOpenAI(
            azure_endpoint="https://ubcnlpgpt4.openai.azure.com/", 
            api_key=key,
            api_version="2023-05-15" 
        )
    return client

def map_affiliations(raw):
    # raw = ast.literal_eval(raw)
    res = { "Victim-aligned": [], "Police-aligned": [] }
    for key, names in raw.items():
        for name in names:
            map_key = AFFILIATION_MAP.get(key)
            if map_key and name not in res[map_key]:
                res[map_key].append(name)
    # print("Mapped affiliations: ", res)
    return res

def clean_name(name):
    if "(" in name and ")" in name:
        return name.split("(")[0].strip().lower()

def evaluate_task1(pred_raw, expected_dict):
    pred = map_affiliations(pred_raw)
    # need to clean the names here
    
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

    print("Predicted Victim-aligned: ", pred_victim)
    print("Actual Victim-aligned: ", set(expected_dict["Victim-aligned"]))
    print("Predicted Police-aligned: ", pred_police)
    print("Actual Police-aligned: ", set(expected_dict["Police-aligned"]))

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



def process_article(client, article_path):
    article_name = os.path.basename(article_path)
    
    try:
        with open(article_path, "r") as f:
            article_data = json.load(f)
        
        # Extract article content - handle both string and list formats
        article_content = article_data.get("article", "")
        if isinstance(article_content, list):
            article_content = "\n".join(article_content)
        
        message = task1_prompt + article_content
        response = client.chat.completions.create(
                model="gpt-4o", # gpt-4o, or you can also try gpt-4o-mini. See here for more details https://platform.openai.com/docs/models. But stick to 4o or 4o-mini
                temperature=0,
                max_tokens = 500, # set this to whatever makes sense. It's the maximum number of tokens (basically, words) that you think are necessary for responding to the prompt
                messages = [
                    {"role": "user", "content" : message}
                ]
        )
        
        # print("Response for ", article_name, ": \n")
        # print(response.choices[0].message.content)

        try:
            response_as_dict = ast.literal_eval(response.choices[0].message.content)
        except (ValueError, SyntaxError) as e:
            print("Error parsing response for ", article_name)
            return {"article": article_name, "error": "Failed to parse response"}

        # Get expected results from the same file
        expected_task1 = article_data.get("task1", {})
        if not expected_task1:
            print("Warning: No expected task1 results found for ", article_name)
            return {"article": article_name, "error": "No expected results"}
        
        evaluation_result, y_true, y_pred = evaluate_task1(response_as_dict, expected_task1)
        
        return {
            "article": article_name,
            "response": response_as_dict,
            "evaluation": evaluation_result,
            "y_true": y_true,
            "y_pred": y_pred
        }
        
    except Exception as e:
        print("Error processing ", article_name, ": ", str(e))
        return {"article": article_name, "error": str(e)}

def main():
    articles_dir = "data/test/"
    article_paths = glob.glob(os.path.join(articles_dir, "*.json"))
    
    if not article_paths:
        print("No articles found in ", articles_dir)
        return
        
    client = load_client()
    results = []
    
    for article_path in article_paths:
        result = process_article(client, article_path)
        results.append(result)

    
    all_y_true = []
    all_y_pred = []
    
    for result in results:
        if result.get('y_true') and result.get('y_pred'):
            all_y_true.extend(result['y_true'])
            all_y_pred.extend(result['y_pred'])
    
    # this is just for printing to terminal
    if all_y_true and all_y_pred:
        overall_report = classification_report(
            all_y_true, 
            all_y_pred, 
            labels=["Victim-aligned", "Police-aligned"], 
            digits=3, 
            zero_division=0
        )
        print(overall_report)
    
    if all_y_true and all_y_pred:    
        report_dict = classification_report(
            all_y_true, 
            all_y_pred, 
            labels=["Victim-aligned", "Police-aligned"], 
            digits=3, 
            zero_division=0,
            output_dict=True
        )
        save_path = os.path.join("data", "test", "results", "task1_evaluation.json")
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print("Classification report saved.")
    else:
        print("No valid predictions found for overall report")
    
    

if __name__ == "__main__":
    main()