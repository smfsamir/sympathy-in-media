import openai
from dotenv import dotenv_values
from collections import defaultdict
import ast
import os
import json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import glob


AFFILIATION_MAP = {
    "Victim-aligned": "Victim-aligned",
    "Government officials": "Police-aligned", #
    "Policing institution officials": "Police-aligned",
    "Police-aligned" : "Police-aligned"
}

'''

# NOTE
## Questions:
# - do we want to identify bodies/orgs even if a spokesperson is not mentioned? unions, watchdogs, police departments, etc.
#   - police say...the police watchdog reports...

# TODO
## move prompts into their own text files
## better logical flow (for evaluation  - have the overall eval fn call the individual eval fns)
## TASK 1:
# better define categories
# resolve issue of neutral parties
# direct

'''
# people only

# task1_prompt = """

# Read the provided article on an incident of police violence to complete the following two tasks. Task 1: Identify each person mentioned in the article and determine their affiliation. The same person could be mentioned many times by different referring expressions.

# For instance, the terms "he", "Bob Dylan", "the singer", and "Dylan" all refer to the same entity.

# Use the article context to recognize these mentions as co-references and list the person only once using their most complete name (e.g. "Bob Dylan"). Do not create separate entities for pronouns, mentions, and names (e.g. "he", "the singer", "Bob Dylan", etc). A person can either be on the side of the police, or the civilian victim.

# For example, government officials including police chiefs, mayors, police oversight officials, police unions officials. Government officials should always be assigned into the police-aligned category, even if their statements are ostensibly sympathetic. 
# There may also be bystanders who take the side of the police. For these bystanders, it is important to take the quotes into account in determining whether they side with the police or not.  

# People who side with the civilian victim, on the other hand, tend to be, of course, the victim's family or friends, their attorney, community members who knew the victim, witnesses, civil rights organization leaders, and more.

# People mentioned from previous cases (whether officers or victims) should also be recorded, even if they are not part of the current case itself. 

# Following each person's name, indicate their role in parentheses, such as (police chief), (witness), (mayor), (victim's mother), (victim's attorney), (police officer) etc. Use all lowercase for the names and roles.

# Return the results in the following format:

# {

# "Victim-aligned": ["person1 (role)", "person2 (role)", "person3 (role)"],

# "Government officials": ["person (role)", "person5 (role)", "person6 (role)"],

# "Policing institution officials": ["person7 (role)", "person8 (role)"]

# }

# Here is the article:


task1_prompt = """
Read the provided article on an incident of police violence to complete the following task.

Task: Identify each entity mentioned in the article and determine their role and affiliation. The entity may be a person or an instituion, like the ACLU or the Special Investigations Unit (SIU). 

Restrictions on who/what to identify:
- Only entities that are identified by either a name (e.g.,idenitfy "Matt Dumas", "Matt", "American Civil Liberties Union" "ACLU", and do not identify "man", "an organization") or relationship should be included (e.g. "victim's brother", "victim's neighbour", "officer's supervisor", etc.)
- Do not include media outlets as entities
- Avoid including entities which are only mentioned generically (e.g. "the officer", "the man") unless it is absolutely necessary to understand the article. If the generic, unnamed entity is the victim mark their name as "victim". 
- Only include civilians if they 1. are mentioned by name or relationship and 2. have been quoted or directly paraphrased in the article
- A person may represent or be a spokeperson for an organization. Identify both the person and the organization as seperate entities.

How to label entities:
- Avoid abbreviations when the full name is mentioned (say "American Civil Liberties Union" and not ACLU)
- The same entity could be mentioned many times by different referring expressions. For instance, the terms "he", "Bob Dylan", "the singer", and "Dylan" all refer to the same person. 
Use the article context to recognize these mentions as co-references and list the entity only once using their most full name or most descriptive name (e.g. "Bob Dylan"). 
Do not create separate entities for pronouns, mentions, and names (e.g. "he", "the singer", "Dylan", etc). 
- Every entity has a role. Following each entity's name, indicate their role in parentheses, such as (police chief), (witness), (mayor), (victim's mother), (legal nonprofit), (police oversight body) (victim's attorney), (police officer), (police union), etc. 
    - For example, if the article mentions "mayor john", "the victim's mother jane" and "police chief bob", you would return "john (mayor)", "jane (victim's mother)", "bob (police chief)"
    - Never include the role in the name, e.g. tag "const. jerry" as "jerry (const.)" and not "const. jerry (const.)" or "const. jerry"
- For unnamed entities described by a relationship, include the full name of the entity with whom they have a relationship. For example, if an article about Bob Dylan mentions "dylan's brother", tag this person as "bob dylan's brother"

Affiliation: An entity can either be on the side of the police, or the civilian victim.

Police-aligned entities:
- Government and policing entities (both organizations and individuals working for them) are always police-aligned. Even if their statements are ostensibly sympathetic, they should always be considered police-aligned. This includes police chiefs, mayors, police oversight bodies, police unions, etc. 
- For individual police officers, only include them if they are mentioned by name. Do not include generic references to a police officer (e.g., "the police officer", "the officer", "the cop", etc.).
- Attempt to identify the specific police department, if it is explicity named. Otherwise use the generic term "police"
- There may also be bystanders who take the side of the police. For these bystanders, it  is important to take their quotes into account to determine their affiliation. If their opinion sympathizes with the police more than the victim, assign them police-aligned. 

Victim-aligned entities:
- Entities who side with the civilian victim, on the other hand, tend to be, of course, the victim's family or friends, their attorney, community members who knew the victim, witnesses, civil rights organization, and more.
- If a person is clearly identified as a family member or friend (e.g., “my dad,” “his sister,” “the victim’s mother”), but their name is not mentioned, you should still include them.
- Bystanders may take the side of the victim.  For these bystanders, it  is important to take their quotes into account to determine their affiliation. If their opinion sympathizes with the victim more than the police, assign them victim-aligned. 

People mentioned from previous cases (whether officers or victims) should also be recorded, even if they are not part of the current case itself.

Output format:
- Use all lowercase for the names and roles
- Use the following format exactly: 
{
"Victim-aligned": ["entity1 (role)", "entity2 (role)", "entity3 (role)"],
"Police-aligned": ["entity4 (role)", "entity5 (role)"]
}

Here is the article:
"""

task2_prompt = """

Next, assign each paragaph with the people whose perspectives are being demonstrated in it. This must be a person who was previously identified in your last response.
There may be zero, one or multiple perspective in a given paragaph. If there are multiple perspectives, list them all. If there are no perspectives, return an empty list.

A perspective can be given through a quote, paraphrase, or a description of the person's opinions or feelings.

Return a list of lists, where the ith inner list corresponds to the ith paragraph in the article. Each inner list should contain the names of the people whose perspectives are being demonstrated in that paragraph.
Use all lowercase.

You must return exactly one list per paragraph**, even if it is an empty list. Do not merge or skip any paragraphs. The article is separated by "\n\n-PARAGRAPH BREAK-\n\n" to indicate where a new paragraph starts.

Example output:

[
    ["person1", "person2"],
    ["person3"],
    [],
    ["person4", "person1"]
]

This represents a 4-paragraph article, where the first paragraph has two perspectives, the second has one, the third has none, and the fourth has two perspectives.
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

### TASK 1 HELPERS ###

# maps affiliations from gpt to our format
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

# remove roles 
def clean_name(name):
    if "(" in name and ")" in name:
        return name.split("(")[0].strip()

# evaluates task 1 for a single article 
def evaluate_task1(pred_raw, expected_dict, article_name):
    # pred is a dict with keys "Victim-aligned" and "Police-aligned"
    # where each entry in the values list is a person's name and (role)
    pred = map_affiliations(pred_raw) 
    
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

    print("------", article_name, "-----TASK 1 EVALUATION----")
    print("\nPredicted Victim-aligned: ", pred_victim)
    print("\nActual Victim-aligned: ", set(expected_dict["Victim-aligned"]))
    print("\nPredicted Police-aligned: ", pred_police)
    print("\nActual Police-aligned: ", set(expected_dict["Police-aligned"]))

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
            output_dict=True,
        )
        save_path = os.path.join("data", "test", "results", "task1_evaluation.json")
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print("Classification report saved.")
    else:
        print("No valid predictions found for overall report")



# Task1: identifies individuals and their affiliations in an article
def task1(client, article_path):
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
        
        evaluation_result, y_true, y_pred = evaluate_task1(response_as_dict, expected_task1, article_name)
        
        return {
            "article": article_name,
            "article_data": article_data,
            "response": response.choices[0].message.content,
            "evaluation": evaluation_result,
            "y_true": y_true,
            "y_pred": y_pred,
        }
        
    except Exception as e:
        print("Error processing ", article_name, ": ", str(e))
        return {"article": article_name, "error": str(e)}

### TASK 2 HELPERS ###

# evaluates task 2 for a single article
def evaluate_task2(expected, predicted):
    # assert len(predicted) == len(expected) 
    if not len(predicted) == len(expected):
        print("EXPECTED LENGTH: ,", len(expected), "PREDICTED LENGTH: ", len(predicted))
        print("\nEXPECTED: ,", expected)
        print("\nPREDICTED: ,", predicted)

        return (None, None, None)

    y_true = []
    y_pred = []

    all_people = set()
    for p in predicted:
        all_people.update(p)
    for p in expected:
        all_people.update(p)

    for exp, pred in zip(expected, predicted):
        for person in all_people:
            y_true.append(1 if person in exp else 0)
            y_pred.append(1 if person in pred else 0)

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

        print("OVERALL TASK 2 REPORT:")
        print(json.dumps(report_dict))

        save_path = os.path.join("data", "test", "results", "task2_evaluation.json")
        with open(save_path, "w") as f:
            json.dump(report_dict, f)
        print("Task 2 report saved.")

    else:
        print("No valid Task 2 predictions found.")


# Task 2: assign paragraphs to individuals    
def task2(client, article_data, task1_response):
    article_text = article_data.get("article", "")

    article_text = article_data.get("article", "")
    if isinstance(article_text, list):
        article_text = "\n\n-PARAGRAPH BREAK-\n\n".join(article_text)
    

    # if isinstance(article_text, list):
    #         article_content = "\n".join(article_text)
    #         print("ARTICLE IS A LIST OF SIZE", len(article_text), "AND IN STRING ITS LENGTH: ", len(article_content))
    
    # print("ARTICLE TEXT:", article_text)

    messages = [
        {"role": "user", "content": task1_prompt + article_text},
        {"role": "assistant", "content": task1_response},
        {"role": "user", "content": task2_prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=500,
        messages=messages
    )

    raw_output = response.choices[0].message.content

    try:
        parsed_res = ast.literal_eval(raw_output)
    except Exception as e:
        print("Failed to parse Task 2 output:", raw_output)
        return {"error": str(e), "raw": raw_output}

    expected_task2 = article_data.get("task2", {})
    if not expected_task2:
        print("Warning: No expected task2 results found")
        return {"error": "No expected task2 labels", "raw": parsed_res}

    eval_metrics, y_true, y_pred = evaluate_task2(expected_task2, parsed_res)

    if eval_metrics is None:
        print("Skipping Task 2 eval due to length mismatch.")
        return {
            "parsed_result": parsed_res,
            "raw_output": raw_output,
            "error": "Length mismatch",
        }


    return {
        "parsed_result": parsed_res,
        "raw_output": raw_output,
        "evaluation": {
            "precision": eval_metrics[0],
            "recall": eval_metrics[1],
            "f1": eval_metrics[2],
            "support": eval_metrics[3],
        },
        "y_true": y_true,
        "y_pred": y_pred
    }


def main():
    articles_dir = "data/test/" #TODO: change after testing is done
    article_paths = glob.glob(os.path.join(articles_dir, "*.json"))
    
    if not article_paths:
        print("No articles found in ", articles_dir)
        return
        
    client = load_client()
    
    # task 1
    task1_results = []
    for article_path in article_paths:
        task1_result = task1(client, article_path)
        # print("TASK 1 --- ", task1_result["article"], ": ", task1_result["response"])
        task1_results.append(task1_result)
    
    task1_evaluation_report(task1_results)

    # task 2
    task2_results = []
    # for task1_result in task1_results:
    #     task2_result = task2(client, task1_result["article_data"], task1_result["response"])
    #     task2_results.append(task2_result)
    #     # print("Task 2 result for ", task1_result["article"], ": ", task2_result)
    #     # print(f"RESPONSE --- ", type(task1_result["response"]))

    # task2_evaluation_report(task2_results)

    
    

if __name__ == "__main__":
    main()