import openai
from dotenv import dotenv_values
from collections import defaultdict
import ast
import os
import json
from sklearn.metrics import classification_report


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

{

"Victim-aligned": [Person1, Person2, Person3],

"Government officials": [Person4, Person5, Person6],

"Policing institution officials": [Person 7, Person8]

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
    print("mapped affiliation is: ", res)
    return res
    


def evaluate_task1(pred_raw, expected_path):

    with open(expected_path, "r") as f:
        expected = json.load(f)

    pred = map_affiliations(pred_raw)

    people = set(pred["Victim-aligned"]) | set(pred["Police-aligned"]) | set(expected["Victim-aligned"]) | set(expected["Police-aligned"])

    y_true = []
    y_pred = []
    for person in people:
        if person in expected["Victim-aligned"]:
            y_true.append("Victim-aligned")
        elif person in expected["Police-aligned"]:
            y_true.append("Police-aligned")
        else:
            y_true.append("Unknown")

        if person in pred["Victim-aligned"]:
            y_pred.append("Victim-aligned")
        elif person in pred["Police-aligned"]:
            y_pred.append("Police-aligned")
        else:
            y_pred.append("Unknown")
    # print(classification_report(y_true, y_pred, labels=["Victim-aligned", "Police-aligned"], digits=3, zero_division=0))
    return classification_report(y_true, y_pred, labels=["Victim-aligned", "Police-aligned"], digits=3, zero_division=0, output_dict=True)
    

def main():
    article_name = "Dexter_Reed-2024_04_10-foody.json"
    article_path = os.path.join("data", "test", "articles", article_name)
    with open(article_path, "r") as f:
        article = f.read()
    message = task1_prompt + article
    client = load_client()
    response = client.chat.completions.create(
            model="gpt-4o", # gpt-4o, or you can also try gpt-4o-mini. See here for more details https://platform.openai.com/docs/models. But stick to 4o or 4o-mini
            temperature=0,
            max_tokens = 500, # set this to whatever makes sense. It's the maximum number of tokens (basically, words) that you think are necessary for responding to the prompt
            messages = [
                {"role": "user", "content" : message}
            ]
    )
    print(response.choices[0].message.content)

    response_as_dict = ast.literal_eval(response.choices[0].message.content)

    expected_name_task1 = "expected_task1_" + article_name
    expected_path_task1 = os.path.join("data", "test", "expected_task1", expected_name_task1)
    evaluate_task1(response_as_dict, expected_path_task1)
    
if __name__ == "__main__":
    main()