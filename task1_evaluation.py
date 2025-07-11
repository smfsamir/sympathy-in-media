import os
import json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import glob
from datetime import datetime

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

def clean_name(name):
    if "(" in name and ")" in name:
        return name.split("(")[0].strip()
    
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