import json
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import classification_report
from splits import TRAIN_FILES, TEST_FILES

def get_civ_entities_training(annotation_obj, include_perspectives_only, count_victim=False):
    if include_perspectives_only:
        victim_aligned_entities = annotation_obj['task1']['Victim-aligned']
        victim_aligned_entities = [entity[:entity.rindex(" (")] for entity in victim_aligned_entities]
        perspective_entities = set([])
        # iterate through task 2 paragraph entities
        for _, entities in annotation_obj['task2'].items():
            for entity in entities:
                if entity in victim_aligned_entities:
                    perspective_entities.update(set([entity]))
        return list(perspective_entities)
    else:
        entities = annotation_obj['task1']['Victim-aligned']
        assert '(victim)' in entities[0] 
        entities = [entity[:entity.rindex(" (")] for entity in entities] # remove the (id) part

    return entities if count_victim else entities[1:]

def get_pb_entities_training(annotation_object, include_perspectives_only: bool):
    if include_perspectives_only:
        police_aligned_entities = annotation_object['task1']['Police-aligned']
        police_aligned_entities = [entity[:entity.rindex(" (")] for entity in police_aligned_entities] # remove the (id) part
        perspective_entities = set([])
        # iterate through task 2 paragraph entities
        for _, entities in annotation_object['task2'].items():
            for entity in entities:
                if entity in police_aligned_entities:
                    perspective_entities.update(set([entity]))
        return list(perspective_entities)
    else:
        entities = annotation_object['task1']['Police-aligned'] 
        entities = [entity[:entity.rindex(" (")] for entity in entities] # remove the (id) part
        return entities

def compute_paragraph_to_affinities(gt_annotations) -> Dict:
    paragraph_to_affinity = {}
    bureaucrats = get_pb_entities_training(gt_annotations, 
                                           include_perspectives_only=True)
    civ_entities = get_civ_entities_training(gt_annotations,
                                             include_perspectives_only=True)
    for paragraph_index, entities in gt_annotations['task2'].items(): 
        entity = entities[0] # entities is a list but usually only has one item.
        if entity in bureaucrats:
            paragraph_to_affinity[int(paragraph_index.split(' ')[1])] = 'police-aligned'
        elif entity in civ_entities:
            paragraph_to_affinity[int(paragraph_index.split(' ')[1])] = 'victim-aligned'
        else:
            raise ValueError(f"Entity {entity} not found in either police-aligned or victim-aligned entities.")
    return paragraph_to_affinity

def load_training_data_annotations_for_person(person_name: str, outlet: str,
                                              identifier: Optional[int] = -1):
    with open("data/training_data.json", 'r') as f:
        training_data_annotations = json.load(f)
    training_data_annotation = None
    matched_annotations = []
    for k, v in training_data_annotations.items():
        if person_name in k and outlet in k:
            if identifier != -1 and (not f"{identifier}" == k.split('_')[0]):
                continue
            training_data_annotation = v
            matched_annotations.append(training_data_annotation)
    assert training_data_annotation is not None, f"Could not find training data annotation for {person_name} and {outlet} and {identifier}."
    assert len(matched_annotations) == 1, f"Found multiple training data annotations for {person_name} and {outlet} and {identifier}: {matched_annotations}."
    return training_data_annotation

def convert_to_ternary_label_list(gt_paragraph_to_affinity, 
               predicted_paragraph_to_affinity, 
               num_paragraphs_in_article):
    gt_paragraph_to_affinity = gt_paragraph_to_affinity.copy()
    predicted_paragraph_to_affinity = predicted_paragraph_to_affinity.copy()
    # add any missing paragraphs as 'no entity', to both dictionaries.
    for i in range(1, num_paragraphs_in_article + 1):
        if i not in gt_paragraph_to_affinity:
            gt_paragraph_to_affinity[i] = 'no entity'
        if i not in predicted_paragraph_to_affinity:
            predicted_paragraph_to_affinity[i] = 'no entity'
    assert len(gt_paragraph_to_affinity) == len(predicted_paragraph_to_affinity)
    y_true = []
    y_pred = []
    for i in range(1, num_paragraphs_in_article + 1):
        y_true.append(gt_paragraph_to_affinity[i])
        y_pred.append(predicted_paragraph_to_affinity[i])
    return y_true, y_pred

def _strip_role(name: str) -> str:
    return name[:name.rindex(" (")] if " (" in name and name.endswith(")") else name

def compute_gpt_paragraph_to_affinities() -> Dict:
    assert 'pred_annotations' in globals(), "Define 'pred_annotations' before calling."

    victim_entities = [_strip_role(e) for e in pred_annotations['task1'].get('Victim-aligned', [])]
    police_entities = [_strip_role(e) for e in pred_annotations['task1'].get('Police-aligned', [])]
    victim_set = set(victim_entities)
    police_set = set(police_entities)

    paragraph_to_affinity = {}

    for paragraph_index, entities in pred_annotations['task2'].items():
        try:
            idx = int(paragraph_index.split(' ')[1])
        except Exception:
            idx = int(paragraph_index)

        if not entities:
            paragraph_to_affinity[idx] = 'no entity'
            continue

        first_entity = _strip_role(entities[0])

        if first_entity in police_set:
            label = 'police-aligned'
        elif first_entity in victim_set:
            label = 'victim-aligned'
        else:
            # fallback: if any listed entity matches either side
            any_police = any(_strip_role(e) in police_set for e in entities)
            any_victim = any(_strip_role(e) in victim_set for e in entities)
            if any_police:
                label = 'police-aligned'
            elif any_victim:
                label = 'victim-aligned'
            else:
                label = 'no entity'

        paragraph_to_affinity[idx] = label

    return paragraph_to_affinity


if __name__ == '__main__':
    PRED_FILE = "v1/2025-10-06_15-02_predictions.json"
    SPLIT = TEST_FILES
    SPLIT_NAME = "TEST"

    with open(PRED_FILE, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    all_y_true, all_y_pred = [], []

    for fname in SPLIT:
        stem = fname[:-5]
        parts = stem.split("_")
        identifier = int(parts[0])
        person_name = parts[1]
        outlet = parts[2]

        # gold
        gt = load_training_data_annotations_for_person(
            person_name=person_name, outlet=outlet, identifier=identifier
        )
        gt_map = compute_paragraph_to_affinities(gt)

        rec = bundle.get(fname)
        if not rec:
            print(f"Missing preds for {fname}. Skipping.")
            continue

        pred_annotations = {
            "task1": rec["task1_prediction"],
            "task2": rec["task2_prediction"],
        }
        pred_map = compute_gpt_paragraph_to_affinities()

        idxs = set(gt_map.keys()) | set(pred_map.keys())
        num_paras = max(idxs) if idxs else 0

        y_true, y_pred = convert_to_ternary_label_list(gt_map, pred_map, num_paras)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    report_str = classification_report(
        all_y_true, all_y_pred,
        labels=["police-aligned", "victim-aligned", "no entity"]
    )

    stem = Path(PRED_FILE).stem 
    tag = stem.replace("_predictions", "")
    out_dir = Path("results") / f"{tag}_eval_{SPLIT_NAME}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(report_str, encoding="utf-8")
    (out_dir / "y_true.json").write_text(json.dumps(all_y_true, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "y_pred.json").write_text(json.dumps(all_y_pred, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "files_evaluated.json").write_text(json.dumps(SPLIT, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report_str)
    print(f"\nSaved to: {out_dir}/classification_report.txt")